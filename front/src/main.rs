//! shodh-front — the Associative Memory dashboard server.
//!
//! A thin front for the shodh backend: it serves the single-page UI + its
//! bundled assets, and reverse-proxies every `/api/*` call to the backend
//! (`SHODH_API_URL`), injecting the API key. Responses are STREAMED, so the
//! Server-Sent-Events endpoint (`/api/events`, the live recall river) forwards
//! without buffering.
//!
//! Env:
//!   SHODH_FRONT_PORT   listen port           (default 8787)
//!   SHODH_API_URL      backend base URL       (default http://127.0.0.1:3030)
//!   SHODH_API_KEY      injected as X-API-Key  (default empty)

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{Html, IntoResponse, Response},
    routing::{any, get},
    Router,
};
use std::net::SocketAddr;

/// The UI and its one asset are embedded in the binary — the front is
/// self-contained and needs no working directory.
const INDEX_HTML: &str = include_str!("../index.html");
const D3_JS: &str = include_str!("../assets/d3.v7.min.js");

#[derive(Clone)]
struct Backend {
    base: String,
    api_key: String,
    client: reqwest::Client,
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("SHODH_FRONT_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8787);
    let base = std::env::var("SHODH_API_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:3030".to_string())
        .trim_end_matches('/')
        .to_string();
    let api_key = std::env::var("SHODH_API_KEY").unwrap_or_default();

    let backend = Backend {
        base: base.clone(),
        api_key,
        client: reqwest::Client::new(),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/assets/d3.v7.min.js", get(d3))
        .route("/api/{*path}", any(proxy))
        .with_state(backend);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("shodh-front: cannot bind {addr}: {e}"));
    println!("shodh-front on http://{addr}  →  backend {base}");
    axum::serve(listener, app).await.unwrap();
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn d3() -> impl IntoResponse {
    ([("content-type", "application/javascript; charset=utf-8")], D3_JS)
}

/// Reverse-proxy `/api/*` to the backend, streaming the response so SSE works.
async fn proxy(
    State(backend): State<Backend>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Preserve the full original path + query ("/api/recall?..." etc.).
    let path_and_query = uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or_else(|| uri.path());
    let target = format!("{}{}", backend.base, path_and_query);

    let mut req = backend.client.request(method, &target).body(body);
    // Forward the request headers the backend cares about; inject the key.
    for name in ["content-type", "accept"] {
        if let Some(v) = headers.get(name) {
            req = req.header(name, v);
        }
    }
    if !backend.api_key.is_empty() {
        req = req.header("X-API-Key", &backend.api_key);
    }

    match req.send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::BAD_GATEWAY);
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/octet-stream")
                .to_string();
            let mut builder = Response::builder()
                .status(status)
                .header("content-type", content_type);
            // Keep SSE responses unbuffered end-to-end.
            if let Some(cc) = resp.headers().get("cache-control") {
                if let Ok(cc) = cc.to_str() {
                    builder = builder.header("cache-control", cc.to_string());
                }
            }
            builder
                .body(Body::from_stream(resp.bytes_stream()))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("shodh-front proxy error → {}: {e}", backend.base),
        )
            .into_response(),
    }
}
