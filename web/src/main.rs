//! shodh-front — the local web frontend for Shodh Memory.
//!
//! A standalone binary (the web analog of `shodh-tui`): every asset is
//! compiled in — the dashboard HTML and a vendored D3 — so it runs fully
//! offline with zero CDN/network dependencies. It serves the UI on its own
//! port with a page-appropriate CSP, while the API server stays pure-JSON
//! under its strict `default-src 'none'` policy.
//!
//!   SHODH_API_URL     API server the browser talks to (default http://127.0.0.1:3030)
//!   SHODH_API_KEY     if set, injected so the dashboard authenticates without prompting
//!   SHODH_FRONT_PORT  listen port (default 8787)

use axum::{
    http::{header, HeaderValue},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};

const DASHBOARD: &str = include_str!("../assets/dashboard.html");
const D3_JS: &str = include_str!("../assets/d3.v7.min.js");
const CDN_TAG: &str =
    r#"<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>"#;

fn api_url() -> String {
    std::env::var("SHODH_API_URL")
        .or_else(|_| std::env::var("SHODH_SERVER_URL"))
        .unwrap_or_else(|_| "http://127.0.0.1:3030".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// Bake the final page: local D3 instead of the CDN, and the API base/key
/// injected via the `window.__SHODH_API_BASE__` hook the dashboard already
/// honors (`const API_BASE = window.__SHODH_API_BASE__ || ""`).
fn bake_dashboard() -> String {
    let api = api_url();
    let key = std::env::var("SHODH_API_KEY").unwrap_or_default();
    let inject = format!(
        "<script src=\"/assets/d3.v7.min.js\"></script>\n<script>window.__SHODH_API_BASE__={};window.__SHODH_API_KEY__={};</script>",
        js_string(&api),
        js_string(&key),
    );
    if !DASHBOARD.contains(CDN_TAG) {
        // The dashboard asset drifted from the CDN tag this binary was built
        // against — fail loudly at startup, not silently in the browser.
        panic!("dashboard.html no longer contains the expected D3 CDN tag; update CDN_TAG");
    }
    DASHBOARD.replace(CDN_TAG, &inject)
}

/// Minimal JS string literal escaping for the injected config values.
fn js_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '<' => out.push_str("\\u003c"),
            '\n' | '\r' => {}
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// UI-appropriate CSP: everything self-hosted; XHR/SSE allowed only to self
/// and the configured API origin. The API server's strict policy is untouched.
async fn security_headers(req: axum::extract::Request, next: Next) -> Response {
    let csp = format!(
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; \
         img-src 'self' data:; connect-src 'self' {}; frame-ancestors 'none'",
        api_url()
    );
    let mut resp = next.run(req).await;
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_str(&csp).unwrap_or_else(|_| HeaderValue::from_static("default-src 'self'")),
    );
    h.insert("X-Content-Type-Options", HeaderValue::from_static("nosniff"));
    h.insert("X-Frame-Options", HeaderValue::from_static("DENY"));
    resp
}

async fn dashboard(axum::extract::State(page): axum::extract::State<&'static str>) -> Html<&'static str> {
    Html(page)
}

async fn d3_asset() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        D3_JS,
    )
}

async fn health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Bake once; leak to get a &'static str for zero-copy responses.
    let page: &'static str = Box::leak(bake_dashboard().into_boxed_str());

    let app = Router::new()
        .route("/", get(dashboard))
        .route("/dashboard", get(dashboard))
        .route("/assets/d3.v7.min.js", get(d3_asset))
        .route("/health", get(health))
        .with_state(page)
        .layer(middleware::from_fn(security_headers));

    let port: u16 = std::env::var("SHODH_FRONT_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8787);
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));

    tracing::info!("shodh-front serving http://{addr} -> API {}", api_url());
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("cannot bind {addr}: {e}"));
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("server error");
}
