//! shodh-front — the Associative Memory dashboard server.
//!
//! A thin front for the shodh backend: it serves the single-page UI (the React
//! app under `front/ui`, embedded as one self-contained file at compile time —
//! see `build.rs`), and reverse-proxies every `/api/*` call to the backend
//! (`SHODH_API_URL`), injecting the API key. Responses are STREAMED, so the
//! Server-Sent-Events endpoint (`/api/events`, the live recall river) forwards
//! without buffering.
//!
//! It also reverse-proxies `/seat/*` to the conversation-seat harness
//! (`SHODH_SEAT_URL`), stripping the `/seat` prefix and injecting its bearer
//! token — the browser holds neither the backend API key nor the seat token,
//! and the seat's SSE message stream forwards unbuffered like the backend's.
//!
//! Env:
//!   SHODH_FRONT_PORT   listen port           (default 8787)
//!   SHODH_API_URL      backend base URL       (default http://127.0.0.1:3030)
//!   SHODH_API_KEY      injected as X-API-Key  (default empty)
//!   SHODH_SEAT_URL     seat harness base URL  (default http://127.0.0.1:3141)
//!   SHODH_SEAT_TOKEN   injected as Authorization: Bearer (default empty)

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{Html, IntoResponse, Response},
    routing::{any, get},
    Router,
};
use std::net::SocketAddr;

/// The UI is embedded in the binary — the front is self-contained and needs no
/// working directory.
///
/// This is the React app under `front/ui`, built by Vite into exactly ONE
/// self-contained file: `vite-plugin-singlefile` inlines every script,
/// stylesheet and asset into the HTML (see front/ui/vite.config.ts), so there
/// is nothing beside it to serve. `dist/` is a build artefact and is not
/// committed; `build.rs` fails with the remedy if it has not been produced.
const INDEX_HTML: &str = include_str!("../ui/dist/index.html");

#[derive(Clone)]
struct Backend {
    base: String,
    api_key: String,
    seat_base: String,
    seat_token: String,
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
    let seat_base = std::env::var("SHODH_SEAT_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:3141".to_string())
        .trim_end_matches('/')
        .to_string();
    let seat_token = std::env::var("SHODH_SEAT_TOKEN").unwrap_or_default();

    let backend = Backend {
        base: base.clone(),
        api_key,
        seat_base,
        seat_token,
        client: reqwest::Client::new(),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/{*path}", any(proxy))
        .route("/seat/{*path}", any(seat_proxy))
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

    forward(req, &backend.base).await
}

/// Reverse-proxy `/seat/*` to the conversation-seat harness, stripping the
/// `/seat` prefix (`/seat/v1/models` → `{SHODH_SEAT_URL}/v1/models`) and
/// injecting the seat bearer token. Streaming end-to-end — the seat's message
/// endpoint is SSE.
async fn seat_proxy(
    State(backend): State<Backend>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let path_and_query = uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or_else(|| uri.path());
    let stripped = path_and_query.strip_prefix("/seat").unwrap_or(path_and_query);
    let target = format!("{}{}", backend.seat_base, stripped);

    let mut req = backend.client.request(method, &target).body(body);
    for name in ["content-type", "accept"] {
        if let Some(v) = headers.get(name) {
            req = req.header(name, v);
        }
    }
    if !backend.seat_token.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", backend.seat_token));
    }

    forward(req, &backend.seat_base).await
}

/// Headers this proxy must NEVER copy from an upstream response.
///
/// The proxy is a *denylist*, not an allowlist. It used to forward exactly
/// `content-type` and `cache-control`, which silently dropped every other
/// header both upstreams set — including the four the backend's
/// `security_headers` middleware stamps on EVERY response
/// (`x-content-type-options`, `x-frame-options`, `content-security-policy`,
/// and `strict-transport-security` in production). The dev proxy forwards
/// everything, so the shipped binary was the only place the dashboard ran
/// without its security headers. An allowlist re-creates that failure mode
/// every time either service starts setting a new header; a denylist fails the
/// other way, which is the safe way for a header the proxy has not heard of.
///
/// Two groups are denied:
///
/// 1. **Hop-by-hop headers** (RFC 9110 §7.6.1). They describe the single
///    connection they arrived on, not the message, and a proxy must not pass
///    them to the next hop.
/// 2. **`content-length`.** The body is re-framed here —
///    [`Body::from_stream`] hands hyper a stream and hyper derives the
///    framing. An upstream `content-length` copied onto a re-framed body is at
///    best redundant and at worst a contradiction.
///
/// `content-encoding` is deliberately NOT denied: it is an end-to-end header
/// and this proxy does not decode bodies (`front`'s reqwest is built with
/// `default-features = false` and no `gzip`/`brotli`/`deflate`/`zstd`
/// feature), so a coded body must keep its label. Nothing in the product emits
/// it today — the backend has no compression layer and this proxy does not
/// forward the browser's `accept-encoding`, so compression is never
/// negotiated.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
];

/// Headers withheld on purpose even though they are end-to-end.
///
/// `set-cookie` is the only header either upstream could send that would leave
/// persistent state on the dashboard's origin, and the browser talks to this
/// proxy on one origin for the UI, the backend (`/api/*`) and the seat
/// (`/seat/*`) alike — so an upstream cookie would be scoped to the page
/// itself. Neither service sets one: the backend authenticates with
/// `X-API-Key` and the seat with a bearer token, both injected here and never
/// exposed to the browser, and neither has a session concept. Withholding a
/// header nobody sends costs nothing and keeps one upstream from writing
/// durable state into the other's origin.
///
/// If a real browser-facing auth flow is ever introduced, this is the line
/// that has to be revisited — not the denylist above.
const WITHHELD: &[&str] = &["set-cookie"];

/// Decide whether a response header from an upstream may be forwarded.
///
/// Split out from [`forward`] so it can be tested without a live upstream.
/// `connection_tokens` are the header names listed in the upstream's own
/// `Connection:` header, which RFC 9110 §7.6.1 makes hop-by-hop for that
/// message even though they are not hop-by-hop in general.
fn may_forward(name: &str, connection_tokens: &[String]) -> bool {
    let lower = name.to_ascii_lowercase();
    if HOP_BY_HOP.contains(&lower.as_str()) || WITHHELD.contains(&lower.as_str()) {
        return false;
    }
    !connection_tokens.contains(&lower)
}

/// The header names an upstream declared hop-by-hop via its `Connection:`
/// header, lowercased. `Connection: close` and `Connection: keep-alive` name
/// no other header, but they are harmless here — both are already denied.
fn connection_tokens(headers: &reqwest::header::HeaderMap) -> Vec<String> {
    headers
        .get_all("connection")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|t| t.trim().to_ascii_lowercase())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Send a proxied request and stream the response back unbuffered.
async fn forward(req: reqwest::RequestBuilder, upstream: &str) -> Response {
    match req.send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::BAD_GATEWAY);
            let mut builder = Response::builder().status(status);

            let tokens = connection_tokens(resp.headers());
            let mut saw_content_type = false;
            for (name, value) in resp.headers() {
                if !may_forward(name.as_str(), &tokens) {
                    continue;
                }
                if name.as_str().eq_ignore_ascii_case("content-type") {
                    saw_content_type = true;
                }
                // `append`, not `insert`: repeated headers are meaningful
                // (`vary`, `set-cookie` were it forwarded) and collapsing them
                // would lose all but the last.
                builder = builder.header(name.as_str(), value);
            }
            // Same fallback the two-header version had: a body with no declared
            // type is opaque bytes, not guessable ones — the dashboard must not
            // be handed something a browser will sniff.
            if !saw_content_type {
                builder = builder.header("content-type", "application/octet-stream");
            }

            builder
                .body(Body::from_stream(resp.bytes_stream()))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("shodh-front proxy error → {upstream}: {e}"),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::{connection_tokens, may_forward};
    use reqwest::header::HeaderMap;

    fn no_tokens() -> Vec<String> {
        Vec::new()
    }

    /// The regression this file exists to prevent: the proxy forwarded exactly
    /// `content-type` and `cache-control`, so every security header the
    /// backend's `security_headers` middleware stamps was dropped before it
    /// reached the browser. Captured live from the running backend
    /// (`GET /api/list/{user}` → 200) on 2026-08-16:
    ///
    /// ```text
    /// content-type: application/json
    /// vary: origin, access-control-request-method, access-control-request-headers
    /// access-control-allow-origin: *
    /// x-content-type-options: nosniff
    /// x-frame-options: DENY
    /// content-security-policy: default-src 'none'
    /// cache-control: no-store
    /// content-length: 387
    /// ```
    ///
    /// Only the first and the `cache-control` line survived the proxy.
    #[test]
    fn security_headers_reach_the_browser() {
        for name in [
            "x-content-type-options",
            "x-frame-options",
            "content-security-policy",
            "strict-transport-security",
        ] {
            assert!(
                may_forward(name, &no_tokens()),
                "{name} is set on every backend response and must not be dropped by the proxy"
            );
        }
    }

    /// Headers whose loss is silent but load-bearing: `allow` is the whole
    /// content of a 405, `retry-after` is how a client learns to back off from
    /// a 429, and `x-accel-buffering: no` (set by the seat on both its SSE
    /// streams) is what stops a downstream proxy buffering a live stream.
    #[test]
    fn conditional_headers_reach_the_browser() {
        for name in [
            "allow",
            "retry-after",
            "x-ratelimit-after",
            "x-accel-buffering",
            "vary",
            "access-control-allow-origin",
            "content-type",
            "cache-control",
            "content-encoding",
        ] {
            assert!(
                may_forward(name, &no_tokens()),
                "{name} must be forwarded"
            );
        }
    }

    /// The body is re-framed by hyper via `Body::from_stream`, so upstream
    /// framing must not be copied onto it, and hop-by-hop headers describe a
    /// connection this response is leaving.
    #[test]
    fn framing_and_hop_by_hop_are_dropped() {
        for name in [
            "content-length",
            "transfer-encoding",
            "connection",
            "keep-alive",
            "upgrade",
            "te",
            "trailer",
            "proxy-authenticate",
            "proxy-authorization",
        ] {
            assert!(
                !may_forward(name, &no_tokens()),
                "{name} must NOT be forwarded"
            );
        }
    }

    /// Withheld on purpose — see `WITHHELD`.
    #[test]
    fn set_cookie_is_withheld() {
        assert!(!may_forward("set-cookie", &no_tokens()));
    }

    /// Header names are case-insensitive; the denylist compares lowercased.
    #[test]
    fn denylist_is_case_insensitive() {
        assert!(!may_forward("Content-Length", &no_tokens()));
        assert!(!may_forward("Transfer-Encoding", &no_tokens()));
        assert!(may_forward("X-Frame-Options", &no_tokens()));
    }

    /// A header the upstream itself declared hop-by-hop via `Connection:` is
    /// hop-by-hop for that message only.
    #[test]
    fn connection_listed_headers_are_dropped() {
        let mut headers = HeaderMap::new();
        headers.insert("connection", "keep-alive, X-Upstream-Only".parse().unwrap());
        let tokens = connection_tokens(&headers);

        assert!(!may_forward("x-upstream-only", &tokens));
        // ...and only for that message.
        assert!(may_forward("x-upstream-only", &no_tokens()));
        // Unrelated headers are unaffected.
        assert!(may_forward("x-frame-options", &tokens));
    }

    #[test]
    fn connection_tokens_absent_is_empty() {
        assert!(connection_tokens(&HeaderMap::new()).is_empty());
    }
}
