//! P1.3: HTTP request tracking middleware for observability
//!
//! Provides:
//! - Request ID generation and propagation
//! - HTTP latency and count metrics
//! - Path normalization to prevent cardinality explosion

use axum::{
    extract::Request,
    http::{header::HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
};
use std::time::Instant;
use uuid::Uuid;

/// Request ID extension for correlation across logs and errors
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

impl RequestId {
    /// Generate a new unique request ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create from existing ID string
    pub fn from_string(id: String) -> Self {
        Self(id)
    }

    /// Get the ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Request ID header name (standard header used by many load balancers)
pub const REQUEST_ID_HEADER: &str = "X-Request-ID";

/// Middleware to add/propagate request IDs for distributed tracing
///
/// Behavior:
/// - If `X-Request-ID` header is present in request, use it
/// - Otherwise, generate a new UUID v4
/// - Add the ID to response headers
/// - Store in request extensions for downstream handlers
pub async fn request_id(mut req: Request, next: Next) -> Response {
    // Extract or generate request ID
    // Sanitize: only allow [a-zA-Z0-9\-_.] to prevent log injection
    let request_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty() && s.len() <= 64)
        .filter(|s| {
            s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        })
        .map(|s| RequestId::from_string(s.to_string()))
        .unwrap_or_else(RequestId::new);

    // Store in extensions for handlers to access
    req.extensions_mut().insert(request_id.clone());

    // Add to tracing span
    let _span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %req.method(),
        path = %req.uri().path()
    );

    // Process request
    let mut response = next.run(req).await;

    // Add request ID to response headers
    if let Ok(header_value) = HeaderValue::from_str(&request_id.0) {
        response
            .headers_mut()
            .insert(REQUEST_ID_HEADER, header_value);
    }

    response
}

/// Middleware to add security response headers
///
/// Adds:
/// - X-Content-Type-Options: nosniff (prevent MIME-type sniffing)
/// - X-Frame-Options: DENY (prevent clickjacking)
/// - Content-Security-Policy: default-src 'none' (restrict resource loading)
/// - Cache-Control: no-store (prevent caching of API responses)
/// - Strict-Transport-Security (HSTS) in production mode only
pub async fn security_headers(req: Request, next: Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();

    headers.insert(
        "X-Content-Type-Options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("X-Frame-Options", HeaderValue::from_static("DENY"));
    headers.insert(
        "Content-Security-Policy",
        HeaderValue::from_static("default-src 'none'"),
    );
    headers.insert("Cache-Control", HeaderValue::from_static("no-store"));

    // HSTS in production only (requires HTTPS to be meaningful)
    if crate::auth::is_production_mode() {
        headers.insert(
            "Strict-Transport-Security",
            HeaderValue::from_static("max-age=31536000; includeSubDomains"),
        );
    }

    response
}

/// Slow request warning threshold (seconds)
const SLOW_REQUEST_THRESHOLD_SECS: f64 = 30.0;

/// P1.3: Middleware to track HTTP request latency and counts
pub async fn track_metrics(req: Request, next: Next) -> Result<Response, StatusCode> {
    let start = Instant::now();
    let method = req.method().to_string();
    let path = req.uri().path().to_string();

    // Process request
    let response = next.run(req).await;

    // Record metrics
    let duration = start.elapsed().as_secs_f64();
    let status_code = response.status();
    let status = status_code.as_u16().to_string();

    // Normalize path to avoid high cardinality (group dynamic IDs)
    let normalized_path = normalize_path(&path);

    // Log timeouts (408) for observability - helps identify which endpoints need attention
    if status_code == StatusCode::REQUEST_TIMEOUT {
        tracing::error!(
            method = %method,
            path = %path,
            normalized_path = %normalized_path,
            duration_secs = %duration,
            "Request timeout - endpoint exceeded configured timeout limit"
        );
    } else if duration > SLOW_REQUEST_THRESHOLD_SECS {
        // Log slow requests that haven't timed out yet
        tracing::warn!(
            method = %method,
            path = %path,
            normalized_path = %normalized_path,
            duration_secs = %format!("{:.2}", duration),
            "Slow request - approaching timeout threshold"
        );
    }

    crate::metrics::HTTP_REQUEST_DURATION
        .with_label_values(&[&method, &normalized_path, &status])
        .observe(duration);

    crate::metrics::HTTP_REQUESTS_TOTAL
        .with_label_values(&[&method, &normalized_path, &status])
        .inc();

    Ok(response)
}

/// Normalize path to prevent metric cardinality explosion
/// /api/users/user123/memories -> /api/users/{id}/memories
fn normalize_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    let mut normalized = Vec::new();

    for part in parts {
        if part.is_empty() {
            continue;
        }

        // Replace UUIDs and IDs with placeholders
        if is_id(part) {
            normalized.push("{id}");
        } else {
            normalized.push(part);
        }
    }

    format!("/{}", normalized.join("/"))
}

/// Known path segments that should NEVER be treated as IDs (SHO-71)
const KNOWN_PATH_SEGMENTS: &[&str] = &[
    "v1",
    "v2",
    "v3",
    "api",
    "api2",
    "health",
    "metrics",
    "status",
    "docs",
    "swagger",
    "remember",
    "recall",
    "forget",
    "stats",
    "stream",
    "events",
    "settings",
    "config",
    "preferences",
    "notifications",
    "webhooks",
    "auth",
    "login",
    "logout",
    "register",
    "oauth",
    "token",
    "refresh",
    "list",
    "create",
    "update",
    "delete",
    "search",
    "query",
    "sync",
    "linear",
    "github",
    "import",
    "export",
    "memories",
    "context",
    "session",
    "working",
    "longterm",
    "consolidate",
    "maintenance",
    "repair",
    "rebuild",
    "graph",
    "edges",
    "entities",
    "reinforce",
    "coactivation",
    "introspection",
    "report",
    "consolidation",
    "learning",
    "tags",
    "date",
    "proactive",
    "verify",
    "index",
];

/// Check if a path segment looks like an ID (UUID, numeric, user ID, etc.) (SHO-71)
///
/// Improved detection to avoid false positives on legitimate path segments.
fn is_id(segment: &str) -> bool {
    // Never treat known path segments as IDs
    let lower = segment.to_lowercase();
    if KNOWN_PATH_SEGMENTS.contains(&lower.as_str()) {
        return false;
    }

    // UUID pattern: 8-4-4-4-12 hex chars with dashes (36 chars total)
    if segment.len() == 36 && segment.matches('-').count() == 4 {
        let parts: Vec<&str> = segment.split('-').collect();
        if parts.len() == 5
            && parts[0].len() == 8
            && parts[1].len() == 4
            && parts[2].len() == 4
            && parts[3].len() == 4
            && parts[4].len() == 12
            && parts
                .iter()
                .all(|p| p.chars().all(|c| c.is_ascii_hexdigit()))
        {
            return true;
        }
    }

    // Pure numeric ID (any length)
    if !segment.is_empty() && segment.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }

    // Hash-like strings: very long alphanumeric (>40 chars, like SHA256)
    if segment.len() > 40 && segment.chars().all(|c| c.is_ascii_alphanumeric()) {
        return true;
    }

    // ID with common prefixes: user_123, mem_abc, id-456
    let id_prefixes = [
        "user_", "user-", "mem_", "mem-", "id_", "id-", "uid_", "uid-", "drone_", "drone-",
        "robot_", "robot-", "session_", "session-", "mission_", "mission-",
    ];
    for prefix in id_prefixes {
        if lower.starts_with(prefix) {
            return true;
        }
    }

    // Short alphanumeric with majority digits (like "abc123", "x99")
    if segment.len() >= 3 && segment.len() <= 20 {
        let digit_count = segment.chars().filter(|c| c.is_ascii_digit()).count();
        let alpha_count = segment.chars().filter(|c| c.is_alphabetic()).count();
        // If >50% digits and has both letters and digits, it's likely an ID
        if digit_count > 0 && alpha_count > 0 && digit_count >= segment.len() / 2 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Process-global lock for tests that manipulate environment variables.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_normalize_path() {
        // User IDs should be normalized
        assert_eq!(
            normalize_path("/api/users/user123/memories"),
            "/api/users/{id}/memories"
        );
        // UUIDs should be normalized
        assert_eq!(
            normalize_path("/api/memories/550e8400-e29b-41d4-a716-446655440000"),
            "/api/memories/{id}"
        );
        // Health check should NOT be normalized
        assert_eq!(normalize_path("/health"), "/health");
        // Numeric IDs should be normalized
        assert_eq!(
            normalize_path("/api/users/12345/stats"),
            "/api/users/{id}/stats"
        );
    }

    #[test]
    fn test_known_paths_not_normalized() {
        // SHO-71: Known path segments should NOT be treated as IDs
        assert_eq!(
            normalize_path("/api/settings/notifications"),
            "/api/settings/notifications"
        );
        assert_eq!(normalize_path("/api/recall/tags"), "/api/recall/tags");
        assert_eq!(
            normalize_path("/api/consolidation/report"),
            "/api/consolidation/report"
        );
        assert_eq!(normalize_path("/api/v2/remember"), "/api/v2/remember");
    }

    #[test]
    fn test_uuid_detection() {
        // Valid UUID should be detected
        assert!(is_id("550e8400-e29b-41d4-a716-446655440000"));
        // Invalid UUID-like strings should not match
        assert!(!is_id("not-a-valid-uuid-at-all"));
    }

    // ── security_headers ──

    #[tokio::test]
    async fn security_headers_present_in_dev_mode() {
        let _guard = ENV_LOCK.lock().unwrap();
        use axum::body::Body;
        use axum::http::{Request as HttpRequest, StatusCode};
        use axum::middleware::from_fn;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        std::env::remove_var("SHODH_ENV");

        let app = Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(from_fn(security_headers));

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("X-Content-Type-Options").unwrap(),
            "nosniff"
        );
        assert_eq!(resp.headers().get("X-Frame-Options").unwrap(), "DENY");
        assert_eq!(
            resp.headers().get("Content-Security-Policy").unwrap(),
            "default-src 'none'"
        );
        assert_eq!(resp.headers().get("Cache-Control").unwrap(), "no-store");
        // HSTS should NOT be present in dev mode
        assert!(
            resp.headers().get("Strict-Transport-Security").is_none(),
            "HSTS should only be set in production"
        );
    }

    #[tokio::test]
    async fn security_headers_hsts_in_production() {
        let _guard = ENV_LOCK.lock().unwrap();
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::middleware::from_fn;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        std::env::set_var("SHODH_ENV", "production");

        let app = Router::new()
            .route("/test", get(|| async { "ok" }))
            .layer(from_fn(security_headers));

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();

        assert!(
            resp.headers().get("Strict-Transport-Security").is_some(),
            "HSTS should be set in production"
        );
        let hsts = resp
            .headers()
            .get("Strict-Transport-Security")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(hsts.contains("max-age="));

        std::env::remove_var("SHODH_ENV");
    }
}
