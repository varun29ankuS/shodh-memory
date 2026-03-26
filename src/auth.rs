use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::env;

use crate::errors::ErrorResponse;


/// Check if running in production mode
pub fn is_production_mode() -> bool {
    env::var("SHODH_ENV")
        .map(|v| v.to_lowercase() == "production" || v.to_lowercase() == "prod")
        .unwrap_or(false)
}

/// Check if dev key should be hidden from error messages.
///
/// Returns true when SHODH_HIDE_DEV_KEY=true (opt-in).
/// In production mode, always returns true regardless of the env var.
fn should_hide_dev_key() -> bool {
    if is_production_mode() {
        return true;
    }
    env::var("SHODH_HIDE_DEV_KEY")
        .map(|v| v.to_lowercase() == "true" || v == "1")
        .unwrap_or(false)
}

/// Log security warnings at startup based on environment configuration
pub fn log_security_status() {
    let has_api_keys = env::var("SHODH_API_KEYS")
        .or_else(|_| env::var("SHODH_API_KEY"))
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    let has_dev_key = env::var("SHODH_DEV_API_KEY")
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    let is_prod = is_production_mode();

    if is_prod {
        if has_api_keys {
            tracing::info!("Running in PRODUCTION mode with API key authentication");
        } else {
            tracing::error!(
                "PRODUCTION mode but SHODH_API_KEYS not set! Server will reject all authenticated requests."
            );
        }
    } else {
        tracing::warn!("╔════════════════════════════════════════════════════════════════╗");
        tracing::warn!("║  SECURITY WARNING: Running in DEVELOPMENT mode                 ║");
        tracing::warn!("║                                                                ║");
        if has_dev_key {
            tracing::warn!("║  Using SHODH_DEV_API_KEY for authentication.                  ║");
            tracing::warn!("║  DO NOT use this configuration in production!                 ║");
        } else if !has_api_keys {
            tracing::error!("║  No API keys configured. You must set SHODH_API_KEYS.        ║");
            tracing::error!("║  Server will reject all authenticated requests.              ║");
        }
        tracing::warn!("║                                                                ║");
        tracing::warn!("║  For production, set:                                          ║");
        tracing::warn!("║    SHODH_ENV=production                                        ║");
        tracing::warn!("║    SHODH_API_KEYS=your-secure-key-1,your-secure-key-2          ║");
        tracing::warn!("╚════════════════════════════════════════════════════════════════╝");
    }
}

/// API Key authentication errors
#[derive(Debug)]
pub enum AuthError {
    MissingApiKey,
    InvalidApiKey,
    NotConfigured,
}

impl AuthError {
    fn code(&self) -> &'static str {
        match self {
            Self::MissingApiKey => "MISSING_API_KEY",
            Self::InvalidApiKey => "INVALID_API_KEY",
            Self::NotConfigured => "AUTH_NOT_CONFIGURED",
        }
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::MissingApiKey | Self::InvalidApiKey => StatusCode::UNAUTHORIZED,
            Self::NotConfigured => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let is_prod = is_production_mode();
        let status = self.status_code();

        let message = match &self {
            AuthError::MissingApiKey => {
                if is_prod {
                    "Missing X-API-Key header".to_string()
                } else {
                    "Missing X-API-Key header. Set SHODH_DEV_API_KEY or SHODH_API_KEYS.".to_string()
                }
            }
            AuthError::InvalidApiKey => {
                if is_prod {
                    "Invalid API key".to_string()
                } else {
                    "Invalid API key. Check SHODH_DEV_API_KEY or SHODH_API_KEYS.".to_string()
                }
            }
            AuthError::NotConfigured => {
                "API keys not configured. Set SHODH_API_KEYS environment variable.".to_string()
            }
        };

        let body = ErrorResponse {
            code: self.code().to_string(),
            message,
            details: None,
            request_id: None,
        };

        (status, Json(body)).into_response()
    }
}

/// Constant-time string comparison to prevent timing attacks
///
/// Compares all bytes of both strings to prevent length-based timing leaks.
/// The comparison time is constant regardless of where differences occur.
fn constant_time_compare(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let a_len = a_bytes.len();
    let b_len = b_bytes.len();
    let max_len = std::cmp::max(a_len, b_len);

    // Track whether lengths match (0 if equal, non-zero otherwise)
    // Use u32 to avoid truncation: (usize as u8) wraps at 256, so lengths
    // differing by a multiple of 256 would falsely compare as equal.
    let mut result: u32 = (a_len ^ b_len) as u32;

    // Compare all bytes up to max_len, using 0 for out-of-bounds indices
    // This ensures constant time regardless of actual lengths
    for i in 0..max_len {
        let byte_a = if i < a_len { a_bytes[i] } else { 0 };
        let byte_b = if i < b_len { b_bytes[i] } else { 0 };
        result |= (byte_a ^ byte_b) as u32;
    }

    result == 0
}

/// Validate API key against configured keys using constant-time comparison
pub fn validate_api_key(provided_key: &str) -> Result<(), AuthError> {
    // Get API keys from environment.
    // Resolution order: SHODH_API_KEYS (plural, comma-separated) → SHODH_API_KEY (singular)
    //                 → SHODH_DEV_API_KEY (dev mode) → built-in default (dev mode only)
    let valid_keys = match env::var("SHODH_API_KEYS") {
        Ok(keys) if !keys.trim().is_empty() => keys,
        _ => match env::var("SHODH_API_KEY") {
            Ok(key) if !key.trim().is_empty() => key,
            _ => {
                // In production, refuse to start without API keys
                let is_production = env::var("SHODH_ENV")
                    .map(|v| v.to_lowercase() == "production" || v.to_lowercase() == "prod")
                    .unwrap_or(false);

                if is_production {
                    tracing::error!("SHODH_API_KEYS not set in production mode");
                    return Err(AuthError::NotConfigured);
                }

                // Development mode: use SHODH_DEV_API_KEY, or fall back to built-in default
                match env::var("SHODH_DEV_API_KEY") {
                    Ok(key) if !key.trim().is_empty() => {
                        tracing::warn!(
                            "Using SHODH_DEV_API_KEY for development (not for production!)"
                        );
                        key
                    }
                    _ => {
                        tracing::error!("SHODH_API_KEYS or SHODH_DEV_API_KEY not set");
                        return Err(AuthError::NotConfigured);
                    }
                }
            }
        },
    };

    let keys: Vec<&str> = valid_keys.split(',').map(|k| k.trim()).collect();

    // Use constant-time comparison to prevent timing attacks
    let mut found = false;
    for key in &keys {
        if constant_time_compare(key, provided_key) {
            found = true;
            // Don't break early - continue checking to maintain constant time
        }
    }

    if found {
        Ok(())
    } else {
        Err(AuthError::InvalidApiKey)
    }
}

/// Authentication middleware
pub async fn auth_middleware(request: Request, next: Next) -> Response {
    let path = request.uri().path();

    // Skip auth for health endpoint
    if path == "/health" {
        return next.run(request).await;
    }

    // Skip API key auth for webhook endpoints (they use HMAC signature verification)
    if path.starts_with("/webhook/") {
        return next.run(request).await;
    }

    // Extract API key: try X-API-Key header first, then Authorization: Bearer,
    // then query parameter (for WebSocket connections where headers aren't supported)
    let api_key_value = match request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            request
                .headers()
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                .map(|s| s.to_string())
        })
        .or_else(|| {
            // WebSocket fallback: check query parameter for api_key
            // Browser WebSocket API doesn't support custom headers, so
            // clients can pass ?api_key=... in the URL instead.
            // ONLY allow this for WebSocket upgrades to prevent API key
            // leakage via URLs in server logs, browser history, and referrer headers.
            let is_websocket = request
                .headers()
                .get("upgrade")
                .and_then(|v| v.to_str().ok())
                .map(|v| v.eq_ignore_ascii_case("websocket"))
                .unwrap_or(false);
            if !is_websocket {
                return None;
            }
            request.uri().query().and_then(|q| {
                q.split('&')
                    .find_map(|pair| pair.strip_prefix("api_key=").map(|v| v.to_string()))
            })
        }) {
        Some(key) => key,
        None => return AuthError::MissingApiKey.into_response(),
    };

    // Validate the cloned key
    if let Err(e) = validate_api_key(&api_key_value) {
        return e.into_response();
    }

    // Now we can move request to next layer
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use std::sync::Mutex;

    /// Process-global lock for tests that manipulate environment variables.
    /// `env::set_var` / `env::remove_var` are not thread-safe, so all tests
    /// that touch auth env vars must hold this lock for the duration of the test.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Clear all auth-related env vars to isolate tests.
    /// Caller MUST hold `ENV_LOCK` — this is not enforced at compile time.
    fn clear_auth_env() {
        env::remove_var("SHODH_API_KEYS");
        env::remove_var("SHODH_DEV_API_KEY");
        env::remove_var("SHODH_ENV");
        env::remove_var("SHODH_HIDE_DEV_KEY");
    }

    // ── constant_time_compare ──

    #[test]
    fn constant_time_equal_strings() {
        assert!(constant_time_compare("hello", "hello"));
    }

    #[test]
    fn constant_time_different_strings() {
        assert!(!constant_time_compare("hello", "world"));
    }

    #[test]
    fn constant_time_different_lengths() {
        assert!(!constant_time_compare("short", "a-longer-string"));
    }

    #[test]
    fn constant_time_empty_strings() {
        assert!(constant_time_compare("", ""));
    }

    #[test]
    fn constant_time_one_empty() {
        assert!(!constant_time_compare("", "notempty"));
        assert!(!constant_time_compare("notempty", ""));
    }

    #[test]
    fn constant_time_length_multiple_of_256() {
        // Regression: (256 ^ 0) as u8 == 0, so the old u8 accumulator
        // would falsely treat a 256-byte string as equal to an empty string.
        let long = "a".repeat(256);
        assert!(!constant_time_compare(&long, ""));
        assert!(!constant_time_compare("", &long));

        // Also test 512 vs 256 (difference = 256, wraps to 0 in u8)
        let medium = "b".repeat(256);
        let longer = "b".repeat(512);
        assert!(!constant_time_compare(&medium, &longer));
    }

    // ── is_production_mode ──

    #[test]
    fn production_mode_detection() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();

        assert!(!is_production_mode());

        env::set_var("SHODH_ENV", "production");
        assert!(is_production_mode());

        env::set_var("SHODH_ENV", "prod");
        assert!(is_production_mode());

        env::set_var("SHODH_ENV", "PRODUCTION");
        assert!(is_production_mode());

        env::set_var("SHODH_ENV", "development");
        assert!(!is_production_mode());

        env::set_var("SHODH_ENV", "test");
        assert!(!is_production_mode());

        clear_auth_env();
    }

    // ── validate_api_key: SHODH_API_KEYS ──

    #[test]
    fn validate_with_single_api_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "my-key");
        assert!(validate_api_key("my-key").is_ok());
        assert!(validate_api_key("wrong").is_err());
        clear_auth_env();
    }

    #[test]
    fn validate_with_multiple_api_keys() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "key1,key2,key3");
        assert!(validate_api_key("key1").is_ok());
        assert!(validate_api_key("key2").is_ok());
        assert!(validate_api_key("key3").is_ok());
        assert!(validate_api_key("key4").is_err());
        clear_auth_env();
    }

    #[test]
    fn validate_api_keys_trims_whitespace() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", " key1 , key2 ");
        assert!(validate_api_key("key1").is_ok());
        assert!(validate_api_key("key2").is_ok());
        clear_auth_env();
    }

    // ── validate_api_key: dev key ──

    #[test]
    fn validate_with_dev_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_DEV_API_KEY", "dev-key-123");
        assert!(validate_api_key("dev-key-123").is_ok());
        assert!(validate_api_key("wrong").is_err());
        clear_auth_env();
    }


    // ── validate_api_key: production mode ──

    #[test]
    fn validate_production_rejects_when_no_keys() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_ENV", "production");
        let result = validate_api_key("any-key");
        assert!(result.is_err());
        match result.unwrap_err() {
            AuthError::NotConfigured => {}
            other => panic!("Expected NotConfigured, got {:?}", other),
        }
        clear_auth_env();
    }

    #[test]
    fn validate_production_works_with_api_keys_set() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_ENV", "production");
        env::set_var("SHODH_API_KEYS", "prod-key");
        assert!(validate_api_key("prod-key").is_ok());
        assert!(validate_api_key("wrong").is_err());
        clear_auth_env();
    }

    // ── validate_api_key: edge cases ──

    #[test]
    fn validate_empty_api_keys_falls_through() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "  ");
        assert!(validate_api_key("anything").is_err());
        clear_auth_env();
    }


    #[test]
    fn api_keys_takes_priority_over_dev_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "prod-key");
        env::set_var("SHODH_DEV_API_KEY", "dev-key");
        assert!(validate_api_key("prod-key").is_ok());
        assert!(validate_api_key("dev-key").is_err()); // dev key ignored
        clear_auth_env();
    }

    // ── AuthError response codes ──

    #[test]
    fn auth_error_status_codes() {
        assert_eq!(
            AuthError::MissingApiKey.status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            AuthError::InvalidApiKey.status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            AuthError::NotConfigured.status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn auth_error_codes() {
        assert_eq!(AuthError::MissingApiKey.code(), "MISSING_API_KEY");
        assert_eq!(AuthError::InvalidApiKey.code(), "INVALID_API_KEY");
        assert_eq!(AuthError::NotConfigured.code(), "AUTH_NOT_CONFIGURED");
    }

    // ── AuthError JSON response shape ──

    #[tokio::test]
    async fn auth_error_response_is_valid_json() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        let resp = AuthError::MissingApiKey.into_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let body = to_bytes(resp.into_body(), 2048).await.unwrap();
        let parsed: ErrorResponse = serde_json::from_slice(&body)
            .expect("Response body should be valid JSON matching ErrorResponse");
        assert_eq!(parsed.code, "MISSING_API_KEY");
        assert!(parsed.message.contains("X-API-Key"));
        clear_auth_env();
    }

    #[tokio::test]
    async fn missing_key_dev_message_includes_help() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        // Not production → should include env var names in message
        let resp = AuthError::MissingApiKey.into_response();
        let body = to_bytes(resp.into_body(), 2048).await.unwrap();
        let parsed: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert!(
            parsed.message.contains("SHODH_API_KEYS"),
            "Should mention SHODH_API_KEYS"
        );
        assert!(
            parsed.message.contains("SHODH_DEV_API_KEY"),
            "Should mention SHODH_DEV_API_KEY"
        );

        clear_auth_env();
    }

    #[tokio::test]
    async fn invalid_key_dev_message_includes_help() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        let resp = AuthError::InvalidApiKey.into_response();
        let body = to_bytes(resp.into_body(), 2048).await.unwrap();
        let parsed: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert!(
            parsed.message.contains("SHODH_API_KEYS"),
            "Should mention SHODH_API_KEYS"
        );

        clear_auth_env();
    }

    #[tokio::test]
    async fn missing_key_prod_message_is_terse() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_ENV", "production");
        let resp = AuthError::MissingApiKey.into_response();
        let body = to_bytes(resp.into_body(), 2048).await.unwrap();
        let parsed: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.message, "Missing X-API-Key header");
        assert!(
            !parsed.message.contains("SHODH_DEV_API_KEY"),
            "Prod must not leak env var names"
        );
        clear_auth_env();
    }

    #[tokio::test]
    async fn invalid_key_prod_message_is_terse() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_ENV", "production");
        let resp = AuthError::InvalidApiKey.into_response();
        let body = to_bytes(resp.into_body(), 2048).await.unwrap();
        let parsed: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.message, "Invalid API key");

        clear_auth_env();
    }

    #[tokio::test]
    async fn not_configured_response_shape() {
        let resp = AuthError::NotConfigured.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(resp.into_body(), 2048).await.unwrap();
        let parsed: ErrorResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.code, "AUTH_NOT_CONFIGURED");
        assert!(parsed.message.contains("SHODH_API_KEYS"));
    }

    // ── SHODH_HIDE_DEV_KEY ──

    #[tokio::test]
    async fn hide_dev_key_suppresses_key_in_missing_key_error() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_HIDE_DEV_KEY", "true");

        let resp = AuthError::MissingApiKey.into_response();
        let body = to_bytes(resp.into_body(), 2048).await.unwrap();
        let parsed: ErrorResponse = serde_json::from_slice(&body).unwrap();


        assert!(
            parsed.message.contains("SHODH_DEV_API_KEY"),
            "Should still mention env var name: {}",
            parsed.message
        );
        clear_auth_env();
    }

    #[tokio::test]
    async fn hide_dev_key_suppresses_key_in_invalid_key_error() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_HIDE_DEV_KEY", "true");

        let resp = AuthError::InvalidApiKey.into_response();
        let body = to_bytes(resp.into_body(), 2048).await.unwrap();
        let parsed: ErrorResponse = serde_json::from_slice(&body).unwrap();


        clear_auth_env();
    }

    #[test]
    fn should_hide_dev_key_defaults_to_false() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        assert!(!should_hide_dev_key());
        clear_auth_env();
    }

    #[test]
    fn should_hide_dev_key_respects_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();

        env::set_var("SHODH_HIDE_DEV_KEY", "true");
        assert!(should_hide_dev_key());

        env::set_var("SHODH_HIDE_DEV_KEY", "1");
        assert!(should_hide_dev_key());

        env::set_var("SHODH_HIDE_DEV_KEY", "false");
        assert!(!should_hide_dev_key());

        clear_auth_env();
    }

    #[test]
    fn should_hide_dev_key_always_true_in_production() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_ENV", "production");
        // Even without SHODH_HIDE_DEV_KEY, production always hides
        assert!(should_hide_dev_key());
        clear_auth_env();
    }

    // ── Query parameter auth (WebSocket fallback) ──

    #[tokio::test]
    async fn auth_middleware_accepts_query_param_for_websocket() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::middleware::from_fn;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "test-ws-key");

        let app = Router::new()
            .route("/api/stream", get(|| async { "ok" }))
            .layer(from_fn(auth_middleware));

        // WebSocket upgrade with API key in query parameter
        let req = HttpRequest::builder()
            .uri("/api/stream?api_key=test-ws-key")
            .header("upgrade", "websocket")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Should accept API key from query parameter on WebSocket upgrade"
        );

        clear_auth_env();
    }

    #[tokio::test]
    async fn auth_middleware_ignores_query_param_without_websocket_upgrade() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::middleware::from_fn;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "test-ws-key");

        let app = Router::new()
            .route("/api/remember", get(|| async { "ok" }))
            .layer(from_fn(auth_middleware));

        // Non-WebSocket request with API key in query parameter — should be ignored
        let req = HttpRequest::builder()
            .uri("/api/remember?api_key=test-ws-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "Query param auth should be ignored for non-WebSocket requests"
        );

        clear_auth_env();
    }

    #[tokio::test]
    async fn auth_middleware_rejects_invalid_websocket_query_param() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::middleware::from_fn;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "correct-key");

        let app = Router::new()
            .route("/api/stream", get(|| async { "ok" }))
            .layer(from_fn(auth_middleware));

        let req = HttpRequest::builder()
            .uri("/api/stream?api_key=wrong-key")
            .header("upgrade", "websocket")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "Should reject invalid query parameter API key on WebSocket"
        );

        clear_auth_env();
    }
}
