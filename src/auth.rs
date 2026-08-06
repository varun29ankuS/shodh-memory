use axum::{
    extract::{MatchedPath, Request},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::env;
use subtle::ConstantTimeEq;

use crate::errors::ErrorResponse;

/// The identity an authenticated API key is bound to.
///
/// Security posture by variant:
/// - [`AuthIdentity::Unscoped`] — legacy single-tenant posture. The key is
///   root over EVERY `user_id` on this server. All keys configured through
///   `SHODH_API_KEYS`, `SHODH_API_KEY`, and `SHODH_DEV_API_KEY` resolve to
///   this identity, which preserves the exact pre-scoping behavior for
///   existing deployments (MCP server, hooks, TUI all authenticate with a
///   shared unscoped key).
/// - [`AuthIdentity::User`] — multi-tenant posture, opt-in via
///   `SHODH_SCOPED_API_KEYS` (comma-separated `user_id:key` entries). The key
///   may only act as that one user: any request that names a different
///   `user_id` — in a JSON body, a query string, or a `{user_id}` path
///   segment — is rejected with 403 by [`scope_enforcement_middleware`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthIdentity {
    /// Full access to every user's store (legacy/default posture).
    Unscoped,
    /// Access restricted to exactly this `user_id`.
    User(String),
}

/// Check if running in production mode
pub fn is_production_mode() -> bool {
    env::var("SHODH_ENV")
        .map(|v| v.to_lowercase() == "production" || v.to_lowercase() == "prod")
        .unwrap_or(false)
}

/// Log security warnings at startup based on environment configuration
pub fn log_security_status() {
    let has_api_keys = env::var("SHODH_API_KEYS")
        .or_else(|_| env::var("SHODH_API_KEY"))
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    let has_scoped_keys = env::var("SHODH_SCOPED_API_KEYS")
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    let has_dev_key = env::var("SHODH_DEV_API_KEY")
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    let is_prod = is_production_mode();

    if has_scoped_keys {
        tracing::info!(
            "User-scoped API keys configured (SHODH_SCOPED_API_KEYS): scoped keys are \
             restricted to their bound user_id; unscoped keys retain full access"
        );
    } else if has_api_keys {
        tracing::warn!(
            "All configured API keys are UNSCOPED: any valid key can read and write \
             every user's memories. For multi-tenant deployments, bind keys to users \
             with SHODH_SCOPED_API_KEYS=user_id:key[,user_id:key...]"
        );
    }

    if is_prod {
        if has_api_keys || has_scoped_keys {
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
        } else if !has_api_keys && !has_scoped_keys {
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
    /// The key authenticated successfully but is scoped to a different
    /// `user_id` than the one the request names. 403, not 401: the caller IS
    /// authenticated — they are simply not authorized for that user's data.
    UserScopeForbidden,
}

impl AuthError {
    fn code(&self) -> &'static str {
        match self {
            Self::MissingApiKey => "MISSING_API_KEY",
            Self::InvalidApiKey => "INVALID_API_KEY",
            Self::NotConfigured => "AUTH_NOT_CONFIGURED",
            Self::UserScopeForbidden => "API_KEY_SCOPE_FORBIDDEN",
        }
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::MissingApiKey | Self::InvalidApiKey => StatusCode::UNAUTHORIZED,
            Self::NotConfigured => StatusCode::SERVICE_UNAVAILABLE,
            Self::UserScopeForbidden => StatusCode::FORBIDDEN,
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
            // Same message in production and development: it names no
            // configuration internals and is directly actionable.
            AuthError::UserScopeForbidden => {
                "API key is not authorized for the requested user_id".to_string()
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

/// Compiler-resistant constant-time comparison for equal-length key material.
pub(crate) fn constant_time_compare(a: &str, b: &str) -> bool {
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

/// Build the full key table — every configured key paired with the identity
/// it is bound to — using the one precedence rule shared by HTTP and local IPC.
///
/// Scoped entries come from `SHODH_SCOPED_API_KEYS` (comma-separated
/// `user_id:key`; the first `:` separates the two, so keys may contain `:`
/// but user IDs may not — `validate_user_id` already forbids it). Malformed
/// entries are logged and skipped, which fails closed: a typo'd key simply
/// never authenticates.
///
/// Unscoped (legacy, full-access) entries keep the exact historical
/// resolution order: SHODH_API_KEYS (plural, comma-separated) →
/// SHODH_API_KEY (singular) → SHODH_DEV_API_KEY (dev mode only).
///
/// Scoped entries are placed FIRST so that a key mistakenly configured in
/// both lists resolves to its scoped (least-privilege) identity.
pub(crate) fn configured_key_table() -> Result<Vec<(String, AuthIdentity)>, AuthError> {
    let mut table: Vec<(String, AuthIdentity)> = Vec::new();
    let mut any_source_present = false;

    if let Ok(scoped) = env::var("SHODH_SCOPED_API_KEYS") {
        if !scoped.trim().is_empty() {
            any_source_present = true;
            for entry in scoped.split(',').map(str::trim).filter(|e| !e.is_empty()) {
                let Some((user_id, key)) = entry.split_once(':') else {
                    tracing::error!(
                        "SHODH_SCOPED_API_KEYS entry without ':' separator ignored \
                         (expected user_id:key)"
                    );
                    continue;
                };
                let user_id = user_id.trim();
                let key = key.trim();
                if key.is_empty() {
                    tracing::error!("SHODH_SCOPED_API_KEYS entry with empty key ignored");
                    continue;
                }
                if let Err(error) = crate::validation::validate_user_id(user_id) {
                    tracing::error!(
                        "SHODH_SCOPED_API_KEYS entry with invalid user_id ignored: {error}"
                    );
                    continue;
                }
                table.push((key.to_owned(), AuthIdentity::User(user_id.to_owned())));
            }
        }
    }

    // Legacy unscoped keys. Every key from these sources is root over every
    // user — this is the documented single-tenant default and MUST stay the
    // behavior for existing deployments (MCP server, hooks, TUI).
    let unscoped_keys = match env::var("SHODH_API_KEYS") {
        Ok(keys) if !keys.trim().is_empty() => Some(keys),
        _ => match env::var("SHODH_API_KEY") {
            Ok(key) if !key.trim().is_empty() => Some(key),
            _ => {
                if is_production_mode() {
                    // Production never falls back to the dev key. Scoped keys
                    // alone are a valid production configuration.
                    None
                } else {
                    // Development mode: use SHODH_DEV_API_KEY when present.
                    match env::var("SHODH_DEV_API_KEY") {
                        Ok(key) if !key.trim().is_empty() => {
                            tracing::warn!(
                                "Using SHODH_DEV_API_KEY for development (not for production!)"
                            );
                            Some(key)
                        }
                        _ => None,
                    }
                }
            }
        },
    };
    if let Some(keys) = unscoped_keys {
        any_source_present = true;
        for key in keys.split(',').map(str::trim).filter(|key| !key.is_empty()) {
            table.push((key.to_owned(), AuthIdentity::Unscoped));
        }
    }

    if !any_source_present {
        if is_production_mode() {
            tracing::error!("SHODH_API_KEYS not set in production mode");
        } else {
            tracing::error!("SHODH_API_KEYS, SHODH_SCOPED_API_KEYS, or SHODH_DEV_API_KEY not set");
        }
        return Err(AuthError::NotConfigured);
    }

    // A source was present but yielded zero usable keys (e.g. only commas or
    // malformed scoped entries). Return the empty table so validation fails
    // with InvalidApiKey — the same observable behavior as before scoping.
    Ok(table)
}

/// Resolve every configured key (scoped and unscoped) without identities.
/// Used by local IPC probe signing, where health-probe responses are signed
/// with all keys so any legitimate client can verify them.
pub(crate) fn configured_api_keys() -> Result<Vec<String>, AuthError> {
    Ok(configured_key_table()?
        .into_iter()
        .map(|(key, _)| key)
        .collect())
}

/// Resolve an API key to the identity it is bound to, using constant-time
/// comparison against every configured key.
pub fn resolve_api_key(provided_key: &str) -> Result<AuthIdentity, AuthError> {
    let table = configured_key_table()?;

    let mut matched: Option<&AuthIdentity> = None;
    for (key, identity) in &table {
        // Don't break early - continue checking to maintain constant time.
        // First match wins; scoped entries precede unscoped ones in the
        // table, so a duplicate key resolves to least privilege.
        if constant_time_compare(key, provided_key) && matched.is_none() {
            matched = Some(identity);
        }
    }

    matched.cloned().ok_or(AuthError::InvalidApiKey)
}

/// Validate API key against configured keys using constant-time comparison.
///
/// Answers only "is this key valid" — callers that go on to touch per-user
/// data must use [`resolve_api_key`] and enforce the returned identity.
pub fn validate_api_key(provided_key: &str) -> Result<(), AuthError> {
    resolve_api_key(provided_key).map(|_| ())
}

/// Authentication middleware
pub async fn auth_middleware(mut request: Request, next: Next) -> Response {
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
            // Query-parameter fallback: browsers cannot attach custom headers to
            // WebSocket (`new WebSocket`) or Server-Sent Events (`new EventSource`)
            // connections, so those clients pass ?api_key=... in the URL instead.
            // Restricted to those two channels to limit API-key leakage via URLs in
            // server logs, browser history, and referrer headers.
            let is_websocket = request
                .headers()
                .get("upgrade")
                .and_then(|v| v.to_str().ok())
                .map(|v| v.eq_ignore_ascii_case("websocket"))
                .unwrap_or(false);
            let is_event_stream = path == "/api/events" || path == "/api/events/sse";
            if !is_websocket && !is_event_stream {
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

    // Resolve the key to the identity it is bound to. The identity travels
    // with the request so that scope_enforcement_middleware (and the
    // WebSocket handshake handlers) can authorize against it.
    let identity = match resolve_api_key(&api_key_value) {
        Ok(identity) => identity,
        Err(e) => return e.into_response(),
    };
    request.extensions_mut().insert(identity);

    // Now we can move request to next layer
    next.run(request).await
}

/// Maximum request-body size the scope-enforcement middleware will buffer for
/// inspection. Matches axum's default body-extraction limit, so a body that
/// exceeds it would have been rejected by the handler's `Json` extractor too.
const MAX_SCANNED_BODY_BYTES: usize = 2 * 1024 * 1024;

/// Decode `%XX` escapes in a path segment. Invalid escape sequences are kept
/// literally — a segment the handler's own `Path` extraction cannot decode
/// will not match any real user either way.
fn percent_decode_path_segment(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = &segment[i + 1..i + 3];
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| segment.to_string())
}

/// Enforce the authenticated identity's user scope against the `user_id` the
/// request names — in the matched route's `{user_id}` path segment, in the
/// query string (the SSE/EventSource and WebSocket channels pass both
/// `api_key` and `user_id` as query parameters), and as the top-level
/// `user_id` field of a JSON body. Every user-scoped route in the router
/// takes `user_id` through exactly one of those three channels, so this
/// single choke point covers the whole route table without per-handler
/// changes. The two WebSocket routes additionally carry `user_id` inside
/// their post-upgrade handshake message; their handlers read the same
/// [`AuthIdentity`] extension and enforce the scope at handshake time.
///
/// Security posture:
/// - [`AuthIdentity::Unscoped`] (legacy keys) passes through untouched — the
///   request body is not even buffered, so single-tenant deployments keep
///   their exact pre-scoping behavior and cost.
/// - [`AuthIdentity::User`] requests naming any other user are rejected with
///   403 (`API_KEY_SCOPE_FORBIDDEN`).
/// - Requests with no identity extension pass through: on HTTP they cannot
///   exist behind `auth_middleware` (which always inserts one), and on the
///   local IPC router they are health probes or public routes that carry no
///   per-user data.
pub async fn scope_enforcement_middleware(request: Request, next: Next) -> Response {
    let scoped_user = match request.extensions().get::<AuthIdentity>() {
        Some(AuthIdentity::User(user_id)) => user_id.clone(),
        Some(AuthIdentity::Unscoped) | None => return next.run(request).await,
    };

    if let Some(matched) = request.extensions().get::<MatchedPath>() {
        // GET /api/users enumerates every user_id on the server — cross-tenant
        // by construction, so scoped keys are denied outright.
        if matched.as_str() == "/api/users" {
            return AuthError::UserScopeForbidden.into_response();
        }

        // A `{user_id}` path segment must name the key's own user.
        if let Some(position) = matched
            .as_str()
            .split('/')
            .position(|segment| segment == "{user_id}")
        {
            let actual = request.uri().path().split('/').nth(position).unwrap_or("");
            if percent_decode_path_segment(actual) != scoped_user {
                return AuthError::UserScopeForbidden.into_response();
            }
        }
    }

    // Query-string `user_id` (SSE event stream, GET list/stats endpoints).
    if let Some(query) = request.uri().query() {
        for (name, value) in form_urlencoded::parse(query.as_bytes()) {
            if name == "user_id" && value != scoped_user {
                return AuthError::UserScopeForbidden.into_response();
            }
        }
    }

    // Top-level `user_id` in a JSON body. The body is buffered, inspected,
    // and reattached byte-for-byte, so handlers see exactly what the client
    // sent. Non-JSON and non-object bodies pass through untouched — the
    // handlers' own extractors reject anything they cannot parse, and no
    // handler reads `user_id` from a non-JSON body.
    let (parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, MAX_SCANNED_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            let body = ErrorResponse {
                code: "PAYLOAD_TOO_LARGE".to_string(),
                message: format!(
                    "Request body exceeds the {} byte limit",
                    MAX_SCANNED_BODY_BYTES
                ),
                details: None,
                request_id: None,
            };
            return (StatusCode::PAYLOAD_TOO_LARGE, Json(body)).into_response();
        }
    };
    if !bytes.is_empty() {
        if let Ok(serde_json::Value::Object(object)) =
            serde_json::from_slice::<serde_json::Value>(&bytes)
        {
            if let Some(serde_json::Value::String(user_id)) = object.get("user_id") {
                if *user_id != scoped_user {
                    return AuthError::UserScopeForbidden.into_response();
                }
            }
        }
    }
    let request = Request::from_parts(parts, axum::body::Body::from(bytes));

    next.run(request).await
}

/// Process-global lock for tests that manipulate environment variables.
/// `env::set_var` / `env::remove_var` are not thread-safe, so every test across the
/// crate that touches auth env vars must hold this one lock for its duration. It is
/// `pub(crate)` so the `local_ipc` auth tests serialize against auth.rs's own env
/// tests rather than racing them.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    /// Clear all auth-related env vars to isolate tests.
    /// Caller MUST hold `ENV_LOCK` — this is not enforced at compile time.
    fn clear_auth_env() {
        env::remove_var("SHODH_API_KEYS");
        env::remove_var("SHODH_API_KEY");
        env::remove_var("SHODH_SCOPED_API_KEYS");
        env::remove_var("SHODH_DEV_API_KEY");
        env::remove_var("SHODH_ENV");
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
        // Scope violations are 403, not 401: the caller is authenticated but
        // not authorized for the named user.
        assert_eq!(
            AuthError::UserScopeForbidden.status_code(),
            StatusCode::FORBIDDEN
        );
    }

    #[test]
    fn auth_error_codes() {
        assert_eq!(AuthError::MissingApiKey.code(), "MISSING_API_KEY");
        assert_eq!(AuthError::InvalidApiKey.code(), "INVALID_API_KEY");
        assert_eq!(AuthError::NotConfigured.code(), "AUTH_NOT_CONFIGURED");
        assert_eq!(
            AuthError::UserScopeForbidden.code(),
            "API_KEY_SCOPE_FORBIDDEN"
        );
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
    async fn auth_middleware_accepts_query_param_for_event_stream() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::middleware::from_fn;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "test-sse-key");

        let app = Router::new()
            .route("/api/events", get(|| async { "ok" }))
            .layer(from_fn(auth_middleware));

        // SSE EventSource cannot set headers, so the key arrives as a query parameter.
        let req = HttpRequest::builder()
            .uri("/api/events?user_id=u&api_key=test-sse-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Should accept API key from query parameter on the SSE event stream"
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

    // ── Key → identity binding (SHODH_SCOPED_API_KEYS) ──

    #[test]
    fn resolve_scoped_key_returns_bound_identity() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "admin-key");
        env::set_var("SHODH_SCOPED_API_KEYS", "alice:alice-key, bob:bob-key");

        assert_eq!(
            resolve_api_key("alice-key").unwrap(),
            AuthIdentity::User("alice".to_string())
        );
        assert_eq!(
            resolve_api_key("bob-key").unwrap(),
            AuthIdentity::User("bob".to_string())
        );
        assert_eq!(
            resolve_api_key("admin-key").unwrap(),
            AuthIdentity::Unscoped
        );
        assert!(matches!(
            resolve_api_key("unknown"),
            Err(AuthError::InvalidApiKey)
        ));

        clear_auth_env();
    }

    #[test]
    fn scoped_only_config_is_valid_in_production() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_ENV", "production");
        env::set_var("SHODH_SCOPED_API_KEYS", "alice:alice-key");

        assert_eq!(
            resolve_api_key("alice-key").unwrap(),
            AuthIdentity::User("alice".to_string())
        );
        // A wrong key is InvalidApiKey (401), not NotConfigured (503).
        assert!(matches!(
            resolve_api_key("wrong"),
            Err(AuthError::InvalidApiKey)
        ));

        clear_auth_env();
    }

    #[test]
    fn malformed_scoped_entries_are_skipped() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        // no separator / invalid user_id / empty user_id / empty key / valid
        env::set_var(
            "SHODH_SCOPED_API_KEYS",
            "noseparator,bad/uid:key1,:key2,carol:,carol:carol-key",
        );

        assert_eq!(
            resolve_api_key("carol-key").unwrap(),
            AuthIdentity::User("carol".to_string())
        );
        assert!(resolve_api_key("noseparator").is_err());
        assert!(resolve_api_key("key1").is_err());
        assert!(resolve_api_key("key2").is_err());

        clear_auth_env();
    }

    #[test]
    fn duplicate_key_resolves_to_least_privilege() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        // Same key configured both scoped and unscoped — a config mistake.
        // The scoped (least-privilege) identity must win.
        env::set_var("SHODH_API_KEYS", "shared-key");
        env::set_var("SHODH_SCOPED_API_KEYS", "alice:shared-key");

        assert_eq!(
            resolve_api_key("shared-key").unwrap(),
            AuthIdentity::User("alice".to_string())
        );

        clear_auth_env();
    }

    #[test]
    fn scoped_keys_included_in_configured_api_keys() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        env::set_var("SHODH_API_KEYS", "admin-key");
        env::set_var("SHODH_SCOPED_API_KEYS", "alice:alice-key");

        let keys = configured_api_keys().unwrap();
        assert!(keys.contains(&"alice-key".to_string()));
        assert!(keys.contains(&"admin-key".to_string()));

        clear_auth_env();
    }

    // ── Scope enforcement middleware ──

    /// Build a router with the production middleware stack: auth (outer)
    /// resolves the key and attaches its identity, scope enforcement (inner)
    /// authorizes the request's user_id against it.
    fn scoped_test_app(routes: axum::Router) -> axum::Router {
        use axum::middleware::from_fn;
        routes
            .layer(from_fn(scope_enforcement_middleware))
            .layer(from_fn(auth_middleware))
    }

    /// Set up the canonical two-key environment: one unscoped legacy key and
    /// one key scoped to user "alice". Caller must hold ENV_LOCK.
    fn set_scoped_env() {
        env::set_var("SHODH_API_KEYS", "admin-key");
        env::set_var("SHODH_SCOPED_API_KEYS", "alice:alice-key");
    }

    /// Drive an async test body from sync code so the ENV_LOCK guard is never
    /// held across an await point inside an async context
    /// (clippy::await_holding_lock).
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(future)
    }

    #[test]
    fn scoped_key_rejected_when_body_names_other_user() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        set_scoped_env();

        block_on(async {
            let app = scoped_test_app(Router::new().route("/api/recall", post(|| async { "ok" })));

            // Scoped key naming ANOTHER user's store: must be 403.
            let req = HttpRequest::builder()
                .method("POST")
                .uri("/api/recall")
                .header("X-API-Key", "alice-key")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"user_id":"bob","query":"secrets"}"#))
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::FORBIDDEN,
                "A scoped key must not read another user's memories"
            );
            let body = to_bytes(resp.into_body(), 2048).await.unwrap();
            let parsed: ErrorResponse = serde_json::from_slice(&body).unwrap();
            assert_eq!(parsed.code, "API_KEY_SCOPE_FORBIDDEN");

            // Same key naming its OWN user: allowed.
            let req = HttpRequest::builder()
                .method("POST")
                .uri("/api/recall")
                .header("X-API-Key", "alice-key")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"user_id":"alice","query":"mine"}"#))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        });

        clear_auth_env();
    }

    #[test]
    fn unscoped_legacy_key_keeps_full_access() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();

        // Legacy single-tenant config: only SHODH_API_KEYS, no scoped keys.
        env::set_var("SHODH_API_KEYS", "admin-key");
        block_on(async {
            let app = scoped_test_app(Router::new().route("/api/recall", post(|| async { "ok" })));
            let req = HttpRequest::builder()
                .method("POST")
                .uri("/api/recall")
                .header("X-API-Key", "admin-key")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"user_id":"anyone","query":"q"}"#))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "Legacy unscoped keys must keep full access (backward compatibility)"
            );
        });

        // Mixed config: the unscoped key still has full access even when
        // scoped keys exist for other users.
        env::set_var("SHODH_SCOPED_API_KEYS", "alice:alice-key");
        block_on(async {
            let app = scoped_test_app(Router::new().route("/api/recall", post(|| async { "ok" })));
            let req = HttpRequest::builder()
                .method("POST")
                .uri("/api/recall")
                .header("X-API-Key", "admin-key")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"user_id":"alice","query":"q"}"#))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        });

        clear_auth_env();
    }

    #[test]
    fn scoped_key_sse_query_param_enforced() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        set_scoped_env();

        block_on(async {
            let app = scoped_test_app(Router::new().route("/api/events", get(|| async { "ok" })));

            // EventSource passes both api_key and user_id as query parameters —
            // the same flaw surface as the JSON body, enforced the same way.
            let req = HttpRequest::builder()
                .uri("/api/events?user_id=bob&api_key=alice-key")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::FORBIDDEN,
                "A scoped key must not subscribe to another user's event stream"
            );

            let req = HttpRequest::builder()
                .uri("/api/events?user_id=alice&api_key=alice-key")
                .body(Body::empty())
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        });

        clear_auth_env();
    }

    #[test]
    fn scoped_key_path_param_enforced() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        set_scoped_env();

        block_on(async {
            let app =
                scoped_test_app(Router::new().route("/api/list/{user_id}", get(|| async { "ok" })));

            let req = HttpRequest::builder()
                .uri("/api/list/bob")
                .header("X-API-Key", "alice-key")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::FORBIDDEN,
                "A scoped key must not list another user's memories via the path"
            );

            let req = HttpRequest::builder()
                .uri("/api/list/alice")
                .header("X-API-Key", "alice-key")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);

            // Percent-encoded segments must not smuggle a different user past
            // the comparison: %62%6f%62 decodes to "bob".
            let req = HttpRequest::builder()
                .uri("/api/list/%62%6f%62")
                .header("X-API-Key", "alice-key")
                .body(Body::empty())
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        });

        clear_auth_env();
    }

    #[test]
    fn scoped_key_cannot_enumerate_users() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::get;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        set_scoped_env();

        block_on(async {
            let app = scoped_test_app(Router::new().route("/api/users", get(|| async { "ok" })));

            // GET /api/users enumerates every user_id — cross-tenant metadata.
            let req = HttpRequest::builder()
                .uri("/api/users")
                .header("X-API-Key", "alice-key")
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::FORBIDDEN);

            // The unscoped admin key retains access.
            let req = HttpRequest::builder()
                .uri("/api/users")
                .header("X-API-Key", "admin-key")
                .body(Body::empty())
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        });

        clear_auth_env();
    }

    #[test]
    fn scoped_enforcement_passes_body_through_byte_identical() {
        use axum::body::{Body, Bytes};
        use axum::http::Request as HttpRequest;
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        let _guard = ENV_LOCK.lock().unwrap();
        clear_auth_env();
        set_scoped_env();

        block_on(async {
            // Echo handler: proves the buffered-and-reattached body reaches
            // the handler exactly as the client sent it (whitespace, ordering,
            // and non-ASCII included).
            let app = scoped_test_app(
                Router::new().route("/api/echo", post(|body: Bytes| async move { body })),
            );

            let payload = "{ \"user_id\": \"alice\",  \"query\": \"café ↦ résumé\" }";
            let req = HttpRequest::builder()
                .method("POST")
                .uri("/api/echo")
                .header("X-API-Key", "alice-key")
                .header("content-type", "application/json")
                .body(Body::from(payload))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let body = to_bytes(resp.into_body(), 4096).await.unwrap();
            assert_eq!(&body[..], payload.as_bytes());
        });

        clear_auth_env();
    }
}
