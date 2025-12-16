use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::env;

/// API Key authentication errors
#[derive(Debug)]
pub enum AuthError {
    MissingApiKey,
    InvalidApiKey,
    NotConfigured,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingApiKey => (StatusCode::UNAUTHORIZED, "Missing X-API-Key header"),
            AuthError::InvalidApiKey => (StatusCode::UNAUTHORIZED, "Invalid API key"),
            AuthError::NotConfigured => (
                StatusCode::SERVICE_UNAVAILABLE,
                "API keys not configured. Set SHODH_API_KEYS environment variable.",
            ),
        };

        (status, message).into_response()
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
    let mut result = (a_len ^ b_len) as u8;

    // Compare all bytes up to max_len, using 0 for out-of-bounds indices
    // This ensures constant time regardless of actual lengths
    for i in 0..max_len {
        let byte_a = if i < a_len { a_bytes[i] } else { 0 };
        let byte_b = if i < b_len { b_bytes[i] } else { 0 };
        result |= byte_a ^ byte_b;
    }

    result == 0
}

/// Validate API key against configured keys using constant-time comparison
pub fn validate_api_key(provided_key: &str) -> Result<(), AuthError> {
    // Get API keys from environment (comma-separated for multiple keys)
    let valid_keys = match env::var("SHODH_API_KEYS") {
        Ok(keys) if !keys.trim().is_empty() => keys,
        _ => {
            // In production, refuse to start without API keys
            let is_production = env::var("SHODH_ENV")
                .map(|v| v.to_lowercase() == "production" || v.to_lowercase() == "prod")
                .unwrap_or(false);

            if is_production {
                tracing::error!("SHODH_API_KEYS not set in production mode");
                return Err(AuthError::NotConfigured);
            }

            // Development mode: require SHODH_DEV_API_KEY to be explicitly set
            match env::var("SHODH_DEV_API_KEY") {
                Ok(key) if !key.trim().is_empty() => {
                    tracing::warn!("Using SHODH_DEV_API_KEY for development (not for production!)");
                    key
                }
                _ => {
                    tracing::error!(
                        "SHODH_API_KEYS not set. Set SHODH_API_KEYS for production or SHODH_DEV_API_KEY for development."
                    );
                    return Err(AuthError::NotConfigured);
                }
            }
        }
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

    // Extract and clone API key value (borrow ends after this expression)
    let api_key_value = match request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
    {
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

    #[test]
    fn test_validate_api_key() {
        env::set_var("SHODH_API_KEYS", "key1,key2,key3");

        assert!(validate_api_key("key1").is_ok());
        assert!(validate_api_key("key2").is_ok());
        assert!(validate_api_key("key3").is_ok());
        assert!(validate_api_key("invalid").is_err());

        env::remove_var("SHODH_API_KEYS");
    }
}
