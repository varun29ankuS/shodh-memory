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
            AuthError::MissingApiKey => (
                StatusCode::UNAUTHORIZED,
                "Missing X-API-Key header",
            ),
            AuthError::InvalidApiKey => (
                StatusCode::UNAUTHORIZED,
                "Invalid API key",
            ),
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
/// Note: This leaks the length of the shorter string, but that's acceptable
/// for API keys where lengths are not secret. For truly constant-time
/// comparison regardless of length, consider using a crypto library.
fn constant_time_compare(a: &str, b: &str) -> bool {
    // XOR all bytes including length difference indicator
    let mut result = (a.len() ^ b.len()) as u8;

    // Compare byte-by-byte using the shorter length
    // This prevents out-of-bounds but maintains timing consistency
    let min_len = std::cmp::min(a.len(), b.len());
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();

    for i in 0..min_len {
        result |= a_bytes[i] ^ b_bytes[i];
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

            // Development mode: warn but allow default key
            tracing::warn!("SHODH_API_KEYS not set - using development key (not for production!)");
            "shodh-dev-key-change-in-production".to_string()
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
pub async fn auth_middleware(
    request: Request,
    next: Next,
) -> Response {
    // Skip auth for health endpoint
    if request.uri().path() == "/health" {
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
