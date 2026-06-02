//! External integrations for syncing data sources to Shodh memory
//!
//! Supports:
//! - Linear: Issue tracking webhooks and bulk sync
//! - GitHub: PR/Issue webhooks and bulk sync

pub mod github;
pub mod linear;

pub use github::{GitHubSyncRequest, GitHubSyncResponse, GitHubWebhook, GitHubWebhookPayload};
pub use linear::{LinearSyncRequest, LinearSyncResponse, LinearWebhook, LinearWebhookPayload};

/// Returns `true` when `url` uses HTTP over a non-localhost host.
pub fn is_insecure_remote_url(url: &str) -> bool {
    if !url.starts_with("http://") {
        return false;
    }
    let rest = &url["http://".len()..];
    let authority = rest.split('/').next().unwrap_or(rest);
    let host_and_port = authority
        .split_once('@')
        .map_or(authority, |(_, after)| after);
    let host = if host_and_port.starts_with('[') {
        host_and_port
            .trim_start_matches('[')
            .split(']')
            .next()
            .unwrap_or(host_and_port)
    } else {
        host_and_port
            .rsplit_once(':')
            .map_or(host_and_port, |(h, _)| h)
    };
    !matches!(host, "127.0.0.1" | "localhost" | "::1" | "0.0.0.0")
}

/// Resolve an operator-supplied API URL override.
///
/// If `SHODH_ENFORCE_HTTPS=true`, insecure non-localhost HTTP overrides are
/// rejected and `default` is used. Otherwise they are accepted with a warning.
pub fn resolve_api_url_override(env_var: &str, default: &str) -> String {
    let candidate = std::env::var(env_var).unwrap_or_default();
    if candidate.trim().is_empty() {
        return default.to_string();
    }
    if is_insecure_remote_url(&candidate) {
        let enforce = std::env::var("SHODH_ENFORCE_HTTPS")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);
        if enforce {
            tracing::warn!(
                "{env_var} override rejected (SHODH_ENFORCE_HTTPS=true): insecure HTTP for non-localhost host; using default"
            );
            return default.to_string();
        }
        tracing::warn!("{env_var} override uses insecure HTTP for non-localhost host");
    }
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_insecure_remote_url() {
        assert!(is_insecure_remote_url("http://remote.example.com/api"));
        assert!(is_insecure_remote_url(
            "http://169.254.169.254/latest/meta-data/"
        ));
        assert!(is_insecure_remote_url(
            "http://user:pass@remote.example.com/"
        ));
        assert!(!is_insecure_remote_url("http://localhost:3030"));
        assert!(!is_insecure_remote_url("http://127.0.0.1:3030"));
        assert!(!is_insecure_remote_url("http://0.0.0.0:3030"));
        assert!(!is_insecure_remote_url("https://remote.example.com/api"));
    }
}
