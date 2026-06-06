//! External integrations for syncing data sources to Shodh memory
//!
//! Supports:
//! - Linear: Issue tracking webhooks and bulk sync
//! - GitHub: PR/Issue webhooks and bulk sync

pub mod github;
pub mod linear;

pub use github::{GitHubSyncRequest, GitHubSyncResponse, GitHubWebhook, GitHubWebhookPayload};
pub use linear::{LinearSyncRequest, LinearSyncResponse, LinearWebhook, LinearWebhookPayload};

/// Resolve an integration API URL from an environment override.
///
/// If `env_var` is unset/empty, returns `default_url`. If it is set to an
/// insecure `http://` URL pointing at a non-localhost host, credentials would
/// travel in cleartext: when `SHODH_ENFORCE_HTTPS` is enabled the override is
/// rejected and the secure default is used instead; otherwise it is used as-is
/// with a warning. This is a flag-gated enforcement, not a blanket http ban —
/// localhost http (proxies, test servers) is always allowed.
pub(crate) fn resolve_api_url_override(env_var: &str, default_url: &str) -> String {
    let override_url = match std::env::var(env_var) {
        Ok(u) if !u.trim().is_empty() => u.trim().to_string(),
        _ => return default_url.to_string(),
    };

    if is_insecure_remote_url(&override_url) {
        let enforce_https = std::env::var("SHODH_ENFORCE_HTTPS")
            .map(|v| {
                let v = v.to_lowercase();
                v == "true" || v == "1"
            })
            .unwrap_or(false);

        if enforce_https {
            tracing::error!(
                "{env_var} points at an insecure http:// non-localhost URL and \
                 SHODH_ENFORCE_HTTPS is enabled — ignoring the override and using \
                 the secure default ({default_url})."
            );
            return default_url.to_string();
        }

        tracing::warn!(
            "{env_var} uses an insecure http:// URL for a non-localhost host — \
             the API token will be transmitted in cleartext. Use https:// or set \
             SHODH_ENFORCE_HTTPS=true to reject insecure overrides."
        );
    }

    override_url
}

/// Returns true if `url` is an `http://` URL whose host is NOT a loopback/local
/// address — i.e. a configuration that would leak credentials over the network.
fn is_insecure_remote_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else {
        return false; // https:// (or anything else) — not an insecure-http override
    };
    // Host is everything before the first '/', ':' (port), or '?'. This does
    // not unwrap `user:pass@host` userinfo — such a URL yields the userinfo as
    // the "host" and is treated as insecure-remote. That errs toward
    // warning/rejecting (the safe direction), so it is acceptable.
    let host = rest.split(['/', ':', '?']).next().unwrap_or("");
    !(host == "localhost" || host == "::1" || host == "0.0.0.0" || host.starts_with("127."))
}

#[cfg(test)]
mod tests {
    use super::is_insecure_remote_url;

    #[test]
    fn insecure_remote_url_detection() {
        assert!(is_insecure_remote_url("http://api.github.com"));
        assert!(is_insecure_remote_url("http://example.com:8080/graphql"));
        // Loopback / local hosts over http are allowed (dev proxies, test servers)
        assert!(!is_insecure_remote_url("http://localhost:11434"));
        assert!(!is_insecure_remote_url("http://127.0.0.1:3030"));
        assert!(!is_insecure_remote_url("http://127.1.2.3"));
        // https is always fine
        assert!(!is_insecure_remote_url("https://api.linear.app/graphql"));
    }
}
