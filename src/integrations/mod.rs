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
/// insecure `http://` URL pointing at a non-localhost host, the API token would
/// travel in cleartext, so the override is REJECTED and the secure default is
/// used instead.
///
/// Enforcement is the DEFAULT; `SHODH_ENFORCE_HTTPS=false` is the opt-out. It
/// previously defaulted to OFF, which inverted secure-by-default: a plain
/// `http://` override was honoured with only a warning, and a warning does not
/// stop a token being sent in cleartext. Anyone who genuinely needs a plaintext
/// remote endpoint now has to say so explicitly.
///
/// Not a blanket http ban — localhost http (proxies, test servers) stays allowed.
pub(crate) fn resolve_api_url_override(env_var: &str, default_url: &str) -> String {
    let override_url = match std::env::var(env_var) {
        Ok(u) if !u.trim().is_empty() => u.trim().to_string(),
        _ => return default_url.to_string(),
    };

    if is_insecure_remote_url(&override_url) {
        // Secure by default; `SHODH_ENFORCE_HTTPS=false` (or `0`) opts out.
        let enforce_https = std::env::var("SHODH_ENFORCE_HTTPS")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v == "false" || v == "0")
            })
            .unwrap_or(true);

        if enforce_https {
            tracing::error!(
                "{env_var} points at an insecure http:// non-localhost URL and \
                 SHODH_ENFORCE_HTTPS is not disabled — ignoring the override and using \
                 the secure default ({default_url})."
            );
            return default_url.to_string();
        }

        tracing::warn!(
            "{env_var} uses an insecure http:// URL for a non-localhost host — \
             the API token would be transmitted in cleartext. Use https://, or set \
             SHODH_ENFORCE_HTTPS=false to allow it anyway."
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

#[cfg(test)]
mod https_default_tests {
    use super::resolve_api_url_override;

    /// `SHODH_ENFORCE_HTTPS` and the override var are process-global.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const DEFAULT_URL: &str = "https://api.example.com/v1";
    const VAR: &str = "SHODH_TEST_INTEGRATION_URL";

    fn run<T>(enforce: Option<&str>, override_url: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        match enforce {
            Some(v) => std::env::set_var("SHODH_ENFORCE_HTTPS", v),
            None => std::env::remove_var("SHODH_ENFORCE_HTTPS"),
        }
        match override_url {
            Some(v) => std::env::set_var(VAR, v),
            None => std::env::remove_var(VAR),
        }
        let out = f();
        std::env::remove_var("SHODH_ENFORCE_HTTPS");
        std::env::remove_var(VAR);
        out
    }

    #[test]
    fn insecure_remote_override_is_rejected_by_default() {
        // The regression this default was flipped to prevent: with enforcement
        // off, this returned the http:// URL and the API token went in cleartext.
        let got = run(None, Some("http://evil.example.com/v1"), || {
            resolve_api_url_override(VAR, DEFAULT_URL)
        });
        assert_eq!(
            got, DEFAULT_URL,
            "insecure remote override must not be honoured by default"
        );
    }

    #[test]
    fn explicit_opt_out_allows_the_insecure_override() {
        for v in ["false", "0", "FALSE", " false "] {
            let got = run(
                Some(v),
                Some("http://legacy.internal.example.com/v1"),
                || resolve_api_url_override(VAR, DEFAULT_URL),
            );
            assert_eq!(
                got, "http://legacy.internal.example.com/v1",
                "opt-out value {v:?} ignored"
            );
        }
    }

    #[test]
    fn any_other_value_still_enforces() {
        // Only an explicit false/0 disables it; a typo must fail SAFE.
        for v in ["true", "1", "yes", "nope", ""] {
            let got = run(Some(v), Some("http://evil.example.com/v1"), || {
                resolve_api_url_override(VAR, DEFAULT_URL)
            });
            assert_eq!(got, DEFAULT_URL, "value {v:?} must not disable enforcement");
        }
    }

    #[test]
    fn localhost_http_is_always_allowed() {
        for url in ["http://localhost:8787/v1", "http://127.0.0.1:3000/v1"] {
            let got = run(None, Some(url), || {
                resolve_api_url_override(VAR, DEFAULT_URL)
            });
            assert_eq!(
                got, url,
                "localhost http must stay usable for proxies and test servers"
            );
        }
    }

    #[test]
    fn https_override_passes_through_and_unset_yields_the_default() {
        let got = run(None, Some("https://custom.example.com/v1"), || {
            resolve_api_url_override(VAR, DEFAULT_URL)
        });
        assert_eq!(got, "https://custom.example.com/v1");

        let got = run(None, None, || resolve_api_url_override(VAR, DEFAULT_URL));
        assert_eq!(got, DEFAULT_URL);
    }
}
