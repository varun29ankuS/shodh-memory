//! External integrations for syncing data sources to Shodh memory
//!
//! Supports:
//! - Linear: Issue tracking webhooks and bulk sync
//! - GitHub: PR/Issue webhooks and bulk sync

pub mod github;
pub mod linear;

pub use github::{GitHubSyncRequest, GitHubSyncResponse, GitHubWebhook, GitHubWebhookPayload};
pub use linear::{LinearSyncRequest, LinearSyncResponse, LinearWebhook, LinearWebhookPayload};

// NOTE: URL-safety helpers (is_insecure_remote_url / resolve_api_url_override /
// SHODH_ENFORCE_HTTPS) intentionally live in PR #284 (security/breakers-hardening),
// which is the owner of this module's HTTP-boundary hardening. This PR's HTTP
// embedder carries its own self-contained guard (see embeddings/http_embedder.rs,
// feature-gated) so the two PRs do not redefine the same symbols here on merge.
