//! Opt-in usage telemetry — anonymous heartbeat for aggregate instance metrics.
//!
//! Sends a lightweight heartbeat (version, OS, user count, memory count) to the
//! configured endpoint once per interval (default: 24 hours). No PII, no memory
//! content, no queries, no user IDs. Disabled by default — requires explicit
//! `SHODH_TELEMETRY=true` to activate.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use tracing::{debug, info, warn};

use crate::config::ServerConfig;
use crate::handlers::MultiUserMemoryManager;

/// Anonymous heartbeat payload — aggregate metrics only.
#[derive(Debug, Serialize)]
pub struct TelemetryPayload {
    /// Persistent instance identifier (UUID v4, stored on disk)
    pub instance_id: String,
    /// Shodh-memory version (from Cargo.toml)
    pub version: &'static str,
    /// Operating system (e.g. "linux", "macos", "windows")
    pub os: &'static str,
    /// CPU architecture (e.g. "x86_64", "aarch64")
    pub arch: &'static str,
    /// Seconds since server started
    pub uptime_secs: u64,
    /// Number of distinct users on disk
    pub user_count: usize,
    /// Total memories across all users (best-effort, cached users only)
    pub total_memories: u64,
    /// Approximate storage directory size in MB
    pub storage_mb: u64,
    /// Active feature flags
    pub features: FeatureFlags,
}

/// Which optional features are active on this instance.
#[derive(Debug, Serialize)]
pub struct FeatureFlags {
    pub backups: bool,
    pub production: bool,
    pub zenoh: bool,
}

/// Read or create a persistent instance ID at `{storage}/telemetry_id`.
///
/// The ID is a UUID v4, generated once and persisted to survive restarts.
/// If the file cannot be read or created, returns a new transient UUID
/// (won't persist, but telemetry still works for this session).
fn get_or_create_instance_id(storage_path: &Path) -> String {
    let id_path = storage_path.join("telemetry_id");

    // Try reading existing ID
    if let Ok(existing) = std::fs::read_to_string(&id_path) {
        let trimmed = existing.trim().to_string();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }

    // Generate new ID
    let new_id = uuid::Uuid::new_v4().to_string();

    // Ensure storage directory exists
    let _ = std::fs::create_dir_all(storage_path);

    // Persist — failure is non-fatal
    if let Err(e) = std::fs::write(&id_path, &new_id) {
        warn!(
            "Could not persist telemetry instance ID to {:?}: {}",
            id_path, e
        );
    }

    new_id
}

/// Collect aggregate metrics for the heartbeat payload.
fn collect_heartbeat(
    manager: &MultiUserMemoryManager,
    instance_id: &str,
    uptime_start: Instant,
) -> TelemetryPayload {
    let config = manager.server_config();

    // Count users on disk (not just cached)
    let user_count = manager.list_users().len();

    // Count memories from cached users only (avoids loading every user from disk)
    let total_memories: u64 = manager
        .list_cached_users()
        .iter()
        .filter_map(|uid| {
            manager
                .get_user_memory(uid)
                .ok()
                .map(|ms| ms.read().stats().total_memories as u64)
        })
        .sum();

    // Approximate storage size (walk top-level directory)
    let storage_mb = dir_size_mb(&config.storage_path);

    TelemetryPayload {
        instance_id: instance_id.to_string(),
        version: env!("CARGO_PKG_VERSION"),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        uptime_secs: uptime_start.elapsed().as_secs(),
        user_count,
        total_memories,
        storage_mb,
        features: FeatureFlags {
            backups: config.backup_enabled,
            production: config.is_production,
            zenoh: cfg!(feature = "zenoh"),
        },
    }
}

/// Approximate directory size in megabytes (non-recursive walk of immediate children).
/// Falls back to 0 on any error — telemetry should never fail the server.
fn dir_size_mb(path: &Path) -> u64 {
    fn walk(path: &Path) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        total = total.saturating_add(meta.len());
                    } else if meta.is_dir() {
                        total = total.saturating_add(walk(&entry.path()));
                    }
                }
            }
        }
        total
    }
    walk(path) / (1024 * 1024)
}

/// Spawn the telemetry heartbeat loop as a background tokio task.
///
/// - Waits 5 minutes before the first heartbeat (server warmup).
/// - Sends one heartbeat per `config.telemetry_interval_secs`.
/// - Silent on network failures (debug-level log only).
pub fn start_telemetry_loop(
    manager: Arc<MultiUserMemoryManager>,
    config: &ServerConfig,
    uptime_start: Instant,
) {
    let url = config.telemetry_url.clone();
    let interval = std::time::Duration::from_secs(config.telemetry_interval_secs);
    let instance_id = get_or_create_instance_id(&config.storage_path);

    info!(
        "Telemetry instance ID: {} (persisted at {:?}/telemetry_id)",
        instance_id, config.storage_path
    );

    tokio::spawn(async move {
        // Warmup: let the server stabilize before first heartbeat
        tokio::time::sleep(std::time::Duration::from_secs(300)).await;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent(format!(
                "shodh-memory/{} ({}/{})",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH
            ))
            .build()
            .unwrap_or_default();

        let mut first_send = true;

        loop {
            let payload = collect_heartbeat(&manager, &instance_id, uptime_start);
            debug!("Telemetry heartbeat: {:?}", payload);

            match client.post(&url).json(&payload).send().await {
                Ok(resp) if resp.status().is_success() => {
                    if first_send {
                        info!("Telemetry: first heartbeat sent successfully");
                        first_send = false;
                    } else {
                        debug!("Telemetry heartbeat sent ({})", resp.status());
                    }
                }
                Ok(resp) => {
                    debug!("Telemetry endpoint returned {}", resp.status());
                }
                Err(e) => {
                    debug!("Telemetry send failed (non-fatal): {}", e);
                }
            }

            tokio::time::sleep(interval).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_id_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let id1 = get_or_create_instance_id(dir.path());
        assert!(!id1.is_empty());
        // Should be a valid UUID
        assert!(uuid::Uuid::parse_str(&id1).is_ok());

        // Second call returns same ID
        let id2 = get_or_create_instance_id(dir.path());
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_instance_id_creates_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("deep").join("nested");
        let id = get_or_create_instance_id(&nested);
        assert!(!id.is_empty());
        assert!(nested.join("telemetry_id").exists());
    }

    #[test]
    fn test_dir_size_mb_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(dir_size_mb(dir.path()), 0);
    }

    #[test]
    fn test_dir_size_mb_with_file() {
        let dir = tempfile::tempdir().unwrap();
        // Write a 2MB file
        let data = vec![0u8; 2 * 1024 * 1024];
        std::fs::write(dir.path().join("big.bin"), &data).unwrap();
        assert_eq!(dir_size_mb(dir.path()), 2);
    }

    #[test]
    fn test_dir_size_mb_nonexistent() {
        let path = Path::new("/nonexistent/telemetry/test/path");
        assert_eq!(dir_size_mb(path), 0);
    }
}
