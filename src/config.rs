//! Configuration management for Shodh-Memory
//!
//! All configurable parameters in one place with environment variable overrides.
//! Follows the principle: sensible defaults, configurable in production.

use std::env;
use std::path::PathBuf;
use tracing::info;

/// Server configuration loaded from environment with defaults
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server port (default: 3030)
    pub port: u16,

    /// Storage path for RocksDB (default: ./shodh_memory_data)
    pub storage_path: PathBuf,

    /// Maximum users to keep in memory LRU cache (default: 1000)
    pub max_users_in_memory: usize,

    /// Maximum audit log entries per user (default: 10000)
    pub audit_max_entries_per_user: usize,

    /// Audit log rotation check interval (default: 100)
    pub audit_rotation_check_interval: usize,

    /// Audit log retention days (default: 30)
    pub audit_retention_days: u64,

    /// Rate limit: requests per second (default: 1000 - LLM-friendly)
    pub rate_limit_per_second: u64,

    /// Rate limit: burst size (default: 2000 - allows rapid agent bursts)
    pub rate_limit_burst: u32,

    /// Maximum concurrent requests (default: 200)
    pub max_concurrent_requests: usize,

    /// Whether running in production mode
    pub is_production: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 3030,
            storage_path: PathBuf::from("./shodh_memory_data"),
            max_users_in_memory: 1000,
            audit_max_entries_per_user: 10_000,
            audit_rotation_check_interval: 100,
            audit_retention_days: 30,
            rate_limit_per_second: 1000,
            rate_limit_burst: 2000,
            max_concurrent_requests: 200,
            is_production: false,
        }
    }
}

impl ServerConfig {
    /// Load configuration from environment variables with defaults
    #[allow(clippy::field_reassign_with_default)] // Environment overrides require mutable config
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Check production mode first
        config.is_production = env::var("SHODH_ENV")
            .map(|v| {
                let v = v.to_lowercase();
                v == "production" || v == "prod"
            })
            .unwrap_or(false);

        // Port
        if let Ok(val) = env::var("SHODH_PORT") {
            if let Ok(port) = val.parse() {
                config.port = port;
            }
        }

        // Storage path
        if let Ok(val) = env::var("SHODH_MEMORY_PATH") {
            config.storage_path = PathBuf::from(val);
        }

        // Max users in memory
        if let Ok(val) = env::var("SHODH_MAX_USERS") {
            if let Ok(n) = val.parse() {
                config.max_users_in_memory = n;
            }
        }

        // Audit settings
        if let Ok(val) = env::var("SHODH_AUDIT_MAX_ENTRIES") {
            if let Ok(n) = val.parse() {
                config.audit_max_entries_per_user = n;
            }
        }

        if let Ok(val) = env::var("SHODH_AUDIT_RETENTION_DAYS") {
            if let Ok(n) = val.parse() {
                config.audit_retention_days = n;
            }
        }

        // Rate limiting
        if let Ok(val) = env::var("SHODH_RATE_LIMIT") {
            if let Ok(n) = val.parse() {
                config.rate_limit_per_second = n;
            }
        }

        if let Ok(val) = env::var("SHODH_RATE_BURST") {
            if let Ok(n) = val.parse() {
                config.rate_limit_burst = n;
            }
        }

        // Concurrency
        if let Ok(val) = env::var("SHODH_MAX_CONCURRENT") {
            if let Ok(n) = val.parse() {
                config.max_concurrent_requests = n;
            }
        }

        config
    }

    /// Log the current configuration
    pub fn log(&self) {
        info!("📋 Configuration:");
        info!(
            "   Mode: {}",
            if self.is_production {
                "PRODUCTION"
            } else {
                "Development"
            }
        );
        info!("   Port: {}", self.port);
        info!("   Storage: {:?}", self.storage_path);
        info!("   Max users in memory: {}", self.max_users_in_memory);
        info!(
            "   Rate limit: {} req/sec (burst: {})",
            self.rate_limit_per_second, self.rate_limit_burst
        );
        info!("   Max concurrent: {}", self.max_concurrent_requests);
        info!("   Audit retention: {} days", self.audit_retention_days);
    }
}

/// Environment variable documentation
#[allow(unused)] // Public API - available for CLI help output
pub fn print_env_help() {
    println!("Shodh-Memory Configuration Environment Variables:");
    println!();
    println!("  SHODH_ENV              - Set to 'production' or 'prod' for production mode");
    println!("  SHODH_PORT             - Server port (default: 3030)");
    println!("  SHODH_MEMORY_PATH      - Storage directory (default: ./shodh_memory_data)");
    println!("  SHODH_API_KEYS         - Comma-separated API keys (required in production)");
    println!("  SHODH_CORS_ORIGINS     - Comma-separated allowed CORS origins");
    println!("  SHODH_MAX_USERS        - Max users in memory LRU (default: 1000)");
    println!("  SHODH_RATE_LIMIT       - Requests per second (default: 50)");
    println!("  SHODH_RATE_BURST       - Burst size (default: 100)");
    println!("  SHODH_MAX_CONCURRENT   - Max concurrent requests (default: 200)");
    println!("  SHODH_AUDIT_MAX_ENTRIES    - Max audit entries per user (default: 10000)");
    println!("  SHODH_AUDIT_RETENTION_DAYS - Audit log retention days (default: 30)");
    println!("  RUST_LOG               - Log level (e.g., info, debug, trace)");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 3030);
        assert_eq!(config.max_users_in_memory, 1000);
        assert!(!config.is_production);
    }

    #[test]
    fn test_env_override() {
        env::set_var("SHODH_PORT", "8080");
        env::set_var("SHODH_MAX_USERS", "500");

        let config = ServerConfig::from_env();
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_users_in_memory, 500);

        env::remove_var("SHODH_PORT");
        env::remove_var("SHODH_MAX_USERS");
    }
}
