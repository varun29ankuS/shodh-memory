//! Configuration management for Shodh-Memory
//!
//! All configurable parameters in one place with environment variable overrides.
//! Follows the principle: sensible defaults, configurable in production.

use std::env;
use std::path::PathBuf;
use tracing::info;

/// CORS configuration
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins (empty = allow all)
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods
    pub allowed_methods: Vec<String>,
    /// Allowed headers
    pub allowed_headers: Vec<String>,
    /// Whether to allow credentials
    pub allow_credentials: bool,
    /// Max age for preflight cache (seconds)
    pub max_age_seconds: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: Vec::new(), // Empty = allow all origins
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Request-ID".to_string(),
            ],
            allow_credentials: false,
            max_age_seconds: 86400, // 24 hours
        }
    }
}

impl CorsConfig {
    /// Load from environment variables with production safety checks
    ///
    /// In production mode (SHODH_ENV=production), warns if CORS origins are not configured.
    /// This prevents accidentally running in production with permissive CORS.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(origins) = env::var("SHODH_CORS_ORIGINS") {
            config.allowed_origins = origins
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(methods) = env::var("SHODH_CORS_METHODS") {
            config.allowed_methods = methods
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(headers) = env::var("SHODH_CORS_HEADERS") {
            config.allowed_headers = headers
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(val) = env::var("SHODH_CORS_CREDENTIALS") {
            config.allow_credentials = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = env::var("SHODH_CORS_MAX_AGE") {
            if let Ok(n) = val.parse() {
                config.max_age_seconds = n;
            }
        }

        // Production safety check: warn if CORS is permissive in production
        let is_production = env::var("SHODH_ENV")
            .map(|v| {
                let v = v.to_lowercase();
                v == "production" || v == "prod"
            })
            .unwrap_or(false);

        if is_production && config.allowed_origins.is_empty() {
            tracing::warn!(
                "⚠️  PRODUCTION WARNING: CORS allows all origins. Set SHODH_CORS_ORIGINS for security."
            );
        }

        config
    }

    /// Check if any origin restrictions are configured
    pub fn is_restricted(&self) -> bool {
        !self.allowed_origins.is_empty()
    }

    /// Convert to tower-http CorsLayer
    pub fn to_layer(&self) -> tower_http::cors::CorsLayer {
        use tower_http::cors::{AllowOrigin, Any, CorsLayer};

        let mut layer = CorsLayer::new();

        // Configure allowed origins
        if self.allowed_origins.is_empty() {
            // Intentionally permissive - no origins configured
            layer = layer.allow_origin(Any);
        } else {
            // Parse configured origins, tracking failures
            let mut valid_origins = Vec::new();
            let mut invalid_origins = Vec::new();

            for origin_str in &self.allowed_origins {
                match origin_str.parse::<axum::http::HeaderValue>() {
                    Ok(origin) => valid_origins.push(origin),
                    Err(_) => invalid_origins.push(origin_str.clone()),
                }
            }

            // Log any invalid origins
            for invalid in &invalid_origins {
                tracing::warn!("CORS: Invalid origin '{}' - skipping", invalid);
            }

            if valid_origins.is_empty() {
                // All configured origins failed to parse - this is a config error
                // Do NOT fall back to permissive - that would be a security hole
                tracing::error!(
                    "CORS: All {} configured origin(s) failed to parse. \
                     Rejecting all cross-origin requests. Fix SHODH_CORS_ORIGINS.",
                    self.allowed_origins.len()
                );
                // Use an impossible origin to effectively deny all CORS
                layer =
                    layer.allow_origin(AllowOrigin::list(Vec::<axum::http::HeaderValue>::new()));
            } else {
                if !invalid_origins.is_empty() {
                    tracing::info!(
                        "CORS: Using {} valid origin(s), {} invalid skipped",
                        valid_origins.len(),
                        invalid_origins.len()
                    );
                }
                layer = layer.allow_origin(AllowOrigin::list(valid_origins));
            }
        }

        // Configure allowed methods
        let methods: Vec<axum::http::Method> = self
            .allowed_methods
            .iter()
            .filter_map(|m| m.parse().ok())
            .collect();
        if methods.is_empty() {
            layer = layer.allow_methods(Any);
        } else {
            layer = layer.allow_methods(methods);
        }

        // Configure allowed headers
        let headers: Vec<axum::http::HeaderName> = self
            .allowed_headers
            .iter()
            .filter_map(|h| h.parse().ok())
            .collect();
        if headers.is_empty() {
            layer = layer.allow_headers(Any);
        } else {
            layer = layer.allow_headers(headers);
        }

        // Configure credentials
        if self.allow_credentials {
            layer = layer.allow_credentials(true);
        }

        // Configure max age
        layer = layer.max_age(std::time::Duration::from_secs(self.max_age_seconds));

        layer
    }
}

/// Server configuration loaded from environment with defaults
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server host address (default: 127.0.0.1)
    /// Set to 0.0.0.0 for Docker or network-accessible deployments
    pub host: String,

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

    /// Rate limit: requests per second (default: 4000 - LLM-friendly)
    pub rate_limit_per_second: u64,

    /// Rate limit: burst size (default: 8000 - allows rapid agent bursts)
    pub rate_limit_burst: u32,

    /// Maximum concurrent requests (default: 200)
    pub max_concurrent_requests: usize,

    /// Request timeout in seconds (default: 60)
    /// Requests exceeding this duration are terminated with 408 status
    pub request_timeout_secs: u64,

    /// Whether running in production mode
    pub is_production: bool,

    /// CORS configuration
    pub cors: CorsConfig,

    /// Memory maintenance interval in seconds (default: 300 = 5 minutes)
    /// Controls how often consolidation and activation decay run
    pub maintenance_interval_secs: u64,

    /// Activation decay factor per maintenance cycle (default: 0.95)
    /// Memories lose 5% activation each cycle: A_new = A_old * 0.95
    pub activation_decay_factor: f32,

    /// Backup configuration
    /// Automatic backup interval in seconds (default: 86400 = 24 hours)
    /// Set to 0 to disable automatic backups
    pub backup_interval_secs: u64,

    /// Maximum backups to keep per user (default: 7)
    /// Older backups are automatically purged
    pub backup_max_count: usize,

    /// Whether backups are enabled (default: true in production, false in dev)
    pub backup_enabled: bool,

    /// Maximum entities extracted per memory for graph insertion (default: 10)
    /// Caps the number of NER/tag/regex entities to prevent O(n²) edge explosion
    /// in the knowledge graph. 10 entities → max 45 co-occurrence edges.
    pub max_entities_per_memory: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3030,
            storage_path: PathBuf::from("./shodh_memory_data"),
            max_users_in_memory: 1000,
            audit_max_entries_per_user: 10_000,
            audit_rotation_check_interval: 100,
            audit_retention_days: 30,
            rate_limit_per_second: 4000,
            rate_limit_burst: 8000,
            max_concurrent_requests: 200,
            request_timeout_secs: 60,
            is_production: false,
            cors: CorsConfig::default(),
            maintenance_interval_secs: 300, // 5 minutes
            activation_decay_factor: 0.95,  // 5% decay per cycle
            backup_interval_secs: 86400,    // 24 hours
            backup_max_count: 7,            // Keep 7 backups (1 week of daily backups)
            backup_enabled: false,          // Disabled by default, auto-enabled in production
            max_entities_per_memory: 10,    // Cap entities per memory (10 → max 45 edges)
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

        // Host (bind address)
        if let Ok(val) = env::var("SHODH_HOST") {
            config.host = val;
        }

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

        // Request timeout
        if let Ok(val) = env::var("SHODH_REQUEST_TIMEOUT") {
            if let Ok(n) = val.parse() {
                config.request_timeout_secs = n;
            }
        }

        // CORS configuration
        config.cors = CorsConfig::from_env();

        // Memory maintenance settings
        if let Ok(val) = env::var("SHODH_MAINTENANCE_INTERVAL") {
            if let Ok(n) = val.parse() {
                config.maintenance_interval_secs = n;
            }
        }

        if let Ok(val) = env::var("SHODH_ACTIVATION_DECAY") {
            if let Ok(n) = val.parse::<f32>() {
                config.activation_decay_factor = n.clamp(0.5, 0.99);
            }
        }

        // Backup configuration
        if let Ok(val) = env::var("SHODH_BACKUP_INTERVAL") {
            if let Ok(n) = val.parse() {
                config.backup_interval_secs = n;
            }
        }

        if let Ok(val) = env::var("SHODH_BACKUP_MAX_COUNT") {
            if let Ok(n) = val.parse() {
                config.backup_max_count = n;
            }
        }

        // Auto-enable backups in production mode unless explicitly disabled
        if let Ok(val) = env::var("SHODH_BACKUP_ENABLED") {
            config.backup_enabled = val.to_lowercase() == "true" || val == "1";
        } else if config.is_production {
            // Auto-enable in production
            config.backup_enabled = true;
        }

        // Entity extraction cap
        if let Ok(val) = env::var("SHODH_MAX_ENTITIES") {
            if let Ok(n) = val.parse::<usize>() {
                config.max_entities_per_memory = n.clamp(1, 50);
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
        if self.rate_limit_per_second > 0 {
            info!(
                "   Rate limit: {} req/sec (burst: {})",
                self.rate_limit_per_second, self.rate_limit_burst
            );
        } else {
            info!("   Rate limit: disabled");
        }
        info!("   Max concurrent: {}", self.max_concurrent_requests);
        info!("   Request timeout: {}s", self.request_timeout_secs);
        info!("   Audit retention: {} days", self.audit_retention_days);
        if self.cors.is_restricted() {
            info!("   CORS origins: {:?}", self.cors.allowed_origins);
        } else {
            info!("   CORS: Permissive (all origins allowed)");
        }
        info!(
            "   Maintenance interval: {}s (decay factor: {:.2})",
            self.maintenance_interval_secs, self.activation_decay_factor
        );
        if self.backup_enabled {
            let interval_hours = self.backup_interval_secs / 3600;
            info!(
                "   Backup: enabled (every {}h, keep {})",
                interval_hours, self.backup_max_count
            );
        } else {
            info!("   Backup: disabled");
        }
    }
}

/// Environment variable documentation
#[allow(unused)] // Public API - available for CLI help output
pub fn print_env_help() {
    println!("Shodh-Memory Configuration Environment Variables:");
    println!();
    println!("  SHODH_ENV              - Set to 'production' or 'prod' for production mode");
    println!(
        "  SHODH_HOST             - Bind address (default: 127.0.0.1, use 0.0.0.0 for Docker)"
    );
    println!("  SHODH_PORT             - Server port (default: 3030)");
    println!("  SHODH_MEMORY_PATH      - Storage directory (default: ./shodh_memory_data)");
    println!("  SHODH_API_KEYS         - Comma-separated API keys (required in production)");
    println!("  SHODH_DEV_API_KEY      - Development API key (required in dev if SHODH_API_KEYS not set)");
    println!("  SHODH_MAX_USERS        - Max users in memory LRU (default: 1000)");
    println!("  SHODH_RATE_LIMIT       - Requests per second (default: 4000)");
    println!("  SHODH_RATE_BURST       - Burst size (default: 8000)");
    println!("  SHODH_MAX_CONCURRENT   - Max concurrent requests (default: 200)");
    println!("  SHODH_REQUEST_TIMEOUT  - Request timeout in seconds (default: 60)");
    println!("  SHODH_AUDIT_MAX_ENTRIES    - Max audit entries per user (default: 10000)");
    println!("  SHODH_AUDIT_RETENTION_DAYS - Audit log retention days (default: 30)");
    println!();
    println!("Integration APIs:");
    println!("  LINEAR_API_URL         - Linear GraphQL API URL (default: https://api.linear.app/graphql)");
    println!("  LINEAR_WEBHOOK_SECRET  - Linear webhook signing secret for HMAC verification");
    println!("  GITHUB_API_URL         - GitHub REST API URL (default: https://api.github.com)");
    println!("  GITHUB_WEBHOOK_SECRET  - GitHub webhook secret for HMAC verification");
    println!();
    println!("CORS Configuration:");
    println!("  SHODH_CORS_ORIGINS     - Comma-separated allowed origins (default: all)");
    println!("  SHODH_CORS_METHODS     - Comma-separated allowed methods (default: GET,POST,PUT,DELETE,OPTIONS)");
    println!("  SHODH_CORS_HEADERS     - Comma-separated allowed headers (default: Content-Type,Authorization,X-Request-ID)");
    println!("  SHODH_CORS_CREDENTIALS - Allow credentials true/false (default: false)");
    println!("  SHODH_CORS_MAX_AGE     - Preflight cache seconds (default: 86400)");
    println!();
    println!("Backup Configuration:");
    println!("  SHODH_BACKUP_ENABLED   - Enable automatic backups true/false (default: auto in production)");
    println!("  SHODH_BACKUP_INTERVAL  - Backup interval in seconds (default: 86400 = 24 hours)");
    println!("  SHODH_BACKUP_MAX_COUNT - Max backups to keep per user (default: 7)");
    println!();
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

    #[test]
    fn test_cors_default_is_permissive() {
        let cors = CorsConfig::default();
        assert!(!cors.is_restricted());
        assert!(cors.allowed_origins.is_empty());
        assert!(!cors.allowed_methods.is_empty());
        assert!(!cors.allowed_headers.is_empty());
    }

    #[test]
    fn test_cors_with_origins_is_restricted() {
        let cors = CorsConfig {
            allowed_origins: vec!["https://example.com".to_string()],
            ..Default::default()
        };
        assert!(cors.is_restricted());
    }

    #[test]
    fn test_cors_to_layer_permissive() {
        let cors = CorsConfig::default();
        let _layer = cors.to_layer(); // Should not panic
    }

    #[test]
    fn test_cors_to_layer_restricted() {
        let cors = CorsConfig {
            allowed_origins: vec!["https://example.com".to_string()],
            ..Default::default()
        };
        let _layer = cors.to_layer(); // Should not panic
    }
}
