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
            layer = layer.allow_origin(Any);
        } else {
            let origins: Vec<_> = self
                .allowed_origins
                .iter()
                .filter_map(|s| s.parse().ok())
                .collect();
            if origins.is_empty() {
                layer = layer.allow_origin(Any);
            } else {
                layer = layer.allow_origin(AllowOrigin::list(origins));
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

    /// CORS configuration
    pub cors: CorsConfig,

    /// Memory maintenance interval in seconds (default: 300 = 5 minutes)
    /// Controls how often consolidation and activation decay run
    pub maintenance_interval_secs: u64,

    /// Activation decay factor per maintenance cycle (default: 0.95)
    /// Memories lose 5% activation each cycle: A_new = A_old * 0.95
    pub activation_decay_factor: f32,
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
            cors: CorsConfig::default(),
            maintenance_interval_secs: 300, // 5 minutes
            activation_decay_factor: 0.95,  // 5% decay per cycle
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
        if self.cors.is_restricted() {
            info!("   CORS origins: {:?}", self.cors.allowed_origins);
        } else {
            info!("   CORS: Permissive (all origins allowed)");
        }
        info!(
            "   Maintenance interval: {}s (decay factor: {:.2})",
            self.maintenance_interval_secs, self.activation_decay_factor
        );
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
    println!("  SHODH_DEV_API_KEY      - Development API key (required in dev if SHODH_API_KEYS not set)");
    println!("  SHODH_MAX_USERS        - Max users in memory LRU (default: 1000)");
    println!("  SHODH_RATE_LIMIT       - Requests per second (default: 1000)");
    println!("  SHODH_RATE_BURST       - Burst size (default: 2000)");
    println!("  SHODH_MAX_CONCURRENT   - Max concurrent requests (default: 200)");
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
        let mut cors = CorsConfig::default();
        cors.allowed_origins = vec!["https://example.com".to_string()];
        assert!(cors.is_restricted());
    }

    #[test]
    fn test_cors_to_layer_permissive() {
        let cors = CorsConfig::default();
        let _layer = cors.to_layer(); // Should not panic
    }

    #[test]
    fn test_cors_to_layer_restricted() {
        let mut cors = CorsConfig::default();
        cors.allowed_origins = vec!["https://example.com".to_string()];
        let _layer = cors.to_layer(); // Should not panic
    }
}
