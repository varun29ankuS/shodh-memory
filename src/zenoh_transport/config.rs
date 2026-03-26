//! Zenoh Transport Configuration
//!
//! Environment-driven configuration for the Zenoh transport layer.
//! All settings are optional — sensible defaults enable zero-config startup.
//!
//! # Environment Variables
//! ```text
//! SHODH_ZENOH_ENABLED=true           # Enable Zenoh transport (default: false)
//! SHODH_ZENOH_MODE=peer              # peer | client | router (default: peer)
//! SHODH_ZENOH_CONNECT=tcp/1.2.3.4:7447  # Connect endpoints (comma-separated)
//! SHODH_ZENOH_LISTEN=tcp/127.0.0.1:7447 # Listen endpoints (comma-separated)
//! SHODH_ZENOH_PREFIX=shodh           # Key expression prefix (default: shodh)
//! SHODH_ZENOH_API_KEY=<secret>       # Shared-secret auth for Zenoh payloads
//! SHODH_ZENOH_AUTO_TOPICS=[...]      # JSON array of AutoTopic configs
//! ```

use serde::{Deserialize, Serialize};

use crate::streaming::{ExtractionConfig, StreamMode};

// =============================================================================
// ZENOH TRANSPORT CONFIG
// =============================================================================

/// Top-level Zenoh transport configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenohConfig {
    /// Whether Zenoh transport is enabled at runtime.
    /// Even with the `zenoh` feature compiled in, transport won't start unless this is true.
    pub enabled: bool,

    /// Zenoh session mode.
    /// - `peer`: Discover and connect to other Zenoh peers on the network (default)
    /// - `client`: Connect to a Zenoh router only (lower overhead, no peer discovery)
    /// - `router`: Act as a Zenoh router (for infrastructure nodes)
    pub mode: ZenohMode,

    /// Endpoints to connect to (e.g., `["tcp/192.168.1.1:7447"]`).
    /// Empty = rely on multicast peer discovery (default for local networks).
    pub connect: Vec<String>,

    /// Endpoints to listen on (e.g., `["tcp/127.0.0.1:7447"]`).
    /// Empty = Zenoh picks an ephemeral port (fine for client/peer mode).
    pub listen: Vec<String>,

    /// Key expression prefix for all shodh-memory topics.
    /// Default: `"shodh"`.
    ///
    /// Topic structure: `{prefix}/{user_id}/{operation}`
    /// Example: `shodh/robot-1/remember`
    pub prefix: String,

    /// Auto-subscribe topics — automatically remember data from external Zenoh sources.
    /// Useful for ingesting ROS2 topics via zenoh-bridge-ros2dds without writing code.
    pub auto_topics: Vec<AutoTopic>,

    /// Shared-secret API key for authenticating Zenoh payloads.
    /// When set, every incoming Zenoh payload must include an `"api_key"` field matching
    /// this value. When `None`, authentication is skipped (suitable for local-only deployments).
    ///
    /// Loaded from `SHODH_ZENOH_API_KEY` environment variable.
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
}

/// Zenoh session mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ZenohMode {
    #[default]
    Peer,
    Client,
    Router,
}

impl ZenohMode {
    /// Convert to the zenoh WhatAmI enum value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Peer => "peer",
            Self::Client => "client",
            Self::Router => "router",
        }
    }
}

// =============================================================================
// AUTO-TOPIC CONFIG
// =============================================================================

/// Configuration for a single auto-subscribed Zenoh topic.
///
/// When the transport starts, a Zenoh subscriber is created for each auto-topic.
/// Incoming samples are converted to `StreamMessage` and piped into the
/// streaming extraction pipeline (same as WebSocket `/api/stream`).
///
/// # Example (ROS2 via zenoh-bridge-ros2dds)
/// ```json
/// {
///   "key_expr": "rt/spot1/status",
///   "user_id": "spot-1",
///   "mode": "sensor",
///   "payload_mode": "passthrough",
///   "extraction_config": { "checkpoint_interval_ms": 10000 }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTopic {
    /// Zenoh key expression to subscribe to.
    /// Can use wildcards: `rt/*/status` matches any robot's status topic.
    pub key_expr: String,

    /// User ID under which memories are stored.
    /// Each robot/agent should have its own user_id for memory isolation.
    pub user_id: String,

    /// Stream mode — determines how the extraction pipeline processes the data.
    #[serde(default)]
    pub mode: StreamMode,

    /// How to interpret the Zenoh payload.
    /// - `passthrough`: Wrap raw payload string as `StreamMessage::Content`
    /// - `structured`: Parse payload as a `StreamMessage` (JSON with `type` discriminator)
    #[serde(default)]
    pub payload_mode: PayloadMode,

    /// Extraction configuration overrides.
    /// If omitted, uses `ExtractionConfig::default()` (5s checkpoint, NER on, dedup on).
    #[serde(default)]
    pub extraction_config: ExtractionConfig,

    /// Optional tags to attach to all memories from this topic.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// How to interpret Zenoh sample payloads for auto-topics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PayloadMode {
    /// Treat the entire payload as a string and store it as a `Content` message.
    /// Best for ROS2 bridge topics where payloads are JSON-serialized sensor data.
    #[default]
    Passthrough,

    /// Parse the payload as a `StreamMessage` (expects `{"type": "content", ...}`).
    /// Best for shodh-native publishers that construct proper stream messages.
    Structured,
}

// =============================================================================
// DEFAULTS
// =============================================================================

impl Default for ZenohConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: ZenohMode::default(),
            connect: Vec::new(),
            listen: Vec::new(),
            prefix: "shodh".to_string(),
            auto_topics: Vec::new(),
            api_key: None,
        }
    }
}

// =============================================================================
// ENVIRONMENT LOADING
// =============================================================================

impl ZenohConfig {
    /// Load configuration from environment variables.
    ///
    /// All variables are optional — defaults produce a disabled config.
    /// Set `SHODH_ZENOH_ENABLED=true` to activate.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("SHODH_ZENOH_ENABLED") {
            config.enabled = val == "true" || val == "1";
        }

        if let Ok(val) = std::env::var("SHODH_ZENOH_MODE") {
            config.mode = match val.to_lowercase().as_str() {
                "client" => ZenohMode::Client,
                "router" => ZenohMode::Router,
                _ => ZenohMode::Peer,
            };
        }

        if let Ok(val) = std::env::var("SHODH_ZENOH_CONNECT") {
            config.connect = val
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(val) = std::env::var("SHODH_ZENOH_LISTEN") {
            config.listen = val
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(val) = std::env::var("SHODH_ZENOH_PREFIX") {
            let trimmed = val.trim().to_string();
            if !trimmed.is_empty() {
                config.prefix = trimmed;
            }
        }

        if let Ok(val) = std::env::var("SHODH_ZENOH_AUTO_TOPICS") {
            match serde_json::from_str::<Vec<AutoTopic>>(&val) {
                Ok(topics) => config.auto_topics = topics,
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse SHODH_ZENOH_AUTO_TOPICS: {}. Expected JSON array.",
                        e
                    );
                }
            }
        }

        if let Ok(val) = std::env::var("SHODH_ZENOH_API_KEY") {
            let trimmed = val.trim().to_string();
            if !trimmed.is_empty() {
                config.api_key = Some(trimmed);
            }
        }

        config
    }

    /// Validate configuration and log warnings for potential issues.
    pub fn validate(&self) {
        if self.prefix.contains('/') {
            tracing::warn!(
                "SHODH_ZENOH_PREFIX contains '/' — this may cause unexpected key expression nesting"
            );
        }

        for (i, topic) in self.auto_topics.iter().enumerate() {
            if topic.key_expr.is_empty() {
                tracing::warn!("Auto-topic [{}] has empty key_expr — will be skipped", i);
            }
            if topic.user_id.is_empty() {
                tracing::warn!("Auto-topic [{}] has empty user_id — will be skipped", i);
            }
        }

        if !self.connect.is_empty() && self.mode == ZenohMode::Router {
            tracing::warn!(
                "Zenoh mode is 'router' but connect endpoints are set — routers typically only listen"
            );
        }

        // Warn when listening on all interfaces without authentication
        let binds_all_interfaces = self
            .listen
            .iter()
            .any(|ep| ep.contains("0.0.0.0") || ep.contains("[::]"));
        if binds_all_interfaces && self.api_key.is_none() {
            tracing::warn!(
                "Zenoh listen endpoints include 0.0.0.0 but no SHODH_ZENOH_API_KEY is set — \
                 any network peer can invoke memory operations. Set SHODH_ZENOH_API_KEY or \
                 bind to 127.0.0.1 for local-only deployments."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ZenohConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.mode, ZenohMode::Peer);
        assert_eq!(config.prefix, "shodh");
        assert!(config.connect.is_empty());
        assert!(config.listen.is_empty());
        assert!(config.auto_topics.is_empty());
    }

    #[test]
    fn test_zenoh_mode_serde() {
        let json = r#""client""#;
        let mode: ZenohMode = serde_json::from_str(json).unwrap();
        assert_eq!(mode, ZenohMode::Client);
    }

    #[test]
    fn test_auto_topic_deserialization() {
        let json = r#"{
            "key_expr": "rt/spot1/status",
            "user_id": "spot-1",
            "mode": "sensor",
            "payload_mode": "passthrough",
            "tags": ["spot", "sensor"]
        }"#;
        let topic: AutoTopic = serde_json::from_str(json).unwrap();
        assert_eq!(topic.key_expr, "rt/spot1/status");
        assert_eq!(topic.user_id, "spot-1");
        assert_eq!(topic.mode, StreamMode::Sensor);
        assert_eq!(topic.payload_mode, PayloadMode::Passthrough);
        assert_eq!(topic.tags, vec!["spot", "sensor"]);
    }

    #[test]
    fn test_auto_topic_defaults() {
        let json = r#"{"key_expr": "test/topic", "user_id": "u1"}"#;
        let topic: AutoTopic = serde_json::from_str(json).unwrap();
        assert_eq!(topic.mode, StreamMode::Conversation);
        assert_eq!(topic.payload_mode, PayloadMode::Passthrough);
        assert!(topic.tags.is_empty());
    }

    #[test]
    fn test_payload_mode_serde() {
        let json = r#""structured""#;
        let mode: PayloadMode = serde_json::from_str(json).unwrap();
        assert_eq!(mode, PayloadMode::Structured);
    }
}
