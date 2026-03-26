//! Zenoh Transport Layer for Shodh-Memory
//!
//! Exposes memory operations over [Eclipse Zenoh](https://zenoh.io/) pub/sub and
//! request/reply primitives. Runs alongside the Axum HTTP server, sharing the same
//! `Arc<MultiUserMemoryManager>` state.
//!
//! # Feature Gate
//! This module is only compiled with `--features zenoh`. The default build is unaffected.
//!
//! # Key Expressions
//! ```text
//! {prefix}/{user_id}/remember       SUB(PUT)    → store memory (with full robotics fields)
//! {prefix}/{user_id}/recall          Queryable   → retrieve memories (spatial, mission, RL modes)
//! {prefix}/{user_id}/forget          SUB(PUT)    → delete memory
//! {prefix}/{user_id}/stream/{mode}   SUB         → streaming ingestion (sensor, event, conversation)
//! {prefix}/{user_id}/mission/start   SUB(PUT)    → begin named mission
//! {prefix}/{user_id}/mission/end     SUB(PUT)    → end mission with summary
//! {prefix}/fleet/**                  Liveliness  → robot join/leave discovery
//! {prefix}/fleet                     Queryable   → fleet roster query
//! {prefix}/health                    Queryable   → health check
//! ```
//!
//! # ROS2 Integration
//! ROS2 robots connect via `zenoh-bridge-ros2dds` or `rmw_zenoh`. Robot topics become
//! Zenoh key expressions that shodh auto-subscribes to via `AutoTopic` configuration.
//! No ROS2 dependencies are required on the shodh side.
//!
//! # Example
//! ```bash
//! # Enable Zenoh transport (local-only)
//! SHODH_ZENOH_ENABLED=true SHODH_ZENOH_LISTEN=tcp/127.0.0.1:7447 shodh server
//!
//! # Enable Zenoh transport (network-accessible, with authentication)
//! SHODH_ZENOH_ENABLED=true SHODH_ZENOH_LISTEN=tcp/0.0.0.0:7447 \
//!   SHODH_ZENOH_API_KEY=my-secret shodh server
//! ```

pub mod config;
pub mod handlers;

pub use config::ZenohConfig;

use std::sync::Arc;

use tokio::sync::watch;
use tracing::{debug, error, info, warn};
use zenoh::Session;

use crate::handlers::state::MultiUserMemoryManager;
use crate::streaming::StreamMessage;

use config::{AutoTopic, PayloadMode};

// =============================================================================
// TRANSPORT
// =============================================================================

/// Handle returned by [`ZenohTransport::start`] for graceful shutdown.
pub struct ZenohTransportHandle {
    shutdown_tx: watch::Sender<bool>,
    session: Session,
}

impl ZenohTransportHandle {
    /// Signal all subscriber/queryable tasks to stop, then close the Zenoh session.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        // Give tasks a moment to wind down before closing the session
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if let Err(e) = self.session.close().await {
            error!("Error closing Zenoh session: {}", e);
        }
        info!("Zenoh transport shut down");
    }
}

/// Start the Zenoh transport layer.
///
/// Opens a Zenoh session, registers subscribers and queryables for memory operations,
/// and returns a handle for graceful shutdown.
///
/// This function spawns multiple tokio tasks (one per subscriber/queryable). Each task
/// exits cleanly when the shutdown signal is sent via the returned handle.
pub async fn start(
    manager: Arc<MultiUserMemoryManager>,
    config: ZenohConfig,
) -> anyhow::Result<ZenohTransportHandle> {
    config.validate();

    // Build Zenoh configuration
    let mut zenoh_config = zenoh::Config::default();

    // Set mode
    match config.mode {
        config::ZenohMode::Peer => {
            zenoh_config
                .insert_json5("mode", r#""peer""#)
                .map_err(|e| anyhow::anyhow!("Failed to set Zenoh mode: {e}"))?;
        }
        config::ZenohMode::Client => {
            zenoh_config
                .insert_json5("mode", r#""client""#)
                .map_err(|e| anyhow::anyhow!("Failed to set Zenoh mode: {e}"))?;
        }
        config::ZenohMode::Router => {
            zenoh_config
                .insert_json5("mode", r#""router""#)
                .map_err(|e| anyhow::anyhow!("Failed to set Zenoh mode: {e}"))?;
        }
    }

    // Set connect endpoints
    if !config.connect.is_empty() {
        let endpoints_json = serde_json::to_string(&config.connect)?;
        zenoh_config
            .insert_json5("connect/endpoints", &endpoints_json)
            .map_err(|e| anyhow::anyhow!("Failed to set connect endpoints: {e}"))?;
    }

    // Set listen endpoints
    if !config.listen.is_empty() {
        let endpoints_json = serde_json::to_string(&config.listen)?;
        zenoh_config
            .insert_json5("listen/endpoints", &endpoints_json)
            .map_err(|e| anyhow::anyhow!("Failed to set listen endpoints: {e}"))?;
    }

    // Open session
    let session = zenoh::open(zenoh_config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to open Zenoh session: {e}"))?;

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let prefix = &config.prefix;
    let api_key = &config.api_key;

    // Register core subscribers and queryables
    register_remember_subscriber(&session, prefix, &manager, api_key.clone(), shutdown_rx.clone())
        .await?;
    register_recall_queryable(&session, prefix, &manager, api_key.clone(), shutdown_rx.clone())
        .await?;
    register_forget_subscriber(&session, prefix, &manager, api_key.clone(), shutdown_rx.clone())
        .await?;
    register_stream_subscribers(&session, prefix, &manager, api_key.clone(), shutdown_rx.clone())
        .await?;
    register_health_queryable(&session, prefix, shutdown_rx.clone()).await?;

    // Register robotics-specific subscribers
    register_mission_subscribers(
        &session,
        prefix,
        &manager,
        api_key.clone(),
        shutdown_rx.clone(),
    )
    .await?;
    register_fleet_discovery(&session, prefix, shutdown_rx.clone()).await?;

    // Register auto-topic subscribers
    for topic in &config.auto_topics {
        if topic.key_expr.is_empty() || topic.user_id.is_empty() {
            warn!(
                key_expr = %topic.key_expr,
                user_id = %topic.user_id,
                "Skipping auto-topic with empty key_expr or user_id"
            );
            continue;
        }
        register_auto_topic(&session, topic, &manager, shutdown_rx.clone()).await?;
    }

    let auto_count = config
        .auto_topics
        .iter()
        .filter(|t| !t.key_expr.is_empty() && !t.user_id.is_empty())
        .count();

    info!(
        mode = %config.mode.as_str(),
        prefix = %prefix,
        auto_topics = auto_count,
        "Zenoh transport started"
    );

    Ok(ZenohTransportHandle {
        shutdown_tx,
        session,
    })
}

// =============================================================================
// SUBSCRIBER REGISTRATION
// =============================================================================

/// Subscribe to `{prefix}/*/remember` — stores memories from Zenoh publishers.
async fn register_remember_subscriber(
    session: &Session,
    prefix: &str,
    manager: &Arc<MultiUserMemoryManager>,
    api_key: Option<String>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let key_expr = format!("{}/*/remember", prefix);
    let subscriber = session
        .declare_subscriber(&key_expr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare remember subscriber: {e}"))?;

    let mgr = Arc::clone(manager);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                sample = subscriber.recv_async() => {
                    match sample {
                        Ok(sample) => {
                            if !handlers::authenticate_payload(sample.payload(), api_key.as_deref()) {
                                continue;
                            }
                            let mgr = Arc::clone(&mgr);
                            tokio::spawn(async move {
                                handlers::handle_remember(sample, mgr).await;
                            });
                        }
                        Err(e) => {
                            error!("Remember subscriber channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    debug!("Remember subscriber shutting down");
                    break;
                }
            }
        }
    });

    debug!(key_expr = %key_expr, "Remember subscriber registered");
    Ok(())
}

/// Subscribe to `{prefix}/*/forget` — deletes memories from Zenoh publishers.
async fn register_forget_subscriber(
    session: &Session,
    prefix: &str,
    manager: &Arc<MultiUserMemoryManager>,
    api_key: Option<String>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let key_expr = format!("{}/*/forget", prefix);
    let subscriber = session
        .declare_subscriber(&key_expr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare forget subscriber: {e}"))?;

    let mgr = Arc::clone(manager);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                sample = subscriber.recv_async() => {
                    match sample {
                        Ok(sample) => {
                            if !handlers::authenticate_payload(sample.payload(), api_key.as_deref()) {
                                continue;
                            }
                            let mgr = Arc::clone(&mgr);
                            tokio::spawn(async move {
                                handlers::handle_forget(sample, mgr).await;
                            });
                        }
                        Err(e) => {
                            error!("Forget subscriber channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    debug!("Forget subscriber shutting down");
                    break;
                }
            }
        }
    });

    debug!(key_expr = %key_expr, "Forget subscriber registered");
    Ok(())
}

/// Subscribe to `{prefix}/*/stream/**` — streaming ingestion from Zenoh publishers.
///
/// Handles Content, Sensor, and Event stream messages. Each unique user_id gets
/// its own streaming session (created on first message).
async fn register_stream_subscribers(
    session: &Session,
    prefix: &str,
    manager: &Arc<MultiUserMemoryManager>,
    api_key: Option<String>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let key_expr = format!("{}/*/stream/**", prefix);
    let subscriber = session
        .declare_subscriber(&key_expr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare stream subscriber: {e}"))?;

    let mgr = Arc::clone(manager);
    let pfx = prefix.to_string();

    // Track active stream sessions per user_id
    let sessions: Arc<tokio::sync::RwLock<std::collections::HashMap<String, String>>> =
        Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));

    tokio::spawn(async move {
        loop {
            tokio::select! {
                sample = subscriber.recv_async() => {
                    match sample {
                        Ok(sample) => {
                            if !handlers::authenticate_payload(sample.payload(), api_key.as_deref()) {
                                continue;
                            }
                            let key = sample.key_expr().as_str();
                            let user_id = match handlers::extract_user_id(key, &pfx) {
                                Some(uid) => uid.to_string(),
                                None => {
                                    warn!(key = %key, "Cannot extract user_id from stream key");
                                    continue;
                                }
                            };

                            // Determine stream mode from key suffix
                            let mode = if key.contains("/stream/sensor") {
                                crate::streaming::StreamMode::Sensor
                            } else if key.contains("/stream/event") {
                                crate::streaming::StreamMode::Event
                            } else {
                                crate::streaming::StreamMode::Conversation
                            };

                            // Get or create stream session
                            let session_id = {
                                let sessions_read = sessions.read().await;
                                sessions_read.get(&user_id).cloned()
                            };

                            let session_id = match session_id {
                                Some(id) => id,
                                None => {
                                    match handlers::create_stream_session(
                                        &user_id,
                                        mode,
                                        crate::streaming::ExtractionConfig::default(),
                                        Vec::new(),
                                        &mgr,
                                    )
                                    .await
                                    {
                                        Ok(id) => {
                                            sessions.write().await.insert(user_id.clone(), id.clone());
                                            debug!(user_id = %user_id, session_id = %id, "Created stream session for Zenoh subscriber");
                                            id
                                        }
                                        Err(e) => {
                                            error!(user_id = %user_id, "Failed to create stream session: {}", e);
                                            continue;
                                        }
                                    }
                                }
                            };

                            // Parse message
                            let payload_str = match sample.payload().try_to_string() {
                                Ok(s) => s.into_owned(),
                                Err(e) => {
                                    warn!("Invalid UTF-8 in stream payload: {}", e);
                                    continue;
                                }
                            };

                            let msg: StreamMessage = match serde_json::from_str(&payload_str) {
                                Ok(m) => m,
                                Err(_) => {
                                    // Not valid JSON StreamMessage — wrap as Content
                                    handlers::wrap_passthrough(&payload_str, &[])
                                }
                            };

                            let memory = match mgr.get_user_memory(&user_id) {
                                Ok(m) => m,
                                Err(e) => {
                                    error!(user_id = %user_id, "Failed to get user memory: {}", e);
                                    continue;
                                }
                            };

                            let mgr_clone = Arc::clone(&mgr);
                            let sid = session_id.clone();
                            tokio::spawn(async move {
                                handlers::handle_stream_message(msg, &sid, memory, &mgr_clone).await;
                            });
                        }
                        Err(e) => {
                            error!("Stream subscriber channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    debug!("Stream subscriber shutting down");
                    // Close all sessions
                    let sessions = sessions.read().await;
                    for (user_id, session_id) in sessions.iter() {
                        if let Some(total) = mgr.streaming_extractor().close_session(session_id).await {
                            debug!(user_id = %user_id, total_memories = total, "Closed stream session");
                        }
                    }
                    break;
                }
            }
        }
    });

    debug!(key_expr = %key_expr, "Stream subscriber registered");
    Ok(())
}

// =============================================================================
// QUERYABLE REGISTRATION
// =============================================================================

/// Register a queryable on `{prefix}/*/recall` — handles recall requests.
async fn register_recall_queryable(
    session: &Session,
    prefix: &str,
    manager: &Arc<MultiUserMemoryManager>,
    api_key: Option<String>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let key_expr = format!("{}/*/recall", prefix);
    let queryable = session
        .declare_queryable(&key_expr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare recall queryable: {e}"))?;

    let mgr = Arc::clone(manager);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                query = queryable.recv_async() => {
                    match query {
                        Ok(query) => {
                            if let Some(payload) = query.payload() {
                                if !handlers::authenticate_payload(payload, api_key.as_deref()) {
                                    let err = serde_json::json!({"error": "Unauthorized", "success": false});
                                    if let Ok(bytes) = serde_json::to_vec(&err) {
                                        let _ = query.reply(query.key_expr(), bytes).await;
                                    }
                                    continue;
                                }
                            }
                            let mgr = Arc::clone(&mgr);
                            tokio::spawn(async move {
                                handlers::handle_recall(query, mgr).await;
                            });
                        }
                        Err(e) => {
                            error!("Recall queryable channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    debug!("Recall queryable shutting down");
                    break;
                }
            }
        }
    });

    debug!(key_expr = %key_expr, "Recall queryable registered");
    Ok(())
}

/// Register a queryable on `{prefix}/health` — returns transport health status.
async fn register_health_queryable(
    session: &Session,
    prefix: &str,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let key_expr = format!("{}/health", prefix);
    let queryable = session
        .declare_queryable(&key_expr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare health queryable: {e}"))?;

    let ke = key_expr.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                query = queryable.recv_async() => {
                    match query {
                        Ok(query) => {
                            let response = handlers::build_health_response();
                            if let Ok(bytes) = serde_json::to_vec(&response) {
                                if let Err(e) = query.reply(&ke, bytes).await {
                                    error!("Failed to reply to health query: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Health queryable channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    debug!("Health queryable shutting down");
                    break;
                }
            }
        }
    });

    debug!(key_expr = %key_expr, "Health queryable registered");
    Ok(())
}

// =============================================================================
// MISSION LIFECYCLE REGISTRATION
// =============================================================================

/// Subscribe to `{prefix}/*/mission/start` and `{prefix}/*/mission/end`.
///
/// Tracks mission boundaries in the session store. When a robot publishes
/// a mission start/end event, shodh stores it as a searchable memory and
/// emits an SSE event for dashboards.
async fn register_mission_subscribers(
    session: &Session,
    prefix: &str,
    manager: &Arc<MultiUserMemoryManager>,
    api_key: Option<String>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    // Mission start subscriber
    let start_key = format!("{}/*/mission/start", prefix);
    let start_sub = session
        .declare_subscriber(&start_key)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare mission start subscriber: {e}"))?;

    let mgr_start = Arc::clone(manager);
    let mut shutdown_start = shutdown_rx.clone();
    let api_key_start = api_key.clone();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                sample = start_sub.recv_async() => {
                    match sample {
                        Ok(sample) => {
                            if !handlers::authenticate_payload(sample.payload(), api_key_start.as_deref()) {
                                continue;
                            }
                            let mgr = Arc::clone(&mgr_start);
                            tokio::spawn(async move {
                                handlers::handle_mission_start(sample, mgr).await;
                            });
                        }
                        Err(e) => {
                            error!("Mission start subscriber channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_start.changed() => {
                    debug!("Mission start subscriber shutting down");
                    break;
                }
            }
        }
    });

    debug!(key_expr = %start_key, "Mission start subscriber registered");

    // Mission end subscriber
    let end_key = format!("{}/*/mission/end", prefix);
    let end_sub = session
        .declare_subscriber(&end_key)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare mission end subscriber: {e}"))?;

    let mgr_end = Arc::clone(manager);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                sample = end_sub.recv_async() => {
                    match sample {
                        Ok(sample) => {
                            if !handlers::authenticate_payload(sample.payload(), api_key.as_deref()) {
                                continue;
                            }
                            let mgr = Arc::clone(&mgr_end);
                            tokio::spawn(async move {
                                handlers::handle_mission_end(sample, mgr).await;
                            });
                        }
                        Err(e) => {
                            error!("Mission end subscriber channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    debug!("Mission end subscriber shutting down");
                    break;
                }
            }
        }
    });

    debug!(key_expr = %end_key, "Mission end subscriber registered");
    Ok(())
}

// =============================================================================
// FLEET DISCOVERY
// =============================================================================

/// Register fleet discovery using Zenoh liveliness tokens.
///
/// Declares a liveliness token at `{prefix}/fleet/shodh-server` so robots can
/// discover this server. Subscribes to `{prefix}/fleet/**` to track robot
/// join/leave events. A queryable at `{prefix}/fleet` returns the current fleet roster.
async fn register_fleet_discovery(
    session: &Session,
    prefix: &str,
    shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    // Declare our own liveliness token — robots can detect the server
    let liveliness_key = format!("{}/fleet/shodh-server", prefix);
    let _token = session
        .liveliness()
        .declare_token(&liveliness_key)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare liveliness token: {e}"))?;

    info!(key_expr = %liveliness_key, "Fleet liveliness token declared");

    // Subscribe to fleet liveliness changes (robot join/leave)
    let fleet_key = format!("{}/fleet/**", prefix);
    let fleet_sub = session
        .liveliness()
        .declare_subscriber(&fleet_key)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare fleet liveliness subscriber: {e}"))?;

    // Track discovered peers
    let peers: Arc<tokio::sync::RwLock<std::collections::HashMap<String, handlers::FleetPeer>>> =
        Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));

    let peers_for_task = peers.clone();
    let pfx = prefix.to_string();
    let mut shutdown_fleet_sub = shutdown_rx.clone();
    let mut shutdown_fleet_query = shutdown_rx;

    tokio::spawn(async move {
        loop {
            tokio::select! {
                sample = fleet_sub.recv_async() => {
                    match sample {
                        Ok(sample) => {
                            let key = sample.key_expr().as_str().to_string();
                            // Extract robot_id from key: shodh/fleet/{robot_id}
                            let robot_id = key
                                .strip_prefix(&pfx)
                                .and_then(|rest| rest.strip_prefix("/fleet/"))
                                .unwrap_or(&key)
                                .to_string();

                            match sample.kind() {
                                zenoh::sample::SampleKind::Put => {
                                    info!(robot_id = %robot_id, "Fleet peer joined");
                                    peers_for_task.write().await.insert(
                                        robot_id,
                                        handlers::FleetPeer {
                                            key_expr: key,
                                            joined_at: chrono::Utc::now(),
                                        },
                                    );
                                }
                                zenoh::sample::SampleKind::Delete => {
                                    info!(robot_id = %robot_id, "Fleet peer left");
                                    peers_for_task.write().await.remove(&robot_id);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Fleet liveliness subscriber channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_fleet_sub.changed() => {
                    debug!("Fleet liveliness subscriber shutting down");
                    break;
                }
            }
        }
    });

    // Register fleet queryable — returns current fleet roster
    let fleet_query_key = format!("{}/fleet", prefix);
    let fleet_queryable = session
        .declare_queryable(&fleet_query_key)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to declare fleet queryable: {e}"))?;

    let fleet_ke = fleet_query_key.clone();
    let peers_for_query = peers;

    tokio::spawn(async move {
        loop {
            tokio::select! {
                query = fleet_queryable.recv_async() => {
                    match query {
                        Ok(query) => {
                            let roster = peers_for_query.read().await;
                            let response = handlers::build_fleet_response(&roster);
                            if let Ok(bytes) = serde_json::to_vec(&response) {
                                if let Err(e) = query.reply(&fleet_ke, bytes).await {
                                    error!("Failed to reply to fleet query: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Fleet queryable channel closed: {}", e);
                            break;
                        }
                    }
                }
                _ = shutdown_fleet_query.changed() => {
                    debug!("Fleet queryable shutting down");
                    break;
                }
            }
        }
    });

    debug!(key_expr = %fleet_query_key, "Fleet discovery registered");
    Ok(())
}

// =============================================================================
// AUTO-TOPIC REGISTRATION
// =============================================================================

/// Register a subscriber for a single auto-topic.
///
/// Creates a streaming session and pipes all incoming samples through the
/// extraction pipeline. Handles both `passthrough` and `structured` payload modes.
async fn register_auto_topic(
    session: &Session,
    topic: &AutoTopic,
    manager: &Arc<MultiUserMemoryManager>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let subscriber = session
        .declare_subscriber(&topic.key_expr)
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to declare auto-topic subscriber for '{}': {e}",
                topic.key_expr
            )
        })?;

    // Create a streaming session for this auto-topic
    let mut extraction_config = topic.extraction_config.clone();
    extraction_config.validate_and_clamp();

    let session_id = handlers::create_stream_session(
        &topic.user_id,
        topic.mode,
        extraction_config,
        topic.tags.clone(),
        manager,
    )
    .await
    .map_err(|e| anyhow::anyhow!("Failed to create stream session for auto-topic: {e}"))?;

    let memory = manager
        .get_user_memory(&topic.user_id)
        .map_err(|e| anyhow::anyhow!("Failed to get user memory for auto-topic: {e}"))?;

    let mgr = Arc::clone(manager);
    let user_id = topic.user_id.clone();
    let key_expr = topic.key_expr.clone();
    let payload_mode = topic.payload_mode;
    let tags = topic.tags.clone();
    let sid = session_id.clone();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                sample = subscriber.recv_async() => {
                    match sample {
                        Ok(sample) => {
                            let payload_str = match sample.payload().try_to_string() {
                                Ok(s) => s.into_owned(),
                                Err(e) => {
                                    warn!(key_expr = %key_expr, "Invalid UTF-8 in auto-topic payload: {}", e);
                                    continue;
                                }
                            };

                            let msg = match payload_mode {
                                PayloadMode::Passthrough => {
                                    handlers::wrap_passthrough(&payload_str, &tags)
                                }
                                PayloadMode::Structured => {
                                    match serde_json::from_str::<StreamMessage>(&payload_str) {
                                        Ok(m) => m,
                                        Err(e) => {
                                            warn!(
                                                key_expr = %key_expr,
                                                "Failed to parse structured payload, falling back to passthrough: {}",
                                                e
                                            );
                                            handlers::wrap_passthrough(&payload_str, &tags)
                                        }
                                    }
                                }
                            };

                            let mem = memory.clone();
                            let mgr_clone = Arc::clone(&mgr);
                            let sid_clone = sid.clone();
                            tokio::spawn(async move {
                                handlers::handle_stream_message(msg, &sid_clone, mem, &mgr_clone).await;
                            });
                        }
                        Err(e) => {
                            error!(
                                key_expr = %key_expr,
                                "Auto-topic subscriber channel closed: {}", e
                            );
                            break;
                        }
                    }
                }
                _ = shutdown_rx.changed() => {
                    debug!(key_expr = %key_expr, "Auto-topic subscriber shutting down");
                    if let Some(total) = mgr.streaming_extractor().close_session(&sid).await {
                        info!(
                            key_expr = %key_expr,
                            user_id = %user_id,
                            total_memories = total,
                            "Auto-topic session closed"
                        );
                    }
                    break;
                }
            }
        }
    });

    info!(
        key_expr = %topic.key_expr,
        user_id = %topic.user_id,
        mode = ?topic.mode,
        payload_mode = ?topic.payload_mode,
        "Auto-topic subscriber registered"
    );

    Ok(())
}
