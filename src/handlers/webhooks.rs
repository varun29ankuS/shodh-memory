//! Webhook and Streaming Handlers
//!
//! Server-Sent Events (SSE) and WebSocket endpoints for real-time
//! memory streaming, context monitoring, and event broadcasting.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
};
use futures::{SinkExt, StreamExt};
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;

use super::state::MultiUserMemoryManager;
use crate::relevance;
use crate::streaming;
use crate::validation;

/// Query parameters for SSE endpoints
#[derive(Debug, serde::Deserialize)]
pub struct SseQuery {
    pub user_id: Option<String>,
}

/// Application state type alias
pub type AppState = std::sync::Arc<MultiUserMemoryManager>;

// =============================================================================
// SSE ENDPOINTS
// =============================================================================

/// SSE endpoint for context status updates (no auth - local status line script)
///
/// Streams context updates to TUI and other subscribers.
pub async fn context_status_sse(
    State(state): State<AppState>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let receiver = state.context_broadcaster.subscribe();
    let stream = BroadcastStream::new(receiver);

    let event_stream = stream.filter_map(|result| async move {
        match result {
            Ok(status) => {
                let data = serde_json::to_string(&status).ok()?;
                Some(Ok(Event::default().event("context").data(data)))
            }
            Err(_) => None,
        }
    });

    Sse::new(event_stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

/// SSE endpoint for real-time memory events
///
/// Streams CREATE, RETRIEVE, DELETE events to connected dashboard clients.
/// Accepts optional `?user_id=X` query parameter to filter events.
/// Without user_id, no events are sent (secure by default).
pub async fn memory_events_sse(
    State(state): State<AppState>,
    Query(params): Query<SseQuery>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let receiver = state.subscribe_events();
    let stream = BroadcastStream::new(receiver);
    let filter_user_id = params.user_id;

    let event_stream = stream.filter_map(move |result| {
        let filter_uid = filter_user_id.clone();
        async move {
            match result {
                Ok(event) => {
                    // Filter by user_id: only send events belonging to the subscribed user
                    if let Some(ref uid) = filter_uid {
                        if event.user_id != *uid {
                            return None;
                        }
                    } else {
                        // No user_id specified — drop event (secure by default)
                        return None;
                    }
                    let json = serde_json::to_string(&event).ok()?;
                    Some(Ok(Event::default().event(&event.event_type).data(json)))
                }
                Err(_) => None,
            }
        }
    });

    Sse::new(event_stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("heartbeat"),
    )
}

// =============================================================================
// WEBSOCKET: STREAMING MEMORY INGESTION
// =============================================================================

/// WebSocket endpoint for streaming memory ingestion
///
/// Enables implicit learning from continuous data streams.
///
/// # Protocol
/// 1. Client connects to WS /api/stream
/// 2. Client sends handshake: { user_id, mode, extraction_config }
/// 3. Client streams messages: { type: "content"|"event"|"sensor", ... }
/// 4. Server responds with extraction results: { memories_created, entities_detected, ... }
///
/// # Modes
/// - conversation: Agent dialogue (high semantic content)
/// - sensor: IoT/robotics data (needs aggregation)
/// - event: Discrete system events
pub async fn streaming_memory_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_streaming_socket(socket, state))
}

/// Handle WebSocket connection for streaming memory ingestion
async fn handle_streaming_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut session_id: Option<String> = None;

    // Wait for handshake message
    while let Some(msg) = receiver.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => {
                tracing::debug!("WebSocket closed before handshake");
                return;
            }
            Ok(_) => continue, // Skip binary/ping/pong
            Err(e) => {
                tracing::warn!("WebSocket error before handshake: {}", e);
                return;
            }
        };

        // Parse handshake
        let handshake: streaming::StreamHandshake = match serde_json::from_str(&msg) {
            Ok(h) => h,
            Err(e) => {
                let error = streaming::ExtractionResult::Error {
                    code: "INVALID_HANDSHAKE".to_string(),
                    message: format!("Failed to parse handshake: {}", e),
                    fatal: true,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                return;
            }
        };

        // Validate user_id
        if let Err(e) = validation::validate_user_id(&handshake.user_id) {
            let error = streaming::ExtractionResult::Error {
                code: "INVALID_USER_ID".to_string(),
                message: format!("Invalid user_id: {}", e),
                fatal: true,
                timestamp: chrono::Utc::now(),
            };
            let _ = sender
                .send(Message::Text(
                    serde_json::to_string(&error)
                        .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                        .into(),
                ))
                .await;
            return;
        }

        // Validate extraction config bounds
        {
            let config = &handshake.extraction_config;
            if config.checkpoint_interval_ms > 0 && config.checkpoint_interval_ms < 1000 {
                let error = streaming::ExtractionResult::Error {
                    code: "INVALID_CONFIG".to_string(),
                    message: "checkpoint_interval_ms must be 0 (disabled) or >= 1000ms".to_string(),
                    fatal: true,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                return;
            }
            if config.max_buffer_size > 1000 {
                let error = streaming::ExtractionResult::Error {
                    code: "INVALID_CONFIG".to_string(),
                    message: "max_buffer_size must be <= 1000".to_string(),
                    fatal: true,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                return;
            }
        }

        // Create session
        let id = match state
            .streaming_extractor
            .create_session(handshake.clone())
            .await
        {
            Ok(id) => id,
            Err(e) => {
                let error = streaming::ExtractionResult::Error {
                    code: "SESSION_LIMIT_REACHED".to_string(),
                    message: e,
                    fatal: true,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                return;
            }
        };
        session_id = Some(id.clone());

        // Send acknowledgement
        let ack = streaming::ExtractionResult::Ack {
            message_type: "handshake".to_string(),
            timestamp: chrono::Utc::now(),
        };
        if sender
            .send(Message::Text(
                serde_json::to_string(&ack)
                    .unwrap_or_else(|_| r#"{"ack":true}"#.to_string())
                    .into(),
            ))
            .await
            .is_err()
        {
            return;
        }

        tracing::info!(
            "Streaming session {} created for user {} in {:?} mode",
            id,
            handshake.user_id,
            handshake.mode
        );
        break;
    }

    let session_id = match session_id {
        Some(id) => id,
        None => return,
    };

    // Get user memory system for storing extracted memories
    let user_memory = {
        let stats = state
            .streaming_extractor
            .get_session_stats(&session_id)
            .await;
        match stats {
            Some(s) => match state.get_user_memory(&s.user_id) {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("Failed to get user memory: {}", e);
                    return;
                }
            },
            None => return,
        }
    };

    // Process messages
    while let Some(msg) = receiver.next().await {
        let text = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => {
                let _ = state.streaming_extractor.close_session(&session_id).await;
                return;
            }
            Ok(Message::Ping(data)) => {
                let _ = sender.send(Message::Pong(data)).await;
                continue;
            }
            Ok(_) => continue,
            Err(e) => {
                tracing::warn!("WebSocket error: {}", e);
                break;
            }
        };

        // Parse message
        let stream_msg: streaming::StreamMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                let error = streaming::ExtractionResult::Error {
                    code: "INVALID_MESSAGE".to_string(),
                    message: format!("Failed to parse message: {}", e),
                    fatal: false,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                continue;
            }
        };

        // Process message
        let result = state
            .streaming_extractor
            .process_message(&session_id, stream_msg, user_memory.clone())
            .await;

        // Send result
        let response = serde_json::to_string(&result)
            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string());
        if sender.send(Message::Text(response.into())).await.is_err() {
            break;
        }

        // Check if session was closed
        if matches!(result, streaming::ExtractionResult::Closed { .. }) {
            break;
        }
    }

    // Cleanup session on disconnect
    if let Some(total) = state.streaming_extractor.close_session(&session_id).await {
        tracing::info!(
            "Streaming session {} closed. Total memories created: {}",
            session_id,
            total
        );
    }
}

// =============================================================================
// WEBSOCKET: CONTEXT MONITORING
// =============================================================================

/// WebSocket endpoint for context monitoring (SHO-29)
///
/// Enables proactive memory surfacing based on streaming context updates.
///
/// # Protocol
/// 1. Client connects to WS /api/context/monitor
/// 2. Client sends handshake: { user_id, config?, debounce_ms? }
/// 3. Client streams context updates: { context, entities?, config? }
/// 4. Server responds with relevant memories when threshold met
pub async fn context_monitor_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_context_monitor_socket(socket, state))
}

/// Handle WebSocket connection for context monitoring
async fn handle_context_monitor_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut user_id: Option<String> = None;
    let mut config = relevance::RelevanceConfig::default();
    let mut _debounce_ms: u64 = 100;
    let mut last_surface_time = std::time::Instant::now();

    // Wait for handshake message
    while let Some(msg) = receiver.next().await {
        let text = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => {
                tracing::debug!("Context monitor WebSocket closed before handshake");
                return;
            }
            Ok(_) => continue,
            Err(e) => {
                tracing::warn!("Context monitor WebSocket error before handshake: {}", e);
                return;
            }
        };

        // Parse handshake
        let handshake: relevance::ContextMonitorHandshake = match serde_json::from_str(&text) {
            Ok(h) => h,
            Err(e) => {
                let error = relevance::ContextMonitorResponse::Error {
                    code: "INVALID_HANDSHAKE".to_string(),
                    message: format!("Failed to parse handshake: {}", e),
                    fatal: true,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                return;
            }
        };

        // Validate user_id
        if let Err(e) = validation::validate_user_id(&handshake.user_id) {
            let error = relevance::ContextMonitorResponse::Error {
                code: "INVALID_USER_ID".to_string(),
                message: format!("Invalid user_id: {}", e),
                fatal: true,
                timestamp: chrono::Utc::now(),
            };
            let _ = sender
                .send(Message::Text(
                    serde_json::to_string(&error)
                        .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                        .into(),
                ))
                .await;
            return;
        }

        user_id = Some(handshake.user_id.clone());
        if let Some(cfg) = handshake.config {
            config = cfg;
        }
        _debounce_ms = handshake.debounce_ms;

        // Send acknowledgement
        let ack = relevance::ContextMonitorResponse::Ack {
            timestamp: chrono::Utc::now(),
        };
        if sender
            .send(Message::Text(
                serde_json::to_string(&ack)
                    .unwrap_or_else(|_| r#"{"ack":true}"#.to_string())
                    .into(),
            ))
            .await
            .is_err()
        {
            return;
        }

        tracing::info!(
            "Context monitor session started for user {}",
            handshake.user_id
        );
        break;
    }

    let user_id = match user_id {
        Some(id) => id,
        None => return,
    };

    // Get user memory and graph systems
    let memory_sys = match state.get_user_memory(&user_id) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!("Failed to get user memory: {}", e);
            return;
        }
    };

    let graph_memory = match state.get_user_graph(&user_id) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("Failed to get user graph: {}", e);
            return;
        }
    };

    let ner = state.get_neural_ner();
    let engine = std::sync::Arc::new(relevance::RelevanceEngine::new(ner));

    // Process context updates
    while let Some(msg) = receiver.next().await {
        let text = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => {
                tracing::debug!("Context monitor closed by client");
                return;
            }
            Ok(Message::Ping(data)) => {
                let _ = sender.send(Message::Pong(data)).await;
                continue;
            }
            Ok(_) => continue,
            Err(e) => {
                tracing::warn!("Context monitor WebSocket error: {}", e);
                break;
            }
        };

        // Parse context update
        let update: relevance::ContextUpdate = match serde_json::from_str(&text) {
            Ok(u) => u,
            Err(e) => {
                let error = relevance::ContextMonitorResponse::Error {
                    code: "INVALID_MESSAGE".to_string(),
                    message: format!("Failed to parse context update: {}", e),
                    fatal: false,
                    timestamp: chrono::Utc::now(),
                };
                let _ = sender
                    .send(Message::Text(
                        serde_json::to_string(&error)
                            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
                            .into(),
                    ))
                    .await;
                continue;
            }
        };

        // Check debounce
        let elapsed = last_surface_time.elapsed().as_millis() as u64;
        if elapsed < _debounce_ms {
            continue;
        }
        last_surface_time = std::time::Instant::now();

        // Use config from update if provided, otherwise use session config
        let effective_config = update.config.unwrap_or_else(|| config.clone());

        // Surface relevant memories
        let response = {
            let memory_sys = memory_sys.clone();
            let graph_memory = graph_memory.clone();
            let engine = engine.clone();
            let context = update.context.clone();

            tokio::task::spawn_blocking(move || {
                let memory_guard = memory_sys.read();
                let graph_guard = graph_memory.read();
                engine.surface_relevant(
                    &context,
                    &memory_guard,
                    Some(&*graph_guard),
                    &effective_config,
                    None,
                )
            })
            .await
        };

        let response = match response {
            Ok(Ok(r)) => relevance::ContextMonitorResponse::Relevant {
                memories: r.memories,
                detected_entities: r.detected_entities,
                latency_ms: r.latency_ms,
                timestamp: chrono::Utc::now(),
            },
            Ok(Err(e)) => relevance::ContextMonitorResponse::Error {
                code: "SURFACE_ERROR".to_string(),
                message: e.to_string(),
                fatal: false,
                timestamp: chrono::Utc::now(),
            },
            Err(e) => relevance::ContextMonitorResponse::Error {
                code: "TASK_PANIC".to_string(),
                message: e.to_string(),
                fatal: false,
                timestamp: chrono::Utc::now(),
            },
        };

        // Send response
        let response_json = serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string());
        if sender
            .send(Message::Text(response_json.into()))
            .await
            .is_err()
        {
            break;
        }
    }
}
