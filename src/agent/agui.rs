//! AG-UI (Agent-User Interaction Protocol) SSE endpoint: `POST /api/agent/run`.
//!
//! AG-UI (<https://docs.ag-ui.com/>) is a single JSON event stream over SSE describing one
//! agent "run": lifecycle events (`RUN_STARTED`/`RUN_FINISHED`/`RUN_ERROR`), text message
//! events (`TEXT_MESSAGE_START`/`_CONTENT`/`_END`), and tool call events
//! (`TOOL_CALL_START`/`_ARGS`/`_END`/`_RESULT`). This handler drives
//! [`crate::agent::turn_loop::run`] and translates its transport-agnostic [`LoopEvent`]s
//! 1:1 onto the wire.
//!
//! # Where these event names/shapes came from
//!
//! Fetched directly from the AG-UI docs (not guessed):
//!
//! - `https://docs.ag-ui.com/sdk/js/core/events` — the full, exact `EventType` string enum
//!   (`RUN_STARTED`, `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END`,
//!   `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, `TOOL_CALL_RESULT`, `RUN_FINISHED`,
//!   `RUN_ERROR`, plus state/activity/reasoning/raw/custom events not used by this endpoint).
//! - `https://docs.ag-ui.com/concepts/events` and
//!   `https://docs.ag-ui.com/sdk/python/core/events` — per-event field lists (e.g.
//!   `RunStartedEvent{ threadId, runId }`, `TextMessageContentEvent{ messageId, delta }`,
//!   `ToolCallStartEvent{ toolCallId, toolCallName, parentMessageId }`,
//!   `ToolCallResultEvent{ messageId, toolCallId, content, role }`) and confirmation that
//!   `BaseEvent` is exactly `{ type, timestamp, raw_event }`.
//! - `https://docs.ag-ui.com/sdk/js/core/types` — the `RunAgentInput` shape
//!   (`threadId, runId, state, messages, tools, context, forwardedProps`) and the `Message`
//!   union (`role` in `"developer"|"system"|"user"|"assistant"|"tool"`, plus per-role fields).
//!
//! The docs describe SSE framing only abstractly (via an `EventEncoder` type), without quoting
//! raw bytes, so the exact `data: <json>\n\n` framing here is standard SSE (what axum's
//! `Sse<Event>` — already used by this crate's other SSE endpoints in
//! `src/handlers/webhooks.rs` — always produces) with the AG-UI event's own `type` field as the
//! discriminator; no named SSE `event:` line is set, matching how an `EventSource`-based AG-UI
//! client consumes an unnamed `onmessage` stream.
//!
//! # Request body
//!
//! Modeled on AG-UI's `RunAgentInput`, with one addition: a required top-level `user_id` for
//! shodh's per-user memory isolation (AG-UI has no built-in user concept; `forwardedProps`
//! is its sanctioned free-form extension point, but a required named field is clearer for
//! callers than an opaque bag). Deliberately simplified from the full spec for this foundation
//! slice: `messages` must be provided (the last one must be `role: "user"`, and it becomes this
//! turn's input) but only that last message is used — prior turns are **not** re-ingested from
//! the client-supplied array. Server-side [`crate::agent::conversation_memory::ShodhConversationMemory`]
//! is the source of truth for history, keyed by `threadId`; this avoids the client and server
//! ever disagreeing about what "the conversation so far" contains. `tools`/`state`/`context`/
//! `forwardedProps` are accepted (so the request validates against the spec's shape) but not
//! yet acted on — shodh registers its own fixed tool set server-side (see
//! `src/agent/tools.rs`) rather than accepting client-supplied tool definitions.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::agent::harness::ContinualHarnessStore;
use crate::agent::provider::{AgentModel, AgentProviderConfig};
use crate::agent::turn_loop::{self, LoopEvent, TurnLoopConfig};
use crate::agent::{tools, ShodhConversationMemory};
use crate::errors::ErrorResponse;
use crate::handlers::AppState;

/// Bounded so a slow/idle SSE consumer can't make the turn loop buffer unbounded memory while
/// waiting to be read; the loop's own sends simply fail (treated as client-disconnect) once a
/// receiver that isn't draining fills this up.
const EVENT_CHANNEL_CAPACITY: usize = 64;

/// One message in the client-supplied `RunAgentInput.messages` array. Only `role` and `content`
/// are read; `id`, `toolCalls`, etc. are intentionally not modeled — see module docs on why
/// prior turns in this array aren't re-ingested.
#[derive(Debug, Deserialize)]
pub struct AgUiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
}

/// `POST /api/agent/run` request body — see module docs.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgUiRunInput {
    pub thread_id: String,
    #[serde(default)]
    pub run_id: Option<String>,
    pub user_id: String,
    pub messages: Vec<AgUiMessage>,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub state: serde_json::Value,
    #[serde(default)]
    pub context: Vec<serde_json::Value>,
    #[serde(default)]
    pub forwarded_props: serde_json::Value,
}

/// One AG-UI protocol event. Rust variant names are `PascalCase`; `rename_all` /
/// `rename_all_fields` serialize them to the spec's exact wire shapes — `RunStarted` ->
/// `{"type":"RUN_STARTED", "threadId":..., "runId":...}` — verified against
/// `https://docs.ag-ui.com/sdk/js/core/events` (see module docs for the full citation).
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "SCREAMING_SNAKE_CASE", rename_all_fields = "camelCase")]
pub enum AgUiEvent {
    RunStarted {
        thread_id: String,
        run_id: String,
    },
    RunFinished {
        thread_id: String,
        run_id: String,
    },
    RunError {
        message: String,
        code: String,
    },
    TextMessageStart {
        message_id: String,
        role: String,
    },
    TextMessageContent {
        message_id: String,
        delta: String,
    },
    TextMessageEnd {
        message_id: String,
    },
    ToolCallStart {
        tool_call_id: String,
        tool_call_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_message_id: Option<String>,
    },
    ToolCallArgs {
        tool_call_id: String,
        delta: String,
    },
    ToolCallEnd {
        tool_call_id: String,
    },
    ToolCallResult {
        message_id: String,
        tool_call_id: String,
        content: serde_json::Value,
        role: String,
    },
}

fn to_ag_ui_event(event: LoopEvent, thread_id: &str, run_id: &str) -> AgUiEvent {
    match event {
        LoopEvent::RunStarted => AgUiEvent::RunStarted {
            thread_id: thread_id.to_string(),
            run_id: run_id.to_string(),
        },
        LoopEvent::RunFinished => AgUiEvent::RunFinished {
            thread_id: thread_id.to_string(),
            run_id: run_id.to_string(),
        },
        LoopEvent::RunError { message, code } => AgUiEvent::RunError { message, code },
        LoopEvent::TextMessageStart { message_id } => AgUiEvent::TextMessageStart {
            message_id,
            role: "assistant".to_string(),
        },
        LoopEvent::TextMessageContent { message_id, delta } => {
            AgUiEvent::TextMessageContent { message_id, delta }
        }
        LoopEvent::TextMessageEnd { message_id } => AgUiEvent::TextMessageEnd { message_id },
        LoopEvent::ToolCallStart {
            tool_call_id,
            tool_call_name,
            parent_message_id,
        } => AgUiEvent::ToolCallStart {
            tool_call_id,
            tool_call_name,
            parent_message_id,
        },
        LoopEvent::ToolCallArgs { tool_call_id, delta } => {
            AgUiEvent::ToolCallArgs { tool_call_id, delta }
        }
        LoopEvent::ToolCallEnd { tool_call_id } => AgUiEvent::ToolCallEnd { tool_call_id },
        LoopEvent::ToolCallResult {
            message_id,
            tool_call_id,
            content,
        } => AgUiEvent::ToolCallResult {
            message_id,
            tool_call_id,
            content,
            role: "tool".to_string(),
        },
    }
}

fn error_response(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorResponse {
            code: code.to_string(),
            message: message.into(),
            details: None,
            request_id: None,
        }),
    )
        .into_response()
}

/// `POST /api/agent/run` — see module docs for the request/response contract. Registered under
/// `build_protected_routes` in `src/handlers/router.rs`, so it sits behind the same API-key
/// auth middleware as every other `/api/*` route.
pub async fn ag_ui_run(
    State(state): State<AppState>,
    Json(input): Json<AgUiRunInput>,
) -> Response {
    // Validated up front, before any SSE bytes are written: a malformed request becomes an
    // ordinary JSON error response, not a stream that opens and then immediately errors.
    if input.thread_id.trim().is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "MISSING_THREAD_ID", "threadId must not be empty");
    }
    if let Err(e) = crate::validation::validate_user_id(&input.user_id) {
        return error_response(StatusCode::BAD_REQUEST, "INVALID_USER_ID", e.to_string());
    }
    let Some(last_message) = input.messages.last() else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "EMPTY_MESSAGES",
            "messages must contain at least one message",
        );
    };
    if last_message.role != "user" {
        return error_response(
            StatusCode::BAD_REQUEST,
            "LAST_MESSAGE_NOT_USER",
            "the last message in `messages` must have role \"user\"",
        );
    }
    let Some(user_text) = last_message
        .content
        .clone()
        .filter(|c| !c.trim().is_empty())
    else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "EMPTY_USER_MESSAGE",
            "the last user message must have non-empty content",
        );
    };

    let provider_config = match AgentProviderConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "PROVIDER_NOT_CONFIGURED",
                e.to_string(),
            )
        }
    };
    let max_turns = provider_config.max_turns;
    let model = match provider_config.build_model() {
        Ok(m) => m,
        Err(e) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "PROVIDER_INIT_FAILED",
                e.to_string(),
            )
        }
    };

    let memory = match state.get_user_memory(&input.user_id) {
        Ok(m) => m,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "MEMORY_INIT_FAILED",
                e.to_string(),
            )
        }
    };

    let thread_id = input.thread_id.clone();
    let run_id = input
        .run_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let harness = Arc::new(ContinualHarnessStore::new(memory.clone(), input.user_id.clone()));
    let conversation = Arc::new(ShodhConversationMemory::new(memory.clone(), input.user_id.clone()));
    let loop_config = TurnLoopConfig {
        max_turns,
        user_id: input.user_id.clone(),
        thread_id: thread_id.clone(),
        scope: input.user_id.clone(),
    };
    let tool_defs = tools::tool_definitions();

    let json_stream = match model {
        AgentModel::Ollama(m) => spawn_and_stream_json(
            m, tool_defs, memory, harness, conversation, loop_config, user_text, thread_id, run_id,
        ),
        AgentModel::OpenAiCompatible(m) => spawn_and_stream_json(
            m, tool_defs, memory, harness, conversation, loop_config, user_text, thread_id, run_id,
        ),
    };

    let event_stream =
        json_stream.map(|json| Ok::<Event, Infallible>(Event::default().data(json)));

    Sse::new(event_stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("ping"))
        .into_response()
}

/// Spawn [`turn_loop::run`] as a background task and return the stream of JSON-encoded AG-UI
/// events it produces, in order, as they're emitted (nothing is buffered until the run
/// completes). Generic over `M: CompletionModel` — the real handler above monomorphizes this
/// once per provider variant; this crate's own tests monomorphize it against a fake model to
/// exercise the exact same stream-construction code the real handler runs, with no network
/// involved (see `src/agent/turn_loop.rs`'s `test_support` module).
#[allow(clippy::too_many_arguments)]
fn spawn_and_stream_json<M>(
    model: M,
    tool_defs: Vec<rig_core::completion::ToolDefinition>,
    memory: Arc<parking_lot::RwLock<crate::memory::MemorySystem>>,
    harness: Arc<ContinualHarnessStore>,
    conversation: Arc<ShodhConversationMemory>,
    loop_config: TurnLoopConfig,
    user_text: String,
    thread_id: String,
    run_id: String,
) -> impl futures::Stream<Item = String>
where
    M: rig_core::completion::CompletionModel,
{
    let (tx, rx) = mpsc::channel::<LoopEvent>(EVENT_CHANNEL_CAPACITY);

    tokio::spawn(turn_loop::run(
        model,
        tool_defs,
        memory,
        harness,
        conversation,
        loop_config,
        user_text,
        tx,
    ));

    ReceiverStream::new(rx).map(move |loop_event| {
        let ag_ui_event = to_ag_ui_event(loop_event, &thread_id, &run_id);
        serde_json::to_string(&ag_ui_event).unwrap_or_else(|e| {
            tracing::error!("agui: failed to serialize AG-UI event: {e}");
            r#"{"type":"RUN_ERROR","message":"internal event serialization failure","code":"SERIALIZATION_FAILED"}"#
                .to_string()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AG-UI event mapping must round-trip the spec's exact `type` string values — this is
    /// the one place a typo in the `EventType` strings fetched from the docs would otherwise
    /// slip through undetected (a `#[derive(Serialize)]` typo doesn't fail to compile).
    #[test]
    fn event_type_strings_match_the_ag_ui_spec_exactly() {
        let cases: Vec<(LoopEvent, &str)> = vec![
            (LoopEvent::RunStarted, "RUN_STARTED"),
            (LoopEvent::RunFinished, "RUN_FINISHED"),
            (
                LoopEvent::RunError {
                    message: "boom".to_string(),
                    code: "X".to_string(),
                },
                "RUN_ERROR",
            ),
            (
                LoopEvent::TextMessageStart {
                    message_id: "m1".to_string(),
                },
                "TEXT_MESSAGE_START",
            ),
            (
                LoopEvent::TextMessageContent {
                    message_id: "m1".to_string(),
                    delta: "hi".to_string(),
                },
                "TEXT_MESSAGE_CONTENT",
            ),
            (
                LoopEvent::TextMessageEnd {
                    message_id: "m1".to_string(),
                },
                "TEXT_MESSAGE_END",
            ),
            (
                LoopEvent::ToolCallStart {
                    tool_call_id: "t1".to_string(),
                    tool_call_name: "memory_recall".to_string(),
                    parent_message_id: None,
                },
                "TOOL_CALL_START",
            ),
            (
                LoopEvent::ToolCallArgs {
                    tool_call_id: "t1".to_string(),
                    delta: "{}".to_string(),
                },
                "TOOL_CALL_ARGS",
            ),
            (
                LoopEvent::ToolCallEnd {
                    tool_call_id: "t1".to_string(),
                },
                "TOOL_CALL_END",
            ),
            (
                LoopEvent::ToolCallResult {
                    message_id: "m2".to_string(),
                    tool_call_id: "t1".to_string(),
                    content: serde_json::json!({}),
                },
                "TOOL_CALL_RESULT",
            ),
        ];

        for (loop_event, expected_type) in cases {
            let ag_ui_event = to_ag_ui_event(loop_event, "thread-1", "run-1");
            let value = serde_json::to_value(&ag_ui_event).expect("serialize AG-UI event");
            assert_eq!(
                value.get("type").and_then(|v| v.as_str()),
                Some(expected_type),
                "unexpected `type` for {value:?}"
            );
        }
    }

    #[test]
    fn run_started_carries_thread_and_run_ids_camel_cased() {
        let value = serde_json::to_value(to_ag_ui_event(LoopEvent::RunStarted, "th-1", "run-1"))
            .expect("serialize");
        assert_eq!(value["threadId"], "th-1");
        assert_eq!(value["runId"], "run-1");
    }

    #[test]
    fn tool_call_start_omits_absent_parent_message_id() {
        let value = serde_json::to_value(to_ag_ui_event(
            LoopEvent::ToolCallStart {
                tool_call_id: "t1".to_string(),
                tool_call_name: "memory_recall".to_string(),
                parent_message_id: None,
            },
            "th-1",
            "run-1",
        ))
        .expect("serialize");
        assert!(value.get("parentMessageId").is_none());
    }

    /// The load-bearing test for the SSE endpoint: drives the exact stream-construction
    /// function `ag_ui_run` uses (`spawn_and_stream_json`) with a trait-level fake model (per
    /// this task's test-scaffolding allowance) covering a tool-call turn followed by a final
    /// text answer, and asserts the resulting AG-UI event sequence is well-formed:
    /// - starts with `RUN_STARTED` and ends with `RUN_FINISHED`;
    /// - every `TOOL_CALL_START` is followed later by a `TOOL_CALL_END` with the same
    ///   `toolCallId`, and a `TOOL_CALL_RESULT` for that same id appears after the `END`;
    /// - every `TEXT_MESSAGE_START` is eventually followed by a `TEXT_MESSAGE_END` with the
    ///   same `messageId`, with only `TEXT_MESSAGE_CONTENT` for that id in between;
    /// - no event type outside the AG-UI vocabulary this endpoint emits appears.
    #[tokio::test]
    async fn sse_stream_emits_a_well_formed_ag_ui_event_sequence_for_a_turn() {
        use crate::agent::turn_loop::test_support::{setup, FakeCompletionModel, FakeResponse};
        use rig_core::streaming::{RawStreamingChoice, RawStreamingToolCall};

        let (memory, harness, conversation, _temp_dir) = setup();

        let tool_call = RawStreamingToolCall::new(
            "call-xyz".to_string(),
            crate::agent::tools::TOOL_MEMORY_RECALL.to_string(),
            serde_json::json!({ "query": "prior context" }),
        );
        let model = FakeCompletionModel::new(vec![
            vec![
                RawStreamingChoice::ToolCall(tool_call),
                RawStreamingChoice::FinalResponse(FakeResponse),
            ],
            vec![
                RawStreamingChoice::Message("Here you go.".to_string()),
                RawStreamingChoice::FinalResponse(FakeResponse),
            ],
        ]);

        let loop_config = TurnLoopConfig {
            max_turns: 4,
            user_id: "test-user".to_string(),
            thread_id: "conv-sse".to_string(),
            scope: "test-user".to_string(),
        };

        let stream = spawn_and_stream_json(
            model,
            crate::agent::tools::tool_definitions(),
            memory,
            harness,
            conversation,
            loop_config,
            "what did we discuss?".to_string(),
            "conv-sse".to_string(),
            "run-sse".to_string(),
        );

        let raw_events: Vec<String> =
            tokio::time::timeout(std::time::Duration::from_secs(10), stream.collect())
                .await
                .expect("SSE event stream must complete within the test timeout");

        assert!(!raw_events.is_empty(), "expected at least one SSE event");

        let events: Vec<serde_json::Value> = raw_events
            .iter()
            .map(|raw| serde_json::from_str(raw).expect("every SSE payload must be valid JSON"))
            .collect();

        let types: Vec<&str> = events
            .iter()
            .map(|e| e["type"].as_str().expect("every event must have a `type` string"))
            .collect();

        assert_eq!(
            types.first(),
            Some(&"RUN_STARTED"),
            "the first event must be RUN_STARTED, got {types:?}"
        );
        assert_eq!(
            types.last(),
            Some(&"RUN_FINISHED"),
            "the run must end in RUN_FINISHED (no RunError expected for this script), got {types:?}"
        );
        assert!(
            events[0]["threadId"] == "conv-sse" && events[0]["runId"] == "run-sse",
            "RUN_STARTED must carry the request's threadId/runId, got {:?}",
            events[0]
        );

        const KNOWN_TYPES: &[&str] = &[
            "RUN_STARTED",
            "RUN_FINISHED",
            "RUN_ERROR",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "TOOL_CALL_RESULT",
        ];
        for t in &types {
            assert!(KNOWN_TYPES.contains(t), "unexpected AG-UI event type: {t}");
        }

        // TOOL_CALL_START / _END / _RESULT pairing, by toolCallId, in the right relative order.
        let tool_call_id = "call-xyz";
        let start_idx = types
            .iter()
            .position(|t| *t == "TOOL_CALL_START")
            .expect("expected a TOOL_CALL_START event");
        assert_eq!(events[start_idx]["toolCallId"], tool_call_id);
        assert_eq!(events[start_idx]["toolCallName"], crate::agent::tools::TOOL_MEMORY_RECALL);

        let end_idx = types
            .iter()
            .enumerate()
            .position(|(i, t)| *t == "TOOL_CALL_END" && i > start_idx)
            .expect("expected a TOOL_CALL_END after TOOL_CALL_START");
        assert_eq!(events[end_idx]["toolCallId"], tool_call_id);

        let result_idx = types
            .iter()
            .enumerate()
            .position(|(i, t)| *t == "TOOL_CALL_RESULT" && i > end_idx)
            .expect("expected a TOOL_CALL_RESULT after TOOL_CALL_END");
        assert_eq!(events[result_idx]["toolCallId"], tool_call_id);
        assert!(
            events[result_idx]["content"]["results"].is_array(),
            "the real memory_recall tool must have actually run: {:?}",
            events[result_idx]
        );

        // TEXT_MESSAGE_START / _CONTENT / _END pairing for the final answer, all sharing one
        // messageId, appearing after the tool-call result.
        let text_start_idx = types
            .iter()
            .enumerate()
            .position(|(i, t)| *t == "TEXT_MESSAGE_START" && i > result_idx)
            .expect("expected TEXT_MESSAGE_START after the tool result");
        let message_id = events[text_start_idx]["messageId"].clone();

        let text_end_idx = types
            .iter()
            .enumerate()
            .position(|(i, t)| *t == "TEXT_MESSAGE_END" && i > text_start_idx)
            .expect("expected a TEXT_MESSAGE_END");
        assert_eq!(events[text_end_idx]["messageId"], message_id);

        let mut full_text = String::new();
        for i in (text_start_idx + 1)..text_end_idx {
            assert_eq!(
                types[i], "TEXT_MESSAGE_CONTENT",
                "only TEXT_MESSAGE_CONTENT may appear between START and END, got {} at index {i}",
                types[i]
            );
            assert_eq!(events[i]["messageId"], message_id);
            full_text.push_str(events[i]["delta"].as_str().unwrap_or_default());
        }
        assert_eq!(full_text, "Here you go.");
    }
}
