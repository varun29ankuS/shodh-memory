//! The agent turn loop: model call -> tool calls -> tool results -> repeat until the model
//! stops calling tools, bounded by a hard turn count.
//!
//! This is built directly on `rig-core`'s low-level [`rig_core::completion::CompletionModel`]
//! trait rather than any `Agent`/runtime type, because `rig-core` 0.41.0 doesn't ship one — see
//! [`crate::agent::provider`]'s module docs for why. [`run`] is generic over `M: CompletionModel`
//! rather than closed over [`crate::agent::provider::AgentModel`]; production call sites
//! monomorphize it once per provider variant (see `src/agent/agui.rs`), and this module's own
//! tests monomorphize it against a small in-process fake `CompletionModel` — the trait-level
//! seam the task's test requirements call for, with no network and no real model involved.
//!
//! [`run`] emits transport-agnostic [`LoopEvent`]s over an `mpsc` channel as they happen —
//! nothing is buffered until the turn completes. `src/agent/agui.rs` is the only thing that
//! knows these map onto AG-UI SSE events; this module has no AG-UI/HTTP/SSE awareness at all.
//!
//! # Per-turn responsibilities
//!
//! - The Continual Harness ([`crate::agent::harness::ContinualHarnessStore`]) is re-rendered
//!   into the system prompt on **every** model call in the loop (not just the first), via
//!   `render_for_prompt` — relevance is computed once per run against the original user query
//!   text (re-running the query itself against shifting tool-result context is a follow-on
//!   refinement, not part of this foundation slice).
//! - Conversation history persists through
//!   [`crate::agent::conversation_memory::ShodhConversationMemory`]: the incoming user message
//!   is appended before the first model call (so it survives even if the model call fails), and
//!   each turn's assistant message / tool-result messages are appended as they're produced.
//! - Tool dispatch ([`crate::agent::tools::execute`]) is synchronous/blocking (RocksDB +
//!   CPU work), so it runs inside `tokio::task::spawn_blocking`, matching every other call site
//!   in this crate that touches `MemorySystem`.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::mpsc;

use rig_core::completion::message::{
    AssistantContent, Message, ToolCall, ToolResult, ToolResultContent, UserContent,
};
use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequest, GetTokenUsage, ToolDefinition,
};
use rig_core::memory::ConversationMemory;
use rig_core::streaming::{
    StreamedAssistantContent, StreamingCompletionResponse, ToolCallDeltaContent,
};
use rig_core::OneOrMany;

use crate::agent::harness::{ContinualHarnessStore, RenderBudget};
use crate::agent::ShodhConversationMemory;
use crate::memory::MemorySystem;

/// Hard ceiling applied even when a caller-supplied `max_turns` is absurdly large, so a
/// misconfigured env var can't turn a single HTTP request into an unbounded background loop.
pub const ABSOLUTE_MAX_TURNS: usize = 64;

/// Bounds and identifiers for one turn-loop run.
pub struct TurnLoopConfig {
    /// Maximum number of model calls in this run. Clamped to [`ABSOLUTE_MAX_TURNS`].
    pub max_turns: usize,
    /// shodh-memory user id — scopes memory, tool dispatch, and the Continual Harness.
    pub user_id: String,
    /// Conversation id, used as the `ConversationMemory` key.
    pub thread_id: String,
    /// Continual Harness scope. Defaults to `user_id` at the call site; kept as a separate
    /// field so callers can scope harness state more narrowly than "everything this user owns"
    /// without this module needing to know why.
    pub scope: String,
}

/// One step of turn-loop activity, transport-agnostic. `src/agent/agui.rs` maps each variant
/// onto an AG-UI SSE event.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopEvent {
    RunStarted,
    TextMessageStart {
        message_id: String,
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
    },
    RunFinished,
    RunError {
        message: String,
        code: String,
    },
}

fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn build_system_prompt(harness_text: &str) -> String {
    if harness_text.trim().is_empty() {
        return "You are the shodh-memory assistant. You have durable-memory tools available \
                (memory_recall, memory_remember) — use them when they would help answer \
                accurately, or when the user asks you to remember something. Only call a tool \
                when it is actually useful; otherwise respond directly."
            .to_string();
    }

    format!(
        "You are the shodh-memory assistant. You have durable-memory tools available \
         (memory_recall, memory_remember) — use them when they would help answer accurately, \
         or when the user asks you to remember something. Only call a tool when it is actually \
         useful; otherwise respond directly.\n\n\
         # Continual Harness (relevant state for this turn)\n{harness_text}"
    )
}

/// Run one AG-UI turn-loop conversation: load history, call the model, dispatch any tool calls
/// it requests, and repeat until it stops calling tools or the turn bound is hit. Emits
/// [`LoopEvent`]s over `tx` as they happen; returns when the run is over (successfully,
/// erroneously, or because the SSE client disconnected and `tx` was dropped).
#[allow(clippy::too_many_arguments)]
pub async fn run<M>(
    model: M,
    tools: Vec<ToolDefinition>,
    memory: Arc<parking_lot::RwLock<MemorySystem>>,
    harness: Arc<ContinualHarnessStore>,
    conversation: Arc<ShodhConversationMemory>,
    config: TurnLoopConfig,
    user_message: String,
    tx: mpsc::Sender<LoopEvent>,
) where
    M: CompletionModel,
{
    if tx.send(LoopEvent::RunStarted).await.is_err() {
        return;
    }

    let max_turns = config.max_turns.clamp(1, ABSOLUTE_MAX_TURNS);

    let mut history = match conversation.load(&config.thread_id).await {
        Ok(h) => h,
        Err(e) => {
            let _ = tx
                .send(LoopEvent::RunError {
                    message: format!("failed to load conversation history: {e}"),
                    code: "HISTORY_LOAD_FAILED".to_string(),
                })
                .await;
            return;
        }
    };

    let user_msg = Message::user(user_message.clone());
    if let Err(e) = conversation
        .append(&config.thread_id, vec![user_msg.clone()])
        .await
    {
        let _ = tx
            .send(LoopEvent::RunError {
                message: format!("failed to persist the incoming user message: {e}"),
                code: "HISTORY_APPEND_FAILED".to_string(),
            })
            .await;
        return;
    }
    history.push(user_msg);

    for turn in 1..=max_turns {
        let harness_clone = Arc::clone(&harness);
        let scope = config.scope.clone();
        let query_text = user_message.clone();
        let rendered = tokio::task::spawn_blocking(move || {
            harness_clone.render_for_prompt(&scope, &query_text, RenderBudget::default())
        })
        .await;

        let harness_text = match rendered {
            Ok(Ok(state)) => state.to_prompt_text(),
            Ok(Err(e)) => {
                tracing::warn!("agent turn loop: harness render_for_prompt failed: {e}");
                String::new()
            }
            Err(e) => {
                tracing::warn!("agent turn loop: harness render task panicked: {e}");
                String::new()
            }
        };

        let mut chat_history = Vec::with_capacity(history.len() + 1);
        chat_history.push(Message::system(build_system_prompt(&harness_text)));
        chat_history.extend(history.iter().cloned());

        let chat_history = OneOrMany::from_iter_optional(chat_history)
            .expect("chat_history always has at least the system message");

        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history,
            documents: Vec::new(),
            tools: tools.clone(),
            temperature: None,
            max_tokens: Some(4096),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let stream = match model.stream(request).await {
            Ok(s) => s,
            Err(e) => {
                let _ = tx
                    .send(LoopEvent::RunError {
                        message: e.to_string(),
                        code: "PROVIDER_ERROR".to_string(),
                    })
                    .await;
                return;
            }
        };

        let outcome = match drain_stream(stream, &tx).await {
            Ok(o) => o,
            Err(TurnError::ChannelClosed) => return,
            Err(TurnError::Completion(e)) => {
                let _ = tx
                    .send(LoopEvent::RunError {
                        message: e.to_string(),
                        code: "PROVIDER_STREAM_ERROR".to_string(),
                    })
                    .await;
                return;
            }
        };

        let assistant_message = Message::Assistant {
            id: None,
            content: outcome.choice.clone(),
        };
        if let Err(e) = conversation
            .append(&config.thread_id, vec![assistant_message.clone()])
            .await
        {
            tracing::warn!("agent turn loop: failed to persist assistant turn: {e}");
        }
        history.push(assistant_message);

        let tool_calls: Vec<ToolCall> = outcome
            .choice
            .iter()
            .filter_map(|item| match item {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect();

        if tool_calls.is_empty() {
            let _ = tx.send(LoopEvent::RunFinished).await;
            return;
        }

        let mut tool_result_messages = Vec::with_capacity(tool_calls.len());
        for call in &tool_calls {
            let mem = Arc::clone(&memory);
            let user_id = config.user_id.clone();
            let tool_name = call.function.name.clone();
            let arguments = call.function.arguments.clone();

            let result_json = tokio::task::spawn_blocking(move || {
                crate::agent::tools::execute(&mem, &user_id, &tool_name, &arguments)
            })
            .await
            .unwrap_or_else(|join_err| {
                serde_json::json!({ "error": format!("tool task panicked: {join_err}") })
            });

            let result_message_id = new_id();
            if tx
                .send(LoopEvent::ToolCallResult {
                    message_id: result_message_id,
                    tool_call_id: call.id.clone(),
                    content: result_json.clone(),
                })
                .await
                .is_err()
            {
                return;
            }

            tool_result_messages.push(Message::User {
                content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                    id: call.id.clone(),
                    call_id: call.call_id.clone(),
                    content: OneOrMany::one(ToolResultContent::Json { value: result_json }),
                })),
            });
        }

        if let Err(e) = conversation
            .append(&config.thread_id, tool_result_messages.clone())
            .await
        {
            tracing::warn!("agent turn loop: failed to persist tool results: {e}");
        }
        history.extend(tool_result_messages);

        if turn == max_turns {
            let _ = tx
                .send(LoopEvent::RunError {
                    message: format!(
                        "turn limit reached ({max_turns}) without a final response — the model \
                         kept requesting tool calls"
                    ),
                    code: "TURN_LIMIT_EXCEEDED".to_string(),
                })
                .await;
            return;
        }
    }
}

struct TurnOutcome {
    choice: OneOrMany<AssistantContent>,
}

enum TurnError {
    Completion(CompletionError),
    ChannelClosed,
}

async fn emit(tx: &mpsc::Sender<LoopEvent>, event: LoopEvent) -> Result<(), TurnError> {
    tx.send(event).await.map_err(|_| TurnError::ChannelClosed)
}

/// Tracks one in-flight tool call's streamed name/arguments as `ToolCallDelta` chunks arrive,
/// so `ToolCallStart` isn't emitted until a name is known (AG-UI's `TOOL_CALL_START` requires
/// `toolCallName`) while still relaying argument deltas live once it has been.
struct ToolAccumulator {
    /// Provider-assigned tool call id — this becomes the AG-UI `toolCallId`, so the model can
    /// correlate the eventual tool-result message back to its own call.
    provider_id: String,
    parent_message_id: Option<String>,
    started: bool,
    ended: bool,
    /// Argument text that arrived before a name was known.
    pending_args: String,
}

impl ToolAccumulator {
    fn new(provider_id: String, parent_message_id: Option<String>) -> Self {
        Self {
            provider_id,
            parent_message_id,
            started: false,
            ended: false,
            pending_args: String::new(),
        }
    }
}

/// Drive one provider stream to completion, translating each item into [`LoopEvent`]s as it
/// arrives (this is what makes the SSE response incremental rather than buffered), and return
/// the fully aggregated assistant content once the provider stream ends.
async fn drain_stream<R>(
    mut stream: StreamingCompletionResponse<R>,
    tx: &mpsc::Sender<LoopEvent>,
) -> Result<TurnOutcome, TurnError>
where
    R: Clone + Unpin + GetTokenUsage,
{
    let mut text_message_id: Option<String> = None;
    let mut accumulators: HashMap<String, ToolAccumulator> = HashMap::new();
    let mut tool_order: Vec<String> = Vec::new();

    while let Some(item) = stream.next().await {
        let item = item.map_err(TurnError::Completion)?;

        match item {
            StreamedAssistantContent::Text(text) => {
                let is_new = text_message_id.is_none();
                let message_id = text_message_id.get_or_insert_with(new_id).clone();
                if is_new {
                    emit(
                        tx,
                        LoopEvent::TextMessageStart {
                            message_id: message_id.clone(),
                        },
                    )
                    .await?;
                }
                emit(
                    tx,
                    LoopEvent::TextMessageContent {
                        message_id,
                        delta: text.text,
                    },
                )
                .await?;
            }

            StreamedAssistantContent::ToolCallDelta {
                id,
                internal_call_id,
                content,
            } => {
                if !accumulators.contains_key(&internal_call_id) {
                    tool_order.push(internal_call_id.clone());
                    accumulators.insert(
                        internal_call_id.clone(),
                        ToolAccumulator::new(id, text_message_id.clone()),
                    );
                }
                // Just inserted above if absent, so this lookup always succeeds.
                let acc = accumulators
                    .get_mut(&internal_call_id)
                    .expect("accumulator inserted immediately above");

                match content {
                    ToolCallDeltaContent::Name(name) => {
                        if !acc.started {
                            acc.started = true;
                            emit(
                                tx,
                                LoopEvent::ToolCallStart {
                                    tool_call_id: acc.provider_id.clone(),
                                    tool_call_name: name,
                                    parent_message_id: acc.parent_message_id.clone(),
                                },
                            )
                            .await?;
                            if !acc.pending_args.is_empty() {
                                let buffered = std::mem::take(&mut acc.pending_args);
                                emit(
                                    tx,
                                    LoopEvent::ToolCallArgs {
                                        tool_call_id: acc.provider_id.clone(),
                                        delta: buffered,
                                    },
                                )
                                .await?;
                            }
                        }
                    }
                    ToolCallDeltaContent::Delta(chunk) => {
                        if acc.started {
                            emit(
                                tx,
                                LoopEvent::ToolCallArgs {
                                    tool_call_id: acc.provider_id.clone(),
                                    delta: chunk,
                                },
                            )
                            .await?;
                        } else {
                            acc.pending_args.push_str(&chunk);
                        }
                    }
                }
            }

            StreamedAssistantContent::ToolCall {
                tool_call,
                internal_call_id,
            } => {
                if !accumulators.contains_key(&internal_call_id) {
                    tool_order.push(internal_call_id.clone());
                    accumulators.insert(
                        internal_call_id.clone(),
                        ToolAccumulator::new(tool_call.id.clone(), text_message_id.clone()),
                    );
                }
                let acc = accumulators
                    .get_mut(&internal_call_id)
                    .expect("accumulator inserted immediately above");

                if !acc.started {
                    acc.started = true;
                    emit(
                        tx,
                        LoopEvent::ToolCallStart {
                            tool_call_id: acc.provider_id.clone(),
                            tool_call_name: tool_call.function.name.clone(),
                            parent_message_id: acc.parent_message_id.clone(),
                        },
                    )
                    .await?;
                    let args_json = serde_json::to_string(&tool_call.function.arguments)
                        .unwrap_or_else(|_| "{}".to_string());
                    emit(
                        tx,
                        LoopEvent::ToolCallArgs {
                            tool_call_id: acc.provider_id.clone(),
                            delta: args_json,
                        },
                    )
                    .await?;
                }
                if !acc.ended {
                    acc.ended = true;
                    emit(
                        tx,
                        LoopEvent::ToolCallEnd {
                            tool_call_id: acc.provider_id.clone(),
                        },
                    )
                    .await?;
                }
            }

            // Deliberately not mapped to AG-UI's REASONING_* events in this foundation slice —
            // see this crate's PR/task description for scope. Neither `ollama` nor an
            // OpenAI-compatible Chat Completions endpoint (the two providers wired up here)
            // typically emits these for tool-capable chat models.
            StreamedAssistantContent::Reasoning(_)
            | StreamedAssistantContent::ReasoningDelta { .. } => {}

            // Unmodeled provider-native output item (e.g. a hosted-tool result rig doesn't
            // parse) — not part of the aggregated assistant message, so there is nothing
            // AG-UI-shaped to relay for it.
            StreamedAssistantContent::Unknown(_) => {}

            // Internal bookkeeping only (populates `stream.response`); no AG-UI event.
            StreamedAssistantContent::Final(_) => {}
        }
    }

    if let Some(message_id) = text_message_id {
        emit(tx, LoopEvent::TextMessageEnd { message_id }).await?;
    }

    // Close out every tool call thread that is still open once the provider stream has fully
    // drained. Two cases land here: (a) deltas streamed but the provider never sent an explicit
    // finalized `ToolCall` to confirm completion, and (b) — defensively — a call whose
    // arguments streamed in before any `Name` delta ever arrived, which never got a
    // `ToolCallStart` at all; that's given one now (with a placeholder name) rather than
    // silently dropping the accumulated arguments.
    for internal_call_id in &tool_order {
        let Some(acc) = accumulators.get_mut(internal_call_id) else {
            continue;
        };
        if !acc.started {
            acc.started = true;
            emit(
                tx,
                LoopEvent::ToolCallStart {
                    tool_call_id: acc.provider_id.clone(),
                    tool_call_name: "unknown_tool".to_string(),
                    parent_message_id: acc.parent_message_id.clone(),
                },
            )
            .await?;
            if !acc.pending_args.is_empty() {
                let buffered = std::mem::take(&mut acc.pending_args);
                emit(
                    tx,
                    LoopEvent::ToolCallArgs {
                        tool_call_id: acc.provider_id.clone(),
                        delta: buffered,
                    },
                )
                .await?;
            }
        }
        if !acc.ended {
            acc.ended = true;
            emit(
                tx,
                LoopEvent::ToolCallEnd {
                    tool_call_id: acc.provider_id.clone(),
                },
            )
            .await?;
        }
    }

    Ok(TurnOutcome {
        choice: stream.choice.clone(),
    })
}

/// Test-only trait-level stand-in for a real provider's `CompletionModel`, and shared setup
/// helpers, used by this module's own tests and by `src/agent/agui.rs`'s SSE-stream test (which
/// needs the exact same fake to drive [`run`] through the real handler-facing stream-building
/// function rather than a reimplementation of it). `pub(crate)` and `#[cfg(test)]`: never
/// reachable from `src/agent/provider.rs`'s real construction path or any non-test build.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{Arc, Mutex};

    use rig_core::completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse, GetTokenUsage,
        Usage,
    };
    use rig_core::streaming::{RawStreamingChoice, StreamingCompletionResponse, StreamingResult};

    use crate::agent::harness::ContinualHarnessStore;
    use crate::agent::ShodhConversationMemory;
    use crate::memory::{MemoryConfig, MemorySystem};

    /// Response type for [`FakeCompletionModel`]. `rig_core::completion::CompletionModel`
    /// requires `Response`/`StreamingResponse` to be `Serialize + DeserializeOwned`; nothing in
    /// this fake ever inspects the value, so it carries no data.
    #[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
    pub(crate) struct FakeResponse;

    impl GetTokenUsage for FakeResponse {
        fn token_usage(&self) -> Usage {
            Usage::new()
        }
    }

    /// A trait-level stand-in for a real provider's `CompletionModel`, driven by a fixed script
    /// of canned per-call raw streaming choices. This is test scaffolding (per this task's own
    /// instructions), not a production stub.
    #[derive(Clone)]
    pub(crate) struct FakeCompletionModel {
        /// Each element is one `.stream()` call's canned response, consumed in order.
        script: Arc<Mutex<Vec<Vec<RawStreamingChoice<FakeResponse>>>>>,
        /// Number of `.stream()` calls actually made, for assertions.
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl FakeCompletionModel {
        pub(crate) fn new(script: Vec<Vec<RawStreamingChoice<FakeResponse>>>) -> Self {
            Self {
                script: Arc::new(Mutex::new(script)),
                calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }

        pub(crate) fn call_count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl CompletionModel for FakeCompletionModel {
        type Response = FakeResponse;
        type StreamingResponse = FakeResponse;
        type Client = ();

        fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
            // Never exercised: tests construct `FakeCompletionModel` directly via `new()`,
            // never through `CompletionClient::completion_model`.
            Self::new(Vec::new())
        }

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            Err(CompletionError::ProviderError(
                "FakeCompletionModel only implements stream(); the turn loop never calls \
                 completion() directly"
                    .to_string(),
            ))
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
        {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut script = self.script.lock().unwrap();
            if script.is_empty() {
                return Err(CompletionError::ProviderError(
                    "FakeCompletionModel script exhausted".to_string(),
                ));
            }
            let choices = script.remove(0);
            let items: Vec<Result<RawStreamingChoice<FakeResponse>, CompletionError>> =
                choices.into_iter().map(Ok).collect();
            let boxed: StreamingResult<FakeResponse> = Box::pin(futures::stream::iter(items));
            Ok(StreamingCompletionResponse::stream(boxed))
        }
    }

    /// Fresh on-disk `MemorySystem` + `ContinualHarnessStore` + `ShodhConversationMemory`,
    /// scoped to `test-user`, backed by a temp dir the caller must keep alive for the test's
    /// duration.
    pub(crate) fn setup() -> (
        Arc<parking_lot::RwLock<MemorySystem>>,
        Arc<ContinualHarnessStore>,
        Arc<ShodhConversationMemory>,
        tempfile::TempDir,
    ) {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let config = MemoryConfig {
            storage_path: temp_dir.path().to_path_buf(),
            working_memory_size: 50,
            session_memory_size_mb: 50,
            max_heap_per_user_mb: 200,
            auto_compress: false,
            compression_age_days: 1,
            importance_threshold: 0.0,
        };
        let system = MemorySystem::new(config, None).expect("memory system");
        let memory = Arc::new(parking_lot::RwLock::new(system));
        let harness = Arc::new(ContinualHarnessStore::new(memory.clone(), "test-user"));
        let conversation = Arc::new(ShodhConversationMemory::new(memory.clone(), "test-user"));
        (memory, harness, conversation, temp_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_support::{setup, FakeCompletionModel, FakeResponse};

    use rig_core::streaming::{RawStreamingChoice, RawStreamingToolCall};

    fn default_config(thread_id: &str) -> TurnLoopConfig {
        TurnLoopConfig {
            max_turns: 4,
            user_id: "test-user".to_string(),
            thread_id: thread_id.to_string(),
            scope: "test-user".to_string(),
        }
    }

    async fn collect_events(mut rx: mpsc::Receiver<LoopEvent>) -> Vec<LoopEvent> {
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }
        events
    }

    /// A single model turn that answers directly with no tool call must stream text deltas and
    /// finish cleanly with exactly one `RunFinished`, with no tool-call events at all.
    #[tokio::test]
    async fn text_only_turn_streams_deltas_and_finishes() {
        let (memory, harness, conversation, _temp_dir) = setup();
        let model = FakeCompletionModel::new(vec![vec![
            RawStreamingChoice::Message("Hello".to_string()),
            RawStreamingChoice::Message(", world.".to_string()),
            RawStreamingChoice::FinalResponse(FakeResponse),
        ]]);

        let (tx, rx) = mpsc::channel(32);
        run(
            model.clone(),
            Vec::new(),
            memory,
            harness,
            conversation,
            default_config("conv-text-only"),
            "hi there".to_string(),
            tx,
        )
        .await;

        let events = collect_events(rx).await;
        assert_eq!(model.call_count(), 1, "a text-only turn must call the model exactly once");
        assert_eq!(events.first(), Some(&LoopEvent::RunStarted));
        assert_eq!(events.last(), Some(&LoopEvent::RunFinished));
        assert!(events
            .iter()
            .any(|e| matches!(e, LoopEvent::TextMessageStart { .. })));
        assert!(events.iter().any(
            |e| matches!(e, LoopEvent::TextMessageContent { delta, .. } if delta == "Hello")
        ));
        assert!(events.iter().any(
            |e| matches!(e, LoopEvent::TextMessageContent { delta, .. } if delta == ", world.")
        ));
        assert!(events
            .iter()
            .any(|e| matches!(e, LoopEvent::TextMessageEnd { .. })));
        assert!(
            !events.iter().any(|e| matches!(e, LoopEvent::ToolCallStart { .. })),
            "a text-only turn must not emit any tool-call events"
        );
    }

    /// Tool call -> tool result -> a second model call that answers with text: the loop must
    /// call the model twice, dispatch the real `memory_remember` tool in between, and finish
    /// with one `RunFinished` — proving multi-turn tool dispatch, not just single-shot text.
    #[tokio::test]
    async fn tool_call_then_final_answer_drives_exactly_two_model_calls() {
        let (memory, harness, conversation, _temp_dir) = setup();

        let tool_call = RawStreamingToolCall::new(
            "call-1".to_string(),
            crate::agent::tools::TOOL_MEMORY_REMEMBER.to_string(),
            serde_json::json!({ "content": "the sky is blue" }),
        );

        let model = FakeCompletionModel::new(vec![
            vec![
                RawStreamingChoice::ToolCall(tool_call),
                RawStreamingChoice::FinalResponse(FakeResponse),
            ],
            vec![
                RawStreamingChoice::Message("Noted.".to_string()),
                RawStreamingChoice::FinalResponse(FakeResponse),
            ],
        ]);

        let (tx, rx) = mpsc::channel(32);
        run(
            model.clone(),
            crate::agent::tools::tool_definitions(),
            memory,
            harness,
            conversation,
            default_config("conv-tool-call"),
            "remember that the sky is blue".to_string(),
            tx,
        )
        .await;

        let events = collect_events(rx).await;
        assert_eq!(
            model.call_count(),
            2,
            "must call the model again after the tool result, then stop once it answers with text"
        );
        assert_eq!(events.first(), Some(&LoopEvent::RunStarted));
        assert_eq!(events.last(), Some(&LoopEvent::RunFinished));

        let start = events
            .iter()
            .find(|e| matches!(e, LoopEvent::ToolCallStart { .. }))
            .expect("expected a ToolCallStart event");
        if let LoopEvent::ToolCallStart {
            tool_call_id,
            tool_call_name,
            ..
        } = start
        {
            assert_eq!(tool_call_id, "call-1");
            assert_eq!(tool_call_name, crate::agent::tools::TOOL_MEMORY_REMEMBER);
        }

        let result = events
            .iter()
            .find(|e| matches!(e, LoopEvent::ToolCallResult { .. }))
            .expect("expected a ToolCallResult event");
        if let LoopEvent::ToolCallResult {
            tool_call_id,
            content,
            ..
        } = result
        {
            assert_eq!(tool_call_id, "call-1");
            assert!(
                content.get("memory_id").is_some(),
                "the real memory_remember tool must have actually run: {content}"
            );
        }

        assert!(events
            .iter()
            .any(|e| matches!(e, LoopEvent::ToolCallEnd { tool_call_id } if tool_call_id == "call-1")));
    }

    /// A model that keeps calling tools forever must not loop forever: the turn loop stops at
    /// its configured bound and reports `TURN_LIMIT_EXCEEDED` instead of hanging or looping
    /// unboundedly. This is the load-bearing test for "bound the turn count".
    #[tokio::test]
    async fn loop_terminates_at_the_turn_bound_instead_of_looping_forever() {
        let (memory, harness, conversation, _temp_dir) = setup();

        // Every single call returns another tool call — nothing in this script ever answers
        // with plain text, so an unbounded loop would never terminate on its own.
        let always_calls_a_tool = || {
            vec![
                RawStreamingChoice::ToolCall(RawStreamingToolCall::new(
                    "call-loop".to_string(),
                    crate::agent::tools::TOOL_MEMORY_RECALL.to_string(),
                    serde_json::json!({ "query": "anything" }),
                )),
                RawStreamingChoice::FinalResponse(FakeResponse),
            ]
        };
        let max_turns = 3;
        let script: Vec<_> = (0..max_turns + 2).map(|_| always_calls_a_tool()).collect();
        let model = FakeCompletionModel::new(script);

        let mut config = default_config("conv-turn-limit");
        config.max_turns = max_turns;

        let (tx, rx) = mpsc::channel(64);
        let run_future = run(
            model.clone(),
            crate::agent::tools::tool_definitions(),
            memory,
            harness,
            conversation,
            config,
            "keep going forever".to_string(),
            tx,
        );

        // The whole point of this test: it must complete on its own within a short, bounded
        // wait. If the loop didn't respect `max_turns`, this would hang until the outer test
        // harness's own timeout killed it — this explicit timeout turns that failure mode into
        // a fast, clear assertion failure instead of a multi-minute hang.
        tokio::time::timeout(std::time::Duration::from_secs(10), run_future)
            .await
            .expect("turn loop must terminate on its own well within the turn bound, not hang");

        let events = collect_events(rx).await;

        assert_eq!(
            model.call_count(),
            max_turns,
            "the model must be called exactly max_turns times, never more"
        );

        let last = events.last().expect("expected at least one event");
        match last {
            LoopEvent::RunError { code, .. } => {
                assert_eq!(code, "TURN_LIMIT_EXCEEDED");
            }
            other => panic!("expected the run to end in RunError(TURN_LIMIT_EXCEEDED), got {other:?}"),
        }
        assert!(
            !events.iter().any(|e| matches!(e, LoopEvent::RunFinished)),
            "a run that hit the turn limit must not also report RunFinished"
        );
    }
}
