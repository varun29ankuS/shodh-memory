//! `rig_core::memory::ConversationMemory` backed by shodh-memory's persistent store.
//!
//! ## How a turn maps onto shodh storage (deliberate design decisions)
//!
//! **One shodh `Memory` per rig `Message`, not one per `append()` batch.** rig calls
//! `append()` once per turn with the whole turn's messages (user prompt, assistant
//! response, any tool-call/tool-result pairs). Storing per-message rather than
//! per-batch means every utterance becomes its own embedded, independently recallable
//! memory — the same substrate the rest of shodh-memory searches with `recall()`. A
//! conversation is not a shadow store bolted onto the side of the memory system; it
//! becomes memory, on equal footing with anything else `remember()`d.
//!
//! **Exact ordering via `RichContext.episode`, not tags.** Reading `src/memory/mod.rs`'s
//! `recall()` pipeline shows `Query.tags` is threaded all the way through (copied onto the
//! internal `vector_query`, `mod.rs` around line 3310) but never actually applied as a
//! filter predicate anywhere in the semantic/graph/BM25 legs — the only code path that
//! treats tags as a hard filter is the separate exact-match `recall_by_tags` /
//! `SearchCriteria::ByTags` path. That's fine for approximate scoping, but wrong for
//! reconstructing conversation history, which must be exact and ordered. The system
//! already has a purpose-built mechanism for exactly that:
//! `EpisodeContext { episode_id, sequence_number, .. }`, which `storage.rs` indexes at
//! `episode_seq:{episode_id}:{seq:010}:{memory_id}` (storage.rs ~line 1567-1571) and which
//! `SearchCriteria::ByEpisodeSequence` returns already sorted in numeric sequence order
//! (storage.rs ~line 2230-2275), independent of RocksDB iteration/hash order. Conversation
//! id `c` maps to `episode_id = "rig-conv:{c}"`; the sequence number is the running
//! per-conversation message count.
//!
//! **Content-hash dedup hazard, fixed at write time.** `MemorySystem::remember()`
//! deduplicates on a hash of `experience.content` (mod.rs ~line 917-926) and, on a hit,
//! returns the *existing* `MemoryId` instead of writing a new memory — the index is not
//! scoped per conversation. Two turns with identical rendered text ("yes", "Done.", …),
//! even across two different conversations, would otherwise collide: the second
//! `remember()` call writes nothing, no new `episode_seq` entry is created, and `load()`
//! would silently drop a message. Every stored message is therefore prefixed with a
//! `[conv:{id} seq:{n} role:{role}]` header before hashing/embedding, which guarantees
//! uniqueness by construction. The header only ever lives in `experience.content` (used for
//! embedding/hashing/BM25); it is never part of what `load()` hands back to rig.
//!
//! **Lossy text for embeddings, exact JSON for reconstruction.** `experience.content` holds
//! a human/embedding-friendly rendering of the message. The exact rig `Message` — including
//! tool calls, structured content blocks, provider-specific fields — round-trips through
//! `serde_json` and is stored verbatim in `experience.metadata["rig_message_json"]`.
//! `load()` always reconstructs from that JSON field, never by re-parsing the rendered text.
//!
//! **`append()` uses the light `MemorySystem::remember()` path, not the full HTTP
//! `remember` handler's NER + knowledge-graph ingestion pipeline**
//! (`src/handlers/remember.rs`, which runs NER, YAKE keyword extraction, and
//! `process_experience_into_graph` synchronously on every call). rig's contract requires
//! `append` to stay cheap — "it runs inline before the agent returns its response" (rig-core
//! `src/memory.rs` doc comment on `ConversationMemory`) — and the full ingestion pipeline is
//! not cheap. Conversation memories are persisted, embedded, and semantically searchable via
//! `recall()`; they are not yet wired into knowledge-graph entity/relation extraction. That
//! is a deliberate, documented scope cut for this foundation slice, not an oversight — see
//! the crate-level report for the tradeoff.
//!
//! **`clear()` deletes by id, not by a bulk criteria.** `ForgetCriteria` (types.rs) has no
//! episode-scoped variant, so `clear()` enumerates the episode's memory ids via
//! `SearchCriteria::ByEpisode` and deletes each with `ForgetCriteria::ById`. Reading
//! `MemorySystem::forget` and `MemoryStorage::remove_from_indices` confirms `ById` also
//! removes the `episode:`/`episode_seq:` index entries, so a cleared conversation leaves no
//! dangling index rows behind.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};

use rig_core::completion::message::{ToolResultContent, UserContent};
use rig_core::completion::{AssistantContent, Message};
use rig_core::memory::{ConversationMemory, MemoryError};
use rig_core::wasm_compat::WasmBoxedFuture;

use crate::memory::storage::SearchCriteria;
use crate::memory::types::{
    ContextId, ConversationContext, EpisodeContext, Experience, ExperienceType, ForgetCriteria,
    RichContext,
};
use crate::memory::MemorySystem;

/// Namespace prefix for the `episode_id` used to group a rig conversation's messages.
/// Keeps rig-originated episodes distinguishable from episodes written by other callers
/// (e.g. `handlers/remember.rs`'s own `episode_id`/`sequence_number` request fields).
const EPISODE_NAMESPACE: &str = "rig-conv";

/// Metadata key under which the exact serialized `rig_core::completion::Message` is stored.
const MESSAGE_JSON_KEY: &str = "rig_message_json";

/// Metadata key recording the message's role, duplicated out of the JSON blob so it can be
/// read back (e.g. for diagnostics) without a full deserialize.
const MESSAGE_ROLE_KEY: &str = "rig_message_role";

fn episode_id_for(conversation_id: &str) -> String {
    format!("{EPISODE_NAMESPACE}:{conversation_id}")
}

fn role_of(message: &Message) -> &'static str {
    match message {
        Message::System { .. } => "system",
        Message::User { .. } => "user",
        Message::Assistant { .. } => "assistant",
    }
}

fn render_tool_result_content(item: &ToolResultContent) -> String {
    match item {
        ToolResultContent::Text(text) => text.text.clone(),
        ToolResultContent::Image(_) => "[image]".to_string(),
        ToolResultContent::Json { value } => value.to_string(),
    }
}

fn render_user_content(item: &UserContent) -> String {
    match item {
        UserContent::Text(text) => text.text.clone(),
        UserContent::ToolResult(result) => {
            let rendered: Vec<String> = result
                .content
                .iter()
                .map(render_tool_result_content)
                .collect();
            format!("[tool_result id={}]\n{}", result.id, rendered.join("\n"))
        }
        UserContent::Image(_) => "[image]".to_string(),
        UserContent::Audio(_) => "[audio]".to_string(),
        UserContent::Video(_) => "[video]".to_string(),
        UserContent::Document(_) => "[document]".to_string(),
    }
}

fn render_assistant_content(item: &AssistantContent) -> String {
    match item {
        AssistantContent::Text(text) => text.text.clone(),
        AssistantContent::ToolCall(call) => {
            format!(
                "[tool_call {}({})]",
                call.function.name, call.function.arguments
            )
        }
        AssistantContent::Reasoning(reasoning) => {
            format!("[reasoning]\n{}", reasoning.display_text())
        }
        AssistantContent::Image(_) => "[image]".to_string(),
    }
}

/// Render a `Message` to plain, embedding/BM25-friendly text. Lossy by design — the exact
/// message is preserved separately as JSON (see module docs).
fn render_plain_text(message: &Message) -> String {
    match message {
        Message::System { content } => content.clone(),
        Message::User { content } => content
            .iter()
            .map(render_user_content)
            .collect::<Vec<_>>()
            .join("\n"),
        Message::Assistant { content, .. } => content
            .iter()
            .map(render_assistant_content)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn conversation_context(
    conversation_id: &str,
    episode_id: &str,
    sequence_number: u32,
    preceding_memory_id: Option<String>,
) -> RichContext {
    let now = chrono::Utc::now();
    RichContext {
        id: ContextId(uuid::Uuid::new_v4()),
        conversation: ConversationContext {
            conversation_id: Some(conversation_id.to_string()),
            ..Default::default()
        },
        user: Default::default(),
        project: Default::default(),
        temporal: Default::default(),
        semantic: Default::default(),
        code: Default::default(),
        document: Default::default(),
        environment: Default::default(),
        emotional: Default::default(),
        source: Default::default(),
        episode: EpisodeContext {
            episode_id: Some(episode_id.to_string()),
            sequence_number: Some(sequence_number),
            preceding_memory_id,
            episode_type: Some("rig_conversation".to_string()),
            episode_start: None,
            parent_episode_id: None,
        },
        parent: None,
        embeddings: None,
        decay_rate: 1.0,
        created_at: now,
        updated_at: now,
    }
}

/// A [`rig_core::memory::ConversationMemory`] backend that persists every message as a
/// shodh-memory `Memory`, grouped and ordered via `RichContext.episode`. See the module
/// docs for the full set of design decisions and their justification.
pub struct ShodhConversationMemory {
    memory: Arc<RwLock<MemorySystem>>,
    user_id: String,
    /// Serializes `append()`/`clear()` per conversation id so "read the current message
    /// count, then write N new sequence numbers" is atomic even under concurrent turns for
    /// the *same* conversation. Different conversations proceed fully independently — this
    /// is a per-key lock, not a single global one.
    conversation_locks: DashMap<String, Arc<Mutex<()>>>,
}

impl ShodhConversationMemory {
    /// Build a conversation-memory backend scoped to one shodh-memory user.
    ///
    /// `memory` is the same per-user handle `src/handlers/state.rs::get_user_memory` hands
    /// out to HTTP handlers — this type does not open its own storage or spin up its own
    /// `MultiUserMemoryManager`; it is a thin adapter over whatever `MemorySystem` the
    /// caller already owns.
    pub fn new(memory: Arc<RwLock<MemorySystem>>, user_id: impl Into<String>) -> Self {
        Self {
            memory,
            user_id: user_id.into(),
            conversation_locks: DashMap::new(),
        }
    }

    fn conversation_lock(&self, conversation_id: &str) -> Arc<Mutex<()>> {
        self.conversation_locks
            .entry(conversation_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

impl std::fmt::Debug for ShodhConversationMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShodhConversationMemory")
            .field("user_id", &self.user_id)
            .finish_non_exhaustive()
    }
}

impl ConversationMemory for ShodhConversationMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        let memory = Arc::clone(&self.memory);
        let user_id = self.user_id.clone();
        let conversation_id_owned = conversation_id.to_string();

        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let memory_guard = memory.read();
                let episode_id = episode_id_for(&conversation_id_owned);

                let stored = memory_guard
                    .advanced_search(SearchCriteria::ByEpisodeSequence {
                        episode_id,
                        min_sequence: None,
                        max_sequence: None,
                    })
                    .map_err(|e| MemoryError::backend(e.to_string()))?;

                let mut messages = Vec::with_capacity(stored.len());
                for mem in &stored {
                    let raw = mem
                        .experience
                        .metadata
                        .get(MESSAGE_JSON_KEY)
                        .ok_or_else(|| {
                            MemoryError::Internal(format!(
                                "conversation memory {} is missing its {} metadata field",
                                mem.id.0, MESSAGE_JSON_KEY
                            ))
                        })?;
                    let message: Message = serde_json::from_str(raw)
                        .map_err(|e| MemoryError::backend(e.to_string()))?;
                    messages.push(message);
                }

                tracing::debug!(
                    user_id = %user_id,
                    conversation_id = %conversation_id_owned,
                    message_count = messages.len(),
                    "shodh: conversation load"
                );

                Ok(messages)
            })
            .await
            .map_err(|e| MemoryError::Internal(format!("load task panicked: {e}")))?
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        if messages.is_empty() {
            return Box::pin(async { Ok(()) });
        }

        let memory = Arc::clone(&self.memory);
        let user_id = self.user_id.clone();
        let conversation_id_owned = conversation_id.to_string();
        let lock = self.conversation_lock(conversation_id);

        Box::pin(async move {
            tokio::task::spawn_blocking(move || {
                let _serialize_conversation = lock.lock();
                let memory_guard = memory.read();
                let episode_id = episode_id_for(&conversation_id_owned);

                // Exact, ordered lookup — the true message count and the most recent
                // message's id, used to seed new sequence numbers and the causal
                // `preceding_memory_id` chain.
                let existing = memory_guard
                    .advanced_search(SearchCriteria::ByEpisodeSequence {
                        episode_id: episode_id.clone(),
                        min_sequence: None,
                        max_sequence: None,
                    })
                    .map_err(|e| MemoryError::backend(e.to_string()))?;

                let message_count = messages.len();
                let starting_seq = existing.len() as u32;
                let mut preceding_memory_id = existing.last().map(|m| m.id.0.to_string());

                for (next_seq, message) in (starting_seq..).zip(messages) {
                    let role = role_of(&message);
                    let rendered = render_plain_text(&message);
                    // Content-hash dedup guard — see module docs. Prefixing with the
                    // conversation id and sequence guarantees `experience.content`
                    // uniqueness even when the rendered text repeats verbatim, both
                    // inside one conversation and across different ones.
                    let content = format!(
                        "[conv:{conversation_id_owned} seq:{next_seq} role:{role}]\n{rendered}"
                    );
                    let message_json = serde_json::to_string(&message)
                        .map_err(|e| MemoryError::backend(e.to_string()))?;

                    let mut metadata = HashMap::new();
                    metadata.insert(MESSAGE_JSON_KEY.to_string(), message_json);
                    metadata.insert(MESSAGE_ROLE_KEY.to_string(), role.to_string());

                    let context = conversation_context(
                        &conversation_id_owned,
                        &episode_id,
                        next_seq,
                        preceding_memory_id.clone(),
                    );

                    let experience = Experience {
                        experience_type: ExperienceType::Conversation,
                        content,
                        context: Some(context),
                        metadata,
                        tags: vec![
                            "rig-conversation".to_string(),
                            format!("rig-conv:{conversation_id_owned}"),
                        ],
                        ..Default::default()
                    };

                    let memory_id = memory_guard
                        .remember(experience, None)
                        .map_err(|e| MemoryError::backend(e.to_string()))?;

                    preceding_memory_id = Some(memory_id.0.to_string());
                }

                tracing::debug!(
                    user_id = %user_id,
                    conversation_id = %conversation_id_owned,
                    message_count,
                    "shodh: conversation append"
                );

                Ok(())
            })
            .await
            .map_err(|e| MemoryError::Internal(format!("append task panicked: {e}")))?
        })
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        let memory = Arc::clone(&self.memory);
        let user_id = self.user_id.clone();
        let conversation_id_owned = conversation_id.to_string();
        let lock = self.conversation_lock(conversation_id);

        Box::pin(async move {
            let result = tokio::task::spawn_blocking({
                let conversation_id_owned = conversation_id_owned.clone();
                move || {
                    let _serialize_conversation = lock.lock();
                    let memory_guard = memory.read();
                    let episode_id = episode_id_for(&conversation_id_owned);

                    let stored = memory_guard
                        .advanced_search(SearchCriteria::ByEpisode(episode_id))
                        .map_err(|e| MemoryError::backend(e.to_string()))?;

                    let deleted = stored.len();
                    for mem in stored {
                        memory_guard
                            .forget(ForgetCriteria::ById(mem.id))
                            .map_err(|e| MemoryError::backend(e.to_string()))?;
                    }

                    tracing::debug!(
                        user_id = %user_id,
                        conversation_id = %conversation_id_owned,
                        deleted,
                        "shodh: conversation cleared"
                    );

                    Ok(())
                }
            })
            .await
            .map_err(|e| MemoryError::Internal(format!("clear task panicked: {e}")))?;

            // Hygiene: drop the per-conversation append lock once cleared, so a
            // long-running process that creates/clears many short-lived conversations
            // doesn't grow this map unboundedly. Safe even if another `append`/`clear`
            // races in concurrently afterward — `conversation_lock` lazily recreates the
            // entry on next use.
            self.conversation_locks.remove(&conversation_id_owned);

            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryConfig, MemorySystem};
    use rig_core::completion::message::{Text, ToolCall, ToolFunction, ToolResult};
    use rig_core::OneOrMany;

    /// Build a fresh on-disk `MemorySystem` rooted at `storage_path`. Returns the config too,
    /// so the caller can reopen the *same* storage path after dropping the first instance —
    /// that reopen is what makes the round-trip test below an actual restart simulation
    /// instead of just exercising the in-process `MemorySystem` handle.
    fn config_at(storage_path: std::path::PathBuf) -> MemoryConfig {
        MemoryConfig {
            storage_path,
            working_memory_size: 50,
            session_memory_size_mb: 50,
            max_heap_per_user_mb: 200,
            auto_compress: false,
            compression_age_days: 1,
            importance_threshold: 0.0,
        }
    }

    fn open(config: &MemoryConfig) -> Arc<RwLock<MemorySystem>> {
        Arc::new(RwLock::new(
            MemorySystem::new(config.clone(), None).expect("open memory system"),
        ))
    }

    /// A tool-call/tool-result pair — the case where a lossy text render and the exact stored
    /// JSON diverge. `AssistantContent` is `#[serde(untagged)]` (rig-core message.rs), which
    /// is the one shape where a deserialize could plausibly pick the wrong variant, so this
    /// is worth exercising for real rather than assuming it round-trips.
    fn tool_call_turn() -> (Message, Message) {
        let call = Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                "call-1".to_string(),
                ToolFunction {
                    name: "get_weather".to_string(),
                    arguments: serde_json::json!({"city": "Madrid"}),
                },
            ))),
        };
        let result = Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: "call-1".to_string(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::Text(Text::new("18C, partly cloudy"))),
            })),
        };
        (call, result)
    }

    /// Without persistence, `load()` after a process restart would return nothing — the
    /// whole point of `ShodhConversationMemory` over rig's `InMemoryConversationMemory`
    /// default. This test fails against an in-memory-only backend and passes only if
    /// conversation turns actually survive a dropped-and-reopened `MemorySystem` pointed at
    /// the same on-disk path.
    ///
    /// Asserts full `Message` equality (rig-core derives `PartialEq` on it), not just
    /// rendered text — the module docs promise tool calls, structured content, and
    /// provider-specific fields round-trip through the stored JSON verbatim, and a naive
    /// implementation that only persisted `render_plain_text()` output would pass a
    /// text-only comparison but fail this one.
    #[tokio::test]
    async fn conversation_round_trips_across_a_simulated_restart() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let config = config_at(temp_dir.path().to_path_buf());

        let (tool_call, tool_result) = tool_call_turn();
        let expected = vec![
            Message::user("What is the capital of France?"),
            Message::assistant("The capital of France is Paris."),
            Message::user("And the weather in Madrid?"),
            tool_call,
            tool_result,
            Message::assistant("It's 18C and partly cloudy in Madrid."),
        ];

        // "Before restart": open storage, append every turn, then drop every handle to it.
        // `MemorySystem`'s `Drop` impl flushes the RocksDB WAL (see mod.rs), so nothing here
        // relies on an explicit shutdown call the real server would also not always get.
        {
            let memory = open(&config);
            let conv_memory = ShodhConversationMemory::new(memory, "test-user");

            let before = conv_memory.load("conv-1").await.expect("initial load");
            assert!(before.is_empty(), "fresh conversation must start empty");

            conv_memory
                .append("conv-1", expected[0..2].to_vec())
                .await
                .expect("append turn 1");
            conv_memory
                .append("conv-1", expected[2..5].to_vec())
                .await
                .expect("append turn 2 (tool call + tool result)");
            conv_memory
                .append("conv-1", expected[5..6].to_vec())
                .await
                .expect("append turn 3");
            // `memory` and `conv_memory` are dropped here, at end of scope.
        }

        // "After restart": open a brand new `MemorySystem` against the same on-disk path —
        // nothing carries over in-process; every byte must come back off disk.
        let reopened_config = config;
        let memory_after_restart = open(&reopened_config);
        let conv_memory_after_restart =
            ShodhConversationMemory::new(memory_after_restart, "test-user");

        let loaded = conv_memory_after_restart
            .load("conv-1")
            .await
            .expect("load after restart");

        // Compare against `expected` *after* an independent `serde_json` round-trip, not the
        // freshly-constructed value. `rig_core::completion::message::Text::additional_params`
        // is `#[serde(flatten)] Option<serde_json::Value>` — serde_json's flatten mechanism
        // always deserializes an absent flattened remainder as `Some(Object {})`, never `None`
        // (confirmed empirically: this failed on first write when compared against the
        // freshly-constructed `expected`, before this normalization was added). That is a
        // property of `Message`'s own serde shape, present for *any* JSON-based storage of
        // it, not something this module introduces or is responsible for correcting. What
        // this module owns — and what this assertion actually verifies — is that `load()`
        // reproduces exactly what an independent `serde_json` round-trip of the original
        // message produces: no additional loss on top of `Message`'s own serde behavior.
        let expected_via_json: Vec<Message> = expected
            .iter()
            .map(|m| {
                let json = serde_json::to_string(m).expect("serialize expected message");
                serde_json::from_str(&json).expect("deserialize expected message")
            })
            .collect();

        assert_eq!(
            loaded, expected_via_json,
            "every message, including the tool call/result pair, must round-trip through a \
             restart exactly as it would through a bare serde_json round-trip — no additional \
             loss from storage"
        );
    }

    /// `clear()` must remove everything for a conversation, including the episode index rows
    /// (verified indirectly: a `load()` immediately after `clear()` must come back empty, not
    /// error or partially populated).
    #[tokio::test]
    async fn clear_removes_all_messages_for_the_conversation() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let config = config_at(temp_dir.path().to_path_buf());
        let memory = open(&config);
        let conv_memory = ShodhConversationMemory::new(memory, "test-user");

        conv_memory
            .append("conv-a", vec![Message::user("hello")])
            .await
            .expect("append");
        conv_memory
            .append("conv-b", vec![Message::user("hi there")])
            .await
            .expect("append other conversation");

        conv_memory.clear("conv-a").await.expect("clear");

        assert!(conv_memory.load("conv-a").await.expect("load a").is_empty());
        assert_eq!(
            conv_memory.load("conv-b").await.expect("load b").len(),
            1,
            "clearing one conversation must not touch another"
        );
    }
}
