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

use rig_core::completion::{AssistantContent, Message, ToolResultContent, UserContent};
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
            format!("[tool_call {}({})]", call.function.name, call.function.arguments)
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
                    let raw = mem.experience.metadata.get(MESSAGE_JSON_KEY).ok_or_else(|| {
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
                let mut next_seq = existing.len() as u32;
                let mut preceding_memory_id = existing.last().map(|m| m.id.0.to_string());

                for message in messages {
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
                    next_seq += 1;
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
