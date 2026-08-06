//! Agent-loop foundation: a `rig-core`-compatible `ConversationMemory` backend and a
//! shodh-memory-backed "Continual Harness" state store.
//!
//! # Why this exists
//!
//! `PrimeIntellect-ai/prime-agent`'s Continual Harness persists four kinds of
//! self-refining agent state (`prompt`, `memory`, `skill`, `subagent`) plus a log of
//! refinement events in a flat `harness_state.json`, and surfaces that state to the model
//! with `formatHarnessStateForPrompt`: `localeCompare` on `(path, title, id)`, then
//! `slice(0, 6)`. Six entries per kind, alphabetically, regardless of relevance to the
//! current turn — while printing the *true* entry count next to the truncated list. That
//! is the exact failure mode this module avoids: [`harness::ContinualHarnessStore`] selects
//! entries by relevance to the current query using shodh-memory's real retrieval pipeline
//! (vector + graph + BM25 + rerank fusion via [`crate::memory::MemorySystem::recall`]), and
//! when the configured budget still leaves entries unshown, it says so — split by *why*
//! they were left out (ranked below the per-kind budget vs. not returned by retrieval at
//! all for this query), not just a bare, uninterpretable count.
//!
//! # Scope of this module (foundation slice)
//!
//! This module provides exactly two things:
//!
//! 1. [`conversation_memory::ShodhConversationMemory`] — an implementation of
//!    `rig_core::memory::ConversationMemory` that persists conversation turns as
//!    shodh-memory `Memory` records instead of an in-process `HashMap` (rig's own
//!    `InMemoryConversationMemory` default), so history survives a process restart.
//! 2. [`harness::ContinualHarnessStore`] — CRUD, scoping, and relevance-ranked rendering
//!    for the four Continual Harness entry kinds plus refinement events, likewise backed by
//!    shodh-memory rather than a JSON file.
//!
//! Deliberately **not** built here (see the crate's task tracker / PR description for the
//! follow-on slices): the AG-UI SSE endpoint, the agent turn loop / tool dispatch / provider
//! wiring, and any code-execution substrate. This module also does not modify
//! `MemorySystem::recall`'s ranking behavior in any way — it is a consumer of the existing
//! pipeline, called through its public API exactly as `src/handlers/recall.rs` calls it.
//!
//! # Feature gate
//!
//! Everything here lives behind the `agent-harness` Cargo feature (see `Cargo.toml`), which
//! pulls in `rig-core` — pinned to an exact version, not `^`, because rig has shipped 48
//! breaking-change entries across its last two releases in ~40 days under a single dominant
//! maintainer. It is not part of the edge-device `default` feature set.

pub mod conversation_memory;
pub mod harness;

pub use conversation_memory::ShodhConversationMemory;
pub use harness::{
    ContinualHarnessStore, HarnessEntry, HarnessEntryDraft, HarnessKind, KindRender,
    RefinementEvent, RenderBudget, RenderedEntry, RenderedHarnessState,
};
