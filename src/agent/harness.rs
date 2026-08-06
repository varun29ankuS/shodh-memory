//! Continual Harness state, backed by shodh-memory: four entry kinds (`prompt`, `memory`,
//! `skill`, `subagent`) plus a refinement-event log, with CRUD, scoping, and a
//! `render_for_prompt` that selects entries by relevance rather than by a fixed
//! alphabetical slice. See `src/agent/mod.rs` for why this exists.
//!
//! ## Storage mapping
//!
//! Each harness entry is one shodh-memory `Memory`, created/updated through
//! `MemorySystem::upsert()` keyed by a caller-stable (or generated) string id used
//! directly as the shodh `external_id` — `upsert` already gives create-or-update-with-
//! history semantics for free (mod.rs `impl MemorySystem::upsert`), which is exactly what
//! mutable harness entries need and plain `remember()` (content-addressed, immutable) does
//! not provide.
//!
//! Scoping uses a *single compound tag* `harness:{scope}:{kind}` as the source of truth for
//! "every entry of this kind in this scope", looked up via the exact tag index
//! (`SearchCriteria::ByTags`) — not `Query.tags` on `recall()`, which (see
//! `conversation_memory.rs` module docs) is plumbed through the retrieval pipeline but never
//! enforced as a filter. `ByTags` matches ANY of the given tags (OR, not AND) — see
//! `storage.rs::search_by_tags` — so encoding `(scope, kind)` as one compound tag avoids
//! needing `SearchCriteria::Combined`'s AND semantics for what is otherwise a simple
//! equality lookup. Separate `harness-kind:{kind}` / `harness-scope:{scope}` tags are also
//! written, for operator debugging via the existing `recall_by_tags` / `forget` admin paths,
//! but the compound tag is what this module's own scoping relies on.
//!
//! Refinement events are plain (immutable, content-addressed) `remember()`s tagged with a
//! `harness-event-scope:{scope}` compound tag, filtered by `target_id` client-side after
//! fetch — refinement-event volume for a harness is expected to be modest, not a firehose,
//! so this is not the same over-fetch-then-intersect concern `render_for_prompt` has to
//! solve for relevance ranking.
//!
//! ## Relevance selection (`render_for_prompt`)
//!
//! For each kind, the exact candidate set for `(scope, kind)` is fetched via the compound
//! tag (cheap, exact, no embedding). Separately, `MemorySystem::recall()` — the same
//! multi-layer (vector + graph + BM25 + rerank) pipeline `src/handlers/recall.rs` calls for
//! every `/api/recall` request — is run against the current query text, with `max_results`
//! sized off the candidate count so the harness's (typically small) candidate set has a real
//! chance of surfacing inside the pipeline's own ranked window. The final shown list is the
//! intersection of "is a candidate for this (scope, kind)" and "appeared in the ranked
//! results", taken in the pipeline's own rank order and cut at the configured per-kind
//! budget. Three counts are reported, not one: how many candidates exist in total, how many
//! of those were ranked at all by `recall()` for this query, and — of those ranked — how
//! many didn't make the budget. `render_for_prompt`'s prose output states all three, instead
//! of hiding the gap the way prime-agent's `formatHarnessStateForPrompt` does (it prints the
//! true total next to a silently-truncated alphabetical slice).

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::memory::storage::SearchCriteria;
use crate::memory::types::{ChangeType, Experience, ExperienceType, ForgetCriteria, Memory, Query};
use crate::memory::MemorySystem;

/// Default number of entries shown per kind by [`ContinualHarnessStore::render_for_prompt`]
/// when the caller doesn't override [`RenderBudget`]. Matches prime-agent's
/// `DEFAULT_OVERVIEW_ENTRY_LIMIT` (6) for a like-for-like comparison — the number is the
/// same, but here it bounds a relevance-ranked selection instead of an alphabetical slice,
/// and it's a config knob rather than a hardcoded constant in the render path.
pub const DEFAULT_PER_KIND_ENTRY_BUDGET: usize = 6;

/// Hard ceiling on how many results a single `render_for_prompt` kind-lookup will ask
/// `recall()` for, regardless of how large a scope's candidate set grows. Protects the
/// pipeline from an unbounded over-fetch if a harness scope accumulates thousands of
/// entries; a harness realistically holds dozens, not thousands.
const HARNESS_RECALL_MAX_RESULTS: usize = 200;

/// One of the four Continual Harness entry kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessKind {
    Prompt,
    Memory,
    Skill,
    Subagent,
}

/// Fixed rendering order for [`ContinualHarnessStore::render_for_prompt`]. Order of the
/// *kinds* is fixed for predictable prompt layout; order of *entries within* a kind is the
/// whole point of this module and is never fixed.
pub const ALL_HARNESS_KINDS: [HarnessKind; 4] = [
    HarnessKind::Prompt,
    HarnessKind::Memory,
    HarnessKind::Skill,
    HarnessKind::Subagent,
];

impl HarnessKind {
    pub fn as_str(self) -> &'static str {
        match self {
            HarnessKind::Prompt => "prompt",
            HarnessKind::Memory => "memory",
            HarnessKind::Skill => "skill",
            HarnessKind::Subagent => "subagent",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "prompt" => Ok(HarnessKind::Prompt),
            "memory" => Ok(HarnessKind::Memory),
            "skill" => Ok(HarnessKind::Skill),
            "subagent" => Ok(HarnessKind::Subagent),
            other => bail!("unknown harness entry kind: '{other}'"),
        }
    }
}

impl std::fmt::Display for HarnessKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

fn compound_tag(scope: &str, kind: HarnessKind) -> String {
    format!("harness:{scope}:{}", kind.as_str())
}

fn event_scope_tag(scope: &str) -> String {
    format!("harness-event-scope:{scope}")
}

const META_KIND: &str = "harness_kind";
const META_SCOPE: &str = "harness_scope";
const META_TITLE: &str = "harness_title";
const META_PATH: &str = "harness_path";
const META_EVENT_TARGET_KIND: &str = "harness_event_target_kind";
const META_EVENT_TARGET_ID: &str = "harness_event_target_id";

/// A stored Continual Harness entry (one of `prompt` / `memory` / `skill` / `subagent`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessEntry {
    /// Stable id, also the shodh `external_id` used for upsert/lookup/delete.
    pub id: String,
    pub kind: HarnessKind,
    pub scope: String,
    pub title: String,
    /// Free-form hierarchical path, for parity with prime-agent's `(path, title, id)` sort
    /// key and for display; defaults to `title` when not supplied.
    pub path: String,
    pub content: String,
    pub version: u32,
    pub updated_at: DateTime<Utc>,
}

/// Input to [`ContinualHarnessStore::upsert_entry`].
#[derive(Debug, Clone)]
pub struct HarnessEntryDraft {
    /// `None` generates a new id (create-only). `Some` upserts: creates if the id is new,
    /// updates (with history tracking) if it already exists.
    pub id: Option<String>,
    pub kind: HarnessKind,
    pub scope: String,
    pub title: String,
    pub path: Option<String>,
    pub content: String,
}

/// A logged refinement action against a harness entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementEvent {
    pub id: String,
    pub scope: String,
    pub target_kind: HarnessKind,
    pub target_id: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
}

/// One entry selected for the prompt, with the relevance score `recall()` assigned it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderedEntry {
    pub id: String,
    pub kind: HarnessKind,
    pub title: String,
    pub path: String,
    pub content: String,
    pub relevance_score: f32,
}

/// Relevance-ranked rendering of a single kind, with an honest breakdown of what wasn't
/// shown and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KindRender {
    pub kind: HarnessKind,
    /// Entries selected for the prompt, in descending relevance order, `shown.len() <=`
    /// the configured budget.
    pub shown: Vec<RenderedEntry>,
    /// Every entry that exists for this `(scope, kind)`, regardless of relevance.
    pub total_candidates: usize,
    /// Of `total_candidates`, how many `recall()` actually returned in its ranked window
    /// for this query (i.e. were scored at all).
    pub scored_candidates: usize,
    /// Scored but not shown — ranked below the per-kind budget cutoff.
    pub omitted_below_budget: usize,
    /// Never scored — did not appear in `recall()`'s ranked window for this query at all
    /// (irrelevant to the query, or lost to the pipeline's own internal candidate-pool
    /// sizing before the harness's `max_results` request kicks in).
    pub omitted_below_cutoff: usize,
}

impl KindRender {
    /// Render this kind's block for prompt injection. Always states the true total and,
    /// when anything was left out, splits the remainder by cause — this is the honesty
    /// requirement this module exists to satisfy.
    pub fn to_prompt_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "## {} ({} of {} shown, by relevance)\n",
            capitalize(self.kind.as_str()),
            self.shown.len(),
            self.total_candidates
        ));

        if self.shown.is_empty() {
            out.push_str("(none relevant to the current query)\n");
        } else {
            for entry in &self.shown {
                out.push_str(&format!(
                    "- [{}] {} (relevance {:.2})\n  {}\n",
                    entry.id,
                    entry.title,
                    entry.relevance_score,
                    truncate_for_display(&entry.content, 240)
                ));
            }
        }

        if self.omitted_below_budget > 0 || self.omitted_below_cutoff > 0 {
            out.push_str(&format!(
                "({} more {} entries exist for this scope: {} ranked below the top-{} shown here, {} did not surface for this query (scored below the retrieval cutoff).)\n",
                self.omitted_below_budget + self.omitted_below_cutoff,
                self.kind.as_str(),
                self.omitted_below_budget,
                self.shown.len(),
                self.omitted_below_cutoff,
            ));
        }

        out
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn truncate_for_display(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars).collect();
    format!("{truncated}…")
}

/// The full rendered harness state, in fixed kind order (see [`ALL_HARNESS_KINDS`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderedHarnessState {
    pub kinds: Vec<KindRender>,
}

impl RenderedHarnessState {
    pub fn to_prompt_text(&self) -> String {
        self.kinds
            .iter()
            .map(KindRender::to_prompt_text)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Per-kind entry budget for [`ContinualHarnessStore::render_for_prompt`].
#[derive(Debug, Clone, Copy)]
pub struct RenderBudget {
    pub per_kind_limit: usize,
}

impl Default for RenderBudget {
    fn default() -> Self {
        Self {
            per_kind_limit: DEFAULT_PER_KIND_ENTRY_BUDGET,
        }
    }
}

/// CRUD + scoping + relevance-ranked rendering for Continual Harness state, backed by
/// shodh-memory. See the module docs for the storage mapping and selection algorithm.
pub struct ContinualHarnessStore {
    memory: Arc<RwLock<MemorySystem>>,
    user_id: String,
}

impl ContinualHarnessStore {
    /// Build a harness store scoped to one shodh-memory user. `memory` is the same
    /// per-user handle used elsewhere (see `ShodhConversationMemory::new`).
    pub fn new(memory: Arc<RwLock<MemorySystem>>, user_id: impl Into<String>) -> Self {
        Self {
            memory,
            user_id: user_id.into(),
        }
    }

    /// Create a new entry, or update an existing one if `draft.id` names one that already
    /// exists (shodh `upsert()` semantics — content is versioned with audit history on
    /// update). Returns `(id, was_update)`.
    pub fn upsert_entry(&self, draft: HarnessEntryDraft) -> Result<(String, bool)> {
        if draft.title.trim().is_empty() {
            bail!("harness entry title must not be empty");
        }
        if draft.content.trim().is_empty() {
            bail!("harness entry content must not be empty");
        }
        if draft.scope.trim().is_empty() {
            bail!("harness entry scope must not be empty");
        }

        let id = draft
            .id
            .unwrap_or_else(|| format!("harness-{}", uuid::Uuid::new_v4()));
        let path = draft.path.unwrap_or_else(|| draft.title.clone());

        let mut metadata = HashMap::new();
        metadata.insert(META_KIND.to_string(), draft.kind.as_str().to_string());
        metadata.insert(META_SCOPE.to_string(), draft.scope.clone());
        metadata.insert(META_TITLE.to_string(), draft.title.clone());
        metadata.insert(META_PATH.to_string(), path);

        let experience = Experience {
            experience_type: ExperienceType::Context,
            content: draft.content,
            metadata,
            tags: vec![
                "shodh-harness".to_string(),
                format!("harness-kind:{}", draft.kind.as_str()),
                format!("harness-scope:{}", draft.scope),
                compound_tag(&draft.scope, draft.kind),
            ],
            ..Default::default()
        };

        // `upsert()` only consults `change_type` on its UPDATE path (it labels the
        // `MemoryRevision` pushed into history); the CREATE path ignores the argument
        // entirely. `ContentUpdated` is therefore correct in both cases without needing a
        // racy pre-check of whether the id already exists.
        let memory_guard = self.memory.read();
        let (_memory_id, was_update) = memory_guard
            .upsert(
                id.clone(),
                experience,
                ChangeType::ContentUpdated,
                Some(format!("agent-harness:{}", self.user_id)),
                None,
            )
            .with_context(|| format!("failed to upsert harness entry '{id}'"))?;

        Ok((id, was_update))
    }

    /// Fetch a single entry by id, if it exists.
    pub fn get_entry(&self, id: &str) -> Result<Option<HarnessEntry>> {
        let memory_guard = self.memory.read();
        match memory_guard.find_by_external_id(id)? {
            Some(mem) => Ok(Some(harness_entry_from_memory(&mem)?)),
            None => Ok(None),
        }
    }

    /// Delete an entry by id. Returns `true` if an entry was found and deleted.
    pub fn delete_entry(&self, id: &str) -> Result<bool> {
        let memory_guard = self.memory.read();
        let Some(mem) = memory_guard.find_by_external_id(id)? else {
            return Ok(false);
        };
        memory_guard
            .forget(ForgetCriteria::ById(mem.id))
            .with_context(|| format!("failed to delete harness entry '{id}'"))?;
        Ok(true)
    }

    /// List every entry for `(scope, kind)`, in a stable — but not relevance — order:
    /// `(path, title, id)`, matching prime-agent's own sort key. This is an admin/debug
    /// listing, not a selection policy: use [`Self::render_for_prompt`] to choose what to
    /// show a model.
    pub fn list_entries(&self, scope: &str, kind: HarnessKind) -> Result<Vec<HarnessEntry>> {
        let memory_guard = self.memory.read();
        let stored = memory_guard.advanced_search(SearchCriteria::ByTags(vec![compound_tag(
            scope, kind,
        )]))?;
        let mut entries = stored
            .iter()
            .map(harness_entry_from_memory)
            .collect::<Result<Vec<_>>>()?;
        entries.sort_by(|a, b| (&a.path, &a.title, &a.id).cmp(&(&b.path, &b.title, &b.id)));
        Ok(entries)
    }

    /// Record a refinement event against `target_id` (assumed to be an entry of
    /// `target_kind` in `scope`, though this is not enforced — the event log is a record of
    /// what happened, not a foreign-key-checked ledger). Returns the event id.
    pub fn record_refinement(
        &self,
        scope: &str,
        target_kind: HarnessKind,
        target_id: &str,
        description: &str,
    ) -> Result<String> {
        if description.trim().is_empty() {
            bail!("refinement event description must not be empty");
        }

        let mut metadata = HashMap::new();
        metadata.insert(META_SCOPE.to_string(), scope.to_string());
        metadata.insert(
            META_EVENT_TARGET_KIND.to_string(),
            target_kind.as_str().to_string(),
        );
        metadata.insert(META_EVENT_TARGET_ID.to_string(), target_id.to_string());

        let experience = Experience {
            experience_type: ExperienceType::Learning,
            content: description.to_string(),
            metadata,
            tags: vec![
                "shodh-harness-event".to_string(),
                event_scope_tag(scope),
            ],
            ..Default::default()
        };

        let memory_guard = self.memory.read();
        let memory_id = memory_guard
            .remember(experience, None)
            .context("failed to record refinement event")?;
        Ok(memory_id.0.to_string())
    }

    /// List refinement events for `scope`, optionally filtered to one `target_id`, newest
    /// first, capped at `limit`. Returns `(events, total_matching_before_limit)` — the
    /// count is never hidden, matching this module's honesty requirement.
    pub fn list_refinement_events(
        &self,
        scope: &str,
        target_id: Option<&str>,
        limit: usize,
    ) -> Result<(Vec<RefinementEvent>, usize)> {
        let memory_guard = self.memory.read();
        let stored = memory_guard.advanced_search(SearchCriteria::ByTags(vec![event_scope_tag(
            scope,
        )]))?;

        let mut events = stored
            .iter()
            .filter(|mem| {
                target_id.is_none_or(|tid| {
                    mem.experience
                        .metadata
                        .get(META_EVENT_TARGET_ID)
                        .map(|v| v == tid)
                        .unwrap_or(false)
                })
            })
            .map(refinement_event_from_memory)
            .collect::<Result<Vec<_>>>()?;

        events.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        let total = events.len();
        events.truncate(limit);
        Ok((events, total))
    }

    /// Select and render harness state for injection into the current turn's prompt,
    /// ranking each kind's entries by relevance to `query_text` via `MemorySystem::recall()`
    /// — see module docs for the full algorithm and the honesty guarantees on truncation.
    pub fn render_for_prompt(
        &self,
        scope: &str,
        query_text: &str,
        budget: RenderBudget,
    ) -> Result<RenderedHarnessState> {
        if query_text.trim().is_empty() {
            bail!("render_for_prompt requires a non-empty query_text");
        }

        let mut kinds = Vec::with_capacity(ALL_HARNESS_KINDS.len());
        for kind in ALL_HARNESS_KINDS {
            kinds.push(self.render_kind(scope, kind, query_text, budget.per_kind_limit)?);
        }
        Ok(RenderedHarnessState { kinds })
    }

    fn render_kind(
        &self,
        scope: &str,
        kind: HarnessKind,
        query_text: &str,
        per_kind_limit: usize,
    ) -> Result<KindRender> {
        let memory_guard = self.memory.read();

        // Exact candidate set for (scope, kind) — the source of truth for what belongs to
        // this kind. `recall()` below is used purely for ranking, not for scoping.
        let candidates =
            memory_guard.advanced_search(SearchCriteria::ByTags(vec![compound_tag(scope, kind)]))?;
        let total_candidates = candidates.len();

        if total_candidates == 0 {
            return Ok(KindRender {
                kind,
                shown: Vec::new(),
                total_candidates: 0,
                scored_candidates: 0,
                omitted_below_budget: 0,
                omitted_below_cutoff: 0,
            });
        }

        let candidate_ids: HashSet<_> = candidates.iter().map(|m| m.id.clone()).collect();

        // Over-fetch proportional to the candidate set, floored so small budgets still get
        // a wide-enough ranked window, capped so a large scope can't blow up the pipeline.
        let floor = per_kind_limit.saturating_mul(4).max(1);
        let raw = total_candidates.saturating_mul(2).max(floor);
        let max_results = raw.min(HARNESS_RECALL_MAX_RESULTS.max(floor));

        let query = Query {
            user_id: Some(self.user_id.clone()),
            query_text: Some(query_text.to_string()),
            max_results,
            ..Default::default()
        };

        // The real retrieval engine: vector + graph + BM25 + rerank fusion, exactly as
        // `src/handlers/recall.rs` calls it for `/api/recall`. This module does not modify
        // its ranking behavior in any way.
        let ranked = memory_guard
            .recall(&query)
            .with_context(|| format!("recall() failed while rendering harness kind '{kind}'"))?;

        let matched_in_rank_order: Vec<_> = ranked
            .iter()
            .filter(|m| candidate_ids.contains(&m.id))
            .collect();
        let scored_candidates = matched_in_rank_order.len();

        let candidates_by_id: HashMap<_, &Memory> =
            candidates.iter().map(|m| (m.id.clone(), m)).collect();

        let shown: Vec<RenderedEntry> = matched_in_rank_order
            .iter()
            .take(per_kind_limit)
            .filter_map(|m| {
                let source = candidates_by_id.get(&m.id)?;
                let entry = harness_entry_from_memory(source).ok()?;
                Some(RenderedEntry {
                    id: entry.id,
                    kind: entry.kind,
                    title: entry.title,
                    path: entry.path,
                    content: entry.content,
                    relevance_score: m.score.unwrap_or(0.0),
                })
            })
            .collect();

        let omitted_below_budget = scored_candidates.saturating_sub(shown.len());
        let omitted_below_cutoff = total_candidates.saturating_sub(scored_candidates);

        Ok(KindRender {
            kind,
            shown,
            total_candidates,
            scored_candidates,
            omitted_below_budget,
            omitted_below_cutoff,
        })
    }
}

fn harness_entry_from_memory(mem: &Memory) -> Result<HarnessEntry> {
    let id = mem
        .external_id
        .clone()
        .ok_or_else(|| anyhow::anyhow!("harness memory {} is missing external_id", mem.id.0))?;
    let kind_str = mem
        .experience
        .metadata
        .get(META_KIND)
        .ok_or_else(|| anyhow::anyhow!("harness memory {id} is missing {META_KIND} metadata"))?;
    let kind = HarnessKind::parse(kind_str)?;
    let scope = mem
        .experience
        .metadata
        .get(META_SCOPE)
        .cloned()
        .unwrap_or_default();
    let title = mem
        .experience
        .metadata
        .get(META_TITLE)
        .cloned()
        .unwrap_or_default();
    let path = mem
        .experience
        .metadata
        .get(META_PATH)
        .cloned()
        .unwrap_or_else(|| title.clone());

    // `upsert()`'s update path never touches `Memory.created_at` — it only appends a
    // `MemoryRevision` to `history` (mod.rs `impl MemorySystem::upsert`). The most recent
    // revision's `changed_at` is the true "last updated" time; an entry that has never been
    // updated has no history, so its creation time is correct.
    let updated_at = mem
        .history
        .last()
        .map(|revision| revision.changed_at)
        .unwrap_or(mem.created_at);

    Ok(HarnessEntry {
        id,
        kind,
        scope,
        title,
        path,
        content: mem.experience.content.clone(),
        version: mem.version,
        updated_at,
    })
}

fn refinement_event_from_memory(mem: &Memory) -> Result<RefinementEvent> {
    let scope = mem
        .experience
        .metadata
        .get(META_SCOPE)
        .cloned()
        .unwrap_or_default();
    let target_kind_str = mem
        .experience
        .metadata
        .get(META_EVENT_TARGET_KIND)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "refinement event {} is missing {META_EVENT_TARGET_KIND} metadata",
                mem.id.0
            )
        })?;
    let target_kind = HarnessKind::parse(target_kind_str)?;
    let target_id = mem
        .experience
        .metadata
        .get(META_EVENT_TARGET_ID)
        .cloned()
        .unwrap_or_default();

    Ok(RefinementEvent {
        id: mem.id.0.to_string(),
        scope,
        target_kind,
        target_id,
        description: mem.experience.content.clone(),
        created_at: mem.created_at,
    })
}
