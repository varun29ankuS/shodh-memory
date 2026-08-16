/**
 * Wire types, each mirroring a Rust struct that was read before it was typed.
 * The citation on each one is load-bearing: if a field is not listed in the
 * handler it is not listed here, and nothing is added speculatively.
 */

/** `RecallExperience` — src/handlers/types.rs:255 */
export interface RecallExperience {
  content: string;
  /** Debug rendering of a closed enum, Title-Cased — recall serialises
   *  `format!("{:?}", experience_type)` (src/handlers/recall.rs:822). The set is
   *  the one `/api/remember` validates against: Observation, Decision, Learning,
   *  Error, Discovery, Pattern, Context, Task, CodeEdit, FileAccess, Search,
   *  Command, Conversation, Intention. Typed as `string` because the wire type
   *  is `Option<String>` and a union here would turn a server-side enum
   *  addition into a compile error in the client. */
  memory_type: string | null;
  tags: string[];
  /** `[lat, lon, alt]` WGS84. Absent unless the memory carries coordinates —
   *  nothing in the Rust path derives them, they are caller-supplied only. */
  geo_location?: [number, number, number];
}

/**
 * `ScoreAttribution` — src/memory/types.rs.
 *
 * Present only when the request set `debug: true`. There is no endpoint to
 * fetch it for one memory afterwards; it is all-or-nothing per query.
 */
export interface ScoreAttribution {
  memory_id: string;
  rrf_base: number;
  graph_rrf: number;
  hybrid_rrf: number;
  hebbian_boost: number;
  attribute_boost: number;
  temporal_prefilter_boost: number;
  temporal_fact_boost: number;
  interference_adjustment: number;
  prospective_boost: number;
  fact_source_boost: number;
  ontological_boost: number;
  importance_factor: number;
  recency_factor: number;
  arousal_factor: number;
  credibility_factor: number;
  feedback_multiplier: number;
  quality_gate: number;
  final_score: number;
  /** Which retrieval legs contributed this memory. */
  sources: string[];
}

/** `RecallMemory` — src/handlers/types.rs:243 */
export interface RecallMemory {
  id: string;
  experience: RecallExperience;
  importance: number;
  created_at: string;
  score: number;
  /** Consolidation tier. `format!("{:?}", m.tier)` — the Debug rendering of
   *  `MemoryTier` (src/memory/types.rs:1047-1076), so the values are `Working`,
   *  `Session` and `LongTerm`, plus the retired-but-undeletable `Archive`.
   *  Populated at every `RecallMemory` construction site (src/handlers/recall.rs:830,
   *  :3172, :3581) and carries no `skip_serializing_if` (src/handlers/types.rs:249),
   *  so it is always present.
   *
   *  Typed as `string` rather than a union for the same reason `memory_type` is:
   *  the wire type is a `String`, and a union here would turn a server-side enum
   *  addition into a client compile error. `memoryTier()` in
   *  features/recall/tier.ts narrows it. */
  tier: string;
  score_attribution?: ScoreAttribution;
}

/**
 * `RetrievalStats` — src/memory/types.rs:3128. Only the fields this UI renders;
 * the struct also carries candidate counts, per-leg weights, hop counts and a
 * `stage_timings` breakdown, none of which belong on an analyst's first screen.
 *
 * Present whenever the request set `debug: true`, which this client always does
 * (lib/api/recall.ts) — src/handlers/recall.rs:601-611 populates it from
 * `recall_with_diagnostics` on that branch and passes `None` otherwise, and
 * src/handlers/types.rs:169 skips it when absent. Optional here for that reason.
 */
export interface RetrievalStats {
  /** Total time the server spent retrieving, in MICROseconds
   *  (src/memory/types.rs:3160-3161). This is the retrieval itself — it does not
   *  include the network hop, JSON encoding or anything the browser then does,
   *  so it must never be presented as an end-to-end figure. */
  retrieval_time_us: number;
}

/** `RecallFact` — src/handlers/types.rs */
export interface RecallFact {
  id: string;
  fact: string;
  confidence: number;
  support_count: number;
  related_entities: string[];
}

/** `RecallTodo` — src/handlers/types.rs */
export interface RecallTodo {
  id: string;
  short_id: string;
  content: string;
  status: string;
  priority: string;
  project: string | null;
  due_date: string | null;
  score: number;
}

/** `RecallLineageEdge` — src/handlers/types.rs. Causal edges between recalled
 *  memories: this is the trust chain Chain 2 walks. */
export interface RecallLineageEdge {
  from: string;
  to: string;
  relation: string;
  confidence: number;
}

/**
 * `RecallResponse` — src/handlers/types.rs:164.
 *
 * The collection fields carry `skip_serializing_if = "Vec::is_empty"` on the
 * Rust side, so they are genuinely absent rather than empty when there is
 * nothing to send. Every one is optional here for that reason.
 */
export interface RecallResponse {
  memories: RecallMemory[];
  count: number;
  retrieval_stats?: RetrievalStats;
  todos?: RecallTodo[];
  todo_count?: number;
  facts?: RecallFact[];
  fact_count?: number;
  lineage?: RecallLineageEdge[];
  lineage_count?: number;
}

/** `RecallRequest` — src/handlers/types.rs. Only the fields this UI sends. */
export interface RecallRequest {
  user_id: string;
  query: string;
  limit?: number;
  mode?: "hybrid" | "semantic" | "associative" | "temporal" | "spatial";
  /** Produces `score_attribution` on every returned memory. */
  debug?: boolean;
  offset?: number;
}

/** `TodoStatus` — src/memory/types.rs:3406. `#[serde(rename_all =
 *  "snake_case")]`, so the wire values are these, not the PascalCase variant
 *  names. */
export type TodoStatus = "backlog" | "todo" | "in_progress" | "blocked" | "done" | "cancelled";

/** `TodoPriority` — src/memory/types.rs:3452. Same snake_case rule. */
export type TodoPriority = "urgent" | "high" | "medium" | "low" | "none";

/**
 * `Todo` — src/memory/types.rs:3646. Only the fields this UI renders; the
 * struct also carries `recurrence`, `comments` and `related_memory_ids`, none
 * of which the list view needs.
 *
 * The struct's 384-float `embedding` is deliberately absent from the wire: the
 * todo response types strip it (src/handlers/todos.rs, `todo_wire`), which cut
 * a 50-todo list from 287KB to 52KB. It is still persisted on the stored
 * record, so do not expect it in any API response.
 *
 * `id` is `#[serde(transparent)]` over a `Uuid` (types.rs:3358-3361), so it
 * serialises as a bare string. There is no `short_id` field on the wire —
 * `Todo::short_id()` (types.rs:3776) is a Rust-side method, not part of the
 * JSON. The display id is composed client-side from `project_prefix` +
 * `seq_num`, mirroring that method exactly (including its "SHO" fallback and
 * its first-4-hex-chars fallback for legacy todos with `seq_num === 0`).
 */
export interface Todo {
  id: string;
  seq_num: number;
  project_prefix: string | null;
  user_id: string;
  content: string;
  status: TodoStatus;
  priority: TodoPriority;
  project_id: string | null;
  parent_id: string | null;
  contexts: string[];
  tags: string[];
  due_date: string | null;
  blocked_on: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  sort_order: number;
}

/** `Project` — src/memory/types.rs:3911. Only the fields this UI renders. */
export interface Project {
  id: string;
  user_id: string;
  name: string;
  prefix: string | null;
  color: string | null;
}

/** `ListTodosRequest` — src/handlers/todos.rs:223. Only the fields this UI
 *  sends; `include_completed` defaults to `false` server-side, which is what
 *  the Tasks view wants for its primary list. */
export interface ListTodosRequest {
  user_id: string;
  status?: TodoStatus[];
  include_completed?: boolean;
  limit?: number;
}

/** `TodoListResponse` — src/handlers/todos.rs:204. `count` is the total
 *  before pagination, not `todos.length` — src/handlers/todos.rs:1335-1358
 *  sets it from `todos.len()` before `.truncate(limit)` runs. */
export interface TodoListResponse {
  success: boolean;
  count: number;
  todos: Todo[];
  projects: Project[];
  formatted: string;
}
