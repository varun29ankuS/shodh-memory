/**
 * MCP tool annotations and output schemas — single source of truth.
 *
 * Two protocol affordances live here:
 *
 * 1. `annotations` (MCP `ToolAnnotations`): behavioural hints a client uses to
 *    decide what may be auto-approved. `forget`, `purge_facts`, `delete_todo`,
 *    `backup_restore` and friends must be distinguishable from `recall` and
 *    `memory_stats`, or a harness has to choose between prompting on
 *    everything (friction) or running everything (data-loss hazard).
 *
 * 2. `outputSchema` (MCP `Tool.outputSchema`): a machine-readable channel next
 *    to the emoji-formatted text, so a model does not have to scrape layout to
 *    get a memory id or a score. Text is NOT replaced — the spec expects
 *    `content` to stay populated, and the formatted text is genuinely better
 *    for a small result read inline.
 *
 * IMPORTANT — the read-only set is enforced, not merely asserted.
 * `index.ts` gates its two ambient-ingestion paths (`autoStreamContext` and
 * `streamToolCall`) on `isReadOnlyTool()`. Without that gate every tool call
 * with a >=50-char argument or result streams a memory to the backend, which
 * would make `readOnlyHint: true` a lie on every read tool. Adding a tool to
 * READ_ONLY here is therefore a behavioural commitment: that tool will not
 * ambient-ingest. Verify the handler before moving a tool into it.
 *
 * Spec defaults that bite (per @modelcontextprotocol/sdk `ToolAnnotations`):
 *   - `destructiveHint` defaults to TRUE when absent
 *   - `openWorldHint`  defaults to TRUE when absent
 * so both are set explicitly on every tool. The memory store is a closed local
 * world, so `openWorldHint` is false everywhere.
 */

/** MCP `ToolAnnotations` — mirrors the SDK interface (sdk/spec.types.d.ts). */
export interface ToolAnnotations {
  title?: string;
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
}

/** MCP `Tool.outputSchema` — JSON Schema, restricted to an object at the root. */
export interface JsonSchemaObject {
  type: "object";
  properties?: Record<string, unknown>;
  required?: string[];
  [key: string]: unknown;
}

/** Minimal shape of a tool definition in the `tools/list` literal. */
export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: { type: "object"; properties?: Record<string, unknown>; required?: string[] };
}

/** A tool definition decorated with its protocol affordances. */
export interface DecoratedTool extends ToolDefinition {
  annotations: ToolAnnotations;
  outputSchema?: JsonSchemaObject;
}

// =============================================================================
// ANNOTATIONS
// =============================================================================

type BehaviourSpec =
  | { title: string; readOnly: true }
  | { title: string; readOnly: false; destructive: boolean; idempotent: boolean };

/**
 * Per-tool behaviour, verified against the handler in `index.ts` and the Rust
 * route it calls. Reasoning is recorded inline only where the honest answer
 * differs from what the tool name suggests.
 */
const TOOL_BEHAVIOUR: Record<string, BehaviourSpec> = {
  // ---------------------------------------------------------------------
  // Core memory
  // ---------------------------------------------------------------------
  remember: { title: "Store Memory", readOnly: false, destructive: false, idempotent: false },
  recall: { title: "Recall Memories & Todos", readOnly: true },
  recall_by_tags: { title: "Recall by Tags", readOnly: true },
  context_summary: { title: "Session Context Summary", readOnly: true },
  list_memories: { title: "List Memories", readOnly: true },
  read_memory: { title: "Read Full Memory", readOnly: true },
  // Deletes a memory. Repeating with the same id leaves the store in the same
  // state (the second call 404s without further effect) => idempotent.
  forget: { title: "Delete Memory", readOnly: false, destructive: true, idempotent: true },
  memory_stats: { title: "Memory Statistics", readOnly: true },

  // ---------------------------------------------------------------------
  // Index integrity
  // ---------------------------------------------------------------------
  // POST, but `verify_index_integrity` (src/handlers/…) takes a read lock and
  // returns a report — no mutation. POST is transport, not semantics.
  verify_index: { title: "Verify Vector Index", readOnly: true },
  // Re-indexes orphaned memories: adds missing index entries, removes nothing.
  // Running it again once healthy is a no-op.
  repair_index: { title: "Repair Vector Index", readOnly: false, destructive: false, idempotent: true },

  // ---------------------------------------------------------------------
  // Backup & restore
  // ---------------------------------------------------------------------
  // Each call writes a new backup artifact => not idempotent, but additive.
  backup_create: { title: "Create Backup", readOnly: false, destructive: false, idempotent: false },
  backup_list: { title: "List Backups", readOnly: true },
  // Checksum comparison only.
  backup_verify: { title: "Verify Backup Integrity", readOnly: true },
  backup_purge: { title: "Purge Old Backups", readOnly: false, destructive: true, idempotent: true },
  // Replaces all current user data with the backup contents.
  backup_restore: { title: "Restore Backup", readOnly: false, destructive: true, idempotent: true },

  // ---------------------------------------------------------------------
  // Consolidation / session / facts
  // ---------------------------------------------------------------------
  consolidation_report: { title: "Consolidation Report", readOnly: true },
  // NOT read-only despite reading like a query: `auto_ingest` defaults to true,
  // so the default call stores the supplied context as a Conversation memory.
  proactive_context: { title: "Surface Proactive Context", readOnly: false, destructive: false, idempotent: false },
  token_status: { title: "MCP Token Throughput", readOnly: true },
  // NOT read-only despite the name: POST /api/sessions/context-compressed
  // persists a session-digest Context memory on every call (sessions.rs
  // `context_compressed` -> `build_active_digest` -> store).
  reset_token_session: { title: "Reset Token Session Counter", readOnly: false, destructive: false, idempotent: false },
  session_digest: { title: "Session Digest", readOnly: true },
  session_history: { title: "Session History", readOnly: true },
  fact_narratives: { title: "Fact Narratives", readOnly: true },
  // dry_run=true is read-only, but annotations are static per tool, so the
  // hint describes the worst case: the default (dry_run=false) deletes facts.
  purge_facts: { title: "Purge Facts by Pattern", readOnly: false, destructive: true, idempotent: true },

  // ---------------------------------------------------------------------
  // Reminders
  // ---------------------------------------------------------------------
  set_reminder: { title: "Set Reminder", readOnly: false, destructive: false, idempotent: false },
  list_reminders: { title: "List Reminders", readOnly: true },
  // Status transition pending -> dismissed; re-dismissing changes nothing.
  dismiss_reminder: { title: "Dismiss Reminder", readOnly: false, destructive: false, idempotent: true },

  // ---------------------------------------------------------------------
  // Todos & projects
  // ---------------------------------------------------------------------
  add_todo: { title: "Add Todo", readOnly: false, destructive: false, idempotent: false },
  list_todos: { title: "List / Search Todos", readOnly: true },
  // Overwrites existing field values rather than only appending => destructive
  // in the spec's "not purely additive" sense; same args converge on one state.
  // Still idempotent with status="done": settlement fires only on the
  // transition into a settled state, so a repeat call spawns no second
  // recurrence occurrence (src/handlers/todos.rs, `settling`).
  update_todo: { title: "Update Todo", readOnly: false, destructive: true, idempotent: true },
  // NOT idempotent: `TodoStore::complete_todo` re-completes unconditionally,
  // and each call spawns another recurrence occurrence plus another completion
  // memory and activity entry (src/memory/todos.rs, src/handlers/todos.rs).
  complete_todo: { title: "Complete Todo", readOnly: false, destructive: false, idempotent: false },
  delete_todo: { title: "Delete Todo", readOnly: false, destructive: true, idempotent: true },
  // Each call shifts the todo one more position => repeating has further effect.
  reorder_todo: { title: "Reorder Todo", readOnly: false, destructive: false, idempotent: false },
  // `create_project` mints a fresh UUID per call with no name dedup, so calling
  // twice yields two projects.
  add_project: { title: "Add Project", readOnly: false, destructive: false, idempotent: false },
  list_projects: { title: "List Projects", readOnly: true },
  // Hides but preserves; restorable => not destructive.
  archive_project: { title: "Archive Project", readOnly: false, destructive: false, idempotent: true },
  delete_project: { title: "Delete Project", readOnly: false, destructive: true, idempotent: true },
  todo_stats: { title: "Todo Statistics", readOnly: true },
  list_subtasks: { title: "List Subtasks", readOnly: true },
  add_todo_comment: { title: "Add Todo Comment", readOnly: false, destructive: false, idempotent: false },
  list_todo_comments: { title: "List Todo Comments", readOnly: true },
  update_todo_comment: { title: "Update Todo Comment", readOnly: false, destructive: true, idempotent: true },
  delete_todo_comment: { title: "Delete Todo Comment", readOnly: false, destructive: true, idempotent: true },

  // ---------------------------------------------------------------------
  // Causal lineage / knowledge graph / anomalies / facts
  // ---------------------------------------------------------------------
  trace_lineage: { title: "Trace Causal Lineage", readOnly: true },
  list_causal_edges: { title: "Survey Causal Edges", readOnly: true },
  // Deduped server-side, but a repeat call `reinforce()`s the existing edge's
  // confidence (src/memory/lineage.rs `add_explicit_edge`) => further effect.
  add_causal_link: { title: "Add Causal Link", readOnly: false, destructive: false, idempotent: false },
  // verdict="reject" deletes the edge, so the tool may destroy graph structure.
  validate_causal_link: { title: "Confirm or Reject Causal Link", readOnly: false, destructive: true, idempotent: true },
  explore_entity: { title: "Explore Entity Neighbourhood", readOnly: true },
  list_entities: { title: "List Graph Entities", readOnly: true },
  list_anomalies: { title: "List Statistical Anomalies", readOnly: true },
  search_facts: { title: "Search Distilled Facts", readOnly: true },
  // Adjusts importance weights and association strengths in a designed feedback
  // loop; nothing is deleted and "helpful" reverses "misleading", so it is not
  // destructive. Boosts compound across calls, so it is not idempotent.
  reinforce_memories: { title: "Reinforce Memories (Hebbian Feedback)", readOnly: false, destructive: false, idempotent: false },
};

/** Fully-resolved annotations per tool, with spec defaults made explicit. */
export const TOOL_ANNOTATIONS: Record<string, ToolAnnotations> = Object.fromEntries(
  Object.entries(TOOL_BEHAVIOUR).map(([name, spec]): [string, ToolAnnotations] => {
    if (spec.readOnly) {
      return [
        name,
        {
          title: spec.title,
          readOnlyHint: true,
          // destructiveHint/idempotentHint are defined by the spec to be
          // meaningful only when readOnlyHint is false, so they are omitted.
          openWorldHint: false,
        },
      ];
    }
    return [
      name,
      {
        title: spec.title,
        readOnlyHint: false,
        destructiveHint: spec.destructive,
        idempotentHint: spec.idempotent,
        openWorldHint: false,
      },
    ];
  }),
);

/** The set of tools that do not modify the memory store. */
export const READ_ONLY_TOOLS: ReadonlySet<string> = new Set(
  Object.entries(TOOL_BEHAVIOUR)
    .filter(([, spec]) => spec.readOnly)
    .map(([name]) => name),
);

/**
 * True when the tool is annotated `readOnlyHint: true`.
 *
 * Ambient ingestion in `index.ts` is gated on this so the annotation is true by
 * construction rather than by assertion.
 */
export function isReadOnlyTool(name: string): boolean {
  return READ_ONLY_TOOLS.has(name);
}

// =============================================================================
// OUTPUT SCHEMAS
// =============================================================================

// Reusable fragments. Every schema keeps `additionalProperties` open by
// omission so a backend field added later does not fail client-side validation
// (the SDK client validates structuredContent against these with Ajv).

const MEMORY_ITEM = {
  type: "object",
  properties: {
    id: { type: "string", description: "Memory UUID" },
    content: { type: "string", description: "Memory body; a preview unless full_content was requested" },
    content_truncated: { type: "boolean", description: "True when `content` is a truncated preview" },
    memory_type: { type: "string", description: "Observation | Decision | Learning | Error | ..." },
    tags: { type: "array", items: { type: "string" } },
    score: { type: "number", description: "Retrieval score, when the tool ranks results" },
    importance: { type: "number" },
    created_at: { type: "string", description: "ISO 8601 timestamp" },
    tier: { type: "string", description: "working | session | long_term" },
  },
  required: ["id", "content"],
} as const;

const TODO_ITEM = {
  type: "object",
  properties: {
    id: { type: "string" },
    short_id: { type: "string", description: "Human-facing prefix id, e.g. SHO-1a2b" },
    content: { type: "string" },
    status: { type: "string" },
    priority: { type: "string" },
    project: { type: "string" },
    score: { type: "number", description: "Present only for semantic todo search" },
    created_at: { type: "string" },
    due_date: { type: "string" },
  },
  required: ["id", "content", "status"],
} as const;

// `edge_id` rather than the wire's `id`: the formatted text has always labelled
// it "edge_id", and validate_causal_link's own parameter is named edge_id, so
// the structured channel uses the name the rest of the surface already uses.
const LINEAGE_EDGE = {
  type: "object",
  properties: {
    edge_id: { type: "string", description: "Pass to validate_causal_link" },
    from: { type: "string", description: "Cause / origin memory id" },
    to: { type: "string", description: "Effect memory id" },
    relation: {
      type: "string",
      description: "Caused | ResolvedBy | InformedBy | SupersededBy | TriggeredBy | BranchedFrom | RelatedTo",
    },
    confidence: { type: "number" },
    source: { type: "string", description: "Inferred | Confirmed | Explicit" },
    created_at: { type: "string" },
    reinforcement_count: { type: "number" },
  },
  required: ["edge_id", "from", "to", "relation"],
} as const;

// Knowledge-graph entity. `name_embedding` is on the wire but deliberately
// never surfaced — it is a vector, not information for a reader.
const GRAPH_ENTITY = {
  type: "object",
  properties: {
    id: { type: "string", description: "Entity UUID" },
    name: { type: "string", description: "Pass to explore_entity" },
    entity_type: { type: "string", description: "fine_type when known, else the labels, else Concept" },
    labels: { type: "array", items: { type: "string" } },
    salience: { type: "number", description: "Learned importance" },
    mention_count: { type: "number" },
    kb_id: { type: "string", description: "Knowledge-base identifier when the entity is linked" },
    hop_distance: { type: "number", description: "Hops from the starting entity (explore_entity only)" },
  },
  required: ["name"],
} as const;

const arrayOf = (item: unknown, description: string) => ({
  type: "array",
  items: item,
  description,
});

/**
 * Tools that return data worth parsing get a schema. Tools that return a
 * one-line confirmation ("Memory deleted", "Backup restored") or a prose
 * narrative deliberately do not — a schema there buys nothing and has to be
 * maintained forever.
 */
export const TOOL_OUTPUT_SCHEMAS: Record<string, JsonSchemaObject> = {
  remember: {
    type: "object",
    properties: {
      id: { type: "string", description: "UUID of the stored memory" },
      memory_type: { type: "string" },
      tags: { type: "array", items: { type: "string" } },
    },
    required: ["id"],
  },

  recall: {
    type: "object",
    properties: {
      query: { type: "string" },
      mode: { type: "string" },
      memories: arrayOf(MEMORY_ITEM, "Matching memories, best first"),
      todos: arrayOf(TODO_ITEM, "Matching todos, best first"),
      lineage: arrayOf(
        {
          type: "object",
          properties: {
            from: { type: "string" },
            to: { type: "string" },
            relation: { type: "string" },
            confidence: { type: "number" },
          },
          required: ["from", "to", "relation"],
        },
        "Causal edges connecting the recalled memories",
      ),
      memory_count: { type: "number" },
      todo_count: { type: "number" },
    },
    required: ["query", "mode", "memories", "todos", "memory_count", "todo_count"],
  },

  recall_by_tags: {
    type: "object",
    properties: {
      tags: { type: "array", items: { type: "string" }, description: "Tags that were searched" },
      memories: arrayOf(MEMORY_ITEM, "Memories matching ANY of the tags"),
      count: { type: "number" },
    },
    required: ["tags", "memories", "count"],
  },

  list_memories: {
    type: "object",
    properties: {
      memories: arrayOf(MEMORY_ITEM, "Stored memories, most recent first"),
      count: { type: "number" },
    },
    required: ["memories", "count"],
  },

  // `found` exists so the "memory not found" path — which the tool reports as a
  // successful call, not an error — still has an honest structured form. Only
  // `found` and the requested `id` are guaranteed; everything else is present
  // only when the lookup succeeded.
  read_memory: {
    type: "object",
    properties: {
      found: { type: "boolean" },
      id: { type: "string", description: "Resolved memory id, or the id that was requested when not found" },
      content: { type: "string", description: "Complete, untruncated memory body" },
      memory_type: { type: "string" },
      entities: { type: "array", items: { type: "string" }, description: "Entities extracted from this memory" },
      created_at: { type: "string" },
      importance: { type: "number" },
      tier: { type: "string" },
      parent_id: { type: "string" },
      children_ids: { type: "array", items: { type: "string" } },
      children_count: { type: "number" },
    },
    required: ["found", "id"],
  },

  memory_stats: {
    type: "object",
    properties: {
      total_memories: { type: "number" },
      working_memory_count: { type: "number" },
      session_memory_count: { type: "number" },
      long_term_memory_count: { type: "number" },
      vector_index_count: { type: "number" },
      average_importance: { type: "number" },
      total_retrievals: { type: "number" },
      graph_nodes: { type: "number" },
      graph_edges: { type: "number" },
    },
    required: ["total_memories"],
  },

  // Field names mirror the backend `IndexIntegrityReport` wire shape rather
  // than inventing prettier ones — a renaming layer would drift the moment the
  // report changes.
  verify_index: {
    type: "object",
    properties: {
      is_healthy: { type: "boolean" },
      total_storage: { type: "number", description: "Memories in storage" },
      total_indexed: { type: "number", description: "Vectors in the search index" },
      orphaned_count: { type: "number", description: "Stored but not searchable" },
      orphaned_ids: { type: "array", items: { type: "string" } },
    },
    required: ["is_healthy", "total_storage", "total_indexed", "orphaned_count"],
  },

  list_reminders: {
    type: "object",
    properties: {
      status_filter: { type: "string" },
      reminders: arrayOf(
        {
          type: "object",
          properties: {
            id: { type: "string" },
            content: { type: "string" },
            status: { type: "string", description: "pending | triggered | dismissed" },
            trigger_type: { type: "string", description: "time | duration | context" },
            due_at: { type: "string", description: "ISO 8601, for time/duration triggers" },
            created_at: { type: "string" },
            priority: { type: "number", description: "1-5, 5 = highest" },
            overdue_seconds: { type: "number" },
          },
          required: ["id", "content"],
        },
        "Reminders matching the status filter",
      ),
      count: { type: "number" },
    },
    required: ["reminders", "count"],
  },

  // Same projection as the todo items inside list_todos, so a caller can hand
  // the created todo straight back to update_todo / add_todo_comment.
  add_todo: {
    type: "object",
    properties: TODO_ITEM.properties,
    required: ["id"],
  },

  list_todos: {
    type: "object",
    properties: {
      todos: arrayOf(TODO_ITEM, "Todos matching the filters or semantic query"),
      count: { type: "number", description: "Number of todos in this page" },
      total: { type: "number", description: "Total matching todos before pagination" },
    },
    required: ["todos", "count"],
  },

  list_subtasks: {
    type: "object",
    properties: {
      parent_id: { type: "string" },
      subtasks: arrayOf(TODO_ITEM, "Direct children of the parent todo"),
      count: { type: "number" },
    },
    required: ["subtasks", "count"],
  },

  list_todo_comments: {
    type: "object",
    properties: {
      todo_id: { type: "string" },
      comments: arrayOf(
        {
          type: "object",
          properties: {
            id: { type: "string" },
            content: { type: "string" },
            author: { type: "string" },
            comment_type: { type: "string" },
            created_at: { type: "string" },
          },
          required: ["id", "content"],
        },
        "Comments and activity entries, oldest first",
      ),
      count: { type: "number" },
    },
    required: ["comments", "count"],
  },

  // Flat status counts, mirroring the backend `UserTodoStats` wire shape.
  todo_stats: {
    type: "object",
    properties: {
      total: { type: "number" },
      backlog: { type: "number" },
      todo: { type: "number" },
      in_progress: { type: "number" },
      blocked: { type: "number" },
      done: { type: "number" },
      cancelled: { type: "number" },
      overdue: { type: "number" },
      due_today: { type: "number" },
      projects: { type: "number", description: "Number of projects" },
    },
    required: ["total"],
  },

  // The backend returns `projects` as (Project, ProjectStats) tuples; the
  // structured payload flattens each pair into one object so consumers do not
  // have to index into a two-element array.
  list_projects: {
    type: "object",
    properties: {
      projects: arrayOf(
        {
          type: "object",
          properties: {
            id: { type: "string" },
            name: { type: "string" },
            prefix: { type: "string", description: "Todo id prefix, e.g. BOLT" },
            description: { type: "string" },
            status: { type: "string", description: "active | archived" },
            parent_id: { type: "string" },
            stats: {
              type: "object",
              description: "Todo counts for this project",
              properties: {
                total: { type: "number" },
                backlog: { type: "number" },
                todo: { type: "number" },
                in_progress: { type: "number" },
                blocked: { type: "number" },
                done: { type: "number" },
                cancelled: { type: "number" },
              },
            },
          },
          required: ["id", "name"],
        },
        "Projects with their todo counts",
      ),
      count: { type: "number" },
    },
    required: ["projects", "count"],
  },

  backup_list: {
    type: "object",
    properties: {
      backups: arrayOf(
        {
          type: "object",
          properties: {
            backup_id: { type: "number", description: "Pass to backup_verify / backup_restore" },
            created_at: { type: "string" },
            backup_type: { type: "string" },
            size_bytes: { type: "number" },
            memory_count: { type: "number" },
            checksum: { type: "string" },
            sequence_number: { type: "number" },
          },
          required: ["backup_id"],
        },
        "Available backups, newest first",
      ),
      count: { type: "number" },
    },
    required: ["backups", "count"],
  },

  proactive_context: {
    type: "object",
    properties: {
      memories: arrayOf(
        {
          type: "object",
          properties: {
            id: { type: "string" },
            content: { type: "string" },
            content_truncated: { type: "boolean" },
            memory_type: { type: "string" },
            score: { type: "number", description: "Relevance score for this context" },
            importance: { type: "number" },
            tags: { type: "array", items: { type: "string" } },
            relevance_reason: { type: "string", description: "Why the engine surfaced this memory" },
            matched_entities: { type: "array", items: { type: "string" } },
            created_at: { type: "string" },
          },
          required: ["id", "content"],
        },
        "Memories surfaced as relevant to the supplied context",
      ),
      detected_entities: arrayOf(
        {
          type: "object",
          properties: { name: { type: "string" }, entity_type: { type: "string" } },
          required: ["name"],
        },
        "Entities extracted from the supplied context",
      ),
      todos: arrayOf(TODO_ITEM, "Todos judged relevant to the context"),
      facts: arrayOf(
        {
          type: "object",
          properties: {
            id: { type: "string" },
            fact: { type: "string" },
            confidence: { type: "number" },
            support_count: { type: "number" },
          },
          required: ["fact"],
        },
        "Distilled facts judged relevant to the context",
      ),
      count: { type: "number", description: "Number of surfaced memories" },
      // Deliberately NOT a boolean "was it ingested". The backend spawns the
      // ingest as a background task and only waits 50ms to read the id back
      // (handlers/recall.rs), so a missing id routinely accompanies a write
      // that did happen. Reporting that as `auto_ingested: false` would be a
      // false negative — the exact wrong signal for a capture path.
      ingest_requested: {
        type: "boolean",
        description:
          "Whether ingestion of the supplied context was requested (the auto_ingest argument, default true). The backend applies its own filters (duplicate, length, bare-question, noise), so this is the request, not a guarantee.",
      },
      ingested_memory_id: {
        type: "string",
        description:
          "Id of the stored context memory when the backend returned one in time. ABSENCE IS NOT PROOF OF NO WRITE: the ingest runs in the background and the id is only awaited briefly.",
      },
      latency_ms: { type: "number" },
    },
    required: ["memories", "detected_entities", "todos", "facts", "count", "ingest_requested"],
  },

  trace_lineage: {
    type: "object",
    properties: {
      memory_id: { type: "string", description: "Resolved full UUID of the traced memory" },
      direction: { type: "string", enum: ["backward", "forward", "both"] },
      root_cause_id: {
        type: "string",
        description: "Oldest ancestor. Absent when tracing forward, or when the memory starts its own chain.",
      },
      edges: arrayOf(LINEAGE_EDGE, "Causal edges on the traced chain — ALL of them, not the display-capped subset"),
      path: { type: "array", items: { type: "string" }, description: "Memory ids along the traced path" },
      depth_reached: { type: "number" },
      edge_count: { type: "number" },
    },
    required: ["memory_id", "direction", "edges", "edge_count"],
  },

  list_causal_edges: {
    type: "object",
    properties: {
      edges: arrayOf(LINEAGE_EDGE, "Highest-confidence edges, up to the requested limit"),
      count: { type: "number", description: "Edges returned here" },
      total: { type: "number", description: "Total edges in the lineage graph" },
      stats: {
        type: "object",
        description: "Graph-wide totals. Absent when the stats endpoint is unavailable.",
        properties: {
          total_edges: { type: "number" },
          inferred_edges: { type: "number" },
          confirmed_edges: { type: "number" },
          explicit_edges: { type: "number" },
          total_branches: { type: "number" },
          active_branches: { type: "number" },
          avg_confidence: { type: "number" },
          edges_by_relation: { type: "object", additionalProperties: { type: "number" } },
        },
      },
    },
    required: ["edges", "count"],
  },

  explore_entity: {
    type: "object",
    properties: {
      query: { type: "string", description: "Entity name exactly as supplied" },
      found: { type: "boolean", description: "False when no entity matched the name" },
      matched_entity: {
        type: "string",
        description: "Name the fuzzy match actually resolved to — may differ from `query`",
      },
      max_depth: { type: "number" },
      entities: arrayOf(GRAPH_ENTITY, "Connected entities, nearest hop and highest salience first"),
      relationships: arrayOf(
        {
          type: "object",
          properties: {
            id: { type: "string" },
            from: { type: "string", description: "Source entity name" },
            to: { type: "string", description: "Target entity name" },
            from_id: { type: "string" },
            to_id: { type: "string" },
            relation_type: { type: "string", description: "Causes | Triggers | DependsOn | CoOccurs | custom" },
            strength: { type: "number" },
            context: { type: "string", description: "Sentence the relation was extracted from" },
            typed: {
              type: "boolean",
              description: "False for bulk co-occurrence edges (CoOccurs / RelatedTo)",
            },
          },
          required: ["from", "to", "relation_type"],
        },
        "Live relationships between the returned entities; invalidated edges are excluded",
      ),
      entity_count: { type: "number" },
      relationship_count: { type: "number", description: "Live relationships" },
      invalidated_count: { type: "number", description: "Relationships excluded because they were invalidated" },
    },
    required: ["query", "found", "entities", "relationships", "entity_count", "relationship_count"],
  },

  list_entities: {
    type: "object",
    properties: {
      entities: arrayOf(GRAPH_ENTITY, "Entities ranked by salience"),
      count: { type: "number" },
      total: { type: "number", description: "Total entities in the graph" },
    },
    required: ["entities", "count"],
  },

  list_anomalies: {
    type: "object",
    properties: {
      min_sigma: { type: "number", description: "Threshold the backend actually applied" },
      anomalies: arrayOf(
        {
          type: "object",
          properties: {
            memory_id: { type: "string" },
            content_preview: { type: "string" },
            max_abs_z: { type: "number", description: "Largest absolute z-score across components" },
            flagged: { type: "boolean", description: "True when max_abs_z >= min_sigma" },
            explanation: { type: "string", description: "Deterministic reason the entry deviates" },
            entities: arrayOf(
              {
                type: "object",
                properties: { id: { type: "string" }, name: { type: "string" } },
                required: ["name"],
              },
              "Entities in the anomalous memory",
            ),
            created_at: { type: "string" },
          },
          required: ["memory_id", "max_abs_z", "flagged"],
        },
        "Memories ranked by deviation from the user's own rolling baseline",
      ),
      count: { type: "number" },
      flagged_count: { type: "number" },
      episodes_scored: {
        type: "number",
        description: "Scored episodes available. Below 10 the feed is empty by design, not because nothing deviates.",
      },
      baseline_window: { type: "number", description: "Size of the rolling baseline window" },
    },
    required: ["anomalies", "count", "episodes_scored"],
  },

  search_facts: {
    type: "object",
    properties: {
      query: { type: "string", description: "Echoed when keyword search was used" },
      entity: { type: "string", description: "Echoed when entity lookup was used" },
      facts: arrayOf(
        {
          type: "object",
          properties: {
            id: { type: "string" },
            fact: { type: "string", description: "The distilled statement" },
            fact_type: { type: "string" },
            confidence: { type: "number" },
            support_count: { type: "number", description: "Memories supporting this fact" },
            related_entities: { type: "array", items: { type: "string" } },
            created_at: { type: "string" },
            invalidated_at: {
              type: "string",
              description: "Set when the fact was superseded but retained for audit",
            },
          },
          required: ["fact"],
        },
        "Distilled semantic facts",
      ),
      count: { type: "number" },
    },
    required: ["facts", "count"],
  },
};

// =============================================================================
// DECORATION
// =============================================================================

/**
 * Attach annotations (and an outputSchema where one is declared) to a tool
 * definition.
 *
 * Throws when a tool has no entry in TOOL_BEHAVIOUR. That is deliberate: a
 * silent default would mean a newly added tool ships with the spec defaults
 * (`destructiveHint: true`, `openWorldHint: true`) and, worse, would be treated
 * as non-read-only by the ambient-ingestion gate without anyone deciding so.
 * Failing at startup forces the judgement call to be made once, here.
 */
export function decorateTool(def: ToolDefinition): DecoratedTool {
  const annotations = TOOL_ANNOTATIONS[def.name];
  if (!annotations) {
    throw new Error(
      `Tool "${def.name}" has no entry in TOOL_BEHAVIOUR (mcp-server/tool-metadata.ts). ` +
        `Add one — annotations are not optional, and the read-only set gates ambient ingestion.`,
    );
  }
  const outputSchema = TOOL_OUTPUT_SCHEMAS[def.name];
  return outputSchema ? { ...def, annotations, outputSchema } : { ...def, annotations };
}
