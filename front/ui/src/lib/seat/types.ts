/**
 * Wire types for the conversation-seat harness, each transcribed from the
 * seat's own source (seat/src/*.ts) the same way lib/api/types.ts transcribes
 * the Rust handlers. Nothing here is guessed; if a field is not in the seat
 * source it is not here.
 *
 * The recall shapes (RecallMemory, ScoreAttribution, …) are re-exported from
 * lib/api/types.ts rather than redeclared: the seat forwards the Rust
 * backend's own structures verbatim (seat/src/backend.ts mirrors
 * src/handlers/types.rs), so one declaration serves both surfaces.
 */

import type {
  RecallFact,
  RecallLineageEdge,
  RecallMemory,
  RecallTodo,
} from "@/lib/api/types";
import type { ViewDimension } from "@/lib/view/authority";
import type { ViewOutcomeState } from "@/lib/view/outcome";

export type {
  RecallFact,
  RecallLineageEdge,
  RecallMemory,
  RecallTodo,
  ScoreAttribution,
} from "@/lib/api/types";

/** seat/src/events.ts `MemoryScope` */
export type MemoryScope = "user" | "harness";

/** seat/src/events.ts `ModelRef` */
export interface ModelRef {
  provider: string;
  id: string;
  name: string;
}

/** seat/src/events.ts `UsagePayload` — pi's per-message Usage, verbatim. */
export interface UsagePayload {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning?: number;
  totalTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
  };
}

/** seat/src/backend.ts `ProactiveSurfacedMemory` (src/handlers/recall.rs). */
export interface ProactiveSurfacedMemory {
  id: string;
  content: string;
  memory_type: string;
  score: number;
  importance: number;
  created_at: string;
  tags: string[];
  tier: string;
  relevance_reason: string;
  matched_entities?: string[];
}

/** seat/src/backend.ts `FeedbackProcessed` (src/handlers/recall.rs). */
export interface FeedbackProcessed {
  memories_evaluated: number;
  reinforced: string[];
  weakened: string[];
}

/** seat/src/backend.ts `ReinforceOutcome` / `ReinforceStats`. */
export type ReinforceOutcome = "helpful" | "misleading" | "neutral";
export interface ReinforceStats {
  memories_processed: number;
  associations_strengthened: number;
  importance_boosts: number;
  importance_decays: number;
}

/** seat/src/events.ts `ReinforceTrigger` */
export type ReinforceTrigger =
  | { kind: "response_overlap"; overlaps: Record<string, number>; threshold: number }
  | { kind: "citation"; cited: string[] }
  | { kind: "negative_followup"; keywords: string[] }
  | { kind: "revert"; of: string };

/** seat/src/events.ts `SeatEvent` — the discriminated union, verbatim. */
export type SeatEvent =
  | { type: "conversation_created"; conversation_id: string; user_id: string; model: ModelRef }
  | { type: "turn_start"; turn: number }
  | { type: "text_delta"; delta: string }
  | { type: "thinking_delta"; delta: string }
  | { type: "tool_call_start"; tool_call_id: string; tool_name: string; args: unknown }
  | { type: "tool_call_end"; tool_call_id: string; tool_name: string; is_error: boolean }
  | {
      type: "memory_recall";
      scope: MemoryScope;
      tool_call_id?: string;
      query: string;
      mode: string;
      memories: RecallMemory[];
      facts: RecallFact[];
      todos: RecallTodo[];
      lineage: RecallLineageEdge[];
      took_ms: number;
    }
  | {
      type: "memory_write";
      scope: MemoryScope;
      memory_id: string;
      memory_type: string;
      content_preview: string;
      ledger_event_id: string;
    }
  | {
      type: "memory_reinforce";
      scope: MemoryScope;
      outcome: ReinforceOutcome;
      memory_ids: string[];
      stats: ReinforceStats;
      trigger: ReinforceTrigger;
      ledger_event_id: string;
    }
  | {
      type: "proactive_context";
      scope: "user";
      query: string;
      memories: ProactiveSurfacedMemory[];
      injected_memory_ids: string[];
      feedback: FeedbackProcessed | null;
      temporal_credits_applied: number | null;
      took_ms: number;
    }
  | {
      /**
       * The model asked to move this view (seat/src/view-tools.ts `direct_view`).
       *
       * ALREADY VALIDATED WHEN IT ARRIVES. `destination` is a real path and
       * every name in `entities` was resolved against this profile's graph by
       * the seat, so `lib/view/commands.ts` translates it without re-checking
       * anything. `unresolved` carries the terms that named nothing — kept on
       * the wire because a command that framed three of five things must not be
       * indistinguishable from one that framed five.
       */
      type: "view_command";
      tool_call_id: string;
      /** Why, in the model's own words. Shown to the person verbatim. */
      reason: string;
      /** Destination path, or null when the move only frames. */
      destination: string | null;
      entities: string[];
      unresolved: string[];
      /** The one entity to open in the inspector, or null. `id` is the graph's
       *  `uuid`, which is what `UniverseStar.id` and therefore
       *  `selectedEntityId` are — the seat resolved it, so it selects directly. */
      focus: { id: string; name: string } | null;
    }
  | {
      /**
       * The seat asking this browser what is on screen (`inspect_view`).
       *
       * Carries nothing but a correlation id, and that is the guarantee: there
       * is no dimension, no path and no entity in it, so there is nothing a
       * probe could be misread as instructing. The answer goes back on
       * `POST /v1/conversations/{id}/view-report` quoting `probe_id`.
       */
      type: "view_probe";
      probe_id: string;
    }
  | {
      /**
       * What this browser did with a view command — written by the seat's
       * view-report route, never streamed.
       *
       * It appears in a conversation's stored events (and so in `buildTurns`)
       * because it is durable audit material; it is not something the browser
       * learns from the wire, since the browser is what produced it.
       */
      type: "view_outcome";
      tool_call_id: string;
      dimension: ViewDimension;
      state: ViewOutcomeState;
      /** The path the browser was on when it decided. */
      at: string;
    }
  | { type: "harness_learning_applied"; memories: { id: string; content: string; score: number }[] }
  | { type: "model_changed"; model: ModelRef }
  | { type: "usage"; model: ModelRef; usage: UsagePayload }
  | { type: "turn_end"; turn: number; stop_reason: string; error_message?: string }
  | { type: "agent_end" }
  | { type: "error"; message: string };

/** seat/src/models-registry.ts `ModelInfo`. `billing` is what a token MEANS
 *  under the model's effective credential — the seat computes it from pi's
 *  auth resolution, the UI only ever displays it:
 *  "none" local (nothing leaves the machine) · "subscription" flat-rate plan
 *  (pi's cost numbers do not describe a bill) · "metered" API key (they do). */
export interface SeatModelInfo {
  provider: string;
  id: string;
  name: string;
  context_window: number;
  max_tokens: number;
  reasoning: boolean;
  local: boolean;
  billing: "none" | "subscription" | "metered";
}

/** seat/src/models-registry.ts `ProviderInfo` */
export interface ProviderInfo {
  id: string;
  name: string;
  configured: boolean;
  source: string | null;
  auth_type: "api_key" | "oauth" | null;
  stored: boolean;
  accepts_api_key: boolean;
  oauth_available: boolean;
  oauth_subscription: boolean;
  oauth_label: string | null;
  model_count: number;
  local: boolean;
}

/** seat/src/mcp.ts `McpTransportKind` — which wire a tool server is reached
 *  over. "stdio" is a program the seat starts on this machine; "http" is the
 *  current remote transport (streamable HTTP); "sse" is the superseded one,
 *  still the only thing some deployed servers speak. */
export type McpTransportKind = "stdio" | "http" | "sse";

/** seat/src/mcp.ts `McpServerStatus`. "failed" and "disconnected" are kept
 *  apart on purpose: one never came up, the other was working and went away,
 *  and they do not have the same remedy. */
export type McpServerStatus = "connecting" | "ready" | "failed" | "disconnected";

/** seat/src/mcp.ts `McpToolInfo` — a tool as its server DESCRIBES it. Nothing
 *  in this shape says the tool has ever been called. */
export interface McpToolInfo {
  name: string;
  title: string | null;
  description: string | null;
}

/** seat/src/mcp.ts `McpServerInfo` — GET /seat/v1/mcp/servers. Header values
 *  and URL query strings are stripped by the seat; only header names and the
 *  scheme/host/path reach here. */
export interface McpServerInfo {
  name: string;
  status: McpServerStatus;
  transport: McpTransportKind;
  tool_count: number;
  tools: McpToolInfo[];
  error: string | null;
  endpoint: string | null;
  command: string | null;
  auth_header_names: string[];
  server_name: string | null;
  server_version: string | null;
  connected_at: string | null;
  tools_listed_at: string | null;
  last_attempt_at: string | null;
}

/** OAuth-bridge stream frames — seat/src/server.ts handleOAuthStart. */
export type OAuthFlowEvent =
  | {
      kind: "notify";
      event:
        | { type: "info"; message: string; links?: { url: string; label?: string }[] }
        | { type: "auth_url"; url: string; instructions?: string }
        | {
            type: "device_code";
            userCode: string;
            verificationUri: string;
            intervalSeconds?: number;
            expiresInSeconds?: number;
          }
        | { type: "progress"; message: string };
    }
  | {
      kind: "prompt";
      prompt_id: string;
      type: "text" | "secret" | "select" | "manual_code";
      message: string;
      placeholder?: string;
      options?: { id: string; label: string; description?: string }[];
    }
  | { kind: "prompt_cancelled"; prompt_id: string }
  | { kind: "complete"; provider: ProviderInfo | null }
  | { kind: "error"; message: string };

/** seat/src/store.ts `UsageTotals` — accumulated per conversation. */
export interface UsageTotals {
  input: number;
  output: number;
  cache_read: number;
  cache_write: number;
  reasoning: number;
  total_tokens: number;
  cost_total: number;
}

/** seat/src/server.ts `conversationSummary` — one row of the session list. */
export interface ConversationSummary {
  conversation_id: string;
  user_id: string;
  title: string | null;
  model: ModelRef;
  created_at: string;
  updated_at: string;
  turns: number;
  usage: UsageTotals;
  busy: boolean;
}

/** seat/src/store.ts `StoredEvent` — durable event with its turn position. */
export interface StoredEvent {
  turn: number;
  ts: string;
  event: SeatEvent;
}

/**
 * pi message shapes, as persisted in the transcript —
 * pi packages/ai/src/types.ts (UserMessage / AssistantMessage /
 * ToolResultMessage), fields this UI renders only.
 */
export interface PiTextContent {
  type: "text";
  text: string;
}
export interface PiThinkingContent {
  type: "thinking";
  thinking: string;
}
export interface PiToolCallContent {
  type: "toolCall";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}
export interface PiUserMessage {
  role: "user";
  content: string | ({ type: string } & Record<string, unknown>)[];
  timestamp: number;
}
export interface PiAssistantMessage {
  role: "assistant";
  content: (PiTextContent | PiThinkingContent | PiToolCallContent)[];
  provider: string;
  model: string;
  usage: UsagePayload;
  stopReason: string;
  errorMessage?: string;
  timestamp: number;
}
export interface PiToolResultMessage {
  role: "toolResult";
  toolCallId: string;
  toolName: string;
  content: ({ type: string } & Record<string, unknown>)[];
  isError: boolean;
  timestamp: number;
}
export type PiMessage = PiUserMessage | PiAssistantMessage | PiToolResultMessage;

/** GET /seat/v1/conversations/{id} — summary + transcript + durable events. */
export interface ConversationDetail extends ConversationSummary {
  messages: PiMessage[];
  events: StoredEvent[];
}

/**
 * seat/src/ledger.ts `LedgerActor` — WHO caused an entry, as opposed to
 * `MemoryScope`, which says which namespace was touched. Both spell "user" and
 * they mean different things: an `actor: "agent"`, `scope: "user"` entry is the
 * model writing into the human's memory.
 */
export type LedgerActor = "user" | "agent" | "system";

/**
 * seat/src/ledger.ts `LedgerActorView` — the READ-side actor, widened by one
 * value the write side cannot produce.
 *
 * Entries written before `actor` existed report "unknown" and are deliberately
 * NOT backfilled: inferring an actor for a historical entry and writing it down
 * as fact is exactly what an audit log exists to prevent. Every surface that
 * renders this must render the gap, never a default.
 */
export type LedgerActorView = LedgerActor | "unknown";

/** seat/src/audit.ts `AuditSource` — which store a row came from. `view` covers
 *  both the ask (`kind: "view_command"`) and what the browser did about it
 *  (`kind: "view_outcome"`); they are one source and two kinds, because they
 *  come from the same event store and are told apart by what they claim. */
export type AuditSource = "ledger" | "tool_call" | "retrieval" | "view";

/**
 * seat/src/view-link.ts `ViewSnapshot` — what this browser tells the seat is on
 * screen, and the shape `inspect_view` answers with.
 *
 * The absences are the contract: no memory text, no recall results, no
 * conversation content, no credentials, no pixels. See the seat-side note.
 */
export interface ViewSnapshotWire {
  destination: string;
  profile: string | null;
  cue: { text: string; entities: string[]; author: "user" | "agent" } | null;
  focus: { id: string; name: string | null } | null;
  claimed: ViewDimension[];
  offers: { dimension: ViewDimension; reason: string }[];
}

/** One `POST /seat/v1/conversations/{id}/view-report` body — seat/src/view-link.ts
 *  `parseViewReport`, which rejects anything outside these closed sets. */
export interface ViewReportWire {
  probe_id: string | null;
  outcomes: { tool_call_id: string; dimension: ViewDimension; state: ViewOutcomeState }[];
  view: ViewSnapshotWire;
}

/**
 * seat/src/audit.ts `AuditRow` — one line of the audit trail, flat and uniform
 * across all three sources so it survives a spreadsheet and `jq` alike.
 *
 * This is the row shape of `GET /v1/audit/export`, in the order
 * `buildAuditRows` sorts it: the total order (ts, source, ref). That order is
 * the artefact's citability — two exports of the same window are byte-identical
 * — so nothing on this side may re-sort a parsed trail.
 */
export interface AuditRow {
  /** ISO-8601 UTC. */
  ts: string;
  source: AuditSource;
  actor: LedgerActorView;
  /** Ledger kind, tool name, or event type. */
  kind: string;
  user_id: string;
  conversation_id: string;
  turn: number;
  /** Ledger entry id, tool call id, or memory-operation identity. */
  ref: string;
  /** Source-specific payload, JSON-encoded so one column holds every shape. */
  detail: string;
}

/**
 * seat/src/audit.ts `AUDIT_COLUMNS`, verbatim and in order.
 *
 * Transcribed rather than derived from `keyof AuditRow`: the seat's list is
 * FIXED because an export whose columns move is not citable, and a list derived
 * from a TypeScript interface would silently follow a field rename here. This
 * copy exists so `parseAuditJsonl` can check that what arrived is what the
 * seat's writer emits.
 */
export const AUDIT_COLUMNS = [
  "ts",
  "source",
  "actor",
  "kind",
  "user_id",
  "conversation_id",
  "turn",
  "ref",
  "detail",
] as const satisfies readonly (keyof AuditRow)[];

/**
 * seat/src/audit.ts `ToolCallRecord`'s tail, as it is encoded into
 * `AuditRow.detail` by `toolCallRow`.
 *
 * THE NULLS ARE LOAD-BEARING AND ARE NOT OPTIONAL FIELDS. A call whose turn
 * ended before it returned (abort, crash, kill) keeps `ended_at`, `duration_ms`
 * and `is_error` at null — `is_error: false` would assert a success that never
 * happened, and those rows are the most audit-relevant ones in the set.
 */
export interface ToolCallDetail {
  args: unknown;
  ended_at: string | null;
  duration_ms: number | null;
  is_error: boolean | null;
}

/** seat/src/ledger.ts `LedgerEntry` / `LedgerEntryView` — fields the UI shows. */
export interface LedgerEntryView {
  entry: {
    id: string;
    ts: string;
    /** Absent on entries written before the field existed; render as unknown,
     *  never as a default actor. */
    actor?: LedgerActor;
    kind: "memory_write" | "reinforce" | "implicit_feedback" | "revert";
    scope: MemoryScope;
    user_id: string;
    conversation_id: string;
    turn: number;
    data: Record<string, unknown>;
  };
  reverted_by?: string;
}

/** GET /seat/healthz (seat/src/server.ts handleHealth). 200 when the backend
 *  answers, 503 when it does not — the seat itself is up in both cases. */
export interface SeatHealthResponse {
  seat: "ok";
  backend: { ok: boolean; detail: string };
  conversations: number;
  /** seat/src/mcp.ts `McpServerHealth`. This route is unauthenticated, so it
   *  carries liveness only — endpoints and connection errors are on
   *  GET /seat/v1/mcp/servers, behind the seat's bearer token. */
  mcp_servers: { name: string; status: McpServerStatus; tool_count: number }[];
}

/** Reachability of the seat process, distinguished the same way the backend's
 *  is (lib/api/health.ts): different states need different remedies. */
export type SeatReachability =
  | { state: "online"; backendOk: boolean; backendDetail: string }
  | { state: "offline"; detail: string };
