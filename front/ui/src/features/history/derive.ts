import {
  AUDIT_COLUMNS,
  type AuditRow,
  type AuditSource,
  type LedgerActorView,
  type ToolCallDetail,
} from "@/lib/seat/types";

/**
 * The reading of the audit trail, kept out of the component so every claim the
 * History screen makes can be pinned by a test.
 *
 * WHY THE SCREEN READS THE EXPORT FILE RATHER THAN A LIST ENDPOINT. The seat
 * serves three audit reads and only one of them is the whole trail:
 * `GET /v1/audit/tool-calls` returns tool calls alone, `GET /v1/learning/events`
 * returns ledger entries alone and cannot filter by time, and
 * `GET /v1/audit/export` returns ledger + paired tool calls + retrievals merged
 * and sorted (seat/src/audit.ts `buildAuditRows`). Building the timeline from
 * the first two would put LESS on screen than the file the reviewer receives,
 * and the difference would be invisible — so this screen parses the export
 * itself. What is displayed and what is downloaded are then the same bytes by
 * construction, and cannot drift.
 *
 * NOTHING HERE RE-SERIALIZES A ROW. The download saves the server's response
 * body untouched. A second CSV writer on this side would be a second definition
 * of the artefact, and the one property the export is built for — two exports
 * of the same window are byte-identical, so they can be diffed — belongs to the
 * seat's writer alone.
 *
 * ORDER IS NEVER TOUCHED. `buildAuditRows` sorts on the total order
 * (ts, source, ref) precisely so the artefact is citable; re-sorting a parsed
 * trail here would make the screen disagree with the file it exports.
 */

/* ------------------------------------------------------------------ *
 * THE QUERY
 *
 * Only what `GET /v1/audit/export` actually accepts: `format`, `user_id`,
 * `conversation_id`, `since`, `until` (seat/src/server.ts handleAuditExport +
 * auditWindow). Everything else this screen offers narrows rows already in
 * hand, and is labelled as such on the surface, because the file carries the
 * server-side narrowing and not the view's.
 *
 * `user_id` is deliberately NOT offered. It is not one filter but two: the
 * event leg matches it against the conversation's owner (store.ts `queryEvents`
 * joins `conversations.user_id`), while the ledger leg matches it against the
 * backend namespace the operation ran against — which for harness-scope entries
 * is the derived `<user>.seat-harness` (ledger.ts `LedgerEntryBase.user_id`).
 * A profile picker would therefore drop harness-scope ledger writes from a
 * trail that still showed their events, which is a silently incomplete audit
 * log. The column stays visible on the row instead.
 * ------------------------------------------------------------------ */

/**
 * The windows offered, and the only source of their durations.
 *
 * "Everything" is the default and is not a placeholder: the seat's own read
 * ceiling is 50,000 EVENTS per export (server.ts `MAX_AUDIT_EXPORT_EVENTS`), so
 * the unbounded window is already bounded, and a screen that opened onto a
 * narrow window would greet most installs with an empty audit log — which reads
 * as "nothing was recorded" rather than as "you are looking at a slice".
 */
export const AUDIT_WINDOWS = [
  { id: "all", label: "Everything", ms: null },
  { id: "month", label: "30 days", ms: 30 * 86_400_000 },
  { id: "week", label: "7 days", ms: 7 * 86_400_000 },
  { id: "day", label: "24 hours", ms: 86_400_000 },
] as const;

export type AuditWindowId = (typeof AUDIT_WINDOWS)[number]["id"];

export interface AuditQuery {
  window: AuditWindowId;
  /** A conversation id present in the trail, or null for every conversation. */
  conversationId: string | null;
}

export type AuditFormat = "jsonl" | "csv";

/**
 * The `since` bound for a window, or null for an unbounded one.
 *
 * `now` is a parameter rather than a `Date.now()` call so the boundary is
 * testable, and so the screen can hold one instant across the trail read and
 * the export that follows it — a download whose window had silently advanced
 * since the rows on screen were fetched would not be the file the reader is
 * looking at.
 *
 * NO `until` IS EVER SENT. An audit read has no upper bound: it runs up to
 * whatever the seat has written by the time it answers. Sending `until = now`
 * would exclude anything recorded between the click and the read, which is the
 * one class of row a reviewer is most likely to be looking for.
 */
export function windowSince(id: AuditWindowId, now: number): string | null {
  const window = AUDIT_WINDOWS.find((candidate) => candidate.id === id);
  if (!window || window.ms === null) return null;
  return new Date(now - window.ms).toISOString();
}

/** The seat path for one audit read. Same builder for the on-screen trail and
 *  the download, so the file can never cover a different window than the rows. */
export function auditExportPath(query: AuditQuery, format: AuditFormat, now: number): string {
  const params = new URLSearchParams({ format });
  const since = windowSince(query.window, now);
  if (since !== null) params.set("since", since);
  if (query.conversationId !== null) params.set("conversation_id", query.conversationId);
  return `/seat/v1/audit/export?${params}`;
}

/**
 * The filename to save the download under.
 *
 * Rebuilt here rather than read from `Content-Disposition`, because in the
 * shipped product that header does not arrive: the shodh-front proxy forwards
 * exactly two response headers, `content-type` and `cache-control`
 * (front/src/main.rs `forward`), and drops the rest. The dev proxy passes it
 * through, so the caller prefers the real header where there is one and falls
 * back to this — see `client.ts` `fetchAuditFile`.
 *
 * The format mirrors seat/src/server.ts handleAuditExport exactly
 * (`shodh-audit-<ISO with ':' replaced, to seconds>.<format>`).
 *
 * `now` is the instant the DOWNLOAD is taken, not the instant the window was
 * chosen — the seat stamps its own name when it answers, and two exports of the
 * same window taken an hour apart are two artefacts. Passing the screen's
 * frozen mount instant here would give them one name and leave the browser to
 * disambiguate them with "(1)".
 */
export function exportFilename(format: AuditFormat, now: number): string {
  const stamp = new Date(now).toISOString().replaceAll(":", "-").slice(0, 19);
  return `shodh-audit-${stamp}.${format}`;
}

/* ------------------------------------------------------------------ *
 * PARSING
 * ------------------------------------------------------------------ */

const AUDIT_SOURCES: ReadonlySet<string> = new Set<AuditSource>([
  "ledger",
  "tool_call",
  "retrieval",
  "view",
]);

const ACTOR_VIEWS: ReadonlySet<string> = new Set<LedgerActorView>([
  "user",
  "agent",
  "system",
  "unknown",
]);

export interface ParsedTrail {
  /** In the server's order, which is the export's order. Never re-sorted. */
  rows: AuditRow[];
  /**
   * Lines this build could not read: torn JSON, a missing or wrongly typed
   * column, or a `source`/`actor` value newer than this UI.
   *
   * COUNTED AND SHOWN, NOT SWALLOWED. Two different failures land here and
   * both matter: a truncated file, and an embedded UI older than the seat it
   * is talking to. Either way the screen is showing fewer rows than the export
   * contains, and a trail that quietly shrinks is worse than one that says so.
   */
  unreadable: number;
}

/**
 * Parse the JSONL body of `GET /v1/audit/export`.
 *
 * A row whose `actor` is a string this build does not know is UNREADABLE, not
 * "unknown". "unknown" is a specific fact — the seat writes it for entries
 * predating the `actor` field and refuses to backfill them (ledger.ts
 * `entryActor`) — and relabelling an actor this build merely failed to
 * recognise would manufacture that fact for a row that has a real one.
 */
export function parseAuditJsonl(text: string): ParsedTrail {
  const rows: AuditRow[] = [];
  let unreadable = 0;

  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      unreadable += 1;
      continue;
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      unreadable += 1;
      continue;
    }
    const record = parsed as Record<string, unknown>;
    const shaped = AUDIT_COLUMNS.every((column) =>
      column === "turn" ? typeof record.turn === "number" : typeof record[column] === "string",
    );
    if (
      !shaped ||
      !AUDIT_SOURCES.has(record.source as string) ||
      !ACTOR_VIEWS.has(record.actor as string)
    ) {
      unreadable += 1;
      continue;
    }
    rows.push(record as unknown as AuditRow);
  }

  return { rows, unreadable };
}

/**
 * The tool-call tail of a row's `detail`, or null when the row is not a tool
 * call or its payload cannot be read.
 *
 * The three nulls are passed through as nulls. `duration_ms ?? 0` and
 * `is_error ?? false` are both one character away from here and both would turn
 * "this tool was invoked and never returned" into "this tool returned instantly
 * and succeeded" — the single most misleading statement this screen could make.
 */
export function toolCallDetail(row: AuditRow): ToolCallDetail | null {
  if (row.source !== "tool_call") return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(row.detail);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const record = parsed as Record<string, unknown>;
  const endedAt = record.ended_at;
  const duration = record.duration_ms;
  const isError = record.is_error;
  return {
    args: record.args,
    ended_at: typeof endedAt === "string" ? endedAt : null,
    duration_ms: typeof duration === "number" ? duration : null,
    is_error: typeof isError === "boolean" ? isError : null,
  };
}

/**
 * What became of a row.
 *
 * Only tool calls have one. A ledger entry records a change that happened and a
 * retrieval records evidence that was returned; neither has a success/failure
 * axis, and inventing "ok" for them would let a reader count successes that
 * were never measured.
 */
export type Outcome = "ok" | "error" | "unterminated";

export function outcomeOf(row: AuditRow): Outcome | null {
  const detail = toolCallDetail(row);
  if (!detail) return null;
  if (detail.is_error === null) return "unterminated";
  return detail.is_error ? "error" : "ok";
}

/* ------------------------------------------------------------------ *
 * WHAT THE TRAIL SAYS
 * ------------------------------------------------------------------ */

export interface TrailSummary {
  rows: number;
  /** Every actor is present as a key, including zeroes: a screen that offers
   *  an actor filter must be able to say "none by this actor" rather than
   *  omitting the option and leaving a reader to assume it was never asked. */
  actors: Record<LedgerActorView, number>;
  sources: Record<AuditSource, number>;
  /** Distinct conversation ids in the window. */
  conversations: number;
  toolCalls: number;
  /** Tool calls the seat recorded as having returned an error. */
  failed: number;
  /** Tool calls with no end event — see `toolCallDetail`. */
  unterminated: number;
  /** Nearest-rank median over COMPLETED calls only. */
  durationP50: number | null;
  durationMax: number | null;
  /** Oldest and newest `ts` present, or null on an empty trail. */
  span: { from: string; to: string } | null;
}

/**
 * Nearest-rank percentile: the smallest observed value at or above the rank.
 *
 * Not an interpolating median. Every figure on this screen must be a duration
 * some call actually took — an interpolated 15ms between a 10ms call and a 20ms
 * one is a number that appears nowhere in the log a reviewer is holding.
 */
function nearestRank(sorted: readonly number[], quantile: number): number | null {
  if (sorted.length === 0) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(quantile * sorted.length) - 1));
  return sorted[index];
}

export function summarise(rows: readonly AuditRow[]): TrailSummary {
  const actors: Record<LedgerActorView, number> = { user: 0, agent: 0, system: 0, unknown: 0 };
  const sources: Record<AuditSource, number> = { ledger: 0, tool_call: 0, retrieval: 0, view: 0 };
  const conversations = new Set<string>();
  const durations: number[] = [];
  let failed = 0;
  let unterminated = 0;
  let from: string | null = null;
  let to: string | null = null;

  for (const row of rows) {
    actors[row.actor] += 1;
    sources[row.source] += 1;
    conversations.add(row.conversation_id);
    if (from === null || row.ts < from) from = row.ts;
    if (to === null || row.ts > to) to = row.ts;

    const detail = toolCallDetail(row);
    if (!detail) continue;
    if (detail.duration_ms !== null) durations.push(detail.duration_ms);
    if (detail.is_error === null) unterminated += 1;
    else if (detail.is_error) failed += 1;
  }

  durations.sort((a, b) => a - b);

  return {
    rows: rows.length,
    actors,
    sources,
    conversations: conversations.size,
    toolCalls: sources.tool_call,
    failed,
    unterminated,
    durationP50: nearestRank(durations, 0.5),
    durationMax: durations.length > 0 ? durations[durations.length - 1] : null,
    span: from !== null && to !== null ? { from, to } : null,
  };
}

export interface ToolStat {
  name: string;
  calls: number;
  failed: number;
  unterminated: number;
  /** Nearest-rank median and slowest, over this tool's COMPLETED calls. */
  p50: number | null;
  max: number | null;
}

/**
 * Which tool was used, how often, and how long it took — the aggregate answer
 * to the question this screen exists for.
 *
 * Ordered by call count, then by name. The name tiebreak is not cosmetic: a
 * sort with ties resolved by input order would reshuffle the list between two
 * reads of the same window, and a reader comparing this week's census to last
 * week's would see movement that is not there.
 */
export function toolCensus(rows: readonly AuditRow[]): ToolStat[] {
  const byName = new Map<string, { stat: ToolStat; durations: number[] }>();

  for (const row of rows) {
    const detail = toolCallDetail(row);
    if (!detail) continue;
    let entry = byName.get(row.kind);
    if (!entry) {
      entry = {
        stat: { name: row.kind, calls: 0, failed: 0, unterminated: 0, p50: null, max: null },
        durations: [],
      };
      byName.set(row.kind, entry);
    }
    entry.stat.calls += 1;
    if (detail.duration_ms !== null) entry.durations.push(detail.duration_ms);
    if (detail.is_error === null) entry.stat.unterminated += 1;
    else if (detail.is_error) entry.stat.failed += 1;
  }

  const stats: ToolStat[] = [];
  for (const { stat, durations } of byName.values()) {
    durations.sort((a, b) => a - b);
    stat.p50 = nearestRank(durations, 0.5);
    stat.max = durations.length > 0 ? durations[durations.length - 1] : null;
    stats.push(stat);
  }
  stats.sort((a, b) => (b.calls !== a.calls ? b.calls - a.calls : a.name < b.name ? -1 : 1));
  return stats;
}

/** Conversation ids present in the trail, oldest first appearance first — the
 *  picker's options, taken from the rows rather than from
 *  `GET /v1/conversations`, which is keyed on the forked `user_id` this screen
 *  refuses to filter by, and which lists conversations that produced no audit
 *  rows at all. */
export function conversationsIn(rows: readonly AuditRow[]): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const row of rows) {
    if (seen.has(row.conversation_id)) continue;
    seen.add(row.conversation_id);
    ids.push(row.conversation_id);
  }
  return ids;
}

/* ------------------------------------------------------------------ *
 * VIEW NARROWING
 *
 * Applied to rows already in hand, and NOT carried into the export — the file
 * covers the window and conversation the server was asked for. The surface
 * states that in as many words; a reviewer who thought the download matched
 * what was on screen would hand over a wider file than they meant to.
 * ------------------------------------------------------------------ */

export interface ViewFilter {
  /** Empty means every actor. */
  actors: ReadonlySet<LedgerActorView>;
  /** Empty means every source. */
  sources: ReadonlySet<AuditSource>;
}

export function matchesView(row: AuditRow, filter: ViewFilter): boolean {
  if (filter.actors.size > 0 && !filter.actors.has(row.actor)) return false;
  if (filter.sources.size > 0 && !filter.sources.has(row.source)) return false;
  return true;
}

/** Toggle one member of a filter set, returning a new set. */
export function toggle<T>(set: ReadonlySet<T>, value: T): Set<T> {
  const next = new Set(set);
  if (!next.delete(value)) next.add(value);
  return next;
}

export interface DayGroup {
  /** Local calendar day, `YYYY-MM-DD`. Stable as a React key and sortable. */
  day: string;
  rows: AuditRow[];
}

/**
 * Split the trail into local calendar days, preserving row order within each.
 *
 * LOCAL, NOT UTC, and the difference is not cosmetic. Every `ts` here is UTC,
 * and a reader east of Greenwich answering "what happened on Tuesday" means
 * their Tuesday — grouping on the ISO string's date prefix would file the first
 * five and a half hours of their day under the day before, on every row, in the
 * one product surface whose whole job is to say when something happened.
 *
 * A run of rows that leaves a day and comes back would open a second group for
 * it. That cannot arise from the export, which is sorted by `ts`, and the
 * alternative — keying groups by day and merging — would silently reorder rows
 * to fit, which this module never does.
 */
export function groupByDay(rows: readonly AuditRow[]): DayGroup[] {
  const groups: DayGroup[] = [];
  for (const row of rows) {
    const date = new Date(row.ts);
    const day = Number.isNaN(date.getTime())
      ? row.ts
      : `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(
          date.getDate(),
        ).padStart(2, "0")}`;
    const last = groups[groups.length - 1];
    if (last && last.day === day) last.rows.push(row);
    else groups.push({ day, rows: [row] });
  }
  return groups;
}

/* ------------------------------------------------------------------ *
 * WORDS AND FIGURES
 * ------------------------------------------------------------------ */

/**
 * A duration, or null when there is none to state.
 *
 * Null is returned rather than a dash so the caller has to decide what an
 * absent duration means in its context — on a tool call it means the call never
 * returned, which deserves a word and not a punctuation mark.
 *
 * The precision steps down as the number grows because that is what a reader
 * uses it for: 793ms and 812ms are different calls, 30.0s and 30.1s are the
 * same hang.
 */
export function formatDuration(ms: number | null): string | null {
  if (ms === null) return null;
  if (Math.abs(ms) < 1000) return `${ms}ms`;
  if (Math.abs(ms) < 10_000) return `${(ms / 1000).toFixed(1)}s`;
  if (Math.abs(ms) < 60_000) return `${Math.round(ms / 1000)}s`;
  const seconds = Math.round(Math.abs(ms) / 1000);
  const sign = ms < 0 ? "-" : "";
  return `${sign}${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

/**
 * What happened, in the words a person would use.
 *
 * A tool call is labelled with the TOOL'S OWN NAME and nothing else: the name
 * is the answer to "which tool was used", and wrapping it in a phrase would
 * bury the one string the reader came for. The other two sources carry event
 * type names that mean nothing outside this codebase, so they get a plain-word
 * label — and an unrecognised kind falls through to its raw value rather than
 * to "other", because an audit surface must not hide a kind it does not know.
 */
export function kindLabel(row: AuditRow): string {
  if (row.source === "tool_call") return row.kind;
  switch (row.kind) {
    case "memory_write":
      return "Wrote a memory";
    case "reinforce":
      return "Reinforced memories";
    case "implicit_feedback":
      return "Server adjusted memories";
    case "revert":
      return "Reverted a change";
    case "memory_recall":
      return "Searched memory";
    case "proactive_context":
      return "Surfaced context unasked";
    // NOT "Moved the view". The seat records the request and never learns the
    // verdict — the authority ledger that decides whether a command applied or
    // waited as a Follow lives in this browser and reports to nobody. A label
    // asserting the view moved would be the trail claiming an outcome it has no
    // evidence for, on the one screen whose entire value is that it does not.
    case "view_command":
      return "Asked to move the view";
    default:
      return row.kind;
  }
}

/**
 * Who acted.
 *
 * "unknown" keeps its own word and is never folded into "system". It marks a
 * ledger entry written before the seat recorded an actor at all, which the seat
 * deliberately does not backfill; a reader must be able to tell "nobody
 * recorded who did this" from "an automatic loop did this".
 */
export function actorLabel(actor: LedgerActorView): string {
  switch (actor) {
    case "user":
      return "Person";
    case "agent":
      return "Model";
    case "system":
      return "Automatic";
    case "unknown":
      return "Unknown";
  }
}

/** A long opaque id, shortened for a dense row. The full value stays on the
 *  row's detail, because a truncated id cannot be looked up anywhere. */
export function shortRef(id: string): string {
  return id.length <= 12 ? id : `${id.slice(0, 12)}…`;
}

/**
 * Time of day, to the second. The date is the group header above the row —
 * repeating it on ninety consecutive lines spends the width and is read once.
 *
 * A timestamp the platform cannot parse is returned VERBATIM. `new Date(x)`
 * yields an Invalid Date for anything malformed and every formatter then prints
 * the string "Invalid Date", which on an audit row reads as a rendering bug
 * rather than as a value worth looking at — and the raw string is the only
 * thing that would let anyone find the row in the exported file.
 */
export function clock(ts: string): string {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return ts;
  return date.toLocaleTimeString(undefined, { hour12: false });
}

/**
 * A day-group heading, from the `YYYY-MM-DD` key `groupByDay` produced.
 *
 * Parsed WITHOUT a zone suffix, which JavaScript reads as local midnight —
 * matching the local-day boundary the grouping itself used. Appending "Z" here
 * would name the group after a different day than the one it holds, for every
 * reader west of Greenwich.
 *
 * Falls back to the key for the same reason `clock` falls back to its input:
 * `groupByDay` puts an unparseable `ts` through as its own key, and that value
 * has to survive to the screen intact.
 */
export function dayLabel(day: string): string {
  const date = new Date(`${day}T00:00:00`);
  if (Number.isNaN(date.getTime())) return day;
  return date.toLocaleDateString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}
