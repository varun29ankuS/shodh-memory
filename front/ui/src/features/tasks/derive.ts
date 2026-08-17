import type { Todo, TodoPriority, TodoStatus } from "@/lib/api";
import { parseTime } from "@/features/briefing/derive";
import type { LinkedMemory, TriageProject, TriageTodo } from "./api";

/**
 * The arithmetic behind the Tasks screen, kept out of the component so it can
 * be pinned by tests.
 *
 * Everything here is a CLAIM the screen makes about a real list — a count, a
 * "12 overdue", a "showing 200 of 431" — and every one of them fails
 * invisibly. A miscounted overdue renders as a perfectly plausible number and
 * nobody ever checks it against the rows.
 */

/**
 * The row limit this screen asks the server for. Stated here because the
 * reduction notice has to quote the same number the request used, and two
 * copies of it is how a screen ends up claiming "first 200 of 431" while
 * showing 100 rows.
 *
 * RAISED FROM 200 BECAUSE EVERY DENOMINATOR ON THIS SCREEN NOW DEPENDS ON IT.
 * The completion meters count settled against total over the rows in hand, and
 * the server paginates AFTER sorting by `sort_order`, priority and due date
 * (src/handlers/todos.rs:1007-1012, 1419-1432) — an order that has nothing to
 * do with project or status. So a truncated response is an arbitrary slice
 * across every project at once, and "13 of 13 done" computed over it would be
 * a fabricated number of exactly the kind this module exists to refuse.
 *
 * 1000 is the smallest round figure that is not a real constraint here: the
 * server's own ceiling is 10,000 (`MAX_LIMIT`, src/validation.rs:445 — verified
 * live, a limit of 100,000 is rejected with INVALID_INPUT), and the largest
 * profile on this instance holds 93 todos. Truncation is still reported rather
 * than assumed away, and it still disables the meters — see `boardOf`.
 */
export const TASK_LIMIT = 1000;

/**
 * Mirrors `Todo::short_id()` (src/memory/types.rs:3776-3784) exactly, including
 * its "SHO" fallback prefix and its first-4-hex-chars fallback for legacy todos
 * with no `seq_num`. Neither is invented here.
 */
export function shortId(todo: Todo): string {
  if (todo.seq_num > 0) return `${todo.project_prefix ?? "SHO"}-${todo.seq_num}`;
  return `SHO-${todo.id.slice(0, 4)}`;
}

export interface DueMeta {
  label: string;
  tone: "destructive" | "warn" | "muted";
  /** Past its due date. Counted in the header, so it is derived once here
   *  rather than re-tested against a second, slightly different rule. */
  overdue: boolean;
}

/** Local midnight for an instant. Two dates are "the same day" to a reader if
 *  they share this, and to nobody if they are within 86,400,000ms of each
 *  other — which is what the arithmetic this replaces was actually testing. */
function startOfLocalDay(ms: number): number {
  const d = new Date(ms);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

/**
 * `Todo::is_overdue()` (src/memory/types.rs:4216-4224), re-derived: the server
 * does not send a boolean, only `due_date`. Its rule is `Utc::now() > due`,
 * strictly, with no day of grace.
 *
 * THE ARITHMETIC IS CALENDAR DAYS, NOT 24-HOUR WINDOWS. A `Math.ceil` over the
 * raw millisecond difference got both ends of the scale wrong on the same
 * clock: a todo due at 23:00 today read "Due in 1d", and a todo that went past
 * its due date an hour ago read "Due today" in the warn colour — an OVERDUE
 * item rendered as one still in hand, which is the failure that matters. Days
 * are now counted between local midnights, and lateness is the server's own
 * strict comparison.
 *
 * `now` is a parameter rather than a call to `Date.now()` so those boundaries
 * can be tested at all. A malformed or absent date yields null rather than an
 * "Invalid Date" chip.
 */
export function dueMeta(todo: Todo, now: number): DueMeta | null {
  if (!todo.due_date) return null;
  const due = new Date(todo.due_date);
  const dueMs = due.getTime();
  if (Number.isNaN(dueMs)) return null;

  if (now > dueMs) {
    const late = Math.floor((startOfLocalDay(now) - startOfLocalDay(dueMs)) / 86_400_000);
    // Under a day late has no useful number in it; "Overdue 0d" is worse than
    // the word on its own.
    return {
      label: late >= 1 ? `Overdue ${late}d` : "Overdue",
      tone: "destructive",
      overdue: true,
    };
  }

  // Math.round, not floor: a daylight-saving boundary makes one of these
  // spans 23 or 25 hours, and a floor turns that into an off-by-one day.
  const days = Math.round((startOfLocalDay(dueMs) - startOfLocalDay(now)) / 86_400_000);
  if (days === 0) return { label: "Due today", tone: "warn", overdue: false };
  if (days <= 3) return { label: `Due in ${days}d`, tone: "warn", overdue: false };
  return { label: `Due ${due.toLocaleDateString()}`, tone: "muted", overdue: false };
}

/** Priority as a token every row can carry. `none` is the server's own
 *  "unset" (src/memory/types.rs:3452) and is rendered as nothing at all —
 *  printing "none" on a row would read as a deliberate downgrade. */
export function priorityLabel(priority: TodoPriority): string | null {
  switch (priority) {
    case "urgent":
      return "urgent";
    case "high":
      return "high";
    case "medium":
      return "med";
    case "low":
      return "low";
    case "none":
      return null;
  }
}

export interface TaskSummary {
  /** Rows actually on screen. */
  shown: number;
  /** `TodoListResponse.count` — the total BEFORE the server's `.truncate(limit)`
   *  (src/handlers/todos.rs:1419-1432), so this is the only place the screen can
   *  learn that it is showing a slice. */
  total: number;
  truncated: boolean;
  urgent: number;
  high: number;
  overdue: number;
  /** Distinct project prefixes present among the rows. One project means the
   *  chip repeats the same three letters down fifty rows and is dropped. */
  projects: number;
}

/* ------------------------------------------------------------------ *
 * PROVENANCE
 *
 * A todo's `related_memory_ids` is documented server-side as "the 'why does
 * this task exist' link" (src/handlers/todos.rs:195-196). It is not only that,
 * and the difference is the whole reason this classifier exists.
 *
 * Two things write that list, and nothing on the wire tells them apart:
 *
 *   1. A CALLER passing memory ids at create or update time
 *      (src/handlers/todos.rs:1078-1079, 1632-1633). This is a real source.
 *      It is also entirely optional, and no first-party client sends it — the
 *      session hook that creates todos from Claude Code task events
 *      (hooks/memory-hook.ts:976-1006) passes none.
 *
 *   2. The SERVER, on every create, update, complete and comment, which
 *      writes a new memory RESTATING the todo and links it back
 *      (src/handlers/todos.rs:1174-1180, 1773-1777, 1912-1916, 2304-2307).
 *      The content of one reads "[SHO-1] Todo created: <the todo's title>".
 *
 * So the default state of a task created through this product is a memory link
 * that leads to a restatement of itself. Rendering that under "where did this
 * come from" would answer the question with the question. These are separated
 * by the tags the four echo writers set (todos.rs:1132-1138, 1728-1737,
 * 1867-1876, 2267-2275) — the only signature they leave.
 * ------------------------------------------------------------------ */

/** Exact lifecycle markers written by the four echo sites. `todo-comment:` is
 *  a prefix (`todo-comment:resolution` and friends), the rest are literal. */
const ECHO_TAGS = ["todo-created", "todo-updated", "todo-completed"];
const ECHO_TAG_PREFIX = "todo-comment:";

export type LinkKind =
  /** A memory this task's own lifecycle wrote. Not provenance. */
  | "echo"
  /** Not written by any todo lifecycle: a memory that existed independently. */
  | "source";

/**
 * Classify one linked memory.
 *
 * Judged on the lifecycle tags alone, NOT on `experience_type`: the create and
 * complete echoes are `Task`-typed but the update echo is `Context`-typed
 * (todos.rs:1741, 1879), and a genuine source memory is free to be any of
 * them. Type is not the signature; the tags are.
 */
export function classifyLink(memory: LinkedMemory): LinkKind {
  const tags = memory.experience?.tags ?? [];
  for (const tag of tags) {
    if (ECHO_TAGS.includes(tag)) return "echo";
    if (tag.startsWith(ECHO_TAG_PREFIX)) return "echo";
  }
  return "source";
}

export interface Provenance {
  /** Memories that existed independently of this task. The answer to "where
   *  did this come from", when there is one. */
  sources: LinkedMemory[];
  /** Links that are this task's own lifecycle restating itself. Counted so the
   *  screen can say what the other links were, rather than hiding them and
   *  leaving a "3 linked" chip unaccounted for. */
  echoes: number;
}

export function provenanceOf(memories: LinkedMemory[]): Provenance {
  const sources: LinkedMemory[] = [];
  let echoes = 0;
  for (const memory of memories) {
    if (classifyLink(memory) === "echo") echoes += 1;
    else sources.push(memory);
  }
  return { sources, echoes };
}

/* ------------------------------------------------------------------ *
 * ORIGIN
 *
 * There is NO field recording whether a todo was typed by a person or written
 * by an agent. The `Todo` struct (src/memory/types.rs:4067-4162) has no
 * `origin`, `source`, `confidence` or span of any kind, and nothing in the
 * backend extracts a todo from memory text — every todo is created by an
 * explicit call.
 *
 * One partial signal does exist, and only for one writer: the session hook
 * stamps `external_id: "claude-task:{id}"` and tags `source:hook`
 * (hooks/memory-hook.ts:1004-1005). That is reported where present and nothing
 * is inferred where it is absent — an unstamped todo is unknown, not "typed by
 * a human".
 * ------------------------------------------------------------------ */

export type Origin =
  | { kind: "session-hook"; label: string }
  | { kind: "external"; label: string }
  | { kind: "unrecorded" };

export function originOf(todo: TriageTodo): Origin {
  const external = todo.external_id;
  if (external && external.startsWith("claude-task:")) {
    return { kind: "session-hook", label: "recorded by a session hook" };
  }
  if (todo.tags.includes("source:hook")) {
    return { kind: "session-hook", label: "recorded by a session hook" };
  }
  if (external) {
    // "todoist:123", "linear:SHO-39" — the scheme is the system of record.
    const scheme = external.slice(0, external.indexOf(":"));
    return { kind: "external", label: `synced from ${scheme || external}` };
  }
  return { kind: "unrecorded" };
}

/* ------------------------------------------------------------------ *
 * SETTLED WORK
 * ------------------------------------------------------------------ */

/**
 * The reason a task was settled, if one was recorded.
 *
 * Written as a `resolution` comment (src/memory/types.rs:4061-4062) and read
 * back off the `comments` array that ships inline on every listed todo
 * (types.rs:4143), so no extra request is needed to show it.
 *
 * The LAST resolution wins: a task dismissed, reopened and dismissed again has
 * two, and the current one is the reason it is settled now. Returns null when
 * none was recorded — which is the normal case for anything cancelled outside
 * this screen, and must read as "no reason recorded" rather than as a blank
 * that implies one.
 */
export function settledReason(todo: TriageTodo): string | null {
  const comments = todo.comments ?? [];
  for (let i = comments.length - 1; i >= 0; i -= 1) {
    const comment = comments[i];
    if (comment.comment_type === "resolution") {
      const text = comment.content.trim();
      if (text.length > 0) return text;
    }
  }
  return null;
}

/**
 * Is this list a slice, and of what?
 *
 * `TodoListResponse.count` is the total BEFORE the server truncates
 * (src/handlers/todos.rs:1421), so it is the only way any list on this screen
 * learns it is showing part of one. Both lists here are capped, so both need
 * this — a Settled section that renders 200 of 300 dismissals under a heading
 * reading "200" is the same unmarked reduction the open list refuses to make.
 *
 * A server reporting a smaller total than the rows it sent would make
 * `truncated` nonsense, so the total is clamped rather than rendered as
 * "showing 50 of 12".
 */
export function truncation(shown: number, total: number): { total: number; truncated: boolean } {
  return { total: Math.max(total, shown), truncated: total > shown };
}

export function summarise(todos: Todo[], total: number, now: number): TaskSummary {
  const prefixes = new Set<string>();
  let urgent = 0;
  let high = 0;
  let overdue = 0;
  for (const todo of todos) {
    if (todo.project_prefix) prefixes.add(todo.project_prefix);
    if (todo.priority === "urgent") urgent += 1;
    if (todo.priority === "high") high += 1;
    if (dueMeta(todo, now)?.overdue) overdue += 1;
  }
  return {
    shown: todos.length,
    ...truncation(todos.length, total),
    urgent,
    high,
    overdue,
    projects: prefixes.size,
  };
}

/* ================================================================== *
 * WHEN THE WORK MOVED
 *
 * THE FIELDS DO NOT RECORD THIS AND THE ACTIVITY LOG DOES. That is the
 * measurement this whole section exists for, and it is not a nicety:
 *
 *   - `updated_at` is the LAST change of any kind, so it cannot say when work
 *     began.
 *   - `completed_at` is set in exactly one place — `Todo::complete()`
 *     (src/memory/types.rs:4251) — which only `POST /todos/{id}/complete`
 *     reaches. Setting `status: "done"` through the UPDATE handler never
 *     touches it, and cancelling never touches it at all.
 *
 * Measured on the live `claude-code` profile: of 82 settled todos, 39 carry
 * `completed_at` and 43 DO NOT — 20 marked done through update, 23 cancelled.
 * For those 43 the activity log is the only record that they ever settled, let
 * alone when. A screen that read `completed_at` alone would show more than half
 * of the finished work as having no completion date, and any cycle-time figure
 * over it would be computed from the unrepresentative half that happened to use
 * one of the two endpoints.
 *
 * Every one of the 82 is recoverable from the two sources together — verified
 * live, zero unaccounted.
 * ================================================================== */

/** The Debug-formatted `TodoStatus` names the server writes into its activity
 *  text (`format!("status → {:?}", todo.status)`, src/handlers/todos.rs:1682),
 *  mapped to the snake_case values the same enum serialises as on the wire.
 *  The two spellings are genuinely different and both come from one enum. */
const DEBUG_STATUS: Record<string, TodoStatus> = {
  Backlog: "backlog",
  Todo: "todo",
  InProgress: "in_progress",
  Blocked: "blocked",
  Done: "done",
  Cancelled: "cancelled",
};

/** `Updated: status → InProgress, priority → High` — the status clause is
 *  always first when present (the change list is built in field order,
 *  todos.rs:1680-1707) and the rest is ignored here. U+2192, not "->". */
const STATUS_LINE = /^Updated: status → (\w+)/;

/** `Marked complete after 0.0 days` (todos.rs:1855). The DURATION IN IT IS
 *  NEVER PARSED — it is `{:.1}` rounded, so "0.0 days" covers everything under
 *  about 72 minutes, and a screen quoting it would be restating the server's
 *  rounding as a measurement. Only the fact and the comment's own timestamp are
 *  taken. */
const COMPLETE_LINE = /^Marked complete after /;

export interface StatusChange {
  at: number;
  to: TodoStatus;
}

/**
 * Every recorded status change on one todo, oldest first.
 *
 * GATED ON THE AUTHOR, NOT ONLY THE TEXT. `TodoComment::system_activity`
 * (src/memory/types.rs:4037-4047) hardcodes `author: "system"` and
 * `comment_type: Activity`, and it is the only writer of these strings
 * server-side. Callers may set both fields freely
 * (`AddCommentRequest.author`/`comment_type`, todos.rs:339-346), so a person or
 * an agent can post a comment reading "Updated: status → Done" — and without
 * this gate that comment would be read back as a state transition that never
 * happened. Author is the one field on a comment the server sets itself.
 *
 * SORTED BY TIMESTAMP RATHER THAN TRUSTING ARRAY ORDER. The array is appended
 * to (`Todo::add_activity`, types.rs:4301-4305) so it is in order today, but
 * ordering is what "first started" and "last settled" below are read off, and
 * inheriting that assumption silently would make both wrong the day anything
 * rewrites the list.
 *
 * A creation entry is NOT a status change. "Created in project 'X'" records no
 * status, and `CreateTodoRequest.status` (todos.rs:170) means a todo can be
 * created directly as `in_progress` or `done` — so the initial status is
 * genuinely unknown from the log and is not guessed at.
 */
export function statusChanges(todo: TriageTodo): StatusChange[] {
  const changes: StatusChange[] = [];
  for (const comment of todo.comments ?? []) {
    if (comment.author !== "system" || comment.comment_type !== "activity") continue;
    const at = parseTime(comment.created_at);
    if (at === null) continue;

    const matched = STATUS_LINE.exec(comment.content);
    if (matched) {
      const to = DEBUG_STATUS[matched[1]];
      // An unrecognised variant name is dropped rather than coerced. It means
      // this build is older than the server's enum, and inventing a status for
      // it would be worse than showing one fewer transition.
      if (to) changes.push({ at, to });
      continue;
    }
    if (COMPLETE_LINE.test(comment.content)) changes.push({ at, to: "done" });
  }
  return changes.sort((a, b) => a.at - b.at);
}

export interface Lifeline {
  /** `created_at`. Always present — it is a required field. */
  recorded: number;
  /** The FIRST move into `in_progress`. First, not last: a todo can be pushed
   *  back to `todo` and picked up again (live data carries such a reversal),
   *  and the question this answers is when work began, not when it last
   *  resumed. Null when no start was ever recorded — including for a todo
   *  created directly as `in_progress`, whose start leaves no entry. */
  started: number | null;
  /** When it reached `done` or `cancelled`: `completed_at` when the server set
   *  one, else the last settling transition in the log. Null while open —
   *  INCLUDING for a task that was completed once and reopened, see below. */
  settled: number | null;
  /** True when `settled` came only from the log because no `completed_at` was
   *  ever written. 43 of 82 settled todos on the live profile. */
  settledFromLog: boolean;
}

export function lifelineOf(todo: TriageTodo): Lifeline | null {
  const recorded = parseTime(todo.created_at);
  if (recorded === null) return null;

  const changes = statusChanges(todo);
  const started = changes.find((c) => c.to === "in_progress")?.at ?? null;

  /*
   * THE CURRENT STATUS GATES BOTH SOURCES, AND `completed_at` ESPECIALLY.
   *
   * `completed_at` is set by `Todo::complete()` and is never CLEARED by
   * anything: there is no assignment to it anywhere in src/handlers/todos.rs,
   * and the only two `completed_at = None` sites in the tree are a recurrence
   * rollover building a fresh todo (types.rs:4266) and a Project (todos.rs:1361).
   *
   * So a task completed through the complete endpoint and then reopened keeps
   * its old stamp for good — and reopening is a button on this very screen,
   * which routes through the update handler. Reading the stamp without checking
   * the status would put "took 2d" on a task sitting in To do, and would let
   * `laneCurve` count it as settled while the status-based meter beside it did
   * not. The curve and the number would disagree about the same project.
   */
  const isSettled = todo.status === "done" || todo.status === "cancelled";
  let settled: number | null = null;
  let settledFromLog = false;
  if (isSettled) {
    settled = parseTime(todo.completed_at);
    if (settled === null) {
      for (const change of changes) {
        if (change.to === "done" || change.to === "cancelled") {
          settled = change.at;
          settledFromLog = true;
        }
      }
    }
  }

  return { recorded, started, settled, settledFromLog };
}

/** Has this task ever changed state? The predicate the whole screen turns on:
 *  a profile where this is false everywhere is a list of rows nobody has ever
 *  acted on, which is a fact worth stating outright rather than drawing. */
export function hasMoved(todo: TriageTodo): boolean {
  return statusChanges(todo).length > 0;
}

/* ================================================================== *
 * BLOCKED — TWO DIFFERENT THINGS THAT MUST NOT SHARE A TREATMENT
 *
 * Linear settles this by making blocking a RELATION and not a status at all:
 * its four relation types are Related, Blocked by, Blocks and Duplicate, drawn
 * as flags in the issue's properties sidebar, and "blocked" appears nowhere in
 * its workflow states (linear.app/docs/issue-relations).
 *
 * This model has BOTH, plus a third thing, and they answer different questions:
 *
 *   - `status: "blocked"` — the enum's own "Waiting for someone/something"
 *     (src/memory/types.rs:3835-3836). A declaration, with no object.
 *   - `blocked_on: Option<String>` — free text. WHO or WHAT is being waited on
 *     when it is not another task: a person, a vendor, a decision.
 *   - `blocked_by: Vec<TodoId>` — real todo references, resolved and
 *     cycle-checked server-side (src/memory/todos.rs:804-867).
 *
 * "Waiting on a person" and "waiting on another task" are therefore separable
 * here in a way they are not in most trackers, and the second is actionable in
 * a way the first is not — a blocker task has a status of its own, so the
 * screen can say whether the thing being waited on is itself moving.
 * ================================================================== */

export type Blocker =
  /** A real todo, resolved against the rows in hand. */
  | { kind: "task"; id: string; todo: TriageTodo }
  /** A todo reference that is not in the fetched set. NOT dropped: a dependency
   *  that exists and cannot be shown is different from no dependency, and
   *  silently omitting it would understate what a task is waiting on. */
  | { kind: "task-missing"; id: string }
  /** The free-text `blocked_on`. A person or a thing, not a task. */
  | { kind: "waiting"; text: string };

export function blockersOf(todo: TriageTodo, byId: ReadonlyMap<string, TriageTodo>): Blocker[] {
  const blockers: Blocker[] = [];
  for (const id of todo.blocked_by ?? []) {
    const found = byId.get(id);
    blockers.push(found ? { kind: "task", id, todo: found } : { kind: "task-missing", id });
  }
  // Trimmed and emptiness-checked because the update handler writes the field
  // through verbatim, and clearing it produces "" rather than null — live data
  // carries an activity entry reading "blocked on: " with nothing after it.
  const waiting = todo.blocked_on?.trim();
  if (waiting) blockers.push({ kind: "waiting", text: waiting });
  return blockers;
}

/** A blocker task that is itself settled is no longer holding anything up, and
 *  the server agrees — `unblocked_by_completion` (src/memory/todos.rs:869-895)
 *  treats Done and Cancelled blockers as satisfied. A chain still listing one
 *  is stale, not blocking, and reads differently. */
export function blockerIsSatisfied(blocker: Blocker): boolean {
  return blocker.kind === "task" && (blocker.todo.status === "done" || blocker.todo.status === "cancelled");
}

/* ================================================================== *
 * HOW MUCH IS DONE
 *
 * THE ONLY HONEST DENOMINATOR ON THIS CORPUS IS A COUNT OF TASKS. There is no
 * estimate, no story point and no size field anywhere on `Todo`
 * (src/memory/types.rs:4067-4162), so there is nothing to weight by.
 *
 * Linear lands in the same place whenever a team has not enabled estimates:
 * "When estimates are not enabled, we calculate statistics using a default
 * value of 1 estimate point per issue" (linear.app/docs/estimates), and its
 * project graph "treats all issues as 1 estimate point" in that case
 * (linear.app/docs/project-graph). Our permanent condition is their fallback,
 * so a count-based ratio is the same measure they ship rather than a
 * simplification of it.
 *
 * WHAT IS DELIBERATELY NOT BORROWED: Linear's project graph applies "a 1/4
 * modifier for any in-progress issues". That number exists to smooth a VELOCITY
 * FORECAST, not to describe a task — and this screen makes no forecast, having
 * neither estimates nor cycles. Rendering an in-progress task as 25% done would
 * put a fraction on work whose partial progress nobody measured. Underway is
 * counted and named; it is not scored.
 *
 * NO TASK EVER GETS A PERCENTAGE. Status is the whole of what is known about
 * one task. A number appears only where there is a real population to count
 * over: a project, or a parent's subtasks.
 * ================================================================== */

/** The lane key for todos belonging to no project. Empty string cannot collide
 *  with a UUID. */
export const NO_PROJECT = "";

export interface Lane {
  /** `project_id`, or `NO_PROJECT`. */
  key: string;
  name: string;
  prefix: string | null;
  archived: boolean;
  todos: TriageTodo[];
  total: number;
  done: number;
  cancelled: number;
  /** done + cancelled. Both are settled: a cancelled task is not outstanding
   *  work, and counting it as open would make a project that abandoned half its
   *  scope look permanently unfinished. They are reported separately too,
   *  because "we finished it" and "we dropped it" are not the same outcome. */
  settled: number;
  open: number;
  underway: number;
  blocked: number;
  /** Earliest `created_at` and latest movement among these todos. */
  from: number;
  to: number;
  /** How many of these have ever changed state. */
  moved: number;
}

/**
 * Group todos into project lanes.
 *
 * KEYED ON `project_id`, NEVER ON `project_prefix`, AND THE DIFFERENCE IS LIVE.
 * The `claude-code` profile has two distinct projects both carrying the prefix
 * "SHOD" — "shodh-memory" (archived, 39 todos) and "Shodh-redb" (active, 1) —
 * because the prefix is derived from the name (`Project::derive_prefix`) and is
 * not unique. Grouping by prefix would merge a finished project into a running
 * one and report a single wrong ratio over both.
 *
 * The prefix is still carried, because it is what the short ids on the rows are
 * built from and a lane whose rows all read "SHOD-n" must say so.
 *
 * Projects with no todos in hand are NOT given empty lanes: the profile's whole
 * project list ships with every response (todos.rs:1435-1438), and six of the
 * nine on the live profile would otherwise draw a lane reading "0 of 0".
 */
export function lanesOf(todos: TriageTodo[], projects: readonly TriageProject[]): Lane[] {
  const meta = new Map(projects.map((p) => [p.id, p]));
  const lanes = new Map<string, Lane>();

  for (const todo of todos) {
    const key = todo.project_id ?? NO_PROJECT;
    let lane = lanes.get(key);
    if (!lane) {
      const project = key === NO_PROJECT ? undefined : meta.get(key);
      lane = {
        key,
        // A todo can name a project the project list does not contain; the id
        // is then all there is and the lane says exactly that rather than
        // rendering a blank heading.
        name: key === NO_PROJECT ? "No project" : (project?.name ?? "Unnamed project"),
        prefix: project?.prefix ?? todo.project_prefix ?? null,
        archived: project?.status === "archived" || project?.status === "completed",
        todos: [],
        total: 0,
        done: 0,
        cancelled: 0,
        settled: 0,
        open: 0,
        underway: 0,
        blocked: 0,
        from: Number.POSITIVE_INFINITY,
        to: Number.NEGATIVE_INFINITY,
        moved: 0,
      };
      lanes.set(key, lane);
    }

    lane.todos.push(todo);
    lane.total += 1;
    if (todo.status === "done") lane.done += 1;
    else if (todo.status === "cancelled") lane.cancelled += 1;
    else {
      lane.open += 1;
      if (todo.status === "in_progress") lane.underway += 1;
      if (todo.status === "blocked") lane.blocked += 1;
    }

    const line = lifelineOf(todo);
    if (line) {
      lane.from = Math.min(lane.from, line.recorded);
      lane.to = Math.max(lane.to, line.settled ?? line.started ?? line.recorded);
    }
    if (hasMoved(todo)) lane.moved += 1;
  }

  for (const lane of lanes.values()) {
    lane.settled = lane.done + lane.cancelled;
    // A lane whose every todo had an unreadable created_at would otherwise
    // carry infinities into the axis arithmetic.
    if (!Number.isFinite(lane.from)) lane.from = 0;
    if (!Number.isFinite(lane.to)) lane.to = lane.from;
  }

  return [...lanes.values()].sort(laneOrder);
}

/**
 * Lane order: running work first, finished work after it, unfiled last.
 *
 * Linear's My Issues groups in the same spirit — "urgent work, SLA-bound work,
 * blockers, cycle work, other active work, triage, backlog, and completed work"
 * (linear.app/docs/my-issues) — with completed work last. The principle taken
 * is that ordering follows what still needs a decision, not alphabetical or
 * chronological neatness. The specific groups are not taken: there is no SLA,
 * cycle or triage concept in this model and inventing lanes for them would be
 * chrome with nothing behind it.
 *
 * Within a tier, most recently active first, so a lane that moved this week
 * outranks one that has been quiet for months.
 */
function laneOrder(a: Lane, b: Lane): number {
  const tier = (lane: Lane) => (lane.key === NO_PROJECT ? 2 : lane.archived ? 1 : 0);
  const byTier = tier(a) - tier(b);
  if (byTier !== 0) return byTier;
  return b.to - a.to;
}

export interface SubtaskProgress {
  done: number;
  total: number;
}

/**
 * Completion across a parent's subtasks, or null when it has none.
 *
 * NULL RATHER THAN ZERO, AND ONLY WHERE CHILDREN EXIST. This is the one place
 * a task-level ratio has a real population behind it. Every other task gets no
 * number, because a percentage derived from a single task's status would be
 * invented — the reason `in_progress` is reported as a word and never as a
 * fraction.
 *
 * NOTHING ON THE LIVE INSTANCE USES `parent_id`: zero of 143 todos across four
 * profiles carry one. This is written against the field rather than against the
 * corpus because the field is real, the create and update handlers both accept
 * it (todos.rs:182, 288), and the list endpoint returns subtasks inline with
 * everything else — but nothing on screen will render from it today, and the
 * surface must not imply otherwise.
 */
export function subtaskProgress(parent: TriageTodo, todos: readonly TriageTodo[]): SubtaskProgress | null {
  let done = 0;
  let total = 0;
  for (const todo of todos) {
    if (todo.parent_id !== parent.id) continue;
    total += 1;
    // Cancelled counts as settled for the same reason it does in a lane: a
    // dropped subtask is not outstanding, and leaving it in the denominator
    // would strand a parent at "3 of 4" forever.
    if (todo.status === "done" || todo.status === "cancelled") done += 1;
  }
  return total === 0 ? null : { done, total };
}

/* ================================================================== *
 * WHAT THE HEADER IS ALLOWED TO SAY
 * ================================================================== */

export interface Board {
  shown: number;
  total: number;
  /** When true EVERY RATIO ON THIS SCREEN IS SUPPRESSED. The server paginates
   *  after sorting by manual order, priority and due date — never by project —
   *  so a truncated response is an arbitrary slice across all projects at once
   *  and no lane's denominator is its own. */
  truncated: boolean;
  open: number;
  underway: number;
  blocked: number;
  settled: number;
  done: number;
  cancelled: number;
  /** Lanes with at least one todo in hand. */
  projects: number;
  /** Todos that have ever changed state. Zero means nothing here has moved. */
  moved: number;
  /** Todos naming another todo as a blocker. Zero across every live profile. */
  dependencies: number;
  /** Todos carrying free-text `blocked_on`. */
  waiting: number;
  /** Span of `created_at` across the rows, for the "recorded in one sitting"
   *  finding. Null when nothing could be read. */
  from: number | null;
  to: number | null;
}

export function boardOf(todos: TriageTodo[], total: number, lanes: readonly Lane[]): Board {
  let open = 0;
  let underway = 0;
  let blocked = 0;
  let done = 0;
  let cancelled = 0;
  let moved = 0;
  let dependencies = 0;
  let waiting = 0;
  let from: number | null = null;
  let to: number | null = null;

  for (const todo of todos) {
    if (todo.status === "done") done += 1;
    else if (todo.status === "cancelled") cancelled += 1;
    else {
      open += 1;
      if (todo.status === "in_progress") underway += 1;
      if (todo.status === "blocked") blocked += 1;
    }
    if (hasMoved(todo)) moved += 1;
    if ((todo.blocked_by ?? []).length > 0) dependencies += 1;
    if (todo.blocked_on?.trim()) waiting += 1;

    const at = parseTime(todo.created_at);
    if (at !== null) {
      from = from === null ? at : Math.min(from, at);
      to = to === null ? at : Math.max(to, at);
    }
  }

  return {
    shown: todos.length,
    ...truncation(todos.length, total),
    open,
    underway,
    blocked,
    settled: done + cancelled,
    done,
    cancelled,
    projects: lanes.length,
    moved,
    dependencies,
    waiting,
    from,
    to,
  };
}

/**
 * The time axis the lane strips share, or null when there is nothing to draw.
 *
 * NULL WHEN NOTHING HAS EVER MOVED, and that is the whole rule. A strip is a
 * picture of work progressing; with no transitions anywhere, all it can plot is
 * the instants at which rows were written down, which is a picture of an import
 * and reads as one only if you already know that. The live `claude` profile is
 * exactly this case — 50 todos, all created inside 33 minutes on one day, not
 * one of them ever moved — and it gets a sentence stating that instead, which
 * says more than the drawing could.
 *
 * A SHARED AXIS ACROSS ALL LANES, not one per lane. Per-lane axes would draw a
 * project that ran for a day and one that ran for four months at the same
 * width, which inverts the comparison the strips exist to support.
 */
export interface Axis {
  from: number;
  to: number;
  span: number;
}

export function axisOf(lanes: readonly Lane[], board: Board): Axis | null {
  if (board.moved === 0) return null;
  let from = Number.POSITIVE_INFINITY;
  let to = Number.NEGATIVE_INFINITY;
  for (const lane of lanes) {
    if (lane.total === 0) continue;
    from = Math.min(from, lane.from);
    to = Math.max(to, lane.to);
  }
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null;
  // A zero span would divide by zero in every position below. One millisecond
  // of width puts every mark at the left edge, which is truthful for a set of
  // events that genuinely share an instant.
  const span = Math.max(to - from, 1);
  return { from, to, span };
}

/** Where an instant sits on the axis, as a 0..1 fraction. Clamped: a todo
 *  settled before it was recorded (clock skew between two writes) would
 *  otherwise place a mark outside the track. */
export function positionOn(axis: Axis, at: number): number {
  return Math.min(1, Math.max(0, (at - axis.from) / axis.span));
}

/* ================================================================== *
 * WHO WROTE A COMMENT, AND WHAT THAT IS WORTH
 * ================================================================== */

export type AuthorKind =
  /** Set by the server itself. `TodoComment::system_activity`
   *  (src/memory/types.rs:4037-4047) hardcodes it, so it is the one author
   *  value on the wire that is evidence rather than a claim. */
  | { kind: "server" }
  /** Written from this dashboard, which is the only thing that sets it. */
  | { kind: "dashboard" }
  /** Written by a model through the seat's todo tools, which sign every
   *  mutation `agent:…` because `Todo` has no assignee, executor or actor field
   *  to sign in (seat/src/todo-tools.ts). `model` is whatever follows the
   *  prefix, or null when nothing does. */
  | { kind: "agent"; model: string | null }
  /** Any other name. `AddCommentRequest.author` defaults to `user_id`
   *  (src/handlers/todos.rs:2233), so most of these are just the profile name,
   *  and none of them is verified by anything. */
  | { kind: "caller"; name: string };

/** The prefix the seat's todo tools sign with (`agentAuthor`,
 *  seat/src/todo-tools.ts). Matched as a PREFIX rather than an exact string
 *  deliberately: the intent there is to append the model that did the work, so
 *  this must keep reading both the bare marker and a fuller one. */
const AGENT_PREFIX = "agent:";

/** The name this dashboard signs with. Duplicated from api.ts's
 *  `DASHBOARD_AUTHOR` as a value comparison rather than imported, so that
 *  derive stays free of the request layer. */
const DASHBOARD = "shodh-dashboard";

export function authorKind(author: string): AuthorKind {
  if (author === "system") return { kind: "server" };
  if (author === DASHBOARD) return { kind: "dashboard" };
  if (author.startsWith(AGENT_PREFIX)) {
    const model = author.slice(AGENT_PREFIX.length).trim();
    return { kind: "agent", model: model.length > 0 ? model : null };
  }
  return { kind: "caller", name: author };
}

/**
 * How long something took, in the coarsest unit that still says something.
 *
 * BUILT ON TOTAL MINUTES WITH FLOOR AND MODULO, NOT ON NESTED ROUNDING. The
 * failure this shape avoids has bitten this codebase before: rounding a
 * remainder independently of the unit above it produces "1h 60m", which is
 * wrong and reads as plausible. Here the rounding happens exactly once, to
 * minutes, and every larger unit is derived from that single number, so a
 * remainder can never reach its own base.
 *
 * Null for a negative span. Two writes on different clocks can settle a todo
 * before it was recorded, and "-3h" beside a finished task is worse than
 * nothing at all.
 */
export function elapsedLabel(from: number, to: number): string | null {
  const ms = to - from;
  if (!Number.isFinite(ms) || ms < 0) return null;
  const minutes = Math.round(ms / 60_000);
  if (minutes < 1) return "under a minute";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const rest = minutes % 60;
    return rest === 0 ? `${hours}h` : `${hours}h ${rest}m`;
  }
  const days = Math.floor(hours / 24);
  const rest = hours % 24;
  return rest === 0 ? `${days}d` : `${days}d ${rest}h`;
}

export interface CurvePoint {
  /** 0..1 along the shared axis. */
  x: number;
  /** 0..1 of the lane's total task count. */
  y: number;
}

export interface LaneCurve {
  /** How much work existed by time t. Linear calls this scope, and plots it
   *  for the same reason: a project that "finished 13 of 13" having quietly
   *  grown from 4 is a different story from one that never moved the target. */
  scope: CurvePoint[];
  /** How much of it had settled by time t. */
  settled: CurvePoint[];
}

/**
 * The two step functions behind a lane's strip.
 *
 * WHY TWO SERIES AND NOT A SINGLE PERCENTAGE. A meter reading "31 of 40" is one
 * number at one instant and cannot distinguish a project that shipped steadily
 * from one that sat still for three months and then closed everything in a day
 * — and on this corpus that difference is real and visible. Linear's project
 * graph plots scope alongside completed work for the same reason
 * (linear.app/docs/project-graph); this is the same pair, reduced to what a
 * count-only model can support.
 *
 * WHAT IS DELIBERATELY NOT PLOTTED: Linear's third series is a velocity-based
 * FORECAST, wrapped in "a buffer of about ±40%". Nothing here forecasts. There
 * are no estimates, no cycles and no velocity to extrapolate from, so a
 * projected finish would be a line with no measurement under it — the exact
 * failure this module exists to refuse. Both series stop at the present.
 *
 * BOTH ARE STEP FUNCTIONS AND ARE RETURNED AS SUCH. Interpolating between two
 * completions would draw work progressing on days when nothing happened. The
 * caller draws the steps; joining these points with straight lines would
 * silently reintroduce exactly that lie.
 */
export function laneCurve(lane: Lane, axis: Axis): LaneCurve {
  const series = (times: number[]): CurvePoint[] => {
    if (lane.total === 0) return [];
    const sorted = [...times].sort((a, b) => a - b);
    const points: CurvePoint[] = [{ x: 0, y: 0 }];
    let count = 0;
    for (const time of sorted) {
      const x = positionOn(axis, time);
      count += 1;
      // Two events at the same position collapse to the taller step rather
      // than stacking a zero-width segment between them.
      const last = points[points.length - 1];
      if (last.x === x) last.y = count / lane.total;
      else points.push({ x, y: count / lane.total });
    }
    // Carried to the right edge: the last known level holds until now, and a
    // curve stopping mid-track would read as work that stopped being counted.
    points.push({ x: 1, y: count / lane.total });
    return points;
  };

  const recorded: number[] = [];
  const settled: number[] = [];
  for (const todo of lane.todos) {
    const line = lifelineOf(todo);
    if (!line) continue;
    recorded.push(line.recorded);
    if (line.settled !== null) settled.push(line.settled);
  }

  return { scope: series(recorded), settled: series(settled) };
}

/** The stepped SVG path for one series, in a `width` × `height` box with y
 *  inverted (SVG's origin is top-left, the curve grows upward). Empty for a
 *  series with nothing in it, which draws nothing rather than a flat line at
 *  zero that looks like a measurement. */
export function stepPath(points: readonly CurvePoint[], width: number, height: number): string {
  if (points.length === 0) return "";
  const px = (p: CurvePoint) => `${(p.x * width).toFixed(2)},${(height - p.y * height).toFixed(2)}`;
  let path = `M ${px(points[0])}`;
  for (let i = 1; i < points.length; i += 1) {
    // Horizontal to the new x at the OLD y, then vertical: the step itself.
    path += ` L ${(points[i].x * width).toFixed(2)},${(height - points[i - 1].y * height).toFixed(2)}`;
    path += ` L ${px(points[i])}`;
  }
  return path;
}
