import type { Todo, TodoPriority } from "@/lib/api";
import type { LinkedMemory, TriageTodo } from "./api";

/**
 * The arithmetic behind the Tasks screen, kept out of the component so it can
 * be pinned by tests.
 *
 * Everything here is a CLAIM the screen makes about a real list — a count, a
 * "12 overdue", a "showing 200 of 431" — and every one of them fails
 * invisibly. A miscounted overdue renders as a perfectly plausible number and
 * nobody ever checks it against the rows.
 */

/** The row limit this screen asks the server for. Stated here because the
 *  reduction notice has to quote the same number the request used, and two
 *  copies of it is how a screen ends up claiming "first 200 of 431" while
 *  showing 100 rows. */
export const TASK_LIMIT = 200;

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
