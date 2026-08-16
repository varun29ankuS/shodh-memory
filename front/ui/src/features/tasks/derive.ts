import type { Todo, TodoPriority } from "@/lib/api";

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
    // A server that reported a smaller total than it sent rows for would make
    // `truncated` nonsense; clamp rather than render "showing 50 of 12".
    total: Math.max(total, todos.length),
    truncated: total > todos.length,
    urgent,
    high,
    overdue,
    projects: prefixes.size,
  };
}
