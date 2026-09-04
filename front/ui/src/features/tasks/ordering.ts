import type { Todo } from "@/lib/api";

/**
 * The selection and ordering rules behind the tasks board.
 *
 * Pure and exported because each one IS a claim the screen makes. "Next up"
 * asserts that these are the items demanding attention now, out of everything
 * open; "Completed" asserts an order over records where the field that should
 * carry the order is missing on half of them. Both are the kind of rule that
 * looks obviously right inline and is quietly wrong for a year.
 *
 * `now` is a parameter rather than a call to `Date.now()` inside, so the due
 * -date rules can be tested against fixed dates instead of against whenever the
 * suite happens to run.
 */

/** Whole days until `due_date`. Negative is overdue. Null when there is no
 *  usable date — an unparseable one is treated as absent rather than as the
 *  epoch, which would make every such todo maximally overdue. */
export function daysUntilDue(t: Todo, now: number = Date.now()): number | null {
  if (!t.due_date) return null;
  const due = new Date(t.due_date).getTime();
  if (Number.isNaN(due)) return null;
  return Math.ceil((due - now) / 86_400_000);
}

export function isOverdue(t: Todo, now: number = Date.now()): boolean {
  const days = daysUntilDue(t, now);
  return days !== null && days < 0;
}

/** Due within this many days still counts as needing attention now. */
const SOON_DAYS = 3;

/**
 * Whether a todo belongs in "Next up".
 *
 * The rule, stated so it can be argued with: something is next up when it is
 * overdue, due within three days, or marked urgent -- AND is not already in
 * progress. The exclusion is the part worth defending: work someone has
 * already started does not need to be told to them again, and a focus list
 * that repeats the in-progress column is just the board twice.
 *
 * Blocked work is NOT excluded. A blocked item that is overdue is precisely
 * the thing that rots unattended, which is the case this list exists for.
 */
export function isNextUp(t: Todo, now: number = Date.now()): boolean {
  if (t.status === "in_progress" || t.status === "done" || t.status === "cancelled") return false;
  const days = daysUntilDue(t, now);
  const dated = days !== null && days <= SOON_DAYS;
  return dated || t.priority === "urgent";
}

/** Most pressing first: by days until due, undated urgent work after dated
 *  work, then a total order on seq_num so two runs cannot disagree. */
export function nextUp(todos: Todo[], now: number = Date.now()): Todo[] {
  return todos.filter((t) => isNextUp(t, now)).sort((a, b) => {
    const da = daysUntilDue(a, now);
    const db = daysUntilDue(b, now);
    if (da !== null && db !== null && da !== db) return da - db;
    if (da !== null && db === null) return -1;
    if (da === null && db !== null) return 1;
    return a.seq_num - b.seq_num;
  });
}

/**
 * Completed work, most recently finished first.
 *
 * `completed_at` is missing on roughly half of these records -- 43 of 82 on the
 * profile this was built against -- so it cannot be sorted on alone. Sorting a
 * field that is undefined for half the rows leaves those rows wherever the
 * engine's sort happens to put them, which is the same class of nondeterminism
 * swept out of the Rust ranking paths this week: an order that is stable only
 * by luck. `updated_at` stands in, and seq_num breaks the remaining ties.
 */
export function completedOrder(todos: Todo[]): Todo[] {
  const finishedAt = (t: Todo) => {
    const stamp = Date.parse(t.completed_at ?? t.updated_at);
    return Number.isNaN(stamp) ? 0 : stamp;
  };
  return [...todos].sort((a, b) => finishedAt(b) - finishedAt(a) || b.seq_num - a.seq_num);
}
