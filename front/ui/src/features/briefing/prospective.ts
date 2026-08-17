import { parseTime } from "./derive";

/**
 * What you asked to be told about, and when it is owed.
 *
 * Pure and separate from the view for the same reason the rest of this
 * directory is: every token in the right-hand gutter is a claim about a real
 * reminder, and a claim that needs a query client around it to read is a claim
 * nobody checks.
 *
 * TWO TRIGGER KINDS, AND THEY ARE NOT THE SAME OBJECT. `ProspectiveTrigger`
 * (src/memory/types.rs:3596-3611) has three variants and `due_at()`
 * (types.rs:3632-3640) answers for only two of them: `AtTime` returns its
 * instant, `AfterDuration` returns `from + seconds`, and `OnContext` returns
 * `None`. A context reminder therefore has NO due instant, can never be late,
 * and must never be drawn as if it were — which is the single easiest way this
 * section could lie, because on the live `claude-code` profile every standing
 * reminder is context-triggered with `due_at: null`.
 *
 * LATENESS IS COMPUTED HERE, NOT READ OFF `overdue_seconds`. The server's field
 * (types.rs:3764-3772) is `Utc::now()` at the instant the response was built.
 * The briefing holds a screen clock that ticks every sixty seconds and re-reads
 * on a stale query, so a page left open would print a gutter that disagrees
 * with its own dateline. `due_at` and the screen's own `now` is the same
 * arithmetic Tasks does (`lateDays`), so the two surfaces cannot drift apart.
 */

/** `ReminderItem.trigger_type` — the three strings `list_reminders` maps the
 *  trigger enum onto (src/handlers/todos.rs:679-683). */
export type ReminderTriggerType = "time" | "duration" | "context";

/** `ProspectiveTaskStatus` (src/memory/types.rs:3665-3675), lowercased by
 *  `format!("{:?}", …).to_lowercase()` at todos.rs:684. */
export type ReminderStatus = "pending" | "triggered" | "dismissed" | "expired";

/**
 * `ReminderItem` — src/handlers/todos.rs:92-104 — reduced to what is read.
 *
 * WHAT THE SERVER SENDS AND THIS DOES NOT DECLARE, each for its own reason,
 * on the precedent `features/tasks/api.ts` set for `todo_counts`: declaring a
 * field is a claim that something renders it.
 *
 *   - `overdue_seconds` — a snapshot of the server's clock at response time.
 *     Lateness is computed from `due_at` against the screen's own clock (see
 *     the header above), so reading this would be a second answer to a question
 *     already answered, free to drift from the first.
 *   - `dismissed_at` — carries nothing `status === "dismissed"` does not, and
 *     a dismissed reminder is not drawn at all.
 *   - `priority` — a 1-5 integer with no editor anywhere in the product and no
 *     effect on when a reminder fires. A front page that ranked four rows by an
 *     unmaintained number would be inventing an order.
 *   - `tags` — empty on every reminder this instance holds. Two rows is not
 *     enough to call the field dead, so this claims only that there is nothing
 *     to draw, not that nothing could ever write one.
 *
 * AND NOTE WHAT THE SERVER ITSELF DROPS. The `OnContext` variant carries
 * `keywords` and a `threshold`, and neither survives into this response: the
 * handler collapses the whole trigger to the single word `"context"`
 * (todos.rs:681). So this surface can say a reminder is context-triggered and
 * can never say WHAT it listens for, and no phrasing here may imply otherwise.
 */
export interface ReminderItem {
  id: string;
  content: string;
  trigger_type: ReminderTriggerType;
  status: ReminderStatus;
  due_at: string | null;
  created_at: string;
  triggered_at: string | null;
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const WEEK = 7 * DAY;
const MONTH = 30 * DAY;

/**
 * A span of time, spelled, for a gutter that has to be read rather than
 * scanned.
 *
 * `longAgo` in this directory's own `derive.ts` stops at days, which is right
 * for a footer reporting the last write and wrong here: the standing reminders
 * on the live profile were asked for in February and March, and "178 days ago"
 * is a number a reader has to convert. Months are the unit at that distance.
 *
 * Never zero-valued, and never negative. A span under a minute is "moments",
 * because "0 minutes late" reads as a rounding error rather than as a deadline
 * that has just passed — and the same floor absorbs a negative span, which is
 * what a clock skew between this machine and the server produces. There is
 * deliberately no `Math.max(0, ms)` above it: the floor already covers every
 * negative input, and a clamp that no input can reach is a line that looks
 * load-bearing and is not.
 */
export function spanLabel(ms: number): string {
  if (ms < MINUTE) return "moments";
  if (ms < HOUR) {
    const n = Math.floor(ms / MINUTE);
    return n === 1 ? "a minute" : `${n} minutes`;
  }
  if (ms < DAY) {
    const n = Math.floor(ms / HOUR);
    return n === 1 ? "an hour" : `${n} hours`;
  }
  if (ms < WEEK) {
    const n = Math.floor(ms / DAY);
    return n === 1 ? "a day" : `${n} days`;
  }
  if (ms < MONTH) {
    const n = Math.floor(ms / WEEK);
    return n === 1 ? "a week" : `${n} weeks`;
  }
  const n = Math.floor(ms / MONTH);
  return n === 1 ? "a month" : `${n} months`;
}

/**
 * Still owed, or done with.
 *
 * `Pending` is waiting for its trigger and `Triggered` means the condition was
 * met and it was raised — but nothing has acknowledged it, so it is still
 * something you are owed and it belongs on the page. `Dismissed` and `Expired`
 * are closed, and drawing them would make a section about outstanding
 * intentions into a log.
 */
export function isStanding(item: ReminderItem): boolean {
  return item.status === "pending" || item.status === "triggered";
}

/**
 * How late a reminder is, in milliseconds, or `null` when it cannot be late.
 *
 * `null` for a context reminder — it has no due instant — and `null` for a
 * timed one whose instant has not passed. Those two are the same answer to the
 * question "how late is this" and a different answer to "when is this owed",
 * which is why the caller asks both.
 */
export function lateBy(item: ReminderItem, now: number): number | null {
  const due = parseTime(item.due_at);
  if (due === null) return null;
  const delta = now - due;
  return delta > 0 ? delta : null;
}

export interface StandingRow {
  id: string;
  text: string;
  /** The right-hand gutter, in the order it is printed. At most two tokens. */
  meta: string[];
  /**
   * A timed reminder whose instant has passed. THE ONLY THING ON THIS SECTION
   * THAT MAY TAKE `--destructive`: the token means late-or-wrong across the
   * product, and a context reminder that has sat quietly since February is
   * neither.
   */
  late: boolean;
}

/**
 * The gutter for one reminder — when it is owed, and when you asked for it.
 *
 * FIRST TOKEN: WHAT IS OWED. A timed reminder that has passed says how late; one
 * still ahead says how long is left; a context reminder says "when it comes up",
 * which is exactly what `OnContext` means and is all this response supports.
 *
 * SECOND TOKEN: WHEN YOU ASKED, or when it was raised if it has been. This is
 * the quiet finding on the live profile and it costs no prose to state — a
 * reminder reading "asked 4 months ago" with no "raised" beside it has been
 * standing since March and has never once surfaced, and that is the whole
 * reason this section exists. Both of `claude-code`'s reminders read that way.
 */
export function reminderMeta(item: ReminderItem, now: number): string[] {
  const meta: string[] = [];

  const late = lateBy(item, now);
  const due = parseTime(item.due_at);
  if (late !== null) {
    meta.push(`${spanLabel(late)} late`);
  } else if (due !== null) {
    meta.push(`due in ${spanLabel(due - now)}`);
  } else if (item.trigger_type === "context") {
    meta.push("when it comes up");
  }

  const raised = parseTime(item.triggered_at);
  const asked = parseTime(item.created_at);
  if (raised !== null) meta.push(`raised ${spanLabel(now - raised)} ago`);
  else if (asked !== null) meta.push(`asked ${spanLabel(now - asked)} ago`);

  return meta;
}

/**
 * Everything with a deadline, soonest first; then everything without one,
 * newest ask first.
 *
 * TWO POPULATIONS, NOT COMPARED. A lateness in milliseconds and a "when it
 * comes up" are not the same quantity, so they are never ranked against each
 * other: every reminder that carries a due instant sorts above every reminder
 * that cannot have one, and each band is ordered by the one measure it has.
 *
 * NO SEPARATE BAND FOR LATE WORK, AND THAT IS DELIBERATE. An earlier draft
 * ranked late reminders above merely-scheduled ones as its own first pass, and
 * a mutation test showed the pass could be deleted without changing a single
 * ordering — sorting every due instant ascending already puts everything in the
 * past above everything in the future, most overdue first, because that is what
 * "past" means. The band was a second expression of the same rule and the two
 * could have drifted apart. The alarm marker still keys on `lateBy`, which is
 * the judgement; this is only the order.
 *
 * Ties fall back to the id so the order is stable across renders rather than
 * left to the sort's own.
 */
function standingOrder(a: ReminderItem, b: ReminderItem): number {
  const dueA = parseTime(a.due_at);
  const dueB = parseTime(b.due_at);
  if (dueA !== null || dueB !== null) {
    if (dueA === null) return 1;
    if (dueB === null) return -1;
    if (dueA !== dueB) return dueA - dueB;
    return a.id.localeCompare(b.id);
  }

  const askedA = parseTime(a.created_at) ?? 0;
  const askedB = parseTime(b.created_at) ?? 0;
  if (askedA !== askedB) return askedB - askedA;
  return a.id.localeCompare(b.id);
}

/**
 * How many rows the front page will print before it stops.
 *
 * Four, because this is a front page and not the reminder screen — a section
 * that can grow without bound turns the briefing into the thing it says in its
 * own header it is not. The remainder is counted rather than dropped, so the
 * page never quietly shows a subset as though it were the whole.
 */
export const STANDING_LIMIT = 4;

export interface Standing {
  rows: StandingRow[];
  /** Standing reminders past the display cap. Zero on every profile today. */
  hidden: number;
}

/**
 * Everything still owed, ordered, capped, and ready to print — or `null`.
 *
 * `null`, NOT AN EMPTY BOARD. The briefing's governing rule is that an empty
 * page and a broken one must never look the same, and the resolution of that
 * rule for a section with nothing in it is to render no section: a heading over
 * no rows is furniture, and "0 reminders" is a claim about a subsystem most
 * profiles have never used. A profile with nothing standing simply does not
 * have this section, and a read that failed does not either — the caller keeps
 * those apart, because only one of them is a fact.
 */
export function standingReminders(
  items: readonly ReminderItem[],
  now: number,
  limit: number = STANDING_LIMIT,
): Standing | null {
  const standing = items.filter(isStanding).sort(standingOrder);
  if (standing.length === 0) return null;

  const rows = standing.slice(0, limit).map<StandingRow>((item) => ({
    id: item.id,
    // Collapsed to one line. A reminder is stored as free text and a newline in
    // it would break the row grammar this sits in without adding a word.
    text: item.content.replace(/\s+/g, " ").trim(),
    meta: reminderMeta(item, now),
    late: lateBy(item, now) !== null,
  }));

  return { rows, hidden: Math.max(0, standing.length - rows.length) };
}
