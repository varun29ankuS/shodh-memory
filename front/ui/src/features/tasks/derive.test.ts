import { describe, expect, it } from "vitest";
import type { Todo } from "@/lib/api";
import { dueMeta, priorityLabel, shortId, summarise } from "./derive";

/**
 * The Tasks screen states figures about real work. The ones worth pinning are
 * the ones where being wrong is INVISIBLE — an overdue count that silently
 * includes today, a "showing 200 of 431" that appears when nothing was
 * truncated, a short id that drops its project prefix. Each of those renders
 * as a plausible number nobody re-checks against the rows.
 */

/** Built in LOCAL time, and every due date below with it. The rule under test
 *  counts calendar days, so a UTC literal would make these assertions pass or
 *  fail on the runner's timezone rather than on the arithmetic. */
const local = (y: number, m: number, d: number, h = 0, min = 0) =>
  new Date(y, m - 1, d, h, min, 0, 0);
const iso = (date: Date) => date.toISOString();

const NOW = local(2026, 8, 16, 12).getTime();

const todo = (extra: Partial<Todo> = {}): Todo => ({
  id: "0123abcd-0000-0000-0000-000000000000",
  seq_num: 1,
  project_prefix: "SHOD",
  user_id: "claude",
  content: "x",
  status: "todo",
  priority: "medium",
  project_id: null,
  parent_id: null,
  contexts: [],
  tags: [],
  due_date: null,
  blocked_on: null,
  notes: null,
  created_at: "2026-08-01T00:00:00Z",
  updated_at: "2026-08-01T00:00:00Z",
  completed_at: null,
  sort_order: 0,
  ...extra,
});

describe("shortId", () => {
  it("uses the project prefix and sequence number when the server sent one", () => {
    expect(shortId(todo({ seq_num: 7, project_prefix: "SHOD" }))).toBe("SHOD-7");
  });

  it("falls back to SHO for a todo with a sequence but no project", () => {
    expect(shortId(todo({ seq_num: 7, project_prefix: null }))).toBe("SHO-7");
  });

  it("falls back to the id's first four hex characters for a legacy todo", () => {
    // seq_num 0 is the server's "never assigned", not a valid first todo.
    expect(shortId(todo({ seq_num: 0, project_prefix: "SHOD" }))).toBe("SHO-0123");
  });
});

describe("dueMeta", () => {
  it("has no opinion about a todo with no due date", () => {
    expect(dueMeta(todo(), NOW)).toBeNull();
  });

  it("refuses a malformed date rather than rendering Invalid Date", () => {
    expect(dueMeta(todo({ due_date: "not a date" }), NOW)).toBeNull();
  });

  it("calls something due later the same day today, not tomorrow", () => {
    // The millisecond arithmetic this replaces ceil'd 11 hours to "Due in 1d".
    const today = dueMeta(todo({ due_date: iso(local(2026, 8, 16, 23)) }), NOW);
    expect(today).toEqual({ label: "Due today", tone: "warn", overdue: false });
  });

  it("treats an hour past the due date as overdue, as the server does", () => {
    // `Todo::is_overdue()` is a strict `now > due`. The previous rule needed a
    // full 24 hours before it said so, and painted this one warn — an item
    // already late, rendered as one still in hand.
    const past = dueMeta(todo({ due_date: iso(local(2026, 8, 16, 11)) }), NOW);
    expect(past).toEqual({ label: "Overdue", tone: "destructive", overdue: true });
  });

  it("counts lateness in whole calendar days once there is one to count", () => {
    expect(dueMeta(todo({ due_date: iso(local(2026, 8, 14, 23, 59)) }), NOW)?.label).toBe(
      "Overdue 2d",
    );
  });

  it("counts the near horizon in days and hands the rest to a plain date", () => {
    expect(dueMeta(todo({ due_date: iso(local(2026, 8, 19, 1)) }), NOW)?.label).toBe("Due in 3d");
    expect(dueMeta(todo({ due_date: iso(local(2026, 8, 20, 23)) }), NOW)?.tone).toBe("muted");
  });
});

describe("priorityLabel", () => {
  it("renders nothing for the server's unset priority", () => {
    // "none" on a row reads as a deliberate downgrade rather than an absence.
    expect(priorityLabel("none")).toBeNull();
  });

  it("shortens medium so it fits a row without competing with the title", () => {
    expect(priorityLabel("medium")).toBe("med");
    expect(priorityLabel("urgent")).toBe("urgent");
  });
});

describe("summarise", () => {
  it("reports truncation only when the server's total exceeds the rows sent", () => {
    const rows = [todo({ id: "a" }), todo({ id: "b" })];
    expect(summarise(rows, 2, NOW).truncated).toBe(false);
    const cut = summarise(rows, 431, NOW);
    expect(cut.truncated).toBe(true);
    expect(cut.shown).toBe(2);
    expect(cut.total).toBe(431);
  });

  it("never claims a total smaller than the rows it was handed", () => {
    // A total below the row count would render "showing 2 of 1".
    const s = summarise([todo({ id: "a" }), todo({ id: "b" })], 1, NOW);
    expect(s.total).toBe(2);
    expect(s.truncated).toBe(false);
  });

  it("counts overdue by the same rule the rows render, not by due_date presence", () => {
    const rows = [
      todo({ id: "a", due_date: iso(local(2026, 8, 1)) }), // overdue
      todo({ id: "b", due_date: iso(local(2026, 8, 16, 23)) }), // due today
      todo({ id: "c", due_date: iso(local(2026, 9, 1)) }), // future
      todo({ id: "d" }), // none
    ];
    expect(summarise(rows, 4, NOW).overdue).toBe(1);
  });

  it("counts urgent and high separately so neither hides inside the other", () => {
    const rows = [
      todo({ id: "a", priority: "urgent" }),
      todo({ id: "b", priority: "high" }),
      todo({ id: "c", priority: "high" }),
      todo({ id: "d", priority: "low" }),
    ];
    const s = summarise(rows, 4, NOW);
    expect(s.urgent).toBe(1);
    expect(s.high).toBe(2);
  });

  it("counts distinct project prefixes, ignoring todos that carry none", () => {
    const rows = [
      todo({ id: "a", project_prefix: "SHOD" }),
      todo({ id: "b", project_prefix: "SHOD" }),
      todo({ id: "c", project_prefix: null }),
    ];
    expect(summarise(rows, 3, NOW).projects).toBe(1);
  });
});
