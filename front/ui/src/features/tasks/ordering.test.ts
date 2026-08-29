import { describe, expect, it } from "vitest";

import type { Todo, TodoPriority, TodoStatus } from "@/lib/api";
import { completedOrder, daysUntilDue, isNextUp, isOverdue, nextUp } from "./ordering";

/**
 * "Next up" claims these are the items demanding attention out of everything
 * open, and "Completed" claims an order over records where the ordering field
 * is missing on half the rows. Both are rules a reader trusts without being
 * able to check, which is exactly the kind that has to be checked here.
 *
 * NOW is fixed so the due-date rules are tested against known dates rather than
 * against whenever the suite runs.
 */
const NOW = Date.parse("2026-08-29T12:00:00Z");
const daysFromNow = (n: number) => new Date(NOW + n * 86_400_000).toISOString();

function todo(over: Partial<Todo> & { seq_num: number }): Todo {
  return {
    id: `id-${over.seq_num}`,
    project_prefix: "SHOD",
    user_id: "test",
    content: `task ${over.seq_num}`,
    status: "todo" as TodoStatus,
    priority: "medium" as TodoPriority,
    project_id: null,
    parent_id: null,
    contexts: [],
    tags: [],
    due_date: null,
    blocked_on: null,
    notes: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    completed_at: null,
    sort_order: 0,
    ...over,
  };
}

describe("daysUntilDue", () => {
  it("is negative for a past date and positive for a future one", () => {
    expect(daysUntilDue(todo({ seq_num: 1, due_date: daysFromNow(-5) }), NOW)).toBeLessThan(0);
    expect(daysUntilDue(todo({ seq_num: 2, due_date: daysFromNow(5) }), NOW)).toBeGreaterThan(0);
  });

  it("treats an unparseable date as absent rather than as the epoch", () => {
    // The dangerous alternative: NaN or 0 would make every malformed row read
    // as decades overdue and push it to the top of the board.
    const bad = todo({ seq_num: 1, due_date: "not a date" });
    expect(daysUntilDue(bad, NOW)).toBeNull();
    expect(isOverdue(bad, NOW)).toBe(false);
  });

  it("is null, not zero, when there is no due date at all", () => {
    expect(daysUntilDue(todo({ seq_num: 1 }), NOW)).toBeNull();
    expect(isOverdue(todo({ seq_num: 1 }), NOW)).toBe(false);
  });
});

describe("isNextUp", () => {
  it("takes overdue work", () => {
    expect(isNextUp(todo({ seq_num: 1, due_date: daysFromNow(-1) }), NOW)).toBe(true);
  });

  it("takes work due within three days but not work due later", () => {
    expect(isNextUp(todo({ seq_num: 1, due_date: daysFromNow(2) }), NOW)).toBe(true);
    expect(isNextUp(todo({ seq_num: 2, due_date: daysFromNow(30) }), NOW)).toBe(false);
  });

  it("takes urgent work with no date", () => {
    expect(isNextUp(todo({ seq_num: 1, priority: "urgent" }), NOW)).toBe(true);
  });

  it("excludes work already in progress, however urgent", () => {
    // The exclusion is the whole point: a focus list that repeats the
    // in-progress column is the board printed twice.
    expect(
      isNextUp(todo({ seq_num: 1, status: "in_progress", priority: "urgent", due_date: daysFromNow(-9) }), NOW),
    ).toBe(false);
  });

  it("includes BLOCKED work that is overdue", () => {
    // Deliberately not excluded. Blocked and overdue is the exact combination
    // that rots unattended, which is what this list is for.
    expect(
      isNextUp(todo({ seq_num: 1, status: "blocked", due_date: daysFromNow(-4) }), NOW),
    ).toBe(true);
  });

  it("excludes finished work", () => {
    for (const status of ["done", "cancelled"] as TodoStatus[]) {
      expect(isNextUp(todo({ seq_num: 1, status, priority: "urgent" }), NOW)).toBe(false);
    }
  });

  it("ignores ordinary undated work", () => {
    expect(isNextUp(todo({ seq_num: 1, priority: "high" }), NOW)).toBe(false);
  });
});

describe("nextUp", () => {
  it("orders by how overdue, most pressing first", () => {
    const order = nextUp(
      [
        todo({ seq_num: 1, due_date: daysFromNow(2) }),
        todo({ seq_num: 2, due_date: daysFromNow(-10) }),
        todo({ seq_num: 3, due_date: daysFromNow(-1) }),
      ],
      NOW,
    ).map((t) => t.seq_num);
    expect(order).toEqual([2, 3, 1]);
  });

  it("puts dated work ahead of undated urgent work", () => {
    const order = nextUp(
      [todo({ seq_num: 1, priority: "urgent" }), todo({ seq_num: 2, due_date: daysFromNow(3) })],
      NOW,
    ).map((t) => t.seq_num);
    expect(order).toEqual([2, 1]);
  });

  it("breaks a full tie on seq_num rather than on arrival order", () => {
    const a = todo({ seq_num: 9, priority: "urgent" });
    const b = todo({ seq_num: 4, priority: "urgent" });
    expect(nextUp([a, b], NOW).map((t) => t.seq_num)).toEqual([4, 9]);
    expect(nextUp([b, a], NOW).map((t) => t.seq_num)).toEqual([4, 9]);
  });
});

describe("completedOrder", () => {
  it("puts the most recently finished first", () => {
    const order = completedOrder([
      todo({ seq_num: 1, status: "done", completed_at: "2026-02-01T00:00:00Z" }),
      todo({ seq_num: 2, status: "done", completed_at: "2026-08-01T00:00:00Z" }),
    ]).map((t) => t.seq_num);
    expect(order).toEqual([2, 1]);
  });

  it("falls back to updated_at when completed_at is missing", () => {
    // Roughly half of these records carry no completed_at. Sorting on it alone
    // leaves those rows wherever the engine puts them.
    const order = completedOrder([
      todo({ seq_num: 1, status: "done", completed_at: null, updated_at: "2026-01-01T00:00:00Z" }),
      todo({ seq_num: 2, status: "done", completed_at: null, updated_at: "2026-07-01T00:00:00Z" }),
    ]).map((t) => t.seq_num);
    expect(order).toEqual([2, 1]);
  });

  it("interleaves dated and undated rows by the timestamp each actually has", () => {
    const order = completedOrder([
      todo({ seq_num: 1, status: "done", completed_at: "2026-05-01T00:00:00Z" }),
      todo({ seq_num: 2, status: "done", completed_at: null, updated_at: "2026-09-01T00:00:00Z" }),
      todo({ seq_num: 3, status: "done", completed_at: "2026-03-01T00:00:00Z" }),
    ]).map((t) => t.seq_num);
    expect(order).toEqual([2, 1, 3]);
  });

  it("is deterministic when two rows carry the same timestamp", () => {
    const stamp = "2026-04-01T00:00:00Z";
    const a = todo({ seq_num: 3, status: "done", completed_at: stamp });
    const b = todo({ seq_num: 8, status: "done", completed_at: stamp });
    expect(completedOrder([a, b]).map((t) => t.seq_num)).toEqual([8, 3]);
    expect(completedOrder([b, a]).map((t) => t.seq_num)).toEqual([8, 3]);
  });

  it("does not mutate its argument", () => {
    const input = [
      todo({ seq_num: 1, status: "done", completed_at: "2026-01-01T00:00:00Z" }),
      todo({ seq_num: 2, status: "done", completed_at: "2026-09-01T00:00:00Z" }),
    ];
    completedOrder(input);
    expect(input.map((t) => t.seq_num)).toEqual([1, 2]);
  });
});
