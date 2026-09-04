import { describe, expect, it } from "vitest";

import type { Todo, TodoPriority, TodoStatus } from "@/lib/api";
import { briefingOrder, isBlocked, isOpen } from "./WorkPanel";

/**
 * The briefing shows five of what may be two hundred open items, so the order
 * is the entire claim. If it is wrong the panel is not merely untidy — it is
 * confidently showing the wrong five and hiding the item the reader needed.
 */

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
    created_at: "2026-08-01T00:00:00Z",
    updated_at: "2026-08-01T00:00:00Z",
    completed_at: null,
    sort_order: 0,
    ...over,
  };
}

const seqs = (todos: Todo[]) => briefingOrder(todos).map((t) => t.seq_num);

describe("isOpen", () => {
  it("counts everything not yet finished, including blocked", () => {
    for (const status of ["backlog", "todo", "in_progress", "blocked"] as TodoStatus[]) {
      expect(isOpen(todo({ seq_num: 1, status }))).toBe(true);
    }
  });

  it("excludes done and cancelled", () => {
    // A briefing that counts finished work as open reports a backlog that does
    // not exist, and the number is the first thing on the panel.
    expect(isOpen(todo({ seq_num: 1, status: "done" }))).toBe(false);
    expect(isOpen(todo({ seq_num: 2, status: "cancelled" }))).toBe(false);
  });
});

describe("isBlocked", () => {
  it("takes the status", () => {
    expect(isBlocked(todo({ seq_num: 1, status: "blocked" }))).toBe(true);
  });

  it("also takes a dependency the store is tracking on an otherwise ordinary todo", () => {
    // The two facts are set by different things — a person sets the status, the
    // store sets blocked_on — so reading only one silently under-reports.
    expect(isBlocked(todo({ seq_num: 1, status: "todo", blocked_on: "SHOD-9" }))).toBe(true);
  });

  it("is false for work merely waiting in the backlog", () => {
    expect(isBlocked(todo({ seq_num: 1, status: "backlog" }))).toBe(false);
  });
});

describe("briefingOrder", () => {
  it("puts blocked work first, ahead of higher-priority unblocked work", () => {
    // The load-bearing case. An urgent in-progress task outranks a low-priority
    // blocked one on every other axis, and the briefing still leads with the
    // blocked one, because that is the item that goes stale unattended.
    const order = seqs([
      todo({ seq_num: 1, status: "in_progress", priority: "urgent" }),
      todo({ seq_num: 2, status: "todo", priority: "low", blocked_on: "SHOD-1" }),
    ]);
    expect(order).toEqual([2, 1]);
  });

  it("puts work already underway ahead of work not started", () => {
    expect(
      seqs([
        todo({ seq_num: 1, status: "backlog" }),
        todo({ seq_num: 2, status: "todo" }),
        todo({ seq_num: 3, status: "in_progress" }),
      ]),
    ).toEqual([3, 2, 1]);
  });

  it("breaks a status tie on priority", () => {
    expect(
      seqs([
        todo({ seq_num: 1, status: "todo", priority: "none" }),
        todo({ seq_num: 2, status: "todo", priority: "urgent" }),
        todo({ seq_num: 3, status: "todo", priority: "medium" }),
      ]),
    ).toEqual([2, 3, 1]);
  });

  it("breaks a full tie on seq_num rather than on arrival order", () => {
    // Same reason the recall sorts carry a total order: two runs over the same
    // data must not disagree, and the server's ordering is not a guarantee.
    const a = todo({ seq_num: 7 });
    const b = todo({ seq_num: 3 });
    expect(seqs([a, b])).toEqual([3, 7]);
    expect(seqs([b, a])).toEqual([3, 7]);
  });

  it("does not mutate its argument", () => {
    // The caller passes a filtered view of a TanStack cache entry; sorting it
    // in place would reorder the array /tasks renders from.
    const input = [todo({ seq_num: 2 }), todo({ seq_num: 1 })];
    briefingOrder(input);
    expect(input.map((t) => t.seq_num)).toEqual([2, 1]);
  });

  it("orders an unknown status and priority last rather than dropping it", () => {
    // The backend owns these enums and can add a variant. An unrecognised one
    // must still appear — a briefing that silently omits work is worse than
    // one that ranks it badly.
    const order = seqs([
      todo({ seq_num: 1, status: "future_state" as TodoStatus }),
      todo({ seq_num: 2, status: "todo" }),
    ]);
    expect(order).toEqual([2, 1]);
    expect(order).toHaveLength(2);
  });
});
