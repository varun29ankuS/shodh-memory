import { describe, expect, it } from "vitest";
import type { Todo } from "@/lib/api";
import type { LinkedMemory, TodoComment, TriageTodo } from "./api";
import {
  classifyLink,
  dueMeta,
  originOf,
  priorityLabel,
  provenanceOf,
  settledReason,
  shortId,
  summarise,
} from "./derive";

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

/* ------------------------------------------------------------------ *
 * Triage: provenance, origin, and the reason a task was settled.
 *
 * These three decide what a row CLAIMS about where work came from, and every
 * one of them fails silently in the direction of claiming too much. A
 * misclassified echo renders as a source sentence — a quoted blockquote under
 * "Where this came from" that is really the task restating itself. An origin
 * guessed from a missing field renders as a confident provenance label. A
 * missed resolution comment renders a dismissal as unexplained.
 * ------------------------------------------------------------------ */

const memory = (tags: string[], content = "some memory"): LinkedMemory => ({
  id: "m1",
  experience: { content, experience_type: "Observation", tags },
  created_at: "2026-08-01T00:00:00Z",
});

const triage = (extra: Partial<TriageTodo> = {}): TriageTodo => ({
  ...todo(),
  ...extra,
});

describe("classifyLink", () => {
  it("calls a memory with no lifecycle tag a source", () => {
    expect(classifyLink(memory(["logistics", "contract"]))).toBe("source");
  });

  it.each(["todo-created", "todo-updated", "todo-completed"])(
    "calls a %s memory an echo — the server wrote it about the task",
    (tag) => {
      expect(classifyLink(memory(["todo:SHO-1", tag]))).toBe("echo");
    },
  );

  it("treats todo-comment: as a prefix, since the comment type is appended", () => {
    expect(classifyLink(memory(["todo:SHO-1", "todo-comment:resolution"]))).toBe("echo");
    expect(classifyLink(memory(["todo:SHO-1", "todo-comment:progress"]))).toBe("echo");
  });

  it("does not judge on experience_type: a Task-typed memory with no lifecycle tag is a source", () => {
    // The create and complete echoes are Task-typed but the update echo is
    // Context-typed, so type is not the signature and using it would both
    // miss echoes and swallow genuine Task-typed sources.
    const m: LinkedMemory = {
      id: "m2",
      experience: { content: "Sprint planning", experience_type: "Task", tags: ["planning"] },
      created_at: "2026-08-01T00:00:00Z",
    };
    expect(classifyLink(m)).toBe("source");
  });

  it("does not mistake an unrelated tag that merely starts with todo", () => {
    expect(classifyLink(memory(["todo:SHO-1"]))).toBe("source");
    expect(classifyLink(memory(["todoist-sync"]))).toBe("source");
  });

  it("survives a memory whose tags are absent rather than empty", () => {
    const m = { id: "m3", experience: { content: "x", experience_type: "Observation" }, created_at: "2026-08-01T00:00:00Z" } as unknown as LinkedMemory;
    expect(classifyLink(m)).toBe("source");
  });
});

describe("provenanceOf", () => {
  it("separates a real source from the creation echo that always accompanies it", () => {
    const result = provenanceOf([
      memory(["contract"], "Meridian need the signed addendum before the 30th."),
      memory(["todo:SHO-1", "todo-created"], "[SHO-1] Todo created: Send signed addendum"),
    ]);
    expect(result.sources.map((m) => m.experience.content)).toEqual([
      "Meridian need the signed addendum before the 30th.",
    ]);
    expect(result.echoes).toBe(1);
  });

  it("reports zero sources when every link is the task's own history", () => {
    const result = provenanceOf([
      memory(["todo:SHO-2", "todo-created"]),
      memory(["todo:SHO-2", "todo-updated"]),
    ]);
    expect(result.sources).toEqual([]);
    expect(result.echoes).toBe(2);
  });

  it("counts nothing for an empty link list", () => {
    expect(provenanceOf([])).toEqual({ sources: [], echoes: 0 });
  });
});

describe("originOf", () => {
  it("reports the session hook from its external_id stamp", () => {
    expect(originOf(triage({ external_id: "claude-task:abc123" })).kind).toBe("session-hook");
  });

  it("reports the session hook from its tag when external_id is absent", () => {
    expect(originOf(triage({ tags: ["source:hook", "claude-task"] })).kind).toBe("session-hook");
  });

  it("names the system of record for an external sync key", () => {
    const origin = originOf(triage({ external_id: "linear:SHO-39" }));
    expect(origin.kind).toBe("external");
    expect(origin).toHaveProperty("label", "synced from linear");
  });

  it("says UNRECORDED rather than guessing when nothing was stamped", () => {
    // The backend has no origin field. An unstamped todo is not "typed by a
    // human" — it is unknown, and the screen must not render a label for it.
    expect(originOf(triage()).kind).toBe("unrecorded");
  });
});

describe("settledReason", () => {
  const comment = (
    comment_type: TodoComment["comment_type"],
    content: string,
  ): TodoComment => ({
    id: `c-${content}`,
    todo_id: "t",
    author: "someone",
    content,
    comment_type,
    created_at: "2026-08-01T00:00:00Z",
    updated_at: null,
  });

  it("reads the resolution comment past the system activity ones around it", () => {
    // A real dismissal ships exactly this shape: system "Created", the
    // reason, then system "Updated: status -> Cancelled".
    const t = triage({
      comments: [
        comment("activity", "Created"),
        comment("resolution", "Not ours — Meridian accept an e-signature."),
        comment("activity", "Updated: status → Cancelled"),
      ],
    });
    expect(settledReason(t)).toBe("Not ours — Meridian accept an e-signature.");
  });

  it("takes the LAST resolution, which is why it is settled now", () => {
    const t = triage({
      comments: [comment("resolution", "first call"), comment("resolution", "after reopening")],
    });
    expect(settledReason(t)).toBe("after reopening");
  });

  it("returns null when only system activity was recorded", () => {
    const t = triage({ comments: [comment("activity", "Created")] });
    expect(settledReason(t)).toBeNull();
  });

  it("returns null for a whitespace-only reason rather than rendering a blank", () => {
    const t = triage({ comments: [comment("resolution", "   ")] });
    expect(settledReason(t)).toBeNull();
  });

  it("returns null when the server sent no comments array at all", () => {
    expect(settledReason(triage())).toBeNull();
  });
});
