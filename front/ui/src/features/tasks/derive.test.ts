import { describe, expect, it } from "vitest";
import type { Todo } from "@/lib/api";
import type { LinkedMemory, TodoComment, TriageProject, TriageTodo } from "./api";
import {
  authorKind,
  axisOf,
  blockerIsSatisfied,
  blockersOf,
  boardOf,
  classifyLink,
  dueMeta,
  elapsedLabel,
  lanesOf,
  lifelineOf,
  originOf,
  laneCurve,
  positionOn,
  priorityLabel,
  provenanceOf,
  settledReason,
  shortId,
  statusChanges,
  stepPath,
  subtaskProgress,
  summarise,
  truncation,
  type Board,
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

describe("truncation", () => {
  it("reports a slice when the server's total exceeds the rows it sent", () => {
    expect(truncation(200, 431)).toEqual({ total: 431, truncated: true });
  });

  it("reports no slice when everything came back", () => {
    expect(truncation(12, 12)).toEqual({ total: 12, truncated: false });
  });

  it("clamps a total smaller than the rows sent rather than saying 'showing 50 of 12'", () => {
    // A server disagreeing with itself must not produce a nonsense notice.
    expect(truncation(50, 12)).toEqual({ total: 50, truncated: false });
  });

  it("says nothing about an empty list", () => {
    expect(truncation(0, 0)).toEqual({ total: 0, truncated: false });
  });
});

/* ==================================================================== *
 * MOVEMENT, LANES AND BLOCKERS
 *
 * These pin the claims that are new to this screen and that fail silently: a
 * completion ratio computed over the wrong denominator, a "started" read off
 * the wrong transition, a lane that merged two projects sharing a prefix, and a
 * strip drawn over a corpus that never moved. Each renders as a plausible
 * picture nobody re-derives from the rows.
 *
 * The fixtures are shaped from the live `claude-code` profile — both settlement
 * paths, the SHOD prefix collision — because those are the cases that actually
 * occur rather than the ones that are easy to construct.
 * ==================================================================== */

/** The server's own activity text, verbatim from src/handlers/todos.rs. Written
 *  out rather than generated so a change to the real format shows up here as a
 *  literal that no longer matches. */
const sysActivity = (content: string, created_at: string): TodoComment => ({
  id: `sys-${content}-${created_at}`,
  todo_id: "t",
  author: "system",
  content,
  comment_type: "activity",
  created_at,
  updated_at: null,
});

const at = (day: number, hour = 0) => new Date(Date.UTC(2026, 1, day, hour)).toISOString();

describe("statusChanges", () => {
  it("reads the server's Debug-cased status names into the wire values", () => {
    // `format!("status → {:?}")` prints the Rust variant name; the same enum
    // serialises snake_case. Both spellings are real and they differ.
    const t = triage({
      comments: [
        sysActivity("Updated: status → InProgress", at(2)),
        sysActivity("Updated: status → Cancelled", at(3)),
      ],
    });
    expect(statusChanges(t)).toEqual([
      { at: Date.parse(at(2)), to: "in_progress" },
      { at: Date.parse(at(3)), to: "cancelled" },
    ]);
  });

  it("reads the complete endpoint's own wording as a move to done", () => {
    // The complete path never writes "status → Done"; it writes this. 39 of the
    // 82 settled todos on the live profile carry only this line.
    const t = triage({ comments: [sysActivity("Marked complete after 22.4 days", at(9))] });
    expect(statusChanges(t)).toEqual([{ at: Date.parse(at(9)), to: "done" }]);
  });

  it("takes the status clause when other fields changed in the same update", () => {
    const t = triage({ comments: [sysActivity("Updated: status → Done, content updated", at(4))] });
    expect(statusChanges(t)).toEqual([{ at: Date.parse(at(4)), to: "done" }]);
  });

  it("IGNORES A COMMENT THAT ONLY LOOKS LIKE ONE — the author is the gate", () => {
    // `author` and `comment_type` are both caller-supplied, so anyone can post
    // this text. Only `TodoComment::system_activity` sets author "system", and
    // it is the sole writer of these strings server-side. Without the gate this
    // reads back as a transition that never happened.
    const t = triage({
      comments: [
        { ...sysActivity("Updated: status → Done", at(5)), author: "claude-code" },
        { ...sysActivity("Updated: status → Done", at(6)), comment_type: "comment" },
      ],
    });
    expect(statusChanges(t)).toEqual([]);
  });

  it("orders by timestamp rather than trusting the array's order", () => {
    const t = triage({
      comments: [
        sysActivity("Updated: status → Done", at(8)),
        sysActivity("Updated: status → InProgress", at(2)),
      ],
    });
    expect(statusChanges(t).map((c) => c.to)).toEqual(["in_progress", "done"]);
  });

  it("drops a status name this build does not know rather than inventing one", () => {
    const t = triage({ comments: [sysActivity("Updated: status → Superseded", at(2))] });
    expect(statusChanges(t)).toEqual([]);
  });

  it("does not read a creation entry as a status change", () => {
    // "Created in project 'X'" records no status, and CreateTodoRequest accepts
    // an initial one — so the starting status is genuinely unknown.
    const t = triage({ comments: [sysActivity("Created in project 'Codebase Audit'", at(1))] });
    expect(statusChanges(t)).toEqual([]);
  });
});

describe("lifelineOf", () => {
  it("takes the FIRST start, so a task picked up twice reports when work began", () => {
    const t = triage({
      comments: [
        sysActivity("Updated: status → InProgress", at(2)),
        sysActivity("Updated: status → Todo", at(3)),
        sysActivity("Updated: status → InProgress", at(7)),
      ],
    });
    expect(lifelineOf(t)?.started).toBe(Date.parse(at(2)));
  });

  it("prefers the server's completed_at when it set one", () => {
    const t = triage({
      status: "done",
      completed_at: at(6),
      comments: [sysActivity("Marked complete after 0.0 days", at(6, 1))],
    });
    const line = lifelineOf(t);
    expect(line?.settled).toBe(Date.parse(at(6)));
    expect(line?.settledFromLog).toBe(false);
  });

  it("RECOVERS A SETTLED TIME FROM THE LOG WHEN NO completed_at WAS WRITTEN", () => {
    // `Todo::complete()` is the only writer of completed_at, so setting status
    // "done" through the update handler leaves it null — and cancelling always
    // does. 43 of 82 settled todos on the live profile are in this state and
    // would otherwise have no completion date at all.
    const t = triage({
      status: "cancelled",
      completed_at: null,
      comments: [sysActivity("Updated: status → Cancelled", at(5))],
    });
    const line = lifelineOf(t);
    expect(line?.settled).toBe(Date.parse(at(5)));
    expect(line?.settledFromLog).toBe(true);
  });

  it("leaves an open task unsettled even if it once passed through done", () => {
    const t = triage({
      status: "todo",
      completed_at: null,
      comments: [
        sysActivity("Updated: status → Done", at(4)),
        sysActivity("Updated: status → Todo", at(5)),
      ],
    });
    expect(lifelineOf(t)?.settled).toBeNull();
  });

  it("has no opinion when created_at cannot be read", () => {
    expect(lifelineOf(triage({ created_at: "not a date" }))).toBeNull();
  });
});

describe("blockersOf", () => {
  const index = (rows: TriageTodo[]) => new Map(rows.map((r) => [r.id, r]));

  it("separates waiting on a task from waiting on a person", () => {
    // Different objects in this model, and they must not share a treatment: one
    // has a status of its own and can be chased, the other is prose.
    const blocker = triage({ id: "b1", content: "Sign the MOU", status: "in_progress" });
    const t = triage({ id: "t1", blocked_by: ["b1"], blocked_on: "Meridian's counsel" });
    expect(blockersOf(t, index([blocker]))).toEqual([
      { kind: "task", id: "b1", todo: blocker },
      { kind: "waiting", text: "Meridian's counsel" },
    ]);
  });

  it("KEEPS A BLOCKER IT CANNOT RESOLVE rather than dropping it silently", () => {
    // A dependency that exists and is off-screen is not the same as none, and
    // dropping it would understate what the task is waiting on.
    const t = triage({ id: "t1", blocked_by: ["gone"] });
    expect(blockersOf(t, index([]))).toEqual([{ kind: "task-missing", id: "gone" }]);
  });

  it("ignores a blocked_on the update handler cleared to an empty string", () => {
    // Clearing writes "" rather than null — live activity carries an entry
    // reading "blocked on: " with nothing after it.
    expect(blockersOf(triage({ blocked_on: "   " }), index([]))).toEqual([]);
  });

  it("treats a settled blocker as satisfied, as unblocked_by_completion does", () => {
    const done = triage({ id: "b1", status: "done" });
    const open = triage({ id: "b2", status: "todo" });
    const map = index([done, open]);
    expect(blockerIsSatisfied(blockersOf(triage({ blocked_by: ["b1"] }), map)[0])).toBe(true);
    expect(blockerIsSatisfied(blockersOf(triage({ blocked_by: ["b2"] }), map)[0])).toBe(false);
    expect(blockerIsSatisfied({ kind: "waiting", text: "counsel" })).toBe(false);
  });
});

describe("lanesOf", () => {
  const project = (extra: Partial<TriageProject> & { id: string }): TriageProject => ({
    user_id: "claude-code",
    name: "Untitled",
    prefix: null,
    status: "active",
    ...extra,
  });

  it("KEEPS TWO PROJECTS SHARING A PREFIX APART — this collision is live", () => {
    // claude-code carries "shodh-memory" (archived, 39 todos) and "Shodh-redb"
    // (active, 1), both prefix SHOD, because the prefix is derived from the
    // name and is not unique. Grouping by prefix merges a finished project into
    // a running one and reports one wrong ratio over both.
    const lanes = lanesOf(
      [
        triage({ id: "a", project_id: "p-old", project_prefix: "SHOD", status: "done" }),
        triage({ id: "b", project_id: "p-new", project_prefix: "SHOD", status: "todo" }),
      ],
      [
        project({ id: "p-old", name: "shodh-memory", prefix: "SHOD", status: "archived" }),
        project({ id: "p-new", name: "Shodh-redb", prefix: "SHOD", status: "active" }),
      ],
    );
    expect(lanes.map((l) => l.name)).toEqual(["Shodh-redb", "shodh-memory"]);
    expect(lanes.map((l) => l.total)).toEqual([1, 1]);
  });

  it("counts cancelled as settled but reports it separately from done", () => {
    // A project that abandoned half its scope is not permanently unfinished,
    // and "we finished it" is not "we dropped it".
    const lanes = lanesOf(
      [
        triage({ id: "a", project_id: "p", status: "done" }),
        triage({ id: "b", project_id: "p", status: "cancelled" }),
        triage({ id: "c", project_id: "p", status: "in_progress" }),
        triage({ id: "d", project_id: "p", status: "blocked" }),
      ],
      [project({ id: "p", name: "Pipeline" })],
    );
    expect(lanes[0]).toMatchObject({
      total: 4,
      done: 1,
      cancelled: 1,
      settled: 2,
      open: 2,
      underway: 1,
      blocked: 1,
    });
  });

  it("gives unfiled work its own lane so the lanes reconcile with the total", () => {
    const rows = [triage({ id: "a", project_id: "p" }), triage({ id: "b", project_id: null })];
    const lanes = lanesOf(rows, [project({ id: "p", name: "Pipeline" })]);
    expect(lanes.map((l) => l.name)).toEqual(["Pipeline", "No project"]);
    expect(lanes.reduce((sum, l) => sum + l.total, 0)).toBe(rows.length);
  });

  it("draws no lane for a project with nothing in it", () => {
    // The whole project list ships with every response; six of the nine on the
    // live profile hold nothing and would otherwise read "0 of 0".
    const lanes = lanesOf(
      [triage({ id: "a", project_id: "p" })],
      [project({ id: "p", name: "Pipeline" }), project({ id: "empty", name: "Canopy" })],
    );
    expect(lanes.map((l) => l.name)).toEqual(["Pipeline"]);
  });

  it("ranks running work above archived, and unfiled last", () => {
    const lanes = lanesOf(
      [
        triage({ id: "a", project_id: "old", created_at: at(20) }),
        triage({ id: "b", project_id: "live", created_at: at(1) }),
        triage({ id: "c", project_id: null, created_at: at(25) }),
      ],
      [
        project({ id: "old", name: "Codebase Audit", status: "archived" }),
        project({ id: "live", name: "recall-harness", status: "active" }),
      ],
    );
    // The archived lane is the more recent and still sits below the active one.
    expect(lanes.map((l) => l.name)).toEqual(["recall-harness", "Codebase Audit", "No project"]);
  });

  it("names a project the project list does not carry rather than heading it blank", () => {
    expect(lanesOf([triage({ id: "a", project_id: "ghost" })], [])[0].name).toBe("Unnamed project");
  });
});

describe("subtaskProgress", () => {
  it("has no opinion about a task with no subtasks", () => {
    // Null, not {done: 0, total: 0}: a zero would render a meter on every row.
    expect(subtaskProgress(triage({ id: "p" }), [triage({ id: "p" })])).toBeNull();
  });

  it("counts only its own children, and counts cancelled ones as settled", () => {
    const parent = triage({ id: "p" });
    const rows = [
      parent,
      triage({ id: "c1", parent_id: "p", status: "done" }),
      triage({ id: "c2", parent_id: "p", status: "cancelled" }),
      triage({ id: "c3", parent_id: "p", status: "todo" }),
      triage({ id: "x", parent_id: "other", status: "done" }),
    ];
    // A dropped subtask would otherwise strand the parent at "2 of 3" forever.
    expect(subtaskProgress(parent, rows)).toEqual({ done: 2, total: 3 });
  });
});

describe("boardOf", () => {
  it("counts open work without counting settled work as open", () => {
    const rows = [
      triage({ id: "a", status: "in_progress" }),
      triage({ id: "b", status: "blocked" }),
      triage({ id: "c", status: "todo" }),
      triage({ id: "d", status: "done" }),
      triage({ id: "e", status: "cancelled" }),
    ];
    expect(boardOf(rows, rows.length, [])).toMatchObject({
      open: 3,
      underway: 1,
      blocked: 1,
      settled: 2,
      done: 1,
      cancelled: 1,
    });
  });

  it("separates a declared block from a recorded dependency", () => {
    // "Nothing is blocked" is two claims: no task carries the status, AND no
    // dependency was ever recorded — and it is the second that makes the first
    // weak evidence rather than proof.
    const rows = [
      triage({ id: "a", status: "blocked", blocked_on: "counsel" }),
      triage({ id: "b", blocked_by: ["a"] }),
    ];
    const board = boardOf(rows, rows.length, []);
    expect(board.blocked).toBe(1);
    expect(board.waiting).toBe(1);
    expect(board.dependencies).toBe(1);
  });

  it("reports movement as zero when nothing has ever changed state", () => {
    // The live `claude` profile: 50 todos recorded inside 33 minutes on one
    // day, not one of them ever moved.
    const rows = [triage({ id: "a" }), triage({ id: "b" })];
    expect(boardOf(rows, rows.length, []).moved).toBe(0);
  });

  it("carries the truncation fact, which is what suppresses every ratio", () => {
    // The server paginates after sorting by manual order, priority and due date
    // — never by project — so a slice is arbitrary across all lanes at once.
    const rows = [triage({ id: "a" })];
    expect(boardOf(rows, 431, []).truncated).toBe(true);
    expect(boardOf(rows, 1, []).truncated).toBe(false);
  });
});

describe("axisOf", () => {
  const board = (moved: number): Board => ({
    shown: 0,
    total: 0,
    truncated: false,
    open: 0,
    underway: 0,
    blocked: 0,
    settled: 0,
    done: 0,
    cancelled: 0,
    projects: 0,
    moved,
    dependencies: 0,
    waiting: 0,
    from: null,
    to: null,
  });

  it("REFUSES TO DRAW WHEN NOTHING HAS EVER MOVED", () => {
    // With no transitions anywhere a strip can only plot when rows were written
    // down — a picture of an import that reads as one of progress.
    expect(axisOf(lanesOf([triage({ id: "a", created_at: at(1) })], []), board(0))).toBeNull();
  });

  it("spans every lane so two projects are drawn on one comparable scale", () => {
    const lanes = lanesOf(
      [
        triage({ id: "a", project_id: "p", created_at: at(1) }),
        triage({
          id: "b",
          project_id: "q",
          created_at: at(4),
          status: "done",
          completed_at: at(10),
          comments: [sysActivity("Marked complete after 6.0 days", at(10))],
        }),
      ],
      [],
    );
    expect(axisOf(lanes, board(1))).toEqual({
      from: Date.parse(at(1)),
      to: Date.parse(at(10)),
      span: Date.parse(at(10)) - Date.parse(at(1)),
    });
  });

  it("never yields a zero span, which would divide by zero at every mark", () => {
    expect(axisOf(lanesOf([triage({ id: "a", created_at: at(3) })], []), board(1))?.span).toBe(1);
  });
});

describe("positionOn", () => {
  const axis = { from: 100, to: 200, span: 100 };

  it("places an instant proportionally along the track", () => {
    expect(positionOn(axis, 150)).toBe(0.5);
  });

  it("clamps a mark that clock skew put outside the track", () => {
    // Two writes on different clocks can settle a todo before it was recorded.
    expect(positionOn(axis, 40)).toBe(0);
    expect(positionOn(axis, 900)).toBe(1);
  });
});

describe("authorKind", () => {
  it("treats only the server's own marker as evidence", () => {
    // `TodoComment::system_activity` hardcodes "system"; everything else on
    // this field is caller-supplied and unverified.
    expect(authorKind("system")).toEqual({ kind: "server" });
    expect(authorKind("shodh-dashboard")).toEqual({ kind: "dashboard" });
  });

  it("reads the model out of an agent signature when one is appended", () => {
    expect(authorKind("agent:anthropic/claude-opus-4")).toEqual({
      kind: "agent",
      model: "anthropic/claude-opus-4",
    });
  });

  it("STILL RECOGNISES A BARE agent: MARKER AS AN AGENT", () => {
    // The seat's `agentAuthor` currently returns the bare prefix with no model
    // appended, so matching an exact "agent:<provider>/<model>" shape would
    // classify every real agent write as an anonymous caller.
    expect(authorKind("agent:")).toEqual({ kind: "agent", model: null });
  });

  it("does not mistake a profile name for anything trustworthy", () => {
    // `author` defaults to `user_id`, so this is the common case and it means
    // only "whatever called the API under that name".
    expect(authorKind("claude-code")).toEqual({ kind: "caller", name: "claude-code" });
  });
});

describe("elapsedLabel", () => {
  const span = (minutes: number) => elapsedLabel(0, minutes * 60_000);

  it("NEVER PRODUCES A REMAINDER THAT REACHED ITS OWN BASE", () => {
    // "1h 60m" is the shape of this bug, and it reads as plausible. Rounding
    // happens once, to minutes; every larger unit is derived from that.
    expect(span(120)).toBe("2h");
    expect(span(119)).toBe("1h 59m");
    expect(span(60)).toBe("1h");
    expect(span(59.6)).toBe("1h");
    expect(span(24 * 60)).toBe("1d");
    expect(span(24 * 60 - 1)).toBe("23h 59m");
    expect(span(48 * 60 - 60)).toBe("1d 23h");
  });

  it("says so rather than printing 0m for a span under a minute", () => {
    expect(span(0)).toBe("under a minute");
  });

  it("refuses a negative span instead of rendering '-3h' beside a finished task", () => {
    // Two writes on different clocks can settle a todo before it was recorded.
    expect(elapsedLabel(1000, 0)).toBeNull();
  });
});

describe("laneCurve", () => {
  const axis = { from: Date.parse(at(1)), to: Date.parse(at(11)), span: Date.parse(at(11)) - Date.parse(at(1)) };

  const settledOn = (id: string, created: string, done: string) =>
    triage({
      id,
      project_id: "p",
      created_at: created,
      status: "done",
      completed_at: done,
      comments: [sysActivity("Marked complete after 1.0 days", done)],
    });

  it("steps scope up as work is recorded and settled up as it lands", () => {
    const lanes = lanesOf(
      [
        settledOn("a", at(1), at(6)),
        triage({ id: "b", project_id: "p", created_at: at(6), status: "todo" }),
      ],
      [],
    );
    const curve = laneCurve(lanes[0], axis);
    // Scope reaches everything recorded; settled stops at the half that landed.
    expect(curve.scope[curve.scope.length - 1]).toEqual({ x: 1, y: 1 });
    expect(curve.settled[curve.settled.length - 1]).toEqual({ x: 1, y: 0.5 });
  });

  it("collapses simultaneous events into one taller step, not a zero-width one", () => {
    // Thirteen tasks recorded in the same minute is the live AUDIT project.
    const lanes = lanesOf(
      [
        triage({ id: "a", project_id: "p", created_at: at(3) }),
        triage({ id: "b", project_id: "p", created_at: at(3) }),
      ],
      [],
    );
    const { scope } = laneCurve(lanes[0], axis);
    expect(scope).toEqual([
      { x: 0, y: 0 },
      { x: positionOn(axis, Date.parse(at(3))), y: 1 },
      { x: 1, y: 1 },
    ]);
  });

  it("carries the last level to the right edge rather than stopping mid-track", () => {
    // A curve ending early reads as work that stopped being counted.
    const lanes = lanesOf([settledOn("a", at(1), at(2))], []);
    expect(laneCurve(lanes[0], axis).settled.at(-1)).toEqual({ x: 1, y: 1 });
  });

  it("draws nothing for a lane where nothing has settled", () => {
    // Not a flat line at zero, which looks like a measurement of no progress
    // when it is really the absence of any settled task to plot.
    const lanes = lanesOf([triage({ id: "a", project_id: "p", created_at: at(2) })], []);
    expect(laneCurve(lanes[0], axis).settled).toEqual([
      { x: 0, y: 0 },
      { x: 1, y: 0 },
    ]);
  });
});

describe("stepPath", () => {
  it("moves horizontally at the OLD level before stepping up", () => {
    // A straight line between two completions would draw work progressing on
    // days when nothing happened.
    const path = stepPath(
      [
        { x: 0, y: 0 },
        { x: 0.5, y: 1 },
      ],
      100,
      10,
    );
    expect(path).toBe("M 0.00,10.00 L 50.00,10.00 L 50.00,0.00");
  });

  it("inverts y, because SVG grows downward and the curve grows up", () => {
    expect(stepPath([{ x: 0, y: 1 }], 100, 10)).toBe("M 0.00,0.00");
  });

  it("draws nothing at all for an empty series", () => {
    expect(stepPath([], 100, 10)).toBe("");
  });
});
