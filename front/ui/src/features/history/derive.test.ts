import { describe, expect, it } from "vitest";
import type { AuditRow, AuditSource, LedgerActorView } from "@/lib/seat/types";
import {
  actorLabel,
  auditExportPath,
  conversationsIn,
  exportFilename,
  formatDuration,
  groupByDay,
  kindLabel,
  matchesView,
  outcomeOf,
  parseAuditJsonl,
  summarise,
  toggle,
  toolCensus,
  toolCallDetail,
  windowSince,
} from "./derive";

/**
 * The History screen is an audit surface, so every failure mode worth pinning
 * here is a failure that renders as a PLAUSIBLE, CONFIDENT STATEMENT: a tool
 * call that never returned shown as an instant success, an actor this build did
 * not recognise relabelled "unknown", a median interpolated between two
 * durations that no call actually took, a census that reshuffles between reads.
 * None of those look wrong on screen, which is exactly why they are tested.
 */

const row = (over: Partial<AuditRow> = {}): AuditRow => ({
  ts: "2026-08-16T10:00:00.000Z",
  source: "ledger",
  actor: "system",
  kind: "reinforce",
  user_id: "varun",
  conversation_id: "c1",
  turn: 1,
  ref: "e1",
  detail: '{"scope":"user"}',
  ...over,
});

/** A tool-call row with the tail `seat/src/audit.ts` `toolCallRow` encodes. */
const call = (over: {
  ts?: string;
  name?: string;
  conversation?: string;
  ref?: string;
  ended?: string | null;
  duration?: number | null;
  error?: boolean | null;
}): AuditRow =>
  row({
    ts: over.ts ?? "2026-08-16T10:00:00.000Z",
    source: "tool_call",
    actor: "agent",
    kind: over.name ?? "recall_memory",
    conversation_id: over.conversation ?? "c1",
    ref: over.ref ?? "toolu_1",
    detail: JSON.stringify({
      args: { query: "x" },
      ended_at: over.ended ?? null,
      duration_ms: over.duration ?? null,
      is_error: over.error ?? null,
    }),
  });

const jsonl = (rows: AuditRow[]): string => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";

describe("parseAuditJsonl", () => {
  it("keeps the server's order, which is the export's order", () => {
    // buildAuditRows sorts on (ts, source, ref) so two exports of a window are
    // byte-identical. A parser that re-sorted would make the screen disagree
    // with the file it downloads — and both orders look fine in isolation.
    const late = row({ ts: "2026-08-16T12:00:00.000Z", ref: "b" });
    const early = row({ ts: "2026-08-16T09:00:00.000Z", ref: "a" });
    const parsed = parseAuditJsonl(jsonl([late, early]));
    expect(parsed.rows.map((r) => r.ref)).toEqual(["b", "a"]);
  });

  it("counts a torn line instead of dropping it", () => {
    const parsed = parseAuditJsonl(`${JSON.stringify(row())}\n{"ts":"2026-08\n`);
    expect(parsed.rows).toHaveLength(1);
    expect(parsed.unreadable).toBe(1);
  });

  it("refuses an actor this build does not know rather than calling it unknown", () => {
    // "unknown" is a FACT the seat writes for entries predating the actor field
    // and refuses to backfill. Relabelling an actor we merely failed to
    // recognise would manufacture that fact for a row that has a real one.
    const parsed = parseAuditJsonl(jsonl([row({ actor: "operator" as LedgerActorView })]));
    expect(parsed.rows).toHaveLength(0);
    expect(parsed.unreadable).toBe(1);
  });

  it("refuses a source this build does not know", () => {
    const parsed = parseAuditJsonl(jsonl([row({ source: "backend" as AuditSource })]));
    expect(parsed.unreadable).toBe(1);
  });

  it("requires every column, at the type the seat writes", () => {
    const { turn: _turn, ...missing } = row();
    const wrongType = { ...row(), turn: "1" };
    const parsed = parseAuditJsonl(
      `${JSON.stringify(missing)}\n${JSON.stringify(wrongType)}\n${JSON.stringify([row()])}\n`,
    );
    expect(parsed.rows).toHaveLength(0);
    expect(parsed.unreadable).toBe(3);
  });

  it("reads the seat's trailing newline and its empty body without inventing a row", () => {
    // toJsonl emits a trailing newline for a non-empty trail and "" for an
    // empty one (seat/src/audit.ts).
    expect(parseAuditJsonl("")).toEqual({ rows: [], unreadable: 0 });
    expect(parseAuditJsonl(jsonl([row()])).rows).toHaveLength(1);
  });
});

describe("toolCallDetail", () => {
  it("passes the unterminated nulls through untouched", () => {
    // The single most misleading thing this screen could do is default these:
    // `duration_ms ?? 0` and `is_error ?? false` turn "invoked and never
    // returned" into "returned instantly and succeeded".
    const detail = toolCallDetail(call({}));
    expect(detail).toEqual({
      args: { query: "x" },
      ended_at: null,
      duration_ms: null,
      is_error: null,
    });
  });

  it("is null for a row that is not a tool call", () => {
    expect(toolCallDetail(row())).toBeNull();
    expect(toolCallDetail(row({ source: "retrieval", kind: "memory_recall" }))).toBeNull();
  });

  it("is null rather than a partial record when the payload cannot be read", () => {
    expect(toolCallDetail(row({ source: "tool_call", detail: "{not json" }))).toBeNull();
  });
});

describe("outcomeOf", () => {
  it("separates a hang from a failure and from a success", () => {
    expect(outcomeOf(call({}))).toBe("unterminated");
    expect(outcomeOf(call({ ended: "2026-08-16T10:00:01.000Z", duration: 1000, error: true }))).toBe(
      "error",
    );
    expect(outcomeOf(call({ ended: "2026-08-16T10:00:01.000Z", duration: 1000, error: false }))).toBe(
      "ok",
    );
  });

  it("gives ledger and retrieval rows no outcome at all", () => {
    // Neither has a success/failure axis. An "ok" here would let a reader count
    // successes that were never measured.
    expect(outcomeOf(row())).toBeNull();
    expect(outcomeOf(row({ source: "retrieval", kind: "proactive_context" }))).toBeNull();
  });
});

describe("summarise", () => {
  const trail = [
    row({ ts: "2026-08-16T09:00:00.000Z", actor: "unknown", conversation_id: "c1" }),
    call({ ts: "2026-08-16T10:00:00.000Z", duration: 10, ended: "x", error: false }),
    call({ ts: "2026-08-16T10:01:00.000Z", duration: 20, ended: "x", error: false, ref: "t2" }),
    call({ ts: "2026-08-16T10:02:00.000Z", duration: 30, ended: "x", error: true, ref: "t3" }),
    call({ ts: "2026-08-16T10:03:00.000Z", conversation: "c2", ref: "t4" }),
    row({ ts: "2026-08-16T11:00:00.000Z", source: "retrieval", kind: "memory_recall", actor: "agent", conversation_id: "c3" }),
  ];

  it("reports a median some call actually took", () => {
    // Nearest rank over [10, 20, 30]. An interpolating median would be legal
    // arithmetic and a duration that appears nowhere in the reviewer's file.
    const summary = summarise(trail);
    expect(summary.durationP50).toBe(20);
    expect(summary.durationMax).toBe(30);
  });

  it("leaves an unterminated call out of the duration figures entirely", () => {
    // Counting it as 0 would drag the median down; counting it as the window
    // length would invent a duration. It is absent from both, and counted
    // separately where it can be seen.
    expect(summarise([call({}), call({ duration: 500, ended: "x", error: false, ref: "t2" })]))
      .toMatchObject({ durationP50: 500, durationMax: 500, unterminated: 1, toolCalls: 2 });
  });

  it("counts failures and hangs apart", () => {
    expect(summarise(trail)).toMatchObject({ failed: 1, unterminated: 1, toolCalls: 4 });
  });

  it("counts distinct conversations, not rows", () => {
    expect(summarise(trail).conversations).toBe(3);
  });

  it("takes the span from the extremes, not from the ends of the array", () => {
    const shuffled = [trail[3], trail[0], trail[5]];
    expect(summarise(shuffled).span).toEqual({
      from: "2026-08-16T09:00:00.000Z",
      to: "2026-08-16T11:00:00.000Z",
    });
  });

  it("keeps a zero for every actor and source so the filter can say 'none'", () => {
    const summary = summarise([row({ actor: "system" })]);
    expect(summary.actors).toEqual({ user: 0, agent: 0, system: 1, unknown: 0 });
    expect(summary.sources).toEqual({ ledger: 1, tool_call: 0, retrieval: 0 });
  });

  it("has no span on an empty trail rather than an invalid one", () => {
    expect(summarise([]).span).toBeNull();
    expect(summarise([]).durationP50).toBeNull();
  });
});

describe("toolCensus", () => {
  const trail = [
    call({ name: "recall_memory", duration: 100, ended: "x", error: false, ref: "a" }),
    call({ name: "recall_memory", duration: 900, ended: "x", error: false, ref: "b" }),
    call({ name: "recall_memory", duration: 500, ended: "x", error: true, ref: "c" }),
    call({ name: "remember_memory", duration: 40, ended: "x", error: false, ref: "d" }),
    call({ name: "read_file", ref: "e" }),
    call({ name: "write_file", duration: 40, ended: "x", error: false, ref: "f" }),
    row(),
  ];

  it("ranks by calls and breaks ties by name, so two reads agree", () => {
    // Ties resolved by input order would reshuffle between reads of the same
    // window, and a reader comparing censuses would see movement that is not
    // in the data.
    expect(toolCensus(trail).map((t) => t.name)).toEqual([
      "recall_memory",
      "read_file",
      "remember_memory",
      "write_file",
    ]);
  });

  it("reports each tool's own completed durations", () => {
    const [recall] = toolCensus(trail);
    expect(recall).toMatchObject({ calls: 3, failed: 1, unterminated: 0, p50: 500, max: 900 });
  });

  it("keeps a tool that only ever hung, with no duration claimed for it", () => {
    const hung = toolCensus(trail).find((t) => t.name === "read_file");
    expect(hung).toMatchObject({ calls: 1, unterminated: 1, p50: null, max: null });
  });

  it("counts no ledger or retrieval row as a tool call", () => {
    expect(toolCensus([row(), row({ source: "retrieval", kind: "memory_recall" })])).toEqual([]);
  });
});

describe("conversationsIn", () => {
  it("lists each conversation once, in the order it first appears", () => {
    const ids = conversationsIn([
      row({ conversation_id: "c2" }),
      row({ conversation_id: "c1" }),
      row({ conversation_id: "c2" }),
    ]);
    expect(ids).toEqual(["c2", "c1"]);
  });
});

describe("groupByDay", () => {
  /** Built from local components so the assertion does not depend on the
   *  runner's time zone — which is the whole point of the function. */
  const atLocal = (y: number, m: number, d: number, h: number, min = 0): string =>
    new Date(y, m - 1, d, h, min).toISOString();

  it("splits on the reader's midnight, not on the UTC date in the string", () => {
    // A reader east of Greenwich asking "what happened on Tuesday" means their
    // Tuesday. Keying on `ts.slice(0, 10)` files their morning under Monday.
    const groups = groupByDay([
      row({ ts: atLocal(2026, 8, 16, 23, 59) }),
      row({ ts: atLocal(2026, 8, 17, 0, 1) }),
    ]);
    expect(groups.map((g) => g.day)).toEqual(["2026-08-16", "2026-08-17"]);
  });

  it("keeps rows in the order they arrived", () => {
    const groups = groupByDay([
      row({ ts: atLocal(2026, 8, 16, 9), ref: "b" }),
      row({ ts: atLocal(2026, 8, 16, 8), ref: "a" }),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].rows.map((r) => r.ref)).toEqual(["b", "a"]);
  });

  it("groups nothing into nothing", () => {
    expect(groupByDay([])).toEqual([]);
  });
});

describe("matchesView", () => {
  const none = { actors: new Set<LedgerActorView>(), sources: new Set<AuditSource>() };

  it("narrows nothing when a set is empty", () => {
    expect(matchesView(row(), none)).toBe(true);
  });

  it("requires both axes at once", () => {
    const filter = {
      actors: new Set<LedgerActorView>(["agent"]),
      sources: new Set<AuditSource>(["tool_call"]),
    };
    expect(matchesView(call({}), filter)).toBe(true);
    expect(matchesView(row({ actor: "agent" }), filter)).toBe(false);
    expect(matchesView(row({ source: "tool_call", actor: "system" }), filter)).toBe(false);
  });
});

describe("toggle", () => {
  it("adds a missing member and removes a present one, without mutating", () => {
    const start: ReadonlySet<string> = new Set(["a"]);
    expect([...toggle(start, "b")]).toEqual(["a", "b"]);
    expect([...toggle(start, "a")]).toEqual([]);
    expect([...start]).toEqual(["a"]);
  });
});

describe("windowSince", () => {
  const NOW = Date.parse("2026-08-16T12:00:00.000Z");

  it("has no lower bound for Everything", () => {
    expect(windowSince("all", NOW)).toBeNull();
  });

  it("subtracts the window's own duration", () => {
    expect(windowSince("day", NOW)).toBe("2026-08-15T12:00:00.000Z");
    expect(windowSince("week", NOW)).toBe("2026-08-09T12:00:00.000Z");
    expect(windowSince("month", NOW)).toBe("2026-07-17T12:00:00.000Z");
  });
});

describe("auditExportPath", () => {
  const NOW = Date.parse("2026-08-16T12:00:00.000Z");

  it("sends no since and no conversation for the widest read", () => {
    expect(auditExportPath({ window: "all", conversationId: null }, "jsonl", NOW)).toBe(
      "/seat/v1/audit/export?format=jsonl",
    );
  });

  it("never sends an until — an audit read has no upper bound", () => {
    // until=now would exclude anything written between the click and the read.
    expect(auditExportPath({ window: "day", conversationId: null }, "csv", NOW)).not.toContain(
      "until",
    );
  });

  it("carries the window and the conversation, encoded", () => {
    expect(auditExportPath({ window: "day", conversationId: "c 1/2" }, "csv", NOW)).toBe(
      "/seat/v1/audit/export?format=csv&since=2026-08-15T12%3A00%3A00.000Z&conversation_id=c+1%2F2",
    );
  });
});

describe("exportFilename", () => {
  it("mirrors the name the seat puts on the attachment", () => {
    // seat/src/server.ts: ISO with ':' replaced, sliced to seconds. Rebuilt
    // here because the shodh-front proxy drops Content-Disposition.
    expect(exportFilename("csv", Date.parse("2026-08-16T15:57:30.123Z"))).toBe(
      "shodh-audit-2026-08-16T15-57-30.csv",
    );
  });
});

describe("formatDuration", () => {
  it("returns null for an absent duration rather than a dash", () => {
    expect(formatDuration(null)).toBeNull();
  });

  it("steps precision down as the number grows", () => {
    expect(formatDuration(793)).toBe("793ms");
    expect(formatDuration(1240)).toBe("1.2s");
    expect(formatDuration(30_016)).toBe("30s");
    expect(formatDuration(125_400)).toBe("2m 05s");
  });

  it("shows a zero duration as a measurement, not as nothing", () => {
    expect(formatDuration(0)).toBe("0ms");
  });
});

describe("kindLabel", () => {
  it("labels a tool call with the tool's own name and nothing else", () => {
    // The name is the answer to "which tool was used"; a phrase around it
    // buries the one string the reader came for.
    expect(kindLabel(call({ name: "recall_memory" }))).toBe("recall_memory");
  });

  it("puts the seat's event names into words", () => {
    expect(kindLabel(row({ kind: "implicit_feedback" }))).toBe("Server adjusted memories");
    expect(kindLabel(row({ source: "retrieval", kind: "proactive_context" }))).toBe(
      "Surfaced context unasked",
    );
  });

  it("shows a kind it does not recognise rather than hiding it", () => {
    expect(kindLabel(row({ kind: "consolidation_sweep" }))).toBe("consolidation_sweep");
  });
});

describe("actorLabel", () => {
  it("keeps unknown as its own word, never folded into a real actor", () => {
    expect(actorLabel("unknown")).toBe("Unknown");
    expect(actorLabel("system")).toBe("Automatic");
    expect(actorLabel("agent")).toBe("Model");
    expect(actorLabel("user")).toBe("Person");
  });
});
