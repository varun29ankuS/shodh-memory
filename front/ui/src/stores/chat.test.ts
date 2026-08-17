import { describe, expect, it } from "vitest";
import type { SeatEvent } from "@/lib/seat/types";
import { applyEvent, type ConvoLive } from "./chat";

/**
 * The wire→store fold, tested at the ONE hop that has no other witness.
 *
 * `app/useAgentView.ts` reads the live turn's `ops` and nothing else. That list
 * is therefore the entire channel between an event arriving on the stream and
 * the view bus acting on it — and a type left out of it does not fail, does not
 * warn, and does not typecheck differently. It simply never arrives, and the
 * seat waits out its timeout over an answer this browser could have given
 * instantly.
 *
 * That is exactly the bug these tests were written for: `view_probe` was routed
 * to the "not evidence" group, which made the probe handler in `useAgentView`
 * dead code and `inspect_view` permanently blind. Neither side's tests could see
 * it — the browser tests drove the bus directly and the seat tests posted the
 * answer by hand, so the broken hop was the one hop nothing crossed.
 */

const EMPTY: ConvoLive = {
  turns: [
    {
      turn: 1,
      userText: "where did this come from?",
      ops: [],
      assistantText: "",
      thinkingText: "",
      usage: null,
      pending: true,
    },
  ],
  streaming: true,
  model: null,
  totals: {
    input: 0,
    output: 0,
    cache_read: 0,
    cache_write: 0,
    reasoning: 0,
    total_tokens: 0,
    cost_total: 0,
  },
  transportError: null,
};

const opTypes = (convo: ConvoLive): string[] => convo.turns[0].ops.map((op) => op.type);

describe("applyEvent — what reaches the view bus", () => {
  it("delivers a view_probe to the live turn's ops, or inspect_view is blind", () => {
    const probe: SeatEvent = { type: "view_probe", probe_id: "probe-1" };
    expect(opTypes(applyEvent(EMPTY, probe))).toEqual(["view_probe"]);
  });

  it("delivers a view_command, which is what moves the view at all", () => {
    const command: SeatEvent = {
      type: "view_command",
      tool_call_id: "call-1",
      reason: "these cluster on the coast",
      destination: "/geo",
      entities: ["Dali"],
      unresolved: [],
      focus: null,
    };
    expect(opTypes(applyEvent(EMPTY, command))).toEqual(["view_command"]);
  });

  it("does NOT deliver a view_outcome — that is this browser's own decision", () => {
    // It reaches the store only as a durable row when a conversation is
    // reopened. Rendered in the evidence panel it would play the reader's own
    // click back to them as something the model did.
    const outcome: SeatEvent = {
      type: "view_outcome",
      tool_call_id: "call-1",
      dimension: "destination",
      state: "declined",
      at: "/chat",
    };
    expect(opTypes(applyEvent(EMPTY, outcome))).toEqual([]);
  });

  it("keeps the two apart in one stream, in arrival order", () => {
    let convo = EMPTY;
    for (const event of [
      { type: "view_probe", probe_id: "p1" },
      { type: "view_outcome", tool_call_id: "c", dimension: "cue", state: "applied", at: "/graph" },
      { type: "view_probe", probe_id: "p2" },
    ] as SeatEvent[]) {
      convo = applyEvent(convo, event);
    }
    expect(opTypes(convo)).toEqual(["view_probe", "view_probe"]);
  });

  it("drops an event that arrives with no turn to hold it, rather than throwing", () => {
    // A probe can in principle arrive before `turn_start` has produced a turn.
    // The bus would have nothing to attach it to either way; what must not
    // happen is the stream handler dying mid-answer.
    const noTurns: ConvoLive = { ...EMPTY, turns: [] };
    expect(() => applyEvent(noTurns, { type: "view_probe", probe_id: "p1" })).not.toThrow();
    expect(applyEvent(noTurns, { type: "view_probe", probe_id: "p1" }).turns).toEqual([]);
  });
});
