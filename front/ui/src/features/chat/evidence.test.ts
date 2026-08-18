import { describe, expect, it } from "vitest";
import type { ChatOp, ChatTurn } from "@/stores/chat";
import type { RecallMemory } from "@/lib/api/types";
import { resolveSelection } from "./EvidencePanel";

/**
 * The lookup behind every click in the evidence panel.
 *
 * These are written from a measured failure, not from a hypothetical. On the
 * live `claude-code` seat, one conversation logged 252 React duplicate-key
 * errors; five turns held 42 rows whose identity collided, and every colliding
 * pair carried a DIFFERENT score. Memory `a8ca63ff` was returned by two of turn
 * 3's searches, at 0.0816 and at 0.95. Clicking the row drawn at 95% opened a
 * breakdown reading `Vector + keyword 0.0589 · Final score 0.0418` — the other
 * search's working. The fixture below is that turn, reduced.
 *
 * Every test was checked to fail by mutating the line it covers; the mutation
 * is named in each.
 */

function memory(id: string, score: number, content: string): RecallMemory {
  return {
    id,
    experience: { content, memory_type: "Task", tags: [] },
    importance: 0.5,
    created_at: "2026-04-16T10:03:41Z",
    score,
    tier: "Working",
    score_attribution: {
      memory_id: id,
      rrf_base: score,
      graph_rrf: 0,
      hybrid_rrf: score,
    } as RecallMemory["score_attribution"],
  };
}

function recallOp(query: string, memories: RecallMemory[]): ChatOp {
  return {
    type: "memory_recall",
    scope: "user",
    query,
    mode: "hybrid",
    memories,
    facts: [],
    todos: [],
    lineage: [],
    took_ms: 12,
  };
}

const SHARED = "a8ca63ff-54e7-45a9-a195-f4a23fccccba";

/** Turn 3 as measured: two searches, both of which returned `SHARED`. */
const TURN_THREE: ChatTurn = {
  turn: 3,
  userText: "how does the PQ codebook work?",
  ops: [
    recallOp("IVF-PQ index codebook adc kmeans quantization src/ivfpq", [
      memory(SHARED, 0.0816069, "[SHOD-50] No benchmarks for critical hot paths"),
      memory("48d4ac7d-7011-475d-a323-cdf0e2f1fa06", 0.8338, "[SHOD-15] Corrupted misuse"),
    ]),
    recallOp("product quantization codebook centroid probe rerank distance vector", [
      memory(SHARED, 0.95, "[SHOD-50] No benchmarks for critical hot paths"),
    ]),
  ],
  assistantText: "",
  thinkingText: "",
  usage: null,
  pending: false,
};

/** A turn whose seat-assigned label does not equal its position in the array.
 *  `applyEvent`'s `turn_start` writes `event.turn` onto whichever turn is last,
 *  from a counter the seat owns (`this.turn += 1`, restored from persistence),
 *  so nothing guarantees label === position + 1. */
const RELABELLED: ChatTurn[] = [
  { ...TURN_THREE, turn: 9, ops: [recallOp("first", [memory("aaa", 0.1, "first turn's memory")])] },
  { ...TURN_THREE, turn: 9 },
];

describe("resolveSelection", () => {
  it("returns the op the reader clicked, not the first op that happens to hold the memory", () => {
    // THE DEFECT, DIRECTLY. Both ops hold `SHARED`. A resolver that scans the
    // turn returns op 0 for both rows, so the 0.95 row shows 0.0816's working.
    // Mutation caught: replacing `turns[turnIndex]?.ops[opIndex]` with a scan
    // over `turns[turnIndex].ops` returns 0.0816 here.
    const strong = resolveSelection([TURN_THREE], 0, 1, SHARED);
    expect(strong?.memory?.score).toBe(0.95);
    expect(strong?.memory?.score_attribution?.rrf_base).toBe(0.95);

    const weak = resolveSelection([TURN_THREE], 0, 0, SHARED);
    expect(weak?.memory?.score).toBe(0.0816069);
  });

  it("carries the clicked op's own result set as siblings", () => {
    // `siblings` feeds the lineage hops, so resolving to the wrong op sends a
    // reader into a different search's result set. Mutation caught: returning
    // `turns[turnIndex].ops[0].memories` regardless of `opIndex` gives 2.
    expect(resolveSelection([TURN_THREE], 0, 1, SHARED)?.siblings).toHaveLength(1);
    expect(resolveSelection([TURN_THREE], 0, 0, SHARED)?.siblings).toHaveLength(2);
  });

  it("addresses the turn by position, so a relabelled turn still resolves to itself", () => {
    // Both turns here carry the label 9. Mutation caught: restoring
    // `turns[turn - 1]` and passing the label reaches `turns[8]`, which is
    // undefined, and the panel silently renders nothing where evidence exists.
    expect(resolveSelection(RELABELLED, 0, 0, "aaa")?.memory?.score).toBe(0.1);
    expect(resolveSelection(RELABELLED, 1, 1, SHARED)?.memory?.score).toBe(0.95);
  });

  it("returns null when the named op does not hold the named memory", () => {
    // A stale selection must not fall through to a neighbouring op and render
    // a record the reader never chose. Mutation caught: deleting the
    // `if (!memory) return null` and letting the function continue.
    expect(resolveSelection([TURN_THREE], 0, 1, "48d4ac7d-7011-475d-a323-cdf0e2f1fa06")).toBeNull();
  });

  it("returns null for an out-of-range turn or op rather than throwing", () => {
    // Mutation caught: dropping the `?.` makes an out-of-range turn a
    // TypeError, which unmounts the panel instead of clearing the selection.
    expect(resolveSelection([TURN_THREE], 4, 0, SHARED)).toBeNull();
    expect(resolveSelection([TURN_THREE], 0, 7, SHARED)).toBeNull();
  });

  it("resolves a proactively surfaced memory as surfaced, not as recalled", () => {
    // The two kinds render differently — a surfaced memory has no score
    // attribution to show. Mutation caught: collapsing the `proactive_context`
    // branch into the recall one returns kind "recalled" with a null memory.
    const turn: ChatTurn = {
      ...TURN_THREE,
      ops: [
        {
          type: "proactive_context",
          scope: "user",
          query: "codebook",
          memories: [
            {
              id: SHARED,
              content: "surfaced before the model answered",
              memory_type: "Task",
              score: 0.42,
              importance: 0.5,
              created_at: "2026-04-16T10:03:41Z",
              tags: [],
              tier: "Working",
              relevance_reason: "semantic",
            },
          ],
          injected_memory_ids: [],
          feedback: null,
          temporal_credits_applied: null,
          took_ms: 8,
        },
      ],
    };
    const resolved = resolveSelection([turn], 0, 0, SHARED);
    expect(resolved?.kind).toBe("surfaced");
    expect(resolved?.proactive?.score).toBe(0.42);
    expect(resolved?.memory).toBeNull();
  });

  it("returns null for an op that carries no memories at all", () => {
    // Every other op type in a turn — a write, a tool call, a model swap — is
    // addressable by index and must not resolve. Mutation caught: a fallthrough
    // `return { kind: "recalled", ... }` at the end of the function.
    const turn: ChatTurn = {
      ...TURN_THREE,
      ops: [
        {
          type: "memory_write",
          scope: "user",
          memory_id: SHARED,
          memory_type: "Task",
          content_preview: "written, not recalled",
          ledger_event_id: "led-1",
        },
      ],
    };
    expect(resolveSelection([turn], 0, 0, SHARED)).toBeNull();
  });
});
