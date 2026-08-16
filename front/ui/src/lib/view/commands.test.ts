import { describe, expect, it } from "vitest";
import type { SeatEvent } from "@/lib/seat/types";
import type { RecallFact, RecallMemory } from "@/lib/api/types";
import { ENTITY_LIMIT, commandsFromOp, describeCommands, dimensionsOf } from "./commands";

/**
 * The event→command translation is the contract between the conversation and
 * the interface, so these are the tests that must be able to fail: every one of
 * them asserts a specific command list, not merely that something happened.
 */

function memory(id: string, geo?: [number, number, number]): RecallMemory {
  return {
    id,
    experience: {
      content: `memory ${id}`,
      memory_type: "Observation",
      tags: [],
      ...(geo ? { geo_location: geo } : {}),
    },
    importance: 0.5,
    created_at: "2026-08-14T00:00:00Z",
    score: 0.9,
    tier: "Working",
  };
}

function fact(id: string, entities: string[]): RecallFact {
  return { id, fact: `fact ${id}`, confidence: 0.8, support_count: 1, related_entities: entities };
}

function recall(over: Partial<Extract<SeatEvent, { type: "memory_recall" }>>): SeatEvent {
  return {
    type: "memory_recall",
    scope: "user",
    query: "baltimore port",
    mode: "hybrid",
    memories: [],
    facts: [],
    todos: [],
    lineage: [],
    took_ms: 12,
    ...over,
  };
}

describe("commandsFromOp", () => {
  it("turns a recall into a cue carrying the query and the entities behind it", () => {
    const commands = commandsFromOp(
      recall({ memories: [memory("m1")], facts: [fact("f1", ["Maersk", "Patapsco"])] }),
      "/graph",
    );

    expect(commands).toEqual([
      { dimension: "cue", text: "baltimore port", entities: ["Maersk", "Patapsco"] },
      { dimension: "frame", entities: ["Maersk", "Patapsco"] },
    ]);
  });

  it("ignores harness-scope recalls — the seat's own bookkeeping is not a question the user asked", () => {
    const op = recall({ scope: "harness", memories: [memory("m1")], facts: [fact("f1", ["x"])] });
    expect(commandsFromOp(op, "/graph")).toEqual([]);
  });

  it("ignores everything that is not a recall", () => {
    const write: SeatEvent = {
      type: "memory_write",
      scope: "user",
      memory_id: "m1",
      memory_type: "Observation",
      content_preview: "…",
      ledger_event_id: "l1",
    };
    expect(commandsFromOp(write, "/graph")).toEqual([]);
    expect(commandsFromOp({ type: "turn_start", turn: 1 }, "/graph")).toEqual([]);
  });

  it("changes the cue but never the frame or the destination when a recall returned nothing", () => {
    // "A recall that returned nothing must NOT blank the view." The cue moves
    // so the interface states what was asked; the corpus stays framed.
    const commands = commandsFromOp(recall({ memories: [], facts: [] }), "/");

    expect(commands).toEqual([{ dimension: "cue", text: "baltimore port", entities: [] }]);
    expect(dimensionsOf(commands)).toEqual(["cue"]);
  });

  it("emits no frame when the facts carry no entity terms, rather than framing on nothing", () => {
    const commands = commandsFromOp(
      recall({ memories: [memory("m1")], facts: [fact("f1", [])] }),
      "/graph",
    );

    // The cue survives on the query text alone — that channel costs nothing and
    // is the one that works on a corpus whose fact extraction is thin.
    expect(commands).toEqual([{ dimension: "cue", text: "baltimore port", entities: [] }]);
  });

  it("opens the map when the answer is located, and the graph when it is not", () => {
    const located = commandsFromOp(
      recall({ memories: [memory("m1", [39.26, -76.57, 0])] }),
      "/graph",
    );
    expect(located).toContainEqual({ dimension: "destination", path: "/geo" });

    const plain = commandsFromOp(recall({ memories: [memory("m1")] }), "/");
    expect(plain).toContainEqual({ dimension: "destination", path: "/graph" });
  });

  it("does not ask to open the destination it is already on", () => {
    const commands = commandsFromOp(recall({ memories: [memory("m1")] }), "/graph");
    expect(dimensionsOf(commands)).toEqual(["cue"]);
  });

  it("drops a blank query — there is nothing to state and nothing to match", () => {
    expect(commandsFromOp(recall({ query: "   ", memories: [memory("m1")] }), "/")).toEqual([]);
  });

  it("de-duplicates entity terms case-insensitively and keeps retrieval order", () => {
    const commands = commandsFromOp(
      recall({
        memories: [memory("m1")],
        facts: [fact("f1", ["Maersk", "maersk", " Patapsco "]), fact("f2", ["MAERSK", "Dali"])],
      }),
      "/graph",
    );
    expect(commands[0]).toEqual({
      dimension: "cue",
      text: "baltimore port",
      entities: ["Maersk", "Patapsco", "Dali"],
    });
  });

  it("caps the entity list, so a cue narrows instead of lighting the whole graph", () => {
    const many = Array.from({ length: ENTITY_LIMIT + 15 }, (_, i) => `entity-${i}`);
    const commands = commandsFromOp(
      recall({ memories: [memory("m1")], facts: [fact("f1", many)] }),
      "/graph",
    );
    const cue = commands[0] as { entities: string[] };
    expect(cue.entities).toHaveLength(ENTITY_LIMIT);
    expect(cue.entities[0]).toBe("entity-0");
  });
});

describe("describeCommands", () => {
  it("says what following would do, in the words a person would use", () => {
    expect(
      describeCommands([
        { dimension: "cue", text: "baltimore port", entities: [] },
        { dimension: "destination", path: "/geo" },
      ]),
    ).toBe("show what it recalled for “baltimore port” and open the map");
  });

  it("is empty for no commands, so the offer cannot render a bare frame", () => {
    expect(describeCommands([])).toBe("");
  });
});
