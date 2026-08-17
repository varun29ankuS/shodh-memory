import { describe, expect, it } from "vitest";
import type { SeatEvent } from "@/lib/seat/types";
import type { RecallFact, RecallMemory } from "@/lib/api/types";
import { ENTITY_LIMIT, commandsFromOp, describeCommands, dimensionsOf, reasonOf } from "./commands";

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
  it("turns a recall into a cue carrying the query, the entities behind it, and its own words", () => {
    const commands = commandsFromOp(
      recall({ memories: [memory("m1")], facts: [fact("f1", ["Maersk", "Patapsco"])] }),
      "/graph",
    );

    // Facts first — they are the better signal where they exist. The query's
    // own words follow, because they are the signal that always exists.
    expect(commands).toEqual([
      {
        dimension: "cue",
        text: "baltimore port",
        entities: ["Maersk", "Patapsco", "baltimore", "port"],
      },
      { dimension: "frame", entities: ["Maersk", "Patapsco", "baltimore", "port"] },
    ]);
  });

  it("splits the model's query, because a phrase is a substring of no entity name", () => {
    // Measured on defence-live: this exact query returned nine memories, zero
    // facts, and matched nothing whole — the narrowing was a claim in a chip
    // over an unchanged picture until the phrase was split.
    const commands = commandsFromOp(
      recall({ query: "Hindustan Aeronautics Limited HAL", memories: [memory("m1")] }),
      "/graph",
    );
    expect(commands[0]).toEqual({
      dimension: "cue",
      text: "Hindustan Aeronautics Limited HAL",
      entities: ["Hindustan", "Aeronautics", "Limited", "HAL"],
    });
  });

  it("drops function words but never domain nouns", () => {
    const commands = commandsFromOp(
      recall({ query: "what do we know about the Tejas programme", memories: [memory("m1")] }),
      "/graph",
    );
    const cue = commands[0] as { entities: string[] };
    // "do" and "we" fall to the matcher's own length floor; the rest are named.
    expect(cue.entities).toEqual(["do", "we", "Tejas", "programme"]);
  });

  it("keeps hyphenated designations whole — MiG-21 is one name, not two", () => {
    const commands = commandsFromOp(
      recall({ query: "MiG-21 and Su-30MKI fleet", memories: [memory("m1")] }),
      "/graph",
    );
    const cue = commands[0] as { entities: string[] };
    expect(cue.entities).toEqual(["MiG-21", "Su-30MKI", "fleet"]);
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
    // so the interface states what was asked; the corpus stays framed and the
    // stage stays put.
    const commands = commandsFromOp(recall({ memories: [], facts: [] }), "/");

    expect(dimensionsOf(commands)).toEqual(["cue"]);
  });

  it("emits no frame when the query is all function words, rather than framing on nothing", () => {
    const commands = commandsFromOp(
      recall({ query: "what about them", memories: [memory("m1")], facts: [fact("f1", [])] }),
      "/graph",
    );

    expect(commands).toEqual([{ dimension: "cue", text: "what about them", entities: [] }]);
  });

  it("opens the graph, and never the map — even for a located answer", () => {
    // The map raises points by the COMMITTED query (features/geo/GeoView.tsx)
    // and has no cue channel, so arriving there from a recall would show the
    // whole corpus, unmoved, under a chip claiming the view is following the
    // conversation. Refusing to offer it is the honest option until the map
    // consumes the cue. This test is the thing that must be changed then.
    const located = commandsFromOp(
      recall({ memories: [memory("m1", [39.26, -76.57, 0])] }),
      "/",
    );
    expect(located).toContainEqual({ dimension: "destination", path: "/graph" });
    expect(located.every((c) => c.dimension !== "destination" || c.path !== "/geo")).toBe(true);

    const plain = commandsFromOp(recall({ memories: [memory("m1")] }), "/");
    expect(plain).toContainEqual({ dimension: "destination", path: "/graph" });
  });

  it("does not ask to open the destination it is already on", () => {
    const commands = commandsFromOp(recall({ memories: [memory("m1")] }), "/graph");
    expect(dimensionsOf(commands)).toEqual(["cue", "frame"]);
  });

  it("drops a blank query — there is nothing to state and nothing to match", () => {
    expect(commandsFromOp(recall({ query: "   ", memories: [memory("m1")] }), "/")).toEqual([]);
  });

  it("de-duplicates terms case-insensitively and keeps retrieval order", () => {
    const commands = commandsFromOp(
      recall({
        query: "Dali",
        memories: [memory("m1")],
        facts: [fact("f1", ["Maersk", "maersk", " Patapsco "]), fact("f2", ["MAERSK", "Dali"])],
      }),
      "/graph",
    );
    expect(commands[0]).toEqual({
      dimension: "cue",
      text: "Dali",
      entities: ["Maersk", "Patapsco", "Dali"],
    });
  });

  it("caps the term list, so a cue narrows instead of lighting the whole graph", () => {
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
        { dimension: "destination", path: "/graph" },
      ]),
    ).toBe("follow its cue “baltimore port” and open the graph");
  });

  it("is empty for no commands, so the offer cannot render a bare frame", () => {
    expect(describeCommands([])).toBe("");
  });
});

/* -------------------------------------------------------------------------- *
 * THE MODEL ASKING, RATHER THAN THE BROWSER INFERRING
 * -------------------------------------------------------------------------- */

function request(
  over: Partial<Extract<SeatEvent, { type: "view_command" }>> = {},
): SeatEvent {
  return {
    type: "view_command",
    tool_call_id: "call-1",
    reason: "these 12 memories cluster on the Malabar coast",
    destination: "/geo",
    entities: ["Malabar Coast", "Dali"],
    unresolved: [],
    focus: null,
    ...over,
  };
}

describe("commandsFromOp — view_command", () => {
  it("carries the model's reason on every command the request produced", () => {
    const commands = commandsFromOp(request(), "/chat");

    expect(commands).toEqual([
      {
        dimension: "cue",
        text: "Malabar Coast, Dali",
        entities: ["Malabar Coast", "Dali"],
        reason: "these 12 memories cluster on the Malabar coast",
        origin: "call-1",
      },
      {
        dimension: "frame",
        entities: ["Malabar Coast", "Dali"],
        reason: "these 12 memories cluster on the Malabar coast",
        origin: "call-1",
      },
      {
        dimension: "destination",
        path: "/geo",
        reason: "these 12 memories cluster on the Malabar coast",
        origin: "call-1",
      },
    ]);
  });

  it("puts the ENTITIES in the cue field, never the reason", () => {
    // `dispatch` writes a cue through to the visible search field. The reason is
    // prose about evidence; a person watching their own search box fill with a
    // sentence they did not type has been handed a different app. The field has
    // to hold something they could have typed and can check the picture against.
    const [cue] = commandsFromOp(request({ reason: "a sentence about evidence" }), "/chat");
    expect(cue).toMatchObject({ dimension: "cue", text: "Malabar Coast, Dali" });
  });

  it("sends BOTH the cue and the frame, because they light different things", () => {
    // The cue is what recedes the unmatched (and it is the map's only channel);
    // the frame is the graph's camera. One without the other either aims at
    // entities that were never distinguished, or distinguishes entities that are
    // off screen.
    expect(dimensionsOf(commandsFromOp(request({ destination: null }), "/graph"))).toEqual([
      "cue",
      "frame",
    ]);
  });

  it("does not navigate to the surface the person is already on", () => {
    const commands = commandsFromOp(request({ destination: "/geo" }), "/geo");
    expect(dimensionsOf(commands)).toEqual(["cue", "frame"]);
  });

  it("frames nothing when the seat resolved nothing, and still opens the surface", () => {
    // The seat checked every term against the graph and none of them named
    // anything. A frame command with no subject would reset the camera to the
    // whole corpus while claiming to have narrowed it.
    const commands = commandsFromOp(
      request({ entities: [], unresolved: ["Atlantis", "Shangri-La"] }),
      "/chat",
    );
    expect(commands).toEqual([
      {
        dimension: "destination",
        path: "/geo",
        reason: "these 12 memories cluster on the Malabar coast",
        origin: "call-1",
      },
    ]);
  });

  it("does nothing at all when there is neither a destination nor an entity", () => {
    expect(commandsFromOp(request({ destination: null, entities: [] }), "/chat")).toEqual([]);
  });

  it("refuses a command that gives no account of itself", () => {
    // The tool requires a reason and the store renders it; an event without one
    // is a wire-level malformation. Moving the view on it would produce exactly
    // the unexplained lurch this whole mechanism exists to abolish.
    expect(commandsFromOp(request({ reason: "   " }), "/chat")).toEqual([]);
  });

  it("drops blank entity names rather than framing on an empty string", () => {
    const [cue] = commandsFromOp(request({ entities: ["Dali", "  ", ""] }), "/chat");
    expect(cue).toMatchObject({ entities: ["Dali"], text: "Dali" });
  });

  it("leaves a recall's commands with no reason, because nobody gave one", () => {
    // Inventing "because you asked about X" would put words in the model's
    // mouth that it never said, in a chip that quotes it.
    const commands = commandsFromOp(recall({ memories: [memory("m1")] }), "/chat");
    expect(commands.every((command) => command.reason === undefined)).toBe(true);
  });
});

describe("reasonOf", () => {
  it("finds the reason the commands share", () => {
    expect(reasonOf(commandsFromOp(request(), "/chat"))).toBe(
      "these 12 memories cluster on the Malabar coast",
    );
  });

  it("is null when nothing was justified, so the chip cannot quote an empty string", () => {
    expect(reasonOf(commandsFromOp(recall({ memories: [memory("m1")] }), "/chat"))).toBeNull();
    expect(reasonOf([{ dimension: "frame", entities: ["Dali"], reason: "  " }])).toBeNull();
    expect(reasonOf([])).toBeNull();
  });
});

describe("describeCommands — the ten surfaces", () => {
  it("names every destination the model can choose", () => {
    // The list used to hold one entry because one destination could be produced.
    // A surface the model can name but this table cannot would render as a raw
    // path in the offer the person is asked to accept.
    const nouns = [
      ["/", "the briefing"],
      ["/chat", "the conversation"],
      ["/recall", "recall"],
      ["/graph", "the graph"],
      ["/geo", "the map"],
      ["/anomalies", "anomalies"],
      ["/tasks", "tasks"],
      ["/history", "history"],
      ["/sources", "sources"],
      ["/providers", "providers"],
    ] as const;
    for (const [path, noun] of nouns) {
      expect(describeCommands([{ dimension: "destination", path }])).toBe(`open ${noun}`);
    }
  });

  it("falls back to the path for a surface added to the router but not to the table", () => {
    expect(describeCommands([{ dimension: "destination", path: "/newthing" }])).toBe(
      "open /newthing",
    );
  });
});
