import { describe, expect, it } from "vitest";
import type { ViewCommand } from "./commands";
import {
  arrived,
  axisStateLabel,
  returnTarget,
  traceKey,
  traceOf,
  viewDimensionLabel,
} from "./presence";

/**
 * The rules behind the live account of a move.
 *
 * What each of these is guarding is a specific false statement the block could
 * otherwise make on screen: one request's axes under another request's words, an
 * applied half eclipsed by a waiting one, a block that re-opens on a turn
 * boundary as if something had happened, and a "back" button that navigates
 * somewhere the person never was.
 */

const REASON = "these 12 memories cluster on the Malabar coast";
const OTHER = "the import you ran in March produced all four";

function offer(over: Partial<Extract<ViewCommand, { dimension: "destination" }>> = {}): ViewCommand {
  return { dimension: "destination", path: "/geo", from: "/chat", reason: REASON, origin: "call-1", ...over };
}

describe("traceOf", () => {
  it("shows an applied axis and a waiting axis together, under one reason", () => {
    const trace = traceOf(
      { reason: REASON, dimensions: ["cue", "frame", "focus"] },
      { destination: offer() },
    );
    expect(trace).toEqual({
      reason: REASON,
      axes: [
        { dimension: "cue", state: "applied" },
        { dimension: "frame", state: "applied" },
        { dimension: "destination", state: "waiting" },
        { dimension: "focus", state: "applied" },
      ],
    });
  });

  it("orders axes canonically rather than by how the store happens to hold them", () => {
    const trace = traceOf({ reason: REASON, dimensions: ["focus", "cue"] }, {});
    expect(trace?.axes.map((axis) => axis.dimension)).toEqual(["cue", "focus"]);
  });

  it("accounts for an applied move with no offer standing", () => {
    expect(traceOf({ reason: REASON, dimensions: ["destination"] }, {})).toEqual({
      reason: REASON,
      axes: [{ dimension: "destination", state: "applied" }],
    });
  });

  it("accounts for a wholly declined request", () => {
    expect(traceOf(null, { destination: offer() })).toEqual({
      reason: REASON,
      axes: [{ dimension: "destination", state: "waiting" }],
    });
  });

  it("lets the waiting reason outrank an applied one from a different request", () => {
    const trace = traceOf(
      { reason: OTHER, dimensions: ["cue"] },
      { destination: offer({ reason: REASON }) },
    );
    expect(trace?.reason).toBe(REASON);
    // The older request's applied cue is NOT listed under these words.
    expect(trace?.axes).toEqual([{ dimension: "destination", state: "waiting" }]);
  });

  it("is null when nothing carries a reason — the recall-derived path", () => {
    const cue: ViewCommand = { dimension: "cue", text: "HAL", entities: ["HAL"] };
    expect(traceOf(null, { cue })).toBeNull();
  });

  it("is null when a reason is present but blank", () => {
    expect(traceOf({ reason: "   ", dimensions: ["cue"] }, {})).toBeNull();
  });

  it("is null when a notice covers no axis", () => {
    expect(traceOf({ reason: REASON, dimensions: [] }, {})).toBeNull();
  });

  it("matches an untrimmed offer reason against the trimmed notice", () => {
    const trace = traceOf(
      { reason: REASON, dimensions: ["cue"] },
      { destination: offer({ reason: `  ${REASON}  ` }) },
    );
    expect(trace?.axes).toEqual([
      { dimension: "cue", state: "applied" },
      { dimension: "destination", state: "waiting" },
    ]);
  });
});

describe("traceKey", () => {
  it("separates two requests whose axes coincide", () => {
    const a = traceKey({ reason: REASON, axes: [{ dimension: "cue", state: "applied" }] });
    const b = traceKey({ reason: OTHER, axes: [{ dimension: "cue", state: "applied" }] });
    expect(a).not.toBe(b);
  });

  it("changes when an axis changes fate, so accepting an offer re-reads", () => {
    const waiting = traceKey({ reason: REASON, axes: [{ dimension: "destination", state: "waiting" }] });
    const applied = traceKey({ reason: REASON, axes: [{ dimension: "destination", state: "applied" }] });
    expect(waiting).not.toBe(applied);
  });

  it("changes when an axis is dropped, so an expired offer closes the block", () => {
    const both = traceKey({
      reason: REASON,
      axes: [
        { dimension: "cue", state: "applied" },
        { dimension: "destination", state: "waiting" },
      ],
    });
    const one = traceKey({ reason: REASON, axes: [{ dimension: "cue", state: "applied" }] });
    expect(both).not.toBe(one);
  });

  it("is stable for the same trace", () => {
    const trace = { reason: REASON, axes: [{ dimension: "cue" as const, state: "applied" as const }] };
    expect(traceKey(trace)).toBe(traceKey({ ...trace, axes: [...trace.axes] }));
  });
});

describe("arrived", () => {
  it("fires when a command applies", () => {
    expect(arrived({ seq: 3, offers: {} }, { seq: 4, offers: {} })).toBe(true);
  });

  it("fires when a command is held as an offer, which never touches the sequence", () => {
    expect(arrived({ seq: 3, offers: {} }, { seq: 3, offers: { destination: offer() } })).toBe(true);
  });

  it("fires when a later request replaces a standing offer on the same axis", () => {
    const previous = { seq: 3, offers: { destination: offer() } };
    const next = { seq: 3, offers: { destination: offer({ origin: "call-2", reason: OTHER }) } };
    expect(arrived(previous, next)).toBe(true);
  });

  it("does not fire when an offer expires at a turn boundary", () => {
    expect(arrived({ seq: 3, offers: { destination: offer() } }, { seq: 3, offers: {} })).toBe(false);
  });

  it("does not fire when an offer is refused", () => {
    const held = offer();
    expect(arrived({ seq: 7, offers: { destination: held } }, { seq: 7, offers: {} })).toBe(false);
  });

  it("does not fire when an unchanged offer is carried across a set", () => {
    const held = offer();
    expect(arrived({ seq: 3, offers: { destination: held } }, { seq: 3, offers: { destination: held } })).toBe(
      false,
    );
  });
});

describe("returnTarget", () => {
  it("offers the stage the move took the person off", () => {
    expect(returnTarget({ path: "/geo", from: "/chat" }, "/geo")).toBe("/chat");
  });

  it("offers nothing when no destination command has ever landed", () => {
    expect(returnTarget(null, "/geo")).toBeNull();
  });

  it("offers nothing for the return trip itself, so it cannot become a toggle", () => {
    expect(returnTarget({ path: "/chat", from: null }, "/chat")).toBeNull();
  });

  it("offers nothing once the person has navigated away themselves", () => {
    expect(returnTarget({ path: "/geo", from: "/chat" }, "/tasks")).toBeNull();
  });

  it("offers nothing when the way back is where they already are", () => {
    expect(returnTarget({ path: "/geo", from: "/geo" }, "/geo")).toBeNull();
  });
});

describe("words", () => {
  it("names each axis", () => {
    expect(viewDimensionLabel("cue")).toBe("the narrowing");
    expect(viewDimensionLabel("frame")).toBe("the camera");
    expect(viewDimensionLabel("destination")).toBe("the destination");
    expect(viewDimensionLabel("focus")).toBe("the opened entity");
  });

  it("passes an axis it does not recognise through verbatim", () => {
    expect(viewDimensionLabel("lens")).toBe("lens");
  });

  it("names the person as the one a held request is waiting on", () => {
    expect(axisStateLabel("waiting")).toBe("waiting on you");
    expect(axisStateLabel("applied")).toBe("applied");
  });
});
