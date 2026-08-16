import { describe, expect, it } from "vitest";
import { VIEW_DIMENSIONS, decide, holdsAt, type ViewDimension } from "./authority";

describe("decide", () => {
  it("always applies a user command — this rule protects the person, not the model", () => {
    for (const dimension of VIEW_DIMENSIONS) {
      expect(decide("user", dimension, VIEW_DIMENSIONS)).toBe("apply");
    }
  });

  it("applies an agent command while the human has not touched that dimension", () => {
    for (const dimension of VIEW_DIMENSIONS) {
      expect(decide("agent", dimension, [])).toBe("apply");
    }
  });

  it("offers rather than applies once the human has touched that dimension", () => {
    expect(decide("agent", "cue", ["cue"])).toBe("offer");
    expect(decide("agent", "frame", ["frame"])).toBe("offer");
    expect(decide("agent", "destination", ["destination"])).toBe("offer");
  });

  it("tracks dimensions independently: a hand on the camera does not mute the rest", () => {
    // The failure this prevents: one "the user is driving" flag, so framing the
    // graph by hand also stops the model opening a different destination.
    const claimed: ViewDimension[] = ["frame"];
    expect(decide("agent", "frame", claimed)).toBe("offer");
    expect(decide("agent", "destination", claimed)).toBe("apply");
    expect(decide("agent", "cue", claimed)).toBe("apply");
  });
});

describe("holdsAt", () => {
  it("holds the destination on /chat, where moving would take the answer off screen", () => {
    expect(holdsAt("/chat")).toEqual(["destination"]);
    expect(decide("agent", "destination", holdsAt("/chat"))).toBe("offer");
  });

  it("holds nothing anywhere else — every other stage can be moved to", () => {
    for (const path of ["/", "/graph", "/geo", "/recall", "/anomalies", "/tasks"]) {
      expect(holdsAt(path)).toEqual([]);
    }
  });

  it("still lets the model narrow the cue on /chat, only not move you", () => {
    expect(decide("agent", "cue", holdsAt("/chat"))).toBe("apply");
    expect(decide("agent", "frame", holdsAt("/chat"))).toBe("apply");
  });
});
