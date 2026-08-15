import { describe, expect, it } from "vitest";
import { RAIL_OFFSET, RAIL_WIDTH_PX } from "@/components/layout/destinations";
import {
  ROOT,
  backHref,
  hrefFor,
  paneByPath,
  parseTrail,
  promoteHref,
  promoteTrail,
  railHref,
} from "./trail";

/** Trails read as the ids a person would see spines for, briefing first. */
const ids = (trail: readonly { id: string }[]) => trail.map((p) => p.id);

const trailOf = (...paths: string[]) =>
  paths.map((p) => {
    const pane = paneByPath(p);
    if (!pane) throw new Error(`no pane for ${p}`);
    return pane;
  });

describe("parseTrail", () => {
  it("roots every trail at the briefing, so no screen is a dead end", () => {
    expect(ids(parseTrail("/", ""))).toEqual(["briefing"]);
    expect(ids(parseTrail("/recall", ""))).toEqual(["briefing", "recall"]);
    expect(ids(parseTrail("/tasks", "?via="))).toEqual(["briefing", "tasks"]);
  });

  it("reads the ancestors out of ?via=", () => {
    expect(ids(parseTrail("/graph", "?via=recall"))).toEqual([
      "briefing",
      "recall",
      "graph",
    ]);
    expect(ids(parseTrail("/chat", "?via=recall,graph"))).toEqual([
      "briefing",
      "recall",
      "graph",
      "chat",
    ]);
  });

  it("ignores ?via= on the briefing — it is always the base, never a descendant", () => {
    expect(ids(parseTrail("/", "?via=recall,graph"))).toEqual(["briefing"]);
  });

  it("drops ids it does not recognise rather than drawing a nameless spine", () => {
    expect(ids(parseTrail("/graph", "?via=recall,nonsense,"))).toEqual([
      "briefing",
      "recall",
      "graph",
    ]);
  });

  it("drops duplicates, the briefing and the primary out of ?via=", () => {
    // Two spines with the same title are indistinguishable to a reader.
    expect(ids(parseTrail("/graph", "?via=recall,recall"))).toEqual([
      "briefing",
      "recall",
      "graph",
    ]);
    expect(ids(parseTrail("/graph", "?via=briefing,graph,recall"))).toEqual([
      "briefing",
      "recall",
      "graph",
    ]);
  });

  it("yields the briefing alone for a path the app does not have", () => {
    // The router sends unknown hashes home; the trail agrees with it instead
    // of drawing a spine for a pane that is not on screen.
    expect(ids(parseTrail("/does-not-exist", "?via=recall"))).toEqual(["briefing"]);
  });
});

describe("hrefFor", () => {
  it("is bookmarkable: the link it makes parses back to the same trail", () => {
    const trail = trailOf("/", "/recall", "/graph", "/chat");
    for (let i = 0; i < trail.length; i += 1) {
      const href = hrefFor(trail, i);
      const [pathname, search] = href.split("?");
      expect(ids(parseTrail(pathname, search ? `?${search}` : ""))).toEqual(
        ids(trail.slice(0, i + 1)),
      );
    }
  });

  it("always states the ancestry, even when it is empty", () => {
    // An ABSENT via means "open this from where I am"; an EMPTY one means
    // "this has no ancestors". A link that dropped the parameter would be
    // re-read as a promotion the next time it was followed from elsewhere.
    expect(hrefFor(trailOf("/", "/recall"), 1)).toBe("/recall?via=");
    expect(hrefFor(trailOf("/", "/recall", "/graph"), 2)).toBe("/graph?via=recall");
  });

  it("goes to the bare root for the briefing, whatever is downstream", () => {
    expect(hrefFor(trailOf("/", "/recall", "/graph"), 0)).toBe("/");
  });

  it("discards everything downstream — a spine click returns, it does not keep tabs", () => {
    const trail = trailOf("/", "/recall", "/graph", "/chat");
    expect(hrefFor(trail, 1)).toBe("/recall?via=");
    expect(ids(parseTrail("/recall", "?via="))).toEqual(["briefing", "recall"]);
  });

  it("survives an index that is not in the trail", () => {
    expect(hrefFor(trailOf("/", "/recall"), 9)).toBe("/");
    expect(hrefFor([], 0)).toBe("/");
  });
});

describe("promoteTrail", () => {
  it("appends to the pane it was opened from", () => {
    const trail = trailOf("/", "/recall");
    expect(ids(promoteTrail(trail, 1, "graph"))).toEqual([
      "briefing",
      "recall",
      "graph",
    ]);
  });

  it("truncates everything downstream of the pane it was opened from", () => {
    // The rule that makes this a trail and not a tab bar.
    const trail = trailOf("/", "/recall", "/graph", "/chat");
    expect(ids(promoteTrail(trail, 1, "geo"))).toEqual(["briefing", "recall", "geo"]);
    expect(ids(promoteTrail(trail, 0, "geo"))).toEqual(["briefing", "geo"]);
  });

  it("goes to a pane already in the trail rather than opening a second copy", () => {
    const trail = trailOf("/", "/recall", "/graph");
    expect(ids(promoteTrail(trail, 2, "recall"))).toEqual(["briefing", "recall"]);
  });

  it("treats opening the briefing as a reset", () => {
    const trail = trailOf("/", "/recall", "/graph");
    expect(ids(promoteTrail(trail, 2, "briefing"))).toEqual(["briefing"]);
  });

  it("leaves you where you are when the target is not a pane", () => {
    const trail = trailOf("/", "/recall", "/graph");
    expect(ids(promoteTrail(trail, 1, "nonsense"))).toEqual(["briefing", "recall"]);
  });

  it("clamps an out-of-range origin instead of producing a hole", () => {
    const trail = trailOf("/", "/recall");
    expect(ids(promoteTrail(trail, 47, "graph"))).toEqual([
      "briefing",
      "recall",
      "graph",
    ]);
    expect(ids(promoteTrail(trail, -3, "graph"))).toEqual(["briefing", "graph"]);
  });

  it("never holds the same pane twice, however deep the walk", () => {
    let trail = [ROOT];
    for (const id of ["recall", "graph", "recall", "chat", "graph", "recall"]) {
      trail = promoteTrail(trail, trail.length - 1, id);
      expect(new Set(ids(trail)).size).toBe(trail.length);
    }
  });
});

describe("promoteHref", () => {
  it("is the link for the promoted trail", () => {
    expect(promoteHref(trailOf("/"), 0, "recall")).toBe("/recall?via=");
    expect(promoteHref(trailOf("/", "/recall"), 1, "graph")).toBe("/graph?via=recall");
    expect(promoteHref(trailOf("/", "/recall", "/graph"), 2, "recall")).toBe(
      "/recall?via=",
    );
  });
});

describe("railHref", () => {
  it("states an empty ancestry, so a rail click resets the trail", () => {
    expect(railHref("/recall")).toBe("/recall?via=");
    expect(railHref("/")).toBe("/");
    expect(ids(parseTrail("/recall", "?via="))).toEqual(["briefing", "recall"]);
  });
});

describe("backHref", () => {
  it("goes one level, and one level only", () => {
    expect(backHref(trailOf("/", "/recall", "/graph"))).toBe("/recall?via=");
    expect(backHref(trailOf("/", "/recall"))).toBe("/");
  });

  it("is null at the briefing, so Escape there does nothing at all", () => {
    // Not "/" — re-navigating to the screen already on display would push a
    // history entry and reset the briefing's scroll for no movement.
    expect(backHref(trailOf("/"))).toBeNull();
    expect(backHref([])).toBeNull();
  });

  it("walks the whole way home one press at a time", () => {
    let trail = trailOf("/", "/recall", "/graph", "/chat");
    const seen: string[] = [];
    for (;;) {
      const href = backHref(trail);
      if (!href) break;
      seen.push(href);
      const [pathname, search] = href.split("?");
      trail = parseTrail(pathname, search ? `?${search}` : "");
    }
    expect(seen).toEqual(["/graph?via=recall", "/recall?via=", "/"]);
    expect(ids(trail)).toEqual(["briefing"]);
  });
});

describe("the rail's width", () => {
  it("is the same number in the offset utility as in the element", () => {
    // Tailwind v4 emits utilities by scanning source text, so the padding has
    // to be written out and cannot be interpolated from the number. This is
    // the check that makes writing it out safe: get them out of step and the
    // rail sits on top of the stage in the built product only.
    expect(RAIL_OFFSET).toBe(`pl-[${RAIL_WIDTH_PX}px]`);
  });
});
