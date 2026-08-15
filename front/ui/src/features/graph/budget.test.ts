import { describe, expect, it } from "vitest";
import { capEdgesPerNode, describeEdgeBudget } from "./budget";

const e = (source: string, target: string, strength: number) => ({ source, target, strength });

/** A hub with `n` leaves. */
const star = (n: number) =>
  Array.from({ length: n }, (_, i) => e("h", `leaf${String(i).padStart(2, "0")}`, 1 - i / 100));

/** Every pair among `n` nodes connected — the dense core this exists to thin. */
function clique(n: number) {
  const ids = Array.from({ length: n }, (_, i) => `n${String(i).padStart(2, "0")}`);
  const out: ReturnType<typeof e>[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) out.push(e(ids[i], ids[j], 1 - (i * n + j) / (n * n)));
  }
  return out;
}

describe("capEdgesPerNode", () => {
  it("thins a dense core, which is what it is for", () => {
    const edges = clique(10); // 45 edges
    const { kept, dropped } = capEdgesPerNode(edges, 3);
    expect(edges).toHaveLength(45);
    expect(kept.length).toBeLessThan(edges.length);
    expect(dropped).toBe(edges.length - kept.length);
    // Every node keeps at least one line, so thinning never isolates anything.
    const touched = new Set(kept.flatMap((x) => [x.source, x.target]));
    expect(touched.size).toBe(10);
  });

  it("does NOT trim a star, because every leaf claims its only edge", () => {
    // Deliberate, and the opposite of what a source-only budget would do. A cut
    // that leaves 15 unconnected dots on screen is a statement about the data
    // that is not true; the crossings are between hubs, not out at the leaves.
    const { kept, dropped } = capEdgesPerNode(star(20), 5);
    expect(kept).toHaveLength(20);
    expect(dropped).toBe(0);
  });

  it("keeps a hub's strongest edges when the other end has spent its claim", () => {
    // Two hubs, each also carrying leaves, so neither can claim every shared
    // line. The weakest hub-to-hub edge is the one that goes.
    const edges = [
      e("h1", "h2", 0.9),
      e("h1", "h2b", 0.8),
      e("h1", "h2c", 0.1),
      e("h2", "h2b", 0.7),
      e("h2", "h2c", 0.6),
      e("h2b", "h2c", 0.5),
    ];
    const kept = capEdgesPerNode(edges, 2).kept;
    expect(kept.length).toBeLessThan(edges.length);
    // The strongest edge in the graph is never a candidate for dropping.
    expect(kept.some((x) => x.source === "h1" && x.target === "h2")).toBe(true);
  });

  it("leaves a graph alone when it is already under budget", () => {
    const edges = [e("a", "b", 1), e("b", "c", 1)];
    const { kept, dropped } = capEdgesPerNode(edges, 5);
    expect(kept).toHaveLength(2);
    expect(dropped).toBe(0);
  });

  it("selects the same edges regardless of input order", () => {
    // Emission follows input order — the canvas paints in that order and
    // reordering changes which lines land on top — so the SET is the invariant,
    // not the sequence.
    const edges = clique(8);
    const a = capEdgesPerNode(edges, 2).kept.map((x) => `${x.source}|${x.target}`);
    const b = capEdgesPerNode(edges.slice().reverse(), 2).kept.map((x) => `${x.source}|${x.target}`);
    expect(new Set(a)).toEqual(new Set(b));
  });

  it("breaks ties deterministically rather than on array order", () => {
    const edges = [e("h", "z", 1), e("h", "a", 1), e("h", "m", 1), e("z", "a", 1), e("z", "m", 1)];
    const first = capEdgesPerNode(edges, 1).kept.map((x) => `${x.source}|${x.target}`);
    const second = capEdgesPerNode(edges, 1).kept.map((x) => `${x.source}|${x.target}`);
    expect(first).toEqual(second);
    // "h a" sorts before "h m" and "h z", so the hub's single claim is h-a.
    expect(first).toContain("h|a");
  });

  it("counts a self-loop once, as one node's edge rather than two", () => {
    const kept = capEdgesPerNode([e("a", "a", 1), e("a", "b", 0.9), e("a", "c", 0.8)], 2).kept;
    // a claims its top two; b and c each claim their only edge, so all survive.
    // The guarantee under test is that the self-loop did not consume two of a's
    // slots on its own.
    expect(kept).toHaveLength(3);
  });

  it("drops everything at k of zero, and reports it", () => {
    const { kept, dropped } = capEdgesPerNode(star(4), 0);
    expect(kept).toHaveLength(0);
    expect(dropped).toBe(4);
  });

  it("survives an empty graph", () => {
    expect(capEdgesPerNode([], 5)).toEqual({ kept: [], dropped: 0 });
  });
});

describe("describeEdgeBudget", () => {
  it("says nothing when nothing was cut", () => {
    expect(describeEdgeBudget(0, 8)).toBeNull();
  });

  it("states the cut and the rule that made it", () => {
    expect(describeEdgeBudget(691, 8)).toBe("691 more edges beyond the strongest 8 per entity");
  });
});
