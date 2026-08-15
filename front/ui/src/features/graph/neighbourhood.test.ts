import { describe, expect, it } from "vitest";
import { describeNeighbourhood, edgeInNeighbourhood, neighbourhood, rankSeeds } from "./neighbourhood";
import type { EntityNode, UniverseModel } from "./universe";

/** A chain a-b-c-d-e plus an isolate, with weightedDegree descending by
 *  position so ranking and ring-admission order are both checkable. */
function chain(n = 5, weights?: number[]): UniverseModel {
  const letters = "abcdefghij".slice(0, n).split("");
  const nodes: EntityNode[] = letters.map((id, i) => ({
    id,
    name: id.toUpperCase(),
    type: "Thing",
    salience: 0,
    mentions: 0,
    properNoun: true,
    degree: 0,
    weightedDegree: weights ? weights[i] : n - i,
    community: 0,
  }));
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const adjacency: Map<number, number>[] = nodes.map(() => new Map());
  for (let i = 0; i + 1 < n; i++) {
    adjacency[i].set(i + 1, 1);
    adjacency[i + 1].set(i, 1);
  }
  return {
    nodes,
    edges: [],
    index,
    adjacency,
    clusters: [],
    clusterLinks: [],
    totalEntities: n,
    totalConnections: n - 1,
    edgesDropped: 0,
  };
}

describe("rankSeeds", () => {
  it("takes the most connected first", () => {
    expect(rankSeeds(chain(5), 2)).toEqual(["a", "b"]);
  });

  it("breaks ties on id so the same corpus draws the same picture twice", () => {
    // Without a tie-break these resolve on array order, which is store order.
    const flat = chain(4, [1, 1, 1, 1]);
    const reversed = { ...flat, nodes: flat.nodes.slice().reverse() };
    expect(rankSeeds(flat, 2)).toEqual(["a", "b"]);
    expect(rankSeeds(reversed, 2)).toEqual(["a", "b"]);
  });

  it("survives k larger than the corpus, and k of zero", () => {
    expect(rankSeeds(chain(3), 99)).toEqual(["a", "b", "c"]);
    expect(rankSeeds(chain(3), 0)).toEqual([]);
  });
});

describe("neighbourhood", () => {
  it("expands exactly the requested number of hops", () => {
    const m = chain(5);
    expect([...neighbourhood(m, ["a"], { depth: 1 }).ids]).toEqual(["a", "b"]);
    expect([...neighbourhood(m, ["a"], { depth: 2 }).ids].sort()).toEqual(["a", "b", "c"]);
  });

  it("records hop distance from the nearest seed", () => {
    const h = neighbourhood(chain(5), ["a"], { depth: 2 });
    expect(h.hops.get("a")).toBe(0);
    expect(h.hops.get("b")).toBe(1);
    expect(h.hops.get("c")).toBe(2);
  });

  it("takes the nearest seed when two seeds reach the same node", () => {
    // c is 1 from d and 2 from a; the closer one must win.
    const h = neighbourhood(chain(5), ["a", "d"], { depth: 2 });
    expect(h.hops.get("d")).toBe(0);
    expect(h.hops.get("c")).toBe(1);
  });

  it("drops seeds the model does not contain rather than inventing them", () => {
    const h = neighbourhood(chain(3), ["a", "does-not-exist"], { depth: 1 });
    expect(h.seeds).toEqual(["a"]);
    expect(h.ids.has("does-not-exist")).toBe(false);
  });

  it("keeps the seeds when the cap bites, never orphaning what was asked about", () => {
    // This is the failure a naive slice produces: the cap lands mid-expansion
    // and the seed itself is what got sorted out.
    const h = neighbourhood(chain(5), ["e"], { depth: 3, maxNodes: 2 });
    expect(h.ids.has("e")).toBe(true);
    expect(h.ids.size).toBe(2);
  });

  it("truncates ring by ring, keeping the most connected within a partial ring", () => {
    const m = chain(5);
    // a admits, then ring 1 is {b}, ring 2 is {c}: cap of 2 cuts at hop 2.
    const h = neighbourhood(m, ["a"], { depth: 2, maxNodes: 2 });
    expect([...h.ids].sort()).toEqual(["a", "b"]);
    expect(h.truncatedAtHop).toBe(2);
    expect(h.dropped).toBe(1);
  });

  it("reports no truncation when everything reachable fits", () => {
    const h = neighbourhood(chain(3), ["a"], { depth: 5, maxNodes: 100 });
    expect(h.dropped).toBe(0);
    expect(h.truncatedAtHop).toBeNull();
  });

  it("stops when the graph runs out before the depth does", () => {
    const h = neighbourhood(chain(3), ["a"], { depth: 10 });
    expect(h.ids.size).toBe(3);
    expect(h.truncatedAtHop).toBeNull();
  });

  it("returns empty for no seeds, a zero cap, or negative depth", () => {
    const m = chain(4);
    expect(neighbourhood(m, [], { depth: 2 }).ids.size).toBe(0);
    expect(neighbourhood(m, ["a"], { maxNodes: 0 }).ids.size).toBe(0);
    expect(neighbourhood(m, ["a"], { depth: -1 }).ids.size).toBe(0);
  });

  it("is deterministic under a partial ring", () => {
    // Equal weights force the id tie-break to decide who survives the cap.
    const m = chain(5, [1, 1, 1, 1, 1]);
    const a = neighbourhood(m, ["c"], { depth: 1, maxNodes: 2 });
    const b = neighbourhood(m, ["c"], { depth: 1, maxNodes: 2 });
    expect([...a.ids]).toEqual([...b.ids]);
  });
});

describe("edgeInNeighbourhood", () => {
  it("requires both endpoints, so no edge is drawn to nothing", () => {
    const h = neighbourhood(chain(5), ["a"], { depth: 1 });
    expect(edgeInNeighbourhood({ source: "a", target: "b" }, h)).toBe(true);
    expect(edgeInNeighbourhood({ source: "b", target: "c" }, h)).toBe(false);
  });
});

describe("describeNeighbourhood", () => {
  it("states the shape without mentioning a cut that did not happen", () => {
    const h = neighbourhood(chain(3), ["a"], { depth: 2, maxNodes: 100 });
    expect(describeNeighbourhood(h, 2)).toBe("3 entities within 2 hops of 1 seed");
  });

  it("says so when the picture is a sample rather than the whole answer", () => {
    const h = neighbourhood(chain(5), ["a"], { depth: 2, maxNodes: 2 });
    expect(describeNeighbourhood(h, 2)).toContain("1 more reachable, not drawn");
  });
});
