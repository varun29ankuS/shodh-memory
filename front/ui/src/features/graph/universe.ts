import type { MemoryUniverse, UniverseStar, GravitationalConnection } from "@/lib/api/graph";

/**
 * Turning the raw universe into something a screen can hold.
 *
 * Ported from the vanilla dashboard (front/index.html:763-891), which is the
 * only implementation of this that has been run against a real corpus. The
 * numbers in here are not taste — each was set against the live GDELT/defence
 * graphs and the comments record what went wrong at the settings either side.
 *
 * That file has since been deleted, so every front/index.html line number in
 * this module resolves in git history rather than in the tree. The citations
 * stay because they are where these constants came from, and a constant with
 * no provenance is one nobody can safely change.
 *
 * The problem being solved is scale, and it is real: `get_universe` applies no
 * cap (src/graph_memory.rs:7104-7220), so this can be 5k entities and 126k
 * edges. The old file measured 126k raw edges freezing the client build for
 * >75s (front/index.html:728-735). Nothing here is premature.
 *
 * Everything in this module is pure — no React, no canvas, no DOM. That is
 * deliberate: it is the part with actual algorithmic content, and keeping it
 * separable is what makes it checkable.
 */

/** Generic bulk relations. These accumulate enormous strength and would
 *  otherwise mask the rare typed edges (Causes, LocatedIn, Struck) that carry
 *  the signal — front/index.html:772-775. */
const GENERIC_RELATION = /^(co.?occurs|co.?retrieved|related.?to|associated.?with)$/i;

/**
 * Keep the strongest edges up to a budget. Measured: 126k raw edges froze the
 * client-side build for >75s; 40k builds in seconds and the strongest-edge
 * subset preserves cluster structure (front/index.html:728-748).
 *
 * The cut is CLASS-AWARE, not a pure strength sort. Typed and causal edges are
 * the product's signal and are rare; a pure strength cut deletes them en masse
 * because generic co-occurrence wins on bulk. Every non-generic edge is kept
 * and the remainder is filled with the strongest generics.
 */
const EDGE_BUDGET = 40_000;

/** Below this many entities, render the actual entity constellation rather than
 *  cluster bubbles. Cluster abstraction only earns its keep at scale; a demo or
 *  young corpus reads far better as itself (front/index.html:905-913). */
export const SMALL_GRAPH_MAX = 700;

/** Louvain resolution. Higher = more, smaller communities. */
const RESOLUTION = 1.6;

/** Hard cap on overview bubbles: the overview must stay readable whatever the
 *  community structure looks like (front/index.html:836-840). */
const MAX_OVERVIEW = 60;

/**
 * Minimum weight for a GENERIC co-occurrence edge to be drawn.
 *
 * Co-occurrence is produced pairwise, so a memory mentioning n entities emits
 * ~n²/2 of these. On the verification corpus that is 416 CoOccurs against 5
 * typed relations — the typed edges, which are the entire point, are one
 * percent of the ink. Below this weight a co-occurrence says little more than
 * "these appeared in the same paragraph once".
 *
 * Typed relations are NEVER floored, however weak: a LocatedIn at 0.1 is still
 * a claim about the world, and the whole argument for this product is that
 * those claims are the signal. The floor applies to bulk only, and the number
 * hidden is stated on screen — a silent cut would misreport the corpus.
 */
/**
 * How many generic edges to draw, as a multiple of the node count.
 *
 * Deliberately a BUDGET, not a fixed weight threshold. A constant floor cannot
 * work across corpora: on the verification corpus a 0.5 cut hid 407 of 416
 * pairs and rendered a dense graph as nine lines, which misreports the data as
 * badly as drawing the hairball does. Edge weights are not calibrated to any
 * absolute scale — they depend on extraction density and decay — so the only
 * stable question is "how much tissue can this many nodes carry before it stops
 * being readable", and that scales with N.
 *
 * ~2 generic edges per node keeps the graph connected enough to show structure
 * while leaving the typed relations visible on top of it.
 */
const GENERIC_PER_NODE = 2;

/** Strength at or above which a generic edge is drawn, derived from the corpus
 *  rather than assumed. Returns 0 when everything fits, so nothing is hidden
 *  on a small graph. */
export function cooccurFloor(model: UniverseModel): number {
  const budget = Math.max(20, Math.round(model.nodes.length * GENERIC_PER_NODE));
  const generic = model.edges.filter((e) => e.generic);
  if (generic.length <= budget) return 0;
  const sorted = generic.map((e) => e.strength).sort((a, b) => b - a);
  return sorted[budget - 1];
}

/** Whether an edge is drawn at all. Shared by the canvas and the footer count
 *  so the two cannot disagree about what was hidden. Typed relations are never
 *  floored, however weak — a LocatedIn at 0.1 is still a claim about the world,
 *  and those claims are the entire argument for this product. */
export function isEdgeRendered(edge: EntityEdge, floor: number): boolean {
  return !edge.generic || edge.strength >= floor;
}

export interface EntityNode {
  id: string;
  name: string;
  /** `entity_type` with `null` normalised to "Unlabelled" so grouping and the
   *  legend never have to special-case it. */
  type: string;
  salience: number;
  mentions: number;
  properNoun: boolean;
  /** Neighbour count after the edge budget. */
  degree: number;
  /** Summed edge strength. */
  weightedDegree: number;
  /** Community index, assigned by `clusterUniverse`. */
  community: number;
}

export type EdgeTier = "L1Working" | "L2Episodic" | "L3Semantic";

export interface EntityEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  strength: number;
  generic: boolean;
  /** Consolidation tier of the underlying edge. When a pair carries several
   *  relations the most consolidated one wins, so the drawn edge never
   *  understates how settled the connection is. */
  tier: EdgeTier;
}

/** L1 → L2 → L3 as an order, for "keep the most consolidated" comparisons. */
const TIER_RANK: Record<EdgeTier, number> = { L1Working: 0, L2Episodic: 1, L3Semantic: 2 };

export function tierToken(tier: EdgeTier): string {
  return tier === "L3Semantic" ? "--edge-l3" : tier === "L2Episodic" ? "--edge-l2" : "--edge-l1";
}

export const TIER_LABEL: Record<EdgeTier, string> = {
  L1Working: "L1 working",
  L2Episodic: "L2 episodic",
  L3Semantic: "L3 semantic",
};

export const TIER_ORDER: EdgeTier[] = ["L1Working", "L2Episodic", "L3Semantic"];

export interface Cluster {
  id: number;
  /** Indices into `nodes`. */
  members: number[];
  size: number;
  label: string;
  /** True when this bucket is mostly folded-in long tail, in which case naming
   *  it after one member would mislead (front/index.html:874-878). `label`
   *  already reflects this — read `label`, not this flag, when rendering text;
   *  the flag is for styling (muted colour). */
  longTail: boolean;
  dominantType: string;
}

export interface ClusterLink {
  source: number;
  target: number;
  weight: number;
  count: number;
}

export interface UniverseModel {
  nodes: EntityNode[];
  edges: EntityEdge[];
  /** node id → index in `nodes`. */
  index: Map<string, number>;
  /** Adjacency: per node index, neighbour index → strongest edge strength. */
  adjacency: Map<number, number>[];
  clusters: Cluster[];
  clusterLinks: ClusterLink[];
  /** What the server said it holds, before any client budgeting. */
  totalEntities: number;
  totalConnections: number;
  /** How many edges the budget dropped. Surfaced in the UI — a silent cut is
   *  the thing this whole module exists to avoid pretending about. */
  edgesDropped: number;
}

/** Best relation label for a pair, preferring typed semantics over generic
 *  bulk — front/index.html:781-784. */
function betterRelation(current: string | undefined, candidate: string): string {
  if (current === undefined) return candidate;
  if (GENERIC_RELATION.test(current) && !GENERIC_RELATION.test(candidate)) return candidate;
  return current;
}

function applyEdgeBudget(connections: GravitationalConnection[]): {
  kept: GravitationalConnection[];
  dropped: number;
} {
  if (connections.length <= EDGE_BUDGET) return { kept: connections, dropped: 0 };
  const typed: GravitationalConnection[] = [];
  const generic: GravitationalConnection[] = [];
  for (const c of connections) {
    (GENERIC_RELATION.test(c.relation_type) ? generic : typed).push(c);
  }
  generic.sort((a, b) => (b.strength ?? 0) - (a.strength ?? 0));
  const kept = typed.concat(generic.slice(0, Math.max(0, EDGE_BUDGET - typed.length)));
  return { kept, dropped: connections.length - kept.length };
}

export function buildUniverse(universe: MemoryUniverse): UniverseModel {
  const stars: UniverseStar[] = universe.stars ?? [];
  const { kept, dropped } = applyEdgeBudget(universe.connections ?? []);

  const index = new Map<string, number>();
  stars.forEach((s, i) => index.set(s.id, i));

  const nodes: EntityNode[] = stars.map((s) => ({
    id: s.id,
    name: s.name || "(unnamed)",
    type: s.entity_type ?? "Unlabelled",
    salience: s.salience ?? 0,
    mentions: s.mention_count ?? 0,
    properNoun: !!s.is_proper_noun,
    degree: 0,
    weightedDegree: 0,
    community: -1,
  }));

  const adjacency: Map<number, number>[] = Array.from({ length: nodes.length }, () => new Map());
  const relationOf = new Map<string, string>();
  const strengthOf = new Map<string, number>();
  const edgeIdOf = new Map<string, string>();
  const tierOf = new Map<string, EdgeTier>();

  for (const c of kept) {
    const a = index.get(c.from_id);
    const b = index.get(c.to_id);
    // Self-loops and edges to entities not in `stars` carry no layout meaning.
    if (a === undefined || b === undefined || a === b) continue;
    const w = typeof c.strength === "number" ? c.strength : 0.5;
    if ((adjacency[a].get(b) ?? 0) < w) {
      adjacency[a].set(b, w);
      adjacency[b].set(a, w);
    }
    const key = a < b ? `${a}|${b}` : `${b}|${a}`;
    relationOf.set(key, betterRelation(relationOf.get(key), c.relation_type));
    strengthOf.set(key, Math.max(strengthOf.get(key) ?? 0, w));
    if (!edgeIdOf.has(key)) edgeIdOf.set(key, c.id);
    const tier = (c.tier ?? "L1Working") as EdgeTier;
    const prevTier = tierOf.get(key);
    if (prevTier === undefined || TIER_RANK[tier] > TIER_RANK[prevTier]) tierOf.set(key, tier);
  }

  for (let i = 0; i < nodes.length; i++) {
    nodes[i].degree = adjacency[i].size;
    let sum = 0;
    for (const [, w] of adjacency[i]) sum += w;
    nodes[i].weightedDegree = sum;
  }

  // One edge per PAIR, carrying the best relation label for it. The raw payload
  // can hold several relations between the same two entities; drawing all of
  // them stacks identical lines and makes the strongest look no different.
  const edges: EntityEdge[] = [];
  for (const [key, relation] of relationOf) {
    const [a, b] = key.split("|").map(Number);
    edges.push({
      id: edgeIdOf.get(key) ?? key,
      source: nodes[a].id,
      target: nodes[b].id,
      relation,
      strength: strengthOf.get(key) ?? 0.5,
      generic: GENERIC_RELATION.test(relation),
      tier: tierOf.get(key) ?? "L1Working",
    });
  }

  const model: UniverseModel = {
    nodes,
    edges,
    index,
    adjacency,
    clusters: [],
    clusterLinks: [],
    totalEntities: universe.total_entities ?? stars.length,
    totalConnections: universe.total_connections ?? (universe.connections ?? []).length,
    edgesDropped: dropped,
  };

  clusterUniverse(model);
  return model;
}

/**
 * Louvain community detection, one level of local moving.
 *
 * Ported from front/index.html:794-891, where it was measured at "well under
 * 100ms on 87k edges, yielding 20-30 communities at resolution 1.6".
 */
export function clusterUniverse(model: UniverseModel): void {
  const { nodes, adjacency } = model;
  const N = nodes.length;
  if (N === 0) return;

  const k = new Float64Array(N);
  let m = 0;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (const [j, w] of adjacency[i]) {
      s += w;
      if (i < j) m += w;
    }
    k[i] = s;
  }
  const twoM = 2 * (m || 1);

  const comm = new Int32Array(N);
  for (let i = 0; i < N; i++) comm[i] = i;
  const sigmaTot = new Float64Array(N);
  for (let i = 0; i < N; i++) sigmaTot[i] = k[i];

  let improved = true;
  let passes = 0;
  while (improved && passes < 30) {
    improved = false;
    passes++;
    for (let i = 0; i < N; i++) {
      const ci = comm[i];
      sigmaTot[ci] -= k[i];
      const wc = new Map<number, number>();
      wc.set(ci, 0);
      for (const [nb, w] of adjacency[i]) {
        const cn = comm[nb];
        wc.set(cn, (wc.get(cn) ?? 0) + w);
      }
      let best = ci;
      let bestGain = -Infinity;
      for (const [c, wic] of wc) {
        const gain = wic - (RESOLUTION * sigmaTot[c] * k[i]) / twoM;
        if (gain > bestGain) {
          bestGain = gain;
          best = c;
        }
      }
      sigmaTot[best] += k[i];
      comm[i] = best;
      if (best !== ci) improved = true;
    }
  }

  // Absorb micro-clusters. Communities of a few members are visual noise — lone
  // satellites that convey no structure. MIN_CLUSTER scales with corpus size:
  // on a 5k-entity news graph a fixed 3 left thousands of 4-10-member
  // communities alive, a 3,157-label wordball that hung the tab
  // (front/index.html:817-822).
  const MIN_CLUSTER = Math.max(3, Math.ceil(N / 400));
  const csize = new Map<number, number>();
  for (let i = 0; i < N; i++) csize.set(comm[i], (csize.get(comm[i]) ?? 0) + 1);

  for (let pass = 0; pass < 6; pass++) {
    let moved = false;
    for (let i = 0; i < N; i++) {
      if ((csize.get(comm[i]) ?? 0) > MIN_CLUSTER) continue;
      const wc = new Map<number, number>();
      for (const [nb, w] of adjacency[i]) {
        const cn = comm[nb];
        if (cn === comm[i]) continue;
        wc.set(cn, (wc.get(cn) ?? 0) + w);
      }
      let best: number | null = null;
      let bw = -Infinity;
      for (const [c, w] of wc) {
        if ((csize.get(c) ?? 0) > MIN_CLUSTER && w > bw) {
          bw = w;
          best = c;
        }
      }
      if (best !== null && best !== comm[i]) {
        csize.set(comm[i], (csize.get(comm[i]) ?? 0) - 1);
        csize.set(best, (csize.get(best) ?? 0) + 1);
        comm[i] = best;
        moved = true;
      }
    }
    if (!moved) break;
  }

  // Hard cap on overview bubbles. Keep the biggest MAX_OVERVIEW communities and
  // fold every smaller one into the kept community it shares the most edge
  // weight with.
  const foldedInto = new Map<number, number>();
  {
    const ids = [...csize.entries()]
      .filter(([, s]) => s > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([c]) => c);
    if (ids.length > MAX_OVERVIEW) {
      const keep = new Set(ids.slice(0, MAX_OVERVIEW));
      const flow = new Map<number, Map<number, number>>();
      for (let i = 0; i < N; i++) {
        const ca = comm[i];
        if (keep.has(ca)) continue;
        let f = flow.get(ca);
        if (!f) {
          f = new Map();
          flow.set(ca, f);
        }
        for (const [nb, w] of adjacency[i]) {
          const cb = comm[nb];
          if (keep.has(cb)) f.set(cb, (f.get(cb) ?? 0) + w);
        }
      }
      const fallback = ids[MAX_OVERVIEW - 1];
      const dest = new Map<number, number>();
      for (const [cid, f] of flow) {
        let best = fallback;
        let bw = -Infinity;
        for (const [c, w] of f) {
          if (w > bw) {
            bw = w;
            best = c;
          }
        }
        dest.set(cid, best);
      }
      for (let i = 0; i < N; i++) {
        const c = comm[i];
        if (!keep.has(c)) {
          const d = dest.get(c) ?? fallback;
          comm[i] = d;
          foldedInto.set(d, (foldedInto.get(d) ?? 0) + 1);
        }
      }
    }
  }

  // Compact community ids and build metadata.
  const remap = new Map<number, number>();
  const members: number[][] = [];
  const originalComm: number[] = [];
  for (let i = 0; i < N; i++) {
    const cid = comm[i];
    if (!remap.has(cid)) {
      remap.set(cid, members.length);
      members.push([]);
      originalComm.push(cid);
    }
    const m2 = remap.get(cid)!;
    nodes[i].community = m2;
    members[m2].push(i);
  }

  model.clusters = members.map((mem, cid) => {
    // Label: prefer a salient, proper-noun, high-degree entity. Document
    // filenames are poor labels so they rank last (front/index.html:862-870).
    let best = mem[0];
    let bestScore = -Infinity;
    const typeCount = new Map<string, number>();
    for (const i of mem) {
      const n = nodes[i];
      typeCount.set(n.type, (typeCount.get(n.type) ?? 0) + 1);
      const isDoc = /\.(txt|md|pdf|json|csv)$/i.test(n.name);
      const score = n.degree * (isDoc ? 0.15 : 1) * (n.properNoun ? 1.3 : 1) * (1 + n.salience);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    let dominantType = "Unlabelled";
    let domN = -1;
    for (const [t, c] of typeCount) {
      if (c > domN) {
        domN = c;
        dominantType = t;
      }
    }
    const folded = foldedInto.get(originalComm[cid]) ?? 0;
    const longTail = folded > mem.length * 0.5;
    // A mixed bucket must not be named after one member — but it must still be
    // named after something real. The placeholder that used to sit here read
    // "long tail", which on screen is indistinguishable from an entity called
    // "long tail": it is internal vocabulary for *why we could not name it*,
    // shown where the reader expects *what is in it*. What is in it is the
    // dominant type, so say that. `dominantType` is "Unlabelled" only when the
    // members carry no type at all, and then the best member's name is still a
    // better answer than jargon.
    const label = longTail
      ? dominantType !== "Unlabelled"
        ? `mostly ${dominantType}`
        : nodes[best].name
      : nodes[best].name;
    return {
      id: cid,
      members: mem,
      size: mem.length,
      label,
      longTail,
      dominantType,
    };
  });

  // Aggregate inter-cluster edge weights for the overview.
  const inter = new Map<string, { w: number; n: number }>();
  for (let i = 0; i < N; i++) {
    const ca = nodes[i].community;
    for (const [j, w] of adjacency[i]) {
      if (i >= j) continue;
      const cb = nodes[j].community;
      if (ca === cb) continue;
      const key = ca < cb ? `${ca}|${cb}` : `${cb}|${ca}`;
      const e = inter.get(key);
      if (e) {
        e.w += w;
        e.n++;
      } else {
        inter.set(key, { w, n: 1 });
      }
    }
  }
  model.clusterLinks = [...inter.entries()].map(([key, v]) => {
    const [a, b] = key.split("|").map(Number);
    return { source: a, target: b, weight: v.w, count: v.n };
  });
}

/**
 * Entity type → CSS custom property.
 *
 * The design system names exactly four ontology classes
 * (`--node-technology/org/location/person`, index.css:143-146) and they exist
 * for exactly this. The server's `EntityLabel` has ~35 variants
 * (src/graph_memory.rs:254-288), so the other 31 cannot each get a hue without
 * inventing a palette — and the four `--node-*` tokens are already identical to
 * `--chart-1..4`, leaving exactly one free chart hue. Everything outside the
 * named four therefore shares `--chart-5` and reads as "other"; the exact type
 * is never hidden, it is stated on hover and in the Inspector.
 */
export const NAMED_ENTITY_TYPES = ["Person", "Organization", "Location", "Technology"] as const;

export function entityTypeToken(type: string): string {
  switch (type) {
    case "Person":
      return "--node-person";
    case "Organization":
      return "--node-org";
    case "Location":
      return "--node-location";
    case "Technology":
      return "--node-technology";
    default:
      return "--chart-5";
  }
}
