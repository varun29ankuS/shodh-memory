import type { UniverseModel } from "./universe";

/**
 * The part of the graph a question is about.
 *
 * WHY THIS EXISTS. Drawing every entity produces a hairball, and a hairball
 * communicates exactly one fact: there is a dense part and a sparse part.
 * Measured on the running corpus: 1,008 entities and 18,059 relations, against
 * a human edge-tracing limit somewhere near twenty. Nothing about that picture
 * answers a question anyone brought to it.
 *
 * The fix is not a prettier layout, it is drawing less: seed from what the view
 * is actually about and expand a bounded number of hops. Obsidian ships both
 * answers in one product — a global graph nobody opens daily and a local graph
 * that does — and the local one is the seeded neighbourhood.
 *
 * TRUNCATION IS HOP-ORDERED, WHICH IS THE WHOLE POINT. When the cap bites, a
 * naive slice drops whatever sorted last and can orphan a seed from its own
 * neighbours. Admitting ring by ring keeps the result connected and keeps the
 * seeds, so the picture always contains the thing you asked about.
 *
 * Everything here is pure and takes no React, no d3 and no canvas, so the
 * selection can be tested without a DOM.
 */

export interface NeighbourhoodOptions {
  /** How many rings out from the seeds. 1 is the neighbours; 2 is their
   *  neighbours, which is where "and what is that connected to" is answered. */
  depth?: number;
  /** Hard ceiling on nodes admitted, seeds included. */
  maxNodes?: number;
}

export interface Neighbourhood {
  /** Admitted node ids. */
  ids: Set<string>;
  /** id -> hop distance from the nearest seed; 0 for a seed. */
  hops: Map<string, number>;
  /** The seeds actually used, after dropping ids absent from the model. */
  seeds: string[];
  /** Nodes reachable within `depth` that the cap excluded. */
  dropped: number;
  /** The ring at which the cap bit, or null if everything reachable fit. A
   *  partially admitted ring is the one thing a reader must be told about,
   *  because it is the only case where the picture is a sample rather than a
   *  complete answer. */
  truncatedAtHop: number | null;
}

/**
 * Rank nodes for seeding: most connected first.
 *
 * `weightedDegree` rather than `mentions` — mentions is a popularity count, so
 * on a news corpus the biggest number belongs to the most repeated word, which
 * is usually the least informative one. Summed edge strength at least measures
 * how much of the graph hangs off a node.
 *
 * Ties break on `id`, ascending. That is not decoration: without it, equal
 * scores resolve on whatever order the store happened to return, so the same
 * corpus draws a different picture on reload. A graph that cannot be redrawn
 * identically cannot be cited.
 */
export function rankSeeds(model: UniverseModel, k: number): string[] {
  return model.nodes
    .slice()
    .sort((a, b) => b.weightedDegree - a.weightedDegree || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
    .slice(0, Math.max(0, k))
    .map((n) => n.id);
}

/**
 * Breadth-first expansion from `seedIds`, bounded by depth and node count.
 *
 * Within a ring, nodes are admitted in descending `weightedDegree` with the
 * same `id` tie-break, so a partially admitted ring keeps the most connected
 * members and keeps them deterministically.
 */
export function neighbourhood(
  model: UniverseModel,
  seedIds: readonly string[],
  { depth = 2, maxNodes = 500 }: NeighbourhoodOptions = {},
): Neighbourhood {
  const ids = new Set<string>();
  const hops = new Map<string, number>();

  // Seeds the model does not contain are dropped rather than faked. A stale
  // selection must not invent a node.
  const seeds = seedIds.filter((id) => model.index.has(id));
  const cap = Math.max(0, maxNodes);

  if (seeds.length === 0 || cap === 0 || depth < 0) {
    return { ids, hops, seeds, dropped: 0, truncatedAtHop: null };
  }

  let dropped = 0;
  let truncatedAtHop: number | null = null;

  /** Admit in descending weightedDegree, ties on id. Returns what did not fit. */
  const admit = (candidates: number[], hop: number): number[] => {
    const ordered = candidates.slice().sort((a, b) => {
      const na = model.nodes[a];
      const nb = model.nodes[b];
      return nb.weightedDegree - na.weightedDegree || (na.id < nb.id ? -1 : na.id > nb.id ? 1 : 0);
    });
    const admitted: number[] = [];
    for (const i of ordered) {
      if (ids.size >= cap) {
        dropped += 1;
        if (truncatedAtHop === null) truncatedAtHop = hop;
        continue;
      }
      const node = model.nodes[i];
      ids.add(node.id);
      hops.set(node.id, hop);
      admitted.push(i);
    }
    return admitted;
  };

  let frontier = admit(
    seeds.map((id) => model.index.get(id)!),
    0,
  );

  for (let hop = 1; hop <= depth; hop++) {
    if (frontier.length === 0) break;
    const next = new Set<number>();
    for (const i of frontier) {
      const adjacency = model.adjacency[i];
      if (!adjacency) continue;
      for (const j of adjacency.keys()) {
        if (!ids.has(model.nodes[j].id)) next.add(j);
      }
    }
    if (next.size === 0) break;
    frontier = admit([...next], hop);
  }

  return { ids, hops, seeds, dropped, truncatedAtHop };
}

/**
 * Whether an edge belongs in the drawn neighbourhood.
 *
 * Both endpoints must be admitted. An edge with one end outside would draw a
 * line to nothing, which reads as a rendering fault rather than as a boundary.
 */
export function edgeInNeighbourhood(
  edge: { source: string; target: string },
  hood: Neighbourhood,
): boolean {
  return hood.ids.has(edge.source) && hood.ids.has(edge.target);
}

/**
 * What the footer must say about this picture, in plain words.
 *
 * A view that silently drew a sample of the graph while looking like all of it
 * is the failure this module exists to prevent, so the caption is part of the
 * module rather than left to a caller to remember.
 */
export function describeNeighbourhood(hood: Neighbourhood, depth: number): string {
  const shown = hood.ids.size;
  const seeds = hood.seeds.length;
  const ring = depth === 1 ? "1 hop" : `${depth} hops`;
  const base = `${shown} ${shown === 1 ? "entity" : "entities"} within ${ring} of ${seeds} ${
    seeds === 1 ? "seed" : "seeds"
  }`;
  if (hood.dropped === 0) return base;
  return `${base} · ${hood.dropped} more reachable, not drawn`;
}
