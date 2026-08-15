/**
 * Edge budgeting: cap what any one node contributes to the picture.
 *
 * WHY, MEASURED. On gdelt-bridge the canvas draws 138 entities and 1,319
 * relations, with 691 more already hidden by the co-occurrence floor. The
 * unreadability is in the EDGES, not the nodes: 138 dots is nothing, 628
 * crossing lines is a hairball. And it is not spread evenly -- these graphs are
 * hub-and-spoke, so a handful of nodes contribute most of the crossings.
 *
 * A global cap (universe.ts's EDGE_BUDGET) cannot help with that. It trims the
 * tail while the hubs keep every line they had. A per-node cap does the
 * opposite: it takes the bulk off the hubs and leaves a sparse node's only
 * connection alone, which is the one it most needs to keep.
 *
 * Adapted from LATRACE's generation-time budget (group by source and relation,
 * keep the top by weight), with one change: an edge survives if it is in the
 * top k of EITHER endpoint. Budgeting by source alone silently deletes a
 * leaf's single connection whenever the hub at the other end is busy, which
 * reads as the leaf being unconnected rather than as the picture being
 * budgeted.
 */

export interface BudgetableEdge {
  source: string;
  target: string;
  strength: number;
}

export interface EdgeBudget<E> {
  kept: E[];
  dropped: number;
}

/**
 * Keep, for every node, its `k` strongest edges. An edge survives if either
 * endpoint claims it.
 *
 * A RANK RULE, NOT A QUOTA RULE, and the difference is the whole mechanism.
 * A first version walked the edges strongest-first and kept any edge whose
 * source or target still had budget remaining. That cannot trim a star: by the
 * time the hub is full every leaf is still untouched, so each leaf's own
 * budget re-admits the edge the hub could not afford and nothing is cut. The
 * tests caught it.
 *
 * Instead each node ranks ITS OWN edges and claims its top k. Two hubs that
 * have both spent their claim drop the edge between them, which is exactly the
 * crossing-lines problem; a leaf always keeps its only connection, because
 * that edge is trivially within its own top k.
 *
 * Ties break on the endpoint pair, so the same graph budgets to the same
 * picture twice. Without that, equal-strength edges resolve on array order and
 * the drawing changes between loads of identical data.
 */
export function capEdgesPerNode<E extends BudgetableEdge>(
  edges: readonly E[],
  k: number,
): EdgeBudget<E> {
  if (k <= 0) return { kept: [], dropped: edges.length };

  const pairKey = (edge: E) => `${edge.source} ${edge.target}`;
  const stronger = (a: E, b: E) => b.strength - a.strength || (pairKey(a) < pairKey(b) ? -1 : 1);

  const byNode = new Map<string, E[]>();
  const add = (id: string, edge: E) => {
    const list = byNode.get(id);
    if (list) list.push(edge);
    else byNode.set(id, [edge]);
  };
  for (const edge of edges) {
    add(edge.source, edge);
    // A self-loop is one node's edge, not two.
    if (edge.target !== edge.source) add(edge.target, edge);
  }

  const claimed = new Set<E>();
  for (const list of byNode.values()) {
    for (const edge of list.slice().sort(stronger).slice(0, k)) claimed.add(edge);
  }

  // Emitted in input order: the caller's ordering is what the canvas paints,
  // and reordering edges changes which lines land on top of which.
  const kept = edges.filter((edge) => claimed.has(edge));
  return { kept, dropped: edges.length - kept.length };
}

/**
 * What the footer must say when the budget bit.
 *
 * A picture that quietly dropped half its edges while looking complete is the
 * failure this module exists to avoid, so the sentence ships with the
 * mechanism rather than being left to a caller to remember.
 */
export function describeEdgeBudget(dropped: number, perNode: number): string | null {
  if (dropped <= 0) return null;
  return `${dropped} more edges beyond the strongest ${perNode} per entity`;
}
