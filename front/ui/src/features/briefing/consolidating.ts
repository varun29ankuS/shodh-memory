import type { ConsolidationEvent, LineageStats, TierCensus } from "@/lib/api/consolidation";

/**
 * Turning three unrendered endpoints into what the front page can honestly say
 * about whether the store is doing anything to what it holds.
 *
 * THE RULE THROUGHOUT: every figure below is a count the server returned or a
 * share of one. Nothing is inferred from a status, nothing is scored, and where
 * a corpus has not been consolidated the functions return nothing rather than
 * a zero that would read as a measurement.
 */

// =============================================================================
// TIER CENSUS
// =============================================================================

/** One consolidation tier, with its size and how strong its edges are. */
export interface TierBand {
  label: string;
  n: number;
  /** Mean edge strength in this tier, as the server computed it. */
  strength: number;
}

/**
 * The three tiers, always in consolidation order, and only when there is
 * something to report.
 *
 * ORDER IS FIXED AND NOT SORTED BY SIZE. L1→L2→L3 is a progression — working
 * memory promotes to episodic promotes to semantic — and sorting by count would
 * scramble the one thing the row is for. It is what makes two profiles
 * comparable at a glance: `gdelt-bridge` reads 164/52/22, a corpus still
 * arriving, and `claude-code` reads 62/67/67,521, a corpus almost entirely
 * consolidated. Same three labels, opposite shapes.
 *
 * Returns `[]` when nothing was scanned, so the caller renders no row rather
 * than three zeroes. Verified live: the `claude` profile returns
 * `total_scanned: 0` with every field zero.
 */
export function tierBands(census: TierCensus): TierBand[] {
  if (census.total_scanned === 0) return [];
  return [
    { label: "working", n: census.l1_working, strength: census.l1_mean_strength },
    { label: "episodic", n: census.l2_episodic, strength: census.l2_mean_strength },
    { label: "semantic", n: census.l3_semantic, strength: census.l3_mean_strength },
  ];
}

/**
 * The share of the edge set already weak enough for maintenance to drop, as a
 * whole percentage — or `null` when nothing is.
 *
 * `null` rather than 0 because "no edge is near the floor" and "58% of this
 * graph is" are different findings, and a permanent "0% decayed" on a healthy
 * corpus is a line of chrome that never changes. Live: 39,139 of 67,650 on
 * `claude-code` reads 58; `gdelt-bridge` and `defence-live` return 0 and get no
 * line at all.
 */
export function prunableShare(census: TierCensus): number | null {
  if (census.total_scanned === 0 || census.below_prune_threshold === 0) return null;
  return Math.round((census.below_prune_threshold / census.total_scanned) * 100);
}

// =============================================================================
// LINEAGE
// =============================================================================

/** What the causal edge set is made of, and what it is missing. */
export interface LineageVerdict {
  total: number;
  /** Share of edges that were inferred rather than stated or confirmed, 0–100. */
  inferredShare: number;
  confirmed: number;
  explicit: number;
  /** True when NOTHING has been confirmed and nothing was ever explicit. */
  allInferred: boolean;
}

/**
 * The causal backbone, as it actually is.
 *
 * THIS IS THE FINDING THE PRODUCT HAS BEEN HIDING FROM ITSELF. Every profile on
 * this server reports `confirmed_edges: 0` and `explicit_edges: 0` —
 * claude-code 10,000 edges, gdelt-bridge 4,805, defence-live 2,434, all
 * inferred. The confirm path exists and the explicit path exists; neither has
 * ever fired. A screen that showed a causal graph without this would be
 * presenting a structure of guesses as a structure of facts.
 *
 * `null` on an empty lineage set, because 0 edges is not a verdict about
 * causality — it is the absence of one.
 */
export function lineageVerdict(stats: LineageStats): LineageVerdict | null {
  if (stats.total_edges === 0) return null;
  return {
    total: stats.total_edges,
    inferredShare: Math.round((stats.inferred_edges / stats.total_edges) * 100),
    confirmed: stats.confirmed_edges,
    explicit: stats.explicit_edges,
    allInferred: stats.confirmed_edges === 0 && stats.explicit_edges === 0,
  };
}

// =============================================================================
// CONSOLIDATION EVENTS
// =============================================================================

/** One kind of thing the store did, and how often it did it. */
export interface EventCount {
  type: string;
  count: number;
}

/**
 * What happened, by kind, commonest first.
 *
 * NO INVENTED GROUPING. An earlier shape of this bucketed the 21 event
 * variants into "strengthening", "forgetting" and "forming", which reads well
 * and is a judgement this surface has no standing to make — the assignment of
 * `retrieval_competition` to one bucket or another is an editorial claim
 * dressed as a count. The raw kinds are what the server reported and are what
 * is shown; a reader who wants them summarised can do the summarising.
 *
 * Ties break on name so the row does not reorder between two refreshes that
 * returned the same counts.
 */
export function eventCensus(events: readonly ConsolidationEvent[]): EventCount[] {
  const counts = new Map<string, number>();
  for (const event of events) {
    counts.set(event.type, (counts.get(event.type) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
}

/**
 * The seven `ConsolidationEvent` kinds seen live, as [singular, plural].
 *
 * A PAIR RATHER THAN ONE STRING, because these labels sit directly after a
 * count and every one of them can legitimately arrive as 1: the live
 * `claude-code` feed reads "276 memories faded" and "1 maintenance pass" in the
 * same row, and a single plural form renders the second as "1 maintenance
 * passes". The `claude` profile's entire section is one such row.
 */
const EVENT_WORDS: Record<string, [string, string]> = {
  memory_decayed: ["memory faded", "memories faded"],
  memory_strengthened: ["memory strengthened", "memories strengthened"],
  memory_weakened: ["memory weakened", "memories weakened"],
  interference_detected: ["interference found", "interferences found"],
  retrieval_competition: ["recall contested", "recalls contested"],
  pattern_triggered_replay: ["pattern replayed", "patterns replayed"],
  maintenance_cycle_completed: ["maintenance pass", "maintenance passes"],
};

/**
 * A `ConsolidationEvent` discriminant in words, agreeing with its count.
 *
 * The enum has 21 variants (src/memory/introspection.rs:20-220) and the seven
 * seen live are named explicitly. The rest fall through to a de-snaked form of
 * the discriminant itself, which is accurate for every remaining variant —
 * `edge_potentiated` reads "edge potentiated" — and is already count-neutral, so
 * it needs no pair. More importantly it means a variant added server-side
 * appears on screen under its own name rather than being dropped by a switch
 * that did not know about it.
 */
export function eventLabel(type: string, count: number): string {
  const words = EVENT_WORDS[type];
  if (!words) return type.replace(/_/g, " ");
  return count === 1 ? words[0] : words[1];
}
