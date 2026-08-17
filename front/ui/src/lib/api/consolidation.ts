import { api } from "./client";

/**
 * The three reads that answer "is this thing actually learning?".
 *
 * All three are live, all three were unrendered anywhere in the product, and
 * between them they are the only evidence a person could have that the store
 * does anything to a memory after writing it.
 */

/**
 * `GET /api/graph/{user_id}/tier-census` — router.rs:238.
 *
 * The consolidation state of the whole edge set. One reference in the entire
 * repository before this (the MCP server) and none in the UI.
 *
 * `below_prune_threshold` is the figure that carries: on `claude-code` it is
 * 39,139 of 67,650 edges, so more than half the graph is already weak enough to
 * be dropped by maintenance. That is not a fault — it is decay working — but it
 * is invisible, and a graph screen that draws the survivors without it implies
 * a far more solid structure than exists.
 */
export interface TierCensus {
  l1_working: number;
  l2_episodic: number;
  l3_semantic: number;
  l1_mean_strength: number;
  l2_mean_strength: number;
  l3_mean_strength: number;
  below_prune_threshold: number;
  total_scanned: number;
}

export function fetchTierCensus(userId: string, signal?: AbortSignal): Promise<TierCensus> {
  return api.get<TierCensus>(`/api/graph/${encodeURIComponent(userId)}/tier-census`, signal);
}

/**
 * `POST /api/lineage/stats` — router.rs:226.
 *
 * The causal backbone, counted. The headline this endpoint has been reporting
 * to nobody is that `confirmed_edges` and `explicit_edges` are ZERO on every
 * profile on this server — claude-code (10,000 edges), gdelt-bridge (4,805),
 * defence-live (2,434). Every causal link in the product is inferred; none was
 * ever stated by a source or confirmed by a person.
 */
export interface LineageStats {
  total_edges: number;
  inferred_edges: number;
  confirmed_edges: number;
  explicit_edges: number;
  total_branches: number;
  active_branches: number;
  edges_by_relation: Record<string, number>;
  avg_confidence: number;
}

export function fetchLineageStats(userId: string, signal?: AbortSignal): Promise<LineageStats> {
  return api.post<LineageStats>("/api/lineage/stats", { user_id: userId }, signal);
}

/**
 * `GET /api/consolidation/events` — router.rs:198. **GET with query params**,
 * not POST; a POST returns 405.
 *
 * `ConsolidationEvent` (src/memory/introspection.rs:20) is a 21-variant enum
 * tagged `#[serde(tag = "type", rename_all = "snake_case")]`, so every event is
 * an object with a snake_case `type` discriminant and its own payload. Only the
 * discriminant and the timestamp are typed here: this surface counts what
 * happened by kind and does not read any variant's fields, and typing 21 payload
 * shapes to use none of them would be twenty-one chances to be wrong about a
 * struct nothing reads.
 *
 * THE WINDOW IS THE SERVER'S DEFAULT AND IS NOT A CONTROL. Omitting `since`
 * gives `now - 1 hour` (consolidation.rs:819). That default is sent by omission
 * and the surface states the window in words, because a time-range picker in
 * component state would be a channel a person could drive and the agent could
 * not.
 */
export interface ConsolidationEvent {
  type: string;
  timestamp: string;
}

export function fetchConsolidationEvents(
  userId: string,
  signal?: AbortSignal,
): Promise<ConsolidationEvent[]> {
  return api.get<ConsolidationEvent[]>(
    `/api/consolidation/events?user_id=${encodeURIComponent(userId)}`,
    signal,
  );
}
