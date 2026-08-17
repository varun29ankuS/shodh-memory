import { api } from "./client";

/**
 * `POST /api/anomalies` — src/handlers/router.rs:201, handler
 * src/handlers/anomalies.rs.
 *
 * WHAT THIS IS, AND WHY IT IS NOT THE SAME THING THE ANOMALIES SCREEN ALREADY
 * COMPUTED. The three lenses in features/anomalies/measures.ts read a memory's
 * CONTENT — where it says it is, what numbers it states, whether it names
 * anything else in the corpus names. This endpoint reads a memory's
 * EXTRACTION STATISTICS: the `SurpriseComponents` captured at ingest, scored at
 * read time as per-component z-scores against the profile's own rolling
 * baseline. Two different objects, both correctly called anomalies, and the
 * screen names both rather than letting one impersonate the other.
 *
 * The design is stated in the handler's own module doc and it is the reason
 * this is worth surfacing: "ingest stores FACTS (the episode's statistical
 * shape); this endpoint computes DEVIATION". Thresholds are tunable without
 * re-ingesting, and every flag is explainable component-by-component,
 * "deterministically and without any LLM in the loop" — which is the same claim
 * the screen's own header makes, arrived at independently on the server.
 *
 * A CORRECTION TO THE CAPABILITY AUDIT, verified live before building on it.
 * The audit recorded "anomaly detection has exactly one kind,
 * `TemporalAnomalyKind::DormantReactivation` (src/graph_memory.rs:2458)". That
 * is a different subsystem. What `/api/anomalies` serves is this five-axis
 * deviation feed, and it is materially richer than the note implied: live on
 * `gdelt-bridge` it returns z-scores from 1.83 to 8.86 with per-component
 * baselines and a written explanation for each.
 */

/**
 * `SurpriseComponents` — the episode's statistical shape at ingest.
 *
 * `pairs_scored` and `entities_total` are the extraction's own counts and are
 * NOT scored as axes (the five that are appear in `AXES`,
 * src/handlers/anomalies.rs:92-98); they are context for reading the ratios,
 * because a ratio over one entity is a different kind of number from a ratio
 * over forty.
 */
export interface SurpriseComponents {
  mean_pmi: number;
  novel_entity_ratio: number;
  untyped_ratio: number;
  pmi_gated_ratio: number;
  low_selectivity_share: number;
  pairs_scored: number;
  entities_total: number;
}

/** `ComponentDeviation` — src/handlers/anomalies.rs:50-57. One axis, its value,
 *  and the baseline it was judged against. Kept explicit by the server "for
 *  explainability", so nothing here has to be recomputed to be shown. */
export interface ComponentDeviation {
  component: string;
  value: number;
  baseline_mean: number;
  baseline_std: number;
  z: number;
}

/** `AnomalyEntity` — src/handlers/anomalies.rs:63-66. The entity uuid matches a
 *  `UniverseStar.id`, which is what makes a flag projectable onto the graph. */
export interface AnomalyEntity {
  id: string;
  name: string;
}

/** `AnomalyEntry` — src/handlers/anomalies.rs:68-81. */
export interface AnomalyEntry {
  memory_id: string;
  created_at: string;
  content_preview: string;
  components: SurpriseComponents;
  deviations: ComponentDeviation[];
  max_abs_z: number;
  /**
   * `max_abs_z >= min_sigma`. NOT every returned entry is flagged — the feed is
   * the top `limit` ranked by max |z| whether or not any of them cleared the
   * line, so on `gdelt-bridge` 6 of 20 come back false. A screen that called
   * all twenty anomalies would be overstating fourteen of them.
   */
  flagged: boolean;
  /** The server's own deterministic, component-by-component account of the
   *  flag. Rendered verbatim: recomputing the sentence client-side would be a
   *  second implementation of the thing that is supposed to be auditable. */
  explanation: string;
  entities: AnomalyEntity[];
}

/** `AnomalyListResponse` — src/handlers/anomalies.rs:83-89. */
export interface AnomalyListResponse {
  anomalies: AnomalyEntry[];
  /**
   * How many scored episodes the baseline was built from. Below
   * `MIN_BASELINE_EPISODES` (10, anomalies.rs:32) the server returns an empty
   * feed rather than z-scores against noise — so 0 anomalies with a low
   * `episodes_scored` is "no baseline", not "nothing is unusual", and the
   * surface must not report one as the other.
   */
  episodes_scored: number;
  baseline_window: number;
  min_sigma: number;
}

/**
 * Defaults are the SERVER'S defaults, sent by omission rather than restated
 * here — `window` 200, `limit` 20, `min_sigma` 2.0 (anomalies.rs:25-30).
 *
 * The request deliberately carries nothing but `user_id`. Every one of the
 * three tunables is a control, and a control that lives in a component's local
 * state is a channel a person can drive and the agent cannot. Until the view
 * bus exposes a dimension for it, the honest position is to have no control at
 * all and to print the parameters the server used, which is what the section
 * header does.
 */
export function fetchAnomalies(
  userId: string,
  signal?: AbortSignal,
): Promise<AnomalyListResponse> {
  return api.post<AnomalyListResponse>("/api/anomalies", { user_id: userId }, signal);
}
