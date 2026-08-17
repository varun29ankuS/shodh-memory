import type { AnomalyEntry, AnomalyListResponse } from "@/lib/api/anomalies";

/**
 * Reading the server's deviation feed onto the screen.
 *
 * Everything here is a READING of what the server already decided. No z-score
 * is recomputed, no threshold is re-applied and no explanation is rewritten:
 * the endpoint exists precisely so that the flag is auditable, and a client
 * that recomputed it would be a second implementation to disagree with the
 * first. What this module does is decide what the section can honestly CLAIM,
 * which is a different job and is the one the screen was getting wrong by not
 * asking the server at all.
 */

/** What the section can say, mirroring `measures.ts`'s three states so the two
 *  kinds of lens read the same way even though one is fetched. */
export type SurpriseState =
  | { state: "findings"; flagged: AnomalyEntry[]; ranked: AnomalyEntry[] }
  | { state: "clear"; ranked: AnomalyEntry[] }
  | { state: "insufficient"; reason: string };

/**
 * The baseline the server used, as tokens, read off the response rather than
 * assumed.
 *
 * These are the half of every row's comparison that does not change, which is
 * the same argument `measures.ts` makes for its own `facts` row — stating them
 * once above is what lets each row below be a magnitude and a sentence instead
 * of a paragraph.
 */
export function surpriseFacts(response: AnomalyListResponse): string[] {
  return [
    `${response.episodes_scored} episodes scored`,
    `baseline ${response.baseline_window} most recent`,
    `flag at ${response.min_sigma}σ`,
  ];
}

/**
 * Minimum scored episodes before the server will compute deviation at all —
 * `MIN_BASELINE_EPISODES`, src/handlers/anomalies.rs:32. Mirrored here so the
 * surface can tell "no baseline" from "nothing unusual", which are different
 * findings and must never be rendered as the same empty panel.
 */
export const MIN_BASELINE_EPISODES = 10;

/**
 * What this lens concluded.
 *
 * THREE OUTCOMES, AND THE ONE THAT MATTERS IS THE DIFFERENCE BETWEEN THE LAST
 * TWO. An empty feed on a profile with 200 scored episodes means the corpus is
 * statistically unremarkable. An empty feed on a profile with 3 means the
 * server declined to score anything, because z-scores against three samples are
 * noise. Both arrive as `anomalies: []`; only `episodes_scored` tells them
 * apart, and a screen that drew one blank panel for both would be reporting a
 * refusal as a clean bill of health. Verified live: `spot-1` returns
 * `episodes_scored: 0`.
 *
 * UNFLAGGED ENTRIES ARE KEPT, NOT DISCARDED. The feed is the top `limit` ranked
 * by max |z| regardless of whether any cleared `min_sigma` — 6 of 20 come back
 * `flagged: false` on `gdelt-bridge`. They are the nearest ordinary episodes to
 * the line, which is exactly the population the flagged ones are being judged
 * against, so they are what this lens has in place of a distribution plot.
 * Reporting them as anomalies would overstate them; dropping them would leave
 * the flagged rows with nothing to be compared to.
 */
export function readSurprise(response: AnomalyListResponse): SurpriseState {
  if (response.episodes_scored < MIN_BASELINE_EPISODES) {
    return {
      state: "insufficient",
      reason:
        response.episodes_scored === 0
          ? "No episode in this profile carries the ingest-time statistics this measure reads, so there is no baseline to deviate from."
          : `Only ${response.episodes_scored} scored episodes; the server needs ${MIN_BASELINE_EPISODES} before a deviation means anything and returns nothing below that.`,
    };
  }

  const flagged = response.anomalies.filter((a) => a.flagged);
  if (flagged.length === 0) return { state: "clear", ranked: response.anomalies };
  return { state: "findings", flagged, ranked: response.anomalies };
}

/**
 * True when every flagged episode sits at the same distance from the baseline.
 *
 * This is a real and reportable shape rather than a rounding artefact: on
 * `claude-code` all twenty returned entries carry `max_abs_z` of exactly 2.05,
 * because they are twenty instances of one repeated hook-written memory with
 * identical `SurpriseComponents`. A reader looking at twenty rows with the same
 * magnitude should be told that is one pattern seen twenty times, not twenty
 * independent findings — the same claim `measures.ts` makes with its pattern
 * brackets, reached here from the numbers instead.
 *
 * Compared exactly. These are `f32`s serialised from one computation over
 * identical inputs, so genuinely identical episodes produce bit-identical
 * values; a tolerance would start collapsing distinct findings together.
 */
export function uniformMagnitude(flagged: readonly AnomalyEntry[]): boolean {
  if (flagged.length < 2) return false;
  return flagged.every((a) => a.max_abs_z === flagged[0].max_abs_z);
}

/**
 * The axis that drove a flag: the deviation with the largest |z|.
 *
 * The server already writes a sentence naming the top components, and that
 * sentence is what the row renders. This picks out the single axis so the row
 * can also carry it as a token — a scannable column of axis names beside a
 * column of magnitudes is what lets a reader see that fourteen flags are all
 * `novel_entity_ratio` without reading fourteen sentences.
 *
 * `null` for an entry with no deviations rather than a fabricated axis.
 */
export function drivingAxis(entry: AnomalyEntry): string | null {
  let top: string | null = null;
  let best = -1;
  for (const deviation of entry.deviations) {
    const magnitude = Math.abs(deviation.z);
    if (magnitude > best) {
      best = magnitude;
      top = deviation.component;
    }
  }
  return top;
}

/** A `SurpriseComponents` axis name in words. Unrecognised axes are returned
 *  verbatim so a server-side addition shows up rather than disappearing. */
export function axisLabel(component: string): string {
  switch (component) {
    case "mean_pmi":
      return "entity association";
    case "novel_entity_ratio":
      return "unseen entities";
    case "untyped_ratio":
      return "untyped relations";
    case "pmi_gated_ratio":
      return "gated pairs";
    case "low_selectivity_share":
      return "generic entities";
    default:
      return component;
  }
}
