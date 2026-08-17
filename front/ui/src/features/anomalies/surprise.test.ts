import { describe, expect, it } from "vitest";
import type { AnomalyEntry, AnomalyListResponse } from "@/lib/api/anomalies";
import {
  MIN_BASELINE_EPISODES,
  axisLabel,
  drivingAxis,
  readSurprise,
  surpriseFacts,
  uniformMagnitude,
} from "./surprise";

function entry(maxZ: number, flagged: boolean, deviations: Array<[string, number]> = []): AnomalyEntry {
  return {
    memory_id: `m-${maxZ}-${flagged}-${deviations.length}`,
    created_at: "2026-08-17T10:00:34Z",
    content_preview: "preview",
    components: {
      mean_pmi: 0,
      novel_entity_ratio: 1,
      untyped_ratio: 0,
      pmi_gated_ratio: 0,
      low_selectivity_share: 0,
      pairs_scored: 0,
      entities_total: 1,
    },
    deviations: deviations.map(([component, z]) => ({
      component,
      value: 0,
      baseline_mean: 0,
      baseline_std: 1,
      z,
    })),
    max_abs_z: maxZ,
    flagged,
    explanation: "because",
    entities: [],
  };
}

function response(anomalies: AnomalyEntry[], scored: number): AnomalyListResponse {
  return { anomalies, episodes_scored: scored, baseline_window: 200, min_sigma: 2 };
}

describe("readSurprise", () => {
  it("distinguishes no baseline from nothing unusual", () => {
    // Both arrive as `anomalies: []`. Only episodes_scored separates a refusal
    // to score from a clean corpus, and reporting one as the other is the exact
    // failure this branch exists to prevent. spot-1 returns 0 live.
    const noBaseline = readSurprise(response([], 0));
    expect(noBaseline.state).toBe("insufficient");

    const clean = readSurprise(response([], 200));
    expect(clean.state).toBe("clear");
  });

  it("refuses to score right up to the server's own minimum", () => {
    expect(readSurprise(response([], MIN_BASELINE_EPISODES - 1)).state).toBe("insufficient");
    expect(readSurprise(response([], MIN_BASELINE_EPISODES)).state).toBe("clear");
  });

  it("says how many episodes fell short when some were scored", () => {
    const result = readSurprise(response([], 3));
    expect(result.state).toBe("insufficient");
    if (result.state !== "insufficient") throw new Error("unreachable");
    expect(result.reason).toContain("3");
  });

  it("separates flagged entries from the ranked feed they arrived in", () => {
    // gdelt-bridge live: 14 flagged of 20 returned. The unflagged six are the
    // nearest ordinary episodes and must survive as the comparison population.
    const result = readSurprise(
      response([entry(8.86, true), entry(2.4, true), entry(1.83, false)], 200),
    );
    expect(result.state).toBe("findings");
    if (result.state !== "findings") throw new Error("unreachable");
    expect(result.flagged).toHaveLength(2);
    expect(result.ranked).toHaveLength(3);
  });

  it("is clear, not findings, when the feed has entries but none cleared the line", () => {
    const result = readSurprise(response([entry(1.9, false), entry(1.5, false)], 200));
    expect(result.state).toBe("clear");
    if (result.state !== "clear") throw new Error("unreachable");
    // The near-misses are still carried — "nothing flagged" over a drawn
    // population is a result, over an empty panel it is a shrug.
    expect(result.ranked).toHaveLength(2);
  });
});

describe("uniformMagnitude", () => {
  it("is false below two entries — one finding has no uniformity to report", () => {
    expect(uniformMagnitude([])).toBe(false);
    expect(uniformMagnitude([entry(2.05, true)])).toBe(false);
  });

  it("detects the repeated-memory case seen live on claude-code", () => {
    // All twenty returned entries carry max_abs_z 2.051865 exactly, because
    // they are twenty instances of one hook-written memory shape.
    expect(uniformMagnitude([entry(2.051865, true), entry(2.051865, true), entry(2.051865, true)])).toBe(
      true,
    );
  });

  it("is false when magnitudes genuinely differ", () => {
    expect(uniformMagnitude([entry(8.86, true), entry(2.14, true)])).toBe(false);
  });

  it("does not treat near-equal magnitudes as uniform", () => {
    // A tolerance here would start collapsing distinct findings together.
    expect(uniformMagnitude([entry(2.05, true), entry(2.06, true)])).toBe(false);
  });
});

describe("drivingAxis", () => {
  it("picks the largest absolute z, not the largest signed z", () => {
    // mean_pmi deviates NEGATIVELY and hardest in the live claude-code feed;
    // a signed max would report the weaker positive axis instead.
    expect(drivingAxis(entry(3.2, true, [["novel_entity_ratio", 2.05], ["mean_pmi", -3.2]]))).toBe(
      "mean_pmi",
    );
  });

  it("returns the single axis when there is only one", () => {
    expect(drivingAxis(entry(2.1, true, [["untyped_ratio", 2.1]]))).toBe("untyped_ratio");
  });

  it("is null rather than a fabricated axis when there are no deviations", () => {
    expect(drivingAxis(entry(0, false, []))).toBeNull();
  });

  it("keeps the first axis on an exact tie rather than the last", () => {
    expect(drivingAxis(entry(2, true, [["mean_pmi", -2], ["untyped_ratio", 2]]))).toBe("mean_pmi");
  });
});

describe("surpriseFacts", () => {
  it("reads every token off the response instead of restating a default", () => {
    expect(surpriseFacts({ anomalies: [], episodes_scored: 188, baseline_window: 200, min_sigma: 2 }))
      .toEqual(["188 episodes scored", "baseline 200 most recent", "flag at 2σ"]);
  });

  it("reflects a server that answered with different parameters", () => {
    expect(surpriseFacts({ anomalies: [], episodes_scored: 12, baseline_window: 50, min_sigma: 3.5 }))
      .toEqual(["12 episodes scored", "baseline 50 most recent", "flag at 3.5σ"]);
  });
});

describe("axisLabel", () => {
  it("names the five scored axes", () => {
    expect(axisLabel("mean_pmi")).toBe("entity association");
    expect(axisLabel("novel_entity_ratio")).toBe("unseen entities");
    expect(axisLabel("untyped_ratio")).toBe("untyped relations");
    expect(axisLabel("pmi_gated_ratio")).toBe("gated pairs");
    expect(axisLabel("low_selectivity_share")).toBe("generic entities");
  });

  it("returns an unrecognised axis verbatim", () => {
    expect(axisLabel("some_new_axis")).toBe("some_new_axis");
  });
});
