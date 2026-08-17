import { describe, expect, it } from "vitest";
import type { ConsolidationEvent, LineageStats, TierCensus } from "@/lib/api/consolidation";
import {
  eventCensus,
  eventLabel,
  lineageVerdict,
  prunableShare,
  tierBands,
} from "./consolidating";

function census(over: Partial<TierCensus> = {}): TierCensus {
  return {
    l1_working: 0,
    l2_episodic: 0,
    l3_semantic: 0,
    l1_mean_strength: 0,
    l2_mean_strength: 0,
    l3_mean_strength: 0,
    below_prune_threshold: 0,
    total_scanned: 0,
    ...over,
  };
}

function lineage(over: Partial<LineageStats> = {}): LineageStats {
  return {
    total_edges: 0,
    inferred_edges: 0,
    confirmed_edges: 0,
    explicit_edges: 0,
    total_branches: 0,
    active_branches: 0,
    edges_by_relation: {},
    avg_confidence: 0,
    ...over,
  };
}

function ev(type: string): ConsolidationEvent {
  return { type, timestamp: "2026-08-17T10:00:00Z" };
}

describe("tierBands", () => {
  it("is empty when nothing was scanned, so no row of zeroes is drawn", () => {
    // The `claude` profile returns total_scanned 0 with every field zero.
    expect(tierBands(census())).toEqual([]);
  });

  it("keeps consolidation order rather than sorting by size", () => {
    // claude-code live: 62 / 67 / 67,521. Sorted by count this would read
    // semantic, episodic, working and destroy the progression the row is for.
    const bands = tierBands(
      census({ l1_working: 62, l2_episodic: 67, l3_semantic: 67521, total_scanned: 67650 }),
    );
    expect(bands.map((b) => b.label)).toEqual(["working", "episodic", "semantic"]);
    expect(bands.map((b) => b.n)).toEqual([62, 67, 67521]);
  });

  it("keeps the same order for an inverted corpus", () => {
    // gdelt-bridge live: 164 / 52 / 22 — the opposite shape, same labels.
    const bands = tierBands(
      census({ l1_working: 164, l2_episodic: 52, l3_semantic: 22, total_scanned: 238 }),
    );
    expect(bands.map((b) => b.n)).toEqual([164, 52, 22]);
  });

  it("carries each tier's own mean strength, not a shared one", () => {
    const bands = tierBands(
      census({
        l1_working: 1,
        l2_episodic: 1,
        l3_semantic: 1,
        l1_mean_strength: 0.13,
        l2_mean_strength: 0.03,
        l3_mean_strength: 0.28,
        total_scanned: 3,
      }),
    );
    expect(bands.map((b) => b.strength)).toEqual([0.13, 0.03, 0.28]);
  });
});

describe("prunableShare", () => {
  it("is null when nothing is near the floor, so no permanent 0% line is drawn", () => {
    // gdelt-bridge and defence-live both return below_prune_threshold 0.
    expect(prunableShare(census({ total_scanned: 238, below_prune_threshold: 0 }))).toBeNull();
  });

  it("is null when nothing was scanned rather than dividing by zero", () => {
    expect(prunableShare(census())).toBeNull();
  });

  it("reports the real share on claude-code", () => {
    // 39,139 of 67,650 = 57.85% -> 58.
    expect(prunableShare(census({ total_scanned: 67650, below_prune_threshold: 39139 }))).toBe(58);
  });

  it("rounds rather than truncating", () => {
    // 7/10 of a percent past the boundary must round up, not floor to 57.
    expect(prunableShare(census({ total_scanned: 1000, below_prune_threshold: 576 }))).toBe(58);
  });
});

describe("lineageVerdict", () => {
  it("is null on an empty lineage set — zero edges is not a verdict", () => {
    expect(lineageVerdict(lineage())).toBeNull();
  });

  it("reports the all-inferred backbone every live profile has", () => {
    // claude-code: 10,000 edges, 10,000 inferred, 0 confirmed, 0 explicit.
    const verdict = lineageVerdict(
      lineage({ total_edges: 10000, inferred_edges: 10000, confirmed_edges: 0, explicit_edges: 0 }),
    );
    expect(verdict).toEqual({
      total: 10000,
      inferredShare: 100,
      confirmed: 0,
      explicit: 0,
      allInferred: true,
    });
  });

  it("stops claiming all-inferred as soon as one edge is confirmed", () => {
    const verdict = lineageVerdict(
      lineage({ total_edges: 100, inferred_edges: 99, confirmed_edges: 1, explicit_edges: 0 }),
    );
    expect(verdict!.allInferred).toBe(false);
    expect(verdict!.inferredShare).toBe(99);
  });

  it("stops claiming all-inferred when an edge is explicit even with none confirmed", () => {
    const verdict = lineageVerdict(
      lineage({ total_edges: 100, inferred_edges: 99, confirmed_edges: 0, explicit_edges: 1 }),
    );
    expect(verdict!.allInferred).toBe(false);
  });
});

describe("eventCensus", () => {
  it("is empty for no events", () => {
    expect(eventCensus([])).toEqual([]);
  });

  it("counts by kind, commonest first", () => {
    // claude-code live, last hour: 258 decayed, 14 interference, 14 weakened,
    // 8 strengthened, 1 replay, 1 maintenance.
    const events = [
      ...Array.from({ length: 258 }, () => ev("memory_decayed")),
      ...Array.from({ length: 14 }, () => ev("interference_detected")),
      ...Array.from({ length: 8 }, () => ev("memory_strengthened")),
    ];
    expect(eventCensus(events)).toEqual([
      { type: "memory_decayed", count: 258 },
      { type: "interference_detected", count: 14 },
      { type: "memory_strengthened", count: 8 },
    ]);
  });

  it("breaks ties on name so two identical refreshes do not reorder the row", () => {
    expect(eventCensus([ev("memory_weakened"), ev("interference_detected")])).toEqual([
      { type: "interference_detected", count: 1 },
      { type: "memory_weakened", count: 1 },
    ]);
  });

  it("counts a kind it has never seen rather than dropping it", () => {
    expect(eventCensus([ev("semantic_cluster_formed")])).toEqual([
      { type: "semantic_cluster_formed", count: 1 },
    ]);
  });
});

describe("eventLabel", () => {
  it("names the kinds seen live, in the plural", () => {
    expect(eventLabel("memory_decayed", 276)).toBe("memories faded");
    expect(eventLabel("memory_strengthened", 5)).toBe("memories strengthened");
    expect(eventLabel("interference_detected", 2)).toBe("interferences found");
    expect(eventLabel("retrieval_competition", 2)).toBe("recalls contested");
    expect(eventLabel("pattern_triggered_replay", 3)).toBe("patterns replayed");
    expect(eventLabel("maintenance_cycle_completed", 4)).toBe("maintenance passes");
  });

  it("agrees with a count of one", () => {
    // The live claude-code row carries 276 and 1 side by side, and the whole
    // `claude` profile section is a single "1 maintenance pass".
    expect(eventLabel("maintenance_cycle_completed", 1)).toBe("maintenance pass");
    expect(eventLabel("memory_decayed", 1)).toBe("memory faded");
    expect(eventLabel("pattern_triggered_replay", 1)).toBe("pattern replayed");
    expect(eventLabel("interference_detected", 1)).toBe("interference found");
  });

  it("does not treat zero as singular", () => {
    expect(eventLabel("maintenance_cycle_completed", 0)).toBe("maintenance passes");
  });

  it("de-snakes an unseen variant rather than hiding it, at any count", () => {
    // 21 variants exist; a switch that dropped the unlisted ones would make a
    // real event invisible. The de-snaked form is already count-neutral.
    expect(eventLabel("edge_potentiated", 1)).toBe("edge potentiated");
    expect(eventLabel("temporal_cluster_formed", 9)).toBe("temporal cluster formed");
  });
});
