import { describe, expect, it } from "vitest";
import type { RecallFact } from "@/lib/api";
import { confidenceSpread, supportLabel } from "./consolidated";

function fact(confidence: number, support = 1): RecallFact {
  return {
    id: `f-${confidence}-${support}`,
    fact: "text",
    confidence,
    support_count: support,
    related_entities: [],
  };
}

describe("confidenceSpread", () => {
  it("is null for an empty set, so the caveat is absent rather than 0–0", () => {
    expect(confidenceSpread([])).toBeNull();
  });

  it("reports the true low and high regardless of order", () => {
    // Deliberately NOT sorted. The server sorts by confidence descending
    // (src/handlers/recall.rs:953), but a helper that only worked on sorted
    // input would silently produce a wrong range the day that changes.
    expect(confidenceSpread([fact(0.5), fact(0.98), fact(0.11)])).toEqual({
      low: 0.11,
      high: 0.98,
    });
  });

  it("collapses to one value when the set is uniform", () => {
    expect(confidenceSpread([fact(0.4), fact(0.4)])).toEqual({ low: 0.4, high: 0.4 });
  });

  it("handles a single fact", () => {
    expect(confidenceSpread([fact(0.73)])).toEqual({ low: 0.73, high: 0.73 });
  });

  it("does not seed the range from a hard-coded bound", () => {
    // Every value is above 0 and below 1, so an implementation seeded with
    // `low = 0` / `high = 1` would pass the uniform case above and fail here.
    expect(confidenceSpread([fact(0.6), fact(0.7)])).toEqual({ low: 0.6, high: 0.7 });
  });
});

describe("supportLabel", () => {
  it("refuses to call a freshly minted fact corroborated", () => {
    // support_count is minted at 1 meaning "seen once (this extraction)"
    // — src/memory/compression.rs:738-742. Calling that "confirmed 1×" would
    // present the extraction itself as independent confirmation.
    expect(supportLabel(1)).toBe("not yet re-confirmed");
  });

  it("counts re-confirmations above 1", () => {
    expect(supportLabel(2)).toBe("confirmed 2×");
    expect(supportLabel(406)).toBe("confirmed 406×");
  });

  it("treats a 0 or absent count as unconfirmed rather than printing it", () => {
    expect(supportLabel(0)).toBe("not yet re-confirmed");
  });
});
