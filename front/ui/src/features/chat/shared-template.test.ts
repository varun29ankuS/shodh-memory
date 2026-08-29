import { describe, expect, it } from "vitest";

import { templateStripper } from "./shared-template";

describe("templateStripper", () => {
  const corpus = [
    "Task: [SHOD-15] Todo created in shodh-redb: [shodh-redb audit] M3: StorageError misused for API misuse",
    "Task: [SHOD-22] Todo created in shodh-redb: [shodh-redb audit] H5: Guard clause drops the error",
    "Task: [SHOD-9] Todo created in shodh-redb: [shodh-redb audit] H17: FixedVec zero-pads on mismatch",
  ];

  it("keeps the identifier and returns the part that differs", () => {
    const strip = templateStripper(corpus);
    expect(strip(corpus[0])).toBe("[SHOD-15] M3: StorageError misused for API misuse");
    expect(strip(corpus[1])).toBe("[SHOD-22] H5: Guard clause drops the error");
  });

  it("matches across differing identifier widths", () => {
    // SHOD-9 is one digit where the others are two: a plain string prefix stops
    // at "SHOD-", which would leave the template in place.
    expect(templateStripper(corpus)(corpus[2])).toBe("[SHOD-9] H17: FixedVec zero-pads on mismatch");
  });


  it("survives outliers that share no template", () => {
    // Measured failure: 44 of 46 memories carried the template and two
    // stragglers reduced the global common prefix to "".
    const withNoise = [
      ...corpus,
      "Recall found nothing useful for cue \"Bangalore\" (10 results, best 0.11)",
      "Tool recall_memory failed: connection refused",
    ];
    const strip = templateStripper(withNoise);
    expect(strip(corpus[0])).toBe("[SHOD-15] M3: StorageError misused for API misuse");
    // The outliers are left exactly as they are.
    expect(strip(withNoise[4])).toBe(withNoise[4]);
  });

  it("handles two templates in one conversation", () => {
    const mixed = [
      ...corpus,
      "GDELT 2024-03-26 Baltimore: vessel Dali struck the Francis Scott Key Bridge",
      "GDELT 2024-03-27 Baltimore: salvage crews cut the collapsed truss sections",
      "GDELT 2024-03-28 Baltimore: channel reopened to limited traffic",
    ];
    const strip = templateStripper(mixed);
    expect(strip(corpus[1])).toBe("[SHOD-22] H5: Guard clause drops the error");
    expect(strip(mixed[3])).toContain("vessel Dali struck");
    expect(strip(mixed[3])).not.toContain("GDELT");
    // The date survives as the identifier, the way [SHOD-22] does.
    expect(strip(mixed[3])).toMatch(/^2024-03-26 /);
  });

  it("does not eat the first letter of the word it keeps", () => {
    // Sorted neighbours share content by chance: Btree/Buddy/Blob all start
    // with B, so the naive prefix runs one character into the real text.
    const bs = [
      "[SHOD-11] audit note: M11: Btree leaf split may leave an empty leaf",
      "[SHOD-13] audit note: M13: Buddy allocator coalesce does not verify",
      "[SHOD-15] audit note: M15: Blob dedup collision has no second check",
    ];
    const strip = templateStripper(bs);
    expect(strip(bs[0])).toContain("Btree");
    expect(strip(bs[1])).toContain("Buddy");
    expect(strip(bs[2])).toContain("Blob");
  });
  it("leaves unrelated labels alone", () => {
    const mixed = ["the bridge collapsed", "coal exports halted", "Seagirt terminal closed"];
    const strip = templateStripper(mixed);
    for (const label of mixed) expect(strip(label)).toBe(label);
  });

  it("does nothing on too few samples", () => {
    const two = ["Task: [A-1] boilerplate here x", "Task: [A-2] boilerplate here y"];
    expect(templateStripper(two)(two[0])).toBe(two[0]);
  });

  it("does nothing when the shared prefix is trivially short", () => {
    const short = ["ab cat", "ab dog", "ab fish"];
    expect(templateStripper(short)("ab cat")).toBe("ab cat");
  });

  it("never returns an empty label", () => {
    const identical = ["Todo created in shodh-redb", "Todo created in shodh-redb", "Todo created in shodh-redb"];
    expect(templateStripper(identical)(identical[0])).toBe(identical[0]);
  });
});
