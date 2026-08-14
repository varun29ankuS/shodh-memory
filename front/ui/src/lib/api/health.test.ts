import { describe, expect, it } from "vitest";
import { isHumanProfile } from "./health";

/**
 * `isHumanProfile` decides which backend profiles a person is ever shown —
 * including the auto-select that picks one on first load. When it was too
 * permissive the app opened on `.mcp-shims`, and every surface then truthfully
 * reported an empty profile: the graph failed to load, geo had no coordinates,
 * tasks answered 400. That reads as a broken product when the only thing wrong
 * is the selection, so each exclusion below is pinned individually.
 */
describe("isHumanProfile", () => {
  it("accepts ordinary human profiles", () => {
    for (const p of ["varun", "analyst-1", "gdelt-bridge", "demo", "a"]) {
      expect(isHumanProfile(p)).toBe(true);
    }
  });

  it("rejects the seat harness store", () => {
    // The seat writes its lessons to `<user>.seat-harness`, a real per-user
    // store — so /api/users lists it like any other profile. A session opened
    // on it would read machinery as memory.
    expect(isHumanProfile("varun.seat-harness")).toBe(false);
    expect(isHumanProfile(".seat-harness")).toBe(false);
  });

  it("rejects dot-prefixed internal directories", () => {
    // The regression: `.mcp-shims` sorted first, so first load selected it.
    expect(isHumanProfile(".mcp-shims")).toBe(false);
    expect(isHumanProfile(".internal")).toBe(false);
    expect(isHumanProfile(".")).toBe(false);
  });

  it("rejects test fixtures left behind by PR work", () => {
    expect(isHumanProfile("test")).toBe(false);
    expect(isHumanProfile("test-recall")).toBe(false);
    expect(isHumanProfile("smoke-test")).toBe(false);
  });

  it("does not reject human profiles that merely contain the word test", () => {
    // Guards against widening the filter into a substring match, which would
    // silently hide a real corpus.
    expect(isHumanProfile("contested")).toBe(true);
    expect(isHumanProfile("latest-intel")).toBe(true);
    expect(isHumanProfile("protest-monitoring")).toBe(true);
  });

  it("does not reject profiles that merely contain a dot", () => {
    expect(isHumanProfile("varun.sharma")).toBe(true);
    expect(isHumanProfile("gdelt.2026")).toBe(true);
  });

  it("currently accepts the empty string — documented, not endorsed", () => {
    // None of the five predicates excludes "", so it passes. The backend is not
    // expected to return it, and this test exists to make that assumption
    // visible: if /api/users ever yields "", the switcher gets a blank row
    // rather than an error, and this is where that gets noticed.
    expect(isHumanProfile("")).toBe(true);
  });
});
