import { describe, expect, it } from "vitest";
import { isHumanProfile, outageOf } from "./health";

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

/**
 * The defect this pins, measured in the browser: with a wrong key the status
 * strip read `Key rejected — 401 — set SHODH_API_KEY to the server's key` while
 * the body of the SAME screen read "Recorded work appears here once the memory
 * server is running." The server was running; the key was wrong. Two diagnoses
 * on one screen and the larger one was wrong, in five views at once, because
 * every one of them gated on `state !== "online"` and threw away the
 * distinction `probeBackend` had gone out of its way to establish.
 */
describe("outageOf", () => {
  const absent = "Recorded work appears here once the memory server is running.";

  it("says nothing at all when the backend is reachable and authorized", () => {
    // Null is the caller's signal to carry on and decide for itself whether it
    // has data — "this profile is empty" is a claim only the view can make.
    expect(outageOf({ state: "online", profiles: ["claude"] }, absent)).toBeNull();
    expect(outageOf({ state: "online", profiles: [] }, absent)).toBeNull();
  });

  it("does NOT tell the reader to start a server that is already running", () => {
    const out = outageOf({ state: "unauthorized", status: 401 }, absent);
    expect(out?.title).toBe("Key rejected");
    expect(out?.body).toContain("is running");
    // The regression, stated as an assertion: the offline sentence must not
    // appear over a rejected key.
    expect(out?.body).not.toContain(absent);
    expect(`${out?.body} ${out?.more}`).not.toMatch(/once the memory server is running/);
  });

  it("quotes the status the server actually answered, as the evidence", () => {
    expect(outageOf({ state: "unauthorized", status: 401 }, absent)?.body).toContain("401");
    // 403 is the other auth failure the client classifies; it must not be
    // reported as a 401.
    expect(outageOf({ state: "unauthorized", status: 403 }, absent)?.body).toContain("403");
  });

  it("names the variable and the fix for a rejected key", () => {
    expect(outageOf({ state: "unauthorized", status: 401 }, absent)?.more).toContain(
      "SHODH_API_KEY",
    );
  });

  it("keeps the caller's own sentence for the offline case, where it is true", () => {
    const out = outageOf({ state: "offline", detail: "fetch failed" }, absent);
    expect(out?.title).toBe("Not connected");
    expect(out?.body).toBe(absent);
    expect(out?.more).toContain("fetch failed");
  });

  it("refuses to report an unreachable server as an empty profile", () => {
    const out = outageOf({ state: "offline", detail: "backend returned 502" }, absent);
    expect(out?.more).toContain("not reporting an empty profile");
  });

  it("uses the status strip's own state names, so one product gives one verdict", () => {
    // StatusStrip.tsx renders exactly these two strings for these two states.
    expect(outageOf({ state: "unauthorized", status: 401 }, absent)?.title).toBe("Key rejected");
    expect(outageOf({ state: "offline", detail: "x" }, absent)?.title).toBe("Not connected");
  });
});
