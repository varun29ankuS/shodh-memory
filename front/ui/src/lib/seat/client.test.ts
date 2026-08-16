import { describe, expect, it } from "vitest";
import { filenameFromDisposition } from "./client";

/**
 * The name an audit export is saved under.
 *
 * This parses a header the client does not control, and every way it can be
 * wrong is quiet: a name that fails to parse silently becomes the caller's
 * fallback and nobody notices, and a name that parses too eagerly reaches
 * `<a download>` — the one place in this product where an attacker-shaped
 * string would be handed to the filesystem.
 */
describe("filenameFromDisposition", () => {
  it("reads the quoted form the seat writes", () => {
    // seat/src/server.ts handleAuditExport, verbatim.
    expect(
      filenameFromDisposition('attachment; filename="shodh-audit-2026-08-16T15-57-30.jsonl"'),
    ).toBe("shodh-audit-2026-08-16T15-57-30.jsonl");
  });

  it("reads an unquoted value, trimmed", () => {
    expect(filenameFromDisposition("attachment; filename = trail.csv ")).toBe("trail.csv");
  });

  it("is null when the header never arrived", () => {
    // The shipped case: the shodh-front proxy forwards only content-type and
    // cache-control, so this is what the browser sees in the Rust binary.
    expect(filenameFromDisposition(null)).toBeNull();
    expect(filenameFromDisposition("attachment")).toBeNull();
    expect(filenameFromDisposition('attachment; filename=""')).toBeNull();
  });

  it("refuses a name carrying a path separator instead of sanitising it", () => {
    expect(filenameFromDisposition('attachment; filename="../../etc/passwd"')).toBeNull();
    expect(filenameFromDisposition('attachment; filename="..\\\\windows\\\\system32"')).toBeNull();
  });

  it("does not read a parameter that merely ends in 'filename'", () => {
    expect(filenameFromDisposition('attachment; xfilename="taken.csv"')).toBeNull();
  });

  it("leaves the RFC 5987 form to the caller's own name", () => {
    // `filename*=` is charset-tagged and percent-encoded. The seat never emits
    // it, and half-decoding it would be worse than not reading it.
    expect(filenameFromDisposition("attachment; filename*=UTF-8''trail.csv")).toBeNull();
  });
});
