import { describe, expect, it } from "vitest";
import { MEMORY_PREVIEW_MAX, renderContent } from "../memory-format";

describe("MEMORY_PREVIEW_MAX", () => {
  it("defaults the preview cap to 500 chars (raised from 150 per #396)", () => {
    expect(MEMORY_PREVIEW_MAX).toBe(500);
  });
});

describe("renderContent", () => {
  it("returns short content verbatim with no marker (no bare '...')", () => {
    const body = "a short memory body";
    const out = renderContent(body, "mem-1", 500, false);
    expect(out).toBe(body);
    expect(out).not.toContain("truncated");
    expect(out).not.toContain("...");
  });

  it("returns content verbatim when exactly at the cap (boundary, no marker)", () => {
    const body = "x".repeat(500);
    const out = renderContent(body, "mem-1", 500, false);
    expect(out).toBe(body);
    expect(out).not.toContain("truncated");
  });

  it("appends an explicit marker with real lengths and read_memory hint when truncated", () => {
    const body = "y".repeat(2340);
    const out = renderContent(body, "abc123", 500, false);
    expect(out.startsWith("y".repeat(500))).toBe(true);
    expect(out).toContain('…[truncated 500/2340 chars — read_memory("abc123") for full]');
  });

  it("omits the read_memory hint when no id is available but still marks truncation", () => {
    const body = "z".repeat(700);
    const out = renderContent(body, undefined, 180, false);
    expect(out.startsWith("z".repeat(180))).toBe(true);
    expect(out).toContain("…[truncated 180/700 chars]");
    expect(out).not.toContain("read_memory");
  });

  it("treats an empty-string id as no id (no read_memory hint)", () => {
    const body = "q".repeat(600);
    const out = renderContent(body, "", 500, false);
    expect(out).toContain("…[truncated 500/600 chars]");
    expect(out).not.toContain("read_memory");
  });

  it("returns the full body verbatim when full=true, even past the cap (no marker)", () => {
    const body = "w".repeat(3000);
    const out = renderContent(body, "mem-1", 500, true);
    expect(out).toBe(body);
    expect(out).not.toContain("truncated");
  });
});
