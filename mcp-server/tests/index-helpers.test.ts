import { describe, expect, it } from "vitest";
import {
  buildProgressBar,
  extractStringContextFromArgs,
  formatResetTokenSessionText,
  formatTokenStatusText,
  normalizeLimit,
  shouldAppendProactiveContext,
  describeUserId,
} from "../index-helpers";

describe("buildProgressBar", () => {
  it("builds 20-char bar", () => {
    const bar = buildProgressBar(50);
    expect(bar.length).toBe(20);
    expect(bar).toContain("█");
    expect(bar).toContain("░");
  });

  it("clamps percent bounds", () => {
    expect(buildProgressBar(-10)).toBe("░".repeat(20));
    expect(buildProgressBar(150)).toBe("█".repeat(20));
  });
});

describe("formatTokenStatusText", () => {
  it("formats healthy status", () => {
    const text = formatTokenStatusText(
      { tokens: 100, budget: 1000, percent: 0.1, alert: null },
      5,
    );
    expect(text).toContain("Token Status");
    expect(text).toContain("Used: 100 tokens");
    expect(text).toContain("Budget: 1,000 tokens");
    expect(text).toContain("Remaining: 900 tokens");
    expect(text).toContain("Context window healthy");
  });

  it("formats alert status", () => {
    const text = formatTokenStatusText(
      { tokens: 950, budget: 1000, percent: 0.95, alert: "context_90_percent" },
      12,
    );
    expect(text).toContain("ALERT");
    expect(text).toContain("95% used");
  });
});

describe("normalizeLimit", () => {
  it("defaults when undefined", () => {
    expect(normalizeLimit(undefined)).toBe(10);
  });

  it("enforces minimum", () => {
    expect(normalizeLimit(0)).toBe(1);
    expect(normalizeLimit(-5)).toBe(1);
  });

  it("enforces maximum", () => {
    expect(normalizeLimit(999, 250)).toBe(250);
  });

  it("floors decimals", () => {
    expect(normalizeLimit(7.9)).toBe(7);
  });
});

describe("formatResetTokenSessionText", () => {
  it("formats reset text with prior usage and budget", () => {
    const text = formatResetTokenSessionText(12500, 100000);
    expect(text).toContain("Token Session Reset");
    expect(text).toContain("Previous: 12,500 tokens");
    expect(text).toContain("Current: 0 tokens");
    expect(text).toContain("Budget: 100,000 tokens");
    expect(text).toContain("Counter cleared");
  });
});

describe("shouldAppendProactiveContext", () => {
  it("returns false for excluded memory-management tools", () => {
    expect(shouldAppendProactiveContext("remember")).toBe(false);
    expect(shouldAppendProactiveContext("recall")).toBe(false);
    expect(shouldAppendProactiveContext("memory_stats")).toBe(false);
  });

  it("returns true for normal tools", () => {
    expect(shouldAppendProactiveContext("search_code")).toBe(true);
    expect(shouldAppendProactiveContext("read_file")).toBe(true);
  });
});

describe("extractStringContextFromArgs", () => {
  it("extracts only sufficiently long string values", () => {
    const context = extractStringContextFromArgs({
      short: "abc",
      good: "this should be included",
      count: 42,
      nested: { x: 1 },
      alsoGood: "another long value",
    });
    expect(context).toContain("this should be included");
    expect(context).toContain("another long value");
    expect(context).not.toContain("abc");
  });

  it("caps output length", () => {
    const veryLong = "x".repeat(3000);
    const context = extractStringContextFromArgs({ a: veryLong }, 1, 1000);
    expect(context.length).toBe(1000);
  });

  it("returns empty string for non-object args", () => {
    expect(extractStringContextFromArgs(null)).toBe("");
    expect(extractStringContextFromArgs("text")).toBe("");
  });
});

describe("describeUserId", () => {
  const EMAIL = "alice@example.com";

  it("passes the default through unchanged — it is not sensitive", () => {
    expect(describeUserId("claude-code")).toBe("claude-code");
  });

  // The regression this pins: the banner is written to stderr, which clients
  // persist. An operator who puts an email in SHODH_USER_ID must not find it
  // there, and no substring of it either — a prefix leaks the local-part.
  it("never emits the raw id or any substring of it", () => {
    const out = describeUserId(EMAIL);
    expect(out).not.toContain(EMAIL);
    expect(out).not.toContain("alice");
    expect(out).not.toContain("example.com");
    expect(out).not.toContain("@");
  });

  // Only the hash may be derived from userId. .length is not a sanitiser and
  // reintroduced a js/clear-text-logging finding when it was included.
  it("emits the digest and nothing else derived from the id", () => {
    expect(describeUserId(EMAIL)).toBe("custom (sha256:ff8d9819)");
    expect(describeUserId(EMAIL)).not.toMatch(/len|length|\d+ chars/);
  });

  it("is reproducible, so an operator can confirm which id is live", () => {
    expect(describeUserId(EMAIL)).toBe(describeUserId(EMAIL));
    expect(describeUserId("prod-team")).not.toBe(describeUserId(EMAIL));
  });

  // Trailing whitespace from a shell export is the common misconfiguration.
  // The digest alone distinguishes it, which is why the length was redundant.
  it("distinguishes a trailing newline without reporting the length", () => {
    expect(describeUserId(EMAIL + "\n")).not.toBe(describeUserId(EMAIL));
  });

  it("handles the empty string without emitting it", () => {
    const out = describeUserId("");
    expect(out).toMatch(/^custom \(sha256:[0-9a-f]{8}\)$/);
  });
});
