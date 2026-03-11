import { describe, expect, it } from "vitest";
import {
  isLocalHostFromUrl,
  shouldWarnInsecureApiUrl,
  serializeAndValidateBody,
  nextReconnectDelay,
} from "../security-utils";

// =============================================================================
// isLocalHostFromUrl
// =============================================================================
describe("isLocalHostFromUrl", () => {
  it("returns true for localhost", () => {
    expect(isLocalHostFromUrl("http://localhost:3030")).toBe(true);
  });

  it("returns true for 127.0.0.1", () => {
    expect(isLocalHostFromUrl("http://127.0.0.1:3030")).toBe(true);
  });

  it("returns false for ::1 with brackets (URL parser includes brackets)", () => {
    // Note: new URL("http://[::1]:3030").hostname === "[::1]" in Node.js
    // The implementation checks for "::1" without brackets, so this is false
    expect(isLocalHostFromUrl("http://[::1]:3030")).toBe(false);
  });

  it("returns true for 0.0.0.0", () => {
    expect(isLocalHostFromUrl("http://0.0.0.0:3030")).toBe(true);
  });

  it("returns false for remote URLs", () => {
    expect(isLocalHostFromUrl("https://example.com/api")).toBe(false);
  });

  it("returns false for invalid URL (catch branch)", () => {
    expect(isLocalHostFromUrl("not a url")).toBe(false);
  });

  it("returns false for empty string", () => {
    expect(isLocalHostFromUrl("")).toBe(false);
  });
});

// =============================================================================
// shouldWarnInsecureApiUrl
// =============================================================================
describe("shouldWarnInsecureApiUrl", () => {
  it("warns for remote http URLs by default", () => {
    expect(shouldWarnInsecureApiUrl("http://example.com/api")).toBe(true);
  });

  it("does not warn when allowHttpEnv is 'true'", () => {
    expect(shouldWarnInsecureApiUrl("http://example.com/api", "true")).toBe(false);
  });

  it("does not warn for localhost over http", () => {
    expect(shouldWarnInsecureApiUrl("http://localhost:3030")).toBe(false);
  });

  it("does not warn for https URLs (even remote)", () => {
    expect(shouldWarnInsecureApiUrl("https://example.com/api")).toBe(false);
  });

  it("does not warn for 127.0.0.1 over http", () => {
    expect(shouldWarnInsecureApiUrl("http://127.0.0.1:3030")).toBe(false);
  });

  it("warns when allowHttpEnv is undefined (default)", () => {
    expect(shouldWarnInsecureApiUrl("http://remote.server.com")).toBe(true);
  });

  it("warns when allowHttpEnv is 'false' (not 'true')", () => {
    expect(shouldWarnInsecureApiUrl("http://remote.server.com", "false")).toBe(true);
  });
});

// =============================================================================
// serializeAndValidateBody
// =============================================================================
describe("serializeAndValidateBody", () => {
  it("accepts a small payload", () => {
    const result = serializeAndValidateBody({ message: "ok" }, 100);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.serialized).toContain("message");
    }
  });

  it("rejects an oversized payload", () => {
    const result = serializeAndValidateBody({ data: "x".repeat(100) }, 20);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("exceeds maximum length");
    }
  });

  it("accepts payload at exact boundary", () => {
    const obj = { a: "b" };
    const expectedLen = JSON.stringify(obj).length;
    const result = serializeAndValidateBody(obj, expectedLen);
    expect(result.ok).toBe(true);
  });

  it("rejects payload one byte over boundary", () => {
    const obj = { a: "b" };
    const exactLen = JSON.stringify(obj).length;
    const result = serializeAndValidateBody(obj, exactLen - 1);
    expect(result.ok).toBe(false);
  });

  it("handles empty object", () => {
    const result = serializeAndValidateBody({}, 100);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.serialized).toBe("{}");
    }
  });

  it("handles nested objects", () => {
    const result = serializeAndValidateBody({ a: { b: { c: 1 } } }, 10000);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(JSON.parse(result.serialized)).toEqual({ a: { b: { c: 1 } } });
    }
  });
});

// =============================================================================
// nextReconnectDelay
// =============================================================================
describe("nextReconnectDelay", () => {
  it("doubles reconnect delay from 1s", () => {
    expect(nextReconnectDelay(1000, 60000)).toBe(2000);
  });

  it("doubles from 2s to 4s", () => {
    expect(nextReconnectDelay(2000, 60000)).toBe(4000);
  });

  it("caps reconnect delay at max", () => {
    expect(nextReconnectDelay(60000, 60000)).toBe(60000);
  });

  it("caps when current exceeds max", () => {
    expect(nextReconnectDelay(70000, 60000)).toBe(60000);
  });

  it("enforces minimum of 1000ms before doubling", () => {
    // safeCurrent = Math.max(500, 1000) = 1000, then 1000*2 = 2000
    expect(nextReconnectDelay(500, 60000)).toBe(2000);
  });

  it("enforces minimum for zero input", () => {
    expect(nextReconnectDelay(0, 60000)).toBe(2000);
  });

  it("enforces minimum for negative input", () => {
    expect(nextReconnectDelay(-100, 60000)).toBe(2000);
  });

  it("handles small max delay", () => {
    expect(nextReconnectDelay(1000, 1500)).toBe(1500);
  });
});
