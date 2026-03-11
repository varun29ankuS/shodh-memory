import { describe, expect, it } from "vitest";
import {
  isLocalHostFromUrl,
  shouldWarnInsecureApiUrl,
  serializeAndValidateBody,
  nextReconnectDelay,
} from "../security-utils";

describe("isLocalHostFromUrl", () => {
  it("returns true for localhost-style URLs", () => {
    expect(isLocalHostFromUrl("http://localhost:3030")).toBe(true);
    expect(isLocalHostFromUrl("http://127.0.0.1:3030")).toBe(true);
  });

  it("returns false for remote URLs", () => {
    expect(isLocalHostFromUrl("https://example.com/api")).toBe(false);
  });
});

describe("shouldWarnInsecureApiUrl", () => {
  it("warns for remote http URLs by default", () => {
    expect(shouldWarnInsecureApiUrl("http://example.com/api")).toBe(true);
  });

  it("does not warn when allowHttpEnv is true", () => {
    expect(shouldWarnInsecureApiUrl("http://example.com/api", "true")).toBe(false);
  });

  it("does not warn for localhost over http", () => {
    expect(shouldWarnInsecureApiUrl("http://localhost:3030")).toBe(false);
  });
});

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
});

describe("nextReconnectDelay", () => {
  it("doubles reconnect delay", () => {
    expect(nextReconnectDelay(1000, 60000)).toBe(2000);
    expect(nextReconnectDelay(2000, 60000)).toBe(4000);
  });

  it("caps reconnect delay at max", () => {
    expect(nextReconnectDelay(60000, 60000)).toBe(60000);
    expect(nextReconnectDelay(70000, 60000)).toBe(60000);
  });
});
