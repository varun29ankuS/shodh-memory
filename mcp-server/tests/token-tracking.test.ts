import { describe, expect, it, beforeEach } from "vitest";
import { TokenTracker } from "../token-tracking";

describe("TokenTracker", () => {
  let tracker: TokenTracker;

  beforeEach(() => {
    tracker = new TokenTracker(100_000, 0.9, () => 1000);
  });

  // =========================================================================
  // estimateTokens
  // =========================================================================
  describe("estimateTokens", () => {
    it("estimates ~4 chars per token", () => {
      expect(tracker.estimateTokens("abcd")).toBe(1);
      expect(tracker.estimateTokens("abcdefgh")).toBe(2);
    });

    it("rounds up for partial tokens", () => {
      expect(tracker.estimateTokens("ab")).toBe(1); // ceil(2/4) = 1
      expect(tracker.estimateTokens("abcde")).toBe(2); // ceil(5/4) = 2
    });

    it("returns 0 for empty string", () => {
      expect(tracker.estimateTokens("")).toBe(0);
    });

    it("handles very long text", () => {
      const text = "x".repeat(400_000);
      expect(tracker.estimateTokens(text)).toBe(100_000);
    });
  });

  // =========================================================================
  // trackTokens
  // =========================================================================
  describe("trackTokens", () => {
    it("accumulates tokens across calls", () => {
      tracker.trackTokens("abcd"); // 1 token
      tracker.trackTokens("abcdefgh"); // 2 tokens
      expect(tracker.getSessionTokens()).toBe(3);
    });

    it("returns count for current tracking call", () => {
      const count = tracker.trackTokens("abcdefgh");
      expect(count).toBe(2);
    });
  });

  // =========================================================================
  // getStatus
  // =========================================================================
  describe("getStatus", () => {
    it("returns zero usage initially", () => {
      const status = tracker.getStatus();
      expect(status.tokens).toBe(0);
      expect(status.budget).toBe(100_000);
      expect(status.percent).toBe(0);
      expect(status.alert).toBeNull();
    });

    it("calculates percent correctly", () => {
      // Track 50k tokens (50% of 100k budget)
      for (let i = 0; i < 50_000; i++) {
        tracker.trackTokens("abcd"); // 1 token each
      }
      const status = tracker.getStatus();
      expect(status.tokens).toBe(50_000);
      expect(status.percent).toBe(0.5);
      expect(status.alert).toBeNull();
    });

    it("triggers alert at threshold", () => {
      // Track 90k tokens (90% of 100k budget) — at threshold
      const tracker90 = new TokenTracker(100, 0.9, () => 1000);
      for (let i = 0; i < 90; i++) {
        tracker90.trackTokens("abcd"); // 1 token each
      }
      const status = tracker90.getStatus();
      expect(status.alert).toBe("context_90_percent");
    });

    it("triggers alert above threshold", () => {
      const tracker95 = new TokenTracker(100, 0.9, () => 1000);
      for (let i = 0; i < 95; i++) {
        tracker95.trackTokens("abcd");
      }
      expect(tracker95.getStatus().alert).toBe("context_90_percent");
    });

    it("no alert below threshold", () => {
      const t = new TokenTracker(100, 0.9, () => 1000);
      for (let i = 0; i < 89; i++) {
        t.trackTokens("abcd");
      }
      expect(t.getStatus().alert).toBeNull();
    });

    it("respects custom threshold", () => {
      const t = new TokenTracker(100, 0.5, () => 1000);
      for (let i = 0; i < 50; i++) {
        t.trackTokens("abcd");
      }
      expect(t.getStatus().alert).toBe("context_50_percent");
    });
  });

  // =========================================================================
  // reset
  // =========================================================================
  describe("reset", () => {
    it("clears token count", () => {
      tracker.trackTokens("abcdefgh");
      expect(tracker.getSessionTokens()).toBeGreaterThan(0);
      tracker.reset();
      expect(tracker.getSessionTokens()).toBe(0);
    });

    it("resets session start time", () => {
      let time = 1000;
      const t = new TokenTracker(100_000, 0.9, () => time);
      expect(t.getSessionStartTime()).toBe(1000);
      time = 2000;
      t.reset();
      expect(t.getSessionStartTime()).toBe(2000);
    });
  });

  // =========================================================================
  // constructor
  // =========================================================================
  describe("constructor", () => {
    it("uses defaults when no args", () => {
      const t = new TokenTracker();
      expect(t.budget).toBe(100_000);
      expect(t.alertThreshold).toBe(0.9);
    });

    it("accepts custom budget and threshold", () => {
      const t = new TokenTracker(50_000, 0.8);
      expect(t.budget).toBe(50_000);
      expect(t.alertThreshold).toBe(0.8);
    });

    it("uses injectable clock", () => {
      const t = new TokenTracker(100, 0.9, () => 42);
      expect(t.getSessionStartTime()).toBe(42);
    });
  });
});
