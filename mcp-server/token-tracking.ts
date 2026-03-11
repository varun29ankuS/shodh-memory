/**
 * Token tracking for context window awareness (SHO-115).
 *
 * All functions are pure or near-pure for testability —
 * state is held in the TokenTracker class rather than module globals.
 */

export interface TokenStatus {
  tokens: number;
  budget: number;
  percent: number;
  alert: string | null;
}

export class TokenTracker {
  private sessionTokens = 0;
  private sessionStartTime: number;

  constructor(
    readonly budget: number = 100_000,
    readonly alertThreshold: number = 0.9,
    private clock: () => number = Date.now
  ) {
    this.sessionStartTime = clock();
  }

  /** Rough token estimation: ~4 chars per token */
  estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  /** Track tokens consumed by a text */
  trackTokens(text: string): number {
    const count = this.estimateTokens(text);
    this.sessionTokens += count;
    return count;
  }

  /** Get current token status */
  getStatus(): TokenStatus {
    const percent = this.sessionTokens / this.budget;
    return {
      tokens: this.sessionTokens,
      budget: this.budget,
      percent: Math.round(percent * 100) / 100,
      alert:
        percent >= this.alertThreshold
          ? `context_${Math.round(this.alertThreshold * 100)}_percent`
          : null,
    };
  }

  /** Reset session counters */
  reset(): void {
    this.sessionTokens = 0;
    this.sessionStartTime = this.clock();
  }

  /** Get session start time */
  getSessionStartTime(): number {
    return this.sessionStartTime;
  }

  /** Get raw token count */
  getSessionTokens(): number {
    return this.sessionTokens;
  }
}
