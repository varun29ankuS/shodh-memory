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

  /**
   * Content-aware token estimation.
   *
   * Three modes: CJK (1.5 tokens/char), code (bytes*10/32), prose (bytes/4).
   * Classifies via single-pass sampling of first 512 chars.
   */
  estimateTokens(text: string): number {
    if (text.length === 0) return 0;

    const sampleEnd = Math.min(text.length, 512);
    let syntaxCount = 0;
    let highCharCount = 0;

    for (let i = 0; i < sampleEnd; i++) {
      const code = text.charCodeAt(i);
      if (code > 0x7F) {
        highCharCount++;
      } else if (
        code === 0x7B || code === 0x7D || // { }
        code === 0x5B || code === 0x5D || // [ ]
        code === 0x28 || code === 0x29 || // ( )
        code === 0x3B || code === 0x3D || // ; =
        code === 0x3C || code === 0x3E || // < >
        code === 0x7C || code === 0x26 || // | &
        code === 0x23 || code === 0x40 || // # @
        code === 0x21 || code === 0x7E || // ! ~
        code === 0x5E || code === 0x5C || // ^ \
        code === 0x22 || code === 0x27    // " '
      ) {
        syntaxCount++;
      }
    }

    // CJK detection: high ratio of non-ASCII characters
    if (highCharCount > 0) {
      const charCount = [...text].length;
      const byteLen = new TextEncoder().encode(text).length;
      if (charCount > 0 && byteLen > charCount * 2.5) {
        return Math.ceil(charCount * 1.5);
      }
    }

    // Code detection: 8%+ syntax characters in sample
    if (sampleEnd > 0 && (syntaxCount / sampleEnd) >= 0.08) {
      const byteLen = new TextEncoder().encode(text).length;
      return Math.ceil(byteLen * 10 / 32);
    }

    // Prose mode (default)
    const byteLen = new TextEncoder().encode(text).length;
    return Math.ceil(byteLen / 4);
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
