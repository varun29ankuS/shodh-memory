// Smoke test only — full DOM/WebGL rendering verified in Playwright (Task 25).
import { describe, it, expect } from "bun:test";

describe('renderer wiring', () => {
  it('exports a mount function', async () => {
    const mod = await import('../../js/graph/renderer.js');
    expect(typeof mod.mount).toBe('function');
  });
});
