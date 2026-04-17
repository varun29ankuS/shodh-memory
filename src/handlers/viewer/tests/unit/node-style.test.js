import { describe, it, expect } from "bun:test";
import { nodeReducer } from '../../js/domain/node-style.js';

describe('nodeReducer', () => {
  it('colors working-tier nodes warm and longterm cool', () => {
    const now = Date.now();
    const working = nodeReducer('n1', {
      type: 'memory',
      tier: 'Working',
      importance: 0.5,
      activation: 0.0,
      access_count: 1,
      last_accessed: new Date(now).toISOString(),
    }, { now });
    const longterm = nodeReducer('n2', {
      type: 'memory',
      tier: 'Longterm',
      importance: 0.5,
      activation: 0.0,
      access_count: 1,
      last_accessed: new Date(now).toISOString(),
    }, { now });

    // Warm = higher R channel, cool = higher B channel
    expect(parseInt(working.color.slice(1, 3), 16)).toBeGreaterThan(
      parseInt(longterm.color.slice(1, 3), 16)
    );
  });

  it('maps importance linearly to size in 6–24px', () => {
    const attrs = { type: 'memory', tier: 'Longterm', activation: 0, access_count: 0,
                    last_accessed: new Date().toISOString() };
    const low = nodeReducer('n1', { ...attrs, importance: 0.0 }, { now: Date.now() });
    const high = nodeReducer('n2', { ...attrs, importance: 1.0 }, { now: Date.now() });
    expect(low.size).toBeCloseTo(6, 1);
    expect(high.size).toBeCloseTo(24, 1);
  });

  it('boosts halo when activation > 0.7', () => {
    const attrs = { type: 'memory', tier: 'Session', importance: 0.5, access_count: 1,
                    last_accessed: new Date().toISOString() };
    const low = nodeReducer('n1', { ...attrs, activation: 0.1 }, { now: Date.now() });
    const high = nodeReducer('n2', { ...attrs, activation: 0.8 }, { now: Date.now() });
    expect(high.haloPulse).toBe(true);
    expect(low.haloPulse).toBe(false);
  });

  it('shapes differ for entity (square) and episode (diamond)', () => {
    const e = nodeReducer('e', { type: 'entity' }, { now: Date.now() });
    const ep = nodeReducer('ep', { type: 'episode' }, { now: Date.now() });
    expect(e.shape).toBe('square');
    expect(ep.shape).toBe('diamond');
  });
});
