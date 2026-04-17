import { describe, it, expect } from "bun:test";
import { edgeReducer } from '../../js/domain/edge-style.js';

describe('edgeReducer', () => {
  it('maps weight to thickness in 0.5–5px', () => {
    const now = Date.now();
    const thin = edgeReducer('e1', { weight: 0.0, tier: 'L1Working', ltp_status: 'None' }, { now });
    const thick = edgeReducer('e2', { weight: 1.0, tier: 'L1Working', ltp_status: 'None' }, { now });
    expect(thin.size).toBeCloseTo(0.5, 1);
    expect(thick.size).toBeCloseTo(5.0, 1);
  });

  it('dashes pending-LTP edges, solid for consolidated', () => {
    const now = Date.now();
    const pending = edgeReducer('e1', { weight: 0.5, tier: 'L1Working', ltp_status: 'Pending' }, { now });
    const solid = edgeReducer('e2', { weight: 0.5, tier: 'L1Working', ltp_status: 'Consolidated' }, { now });
    expect(pending.type).toBe('dashed');
    expect(solid.type).toBe('line');
  });

  it('hue differs per tier (L1 warm, L3 cool)', () => {
    const now = Date.now();
    const l1 = edgeReducer('e1', { weight: 0.5, tier: 'L1Working', ltp_status: 'None' }, { now });
    const l3 = edgeReducer('e2', { weight: 0.5, tier: 'L3Semantic', ltp_status: 'None' }, { now });
    expect(parseInt(l1.color.slice(1, 3), 16)).toBeGreaterThan(
      parseInt(l3.color.slice(1, 3), 16)
    );
  });

  it('marks recently-activated edges for pulse animation', () => {
    const now = Date.now();
    const recent = edgeReducer('e1',
      { weight: 0.5, tier: 'L1Working', ltp_status: 'None',
        last_activated: new Date(now - 1000).toISOString() },
      { now });
    const old = edgeReducer('e2',
      { weight: 0.5, tier: 'L1Working', ltp_status: 'None',
        last_activated: new Date(now - 60000).toISOString() },
      { now });
    expect(recent.pulse).toBe(true);
    expect(old.pulse).toBe(false);
  });
});
