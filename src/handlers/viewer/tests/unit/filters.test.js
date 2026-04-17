import { describe, it, expect } from "bun:test";
import { matchesFilters } from '../../js/domain/filters.js';

const node = (overrides = {}) => ({
  type: 'memory', tier: 'Working', activation: 0.3,
  last_accessed: new Date().toISOString(), ...overrides,
});
const edge = (overrides = {}) => ({
  tier: 'L1Working', weight: 0.4, ltp_status: 'None',
  last_activated: new Date().toISOString(), ...overrides,
});

describe('matchesFilters (node)', () => {
  it('includes node when its tier is in activeTiers', () => {
    const f = { activeTiers: new Set(['Working']), activeTypes: new Set(['memory']),
                minActivation: 0 };
    expect(matchesFilters.node(node(), f)).toBe(true);
  });
  it('excludes node when tier is hidden', () => {
    const f = { activeTiers: new Set(['Longterm']), activeTypes: new Set(['memory']),
                minActivation: 0 };
    expect(matchesFilters.node(node({ tier: 'Working' }), f)).toBe(false);
  });
  it('excludes node below minActivation', () => {
    const f = { activeTiers: new Set(['Working']), activeTypes: new Set(['memory']),
                minActivation: 0.5 };
    expect(matchesFilters.node(node({ activation: 0.2 }), f)).toBe(false);
  });
});

describe('matchesFilters (edge)', () => {
  it('excludes edge below minWeight', () => {
    const f = { activeTiers: new Set(['L1Working']), minWeight: 0.5, activeLtp: new Set(['None']) };
    expect(matchesFilters.edge(edge({ weight: 0.2 }), f)).toBe(false);
  });
  it('includes edge matching tier + LTP + weight', () => {
    const f = { activeTiers: new Set(['L1Working']), minWeight: 0.3, activeLtp: new Set(['None']) };
    expect(matchesFilters.edge(edge(), f)).toBe(true);
  });
});
