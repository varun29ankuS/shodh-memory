import { describe, it, expect } from "bun:test";
import { isWeak, isOrphan, isDeadEdge } from '../../js/domain/curation.js';

describe('curation', () => {
  it('weak = importance < 0.2 AND access_count < 5', () => {
    expect(isWeak({ importance: 0.1, access_count: 2 })).toBe(true);
    expect(isWeak({ importance: 0.1, access_count: 6 })).toBe(false);
    expect(isWeak({ importance: 0.3, access_count: 2 })).toBe(false);
  });
  it('orphan = zero-degree', () => {
    const graph = { degree: (id) => id === 'a' ? 0 : 3 };
    expect(isOrphan(graph, 'a')).toBe(true);
    expect(isOrphan(graph, 'b')).toBe(false);
  });
  it('deadEdge = last_activated > 7d OR activation_count == 0', () => {
    const now = Date.now();
    const tenDaysAgo = new Date(now - 10*24*60*60*1000).toISOString();
    const oneDayAgo = new Date(now - 24*60*60*1000).toISOString();
    expect(isDeadEdge({ last_activated: tenDaysAgo, activation_count: 2 }, now)).toBe(true);
    expect(isDeadEdge({ last_activated: oneDayAgo, activation_count: 0 }, now)).toBe(true);
    expect(isDeadEdge({ last_activated: oneDayAgo, activation_count: 2 }, now)).toBe(false);
  });
});
