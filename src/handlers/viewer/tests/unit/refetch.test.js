import { describe, it, expect, jest, mock, beforeEach, afterEach } from "bun:test";
import { createRefetcher, diffGraphs } from '../../js/live/refetch.js';

describe('diffGraphs', () => {
  it('computes added / removed / common nodes', () => {
    const prev = { nodes: { a: {}, b: {} }, edges: {} };
    const next = { nodes: { b: {}, c: {} }, edges: {} };
    const d = diffGraphs(prev, next);
    expect(d.addedNodes).toEqual(['c']);
    expect(d.removedNodes).toEqual(['a']);
    expect(d.commonNodes).toEqual(['b']);
  });
  it('same for edges', () => {
    const prev = { nodes: {}, edges: { x: {}, y: {} } };
    const next = { nodes: {}, edges: { y: {}, z: {} } };
    const d = diffGraphs(prev, next);
    expect(d.addedEdges).toEqual(['z']);
    expect(d.removedEdges).toEqual(['x']);
    expect(d.commonEdges).toEqual(['y']);
  });
});

describe('createRefetcher', () => {
  beforeEach(() => { jest.useFakeTimers(); });
  afterEach(() => { jest.useRealTimers(); });

  it('debounces trigger calls by 2000ms', async () => {
    const fetchImpl = mock().mockResolvedValue({ unchanged: true });
    const r = createRefetcher({ fetchImpl, applyDiff: mock(), debounceMs: 2000 });
    r.trigger(); r.trigger(); r.trigger();
    expect(fetchImpl).not.toHaveBeenCalled();
    jest.advanceTimersByTime(1999);
    expect(fetchImpl).not.toHaveBeenCalled();
    jest.advanceTimersByTime(2);
    // Flush the microtask queue so the async run() can observe the timer fire.
    await Promise.resolve();
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('skips applyDiff when response is unchanged (304)', async () => {
    const applyDiff = mock();
    const fetchImpl = mock().mockResolvedValue({ unchanged: true });
    const r = createRefetcher({ fetchImpl, applyDiff, debounceMs: 0 });
    r.trigger();
    jest.runAllTimers();
    // Let the awaited fetchImpl resolve before we assert.
    await Promise.resolve();
    await Promise.resolve();
    expect(applyDiff).not.toHaveBeenCalled();
  });

  it('applies diff on fresh response', async () => {
    const applyDiff = mock();
    const fresh = { graph: { nodes: { a: {} }, edges: {} }, etag: 'W/"1"' };
    const fetchImpl = mock().mockResolvedValue(fresh);
    const r = createRefetcher({ fetchImpl, applyDiff, debounceMs: 0 });
    r.trigger();
    jest.runAllTimers();
    await Promise.resolve();
    await Promise.resolve();
    expect(applyDiff).toHaveBeenCalledTimes(1);
  });
});
