import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { startFa2 } from '../../js/graph/layout.js';

function mockGraphologyLibrary() {
  const calls = { inferSettings: 0, assign: 0 };
  globalThis.graphologyLibrary = {
    layoutForceAtlas2: {
      inferSettings: () => { calls.inferSettings++; return { scalingRatio: 1 }; },
      assign: () => { calls.assign++; },
    },
    // No FA2Layout — forces the synchronous fallback path.
  };
  return calls;
}

function makeGraph(nodes) {
  const attrs = new Map(nodes.map((n) => [n.id, { ...n }]));
  return {
    get order() { return attrs.size; },
    forEachNode(fn) { for (const [id, a] of attrs) fn(id, a); },
    mergeNodeAttributes(id, patch) { Object.assign(attrs.get(id), patch); },
    _attrs: attrs,
  };
}

describe('startFa2', () => {
  let prevWorker;
  let rafCallbacks;

  beforeEach(() => {
    prevWorker = globalThis.Worker;
    globalThis.Worker = undefined;
    rafCallbacks = [];
    globalThis.requestAnimationFrame = (fn) => { rafCallbacks.push(fn); return rafCallbacks.length; };
    globalThis.cancelAnimationFrame = () => {};
  });
  afterEach(() => {
    globalThis.Worker = prevWorker;
    delete globalThis.graphologyLibrary;
    delete globalThis.requestAnimationFrame;
    delete globalThis.cancelAnimationFrame;
  });

  it('seeds random positions for nodes missing x/y', () => {
    mockGraphologyLibrary();
    const graph = makeGraph([{ id: 'a' }, { id: 'b', x: 10, y: 20 }]);
    startFa2(graph);
    const a = graph._attrs.get('a');
    const b = graph._attrs.get('b');
    expect(typeof a.x).toBe('number');
    expect(typeof a.y).toBe('number');
    expect(b.x).toBe(10);
    expect(b.y).toBe(20);
  });

  it('returns a no-op handle for empty graphs', () => {
    mockGraphologyLibrary();
    const graph = makeGraph([]);
    const handle = startFa2(graph);
    expect(typeof handle.stop).toBe('function');
    handle.stop(); // should not throw
  });

  it('falls back to synchronous RAF loop when FA2Layout/Worker are absent', () => {
    const calls = mockGraphologyLibrary();
    const graph = makeGraph([{ id: 'a' }, { id: 'b' }]);
    const handle = startFa2(graph);
    expect(rafCallbacks.length).toBe(1);
    rafCallbacks[0]();
    expect(calls.assign).toBe(1);
    handle.stop();
    rafCallbacks[1]?.();
    expect(calls.assign).toBe(1);
  });
});
