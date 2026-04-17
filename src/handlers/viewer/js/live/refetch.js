export function diffGraphs(prev, next) {
  const prevNodeIds = new Set(Object.keys(prev.nodes || {}));
  const nextNodeIds = new Set(Object.keys(next.nodes || {}));
  const prevEdgeIds = new Set(Object.keys(prev.edges || {}));
  const nextEdgeIds = new Set(Object.keys(next.edges || {}));

  const addedNodes = [...nextNodeIds].filter(i => !prevNodeIds.has(i));
  const removedNodes = [...prevNodeIds].filter(i => !nextNodeIds.has(i));
  const commonNodes = [...nextNodeIds].filter(i => prevNodeIds.has(i));
  const addedEdges = [...nextEdgeIds].filter(i => !prevEdgeIds.has(i));
  const removedEdges = [...prevEdgeIds].filter(i => !nextEdgeIds.has(i));
  const commonEdges = [...nextEdgeIds].filter(i => prevEdgeIds.has(i));

  return { addedNodes, removedNodes, commonNodes, addedEdges, removedEdges, commonEdges };
}

export function applyDiffToGraph(sigma, graph, next, diff) {
  for (const id of diff.removedNodes) graph.dropNode(id);
  for (const id of diff.addedNodes) {
    const attrs = next.nodes[id];
    const neighbor = findAnyNeighborId(next, id);
    const base = neighbor && graph.hasNode(neighbor)
      ? graph.getNodeAttributes(neighbor)
      : { x: 0, y: 0 };
    const jitter = () => (Math.random() - 0.5) * 20;
    graph.addNode(id, { ...attrs, x: (base.x || 0) + jitter(), y: (base.y || 0) + jitter() });
  }
  for (const id of diff.commonNodes) graph.mergeNodeAttributes(id, next.nodes[id]);
  for (const id of diff.removedEdges) graph.dropEdge(id);
  for (const id of diff.addedEdges) {
    const e = next.edges[id];
    graph.addEdgeWithKey(id, e.source, e.target, e);
  }
  for (const id of diff.commonEdges) graph.mergeEdgeAttributes(id, next.edges[id]);
  sigma.refresh();
}

function findAnyNeighborId(next, nodeId) {
  for (const [_, e] of Object.entries(next.edges || {})) {
    if (e.source === nodeId) return e.target;
    if (e.target === nodeId) return e.source;
  }
  return null;
}

export function createRefetcher({ fetchImpl, applyDiff, debounceMs = 2000 }) {
  let timer = null;
  let inFlight = false;
  let pending = false;

  async function run() {
    if (inFlight) { pending = true; return; }
    inFlight = true;
    try {
      const result = await fetchImpl();
      if (!result.unchanged) applyDiff(result);
    } catch (e) {
      console.error('[refetch] failed', e);
    } finally {
      inFlight = false;
      if (pending) { pending = false; schedule(); }
    }
  }

  function schedule() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(run, debounceMs);
  }

  function trigger() {
    if (inFlight) { pending = true; return; }
    schedule();
  }

  return { trigger };
}
