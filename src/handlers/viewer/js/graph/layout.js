// FA2 layout wrapper — web-worker-backed when available, sync fallback otherwise.
//
// Exports `startFa2(graph)` which:
//   1. Seeds random positions for any node missing `x`/`y` (GEXF export emits no
//      coordinates; without seeds, FA2's displacement vectors are all zero).
//   2. Picks per-graph FA2 settings via `inferSettings` and starts the worker.
//   3. Returns a handle with `.stop()` to terminate cleanly.

const SEED_SCALE = 100; // Position range: roughly [-SEED_SCALE, +SEED_SCALE].

function seedRandomPositions(graph) {
  graph.forEachNode((id, attrs) => {
    if (typeof attrs.x !== 'number' || typeof attrs.y !== 'number') {
      graph.mergeNodeAttributes(id, {
        x: (Math.random() - 0.5) * 2 * SEED_SCALE,
        y: (Math.random() - 0.5) * 2 * SEED_SCALE,
      });
    }
  });
}

export function startFa2(graph) {
  seedRandomPositions(graph);
  if (graph.order === 0) return { stop: () => {} };

  const fa2 = graphologyLibrary.layoutForceAtlas2;
  const FA2Layout = graphologyLibrary.FA2Layout;
  const settings = fa2.inferSettings(graph);

  if (typeof FA2Layout === 'function' && typeof Worker !== 'undefined') {
    const layout = new FA2Layout(graph, { settings });
    layout.start();
    return {
      stop: () => {
        if (layout.isRunning()) layout.stop();
        layout.kill();
      },
    };
  }

  // Fallback: synchronous iteration loop via requestAnimationFrame.
  let rafId = null;
  let running = true;
  const ITERS_PER_FRAME = 5;
  function tick() {
    if (!running) return;
    fa2.assign(graph, { iterations: ITERS_PER_FRAME, settings });
    rafId = requestAnimationFrame(tick);
  }
  rafId = requestAnimationFrame(tick);
  return {
    stop: () => {
      running = false;
      if (rafId != null) cancelAnimationFrame(rafId);
    },
  };
}
