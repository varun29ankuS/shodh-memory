import { nodeReducer as styleNode } from '../domain/node-style.js';
import { edgeReducer as styleEdge } from '../domain/edge-style.js';

export function mount(graph, container, opts = {}) {
  // State owned by the renderer, mutated by interaction handlers.
  const state = { hoveredNode: null, selectedNode: null, manuallyHidden: new Set() };

  const sigma = new Sigma(graph, container, {
    nodeReducer: (id, attrs) => {
      const base = styleNode(id, attrs, { now: Date.now() });
      if (state.manuallyHidden.has(id)) return { ...base, hidden: true };
      if (state.hoveredNode && state.hoveredNode !== id && !graph.areNeighbors(state.hoveredNode, id)) {
        return { ...base, color: 'rgba(0,0,0,0.1)', label: '', zIndex: 0 };
      }
      if (state.hoveredNode === id || state.selectedNode === id) {
        return { ...base, highlighted: true, zIndex: 1 };
      }
      return base;
    },
    edgeReducer: (id, attrs) => {
      const base = styleEdge(id, attrs, { now: Date.now() });
      if (state.hoveredNode) {
        const [s, t] = graph.extremities(id);
        const touches = s === state.hoveredNode || t === state.hoveredNode;
        if (!touches) return { ...base, hidden: true };
      }
      return base;
    },
    ...opts,
  });

  // FA2 loading overlay
  const overlay = document.createElement('div');
  overlay.className = 'fa2-overlay';
  overlay.textContent = 'Laying out\u2026';
  container.appendChild(overlay);

  // Hide overlay after FA2 converges (best-effort: hide on first stall, or 10s timeout)
  const fa2Worker = startFa2(graph);
  let lastTick = performance.now();
  const checkConverged = setInterval(() => {
    if (performance.now() - lastTick > 1500) {
      overlay.remove();
      clearInterval(checkConverged);
      fa2Worker.stop();
    }
  }, 500);
  setTimeout(() => {
    overlay.remove();
    clearInterval(checkConverged);
  }, 10_000);

  // Animation loop: pulses require refresh
  let rafId;
  function tick() {
    sigma.refresh();
    rafId = requestAnimationFrame(tick);
  }
  rafId = requestAnimationFrame(tick);

  return { sigma, state, stop: () => { cancelAnimationFrame(rafId); fa2Worker.stop(); } };
}

function startFa2(_graph) {
  // Placeholder — real FA2 worker is wired in a later task.
  return { stop: () => {} };
}
