import { nodeReducer as styleNode } from '../domain/node-style.js';
import { edgeReducer as styleEdge } from '../domain/edge-style.js';
import { matchesFilters } from '../domain/filters.js';
import { isWeak, isOrphan, isDeadEdge } from '../domain/curation.js';
import { startFa2 } from './layout.js';

const DEFAULT_CURATION = Object.freeze({ showWeak: false, showOrphans: false, showDeadEdges: false });
const CURATION_HIGHLIGHT = '#ff4444';

// Module-scope constants for the default filter state — built once, never mutated.
const ALL_TIERS = Object.freeze(new Set(['Working', 'Session', 'Longterm', 'L1Working', 'L2Episodic', 'L3Semantic']));
const ALL_TYPES = Object.freeze(new Set(['memory', 'entity', 'episode']));
const ALL_LTP   = Object.freeze(new Set(['None', 'Pending', 'Consolidated', 'JustPromoted']));
const DEFAULT_FILTER = Object.freeze({
  activeTiers: ALL_TIERS,
  activeTypes: ALL_TYPES,
  activeLtp: ALL_LTP,
  minActivation: 0,
  minWeight: 0,
  recencyWindowMs: null,
});

export function defaultFilterState() { return DEFAULT_FILTER; }
export function defaultCurationState() { return DEFAULT_CURATION; }

export function mount(graph, container, opts = {}) {
  const {
    filterState = defaultFilterState,
    curationState = defaultCurationState,
    ...sigmaOpts
  } = opts;

  // State owned by the renderer, mutated by interaction handlers.
  const state = { hoveredNode: null, selectedNode: null, manuallyHidden: new Set() };

  const sigma = new Sigma(graph, container, {
    nodeReducer: (id, attrs) => {
      const f = filterState();
      if (!matchesFilters.node(attrs, f)) return { hidden: true };
      let base = styleNode(id, attrs, { now: Date.now() });
      const c = curationState();
      const highlightWeak = c.showWeak && isWeak(attrs);
      const highlightOrphan = c.showOrphans && isOrphan(graph, id);
      if (highlightWeak || highlightOrphan) {
        base = { ...base, color: CURATION_HIGHLIGHT, highlighted: true, zIndex: 2 };
      }
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
      const f = filterState();
      if (!matchesFilters.edge(attrs, f)) return { hidden: true };
      let base = styleEdge(id, attrs, { now: Date.now() });
      const c = curationState();
      if (c.showDeadEdges && isDeadEdge(attrs)) {
        base = { ...base, color: CURATION_HIGHLIGHT, zIndex: 2 };
      }
      if (state.hoveredNode) {
        const [s, t] = graph.extremities(id);
        const touches = s === state.hoveredNode || t === state.hoveredNode;
        if (!touches) return { ...base, hidden: true };
      }
      return base;
    },
    ...sigmaOpts,
  });

  // FA2 loading overlay — shown for an initial settling window, then hidden.
  // The layout worker keeps running so interactions (drag, refetch) refine
  // positions further; it is only torn down when the renderer is stopped.
  const overlay = document.createElement('div');
  overlay.className = 'fa2-overlay';
  overlay.textContent = 'Laying out\u2026';
  container.appendChild(overlay);

  const fa2Worker = startFa2(graph);
  const overlayTimer = setTimeout(() => overlay.remove(), 2500);

  // Animation loop: pulses require refresh
  let rafId;
  function tick() {
    sigma.refresh();
    rafId = requestAnimationFrame(tick);
  }
  rafId = requestAnimationFrame(tick);

  return {
    sigma,
    state,
    stop: () => {
      cancelAnimationFrame(rafId);
      clearTimeout(overlayTimer);
      overlay.remove();
      fa2Worker.stop();
    },
  };
}
