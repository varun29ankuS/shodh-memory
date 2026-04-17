import { createApiClient } from './config/api-client.js';
import { createLoader } from './graph/loader.js';
import { mount, defaultFilterState, defaultCurationState } from './graph/renderer.js';
import { createSseClient } from './live/sse-client.js';
import { createRefetcher, diffGraphs, applyDiffToGraph } from './live/refetch.js';
import { renderSidebar } from './ui/sidebar.js';
import { renderLegend } from './ui/legend.js';
import { createDetailPanel } from './ui/detail-panel.js';
import { exportIdsAsText, exportVisibleAsGexf, download } from './domain/export.js';

function detectMode() {
  const params = new URLSearchParams(window.location.search);
  const userId = params.get('user_id') || window.SHODH_USER_ID;
  const file = params.get('file');
  if (file) return { mode: 'snapshot-remote', file };
  if (userId && userId !== 'default') return { mode: 'live', userId };
  return { mode: 'drop-zone' };
}

function nodesOf(g) {
  const out = {};
  g.forEachNode((id, a) => { out[id] = a; });
  return out;
}

function edgesOf(g) {
  const out = {};
  g.forEachEdge((id, a, src, tgt) => { out[id] = { ...a, source: src, target: tgt }; });
  return out;
}

function renderDropZone(container, onText) {
  container.innerHTML = '<div class="drop-zone">Drop a .gexf file here</div>';
  container.addEventListener('dragover', (e) => e.preventDefault());
  container.addEventListener('drop', async (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const text = await file.text();
    onText(text);
  });
}

async function main() {
  const mode = detectMode();
  const apiClient = createApiClient({
    baseUrl: window.location.origin,
    apiKey: window.SHODH_API_KEY,
  });
  const loader = createLoader({
    apiClient,
    gexfParser: (G, xml) => graphologyLibrary.gexf.parse(G, xml),
    GraphClass: graphology.Graph,
  });

  const container = document.getElementById('graph-container');
  const sidebarEl = document.getElementById('sidebar');
  const legendEl  = document.getElementById('legend');
  const detailEl  = document.getElementById('detail');

  renderLegend(legendEl);

  if (mode.mode === 'live') {
    let prevEtag = null;
    const first = await loader.fetchFromApi(mode.userId);
    if (first.unchanged) throw new Error('initial fetch returned 304');
    prevEtag = first.etag;

    let currentFilter = defaultFilterState();
    let currentCuration = defaultCurationState();

    const { sigma, state } = mount(first.graph, container, {
      filterState: () => currentFilter,
      curationState: () => currentCuration,
    });

    function invalidate() { sigma.refresh(); }

    const detail = createDetailPanel({ container: detailEl, apiClient, userId: mode.userId });
    sigma.on('clickNode', ({ node }) => detail.show(first.graph, node));

    const refetcher = createRefetcher({
      fetchImpl: async () => {
        const r = await loader.fetchFromApi(mode.userId, prevEtag);
        if (!r.unchanged) prevEtag = r.etag;
        return r;
      },
      applyDiff: (r) => {
        const prev = { nodes: nodesOf(first.graph), edges: edgesOf(first.graph) };
        const next = { nodes: nodesOf(r.graph), edges: edgesOf(r.graph) };
        applyDiffToGraph(sigma, first.graph, next, diffGraphs(prev, next));
      },
      debounceMs: 2000,
    });

    const sidebar = renderSidebar(sidebarEl, {
      stats: { node_count: first.graph.order, edge_count: first.graph.size },
      onFilterChange: (f) => { currentFilter = f; invalidate(); },
      onCurationChange: (c) => { currentCuration = c; invalidate(); },
      onExportSubgraph: () => {
        const isVisible = (id) => !state.manuallyHidden.has(id);
        const xml = exportVisibleAsGexf(
          first.graph,
          graphology.Graph,
          graphologyLibrary.gexf.write,
          isVisible,
        );
        download(xml, 'subgraph.gexf', 'application/gexf+xml');
      },
      onExportIds: () => {
        const ids = [];
        first.graph.forEachNode((id) => {
          if (!state.manuallyHidden.has(id)) ids.push(id);
        });
        download(exportIdsAsText(ids), 'ids.txt', 'text/plain');
      },
    });

    const sse = createSseClient({
      url: apiClient.sseUrl(mode.userId),
      onMessage: () => refetcher.trigger(),
      onReconnect: () => refetcher.trigger(),
      onStatusChange: (s) => sidebar.setLiveStatus(s),
    });
    sse.connect();
  } else if (mode.mode === 'snapshot-remote') {
    const resp = await fetch(mode.file);
    if (!resp.ok) throw new Error(`snapshot fetch failed: ${resp.status}`);
    const text = await resp.text();
    const graph = await loader.parseFromText(text);
    mount(graph, container);
    renderSidebar(sidebarEl, {
      stats: { node_count: graph.order, edge_count: graph.size },
      onFilterChange: () => {},
    });
  } else {
    renderSidebar(sidebarEl, {
      stats: { node_count: 0, edge_count: 0 },
      onFilterChange: () => {},
    });
    renderDropZone(container, async (text) => {
      const graph = await loader.parseFromText(text);
      mount(graph, container);
    });
  }
}

main().catch((e) => {
  console.error('[boot] failed', e);
  const banner = document.createElement('div');
  banner.className = 'banner';
  banner.textContent = 'Failed to load: ' + String(e.message || e);
  document.body.appendChild(banner);
});
