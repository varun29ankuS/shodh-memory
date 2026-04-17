const LIVE_LABELS = {
  connecting:    'Connecting\u2026',
  connected:     'Live',
  disconnected:  'Disconnected',
  closed:        'Closed',
};

const LIVE_CLASS = {
  connecting:    'disconnected',
  connected:     'connected',
  disconnected:  'disconnected',
  closed:        'disconnected',
};

/**
 * Render shodh-memory filter controls, a stats panel, and a live-status
 * indicator into `container`.
 *
 * @param {HTMLElement} container  The #sidebar element (or any wrapper).
 * @param {{ onFilterChange: (state: object) => void, stats?: { node_count?: number, edge_count?: number } }} opts
 * @returns {{ setLiveStatus: (status: 'connecting'|'connected'|'disconnected'|'closed') => void }}
 */
export function renderSidebar(container, { onFilterChange, onCurationChange, stats }) {
  // ------------------------------------------------------------------
  // Filter state — shape matches matchesFilters expectations.
  // activeTiers holds BOTH memory tiers and edge tiers in one Set.
  // ------------------------------------------------------------------
  const filterState = {
    activeTiers:    new Set(['Working', 'Session', 'Longterm', 'L1Working', 'L2Episodic', 'L3Semantic']),
    activeTypes:    new Set(['memory', 'entity', 'episode']),
    activeLtp:      new Set(['None', 'Pending', 'Consolidated', 'JustPromoted']),
    minActivation:  0,
    minWeight:      0,
    recencyWindowMs: null,
  };

  function emit() {
    onFilterChange({
      activeTiers:     new Set(filterState.activeTiers),
      activeTypes:     new Set(filterState.activeTypes),
      activeLtp:       new Set(filterState.activeLtp),
      minActivation:   filterState.minActivation,
      minWeight:       filterState.minWeight,
      recencyWindowMs: filterState.recencyWindowMs,
    });
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function section(title) {
    const sec = document.createElement('section');
    const h2 = document.createElement('h2');
    h2.textContent = title;
    sec.appendChild(h2);
    return sec;
  }

  function checkGroup(items, getSet) {
    const wrap = document.createElement('div');
    wrap.className = 'filter-checks';
    items.forEach(({ value, display }) => {
      const lbl = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.addEventListener('change', () => {
        if (cb.checked) { getSet().add(value); } else { getSet().delete(value); }
        emit();
      });
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode('\u00a0' + (display || value)));
      wrap.appendChild(lbl);
    });
    return wrap;
  }

  function sliderRow(labelText, min, max, step, initial, onChange) {
    const row = document.createElement('div');
    row.className = 'setting-row';
    const lbl = document.createElement('label');
    const span = document.createElement('span');
    span.textContent = initial.toFixed(2);
    lbl.appendChild(document.createTextNode(labelText));
    lbl.appendChild(span);
    row.appendChild(lbl);
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(initial);
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      span.textContent = v.toFixed(2);
      onChange(v);
      emit();
    });
    row.appendChild(slider);
    return row;
  }

  // ------------------------------------------------------------------
  // Build DOM
  // ------------------------------------------------------------------
  container.innerHTML = '';

  // Title
  const title = document.createElement('h1');
  title.textContent = 'shodh-memory';
  container.appendChild(title);

  // Live status
  const liveSpan = document.createElement('span');
  liveSpan.className = 'live-indicator disconnected';
  liveSpan.id = 'live-status';
  liveSpan.textContent = LIVE_LABELS.disconnected;
  container.appendChild(liveSpan);

  // Stats panel
  const statsSec = section('Graph');
  const statsEl = document.createElement('div');
  statsEl.id = 'graph-stats';
  const nc = stats && stats.node_count != null ? String(Number(stats.node_count)) : '\u2014';
  const ec = stats && stats.edge_count  != null ? String(Number(stats.edge_count))  : '\u2014';
  statsEl.textContent = nc + ' nodes \u00b7 ' + ec + ' edges';
  statsSec.appendChild(statsEl);
  container.appendChild(statsSec);

  // Node types
  const typeSec = section('Node Types');
  typeSec.appendChild(checkGroup(
    [{ value: 'memory' }, { value: 'entity' }, { value: 'episode' }],
    () => filterState.activeTypes,
  ));
  container.appendChild(typeSec);

  // Memory tiers
  const memTierSec = section('Memory Tiers');
  memTierSec.appendChild(checkGroup(
    [{ value: 'Working' }, { value: 'Session' }, { value: 'Longterm' }],
    () => filterState.activeTiers,
  ));
  container.appendChild(memTierSec);

  // Edge tiers
  const edgeTierSec = section('Edge Tiers');
  edgeTierSec.appendChild(checkGroup(
    [{ value: 'L1Working' }, { value: 'L2Episodic' }, { value: 'L3Semantic' }],
    () => filterState.activeTiers,
  ));
  container.appendChild(edgeTierSec);

  // LTP status
  const ltpSec = section('LTP Status');
  ltpSec.appendChild(checkGroup(
    [{ value: 'None' }, { value: 'Pending' }, { value: 'Consolidated' }, { value: 'JustPromoted', display: 'Just Promoted' }],
    () => filterState.activeLtp,
  ));
  container.appendChild(ltpSec);

  // Min activation slider
  const actSec = section('Min Activation');
  actSec.appendChild(sliderRow('', 0, 1, 0.05, 0, v => { filterState.minActivation = v; }));
  container.appendChild(actSec);

  // Min weight slider
  const wtSec = section('Min Weight');
  wtSec.appendChild(sliderRow('', 0, 1, 0.05, 0, v => { filterState.minWeight = v; }));
  container.appendChild(wtSec);

  // Recency window
  const recencySec = section('Recency');
  const recencyRow = document.createElement('div');
  recencyRow.className = 'setting-row';
  const recencySelect = document.createElement('select');
  [
    { label: 'All time',   value: '' },
    { label: 'Last hour',  value: '3600000' },
    { label: 'Last day',   value: '86400000' },
    { label: 'Last week',  value: '604800000' },
  ].forEach(({ label, value }) => {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    recencySelect.appendChild(opt);
  });
  recencySelect.addEventListener('change', () => {
    filterState.recencyWindowMs = recencySelect.value === '' ? null : parseInt(recencySelect.value, 10);
    emit();
  });
  recencyRow.appendChild(recencySelect);
  recencySec.appendChild(recencyRow);
  container.appendChild(recencySec);

  // Curation toggles
  const curationState = { showWeak: false, showOrphans: false, showDeadEdges: false };

  function emitCuration() {
    if (onCurationChange) {
      onCurationChange({
        showWeak:      curationState.showWeak,
        showOrphans:   curationState.showOrphans,
        showDeadEdges: curationState.showDeadEdges,
      });
    }
  }

  function boolCheckbox(labelText, key) {
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = false;
    cb.addEventListener('change', () => {
      curationState[key] = cb.checked;
      emitCuration();
    });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode('\u00a0' + labelText));
    return lbl;
  }

  const curationSec = section('Curation');
  const curationWrap = document.createElement('div');
  curationWrap.className = 'filter-checks';
  curationWrap.appendChild(boolCheckbox('Highlight weak',       'showWeak'));
  curationWrap.appendChild(boolCheckbox('Highlight orphans',    'showOrphans'));
  curationWrap.appendChild(boolCheckbox('Highlight dead edges', 'showDeadEdges'));
  curationSec.appendChild(curationWrap);
  container.appendChild(curationSec);

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------
  return {
    setLiveStatus(status) {
      liveSpan.textContent = LIVE_LABELS[status] || status;
      liveSpan.className = 'live-indicator ' + (LIVE_CLASS[status] || 'disconnected');
    },
  };
}
