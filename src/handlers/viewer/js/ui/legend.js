export function renderLegend(container) {
  container.innerHTML = `
    <div class="legend">
      <h4>Nodes</h4>
      <ul>
        <li><span class="swatch swatch-circle" style="background:#ff6b2c"></span> Memory (working)</li>
        <li><span class="swatch swatch-circle" style="background:#f5b73b"></span> Memory (session)</li>
        <li><span class="swatch swatch-circle" style="background:#4b8bb5"></span> Memory (longterm)</li>
        <li><span class="swatch swatch-square" style="background:#7d74c9"></span> Entity</li>
        <li><span class="swatch swatch-diamond" style="background:#39a887"></span> Episode</li>
      </ul>
      <h4>Edges</h4>
      <ul>
        <li><span class="line line-l1"></span> L1 / Working</li>
        <li><span class="line line-l2"></span> L2 / Episodic</li>
        <li><span class="line line-l3"></span> L3 / Semantic</li>
        <li><span class="line line-dashed"></span> LTP pending</li>
        <li>Thickness ∝ Hebbian weight</li>
        <li>Pulse ⇒ fired in last 5 s</li>
      </ul>
    </div>
  `;
}
