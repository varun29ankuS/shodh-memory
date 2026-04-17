const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

export function isWeak(attrs) {
  return (attrs.importance || 0) < 0.2 && (attrs.access_count || 0) < 5;
}

export function isOrphan(graph, id) {
  return graph.degree(id) === 0;
}

export function isDeadEdge(attrs, now = Date.now()) {
  if ((attrs.activation_count || 0) === 0) return true;
  if (attrs.last_activated) {
    const ts = Date.parse(attrs.last_activated);
    if (!Number.isNaN(ts) && (now - ts) > WEEK_MS) return true;
  }
  return false;
}
