// Color tables. Warm → cool = hot → established.
const TIER_COLOR = {
  Working:  '#ff6b2c',
  Session:  '#f5b73b',
  Longterm: '#4b8bb5',
};
const ENTITY_COLOR = '#7d74c9';
const EPISODE_COLOR = '#39a887';
const DEFAULT_COLOR = '#888';

/**
 * Map a graph node's attributes to a sigma reducer result.
 * Pure function; call it from the sigma reducer with the current clock.
 *
 * @param {string} id
 * @param {object} attrs  graphology node attributes
 * @param {{now:number}} ctx  ctx.now is ms since epoch
 * @returns {object} sigma node spec: {size,color,shape,...,haloPulse}
 */
export function nodeReducer(id, attrs, ctx) {
  const type = attrs.type || 'memory';
  const importance = typeof attrs.importance === 'number' ? attrs.importance : 0.5;
  const activation = typeof attrs.activation === 'number' ? attrs.activation : 0.0;

  const size = 6 + 18 * Math.min(1, Math.max(0, importance));
  let color = DEFAULT_COLOR;
  let shape = 'circle';
  if (type === 'memory') {
    color = TIER_COLOR[attrs.tier] || DEFAULT_COLOR;
  } else if (type === 'entity') {
    color = ENTITY_COLOR;
    shape = 'square';
  } else if (type === 'episode') {
    color = EPISODE_COLOR;
    shape = 'diamond';
  }

  const haloPulse = activation > 0.7;

  // Recency badge: last_accessed within 60s
  let recencyBadge = false;
  if (attrs.last_accessed) {
    const ts = Date.parse(attrs.last_accessed);
    if (!Number.isNaN(ts) && (ctx.now - ts) < 60_000) recencyBadge = true;
  }

  return { size, color, shape, haloPulse, recencyBadge, type, label: attrs.label || id };
}
