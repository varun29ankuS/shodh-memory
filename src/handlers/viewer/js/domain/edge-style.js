// Tier hue table. Warm → cool = recent/working → established/semantic.
const TIER_HUE = {
  L1Working:  '#d33b2c',
  L2Episodic: '#e69f12',
  L3Semantic: '#3c78b5',
};
const DEFAULT_HUE = '#888';
const PULSE_WINDOW_MS = 5000;
const MIN_THICKNESS_PX = 0.5;
const MAX_THICKNESS_PX = 5.0;
const JUST_PROMOTED_SIZE_MULT = 1.4;

/**
 * Map a graph edge's attributes to a sigma reducer result.
 * Pure function; call it from the sigma reducer with the current clock.
 *
 * @param {string} id
 * @param {object} attrs  graphology edge attributes
 * @param {{now:number}} ctx  ctx.now is ms since epoch
 * @returns {object} sigma edge spec: {size,color,type,pulse,label}
 */
export function edgeReducer(id, attrs, ctx) {
  const weight = typeof attrs.weight === 'number' ? attrs.weight : 0.5;
  const clamped = Math.min(1, Math.max(0, weight));
  const baseSize = MIN_THICKNESS_PX + (MAX_THICKNESS_PX - MIN_THICKNESS_PX) * clamped;
  const color = TIER_HUE[attrs.tier] || DEFAULT_HUE;

  const type = attrs.ltp_status === 'Pending' ? 'dashed' : 'line';
  const emphasized = attrs.ltp_status === 'JustPromoted';

  let pulse = false;
  if (attrs.last_activated) {
    const ts = Date.parse(attrs.last_activated);
    if (!Number.isNaN(ts) && (ctx.now - ts) < PULSE_WINDOW_MS) pulse = true;
  }

  return {
    size: emphasized ? baseSize * JUST_PROMOTED_SIZE_MULT : baseSize,
    color,
    type,
    pulse,
    label: attrs.label || attrs.relation_type || '',
  };
}
