/**
 * Framing a point set into a viewport.
 *
 * The field's rest state is the whole corpus, framed. Both canvases previously
 * started at `zoomIdentity` and never computed a fit, so a graph wider than the
 * force layout's centre — which is most of them — opened with nodes clipped off
 * every edge. Verified on gdelt-bridge: 136 entities, nodes cut off the top,
 * right and bottom on first paint.
 *
 * Kept free of d3 so both canvases share it and so it tests in vitest's default
 * node environment. The caller turns the result into a `ZoomTransform` — see
 * the note at the call site about applying it through the zoom behaviour rather
 * than assigning the ref.
 */

export interface Extent {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface Viewport {
  width: number;
  height: number;
}

export interface FitOptions {
  /** Breathing room in screen pixels on every side. */
  padding?: number;
  /** The canvas's own d3 scaleExtent, so the fit cannot produce a scale d3 would clamp. */
  scaleExtent?: [number, number];
}

/**
 * The bounding box of every point that has both coordinates.
 *
 * d3-force nodes are `SimulationNodeDatum`, whose `x` and `y` are optional and
 * genuinely absent before the first tick. A partial point is skipped rather
 * than coerced, because treating a missing coordinate as 0 drags the extent to
 * the origin and frames a box the nodes are not in.
 */
export function extentOf(points: readonly { x?: number; y?: number }[]): Extent | null {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let seen = false;

  for (const p of points) {
    if (typeof p.x !== "number" || typeof p.y !== "number") continue;
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    seen = true;
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  return seen ? { minX, minY, maxX, maxY } : null;
}

/**
 * The extent of the bulk of a point set, ignoring a fraction at each edge.
 *
 * A force layout pushes weakly-connected nodes far from the mass, so the raw
 * extent is set by whichever handful of nodes drifted furthest. Framing that
 * squashes the part anyone came to look at into an illegible knot in the
 * middle — verified on gdelt-bridge, where about twenty degree-0 entities
 * (hashes, bare floats, stray tokens) owned the camera for all 136.
 *
 * `q` is trimmed from each end of each axis independently, so a node extreme
 * in x but ordinary in y only forfeits its x. Trimmed points are still drawn —
 * they simply sit outside the opening frame and are reached by zooming out.
 *
 * q = 0 is exactly `extentOf`.
 */
export function coreExtentOf(
  points: readonly { x?: number; y?: number }[],
  q = 0.05,
): Extent | null {
  const xs: number[] = [];
  const ys: number[] = [];
  for (const p of points) {
    if (typeof p.x !== "number" || typeof p.y !== "number") continue;
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    xs.push(p.x);
    ys.push(p.y);
  }
  if (xs.length === 0) return null;

  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);

  // With few points, trimming would discard a meaningful share of a set that
  // has no outlier problem to solve. Below the threshold the raw extent is
  // both correct and safer.
  const clamped = Math.min(Math.max(q, 0), 0.49);
  const cut = xs.length < 20 ? 0 : Math.floor(xs.length * clamped);
  const last = xs.length - 1 - cut;

  return { minX: xs[cut], maxX: xs[last], minY: ys[cut], maxY: ys[last] };
}

/**
 * The transform that frames `extent` inside `viewport`.
 *
 * Screen position is `p * k + offset`, matching d3's `ZoomTransform.apply`.
 */
export function fitTransform(
  extent: Extent,
  viewport: Viewport,
  { padding = 0, scaleExtent }: FitOptions = {},
): { k: number; x: number; y: number } {
  const spanX = extent.maxX - extent.minX;
  const spanY = extent.maxY - extent.minY;

  // A viewport smaller than its own padding would ask for a non-positive
  // available box and produce a negative or non-finite scale. Clamp to a
  // positive floor: an unmeasured container (width 0 before layout) must still
  // yield a usable transform rather than NaN painted onto the canvas.
  const availableWidth = Math.max(viewport.width - padding * 2, 1);
  const availableHeight = Math.max(viewport.height - padding * 2, 1);

  const scaleX = spanX > 0 ? availableWidth / spanX : Number.POSITIVE_INFINITY;
  const scaleY = spanY > 0 ? availableHeight / spanY : Number.POSITIVE_INFINITY;

  // Both infinite means a single point, or a set collapsed onto one position:
  // there is no span to fit, so a scale is meaningless and 1 is honest.
  let k = Math.min(scaleX, scaleY);
  if (!Number.isFinite(k) || k <= 0) k = 1;

  if (scaleExtent) {
    const [lo, hi] = scaleExtent;
    k = Math.min(Math.max(k, lo), hi);
  }

  const centreX = (extent.minX + extent.maxX) / 2;
  const centreY = (extent.minY + extent.maxY) / 2;

  return {
    k,
    x: viewport.width / 2 - k * centreX,
    y: viewport.height / 2 - k * centreY,
  };
}
