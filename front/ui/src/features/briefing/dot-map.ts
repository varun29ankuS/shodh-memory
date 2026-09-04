/**
 * Dot-matrix map renderer.
 *
 * Landmasses are filled into an off-screen buffer and then *sampled* on a fixed
 * grid, so what you see is a mesh of dots rather than a filled shape. That is
 * not decoration: at briefing size a filled choropleth reads as a solid blob and
 * a stroked outline reads as noise, while a mesh stays legible at 240px wide and
 * makes located memories — drawn as solid discs on top — the only continuous
 * marks on the canvas. The eye goes to them first, which is the point.
 *
 * Every colour is read from CSS custom properties at draw time, so the map
 * follows the Paper/Night ground with no palette passed in and no second source
 * of truth. Callers must redraw on theme change; `DotMap` does.
 */

import type { Bounds, Ring } from "./geo-shapes";

/** A located memory: longitude, latitude, and how many memories sit there. */
export type PlacedPoint = [lon: number, lat: number, count: number];

/** Device pixel ratio is capped: past 2x the dots stop resolving and we just pay. */
const MAX_DPR = 2;

/** Alpha above which a sampled pixel counts as land. */
const LAND_ALPHA = 128;

export type DrawDotMapOptions = {
  canvas: HTMLCanvasElement;
  shapes: Ring[];
  bounds: Bounds;
  points: PlacedPoint[];
  /** Dot pitch in CSS px. */
  cell: number;
};

/**
 * Draws one map. Sizes the canvas from its own `clientWidth`, so the caller
 * controls the width through CSS and the height follows the bounds' aspect.
 *
 * Returns the CSS height it settled on, which the caller can use to reserve
 * space and avoid a layout jump on first paint.
 */
export function drawDotMap({
  canvas,
  shapes,
  bounds,
  points,
  cell,
}: DrawDotMapOptions): number {
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
  const w = canvas.clientWidth || 240;
  const h = Math.round(w * (bounds.h / bounds.w));

  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext("2d");
  if (!ctx) return h;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // Equirectangular. Equal-area matters on a globe; at this size, on a window
  // this narrow, the distortion is below the dot pitch and the cheaper
  // projection keeps the maths inspectable.
  const project = (lon: number, lat: number): [number, number] => [
    ((lon - bounds.x) / bounds.w) * w,
    ((bounds.y - lat) / bounds.h) * h,
  ];

  // Fill the landmasses off-screen, then sample. Filling and sampling separately
  // is what lets a dot sit exactly on the grid regardless of the coastline.
  const buffer = document.createElement("canvas");
  buffer.width = w;
  buffer.height = h;
  const bx = buffer.getContext("2d");
  if (!bx) return h;

  bx.fillStyle = "#000";
  for (const ring of shapes) {
    bx.beginPath();
    ring.forEach((pt, i) => {
      const [x, y] = project(pt[0], pt[1]);
      if (i === 0) bx.moveTo(x, y);
      else bx.lineTo(x, y);
    });
    bx.closePath();
    bx.fill();
  }
  const { data } = bx.getImageData(0, 0, w, h);

  const styles = getComputedStyle(document.documentElement);
  const token = (name: string, fallback: string) =>
    styles.getPropertyValue(name).trim() || fallback;

  const ink = token("--paper-ink-4", "#9a9484");
  const outline = token("--paper-ink-3", "#6d6959");
  const ground = token("--paper-bg", "#e9e6dc");
  const accent = token("--paper-accent", "#c0391a");

  ctx.fillStyle = ink;
  for (let y = cell / 2; y < h; y += cell) {
    for (let x = cell / 2; x < w; x += cell) {
      const idx = (Math.floor(y) * w + Math.floor(x)) * 4;
      if (data[idx + 3] > LAND_ALPHA) {
        // The half-pixel offset lands the 1.4px dot on the device grid; without
        // it every dot is antialiased across two pixels and the mesh goes grey.
        ctx.fillRect(Math.round(x) - 0.5, Math.round(y) - 0.5, 1.4, 1.4);
      }
    }
  }

  // The silhouette, drawn over the mesh.
  //
  // A dot matrix carries density well and shape badly: at this size India is
  // about 114 cells across, and the peninsula, the Kutch and the north-east
  // arm all dissolve into texture. The pitch is in CSS pixels, so a retina
  // screen makes the dots sharper without making them more numerous — the
  // shape does not come back by itself.
  //
  // A hairline over the top restores the outline at any size while the mesh
  // keeps doing what it is good at. Stroked after the fill so the coast reads
  // as an edge rather than as another row of dots.
  // Split at the antimeridian. Two of the world rings -- the Eurasia/Americas
  // landmass and Antarctica -- run a full 360 degrees of longitude, because a
  // ring that crosses 180E re-enters at 180W. Drawn segment by segment in an
  // equirectangular projection that crossing is a jump the width of the plate,
  // so a naive stroke lays a straight line edge to edge across the map.
  //
  // The FILL never showed this, which is why the shapes shipped looking fine:
  // the sampler asks whether a cell centre is inside the path, and a chord of
  // zero area contains nothing. A stroke draws every segment it is given.
  //
  // So a segment jumping more than half the world starts a new sub-path, and
  // only a ring that never wrapped is closed -- closing a wrapped one would
  // reintroduce the same chord between the two loose ends.
  ctx.strokeStyle = outline;
  ctx.lineWidth = 0.6;
  ctx.lineJoin = "round";
  for (const ring of shapes) {
    let wrapped = false;
    ctx.beginPath();
    for (let i = 0; i < ring.length; i++) {
      const [x, y] = project(ring[i][0], ring[i][1]);
      if (i > 0 && Math.abs(ring[i][0] - ring[i - 1][0]) > 180) {
        wrapped = true;
        ctx.moveTo(x, y);
      } else if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    if (!wrapped) ctx.closePath();
    ctx.stroke();
  }

  // Located memories, as discs area-scaled by count. The halo is the *ground*
  // colour rather than a paper literal, so a mark stays a disc instead of
  // merging into the mesh — and it has to follow the theme to keep doing that.
  for (const [lon, lat, count] of points) {
    const [x, y] = project(lon, lat);
    const r = 1.8 + Math.sqrt(count) * 0.55;

    ctx.beginPath();
    ctx.arc(x, y, r + 1.6, 0, Math.PI * 2);
    ctx.fillStyle = ground;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = accent;
    ctx.fill();
  }

  return h;
}
