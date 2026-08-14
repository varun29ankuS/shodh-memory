import { useEffect, useRef } from "react";
import { geoEquirectangular, geoPath, type GeoPermissibleObjects } from "d3";
import { useGround } from "@/lib/ground";

/**
 * A landmass rendered as a dot matrix.
 *
 * WHY DOTS AND NOT A COASTLINE. These maps are 300-odd pixels wide inside a
 * column of a briefing, and at that size a stroked coastline turns to mud —
 * every fjord and island becomes one grey smear and the shape stops being
 * recognisable, which is the only job a map this small has. Filling the land
 * to an offscreen buffer, sampling it on a fixed grid and painting the hits as
 * dots is how e-ink renders a fill, and it degrades in the right direction:
 * the outline coarsens rather than blurring, so a continent still reads as
 * itself. It is also the same texture as the paper tooth behind the page, so
 * the map sits ON the sheet instead of on top of it.
 *
 * The grid is in CSS pixels and so is the sampling buffer. Sampling at device
 * resolution would put more dots on a retina screen than on a laptop and the
 * map would render at a different density per machine — the matrix is a design
 * decision, not a function of the display.
 *
 * NO NETWORK, EVER. Both geometries are vendored assets decoded in-process
 * (see GeoMap.tsx for their provenance and licences). The whole product ships
 * as one embedded index.html, so a tile request would break the offline
 * guarantee the first time it ran somewhere without egress.
 *
 * EQUIRECTANGULAR, deliberately, where the analyst surface at /geo is Equal
 * Earth. That surface answers "how far apart are these really" and needs equal
 * area; this one is a shape to be recognised at a glance in a fixed column, and
 * a plate carrée keeps the graticule square so the dot grid lands on a regular
 * lattice rather than on a curved one. Equal Earth's bowed parallels put the
 * sample points off the lattice and the matrix stops reading as a matrix.
 */

/** One place, already aggregated. `count` is how many memories are there —
 *  marks are sized by its square root, so ten memories at one site reads as
 *  larger than one and not as ten times larger. */
export interface DotMapPoint {
  lon: number;
  lat: number;
  count: number;
}

/** `[[west, south], [east, north]]`, in degrees. */
export type DotMapExtent = [[number, number], [number, number]];

/**
 * `#rrggbb` (or `#rgb`) at an alpha, as a canvas colour.
 *
 * Every colour on this canvas is READ FROM A TOKEN — there is no literal hue
 * in this file — and the token ramps do not include every value the drawing
 * wants. The dot ink is one step lighter than `--muted-foreground` on paper
 * and one step darker on night; composited alpha over the ground gets there
 * from the token itself rather than by adding a token that exists for one
 * canvas. An unparseable value is returned unchanged rather than replaced by a
 * literal, so a token that ever stops being hex degrades to full opacity
 * instead of to a hardcoded grey.
 */
function withAlpha(color: string, alpha: number): string {
  const short = /^#([\da-f])([\da-f])([\da-f])$/i.exec(color);
  const hex = short ? `#${short[1]}${short[1]}${short[2]}${short[2]}${short[3]}${short[3]}` : color;
  const m = /^#([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex);
  if (!m) return color;
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${alpha})`;
}

/** Composited over the ground this lands within a shade of the mockup's
 *  fourth ink step on both grounds — present enough to hold a shape, quiet
 *  enough that the marks on top of it are never in competition. */
const INK_ALPHA = 0.6;

/** Device pixels per CSS pixel, capped. Past 2 the extra resolution is not
 *  visible at a 1.4px dot and the buffer allocation grows quadratically. */
const MAX_DPR = 2;

export function DotMap({
  shapes,
  extent,
  points,
  cell,
}: {
  /** Filled into the sampling buffer in order, and unioned by construction —
   *  overlapping fills are both opaque, so drawing India's own boundary after
   *  Natural Earth's land is what makes the world map inherit the correction
   *  rather than needing its own basemap. */
  shapes: GeoPermissibleObjects[];
  extent: DotMapExtent;
  points: DotMapPoint[];
  /** Grid pitch in CSS pixels. Smaller means a finer coastline and more dots;
   *  the world can take a coarser grid than a single country because it is
   *  showing continents rather than a national outline. */
  cell: number;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Not used in the drawing, but the tokens it reads are a function of it: a
  // ground change has to repaint a canvas, which no amount of CSS reaches.
  const { ground } = useGround();

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const [[west, south], [east, north]] = extent;
    const spanLon = east - west;
    const spanLat = north - south;
    if (!(spanLon > 0) || !(spanLat > 0)) return;
    // A plate carrée is linear in both axes, so the frame's aspect IS the
    // extent's aspect. Deriving the height rather than accepting one means the
    // projection never letterboxes and the dot grid stays square.
    const aspect = spanLat / spanLon;

    // The extent as a geometry, so d3 does the fitting. A two-corner MultiPoint
    // has exactly the bounding box asked for.
    const frame = {
      type: "MultiPoint",
      coordinates: [
        [west, south],
        [east, north],
      ],
    } as unknown as GeoPermissibleObjects;

    function draw() {
      const w = Math.max(1, Math.round(wrap!.clientWidth));
      const h = Math.max(1, Math.round(w * aspect));
      const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);

      canvas!.width = Math.round(w * dpr);
      canvas!.height = Math.round(h * dpr);
      canvas!.style.height = `${h}px`;

      const ctx = canvas!.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const projection = geoEquirectangular().fitExtent(
        [
          [0, 0],
          [w, h],
        ],
        frame,
      );

      // The stencil. Never painted, never seen: only its alpha channel is
      // read, which is why no fill colour is set — the canvas default is
      // opaque black and any opaque value would do.
      const buffer = document.createElement("canvas");
      buffer.width = w;
      buffer.height = h;
      const bctx = buffer.getContext("2d", { willReadFrequently: true });
      if (!bctx) return;
      const stencil = geoPath(projection, bctx);
      for (const shape of shapes) {
        bctx.beginPath();
        stencil(shape);
        bctx.fill();
      }
      const alpha = bctx.getImageData(0, 0, w, h).data;

      const cs = getComputedStyle(document.documentElement);
      const token = (name: string) => cs.getPropertyValue(name).trim();
      const ink = token("--muted-foreground");
      const groundColour = token("--background");
      const mark = token("--primary");

      ctx.fillStyle = withAlpha(ink, INK_ALPHA);
      for (let y = cell / 2; y < h; y += cell) {
        for (let x = cell / 2; x < w; x += cell) {
          const idx = (Math.floor(y) * w + Math.floor(x)) * 4;
          if (alpha[idx + 3] > 128) {
            // Offset by half a pixel so the dot lands on the device grid
            // instead of straddling two rows and rendering as a soft smudge.
            ctx.fillRect(Math.round(x) - 0.5, Math.round(y) - 0.5, 1.4, 1.4);
          }
        }
      }

      // Located memories, as filled marks. The halo is the GROUND colour, not
      // a white: a mark sitting on the dot matrix has to stay a disc rather
      // than merging into the texture, and painting it in a paper literal
      // would leave a light ring around every point on the night ground.
      for (const p of points) {
        const xy = projection([p.lon, p.lat]);
        if (!xy) continue;
        const r = 1.8 + Math.sqrt(Math.max(1, p.count)) * 0.55;
        ctx.beginPath();
        ctx.arc(xy[0], xy[1], r + 1.6, 0, Math.PI * 2);
        ctx.fillStyle = groundColour;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(xy[0], xy[1], r, 0, Math.PI * 2);
        ctx.fillStyle = mark;
        ctx.fill();
      }
    }

    draw();

    const observer = new ResizeObserver(() => draw());
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [shapes, extent, points, cell, ground]);

  return (
    <div ref={wrapRef} className="relative">
      {/* The door around this canvas carries the text alternative with the real
          counts. A canvas labelled as well would announce the same map twice
          and give a screen reader a control it cannot use. */}
      <canvas ref={canvasRef} aria-hidden="true" className="block h-auto w-full" />
    </div>
  );
}
