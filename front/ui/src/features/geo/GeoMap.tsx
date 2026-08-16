import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  geoBounds,
  geoEqualEarth,
  geoPath,
  geoGraticule10,
  select,
  zoom,
  zoomIdentity,
  type GeoPermissibleObjects,
  type ZoomTransform,
} from "d3";
import type { RecallMemory } from "@/lib/api";
import { useSession } from "@/stores/session";
// Decoded once for the whole product — see lib/atlas.ts for the provenance of
// both files and for why India is never drawn from Natural Earth.
import { LAND, BORDERS, INDIA } from "@/lib/atlas";

/**
 * The world basemap and the plotted points.
 *
 * NO TILE SERVER, NO NETWORK. The whole app ships as one self-contained
 * index.html embedded in the Rust binary (front/src/main.rs), which is the
 * reason vite.config.ts inlines every asset; a map that fetched tiles would
 * break that guarantee the first time it ran somewhere without egress, and
 * quietly leak the fact that someone is looking at a place. The basemap is
 * therefore vendored: src/assets/world-countries-50m.json, Natural Earth
 * 1:50m via world-atlas (its licence sits beside it), 756 kB of quantised
 * TopoJSON decoded in-process.
 *
 * 1:50m, not 1:110m, and not 1:10m. The map is fitted to the CORPUS, so a
 * regional view is the normal case, and 110m is a world dataset -- 8,246
 * points for the whole planet, which renders Sri Lanka and the Gujarat coast
 * as visible facets at India zoom. 50m carries 80,617 points, ten times the
 * detail for seven times the bytes.
 *
 * 10m exists (477,295 points, 3.66 MB) and was rejected on render cost rather
 * than size: this canvas repaints the entire basemap on every pan and zoom
 * frame, and half a million points per frame drops frames on the one surface
 * most likely to be driven live in front of someone.
 *
 * Equal Earth, not Mercator. This is an analyst surface where the question is
 * "where did this happen and what else is near it"; Mercator inflates high
 * latitudes by a factor of several and would make a cluster of northern points
 * look far more spread out than a matching cluster near the equator. Equal
 * Earth is equal-area, so density on screen is density in the world.
 *
 * COORDINATE ORDER IS A REAL HAZARD HERE. The wire carries `[lat, lon, alt]`
 * (src/validation.rs:293-294, whose own test uses San Francisco as
 * [37.7749, -122.4194]), and every d3-geo entry point takes [lon, lat]. The
 * swap happens exactly once, in `project`, so no other code in this file can
 * get it wrong — transposed coordinates do not throw, they silently plot
 * Baltimore in China.
 */

const GRATICULE = geoGraticule10() as unknown as GeoPermissibleObjects;

interface GeoPoint {
  id: string;
  label: string;
  lat: number;
  lon: number;
  type: string | null;
  score: number;
}

interface Hover {
  point: GeoPoint;
  x: number;
  y: number;
}

function readTokens(el: HTMLElement) {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    chart: [
      v("--chart-1", "#a599ff"),
      v("--chart-2", "#4ea7fc"),
      v("--chart-3", "#4cb782"),
      v("--chart-4", "#ec6f9e"),
      v("--chart-5", "#39b8b0"),
    ],
    active: v("--node-active", "#f4622e"),
    muted: v("--muted-foreground", "#8a8f98"),
    border: v("--border", "#23252a"),
  };
}

function hexA(hex: string, a: number): string {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex.trim());
  if (!m) return `rgba(138,143,152,${a})`;
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
}

const HIT_RADIUS_PX = 9;

export function GeoMap({
  memories,
  types,
  dimmed,
}: {
  memories: RecallMemory[];
  types: string[];
  /** Ids drawn as quiet context — present on the map, visibly not part of the
   *  current answer. Absent set = every point is a first-class result. */
  dimmed?: Set<string>;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const selectedId = useSession((s) => s.selectedMemoryId);
  const selectMemory = useSession((s) => s.select);
  const [hover, setHover] = useState<Hover | null>(null);

  const transformRef = useRef<ZoomTransform>(zoomIdentity);
  const selectedRef = useRef<string | null>(selectedId);
  const drawRef = useRef<() => void>(() => {});
  /** Screen positions of the plotted points, refreshed on every paint so the
   *  hit-test never has to re-project or duplicate the projection's state. */
  const screenRef = useRef<Array<{ point: GeoPoint; x: number; y: number }>>([]);

  const points = useMemo<GeoPoint[]>(
    () =>
      memories.flatMap((m) => {
        const geo = m.experience.geo_location;
        if (!geo) return [];
        const [lat, lon] = geo;
        // A memory whose coordinates are out of range is corrupt, not
        // interesting — plotting it would put a point on a meaningless part of
        // the map and imply the data is fine.
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return [];
        if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return [];
        return [
          {
            id: m.id,
            label: m.experience.content,
            lat,
            lon,
            type: m.experience.memory_type,
            score: m.score,
          },
        ];
      }),
    [memories],
  );

  useEffect(() => {
    selectedRef.current = selectedId;
    drawRef.current();
  }, [selectedId]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const tokens = readTokens(wrap);
    const hueFor = (t: string | null) =>
      t && types.indexOf(t) >= 0 ? tokens.chart[types.indexOf(t) % tokens.chart.length] : tokens.muted;

    let width = 0;
    let height = 0;
    const projection = geoEqualEarth();
    /* The basemap is re-tessellated ONCE PER FIT, not once per frame.

       Zoom and pan are applied as a canvas transform, so the projected
       geometry does not change while you drag -- yet every frame was calling
       geoPath and re-projecting the whole world. At 1:110m that was 8,246
       points and got away with it; at 1:50m it is 80,617, five times over
       (graticule, land, borders, the India clip and India's own outline), and
       the map went sluggish under the hand.

       geoPath without a context returns an SVG path string, which Path2D
       accepts. Building them at fit time and replaying them per frame removes
       the projection from the hot path entirely. They are rebuilt whenever the
       projection changes -- which is a resize or a new fit, never a gesture. */
    const pathString = geoPath(projection);
    let basemap: {
      graticule: Path2D;
      land: Path2D;
      borders: Path2D;
      india: Path2D;
      /** Full plane minus India's interior, for the de-facto border clip. */
      outsideIndia: Path2D;
    } | null = null;

    function buildBasemap() {
      const p2 = (o: GeoPermissibleObjects) => new Path2D(pathString(o) ?? "");
      const india = p2(INDIA);
      const outsideIndia = new Path2D();
      outsideIndia.rect(-1e6, -1e6, 2e6, 2e6);
      outsideIndia.addPath(india);
      basemap = {
        graticule: p2(GRATICULE),
        land: p2(LAND),
        borders: p2(BORDERS),
        india,
        outsideIndia,
      };
    }

    /** The one place [lat, lon] becomes [lon, lat]. */
    const project = (p: GeoPoint) => projection([p.lon, p.lat]);

    /**
     * What the projection is fitted to: the memories, not the planet.
     *
     * Fitting to `LAND` draws the whole world every time, so a corpus that sits
     * in one harbour renders as an empty globe with a dot on it — the map
     * answers "where is Earth" when the question is "where is my memory".
     * Fitting to the points opens at the scale the data actually occupies.
     * Zoom and pan are untouched, so the world stays one scroll away.
     *
     * Two degenerate cases the extent must survive:
     *  - NO points: nothing to fit, fall back to the world. That is also the
     *    honest thing to draw for a corpus carrying no coordinates.
     *  - ONE point, or several at the same place: a zero-area extent, which
     *    `fitExtent` would resolve to an infinite scale. Padding to a minimum
     *    span fixes it, and the span is generous deliberately — a single
     *    located memory reads better with its region around it than magnified
     *    to street level.
     */
    const MIN_SPAN_DEGREES = 4;

    /** How far out the zoom may go, recomputed on every fit.
     *
     *  The projection is fitted to the DATA, so k=1 is the corpus, not the
     *  planet — with a floor of 1 there was simply nothing below the opening
     *  view and the world was unreachable. A corpus spanning 30 degrees needs
     *  to shrink about twelvefold before Asia is on screen, so the floor is
     *  derived from the span rather than guessed: k = span/360 puts the whole
     *  world in the frame. A corpus that already spans the globe gets a floor
     *  of 1, because it is showing the world at rest and there is nothing
     *  further out to see. */
    const minScaleRef = { current: 1 };

    /** The longitude to put under the central meridian.
     *
     *  Read off the same object that is about to be fitted, so the centring
     *  and the framing can never disagree. `geoBounds` returns
     *  [[west, south], [east, north]] and handles the antimeridian, which a
     *  mean of raw longitudes does not. */
    function fitCentreLon(target: GeoPermissibleObjects): number {
      const [[west], [east]] = geoBounds(target as never);
      if (!Number.isFinite(west) || !Number.isFinite(east)) return 0;
      // geoBounds already normalises a box that crosses the antimeridian into
      // east < west; the midpoint of that wrapped span is still the centre.
      const span = east < west ? east + 360 - west : east - west;
      const mid = west + span / 2;
      return ((mid + 180) % 360) - 180;
    }

    function fitTarget(): GeoPermissibleObjects {
      if (points.length === 0) return LAND;

      /* FIT WHAT WAS RAISED. When a cue is live, `dimmed` names everything the
         cue did not reach, so the complement is the answer — and the map
         should travel to it rather than dim in place and leave the reader
         hunting for the highlight. Falls back to the whole set when nothing is
         dimmed, which is the resting case. */
      const raised = dimmed ? points.filter((p) => !dimmed.has(p.id)) : points;
      const subject = raised.length > 0 ? raised : points;

      /* FIT THE MASS, NOT THE OUTLIERS.
         The raw extent is set by whichever two points are furthest apart, so a
         corpus of 24 memories around Bangalore plus one in Washington spans
         170 degrees of longitude and opens on the Atlantic with the actual
         subject squeezed against the edge, half off-screen. Measured on
         geo-demo, which is exactly that shape.

         Each axis is trimmed independently, so a point extreme in longitude
         but ordinary in latitude only forfeits its longitude. Trimmed points
         are still DRAWN -- they simply start outside the opening frame, one
         zoom-out away, and the footer states the count either way.

         Below the threshold no trim is applied: on a handful of points a
         percentile is not a distribution, and discarding one of six is a
         bigger distortion than the spread it corrects. */
      const TRIM_MIN_POINTS = 8;
      // 0.08 was not enough on the shape this is for: 28 points, 24 of them
      // in India, cut two per side and Tel Aviv still anchored the west edge
      // so Arabia filled the frame. 0.15 clears the whole international tail
      // and opens on the subject. A genuinely global corpus still shows its
      // bulk -- 70% of a spread is a view of it; the extremes are one
      // zoom-out away and counted in the footer regardless.
      const TRIM_Q = 0.15;

      const axis = (values: number[]): [number, number] => {
        const sorted = values.slice().sort((a, b) => a - b);
        const cut = sorted.length < TRIM_MIN_POINTS ? 0 : Math.floor(sorted.length * TRIM_Q);
        return [sorted[cut], sorted[sorted.length - 1 - cut]];
      };

      const [minLon, maxLon] = axis(subject.map((p) => p.lon));
      const [minLat, maxLat] = axis(subject.map((p) => p.lat));

      // Grow whichever axis is under the minimum span around its own centre,
      // then clamp to the coordinate domain so padding cannot push the box off
      // the projection.
      const padAxis = (min: number, max: number, limit: number): [number, number] => {
        if (max - min >= MIN_SPAN_DEGREES) return [min, max];
        const centre = (min + max) / 2;
        const half = MIN_SPAN_DEGREES / 2;
        return [Math.max(-limit, centre - half), Math.min(limit, centre + half)];
      };
      const [lon0, lon1] = padAxis(minLon, maxLon, 180);
      const [lat0, lat1] = padAxis(minLat, maxLat, 90);

      return {
        type: "MultiPoint",
        coordinates: [
          [lon0, lat0],
          [lon1, lat1],
        ],
      } as unknown as GeoPermissibleObjects;
    }

    function sizeCanvas() {
      const rect = wrap!.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      width = rect.width;
      height = rect.height;
      canvas!.width = Math.round(width * dpr);
      canvas!.height = Math.round(height * dpr);
      /* CENTRE THE PROJECTION ON THE DATA BEFORE FITTING IT.
         Equal Earth is pseudocylindrical: its meridians curve away from the
         central meridian, which defaults to 0 degrees. A corpus sitting at
         68-97E is therefore drawn 80 degrees off-centre, where that curvature
         is at its worst, and the whole region arrives visibly sheared -- the
         map looked tilted because it was.

         Rotating the central meridian onto the data removes the shear for
         whatever is being looked at, and costs nothing: the projection stays
         equal-area, so the property the file chose it for -- density on screen
         is density in the world -- is untouched. Longitude only; Equal Earth
         has no oblique form, and rotating latitude would tilt the graticule
         instead of straightening it.

         Rotation must precede fitExtent, because fitting measures the
         projected result and a later rotate would invalidate it. */
      const centre = fitTarget();
      projection.rotate([-fitCentreLon(centre), 0]);

      {
        const [[west, south], [east, north]] = geoBounds(centre as never);
        const lonSpan = east < west ? east + 360 - west : east - west;
        // Latitude counts too: a tall, narrow corpus is just as far from a
        // world view as a wide one, and the binding axis is whichever fills
        // the frame first.
        const span = Math.max(lonSpan || 360, (north - south) * 2 || 360);
        minScaleRef.current = Math.min(1, Math.max(0.01, span / 360));
      }
      projection.fitExtent(
        [
          [12, 12],
          [Math.max(24, width - 12), Math.max(24, height - 12)],
        ],
        centre,
      );
      buildBasemap();
      return dpr;
    }

    function draw() {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const t = transformRef.current;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx!.clearRect(0, 0, width, height);
      ctx!.save();
      ctx!.translate(t.x, t.y);
      ctx!.scale(t.k, t.k);

      // Graticule first, faintest: it is a reading aid for latitude, not a
      // feature of the world.
      if (!basemap) buildBasemap();
      const bm = basemap!;

      ctx!.strokeStyle = hexA(tokens.muted, 0.09);
      ctx!.lineWidth = 0.5 / t.k;
      ctx!.stroke(bm.graticule);

      // Landmass as a quiet ground. The map is a backdrop for the points; if
      // it competes with them it is doing the wrong job.
      ctx!.fillStyle = hexA(tokens.muted, 0.1);
      ctx!.fill(bm.land);
      ctx!.strokeStyle = hexA(tokens.muted, 0.32);
      ctx!.lineWidth = 0.6 / t.k;
      ctx!.stroke();

      // National borders, with India's interior clipped out.
      //
      // Natural Earth draws boundaries on lines of DE-FACTO CONTROL. For India
      // that splits Jammu & Kashmir, puts Aksai Chin outside the country and
      // treats Arunachal Pradesh as disputed — so the mesh cannot be drawn
      // as-is. It is still correct for every other country, so rather than
      // replace it, the claimed area is excluded from it and India's own
      // boundary is stroked over the gap.
      //
      // The clip is a full-plane rectangle plus the India path under the
      // even-odd rule, which leaves everything EXCEPT India's interior
      // paintable. The rectangle is deliberately enormous: this runs inside
      // the zoom transform, so it must still cover the plane at k = 24.
      ctx!.save();
      ctx!.clip(bm.outsideIndia, "evenodd");
      ctx!.strokeStyle = hexA(tokens.muted, 0.2);
      ctx!.lineWidth = 0.5 / t.k;
      ctx!.stroke(bm.borders);
      ctx!.restore();

      // India, from LGD. Drawn at the landmass's weight rather than the border
      // weight: it is the same kind of line as a coastline here, not a fainter
      // internal division.
      ctx!.strokeStyle = hexA(tokens.muted, 0.32);
      ctx!.lineWidth = 0.6 / t.k;
      ctx!.stroke(bm.india);

      ctx!.restore();

      // Points are drawn in SCREEN space, after restore, so their radius stays
      // constant under zoom: a point marks a location, and a location has no
      // area to scale. Zooming in should reveal separation between points, not
      // inflate the dots until they merge again.
      const sel = selectedRef.current;
      const screen: Array<{ point: GeoPoint; x: number; y: number }> = [];
      for (const p of points) {
        const xy = project(p);
        if (!xy) continue;
        const x = xy[0] * t.k + t.x;
        const y = xy[1] * t.k + t.y;
        screen.push({ point: p, x, y });
      }
      screenRef.current = screen;

      for (const { point, x, y } of screen) {
        const isSelected = point.id === sel;
        // Selection outranks dimming: a person who clicked a context point is
        // asking about it, and the map must answer at full strength.
        const isDim = !isSelected && (dimmed?.has(point.id) ?? false);
        // Coordinates are DATA, so a point takes a category hue. The accent is
        // reserved for focus — the selected point and nothing else.
        const hue = isSelected ? tokens.active : hueFor(point.type);
        const r = isDim ? 2.5 : 4 + Math.sqrt(Math.max(0, point.score)) * 3;

        if (!isDim) {
          // A soft halo so a single point on an empty ocean is still findable.
          // Context points get none — a halo is a claim on attention.
          ctx!.beginPath();
          ctx!.arc(x, y, r + (isSelected ? 7 : 4), 0, 2 * Math.PI);
          ctx!.fillStyle = hexA(hue, isSelected ? 0.28 : 0.14);
          ctx!.fill();
        }

        ctx!.beginPath();
        ctx!.arc(x, y, r, 0, 2 * Math.PI);
        ctx!.fillStyle = hexA(hue, isDim ? 0.35 : 0.9);
        ctx!.fill();
        ctx!.lineWidth = isSelected ? 2 : isDim ? 0.75 : 1;
        ctx!.strokeStyle = isSelected ? tokens.active : hexA(hue, isDim ? 0.45 : 1);
        ctx!.stroke();
      }
    }

    drawRef.current = draw;
    sizeCanvas();
    draw();

    const zoomBehavior = zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([minScaleRef.current, 24])
      .on("zoom", (event) => {
        transformRef.current = event.transform;
        draw();
      });
    const sel = select(canvas);
    sel.call(zoomBehavior);

    const observer = new ResizeObserver(() => {
      sizeCanvas();
      zoomBehavior.scaleExtent([minScaleRef.current, 24]);
      draw();
    });
    observer.observe(wrap);

    return () => {
      observer.disconnect();
      sel.on(".zoom", null);
    };
  }, [points, types, dimmed]);

  const pointAt = useCallback((sx: number, sy: number) => {
    let hit: { point: GeoPoint; x: number; y: number } | null = null;
    let best = Infinity;
    for (const s of screenRef.current) {
      const dx = s.x - sx;
      const dy = s.y - sy;
      const d = dx * dx + dy * dy;
      if (d <= HIT_RADIUS_PX * HIT_RADIUS_PX && d < best) {
        best = d;
        hit = s;
      }
    }
    return hit;
  }, []);

  const downRef = useRef<[number, number] | null>(null);

  /**
   * Keyboard selection.
   *
   * Recall's canvas can be pointer-only because the result list beside it is a
   * column of real buttons — every memory the graph draws is reachable by Tab,
   * and selecting there lights the graph. Geo has no such list: the map is the
   * only way to reach a point, so without this the destination is unusable
   * without a mouse, which is the exact failure DIRECTION.md rules out for the
   * rail and is no more acceptable here.
   *
   * Points are ordered west to east so the arrow keys track what the eye does
   * across the map rather than replaying insertion order.
   */
  const ordered = useMemo(() => [...points].sort((a, b) => a.lon - b.lon), [points]);

  const step = useCallback(
    (delta: number) => {
      if (ordered.length === 0) return;
      const at = ordered.findIndex((p) => p.id === selectedId);
      // Nothing selected yet: enter from the appropriate end rather than
      // jumping to the middle.
      const next =
        at === -1
          ? delta > 0
            ? 0
            : ordered.length - 1
          : (at + delta + ordered.length) % ordered.length;
      selectMemory(ordered[next].id);
    },
    [ordered, selectedId, selectMemory],
  );

  return (
    <div ref={wrapRef} className="absolute inset-0">
      <canvas
        ref={canvasRef}
        className="focus-visible:ring-ring size-full focus-visible:ring-2 focus-visible:outline-none"
        // `application`, not `img`: this reports its own keyboard contract, and
        // the label states it because a canvas exposes nothing else to a
        // screen reader.
        role="application"
        aria-label={`World map, ${points.length} located ${
          points.length === 1 ? "memory" : "memories"
        }. Use the left and right arrow keys to move between them.`}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight" || e.key === "ArrowDown") {
            e.preventDefault();
            step(1);
          } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
            e.preventDefault();
            step(-1);
          } else if (e.key === "Escape") {
            selectMemory(null);
          }
        }}
        style={{ cursor: hover ? "pointer" : "grab" }}
        onPointerDown={(e) => {
          downRef.current = [e.nativeEvent.offsetX, e.nativeEvent.offsetY];
        }}
        onClick={(e) => {
          const down = downRef.current;
          const x = e.nativeEvent.offsetX;
          const y = e.nativeEvent.offsetY;
          if (down && (Math.abs(x - down[0]) > 4 || Math.abs(y - down[1]) > 4)) return;
          const hit = pointAt(x, y);
          selectMemory(hit ? hit.point.id : null);
        }}
        onPointerMove={(e) => {
          const x = e.nativeEvent.offsetX;
          const y = e.nativeEvent.offsetY;
          const hit = pointAt(x, y);
          setHover(hit ? { point: hit.point, x, y } : null);
        }}
        onPointerLeave={() => setHover(null)}
      />

      {hover ? (
        <div
          className="border-border bg-popover text-popover-foreground pointer-events-none absolute z-20 w-[240px] rounded-md border p-2.5 shadow-xl"
          style={{
            left: hover.x,
            top: hover.y,
            transform: `translate(${hover.x > 300 ? "calc(-100% - 14px)" : "14px"}, ${
              hover.y > 180 ? "calc(-100% - 14px)" : "14px"
            })`,
          }}
        >
          <p className="line-clamp-3 text-[12px] leading-relaxed">{hover.point.label}</p>
          <p className="text-muted-foreground mono mt-1.5 text-[10px]">
            {hover.point.lat.toFixed(4)}, {hover.point.lon.toFixed(4)}
          </p>
        </div>
      ) : null}
    </div>
  );
}
