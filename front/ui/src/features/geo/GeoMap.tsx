import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  geoEqualEarth,
  geoPath,
  geoGraticule10,
  select,
  zoom,
  zoomIdentity,
  type GeoPermissibleObjects,
  type ZoomTransform,
} from "d3";
import { feature, mesh } from "topojson-client";
import type { Topology, GeometryCollection } from "topojson-specification";
import type { RecallMemory } from "@/lib/api";
import { useSession } from "@/stores/session";
import { createPulseRunner, pulsePhase, subscribePulse } from "@/stores/activity";
import worldTopology from "@/assets/world-countries-110m.json";

/**
 * The world basemap and the plotted points.
 *
 * NO TILE SERVER, NO NETWORK. The whole app ships as one self-contained
 * index.html embedded in the Rust binary (front/src/main.rs), which is the
 * reason vite.config.ts inlines every asset; a map that fetched tiles would
 * break that guarantee the first time it ran somewhere without egress, and
 * quietly leak the fact that someone is looking at a place. The basemap is
 * therefore vendored: src/assets/world-countries-110m.json, Natural Earth
 * 1:110m via world-atlas (its licence sits beside it), 107.7 kB of quantised
 * TopoJSON decoded in-process.
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
 *
 * THE CAMERA ANSWERS TO THE CONVERSATION. When the seat surfaces a memory that
 * is plotted here, the map brings it into view and the point pulses. Only
 * plotted ids count: a memory the conversation reached that carries no
 * coordinates, or that is not in the current recall result, moves nothing —
 * this view is a second rendering of that result set, not a second retrieval,
 * and it cannot show a point it was never given. `ensureVisible` also declines
 * to move when the points are already comfortably on screen.
 */

/** The vendored file's `objects` — named so the decode below is not `any`. */
type WorldTopology = Topology<{
  countries: GeometryCollection<{ name: string }>;
  land: GeometryCollection;
}>;

const world = worldTopology as unknown as WorldTopology;

/** Decoded once at module scope: the topology is a constant, and re-deriving
 *  it per mount would re-walk 177 country geometries on every navigation. */
const LAND = feature(world, world.objects.land) as unknown as GeoPermissibleObjects;
/** Interior borders only — `(a, b) => a !== b` drops the coastline, which LAND
 *  already draws. Drawing both would double-stroke every shore. */
const BORDERS = mesh(world, world.objects.countries, (a, b) => a !== b) as GeoPermissibleObjects;
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

/** Keep a point this far inside the pane before calling it "on screen" — a dot
 *  touching the frame is technically visible and practically missed. */
const CAMERA_PAD_PX = 48;

/** Ceiling on the scale the camera will choose for itself. `scaleExtent` allows
 *  24×, but a single surfaced memory does not justify diving to street level:
 *  the surrounding points are the context that makes a location mean something,
 *  and they are exactly what a hard zoom throws away. */
const CAMERA_MAX_K = 8;

/** Camera glide. Long enough to read as one continuous move. */
const CAMERA_MS = 520;

export function GeoMap({ memories, types }: { memories: RecallMemory[]; types: string[] }) {
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

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const pulse = createPulseRunner(() => drawRef.current());
    /** In flight camera move, so a second surfaced memory cancels the first
     *  rather than fighting it frame by frame. */
    let cameraRaf = 0;

    let width = 0;
    let height = 0;
    const projection = geoEqualEarth();
    const path = geoPath(projection, ctx);

    /** The one place [lat, lon] becomes [lon, lat]. */
    const project = (p: GeoPoint) => projection([p.lon, p.lat]);

    function sizeCanvas() {
      const rect = wrap!.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      width = rect.width;
      height = rect.height;
      canvas!.width = Math.round(width * dpr);
      canvas!.height = Math.round(height * dpr);
      // Refit rather than rescale: fitExtent recomputes both scale and centre,
      // so the map fills a resized pane instead of drifting off one edge.
      projection.fitExtent(
        [
          [12, 12],
          [Math.max(24, width - 12), Math.max(24, height - 12)],
        ],
        LAND,
      );
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
      ctx!.beginPath();
      path(GRATICULE);
      ctx!.strokeStyle = hexA(tokens.muted, 0.09);
      ctx!.lineWidth = 0.5 / t.k;
      ctx!.stroke();

      // Landmass as a quiet ground. The map is a backdrop for the points; if
      // it competes with them it is doing the wrong job.
      ctx!.beginPath();
      path(LAND);
      ctx!.fillStyle = hexA(tokens.muted, 0.1);
      ctx!.fill();
      ctx!.strokeStyle = hexA(tokens.muted, 0.32);
      ctx!.lineWidth = 0.6 / t.k;
      ctx!.stroke();

      ctx!.beginPath();
      path(BORDERS);
      ctx!.strokeStyle = hexA(tokens.muted, 0.2);
      ctx!.lineWidth = 0.5 / t.k;
      ctx!.stroke();

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

      const lit = pulse.ref.current;
      const now = lit ? performance.now() : 0;

      for (const { point, x, y } of screen) {
        const isSelected = point.id === sel;
        // Coordinates are DATA, so a point takes a category hue. The accent is
        // reserved for focus — the selected point and nothing else.
        const hue = isSelected ? tokens.active : hueFor(point.type);
        const r = 4 + Math.sqrt(Math.max(0, point.score)) * 3;

        // A soft halo so a single point on an empty ocean is still findable.
        ctx!.beginPath();
        ctx!.arc(x, y, r + (isSelected ? 7 : 4), 0, 2 * Math.PI);
        ctx!.fillStyle = hexA(hue, isSelected ? 0.28 : 0.14);
        ctx!.fill();

        ctx!.beginPath();
        ctx!.arc(x, y, r, 0, 2 * Math.PI);
        ctx!.fillStyle = hexA(hue, 0.9);
        ctx!.fill();
        ctx!.lineWidth = isSelected ? 2 : 1;
        ctx!.strokeStyle = isSelected ? tokens.active : hexA(hue, 1);
        ctx!.stroke();

        // Screen space like the point itself, so the ring is the same size
        // wherever the camera is: it marks an event, not an area.
        if (lit && lit.ids.has(point.id)) {
          const phase = pulsePhase(lit, now);
          ctx!.beginPath();
          ctx!.arc(x, y, r + 6 + phase * 24, 0, 2 * Math.PI);
          ctx!.strokeStyle = hexA(tokens.active, 0.85 * (1 - phase));
          ctx!.lineWidth = 2;
          ctx!.stroke();
        }
      }
    }

    drawRef.current = draw;
    sizeCanvas();
    draw();

    const zoomBehavior = zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([1, 24])
      .on("zoom", (event) => {
        transformRef.current = event.transform;
        draw();
      });
    const sel = select(canvas);
    sel.call(zoomBehavior);

    /**
     * Bring `hits` into view.
     *
     * Every move goes through `zoomBehavior.transform`, never through
     * `transformRef` directly: d3 keeps the authoritative transform on the DOM
     * node, so writing the ref behind its back leaves the two disagreeing and
     * the user's next scroll or drag jumps back to where d3 still thinks the
     * camera is.
     *
     * The camera is also conservative on purpose. If the points are already
     * comfortably on screen it does not move at all — motion that changes
     * nothing is the kind of animation that teaches people to ignore the
     * surface — and it never zooms IN past what the person had chosen, only out
     * far enough to fit what arrived.
     */
    function ensureVisible(hits: GeoPoint[]) {
      const t = transformRef.current;
      const base: Array<[number, number]> = [];
      for (const p of hits) {
        const xy = project(p);
        if (xy) base.push(xy);
      }
      if (base.length === 0) return;

      const onScreen = base.every(([bx, by]) => {
        const x = bx * t.k + t.x;
        const y = by * t.k + t.y;
        return (
          x >= CAMERA_PAD_PX &&
          x <= width - CAMERA_PAD_PX &&
          y >= CAMERA_PAD_PX &&
          y <= height - CAMERA_PAD_PX
        );
      });
      if (onScreen) return;

      let x0 = Infinity;
      let y0 = Infinity;
      let x1 = -Infinity;
      let y1 = -Infinity;
      for (const [bx, by] of base) {
        x0 = Math.min(x0, bx);
        y0 = Math.min(y0, by);
        x1 = Math.max(x1, bx);
        y1 = Math.max(y1, by);
      }
      const boxW = Math.max(x1 - x0, 1e-6);
      const boxH = Math.max(y1 - y0, 1e-6);
      const fits = Math.min(
        (width - 2 * CAMERA_PAD_PX) / boxW,
        (height - 2 * CAMERA_PAD_PX) / boxH,
      );
      // Hold the current scale where it still fits, otherwise back off to the
      // scale that does. `scaleExtent` is [1, 24]; 1 is the whole world.
      const k = Math.min(CAMERA_MAX_K, Math.max(1, Math.min(fits, Math.max(t.k, 1))));
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      const target = zoomIdentity.translate(width / 2 - cx * k, height / 2 - cy * k).scale(k);

      if (cameraRaf) cancelAnimationFrame(cameraRaf);
      if (reduceMotion) {
        // Jump. The glide is the part that causes trouble, not the destination.
        zoomBehavior.transform(sel, target);
        return;
      }

      const from = t;
      const startedAt = performance.now();
      const step = () => {
        const u = Math.min(1, (performance.now() - startedAt) / CAMERA_MS);
        // Cubic ease-out: leaves fast, arrives settled.
        const e = 1 - Math.pow(1 - u, 3);
        zoomBehavior.transform(
          sel,
          zoomIdentity
            .translate(from.x + (target.x - from.x) * e, from.y + (target.y - from.y) * e)
            .scale(from.k + (target.k - from.k) * e),
        );
        cameraRaf = u < 1 ? requestAnimationFrame(step) : 0;
      };
      cameraRaf = requestAnimationFrame(step);
    }

    const drawn = new Map(points.map((p) => [p.id, p]));
    const unsubscribePulse = subscribePulse(
      (id) => drawn.has(id),
      (hit) => {
        pulse.start(hit);
        ensureVisible([...hit].map((id) => drawn.get(id)!));
      },
    );

    const observer = new ResizeObserver(() => {
      sizeCanvas();
      draw();
    });
    observer.observe(wrap);

    return () => {
      observer.disconnect();
      sel.on(".zoom", null);
      unsubscribePulse();
      pulse.cancel();
      if (cameraRaf) cancelAnimationFrame(cameraRaf);
    };
  }, [points, types]);

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
