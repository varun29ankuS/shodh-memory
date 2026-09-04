import { useEffect, useRef } from "react";

import { drawDotMap, type PlacedPoint } from "./dot-map";
import type { Bounds, Ring } from "./geo-shapes";

/**
 * Canvas wrapper for `drawDotMap`.
 *
 * A canvas is not reactive, so this component owns three redraw triggers and
 * nothing else:
 *
 *  1. the data changed,
 *  2. the element resized — the map sizes itself from `clientWidth`,
 *  3. the ground changed — every colour is read from CSS custom properties at
 *     draw time, so a theme switch leaves a stale bitmap until we redraw.
 *
 * (3) has two sources: an explicit `data-theme` on `<html>`, and the system
 * preference when no explicit choice is set. Both are watched, because missing
 * the second means the map silently keeps paper-coloured dots on a night ground
 * for anyone who never touches the toggle.
 */
export function DotMap({
  shapes,
  bounds,
  points,
  cell,
  label,
}: {
  shapes: Ring[];
  bounds: Bounds;
  points: PlacedPoint[];
  cell: number;
  /** Announced to screen readers; the canvas itself carries no text. */
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    let frame = 0;
    const draw = () => {
      cancelAnimationFrame(frame);
      // Coalesce: a theme switch plus a resize in the same tick is one redraw,
      // and sampling the whole landmass buffer twice is the expensive half.
      frame = requestAnimationFrame(() => {
        drawDotMap({ canvas, shapes, bounds, points, cell });
      });
    };

    draw();

    const resize = new ResizeObserver(draw);
    resize.observe(canvas);

    const theme = new MutationObserver(draw);
    theme.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    const scheme = window.matchMedia("(prefers-color-scheme: dark)");
    scheme.addEventListener("change", draw);

    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      theme.disconnect();
      scheme.removeEventListener("change", draw);
    };
  }, [shapes, bounds, points, cell]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={label}
      className="w-full"
      // Height is set by the renderer from the bounds' aspect ratio.
      style={{ height: 0 }}
    />
  );
}
