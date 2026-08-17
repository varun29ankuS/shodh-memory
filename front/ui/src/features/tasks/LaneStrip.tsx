import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { laneCurve, stepPath, type Axis, type Lane } from "./derive";

/**
 * How one project's work accumulated, and how much of it landed.
 *
 * WHY A CURVE AND NOT A BAR. A bar reading "31 of 40" is one number at one
 * instant, and on this corpus that hides the thing worth seeing: the live
 * `claude-code` profile holds a project that recorded thirteen tasks and closed
 * all thirteen inside a day, and another that has been open since January. Both
 * would draw the same bar. The shape is the finding.
 *
 * TWO SERIES, BECAUSE SCOPE MOVES TOO. The upper line is how much work existed
 * by that point, the filled area is how much of it had settled. A project that
 * "finished everything" having quietly tripled its scope is a different story
 * from one that never moved the target, and one series cannot tell them apart.
 * Linear's project graph plots the same pair for the same reason
 * (linear.app/docs/project-graph).
 *
 * NOTHING IS FORECAST. Linear's third series extrapolates a velocity and wraps
 * it in a ±40% buffer. There are no estimates, cycles or velocity here, so a
 * projected finish would be a line with no measurement under it. Both series
 * stop at the present.
 *
 * IT IS DRAWN AS STEPS. Interpolating between two completions would show work
 * progressing on days when nothing happened — the drawing would be making a
 * claim the data does not.
 *
 * NOT INTERACTIVE, AND NOTHING IS REACHABLE ONLY BY POINTING AT IT. Every
 * figure the curve encodes — total, settled, underway — is printed beside it,
 * following the same rule as the tool census on /history. The curve adds shape
 * at a glance and gates nothing.
 */
export function LaneStrip({
  lane,
  axis,
  className,
}: {
  lane: Lane;
  axis: Axis;
  className?: string;
}) {
  // A viewBox in abstract units with `preserveAspectRatio="none"`: the strip is
  // laid out by CSS at whatever width the column has, and the path arithmetic
  // must not care. Height stays small — this is a sparkline in a table row, not
  // a chart with an axis of its own.
  const W = 100;
  const H = 12;

  const { scope, settled } = useMemo(() => laneCurve(lane, axis), [lane, axis]);
  const scopeLine = useMemo(() => stepPath(scope, W, H), [scope]);
  const settledLine = useMemo(() => stepPath(settled, W, H), [settled]);
  // The area is the settled line closed down to the baseline. Built from the
  // same path string so the two can never disagree about where the curve is.
  const settledArea = useMemo(
    () => (settledLine ? `${settledLine} L ${W},${H} L 0,${H} Z` : ""),
    [settledLine],
  );

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      className={cn("h-3 w-full overflow-visible", className)}
    >
      {/* The baseline is drawn always, so an empty lane reads as an empty lane
          rather than as a missing element. */}
      <line
        x1="0"
        y1={H}
        x2={W}
        y2={H}
        stroke="var(--border)"
        strokeWidth="0.5"
        vectorEffect="non-scaling-stroke"
      />
      {settledArea ? <path d={settledArea} fill="var(--primary)" opacity="0.16" /> : null}
      {scopeLine ? (
        <path
          d={scopeLine}
          fill="none"
          stroke="var(--muted-foreground)"
          strokeWidth="1"
          opacity="0.4"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
      {settledLine ? (
        <path
          d={settledLine}
          fill="none"
          stroke="var(--primary)"
          strokeWidth="1.25"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
    </svg>
  );
}
