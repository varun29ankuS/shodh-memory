import { cn } from "@/lib/utils";
import type { ServiceReading, Tone } from "./systemHealth";
import { RAIL_OFFSET } from "./destinations";

/**
 * Tone as colour, in one table, so no state can be coloured by hand.
 *
 * `--live` has no `@theme` mapping (it is a raw custom property used by the
 * graph canvas as well), which is why it appears as an arbitrary value while
 * `warn` and `destructive` appear as utilities. `unknown` takes
 * `--muted-foreground` — the colour of ordinary label text — because the
 * honest rendering of "this page cannot vouch for the reading" is not an alarm.
 */
const TONE_TEXT: Record<Tone, string> = {
  live: "text-[var(--live)]",
  warn: "text-warn",
  alarm: "text-destructive",
  unknown: "text-muted-foreground",
};

const TONE_BG: Record<Tone, string> = {
  live: "bg-[var(--live)]",
  warn: "bg-warn",
  alarm: "bg-destructive",
  unknown: "bg-muted-foreground",
};

const TONE_BORDER: Record<Tone, string> = {
  live: "border-[var(--live)]",
  warn: "border-warn",
  alarm: "border-destructive",
  unknown: "border-muted-foreground",
};

/**
 * The ribbon, when everything is fine: a 3px rule across the top of the window.
 *
 * WHY IT IS THREE PIXELS AND NOT A BAR. The brief for this screen was that it
 * "takes time to understand what it is saying", and a pass was spent cutting
 * the prose above the fold from ~175 words to ~47. Health is the state this app
 * is in almost all of the time, so a permanent full-width coloured bar would
 * spend that entire budget back on a fact that is nearly always "fine" — and a
 * banner that is always there is a banner nobody reads on the day it matters.
 * Three pixels of colour is enough for a glance to confirm life, and it costs
 * no words at all. The words for the healthy case live in the chips in the
 * header, which say which service and what state in three words each.
 *
 * ONE LINE IN THE WORST TONE, NOT ONE SEGMENT PER SERVICE. A split bar would
 * carry marginally more information and would need a legend to decode, which is
 * not a glance. The line answers "is anything wrong"; the chips forty pixels
 * below it answer "what".
 *
 * RENDERED OUTSIDE THE HEADER ON PURPOSE. The header is `z-20` and the rail is
 * `z-30`, so a line nested inside the header would be painted over for its
 * first 244px and would stop short of the left edge of the window. At `z-40` as
 * a sibling it spans the full width, over the rail, which is the only placement
 * that reads as "the top of this window" rather than "the top of the content".
 */
export function SystemPulse({ tone }: { tone: Tone }) {
  return (
    <div
      // Decorative: every word it could carry is carried by the chips and the
      // banner, both of which are real text. Announcing a colour would add
      // nothing to a screen reader and would repeat the chips.
      aria-hidden="true"
      className={cn(
        "pointer-events-none absolute inset-x-0 top-0 z-40 h-[3px]",
        // A colour change here is a state change, not an animation, but the
        // change is abrupt enough at full width to read as a flash. The global
        // `prefers-reduced-motion` rule in index.css collapses this to 0.01ms,
        // so no branch is needed here.
        "transition-colors duration-500",
        TONE_BG[tone],
      )}
    />
  );
}

/**
 * The ribbon when something is wrong: real height, real colour, real sentences.
 *
 * It appears only for `warn` and `alarm` — never for `unknown`, so it cannot
 * flash an explanation on the first frame of every page load, before either
 * probe has answered.
 *
 * IT OVERLAYS THE TOP OF THE STAGE RATHER THAN PUSHING IT DOWN, and that is a
 * constraint rather than a preference: the shell positions the header, the rail
 * and the inspector absolutely and reserves the header's height with a fixed
 * `pt-12` on `main` (App.tsx). Claiming flow height here would need that
 * padding to become a function of this component's state, in a file this change
 * does not own. The cost is bounded — one line per unhealthy service, at the
 * top of a stage which, for the memory-server failures, is already showing an
 * outage explanation and nothing else.
 */
export function SystemBanner({ alerts }: { alerts: ServiceReading[] }) {
  if (alerts.length === 0) return null;

  return (
    <div
      // A live region, so the state is announced rather than only shown. The
      // one string in here that changes every second — the age — is hidden from
      // it below, or this would be read aloud on every tick.
      role="status"
      className={cn(
        "border-border bg-card absolute inset-x-0 top-12 z-20 border-b",
        RAIL_OFFSET,
      )}
    >
      <ul className="flex flex-col">
        {alerts.map((a) => (
          <li
            key={a.id}
            className={cn(
              "flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5 border-l-2 py-1.5 pr-4 pl-3 text-[12px]",
              TONE_BORDER[a.tone],
            )}
          >
            {/* Service and state lead, in the tone, because "which of the two
                things is broken" is the first question. The mark is the word —
                colour is never the only carrier. */}
            <span className={cn("mono shrink-0 font-medium", TONE_TEXT[a.tone])}>
              {a.service} · {a.state}
            </span>
            {/* The consequence, which is the half a person can act on without
                already knowing how this product is put together. */}
            {a.consequence ? <span className="min-w-0">{a.consequence}</span> : null}
            {a.remedy ? (
              <span className="text-muted-foreground min-w-0 font-medium">{a.remedy}</span>
            ) : null}
            <span className="text-muted-foreground mono min-w-0 shrink-0 text-[11px]">
              {a.evidence} ·{" "}
              {/* The only ticking string on screen, kept out of the live region
                  above so it is not announced once a second. */}
              <span aria-hidden="true">{a.checked}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
