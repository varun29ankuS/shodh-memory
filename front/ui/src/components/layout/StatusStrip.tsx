import { cn } from "@/lib/utils";
import type { ServiceReading, Tone } from "./systemHealth";

const TONE_TEXT: Record<Tone, string> = {
  live: "text-[var(--live)]",
  warn: "text-warn",
  alarm: "text-destructive",
  unknown: "text-muted-foreground",
};

/**
 * Which service, and what state, said in the header for both of them.
 *
 * TWO SEGMENTS, NOT ONE, and that is the correction this component needed most.
 * It reported the memory server alone, so it displayed "Connected" over a dead
 * seat — and while the seat is down the assistant cannot answer, cannot move
 * this view and cannot touch tasks. Being told the system is connected while
 * half of it is unusable is worse than being told nothing, because it is
 * believed.
 *
 * NO LONGER `hidden sm:flex`. A liveness indicator that disappears exactly when
 * the window gets cramped is not an indicator; and since the rail is a fixed
 * 244px at every width, `sm` was never the width at which this app becomes
 * usable anyway — the strip was vanishing well inside the range where it had
 * room. The remedy text, which used to need `lg`, has moved to the banner,
 * where it has a whole line and does not have to compete with the title.
 *
 * The dot is not the state. It is a second encoding of a state that is also
 * spelled out in a word beside it, so the reading survives being colour-blind,
 * being on a projector, or being a screenshot in a slide deck.
 */
export function StatusStrip({ readings }: { readings: ServiceReading[] }) {
  return (
    <div className="bg-muted mono flex h-[26px] min-w-0 shrink items-center overflow-hidden rounded-md text-[11px]">
      {readings.map((r, i) => (
        <span
          key={r.id}
          // The whole reading, for a pointer. The words on screen are the
          // primary carrier — this is the evidence behind them, not a
          // substitute for them.
          title={`${r.service}: ${r.state} — ${r.evidence}, ${r.checked}${
            r.consequence ? `. ${r.consequence}` : ""
          }`}
          className={cn(
            "flex h-full min-w-0 items-center gap-1.5 px-2.5",
            i > 0 && "border-sidebar border-l-2",
          )}
        >
          <span className={cn("size-1.5 shrink-0 rounded-full bg-current", TONE_TEXT[r.tone])} />
          <span className="text-muted-foreground shrink-0">{r.service}</span>
          <span className={cn("truncate font-medium", TONE_TEXT[r.tone])}>{r.state}</span>
        </span>
      ))}
    </div>
  );
}
