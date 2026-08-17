import { useEffect, useMemo, useState } from "react";
import { Sparkle, Undo2 } from "lucide-react";
import { useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { destinationNoun } from "@/lib/view/commands";
import {
  TRACE_DWELL_MS,
  type AxisState,
  arrived,
  axisStateLabel,
  returnTarget,
  traceAnnouncement,
  traceKey,
  traceOf,
  viewDimensionLabel,
} from "@/lib/view/presence";
import { useTrace } from "@/stores/trace";
import { useView } from "@/stores/view";

/**
 * The conversation operating the workbench, visible while it happens.
 *
 * THE THING THAT WAS MISSING. The loop was already whole — the model asks, the
 * authority ledger decides per axis, the seat is told, `/history` keeps the row —
 * and none of it was on screen at the moment it ran. An applied change was a
 * picture rearranging itself under a one-line statement in the header that a
 * reader watching the picture never looked at; a declined one was a Follow chip;
 * and a request that landed on four axes with different fates rendered as the
 * one axis that was refused, because `FollowOffer` returns its offer branch
 * whenever anything is pending. The most informative event this system produces
 * was the one it showed least.
 *
 * THE REASON IS THE PAYLOAD AND IT IS WHY THIS IS A BLOCK RATHER THAN A CHIP.
 * Every `direct_view` carries a required 8–400 character account of the EVIDENCE
 * — "these 12 memories cluster on the Malabar coast", never "opening Geo" — and
 * the header's one line at 12px truncates it. A sentence a person cannot finish
 * reading is a sentence that was not delivered.
 *
 * IT IS ANCHORED UNDER THE HEADER, NOT FLOATED IN A CORNER, and that was forced
 * rather than chosen. Every canvas surface in this product pins a hint strip
 * across `inset-x-4 bottom-3` (GraphView, GeoView, GraphStage) and the
 * conversation docks bottom-right, so the bottom edge is spoken for on the
 * screens this most often fires on. The top of the stage under the header is the
 * one region that is free on all ten destinations, it is directly below the
 * header line this collapses into — so the eye that read the block knows where
 * the residue lives — and it is the same band `SystemBanner` already uses, which
 * is why the two are stacked in one column by `TopBar` rather than each claiming
 * `top-12` and overlapping in the state where something is also wrong.
 *
 * IT COSTS NOTHING WHEN THE CONVERSATION IS IDLE: no reason, no block, `null`,
 * zero words and zero pixels. It costs nothing after the moment has passed
 * either — it collapses on its own, and the header line carries what remains.
 *
 * NON-BLOCKING BY CONSTRUCTION. The card takes no pointer events; only the row
 * of controls does. A person who wants to keep working clicks straight through
 * the prose to the stage underneath, and the only pixels this takes away are the
 * ones that exist to be pressed.
 */

/** The label column, wide enough for the longest axis name at 11px. */
const AXIS_COLUMN = "w-[92px]";

/**
 * One tone per state, in a table, so no row can be coloured by hand.
 *
 * `applied` and `already` share the label colour because neither is anything to
 * do: one moved, one was already right, and both are statements of fact. Only
 * the axis that is waiting on a person is marked, and the WORD carries it — the
 * colour is a second channel, never the only one.
 */
const AXIS_TONE: Record<AxisState, string> = {
  applied: "text-muted-foreground",
  already: "text-muted-foreground",
  waiting: "text-warn font-medium",
};

export function AgentTrace() {
  const notice = useView((s) => s.notice);
  const offers = useView((s) => s.offers);
  const destination = useView((s) => s.destination);
  const follow = useView((s) => s.follow);
  const dismiss = useView((s) => s.dismiss);
  const back = useView((s) => s.back);
  const shown = useTrace((s) => s.shown);
  const close = useTrace((s) => s.close);
  const { pathname } = useLocation();

  /**
   * Paused while a hand or the keyboard is on the controls.
   *
   * A control that vanishes from under a cursor on its way to it is worse than
   * one that was never offered: the person's press lands on whatever the collapse
   * revealed. `Follow` and `Not now` are a decision, and a decision does not get
   * a countdown.
   */
  const [held, setHeld] = useState(false);

  /**
   * The arrival, watched THROUGH THE STORE rather than through a render.
   *
   * `arrived` is deliberately not "the trace changed". A trace SHRINKS for
   * reasons that are not arrivals — a turn boundary expiring an offer, a refusal,
   * a hand taking an axis — and opening on any change would flash an
   * applied-only summary at the start of every turn the person typed into. That
   * is the app twitching, which is the failure this whole mechanism exists to
   * replace.
   */
  useEffect(
    () =>
      useView.subscribe((state, previous) => {
        if (!arrived(previous, state)) return;
        const next = traceOf(state.notice, state.offers);
        if (next) useTrace.getState().open(traceKey(next));
      }),
    [],
  );

  const trace = useMemo(() => traceOf(notice, offers), [notice, offers]);
  const key = trace === null ? null : traceKey(trace);
  const live = key !== null && key === shown;

  /**
   * A new trace starts unheld, and that is a bug fix rather than tidiness.
   *
   * `held` is set by focus and cleared by blur — but activating `Follow` from
   * the keyboard UNMOUNTS the button under the focus that is holding the clock,
   * and a removed element fires no blur. The hold would then never clear: the
   * block would sit open forever AND the header line would stay suppressed
   * behind it, so its `Back` and `Release` were unreachable too. A mouse user
   * recovers on the next `mouseleave`; a keyboard user had no way out at all.
   */
  useEffect(() => setHeld(false), [key]);

  useEffect(() => {
    if (!live || held) return;
    const timer = setTimeout(close, TRACE_DWELL_MS);
    return () => clearTimeout(timer);
    // `key` is a dependency so a second request arriving while the first is up
    // restarts the clock rather than inheriting the remainder of it.
  }, [live, held, key, close]);

  if (trace === null || !live) return null;

  const waiting = trace.axes.some((axis) => axis.state === "waiting");
  const target = returnTarget(destination, pathname);

  return (
    // `pointer-events-none` on the card and `auto` on the controls: the block
    // overlays the top of a stage the person may still be working on, and the
    // only part of it that has any business intercepting a click is the part
    // that is a button.
    <div className="pointer-events-none px-4 pt-2">
      <div
        role="status"
        className={cn(
          "border-border bg-card max-w-[440px] rounded-md border py-2 pr-3 pl-2.5 shadow-lg",
          // Motion is the only thing that says this arrived rather than having
          // always been here. The global reduced-motion rule in index.css
          // collapses it.
          "offer-in",
        )}
      >
        {/* Spoken as prose, because a label column beside a state column reads
            to the ear as an ungrammatical run of words — see
            `traceAnnouncement`. The visual rows below are hidden from the
            region so the same facts are not announced twice. */}
        <p className="sr-only">{traceAnnouncement(trace)}</p>

        <div aria-hidden="true" className="flex min-w-0 gap-2">
          <Sparkle className="text-primary mt-[3px] size-3 shrink-0" strokeWidth={2} />
          <div className="min-w-0 flex-1">
            {/* THE MODEL'S WORDS, QUOTED AND UNEDITED, and given the room to be
                finished. Clamped at three lines because the tool admits 400
                characters and a block that can grow to a paragraph stops being
                a trace; the header line holds the same sentence afterwards for
                anyone who wants the rest of it. */}
            <p className="text-foreground line-clamp-3 text-[12px] leading-relaxed">
              “{trace.reason}”
            </p>
            <ul className="mt-1.5 flex flex-col gap-px">
              {trace.axes.map((axis) => (
                <li key={axis.dimension} className="flex items-baseline gap-2 text-[11px]">
                  <span className={cn("text-muted-foreground shrink-0", AXIS_COLUMN)}>
                    {viewDimensionLabel(axis.dimension)}
                  </span>
                  {/* THE WORD CARRIES IT, NOT THE COLOUR. `--warn` is the token
                      for waiting-on-someone and this is exactly what it is for;
                      `--destructive` would say the request was wrong, which is a
                      different and false claim about a model that asked
                      correctly and was told to wait. */}
                  <span className={AXIS_TONE[axis.state]}>
                    {axisStateLabel(axis.state)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {waiting || target !== null ? (
          <div
            className="pointer-events-auto mt-2 flex flex-wrap items-center gap-1.5 pl-5"
            onMouseEnter={() => setHeld(true)}
            onMouseLeave={() => setHeld(false)}
            onFocus={() => setHeld(true)}
            onBlur={() => setHeld(false)}
          >
            {waiting ? (
              <>
                <button
                  type="button"
                  onClick={follow}
                  aria-label={`Let the conversation take what it is waiting for, because ${trace.reason}`}
                  className="bg-primary text-primary-foreground hover:bg-primary/90 focus-visible:ring-ring rounded px-2 py-[3px] text-[11px] font-medium transition-colors focus-visible:ring-2 focus-visible:outline-none"
                >
                  Follow
                </button>
                <button
                  type="button"
                  onClick={dismiss}
                  aria-label="Keep this view and decline the conversation's change"
                  className="text-muted-foreground hover:text-foreground focus-visible:ring-ring rounded px-1.5 py-[3px] text-[11px] transition-colors focus-visible:ring-2 focus-visible:outline-none"
                >
                  Not now
                </button>
              </>
            ) : null}
            {/* REVERSIBLE FROM WHERE IT LANDS. A destination change is the one
                axis a person cannot undo by doing the ordinary thing, and this
                is the block that appears at the moment it happens — so the way
                back belongs here, spelled with the stage's own name rather than
                as a bare arrow. It survives the collapse: the header line
                carries the same control for as long as the record describes
                where the person is standing. */}
            {target !== null ? (
              <button
                type="button"
                onClick={back}
                aria-label={`Go back to ${destinationNoun(target)}`}
                className="text-muted-foreground hover:text-foreground focus-visible:ring-ring flex items-center gap-1 rounded px-1.5 py-[3px] text-[11px] transition-colors focus-visible:ring-2 focus-visible:outline-none"
              >
                <Undo2 aria-hidden="true" className="size-3" />
                Back to {destinationNoun(target)}
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
