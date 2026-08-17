import { useMemo } from "react";
import { Sparkle } from "lucide-react";
import { cn } from "@/lib/utils";
import { describeCommands, reasonOf } from "@/lib/view/commands";
import { useView } from "@/stores/view";

/**
 * What the conversation did to this view, and what it would still like to do.
 *
 * TWO STATES, ONE PLACE, and they are the two halves of the same claim:
 *
 *  - APPLIED — the model narrowed the view and it moved. The line says so, and
 *    Release hands the whole corpus back. Without it, a graph that dimmed
 *    itself while an answer streamed is a graph that broke, and the only way
 *    out would be to guess that clearing the search field fixes it.
 *  - OFFERED — the model asked and the authority rule declined, because the
 *    person had already taken that dimension this turn. The command is HELD,
 *    not dropped: a model that says "I've pulled that up on the map" over a
 *    view that never moved is worse than one that visibly asks, because the
 *    person is left believing something happened somewhere they are not
 *    looking.
 *
 * An offer outranks an applied line: one is an action waiting on someone, the
 * other is a statement of fact, and there is one 48px bar.
 *
 * IN THE HEADER, WHERE NOTHING IS PUSHED. Every stage in this product has a
 * canvas that fills it and hint strips pinned to its corners, so an overlay
 * would occlude the picture the offer is about. Here it grows the title group,
 * whose caption already truncates first — the title stays left, the status
 * strip and the search field stay right, and nothing the eye is anchored on
 * changes position. It is real buttons in the header's own tab order, so it is
 * reachable by keyboard from the top of the page rather than after a canvas.
 */
/**
 * Whether this component will render anything.
 *
 * Exists so the header can give the offer the CAPTION'S place rather than a
 * place of its own. The caption says what the screen is; while the conversation
 * is doing something to that screen, or asking to, that is the more useful of
 * the two and the caption is a sentence the reader has seen on every visit.
 * Trading them keeps the bar's width constant — otherwise the offer grows the
 * title group and the search field snaps 200px narrower under a hand that may
 * be on its way to it, which is the shift the craft bar forbids.
 */
export function useHasViewNotice(): boolean {
  return useView((s) => Object.keys(s.offers).length > 0 || s.cue !== null || s.notice !== null);
}

export function FollowOffer() {
  const offers = useView((s) => s.offers);
  const cue = useView((s) => s.cue);
  const notice = useView((s) => s.notice);
  const follow = useView((s) => s.follow);
  const dismiss = useView((s) => s.dismiss);
  const release = useView((s) => s.release);

  const pending = useMemo(() => Object.values(offers), [offers]);
  const description = describeCommands(pending);
  /* THE REASON IS QUOTED; THE DESCRIPTION IS NOT. `describeCommands` is this
     app's own account of what a command would do, generated from its shape;
     `reasonOf` is the model's sentence about the evidence, unedited. Rendering
     them as one phrase would make the app's wording read as something the model
     said, which is the same class of misattribution as a chip crediting the
     model for a cue the person typed. So the reason is set off by an em dash
     and — where there is room — carries the quotation the applied line already
     uses for the model's words. */
  const offeredReason = reasonOf(pending);

  if (pending.length > 0 && description.length > 0) {
    return (
      <div
        className={cn(
          "border-primary/40 bg-primary/10 flex min-w-0 shrink items-center gap-2",
          // Capped so a model that composed a long query cannot push the search
          // field off its own edge; the query itself truncates inside.
          "max-w-[560px]",
          "rounded-md border py-0.5 pr-0.5 pl-2",
          // Motion is the only thing that says this arrived rather than having
          // always been here. index.css collapses it under reduced motion.
          "offer-in",
        )}
      >
        <Sparkle aria-hidden="true" className="text-primary size-3 shrink-0" strokeWidth={2} />
        <p className="min-w-0 truncate text-[12px]">
          <span className="text-muted-foreground">The conversation would </span>
          {description}
          {offeredReason ? (
            <span className="text-muted-foreground"> — “{offeredReason}”</span>
          ) : null}
        </p>
        <button
          type="button"
          onClick={follow}
          aria-label={
            offeredReason
              ? `Follow the conversation: ${description}, because ${offeredReason}`
              : `Follow the conversation: ${description}`
          }
          className="bg-primary text-primary-foreground hover:bg-primary/90 focus-visible:ring-ring shrink-0 rounded px-2 py-[3px] text-[11px] font-medium transition-colors focus-visible:ring-2 focus-visible:outline-none"
        >
          Follow
        </button>
        <button
          type="button"
          onClick={dismiss}
          aria-label="Keep this view and decline the conversation's change"
          className="text-muted-foreground hover:text-foreground focus-visible:ring-ring shrink-0 rounded px-1.5 py-[3px] text-[11px] transition-colors focus-visible:ring-2 focus-visible:outline-none"
        >
          Not now
        </button>
      </div>
    );
  }

  /* AN APPLIED MOVE WITH NO CUE IS STILL A MOVE. `direct_view` can open a
     surface without naming a single entity — "these came from an import you ran
     in March", on sources — and before the notice record that produced a silent
     navigation: the stage changed under the reader with nothing anywhere saying
     who did it or why. The cue is no longer the only evidence that the
     conversation touched this view, so it is no longer the gate. */
  if (!cue && !notice) return null;

  return (
    <div
      role="status"
      className="border-border bg-muted/60 flex min-w-0 max-w-[560px] shrink items-center gap-2 rounded-md border py-0.5 pr-0.5 pl-2"
    >
      <Sparkle aria-hidden="true" className="text-primary size-3 shrink-0" strokeWidth={2} />
      <p className="text-muted-foreground min-w-0 truncate text-[12px]">
        {/* "FOLLOWING", NOT "SHOWING", and the difference is honesty rather than
            tone. This line is on every surface, and the cue only visibly
            narrows the graph — and only when its terms name something this
            corpus knows. "Showing what the conversation recalled" over an
            unchanged picture is exactly the invisible claim the Follow
            mechanism exists to prevent, made by the mechanism itself.
            "Following" states what the view bus is doing, which is true on
            every surface and whether or not anything matched.

            The query, not a count: it is the one thing that lets a reader check
            the view against the answer they are reading.

            THE REASON OUTRANKS THE CUE TEXT WHEN THERE IS ONE, and not because
            it is nicer prose. The cue text is already on screen verbatim, in
            the search field two inches to the right — printing it again spends
            the one line this bar has on something the eye can already see,
            while the account of WHY is nowhere else in the interface. When the
            model gave no reason (a narrowing inferred from a recall), the cue
            text is all there is, and it is shown. */}
        Following the conversation —{" "}
        <span className="text-foreground">“{notice ? notice.reason : cue?.text}”</span>
      </p>
      <button
        type="button"
        onClick={release}
        aria-label="Release the conversation's narrowing and show the whole corpus"
        className="text-muted-foreground hover:text-foreground focus-visible:ring-ring shrink-0 rounded px-1.5 py-[3px] text-[11px] transition-colors focus-visible:ring-2 focus-visible:outline-none"
      >
        Release
      </button>
    </div>
  );
}
