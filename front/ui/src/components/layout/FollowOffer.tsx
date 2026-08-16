import { useMemo } from "react";
import { Sparkle } from "lucide-react";
import { cn } from "@/lib/utils";
import { describeCommands } from "@/lib/view/commands";
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
export function FollowOffer() {
  const offers = useView((s) => s.offers);
  const cue = useView((s) => s.cue);
  const follow = useView((s) => s.follow);
  const dismiss = useView((s) => s.dismiss);
  const release = useView((s) => s.release);

  const pending = useMemo(() => Object.values(offers), [offers]);
  const description = describeCommands(pending);

  if (pending.length > 0 && description.length > 0) {
    return (
      <div
        className={cn(
          "border-primary/40 bg-primary/10 flex min-w-0 shrink items-center gap-2",
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
        </p>
        <button
          type="button"
          onClick={follow}
          aria-label={`Follow the conversation: ${description}`}
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

  if (!cue) return null;

  return (
    <div
      role="status"
      className="border-border bg-muted/60 flex min-w-0 shrink items-center gap-2 rounded-md border py-0.5 pr-0.5 pl-2"
    >
      <Sparkle aria-hidden="true" className="text-primary size-3 shrink-0" strokeWidth={2} />
      <p className="text-muted-foreground min-w-0 truncate text-[12px]">
        {/* The query, not a count: it is the one thing that lets a reader check
            the narrowing against the answer they are reading. */}
        Showing what the conversation recalled for{" "}
        <span className="text-foreground">“{cue.text}”</span>
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
