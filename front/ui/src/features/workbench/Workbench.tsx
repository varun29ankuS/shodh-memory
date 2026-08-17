import { useEffect, useRef, useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";
import { useSession } from "@/stores/session";
import { Spine, SPINE_WIDTH_PX } from "./Spine";
import {
  ROOT,
  VIA_PARAM,
  backHref,
  hrefFor,
  paneByPath,
  parseTrail,
  promoteTrail,
  spineOrdinal,
  type Pane,
} from "./trail";
import {
  escapeVerdict,
  hasTransientSurface,
  isConversationExpanded,
  isEditable,
} from "./escape";

/**
 * The workbench — one continuous surface instead of a set of destinations.
 *
 * You land on the briefing. Selecting a door does not navigate away from it:
 * the door's content is PROMOTED to the stage and the briefing COMPRESSES to a
 * spine beside it, which is both the record of where you came from and the way
 * back. Open something from there and that pane compresses in turn. The result
 * is a stack you can read: `Briefing │ Recall │ ▐ graph ▌`.
 *
 * ONE PRIMARY AT A TIME, and it is enforced by construction rather than by
 * discipline: the primary is the last pane in the trail, `<Routes>` renders
 * exactly one view, and every other pane is 40px of spine. There is no
 * arrangement of this component that produces two co-equal panels, which is
 * the point — the rule that keeps a workbench from silently becoming a
 * dashboard has to be structural, because every addition that breaks it is
 * individually reasonable.
 *
 * COMPRESSED IS NOT HIDDEN, AND IT IS NOT CLIPPED. A pane is either full or a
 * spine; there is no partly-occluded state. The view a compressed pane held is
 * unmounted rather than slid under its neighbour, so no word is ever cut
 * through mid-glyph, and its spine states its full title at body size.
 *
 * THE GEOMETRY IS `flex-grow`, TRANSITIONED. Every pane has a 40px flex basis;
 * the primary is the only one with `flex-grow: 1`. Promotion moves that 1 from
 * one pane to another and the browser interpolates it — `flex-grow` is a
 * number, so it animates everywhere. (`grid-template-columns` between `40px`
 * and `1fr` does not: those units do not interpolate and the change would
 * snap.) A newly mounted pane is given the grow on the frame AFTER it appears,
 * so it opens from a 40px sliver instead of taking half the stage on its first
 * frame. `prefers-reduced-motion` is collapsed globally in `index.css`, which
 * turns all of this into an instant state change with no branch here.
 *
 * STICKY, NOT MEASURED. Spines are `position: sticky` at increasing left
 * offsets, so if the row is ever wider than the stage they pin in reading
 * order and the primary scrolls under them rather than the first spine
 * scrolling out of reach. No scroll listener, no `getBoundingClientRect`,
 * nothing that can be one frame out of date. In the arrangement below the row
 * does not normally overflow — the primary takes whatever the spines leave —
 * so this is the guarantee that holds when a view brings an intrinsic minimum
 * with it, not a mechanism that runs on every screen.
 */

/**
 * The trail this location describes, and the URL kept in agreement with it.
 *
 * A bare `navigate("/recall")` — which is what the briefing's doors, the
 * evidence panel and the conversation dock all do, none of which has heard of
 * a trail — carries no `via`, and is read here as "open this from where I am".
 * The promoted trail is computed DURING RENDER rather than in an effect, so
 * the stack draws correctly on the first frame; the effect afterwards only
 * writes the same conclusion into the URL, with `replace`, so the path stays
 * bookmarkable and the history stays one entry per real move.
 */
function useTrail(): Pane[] {
  const { pathname, search } = useLocation();
  const navigate = useNavigate();

  const declared = new URLSearchParams(search).has(VIA_PARAM);
  const target = paneByPath(pathname);
  const previous = useRef<Pane[]>([ROOT]);

  // The briefing is the base of every trail, so arriving at it is a reset and
  // never a promotion — there is no arrangement in which it is not index 0.
  const inherits = !declared && target !== null && target.id !== ROOT.id;
  const trail = inherits
    ? promoteTrail(previous.current, previous.current.length - 1, target.id)
    : parseTrail(pathname, search);

  useEffect(() => {
    previous.current = trail;
  });

  const href = hrefFor(trail, trail.length - 1);
  useEffect(() => {
    if (!inherits) return;
    navigate(href, { replace: true });
  }, [inherits, href, navigate]);

  return trail;
}

/**
 * Escape, and who gets it.
 *
 * Registered in the CAPTURE phase deliberately. React attaches its own
 * handlers at the root container, so a bubble-phase window listener runs after
 * them: with a node selected on the graph canvas, the canvas would clear the
 * selection and this would then read a store with nothing in it and pop a pane
 * — one keypress, two levels. Capturing means the world is read before
 * anything has changed it. `escape.ts` decides; this only carries the verdict
 * out.
 */
function useEscape(trail: Pane[]) {
  const navigate = useNavigate();
  const current = useRef(trail);
  current.current = trail;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      const session = useSession.getState();
      const verdict = escapeVerdict({
        editable: isEditable(e.target),
        transientOpen: hasTransientSurface(document),
        conversationExpanded: isConversationExpanded(document),
        hasSelection:
          session.selectedMemoryId !== null || session.selectedEntityId !== null,
        canGoBack: backHref(current.current) !== null,
      });

      if (verdict === "clear-selection") {
        e.preventDefault();
        // Clears both ids: `select` is the store's one "nothing is open" path.
        session.select(null);
        return;
      }
      if (verdict !== "back") return;
      const href = backHref(current.current);
      if (!href) return;
      e.preventDefault();
      navigate(href);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [navigate]);
}

export function Workbench({ children }: { children: ReactNode }) {
  const trail = useTrail();
  const navigate = useNavigate();
  useEscape(trail);

  const primaryIndex = trail.length - 1;
  const primaryId = trail[primaryIndex].id;

  // Which pane currently holds the grow. Behind `primaryId` by one frame on
  // purpose — see the note above: a pane that mounts already grown takes its
  // share of the stage instantly and there is nothing left to animate.
  const [grownId, setGrownId] = useState(primaryId);
  useEffect(() => {
    const frame = requestAnimationFrame(() => setGrownId(primaryId));
    return () => cancelAnimationFrame(frame);
  }, [primaryId]);

  return (
    <div className="flex h-full min-h-0 overflow-x-auto overflow-y-hidden">
      {trail.map((pane, i) => {
        const isPrimary = i === primaryIndex;
        return (
          <div
            key={pane.id}
            style={{
              flexBasis: `${SPINE_WIDTH_PX}px`,
              flexGrow: pane.id === grownId ? 1 : 0,
              minWidth: `${SPINE_WIDTH_PX}px`,
              // Increasing offsets, so a stack too wide for the stage pins in
              // reading order rather than all at zero.
              left: isPrimary ? undefined : `${i * SPINE_WIDTH_PX}px`,
            }}
            className={cn(
              "h-full min-h-0 overflow-hidden",
              isPrimary ? "relative z-0" : "sticky z-10",
              "transition-[flex-grow] duration-300 ease-[cubic-bezier(0.2,0,0,1)]",
            )}
          >
            {isPrimary ? (
              // Exactly the width the pane has, and NO MINIMUM. A floor here
              // was tried and removed after measuring it: the pane clips its
              // contents, so on a narrow window a floored view lost the
              // difference off its right edge with no scrollbar and no way to
              // reach it — content silently deleted rather than a stage that
              // is merely tight. The views are responsive; a narrow pane is a
              // narrow window, which they already handle.
              <div className="h-full">{children}</div>
            ) : (
              <Spine
                title={pane.title}
                caption={pane.caption}
                ordinal={spineOrdinal(trail.length, i)}
                onOpen={() => navigate(hrefFor(trail, i))}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
