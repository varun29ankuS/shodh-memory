import type { ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import { StatusStrip } from "./StatusStrip";
import { SystemBanner, SystemPulse } from "./SystemRibbon";
import { useSystemHealth } from "./useSystemHealth";
import { FollowOffer, useHasViewNotice } from "./FollowOffer";
import { DESTINATIONS } from "./Sidebar";
import { RAIL_OFFSET } from "./destinations";

/** The rail's width as a padding utility, re-exported so every existing
 *  importer keeps its one import. It is defined next to the number it has to
 *  agree with — see `destinations.ts`. */
export { RAIL_OFFSET };

/**
 * The header — and the one place every destination says what it is.
 *
 * A name is not an explanation. "Geo" tells someone which button they pressed;
 * it does not tell them that the points are memories and that the map is the
 * whole corpus until they search. Arriving somewhere and not knowing what the
 * data means was the single most repeated reaction to this app, and the fix has
 * to be visible without hovering, because a tooltip only pays out to someone who
 * already suspected there was something to learn.
 *
 * The caption is read from `DESTINATIONS` rather than passed in, so the rail and
 * the header cannot describe the same destination two different ways, and a new
 * destination gets its line by existing rather than by remembering to wire one.
 * The `title` prop still comes from the shell, and still wins: it is what the
 * router decided is on screen, and this component is not the place to disagree
 * with it.
 *
 * One line, not a stacked title and subtitle: the bar is 48px, and a two-line
 * block inside it is cramped everywhere and clipped at the sizes where the
 * status strip is also fighting for room.
 */
export function TopBar({
  title,
  reach,
  children,
}: {
  title: string;
  reach: Reachability;
  children?: ReactNode;
}) {
  const { pathname } = useLocation();
  const caption = DESTINATIONS.find((d) => d.path === pathname)?.caption;
  // The caption yields its place to the conversation's notice rather than
  // sharing the bar with it — see `useHasViewNotice`.
  const notice = useHasViewNotice();
  // Both services, in one reading. The header is where the system says whether
  // it is alive, so it is where the ribbon is assembled — see `SystemRibbon`
  // for why the pulse and the banner are siblings of the header and not
  // children of it.
  const { readings, tone, alerts } = useSystemHealth(reach);

  return (
    <>
      <SystemPulse tone={tone} />
      <header
        className={cn(
          "border-sidebar-border bg-sidebar text-sidebar-foreground",
          "absolute inset-x-0 top-0 z-20 flex h-12 items-center gap-3 border-b px-4",
          RAIL_OFFSET,
        )}
      >
        {/* Everything but the title sits right. This used to be a defence
            against the rail expanding over the bar's first 224px on hover; the
            rail no longer expands, so nothing is occluded and the arrangement
            survives on its own merit — the title identifies the screen and reads
            left, the status of the system reads right, and the caption is the
            first thing to give up width because the title alone still says where
            you are. */}
        {/* `items-center`, not `items-baseline`: the offer is a bordered control
            rather than a line of text, and on a baseline it hangs low enough to
            crowd the bar's underline. The title and caption are unaffected —
            both are single-line at the same size. */}
        <div className="flex min-w-0 items-center gap-2.5">
          <h1 className="shrink-0 text-[13px] font-medium tracking-tight">{title}</h1>
          {caption && !notice ? (
            <p className="text-muted-foreground hidden min-w-0 truncate text-[12px] md:block">
              {caption}
            </p>
          ) : null}
          {/* What the conversation did to this view, and what it is still asking
              to do. It sits with the title because it describes the screen, not
              the system — the status strip on the right is about the server. */}
          <FollowOffer />
        </div>
        <div className="ml-auto flex min-w-0 flex-1 items-center justify-end gap-3 pl-3">
          <StatusStrip readings={readings} />
          {children}
        </div>
      </header>

      <SystemBanner alerts={alerts} />
    </>
  );
}
