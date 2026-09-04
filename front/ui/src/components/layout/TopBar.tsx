import type { ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { GroundToggle } from "./GroundToggle";
import type { Reachability } from "@/lib/api";
import { StatusStrip } from "./StatusStrip";
import { DESTINATIONS } from "./Sidebar";

/** Collapsed rail width, as the offset every fixed element reserves. The rail's
 *  *expanded* width is deliberately never reserved: doing so would make the
 *  expansion push content, which is the one thing it must not do. */
export const RAIL_OFFSET = "pl-56";

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

  return (
    <header
      className={cn(
        "border-sidebar-border bg-sidebar text-sidebar-foreground",
        "absolute inset-x-0 top-0 z-20 flex h-12 items-center gap-3 border-b px-4",
        RAIL_OFFSET,
      )}
    >
      {/* Everything but the title sits right. The rail expands over this bar
          from the left, so anything in its first 224px gets occluded on hover
          — and a status strip sliced mid-word reads as a rendering fault. The
          title is the one thing that can afford to go: while the rail is open
          it is showing its own header, which names the product anyway. The
          caption travels with the title for the same reason, and is the first
          thing to give up width because the title alone still identifies the
          screen. */}
      <div className="flex min-w-0 items-baseline gap-2.5">
        <h1 className="shrink-0 text-[13px] font-medium tracking-tight">{title}</h1>
        {caption ? (
          <p className="text-muted-foreground hidden min-w-0 truncate text-[12px] md:block">
            {caption}
          </p>
        ) : null}
      </div>
      <div className="ml-auto flex min-w-0 flex-1 items-center justify-end gap-3 pl-3">
        <StatusStrip reach={reach} />
        {children}
        <GroundToggle />
      </div>
    </header>
  );
}
