import { useEffect } from "react";
import { HashRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Providers } from "./providers";
import { useReachability } from "./useReachability";
import { useSeatHealth } from "./useSeatHealth";
import { Sidebar, DESTINATIONS } from "@/components/layout/Sidebar";
import { TopBar, RAIL_OFFSET } from "@/components/layout/TopBar";
import { SearchField } from "@/components/layout/SearchField";
import { RecallView } from "@/features/recall/RecallView";
import { GeoView } from "@/features/geo/GeoView";
import { GraphView } from "@/features/graph/GraphView";
import { Inspector } from "@/features/inspector/Inspector";
import { TasksView } from "@/features/tasks/TasksView";
import { BriefingView } from "@/features/briefing/BriefingView";
import { ChatView } from "@/features/chat/ChatView";
import { ConversationOverlay } from "@/features/chat/ConversationOverlay";
import { ProvidersView } from "@/features/providers/ProvidersView";
import { AnomaliesView } from "@/features/anomalies/AnomaliesView";
import { initTheme } from "@/stores/theme";
import { useUiCommands } from "./useUiCommands";
import type { Reachability } from "@/lib/api";

/**
 * The shell.
 *
 * Nothing here is in normal flow except the content itself: the rail, the
 * header and the Inspector are all absolutely positioned, and `main` is offset
 * by their widths. That is what lets the rail expand over the stage instead of
 * shoving it, and what will let the graph canvas take the full width with
 * panels floated on top once it is ported. A flex row cannot do either.
 *
 * `HashRouter`, not `BrowserRouter`, and this is forced rather than chosen:
 * front/src/main.rs serves exactly three routes — `/`, the `/api/{*path}`
 * proxy and the `/seat/{*path}` proxy — with no catch-all. A deep path like
 * `/recall` would 404 on refresh under history routing. The app also ships as
 * one embedded `index.html` inside the Rust binary, where a hash is the only
 * routing that survives being opened from anywhere.
 *
 * Gridline's version strip is deliberately absent: there is no version feed
 * behind it, and chrome that displays invented data is worse than chrome that
 * is absent.
 *
 * Its theme switch was absent too, on the grounds that the product was
 * dark-only and the control would toggle nothing. That stopped being true with
 * the briefing, which is a reading surface rather than a console and is
 * recognised by its paper ground. There are now two grounds and a third state
 * that follows the system -- see stores/theme.ts. The rest of the chrome
 * follows for free because it was already built on semantic tokens.
 */

// Kept in lockstep with the Inspector's own width (Inspector.tsx) — must
// match exactly, or `main`'s reserved space either exposes the Inspector's
// backing surface or clips the content underneath it.
const INSPECTOR_OFFSET = "pr-[min(280px,36vw)]";

/** The Inspector is the detail surface for recall results, so it accompanies
 *  the routes that render them. Recall lists that result set and Geo plots the
 *  geotagged part of the same one — both select into this pane, and a click
 *  that selected something with nowhere to show it would be a dead end. On a
 *  destination with no selectable objects the Inspector could show nothing but
 *  an explanation of itself, so it stays off those. */
const ROUTES_WITH_INSPECTOR = ["/recall", "/geo", "/graph", "/anomalies"];

/** Destinations that render a recall result and therefore need the cue that
 *  produced it. Without the field, Geo would depend on the user having visited
 *  Recall first and would look empty for no stated reason. */
const ROUTES_WITH_SEARCH = ["/recall", "/geo"];

function Shell({ reach }: { reach: Reachability }) {
  const { pathname } = useLocation();
  // Applies the stored ground and keeps the "system" state live. Here rather
  // than at module scope so it unsubscribes with the shell.
  useEffect(() => initTheme(), []);
  // Agent-issued screen changes, applied through the same router and stores
  // the rail and the profile switcher use.
  const announcement = useUiCommands();
  const seat = useSeatHealth();
  const destination = DESTINATIONS.find((d) => d.path === pathname);
  const showInspector = ROUTES_WITH_INSPECTOR.includes(pathname) && reach.state === "online";

  return (
    <div className="relative h-svh min-h-0 overflow-hidden">
      <Sidebar reach={reach} />

      <TopBar title={destination?.label ?? "shodh"} reach={reach}>
        {ROUTES_WITH_SEARCH.includes(pathname) ? <SearchField reach={reach} /> : null}
      </TopBar>

      <main className={cn("h-full pt-12", RAIL_OFFSET, showInspector && INSPECTOR_OFFSET)}>
        <Routes>
          {/* The briefing is the only surface that says something without being
              asked, so it is home. Everything else answers a question you
              already had. */}
          <Route path="/" element={<Navigate to="/briefing" replace />} />
          <Route path="/briefing" element={<BriefingView reach={reach} />} />
          <Route path="/chat" element={<ChatView reach={reach} seat={seat} />} />
          <Route path="/recall" element={<RecallView reach={reach} />} />
          <Route path="/geo" element={<GeoView reach={reach} />} />
          <Route path="/graph" element={<GraphView reach={reach} />} />
          <Route path="/anomalies" element={<AnomaliesView reach={reach} />} />
          <Route path="/tasks" element={<TasksView reach={reach} />} />
          <Route path="/providers" element={<ProvidersView seat={seat} />} />
          {/* A hash the app does not know is a typo or a stale link, not an
              error worth a page — send it home rather than showing a dead end. */}
          <Route path="*" element={<Navigate to="/briefing" replace />} />
        </Routes>
      </main>

      {showInspector ? <Inspector /> : null}

      {/* The agent moved the screen. Announced because a view that changes by
          itself with no explanation reads as a fault, and the operator has to
          be able to tell "the agent did this" from "something broke".
          aria-live so it is not a visual-only event. */}
      {announcement ? (
        <div
          role="status"
          aria-live="polite"
          className={cn(
            "border-border bg-popover text-popover-foreground pointer-events-none",
            "absolute bottom-4 left-1/2 z-40 -translate-x-1/2 rounded-sm border px-3 py-1.5",
            "text-[12px] shadow-lg",
          )}
        >
          <span className="text-muted-foreground font-mono text-[11px]">agent · </span>
          {announcement.text}
        </div>
      ) : null}

      {/* Outside <Routes> deliberately: the conversation is available from
          every destination, so it must not unmount when the route changes —
          unmounting it mid-stream would tear down the panel showing the answer.
          It portals to document.body and returns null on /chat, which is the
          conversation at full width already. */}
      <ConversationOverlay seat={seat} />
    </div>
  );
}

/** Split from `App` because `useReachability` needs to be inside `Providers`
 *  to use the query client, and inside the router to be one instance. */
function Routed() {
  const reach = useReachability();
  return <Shell reach={reach} />;
}

export function App() {
  return (
    <Providers>
      <HashRouter>
        <Routed />
      </HashRouter>
    </Providers>
  );
}
