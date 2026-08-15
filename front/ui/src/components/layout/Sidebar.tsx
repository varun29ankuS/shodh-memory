import { NavLink } from "react-router-dom";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { isHumanProfile, type Reachability } from "@/lib/api";
import { useSession } from "@/stores/session";
import shodhMark from "@/assets/shodh-mark.png";
import { railHref } from "@/features/workbench/trail";
import { DESTINATIONS, RAIL_WIDTH_PX } from "./destinations";

/** Re-exported so the header keeps its one import for the table it reads. The
 *  table itself lives in `destinations.ts`, which has no React in it. */
export { DESTINATIONS, RAIL_WIDTH_PX };
export type { DestinationId } from "./destinations";

/**
 * Primary navigation — a permanently labelled column.
 *
 * Names are what the thing is, not what the subsystem is called:
 *   Briefing   — the front page you land on
 *   Recall     — the search surface (was "Live")
 *   Anomalies  — deviations from this user's own baseline (was unqualified)
 *   Tasks      — the todo list (was "Work", which named nothing)
 *
 * IT NO LONGER EXPANDS ON HOVER, and that supersedes `front/ui/DIRECTION.md`
 * §2 of the rail notes, which mandated the opposite. That decision existed to
 * answer a real defect — seven unlabelled glyphs is a memory game — without
 * spending width. Measured against use it failed worse than the defect it
 * fixed: the column animated on every accidental pointer pass, so the rail
 * moved when nobody asked it to, and no decision could be made until the
 * labels had finished arriving. That breaks two rules at once — nothing ever
 * jumps, and hover reveals but never reflows.
 *
 * The density is taken from Linear's shipped client rather than invented:
 * 244px column, 28px × 220px rows inset 12px at an 8px radius, 13px text at
 * weight 450, and NO borders between rows — structure here is rhythm, not
 * rules. A permanently labelled rail costs ~190px and buys an instant,
 * motionless decision, which is the right trade for a surface someone reads
 * all day.
 *
 * With the expansion gone, the machinery that served it goes with it: the
 * width transition, the close delay, the focus-mirrors-hover handler and the
 * clipped-but-present label wrapper all had exactly one job, which no longer
 * exists. Nothing here animates, so `prefers-reduced-motion` has nothing to
 * collapse.
 */

/**
 * BRIEFING IS THE FIRST ROW, AND IT IS A REAL ROW.
 *
 * It used to be reachable only by landing on it: nothing in the rail pointed
 * at `/`, and the wordmark was an inert image, so once you left the front page
 * the only way back was to edit the URL. A labelled row is the fix rather than
 * a logo link, because a row states its destination in a word while a
 * clickable logo is a convention you have to already know — and the logo is a
 * link now too, which costs nothing and pays out to the people who do know it.
 *
 * The table itself is `destinations.ts`.
 */

/**
 * Which profile's memory is on screen.
 *
 * Every option here came from `GET /api/users`; none is typed or guessed. That
 * is a correctness requirement, not tidiness — `get_user_memory`
 * (src/handlers/state.rs) provisions a fresh RocksDB store for any id it has
 * not seen rather than rejecting it, so a free-text profile field would let
 * someone create empty profiles by mistyping.
 *
 * Rendered as a native <select> deliberately. It is one of the few controls
 * that must work before anything else does, and the platform's own is
 * keyboard-complete, screen-reader-correct and impossible to get wrong.
 */
function ProfileSwitcher({ reach }: { reach: Reachability }) {
  const profile = useSession((s) => s.profile);
  const setProfile = useSession((s) => s.setProfile);

  // The seat stores its own lessons under `<user>.seat-harness` — a real
  // backend profile, but machinery, not a person. Offering it here invites
  // exactly the mistake the switcher exists to prevent: a human reading (or
  // writing!) the harness's internal scope as if it were their memory.
  const humanProfiles = reach.state === "online"
    ? reach.profiles.filter(isHumanProfile)
    : [];

  if (reach.state !== "online" || humanProfiles.length === 0 || !profile) return null;

  const single = humanProfiles.length === 1;

  return (
    <div className="px-3 pt-2 pb-3">
      <div className="flex h-7 items-center gap-2 px-2">
        <span className="bg-muted-foreground/40 size-1.5 shrink-0 rounded-full" />
        {single ? (
          <span className="text-muted-foreground min-w-0 flex-1 truncate text-[12px]">
            {profile}
          </span>
        ) : (
          <span className="relative flex min-w-0 flex-1 items-center gap-1">
            <select
              aria-label="Active profile"
              value={profile}
              onChange={(e) => setProfile(e.target.value)}
              className="text-muted-foreground hover:text-foreground focus-visible:ring-ring min-w-0 flex-1 cursor-pointer appearance-none truncate bg-transparent text-[12px] focus-visible:ring-2 focus-visible:outline-none"
            >
              {humanProfiles.map((p) => (
                <option key={p} value={p} className="bg-popover text-popover-foreground">
                  {p}
                </option>
              ))}
            </select>
            <ChevronDown aria-hidden="true" className="text-muted-foreground size-3 shrink-0" />
          </span>
        )}
      </div>
    </div>
  );
}

export function Sidebar({ reach }: { reach: Reachability }) {
  return (
    <aside
      aria-label="Primary navigation"
      style={{ width: `${RAIL_WIDTH_PX}px` }}
      className={cn(
        "border-sidebar-border bg-sidebar text-sidebar-foreground",
        "absolute inset-y-0 left-0 z-30 flex flex-col overflow-hidden border-r",
      )}
    >
      {/* The wordmark is a link to the briefing — the conventional way home,
          kept alongside the labelled row rather than instead of it. 48px to
          sit on the header's baseline, since the two share a rule. */}
      <NavLink
        to="/"
        end
        aria-label="shodh — the briefing"
        className="border-sidebar-border focus-visible:ring-ring flex h-12 shrink-0 items-center gap-2.5 border-b px-3 focus-visible:ring-2 focus-visible:outline-none focus-visible:-outline-offset-2"
      >
        <img src={shodhMark} alt="" aria-hidden="true" className="size-6 shrink-0 object-contain" />
        <span className="text-[13px] font-semibold tracking-tight">shodh</span>
      </NavLink>

      <nav aria-label="Destinations" className="flex flex-col gap-0.5 px-3 py-3">
        {DESTINATIONS.map((d) => {
          const Icon = d.icon;
          return (
            <NavLink
              key={d.id}
              // `railHref`, not the bare path. The rail is a JUMP to a place,
              // not an opening-from-here, so its links state an empty ancestry
              // and reset the trail to `[briefing, destination]`. A bare path
              // means the opposite — see `features/workbench/trail.ts`.
              // `NavLink` matches on pathname alone, so the active row is
              // unaffected by the parameter.
              to={railHref(d.path)}
              // `/` is a prefix of every path, so without `end` the briefing
              // row would read as active on every screen in the product.
              end={d.path === "/"}
              aria-label={`${d.label} — ${d.caption}`}
              className={({ isActive }) =>
                cn(
                  "flex h-7 shrink-0 items-center gap-2 rounded-lg px-2 text-left text-[13px] font-[450]",
                  // Colour only. Nothing here changes size, weight or position
                  // on hover, so the row cannot shift the thing under the
                  // pointer while it is being aimed at.
                  "transition-colors duration-100",
                  "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                )
              }
            >
              <Icon aria-hidden="true" className="size-4 shrink-0" strokeWidth={1.7} />
              <span className="truncate">{d.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="mt-auto shrink-0">
        <ProfileSwitcher reach={reach} />
      </div>
    </aside>
  );
}
