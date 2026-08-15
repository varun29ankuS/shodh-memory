import { NavLink } from "react-router-dom";
import {
  Search,
  TriangleAlert,
  ListChecks,
  ChevronDown,
  MessageSquare,
  KeyRound,
  Globe,
  Share2,
  House,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { isHumanProfile, type Reachability } from "@/lib/api";
import { useSession } from "@/stores/session";
import shodhMark from "@/assets/shodh-mark.png";

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
 * The destinations, and the one line each of them is.
 *
 * A caption is not a tooltip and not a tagline. It is what someone needs in
 * order to know what they are looking at the moment they arrive, so it says
 * what the DATA on that screen is, in the words a person would use. Four rules
 * it is held to, all of them learned from copy that failed them:
 *
 *  - No subsystem words. "corpus", "session store", "spreading activation" name
 *    parts of this program, not things a reader has.
 *  - True of the screen as it opens, not of the screen after work. Geo used to
 *    promise "the current results" and then opened onto the whole map, which
 *    reads as the wrong screen rather than as a fuller one.
 *  - It labels; it does not sell. No adjective that the screen cannot be
 *    checked against.
 *  - A NOUN PHRASE, NOT A SENTENCE, AND UNDER SIX WORDS. These were clauses —
 *    "Search this memory, and see what connects to what" — sitting permanently
 *    beside a one-word title in a 48px bar, on every screen, on every visit.
 *    The caption has to survive being read at a glance by someone who has read
 *    it forty times before, and a clause does not: it gets skipped, which makes
 *    it cost its space and pay nothing. Naming the data in three words is read
 *    every time.
 *
 * Deliberately NOT moved behind an info affordance, and this is the one place
 * in the density pass where that was the wrong tool. The documented failure
 * this caption was added to fix is people arriving somewhere and not knowing
 * what the data means — and by definition those people do not know there is
 * anything to ask about, so an icon pays out to nobody. It gets shorter; it
 * does not get hidden.
 *
 * Read in three places — the rail itself, the header next to the title
 * (TopBar.tsx), and the trail that stacks them (features/workbench/trail.ts).
 * All three read it from here so the app cannot describe one destination two
 * ways, and so a pane's spine cannot be titled differently from its rail row.
 *
 * BRIEFING IS FIRST AND IT IS A REAL ROW. It used to be reachable only by
 * landing on it: nothing in the rail pointed at `/`, and the wordmark was an
 * image, so once you left the front page the only way back was to edit the
 * URL. A labelled row is the fix rather than a logo link because a row states
 * its destination in a word, while a clickable logo is a convention you have to
 * already know — and the logo is a link now too, which costs nothing and helps
 * the people who do know it.
 */
export const DESTINATIONS = [
  {
    id: "briefing",
    path: "/",
    label: "Briefing",
    icon: House,
    caption: "What is in here, and what changed",
  },
  {
    id: "chat",
    path: "/chat",
    label: "Conversations",
    icon: MessageSquare,
    caption: "A model that can read this memory",
  },
  {
    id: "recall",
    path: "/recall",
    label: "Recall",
    icon: Search,
    caption: "Search memory and its connections",
  },
  {
    id: "graph",
    path: "/graph",
    label: "Graph",
    icon: Share2,
    caption: "Entities, and how they relate",
  },
  {
    id: "geo",
    path: "/geo",
    label: "Geo",
    icon: Globe,
    caption: "Where memory happened",
  },
  {
    id: "anomalies",
    path: "/anomalies",
    label: "Anomalies",
    icon: TriangleAlert,
    caption: "What deviates from this profile's normal",
  },
  {
    id: "tasks",
    path: "/tasks",
    label: "Tasks",
    icon: ListChecks,
    caption: "Open work found in memory",
  },
  {
    id: "providers",
    path: "/providers",
    label: "Providers",
    icon: KeyRound,
    caption: "Where models run, and their keys",
  },
] as const;

export type DestinationId = (typeof DESTINATIONS)[number]["id"];

/** The rail's width, and the offset every fixed element beside it reserves.
 *  One number, exported, because the header and the stage both have to agree
 *  with it exactly — a rail 4px wider than the space reserved for it either
 *  clips the stage or shows a seam of the wrong surface. */
export const RAIL_WIDTH_PX = 244;

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
              to={d.path}
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
