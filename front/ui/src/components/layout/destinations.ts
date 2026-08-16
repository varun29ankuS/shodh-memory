import {
  Search,
  TriangleAlert,
  ListChecks,
  MessageSquare,
  KeyRound,
  Globe,
  History,
  Inbox,
  Share2,
  House,
} from "lucide-react";

/**
 * The destinations, and the one line each of them is.
 *
 * SPLIT OUT OF `Sidebar.tsx` so it can be read by things that are not
 * components. The trail (`features/workbench/trail.ts`) titles every pane from
 * this table, and its tests are plain functions — importing the rail component
 * to learn what a pane is called would drag React, an image asset and a
 * zustand store into a test that computes strings.
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
 * Read in three places — the rail (`Sidebar.tsx`), the header next to the
 * title (`TopBar.tsx`), and the trail that stacks them
 * (`features/workbench/trail.ts`). All three read it from here, so the app
 * cannot describe one destination two ways and a pane's spine cannot be
 * titled differently from its rail row.
 *
 * BRIEFING IS FIRST, AND THAT ORDER IS LOAD-BEARING: the trail's root pane is
 * the entry whose path is `/`, so the briefing is the base of every trail and
 * the way back out of every destination. It used to be reachable only by
 * landing on it — nothing in the rail pointed at `/` — so leaving the front
 * page meant editing the URL to return.
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
    // NOT "found in memory". That implied extraction, and there is none:
    // store_todo has three callers -- the create handler, MIF import and
    // recurrence rollover (src/handlers/todos.rs, src/memory/types.rs) -- and
    // nothing anywhere turns memory text into a task. Todos are RECORDED by
    // explicit calls. A caption that credits the product with a capability it
    // does not have is the same class of error as a legend that disagrees with
    // its picture, and it sat on the rail on every screen.
    caption: "Work recorded against this profile",
  },
  {
    id: "history",
    path: "/history",
    label: "History",
    icon: History,
    // NOT "everything that happened", and not "audit log". Two of the four
    // rules bite here. The screen shows the conversation SEAT's trail — the
    // memory server keeps its own, in RocksDB's CF_AUDIT, which is not served
    // over HTTP and rotates on a retention timer — so any caption spanning the
    // whole system would credit the product with completeness it does not
    // have, the same failure the Tasks caption above was corrected for. And
    // "audit log" names a genre rather than the data: the three things on that
    // screen are tool calls, changes to memory, and retrievals, which is what
    // a reader is looking for and what they will recognise on arrival.
    caption: "Tool calls, changes, retrievals",
  },
  {
    id: "sources",
    path: "/sources",
    label: "Sources",
    icon: Inbox,
    // NOT "What feeds this profile", and the difference is the whole screen.
    // Two things write into a profile and only one of them leaves anything
    // readable behind: the Claude Code hook records a summary when a session
    // ends, while MIF import writes memories under their original timestamps
    // with no marker at all and logs its run only to the server's internal
    // audit trail, which no HTTP route serves. A caption spanning every source
    // would promise a completeness that cannot exist, because a stored memory
    // has no origin field to read (src/memory/types.rs) — the same class of
    // error as Tasks' "found in memory". Naming the restriction in the caption
    // is what keeps it true of the screen as it opens.
    caption: "Sources that leave a record",
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

/** The rail's width. The rail no longer expands on hover, so this is its only
 *  width and it is reserved outright by everything beside it. */
export const RAIL_WIDTH_PX = 244;

/**
 * The same number as a padding utility, for the header and the stage.
 *
 * WRITTEN OUT RATHER THAN INTERPOLATED FROM `RAIL_WIDTH_PX`. Tailwind v4
 * generates utilities by scanning source text for complete class names, so
 * `pl-[${n}px]` yields a class that is never emitted and an offset of zero — a
 * failure that appears only in the built product, where the rail then sits on
 * top of the stage. `trail.test.ts` asserts the two agree, which is the check
 * that makes writing it out safe.
 */
export const RAIL_OFFSET = "pl-[244px]";
