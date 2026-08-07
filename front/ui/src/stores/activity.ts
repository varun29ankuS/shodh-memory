import { create } from "zustand";
import type { SeatEvent } from "@/lib/seat/types";

/**
 * What the agent just did, for the surfaces that are not the conversation.
 *
 * The chat store is the transcript: it accumulates every event into turns and
 * is replayed verbatim when a conversation is reopened. This store is the
 * opposite kind of thing — a short-lived signal that something happened NOW, so
 * a graph, a map or a result list can move while the model works instead of
 * sitting inert beside it.
 *
 * Two rules make this honest rather than decorative, and both are enforced here
 * rather than left to each caller:
 *
 *  1. **Only ids the event actually named.** A pulse is a claim that the model
 *     touched *these* memories. `subscribePulse` intersects the event's ids
 *     with what the calling surface draws and stays silent when the
 *     intersection is empty — a canvas holding none of them must do nothing,
 *     because a pulse with no matching node would be an animation pretending to
 *     be evidence.
 *  2. **Only the user scope reaches a canvas.** `memory_recall`,
 *     `memory_write` and `memory_reinforce` all carry
 *     `scope: "user" | "harness"` (seat/src/events.ts), and the harness writes
 *     under its own `user_id` — a different store, a different id space.
 *     Intersecting a harness id against the profile's nodes could only ever
 *     collide by accident. Harness activity is still reported, but as a notice
 *     that says so, never as a highlight on the profile's corpus.
 *
 * Nothing here is durable. The record of a write or a reinforcement — with its
 * revert — lives in the conversation block, which is where an action a person
 * may need to undo belongs.
 */

/** How long a pulse stays lit. Long enough to catch the eye from another pane,
 *  short enough that it cannot be mistaken for persistent state. */
export const PULSE_MS = 1_800;

/** How long a notice stays on screen. */
export const NOTICE_MS = 6_000;

/** Notices are ephemeral; more than a few at once is a log, not a signal. */
const MAX_NOTICES = 3;

export interface Pulse {
  /** Memory ids the event named, before any surface has filtered them. */
  ids: ReadonlySet<string>;
  /** Monotonic. Two consecutive events naming the same memories are still two
   *  events, and subscribers compare pulse identity rather than contents. */
  seq: number;
}

export interface LiveNotice {
  id: string;
  text: string;
  /** Wall clock, so a re-registered timer expires on the original deadline
   *  instead of restarting the clock every time the list changes. */
  at: number;
}

interface ActivityState {
  pulse: Pulse | null;
  notices: LiveNotice[];
  expireNotice: (id: string) => void;
}

export const useActivity = create<ActivityState>((set) => ({
  pulse: null,
  notices: [],
  expireNotice: (id) =>
    set((s) => {
      const notices = s.notices.filter((n) => n.id !== id);
      return notices.length === s.notices.length ? s : { notices };
    }),
}));

let seq = 0;

/**
 * One line about a corpus-changing event, in the words of the fields that
 * carry it. No field is combined with another to imply something neither says.
 */
function noticeFor(event: SeatEvent): string | null {
  switch (event.type) {
    case "memory_write":
      return event.scope === "harness"
        ? `Wrote a harness memory · ${event.memory_type}`
        : `Wrote a memory · ${event.memory_type}`;

    case "memory_reinforce": {
      const n = event.memory_ids.length;
      const noun = n === 1 ? "memory" : "memories";
      const where = event.scope === "harness" ? " in harness memory" : "";
      // The outcome decides both the verb and which stat is meaningful:
      // reporting "boosted" on a weakening pass would misdescribe it.
      switch (event.outcome) {
        case "helpful":
          return `Reinforced ${n} ${noun}${where} · ${event.stats.importance_boosts} boosted`;
        case "misleading":
          return `Weakened ${n} ${noun}${where} · ${event.stats.importance_decays} decayed`;
        default:
          return `Reviewed ${n} ${noun}${where}`;
      }
    }

    case "harness_learning_applied": {
      const n = event.memories.length;
      if (n === 0) return null;
      return `Harness applied ${n} learned ${n === 1 ? "memory" : "memories"}`;
    }

    default:
      return null;
  }
}

/**
 * Memory ids a live event surfaced, or `null` when the event surfaces nothing.
 *
 * Only the two events that report retrieval qualify. A write names an id too,
 * but the memory it names did not exist when the surfaces on screen were drawn,
 * so lighting it up could only ever light nothing.
 */
function pulseIdsFor(event: SeatEvent): string[] | null {
  switch (event.type) {
    case "memory_recall":
      // Harness recall reads the harness's own store — see the scope rule above.
      if (event.scope !== "user") return null;
      return event.memories.map((m) => m.id);
    case "proactive_context":
      return event.injected_memory_ids;
    default:
      return null;
  }
}

/**
 * Fold one live event into the activity signal.
 *
 * Called from the chat store's stream handler, and deliberately NOT from
 * `adoptDetail`: replaying a persisted conversation would otherwise fire every
 * pulse and notice of a conversation that finished hours ago. "The software
 * moves as the agent acts" is a statement about the present tense.
 */
export function noteSeatEvent(event: SeatEvent): void {
  const ids = pulseIdsFor(event);
  const text = noticeFor(event);
  if (!ids?.length && text === null) return;

  seq += 1;
  const at = Date.now();
  useActivity.setState((s) => ({
    pulse: ids?.length ? { ids: new Set(ids), seq } : s.pulse,
    notices:
      text === null
        ? s.notices
        : [...s.notices, { id: `n${seq}`, text, at }].slice(-MAX_NOTICES),
  }));
}

/**
 * Watch for pulses this surface can honestly show.
 *
 * `present` is the surface's own membership test. When none of the event's ids
 * pass it, `onHit` is never called: the correct reaction to activity elsewhere
 * in the corpus is no reaction at all.
 */
export function subscribePulse(
  present: (id: string) => boolean,
  onHit: (ids: ReadonlySet<string>) => void,
): () => void {
  return useActivity.subscribe((state, previous) => {
    const pulse = state.pulse;
    if (pulse === null || pulse === previous.pulse) return;
    const hit = new Set<string>();
    for (const id of pulse.ids) if (present(id)) hit.add(id);
    if (hit.size === 0) return;
    onHit(hit);
  });
}

export interface CanvasPulse {
  ids: ReadonlySet<string>;
  /** `performance.now()` at the start, for phase. */
  at: number;
  /** Motion is reduced: hold one frame rather than interpolate. */
  still: boolean;
}

/**
 * The pulse clock for a canvas.
 *
 * Canvas-side and ref-based on purpose. Both canvases in this app paint
 * manually from refs precisely so a 60 Hz signal never re-renders the React
 * tree; driving the pulse through component state would reintroduce exactly the
 * cost that architecture exists to avoid, and — in the lineage canvas — would
 * risk restarting the force simulation on every frame.
 */
export function createPulseRunner(redraw: () => void) {
  const ref: { current: CanvasPulse | null } = { current: null };
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let raf = 0;
  let timer = 0;

  const stop = () => {
    if (raf) cancelAnimationFrame(raf);
    if (timer) clearTimeout(timer);
    raf = 0;
    timer = 0;
  };

  return {
    ref,
    /** Light `ids`. Callers pass only ids their surface actually draws. */
    start(ids: ReadonlySet<string>) {
      if (ids.size === 0) return;
      stop();
      ref.current = { ids, at: performance.now(), still };
      redraw();

      if (still) {
        // Appears, then is removed. No interpolation, matching the global
        // reduced-motion rule in index.css and the settled-layout treatment
        // the force simulation already gets.
        timer = window.setTimeout(() => {
          ref.current = null;
          redraw();
        }, PULSE_MS);
        return;
      }

      const frame = () => {
        const live = ref.current;
        if (live === null) {
          raf = 0;
          return;
        }
        if (performance.now() - live.at >= PULSE_MS) {
          ref.current = null;
          redraw();
          raf = 0;
          return;
        }
        redraw();
        raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);
    },
    cancel() {
      stop();
      ref.current = null;
    },
  };
}

/** 0 → 1 across a pulse's life; one fixed frame when motion is reduced. */
export function pulsePhase(pulse: CanvasPulse, now: number): number {
  if (pulse.still) return 0.34;
  return Math.min(1, Math.max(0, (now - pulse.at) / PULSE_MS));
}
