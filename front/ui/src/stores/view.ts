import { create } from "zustand";
import { useSession } from "./session";
import { decide, holdsAt, type Verdict, type ViewAuthor, type ViewDimension } from "@/lib/view/authority";
import type { ViewCommand } from "@/lib/view/commands";

/**
 * The view bus — one store, one `dispatch(command, author)`, two producers
 * (spec §5).
 *
 * WHAT THIS OWNS AND WHAT IT DOES NOT. It owns the authority ledger (who has
 * touched what since the turn began), the declined commands waiting as Follow
 * offers, and the two dimensions nothing else owned: the camera (`frame`) and
 * the stage (`destination`). It does NOT own the cue — `stores/session.ts`
 * already does, every surface already reads it from there, and a second copy
 * would be a second truth for the canvas to reconcile. Applying a cue command
 * writes through to the session store, and the record kept here is provenance,
 * not state: it is how the chip can say the model set this and how Release
 * knows there is something to release.
 *
 * NOTHING HERE NAVIGATES OR PAINTS. `destination` and `frame` are records with
 * a monotonic `seq`, and the surfaces that can act on them — the router, the
 * canvas — watch that number. The alternative, handing the store a `navigate`
 * or a canvas ref, makes the bus depend on which components happen to be
 * mounted, and a command dispatched a frame before its consumer mounts would be
 * lost. Records survive that; a callback does not, which is exactly what makes
 * "Follow, then arrive already framed" work.
 */

/** The camera, aimed at the entities a command named. */
export interface FrameRecord {
  entities: string[];
  seq: number;
}

/** The stage a command asked for. */
export interface DestinationRecord {
  path: string;
  seq: number;
}

/** Provenance for a cue the model set, so the chip is never a guess. */
export interface AgentCueRecord {
  text: string;
  entities: string[];
  seq: number;
}

interface ViewState {
  /** Dimensions the human owns for the rest of this turn. */
  claimed: ViewDimension[];
  /** Commands declined by the authority rule, one per dimension, newest wins. */
  offers: Partial<Record<ViewDimension, ViewCommand>>;
  cue: AgentCueRecord | null;
  frame: FrameRecord | null;
  destination: DestinationRecord | null;
  /** Monotonic, shared by every record, so a consumer can key an effect on it. */
  seq: number;

  /** A new turn hands the wheel back, less whatever the current surface holds. */
  beginTurn: (path: string) => void;
  dispatch: (command: ViewCommand, author: ViewAuthor) => Verdict;
  /** The human moved this dimension themselves. */
  touch: (dimension: ViewDimension) => void;
  /** Accept every waiting offer. */
  follow: () => void;
  /** Refuse them, visibly and once. */
  dismiss: () => void;
  /** Drop the model's cue and hand the whole corpus back. */
  release: () => void;
}

export const useView = create<ViewState>((set, get) => ({
  claimed: [],
  offers: {},
  cue: null,
  frame: null,
  destination: null,
  seq: 0,

  beginTurn: (path) =>
    set({
      claimed: holdsAt(path),
      // Offers do not outlive the turn that produced them. A Follow chip
      // offering to show what the model recalled two questions ago is an
      // invitation to a view of the wrong thing.
      offers: {},
    }),

  dispatch: (command, author) => {
    const state = get();
    const verdict = decide(author, command.dimension, state.claimed);

    if (verdict === "offer") {
      set({ offers: { ...state.offers, [command.dimension]: command } });
      return verdict;
    }

    const seq = state.seq + 1;
    // The dimension is no longer contested, so any older offer for it is stale
    // whichever way it was resolved.
    const offers = { ...state.offers };
    delete offers[command.dimension];
    // Following is itself an act of the hand: the person chose this, so they
    // hold the dimension for the rest of the turn and the model does not get to
    // move it again underneath them.
    const claimed =
      author === "user" && !state.claimed.includes(command.dimension)
        ? [...state.claimed, command.dimension]
        : state.claimed;

    switch (command.dimension) {
      case "cue":
        // ORDER IS LOAD-BEARING. The adapter distinguishes a typed cue from
        // this one by comparing the session's cue against this record, and
        // zustand notifies subscribers synchronously — so the record must
        // already be in place when `setCue` fires, or the adapter reads a stale
        // record, concludes a human typed it and claims the dimension.
        set({ cue: { text: command.text, entities: command.entities, seq }, seq, offers, claimed });
        useSession.getState().setCue(command.text, command.entities);
        break;
      case "frame":
        set({ frame: { entities: command.entities, seq }, seq, offers, claimed });
        break;
      case "destination":
        set({ destination: { path: command.path, seq }, seq, offers, claimed });
        break;
    }

    return verdict;
  },

  touch: (dimension) =>
    set((s) => {
      const offers = { ...s.offers };
      delete offers[dimension];
      return {
        claimed: s.claimed.includes(dimension) ? s.claimed : [...s.claimed, dimension],
        offers,
        // A cue the person typed replaces the model's, so the attribution has
        // to go with it — a chip claiming the model narrowed this while the
        // narrowing is the user's own word is the precise lie this whole
        // mechanism exists to prevent.
        cue: dimension === "cue" ? null : s.cue,
      };
    }),

  follow: () => {
    const offers = get().offers;
    // Dispatched as the USER, because that is who is asking: the command was
    // the model's, the decision to apply it is not.
    for (const command of Object.values(offers)) get().dispatch(command, "user");
  },

  // Refusing is a decision, not a shrug: the dimensions it covered become the
  // person's for the rest of the turn, so a second recall in the same answer
  // cannot re-ask a question that has just been answered no.
  dismiss: () =>
    set((s) => {
      const claimed = [...s.claimed];
      for (const dimension of Object.keys(s.offers) as ViewDimension[]) {
        if (!claimed.includes(dimension)) claimed.push(dimension);
      }
      return { claimed, offers: {} };
    }),

  release: () => {
    // Releasing is the person taking the cue back by hand, so it claims the
    // dimension the same way typing into the field does.
    get().touch("cue");
    useSession.getState().setCueDraft("");
    set({ cue: null, frame: null });
  },
}));
