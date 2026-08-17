import { create } from "zustand";
import { useSession } from "./session";
import { decide, holdsAt, type Verdict, type ViewAuthor, type ViewDimension } from "@/lib/view/authority";
import type { ViewCommand } from "@/lib/view/commands";
import { verdictsForDispatch, verdictsForEndedOffers, type ViewVerdict } from "@/lib/view/outcome";

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

/**
 * The stage a command asked for, and the one it took the person off.
 *
 * `from` IS THE INVERSE, HELD WITH THE MOVE. Every other axis this bus owns can
 * be undone by doing the ordinary thing — clear the field, pan the graph, click
 * another node — and the destination could not: the reader whose screen was
 * swapped mid-sentence had to remember where they had been. The stack the hash
 * router keeps is not a substitute, because it is the browser's own history and
 * stepping back through it walks past every navigation the person made by hand.
 *
 * `null` where there is nothing to go back to. See `ViewCommand`'s `from`.
 */
export interface DestinationRecord {
  path: string;
  from: string | null;
  seq: number;
}

/** Provenance for a cue the model set, so the chip is never a guess. */
export interface AgentCueRecord {
  text: string;
  entities: string[];
  seq: number;
}

/**
 * The one object a command opened, and the graph's name for it.
 *
 * THE SELECTION ITSELF LIVES IN `stores/session.ts`, exactly as the cue does,
 * because every surface already reads it from there and a second copy would be a
 * second truth for the Inspector to reconcile. What is kept here is provenance:
 * it is how a hand-made selection is told apart from this one (`useAgentView`
 * compares them), and it is the only place the entity's NAME is held — the
 * session store keeps a bare uuid, and the graph that could name it may not be
 * loaded.
 */
export interface AgentFocusRecord {
  id: string;
  name: string;
  seq: number;
}

/**
 * Why the view is where it is, in the model's own words.
 *
 * THE REASON IS THE PRODUCT, not decoration on it. "Opening Geo" tells a person
 * nothing they cannot see; "these 12 memories cluster on the Malabar coast" is
 * the finding, and the move is merely how it is shown. A view that rearranges
 * itself without one is the app twitching — the reader has to reverse-engineer
 * the intent from the result, which is precisely the failure the Follow
 * mechanism was built to prevent in the declined case and which was left
 * unaddressed in the applied one.
 *
 * `dimensions` is what makes this expire honestly. A reason that explains a cue
 * the person has since retyped is a lie about the picture in front of them, so
 * the record remembers which axes it accounts for and dies when one of them is
 * taken by hand.
 */
export interface AgentNoticeRecord {
  reason: string;
  /** The axes this reason accounts for, accumulated across one request's commands. */
  dimensions: ViewDimension[];
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
  focus: AgentFocusRecord | null;
  /** The model's account of the change it just made, or null. */
  notice: AgentNoticeRecord | null;
  /** Monotonic, shared by every record, so a consumer can key an effect on it. */
  seq: number;
  /**
   * Where verdicts go — the seat, via `app/useAgentView.ts`.
   *
   * A CALLBACK RATHER THAN A QUEUE THIS STORE OWNS, because a verdict is not
   * state: nothing renders it, nobody reads it twice, and a queue would need
   * draining, which means a subscriber writing back into the store it is
   * subscribed to. It is null until the shell wires one, and every call site
   * below tolerates that — an unwired bus still moves the view, it just cannot
   * say so, which is the same "not known" the seat already handles.
   */
  report: ((verdicts: readonly ViewVerdict[]) => void) | null;
  setReporter: (report: ((verdicts: readonly ViewVerdict[]) => void) | null) => void;

  /** A new turn hands the wheel back, less whatever the current surface holds. */
  beginTurn: (path: string) => void;
  dispatch: (command: ViewCommand, author: ViewAuthor) => Verdict;
  /** The human moved this dimension themselves. */
  touch: (dimension: ViewDimension) => void;
  /** Accept every waiting offer. */
  follow: () => void;
  /** Refuse them, visibly and once. */
  dismiss: () => void;
  /** Return to the stage the last agent-applied destination took you off. */
  back: () => void;
  /** Drop the model's cue and hand the whole corpus back. */
  release: () => void;
}

export const useView = create<ViewState>((set, get) => ({
  claimed: [],
  offers: {},
  cue: null,
  frame: null,
  destination: null,
  focus: null,
  notice: null,
  seq: 0,
  report: null,

  setReporter: (report) => set({ report }),

  beginTurn: (path) => {
    const state = get();
    // Reported BEFORE the offers are cleared, and reported as `expired` rather
    // than as a refusal: the person neither accepted nor declined, they simply
    // asked something else. Left unsaid, the seat would hold those asks open
    // forever and the trail would show a question that was never answered.
    state.report?.(verdictsForEndedOffers(state.offers, "expired"));
    set({
      claimed: holdsAt(path),
      // Offers do not outlive the turn that produced them. A Follow chip
      // offering to show what the model recalled two questions ago is an
      // invitation to a view of the wrong thing.
      offers: {},
    });
  },

  dispatch: (command, author) => {
    const state = get();
    const verdict = decide(author, command.dimension, state.claimed);
    const previous = state.offers[command.dimension];
    // Computed against the PRE-dispatch offer, because that offer is one of the
    // two things this dispatch resolves: it is either being followed or being
    // replaced, and after the `set` below there is nothing left to say which.
    const verdicts = verdictsForDispatch({ previous, command, author, verdict });

    if (verdict === "offer") {
      set({ offers: { ...state.offers, [command.dimension]: command } });
      state.report?.(verdicts);
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

    // ACCUMULATED ACROSS ONE REQUEST, KEYED ON THE REASON. A single
    // `direct_view` call arrives as up to three commands and dispatches three
    // times, all carrying the same words; the axes they cover have to end up on
    // one record or the notice would expire when the first of them is touched
    // and survive on the others. Reason equality is the join because it is what
    // the commands actually share — the store never sees the call that produced
    // them.
    //
    // Applied on a USER-authored dispatch too, and that is deliberate: Follow
    // re-dispatches the model's held command as the person, and the reason is
    // still the model's. A user's own hand never carries one.
    const reason = command.reason?.trim();
    const notice: AgentNoticeRecord | null = reason
      ? state.notice?.reason === reason
        ? {
            reason,
            dimensions: state.notice.dimensions.includes(command.dimension)
              ? state.notice.dimensions
              : [...state.notice.dimensions, command.dimension],
            seq,
          }
        : { reason, dimensions: [command.dimension], seq }
      : state.notice;

    switch (command.dimension) {
      case "cue":
        // ORDER IS LOAD-BEARING. The adapter distinguishes a typed cue from
        // this one by comparing the session's cue against this record, and
        // zustand notifies subscribers synchronously — so the record must
        // already be in place when `setCue` fires, or the adapter reads a stale
        // record, concludes a human typed it and claims the dimension.
        set({ cue: { text: command.text, entities: command.entities, seq }, seq, offers, claimed, notice });
        useSession.getState().setCue(command.text, command.entities);
        break;
      case "frame":
        set({ frame: { entities: command.entities, seq }, seq, offers, claimed, notice });
        break;
      case "destination":
        set({ destination: { path: command.path, from: command.from, seq }, seq, offers, claimed, notice });
        break;
      case "focus":
        // Same order rule, and the same reason, as the cue: the record must be
        // in place before `selectEntity` fires, or the watcher that tells a
        // hand-made selection from this one reads a stale record, concludes a
        // person clicked the node, and claims the dimension against the model
        // that just set it.
        set({ focus: { id: command.id, name: command.name, seq }, seq, offers, claimed, notice });
        useSession.getState().selectEntity(command.id);
        break;
    }

    state.report?.(verdicts);
    return verdict;
  },

  touch: (dimension) => {
    const held = get().offers[dimension];
    // Taking an axis by hand answers any offer standing on it, and the answer is
    // no. Reported as `declined` rather than `expired` for the same reason
    // Dismiss is: the person did something, and a trail that could not tell a
    // refusal from a lapse would let "they said no" and "they never looked" be
    // counted as the same event.
    if (held) get().report?.(verdictsForEndedOffers({ [dimension]: held }, "declined"));
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
        // And the account goes with the thing it accounted for. A reason
        // survives a touch of an axis it never claimed — panning the graph does
        // not falsify "I opened the map because these cluster on the coast" —
        // but the moment the person takes an axis the reason covers, the reason
        // is describing a view that is no longer the one it describes.
        //
        // NOT CLEARED AT `beginTurn`, deliberately. The next question does not
        // undo the last answer's narrowing: the view is still where the model
        // put it and still there for the stated reason, so dropping the words
        // would leave a moved view with no account of itself, which is the
        // state this record exists to abolish.
        notice: s.notice?.dimensions.includes(dimension) ? null : s.notice,
        // The model's selection goes the way its cue does when the person takes
        // that axis: a record naming the entity the model opened, sitting behind
        // one the person clicked, would attribute their choice to it.
        focus: dimension === "focus" ? null : s.focus,
      };
    });
  },

  follow: () => {
    const offers = get().offers;
    // Dispatched as the USER, because that is who is asking: the command was
    // the model's, the decision to apply it is not.
    for (const command of Object.values(offers)) get().dispatch(command, "user");
  },

  // Refusing is a decision, not a shrug: the dimensions it covered become the
  // person's for the rest of the turn, so a second recall in the same answer
  // cannot re-ask a question that has just been answered no.
  dismiss: () => {
    const state = get();
    // The only place a person says no in so many words. Reported before the
    // offers are dropped, because afterwards there is nothing left to name.
    state.report?.(verdictsForEndedOffers(state.offers, "declined"));
    set((s) => {
      const claimed = [...s.claimed];
      for (const dimension of Object.keys(s.offers) as ViewDimension[]) {
        if (!claimed.includes(dimension)) claimed.push(dimension);
      }
      return { claimed, offers: {} };
    });
  },

  /**
   * The way back from a stage the conversation opened.
   *
   * A TOUCH FOLLOWED BY A USER DISPATCH, not a bare navigation, and each half is
   * load-bearing. The touch is the truthful part: going back is the person
   * taking the destination by hand, so the model must not move it again
   * underneath them this turn, and the account of the move — a reason that
   * covered the destination — has to die with the move it was accounting for.
   * The dispatch is what makes it a MOVE rather than a state edit: nothing here
   * navigates, `useAgentView` watches the record, and only a record change gets
   * the person to the other stage.
   *
   * The return command carries `from: null`. One inverse, offered once — a
   * return trip that could itself be returned from is a toggle, and a control
   * that swaps between two stages on repeated presses is not the "one obvious
   * action" a moved reader is looking for.
   *
   * Silent when there is nowhere to go. The button that calls this is gated on
   * `returnTarget` (lib/view/presence.ts), which knows the current path and this
   * does not; the guard here is for the case where those two disagree, and it is
   * a no-op rather than a navigation to a stage nobody asked for.
   */
  back: () => {
    const record = get().destination;
    if (!record || record.from === null) return;
    get().touch("destination");
    get().dispatch({ dimension: "destination", path: record.from, from: null }, "user");
  },

  release: () => {
    // Releasing is the person taking the cue back by hand, so it claims the
    // dimension the same way typing into the field does.
    get().touch("cue");
    useSession.getState().setCueDraft("");
    // The notice goes UNCONDITIONALLY, not through `touch`'s dimension test.
    // Release is the whole corpus being handed back; a reason for a narrowing
    // that no longer exists would outlive it whenever the reason covered only
    // the destination.
    //
    // The FOCUS RECORD GOES TOO, and the selection with it. Release is used on
    // a profile change as well as by hand (`useAgentView`), and an entity uuid
    // names nothing in another corpus — the inspector would sit on an id from a
    // graph that is no longer loaded, under a record crediting a conversation
    // about somewhere else.
    get().touch("focus");
    useSession.getState().selectEntity(null);
    set({ cue: null, frame: null, focus: null, notice: null });
  },
}));
