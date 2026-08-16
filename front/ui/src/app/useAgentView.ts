import { useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { commandsFromOp } from "@/lib/view/commands";
import { useChat } from "@/stores/chat";
import { useSession } from "@/stores/session";
import { useView } from "@/stores/view";

/**
 * The conversation's producer on the view bus — mounted ONCE, in the shell.
 *
 * `EvidencePanel` and `MessageList` already read the same `memory_recall` ops
 * and they stay exactly as they are: they render what happened. This is the one
 * thing that turns what happened into what is on screen. Keeping it single is
 * not tidiness — two translators would double-dispatch every op, and each would
 * mount and unmount with its own surface, so which of them was listening would
 * depend on where you happened to be standing.
 *
 * It also carries the other direction: the human's hands do not know the bus
 * exists, so their touches are DETECTED rather than reported. Two of the three
 * dimensions are observable as state — the cue is a field in the session store,
 * the destination is the URL — so they are watched here instead of being
 * instrumented at a dozen call sites that would each have to remember. The
 * third, the camera, is a gesture with no state to watch, so `EntityCanvas`
 * reports it directly at the one place it already detects a real wheel or drag.
 */

/** Where the cursor is in the live turn's op list, so an op is translated once. */
interface Cursor {
  key: string;
  consumed: number;
}

export function useAgentView(): void {
  const { pathname } = useLocation();
  const navigate = useNavigate();

  /** The path as of this render, for effects that must not re-run on a move. */
  const pathRef = useRef(pathname);
  pathRef.current = pathname;

  const convo = useChat((s) => (s.activeId ? (s.conversations[s.activeId] ?? null) : null));
  const activeId = useChat((s) => s.activeId);
  const cursor = useRef<Cursor>({ key: "", consumed: 0 });

  useEffect(() => {
    if (!activeId || !convo) return;
    // KEYED ON POSITION, NOT ON THE TURN NUMBER. `send` appends a turn numbered
    // by array length and the seat's `turn_start` then overwrites it with its
    // own count (stores/chat.ts). Keying on the number makes those two values
    // look like two different turns, which would re-open the authority window
    // mid-answer and throw away a touch the user made in between.
    const index = convo.turns.length - 1;
    const live = convo.turns[index];
    if (!live) return;
    const key = `${activeId}#${index}`;

    if (cursor.current.key !== key) {
      cursor.current = { key, consumed: live.pending ? 0 : live.ops.length };
      // A NEW QUESTION HANDS THE WHEEL BACK. Only a live turn does: `pending`
      // is false for everything `adoptDetail` rebuilt from the transcript, so
      // reopening a conversation from the session list replays its recalls
      // without moving the view. A history click that lurched the graph would
      // make the interface untrustworthy in exactly the place it is meant to
      // become trustworthy.
      if (live.pending) useView.getState().beginTurn(pathRef.current);
    }

    if (!live.pending) {
      cursor.current.consumed = live.ops.length;
      return;
    }
    if (live.ops.length <= cursor.current.consumed) return;

    const fresh = live.ops.slice(cursor.current.consumed);
    cursor.current.consumed = live.ops.length;
    for (const op of fresh) {
      for (const command of commandsFromOp(op, pathRef.current)) {
        useView.getState().dispatch(command, "agent");
      }
    }
  }, [activeId, convo]);

  /**
   * The cue, and who wrote it.
   *
   * Watched rather than instrumented because there are several ways to set one
   * — the header field per keystroke, a cue chip on the empty recall state, a
   * committed search — and a producer that has to be remembered at each of them
   * is a producer that will be forgotten at the next one. A change that does
   * not match what the bus just applied came from a person.
   */
  useEffect(
    () =>
      useSession.subscribe((state, previous) => {
        if (state.profile !== previous.profile) {
          // The model's terms name entities in the corpus it was reading. In
          // another profile they match nothing, so the graph would sit dimmed
          // under a chip crediting a conversation that was about somewhere else.
          useView.getState().release();
          return;
        }
        if (state.cueDraft === previous.cueDraft) return;
        if (state.cueDraft === useView.getState().cue?.text) return;
        useView.getState().touch("cue");
      }),
    [],
  );

  /**
   * The destination, and who moved it.
   *
   * Every navigation in the product is a bare `navigate(path)` from a door, a
   * rail row, a spine or the evidence panel — none of which has heard of the
   * bus. So the URL is the signal: any change that is not the one the bus just
   * applied was a person's.
   */
  const firstPath = useRef(true);
  const applied = useRef<string | null>(null);
  useEffect(() => {
    if (firstPath.current) {
      firstPath.current = false;
      return;
    }
    if (applied.current === pathname) {
      applied.current = null;
      return;
    }
    useView.getState().touch("destination");
  }, [pathname]);

  const destination = useView((s) => s.destination);
  useEffect(() => {
    if (!destination) return;
    if (destination.path === pathRef.current) return;
    applied.current = destination.path;
    navigate(destination.path);
  }, [destination, navigate]);
}
