import type { SeatEvent } from "@/lib/seat/types";

/**
 * Where the adapter has read up to in the live turn.
 *
 * Split out of the adapter because it holds two rules that must be able to fail
 * a test, and both of them are invisible in a component:
 *
 *  1. AN OP IS TRANSLATED ONCE. The chat store's reducer replaces the turn
 *     object on every flush, so "what changed" cannot be read from identity —
 *     it is a position in a growing list.
 *  2. ONLY A LIVE TURN MOVES THE VIEW. `adoptDetail` rebuilds every past turn's
 *     ops from the transcript with `pending: false`; a click in the session list
 *     would otherwise replay a conversation's recalls and lurch the graph
 *     through everything the model ever looked at.
 */

export interface TurnCursor {
  /** Conversation id and the turn's POSITION — never the seat's turn number,
   *  which `turn_start` overwrites mid-answer (stores/chat.ts). */
  key: string;
  consumed: number;
}

export interface LiveTurn {
  key: string;
  pending: boolean;
  ops: readonly SeatEvent[];
}

export interface CursorStep {
  cursor: TurnCursor;
  /** A new question just started: the authority window reopens. */
  beginTurn: boolean;
  /** Ops seen for the first time, and only from a live turn. */
  fresh: readonly SeatEvent[];
}

export const EMPTY_CURSOR: TurnCursor = { key: "", consumed: 0 };

export function advance(cursor: TurnCursor, turn: LiveTurn): CursorStep {
  if (cursor.key !== turn.key) {
    if (!turn.pending) {
      // A turn that is already finished when first seen was replayed, not
      // lived. Everything it holds is marked read without being translated.
      return { cursor: { key: turn.key, consumed: turn.ops.length }, beginTurn: false, fresh: [] };
    }
    return {
      cursor: { key: turn.key, consumed: turn.ops.length },
      beginTurn: true,
      fresh: turn.ops,
    };
  }

  if (!turn.pending) {
    return { cursor: { key: turn.key, consumed: turn.ops.length }, beginTurn: false, fresh: [] };
  }
  if (turn.ops.length <= cursor.consumed) return { cursor, beginTurn: false, fresh: [] };

  return {
    cursor: { key: turn.key, consumed: turn.ops.length },
    beginTurn: false,
    fresh: turn.ops.slice(cursor.consumed),
  };
}
