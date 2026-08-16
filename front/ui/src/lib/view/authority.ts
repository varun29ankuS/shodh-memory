/**
 * Who gets to move the view.
 *
 * The workbench has two producers — the person's hands, and one adapter
 * translating the conversation's `SeatEvent`s (spec §5). They are not peers.
 *
 * THE HUMAN ALWAYS HAS THE WHEEL. An agent-authored change applies only while
 * the human has not touched that dimension since the current turn began. The
 * turn is the window because it is the unit of intent: you ask a question, the
 * model answers it, and anything you did to the view while it was answering is
 * a statement that you want to look at something else. When the next turn
 * starts the wheel is free again — otherwise one pan of the graph, an hour ago,
 * would mute the model for the rest of the session.
 *
 * DIMENSIONS ARE TRACKED SEPARATELY, and that is the whole reason this is a
 * set rather than a boolean. Framing the graph by hand says nothing about where
 * you want to be; it must not stop the model opening a different destination,
 * and a single "the user is driving" flag cannot express that.
 *
 * A DECLINED COMMAND IS NOT DISCARDED — `offer` is a verdict, not a bin. The
 * failure this exists to prevent is a model that says "I've pulled that up on
 * the map" over a view that never moved, which is worse than either applying
 * the change or refusing it visibly: the person is left believing something
 * happened somewhere they are not looking.
 *
 * Pure, and separate from the store, because this is the rule that must be able
 * to fail a test. `stores/view.ts` carries the verdict out.
 */

/**
 * The independently-owned axes of the view.
 *
 * - `cue`         — what is lit: which entities the canvas rings and which recede.
 * - `frame`       — the camera: what the canvas is framed on.
 * - `destination` — which surface is on the stage.
 *
 * `focus` (the selected object) and `lens` are named in the spec and are not
 * here: nothing in a `memory_recall` says which single object to open, and the
 * lens control does not exist yet. A dimension nothing can author is a dimension
 * whose authority rule cannot be checked, so it is added when its producer is.
 */
export const VIEW_DIMENSIONS = ["cue", "frame", "destination"] as const;
export type ViewDimension = (typeof VIEW_DIMENSIONS)[number];

/** The two producers. Nothing else may author a command. */
export type ViewAuthor = "user" | "agent";

/** `apply` — move the view now. `offer` — hold it as a Follow the user can accept. */
export type Verdict = "apply" | "offer";

/**
 * The rule, entire.
 *
 * A user command always applies: this function exists to protect the person
 * from the model, not the other way round.
 */
export function decide(
  author: ViewAuthor,
  dimension: ViewDimension,
  claimed: readonly ViewDimension[],
): Verdict {
  if (author === "user") return "apply";
  return claimed.includes(dimension) ? "offer" : "apply";
}

/**
 * Dimensions the human owns from the moment a turn begins, by where they are.
 *
 * ONE ENTRY, AND IT IS EARNED. On `/chat` the conversation is the primary and
 * it is what the person is reading; navigating away from it demotes that to the
 * dock, which mounts MINIMIZED (`ConversationOverlay`'s `mode` starts at
 * `"minimized"`, and only a `dismissed` panel is un-hidden by a new stream).
 * So an agent destination change from `/chat` would take the answer off screen
 * at the exact moment it is being produced — the reader would watch the text
 * they were reading collapse into a 40px bar. It becomes a Follow instead,
 * which is the same information without the theft.
 *
 * Seeded at the turn boundary rather than consulted at dispatch time, so that
 * navigating away from `/chat` mid-turn does not hand the wheel back: leaving
 * is itself a user touch of the same dimension.
 */
export function holdsAt(path: string): ViewDimension[] {
  return path === "/chat" ? ["destination"] : [];
}
