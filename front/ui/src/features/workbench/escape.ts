/**
 * Who owns Escape.
 *
 * The rule is "Escape always goes back one level, from everywhere", and the
 * word that does the work is LEVEL. Several surfaces in this product already
 * treat Escape as their own — a text field abandons what you typed, a hint
 * panel closes, a model list closes, a canvas clears the selected node. Every
 * one of those IS a level, so Escape belongs to the innermost one that has a
 * level to give up, and only reaches the trail when nothing else does.
 *
 * WITHOUT THIS LADDER ONE KEYPRESS COSTS TWO LEVELS. React attaches its
 * handlers at the root container, so a window listener in the bubble phase
 * runs AFTER them: press Escape with a node selected on the graph canvas and
 * the canvas clears the selection, then the trail handler reads a store that
 * no longer has one and pops a pane. The listener is registered in the CAPTURE
 * phase for that reason — it reads the world before anything has mutated it —
 * and this function decides what happens.
 *
 * Kept as a pure function of five booleans so the precedence can be tested as
 * a table rather than as a browser interaction. The probes that produce those
 * booleans are below, and they are the only part that touches the DOM.
 */
export type EscapeVerdict =
  /** A field owns it: Escape abandons the edit. Nothing else may act. */
  | "editable"
  /** A transient surface is open — a hint, a listbox, the expanded
   *  conversation. It closes, and that is the level being given up. */
  | "transient"
  /** Something is selected. The detail pane is a level; close it first. */
  | "clear-selection"
  /** Nothing more local has a level. Go back one pane. */
  | "back"
  /** The briefing, with nothing selected and nothing open. Escape does
   *  nothing, and doing nothing is the correct answer — re-navigating to the
   *  screen already on display would push a history entry for no movement. */
  | "none";

export interface EscapeContext {
  /** The keystroke landed in a text field, a textarea or a contenteditable. */
  editable: boolean;
  /** A dialog, listbox or note is in the document. */
  transientOpen: boolean;
  /** The conversation dock is expanded, and minimises on Escape. */
  conversationExpanded: boolean;
  /** A memory or an entity is open in the Inspector. */
  hasSelection: boolean;
  /** The trail has somewhere to go back to. */
  canGoBack: boolean;
}

export function escapeVerdict(ctx: EscapeContext): EscapeVerdict {
  if (ctx.editable) return "editable";
  if (ctx.transientOpen || ctx.conversationExpanded) return "transient";
  if (ctx.hasSelection) return "clear-selection";
  return ctx.canGoBack ? "back" : "none";
}

/** Whether the verdict is one the trail acts on. The other three are the
 *  trail standing aside for something more local. */
export function trailActs(verdict: EscapeVerdict): boolean {
  return verdict === "clear-selection" || verdict === "back";
}

// ---------------------------------------------------------------------------
// The probes
// ---------------------------------------------------------------------------

/** A field that owns its own Escape. `SearchField` blurs, `SessionList`
 *  abandons a rename; both would be overridden by a trail pop. */
export function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return (
    tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable
  );
}

/**
 * A transient surface that is open right now.
 *
 * Each of these roles is rendered only while its surface is open —
 * `role="note"` by the info hint, `role="listbox"` by the model picker — so
 * their presence in the document is the same fact as "something is open", with
 * no state to keep in sync. `dialog` is included for surfaces that do not
 * exist yet, so a future one does not have to remember to register here.
 */
export function hasTransientSurface(doc: Document): boolean {
  return doc.querySelector('[role="dialog"], [role="listbox"], [role="note"]') !== null;
}

/**
 * The conversation dock, expanded.
 *
 * DETECTED FROM THE DOM, and the coupling is deliberate rather than lazy: the
 * dock's mode is component-local state in `ConversationOverlay`, it listens on
 * `window` regardless of where focus is, and it is outside this pass's scope
 * to change. Its "Minimize conversation" button is rendered only in the
 * expanded state, which makes it an exact witness for the state that owns
 * Escape. If that label changes, this returns false and Escape costs two
 * levels while the dock is open — which is why the label is named here rather
 * than matched loosely.
 */
export function isConversationExpanded(doc: Document): boolean {
  return doc.querySelector('[aria-label="Minimize conversation"]') !== null;
}
