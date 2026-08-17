import { VIEW_DIMENSIONS, type ViewDimension } from "./authority";
import type { ViewCommand } from "./commands";

/**
 * What the conversation is doing to this view, WHILE it does it.
 *
 * THE GAP THIS CLOSES. Everything the bus decides was already recorded — the
 * authority ledger picks a verdict per axis, the seat is told, `/history` keeps
 * the row. What none of it did was show the person the decision at the moment it
 * was taken. An applied change was a picture that rearranged itself under a
 * one-line statement in the header; a declined one was a Follow chip; a request
 * that landed on four axes with different fates showed only the axis that was
 * refused, because `FollowOffer` returns its offer branch whenever anything is
 * pending. The most informative event this system can produce — the model asked
 * for four things, three happened and one is waiting on you — was the one it
 * rendered least.
 *
 * TWO CLAIMS, KEPT APART, AND THAT IS THE WHOLE SHAPE. An axis the bus APPLIED
 * and an axis the bus is HOLDING are different facts about the same request, and
 * a presence surface that cannot show them side by side will always be lying by
 * omission about one of them. So a trace is a reason plus a per-axis verdict,
 * never a sentence about the request as a whole.
 *
 * WHY THERE IS NO "UNKNOWN" AXIS. `stores/view.ts` decides synchronously, in the
 * same call that performs the transition — by the time anything can render,
 * every axis this browser was asked about has a verdict, and an axis it was
 * never asked about has no row. Absence is how this file says "not known",
 * exactly as `lib/view/outcome.ts` refuses a wire state spelling it. The unknown
 * that DOES exist in this loop is the seat's, not the browser's: the seat waits
 * a bounded moment for a report it may never get. That is unobservable from
 * here — `lib/seat/client.ts` `reportView` is fire-and-forget and does not check
 * `res.ok` — so nothing on this side may claim the conversation was told.
 *
 * Pure, and separate from the component, because these are the rules that must
 * be able to fail a test.
 */

/**
 * How long an arrival stays on screen before it collapses back to the header
 * line.
 *
 * IT COLLAPSES AT ALL because the brief for this product was that the screen
 * "takes time to understand what it is saying", and a pass was already spent
 * cutting the prose above the fold from ~175 words to ~47. A block that stays is
 * a permanent tax on every screen for an event that is over. Nothing is lost by
 * going: an offer that is still waiting is carried by the header exactly as it
 * was before this existed, and the trace itself is one click away for as long as
 * the state it describes survives.
 *
 * NINE SECONDS is read-a-sentence-and-decide, not glance-and-guess: the payload
 * is a clause of the model's own prose plus up to four verdicts, and the two
 * controls beside them are a choice. Hovering or focusing the block stops the
 * clock, so it cannot vanish from under a hand on its way to `Follow`.
 */
export const TRACE_DWELL_MS = 9_000;

/**
 * What the bus did with one axis.
 *
 * `waiting` and not `declined`: nobody has refused anything. The person holds
 * this axis, the model asked for it anyway, and the ask is standing there until
 * they answer or the turn ends. `--warn` is the token for waiting-on-someone and
 * this is what it is for; `--destructive` would say the request was wrong, which
 * is a different and false claim.
 */
export type AxisState = "applied" | "waiting";

export interface TraceAxis {
  dimension: ViewDimension;
  state: AxisState;
}

/** One request's account of itself, as the person should read it. */
export interface Trace {
  /** The model's own words. Never generated, never edited, never summarised. */
  reason: string;
  /** In `VIEW_DIMENSIONS` order, so the list cannot reshuffle between renders. */
  axes: TraceAxis[];
}

/**
 * The trace the view is currently able to account for, or null.
 *
 * KEYED ON ONE REASON, and that is a correctness constraint rather than tidiness.
 * The store holds at most one offer per dimension and one notice, and those can
 * outlive each other: an offer left standing from an earlier request in the same
 * turn sits beside a notice from a later one. Listing both sets of axes under a
 * single quotation would attribute one request's ask to another request's words,
 * which is the exact misattribution the notice record's `dimensions` field was
 * added to prevent.
 *
 * THE WAITING REASON OUTRANKS THE APPLIED ONE when they differ, on the rule
 * `FollowOffer` already states in as many words: an offer is an action waiting on
 * someone and an applied line is a statement of fact. In the ordinary case — one
 * `direct_view`, up to four commands, all carrying the same sentence — they are
 * the same string and the precedence never fires.
 *
 * NULL WHEN NOTHING CARRIES A REASON, which is how the recall-derived path stays
 * exactly as it was. A cue inferred from a search has no author's account of
 * itself; this surface exists to deliver that account, so with none there is
 * nothing here to show and the header's own line remains the whole of it.
 */
export function traceOf(
  notice: { reason: string; dimensions: readonly ViewDimension[] } | null,
  offers: Partial<Record<ViewDimension, ViewCommand>>,
): Trace | null {
  let reason: string | null = null;
  for (const dimension of VIEW_DIMENSIONS) {
    const waiting = offers[dimension]?.reason?.trim();
    if (waiting) {
      reason = waiting;
      break;
    }
  }
  if (reason === null) reason = notice?.reason.trim() ?? null;
  if (reason === null || reason.length === 0) return null;

  const axes: TraceAxis[] = [];
  for (const dimension of VIEW_DIMENSIONS) {
    if (offers[dimension]?.reason?.trim() === reason) {
      axes.push({ dimension, state: "waiting" });
    } else if (notice?.reason.trim() === reason && notice.dimensions.includes(dimension)) {
      axes.push({ dimension, state: "applied" });
    }
  }
  if (axes.length === 0) return null;

  return { reason, axes };
}

/**
 * The identity of a trace — what makes this one a different thing to look at
 * from the last one.
 *
 * WHY IDENTITY RATHER THAN A COUNTER. The block is shown while the key it was
 * opened for is still the key the state produces, so every way a trace can stop
 * being true closes it without anything having to notice: accepting an offer
 * moves an axis from waiting to applied, refusing one drops the axis, a new turn
 * expires it, taking an axis by hand kills the notice that covered it. A counter
 * would need each of those sites to remember to decrement it, and the one that
 * forgot would leave a block on screen describing a view that had moved on.
 *
 * The reason is part of the key, so two requests whose axes coincide are still
 * two traces. Serialised rather than concatenated with a separator character:
 * the reason is arbitrary model prose and any delimiter chosen for it is a
 * delimiter the model can one day emit, which would collide two distinct traces
 * into one and leave a block on screen that never re-read.
 */
export function traceKey(trace: Trace): string {
  const axes = trace.axes.map((axis) => `${axis.dimension}:${axis.state}`);
  return JSON.stringify([trace.reason, axes]);
}

/** The parts of the bus that say a command has just landed. */
export interface ViewArrival {
  seq: number;
  offers: Partial<Record<ViewDimension, ViewCommand>>;
}

/**
 * Whether an agent command just landed — the moment the block opens on.
 *
 * NOT "THE TRACE CHANGED", and the difference is the whole reason this function
 * exists. A trace SHRINKS for reasons that are not arrivals: `beginTurn` expires
 * a standing offer, `dismiss` drops one, a hand on an axis kills the notice that
 * covered it. Opening the block on any change would make it flash an
 * applied-only summary at the start of every turn the person typed into, which is
 * the app twitching — the failure this feature exists to replace, reproduced by
 * the feature itself.
 *
 * TWO SIGNALS, BECAUSE THE STORE HAS TWO OUTCOMES. An applied command bumps
 * `seq`; a held one does not touch it and appears as an offer instead. Watching
 * only the sequence would miss every declined request, which is the half of this
 * that the person most needs to see.
 *
 * Offers are compared BY IDENTITY. `dispatch` builds a fresh offers object and
 * carries unchanged commands across by reference, so a differing reference is
 * exactly a command that arrived or was replaced — no field-by-field comparison
 * can be more accurate than that, and any that tried would have to guess which
 * fields make two asks the same ask.
 */
export function arrived(previous: ViewArrival, next: ViewArrival): boolean {
  if (next.seq > previous.seq) return true;
  for (const dimension of VIEW_DIMENSIONS) {
    const command = next.offers[dimension];
    if (command !== undefined && command !== previous.offers[dimension]) return true;
  }
  return false;
}

/**
 * The stage to offer as the way back, or null when there is none to offer.
 *
 * TWO REFUSALS, AND EACH IS A DIFFERENT WAY THE BUTTON WOULD LIE:
 *
 *  - The record does not describe where they are standing. They navigated since,
 *    by hand or by a later command; a button labelled "back" that took them off
 *    a stage they chose themselves would be a move, not an undo.
 *  - The way back is where they already are. That can only arise from a record
 *    whose `from` equals its `path`, and a control that does nothing is worse
 *    than an absent one — the person presses it and learns the app is broken.
 *
 * A `from` of null needs NO clause of its own and deliberately does not have
 * one: it falls out of the final line, which returns exactly what the record
 * holds. An explicit early return for it would read as a third rule and would be
 * dead — no input can reach it and change the answer — and a line no test can
 * kill is a line that will be wrong one day without anything noticing.
 *
 * The path is passed in rather than read, because this is the one piece of the
 * bus that depends on where the person actually is, and the store deliberately
 * knows nothing about the router.
 */
export function returnTarget(
  destination: { path: string; from: string | null } | null,
  current: string,
): string | null {
  if (destination === null) return null;
  if (destination.path !== current) return null;
  if (destination.from === current) return null;
  return destination.from;
}

/**
 * Which axis a row is about, in a person's words.
 *
 * ONE DEFINITION, TWO SURFACES. `/history` labels the same four axes on its
 * outcome rows and used to own this list; a second copy would let the live
 * account of a move and the permanent record of it call the same axis two
 * different things, which is the drift a reader comparing them would read as two
 * different events. `features/history/derive.ts` re-exports this.
 *
 * An unrecognised value survives verbatim, as everywhere else in this codebase:
 * a surface that reports what the system did must not hide an axis it does not
 * recognise behind a friendly word.
 */
export function viewDimensionLabel(dimension: string): string {
  switch (dimension) {
    case "cue":
      return "the narrowing";
    case "frame":
      return "the camera";
    case "destination":
      return "the destination";
    case "focus":
      return "the opened entity";
    default:
      return dimension;
  }
}

/**
 * What became of an axis, in a person's words.
 *
 * "waiting on you" NAMES THE PERSON, deliberately. The bus's own vocabulary for
 * this is `offered`, which is the model's side of it and reads on screen as
 * something the app is doing. What the reader needs to know is that nothing more
 * will happen until they act — that they are the one holding this, which is the
 * fact that makes the whole authority ledger visible rather than theoretical.
 */
export function axisStateLabel(state: AxisState): string {
  return state === "applied" ? "applied" : "waiting on you";
}

/** A list, as a person would say it. */
function list(items: readonly string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/**
 * The whole trace as one spoken sentence, for the live region.
 *
 * WRITTEN AS PROSE RATHER THAN LET THE ROWS BE READ OUT. The visual block is a
 * label column beside a state column, which a screen reader renders as "the
 * narrowing applied the camera applied the destination waiting on you" — a run
 * of eight words with no grammar, from which the listener has to reconstruct
 * that these are pairs. The eye gets the pairing from the alignment for free;
 * the ear has to be given it.
 *
 * "MOVED" IS SAID ONLY OF AXES THAT MOVED, which is the same discipline
 * `features/history/derive.ts` keeps when it refuses to relabel an ask as a
 * move. A wholly declined request announces the waiting clause and no other, so
 * the listener is never told the view moved when it did not.
 */
export function traceAnnouncement(trace: Trace): string {
  const applied = trace.axes
    .filter((axis) => axis.state === "applied")
    .map((axis) => viewDimensionLabel(axis.dimension));
  const waiting = trace.axes
    .filter((axis) => axis.state === "waiting")
    .map((axis) => viewDimensionLabel(axis.dimension));

  const parts = [`The conversation: “${trace.reason}”.`];
  if (applied.length > 0) parts.push(`It moved ${list(applied)}.`);
  if (waiting.length > 0) parts.push(`It is waiting on you for ${list(waiting)}.`);
  return parts.join(" ");
}
