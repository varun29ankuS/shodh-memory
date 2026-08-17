import type { ViewCommand } from "./commands";
import type { ViewAuthor, ViewDimension, Verdict } from "./authority";

/**
 * What the browser tells the seat it did — the return leg of the view loop.
 *
 * THE ASYMMETRY THIS FIXES. The seat could say "move the view" and never learn
 * whether the view moved. The authority ledger's answer lived and died in this
 * tab: History could only honestly label a `view_command` "Asked to move the
 * view", and the model wrote its sentence to the person without knowing whether
 * there was anything on screen to describe.
 *
 * NOTHING HERE INFERS AN OUTCOME BY WATCHING STATE. Every verdict below is
 * produced by the code that performs the transition, at the moment it performs
 * it, because the cause is knowable only there. A reporter that diffed the store
 * could see an offer disappear and could not tell whether the person accepted
 * it, refused it, or simply started a new turn — three facts that a person would
 * describe with three different sentences, collapsed into one guess.
 *
 * Pure, and separate from the store, so each rule can be made to fail a test.
 * `stores/view.ts` calls these; `app/useAgentView.ts` carries the result to the
 * seat.
 */

/**
 * The closed set, transcribed from seat/src/view-link.ts `VIEW_OUTCOME_STATES`.
 *
 * DUPLICATED ACROSS THE WIRE, and made safe by the seat REJECTING anything it
 * does not recognise with a 400 rather than coercing it. Drift shows up as a
 * loud failure on the first report instead of as a mislabelled audit row.
 *
 * There is no member meaning "unknown". A command whose fate is never reported
 * simply has no row on the other side, and absence is how the whole system says
 * "not known" — a value spelling it would let a row assert something this
 * browser never observed.
 */
export const VIEW_OUTCOME_STATES = [
  "applied",
  "already",
  "offered",
  "followed",
  "declined",
  "expired",
  "superseded",
] as const;

export type ViewOutcomeState = (typeof VIEW_OUTCOME_STATES)[number];

/** One dimension's fate, addressed to the `direct_view` call that asked for it. */
export interface ViewVerdict {
  /** The seat tool call id the command carries. */
  origin: string;
  dimension: ViewDimension;
  state: ViewOutcomeState;
}

/**
 * The verdicts one dispatch produces.
 *
 * TWO COMMANDS CAN BE INVOLVED, which is why this is not a single value. A
 * dispatch lands on a dimension that may already be holding an offer, and that
 * older offer is resolved by the same act: accepted when this dispatch IS it
 * being followed, superseded when it is a different request arriving on top.
 * Reporting only the new command would leave the old one pending forever on the
 * seat's side, and the trail would show an ask that was never answered when in
 * fact it was answered by being replaced.
 *
 * FOLLOWING IS NOT A USER'S OWN MOVE. `follow()` re-dispatches the model's held
 * command as the person, because the decision to apply it is theirs. The command
 * still carries the model's `origin`, and that is the discriminator: a
 * user-authored dispatch of a command with an origin is the person accepting an
 * offer, and it is reported as `followed` rather than as a fresh `applied`.
 *
 * A command with no `origin` produces nothing. Those are derived from a recall —
 * nobody asked for them, so there is nobody to answer.
 */
export function verdictsForDispatch(input: {
  /** The offer this dimension was holding, if any. */
  previous: ViewCommand | undefined;
  command: ViewCommand;
  author: ViewAuthor;
  verdict: Verdict;
}): ViewVerdict[] {
  const { previous, command, author, verdict } = input;
  const dimension = command.dimension;
  const out: ViewVerdict[] = [];

  const following = author === "user" && command.origin !== undefined;
  const replaces = previous?.origin !== undefined && previous.origin !== command.origin;

  if (replaces) {
    out.push({ origin: previous!.origin!, dimension, state: "superseded" });
  }

  if (command.origin === undefined) return out;

  if (verdict === "offer") {
    out.push({ origin: command.origin, dimension, state: "offered" });
    return out;
  }

  out.push({ origin: command.origin, dimension, state: following ? "followed" : "applied" });
  return out;
}

/**
 * The verdicts for offers that end without being applied.
 *
 * `state` is the caller's, not this function's, because only the caller knows
 * WHY: `touch` and `dismiss` are the person declining (with a hand and with a
 * button — the same statement), and `beginTurn` is a turn ending over an offer
 * nobody answered. Folding those into one label would erase the difference
 * between "they said no" and "they never saw it", which is the distinction the
 * whole return path exists to preserve.
 */
export function verdictsForEndedOffers(
  offers: Partial<Record<ViewDimension, ViewCommand>>,
  state: ViewOutcomeState,
): ViewVerdict[] {
  const out: ViewVerdict[] = [];
  for (const [dimension, command] of Object.entries(offers) as [ViewDimension, ViewCommand][]) {
    if (command.origin === undefined) continue;
    out.push({ origin: command.origin, dimension, state });
  }
  return out;
}
