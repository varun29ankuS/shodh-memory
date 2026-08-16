import type { SeatEvent } from "@/lib/seat/types";
import type { ViewDimension } from "./authority";

/**
 * The conversation, translated into view state.
 *
 * THIS IS THE CONTRACT, so it is a pure function over a wire event and it is
 * the thing the tests hold. Today the model can read and write memory and none
 * of it moves the interface: `memory_recall` events already carry the exact
 * memories it retrieved, and the browser only prints them. Every command below
 * is derived from one of those events — nothing here needs a seat change, a new
 * endpoint or a tool the model does not already have.
 *
 * Commands are NAMED AND SERIALIZABLE (spec §5): a declined one is held as a
 * Follow offer and applied later, so it cannot close over a canvas, a store or
 * a moment in time.
 */

export type ViewCommand =
  | { dimension: "cue"; text: string; entities: string[] }
  | { dimension: "frame"; entities: string[] }
  | { dimension: "destination"; path: string };

/**
 * How many entity terms a cue carries.
 *
 * A recall returns up to a few dozen facts and each carries its own keyword
 * list, so the union is unbounded in principle. Past a couple of dozen terms
 * the "matched" set stops being a narrowing — it lights most of the graph and
 * says nothing — so the cue keeps the first terms in retrieval order, which is
 * relevance order.
 */
export const ENTITY_LIMIT = 24;

/**
 * Where an answer is looked at: the graph, and only the graph.
 *
 * THE MAP IS NOT A DESTINATION THIS MAY CHOOSE, and the reason is the same rule
 * the whole Follow mechanism exists to enforce. §3.3's lens rule says geotagged
 * results open on the map — but `GeoView` raises points by matching them
 * against `activeQuery`, the COMMITTED search (features/geo/GeoView.tsx), and a
 * cue is deliberately not that: committing a query would fire a full retrieval
 * nobody asked for. `GeoMap` reads the session only for the selected memory; it
 * has no cue channel at all.
 *
 * So sending someone to `/geo` would land them on the entire corpus, unraised
 * and unmoved, under a chip saying the view is following the conversation —
 * a claim the map cannot keep. That is worse than not offering the map, and it
 * would fire on exactly the corpora that carry coordinates.
 *
 * The graph responds to the cue on every answer, so it is the honest
 * destination for all of them. The map rejoins this function the moment it
 * consumes the cue, and not before.
 */
const STAGE = "/graph";

/**
 * Words that name nothing.
 *
 * Function words and question scaffolding only — deliberately NOT domain nouns.
 * A stop list that starts removing "aircraft" or "programme" because they match
 * a lot is a list that decides what the corpus is about, and it will be wrong
 * on the next corpus. Terms under three characters are dropped by the matcher
 * itself, so nothing here needs to repeat "of", "to", "in" or "is".
 */
const STOP_WORDS = new Set([
  "the", "and", "for", "any", "all", "are", "was", "were", "been", "being",
  "what", "when", "where", "which", "who", "whom", "whose", "how", "why",
  "did", "does", "done", "has", "have", "had", "can", "could", "should",
  "would", "will", "shall", "may", "might", "must", "about", "with", "from",
  "into", "over", "under", "between", "this", "that", "these", "those",
  "their", "them", "they", "there", "here", "its", "our", "your", "more",
  "most", "than", "then", "some", "such", "know", "tell", "show", "find",
]);

/**
 * The terms a cue matches on.
 *
 * TWO SOURCES, AND THE SECOND IS NOT A FALLBACK — it is the one that works.
 * `RecallFact.related_entities` is the better signal when it exists, but facts
 * are minted by consolidation and a corpus can answer a dozen recalls without
 * producing one; measured on `defence-live`, a two-recall turn returned eight
 * and nine memories and zero facts.
 *
 * So the model's own query is tokenised. It has to be: a query is a PHRASE the
 * model composed — "Hindustan Aeronautics Limited HAL" — and testing it whole
 * against entity names matches nothing, because no entity is named that. The
 * typed cue keeps its whole-string test (a person watching themselves type
 * expects a prefix to behave like a prefix); a phrase nobody typed does not get
 * that courtesy, and splitting it is what makes the narrowing real rather than
 * a claim in a chip over an unchanged picture.
 *
 * Case-folded, de-duplicated, order-preserving: retrieval order is relevance
 * order, so the cap keeps the front of the list rather than a sorted slice.
 */
function cueTerms(query: string, facts: readonly { related_entities: string[] }[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];

  const take = (raw: string): boolean => {
    const term = raw.trim();
    if (term.length === 0) return true;
    const key = term.toLowerCase();
    if (seen.has(key)) return true;
    seen.add(key);
    out.push(term);
    return out.length < ENTITY_LIMIT;
  };

  for (const fact of facts) {
    for (const entity of fact.related_entities) if (!take(entity)) return out;
  }
  // Split on everything that is not a letter, a digit or an internal hyphen:
  // entity names in this corpus carry hyphens ("MiG-21", "Su-30MKI") and
  // splitting them produces two terms that match far more than the one did.
  for (const word of query.split(/[^\p{L}\p{N}-]+/u)) {
    const token = word.replace(/^-+|-+$/g, "");
    if (STOP_WORDS.has(token.toLowerCase())) continue;
    if (!take(token)) return out;
  }
  return out;
}

/**
 * One memory operation, as view commands. Empty when the op says nothing about
 * what to look at.
 *
 * HARNESS-SCOPE RECALLS ARE IGNORED. `scope: "harness"` is the seat asking its
 * own questions — retrieving its behavioural learnings before a turn, not
 * answering the person. Moving the view for a question the user never asked
 * reads as the app twitching, and it would fire on turns where the model said
 * nothing about memory at all.
 *
 * A RECALL THAT RETURNED NOTHING STILL SETS THE CUE, AND ONLY THE CUE. The
 * corpus stays framed and the destination stays put: an empty answer is a fact
 * about the query, not an instruction to blank the screen. (The canvas already
 * refuses to recede anything when a cue matches no node — `searching` requires
 * a non-empty match set — so an unmatched cue is visible in the chip and inert
 * on the picture.)
 */
export function commandsFromOp(op: SeatEvent, path: string): ViewCommand[] {
  if (op.type !== "memory_recall") return [];
  if (op.scope !== "user") return [];

  const text = op.query.trim();
  if (text.length === 0) return [];

  const entities = cueTerms(text, op.facts);
  const commands: ViewCommand[] = [{ dimension: "cue", text, entities }];

  const found = op.memories.length + op.facts.length;
  if (found === 0) return commands;

  // Framing needs something to frame ON. A frame command with no subject would
  // either do nothing or, worse, reset the camera to the whole corpus while
  // claiming to have narrowed it.
  if (entities.length > 0) commands.push({ dimension: "frame", entities });

  if (STAGE !== path) commands.push({ dimension: "destination", path: STAGE });

  return commands;
}

/** Where a command would take you, for the Follow offer's own label. One entry,
 *  because `STAGE` produces one destination; the path itself is the fallback so
 *  a new stage reads as a path rather than as nothing. */
const DESTINATION_NOUN: Record<string, string> = { "/graph": "the graph" };

/**
 * The offer, in the words a person would use.
 *
 * Written from the commands rather than stored on them so a command stays pure
 * data: the same declined command is described identically whether it was
 * declined a second ago or is being re-offered after a reload.
 */
export function describeCommands(commands: readonly ViewCommand[]): string {
  const parts: string[] = [];
  for (const command of commands) {
    if (command.dimension === "cue") parts.push(`follow its cue “${command.text}”`);
    else if (command.dimension === "frame") parts.push("frame those entities");
    else parts.push(`open ${DESTINATION_NOUN[command.path] ?? command.path}`);
  }
  if (parts.length === 0) return "";
  if (parts.length === 1) return parts[0];
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

/** The dimensions a command list touches, in order, without duplicates. */
export function dimensionsOf(commands: readonly ViewCommand[]): ViewDimension[] {
  const out: ViewDimension[] = [];
  for (const command of commands) if (!out.includes(command.dimension)) out.push(command.dimension);
  return out;
}
