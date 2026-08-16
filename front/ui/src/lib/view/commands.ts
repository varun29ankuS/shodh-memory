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
 * Where an answer of this shape is best looked at.
 *
 * The same rule the spec states for lenses (§3.3): geotagged results open on
 * the map, everything else on the graph, because an empty map and a broken map
 * look identical. There is no lens control yet, so the destination IS the lens.
 */
function stageFor(memories: readonly { experience: { geo_location?: unknown } }[]): string {
  return memories.some((m) => m.experience.geo_location) ? "/geo" : "/graph";
}

/** Case-folded, de-duplicated, order-preserving. Retrieval order is relevance
 *  order, so the cap keeps the front of the list rather than a sorted slice. */
function entityTerms(facts: readonly { related_entities: string[] }[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const fact of facts) {
    for (const raw of fact.related_entities) {
      const term = raw.trim();
      if (term.length === 0) continue;
      const key = term.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(term);
      if (out.length >= ENTITY_LIMIT) return out;
    }
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

  const commands: ViewCommand[] = [{ dimension: "cue", text, entities: entityTerms(op.facts) }];

  const found = op.memories.length + op.facts.length;
  if (found === 0) return commands;

  // Framing needs something to frame ON. The keyword lists behind
  // `related_entities` are extracted per fact and are routinely empty, so this
  // is a common case rather than an edge one — and a frame command with no
  // subject would either do nothing or, worse, reset the camera to the whole
  // corpus while claiming to have narrowed it.
  const entities = entityTerms(op.facts);
  if (entities.length > 0) commands.push({ dimension: "frame", entities });

  const stage = stageFor(op.memories);
  if (stage !== path) commands.push({ dimension: "destination", path: stage });

  return commands;
}

/** Where a command would take you, for the Follow offer's own label. */
const DESTINATION_NOUN: Record<string, string> = {
  "/geo": "the map",
  "/graph": "the graph",
  "/recall": "the results",
};

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
    if (command.dimension === "cue") parts.push(`show what it recalled for “${command.text}”`);
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
