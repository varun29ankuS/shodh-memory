import type { SeatEvent } from "@/lib/seat/types";
import type { ViewDimension } from "./authority";

/**
 * The conversation, translated into view state.
 *
 * THIS IS THE CONTRACT, so it is a pure function over a wire event and it is
 * the thing the tests hold.
 *
 * TWO PRODUCERS, AND THEY ARE NOT THE SAME KIND OF THING.
 *
 *  - `memory_recall` is an INFERENCE. The model searched; the browser reads the
 *    query it composed and narrows to it. The model never asked for this and
 *    cannot say why it happened, because it did not decide that it should.
 *  - `view_command` is a REQUEST. The model called `direct_view` (seat/src/
 *    view-tools.ts) with a destination, entities and a reason in its own words.
 *    Everything in it was validated seat-side against the real destination list
 *    and this profile's actual graph, so nothing here re-checks it.
 *
 * The second is why this file exists in its current shape. An inference can
 * only ever say "I searched for this"; it cannot open the map because an answer
 * turned out to be geographic. Both still pass through the same authority
 * ledger, and neither seizes anything.
 *
 * Commands are NAMED AND SERIALIZABLE (spec §5): a declined one is held as a
 * Follow offer and applied later, so it cannot close over a canvas, a store or
 * a moment in time.
 */

/**
 * A command, optionally carrying the words that justify it.
 *
 * `reason` IS ON EVERY VARIANT AND SHARED BY EVERY COMMAND FROM ONE REQUEST,
 * because the reason belongs to the MOVE and the move is split across
 * dimensions only as an implementation fact. The store holds declined commands
 * one per dimension, so a reason stored anywhere but on the command itself
 * would be gone by the time a Follow offer is rendered — the offer would ask
 * the person to accept a change with no account of why.
 *
 * Absent on commands derived from a recall, and that absence is honest: nobody
 * gave a reason for those, and inventing one ("because you asked about X") puts
 * words in the model's mouth that it never said.
 */
interface Authored {
  reason?: string;
  /**
   * The seat tool call this command came from, and the address its verdict is
   * sent back to (`lib/view/outcome.ts`).
   *
   * ON THE COMMAND, NOT IN A SIDE TABLE, for exactly the reason `reason` is: a
   * declined command is held as a Follow offer and may be accepted minutes
   * later, long after whatever dispatched it has gone. An origin kept anywhere
   * but here would be missing at the one moment it is needed — when the person
   * finally answers.
   *
   * Absent on commands derived from a recall, and that absence is load-bearing:
   * nobody asked for those, so there is nobody to report to, and inventing a
   * recipient would put a verdict in the trail for a request that was never made.
   */
  origin?: string;
}

export type ViewCommand =
  | ({ dimension: "cue"; text: string; entities: string[] } & Authored)
  | ({ dimension: "frame"; entities: string[] } & Authored)
  | ({ dimension: "destination"; path: string } & Authored)
  | ({ dimension: "focus"; id: string; name: string } & Authored);

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
 * Where a RECALL is looked at: the graph, and only the graph.
 *
 * This is the destination of the inference path, not a ceiling on the product.
 * A recall says what was searched for and nothing about where the answer
 * belongs; the graph responds to a cue on every answer and is therefore the one
 * surface that is honest for all of them. Choosing between the graph and the
 * map on a recall would mean guessing at the shape of an answer nobody
 * described.
 *
 * THE MAP IS NOT ABSENT FROM THE PRODUCT, though it used to be absent from this
 * function, and the note that stood here is now out of date: `GeoView` consumed
 * only the committed search when it was written, so a `/geo` destination landed
 * on the whole corpus under a chip claiming the view was following. It reads
 * `useView(s => s.cue)` today, raises the points whose memory mentions a term,
 * and `GeoMap` re-fits to what was raised. The map rejoined exactly as that
 * note said it would — through `direct_view`, where the model states that the
 * answer is geographic instead of the browser guessing it.
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
 * A `view_command` the model issued, as view commands.
 *
 * NOTHING IS RE-VALIDATED HERE and that is the division of labour: the seat
 * resolved the destination against the real surface list and every entity name
 * against this profile's graph before emitting the event, because those are the
 * checks whose failure must reach the MODEL — a tool that silently framed a
 * name the corpus does not contain would teach it to report a narrowing that
 * never happened. By the time the event is on the wire the questions are
 * answered, and a second opinion here could only disagree with the answer the
 * model was already given.
 *
 * THE CUE AND THE FRAME ARE BOTH SENT, because they are two different things
 * that both need saying. The cue is what is LIT — `EntityCanvas` recedes
 * unmatched nodes, `GeoView` raises the points that mention a term — and it is
 * the only channel the map has. The frame is the graph's CAMERA. Sending only
 * the frame would aim at entities that were never distinguished from their
 * neighbours; sending only the cue would dim the corpus around a subset that
 * may be off screen.
 *
 * THE CUE'S TEXT IS THE ENTITY LIST, NOT THE REASON. `dispatch` writes a cue
 * through to `useSession.setCue`, which fills the visible search field — so the
 * text has to be something a person could have typed and can check the picture
 * against. The reason is prose about evidence; it belongs beside the change,
 * where FollowOffer renders it, not inside the field the person searches with.
 */
function commandsFromRequest(
  op: Extract<SeatEvent, { type: "view_command" }>,
  path: string,
): ViewCommand[] {
  const reason = op.reason.trim();
  // A command with no account of itself is exactly what this whole feature
  // exists to replace, and the tool requires one. An event without it is a
  // wire-level malformation, not a move to make silently.
  if (reason.length === 0) return [];

  const commands: ViewCommand[] = [];
  const entities = op.entities.filter((entity) => entity.trim().length > 0);
  const origin = op.tool_call_id;

  if (entities.length > 0) {
    commands.push({ dimension: "cue", text: entities.join(", "), entities, reason, origin });
    commands.push({ dimension: "frame", entities, reason, origin });
  }

  // A destination equal to where the person already stands is not a move. The
  // guard is the same one the recall path uses, and it matters more here: the
  // model chose this surface deliberately, so a redundant navigation would
  // remount the stage under someone who was already reading it.
  if (op.destination !== null && !isAlreadyThere(op, path)) {
    commands.push({ dimension: "destination", path: op.destination, reason, origin });
  }

  // The one object to open. Sent whether or not the destination moved: the
  // inspector is part of the shell, so an entity opened from the map is open on
  // the map.
  if (op.focus !== null) {
    commands.push({ dimension: "focus", id: op.focus.id, name: op.focus.name, reason, origin });
  }

  return commands;
}

/**
 * Whether the destination this request names is the one already on screen.
 *
 * THE SAME PREDICATE THE GUARD ABOVE USES, exported rather than inlined, because
 * the return path has to report this case and could only otherwise re-derive it.
 * "The view was already there" is not "nothing happened": the person IS looking
 * at what the model asked for, and a model told nothing at all would conclude
 * its request vanished. Two copies of the condition would eventually disagree,
 * and the disagreement would be a command silently dropped while the seat was
 * told it applied.
 */
export function isAlreadyThere(
  op: Extract<SeatEvent, { type: "view_command" }>,
  path: string,
): boolean {
  return op.destination !== null && op.destination === path;
}

/**
 * One memory operation, as view commands. Empty when the op says nothing about
 * what to look at.
 *
 * WHAT IS DELIBERATELY NOT MAPPED, since the absences are decisions:
 *
 *  - `proactive_context` fires on EVERY turn, unasked, and its `query` is the
 *    person's own message. Deriving a cue from it would narrow the view on
 *    every turn including the ones where memory was never mentioned, which
 *    reads as the app twitching rather than as the view following anything.
 *  - `memory_write`, `memory_reinforce`, `harness_learning_applied` are changes
 *    to memory's state. Nothing about storing or strengthening a memory says
 *    where the person should be looking.
 *  - `tool_call_start` / `tool_call_end` name bridged MCP tools whose set
 *    changes as servers connect and disconnect, under names the server chooses
 *    (`mcp__<server>__<tool>`). Mapping `list_anomalies` to `/anomalies` would
 *    be a guess about a string another program owns, and it would break
 *    silently when that program renamed it. A model that wants the anomalies
 *    screen has `direct_view` and can say so.
 *  - `model_changed`, `usage`, `error`, `turn_*` describe the conversation's
 *    machinery, not its subject.
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
  if (op.type === "view_command") return commandsFromRequest(op, path);
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

/**
 * Where a command would take you, for the Follow offer's own label.
 *
 * ALL TEN SURFACES NOW, because `direct_view` can name any of them; this used
 * to hold one entry because `STAGE` was the only destination that could be
 * produced. The path itself remains the fallback, so a surface added to the
 * router before it is added here reads as a path rather than as nothing — an
 * offer that says "open /anomalies" is ugly, and an offer that says "open " is
 * broken.
 *
 * The nouns are how a destination is spoken in a sentence, which is not always
 * its rail label: the row says "Conversations", the sentence says "open the
 * conversation". Kept in step with seat/src/view-tools.ts, whose `noun` field
 * says the same words to the model.
 */
const DESTINATION_NOUN: Record<string, string> = {
  "/": "the briefing",
  "/chat": "the conversation",
  "/recall": "recall",
  "/graph": "the graph",
  "/geo": "the map",
  "/anomalies": "anomalies",
  "/tasks": "tasks",
  "/history": "history",
  "/sources": "sources",
  "/providers": "providers",
};

/**
 * The reason behind a set of commands, or null when they carry none.
 *
 * Separate from {@link describeCommands} rather than folded into its sentence,
 * because the two are different KINDS of claim and the interface must be able
 * to distinguish them. The description is the app's own account of what would
 * happen and it is generated; the reason is the model's words, quoted. Running
 * them together into one string would make the app's phrasing look like
 * something the model said.
 *
 * The first reason present wins: every command from one `direct_view` call
 * carries the same one, and commands from different calls cannot reach here
 * together — the store holds at most one offer per dimension and a later
 * command replaces an earlier one on that dimension.
 */
export function reasonOf(commands: readonly ViewCommand[]): string | null {
  for (const command of commands) {
    const reason = command.reason?.trim();
    if (reason) return reason;
  }
  return null;
}

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
    else if (command.dimension === "focus") parts.push(`open ${command.name}`);
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
