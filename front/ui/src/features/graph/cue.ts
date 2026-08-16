/**
 * What the cue lights.
 *
 * The graph's cue channel has TWO producers and exactly one appearance: a
 * person typing in the header, and the conversation, which publishes the query
 * it asked plus the keyword lists of the facts it got back
 * (`RecallFact.related_entities`). Matching is a local string test over names
 * already in memory, which is why it can run per keystroke and why it costs
 * nothing to run again for a dozen terms.
 *
 * THE TERMS ARE NOT ENTITY IDS AND MUST NOT BE TREATED AS THEM. They are minted
 * by the keyword extractor during consolidation (src/memory/compression.rs,
 * `keyword_extractor.extract_texts`) — free text, cased however the source
 * text was, and frequently a fragment of the entity's name or an extension of
 * it. So a term matches an entity when either contains the other: "Baltimore"
 * has to find "Port of Baltimore", and "Maersk Line" has to find "Maersk".
 * Exact-matching them against `UniverseStar.name` looks stricter and in
 * practice matches almost nothing, which presents as a narrowing feature that
 * silently does not work.
 *
 * Kept out of the canvas so it can be tested without a DOM, a force layout or a
 * corpus: this is the rule that decides what an answer looks like on screen.
 */

/**
 * Terms shorter than this never match a name.
 *
 * The keyword extractor emits plenty of two-letter fragments, and a two-letter
 * substring test lights a third of any corpus — a "narrowing" that dims nothing
 * reads as a broken filter rather than as an answer. The same floor applies to
 * the NAME side, because the containment test runs both ways: without it, an
 * entity called "US" would match every term containing those letters.
 */
export const MIN_TERM_CHARS = 3;

/**
 * Does this entity name belong to the current cue?
 *
 * `text` is the typed cue, already lower-cased and trimmed by the caller (it is
 * read on every node of every frame, so the normalisation is hoisted out of the
 * loop). `entities` are the conversation's terms, normalised here because the
 * list is short and changes rarely.
 */
export function cueMatches(label: string, text: string, entities: readonly string[]): boolean {
  const name = label.toLowerCase();
  // The typed cue is a prefix-as-you-go substring test and keeps no floor: a
  // person typing "ba" means it, and they can see what they typed.
  if (text.length > 0 && name.includes(text)) return true;
  if (name.length < MIN_TERM_CHARS) return false;
  for (const entity of entities) {
    const term = entity.toLowerCase();
    if (term.length < MIN_TERM_CHARS) continue;
    if (name.includes(term) || term.includes(name)) return true;
  }
  return false;
}
