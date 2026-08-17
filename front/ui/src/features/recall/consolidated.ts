import type { RecallFact } from "@/lib/api";

/**
 * Reading a `RecallFact` honestly.
 *
 * Facts ride back on EVERY recall response and, until now, the only thing any
 * screen did with them was count them. The reason to be careful about the words
 * around them is that two of the four fields do not mean what their names
 * suggest, and both were about to be printed with the wrong label:
 *
 * WHAT A FACT IS. `SemanticFact.fact` is the REPRESENTATIVE MEMBER of a cluster
 * of pattern extractions, selected as the highest-confidence one
 * (src/memory/compression.rs:723-737, `select_representative` at :884). It is
 * lifted verbatim out of a memory — not paraphrased, not generated. That is why
 * the surface can say "not written by a model" about this block with the same
 * confidence it says it about a memory row, and it is also why some of them read
 * like fragments: they are fragments, of real records.
 *
 * WHAT `support_count` IS NOT. It is NOT the number of memories that mention the
 * fact. It is minted at 1 and incremented once per later memory that re-matches
 * the pattern (compression.rs:742 and :1694), and it feeds the decay half-life
 * (`180 + 30 × support_count` days, :1721-1733). Live proof that the two are
 * different: a fact on `claude-code` carries `support_count: 406` against a
 * `source_memories` array of length 4. Printing "406 supporting memories" would
 * be a fabrication about a number the server never claimed. "Confirmed 406×" is
 * what the increment actually counts.
 *
 * WHAT `related_entities` IS NOT. It is not an independent set of entity links.
 * It is `keyword_extractor.extract_texts(representative)` (compression.rs:735) —
 * keywords re-extracted FROM THE FACT'S OWN TEXT. On live data the fact "rs
 * Refactor: Extract recall handlers to handlers/recall" yields
 * `["extract","handlersrecall","refactor","recall","rs","handlers",…]`, which is
 * that same sentence with its spaces removed. Rendering it as "related entities"
 * would present the claim's own words back as corroboration of itself, so this
 * module exposes no helper for it and the surface renders none. (It stays useful
 * where it is already used — `stores/session.ts` unions it into the cue's
 * substring match, which is a keyword job and correct.)
 *
 * `confidence` and the fact text are the two fields that mean what they say.
 */

/** The confidence range across a fact set — the honest headline for a block
 *  whose rows are not uniformly trustworthy. */
export interface ConfidenceSpread {
  low: number;
  high: number;
}

/**
 * The lowest and highest confidence in a fact set, or `null` when there is
 * nothing to describe.
 *
 * THIS IS THE BLOCK'S CAVEAT, EXPRESSED AS A MEASUREMENT. A live recall returns
 * facts spanning 0.11 to 0.98 in one response, and a heading that said only "3
 * facts" would invite a reader to extend the top row's credibility to the bottom
 * one. A range is one short token, it is read off the same numbers the rows
 * print, and it collapses to a single value when the set genuinely is uniform —
 * which is the behaviour a sentence-shaped caveat could not have.
 */
export function confidenceSpread(facts: readonly RecallFact[]): ConfidenceSpread | null {
  if (facts.length === 0) return null;
  let low = facts[0].confidence;
  let high = facts[0].confidence;
  for (const fact of facts) {
    if (fact.confidence < low) low = fact.confidence;
    if (fact.confidence > high) high = fact.confidence;
  }
  return { low, high };
}

/**
 * How many times the store re-confirmed a fact, in words that match the
 * increment.
 *
 * A freshly minted fact carries 1 and has been "seen once (this extraction)" in
 * the source's own words (compression.rs:738-742), so 1 is NOT "confirmed once
 * over and above extraction" and must not be dressed as corroboration. Anything
 * above 1 is a later memory matching a pattern that already existed, which is
 * exactly what "confirmed" means.
 */
export function supportLabel(supportCount: number): string {
  if (supportCount <= 1) return "not yet re-confirmed";
  return `confirmed ${supportCount}×`;
}
