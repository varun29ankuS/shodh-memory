import type { RecallFact } from "@/lib/api";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { confidenceSpread, supportLabel } from "./consolidated";

/**
 * What the store has CONCLUDED, above what it has recorded.
 *
 * These arrive on every recall response and have never been rendered: the only
 * consumer in the product read `facts.length` to print the word "3 facts" in the
 * chat transcript. The whole point of a consolidating memory is that it forms
 * durable claims out of repeated episodes, and the claims were being paid for on
 * every request and thrown away.
 *
 * A SECOND KIND OF ANSWER, SO IT IS SET APART FROM THE ROWS. A memory is a
 * record — one thing that happened, once. A fact is a pattern the consolidator
 * found across several of them and has been carrying since, with its own
 * confidence and its own decay clock. Interleaving the two would have made the
 * fact look like an unusually short memory. It sits above the results, inside
 * the same scroll so it never permanently costs the column height, and it is
 * ruled off rather than boxed — the house style on every other surface here.
 *
 * THE CONFIDENCE RANGE IS IN THE HEADING, NOT BEHIND THE ICON. On live data one
 * response carries a fact at 0.98 and a fact at 0.11 — the latter being a Windows
 * file path that the pattern extractor mistook for a claim. A heading that said
 * "3 facts" would lend the top row's credibility to the bottom one. The range is
 * three characters wider and is the one thing a reader needs before reading any
 * row, so it goes where it cannot be missed. That is the same rule the Anomalies
 * screen states: mechanism behind the affordance, caveats never.
 *
 * NOT FILTERED, NOT RE-SORTED, NOT TRUNCATED. The server already dedupes, sorts
 * by confidence descending and caps at 5 (src/handlers/recall.rs:947-955). A
 * client-side quality threshold on top would be this screen quietly deciding
 * what the store is allowed to have concluded, and hiding the weak ones is
 * precisely how a reader ends up trusting the strong ones more than they should.
 * With a server cap of five there is also nothing to page, so this block adds no
 * control of any kind — no filter, no sort, no expander. Nothing here is state.
 */

/**
 * One consolidated claim.
 *
 * NOT SELECTABLE, and that is a deliberate limit rather than an oversight. Every
 * other row in this product sets the one global selection and opens the
 * Inspector on a memory id. A fact is not a memory: `RecallFact` carries no
 * `source_memories` on the wire (src/handlers/types.rs:221-227 projects the
 * stored `SemanticFact` down to five fields and drops the source list), so there
 * is no id here to open anything on. Making the row look pressable and then
 * doing nothing would be worse than a row that does not offer.
 */
function FactRow({ fact }: { fact: RecallFact }) {
  return (
    <div className="border-border/60 border-b px-4 py-2 last:border-b-0">
      {/* The claim, verbatim. Clamped rather than truncated with a slice so the
          browser's own ellipsis does it at the real rendered width — this column
          is `min(340px,42vw)` and a character count would break at the wrong
          place on every viewport but one.

          TWO LINES, MEASURED. At three the block ran 371px in the live browser
          and put the first result at y=479 on a 945px viewport — a section of
          conclusions holding back nearly half the column before a single record.
          Two lines is also the shape of the data: the facts that earn their
          place are short declaratives ("The user prefers Rust for all backend
          services", 1.00, confirmed 1189×) and the ones that run long are the
          extractor's mistakes, so the clamp lands hardest exactly where the
          content is weakest. */}
      <p className="text-foreground/90 line-clamp-2 text-[12px] leading-relaxed">{fact.fact}</p>
      {/* The two numbers that decide whether to believe it. Confidence is the
          store's own ratcheting score; the support label counts re-confirmation,
          which is a different quantity and is worded so it cannot be read as a
          count of supporting memories. */}
      <Meta className="mt-1 text-[10px]">
        <Stat value={fact.confidence.toFixed(2)} label="confidence" />
        <span>{supportLabel(fact.support_count)}</span>
      </Meta>
    </div>
  );
}

export function Facts({ facts }: { facts: RecallFact[] | undefined }) {
  const spread = confidenceSpread(facts ?? []);
  // Absent rather than empty. Most queries surface no facts at all — the server
  // only attaches them for entities that appear in the recalled memories — and a
  // permanent "0 facts" rule above every result set would be a line of chrome
  // reporting the ordinary case. The results head already says what surfaced.
  if (!facts || facts.length === 0 || !spread) return null;

  return (
    <section className="border-border border-b">
      <div className="bg-muted/40 border-border/60 flex items-center gap-2 border-b px-4 py-1.5">
        <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          Concluded
        </span>
        <Meta className="text-[10px]">
          <Stat value={facts.length} label={facts.length === 1 ? "fact" : "facts"} />
          {/* One value when the set is uniform, a range when it is not. A range
              of 0.11–0.98 tells a reader in six characters that these rows are
              not equally good, which is the honest headline for this block. */}
          <Stat
            value={
              spread.low === spread.high
                ? spread.low.toFixed(2)
                : `${spread.low.toFixed(2)}–${spread.high.toFixed(2)}`
            }
            label="confidence"
          />
        </Meta>
        <InfoHint label="how these facts were formed">
          <span className="block">
            Claims the store consolidated from repeated memories, not from this query. Each one is
            the highest-confidence member of a cluster of similar extractions, lifted verbatim from
            a memory rather than rewritten — so nothing here was generated, exactly as on the rows
            below.
          </span>
          <span className="text-muted-foreground mt-1.5 block">
            Shown for entities that appear in these results, best five by confidence. Confidence
            rises each time a later memory re-confirms a fact and decays if none does;
            re-confirmation also lengthens how long the fact resists being forgotten.
          </span>
        </InfoHint>
      </div>
      {facts.map((fact) => (
        <FactRow key={fact.id} fact={fact} />
      ))}
    </section>
  );
}
