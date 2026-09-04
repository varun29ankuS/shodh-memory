import { useMemo } from "react";

import type { CorpusMemory } from "@/lib/api/corpus";
import {
  isolatedMemories,
  offPatternLocations,
  quantityOutliers,
  type LensResult,
} from "@/features/anomalies/measures";

/**
 * "What's worth a look" — the anomaly lenses, summarised.
 *
 * Adds no fetches: the three measures are pure functions over the corpus list
 * this view already holds, which is the same list /anomalies runs them on. So
 * this panel and that destination cannot disagree -- there is one
 * implementation and it lives there.
 *
 * TWO THINGS THIS PANEL MUST NOT DO, both of them tempting:
 *
 * It must not rank findings ACROSS lenses. `Finding.deviation` is documented in
 * measures.ts as comparable only within a lens -- a geographic modified z-score
 * and a dimensionless quantity ratio are different quantities -- so "the single
 * most anomalous memory in the corpus" is not a thing this data can answer.
 * Each lens reports its own leader and the reader compares them, or does not.
 *
 * It must not flatten "clear" into "insufficient". The measures went to some
 * trouble to distinguish "this lens looked and found nothing" from "there was
 * not enough here to have an opinion", and a briefing that prints "no
 * anomalies" over a corpus too small to measure is asserting a clean bill of
 * health nobody issued. Both states are shown, in their own words.
 */

const LENSES = [
  { id: "location", title: "Out of place", run: offPatternLocations },
  { id: "quantity", title: "Numbers that do not fit", run: quantityOutliers },
  { id: "isolation", title: "Connected to nothing", run: isolatedMemories },
] as const;

/** One line of evidence per lens. Long enough to recognise the memory, short
 *  enough that three of them stay a summary rather than a reading list. */
const EXCERPT = 88;

function excerpt(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > EXCERPT ? `${flat.slice(0, EXCERPT - 1).trimEnd()}…` : flat;
}

function LensRow({ title, result }: { title: string; result: LensResult }) {
  // Ordering WITHIN a lens by deviation is exactly what the field is for.
  const top =
    result.state === "findings"
      ? [...result.findings].sort((a, b) => b.deviation - a.deviation)[0]
      : null;

  return (
    <li className="border-t border-border py-2.5 first:border-t-0 first:pt-0">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[13px]">{title}</span>
        <span className="mono shrink-0 text-xs text-muted-foreground">
          {result.state === "findings" ? (
            <span className="text-[var(--node-anomalous)]">
              {result.findings.length} flagged
            </span>
          ) : result.state === "clear" ? (
            "nothing flagged"
          ) : (
            "not enough data"
          )}
        </span>
      </div>

      {top && (
        <p className="mt-1 text-xs leading-snug text-muted-foreground">{excerpt(top.content)}</p>
      )}
      {result.state === "insufficient" && (
        <p className="mt-1 text-xs leading-snug text-muted-foreground">{result.reason}</p>
      )}
    </li>
  );
}

export function AttentionPanel({ memories }: { memories: CorpusMemory[] }) {
  // Memoised on the list rather than on the query object, for the same reason
  // AnomaliesView does it: a background refetch returning identical data must
  // not reshuffle rows under a reader.
  const results = useMemo(
    () => LENSES.map((lens) => ({ ...lens, result: lens.run(memories) })),
    [memories],
  );

  const flagged = results.reduce(
    (n, r) => n + (r.result.state === "findings" ? r.result.findings.length : 0),
    0,
  );
  // A lens that could not run did not look, and the summary must not spend its
  // silence as evidence. "Three lenses, nothing flagged" over a corpus where
  // two of them reported "not enough data" is a clean bill of health that only
  // one lens actually issued -- the same flattening this panel's header warns
  // against, reintroduced one line above the rows that contradict it.
  const ran = results.filter((r) => r.result.state !== "insufficient").length;

  return (
    <section>
      <h2 className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
        What's worth a look
      </h2>

      <p className="mt-3 text-sm">
        {flagged > 0 ? (
          <>
            <span className="font-semibold text-[var(--node-anomalous)]">{flagged}</span> flagged
            {ran < results.length && (
              <span className="text-muted-foreground">
                {" "}
                · {results.length - ran} of {results.length} lenses had too little to measure
              </span>
            )}
          </>
        ) : ran === 0 ? (
          <span className="text-muted-foreground">
            Too little here for any of the {results.length} lenses to measure
          </span>
        ) : ran === results.length ? (
          <span className="text-muted-foreground">
            All {results.length} lenses ran · nothing flagged
          </span>
        ) : (
          <span className="text-muted-foreground">
            {ran} of {results.length} lenses could run · nothing flagged
          </span>
        )}
      </p>

      <ul className="mt-3">
        {results.map((r) => (
          <LensRow key={r.id} title={r.title} result={r.result} />
        ))}
      </ul>

      <a
        href="#/anomalies"
        className="mt-3 inline-block font-mono text-xs text-primary hover:underline"
      >
        how each lens drew its line →
      </a>
    </section>
  );
}
