import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { MapPin, Ruler, Unlink, type LucideIcon } from "lucide-react";
import { ApiError, NetworkError, outageOf, type Reachability } from "@/lib/api";
import { fetchAnomalies } from "@/lib/api/anomalies";
import { useCorpus } from "@/lib/api/corpus";
import { useSession } from "@/stores/session";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { SpatialPlot } from "./SpatialPlot";
import { RatioPlot } from "./RatioPlot";
import { DegreePlot } from "./DegreePlot";
import {
  isolatedMemories,
  offPatternLocations,
  quantityOutliers,
  type Finding,
  type LensChart,
  type LensResult,
  type Pattern,
} from "./measures";
import { SurpriseLens, surpriseKey } from "./SurpriseLens";
import { readSurprise } from "./surprise";

/**
 * Anomalies — what deviates from this profile's own baseline.
 *
 * THE SCREEN'S JOB IS THAT YOU SEE THE FINDINGS BEFORE YOU READ ANYTHING. Two
 * earlier passes computed the right things and then printed them: a vertical
 * list of memory sentences with a number attached, which asks a reader to
 * RECONSTRUCT a distribution from four rows of prose and to notice, unaided,
 * that two of those rows are the same event. Compressing the sentences into
 * tokens made the list shorter and left it a list. Nothing was plotted, nothing
 * showed where a flagged value sat relative to normal, and nothing showed that
 * two findings were one finding.
 *
 * So every lens that reaches a conclusion now DRAWS ITS POPULATION FIRST, with
 * the flagged points in it, and the rows underneath are confirmation. Each
 * graphic is chosen for the shape of its own measure rather than for
 * consistency with the others — a distance-from-centre with a heavy tail, a
 * ratio between two magnitudes, and a degree distribution are three different
 * kinds of quantity, and one house chart type would have flattered exactly none
 * of them. The reasoning for each is in its own file.
 *
 * MEMORY TEXT IS EVIDENCE, NOT THE HEADLINE. It sat first and ran to two or
 * three lines, which is what made the screen a wall: the sentence is the part
 * a reader needs LAST, once the plot and the magnitude have already told them
 * which memory to care about. It is now one truncated line under the numbers,
 * with the Inspector one click away for the whole record.
 *
 * PATTERNS ARE A SEPARATE CLAIM AND ARE MARKED AS ONE. Findings that the
 * measure can show belong together are drawn tied on the plot and bracketed in
 * the list, with the evidence for the grouping stated in the bracket. The rule
 * is deliberately strict and lives in measures.ts; what matters here is that a
 * group never asserts a CAUSE, it states what two findings have in common and
 * lets the reader supply the rest.
 *
 * THREE REGISTERS, AND THE RULE THAT SEPARATES THEM. `facts` (the baseline
 * numbers) and `caveat` (anything that changes how those numbers should be
 * read) are always on screen. `looksFor` and `detail` — what the lens is for,
 * and how the arithmetic works — sit behind the section's info affordance,
 * because they are read once and then known. A caveat never goes behind it: the
 * published finding on info tips is that most people never open them, and a
 * limitation nobody reads makes the screen look like it claimed more than it
 * did.
 *
 * The measures are in measures.ts and are pure arithmetic — a median centre and
 * a median absolute deviation over great-circle distance, unit-matched number
 * comparison, and term overlap. No model is called, here or on the server, and
 * that is the claim the header makes: the store finds these in its own data.
 * The rebuild changed what those measures REPORT — they now return the whole
 * distribution rather than only the points past the line, because a flagged
 * value is meaningless without the ordinary ones it was judged against — and
 * changed nothing about how any of them decides.
 *
 * TWO KINDS OF ANOMALY SHIP HERE, AND BOTH ARE NAMED. The three lenses above
 * read a memory's CONTENT. The fourth, `SurpriseLens`, is `POST /api/anomalies`
 * — the server's own five-axis deviation scoring over the extraction statistics
 * captured at ingest, which had no client but the MCP tool while this screen
 * computed something else under the same word. It is not a replacement: it
 * measures a different object, against a different baseline, and a memory can be
 * ordinary on one and far out on the other. It is last because it is the only
 * section that cannot draw its population — the endpoint returns its ranked
 * findings, not the set it ranked them against — and a section that cannot show
 * its working should not lead.
 *
 * THE HEADER'S "no model, here or on the server" SURVIVES THE ADDITION, which
 * is why the fourth lens could join without weakening the claim: the endpoint's
 * own module doc commits to scoring "deterministically and without any LLM in
 * the loop", and every number it returns is a z-score against a rolling mean.
 *
 * ONE REQUEST OF ITS OWN, NOW. The three client lenses still read `useCorpus`,
 * the same single react-query entry Recall and Geo already populate
 * (lib/api/corpus.ts), so arriving here from either is instant. The fourth
 * fetches `/api/anomalies` on its own key and is the only network cost this
 * destination adds over the shared listing.
 *
 * A section that cannot reach a conclusion says why, with the shortfall in
 * numbers, instead of rendering an empty box that reads as "clean". "Not enough
 * placed memories to have a baseline" and "no memory is out of place" are
 * different findings and this screen never lets one impersonate the other.
 */

interface Lens {
  id: string;
  icon: LucideIcon;
  title: string;
  /** What this looks for, before any data — true even when there is none. Read
   *  once and then known, so it lives behind the section's info affordance
   *  rather than as a permanent line above every list. */
  looksFor: string;
  result: LensResult;
}

function LensGraphic({ chart, patterns }: { chart: LensChart; patterns: Pattern[] }) {
  switch (chart.kind) {
    case "spatial":
      return <SpatialPlot chart={chart} patterns={patterns} />;
    case "ratio":
      return <RatioPlot chart={chart} />;
    case "degree":
      return <DegreePlot chart={chart} />;
  }
}

/**
 * One flagged memory.
 *
 * Selecting sets the one global selection (`useSession().select`), exactly as
 * a recall result row does, so an anomaly is the same kind of object as any
 * other memory in the product. NOTE that this route does not itself mount the
 * Inspector — `ROUTES_WITH_INSPECTOR` is `/recall`, `/geo`, `/graph`
 * (app/App.tsx:53) — so the selection made here is carried BY the session to
 * those destinations rather than opening a pane in place. The row therefore has
 * to show its own selected state or the click would appear to do nothing, and
 * the plot above lights the same memory, so a click in either place is visibly
 * answered in both.
 *
 * THE MAGNITUDE LEADS, IN THE MONO FACE, and the memory's own words follow in
 * one clamped line. That order is the whole difference between this and the
 * wall it replaces: a column of numerals is scannable without reading, which is
 * the same argument that puts numerics in their own column in a dense table,
 * and the sentence is there to confirm a finding the reader has already found
 * rather than to deliver it.
 *
 * SEVERITY IS POSITION, AND ONLY POSITION — no bar, no colour ramp, no badge.
 * A bar was tried here and removed: the underlying magnitudes are unbounded, so
 * on a real corpus where one memory sits 126 robust scales out and the next
 * sits 14, a bar scaled to the largest drew the second as an empty stub. The
 * plot above now carries the magnitude comparison on a scale built to survive
 * that range, which is where an unbounded quantity belongs; the row states its
 * own number and takes its rank from its position in the list.
 */
function FindingRow({ finding }: { finding: Finding }) {
  const selected = useSession((s) => s.selectedMemoryId === finding.memoryId);
  const select = useSession((s) => s.select);

  return (
    <button
      type="button"
      onClick={() => select(finding.memoryId)}
      aria-current={selected ? "true" : undefined}
      className={cn(
        "border-border w-full border-b px-4 py-2 text-left transition-colors duration-100",
        "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        selected ? "bg-primary/10" : "hover:bg-accent/60",
      )}
    >
      {/* The evidence, first. Two tokens, not a sentence: what this one
          measured, and what that measurement is of. The baseline it is measured
          against is the section's, stated once above and drawn once above. */}
      <Meta className="text-[12px]">
        <Stat value={finding.value} />
        <span>{finding.against}</span>
      </Meta>
      <p
        className={cn(
          "mt-0.5 truncate text-[12px] leading-relaxed",
          selected ? "text-foreground/80" : "text-muted-foreground",
        )}
      >
        {finding.content}
      </p>
    </button>
  );
}

/**
 * Findings the measure can show belong together.
 *
 * The bracket is a rule down the left and one line of evidence — not a heading,
 * not a colour, and above all not a stated cause. What the arithmetic knows is
 * that these findings share a term, and a magnitude or a type; what it does not
 * know is why, and the difference between those two is the difference between
 * this screen and one that guesses.
 */
function PatternGroup({ pattern, findings }: { pattern: Pattern; findings: Finding[] }) {
  return (
    // `--node-anomalous` is a graph-canvas custom property, deliberately
    // outside the shadcn set and so outside `@theme inline` — there is no
    // `border-node-anomalous` utility for Tailwind to generate, and writing one
    // would compile clean and render nothing. The token is applied directly.
    <div
      className="border-l-2"
      style={{ borderColor: "color-mix(in oklab, var(--node-anomalous) 45%, transparent)" }}
    >
      <div className="px-4 pt-2">
        <Meta className="text-[11px]">
          <span className="text-foreground/70">
            {findings.length} findings, one pattern
          </span>
          {pattern.evidence.map((e) => (
            <span key={e}>{e}</span>
          ))}
        </Meta>
      </div>
      {findings.map((f) => (
        <FindingRow key={`${f.memoryId}-${f.value}`} finding={f} />
      ))}
    </div>
  );
}

/**
 * A limitation on what the section above could conclude.
 *
 * Marked, indented and always rendered — never folded into the info panel. It
 * is the one thing on this screen that is neither a finding nor an explanation:
 * it says the measure could not reach as far as the reader would assume, and a
 * reader who misses it credits the screen with a stronger result than it has.
 */
function Caveat({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-muted-foreground/80 border-warn/40 mt-2 border-l pl-2.5 text-[11px] leading-relaxed">
      {children}
    </p>
  );
}

function LensSection({ lens }: { lens: Lens }) {
  const Icon = lens.icon;
  const result = lens.result;
  const count = result.state === "findings" ? result.findings.length : 0;

  /**
   * The findings in display order, with grouped ones kept together.
   *
   * A pattern's members take the position of whichever of them ranked highest,
   * so grouping never reorders the section's furthest-first logic more than it
   * has to — the strongest finding stays where a reader expects it and brings
   * its relatives with it.
   */
  const blocks = useMemo(() => {
    if (result.state !== "findings") return [];
    const patternOf = new Map<string, Pattern>();
    for (const p of result.patterns) {
      for (const id of p.memoryIds) patternOf.set(id, p);
    }
    const done = new Set<Pattern>();
    const out: Array<
      { kind: "single"; finding: Finding } | { kind: "pattern"; pattern: Pattern; findings: Finding[] }
    > = [];
    for (const finding of result.findings) {
      const pattern = patternOf.get(finding.memoryId);
      if (!pattern) {
        out.push({ kind: "single", finding });
        continue;
      }
      if (done.has(pattern)) continue;
      done.add(pattern);
      out.push({
        kind: "pattern",
        pattern,
        findings: result.findings.filter((f) => pattern.memoryIds.includes(f.memoryId)),
      });
    }
    return out;
  }, [result]);

  const chart = result.state === "insufficient" ? undefined : result.chart;
  const patterns = result.state === "findings" ? result.patterns : [];

  return (
    <section>
      {/* Title, count, and the affordance that holds everything this section
          used to say in prose — all on the one sticky row, so scrolling past a
          section never loses the ability to ask what it was measuring. */}
      <div className="border-border bg-muted/50 sticky top-0 z-10 flex items-center gap-2 border-b px-4 py-1.5 backdrop-blur-sm">
        <Icon aria-hidden="true" className="text-muted-foreground size-3" strokeWidth={1.8} />
        <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          {lens.title}
        </span>
        {count > 0 ? (
          <span className="text-muted-foreground/60 mono text-[10px]">{count}</span>
        ) : null}
        <InfoHint label={lens.title}>
          <span className="block">{lens.looksFor}</span>
          {result.detail ? (
            <span className="text-muted-foreground mt-1.5 block">{result.detail}</span>
          ) : null}
          {count > 1 ? (
            <span className="text-muted-foreground mt-1.5 block">
              Ordered furthest from the baseline first.
            </span>
          ) : null}
        </InfoHint>
        {/* The section's verdict, right, where the eye finishes the row —
            global identity left, contextual state right, as a status bar
            does. "Clear" and "not enough data" are different claims and each
            gets its own word; neither is ever a blank space. */}
        <span className="text-muted-foreground/60 ml-auto text-[11px]">
          {result.state === "clear"
            ? "Nothing flagged"
            : result.state === "insufficient"
              ? "No baseline"
              : null}
        </span>
      </div>

      <div className="border-border border-b px-4 py-2">
        {/* The baseline: where this lens drew its line on this corpus, as
            numbers. This is the half of every row's comparison that does not
            change, and stating it here is what lets each row below be two
            tokens instead of a sentence. */}
        <Meta>
          {result.facts.map((f) => (
            <span key={f}>{f}</span>
          ))}
        </Meta>
        {result.state === "insufficient" ? <Caveat>{result.reason}</Caveat> : null}
        {result.state !== "insufficient" && result.caveat ? (
          <Caveat>{result.caveat}</Caveat>
        ) : null}
      </div>

      {/* The population, drawn. Above the rows because it is what the rows are
          confirming, and present on a clear result too — "nothing flagged" over
          a drawn distribution is a result, over an empty panel it is a shrug. */}
      {chart ? (
        <div className="border-border border-b">
          <LensGraphic chart={chart} patterns={patterns} />
        </div>
      ) : null}

      {blocks.map((block) =>
        block.kind === "single" ? (
          <FindingRow
            key={`${lens.id}-${block.finding.memoryId}-${block.finding.value}`}
            finding={block.finding}
          />
        ) : (
          <PatternGroup
            key={`${lens.id}-pattern-${block.pattern.memoryIds.join("-")}`}
            pattern={block.pattern}
            findings={block.findings}
          />
        ),
      )}
    </section>
  );
}

function LensSkeleton() {
  return (
    <section>
      <div className="border-border border-b px-4 py-1.5">
        <Skeleton className="h-3 w-32" />
      </div>
      <div className="border-border border-b px-4 py-3">
        <Skeleton className="h-3 w-[70%]" />
      </div>
      <div className="border-border border-b px-4 py-3">
        <Skeleton className="h-[132px] w-full" />
      </div>
    </section>
  );
}

export function AnomaliesView({ reach }: { reach: Reachability }) {
  const { data, error, isFetching, profile } = useCorpus(reach);
  const memories = useMemo(() => data?.memories ?? [], [data]);

  // Three passes over one already-fetched list. Memoised on the corpus rather
  // than on the query object so a background refetch that returns identical
  // data does not recompute — and, more to the point, does not reorder rows
  // under a reader's cursor.
  const lenses = useMemo<Lens[]>(
    () => [
      {
        id: "location",
        icon: MapPin,
        title: "Out of place",
        looksFor:
          "Memories whose coordinates sit outside the cluster the rest of this profile's placed memories form.",
        result: offPatternLocations(memories),
      },
      {
        id: "quantity",
        icon: Ruler,
        title: "Numbers that do not fit",
        looksFor:
          "Figures written in the same unit that disagree — inside one memory, or against the range every other memory in that unit establishes.",
        result: quantityOutliers(memories),
      },
      {
        id: "isolation",
        icon: Unlink,
        title: "Connected to nothing",
        looksFor:
          "Memories that name nothing any other memory here names, so no path in the graph reaches them.",
        result: isolatedMemories(memories),
      },
    ],
    [memories],
  );

  // The server lens's own result, off the key `SurpriseLens` uses. React-query
  // serves both callers from one request; this one exists so the header's count
  // spans every measure on the screen rather than only the three computed here.
  const { data: surprise } = useQuery({
    queryKey: surpriseKey(profile),
    queryFn: ({ signal }) => fetchAnomalies(profile!, signal),
    enabled: profile !== null,
    staleTime: 60_000,
  });

  const flagged = useMemo(() => {
    const ids = new Set<string>();
    for (const lens of lenses) {
      if (lens.result.state !== "findings") continue;
      for (const f of lens.result.findings) ids.add(f.memoryId);
    }
    // A UNION, NOT A SUM. The content measures and the extraction measure can
    // both flag the same memory, and adding their totals would count it twice
    // under a word a reader takes to mean "how many memories here are unusual".
    const server = surprise ? readSurprise(surprise) : null;
    if (server?.state === "findings") {
      for (const entry of server.flagged) ids.add(entry.memory_id);
    }
    return ids.size;
  }, [lenses, surprise]);

  const patternCount = useMemo(
    () =>
      lenses.reduce(
        (a, lens) => a + (lens.result.state === "findings" ? lens.result.patterns.length : 0),
        0,
      ),
    [lenses],
  );

  // A REJECTED KEY IS NOT A STOPPED SERVER — see `outageOf`. The sentence
  // below is the offline case only.
  const outage = outageOf(
    reach,
    "These measures read this profile's memory, which needs the server running.",
  );
  if (outage) return <EmptyState size="page" {...outage} />;

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile to measure"
        body="A baseline is built from one profile's memories."
      />
    );
  }

  if (error) {
    const detail =
      error instanceof ApiError
        ? error.isAuthFailure
          ? "The server rejected this key."
          : `The server answered ${error.status}.`
        : error instanceof NetworkError
          ? "The server stopped responding mid-request."
          : "Something went wrong loading this profile's memories.";
    return <EmptyState size="page" title="Could not read the corpus" body={detail} />;
  }

  if (isFetching && !data) {
    return (
      <div className="mx-auto h-full w-full max-w-2xl">
        <div className="border-border border-b px-4 py-4">
          <Skeleton className="h-3.5 w-[60%]" />
          <Skeleton className="mt-2 h-2.5 w-[85%]" />
        </div>
        <LensSkeleton />
        <LensSkeleton />
      </div>
    );
  }

  if (memories.length === 0) {
    return (
      <EmptyState
        size="page"
        title="Anomalies show what deviates from this profile's normal"
        body="Normal is learned from the memories themselves, so this fills once there are enough of them to have a shape."
        more="Every measure here compares a memory with the others in the same profile, so a baseline needs memories to be built from. These appear as soon as anything is stored."
      />
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-2xl pb-16">
        {/* The header a stranger reads first, and a returning reader skips.
            It is a count, not a claim about the screen: the destination is
            already named in the top bar, and repeating "what stands out in
            this profile's memory" underneath it was two lines of sentence
            saying what one number says.

            The one thing that is NOT obvious from looking — that these
            findings are arithmetic over the store's own contents rather than a
            judgement fetched from somewhere else — stays on screen as a chip,
            because it is the product's actual claim and a claim behind an icon
            is a claim nobody reads. The icon holds only how it works. */}
        <header className="border-border flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b px-4 py-3">
          {/* The count spans all FOUR measures. Reading the server lens's
              result off the same react-query key it uses means this number
              cannot drift from the section below it, and react-query dedupes
              the two callers into one request. Counted as a union of memory
              ids, not a sum: a memory the arithmetic and the server both flag
              is one unusual memory, and adding the two totals would report it
              twice under a word that means "how many things are wrong here". */}
          <Meta className="text-[12px]">
            <Stat value={flagged} label="flagged" />
            {patternCount > 0 ? <Stat value={patternCount} label="pattern" /> : null}
            <Stat value={memories.length} label="memories" />
            <Stat value={lenses.length + 1} label="measures" />
          </Meta>
          {/* Not "No model": the seat's egress badge already occupies that
              exact phrase in the corner of every screen, and two unrelated
              claims wearing one wording is worse than either being longer. */}
          <span className="border-border text-muted-foreground ml-auto flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px]">
            Arithmetic, not a model
            <InfoHint label="how these are found">
              Every finding on this screen is arithmetic over the memories already stored here — a
              median centre and a robust spread, unit-matched number comparison, and term overlap.
              No model is called, here or on the server, and nothing is inferred that the corpus
              does not already state.
            </InfoHint>
          </span>
        </header>

        {lenses.map((lens) => (
          <LensSection key={lens.id} lens={lens} />
        ))}

        {/* The server's own measure, last — see the note at the top of this
            file for why it is a fourth lens rather than a replacement for the
            three above, and why it is the one that goes at the bottom. */}
        <SurpriseLens profile={profile} />
      </div>
    </ScrollArea>
  );
}
