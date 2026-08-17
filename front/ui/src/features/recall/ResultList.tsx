import { cn } from "@/lib/utils";
import type { RecallFact, RecallLineageEdge, RecallMemory } from "@/lib/api";
import { useSession } from "@/stores/session";
import { Meta, Stat } from "@/components/ui/meta";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { whyItSurfaced } from "./why";
import { Facts } from "./Facts";

/**
 * The result column.
 *
 * A row is the entry point to both chains in WORKFLOWS.md — selecting it opens
 * the Inspector, which is the only object-detail surface in the product. So the
 * row's job is to carry just enough to choose between candidates, and nothing
 * more: the content, when it was recorded, how strongly it surfaced, and — the
 * one addition — why.
 *
 * WHY THE REASON IS ON THE ROW. The full attribution has always been one click
 * away in the Inspector, and one click away is where it stayed: a ranked list
 * with no stated reason reads as a black box, and nobody opens a panel to
 * check a list they have already decided not to trust. So the strongest single
 * line comes forward and the arithmetic stays behind (see why.ts, which is also
 * where the constraints on the wording live). It is one line, muted, under the
 * content — a reason competing with the memory for attention would have traded
 * one unreadable row for another.
 *
 * The line is absent, not blank, when the server attributed nothing: corpus
 * rows in the pre-query listing were never retrieved and have no reason to
 * give, and a row of dashes standing in for that would be a fabrication.
 *
 * The score shown is the server's normalised `score`, not the raw fusion
 * output. It is rendered as a bar rather than a number because the useful
 * question at this stage is "how does this compare to the one above it", not
 * "what is its absolute value" — and a decimal invites false precision about a
 * quantity that is only meaningful relative to the top hit.
 */

function relativeDay(iso: string): string {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const days = Math.floor((Date.now() - then.getTime()) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}

function ResultRow({
  memory,
  lineage,
}: {
  memory: RecallMemory;
  lineage: RecallLineageEdge[] | undefined;
}) {
  const selected = useSession((s) => s.selectedMemoryId === memory.id);
  const select = useSession((s) => s.select);
  const why = whyItSurfaced(memory, lineage);

  return (
    <button
      type="button"
      onClick={() => select(memory.id)}
      aria-current={selected ? "true" : undefined}
      className={cn(
        "border-border w-full border-b px-4 py-3 text-left transition-colors duration-100",
        "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        selected ? "bg-primary/10" : "hover:bg-accent/60",
      )}
    >
      <p
        className={cn(
          "line-clamp-3 text-[13px] leading-relaxed",
          selected ? "text-foreground" : "text-foreground/90",
        )}
      >
        {memory.experience.content}
      </p>

      {/* Why it is here, in the human's words. Two clauses at most, and each is
          omitted independently: a memory the graph reached with no lineage to
          the rest of the set says only the first, one in a set with no
          attribution says only the second. */}
      {/* Two tokens, not two clauses. "Connects to 6 others here" said in five
          words what "connects to 6 here" says in four and a mono numeral — and
          this line repeats on all twenty-five rows, so a word saved is
          twenty-five words off the column. The leg phrase is not shortened
          further: "No text matched — the graph reached it" is the product's
          differentiator stated where it happens, and it is the one thing on a
          result row that a reader has never seen before. */}
      {why ? (
        <Meta className="mt-1 text-muted-foreground/70">
          {why.legs}
          {why.links > 0 ? (
            <>
              <span>connects to</span>
              <Stat value={why.links} label="here" />
            </>
          ) : null}
        </Meta>
      ) : null}

      <div className="mt-2 flex items-center gap-2">
        {/* Relative strength against the top hit, which is what the server's
            normalisation already expresses. */}
        <span className="bg-border h-[3px] w-16 shrink-0 overflow-hidden rounded-full">
          <span
            className={cn("block h-full rounded-full", selected ? "bg-primary" : "bg-muted-foreground/60")}
            style={{ width: `${Math.max(2, Math.min(100, memory.score * 100))}%` }}
          />
        </span>
        <span className="text-muted-foreground mono text-[10px]">
          {relativeDay(memory.created_at)}
        </span>
        {memory.experience.geo_location ? (
          <span className="text-muted-foreground mono text-[10px]" title="Has coordinates">
            {memory.experience.geo_location[0].toFixed(2)},
            {memory.experience.geo_location[1].toFixed(2)}
          </span>
        ) : null}
      </div>
    </button>
  );
}

export function ResultList({
  memories,
  lineage,
  facts,
}: {
  memories: RecallMemory[];
  /** The causal edges the same response carried. Optional because the pre-query
   *  corpus listing is not a result set and has none. */
  lineage?: RecallLineageEdge[];
  /** The consolidated claims the same response carried. Optional for the same
   *  reason as `lineage`: the pre-query corpus listing is not a result set. */
  facts?: RecallFact[];
}) {
  return (
    <ScrollArea className="min-h-0 flex-1">
      {/* INSIDE THE SCROLL, ABOVE THE ROWS. The facts are what the store
          concluded and the rows are what it recorded, so the conclusions read
          first — but they are capped at five by the server and would otherwise
          hold up to a third of a 340px column permanently, pushing the first
          result off screen. Scrolling past them costs nothing and returns the
          whole column to the results.

          OUTSIDE THE LIST SEMANTICS. `role="list"` moved off the ScrollArea and
          onto the rows' own container: a section of consolidated claims is not
          one of the results, and leaving it inside the list would have made a
          screen reader announce a set of N+1 items where N surfaced. */}
      <Facts facts={facts} />
      <div role="list">
        {memories.map((m) => (
          <div role="listitem" key={m.id}>
            <ResultRow memory={m} lineage={lineage} />
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}

/**
 * Skeleton rows in the real row's shape — three content lines, a score bar,
 * a date — rather than the "Searching…" text this replaced. `aria-hidden`
 * because a screen reader has nothing useful to announce about placeholder
 * shapes; the search form's own state is what's meaningful.
 */
function SkeletonRow() {
  return (
    <div className="border-border border-b px-4 py-3">
      <Skeleton className="h-[13px] w-full" />
      <Skeleton className="mt-1.5 h-[13px] w-[88%]" />
      <Skeleton className="mt-1.5 h-[13px] w-[55%]" />
      <div className="mt-2.5 flex items-center gap-2">
        <Skeleton className="h-[3px] w-16 rounded-full" />
        <Skeleton className="h-2.5 w-9" />
      </div>
    </div>
  );
}

export function ResultListSkeleton() {
  return (
    <div role="list" aria-hidden="true" className="min-h-0 flex-1 overflow-y-auto">
      {Array.from({ length: 7 }, (_, i) => (
        <SkeletonRow key={i} />
      ))}
    </div>
  );
}
