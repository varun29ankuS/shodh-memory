import { MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";
import type { RecallMemory } from "@/lib/api";
import { useSession } from "@/stores/session";
import { useSurfacedMemoryIds } from "@/stores/chat";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * The result column.
 *
 * A row is the entry point to both chains in WORKFLOWS.md — selecting it opens
 * the Inspector, which is the only object-detail surface in the product. So the
 * row's job is to carry just enough to choose between candidates, and nothing
 * more: the content, when it was recorded, and how strongly it surfaced.
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

function ResultRow({ memory, surfaced }: { memory: RecallMemory; surfaced: boolean }) {
  const selected = useSession((s) => s.selectedMemoryId === memory.id);
  const select = useSession((s) => s.select);

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
        {surfaced ? (
          // The deterministic search and the model reached the same memory.
          // That agreement is worth seeing — it is the whole reason recall sits
          // beside the seat rather than behind it — but it is a note, not a
          // state: `--muted-foreground`, never the accent, because the accent
          // already means "selected" two lines up and a second orange here
          // would make every marked row look chosen.
          <span
            className="text-muted-foreground/70 ml-auto shrink-0"
            title="Also surfaced by the conversation"
          >
            <MessageSquare aria-hidden="true" className="size-3" />
            <span className="sr-only">Also surfaced by the conversation</span>
          </span>
        ) : null}
      </div>
    </button>
  );
}

export function ResultList({ memories }: { memories: RecallMemory[] }) {
  // One derivation for the whole list rather than a subscription per row: the
  // set is rebuilt only when the active conversation's turns change.
  const surfaced = useSurfacedMemoryIds();

  return (
    <ScrollArea role="list" className="min-h-0 flex-1">
      {memories.map((m) => (
        <div role="listitem" key={m.id}>
          <ResultRow memory={m} surfaced={surfaced.has(m.id)} />
        </div>
      ))}
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
