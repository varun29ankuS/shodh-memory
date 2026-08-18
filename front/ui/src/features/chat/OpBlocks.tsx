import { useMutation } from "@tanstack/react-query";
import { ArrowDownToLine, BrainCircuit, RotateCcw, Search, Wrench } from "lucide-react";
import { cn } from "@/lib/utils";
import { revertLedgerEvent } from "@/lib/seat/client";
import type { RecallMemory, SeatEvent } from "@/lib/seat/types";
import { useChat } from "@/stores/chat";
import { Badge } from "@/components/ui/badge";

/**
 * Memory operations as first-class transcript elements.
 *
 * These render inline, expanded by default — hiding the memory system inside
 * a collapsed "used a tool" row would bury the one thing this product does
 * that a chat clone does not. Each block is a summary with real numbers; the
 * full attribution lives one click away in the evidence panel.
 *
 * Reverts are real: every learning block carries the ledger event id the seat
 * recorded before moving on, and the revert button calls
 * POST /seat/v1/learning/revert — the ledger's compensation semantics
 * (exact delete for writes, compensating counter-reinforcement otherwise)
 * are the seat's, not invented here.
 */

function OpShell({
  icon,
  tint,
  children,
  className,
}: {
  icon: React.ReactNode;
  tint?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "border-border bg-card/60 flex gap-2.5 rounded-lg border px-3 py-2.5",
        className,
      )}
    >
      <span className={cn("mt-0.5 shrink-0", tint ?? "text-muted-foreground")}>{icon}</span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

function ScoreBar({ score, className }: { score: number; className?: string }) {
  return (
    <span className={cn("bg-border h-[3px] w-12 shrink-0 overflow-hidden rounded-full", className)}>
      <span
        className="bg-muted-foreground/60 block h-full rounded-full"
        style={{ width: `${Math.max(2, Math.min(100, score * 100))}%` }}
      />
    </span>
  );
}

function MemoryRow({
  id,
  content,
  score,
  sources,
  selected,
  onSelect,
}: {
  id: string;
  content: string;
  score: number;
  sources?: string[];
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-label={`Inspect memory ${id.slice(0, 8)}`}
      className={cn(
        "flex w-full items-start gap-2 rounded-md px-1.5 py-1 text-left transition-colors duration-100",
        "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
        selected ? "bg-primary/10" : "hover:bg-accent/60",
      )}
    >
      <ScoreBar score={score} className="mt-[7px]" />
      <span className="text-foreground/90 line-clamp-2 min-w-0 flex-1 text-[12px] leading-relaxed">
        {content}
      </span>
      {sources?.length ? (
        <span className="mt-0.5 flex shrink-0 gap-1">
          {sources.map((source) => (
            <Badge key={source} className="mono">
              {source}
            </Badge>
          ))}
        </span>
      ) : null}
    </button>
  );
}

export function RecallBlock({
  op,
  turnIndex,
  opIndex,
  conversationId,
}: {
  op: Extract<SeatEvent, { type: "memory_recall" }>;
  /** Position in `ConvoLive.turns`, never the seat's turn label. */
  turnIndex: number;
  /** Position in `ChatTurn.ops` — which of the turn's searches this block is. */
  opIndex: number;
  conversationId: string;
}) {
  const selected = useChat((s) => s.selected);
  const select = useChat((s) => s.select);

  if (op.scope === "harness") {
    // Operating notes about the harness itself — real, but not user memory;
    // one quiet line, not a card competing with the user's evidence.
    return (
      <p className="text-muted-foreground flex items-center gap-2 px-1 text-[11px]">
        <BrainCircuit aria-hidden="true" className="size-3.5 shrink-0" />
        Consulted {op.memories.length} operating note{op.memories.length === 1 ? "" : "s"} from
        previous sessions
      </p>
    );
  }

  const shown = op.memories.slice(0, 3);
  const extra = op.memories.length - shown.length;

  return (
    <OpShell icon={<Search aria-hidden="true" className="size-4" />} tint="text-primary">
      <p className="text-[12px]">
        <span className="text-muted-foreground">Recalled memory for</span>{" "}
        <span className="font-medium">“{op.query}”</span>
      </p>
      <p className="text-muted-foreground mono mt-0.5 text-[10px]">
        {op.memories.length} memor{op.memories.length === 1 ? "y" : "ies"}
        {op.facts.length > 0 ? ` · ${op.facts.length} fact${op.facts.length === 1 ? "" : "s"}` : ""}
        {op.lineage.length > 0 ? ` · ${op.lineage.length} causal edge${op.lineage.length === 1 ? "" : "s"}` : ""}
        {" · "}
        {op.took_ms}ms · {op.mode}
      </p>
      {op.memories.length === 0 ? (
        <p className="text-muted-foreground mt-1.5 text-[11px]">
          Nothing surfaced for that cue — the seat records this as an operating
          note and will rephrase next time.
        </p>
      ) : (
        <div className="mt-1.5 flex flex-col gap-0.5">
          {shown.map((memory: RecallMemory) => (
            <MemoryRow
              key={memory.id}
              id={memory.id}
              content={memory.experience.content}
              score={memory.score}
              sources={memory.score_attribution?.sources}
              // A turn runs several searches and they return overlapping
              // memories at different scores. Comparing on memory id and turn
              // alone lit the same row in every block of the turn, and the
              // selection they produced was indistinguishable, so the panel
              // resolved all of them to whichever search came first.
              selected={
                selected?.memoryId === memory.id &&
                selected.turnIndex === turnIndex &&
                selected.opIndex === opIndex
              }
              onSelect={() =>
                select({ conversationId, turnIndex, opIndex, memoryId: memory.id })
              }
            />
          ))}
          {extra > 0 ? (
            <button
              type="button"
              onClick={() => {
                const next = op.memories[3];
                if (next) select({ conversationId, turnIndex, opIndex, memoryId: next.id });
              }}
              className="text-muted-foreground hover:text-foreground focus-visible:ring-ring self-start rounded px-1.5 py-0.5 text-[11px] focus-visible:ring-2 focus-visible:outline-none"
            >
              +{extra} more in evidence
            </button>
          ) : null}
        </div>
      )}
    </OpShell>
  );
}

export function ProactiveBlock({
  op,
  turnIndex,
  opIndex,
  conversationId,
}: {
  op: Extract<SeatEvent, { type: "proactive_context" }>;
  turnIndex: number;
  opIndex: number;
  conversationId: string;
}) {
  const selected = useChat((s) => s.selected);
  const select = useChat((s) => s.select);
  const feedback = op.feedback;

  if (op.memories.length === 0 && !feedback) return null;

  return (
    <OpShell icon={<BrainCircuit aria-hidden="true" className="size-4" />}>
      {op.memories.length > 0 ? (
        <>
          <p className="text-muted-foreground text-[12px]">
            Surfaced {op.memories.length} memor{op.memories.length === 1 ? "y" : "ies"} before
            responding <span className="mono text-[10px]">({op.took_ms}ms)</span>
          </p>
          <div className="mt-1.5 flex flex-col gap-0.5">
            {op.memories.map((memory) => (
              <MemoryRow
                key={memory.id}
                id={memory.id}
                content={memory.content}
                score={memory.score}
                selected={
                  selected?.memoryId === memory.id &&
                  selected.turnIndex === turnIndex &&
                  selected.opIndex === opIndex
                }
                onSelect={() =>
                  select({ conversationId, turnIndex, opIndex, memoryId: memory.id })
                }
              />
            ))}
          </div>
        </>
      ) : null}
      {feedback && feedback.memories_evaluated > 0 ? (
        <p className={cn("text-muted-foreground text-[11px]", op.memories.length > 0 && "mt-1.5")}>
          Last turn's evidence:{" "}
          {feedback.reinforced.length > 0 ? `${feedback.reinforced.length} reinforced` : null}
          {feedback.reinforced.length > 0 && feedback.weakened.length > 0 ? ", " : null}
          {feedback.weakened.length > 0 ? `${feedback.weakened.length} weakened` : null}
          {feedback.reinforced.length === 0 && feedback.weakened.length === 0
            ? `${feedback.memories_evaluated} evaluated, none moved`
            : null}
        </p>
      ) : null}
    </OpShell>
  );
}

function RevertButton({ ledgerEventId }: { ledgerEventId: string }) {
  const reverted = useChat((s) => Boolean(s.revertedLedgerIds[ledgerEventId]));
  const markReverted = useChat((s) => s.markReverted);
  const revert = useMutation({
    mutationFn: () => revertLedgerEvent(ledgerEventId),
    onSuccess: () => markReverted(ledgerEventId),
  });

  if (reverted) {
    return <span className="text-muted-foreground shrink-0 text-[10px]">reverted</span>;
  }
  return (
    <button
      type="button"
      onClick={() => revert.mutate()}
      disabled={revert.isPending}
      title="Revert this learning update (compensating action via the seat ledger)"
      aria-label="Revert this learning update"
      className={cn(
        "text-muted-foreground hover:text-foreground flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[10px]",
        "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none disabled:opacity-50",
      )}
    >
      <RotateCcw aria-hidden="true" className="size-3" />
      revert
    </button>
  );
}

const OUTCOME_DOT: Record<string, string> = {
  helpful: "bg-[var(--live)]",
  neutral: "bg-muted-foreground/50",
  misleading: "bg-destructive",
};

export function ReinforceBlock({ op }: { op: Extract<SeatEvent, { type: "memory_reinforce" }> }) {
  const trigger =
    op.trigger.kind === "citation"
      ? "cited in the answer"
      : op.trigger.kind === "response_overlap"
        ? op.outcome === "helpful"
          ? "content used in the answer"
          : "surfaced but unused"
        : op.trigger.kind === "negative_followup"
          ? `negative follow-up (${op.trigger.keywords.join(", ")})`
          : "revert";

  return (
    <div className="flex items-center gap-2 px-1 text-[11px]">
      <span
        aria-hidden="true"
        className={cn("size-1.5 shrink-0 rounded-full", OUTCOME_DOT[op.outcome])}
      />
      <span className="text-muted-foreground min-w-0 flex-1 truncate">
        {op.outcome === "helpful" ? "Reinforced" : op.outcome === "misleading" ? "Weakened" : "Noted"}{" "}
        {op.memory_ids.length} memor{op.memory_ids.length === 1 ? "y" : "ies"}
        {op.scope === "harness" ? " (operating notes)" : ""} — {trigger}
      </span>
      <RevertButton ledgerEventId={op.ledger_event_id} />
    </div>
  );
}

export function WriteBlock({ op }: { op: Extract<SeatEvent, { type: "memory_write" }> }) {
  return (
    <div className="flex items-center gap-2 px-1 text-[11px]">
      <ArrowDownToLine aria-hidden="true" className="text-muted-foreground size-3.5 shrink-0" />
      <span className="text-muted-foreground min-w-0 flex-1 truncate">
        Wrote {op.scope === "harness" ? "operating note" : "memory"} ({op.memory_type}): “
        {op.content_preview}”
      </span>
      <RevertButton ledgerEventId={op.ledger_event_id} />
    </div>
  );
}

/** Names the seat's own memory tools, whose calls are already rendered as
 *  richer blocks above (seat/src/conversation.ts MEMORY_TOOL_NAMES). */
const MEMORY_TOOL_NAMES = new Set(["recall_memory", "remember_memory", "record_seat_learning"]);

export function ToolBlock({
  op,
  ops,
}: {
  op: Extract<SeatEvent, { type: "tool_call_start" }>;
  ops: SeatEvent[];
}) {
  if (MEMORY_TOOL_NAMES.has(op.tool_name)) return null;
  const end = ops.find(
    (candidate): candidate is Extract<SeatEvent, { type: "tool_call_end" }> =>
      candidate.type === "tool_call_end" && candidate.tool_call_id === op.tool_call_id,
  );
  return (
    <div className="flex items-center gap-2 px-1 text-[11px]">
      <Wrench aria-hidden="true" className="text-muted-foreground size-3.5 shrink-0" />
      <span className="text-muted-foreground mono min-w-0 flex-1 truncate">{op.tool_name}</span>
      <span
        className={cn(
          "mono shrink-0 text-[10px]",
          end ? (end.is_error ? "text-destructive" : "text-muted-foreground") : "text-muted-foreground animate-pulse",
        )}
      >
        {end ? (end.is_error ? "failed" : "done") : "running…"}
      </span>
    </div>
  );
}

export function HarnessAppliedBlock({
  op,
}: {
  op: Extract<SeatEvent, { type: "harness_learning_applied" }>;
}) {
  return (
    <p className="text-muted-foreground flex items-center gap-2 px-1 text-[11px]" title={op.memories.map((m) => m.content).join("\n")}>
      <BrainCircuit aria-hidden="true" className="size-3.5 shrink-0" />
      Applied {op.memories.length} operating note{op.memories.length === 1 ? "" : "s"} to this turn
    </p>
  );
}

export function ModelChangedBlock({ op }: { op: Extract<SeatEvent, { type: "model_changed" }> }) {
  return (
    <div className="text-muted-foreground flex items-center gap-2 text-[10px]">
      <span className="bg-border h-px flex-1" />
      <span className="mono shrink-0">
        model → {op.model.name} ({op.model.provider})
      </span>
      <span className="bg-border h-px flex-1" />
    </div>
  );
}

export function ErrorBlock({ op }: { op: Extract<SeatEvent, { type: "error" }> }) {
  return (
    <p className="text-destructive px-1 text-[11px] leading-relaxed">{op.message}</p>
  );
}
