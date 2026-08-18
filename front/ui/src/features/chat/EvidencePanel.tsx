import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSession } from "@/stores/session";
import type {
  ProactiveSurfacedMemory,
  RecallLineageEdge,
  RecallMemory,
  SeatEvent,
} from "@/lib/seat/types";
import { type ChatTurn, type ConvoLive, useChat } from "@/stores/chat";
import { ScoreBreakdown } from "@/features/inspector/ScoreBreakdown";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

/**
 * The evidence surface — a peer of the conversation, not an afterthought.
 *
 * Everything here is the retrieval pipeline's own account of itself, streamed
 * per turn: which memories surfaced, the full 20-field ScoreAttribution for
 * tool recalls (rendered by the same ScoreBreakdown the Recall view uses,
 * verified against live data), and the causal edges connecting members of a
 * result set. Proactive memories carry the backend's relevance_reason instead
 * of attribution — the backend genuinely does not compute attribution on that
 * path, and showing less is more honest than deriving something.
 *
 * Two levels: a digest of every turn's evidence (newest first), and a
 * memory-detail view any inline chip, citation or digest row focuses.
 *
 * The strip under the header appears ONLY once this conversation has actually
 * changed model — it is a report of something that happened, not a standing
 * claim. That is what makes it checkable: the reader watched the swap, and
 * the rows underneath are the same rows. It survives a reload because
 * `model_changed` is a durable event (seat/src/store.ts persists every
 * non-delta event) and `buildTurns` replays it into `ops`.
 *
 * "Already retrieved" is the exact scope. Turns taken after the swap do add
 * new rows below, retrieved by a new pass the new model drove — so the line
 * says the swap did not change what was retrieved, and never that the list
 * stopped growing or that any model retrieves the same memories.
 */

interface ResolvedMemory {
  kind: "recalled" | "surfaced";
  memory: RecallMemory | null;
  proactive: ProactiveSurfacedMemory | null;
  /** Causal edges within the recall set this memory arrived in. */
  lineage: RecallLineageEdge[];
  /** The rest of that set, for hopping along lineage edges. */
  siblings: RecallMemory[];
}

/**
 * The one op a selection names, and the memory inside it.
 *
 * ADDRESSES ONE OP, NOT THE WHOLE TURN. This used to take a turn number, index
 * the array with it, and then scan every op in that turn for the first
 * matching memory id — two separate ways of answering with the wrong record.
 *
 * The scan is the one that fired constantly. A turn runs several recalls and
 * they overlap: measured on a live conversation, five turns held 42 rows whose
 * (turn, memory) pair named two different results, and EVERY colliding pair
 * carried a different score. Memory `a8ca63ff` came back at 0.0816 from the
 * search "IVF-PQ index codebook adc kmeans quantization" and at 0.95 from
 * "product quantization codebook centroid probe rerank distance" in the same
 * turn. Clicking the row drawn at 95% opened a breakdown reading
 * `Vector + keyword 0.0589 · Final score 0.0418` — the other search's working,
 * under this search's row. This panel exists to show what retrieval did; it
 * was showing a different retrieval.
 *
 * The index is the other. `turns[turn - 1]` assumed the seat's turn LABEL
 * equals array position, and `applyEvent`'s `turn_start` overwrites that label
 * on the last turn from the server's own counter.
 */
export function resolveSelection(
  turns: ChatTurn[],
  turnIndex: number,
  opIndex: number,
  memoryId: string,
): ResolvedMemory | null {
  const op = turns[turnIndex]?.ops[opIndex];
  if (!op) return null;
  if (op.type === "memory_recall") {
    const memory = op.memories.find((candidate) => candidate.id === memoryId);
    if (!memory) return null;
    return {
      kind: "recalled",
      memory,
      proactive: null,
      lineage: op.lineage.filter((edge) => edge.from === memoryId || edge.to === memoryId),
      siblings: op.memories,
    };
  }
  if (op.type === "proactive_context") {
    const memory = op.memories.find((candidate) => candidate.id === memoryId);
    if (!memory) return null;
    return { kind: "surfaced", memory: null, proactive: memory, lineage: [], siblings: [] };
  }
  return null;
}

/** An entity as a handle onto the Recall surface: click runs a direct,
 *  deterministic search for it — no model, no tokens, instant. */
function EntityChip({ entity }: { entity: string }) {
  const navigate = useNavigate();
  const setActiveQuery = useSession((s) => s.setActiveQuery);
  return (
    <button
      type="button"
      onClick={() => {
        setActiveQuery(entity);
        navigate("/recall");
      }}
      title={`Search memory for “${entity}” (direct recall, no model)`}
      className="focus-visible:ring-ring rounded focus-visible:ring-2 focus-visible:outline-none"
    >
      <Badge className="hover:border-primary/40 hover:text-foreground cursor-pointer transition-colors">
        {entity}
      </Badge>
    </button>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <div className="text-muted-foreground/70 mb-1 text-[10px] tracking-wide uppercase">{label}</div>
      {children}
    </div>
  );
}

function MemoryDetail({
  resolved,
  conversationId,
  turnIndex,
  opIndex,
}: {
  resolved: ResolvedMemory;
  conversationId: string;
  turnIndex: number;
  opIndex: number;
}) {
  const select = useChat((s) => s.select);
  const memory = resolved.memory;
  const proactive = resolved.proactive;
  const content = memory?.experience.content ?? proactive?.content ?? "";
  const tags = memory?.experience.tags ?? proactive?.tags ?? [];
  const createdAt = memory?.created_at ?? proactive?.created_at;
  const tier = memory?.tier ?? proactive?.tier;

  return (
    <div className="px-4 py-3">
      <button
        type="button"
        onClick={() => select(null)}
        className="text-muted-foreground hover:text-foreground focus-visible:ring-ring -ml-1 flex items-center gap-1 rounded px-1 py-0.5 text-[11px] focus-visible:ring-2 focus-visible:outline-none"
      >
        <ArrowLeft aria-hidden="true" className="size-3" />
        All evidence
      </button>

      <p className="mt-2 text-[13px] leading-relaxed">{content}</p>

      <div className="mt-2 flex flex-wrap gap-1">
        <Badge className="mono">{resolved.kind === "recalled" ? "tool recall" : "auto-surfaced"}</Badge>
        {tier ? <Badge className="mono">{tier}</Badge> : null}
        {proactive ? <Badge className="mono">{proactive.memory_type}</Badge> : null}
      </div>

      {tags.length > 0 ? (
        <Field label="Entities">
          {/* Each entity hops into DIRECT recall — the deterministic, model-free
              surface — with the entity as the cue. The conversation is one way
              into memory, never the only one. */}
          <div className="flex flex-wrap gap-1">
            {tags.map((tag) => (
              <EntityChip key={tag} entity={tag} />
            ))}
          </div>
        </Field>
      ) : null}

      {createdAt ? (
        <Field label="Recorded">
          <p className="mono text-[11px]">{new Date(createdAt).toLocaleString()}</p>
        </Field>
      ) : null}

      {proactive ? (
        <Field label="Why it surfaced">
          <p className="text-muted-foreground text-[12px] leading-relaxed">
            {proactive.relevance_reason}
          </p>
          {proactive.matched_entities?.length ? (
            <div className="mt-1 flex flex-wrap gap-1">
              {proactive.matched_entities.map((entity) => (
                <Badge key={entity}>{entity}</Badge>
              ))}
            </div>
          ) : null}
        </Field>
      ) : null}

      {resolved.lineage.length > 0 ? (
        <section className="border-border -mx-4 mt-3 border-t px-4 pt-3">
          <h3 className="text-[12px] font-medium tracking-tight">What it connects to</h3>
          <div className="mt-2 flex flex-col gap-1">
            {resolved.lineage.map((edge, index) => {
              const memoryId = resolved.memory?.id;
              const otherId = edge.from === memoryId ? edge.to : edge.from;
              const other = resolved.siblings.find((candidate) => candidate.id === otherId);
              const outgoing = edge.from === memoryId;
              return (
                <button
                  key={`${edge.from}-${edge.to}-${index}`}
                  type="button"
                  disabled={!other}
                  // Stays inside the op that produced this record: `siblings`
                  // is that one op's result set, so a lineage hop resolved
                  // turn-wide could land on another search's copy of the same
                  // memory, with another search's score attribution.
                  onClick={() =>
                    select({ conversationId, turnIndex, opIndex, memoryId: otherId })
                  }
                  className="hover:bg-accent/60 focus-visible:ring-ring rounded px-2 py-1.5 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none disabled:opacity-50"
                >
                  <span className="text-primary mono text-[10px]">
                    {outgoing ? "→" : "←"} {edge.relation}
                  </span>
                  <span className="text-muted-foreground mt-0.5 line-clamp-2 block text-[11px] leading-relaxed">
                    {other?.experience.content ?? "Outside this result set"}
                  </span>
                </button>
              );
            })}
          </div>
        </section>
      ) : null}

      {resolved.memory?.score_attribution ? (
        <div className="-mx-4 mt-3">
          <ScoreBreakdown attr={resolved.memory.score_attribution} />
        </div>
      ) : null}
    </div>
  );
}

/**
 * One turn's evidence, one row per result.
 *
 * A ROW IS AN (OP, MEMORY) PAIR, NOT A MEMORY. The same memory answering two
 * of a turn's searches is two results with two scores, and both are true — the
 * digest keeps both and draws two bars. What it must not do is give them one
 * identity, which `${turn.turn}-${id}-${label}` did: 252 React duplicate-key
 * errors on a single conversation, and one destination for two different
 * clicks. The op index rides on each row, so the key is unique and the click
 * lands on the search the reader pointed at.
 */
function TurnDigest({
  turn,
  turnIndex,
  conversationId,
}: {
  turn: ChatTurn;
  /** Position in `ConvoLive.turns` — the selection's address, and the key. */
  turnIndex: number;
  conversationId: string;
}) {
  const select = useChat((s) => s.select);
  // Indices are captured against `turn.ops` BEFORE the filter, because that is
  // the array `resolveSelection` indexes back into.
  const groups = turn.ops
    .map((op, opIndex) => ({ op, opIndex }))
    .filter(
      (
        entry,
      ): entry is {
        op: Extract<SeatEvent, { type: "memory_recall" | "proactive_context" }>;
        opIndex: number;
      } =>
        (entry.op.type === "memory_recall" && entry.op.scope === "user") ||
        entry.op.type === "proactive_context",
    );
  const rows = groups.flatMap(({ op, opIndex }) =>
    op.type === "memory_recall"
      ? op.memories.map((memory) => ({
          opIndex,
          id: memory.id,
          content: memory.experience.content,
          score: memory.score,
          label: "recall",
        }))
      : op.memories.map((memory) => ({
          opIndex,
          id: memory.id,
          content: memory.content,
          score: memory.score,
          label: "auto",
        })),
  );
  if (rows.length === 0) return null;

  return (
    <section className="border-border border-b px-4 py-3">
      <h3 className="text-muted-foreground text-[10px] tracking-wide uppercase">
        Turn {turn.turn}
        {groups.some(({ op }) => op.type === "memory_recall") ? (
          <span className="normal-case">
            {" — “"}
            {
              (
                groups.find(({ op }) => op.type === "memory_recall")!.op as Extract<
                  SeatEvent,
                  { type: "memory_recall" }
                >
              ).query
            }
            {"”"}
          </span>
        ) : null}
      </h3>
      <div className="mt-2 flex flex-col gap-0.5">
        {rows.map((row) => (
          <button
            key={`${row.opIndex}-${row.id}-${row.label}`}
            type="button"
            onClick={() =>
              select({ conversationId, turnIndex, opIndex: row.opIndex, memoryId: row.id })
            }
            className="hover:bg-accent/60 focus-visible:ring-ring flex items-start gap-2 rounded px-1.5 py-1 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none"
          >
            <span className="bg-border mt-[7px] h-[3px] w-10 shrink-0 overflow-hidden rounded-full">
              <span
                className="bg-muted-foreground/60 block h-full rounded-full"
                style={{ width: `${Math.max(2, Math.min(100, row.score * 100))}%` }}
              />
            </span>
            <span className="text-foreground/90 line-clamp-2 min-w-0 flex-1 text-[11px] leading-relaxed">
              {row.content}
            </span>
            <Badge className="mono mt-0.5 shrink-0">{row.label}</Badge>
          </button>
        ))}
      </div>
    </section>
  );
}

export function EvidencePanel({
  conversationId,
  convo,
  onClose,
  className,
}: {
  conversationId: string | null;
  convo: ConvoLive | null;
  /** Present when rendered as an overlay (narrow viewports). */
  onClose?: () => void;
  className?: string;
}) {
  const selected = useChat((s) => s.selected);

  const resolved = useMemo(() => {
    if (!selected || !convo || selected.conversationId !== conversationId) return null;
    return resolveSelection(convo.turns, selected.turnIndex, selected.opIndex, selected.memoryId);
  }, [selected, convo, conversationId]);

  // A swap that this conversation actually performed. The op only reaches the
  // client when the seat flushes its pending events at the start of the next
  // turn (seat/src/conversation.ts sendMessage), so this line arrives with the
  // new model's first answer — the moment the reader is looking for the
  // difference the swap made to the evidence.
  const swapped = useMemo(
    () => (convo?.turns ?? []).some((turn) => turn.ops.some((op) => op.type === "model_changed")),
    [convo],
  );

  // Position in `convo.turns` is captured BEFORE the filter and the reverse.
  // Keying on position within this derived array would renumber every digest
  // each time a new evidence-bearing turn arrives — a full remount of the
  // panel per turn, and a fresh wrong answer out of `resolveSelection`.
  const turnsWithEvidence = useMemo(
    () =>
      (convo?.turns ?? [])
        .map((turn, turnIndex) => ({ turn, turnIndex }))
        .filter(({ turn }) =>
          turn.ops.some(
            (op) =>
              (op.type === "memory_recall" && op.scope === "user") ||
              (op.type === "proactive_context" && op.memories.length > 0),
          ),
        )
        .reverse(),
    [convo],
  );

  return (
    <aside
      aria-label="Evidence"
      className={cn("border-border bg-card flex min-h-0 flex-col border-l", className)}
    >
      <header className="border-border flex h-10 shrink-0 items-center justify-between border-b pr-1.5 pl-4">
        <div className="min-w-0">
          <h2 className="text-[12px] font-medium tracking-tight">Evidence</h2>
        </div>
        {onClose ? (
          <Button size="icon" variant="ghost" aria-label="Close evidence panel" onClick={onClose}>
            <X />
          </Button>
        ) : null}
      </header>

      {/* Its own strip rather than a second header line: the header is h-10 to
          stay level with the conversation's, and this sentence has to WRAP at
          the panel's narrow widths (min(340px,26vw) — 266px at 1024) instead of
          truncating into half a claim. */}
      {swapped ? (
        <p className="border-border text-muted-foreground shrink-0 border-b px-4 py-1.5 text-[11px] leading-relaxed">
          Switching models did not change what was already retrieved.
        </p>
      ) : null}

      <ScrollArea className="min-h-0 flex-1">
        {resolved && selected ? (
          <MemoryDetail
            resolved={resolved}
            conversationId={selected.conversationId}
            turnIndex={selected.turnIndex}
            opIndex={selected.opIndex}
          />
        ) : turnsWithEvidence.length > 0 ? (
          turnsWithEvidence.map(({ turn, turnIndex }) => (
            <TurnDigest
              key={turnIndex}
              turn={turn}
              turnIndex={turnIndex}
              conversationId={conversationId ?? ""}
            />
          ))
        ) : (
          <div className="px-4 py-3">
            <p className="text-muted-foreground text-[12px] leading-relaxed">
              Evidence appears here the moment the seat recalls — before the
              model answers. Every memory carries the retrieval pipeline's own
              score attribution, and every answer that uses one cites it.
            </p>
          </div>
        )}
      </ScrollArea>
    </aside>
  );
}
