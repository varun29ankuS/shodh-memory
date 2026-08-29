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
import { templateStripper } from "./shared-template";
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

function resolveSelection(
  turns: ChatTurn[],
  turn: number,
  memoryId: string,
): ResolvedMemory | null {
  const target = turns[turn - 1];
  if (!target) return null;
  for (const op of target.ops) {
    if (op.type === "memory_recall") {
      const memory = op.memories.find((candidate) => candidate.id === memoryId);
      if (memory) {
        return {
          kind: "recalled",
          memory,
          proactive: null,
          lineage: op.lineage.filter((edge) => edge.from === memoryId || edge.to === memoryId),
          siblings: op.memories,
        };
      }
    } else if (op.type === "proactive_context") {
      const memory = op.memories.find((candidate) => candidate.id === memoryId);
      if (memory) return { kind: "surfaced", memory: null, proactive: memory, lineage: [], siblings: [] };
    }
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
  turn,
}: {
  resolved: ResolvedMemory;
  conversationId: string;
  turn: number;
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
                  onClick={() => select({ conversationId, turn, memoryId: otherId })}
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
 * Every memory this conversation surfaced, for the template stripper.
 *
 * Conversation-scoped, not turn-scoped, for the same reason the citation map
 * is: the template is a property of the corpus, and deriving it per turn would
 * make the same memory read two ways in two places on one screen.
 */
function useEvidenceStripper(turns: ChatTurn[]): (s: string) => string {
  return useMemo(() => {
    const contents: string[] = [];
    for (const turn of turns) {
      for (const op of turn.ops) {
        if (op.type === "memory_recall") {
          for (const memory of op.memories) contents.push(memory.experience.content);
        } else if (op.type === "proactive_context") {
          for (const memory of op.memories) contents.push(memory.content);
        }
      }
    }
    return templateStripper(contents.map((c) => c.replace(/\s+/g, " ").trim()));
  }, [turns]);
}

function TurnDigest({
  turn,
  conversationId,
  strip,
}: {
  turn: ChatTurn;
  conversationId: string;
  /** Removes the boilerplate this corpus repeats on every memory. */
  strip: (s: string) => string;
}) {
  const select = useChat((s) => s.select);
  const groups = turn.ops.filter(
    (op): op is Extract<SeatEvent, { type: "memory_recall" | "proactive_context" }> =>
      (op.type === "memory_recall" && op.scope === "user") || op.type === "proactive_context",
  );
  const rows = groups.flatMap((op) =>
    op.type === "memory_recall"
      ? op.memories.map((memory) => ({
          id: memory.id,
          content: memory.experience.content,
          score: memory.score,
          label: "recall",
        }))
      : op.memories.map((memory) => ({
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
        {groups.some((op) => op.type === "memory_recall") ? (
          <span className="normal-case">
            {" — “"}
            {(groups.find((op) => op.type === "memory_recall") as Extract<SeatEvent, { type: "memory_recall" }>).query}
            {"”"}
          </span>
        ) : null}
      </h3>
      <div className="mt-2 flex flex-col gap-0.5">
        {rows.map((row) => (
          <button
            key={`${turn.turn}-${row.id}-${row.label}`}
            type="button"
            onClick={() => select({ conversationId, turn: turn.turn, memoryId: row.id })}
            className="hover:bg-accent/60 focus-visible:ring-ring flex items-start gap-2 rounded px-1.5 py-1 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none"
          >
            {/* The fusion score, and it carries real signal — measured across
                150 rows it spans 5% to 95% with a median of 59% and 51 distinct
                widths. It was drawn in muted-foreground/60 on a border-coloured
                track, which reads as a flat dash on the paper ground and made
                the whole column look like decoration. Ranked evidence whose
                ranking you cannot see is half the claim. */}
            <span
              className="bg-border mt-[7px] h-[3px] w-10 shrink-0 overflow-hidden rounded-full"
              title={`fusion score ${row.score.toFixed(3)}`}
            >
              <span
                className="bg-foreground/70 block h-full rounded-full"
                style={{ width: `${Math.max(2, Math.min(100, row.score * 100))}%` }}
              />
            </span>
            <span className="text-foreground/90 line-clamp-2 min-w-0 flex-1 text-[11px] leading-relaxed">
              {strip(row.content.replace(/\s+/g, " ").trim())}
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
  const strip = useEvidenceStripper(convo?.turns ?? []);

  const resolved = useMemo(() => {
    if (!selected || !convo || selected.conversationId !== conversationId) return null;
    return resolveSelection(convo.turns, selected.turn, selected.memoryId);
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

  const turnsWithEvidence = useMemo(
    () =>
      (convo?.turns ?? [])
        .filter((turn) =>
          turn.ops.some(
            (op) =>
              (op.type === "memory_recall" && op.scope === "user") ||
              (op.type === "proactive_context" && op.memories.length > 0),
          ),
        )
        .slice()
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
          <MemoryDetail resolved={resolved} conversationId={selected.conversationId} turn={selected.turn} />
        ) : turnsWithEvidence.length > 0 ? (
          turnsWithEvidence.map((turn) => (
            <TurnDigest
              key={turn.turn}
              turn={turn}
              conversationId={conversationId ?? ""}
              strip={strip}
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
