import { useEffect, useMemo, useRef } from "react";
import { cn } from "@/lib/utils";
import type { ModelRef } from "@/lib/seat/types";
import { formatCost, formatTokens } from "@/lib/format";
import { type ChatTurn, useChat } from "@/stores/chat";
import { Markdown } from "./Markdown";
import { costIsReal, useBillingLookup } from "./useBilling";
import {
  ErrorBlock,
  HarnessAppliedBlock,
  ModelChangedBlock,
  ProactiveBlock,
  RecallBlock,
  ReinforceBlock,
  ToolBlock,
  WriteBlock,
} from "./OpBlocks";

/**
 * The transcript. Memory operations render inline between the user message
 * and the reply, in arrival order — which is also the honest order: recall
 * completes BEFORE the model starts, so during the seconds a local model
 * spends thinking, the evidence is already on screen. The waiting state
 * therefore never shows a bare spinner; it shows what was retrieved and names
 * the model being waited on, because that is what is actually happening.
 */

/** seat/src/conversation.ts memoryShortId — first 8 hex chars, dashes stripped. */
function shortId(memoryId: string): string {
  return memoryId.replace(/-/g, "").slice(0, 8).toLowerCase();
}

function UsageFooter({ turn }: { turn: ChatTurn }) {
  // Per-turn, per-model: a conversation can mix a metered turn with a local
  // one after a mid-conversation swap, and each row must say what ITS tokens
  // meant — that is the whole point of storing the model ref on the usage.
  const lookupModel = useBillingLookup(true);
  if (!turn.usage) return null;
  const cost = costIsReal(lookupModel(turn.usage.model)) ? formatCost(turn.usage.cost) : null;
  return (
    <p className="text-muted-foreground/70 mono text-[10px]">
      {turn.usage.model.name}
      {" · "}
      {formatTokens(turn.usage.input)} in / {formatTokens(turn.usage.output)} out
      {turn.usage.cacheRead > 0 ? ` / ${formatTokens(turn.usage.cacheRead)} cached` : ""}
      {turn.usage.reasoning > 0 ? ` / ${formatTokens(turn.usage.reasoning)} reasoning` : ""}
      {cost ? ` · ${cost}` : ""}
      {turn.usage.calls > 1 ? ` · ${turn.usage.calls} model calls` : ""}
    </p>
  );
}

type CitationMap = Map<string, { id: string; content: string; memory_type?: string }>;

/**
 * Citation targets for the whole conversation.
 *
 * Scoped to the conversation, not the turn, and that is the whole point: a
 * model cites a memory it saw three turns ago as readily as one from this
 * recall. Measured on a live conversation before this changed — 77 citations,
 * 0 resolved against the turn-scoped map, 77 against this one. The fallback to
 * a raw id was defensible as an edge case and was in fact the norm.
 *
 * Carries the memory itself, not just its id, because the chip shows what the
 * memory SAYS. `mem:4a59ea4b` is the seat/model protocol and means nothing to
 * a reader.
 *
 * The two ops carry different shapes — proactive_context surfaces memories
 * flat, memory_recall nests them under `experience`. One map either way,
 * because a citation does not care which op put the memory in front of it.
 */
function useCitationMap(turns: ChatTurn[]): CitationMap {
  return useMemo(() => {
    const map: CitationMap = new Map();
    for (const turn of turns) {
      for (const op of turn.ops) {
        if (op.type === "memory_recall" || op.type === "proactive_context") {
          for (const memory of op.memories) {
            const proactive = "content" in memory;
            map.set(shortId(memory.id), {
              id: memory.id,
              content: proactive ? memory.content : memory.experience.content,
              memory_type: (proactive ? memory.memory_type : memory.experience.memory_type) ?? undefined,
            });
          }
        }
      }
    }
    return map;
  }, [turns]);
}

function Turn({
  turn,
  conversationId,
  model,
  citationMap,
}: {
  citationMap: CitationMap;
  turn: ChatTurn;
  conversationId: string;
  model: ModelRef | null;
}) {
  const select = useChat((s) => s.select);

  const waiting = turn.pending && turn.assistantText.length === 0;

  return (
    <article aria-label={`Turn ${turn.turn}`} className="flex flex-col gap-2.5">
      <div className="flex justify-end">
        <div className="bg-secondary max-w-[85%] rounded-lg rounded-br-sm px-3.5 py-2.5">
          <p className="text-[13px] leading-relaxed whitespace-pre-wrap">{turn.userText}</p>
        </div>
      </div>

      {turn.ops.length > 0 ? (
        <div className="flex max-w-[92%] flex-col gap-1.5">
          {turn.ops.map((op, index) => {
            const key = `${turn.turn}-${index}`;
            switch (op.type) {
              case "memory_recall":
                return <RecallBlock key={key} op={op} turn={turn.turn} conversationId={conversationId} />;
              case "proactive_context":
                return <ProactiveBlock key={key} op={op} turn={turn.turn} conversationId={conversationId} />;
              case "memory_reinforce":
                return <ReinforceBlock key={key} op={op} />;
              case "memory_write":
                return <WriteBlock key={key} op={op} />;
              case "tool_call_start":
                return <ToolBlock key={key} op={op} ops={turn.ops} />;
              case "harness_learning_applied":
                return <HarnessAppliedBlock key={key} op={op} />;
              case "model_changed":
                return <ModelChangedBlock key={key} op={op} />;
              case "error":
                return <ErrorBlock key={key} op={op} />;
              default:
                return null;
            }
          })}
        </div>
      ) : null}

      {waiting ? (
        <p className="text-muted-foreground flex items-center gap-2 text-[12px]">
          <span aria-hidden="true" className="bg-primary size-1.5 animate-pulse rounded-full" />
          {model ? `Waiting on ${model.name}…` : "Waiting on the model…"}
        </p>
      ) : null}

      {turn.thinkingText ? (
        <details className="max-w-[92%]">
          <summary className="text-muted-foreground cursor-pointer text-[11px] select-none">
            Thinking
          </summary>
          <p className="text-muted-foreground mt-1 text-[12px] leading-relaxed whitespace-pre-wrap">
            {turn.thinkingText}
          </p>
        </details>
      ) : null}

      {turn.assistantText ? (
        <div className="max-w-[92%]">
          <Markdown
            text={turn.assistantText}
            cited={citationMap}
            onCitationClick={(cited) => {
              const hit = citationMap.get(cited.slice(0, 8));
              if (hit) select({ conversationId, turn: turn.turn, memoryId: hit.id });
            }}
          />
          {turn.pending ? (
            <span aria-hidden="true" className="bg-foreground/80 ml-0.5 inline-block h-3.5 w-[2px] animate-pulse align-text-bottom" />
          ) : null}
        </div>
      ) : null}

      {turn.errorMessage ? (
        <p className="text-destructive text-[12px] leading-relaxed">
          The model stopped with an error: {turn.errorMessage}
        </p>
      ) : null}

      {!turn.pending ? <UsageFooter turn={turn} /> : null}
    </article>
  );
}

export function MessageList({
  turns,
  conversationId,
  model,
}: {
  turns: ChatTurn[];
  conversationId: string;
  model: ModelRef | null;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const citationMap = useCitationMap(turns);

  // Follow the stream only while the reader is already at the bottom;
  // scrolling up to re-read must never be fought by the autoscroller.
  const last = turns[turns.length - 1];
  const followKey = `${turns.length}:${last?.assistantText.length ?? 0}:${last?.ops.length ?? 0}`;
  useEffect(() => {
    const node = scrollRef.current;
    if (node && pinnedRef.current) node.scrollTop = node.scrollHeight;
  }, [followKey]);

  return (
    <div
      ref={scrollRef}
      onScroll={(e) => {
        const node = e.currentTarget;
        pinnedRef.current = node.scrollHeight - node.scrollTop - node.clientHeight < 48;
      }}
      className="min-h-0 flex-1 overflow-y-auto"
    >
      <div className={cn("mx-auto flex w-full max-w-[760px] flex-col gap-6 px-4 py-5")}>
        {turns.map((turn) => (
          <Turn
            key={turn.turn}
            turn={turn}
            conversationId={conversationId}
            model={model}
            citationMap={citationMap}
          />
        ))}
      </div>
    </div>
  );
}
