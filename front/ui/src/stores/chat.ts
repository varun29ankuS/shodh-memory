import { create } from "zustand";
import { streamMessage } from "@/lib/seat/client";
import type {
  ConversationDetail,
  ModelRef,
  PiAssistantMessage,
  SeatEvent,
  UsageTotals,
} from "@/lib/seat/types";

/**
 * Live conversation state.
 *
 * The seat streams SeatEvents over SSE; this store is the single reducer for
 * them, and the same shapes are rebuilt from the persisted transcript +
 * durable events when a conversation is reopened (`buildTurns`). One turn =
 * one user message = one `turn_start..turn_end` window on the wire — that
 * invariant comes from the seat itself (Conversation.sendMessage increments
 * `turn` exactly once per user message), which is what makes reconstruction
 * by position sound.
 *
 * Memory operations are kept as first-class entries (`ops`), not folded into
 * text: the evidence surface renders from them directly, live and replayed.
 */

/** The durable, inline-renderable subset of SeatEvent. */
export type ChatOp = Extract<
  SeatEvent,
  {
    type:
      | "memory_recall"
      | "proactive_context"
      | "memory_write"
      | "memory_reinforce"
      | "harness_learning_applied"
      | "tool_call_start"
      | "tool_call_end"
      | "view_command"
      // A QUESTION, NOT EVIDENCE, and it is here anyway. `app/useAgentView.ts`
      // reads the live turn's `ops` and nothing else — that is the only channel
      // between the wire and the view bus — so a probe kept out of this list is
      // a probe that reaches no one, and `inspect_view` would time out over an
      // answer this browser was perfectly able to give. It renders as nothing:
      // MessageList's switch falls through to null and the evidence panel
      // filters for the two retrieval types.
      | "view_probe"
      | "model_changed"
      | "error";
  }
>;

export interface TurnUsage {
  model: ModelRef;
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning: number;
  totalTokens: number;
  cost: number;
  /** Model calls in this turn (tool-use turns make several). */
  calls: number;
}

export interface ChatTurn {
  turn: number;
  userText: string;
  ops: ChatOp[];
  assistantText: string;
  thinkingText: string;
  usage: TurnUsage | null;
  stopReason?: string;
  errorMessage?: string;
  /** Streaming right now (only ever true on the newest turn). */
  pending: boolean;
}

export interface ConvoLive {
  turns: ChatTurn[];
  streaming: boolean;
  model: ModelRef | null;
  /** Accumulated token/cost totals: seeded from the store row, advanced live. */
  totals: UsageTotals;
  /** Stream failed outside the event protocol (connection dropped, 409…). */
  transportError: string | null;
}

/**
 * What the evidence panel is focused on.
 *
 * ADDRESSED BY POSITION, NOT BY DISPLAYED TURN NUMBER, and by the OP inside
 * that turn rather than by the turn alone. Both halves were real defects.
 *
 * `turnIndex` is the index into `ConvoLive.turns`. `ChatTurn.turn` is the
 * number the seat assigns (`this.turn += 1` per sent message, restored from
 * persistence), and `applyEvent`'s `turn_start` writes it onto whichever turn
 * is last — so it is a LABEL, and nothing guarantees it equals position + 1.
 * The resolver indexed the array with it anyway, which is a wrong-turn lookup
 * the moment the two disagree, and the same number was a React key, which is a
 * duplicate key at the same moment.
 *
 * `opIndex` is the index into `ChatTurn.ops`. A single turn routinely runs
 * SEVERAL recalls — five in one measured turn — and they overlap: on the live
 * `claude-code` seat, memory `a8ca63ff` came back from two searches in one turn
 * at scores 0.0816 and 0.95. Identifying evidence by `(turn, memoryId)` cannot
 * tell those two apart, so the resolver returned whichever op scanned first and
 * the panel showed the wrong search's score attribution under the right
 * memory's text.
 */
export interface EvidenceSelection {
  conversationId: string;
  /** Index into `ConvoLive.turns` — never `ChatTurn.turn`. */
  turnIndex: number;
  /** Index into `ChatTurn.ops` — which recall produced this row. */
  opIndex: number;
  memoryId: string;
}

interface ChatState {
  activeId: string | null;
  conversations: Record<string, ConvoLive>;
  selected: EvidenceSelection | null;
  /** Ledger events reverted this session (server refuses double reverts). */
  revertedLedgerIds: Record<string, true>;
  evidenceOpen: boolean;
  sessionsOpen: boolean;

  setActive: (id: string | null) => void;
  adoptDetail: (detail: ConversationDetail) => void;
  send: (conversationId: string, text: string, onSettled?: () => void) => Promise<void>;
  setModel: (conversationId: string, model: ModelRef) => void;
  select: (selection: EvidenceSelection | null) => void;
  markReverted: (ledgerEventId: string) => void;
  toggleEvidence: () => void;
  toggleSessions: () => void;
  forget: (conversationId: string) => void;
}

const EMPTY_TOTALS: UsageTotals = {
  input: 0,
  output: 0,
  cache_read: 0,
  cache_write: 0,
  reasoning: 0,
  total_tokens: 0,
  cost_total: 0,
};

function newTurn(turn: number, userText: string, pending: boolean): ChatTurn {
  return {
    turn,
    userText,
    ops: [],
    assistantText: "",
    thinkingText: "",
    usage: null,
    pending,
  };
}

function addUsage(target: TurnUsage | null, model: ModelRef, usage: SeatEvent & { type: "usage" }): TurnUsage {
  const base: TurnUsage = target ?? {
    model,
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    reasoning: 0,
    totalTokens: 0,
    cost: 0,
    calls: 0,
  };
  return {
    ...base,
    model: usage.model,
    input: base.input + usage.usage.input,
    output: base.output + usage.usage.output,
    cacheRead: base.cacheRead + usage.usage.cacheRead,
    cacheWrite: base.cacheWrite + usage.usage.cacheWrite,
    reasoning: base.reasoning + (usage.usage.reasoning ?? 0),
    totalTokens: base.totalTokens + usage.usage.totalTokens,
    cost: base.cost + usage.usage.cost.total,
    calls: base.calls + 1,
  };
}

/** Fold one wire event into a conversation. Pure; returns a new ConvoLive. */
export function applyEvent(convo: ConvoLive, event: SeatEvent): ConvoLive {
  const turns = convo.turns.slice();
  const last = turns[turns.length - 1];
  const replaceLast = (next: ChatTurn): ConvoLive => {
    turns[turns.length - 1] = next;
    return { ...convo, turns };
  };

  switch (event.type) {
    case "turn_start":
      if (last) return replaceLast({ ...last, turn: event.turn });
      return convo;
    case "text_delta":
      if (!last) return convo;
      return replaceLast({ ...last, assistantText: last.assistantText + event.delta });
    case "thinking_delta":
      if (!last) return convo;
      return replaceLast({ ...last, thinkingText: last.thinkingText + event.delta });
    case "usage": {
      if (!last) return convo;
      const totals: UsageTotals = {
        input: convo.totals.input + event.usage.input,
        output: convo.totals.output + event.usage.output,
        cache_read: convo.totals.cache_read + event.usage.cacheRead,
        cache_write: convo.totals.cache_write + event.usage.cacheWrite,
        reasoning: convo.totals.reasoning + (event.usage.reasoning ?? 0),
        total_tokens: convo.totals.total_tokens + event.usage.totalTokens,
        cost_total: convo.totals.cost_total + event.usage.cost.total,
      };
      return { ...replaceLast({ ...last, usage: addUsage(last.usage, event.model, event) }), totals };
    }
    case "turn_end":
      if (!last) return convo;
      return replaceLast({
        ...last,
        stopReason: event.stop_reason,
        errorMessage: event.error_message,
      });
    case "model_changed": {
      const next = last ? replaceLast({ ...last, ops: [...last.ops, event] }) : { ...convo, turns };
      return { ...next, model: event.model };
    }
    // AN OUTCOME IS NOT EVIDENCE AND NOT NEWS. It is what this browser itself
    // decided; it reaches the store only as a durable row on reload, and
    // rendering it in the evidence panel would show the reader their own click
    // played back as something the model did. The trail is where it belongs.
    //
    // A `view_probe` is deliberately NOT in this group. It is a question rather
    // than evidence, but it has to travel to `app/useAgentView.ts`, which reads
    // the live turn's `ops` and nothing else — see `ChatOp`.
    case "view_outcome":
    // NOR IS THE SEAT TELLING THE MODEL WHAT THIS BROWSER ALREADY KNOWS. A
    // `view_outcome_relayed` says a verdict finally reached the model one turn
    // late; every verdict in it was produced HERE, by this store, from the
    // person's own hand. Rendering it would play their click back to them as an
    // event, captioned with what we said about it. It stays durable and
    // inspectable on the conversation record, where a reader asking "did the
    // assistant know?" can find it.
    case "view_outcome_relayed":
    case "agent_end":
    case "conversation_created":
      return convo;
    default:
      // Every remaining member of the union is a ChatOp.
      if (!last) return convo;
      return replaceLast({ ...last, ops: [...last.ops, event] });
  }
}

/**
 * Rebuild turns from a persisted conversation: user/assistant text from the
 * transcript (the authority for final text), ops/usage/stop reasons from the
 * durable events, joined on the turn number.
 */
export function buildTurns(detail: ConversationDetail): ChatTurn[] {
  const turns: ChatTurn[] = [];
  let current: ChatTurn | null = null;

  for (const message of detail.messages) {
    if (message.role === "user") {
      const text =
        typeof message.content === "string"
          ? message.content
          : message.content
              .filter((block): block is { type: "text"; text: string } => block.type === "text")
              .map((block) => (block as { text: string }).text)
              .join("");
      current = newTurn(turns.length + 1, text, false);
      turns.push(current);
    } else if (message.role === "assistant" && current) {
      const assistant = message as PiAssistantMessage;
      for (const block of assistant.content) {
        if (block.type === "text") current.assistantText += block.text;
        else if (block.type === "thinking") current.thinkingText += block.thinking;
      }
      if (assistant.errorMessage) current.errorMessage = assistant.errorMessage;
    }
    // toolResult messages carry nothing the ops don't already say.
  }

  for (const stored of detail.events) {
    const turn = turns[stored.turn - 1];
    if (!turn) continue;
    const event = stored.event;
    switch (event.type) {
      case "usage":
        turn.usage = addUsage(turn.usage, event.model, event);
        break;
      case "turn_end":
        turn.stopReason = event.stop_reason;
        if (event.error_message) turn.errorMessage = event.error_message;
        break;
      case "turn_start":
      case "agent_end":
      case "conversation_created":
      case "text_delta":
      case "thinking_delta":
      // See `applyEvent`: a reloaded conversation replays its stored outcomes,
      // and they belong in the audit trail rather than in the evidence panel.
      case "view_probe":
      case "view_outcome":
      case "view_outcome_relayed":
        break;
      default:
        turn.ops.push(event);
    }
  }

  return turns;
}

/** Abort handles for in-flight streams, outside the store (not renderable state). */
const inflight = new Map<string, AbortController>();

export const useChat = create<ChatState>((set, get) => ({
  activeId: null,
  conversations: {},
  selected: null,
  revertedLedgerIds: {},
  evidenceOpen: true,
  sessionsOpen: true,

  setActive: (id) => set({ activeId: id, selected: null }),

  adoptDetail: (detail) =>
    set((s) => {
      const existing = s.conversations[detail.conversation_id];
      // Never clobber a conversation that is streaming right now — the live
      // reducer state is ahead of anything the server has persisted.
      if (existing?.streaming) return s;
      return {
        conversations: {
          ...s.conversations,
          [detail.conversation_id]: {
            turns: buildTurns(detail),
            streaming: false,
            model: detail.model,
            totals: detail.usage,
            transportError: null,
          },
        },
      };
    }),

  send: async (conversationId, text, onSettled) => {
    const state = get();
    const convo = state.conversations[conversationId];
    if (!convo || convo.streaming) return;

    set((s) => {
      const target = s.conversations[conversationId];
      if (!target) return s;
      return {
        conversations: {
          ...s.conversations,
          [conversationId]: {
            ...target,
            streaming: true,
            transportError: null,
            turns: [...target.turns, newTurn(target.turns.length + 1, text, true)],
          },
        },
      };
    });

    const controller = new AbortController();
    inflight.set(conversationId, controller);

    // Deltas can arrive faster than a sane render cadence; coalesce them and
    // flush on a short timer so one long completion does not schedule
    // hundreds of store updates per second.
    let queue: SeatEvent[] = [];
    let flushTimer: number | undefined;
    const flush = () => {
      flushTimer = undefined;
      if (queue.length === 0) return;
      const batch = queue;
      queue = [];
      set((s) => {
        let target = s.conversations[conversationId];
        if (!target) return s;
        for (const event of batch) target = applyEvent(target, event);
        return { conversations: { ...s.conversations, [conversationId]: target } };
      });
    };
    const onEvent = (event: SeatEvent) => {
      queue.push(event);
      if (event.type === "text_delta" || event.type === "thinking_delta") {
        flushTimer ??= window.setTimeout(flush, 40);
      } else {
        if (flushTimer !== undefined) window.clearTimeout(flushTimer);
        flush();
      }
    };

    try {
      await streamMessage(conversationId, text, onEvent, controller.signal);
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        const message = error instanceof Error ? error.message : String(error);
        set((s) => {
          const target = s.conversations[conversationId];
          if (!target) return s;
          return {
            conversations: {
              ...s.conversations,
              [conversationId]: { ...target, transportError: message },
            },
          };
        });
      }
    } finally {
      inflight.delete(conversationId);
      if (flushTimer !== undefined) window.clearTimeout(flushTimer);
      flush();
      set((s) => {
        const target = s.conversations[conversationId];
        if (!target) return s;
        const turns = target.turns.slice();
        const last = turns[turns.length - 1];
        if (last) turns[turns.length - 1] = { ...last, pending: false };
        return {
          conversations: {
            ...s.conversations,
            [conversationId]: { ...target, streaming: false, turns },
          },
        };
      });
      onSettled?.();
    }
  },

  setModel: (conversationId, model) =>
    set((s) => {
      const target = s.conversations[conversationId];
      if (!target) return s;
      return {
        conversations: { ...s.conversations, [conversationId]: { ...target, model } },
      };
    }),

  select: (selected) => set({ selected, evidenceOpen: selected ? true : get().evidenceOpen }),

  markReverted: (ledgerEventId) =>
    set((s) => ({ revertedLedgerIds: { ...s.revertedLedgerIds, [ledgerEventId]: true } })),

  toggleEvidence: () => set((s) => ({ evidenceOpen: !s.evidenceOpen })),
  toggleSessions: () => set((s) => ({ sessionsOpen: !s.sessionsOpen })),

  forget: (conversationId) =>
    set((s) => {
      inflight.get(conversationId)?.abort();
      inflight.delete(conversationId);
      const conversations = { ...s.conversations };
      delete conversations[conversationId];
      return {
        conversations,
        activeId: s.activeId === conversationId ? null : s.activeId,
        selected: s.selected?.conversationId === conversationId ? null : s.selected,
      };
    }),
}));

export { EMPTY_TOTALS };
