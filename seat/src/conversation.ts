/**
 * A conversation: one pi Agent wired to shodh-memory, with two learning loops
 * closing at every turn.
 *
 * Loop 1 — memory-level (user scope), two legs with strict ownership:
 *
 * - Implicit/momentum leg: each turn calls POST /api/proactive_context — the
 *   only backend path that writes feedback momentum (feedback_multiplier).
 *   It evaluates the previous turn's proactive-surfaced set against the
 *   previous response, the current user message (followup), and the previous
 *   run's tool actions; it also applies its own reinforce_recall + Hebbian
 *   pass internally (src/handlers/recall.rs:1670-1720). Memories surfaced by
 *   this channel are OWNED by it.
 * - Explicit leg: memories recalled by the recall_memory tool (and not also
 *   proactive-surfaced) are reinforced through POST /api/reinforce according
 *   to citation or token overlap (mirroring src/memory/feedback.rs), with
 *   negative-followup penalties for the previous turn. This leg moves
 *   importance/Hebbian but NOT momentum — a backend seam, not a seat choice.
 *
 * Loop 2 — harness-level (harness scope): operational lessons about retrieval
 * and tool use are stored AS MEMORIES in an isolated namespace
 * (`<user_id>.seat-harness`), surfaced by the same recall machinery before
 * each turn, injected as a labeled system-prompt block, and reinforced by the
 * same rules. One substrate, two scopes; the scopes never share retrieval
 * because the backend keys every store (RocksDB, graph, feedback) by user_id
 * (src/handlers/state.rs get_user_memory / get_user_graph).
 *
 * Every update either loop makes is recorded in the LearningLedger before the
 * conversation continues — reviewable and revertible from the start.
 */

import * as crypto from "node:crypto";
import { Agent, type AgentEvent, type AgentMessage, type AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { RecallMemory, ReinforceOutcome, ShodhBackend, ToolAction } from "./backend.js";
import type { MemoryScope, ModelRef, ReinforceTrigger, SeatEvent, SeatEventSink, UsagePayload } from "./events.js";
import {
	detectNegativeKeywords,
	extractCitations,
	extractTokens,
	memoryOverlap,
	OVERLAP_USED_THRESHOLD,
} from "./feedback.js";
import type { LearningLedger } from "./ledger.js";
import { MEMORY_GUIDANCE } from "./memory-guidance.js";
import { createMemoryTools } from "./memory-tools.js";
import type { ModelRegistry } from "./models-registry.js";

const HARNESS_SUFFIX = ".seat-harness";
/** Backend limit: src/validation.rs MAX_USER_ID_LENGTH = 128. */
const MAX_USER_ID_LENGTH = 128;
const USER_ID_PATTERN = /^[A-Za-z0-9@._-]+$/;
/** Minimum normalized recall score for a harness learning to be injected. */
const HARNESS_INJECT_MIN_SCORE = 0.25;
const HARNESS_INJECT_LIMIT = 3;
/** Caps on automatic harness captures, per conversation. */
const MAX_EMPTY_RECALL_CAPTURES = 5;
const MAX_TOOL_ERROR_CAPTURES = 5;

/** How many proactive memories to surface and inject per turn. Kept equal so
 * the backend's pending-feedback set contains only memories the model actually
 * saw — otherwise the implicit loop penalizes memories that were never shown. */
const PROACTIVE_MAX_RESULTS = 3;

/**
 * Memory-behaviour mechanisms, each measured by seat/eval/memory-guidance-ab.mjs.
 * All default ON — this is the ship configuration. Per-mechanism switches exist
 * for exactly one reason: an A/B evaluation needs a control arm reproducing the
 * pre-mechanism behaviour, and factored arms need single-mechanism attribution
 * (same rationale as ConversationOptions.harnessLearning).
 */
export interface MemoryMechanisms {
	/** Append MEMORY_GUIDANCE (memory-guidance.ts) to the base system prompt. */
	guidance: boolean;
	/**
	 * Frame the proactive block as a partial sample of a larger store instead
	 * of a bare list. Measured failure it targets: the model treated the
	 * 3-memory injected block as the whole of memory (recall_memory called 0
	 * times across 63 baseline cases) and declared facts "not recorded"
	 * without ever searching.
	 */
	proactiveFraming: boolean;
	/** Proactive memories surfaced+injected per turn (owner hypothesis: 3 vs 5). */
	proactiveMax: number;
	/**
	 * Render causal lineage edges in recall_memory results. The backend
	 * returns them on every recall; before this mechanism they were dropped
	 * before the model saw any structure (measured: chain-tracing cases
	 * failed 16/16 with the chain present in the store).
	 */
	recallLineage: boolean;
	/**
	 * Deterministic post-draft verification with one bounded revision pass:
	 * malformed/unknown/unsupported [mem:] citations and absence claims made
	 * without having searched are fed back to the model once.
	 */
	verifyLoop: boolean;
	/**
	 * Exclude bridged MCP tools that duplicate the seat's native memory
	 * surface (recall/remember/proactive_context/quick_recall). Measured
	 * defects of the duplicates: relevance-percentage output the citation
	 * contract cannot parse (models cited "[mem:95%]"), writes that bypass
	 * the ledger, recalls that bypass the reinforcement loop, and MCP
	 * proactive_context auto-ingesting conversation fragments as junk
	 * memories mid-conversation.
	 */
	mcpMemoryToolFilter: boolean;
}

export const DEFAULT_MEMORY_MECHANISMS: MemoryMechanisms = {
	guidance: true,
	proactiveFraming: true,
	proactiveMax: PROACTIVE_MAX_RESULTS,
	recallLineage: true,
	verifyLoop: true,
	mcpMemoryToolFilter: true,
};

/**
 * MCP tool base names (suffix after mcp__<server>__) that duplicate native
 * seat memory tools. Everything else — graph/lineage/entity/todo/fact tools —
 * stays bridged: none of those write memories or auto-ingest (verified against
 * mcp-server/index.ts: proactive_context is the only auto_ingest tool;
 * remember is the only general memory write).
 */
const REDUNDANT_MCP_MEMORY_TOOLS = new Set(["recall", "quick_recall", "proactive_context", "remember"]);

/**
 * Drop the bridged MCP tools that duplicate native memory ops.
 *
 * `deps.mcpTools` is a *function* re-read every turn, so this has to be applied
 * at every call site rather than once at construction — a server that
 * reconnects or changes its tool list mid-conversation would otherwise
 * re-introduce exactly the tools this filter exists to remove.
 */
function withoutRedundantMcpTools(tools: AgentTool<any>[], enabled: boolean): AgentTool<any>[] {
	if (!enabled) return tools;
	return tools.filter((tool) => {
		const baseName = tool.name.startsWith("mcp__") ? tool.name.split("__").slice(2).join("__") : tool.name;
		return !REDUNDANT_MCP_MEMORY_TOOLS.has(baseName);
	});
}

/** Hard cap on verification passes per turn: one revision, never a loop. */
const MAX_VERIFY_PASSES = 1;

/** Citation-shaped tokens that are NOT the contract's [mem:<8 hex>] form
 *  (e.g. "[mem:95%]", "[mem:<full-uuid>]"). */
const MALFORMED_CITATION_RE = /\[mem:(?![0-9a-fA-F]{8}\])[^\]]{1,80}\]/g;

/** Conservative floor for the misattribution check: fire only when a cited
 *  memory's content shares essentially no tokens with the answer — the
 *  measured failure shape is citing an unrelated id for content that came
 *  from a different memory entirely. OVERLAP_USED_THRESHOLD (0.1) is NOT
 *  reused here: it calibrates "was this memory used", the opposite question. */
const MISATTRIBUTION_OVERLAP_FLOOR = 0.05;

/** Deterministic detector for "this is absent from memory" claims. */
const ABSENCE_CLAIM_RE =
	/\b(?:no|any)\s+(?:specific\s+)?(?:records?|memor(?:y|ies)|mentions?|information|evidence)\b|\bnot\s+(?:recorded|captured|included|specified|mentioned)\b|\bisn't\s+(?:recorded|captured|included|mentioned)\b|\bdon't\s+have\s+(?:any\s+)?(?:records?|memor|information)|\bnothing\s+in\s+(?:my\s+)?memor/i;
/** Matches the thresholds the existing callers use (mcp-server/index.ts, hooks/memory-hook.ts). */
const PROACTIVE_SEMANTIC_THRESHOLD = 0.6;

/** Native memory tools are excluded from tool-usage attribution: their inputs
 * (the recall cue) trivially overlap surfaced memory content, which would turn
 * the act of recalling into a fake "usage" signal. */
const MEMORY_TOOL_NAMES = new Set(["recall_memory", "remember_memory", "record_seat_learning"]);

/**
 * The backend keeps ONE pending-feedback slot per user_id (set_pending
 * overwrites, take_pending consumes — src/memory/feedback.rs). Concurrent
 * proactive calls for the same user would corrupt each other's feedback, so
 * feedback fields are skipped when another call for that user is in flight —
 * the same guard mcp-server/index.ts uses (proactiveCallInFlight). This
 * protects seat-internal concurrency only; a separate process (e.g. a Claude
 * Code session on the same user_id) cannot be guarded from here.
 */
const proactiveFeedbackInFlight = new Set<string>();

const BASE_SYSTEM_PROMPT = `You are the shodh-memory conversation seat: an assistant whose persistent memory is visible and inspectable by the user.

Memory discipline:
- Use recall_memory when the user refers to past work, decisions, people, or preferences, or when prior context would materially improve the answer.
- When a recalled memory informs your answer, cite it inline as [mem:<id>] using the id shown in the recall result.
- Use remember_memory sparingly: durable facts, decisions, and learnings only.
- Use record_seat_learning only for operational lessons about retrieval or tool strategy — never for user content.`;

export class ConversationBusyError extends Error {
	constructor() {
		super("Conversation is currently processing a message");
		this.name = "ConversationBusyError";
	}
}

export class UnknownModelError extends Error {
	constructor(provider: string, id: string) {
		super(`Unknown or unavailable model: ${provider}/${id}`);
		this.name = "UnknownModelError";
	}
}

export interface ConversationDeps {
	backend: ShodhBackend;
	registry: ModelRegistry;
	ledger: LearningLedger;
	/**
	 * The bridged MCP tools that are reachable RIGHT NOW, read fresh rather
	 * than handed over once.
	 *
	 * A function and not an array because MCP servers come and go underneath a
	 * long-lived conversation: one is reconnected from the workbench, another
	 * announces a changed tool list, a third dies. A snapshot taken when the
	 * conversation was constructed would keep offering the model tools that
	 * cannot be called, and would hide tools that appeared since — and a
	 * conversation here can outlive several such changes, because it is
	 * rehydrated from the store rather than recreated per request.
	 */
	mcpTools: () => AgentTool<any>[];
}

export interface ConversationOptions {
	userId: string;
	model: Model<Api>;
	systemPrompt?: string;
	/**
	 * Harness-level continuous learning (loop 2): lesson retrieval/injection
	 * before each turn and automatic lesson capture after it. Defaults ON —
	 * disabling exists for exactly one reason: an A/B evaluation needs a
	 * control arm that differs in nothing else. Loop 1 (user-memory
	 * reinforcement) is NOT affected by this switch; it is the product's
	 * substrate, not a treatment under test.
	 */
	harnessLearning?: boolean;
	/**
	 * Per-mechanism overrides of DEFAULT_MEMORY_MECHANISMS. Absent fields keep
	 * their (ON) defaults; like harnessLearning, overrides exist for A/B
	 * evaluation arms and are not persisted across restarts.
	 */
	memoryMechanisms?: Partial<MemoryMechanisms>;
	/**
	 * Rehydration state for a conversation reopened from the store: identity,
	 * transcript, and the turn counter continue exactly where they stopped.
	 * `lastAssistantText` re-arms the momentum leg so the first message after a
	 * restart still delivers previous-response feedback to proactive_context.
	 */
	restore?: {
		id: string;
		createdAt: Date;
		turn: number;
		messages: unknown[];
		lastAssistantText?: string;
	};
}

interface SurfacedMemory {
	scope: MemoryScope;
	content: string;
}

export function deriveHarnessUserId(userId: string): string {
	const derived = `${userId}${HARNESS_SUFFIX}`;
	if (!USER_ID_PATTERN.test(userId) || userId.includes("..") || userId.startsWith(".")) {
		throw new Error(`Invalid user_id "${userId}" (allowed: alphanumeric, -, _, @, .)`);
	}
	if (derived.length > MAX_USER_ID_LENGTH) {
		throw new Error(`user_id too long: harness namespace "${derived}" exceeds ${MAX_USER_ID_LENGTH} chars`);
	}
	return derived;
}

function modelRef(model: Model<Api>): ModelRef {
	return { provider: model.provider, id: model.id, name: model.name };
}

function usagePayload(usage: {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	reasoning?: number;
	totalTokens: number;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
}): UsagePayload {
	return {
		input: usage.input,
		output: usage.output,
		cacheRead: usage.cacheRead,
		cacheWrite: usage.cacheWrite,
		reasoning: usage.reasoning,
		totalTokens: usage.totalTokens,
		cost: { ...usage.cost },
	};
}

function memoryShortId(memoryId: string): string {
	return memoryId.replace(/-/g, "").slice(0, 8).toLowerCase();
}

export class Conversation {
	readonly id: string;
	readonly userId: string;
	readonly harnessUserId: string;
	readonly harnessLearning: boolean;
	readonly mechanisms: MemoryMechanisms;
	readonly createdAt: Date;

	private readonly deps: ConversationDeps;
	private readonly agent: Agent;
	private readonly baseSystemPrompt: string;
	/** The native memory tools, kept so the tool list can be rebuilt around a
	 *  changed set of MCP tools without recreating them (they close over this
	 *  conversation's ids and event sink). */
	private readonly memoryTools: AgentTool<any>[];

	private turn = 0;
	private currentSink?: SeatEventSink;
	/** Events raised outside an active run (e.g. model change), flushed on the next run. */
	private pendingEvents: SeatEvent[] = [];

	/** Memories surfaced during the current run, keyed by memory id. */
	private surfaced = new Map<string, SurfacedMemory>();
	/** Memories surfaced during the previous run (target of negative-followup penalties). */
	private prevSurfaced = new Map<string, SurfacedMemory>();
	/** Ids surfaced by proactive_context this run — owned by the backend's
	 * implicit-feedback loop, excluded from the seat's explicit reinforcement. */
	private proactiveIds = new Set<string>();
	/** Content of proactively injected memories this run (verify-loop misattribution check). */
	private proactiveContents = new Map<string, string>();
	/** Memory ids written this run (remember/seat-learning) — citable, never "unknown". */
	private writtenIds = new Set<string>();
	/** User-scope recall_memory calls this run (verify-loop absence check). */
	private userRecallCount = 0;
	/** Previous run's proactive ids — excluded from explicit negative-followup
	 * penalties because the implicit loop applies its own followup penalty. */
	private prevProactiveIds = new Set<string>();
	/** Final assistant text of the previous run — previous_response for the next proactive call. */
	private lastAssistantText: string | undefined;
	/** Tool actions since the last consumed pending set (feedback attribution window). */
	private pendingToolActions: ToolAction[] = [];
	/** Args captured at tool_execution_start (the end event does not carry them). */
	private toolArgsByCallId = new Map<string, unknown>();
	private weakRecalls: { query: string; resultCount: number; bestFinalScore: number }[] = [];
	private toolErrors: { toolName: string; message: string }[] = [];
	private assistantTexts: string[] = [];
	private lastStopReason = "stop";
	private lastErrorMessage: string | undefined;

	/** Per-conversation dedupe for automatic harness captures. */
	private capturedEmptyRecalls = new Set<string>();
	private capturedToolErrors = new Set<string>();

	constructor(deps: ConversationDeps, options: ConversationOptions) {
		this.id = options.restore?.id ?? crypto.randomUUID();
		this.userId = options.userId;
		this.harnessLearning = options.harnessLearning ?? true;
		this.harnessUserId = deriveHarnessUserId(options.userId);
		this.createdAt = options.restore?.createdAt ?? new Date();
		this.deps = deps;
		if (options.restore) {
			this.turn = options.restore.turn;
			this.lastAssistantText = options.restore.lastAssistantText;
		}
		this.mechanisms = { ...DEFAULT_MEMORY_MECHANISMS, ...options.memoryMechanisms };
		const promptBlocks = [BASE_SYSTEM_PROMPT];
		if (this.mechanisms.guidance) promptBlocks.push(MEMORY_GUIDANCE);
		if (options.systemPrompt?.trim()) promptBlocks.push(options.systemPrompt.trim());
		this.baseSystemPrompt = promptBlocks.join("\n\n");

		this.memoryTools = createMemoryTools({
			backend: deps.backend,
			userId: this.userId,
			harnessUserId: this.harnessUserId,
			conversationId: this.id,
			getTurn: () => this.turn,
			emit: (event) => this.emit(event),
			onSurfaced: (scope, memories) => {
				for (const memory of memories) {
					this.surfaced.set(memory.id, { scope, content: memory.content });
				}
			},
			onWeakRecall: (query, resultCount, bestFinalScore) => {
				this.weakRecalls.push({ query, resultCount, bestFinalScore });
			},
			ledger: deps.ledger,
			renderLineage: this.mechanisms.recallLineage,
		});

		this.agent = new Agent({
			initialState: {
				systemPrompt: this.baseSystemPrompt,
				model: options.model,
				thinkingLevel: "off",
				tools: [
					...this.memoryTools,
					...withoutRedundantMcpTools(deps.mcpTools(), this.mechanisms.mcpMemoryToolFilter),
				],
				// Restored transcripts were produced by this same agent and
				// persisted verbatim (store.ts) — the cast re-labels what the
				// agent itself serialized.
				messages: (options.restore?.messages as AgentMessage[] | undefined) ?? [],
			},
			streamFn: (model, context, streamOptions) => deps.registry.models.streamSimple(model, context, streamOptions),
		});
		this.agent.subscribe((event) => this.onAgentEvent(event));
	}

	get model(): ModelRef {
		return modelRef(this.agent.state.model);
	}

	get isStreaming(): boolean {
		return this.agent.state.isStreaming;
	}

	/** Completed turn count — the turn number the NEXT message will get is this + 1. */
	get turnCount(): number {
		return this.turn;
	}

	private emit(event: SeatEvent): void {
		// Verify-loop bookkeeping, kept here so every emitter (native tools,
		// proactive pass) feeds it without additional plumbing.
		if (event.type === "memory_write") this.writtenIds.add(event.memory_id);
		if (event.type === "memory_recall" && event.scope === "user") this.userRecallCount += 1;
		if (this.currentSink) {
			this.currentSink(event);
		} else {
			this.pendingEvents.push(event);
		}
	}

	private onAgentEvent(event: AgentEvent): void {
		switch (event.type) {
			case "message_update": {
				const streamEvent = event.assistantMessageEvent;
				if (streamEvent.type === "text_delta") {
					this.emit({ type: "text_delta", delta: streamEvent.delta });
				} else if (streamEvent.type === "thinking_delta") {
					this.emit({ type: "thinking_delta", delta: streamEvent.delta });
				}
				break;
			}
			case "message_end": {
				const message = event.message;
				if (typeof message === "object" && message !== null && "role" in message && message.role === "assistant") {
					this.lastStopReason = message.stopReason;
					this.lastErrorMessage = message.errorMessage;
					const text = message.content
						.filter((block): block is { type: "text"; text: string } => block.type === "text")
						.map((block) => block.text)
						.join("");
					if (text) this.assistantTexts.push(text);
					this.emit({ type: "usage", model: this.model, usage: usagePayload(message.usage) });
				}
				break;
			}
			case "tool_execution_start":
				this.emit({
					type: "tool_call_start",
					tool_call_id: event.toolCallId,
					tool_name: event.toolName,
					args: event.args,
				});
				// tool_execution_end does not carry args; keep them for attribution.
				this.toolArgsByCallId.set(event.toolCallId, event.args);
				break;
			case "tool_execution_end": {
				this.emit({
					type: "tool_call_end",
					tool_call_id: event.toolCallId,
					tool_name: event.toolName,
					is_error: event.isError,
				});
				if (event.isError) {
					const message =
						typeof event.result === "string" ? event.result : JSON.stringify(event.result)?.slice(0, 500);
					this.toolErrors.push({ toolName: event.toolName, message: message ?? "unknown error" });
				}
				const args = this.toolArgsByCallId.get(event.toolCallId);
				this.toolArgsByCallId.delete(event.toolCallId);
				this.recordToolAction(event.toolName, args, event.result, event.isError);
				break;
			}
			default:
				break;
		}
	}

	/**
	 * Run one user message through the agent, streaming SeatEvents to `sink`.
	 * Resolves after the run AND the learning loops have completed.
	 */
	async sendMessage(text: string, sink: SeatEventSink): Promise<void> {
		if (this.agent.state.isStreaming || this.currentSink) throw new ConversationBusyError();
		this.currentSink = sink;
		this.turn += 1;

		// Re-read the bridged MCP tools for this turn. `AgentState.tools` is a
		// settable accessor that copies the array it is given
		// (pi-agent-core dist/types.d.ts), so this is the supported way to
		// change what the agent can reach between runs — and it is the only
		// place a server that reconnected, dropped, or changed its tool list
		// since the last turn actually takes effect.
		this.agent.state.tools = [
			...this.memoryTools,
			...withoutRedundantMcpTools(this.deps.mcpTools(), this.mechanisms.mcpMemoryToolFilter),
		];

		// Reset per-run state.
		this.surfaced = new Map();
		this.prevProactiveIds = this.proactiveIds;
		this.proactiveIds = new Set();
		this.proactiveContents = new Map();
		this.writtenIds = new Set();
		this.userRecallCount = 0;
		this.weakRecalls = [];
		this.toolErrors = [];
		this.assistantTexts = [];
		this.lastStopReason = "stop";
		this.lastErrorMessage = undefined;

		try {
			for (const pending of this.pendingEvents) sink(pending);
			this.pendingEvents = [];

			this.emit({ type: "turn_start", turn: this.turn });

			await this.applyNegativeFollowupPenalty(text);
			const proactiveBlock = await this.runProactivePass(text);
			const harnessBlock = this.harnessLearning
				? await this.buildHarnessLearningsBlock(text)
				: undefined;
			this.agent.state.systemPrompt = [this.baseSystemPrompt, proactiveBlock, harnessBlock]
				.filter((block): block is string => Boolean(block))
				.join("\n\n");

			await this.agent.prompt(text);

			if (this.mechanisms.verifyLoop) await this.runVerificationPass();

			await this.closeLearningLoops();
			this.lastAssistantText = this.assistantTexts.join("\n") || undefined;

			this.emit({
				type: "turn_end",
				turn: this.turn,
				stop_reason: this.lastStopReason,
				error_message: this.lastErrorMessage,
			});
			this.emit({ type: "agent_end" });
		} finally {
			this.prevSurfaced = this.surfaced;
			this.currentSink = undefined;
		}
	}

	/** Map a finished tool call into the backend's ToolAction shape for feedback attribution. */
	private recordToolAction(toolName: string, args: unknown, result: unknown, isError: boolean): void {
		if (MEMORY_TOOL_NAMES.has(toolName)) return;
		const inputs: Record<string, string> = {};
		if (args && typeof args === "object") {
			for (const [key, value] of Object.entries(args as Record<string, unknown>)) {
				inputs[key] = (typeof value === "string" ? value : JSON.stringify(value) ?? "").slice(0, 500);
			}
		}
		let outputSnippet: string | undefined;
		if (result && typeof result === "object" && "content" in result) {
			const content = (result as { content?: unknown }).content;
			if (Array.isArray(content)) {
				outputSnippet = content
					.filter(
						(block): block is { type: "text"; text: string } =>
							typeof block === "object" &&
							block !== null &&
							(block as { type?: string }).type === "text" &&
							typeof (block as { text?: unknown }).text === "string",
					)
					.map((block) => block.text)
					.join(" ")
					.slice(0, 200);
			}
		} else if (typeof result === "string") {
			outputSnippet = result.slice(0, 200);
		}
		this.pendingToolActions.push({
			tool_name: toolName,
			inputs,
			success: !isError,
			...(outputSnippet ? { output_snippet: outputSnippet } : {}),
		});
	}

	/**
	 * The momentum leg of the memory-level loop. POST /api/proactive_context is
	 * the only backend path that writes feedback momentum (feedback_multiplier);
	 * /api/reinforce does not. This call:
	 *
	 * 1. Delivers the previous assistant response (+ this user message as the
	 *    followup, + tool actions from the previous run) so the backend
	 *    evaluates the pending surfaced set: momentum, context fingerprints,
	 *    temporal credits, AND its own reinforce_recall/Hebbian pass
	 *    (recall.rs:1670-1720).
	 * 2. Surfaces a new set for this turn, which the backend stores as pending.
	 *    Every surfaced memory is injected into the system prompt — the pending
	 *    set must only contain memories the model actually saw, or the implicit
	 *    loop penalizes memories that never had a chance to be used.
	 *
	 * Ownership rule (prevents double-counting): memories surfaced here belong
	 * to the implicit loop; the seat's explicit /api/reinforce never touches
	 * them (closeLearningLoops and the negative-followup pass filter them out).
	 *
	 * auto_ingest is explicitly false: the backend would otherwise silently
	 * ingest the previous response as memories (its default is true), bypassing
	 * the ledger. Seat writes stay deliberate and ledgered.
	 */
	private async runProactivePass(userText: string): Promise<string | undefined> {
		const feedbackAllowed = !proactiveFeedbackInFlight.has(this.userId);
		if (feedbackAllowed) proactiveFeedbackInFlight.add(this.userId);
		const sendFeedback = feedbackAllowed && this.lastAssistantText !== undefined;
		const toolActions = sendFeedback ? this.pendingToolActions.splice(0, this.pendingToolActions.length) : [];

		try {
			const startedAt = Date.now();
			const response = await this.deps.backend.proactiveContext({
				userId: this.userId,
				context: userText,
				maxResults: this.mechanisms.proactiveMax,
				semanticThreshold: PROACTIVE_SEMANTIC_THRESHOLD,
				autoIngest: false,
				previousResponse: sendFeedback ? this.lastAssistantText : undefined,
				userFollowup: sendFeedback ? userText : undefined,
				toolActions,
			});

			for (const memory of response.memories) {
				this.proactiveIds.add(memory.id);
				this.proactiveContents.set(memory.id, memory.content);
			}

			// The implicit leg just applied real learning updates server-side
			// (reinforce_recall + Hebbian strengthening, recall.rs:1670-1720) and
			// reported exactly what moved. Record it, or the ledger's claim that
			// every learning update is reviewable is false for precisely the
			// conversations where the proactive channel owns all surfaced
			// memories — in which case ALL loop-1 learning is implicit and the
			// ledger would stay empty. Found by the lessons A/B eval.
			const fb = response.feedback_processed;
			if (fb && (fb.reinforced.length > 0 || fb.weakened.length > 0)) {
				await this.deps.ledger.append({
					kind: "implicit_feedback",
					// The backend's momentum pass: neither the human nor the model
					// asked for it, it runs on every proactive round-trip.
					actor: "system",
					scope: "user",
					userId: this.userId,
					conversationId: this.id,
					turn: this.turn,
					data: {
						memories_evaluated: fb.memories_evaluated,
						reinforced: fb.reinforced,
						weakened: fb.weakened,
					},
				});
			}

			const lines = response.memories.map(
				(memory) =>
					`- [mem:${memoryShortId(memory.id)}] (${memory.memory_type}) ${memory.content.slice(0, 400)}`,
			);
			let block: string | undefined;
			if (response.memories.length > 0) {
				block = this.mechanisms.proactiveFraming
					? // Sample framing: the measured failure mode is the model treating
						// this block as the whole of memory — answering "not recorded" or
						// stopping a chain because the next link was not in these few lines.
						`## Memory sample (auto-surfaced — cite [mem:id] if used)\n` +
						`These are only the ${response.memories.length} closest matches to the current message; the persistent store holds far more, and details relevant to the question may not be shown here. ` +
						`Search it with recall_memory before concluding anything is missing, and before answering questions whose evidence these lines do not fully cover.\n` +
						lines.join("\n")
					: `## Possibly relevant memories (auto-surfaced — cite [mem:id] if used)\n${lines.join("\n")}`;
			}

			this.emit({
				type: "proactive_context",
				scope: "user",
				query: userText,
				memories: response.memories,
				injected_memory_ids: response.memories.map((memory) => memory.id),
				// The block verbatim: what the model was actually shown must be
				// inspectable, not reconstructable.
				injected_block: block ?? null,
				feedback: response.feedback_processed ?? null,
				temporal_credits_applied: response.temporal_credits_applied ?? null,
				took_ms: Date.now() - startedAt,
			});

			return block;
		} catch (error) {
			// Momentum loop is an enhancement; its failure must not block the turn.
			// Un-drained tool actions stay queued for the next attempt.
			if (toolActions.length > 0) this.pendingToolActions.unshift(...toolActions);
			this.emit({
				type: "error",
				message: `Proactive context failed: ${error instanceof Error ? error.message : String(error)}`,
			});
			return undefined;
		} finally {
			if (feedbackAllowed) proactiveFeedbackInFlight.delete(this.userId);
		}
	}

	/**
	 * Deterministic answer verification with ONE bounded revision pass.
	 *
	 * Memory answers are checkable without any judge model: every [mem:] token
	 * must be the 8-hex contract form; every cited id must have actually been
	 * shown to the model this run (proactive block, recall result, or its own
	 * write); a cited memory must share at least minimal vocabulary with the
	 * answer (misattribution check, conservative floor); and a claim that
	 * something is absent from memory is only creditable after memory was
	 * actually searched. When any check fires, the specific findings are fed
	 * back to the model exactly once and it revises; the `verification` event
	 * records what fired so evaluations can attribute revisions.
	 */
	private async runVerificationPass(): Promise<void> {
		for (let pass = 0; pass < MAX_VERIFY_PASSES; pass += 1) {
			const issues = this.verifyDraft();
			if (issues.length === 0) return;
			this.emit({ type: "verification", issues, nudged: true });
			const nudge =
				`[automated answer verification — not the user] Issues detected in your draft answer:\n` +
				issues.map((issue) => `- ${issue}`).join("\n") +
				`\nRevise now: cite memories as [mem:<8-hex id>] exactly as shown in recall results or the surfaced-memories block, citing only memories that support the sentence they follow. ` +
				`If you stated that something is not in memory, first search with recall_memory using concrete alternative terms (names, identifiers, short forms); then either use what you find or confirm the absence. ` +
				`Reply with the corrected, complete answer.`;
			await this.agent.prompt(nudge);
		}
	}

	/** Pure checks over this run's draft answer and memory events. */
	private verifyDraft(): string[] {
		const answer = this.assistantTexts.join("\n");
		if (!answer.trim()) return [];
		const issues: string[] = [];

		const malformed = answer.match(MALFORMED_CITATION_RE) ?? [];
		if (malformed.length > 0) {
			issues.push(
				`Malformed citation token(s) ${[...new Set(malformed)].slice(0, 4).join(", ")} — the contract is [mem:<8 hex chars>]; for a long id use its first 8 characters.`,
			);
		}

		const knownShort = new Map<string, string>(); // shortId -> content ("" when unknown)
		for (const [memoryId, memory] of this.surfaced) knownShort.set(memoryShortId(memoryId), memory.content);
		for (const [memoryId, content] of this.proactiveContents) knownShort.set(memoryShortId(memoryId), content);
		for (const memoryId of this.writtenIds) knownShort.set(memoryShortId(memoryId), "");

		const cited = extractCitations(answer);
		const unknown = [...cited].filter((shortId) => !knownShort.has(shortId));
		if (unknown.length > 0) {
			issues.push(
				`Cited memory id(s) ${unknown.map((shortId) => `[mem:${shortId}]`).join(", ")} were never surfaced in this conversation — cite only ids shown to you, or search for the memory first.`,
			);
		}

		const answerTokens = extractTokens(answer);
		for (const shortId of cited) {
			const content = knownShort.get(shortId);
			if (!content) continue; // unknown handled above; written ids have no content here
			if (memoryOverlap(content, answerTokens) < MISATTRIBUTION_OVERLAP_FLOOR) {
				issues.push(
					`[mem:${shortId}] does not contain the content it is cited for — re-check which surfaced memory actually supports that sentence.`,
				);
			}
		}

		if (this.userRecallCount === 0 && ABSENCE_CLAIM_RE.test(answer)) {
			issues.push(
				`The answer claims something is not in memory, but memory was never searched this turn — only the auto-surfaced sample was consulted.`,
			);
		}

		return issues;
	}

	/**
	 * Mirror of the backend's user_followup penalty (src/memory/feedback.rs):
	 * a correction/frustration message penalizes the memories surfaced on the
	 * previous turn.
	 */
	private async applyNegativeFollowupPenalty(userText: string): Promise<void> {
		if (this.prevSurfaced.size === 0) return;
		const keywords = detectNegativeKeywords(userText);
		if (keywords.length === 0) return;

		const byScope = new Map<MemoryScope, string[]>();
		for (const [memoryId, memory] of this.prevSurfaced) {
			// Ownership: memories the proactive channel surfaced last turn get
			// their followup penalty from the backend's implicit loop (this
			// turn's proactive call carries user_followup) — penalizing them
			// here too would double-count.
			if (this.prevProactiveIds.has(memoryId)) continue;
			const ids = byScope.get(memory.scope) ?? [];
			ids.push(memoryId);
			byScope.set(memory.scope, ids);
		}
		for (const [scope, ids] of byScope) {
			await this.reinforceAndRecord(scope, ids, "misleading", {
				kind: "negative_followup",
				keywords,
			});
		}
	}

	/**
	 * Harness-level learning, read side: recall operational lessons from the
	 * harness scope with the user message as cue and return strong matches as
	 * a labeled system-prompt block for this run only.
	 */
	private async buildHarnessLearningsBlock(userText: string): Promise<string | undefined> {
		let memories: RecallMemory[] = [];
		try {
			const startedAt = Date.now();
			const response = await this.deps.backend.recall({
				userId: this.harnessUserId,
				query: userText,
				limit: HARNESS_INJECT_LIMIT,
				mode: "hybrid",
				debug: true,
			});
			memories = response.memories.filter((memory) => memory.score >= HARNESS_INJECT_MIN_SCORE);
			if (memories.length > 0) {
				this.emit({
					type: "memory_recall",
					scope: "harness",
					query: userText,
					mode: "hybrid",
					memories,
					facts: [],
					todos: [],
					lineage: [],
					took_ms: Date.now() - startedAt,
				});
			}
		} catch (error) {
			// Harness recall is an enhancement; its failure must not block the turn.
			this.emit({
				type: "error",
				message: `Harness-scope recall failed: ${error instanceof Error ? error.message : String(error)}`,
			});
		}

		if (memories.length === 0) return undefined;

		for (const memory of memories) {
			this.surfaced.set(memory.id, { scope: "harness", content: memory.experience.content });
		}
		const lines = memories.map((memory) => `- ${memory.experience.content}`);
		this.emit({
			type: "harness_learning_applied",
			memories: memories.map((memory) => ({
				id: memory.id,
				content: memory.experience.content,
				score: memory.score,
			})),
		});
		return `## Learned operating notes (from previous sessions of this assistant)\n${lines.join("\n")}`;
	}

	/** Reinforce + ledger + emit, with failures surfaced as error events. */
	private async reinforceAndRecord(
		scope: MemoryScope,
		memoryIds: string[],
		outcome: ReinforceOutcome,
		trigger: ReinforceTrigger,
	): Promise<void> {
		if (memoryIds.length === 0) return;
		const scopeUserId = scope === "user" ? this.userId : this.harnessUserId;
		try {
			const stats = await this.deps.backend.reinforce(scopeUserId, memoryIds, outcome);
			const ledgerEntry = await this.deps.ledger.append({
				kind: "reinforce",
				// Every trigger that reaches here (citation, response_overlap,
				// negative_followup, revert) is computed by the seat's own
				// deterministic loop, not requested by the model or the human.
				actor: "system",
				scope,
				userId: scopeUserId,
				conversationId: this.id,
				turn: this.turn,
				data: { outcome, memory_ids: memoryIds, trigger, stats },
			});
			this.emit({
				type: "memory_reinforce",
				scope,
				outcome,
				memory_ids: memoryIds,
				stats,
				trigger,
				ledger_event_id: ledgerEntry.id,
			});
		} catch (error) {
			this.emit({
				type: "error",
				message: `Reinforcement (${outcome}) failed for ${scope} scope: ${
					error instanceof Error ? error.message : String(error)
				}`,
			});
		}
	}

	/**
	 * Close both learning loops for the finished run:
	 * 1. Reinforce surfaced memories by usage (citation or token overlap).
	 * 2. Capture deterministic harness learnings (empty recalls, tool errors).
	 */
	private async closeLearningLoops(): Promise<void> {
		const responseText = this.assistantTexts.join("\n");
		if (this.surfaced.size > 0 && responseText.length > 0) {
			const responseTokens = extractTokens(responseText);
			const citations = extractCitations(responseText);

			const groups = new Map<
				string,
				{ scope: MemoryScope; outcome: ReinforceOutcome; ids: string[]; overlaps: Record<string, number>; cited: string[] }
			>();
			for (const [memoryId, memory] of this.surfaced) {
				// Ownership: memories the proactive channel surfaced this turn
				// are evaluated by the backend's implicit loop on the NEXT
				// proactive call (which itself applies reinforce_recall +
				// Hebbian strengthening, recall.rs:1670-1720). Reinforcing them
				// here too would double importance and association updates.
				if (this.proactiveIds.has(memoryId)) continue;
				const cited = citations.has(memoryShortId(memoryId));
				const overlap = memoryOverlap(memory.content, responseTokens);
				const outcome: ReinforceOutcome = cited || overlap >= OVERLAP_USED_THRESHOLD ? "helpful" : "neutral";
				const key = `${memory.scope}:${outcome}`;
				const group =
					groups.get(key) ??
					({ scope: memory.scope, outcome, ids: [], overlaps: {}, cited: [] } as {
						scope: MemoryScope;
						outcome: ReinforceOutcome;
						ids: string[];
						overlaps: Record<string, number>;
						cited: string[];
					});
				group.ids.push(memoryId);
				group.overlaps[memoryId] = Number(overlap.toFixed(4));
				if (cited) group.cited.push(memoryId);
				groups.set(key, group);
			}

			for (const group of groups.values()) {
				const trigger =
					group.cited.length > 0
						? ({ kind: "citation", cited: group.cited } as const)
						: ({ kind: "response_overlap", overlaps: group.overlaps, threshold: OVERLAP_USED_THRESHOLD } as const);
				await this.reinforceAndRecord(group.scope, group.ids, group.outcome, trigger);
			}
		}

		if (this.harnessLearning) await this.captureHarnessLearnings();
	}

	/** Deterministic write side of the harness loop, with per-conversation dedupe and caps. */
	private async captureHarnessLearnings(): Promise<void> {
		for (const { query, resultCount, bestFinalScore } of this.weakRecalls) {
			if (this.capturedEmptyRecalls.size >= MAX_EMPTY_RECALL_CAPTURES) break;
			const normalized = query.trim().toLowerCase();
			if (this.capturedEmptyRecalls.has(normalized)) continue;
			this.capturedEmptyRecalls.add(normalized);
			await this.writeHarnessCapture(
				`Recall found nothing useful for cue "${query.slice(0, 200)}" (${resultCount} results, best fusion score ${bestFinalScore.toFixed(3)}). Rephrase with concrete entity names or broaden the cue before answering without memory.`,
				"learning",
				["seat-harness", "retrieval", "empty-recall"],
				"empty_recall_capture",
			);
		}

		for (const toolError of this.toolErrors) {
			if (this.capturedToolErrors.size >= MAX_TOOL_ERROR_CAPTURES) break;
			if (this.capturedToolErrors.has(toolError.toolName)) continue;
			this.capturedToolErrors.add(toolError.toolName);
			await this.writeHarnessCapture(
				`Tool ${toolError.toolName} failed: ${toolError.message.slice(0, 300)}. Verify arguments and tool availability before relying on it.`,
				"error",
				["seat-harness", "tool-error", toolError.toolName],
				"tool_error_capture",
			);
		}
	}

	private async writeHarnessCapture(
		content: string,
		memoryType: "learning" | "error",
		tags: string[],
		trigger: "empty_recall_capture" | "tool_error_capture",
	): Promise<void> {
		try {
			const response = await this.deps.backend.remember({
				userId: this.harnessUserId,
				content,
				memoryType,
				tags,
			});
			const ledgerEntry = await this.deps.ledger.append({
				kind: "memory_write",
				// Deterministic capture rule (empty recall / tool error) fired by
				// the seat itself — the model never chose to write this.
				actor: "system",
				scope: "harness",
				userId: this.harnessUserId,
				conversationId: this.id,
				turn: this.turn,
				data: {
					memory_id: response.id,
					memory_type: memoryType,
					content_preview: content.slice(0, 200),
					trigger,
				},
			});
			this.emit({
				type: "memory_write",
				scope: "harness",
				memory_id: response.id,
				memory_type: memoryType,
				content_preview: content.slice(0, 200),
				ledger_event_id: ledgerEntry.id,
			});
		} catch (error) {
			this.emit({
				type: "error",
				message: `Harness learning capture failed: ${error instanceof Error ? error.message : String(error)}`,
			});
		}
	}

	/**
	 * Swap the model for future turns. The transcript and all retrieved
	 * evidence stay exactly as they are — only the model producing the next
	 * turn changes.
	 */
	setModel(provider: string, id: string): ModelRef {
		if (this.agent.state.isStreaming) throw new ConversationBusyError();
		const model = this.deps.registry.resolve(provider, id);
		if (!model) throw new UnknownModelError(provider, id);
		this.agent.state.model = model;
		const ref = modelRef(model);
		this.emit({ type: "model_changed", model: ref });
		return ref;
	}

	abort(): void {
		this.agent.abort();
	}

	transcript(): unknown[] {
		return this.agent.state.messages.map((message) => message as unknown);
	}
}
