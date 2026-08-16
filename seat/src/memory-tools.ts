/**
 * shodh-memory as first-class agent tools.
 *
 * These are native tools over the Rust backend's HTTP API (not MCP-framed
 * text): recall runs with debug:true so every result carries per-memory
 * ScoreAttribution, and every operation is emitted as a structured SeatEvent
 * the UI renders as its own element. Memory operations are never opaque.
 */

import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";
import type { MemoryType, RecallLineageEdge, RecallMemory, RecallMode, ShodhBackend } from "./backend.js";
import type { MemoryScope, SeatEvent } from "./events.js";
import type { LearningLedger } from "./ledger.js";

export interface MemoryToolContext {
	backend: ShodhBackend;
	/** The person's memory namespace. */
	userId: string;
	/** The harness's own isolated namespace (separate RocksDB + graph per user_id). */
	harnessUserId: string;
	conversationId: string;
	getTurn(): number;
	emit(event: SeatEvent): void;
	/** Register memories surfaced this turn so the turn-end loop can reinforce them. */
	onSurfaced(scope: MemoryScope, memories: { id: string; content: string }[]): void;
	/** A recall came back empty — candidate harness learning. */
	onWeakRecall(query: string, resultCount: number, bestFinalScore: number): void;
	ledger: LearningLedger;
	/** Render causal lineage edges in recall results (MemoryMechanisms.recallLineage). */
	renderLineage: boolean;
}

/** Absolute fusion-score floor under which a recall counts as a miss for
 *  lesson capture. See the miss-detection comment at the recall tool. */
const RECALL_MISS_FLOOR = 0.15;

const RECALL_MODES: RecallMode[] = [
	"hybrid",
	"semantic",
	"associative",
	"temporal",
	"causal",
	"spatial",
];

const recallParameters = Type.Object({
	query: Type.String({
		minLength: 1,
		maxLength: 2000,
		description: "Natural-language cue. Entity names and concrete terms retrieve better than abstractions.",
	}),
	limit: Type.Optional(
		Type.Integer({ minimum: 1, maximum: 20, description: "Maximum memories to retrieve (default 5)." }),
	),
	mode: Type.Optional(
		Type.Union(
			RECALL_MODES.map((mode) => Type.Literal(mode)),
			{ description: "Retrieval mode (default hybrid: vector + BM25 + graph fusion)." },
		),
	),
});

const rememberParameters = Type.Object({
	content: Type.String({ minLength: 3, maxLength: 10000, description: "The content to remember." }),
	memory_type: Type.Optional(
		Type.Union(
			(["observation", "decision", "learning", "error", "discovery", "pattern", "context", "task"] as const).map(
				(memoryType) => Type.Literal(memoryType),
			),
			{ description: "Memory type (default observation)." },
		),
	),
	tags: Type.Optional(Type.Array(Type.String({ minLength: 1, maxLength: 100 }), { maxItems: 10 })),
});

const seatLearningParameters = Type.Object({
	learning: Type.String({
		minLength: 10,
		maxLength: 2000,
		description: "The operational lesson, stated so it is actionable the next time the situation recurs.",
	}),
	kind: Type.Optional(
		Type.Union([Type.Literal("learning"), Type.Literal("pattern"), Type.Literal("error")], {
			description: "Lesson category (default learning).",
		}),
	),
	tags: Type.Optional(Type.Array(Type.String({ minLength: 1, maxLength: 100 }), { maxItems: 8 })),
});

function shortId(memoryId: string): string {
	return memoryId.replace(/-/g, "").slice(0, 8);
}

/**
 * Prose for a causal relation read from→to. The edge's `from` is always the
 * earlier memory (cause/origin/evidence), `to` the later one; InformedBy
 * therefore reads "from informed to", not the enum's to-perspective name —
 * same semantics as mcp-server/index.ts CAUSAL_RELATION_PROSE (post-#468).
 * RelatedTo is deliberately absent: co-occurrence edges are not causal
 * structure and would dilute the chain signal this block exists to carry.
 */
const LINEAGE_RELATION_PROSE: Record<string, string> = {
	Caused: "caused",
	ResolvedBy: "was resolved by",
	InformedBy: "informed",
	SupersededBy: "was superseded by",
	TriggeredBy: "triggered",
	BranchedFrom: "branched from",
};

/** Cap on rendered edges: every line is context on every recall. */
const LINEAGE_RENDER_CAP = 8;

/**
 * Render the causal edges among the returned memories.
 *
 * Within-results only, and deliberately so: /api/recall's lineage payload
 * contains exclusively edges whose BOTH endpoints are in the returned set
 * (verified empirically against a seeded store — every edge across 20+
 * recalls had both endpoints among the results), so there is no
 * outside-the-results information here to expand. Chain links that fall
 * outside the result set need the lineage trace endpoints; on the probed
 * corpus the binding constraint was the inferred graph itself (the
 * drift→strike edge was absent in 7 of 7 user graphs), which no seat-side
 * rendering can repair.
 */
function formatLineage(edges: RecallLineageEdge[], returned: RecallMemory[]): string[] {
	const returnedIds = new Set(returned.map((memory) => memory.id));
	const causal = edges.filter((edge) => LINEAGE_RELATION_PROSE[edge.relation] !== undefined);
	const inside = causal
		.filter((edge) => returnedIds.has(edge.from) && returnedIds.has(edge.to))
		.sort((a, b) => b.confidence - a.confidence)
		.slice(0, LINEAGE_RENDER_CAP);

	const lines: string[] = [];
	if (inside.length > 0) {
		lines.push("Causal links among these memories (earlier → later):");
		for (const edge of inside) {
			lines.push(`- [mem:${shortId(edge.from)}] ${LINEAGE_RELATION_PROSE[edge.relation]} [mem:${shortId(edge.to)}]`);
		}
	}
	return lines;
}

function formatMemoryForModel(memory: RecallMemory, index: number): string {
	const memoryType = memory.experience.memory_type ?? "Observation";
	const content =
		memory.experience.content.length > 600
			? `${memory.experience.content.slice(0, 600)}…`
			: memory.experience.content;
	return `${index + 1}. [mem:${shortId(memory.id)}] (${memoryType}, score ${memory.score.toFixed(2)}) ${content}`;
}

function textResult<T>(text: string, details: T): AgentToolResult<T> {
	return { content: [{ type: "text", text }], details };
}

export function createMemoryTools(context: MemoryToolContext): AgentTool<any>[] {
	const recallTool: AgentTool<typeof recallParameters> = {
		name: "recall_memory",
		label: "Recall memory",
		description:
			"Search the user's persistent memory (vector + BM25 + knowledge-graph fusion). " +
			"Returns memories with ids and scores, plus related facts and todos. " +
			"When a recalled memory informs your answer, cite it inline as [mem:<id>] using the id shown.",
		parameters: recallParameters,
		execute: async (toolCallId, params) => {
			const startedAt = Date.now();
			const mode = params.mode ?? "hybrid";
			const response = await context.backend.recall({
				userId: context.userId,
				query: params.query,
				limit: params.limit ?? 5,
				mode,
				debug: true,
			});
			const tookMs = Date.now() - startedAt;

			context.onSurfaced(
				"user",
				response.memories.map((memory) => ({ id: memory.id, content: memory.experience.content })),
			);
			context.emit({
				type: "memory_recall",
				scope: "user",
				tool_call_id: toolCallId,
				query: params.query,
				mode,
				memories: response.memories,
				facts: response.facts ?? [],
				todos: response.todos ?? [],
				lineage: response.lineage ?? [],
				took_ms: tookMs,
			});

			// A miss is "nothing USEFUL", not "literally nothing". Semantic recall
			// returns top-K for almost any cue once a corpus exists, so a
			// zero-length check alone never fires and the lesson-capture loop it
			// feeds goes dead — found by the lessons A/B eval, whose teach cues
			// all "hit" a one-memory corpus about something else entirely.
			// final_score is the pipeline's absolute fusion output (present
			// because recall runs debug:true); the normalized `score` cannot be
			// used here — it is relative to the top hit, so the top hit is always
			// ~0.95 no matter how bad it is.
			//
			// The floor is calibrated from one observed corpus (a true semantic
			// match measured 0.66; weak single-leg matches land an order of
			// magnitude lower) and is deliberately conservative. Recorded in the
			// capture text so future tuning has data.
			const bestFinal = response.memories.reduce(
				(best, memory) => Math.max(best, memory.score_attribution?.final_score ?? 0),
				0,
			);
			const miss = response.memories.length === 0 || bestFinal < RECALL_MISS_FLOOR;
			if (miss) {
				context.onWeakRecall(params.query, response.memories.length, bestFinal);
				if (response.memories.length === 0) {
					return textResult(
						"No memories matched this cue. Consider retrying with concrete entity names or a broader phrasing.",
						response,
					);
				}
			}

			const lines: string[] = [`Found ${response.memories.length} memories:`];
			response.memories.forEach((memory, index) => lines.push(formatMemoryForModel(memory, index)));
			if (context.renderLineage && response.lineage && response.lineage.length > 0) {
				lines.push(...formatLineage(response.lineage, response.memories));
			}
			if (response.facts && response.facts.length > 0) {
				lines.push("Related facts:");
				for (const fact of response.facts.slice(0, 5)) {
					lines.push(`- ${fact.fact} (confidence ${fact.confidence.toFixed(2)})`);
				}
			}
			if (response.todos && response.todos.length > 0) {
				lines.push("Related todos:");
				for (const todo of response.todos.slice(0, 5)) {
					lines.push(`- [${todo.status}] ${todo.content}`);
				}
			}
			return textResult(lines.join("\n"), response);
		},
	};

	const rememberTool: AgentTool<typeof rememberParameters> = {
		name: "remember_memory",
		label: "Remember",
		description:
			"Store a durable memory for the user. Use sparingly, for high-value facts, decisions, and learnings — " +
			"not for conversational filler.",
		parameters: rememberParameters,
		execute: async (_toolCallId, params) => {
			const memoryType = (params.memory_type ?? "observation") as MemoryType;
			const response = await context.backend.remember({
				userId: context.userId,
				content: params.content,
				memoryType,
				tags: params.tags ?? [],
			});
			const ledgerEntry = await context.ledger.append({
				kind: "memory_write",
				// The model emitted this tool call; the write is its decision.
				actor: "agent",
				scope: "user",
				userId: context.userId,
				conversationId: context.conversationId,
				turn: context.getTurn(),
				data: {
					memory_id: response.id,
					memory_type: memoryType,
					content_preview: params.content.slice(0, 200),
					trigger: "model_tool_call",
				},
			});
			context.emit({
				type: "memory_write",
				scope: "user",
				memory_id: response.id,
				memory_type: memoryType,
				content_preview: params.content.slice(0, 200),
				ledger_event_id: ledgerEntry.id,
			});
			return textResult(`Remembered as [mem:${shortId(response.id)}].`, { memory_id: response.id });
		},
	};

	const seatLearningTool: AgentTool<typeof seatLearningParameters> = {
		name: "record_seat_learning",
		label: "Record seat learning",
		description:
			"Record an operational lesson about how this assistant should retrieve, phrase cues, or use tools — " +
			"stored in the harness's own memory scope, never in the user's. " +
			"Never store user content or conversation facts here; use remember_memory for those.",
		parameters: seatLearningParameters,
		execute: async (_toolCallId, params) => {
			const memoryType = (params.kind ?? "learning") as MemoryType;
			const response = await context.backend.remember({
				userId: context.harnessUserId,
				content: params.learning,
				memoryType,
				tags: ["seat-harness", ...(params.tags ?? [])],
			});
			const ledgerEntry = await context.ledger.append({
				kind: "memory_write",
				// The model emitted this tool call; the write is its decision.
				actor: "agent",
				scope: "harness",
				userId: context.harnessUserId,
				conversationId: context.conversationId,
				turn: context.getTurn(),
				data: {
					memory_id: response.id,
					memory_type: memoryType,
					content_preview: params.learning.slice(0, 200),
					trigger: "model_tool_call",
				},
			});
			context.emit({
				type: "memory_write",
				scope: "harness",
				memory_id: response.id,
				memory_type: memoryType,
				content_preview: params.learning.slice(0, 200),
				ledger_event_id: ledgerEntry.id,
			});
			return textResult("Seat learning recorded.", { memory_id: response.id });
		},
	};

	return [recallTool, rememberTool, seatLearningTool];
}
