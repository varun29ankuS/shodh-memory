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
import type { MemoryScope, ModelRef, SeatEvent } from "./events.js";
import { agentAuthor } from "./events.js";
import type { LearningLedger } from "./ledger.js";
import { composeDeletionData, deletionNotPerformed } from "./ledger.js";
import { composeToolDescription } from "./tool-descriptions.js";

export interface MemoryToolContext {
	backend: ShodhBackend;
	/** The person's memory namespace. */
	userId: string;
	/** The harness's own isolated namespace (separate RocksDB + graph per user_id). */
	harnessUserId: string;
	conversationId: string;
	getTurn(): number;
	/** Read at call time, like the todo tools': `set_model` swaps the model
	 *  mid-conversation, and a deletion attributed to the model that opened the
	 *  conversation rather than the one that ordered it is a wrong signature. */
	getModel(): ModelRef;
	emit(event: SeatEvent): void;
	/** Register memories surfaced this turn so the turn-end loop can reinforce them. */
	onSurfaced(scope: MemoryScope, memories: { id: string; content: string }[]): void;
	/** A recall came back empty — candidate harness learning. */
	onWeakRecall(query: string, resultCount: number, bestFinalScore: number): void;
	ledger: LearningLedger;
	/** Render causal lineage edges in recall results (MemoryMechanisms.recallLineage). */
	renderLineage: boolean;
	/**
	 * The full uuid and scope behind an 8-character citation the model has been
	 * shown THIS RUN, or null.
	 *
	 * Supplied by the conversation for the same reason `create_todo`'s link
	 * resolver is — "has the model seen this" is not a question the backend can
	 * answer — and load-bearing for a different one: it is what keeps a
	 * model-supplied string out of a destructive request URL. See
	 * {@link forgetRefusal}.
	 */
	resolveShownMemory(shortId: string): { id: string; scope: MemoryScope } | null;
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

const forgetParameters = Type.Object({
	memory_id: Type.String({
		minLength: 8,
		maxLength: 20,
		description:
			"The memory to delete, exactly as you were shown it: [mem:1a2b3c4d] or the bare eight characters. Only " +
			"memories surfaced to you in this turn are accepted — a full uuid or an id from earlier in the " +
			"conversation is refused, so recall it and read it before deleting it.",
	}),
	why: Type.String({
		minLength: 10,
		maxLength: 1000,
		description:
			"Why this memory is being deleted, in your own words — normally the user's own reason, quoted. It is " +
			"written to the learning ledger and becomes the only surviving explanation of the deletion.",
	}),
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
 * A `[mem:xxxxxxxx]` citation, reduced to the eight characters that identify it.
 *
 * THE MODEL ONLY EVER HAS SHORT IDS. Every surface that shows it a memory —
 * recall results, the auto-surfaced block, its own writes — prints the first
 * eight hex characters of the uuid, because that is the citation contract.
 * Backend paths that take memory ids, by contrast, want full uuids (the
 * todo-to-memory link) or accept a hex prefix (`/api/memory/{id}`), so a tool
 * that took ids as typed would either reject every id the model can produce or
 * accept text the model invented.
 *
 * Both spellings are accepted because both are things the model has seen: the
 * bracketed citation is what it writes into prose, the bare eight characters are
 * what it reads out of a listing. Nothing else is — a full uuid included, since
 * the model is never shown one and an id in that shape is one it constructed.
 */
export function memoryCitationKey(raw: string): string | null {
	const trimmed = raw.trim();
	const bracketed = /^\[mem:([0-9a-fA-F]{8})\]$/.exec(trimmed);
	if (bracketed) return bracketed[1]!.toLowerCase();
	return /^[0-9a-fA-F]{8}$/.test(trimmed) ? trimmed.toLowerCase() : null;
}

/**
 * Why this seat will not delete the named memory, or null when it will.
 *
 * THE RESOLUTION IS THE SECURITY BOUNDARY, not a convenience. `resolved` is the
 * seat's own record of what it put in front of the model this run, so the id
 * that reaches the backend URL is a uuid this process already held — the model's
 * string is matched against a map and then discarded. The documented
 * path-traversal class in the MCP forget path cannot recur here, because no
 * model-supplied text is ever interpolated into the request at all.
 *
 * The harness refusal is a scope rule with the same shape as
 * `record_seat_learning`'s: the assistant's own lesson namespace is not the
 * user's corpus, a person asking to forget something means their own, and the
 * two are separate RocksDB stores that only this map can tell apart.
 */
export function forgetRefusal(
	typed: string,
	key: string | null,
	resolved: { id: string; scope: MemoryScope } | null,
): string | null {
	if (key === null) {
		return (
			`"${typed}" is not a memory id in the form this seat accepts. Pass the citation exactly as you were ` +
			"shown it — [mem:1a2b3c4d] or the bare eight characters — not a full uuid and not a description."
		);
	}
	if (resolved === null) {
		return (
			`[mem:${key}] is not a memory you have been shown in this turn, so this seat will not delete it. Recall ` +
			"it first and confirm from the result that it is the one meant: a deletion is permanent, and deleting an " +
			"id you have not read is destroying something you cannot describe afterwards."
		);
	}
	if (resolved.scope !== "user") {
		return (
			`[mem:${key}] is in the assistant's own learning scope, not the user's memory. This tool deletes the ` +
			"user's memories only; a lesson recorded by record_seat_learning is not theirs to lose and cannot be " +
			"removed from here."
		);
	}
	return null;
}

/**
 * What the model is told a deletion actually did.
 *
 * IT STATES THE SIDE EFFECTS THE BACKEND PERFORMS SILENTLY. `forget` removes the
 * memory from every tier, the vector index, the BM25 index AND the knowledge
 * graph episode with its sourced edges (src/memory/mod.rs) — so relations other
 * answers were drawing on can disappear with it. Child memories are the
 * opposite case and the sharper one: they are NOT deleted, they are left with a
 * `parent_id` that no longer resolves, and nothing anywhere reports that.
 *
 * The ledger id is returned because it is now the only handle on what was lost.
 */
export function composeForgetReport(input: {
	shortId: string;
	classification: string;
	preview: string;
	orphanedChildren: number;
	ledgerEventId: string;
}): string {
	const lines = [
		`Deleted [mem:${input.shortId}] (${input.classification}): "${input.preview}"`,
		"It is gone from the store, the vector and keyword indexes, and its graph episode — including the relations " +
			"that episode sourced, so answers that leaned on those links may no longer reach them.",
	];
	if (input.orphanedChildren > 0) {
		lines.push(
			`${input.orphanedChildren} child ${input.orphanedChildren === 1 ? "memory" : "memories"} still exist and ` +
				"now point at a parent that does not: they were NOT deleted with it. Tell the user, because nothing " +
				"else will.",
		);
	}
	lines.push(
		`Recorded in the learning ledger as ${input.ledgerEventId}, which holds who deleted it, why, a ` +
			"200-character preview and a checksum — not the memory. Nothing can restore it, so do not offer to undo.",
	);
	return lines.join("\n");
}

/**
 * What the model is told when the record was written and the deletion was not.
 *
 * THE ONE THING IT MUST NOT SAY IS "DELETED". The entry was appended first on
 * purpose (see the ordering note in ledger.ts), so this is the case where the
 * ledger briefly claims something that did not happen — and the model's job at
 * that moment is to know the memory is INTACT and to not tell the user it is
 * gone. Whether the correcting entry landed is reported separately, because it
 * is a second write that can fail on its own and a reviewer's read of the ledger
 * depends on which of the two states it is in.
 */
export function composeForgetFailureReport(input: {
	shortId: string;
	error: string;
	ledgerEventId: string;
	compensationError: string | null;
}): string {
	const lines = [
		`[mem:${input.shortId}] was NOT deleted: ${input.error}. The memory is intact and still recallable — do ` +
			"not tell the user it is gone.",
	];
	lines.push(
		input.compensationError === null
			? `Ledger event ${input.ledgerEventId} recorded the attempt and has been marked as not performed, so the ` +
					"record and the store agree again."
			: `WARNING: ledger event ${input.ledgerEventId} recorded a deletion that did not happen, and the ` +
					`correction ALSO failed (${input.compensationError}). The ledger currently overstates what was ` +
					"destroyed. Tell the user so a reviewer is not misled.",
	);
	return lines.join("\n");
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
		description: composeToolDescription("recall_memory", {
			does:
				"Searches the user's persistent memory over vector, BM25 and knowledge-graph fusion, and returns the " +
				"matching memories with their ids and scores alongside any facts and todos the same cue surfaced.",
			useWhen:
				"Use it whenever the user refers to past work, people, decisions or preferences, and before answering " +
				"any question whose evidence the auto-surfaced sample does not already fully cover. Above all, use it " +
				"before saying that something is NOT in memory — that sample is a handful of closest matches, not the " +
				"store, and an absence claim made without searching is a claim about a place you never looked.",
			notFor:
				"Do not use it for general knowledge the model already has, or to re-read something already quoted " +
				"earlier in this conversation. Do not use it to check whether an entity exists before naming it in " +
				"direct_view — that tool resolves every name against the graph itself and tells you which ones matched.",
			returns:
				"Each result carries an 8-character id to cite inline as [mem:<id>], a relevance score, and content " +
				"truncated to 600 characters; cite only ids shown to you. It searches the user's scope only — the " +
				"assistant's own learning scope is never returned here — and it does not return full memory text, " +
				"embeddings, or anything you have not been given an id for.",
		}),
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
		description: composeToolDescription("remember_memory", {
			does: "Stores one durable memory in the user's own memory scope, where every later recall can reach it.",
			useWhen:
				"Use it sparingly, when the conversation produces something worth having next month: a decision and " +
				"the reason behind it, a stable preference, a fact about the user's world, or a correction to something " +
				"previously believed.",
			notFor:
				"Do not use it for conversational filler, for restating something a recall already returned, or for " +
				"lessons about how to retrieve and use these tools — those belong in record_seat_learning, which writes " +
				"to a separate scope so they never come back as answers about the user's own corpus.",
			returns:
				"The new memory's short id, which you may cite immediately. The write is recorded in the learning " +
				"ledger under your model identity and appears in the user's history at once; nothing here can undo it, " +
				"so a memory written in error has to be reverted by the user from the workbench.",
		}),
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

	/**
	 * The destructive verb, made attributable.
	 *
	 * WHY IT HAD TO EXIST. The bridged MCP `forget` deletes a memory and writes
	 * nothing anywhere: no comment, no event this seat owns, and above all no
	 * ledger entry. Deletion is the one operation whose subject cannot be
	 * consulted afterwards, so an unrecorded one is unauditable by construction —
	 * and the ledger is the substrate this product's auditability claim rests on.
	 * A store that can prove what it learned but not what it destroyed is proving
	 * the easy half.
	 *
	 * IT SIGNS TO THE LEDGER, NOT TO THE OBJECT. Every other native mutation here
	 * signs the thing it touched (a memory write records its own id, a todo
	 * mutation writes an authored comment) because the thing survives to carry the
	 * signature. Nothing survives a deletion, so the record must outlive its
	 * subject, and the ledger is the only append-only store in this seat that
	 * does.
	 *
	 * THE ORDER IS DELIBERATE AND THE FAILURE MODE IS CHOSEN. Read, then record,
	 * then destroy. Reading first is the only chance to capture what is about to
	 * be lost and the only way to see the children that will be orphaned; if it
	 * fails, nothing has been touched. Recording before destroying means a failure
	 * leaves a ledger that overstates — an entry naming an object still sitting in
	 * the store, which a reviewer catches at a glance and a compensating entry
	 * corrects. The alternative, destroying first, leaves a deletion nothing
	 * records, which no reviewer can catch and no entry can repair.
	 */
	const forgetTool: AgentTool<typeof forgetParameters> = {
		name: "forget_memory",
		label: "Forget a memory",
		description: composeToolDescription("forget_memory", {
			does:
				"PERMANENTLY DELETES one memory from the user's store, together with its vector index entry, its " +
				"keyword index entry and its knowledge-graph episode and that episode's relations, and records the " +
				"deletion in the learning ledger under your model identity with your stated reason.",
			useWhen:
				"Use it only when the user asks for something to be forgotten or removed, and only for a memory you " +
				"have recalled and read in this turn, so that what you are destroying is something you have actually " +
				"seen. Say why in `why`: once the memory is gone, that sentence and a 200-character preview are the " +
				"only account of what was lost.",
			notFor:
				"NOTHING UNDOES THIS — there is no restore, the workbench's revert control refuses deletions, and the " +
				"ledger holds a preview and a checksum rather than the memory. Do not use it to correct a memory " +
				"(remember_memory the correction instead, so both the belief and its revision survive), to tidy up " +
				"memories you judge stale or duplicated, or on anything the user did not specifically ask you to " +
				"forget. If you are not certain the user means this memory, ask them rather than guessing.",
			returns:
				"Confirmation naming the memory, what it was, and the ledger event that now records the deletion. It " +
				"also reports child memories left ORPHANED — a memory's children are not deleted with it and are left " +
				"pointing at a parent that no longer exists, which nothing else surfaces. On failure it says plainly " +
				"that the memory is still there; never report a deletion the result did not confirm.",
		}),
		parameters: forgetParameters,
		execute: async (_toolCallId, params) => {
			const key = memoryCitationKey(params.memory_id);
			const resolved = key === null ? null : context.resolveShownMemory(key);
			const refusal = forgetRefusal(params.memory_id, key, resolved);
			if (refusal !== null || resolved === null) {
				throw new Error(refusal ?? `[mem:${key}] could not be resolved to a memory.`);
			}

			// Read before recording, and record before destroying. The read is also
			// the existence check: a memory another session already deleted fails
			// here, before an entry claiming otherwise is written.
			const detail = await context.backend.getMemory(context.userId, resolved.id);
			const author = agentAuthor(context.getModel());
			const data = composeDeletionData({
				target: "memory",
				targetId: detail.id,
				shortId: shortId(detail.id),
				content: detail.experience.content,
				classification: detail.experience.experience_type,
				tags: detail.experience.tags,
				createdAt: detail.created_at,
				// Children SURVIVE a forget with a dangling parent_id — the opposite
				// of the todo cascade, and recorded as the different fact it is.
				collateral: { relation: "orphaned", ids: detail.children_ids },
				reason: params.why,
				author,
			});
			const ledgerEntry = await context.ledger.append({
				kind: "deletion",
				actor: "agent",
				scope: "user",
				userId: context.userId,
				conversationId: context.conversationId,
				turn: context.getTurn(),
				data,
			});

			try {
				await context.backend.deleteMemory(context.userId, detail.id);
			} catch (error) {
				const reason = error instanceof Error ? error.message : String(error);
				// The record now overstates. Correcting it is the whole point of
				// having chosen this ordering, so the correction is attempted before
				// the model is told anything — and its own failure is reported rather
				// than swallowed, because it decides which of two states the ledger
				// is in.
				let compensationError: string | null = null;
				try {
					await context.ledger.append({
						kind: "revert",
						actor: "system",
						scope: "user",
						userId: context.userId,
						conversationId: context.conversationId,
						turn: context.getTurn(),
						data: {
							of: ledgerEntry.id,
							compensation: deletionNotPerformed(data, reason),
							note:
								"The deletion recorded by the referenced event did NOT happen: the entry is appended " +
								"before the backend call so that a failure overstates rather than hides. The memory " +
								"is intact.",
						},
					});
				} catch (compensationFailure) {
					compensationError =
						compensationFailure instanceof Error ? compensationFailure.message : String(compensationFailure);
				}
				throw new Error(
					composeForgetFailureReport({
						shortId: shortId(detail.id),
						error: reason,
						ledgerEventId: ledgerEntry.id,
						compensationError,
					}),
				);
			}

			return textResult(
				composeForgetReport({
					shortId: shortId(detail.id),
					classification: detail.experience.experience_type,
					preview: data.content_preview,
					orphanedChildren: detail.children_count,
					ledgerEventId: ledgerEntry.id,
				}),
				{ memory_id: detail.id, ledger_event_id: ledgerEntry.id, orphaned_children: detail.children_count },
			);
		},
	};

	const seatLearningTool: AgentTool<typeof seatLearningParameters> = {
		name: "record_seat_learning",
		label: "Record seat learning",
		description: composeToolDescription("record_seat_learning", {
			does:
				"Records one operational lesson about how this assistant should retrieve, phrase cues, or choose tools, " +
				"in the assistant's own memory scope rather than the user's.",
			useWhen:
				"Use it when a retrieval strategy visibly worked or visibly failed, or when you learn something about " +
				"the shape of this workbench that would save a later session the same detour. State the lesson so it is " +
				"actionable the next time the situation recurs, not as a diary entry about this one.",
			notFor:
				"Never put user content here — no names, events, decisions or facts about their world. Those belong in " +
				"remember_memory, and a fact filed in this scope is a fact that will never be found when the user asks " +
				"about their own corpus.",
			returns:
				"The lesson's short id. It is surfaced to future sessions automatically when a message resembles it, " +
				"so it is never returned by recall_memory and never cited to the user as evidence.",
		}),
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
			// The id, not a bare "recorded". The description promises one, and a
			// result that names what it wrote is the difference between a tool the
			// model can reason about afterwards and one it can only hope worked.
			return textResult(`Seat learning recorded as [mem:${shortId(response.id)}] in the assistant's own scope.`, {
				memory_id: response.id,
			});
		},
	};

	// Search, write, destroy, then the assistant's own scope. `forget_memory`
	// sits directly after `remember_memory` because that pairing is how the model
	// should read it — the inverse of a write, not a variety of search.
	return [recallTool, rememberTool, forgetTool, seatLearningTool];
}
