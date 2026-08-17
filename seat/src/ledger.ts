/**
 * Learning ledger: every update the learning loops make to shodh-memory state
 * is recorded as an append-only, reviewable, revertible event BEFORE the UI or
 * anyone else has to ask "what changed and why".
 *
 * Design:
 * - Append-only JSONL. Reverts are themselves appended events referencing the
 *   original (`kind: "revert"`, data.of = original id); nothing is mutated.
 * - Revert semantics are honest about what the backend supports:
 *   - memory writes revert exactly (DELETE /api/memory/{id}).
 *   - "helpful"/"misleading" reinforcements revert by applying the opposite
 *     outcome through the same /api/reinforce path. The backend's momentum
 *     update (EMA with inertia, src/memory/feedback.rs) is not exactly
 *     invertible, so this is a compensating action, not a bitwise undo —
 *     recorded as such in the revert event.
 *   - "neutral" reinforcements record access only; there is nothing to
 *     compensate, and the revert event says so.
 *   - deletions CANNOT be reverted and the attempt is refused, not faked. See
 *     `deletionRevertRefusal`: this file holds a preview of what was destroyed,
 *     never the object, so a "restore" could only fabricate one.
 *
 * ORDERING, FOR THE ONE OPERATION WHERE IT MATTERS. Recoverable updates append
 * after the backend call, because the entry describes something that demonstrably
 * happened. A DELETION appends BEFORE it, and that ordering is chosen for its
 * failure mode: append-then-delete can leave an entry for a deletion that did not
 * occur, which a reviewer detects immediately because the named object is still
 * in the store and can be corrected by a compensating entry
 * (`deletionNotPerformed`); delete-then-append can leave a destruction with no
 * record anywhere, which is undetectable by construction and permanently
 * uncorrectable. The first keeps this ledger a superset of what happened, which
 * is auditable; the second makes it a subset, which cannot be told apart from a
 * complete record. A process that dies between the append and the backend call
 * leaves exactly the detectable phantom, by design.
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import * as path from "node:path";
import type { ReinforceOutcome, ReinforceStats, ShodhBackend } from "./backend.js";
import type { MemoryScope, ReinforceTrigger } from "./events.js";

export interface MemoryWriteData {
	memory_id: string;
	memory_type: string;
	content_preview: string;
	trigger: "model_tool_call" | "empty_recall_capture" | "tool_error_capture";
}

export interface ReinforceData {
	outcome: ReinforceOutcome;
	memory_ids: string[];
	trigger: ReinforceTrigger;
	stats: ReinforceStats;
}

/**
 * A destruction, recorded so that what was destroyed is still reviewable.
 *
 * WHY THIS KIND EXISTS AT ALL. Every other entry in this file records a change
 * that the thing it changed survives: a written memory can be read, a reinforced
 * one can be inspected, and a reviewer who wants to know what an entry means can
 * go and look at its subject. A deletion is the one event whose subject is gone
 * by the time anybody reads about it, so whatever the entry does not carry is
 * not recoverable from anywhere. `{ memory_id }` alone would prove that a
 * deletion happened and leave "what was lost" permanently unanswerable.
 *
 * WHAT IT CARRIES, AND THE PRIVACY LINE. Who (`author`, `reason`), when (the
 * entry's own `ts`), which (`target_id` + `short_id`), and enough of what
 * (`content_preview`, `content_length`, `content_sha256`, `classification`,
 * `tags`, `created_at`) for a reviewer to tell what class of thing was destroyed
 * and recognise it if they hold a copy.
 *
 * It is a 200-character preview and NOT the full content, deliberately:
 * - `MemoryWriteData.content_preview` is already 200 characters, so a deletion
 *   preview is not a new class of exposure for anything the seat itself wrote.
 * - Storing the whole body would make `forget` a lie. The user's one
 *   unambiguous request to destroy something would instead move it to a
 *   plaintext JSONL file that no forget path touches and no retention rule
 *   trims — an undeletable shadow copy created by the act of deletion.
 * - `content_sha256` covers the case the preview cannot: a reviewer who still
 *   holds the text can prove the entry refers to it, without the ledger having
 *   to hold the text to make that possible.
 *
 * The edge this accepts, stated rather than hidden: a memory the seat never
 * wrote — hook-captured, imported, or from before the seat existed — has its
 * first 200 characters written into the ledger for the first time by being
 * deleted. That is the floor a review needs to mean anything, and it is bounded,
 * hashed and attributed rather than open-ended.
 */
export interface DeletionData {
	target: "memory" | "todo";
	/** The resolved full id the destructive call actually ran against. */
	target_id: string;
	/** The handle a person recognises: an 8-char memory citation, or "BOLT-7". */
	short_id: string;
	/** First 200 characters of what was destroyed. */
	content_preview: string;
	/** Full length in characters, so a reviewer sees how much the preview omits. */
	content_length: number;
	/** SHA-256 of the full content: verification without disclosure. */
	content_sha256: string;
	/** Memory type, or todo status — what KIND of thing this was. */
	classification: string;
	tags: string[];
	/** When the destroyed thing was created, which its id does not carry. */
	created_at: string;
	/**
	 * What went with it — and the two cases carry different things ON PURPOSE.
	 *
	 * `orphaned` holds ids alone because those objects SURVIVE: a memory's
	 * children are not deleted, only left with a `parent_id` that no longer
	 * resolves, so a reviewer can still go and read every one of them and an id is
	 * a complete reference.
	 *
	 * `cascade_deleted` holds a preview per item because those objects are GONE:
	 * `TodoStore::delete_todo` destroys every subtask beneath the target, and an
	 * id that resolves to nothing is the same defect this whole entry exists to
	 * prevent, one level down. "BOLT-7 took three subtasks with it" is not a
	 * record of what was lost unless it says what the three were.
	 */
	collateral:
		| { relation: "orphaned"; ids: string[] }
		| {
				relation: "cascade_deleted";
				destroyed: { id: string; short_id: string; content_preview: string }[];
		  };
	/** Why, in the actor's own words. The first question a reviewer asks. */
	reason: string;
	/** The identity that asked — `agentAuthor(model)` for a model-issued call. */
	author: string;
}

export interface RevertData {
	of: string;
	compensation:
		| { kind: "memory_delete"; memory_id: string }
		| { kind: "counter_reinforce"; outcome: ReinforceOutcome; memory_ids: string[]; stats: ReinforceStats }
		/**
		 * The deletion recorded by the referenced entry did NOT happen: the entry
		 * was appended first (see {@link LearningLedger.append} ordering note) and
		 * the backend call that followed it failed.
		 *
		 * A distinct kind rather than `none`, because "nothing needed undoing" and
		 * "the record is wrong and here is the correction" are opposite facts and a
		 * reviewer reading `none` would take the deletion as real.
		 */
		| { kind: "deletion_not_performed"; target: "memory" | "todo"; target_id: string; error: string }
		| { kind: "none" };
	note: string;
}

/**
 * SHA-256 of content, hex — the part of a deletion record that survives the
 * privacy line.
 *
 * Over the exact string that was destroyed, unnormalised: a digest computed
 * over trimmed or case-folded text would not match anything a reviewer could
 * produce from a copy, which is the only thing it is for.
 */
export function contentDigest(content: string): string {
	return crypto.createHash("sha256").update(content, "utf8").digest("hex");
}

/** How much of a destroyed body a ledger entry keeps. Matches
 *  `MemoryWriteData.content_preview` on purpose — see {@link DeletionData}. */
export const DELETION_PREVIEW_CHARS = 200;

/**
 * A deletion entry, from the thing about to be destroyed.
 *
 * Pure and separate from the tool that calls it so the record's shape can be
 * tested without a backend: this is the only description of the deleted object
 * that will exist a second later, and a defect in it is undetectable afterwards
 * by construction.
 */
export function composeDeletionData(input: {
	target: "memory" | "todo";
	targetId: string;
	shortId: string;
	content: string;
	classification: string;
	tags: readonly string[];
	createdAt: string;
	collateral:
		| { relation: "orphaned"; ids: readonly string[] }
		| {
				relation: "cascade_deleted";
				destroyed: readonly { id: string; shortId: string; content: string }[];
		  };
	reason: string;
	author: string;
}): DeletionData {
	return {
		target: input.target,
		target_id: input.targetId,
		short_id: input.shortId,
		content_preview: input.content.slice(0, DELETION_PREVIEW_CHARS),
		content_length: input.content.length,
		content_sha256: contentDigest(input.content),
		classification: input.classification,
		tags: [...input.tags],
		created_at: input.createdAt,
		// Each branch copies its own arrays: an append-only ledger that shares
		// array references with its caller is not append-only.
		collateral:
			input.collateral.relation === "orphaned"
				? { relation: "orphaned", ids: [...input.collateral.ids] }
				: {
						relation: "cascade_deleted",
						destroyed: input.collateral.destroyed.map((item) => ({
							id: item.id,
							short_id: item.shortId,
							// The same 200-character rule as the target's, for the same
							// reason: enough for a reviewer to tell what was lost, not
							// enough to make the ledger a shadow copy of it.
							content_preview: item.content.slice(0, DELETION_PREVIEW_CHARS),
						})),
					},
		reason: input.reason,
		author: input.author,
	};
}

/**
 * The correction appended when a recorded deletion did not actually happen.
 *
 * It rides the `revert` kind because that is what the read side already
 * understands: {@link LearningLedger.list} annotates the original with
 * `reverted_by`, so the deletion entry stops reading as a standing fact the
 * moment this lands. The note says plainly that the item is INTACT — a reviewer
 * who took "reverted" to mean "deleted, then restored" would draw exactly the
 * wrong conclusion about a memory that was never touched.
 */
export function deletionNotPerformed(data: DeletionData, error: string): RevertData["compensation"] {
	return { kind: "deletion_not_performed", target: data.target, target_id: data.target_id, error };
}

/**
 * Why a deletion entry cannot be reverted, or null for an entry that can.
 *
 * NOTHING HERE CAN UNDO A DELETION, and the honest thing is to say so rather
 * than offer a button that fabricates. Every other compensation in this file
 * acts on an object that still exists; this one would have to invent its
 * subject — writing the 200-character preview back under a NEW id, with a fresh
 * `created_at`, no embeddings, no graph episode and no comment history. That is
 * not a restored memory, it is a truncated forgery that reads as recovery, and a
 * ledger whose stated rule is that a missing field is never backfilled by
 * inference cannot also manufacture the object the field described.
 */
export function deletionRevertRefusal(entry: Extract<LedgerEntry, { kind: "deletion" }>): string {
	const what = entry.data.target === "memory" ? "memory" : "todo";
	return (
		`Event ${entry.id} recorded the deletion of ${what} ${entry.data.short_id}, and a deletion cannot be ` +
		"reverted. Nothing in this ledger holds the deleted object — only a 200-character preview, its length and " +
		`its SHA-256 (${entry.data.content_sha256.slice(0, 16)}…) — so a "restore" would write a truncated copy ` +
		"under a new id with a new creation time, which is a fabrication wearing recovery's clothes. This entry " +
		"stands as the record of what was destroyed."
	);
}

/**
 * The backend's implicit-feedback pass, as reported by proactive_context.
 *
 * The implicit leg applies reinforce_recall and Hebbian strengthening
 * server-side (src/handlers/recall.rs:1670-1720) — the seat does not perform
 * these updates, it learns of them from `feedback_processed` on the response.
 * Before this entry existed they were invisible to the ledger, which broke
 * its core claim: with a corpus where every surfaced memory is proactive-owned
 * (found by the lessons A/B eval with a one-memory corpus), ALL loop-1
 * learning flowed through the implicit leg and the ledger recorded nothing.
 *
 * Revert compensates with an opposite explicit reinforce per id, same
 * honesty rule as `reinforce`: EMA momentum is not invertible, the revert
 * event says compensating, not undone.
 */
export interface ImplicitFeedbackData {
	memories_evaluated: number;
	reinforced: string[];
	weakened: string[];
}

/**
 * Who caused this entry to exist. Distinct from `scope`, which says WHICH
 * memory namespace was touched — the two are orthogonal and both spell "user",
 * so they are never passed positionally (see `LedgerAppendInput`).
 *
 * - `agent`  — the model chose it: a tool call it emitted (remember_memory,
 *              record_seat_learning). Attributable to the model's decision.
 * - `system` — an automatic seat loop with no decision by either party: the
 *              citation/overlap/negative-followup reinforcements, the backend's
 *              implicit-feedback pass, the deterministic harness captures.
 * - `user`   — a human acted through the seat's HTTP surface, which today means
 *              POST /v1/learning/revert.
 *
 * "Who did what" is unanswerable without this, and it cannot be recovered after
 * the fact: `trigger` is a good proxy for some kinds and absent on others.
 */
export type LedgerActor = "user" | "agent" | "system";

/**
 * Read-side actor, widened for entries written before `actor` existed. Those
 * are NOT backfilled — inferring an actor for a historical entry and writing it
 * down as fact is exactly the kind of invention an audit log exists to prevent.
 * They report "unknown" and a reviewer can see the gap.
 */
export type LedgerActorView = LedgerActor | "unknown";

const KNOWN_ACTORS: ReadonlySet<string> = new Set<LedgerActor>(["user", "agent", "system"]);

/** The entry's actor, or "unknown" for a legacy or corrupt value. */
export function entryActor(entry: LedgerEntry): LedgerActorView {
	const actor = (entry as { actor?: unknown }).actor;
	return typeof actor === "string" && KNOWN_ACTORS.has(actor) ? (actor as LedgerActor) : "unknown";
}

export type LedgerEntry =
	| LedgerEntryBase<"memory_write", MemoryWriteData>
	| LedgerEntryBase<"reinforce", ReinforceData>
	| LedgerEntryBase<"implicit_feedback", ImplicitFeedbackData>
	| LedgerEntryBase<"deletion", DeletionData>
	| LedgerEntryBase<"revert", RevertData>;

interface LedgerEntryBase<K extends string, D> {
	id: string;
	ts: string;
	kind: K;
	/** Who initiated this update. Absent on entries written before it existed. */
	actor: LedgerActor;
	scope: MemoryScope;
	/** The actual backend user_id the operation ran against (harness scope uses the derived namespace). */
	user_id: string;
	conversation_id: string;
	turn: number;
	data: D;
}

/**
 * One append. An object rather than positional arguments because `actor` and
 * `scope` are adjacent, both string enums, and both admit the literal "user" —
 * positionally they are silently swappable, and a swapped audit attribution is
 * worse than no audit attribution.
 */
export interface LedgerAppendInput<K extends LedgerEntry["kind"]> {
	kind: K;
	actor: LedgerActor;
	scope: MemoryScope;
	userId: string;
	conversationId: string;
	turn: number;
	data: Extract<LedgerEntry, { kind: K }>["data"];
}

export interface LedgerEntryView {
	entry: LedgerEntry;
	reverted_by?: string;
}

/** Filter for {@link LearningLedger.query}. Every field narrows. */
export interface LedgerQuery {
	/** The backend namespace the entry ran against (`user_id`, not the actor). */
	userId?: string;
	conversationId?: string;
	/** ISO-8601 UTC, inclusive lower bound on `ts`. */
	since?: string;
	/** ISO-8601 UTC, exclusive upper bound on `ts`. */
	until?: string;
}

export class LedgerError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "LedgerError";
	}
}

export class LearningLedger {
	private readonly filePath: string;
	/** Serializes appends so entries never interleave. */
	private writeChain: Promise<void> = Promise.resolve();

	constructor(dataDir: string) {
		fs.mkdirSync(dataDir, { recursive: true });
		this.filePath = path.join(dataDir, "learning-ledger.jsonl");
	}

	get file(): string {
		return this.filePath;
	}

	async append<K extends LedgerEntry["kind"]>(input: LedgerAppendInput<K>): Promise<LedgerEntry> {
		const entry = {
			id: crypto.randomUUID(),
			ts: new Date().toISOString(),
			kind: input.kind,
			actor: input.actor,
			scope: input.scope,
			user_id: input.userId,
			conversation_id: input.conversationId,
			turn: input.turn,
			data: input.data,
		} as LedgerEntry;

		const write = this.writeChain.then(() => fsp.appendFile(this.filePath, `${JSON.stringify(entry)}\n`, "utf8"));
		// Keep the chain alive even if a write fails; the failure surfaces to the caller.
		this.writeChain = write.catch(() => {});
		await write;
		return entry;
	}

	private async readAll(): Promise<LedgerEntry[]> {
		let raw: string;
		try {
			raw = await fsp.readFile(this.filePath, "utf8");
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
			throw error;
		}
		const entries: LedgerEntry[] = [];
		for (const line of raw.split("\n")) {
			const trimmed = line.trim();
			if (!trimmed) continue;
			try {
				entries.push(JSON.parse(trimmed) as LedgerEntry);
			} catch {
				// A torn trailing line (crash mid-append) is skipped; everything
				// before it is intact because appends are serialized.
			}
		}
		return entries;
	}

	async list(options: { limit?: number; conversationId?: string } = {}): Promise<LedgerEntryView[]> {
		const limit = options.limit ?? 100;
		const entries = await this.readAll();
		const revertedBy = new Map<string, string>();
		for (const entry of entries) {
			if (entry.kind === "revert") revertedBy.set(entry.data.of, entry.id);
		}
		const filtered = options.conversationId
			? entries.filter((entry) => entry.conversation_id === options.conversationId)
			: entries;
		return filtered
			.slice(-limit)
			.reverse()
			.map((entry) => ({
				entry,
				reverted_by: revertedBy.get(entry.id),
			}));
	}

	/**
	 * Entries matching a filter, oldest first — the read the audit export needs.
	 *
	 * Distinct from {@link list}, which is the UI's review feed: newest-first,
	 * limit-capped, and annotated with `reverted_by`. An export wants the raw
	 * append order over a time window, unannotated and uncapped, because a
	 * silently truncated audit trail is worse than none.
	 *
	 * `since` is inclusive and `until` exclusive, compared as strings: `ts` is
	 * `Date.toISOString()`, so lexicographic order is chronological order.
	 */
	async query(options: LedgerQuery = {}): Promise<LedgerEntry[]> {
		const entries = await this.readAll();
		return entries.filter((entry) => {
			if (options.userId !== undefined && entry.user_id !== options.userId) return false;
			if (options.conversationId !== undefined && entry.conversation_id !== options.conversationId) return false;
			if (options.since !== undefined && entry.ts < options.since) return false;
			if (options.until !== undefined && entry.ts >= options.until) return false;
			return true;
		});
	}

	async get(id: string): Promise<LedgerEntryView | undefined> {
		const entries = await this.readAll();
		const entry = entries.find((candidate) => candidate.id === id);
		if (!entry) return undefined;
		const revert = entries.find(
			(candidate) => candidate.kind === "revert" && candidate.data.of === id,
		);
		return { entry, reverted_by: revert?.id };
	}

	/**
	 * Revert a learning event by applying its compensating action through the
	 * backend, then recording the revert as a new ledger event.
	 *
	 * `actor` is required rather than defaulted: the revert entry attributes a
	 * deliberate corrective action, and the only caller that knows whether a
	 * human or an automated policy asked for it is the one making the call.
	 */
	async revert(id: string, backend: ShodhBackend, actor: LedgerActor): Promise<LedgerEntry> {
		const view = await this.get(id);
		if (!view) throw new LedgerError(`Unknown ledger event: ${id}`);
		if (view.reverted_by) throw new LedgerError(`Event ${id} was already reverted by ${view.reverted_by}`);
		const original = view.entry;
		if (original.kind === "revert") throw new LedgerError("Revert events cannot be reverted");

		// Refused before any backend call, because there is no backend call that
		// could help: the refusal is about what this ledger does and does not hold.
		// Written as a `kind` check rather than a truthiness test on the refusal so
		// the compiler also narrows `original` out of the reinforce branch below.
		if (original.kind === "deletion") throw new LedgerError(deletionRevertRefusal(original));

		let compensation: RevertData["compensation"];
		let note: string;

		if (original.kind === "memory_write") {
			await backend.deleteMemory(original.user_id, original.data.memory_id);
			compensation = { kind: "memory_delete", memory_id: original.data.memory_id };
			note = "Exact revert: the written memory was deleted.";
		} else if (original.kind === "implicit_feedback") {
			// Compensate each direction with its opposite explicit reinforce.
			// Same honesty rule as explicit reinforce: EMA momentum has inertia,
			// so this counters rather than undoes — and the backend's own
			// Hebbian strengthening from the implicit pass is countered only to
			// the extent /api/reinforce reaches the same weights.
			const ids = [...original.data.reinforced, ...original.data.weakened];
			if (ids.length === 0) {
				compensation = { kind: "none" };
				note = "The implicit pass evaluated memories but moved none; nothing to compensate.";
			} else {
				let stats = {
					memories_processed: 0,
					associations_strengthened: 0,
					importance_boosts: 0,
					importance_decays: 0,
				};
				if (original.data.reinforced.length > 0) {
					stats = await backend.reinforce(original.user_id, original.data.reinforced, "misleading");
				}
				if (original.data.weakened.length > 0) {
					const s2 = await backend.reinforce(original.user_id, original.data.weakened, "helpful");
					stats = {
						memories_processed: stats.memories_processed + s2.memories_processed,
						associations_strengthened: stats.associations_strengthened + s2.associations_strengthened,
						importance_boosts: stats.importance_boosts + s2.importance_boosts,
						importance_decays: stats.importance_decays + s2.importance_decays,
					};
				}
				compensation = { kind: "counter_reinforce", outcome: "misleading", memory_ids: ids, stats };
				note =
					"Compensating action: opposite explicit reinforce per direction. The backend's " +
					"implicit momentum and Hebbian updates are countered, not exactly undone.";
			}
		} else {
			const outcome = original.data.outcome;
			if (outcome === "neutral") {
				compensation = { kind: "none" };
				note = "Neutral reinforcement records access only; no compensating action exists.";
			} else {
				const inverse: ReinforceOutcome = outcome === "helpful" ? "misleading" : "helpful";
				const stats = await backend.reinforce(original.user_id, original.data.memory_ids, inverse);
				compensation = { kind: "counter_reinforce", outcome: inverse, memory_ids: original.data.memory_ids, stats };
				note =
					"Compensating action: opposite outcome applied via /api/reinforce. " +
					"The backend momentum update (EMA with inertia) is not exactly invertible.";
			}
		}

		return this.append({
			kind: "revert",
			actor,
			scope: original.scope,
			userId: original.user_id,
			conversationId: original.conversation_id,
			turn: original.turn,
			data: { of: original.id, compensation, note },
		});
	}
}
