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

export interface RevertData {
	of: string;
	compensation:
		| { kind: "memory_delete"; memory_id: string }
		| { kind: "counter_reinforce"; outcome: ReinforceOutcome; memory_ids: string[]; stats: ReinforceStats }
		| { kind: "none" };
	note: string;
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
 * A turn in which tool policy removed something from the model's reach.
 *
 * Recorded because "the agent could not do that" is worth nothing to whoever
 * approves the deployment without a line saying when it was stopped and by
 * which rule. This is the difference between a constraint and a claim about
 * one.
 *
 * Written once per turn rather than once per tool: the turn is the unit an
 * auditor reads, and one line per withheld tool per turn would bury the
 * memory events this file exists for.
 *
 * NOT revertible in the compensating sense — a past turn's reach cannot be
 * widened after the fact — so a revert of one of these carries
 * `compensation: { kind: "none" }` and exists only to annotate.
 */
export interface PolicyWithheldData {
	/** Sorted, so the same policy against the same server logs identically. */
	withheld: { tool: string; by: string; reason: string }[];
	/** How many tools the model WAS offered, so the line reads as a ratio
	 *  rather than an unbounded list of absences. */
	offered: number;
}

export type LedgerEntry =
	| LedgerEntryBase<"memory_write", MemoryWriteData>
	| LedgerEntryBase<"reinforce", ReinforceData>
	| LedgerEntryBase<"implicit_feedback", ImplicitFeedbackData>
	| LedgerEntryBase<"policy_withheld", PolicyWithheldData>
	| LedgerEntryBase<"revert", RevertData>;

interface LedgerEntryBase<K extends string, D> {
	id: string;
	ts: string;
	kind: K;
	scope: MemoryScope;
	/** The actual backend user_id the operation ran against (harness scope uses the derived namespace). */
	user_id: string;
	conversation_id: string;
	turn: number;
	data: D;
}

export interface LedgerEntryView {
	entry: LedgerEntry;
	reverted_by?: string;
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

	async append<K extends LedgerEntry["kind"]>(
		kind: K,
		scope: MemoryScope,
		userId: string,
		conversationId: string,
		turn: number,
		data: Extract<LedgerEntry, { kind: K }>["data"],
	): Promise<LedgerEntry> {
		const entry = {
			id: crypto.randomUUID(),
			ts: new Date().toISOString(),
			kind,
			scope,
			user_id: userId,
			conversation_id: conversationId,
			turn,
			data,
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
	 */
	async revert(id: string, backend: ShodhBackend): Promise<LedgerEntry> {
		const view = await this.get(id);
		if (!view) throw new LedgerError(`Unknown ledger event: ${id}`);
		if (view.reverted_by) throw new LedgerError(`Event ${id} was already reverted by ${view.reverted_by}`);
		const original = view.entry;
		if (original.kind === "revert") throw new LedgerError("Revert events cannot be reverted");

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
		} else if (original.kind === "policy_withheld") {
			// A past turn's reach cannot be widened after the fact, so there is
			// nothing to compensate. The revert exists to annotate — an operator
			// marking that a withholding was wrong is itself a record worth
			// keeping, and silently refusing to revert would leave them no way
			// to say so in the file that is supposed to hold the whole history.
			compensation = { kind: "none" };
			note =
				`Policy withheld ${original.data.withheld.length} tool(s) on this turn. ` +
				"Nothing to undo: the turn has already run with the narrower tool set. " +
				"Widening reach is a change to the policy file, not a revert.";
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

		return this.append("revert", original.scope, original.user_id, original.conversation_id, original.turn, {
			of: original.id,
			compensation,
			note,
		});
	}
}
