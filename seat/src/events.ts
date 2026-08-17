/**
 * Structured events streamed to the client over SSE.
 *
 * The traceability surface is the product: memory operations are never opaque.
 * Every recall carries ids, scores, and full ScoreAttribution; every learning
 * update (write or reinforcement) carries its ledger event id so the UI can
 * review and revert it.
 */

import type {
	FeedbackProcessed,
	ProactiveSurfacedMemory,
	RecallFact,
	RecallLineageEdge,
	RecallMemory,
	RecallTodo,
	ReinforceOutcome,
	ReinforceStats,
} from "./backend.js";
import type { ViewDimension, ViewOutcomeState } from "./view-link.js";

/** Which memory namespace an operation touched. */
export type MemoryScope = "user" | "harness";

export interface ModelRef {
	provider: string;
	id: string;
	name: string;
}

export interface UsagePayload {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	reasoning?: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

export type ReinforceTrigger =
	| { kind: "response_overlap"; overlaps: Record<string, number>; threshold: number }
	| { kind: "citation"; cited: string[] }
	| { kind: "negative_followup"; keywords: string[] }
	| { kind: "revert"; of: string };

export type SeatEvent =
	| { type: "conversation_created"; conversation_id: string; user_id: string; model: ModelRef }
	| { type: "turn_start"; turn: number }
	| { type: "text_delta"; delta: string }
	| { type: "thinking_delta"; delta: string }
	| { type: "tool_call_start"; tool_call_id: string; tool_name: string; args: unknown }
	| { type: "tool_call_end"; tool_call_id: string; tool_name: string; is_error: boolean }
	| {
			type: "memory_recall";
			scope: MemoryScope;
			tool_call_id?: string;
			query: string;
			mode: string;
			memories: RecallMemory[];
			facts: RecallFact[];
			todos: RecallTodo[];
			lineage: RecallLineageEdge[];
			took_ms: number;
	  }
	| {
			type: "memory_write";
			scope: MemoryScope;
			memory_id: string;
			memory_type: string;
			content_preview: string;
			ledger_event_id: string;
	  }
	| {
			type: "memory_reinforce";
			scope: MemoryScope;
			outcome: ReinforceOutcome;
			memory_ids: string[];
			stats: ReinforceStats;
			trigger: ReinforceTrigger;
			ledger_event_id: string;
	  }
	| {
			/**
			 * One proactive_context round-trip: the memories the backend
			 * auto-surfaced for this turn, plus the implicit-feedback outcome
			 * for the PREVIOUS turn's surfaced set (momentum reinforced /
			 * weakened ids), so the momentum loop is as inspectable as the
			 * explicit one.
			 */
			type: "proactive_context";
			scope: "user";
			query: string;
			memories: ProactiveSurfacedMemory[];
			injected_memory_ids: string[];
			/** The system-prompt block verbatim as injected this turn (null when
			 * nothing surfaced) — what the model saw must be inspectable. */
			injected_block: string | null;
			feedback: FeedbackProcessed | null;
			temporal_credits_applied: number | null;
			took_ms: number;
	  }
	| {
			/**
			 * The model asked to move the user's view (view-tools.ts `direct_view`).
			 *
			 * EMITTED AFTER VALIDATION, NOT AT THE CALL. The destination is a real
			 * one and every entity here was resolved against this profile's graph,
			 * so a consumer never has to re-check what the model claimed. The
			 * unresolved terms are recorded too — a command that framed three of
			 * five named things must not read as one that framed five.
			 *
			 * This is an ASK, not an outcome. Whether it applied or became a Follow
			 * offer is decided by the authority ledger in the browser
			 * (front/ui/src/stores/view.ts). The seat now LEARNS that verdict — the
			 * browser reports it back over POST /v1/conversations/{id}/view-report
			 * — but it learns it separately, as `view_outcome`. This event stays
			 * exactly what it was: the ask, with no outcome field, because at the
			 * moment it is emitted there is no outcome to have.
			 */
			type: "view_command";
			tool_call_id: string;
			/** Why, in the model's own words. Shown to the person verbatim. */
			reason: string;
			/** Destination path (e.g. "/geo"), or null when the move only frames. */
			destination: string | null;
			/** Entity names as the graph knows them — resolved, not as typed. */
			entities: string[];
			/** Terms the graph ANSWERED about and does not hold. Never silently dropped. */
			unresolved: string[];
			/**
			 * Terms the graph was never reached to check.
			 *
			 * KEPT APART FROM `unresolved` BECAUSE THEY ARE OPPOSITE CLAIMS. One
			 * says the corpus does not contain a thing; the other says this seat
			 * does not know. A trail that filed a backend outage under "absent"
			 * would be evidence for a fact about the person's data that nobody ever
			 * established — and an ask is exactly the row a reviewer would cite it
			 * from.
			 */
			unchecked: string[];
			/**
			 * The one entity to open in the inspector, or null.
			 *
			 * Carries the graph's `uuid` because that is what the browser selects
			 * by (`UniverseStar.id`, src/graph_memory.rs), and the name because
			 * that is what a person reads. Both are the graph's own, resolved by
			 * the seat before this was emitted — the model's word never travels.
			 */
			focus: { id: string; name: string } | null;
	  }
	| {
			/**
			 * What the browser did with a `view_command` — the return leg.
			 *
			 * ONE EVENT PER DIMENSION, because one command lands on up to four
			 * axes and they can land differently: the cue and the camera apply
			 * while the destination waits, because the person was holding the
			 * destination and nothing else.
			 *
			 * NOT STREAMED, PERSISTED DIRECTLY. An offer accepted after the turn
			 * ends has no open stream to ride, so the route writes these to the
			 * event store itself (server.ts) rather than through a turn's sink.
			 * One path for every outcome, early or late, instead of two that could
			 * disagree.
			 *
			 * ABSENCE IS THE UNKNOWN. There is no state meaning "no verdict": a
			 * command the browser never answered for simply has no row, and every
			 * reader — the audit trail, the History screen — must read the absence
			 * as "not known" rather than as anything having happened.
			 */
			type: "view_outcome";
			/** The `view_command` this answers, by its tool call id. */
			tool_call_id: string;
			/** One of front/ui/src/lib/view/authority.ts `VIEW_DIMENSIONS`. */
			dimension: ViewDimension;
			state: ViewOutcomeState;
			/** The path the browser was on when it decided. Context for a reader
			 *  of the trail, who otherwise cannot tell where "already" was true. */
			at: string;
	  }
	| {
			/**
			 * A verdict finally reaching the model, one turn after it was decided.
			 *
			 * THE DIFFERENCE BETWEEN THIS AND `view_outcome` IS THE AUDIENCE. That
			 * one records what the PERSON's workbench did, and it is written the
			 * moment the browser reports it. This one records that the MODEL was
			 * told — which is a separate fact, happens later, and is exactly the
			 * question a reader has when they see an offer accepted at 14:02 and an
			 * assistant still calling it pending at 14:03.
			 *
			 * `injected_block` is the text verbatim, on the same rule the proactive
			 * pass follows: what the model was actually shown must be inspectable,
			 * not reconstructable from the pieces.
			 *
			 * IT IS NOT AN AUDIT SOURCE. `AUDIT_EVENT_TYPES` (audit.ts) deliberately
			 * does not include it — the trail already carries the ask, the outcome
			 * and the tool call, and a fourth row saying "and then we mentioned it"
			 * would add a line to every export without adding an act to it.
			 */
			type: "view_outcome_relayed";
			outcomes: { tool_call_id: string; dimension: ViewDimension; state: ViewOutcomeState }[];
			injected_block: string;
	  }
	| {
			/**
			 * The seat asking the browser what is on screen (`inspect_view`).
			 *
			 * A REQUEST FOR A READING, and it is the only event in this union that
			 * carries no information about memory at all. The browser answers on
			 * the same route the verdicts use, quoting `probe_id`.
			 *
			 * Deliberately NOT durable-audit material: the `inspect_view` tool call
			 * is already a `tool_call` row in the trail, and a second row saying
			 * the same thing at the same instant is noise in an artefact whose
			 * value is that every line means something.
			 */
			type: "view_probe";
			probe_id: string;
	  }
	| { type: "harness_learning_applied"; memories: { id: string; content: string; score: number }[] }
	| {
			/**
			 * Deterministic post-draft verification (conversation.ts verify loop):
			 * the issues found in the drafted answer and whether a bounded
			 * revision pass was run. Emitted only when at least one issue fired,
			 * so rescoring can count trigger rates and attribute revisions.
			 */
			type: "verification";
			issues: string[];
			nudged: boolean;
	  }
	| { type: "model_changed"; model: ModelRef }
	| { type: "usage"; model: ModelRef; usage: UsagePayload }
	| { type: "turn_end"; turn: number; stop_reason: string; error_message?: string }
	| { type: "agent_end" }
	| { type: "error"; message: string };

export type SeatEventSink = (event: SeatEvent) => void;
