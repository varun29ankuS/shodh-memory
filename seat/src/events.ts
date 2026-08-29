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
	| { type: "error"; message: string }
	/**
	 * The agent moving the operator's screen.
	 *
	 * Carries an intent, never a selector or a coordinate: the client applies it
	 * through the same router and stores its own controls use, so an agent-driven
	 * navigation and a clicked one reach identical state. `reason` is not
	 * decoration -- a screen that changes by itself with no explanation reads as
	 * a fault, so the UI shows it.
	 */
	| { type: "ui_command"; command: UiCommand; reason: string };

/** Destinations the agent may open, mirroring DESTINATIONS in the rail. */
export type UiView = "briefing" | "chat" | "recall" | "graph" | "geo" | "anomalies" | "tasks" | "providers";

export type UiCommand =
	| { kind: "open"; view: UiView }
	| { kind: "select_memory"; memory_id: string }
	| { kind: "select_profile"; profile: string };

export type SeatEventSink = (event: SeatEvent) => void;
