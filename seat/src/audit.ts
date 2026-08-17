/**
 * The audit read path: one flat, sorted, exportable trail over three stores
 * that previously could only be read separately and per-conversation.
 *
 * What it merges, and why each is needed to answer "who did what, when":
 * - the learning ledger (JSONL) — every update to memory state, with `actor`;
 * - `tool_call_start` / `tool_call_end` from the seat event store — WHICH TOOL
 *   WAS USED WHEN, the question the ledger alone cannot answer, joined here
 *   into one row per call so the pair reads as a single action with a duration;
 * - `memory_recall` / `proactive_context` events — what was retrieved and with
 *   what scores, which is the evidence for an answer rather than a change to
 *   state, and therefore never belonged in the ledger.
 *
 * Everything here is a pure function over already-read data. The I/O lives in
 * the caller (server.ts), so the shaping is testable without a store, a
 * backend, or a running seat.
 *
 * DETERMINISM: the row order is a total order (ts, source, ref) and the CSV
 * column list is fixed, so exporting the same underlying data twice produces
 * byte-identical output. An audit artefact that reordered itself between runs
 * would be unciteable.
 */

import { entryActor, type LedgerActorView, type LedgerEntry } from "./ledger.js";
import type { SeatEvent } from "./events.js";
import type { StoredEventRow } from "./store.js";

/**
 * One tool invocation, reassembled from the start/end event pair.
 *
 * The two events are stored separately because they are emitted at different
 * times; nothing durable joined them, so "how long did that tool take" was
 * not answerable from the store. Pairing is keyed on
 * (conversation_id, tool_call_id): tool call ids come from the model provider
 * and are only documented to be unique within a conversation.
 */
export interface ToolCallRecord {
	tool_call_id: string;
	tool_name: string;
	args: unknown;
	conversation_id: string;
	user_id: string;
	turn: number;
	started_at: string;
	/** Null when the turn ended before the tool returned (abort, crash, kill). */
	ended_at: string | null;
	/** Null for the same reason as `ended_at`. */
	duration_ms: number | null;
	/** Null when unterminated — NOT `false`, which would assert a success that never happened. */
	is_error: boolean | null;
}

/**
 * JSON rather than a delimiter join: `tool_call_id` is chosen by the model
 * provider, so no separator character can be assumed absent from it, and a
 * collision here would merge two different calls into one audit row.
 */
function pairKey(conversationId: string, toolCallId: string): string {
	return JSON.stringify([conversationId, toolCallId]);
}

/**
 * Join `tool_call_end` onto `tool_call_start`, oldest first.
 *
 * Unterminated calls are KEPT, with nulls. They are the most audit-relevant
 * rows in the set — a tool that was invoked and never returned is exactly what
 * a reviewer is looking for — and dropping them would make the trail claim
 * that an action which happened did not.
 */
export function pairToolCalls(rows: readonly StoredEventRow[]): ToolCallRecord[] {
	const started = new Map<string, ToolCallRecord>();
	const order: ToolCallRecord[] = [];

	for (const row of rows) {
		if (row.event.type === "tool_call_start") {
			const record: ToolCallRecord = {
				tool_call_id: row.event.tool_call_id,
				tool_name: row.event.tool_name,
				args: row.event.args,
				conversation_id: row.conversation_id,
				user_id: row.user_id,
				turn: row.turn,
				started_at: row.ts,
				ended_at: null,
				duration_ms: null,
				is_error: null,
			};
			started.set(pairKey(row.conversation_id, row.event.tool_call_id), record);
			order.push(record);
		} else if (row.event.type === "tool_call_end") {
			const record = started.get(pairKey(row.conversation_id, row.event.tool_call_id));
			// An end with no start means the window began mid-call; there is no
			// start row to attach it to and inventing one would fabricate a
			// duration. It is dropped, and the start-side row it belonged to
			// simply falls outside the requested range.
			if (!record) continue;
			record.ended_at = row.ts;
			record.duration_ms = Date.parse(row.ts) - Date.parse(record.started_at);
			record.is_error = row.event.is_error;
			started.delete(pairKey(row.conversation_id, row.event.tool_call_id));
		}
	}

	return order;
}

/** Where a row came from. */
export type AuditSource = "ledger" | "tool_call" | "retrieval" | "view";

/**
 * One line of the audit trail. Deliberately flat and uniform across sources:
 * a reviewer sorts, filters and diffs these in a spreadsheet or `jq`, and a
 * shape that varied per source would defeat both.
 */
export interface AuditRow {
	/** ISO-8601 UTC. */
	ts: string;
	source: AuditSource;
	actor: LedgerActorView;
	/** Ledger kind, tool name, or event type. */
	kind: string;
	user_id: string;
	conversation_id: string;
	turn: number;
	/** Ledger entry id, tool call id, or memory-operation identity. */
	ref: string;
	/** Source-specific payload, JSON-encoded so one column holds every shape. */
	detail: string;
}

/** CSV column order. Fixed, because an export whose columns move is not citable. */
export const AUDIT_COLUMNS = [
	"ts",
	"source",
	"actor",
	"kind",
	"user_id",
	"conversation_id",
	"turn",
	"ref",
	"detail",
] as const satisfies readonly (keyof AuditRow)[];

/**
 * Who caused a retrieval.
 *
 * `memory_recall` carries a `tool_call_id` only when it came from the model's
 * `recall_memory` tool (memory-tools.ts); the seat's own harness-learning
 * recall emits the same event type with no tool call id (conversation.ts).
 * That presence/absence is the only durable evidence of which one ran, so it
 * is the discriminator rather than a guess from `scope`.
 */
export function retrievalActor(event: Extract<SeatEvent, { type: "memory_recall" }>): LedgerActorView {
	return event.tool_call_id === undefined ? "system" : "agent";
}

function ledgerRow(entry: LedgerEntry): AuditRow {
	return {
		ts: entry.ts,
		source: "ledger",
		actor: entryActor(entry),
		kind: entry.kind,
		user_id: entry.user_id,
		conversation_id: entry.conversation_id,
		turn: entry.turn,
		ref: entry.id,
		detail: JSON.stringify({ scope: entry.scope, ...entry.data }),
	};
}

function toolCallRow(call: ToolCallRecord): AuditRow {
	return {
		ts: call.started_at,
		source: "tool_call",
		// A tool call exists because the model emitted it. Tools the seat runs
		// on its own behalf never enter the agent's tool loop.
		actor: "agent",
		kind: call.tool_name,
		user_id: call.user_id,
		conversation_id: call.conversation_id,
		turn: call.turn,
		ref: call.tool_call_id,
		detail: JSON.stringify({
			args: call.args,
			ended_at: call.ended_at,
			duration_ms: call.duration_ms,
			is_error: call.is_error,
		}),
	};
}

/**
 * Retrieval rows: what was surfaced, with ids and scores, so an answer can be
 * traced to the evidence that produced it.
 */
function retrievalRows(row: StoredEventRow): AuditRow[] {
	const base = {
		ts: row.ts,
		source: "retrieval" as const,
		user_id: row.user_id,
		conversation_id: row.conversation_id,
		turn: row.turn,
	};
	if (row.event.type === "memory_recall") {
		const event = row.event;
		return [
			{
				...base,
				actor: retrievalActor(event),
				kind: "memory_recall",
				ref: event.tool_call_id ?? `${row.conversation_id}:${row.turn}:harness`,
				detail: JSON.stringify({
					scope: event.scope,
					query: event.query,
					mode: event.mode,
					took_ms: event.took_ms,
					results: event.memories.map((memory) => ({
						id: memory.id,
						score: memory.score,
						importance: memory.importance,
						tier: memory.tier,
					})),
					fact_ids: event.facts.map((fact) => fact.id),
					todo_ids: event.todos.map((todo) => todo.id),
				}),
			},
		];
	}
	if (row.event.type === "proactive_context") {
		const event = row.event;
		return [
			{
				...base,
				// The proactive pass runs on every turn without anyone asking.
				actor: "system",
				kind: "proactive_context",
				ref: `${row.conversation_id}:${row.turn}:proactive`,
				detail: JSON.stringify({
					query: event.query,
					took_ms: event.took_ms,
					injected_memory_ids: event.injected_memory_ids,
					results: event.memories.map((memory) => ({
						id: memory.id,
						score: memory.score,
						importance: memory.importance,
						tier: memory.tier,
					})),
					feedback: event.feedback,
					temporal_credits_applied: event.temporal_credits_applied,
				}),
			},
		];
	}
	return [];
}

/**
 * A view command the model issued, as one row.
 *
 * WHY THIS IS NOT REDUNDANT WITH ITS `tool_call` ROW. The tool call records the
 * ARGUMENTS: the destination id the model typed and the entity names it hoped
 * existed. This records the OUTCOME of validation — the resolved path, the
 * graph's own names for the entities (the resolver folds aliases, so the two
 * lists differ), and the terms that named nothing. "The model asked to frame
 * five things" and "three of them exist in this profile" are different facts,
 * and only the second one describes what the person could have been shown.
 *
 * IT IS STILL AN ASK, NOT A VERDICT, and the trail must not be read as more.
 * Whether the command applied or waited as a Follow offer is decided by the
 * authority ledger in the browser against dimensions the person had touched;
 * the seat never learns which. `detail` therefore states what was requested and
 * says nothing about what appeared on screen.
 */
function viewRow(row: StoredEventRow): AuditRow | null {
	if (row.event.type !== "view_command") return null;
	const event = row.event;
	return {
		ts: row.ts,
		source: "view",
		// A view command exists only as the result of a model tool call; the
		// seat has no path that issues one on its own behalf.
		actor: "agent",
		kind: "view_command",
		user_id: row.user_id,
		conversation_id: row.conversation_id,
		turn: row.turn,
		ref: event.tool_call_id,
		detail: JSON.stringify({
			reason: event.reason,
			destination: event.destination,
			entities: event.entities,
			unresolved: event.unresolved,
		}),
	};
}

/** Event types {@link buildAuditRows} consumes. Also the store filter for the read. */
export const AUDIT_EVENT_TYPES = [
	"tool_call_start",
	"tool_call_end",
	"memory_recall",
	"proactive_context",
	"view_command",
] as const satisfies readonly SeatEvent["type"][];

/**
 * Merge the ledger and the seat event store into one sorted trail.
 *
 * Sort key is (ts, source, ref): a total order over rows, so the same inputs
 * always serialize identically. Ties on `ts` are real — a ledger entry and the
 * event announcing it are written in the same millisecond — and resolving them
 * by a stable, content-derived key rather than by input order is what makes
 * two exports of the same window comparable.
 */
export function buildAuditRows(input: {
	entries: readonly LedgerEntry[];
	events: readonly StoredEventRow[];
}): AuditRow[] {
	const rows: AuditRow[] = [];
	for (const entry of input.entries) rows.push(ledgerRow(entry));
	for (const call of pairToolCalls(input.events)) rows.push(toolCallRow(call));
	for (const event of input.events) rows.push(...retrievalRows(event));
	for (const event of input.events) {
		const row = viewRow(event);
		if (row) rows.push(row);
	}

	rows.sort((a, b) => {
		if (a.ts !== b.ts) return a.ts < b.ts ? -1 : 1;
		if (a.source !== b.source) return a.source < b.source ? -1 : 1;
		if (a.ref !== b.ref) return a.ref < b.ref ? -1 : 1;
		return 0;
	});
	return rows;
}

/** JSONL: one row per line, trailing newline. */
export function toJsonl(rows: readonly AuditRow[]): string {
	return rows.map((row) => JSON.stringify(row)).join("\n") + (rows.length > 0 ? "\n" : "");
}

/**
 * RFC 4180 field escaping. Every `detail` is JSON and therefore full of quotes
 * and commas; unescaped, a single row would shift every column after it and the
 * export would be silently, plausibly wrong rather than obviously broken.
 */
export function csvEscape(value: string): string {
	return /[",\r\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

/** CSV with a header row, CRLF line endings per RFC 4180. */
export function toCsv(rows: readonly AuditRow[]): string {
	const lines = [AUDIT_COLUMNS.join(",")];
	for (const row of rows) {
		lines.push(AUDIT_COLUMNS.map((column) => csvEscape(String(row[column]))).join(","));
	}
	return lines.join("\r\n") + "\r\n";
}
