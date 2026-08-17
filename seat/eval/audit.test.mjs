/**
 * The audit read path: actor attribution, tool-call pairing, and export
 * serialization.
 *
 * Every test here is written to FAIL if the behaviour it names is removed —
 * each was checked by deleting the line it covers and confirming it went red.
 * The bar exists because this repo has shipped tautological asserts that kept
 * an inert layer looking tested.
 *
 * Run: npm run build && npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
	AUDIT_COLUMNS,
	AUDIT_EVENT_TYPES,
	buildAuditRows,
	csvEscape,
	pairToolCalls,
	retrievalActor,
	toCsv,
	toJsonl,
} from "../dist/audit.js";
import { entryActor } from "../dist/ledger.js";

// ── Fixtures ────────────────────────────────────────────────────────────────

function startRow(overrides = {}) {
	return {
		conversation_id: "conv-1",
		user_id: "demo",
		turn: 1,
		ts: "2026-08-16T10:00:00.000Z",
		event: { type: "tool_call_start", tool_call_id: "call-1", tool_name: "read_file", args: { path: "/etc/hosts" } },
		...overrides,
	};
}

function endRow(overrides = {}) {
	return {
		conversation_id: "conv-1",
		user_id: "demo",
		turn: 1,
		ts: "2026-08-16T10:00:00.250Z",
		event: { type: "tool_call_end", tool_call_id: "call-1", tool_name: "read_file", is_error: false },
		...overrides,
	};
}

function ledgerEntry(overrides = {}) {
	return {
		id: "led-1",
		ts: "2026-08-16T10:00:00.000Z",
		actor: "agent",
		kind: "memory_write",
		scope: "user",
		user_id: "demo",
		conversation_id: "conv-1",
		turn: 1,
		data: { memory_id: "mem-1", memory_type: "observation", content_preview: "p", trigger: "model_tool_call" },
		...overrides,
	};
}

// ── entryActor: the read-side tolerance for pre-actor entries ───────────────

test("entryActor returns the recorded actor", () => {
	assert.equal(entryActor(ledgerEntry({ actor: "agent" })), "agent");
	assert.equal(entryActor(ledgerEntry({ actor: "system" })), "system");
	assert.equal(entryActor(ledgerEntry({ actor: "user" })), "user");
});

test("entryActor reports 'unknown' for entries written before actor existed", () => {
	// The 26 live entries on this machine have no `actor` key at all. They must
	// not be silently attributed to anyone.
	const legacy = ledgerEntry();
	delete legacy.actor;
	assert.equal(entryActor(legacy), "unknown");
});

test("entryActor rejects a value outside the enum rather than passing it through", () => {
	// A hand-edited or corrupted line must not be able to invent an actor.
	assert.equal(entryActor(ledgerEntry({ actor: "admin" })), "unknown");
	assert.equal(entryActor(ledgerEntry({ actor: "" })), "unknown");
	assert.equal(entryActor(ledgerEntry({ actor: 7 })), "unknown");
	assert.equal(entryActor(ledgerEntry({ actor: null })), "unknown");
});

// ── pairToolCalls ───────────────────────────────────────────────────────────

test("pairToolCalls joins end onto start and derives the duration", () => {
	const [call] = pairToolCalls([startRow(), endRow()]);
	assert.equal(call.tool_call_id, "call-1");
	assert.equal(call.tool_name, "read_file");
	assert.deepEqual(call.args, { path: "/etc/hosts" });
	assert.equal(call.started_at, "2026-08-16T10:00:00.000Z");
	assert.equal(call.ended_at, "2026-08-16T10:00:00.250Z");
	assert.equal(call.duration_ms, 250);
	assert.equal(call.is_error, false);
});

test("pairToolCalls carries the error flag through", () => {
	const [call] = pairToolCalls([
		startRow(),
		endRow({ event: { type: "tool_call_end", tool_call_id: "call-1", tool_name: "read_file", is_error: true } }),
	]);
	assert.equal(call.is_error, true);
});

test("pairToolCalls keeps an unterminated call with nulls, not a false success", () => {
	// A tool invoked and never returned (aborted turn) is the row a reviewer
	// most wants; is_error must not read as `false`.
	const calls = pairToolCalls([startRow()]);
	assert.equal(calls.length, 1);
	assert.equal(calls[0].ended_at, null);
	assert.equal(calls[0].duration_ms, null);
	assert.equal(calls[0].is_error, null);
});

test("pairToolCalls does not join across conversations sharing a tool_call_id", () => {
	// Tool call ids come from the model provider and are only unique within a
	// conversation. Two conversations interleave, both using "call-1", and the
	// end belongs to the FIRST — keying on the id alone would hand conv-1's
	// outcome and duration to conv-2 and leave conv-1 looking unterminated.
	//
	// The interleaving order matters: if the end arrived for the most recently
	// started conversation, a broken key would coincidentally produce the right
	// answer and this test would pass while asserting nothing.
	const calls = pairToolCalls([
		startRow({ ts: "2026-08-16T10:00:00.000Z" }),
		startRow({ conversation_id: "conv-2", ts: "2026-08-16T10:00:01.000Z" }),
		endRow({
			conversation_id: "conv-1",
			ts: "2026-08-16T10:00:05.000Z",
			event: { type: "tool_call_end", tool_call_id: "call-1", tool_name: "read_file", is_error: true },
		}),
	]);
	assert.equal(calls.length, 2);
	const byConversation = Object.fromEntries(calls.map((call) => [call.conversation_id, call]));
	assert.equal(byConversation["conv-1"].duration_ms, 5000, "the end belongs to conv-1");
	assert.equal(byConversation["conv-1"].is_error, true);
	assert.equal(byConversation["conv-2"].duration_ms, null, "conv-2 is still running");
	assert.equal(byConversation["conv-2"].is_error, null);
});

test("pairToolCalls drops an end whose start is outside the window", () => {
	// No start row means no known begin time; a fabricated one would produce a
	// fabricated duration.
	assert.deepEqual(pairToolCalls([endRow()]), []);
});

test("pairToolCalls does not let a second end overwrite a completed call", () => {
	const calls = pairToolCalls([
		startRow(),
		endRow(),
		endRow({ ts: "2026-08-16T10:00:59.000Z", event: { type: "tool_call_end", tool_call_id: "call-1", tool_name: "read_file", is_error: true } }),
	]);
	assert.equal(calls.length, 1);
	assert.equal(calls[0].duration_ms, 250, "the first end is the real one");
	assert.equal(calls[0].is_error, false);
});

// ── retrievalActor ──────────────────────────────────────────────────────────

test("retrievalActor distinguishes a model recall from the seat's own recall", () => {
	// memory-tools.ts sets tool_call_id; the harness recall in conversation.ts
	// does not. That presence is the only durable evidence of which ran.
	assert.equal(retrievalActor({ type: "memory_recall", tool_call_id: "call-9" }), "agent");
	assert.equal(retrievalActor({ type: "memory_recall" }), "system");
	assert.equal(retrievalActor({ type: "memory_recall", tool_call_id: undefined }), "system");
});

// ── buildAuditRows ──────────────────────────────────────────────────────────

test("buildAuditRows merges ledger, tool calls and retrievals into one trail", () => {
	const rows = buildAuditRows({
		entries: [ledgerEntry()],
		events: [
			startRow(),
			endRow(),
			{
				conversation_id: "conv-1",
				user_id: "demo",
				turn: 1,
				ts: "2026-08-16T10:00:02.000Z",
				event: {
					type: "memory_recall",
					scope: "user",
					tool_call_id: "call-7",
					query: "q",
					mode: "hybrid",
					memories: [{ id: "mem-9", score: 0.42, importance: 0.5, tier: "warm" }],
					facts: [],
					todos: [],
					lineage: [],
					took_ms: 12,
				},
			},
		],
	});
	assert.deepEqual(
		rows.map((row) => row.source),
		["ledger", "tool_call", "retrieval"],
	);
	assert.deepEqual(
		rows.map((row) => row.actor),
		["agent", "agent", "agent"],
	);
	assert.equal(rows[2].kind, "memory_recall");
	// The scored result set is the evidence for the answer — it must survive.
	assert.deepEqual(JSON.parse(rows[2].detail).results, [
		{ id: "mem-9", score: 0.42, importance: 0.5, tier: "warm" },
	]);
});

test("buildAuditRows attributes the automatic proactive pass to the system", () => {
	const rows = buildAuditRows({
		entries: [],
		events: [
			{
				conversation_id: "conv-1",
				user_id: "demo",
				turn: 3,
				ts: "2026-08-16T10:00:00.000Z",
				event: {
					type: "proactive_context",
					scope: "user",
					query: "q",
					memories: [],
					injected_memory_ids: ["mem-4"],
					injected_block: "…",
					feedback: { memories_evaluated: 2, reinforced: ["mem-4"], weakened: [] },
					temporal_credits_applied: 1,
					took_ms: 5,
				},
			},
		],
	});
	assert.equal(rows.length, 1);
	assert.equal(rows[0].actor, "system");
	assert.equal(rows[0].kind, "proactive_context");
	assert.deepEqual(JSON.parse(rows[0].detail).injected_memory_ids, ["mem-4"]);
});

test("buildAuditRows preserves 'unknown' for a legacy ledger entry", () => {
	const legacy = ledgerEntry();
	delete legacy.actor;
	const rows = buildAuditRows({ entries: [legacy], events: [] });
	assert.equal(rows[0].actor, "unknown");
});

test("buildAuditRows produces a total order independent of input order", () => {
	// An export whose rows reorder between runs cannot be diffed or cited.
	const entries = [
		ledgerEntry({ id: "led-b", ts: "2026-08-16T10:00:00.000Z" }),
		ledgerEntry({ id: "led-a", ts: "2026-08-16T10:00:00.000Z" }),
		ledgerEntry({ id: "led-c", ts: "2026-08-16T09:00:00.000Z" }),
	];
	const forward = buildAuditRows({ entries, events: [] }).map((row) => row.ref);
	const reversed = buildAuditRows({ entries: [...entries].reverse(), events: [] }).map((row) => row.ref);
	assert.deepEqual(forward, ["led-c", "led-a", "led-b"], "sorted by ts, then ref");
	assert.deepEqual(forward, reversed, "same rows in, same order out");
});

test("buildAuditRows sorts across sources by timestamp, not by source group", () => {
	const rows = buildAuditRows({
		entries: [ledgerEntry({ id: "late", ts: "2026-08-16T11:00:00.000Z" })],
		events: [startRow({ ts: "2026-08-16T09:00:00.000Z" })],
	});
	assert.deepEqual(
		rows.map((row) => row.ref),
		["call-1", "late"],
	);
});

// ── Serialization ───────────────────────────────────────────────────────────

test("csvEscape quotes only what needs it and doubles inner quotes", () => {
	assert.equal(csvEscape("plain"), "plain");
	assert.equal(csvEscape("a,b"), '"a,b"');
	assert.equal(csvEscape('say "hi"'), '"say ""hi"""');
	assert.equal(csvEscape("line\nbreak"), '"line\nbreak"');
	assert.equal(csvEscape("carriage\rreturn"), '"carriage\rreturn"');
});

test("toCsv keeps every row on the declared column count despite JSON details", () => {
	// `detail` is JSON and therefore full of commas and quotes. Unescaped, one
	// row shifts every column after it and the export is plausibly wrong.
	const rows = buildAuditRows({ entries: [ledgerEntry()], events: [] });
	const csv = toCsv(rows);
	const lines = csv.trimEnd().split("\r\n");
	assert.equal(lines[0], AUDIT_COLUMNS.join(","));
	assert.equal(lines.length, 2);

	// Parse the data line respecting RFC 4180 quoting.
	const fields = [];
	let field = "";
	let inQuotes = false;
	for (let index = 0; index < lines[1].length; index += 1) {
		const char = lines[1][index];
		if (inQuotes) {
			if (char === '"' && lines[1][index + 1] === '"') {
				field += '"';
				index += 1;
			} else if (char === '"') {
				inQuotes = false;
			} else {
				field += char;
			}
		} else if (char === '"') {
			inQuotes = true;
		} else if (char === ",") {
			fields.push(field);
			field = "";
		} else {
			field += char;
		}
	}
	fields.push(field);

	assert.equal(fields.length, AUDIT_COLUMNS.length, "detail must not leak extra columns");
	const record = Object.fromEntries(AUDIT_COLUMNS.map((column, index) => [column, fields[index]]));
	assert.equal(record.actor, "agent");
	assert.equal(record.ref, "led-1");
	assert.deepEqual(JSON.parse(record.detail).memory_id, "mem-1");
});

test("toJsonl emits one parseable row per line and nothing for no rows", () => {
	const rows = buildAuditRows({ entries: [ledgerEntry(), ledgerEntry({ id: "led-2", ts: "2026-08-16T10:00:01.000Z" })], events: [] });
	const jsonl = toJsonl(rows);
	assert.ok(jsonl.endsWith("\n"), "a trailing newline keeps appends line-aligned");
	const lines = jsonl.split("\n").filter(Boolean);
	assert.equal(lines.length, 2);
	assert.deepEqual(
		lines.map((line) => JSON.parse(line).ref),
		["led-1", "led-2"],
	);
	assert.equal(toJsonl([]), "", "no rows must not produce a blank line");
});

// ── View commands in the trail ──────────────────────────────────────────────

function viewRow(overrides = {}) {
	return {
		conversation_id: "conv-1",
		user_id: "demo",
		turn: 1,
		ts: "2026-08-16T10:00:00.100Z",
		event: {
			type: "view_command",
			tool_call_id: "call-9",
			reason: "these 12 memories cluster on the Malabar coast",
			destination: "/geo",
			entities: ["Malabar Coast", "Dali"],
			unresolved: ["Atlantis"],
		},
		...overrides,
	};
}

test("a view command becomes a durable row, so the move survives a reload", () => {
	// It was browser-memory only before: the person reloaded and every record of
	// where the conversation had taken them was gone.
	const rows = buildAuditRows({ entries: [], events: [viewRow()] });
	const view = rows.filter((row) => row.source === "view");
	assert.equal(view.length, 1);
	assert.equal(view[0].kind, "view_command");
	assert.equal(view[0].actor, "agent");
	assert.equal(view[0].ref, "call-9");
});

test("the row records the VALIDATED outcome, which its tool_call row cannot", () => {
	// The tool call carries the arguments — the names the model hoped existed.
	// This carries what the graph actually contained, and the two differ.
	const [row] = buildAuditRows({ entries: [], events: [viewRow()] }).filter((r) => r.source === "view");
	const detail = JSON.parse(row.detail);
	assert.equal(detail.destination, "/geo");
	assert.deepEqual(detail.entities, ["Malabar Coast", "Dali"]);
	assert.deepEqual(detail.unresolved, ["Atlantis"]);
	assert.equal(detail.reason, "these 12 memories cluster on the Malabar coast");
});

test("the row claims no verdict, because the seat never learns one", () => {
	// Whether the command applied or waited as a Follow is decided by the
	// authority ledger in the browser and reported to nobody. A field asserting
	// it would be the trail inventing the one fact it cannot have.
	const [row] = buildAuditRows({ entries: [], events: [viewRow()] }).filter((r) => r.source === "view");
	const detail = JSON.parse(row.detail);
	assert.deepEqual(Object.keys(detail).sort(), ["destination", "entities", "reason", "unresolved"]);
});

test("view_command is in the store filter, or the export would query past it", () => {
	// buildAuditRows can only shape what the read returned; a type missing from
	// this list is a row that is built and never fetched.
	assert.ok(AUDIT_EVENT_TYPES.includes("view_command"));
});

test("a view command and its tool call both appear, and sort deterministically", () => {
	const rows = buildAuditRows({
		entries: [],
		events: [startRow({ ts: "2026-08-16T10:00:00.000Z", event: { type: "tool_call_start", tool_call_id: "call-9", tool_name: "direct_view", args: { destination: "geo" } } }), viewRow()],
	});
	assert.deepEqual(
		rows.map((row) => row.source),
		["tool_call", "view"],
	);
});
