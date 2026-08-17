/**
 * The rules that make an agent's work on a todo attributable.
 *
 * The system has no assignee field, so every check here is load-bearing in a
 * way it would not be if it had one: the short id is the only handle, the
 * comment author is the only signature, and a claim is two calls that can
 * disagree. Every test was checked by deleting the line it covers and
 * confirming it went red.
 *
 * Run: npm run build && npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
	agentAuthor,
	agentComments,
	claimRefusal,
	composeClaimReport,
	composeEmptyListingReport,
	describeTodoFilters,
	formatTodoLine,
	parseTodoStatus,
	shortIdOf,
	TODO_STATUSES,
} from "../dist/todo-tools.js";

function todo(over = {}) {
	return {
		id: "9f3c1b20-0000-4000-8000-000000000001",
		seq_num: 7,
		project_prefix: "BOLT",
		project: null,
		user_id: "u",
		content: "Wire the geo cue",
		status: "todo",
		priority: "high",
		project_id: null,
		parent_id: null,
		contexts: [],
		tags: [],
		due_date: null,
		blocked_on: null,
		notes: null,
		created_at: "2026-08-16T09:00:00.000Z",
		updated_at: "2026-08-16T09:00:00.000Z",
		completed_at: null,
		blocked_by: [],
		related_memory_ids: [],
		comments: [],
		...over,
	};
}

function comment(over = {}) {
	return {
		id: "c1",
		todo_id: "t1",
		author: "agent:anthropic/claude",
		content: "Claimed. Plan: wire it",
		comment_type: "activity",
		created_at: "2026-08-16T10:00:00.000Z",
		updated_at: null,
		...over,
	};
}

// ── The status ladder, exactly as Rust has it ───────────────────────────────

test("the six statuses are the Rust enum, in workflow order", () => {
	assert.deepEqual([...TODO_STATUSES], [
		"backlog",
		"todo",
		"in_progress",
		"blocked",
		"done",
		"cancelled",
	]);
});

test("canonical spellings parse", () => {
	for (const status of TODO_STATUSES) assert.equal(parseTodoStatus(status), status);
	assert.equal(parseTodoStatus("  In_Progress  "), "in_progress");
});

test("the loose synonyms Rust accepts are REFUSED here", () => {
	// TodoStatus::from_str_loose takes these. Accepting them would teach the
	// model six spellings of one status, and the one it picks next is the one
	// the next endpoint rejects.
	for (const loose of ["doing", "waiting", "wont_do", "someday", "next", "complete"]) {
		assert.equal(parseTodoStatus(loose), null, loose);
	}
});

// ── The handle ──────────────────────────────────────────────────────────────

test("the short id is project prefix and sequence, as Todo::short_id builds it", () => {
	// Not served by the API — `short_id` is a Rust method, so /api/todos returns
	// seq_num and project_prefix and leaves this to the caller. It is the handle
	// the model quotes back and passes to the next call.
	assert.equal(shortIdOf(todo()), "BOLT-7");
});

test("a todo with a sequence but no project falls back to SHO", () => {
	assert.equal(shortIdOf(todo({ project_prefix: null })), "SHO-7");
});

test("a legacy todo with no sequence uses the uuid's first four characters", () => {
	// TodoId::short() is "SHO-" + &uuid.to_string()[..4], verbatim.
	assert.equal(shortIdOf(todo({ seq_num: 0 })), "SHO-9f3c");
});

// ── The line the model reads ────────────────────────────────────────────────

test("the line carries the id, status, priority and content", () => {
	assert.match(formatTodoLine(todo()), /^\[BOLT-7\] \(todo, high\) Wire the geo cue/);
});

test("a blocked todo shows BOTH halves of why", () => {
	// blocked_on is free text about who or what; blocked_by references real
	// todos. Showing one and hiding the other makes a blocked task look either
	// unexplained or unblocked.
	const line = formatTodoLine(
		todo({ status: "blocked", blocked_on: "waiting on the vendor", blocked_by: ["a", "b"] }),
	);
	assert.match(line, /blocked on: waiting on the vendor/);
	assert.match(line, /blocked by 2 todo\(s\)/);
});

test("the line never carries an embedding, whatever the backend sent", () => {
	// /api/todos on this branch serialises 384 floats per todo (~287KB for
	// fifty). Everything a tool returns becomes model context.
	const line = formatTodoLine(todo({ embedding: [0.1, 0.2, 0.3] }));
	assert.doesNotMatch(line, /0\.1/);
	assert.ok(line.length < 200);
});

test("a subtask says so, because its parent changes what it means", () => {
	assert.match(formatTodoLine(todo({ parent_id: "p1" })), /subtask/);
});

// ── The signature ───────────────────────────────────────────────────────────

test("the author names the model, not merely 'agent'", () => {
	// Two models moving the same todo on two different days is exactly the
	// distinction an audit exists for, and the seat knows which one is running.
	assert.equal(agentAuthor({ provider: "anthropic", id: "claude-x", name: "Claude" }), "agent:anthropic/claude-x");
});

test("only agent-authored comments count as an agent's trace", () => {
	const found = agentComments(
		todo({ comments: [comment(), comment({ id: "c2", author: "varun" })] }),
	);
	assert.equal(found.length, 1);
	assert.equal(found[0].author, "agent:anthropic/claude");
});

// ── What may not be claimed ─────────────────────────────────────────────────

const ME = "agent:anthropic/me";

test("settled work cannot be claimed", () => {
	for (const status of ["done", "cancelled"]) {
		assert.match(claimRefusal(todo({ status }), ME), new RegExp(`already ${status}`));
	}
});

test("work another model has claimed cannot be taken", () => {
	const refusal = claimRefusal(
		todo({ status: "in_progress", comments: [comment({ author: "agent:openai/other" })] }),
		ME,
	);
	// The refusal has to say WHO and WHEN, because there is no assignee field to
	// look at — the comment is the only evidence the claim exists.
	assert.match(refusal, /agent:openai\/other/);
	assert.match(refusal, /2026-08-16T10:00:00.000Z/);
	assert.match(refusal, /no assignee field/);
});

test("a model re-claiming its own in-progress work is not refused", () => {
	assert.equal(
		claimRefusal(todo({ status: "in_progress", comments: [comment({ author: ME })] }), ME),
		null,
	);
});

test("in-progress work with no agent trace is claimable — a person may have started it", () => {
	// A human moving a todo to in_progress leaves a comment authored by their
	// user id, or none at all. Refusing here would lock the model out of every
	// task anyone had ever touched.
	assert.equal(
		claimRefusal(todo({ status: "in_progress", comments: [comment({ author: "varun" })] }), ME),
		null,
	);
});

test("ordinary open work is claimable", () => {
	for (const status of ["backlog", "todo", "blocked"]) {
		assert.equal(claimRefusal(todo({ status }), ME), null, status);
	}
});

// ── Two calls that can disagree ─────────────────────────────────────────────

test("a claim whose comment failed is reported as a claim with NO attribution", () => {
	// The status moved and the signature did not. A model told "claimed" would
	// work on under an attribution that does not exist.
	const text = composeClaimReport({
		shortId: "BOLT-7",
		previousStatus: "todo",
		statusChanged: true,
		commentError: "backend error 503",
		author: ME,
	});
	assert.match(text, /todo → in_progress/);
	assert.match(text, /WARNING/);
	assert.match(text, /backend error 503/);
	assert.match(text, /Nothing on \[BOLT-7\] records that you took it/);
});

test("a clean claim says who signed it", () => {
	const text = composeClaimReport({
		shortId: "BOLT-7",
		previousStatus: "todo",
		statusChanged: true,
		commentError: null,
		author: ME,
	});
	assert.match(text, new RegExp(`Claim recorded on the todo as ${ME}`));
	assert.doesNotMatch(text, /WARNING/);
});

test("re-claiming already-in-progress work does not report a status change that did not happen", () => {
	const text = composeClaimReport({
		shortId: "BOLT-7",
		previousStatus: "in_progress",
		statusChanged: false,
		commentError: null,
		author: ME,
	});
	assert.match(text, /was already in_progress; status unchanged/);
	assert.doesNotMatch(text, /→/);
});

// ── An empty listing that is not a dead end ─────────────────────────────────
// "No todos match those filters" cannot tell an empty board from an over-narrow
// query, and those call for opposite next moves. Mutations: deleting the
// settled-work default line in describeTodoFilters; collapsing the
// narrowed/not-narrowed branch in composeEmptyListingReport to one string;
// dropping any single filter from describeTodoFilters.

test("the settled-work exclusion is named even though the model never asked for it", () => {
	// This default is the single most likely cause of a surprising empty
	// listing and the one thing the model has no way to see it sent.
	const applied = describeTodoFilters({});
	assert.equal(applied.length, 1);
	assert.match(applied[0], /done and cancelled were excluded/);
});

test("an explicit status replaces the default rather than sitting beside it", () => {
	const applied = describeTodoFilters({ status: ["done"] });
	assert.deepEqual(applied, ["status done"]);
	assert.ok(!applied.some((line) => line.includes("excluded")));
});

test("every filter the model can send is echoed back to it", () => {
	const applied = describeTodoFilters({
		status: ["todo", "blocked"],
		project: "BOLT",
		context: "@computer",
		priority: "urgent",
		query: "vendor",
	});
	assert.deepEqual(applied, [
		"status todo or blocked",
		'project "BOLT"',
		'context "@computer"',
		"priority urgent",
		'text matching "vendor"',
	]);
});

test("an unfiltered empty listing tells the model to STOP, not to retry", () => {
	// Telling a model to "try broader filters" when it used none is how a tool
	// teaches a loop.
	const text = composeEmptyListingReport(describeTodoFilters({}));
	assert.match(text, /no open todos/);
	assert.match(text, /rather than searching again/);
	assert.doesNotMatch(text, /Drop the narrowing filters/);
});

test("a filtered empty listing tells the model to widen before concluding", () => {
	const text = composeEmptyListingReport(describeTodoFilters({ project: "BOLT" }));
	assert.match(text, /Drop the narrowing filters/);
	assert.doesNotMatch(text, /no open todos/);
});

test("the empty report always states what was actually asked for", () => {
	const text = composeEmptyListingReport(describeTodoFilters({ project: "BOLT", priority: "urgent" }));
	assert.match(text, /Filters in force/);
	assert.match(text, /BOLT/);
	assert.match(text, /urgent/);
});
