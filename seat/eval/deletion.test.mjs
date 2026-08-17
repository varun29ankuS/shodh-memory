/**
 * The two verbs that destroy, and the record that has to outlive what they took.
 *
 * WHY THIS FILE IS DIFFERENT FROM THE OTHER TOOL TESTS. Every other rule in this
 * seat is checkable after the fact: a wrong todo status can be read off the
 * board, a missing citation can be compared against the store. A deletion's
 * subject is gone by the time anybody reviews it, so a defect in the record is
 * undetectable afterwards BY CONSTRUCTION — the entry is the only surviving
 * evidence, and if it is wrong nothing else contradicts it. That is the whole
 * reason the record's shape is a pure function with tests rather than a few
 * fields assembled inline at the call site.
 *
 * Every test here was checked by deleting the line it covers and confirming it
 * went red. The mutation is named on each block.
 *
 * Run: npm run build && node --test eval/deletion.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
	composeDeletionData,
	contentDigest,
	deletionNotPerformed,
	deletionRevertRefusal,
	DELETION_PREVIEW_CHARS,
} from "../dist/ledger.js";
import {
	composeForgetFailureReport,
	composeForgetReport,
	forgetRefusal,
	memoryCitationKey,
} from "../dist/memory-tools.js";
import {
	composeDeleteTodoFailureReport,
	composeDeleteTodoReport,
	deleteTodoRefusal,
} from "../dist/todo-tools.js";

function todo(over = {}) {
	return {
		id: "9f3c1b20-0000-4000-8000-000000000001",
		seq_num: 7,
		project_prefix: "BOLT",
		project: "Bolt",
		user_id: "u",
		content: "Ship the thing",
		status: "todo",
		priority: "medium",
		project_id: null,
		parent_id: null,
		contexts: [],
		tags: [],
		due_date: null,
		blocked_on: null,
		notes: null,
		created_at: "2026-08-01T00:00:00Z",
		updated_at: "2026-08-01T00:00:00Z",
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
		todo_id: "9f3c1b20-0000-4000-8000-000000000001",
		author: "agent:anthropic/other-model",
		content: "Taking this",
		comment_type: "activity",
		created_at: "2026-08-02T00:00:00Z",
		updated_at: null,
		...over,
	};
}

const DELETION_INPUT = {
	target: "memory",
	targetId: "9f3c1b20-0000-4000-8000-000000000001",
	shortId: "9f3c1b20",
	content: "The vendor contract renews on the first of March.",
	classification: "Decision",
	tags: ["contracts"],
	createdAt: "2026-07-01T00:00:00Z",
	collateral: { relation: "orphaned", ids: ["child-1"] },
	reason: "The user asked me to forget it.",
	author: "agent:anthropic/claude-x",
};

const CASCADE = {
	relation: "cascade_deleted",
	destroyed: [
		{ id: "sub-1", shortId: "BOLT-8", content: "Sub one" },
		{ id: "sub-2", shortId: "BOLT-9", content: "Sub two" },
	],
};

// ── What a deletion record must contain ─────────────────────────────────────
// Mutations: drop `content_length`; drop `content_sha256`; drop `reason`; drop
// `author`; drop `collateral`; drop `created_at`.

test("the record answers who, when, which and why — not just which", () => {
	// The load-bearing test. An entry carrying only an id proves a deletion
	// happened and leaves "what was destroyed" permanently unanswerable, because
	// the object it named is the thing that no longer exists.
	const data = composeDeletionData(DELETION_INPUT);
	assert.equal(data.target_id, "9f3c1b20-0000-4000-8000-000000000001");
	assert.equal(data.short_id, "9f3c1b20");
	assert.equal(data.author, "agent:anthropic/claude-x");
	assert.equal(data.reason, "The user asked me to forget it.");
	assert.equal(data.created_at, "2026-07-01T00:00:00Z");
	assert.equal(data.classification, "Decision");
	assert.deepEqual(data.tags, ["contracts"]);
});

test("the preview is bounded and the full length is recorded beside it", () => {
	// The privacy line, made checkable: a reviewer must be able to see how much
	// of the destroyed body the entry is NOT showing them. A preview with no
	// length reads as the whole thing.
	const long = "x".repeat(1000);
	const data = composeDeletionData({ ...DELETION_INPUT, content: long });
	assert.equal(data.content_preview.length, DELETION_PREVIEW_CHARS);
	assert.equal(data.content_length, 1000);
});

test("short content is not padded and its length is exact", () => {
	const data = composeDeletionData(DELETION_INPUT);
	assert.equal(data.content_preview, DELETION_INPUT.content);
	assert.equal(data.content_length, DELETION_INPUT.content.length);
});

test("the checksum is over the whole content, so a reviewer holding a copy can prove the match", () => {
	// This is what the preview cannot do: verification without disclosure. It has
	// to be over the FULL string — a digest of the 200-char preview would match
	// nothing anybody could reproduce.
	const long = `${"x".repeat(1000)}tail`;
	const data = composeDeletionData({ ...DELETION_INPUT, content: long });
	assert.equal(data.content_sha256, contentDigest(long));
	assert.notEqual(data.content_sha256, contentDigest(data.content_preview));
});

test("the digest is over the exact string, unnormalised", () => {
	// Trimming or case-folding would produce a digest that matches nothing a
	// reviewer could compute from the copy they hold.
	assert.notEqual(contentDigest(" a "), contentDigest("a"));
	assert.notEqual(contentDigest("A"), contentDigest("a"));
	assert.match(contentDigest("a"), /^[0-9a-f]{64}$/);
});

test("collateral distinguishes what was destroyed from what was merely orphaned", () => {
	// A memory's children SURVIVE with a dangling parent; a todo's subtasks are
	// deleted outright. Recording both as "affected" would tell a reviewer the
	// wrong thing in one of the two cases, every time.
	const orphaning = composeDeletionData(DELETION_INPUT);
	assert.equal(orphaning.collateral.relation, "orphaned");
	assert.deepEqual(orphaning.collateral.ids, ["child-1"]);

	const cascading = composeDeletionData({ ...DELETION_INPUT, target: "todo", collateral: CASCADE });
	assert.equal(cascading.collateral.relation, "cascade_deleted");
});

test("cascade-destroyed items carry a preview, because their ids will resolve to nothing", () => {
	// The same defect as an id-only entry, one level down: "BOLT-7 took three
	// subtasks with it" records that something was lost without recording what,
	// and the subtasks are not there to be consulted. Orphaned children get ids
	// alone because they survive and remain readable — the asymmetry is the point.
	const data = composeDeletionData({ ...DELETION_INPUT, target: "todo", collateral: CASCADE });
	assert.deepEqual(data.collateral.destroyed, [
		{ id: "sub-1", short_id: "BOLT-8", content_preview: "Sub one" },
		{ id: "sub-2", short_id: "BOLT-9", content_preview: "Sub two" },
	]);
});

test("a cascaded item's preview is bounded by the same rule as the target's", () => {
	// Otherwise deleting one parent with ten long subtasks would write more of the
	// user's text into the ledger than deleting the parent alone ever could.
	const data = composeDeletionData({
		...DELETION_INPUT,
		target: "todo",
		collateral: { relation: "cascade_deleted", destroyed: [{ id: "s", shortId: "B-1", content: "y".repeat(900) }] },
	});
	assert.equal(data.collateral.destroyed[0].content_preview.length, DELETION_PREVIEW_CHARS);
});

test("the record copies its inputs, so a later mutation of the caller's arrays cannot rewrite history", () => {
	// An append-only ledger that shares array references with its caller is not
	// append-only.
	const tags = ["contracts"];
	const ids = ["child-1"];
	const data = composeDeletionData({ ...DELETION_INPUT, tags, collateral: { relation: "orphaned", ids } });
	tags.push("leaked");
	ids.push("leaked");
	assert.deepEqual(data.tags, ["contracts"]);
	assert.deepEqual(data.collateral.ids, ["child-1"]);
	// The cascade branch is NOT asserted here on purpose: it is built with
	// `.map()`, which allocates a new array and new objects unconditionally, so an
	// aliasing assertion against it could not be made to fail by any edit to the
	// composer. A test that cannot go red is invisible to a failing-test sweep and
	// worse than no test — what that branch actually needs proving is the content
	// it carries, which the two preview tests above cover.
});

// ── A deletion cannot be reverted, and says so ──────────────────────────────
// Mutations: return null from deletionRevertRefusal; drop the checksum from the
// message; drop the "fabrication" clause.

test("reverting a deletion is refused, and the refusal explains what is missing rather than just saying no", () => {
	// The ledger's own rule is that a field it does not hold is never invented.
	// A "restore" would have to invent the whole object — the 200-char preview
	// written back under a new id with a new creation time, no embeddings, no
	// graph episode — which reads as recovery and is a forgery.
	const message = deletionRevertRefusal({
		id: "evt-1",
		ts: "2026-08-17T00:00:00Z",
		kind: "deletion",
		actor: "agent",
		scope: "user",
		user_id: "u",
		conversation_id: "c",
		turn: 1,
		data: composeDeletionData(DELETION_INPUT),
	});
	assert.match(message, /cannot be reverted/);
	assert.match(message, /9f3c1b20/);
	assert.match(message, /fabrication/);
	assert.match(message, new RegExp(contentDigest(DELETION_INPUT.content).slice(0, 16)));
});

test("a deletion that did not happen is compensated as its own kind, never as 'nothing to do'", () => {
	// `none` means "there was nothing to undo". Using it here would tell a
	// reviewer the deletion was real and simply had no compensating action —
	// the exact opposite of the fact being recorded.
	const compensation = deletionNotPerformed(composeDeletionData(DELETION_INPUT), "backend 503");
	assert.equal(compensation.kind, "deletion_not_performed");
	assert.equal(compensation.target, "memory");
	assert.equal(compensation.target_id, DELETION_INPUT.targetId);
	assert.equal(compensation.error, "backend 503");
});

// ── What forget_memory will not do ──────────────────────────────────────────
// Mutations: return null for `key === null`; return null for `resolved ===
// null`; delete the scope check.

const SHOWN = { id: "9f3c1b20-0000-4000-8000-000000000001", scope: "user" };

test("an id in a shape the model was never shown is refused before anything is looked up", () => {
	// A full uuid included: the model is never shown one, so an id in that shape
	// is one it constructed.
	const uuid = "9f3c1b20-0000-4000-8000-000000000001";
	assert.equal(memoryCitationKey(uuid), null);
	const message = forgetRefusal(uuid, memoryCitationKey(uuid), null);
	assert.match(message, /not a memory id in the form this seat accepts/);
});

test("a memory the model has not been shown this turn is REFUSED, not deleted", () => {
	// The security boundary, not a nicety: because resolution is required, the id
	// that reaches the backend URL is a uuid this process already held, and no
	// model-supplied string is ever interpolated into a destructive request.
	const message = forgetRefusal("[mem:deadbeef]", "deadbeef", null);
	assert.match(message, /not a memory you have been shown/);
	assert.match(message, /Recall it first/);
});

test("a memory in the assistant's own learning scope is refused", () => {
	// Separate RocksDB stores. A person asking to forget something means their
	// own corpus, and only this map can tell the two apart.
	const message = forgetRefusal("[mem:9f3c1b20]", "9f3c1b20", { id: SHOWN.id, scope: "harness" });
	assert.match(message, /assistant's own learning scope/);
});

test("a shown user-scope memory is not refused", () => {
	assert.equal(forgetRefusal("[mem:9f3c1b20]", "9f3c1b20", SHOWN), null);
});

// ── What forget_memory reports ──────────────────────────────────────────────
// Mutations: drop the orphan paragraph; make it unconditional; drop the ledger
// id; drop the graph-episode sentence.

test("the report names the side effects the backend performs silently", () => {
	// forget() also removes the vector entry, the BM25 entry and the graph
	// episode with its sourced relations. A model told only "deleted" would go on
	// to explain a later retrieval gap as a search failure.
	const text = composeForgetReport({
		shortId: "9f3c1b20",
		classification: "Decision",
		preview: "The vendor contract renews",
		orphanedChildren: 0,
		ledgerEventId: "evt-1",
	});
	assert.match(text, /graph episode/);
	assert.match(text, /evt-1/);
});

test("orphaned children are reported, because nothing else surfaces them", () => {
	// The children are NOT deleted; their parent_id is simply left dangling, and
	// no endpoint reports that.
	const text = composeForgetReport({
		shortId: "9f3c1b20",
		classification: "Decision",
		preview: "p",
		orphanedChildren: 2,
		ledgerEventId: "evt-1",
	});
	assert.match(text, /2 child memories still exist/);
	assert.match(text, /NOT deleted/);
});

test("a memory with no children carries no orphan warning", () => {
	const text = composeForgetReport({
		shortId: "9f3c1b20",
		classification: "Decision",
		preview: "p",
		orphanedChildren: 0,
		ledgerEventId: "evt-1",
	});
	assert.doesNotMatch(text, /child/);
});

test("a failed deletion says the memory is INTACT, never an optimistic success", () => {
	// The window the append-first ordering creates. The model's one job here is
	// to not report a deletion that did not happen.
	const text = composeForgetFailureReport({
		shortId: "9f3c1b20",
		error: "backend 503",
		ledgerEventId: "evt-1",
		compensationError: null,
	});
	assert.match(text, /was NOT deleted/);
	assert.match(text, /still recallable/);
	assert.doesNotMatch(text, /WARNING/);
});

test("a failed deletion whose correction ALSO failed says the ledger now overstates", () => {
	// Two different states of the record, and a reviewer's reading depends on
	// which one it is in — so the model is told which.
	const text = composeForgetFailureReport({
		shortId: "9f3c1b20",
		error: "backend 503",
		ledgerEventId: "evt-1",
		compensationError: "disk full",
	});
	assert.match(text, /WARNING/);
	assert.match(text, /disk full/);
	assert.match(text, /overstates/);
});

// ── What delete_todo will not do ────────────────────────────────────────────
// Mutations: delete the in_progress claim branch; delete the cascade branch;
// ignore the `cascade` flag; filter out the author's own comments.

const AUTHOR = "agent:anthropic/claude-x";

test("a todo another agent has claimed is not deletable — the delete would destroy the claim too", () => {
	const message = deleteTodoRefusal(
		todo({ status: "in_progress", comments: [comment()] }),
		[],
		AUTHOR,
		false,
	);
	assert.match(message, /claimed by agent:anthropic\/other-model/);
	assert.match(message, /Ask the user/);
});

test("this model's own claim does not block it from deleting", () => {
	// Otherwise a model could never delete a todo it had claimed itself, which is
	// the ordinary case.
	assert.equal(
		deleteTodoRefusal(todo({ status: "in_progress", comments: [comment({ author: AUTHOR })] }), [], AUTHOR, false),
		null,
	);
});

test("the first delete of a parent is refused and NAMES the subtasks it would destroy", () => {
	// There is no dry-run endpoint and no listing shows a todo's children, so
	// this refusal is the only warning that exists.
	const message = deleteTodoRefusal(
		todo(),
		[todo({ id: "s1", seq_num: 8, content: "Sub one" }), todo({ id: "s2", seq_num: 9, content: "Sub two" })],
		AUTHOR,
		false,
	);
	assert.match(message, /2 subtask/);
	assert.match(message, /\[BOLT-8\] Sub one/);
	assert.match(message, /\[BOLT-9\] Sub two/);
	assert.match(message, /cascade/);
});

test("cascade acknowledged lets the same call proceed", () => {
	assert.equal(deleteTodoRefusal(todo(), [todo({ id: "s1", seq_num: 8 })], AUTHOR, true), null);
});

test("a childless todo needs no cascade acknowledgement", () => {
	// The confirmation is about the hidden destruction, not about deletion in
	// general; demanding it every time would train the model to always pass it.
	assert.equal(deleteTodoRefusal(todo(), [], AUTHOR, false), null);
});

// ── What delete_todo reports ────────────────────────────────────────────────
// Mutations: drop the comment count; make it unconditional; drop the cascade
// line; drop the ledger id.

test("the report counts the destroyed comments, which are the attribution history", () => {
	// Every signed record of who touched this task was a comment, and the delete
	// took all of them. "Deleted BOLT-7" alone understates what happened.
	const text = composeDeleteTodoReport({
		shortId: "BOLT-7",
		content: "Ship the thing",
		status: "todo",
		commentCount: 4,
		cascaded: [],
		ledgerEventId: "evt-2",
	});
	assert.match(text, /4 comment\(s\) were destroyed/);
	assert.match(text, /who worked on this task/);
	assert.match(text, /evt-2/);
});

test("a todo with no comments carries no comment line", () => {
	const text = composeDeleteTodoReport({
		shortId: "BOLT-7",
		content: "c",
		status: "todo",
		commentCount: 0,
		cascaded: [],
		ledgerEventId: "evt-2",
	});
	assert.doesNotMatch(text, /comment/);
});

test("cascaded subtasks are named back, so the model can tell the user what went", () => {
	const text = composeDeleteTodoReport({
		shortId: "BOLT-7",
		content: "c",
		status: "todo",
		commentCount: 0,
		cascaded: ["BOLT-8", "BOLT-9"],
		ledgerEventId: "evt-2",
	});
	assert.match(text, /\[BOLT-8\], \[BOLT-9\]/);
});

test("a request that never reached the backend says the board is unchanged", () => {
	const text = composeDeleteTodoFailureReport({
		shortId: "BOLT-7",
		error: "backend 503",
		alreadyGone: false,
		ledgerEventId: "evt-2",
		compensationError: null,
	});
	assert.match(text, /was NOT deleted/);
	assert.match(text, /still on the board/);
	assert.doesNotMatch(text, /WARNING/);
});

test("a todo another session already removed is NOT reported as still on the board", () => {
	// The opposite fact, and the one a shared message would get wrong. The todo
	// really is gone — it was deleted between this tool's read and its write — so
	// "still on the board" would be false, and "you deleted it" would be a
	// misattribution in the ledger's own subject matter.
	const text = composeDeleteTodoFailureReport({
		shortId: "BOLT-7",
		error: "the backend reported no such todo, so this call removed nothing",
		alreadyGone: true,
		ledgerEventId: "evt-2",
		compensationError: null,
	});
	assert.doesNotMatch(text, /still on the board/);
	assert.match(text, /already been removed/);
	assert.match(text, /not by your hand/);
});

test("a failed todo deletion whose correction failed warns that the ledger overstates", () => {
	const text = composeDeleteTodoFailureReport({
		shortId: "BOLT-7",
		error: "backend 503",
		alreadyGone: false,
		ledgerEventId: "evt-2",
		compensationError: "disk full",
	});
	assert.match(text, /WARNING/);
	assert.match(text, /overstates/);
});
