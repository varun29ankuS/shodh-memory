/**
 * LATE RESOLUTION — the segment of the view loop that was still open.
 *
 * `direct_view` waits two seconds and then reports honestly, which for a held
 * offer is "WAITING". The person answers minutes later, and until now the only
 * thing that changed was the audit trail: the model's belief stayed frozen at
 * WAITING, so it would go on calling an accepted offer pending and could learn
 * otherwise only by happening to call `inspect_view` again.
 *
 * The property under test is the dedupe, and it cuts both ways. A verdict the
 * tool already reported must NOT be announced again — that would re-tell the
 * model something it acted on two seconds ago, every turn, forever. A verdict
 * the tool never reported, because it timed out or because it arrived after,
 * must NOT be swallowed. `notified` is the line between them, and it is exact
 * rather than heuristic: it holds precisely what `await` handed back.
 *
 * Every test here was checked by deleting the line it covers and confirming it
 * went red. The mutation is named on each.
 *
 * Run: npm run build && node --test eval/view-news.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { composeViewNews, VIEW_OUTCOME_STATES, ViewLink } from "../dist/view-link.js";

const VIEW = {
	destination: "/graph",
	profile: "demo",
	cue: null,
	focus: null,
	claimed: [],
	offers: [],
};

function body(outcomes) {
	return { probe_id: null, outcomes, view: VIEW };
}

const outcome = (dimension, state, callId = "call-1") => ({ tool_call_id: callId, dimension, state });

// ── The dedupe ──────────────────────────────────────────────────────────────

test("a verdict the tool already reported is NOT announced again next turn", async () => {
	// Mutation: delete the `notify()` wrapper in `await`, or the
	// `!ask?.notified.has(...)` term in the news filter.
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "the map is where this lives");
	link.report("conv-1", body([outcome("destination", "applied")]));

	const told = await link.await("call-1");
	assert.equal(told.length, 1, "the tool must still get its own verdict");
	assert.deepEqual(link.drainNews("conv-1"), [], "and it must not be repeated as news");
});

test("a verdict that arrives AFTER the tool returned becomes news", async () => {
	// The case the whole mechanism exists for: offered inside the window,
	// accepted long after it closed.
	const link = new ViewLink(20);
	link.open("conv-1", "call-1", "these cluster on the coast");
	link.report("conv-1", body([outcome("destination", "offered")]));
	assert.deepEqual(await link.await("call-1"), [outcome("destination", "offered")]);

	link.report("conv-1", body([outcome("destination", "followed")]));
	const news = link.drainNews("conv-1");
	assert.equal(news.length, 1);
	assert.equal(news[0].state, "followed");
	assert.equal(news[0].reason, "these cluster on the coast", "the ask's own words travel with its verdict");
});

test("offered is never news — it is the state the catch-up exists to resolve", () => {
	// Mutation: drop `isTerminal(...)` from the filter. The model would be told
	// "still waiting" every turn about an offer it already knows is waiting.
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "r.");
	link.report("conv-1", body([outcome("cue", "offered")]));
	assert.deepEqual(link.drainNews("conv-1"), []);
});

test("an ask that TIMED OUT makes even its first verdict news", async () => {
	// The model was told VERDICT NOT KNOWN, so nothing about this ask has been
	// said to it. Mutation: mark notified on the timeout path too.
	const link = new ViewLink(1);
	link.open("conv-1", "call-1", "worth a look.");
	assert.equal(await link.await("call-1"), null);

	link.report("conv-1", body([outcome("destination", "applied")]));
	const news = link.drainNews("conv-1");
	assert.equal(news.length, 1);
	assert.equal(news[0].state, "applied");
});

test("only the dimensions the tool was told about are suppressed", async () => {
	// One command lands on several axes and they can land differently. A report
	// that resolved the destination must not silence a later verdict on the cue.
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "r.");
	link.report("conv-1", body([outcome("destination", "applied")]));
	await link.await("call-1");

	link.report("conv-1", body([outcome("cue", "declined")]));
	const news = link.drainNews("conv-1");
	assert.equal(news.length, 1);
	assert.equal(news[0].dimension, "cue");
});

// ── Whose news it is ────────────────────────────────────────────────────────

test("news survives the seat forgetting the ask — a restart mid-offer still reports", () => {
	// The registry dies with the process; an offer on screen does not. The route
	// validates such a report against the DURABLE store before it reaches here,
	// so an unknown ask is a legitimate late verdict rather than a forgery.
	// Mutation: put `if (!ask) continue;` back above the news computation.
	const link = new ViewLink(50);
	link.report("conv-1", body([outcome("destination", "followed", "call-from-before-the-restart")]));
	const news = link.drainNews("conv-1");
	assert.equal(news.length, 1);
	assert.equal(news[0].tool_call_id, "call-from-before-the-restart");
	assert.equal(news[0].reason, "", "no ask means no reason — empty, never invented");
});

test("another conversation's ask never becomes this conversation's news", () => {
	// Mutation: delete the `ask.conversationId !== conversationId` guard.
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "r.");
	link.report("conv-2", body([outcome("destination", "followed")]));
	assert.deepEqual(link.drainNews("conv-2"), []);
	assert.deepEqual(link.drainNews("conv-1"), []);
});

test("draining takes the news away, so a verdict is announced exactly once", () => {
	// Mutation: delete `this.news.delete(conversationId)`. Every turn for the
	// rest of the session would re-announce the same acceptance.
	const link = new ViewLink(50);
	link.report("conv-1", body([outcome("destination", "declined", "c9")]));
	assert.equal(link.drainNews("conv-1").length, 1);
	assert.deepEqual(link.drainNews("conv-1"), []);
});

test("a conversation that goes away takes its undelivered news with it", () => {
	const link = new ViewLink(50);
	link.report("conv-1", body([outcome("destination", "declined", "c9")]));
	link.forget("conv-1");
	assert.deepEqual(link.drainNews("conv-1"), []);
});

test("the queue is bounded, and it is the OLDEST that goes", () => {
	// A model told about the fortieth-oldest lapsed offer instead of the newest
	// accepted one would be worse informed, not better.
	const link = new ViewLink(50);
	for (let index = 0; index < 40; index += 1) {
		link.report("conv-1", body([outcome("destination", "expired", "call-" + index)]));
	}
	const news = link.drainNews("conv-1");
	assert.ok(news.length <= 16, "queue grew to " + news.length);
	assert.equal(news[news.length - 1].tool_call_id, "call-39", "the newest must survive");
});

// ── What the model reads ────────────────────────────────────────────────────

test("no news is no block at all, not an empty heading", () => {
	// An empty section every turn teaches the model to skip a heading that
	// occasionally matters. Mutation: return "" instead of null.
	assert.equal(composeViewNews([]), null);
});

test("an accepted offer is reported in the model's own words", () => {
	const text = composeViewNews([
		{ tool_call_id: "c1", dimension: "destination", state: "followed", reason: "these cluster on the coast" },
	]);
	assert.match(text, /ACCEPTED/);
	assert.match(text, /the destination/);
	assert.match(text, /these cluster on the coast/);
});

test("a refusal carries the instruction not to route around it", () => {
	// Knowing it was declined must not become a way to retry around the person.
	// Mutation: delete the `refused` clause.
	const text = composeViewNews([{ tool_call_id: "c1", dimension: "cue", state: "declined", reason: "r" }]);
	assert.match(text, /REFUSED/);
	assert.match(text, /Do not re-issue/);
});

test("an acceptance does NOT carry the refusal warning", () => {
	// Mutation: make the clause unconditional. The model would be warned off
	// re-issuing a request that was granted, which is a different instruction.
	const text = composeViewNews([{ tool_call_id: "c1", dimension: "cue", state: "followed", reason: "r" }]);
	assert.doesNotMatch(text, /Do not re-issue/);
});

test("a lapse is not reported as a refusal", () => {
	// The distinction the whole return path exists to preserve: "they said no"
	// and "they never saw it" are different facts about the same person.
	const text = composeViewNews([{ tool_call_id: "c1", dimension: "frame", state: "expired", reason: "r" }]);
	assert.match(text, /LAPSED/);
	assert.match(text, /neither accepted nor refused/);
	assert.doesNotMatch(text, /REFUSED/);
});

test("a verdict that arrives after a NOT KNOWN says it is a correction", () => {
	const text = composeViewNews([{ tool_call_id: "c1", dimension: "destination", state: "applied", reason: "r" }]);
	assert.match(text, /you were told at the time that the verdict was not known/);
});

test("every state in the closed set produces a line, so none can be silently dropped", () => {
	for (const state of VIEW_OUTCOME_STATES) {
		const text = composeViewNews([{ tool_call_id: "c1", dimension: "cue", state, reason: "r" }]);
		assert.ok(text && text.includes("- "), state + " produced no line");
		assert.ok(!text.includes("undefined"), state + " rendered as undefined");
	}
});

test("a missing reason is omitted, never rendered as an empty quotation", () => {
	const text = composeViewNews([{ tool_call_id: "c1", dimension: "cue", state: "followed", reason: "" }]);
	assert.doesNotMatch(text, /your reason was ""/);
});

test("two tabs reporting the same verdict announce it once", () => {
	// Mutation: delete the `latest` collapse in drainNews. One conversation open
	// in two windows would tell the model the same thing twice in one block.
	const link = new ViewLink(50);
	link.report("conv-1", body([outcome("destination", "followed", "c1")]));
	link.report("conv-1", body([outcome("destination", "followed", "c1")]));
	assert.equal(link.drainNews("conv-1").length, 1);
});

test("a dimension that moved twice reports where it ENDED, not the sequence", () => {
	// Offered, then superseded by the model's own later ask, then followed: the
	// model needs the state the person left it in, not a replay of the clicks.
	const link = new ViewLink(50);
	link.report("conv-1", body([outcome("cue", "superseded", "c1")]));
	link.report("conv-1", body([outcome("cue", "followed", "c1")]));
	const news = link.drainNews("conv-1");
	assert.equal(news.length, 1);
	assert.equal(news[0].state, "followed");
});
