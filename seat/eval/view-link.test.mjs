/**
 * The return leg: what the browser is allowed to say, and what the model is told
 * it means.
 *
 * The property under test throughout is HONESTY UNDER ABSENCE. Almost every
 * failure this module can have is the same failure — a verdict that did not
 * arrive being rendered as one that did — so most of these tests are about what
 * happens when nothing answers.
 *
 * Run: node --test eval/view-link.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
	composeVerdict,
	composeViewReport,
	describeOutcome,
	isHonoured,
	isTerminal,
	MAX_OUTCOMES_PER_REPORT,
	parseViewReport,
	VIEW_DIMENSIONS,
	VIEW_OUTCOME_STATES,
	ViewLink,
} from "../dist/view-link.js";

const VIEW = {
	destination: "/graph",
	profile: "demo",
	cue: { text: "Dali", entities: ["Dali"], author: "agent" },
	focus: null,
	claimed: [],
	offers: [],
};

function body(overrides = {}) {
	return { probe_id: null, outcomes: [], view: VIEW, ...overrides };
}

// ── The wire contract ───────────────────────────────────────────────────────

test("a well-formed report parses into exactly what was sent", () => {
	const parsed = parseViewReport(
		body({ outcomes: [{ tool_call_id: "call-1", dimension: "destination", state: "offered" }] }),
	);
	assert.ok("report" in parsed, parsed.error);
	assert.deepEqual(parsed.report.outcomes, [
		{ tool_call_id: "call-1", dimension: "destination", state: "offered" },
	]);
	assert.equal(parsed.report.view.destination, "/graph");
	assert.equal(parsed.report.probe_id, null);
});

test("a body that is not an object is refused, not coerced", () => {
	for (const raw of [null, "{}", 7, [], undefined]) {
		const parsed = parseViewReport(raw);
		assert.ok("error" in parsed, `${JSON.stringify(raw)} must not parse`);
	}
});

test("an unknown dimension is refused AND the valid ones are named", () => {
	// The browser and the seat keep two copies of this list (no import path
	// between the packages). Drift must surface here, loudly, on the first
	// report — not as a mislabelled row in an audit trail nobody re-reads.
	const parsed = parseViewReport(
		body({ outcomes: [{ tool_call_id: "c", dimension: "lens", state: "applied" }] }),
	);
	assert.ok("error" in parsed);
	for (const dimension of VIEW_DIMENSIONS) assert.match(parsed.error, new RegExp(dimension));
});

test("an unknown state is refused rather than mapped onto a neighbour", () => {
	// "dismissed" is a plausible spelling of "declined" and mapping it would put
	// a false statement about a person's decision into a durable row.
	const parsed = parseViewReport(
		body({ outcomes: [{ tool_call_id: "c", dimension: "cue", state: "dismissed" }] }),
	);
	assert.ok("error" in parsed);
	for (const state of VIEW_OUTCOME_STATES) assert.match(parsed.error, new RegExp(state));
});

test("an outcome with no tool call id is refused: it answers nothing", () => {
	const parsed = parseViewReport(body({ outcomes: [{ tool_call_id: "", dimension: "cue", state: "applied" }] }));
	assert.ok("error" in parsed);
	assert.match(parsed.error, /tool_call_id/);
});

test("the outcome list is capped, so one request cannot write unbounded rows", () => {
	const outcomes = Array.from({ length: MAX_OUTCOMES_PER_REPORT + 1 }, (_, i) => ({
		tool_call_id: `c-${i}`,
		dimension: "cue",
		state: "applied",
	}));
	const parsed = parseViewReport(body({ outcomes }));
	assert.ok("error" in parsed);
	assert.match(parsed.error, new RegExp(String(MAX_OUTCOMES_PER_REPORT)));
});

test("view is required — a report with no state is not a report", () => {
	const parsed = parseViewReport({ outcomes: [] });
	assert.ok("error" in parsed);
	assert.match(parsed.error, /view/);
});

test("a claimed axis outside the vocabulary is refused", () => {
	const parsed = parseViewReport(body({ view: { ...VIEW, claimed: ["cue", "zoom"] } }));
	assert.ok("error" in parsed);
	assert.match(parsed.error, /claimed/);
});

test("a cue must say whose it is", () => {
	// "the model narrowed this" over a narrowing the person typed is the precise
	// lie the whole authority mechanism exists to prevent, so the author is not
	// optional and not defaulted.
	const parsed = parseViewReport(body({ view: { ...VIEW, cue: { text: "x", entities: [], author: "someone" } } }));
	assert.ok("error" in parsed);
	assert.match(parsed.error, /author/);
});

test("a focus with no id is refused; a focus with no name is not", () => {
	assert.ok("error" in parseViewReport(body({ view: { ...VIEW, focus: { id: "", name: "Dali" } } })));
	// A null name is a real state: the browser has not loaded the graph that
	// would name the entity it is showing. Refusing it would force a guess.
	const ok = parseViewReport(body({ view: { ...VIEW, focus: { id: "u-1", name: null } } }));
	assert.ok("report" in ok, ok.error);
	assert.equal(ok.report.view.focus.name, null);
});

// ── The rendezvous ──────────────────────────────────────────────────────────

test("an ask that is opened, then answered, resolves with the outcomes", async () => {
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "because the evidence is there");
	const pending = link.await("call-1");
	link.report("conv-1", parseViewReport(
		body({ outcomes: [{ tool_call_id: "call-1", dimension: "cue", state: "applied" }] }),
	).report);
	assert.deepEqual(await pending, [{ tool_call_id: "call-1", dimension: "cue", state: "applied" }]);
});

test("an answer that arrives BEFORE anyone waits is still delivered", async () => {
	// The emit path is synchronous today, so this cannot happen — which is
	// exactly why it must be tested. The day emitting yields, a discarded early
	// verdict would show up as an occasional "not known" over an answer that had
	// already arrived, and nothing would point at this class.
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "because the evidence is there");
	link.report("conv-1", parseViewReport(
		body({ outcomes: [{ tool_call_id: "call-1", dimension: "frame", state: "offered" }] }),
	).report);
	assert.deepEqual(await link.await("call-1"), [
		{ tool_call_id: "call-1", dimension: "frame", state: "offered" },
	]);
});

test("nothing answering yields null — the honest unknown, never an empty success", async () => {
	const link = new ViewLink(10);
	link.open("conv-1", "call-1", "because the evidence is there");
	assert.equal(await link.await("call-1"), null);
});

test("awaiting an ask that was never opened yields null, not a hang", async () => {
	const link = new ViewLink(10);
	assert.equal(await link.await("never-issued"), null);
});

test("a verdict from another conversation cannot answer this ask", async () => {
	// The tool call id comes from the model provider and is only documented to
	// be unique within a conversation. Without this check one conversation's
	// browser could resolve another's question.
	const link = new ViewLink(30);
	link.open("conv-1", "call-1", "because the evidence is there");
	const pending = link.await("call-1");
	link.report("conv-2", parseViewReport(
		body({ outcomes: [{ tool_call_id: "call-1", dimension: "cue", state: "applied" }] }),
	).report);
	assert.equal(await pending, null);
});

const APPLIED = parseViewReport(
	body({ outcomes: [{ tool_call_id: "call-1", dimension: "destination", state: "applied" }] }),
).report;
const OFFERED = parseViewReport(
	body({ outcomes: [{ tool_call_id: "call-1", dimension: "destination", state: "offered" }] }),
).report;

test("the first report wins when two tabs answer the same ask, waiter first", async () => {
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "because the evidence is there");
	const pending = link.await("call-1");
	link.report("conv-1", APPLIED);
	link.report("conv-1", OFFERED);
	assert.deepEqual(await pending, [
		{ tool_call_id: "call-1", dimension: "destination", state: "applied" },
	]);
});

test("the first report wins when BOTH arrive before anyone waits", async () => {
	// The ordering that actually exercises the buffer. With a waiter attached the
	// promise has already settled by the time the second report lands, so that
	// test passes whether or not the buffer is guarded — this one does not.
	// Two tabs on one conversation can hold genuinely different views, and the
	// model must be told the one that answered first rather than the one that
	// happened to be written last.
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "because the evidence is there");
	link.report("conv-1", APPLIED);
	link.report("conv-1", OFFERED);
	assert.deepEqual(await link.await("call-1"), [
		{ tool_call_id: "call-1", dimension: "destination", state: "applied" },
	]);
});

test("one ask's several dimensions resolve together, not one at a time", async () => {
	const link = new ViewLink(50);
	link.open("conv-1", "call-1", "because the evidence is there");
	const pending = link.await("call-1");
	link.report("conv-1", parseViewReport(
		body({
			outcomes: [
				{ tool_call_id: "call-1", dimension: "cue", state: "applied" },
				{ tool_call_id: "call-1", dimension: "destination", state: "offered" },
			],
		}),
	).report);
	const outcomes = await pending;
	assert.equal(outcomes.length, 2);
	assert.deepEqual(outcomes.map((o) => o.state), ["applied", "offered"]);
});

test("knows() is true only for asks this process issued", () => {
	const link = new ViewLink(10);
	link.open("conv-1", "call-1", "because the evidence is there");
	assert.equal(link.knows("call-1"), true);
	assert.equal(link.knows("call-2"), false);
});

test("a probe round trip returns the snapshot; an unanswered one returns null", async () => {
	const link = new ViewLink(30);
	const probeId = link.openProbe("conv-1");
	const pending = link.awaitProbe(probeId);
	link.report("conv-1", parseViewReport(body({ probe_id: probeId })).report);
	const view = await pending;
	assert.equal(view.destination, "/graph");

	const silent = link.openProbe("conv-1");
	assert.equal(await link.awaitProbe(silent), null);
});

test("every report caches the view, and the cache states its age", () => {
	const link = new ViewLink(10);
	assert.equal(link.lastSnapshot("conv-1"), null);
	link.report("conv-1", parseViewReport(body()).report);
	const cached = link.lastSnapshot("conv-1");
	assert.equal(cached.view.destination, "/graph");
	assert.ok(cached.ageMs >= 0);
});

test("forgetting a conversation drops its snapshot and its asks", () => {
	const link = new ViewLink(10);
	link.open("conv-1", "call-1", "because the evidence is there");
	link.report("conv-1", parseViewReport(body()).report);
	link.forget("conv-1");
	assert.equal(link.lastSnapshot("conv-1"), null);
	assert.equal(link.knows("call-1"), false);
});

// ── What the model is told ──────────────────────────────────────────────────

test("no verdict is stated as NOT KNOWN, with an instruction not to claim a move", () => {
	// The failure this text exists to prevent is one sentence long and the model
	// writes it to a person: "I've pulled that up on the map."
	const text = composeVerdict(null);
	assert.match(text, /NOT KNOWN/);
	assert.match(text, /Do NOT tell the person the view moved/);
});

test("a verdict of no outcomes is not silence and is not success", () => {
	const text = composeVerdict([]);
	assert.match(text, /nothing was actionable/);
	assert.doesNotMatch(text, /MOVED/);
});

test("a waiting offer is reported as not applied, and re-asking is forbidden", () => {
	// Knowing it was declined must not become a route around the person.
	const text = composeVerdict([{ tool_call_id: "c", dimension: "destination", state: "offered" }]);
	assert.match(text, /WAITING/);
	assert.match(text, /has NOT been applied/);
	assert.match(text, /Do not re-issue/);
});

test("an applied verdict says the person is looking at it, and adds no offer warning", () => {
	const text = composeVerdict([{ tool_call_id: "c", dimension: "cue", state: "applied" }]);
	assert.match(text, /MOVED/);
	assert.doesNotMatch(text, /Do not re-issue/);
});

test("every state has its own sentence, and no two of them read alike", () => {
	// A flattened label is how "the person refused" and "the person never saw
	// it" become the same fact.
	const seen = new Set();
	for (const state of VIEW_OUTCOME_STATES) {
		const text = describeOutcome({ tool_call_id: "c", dimension: "cue", state });
		assert.ok(text.length > 0, state);
		assert.equal(seen.has(text), false, `${state} reads the same as another state`);
		seen.add(text);
	}
});

test("declined and expired are told apart, because the person did different things", () => {
	const declined = describeOutcome({ tool_call_id: "c", dimension: "cue", state: "declined" });
	const expired = describeOutcome({ tool_call_id: "c", dimension: "cue", state: "expired" });
	assert.match(declined, /REFUSED/);
	assert.match(expired, /UNANSWERED/);
});

test("honoured states are exactly the ones where the person can see the result", () => {
	assert.deepEqual(
		VIEW_OUTCOME_STATES.filter(isHonoured),
		["applied", "already", "followed"],
	);
	assert.equal(isTerminal("offered"), false);
	assert.equal(VIEW_OUTCOME_STATES.filter(isTerminal).length, VIEW_OUTCOME_STATES.length - 1);
});

// ── What perception says ────────────────────────────────────────────────────

test("the view report names the surface and the profile", () => {
	const text = composeViewReport({ view: VIEW, ageMs: null, destinationNoun: "the graph" });
	assert.match(text, /the graph/);
	assert.match(text, /\/graph/);
	assert.match(text, /demo/);
});

test("a path this build cannot name is quoted raw rather than invented", () => {
	const text = composeViewReport({
		view: { ...VIEW, destination: "/timeline" },
		ageMs: null,
		destinationNoun: null,
	});
	assert.match(text, /\/timeline/);
	assert.match(text, /no name for/);
});

test("held axes are stated as a prediction the model can act on", () => {
	const text = composeViewReport({
		view: { ...VIEW, claimed: ["destination"] },
		ageMs: null,
		destinationNoun: "the graph",
	});
	assert.match(text, /THE PERSON HOLDS/);
	assert.match(text, /WAIT as an offer/);
});

test("no held axis says so, rather than leaving the model to infer it from silence", () => {
	const text = composeViewReport({ view: VIEW, ageMs: null, destinationNoun: "the graph" });
	assert.match(text, /has not taken any axis/);
});

test("a cue is attributed, so the model can tell its own narrowing from the person's", () => {
	const mine = composeViewReport({ view: VIEW, ageMs: null, destinationNoun: "the graph" });
	assert.match(mine, /set by you/);
	const theirs = composeViewReport({
		view: { ...VIEW, cue: { text: "frigate", entities: [], author: "user" } },
		ageMs: null,
		destinationNoun: "the graph",
	});
	assert.match(theirs, /the person's own/);
});

test("no cue is reported as the whole corpus, not as an empty string", () => {
	const text = composeViewReport({
		view: { ...VIEW, cue: null },
		ageMs: null,
		destinationNoun: "the graph",
	});
	assert.match(text, /whole corpus/);
});

test("a stale reading is labelled with its age and never passed off as current", () => {
	const text = composeViewReport({ view: VIEW, ageMs: 42_000, destinationNoun: "the graph" });
	assert.match(text, /42s ago/);
	assert.match(text, /not a fresh reading/);
});

test("a live reading says nothing about age, because there is none to state", () => {
	const text = composeViewReport({ view: VIEW, ageMs: null, destinationNoun: "the graph" });
	assert.doesNotMatch(text, /ago/);
});

test("an unnamed focused entity is reported as unnamed, not omitted", () => {
	const text = composeViewReport({
		view: { ...VIEW, focus: { id: "u-9", name: null } },
		ageMs: null,
		destinationNoun: "the graph",
	});
	assert.match(text, /u-9/);
	assert.match(text, /has not loaded the graph/);
});

test("a waiting offer is surfaced with the reason it carries", () => {
	const text = composeViewReport({
		view: { ...VIEW, offers: [{ dimension: "destination", reason: "these cluster on the coast" }] },
		ageMs: null,
		destinationNoun: "the graph",
	});
	assert.match(text, /Waiting on screen/);
	assert.match(text, /these cluster on the coast/);
});
