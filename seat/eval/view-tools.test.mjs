/**
 * The rules `direct_view` enforces before anything reaches the browser.
 *
 * These are the honesty checks: what the model is allowed to name, what it is
 * told when it names something that does not exist, and what the report may
 * claim about a surface that cannot show a narrowing. Every test here was
 * checked by deleting the line it covers and confirming it went red — the bar
 * exists because this repo has shipped tautological asserts that kept an inert
 * layer looking tested.
 *
 * Run: npm run build && npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
	collectHighlights,
	composeDirectViewReport,
	DESTINATION_IDS,
	HIGHLIGHT_LIMIT,
	normalizeHighlights,
	resolveDestination,
	WORKBENCH_DESTINATIONS,
} from "../dist/view-tools.js";

// ── The surfaces ────────────────────────────────────────────────────────────

test("every destination in the table resolves to its own path", () => {
	for (const destination of WORKBENCH_DESTINATIONS) {
		assert.equal(resolveDestination(destination.id)?.path, destination.path);
	}
});

test("the paths are the router's, not a plausible guess at them", () => {
	// Transcribed from front/ui/src/components/layout/destinations.ts. A drifted
	// path is a door the model can name and the person lands on a 404 behind.
	assert.equal(resolveDestination("briefing")?.path, "/");
	assert.equal(resolveDestination("geo")?.path, "/geo");
	assert.equal(resolveDestination("tasks")?.path, "/tasks");
	assert.equal(resolveDestination("providers")?.path, "/providers");
});

test("a name that is not a surface resolves to null rather than to something near it", () => {
	// There is deliberately no nearest-match fallback: "I could not open that so
	// I opened something else" is a worse answer than an error.
	assert.equal(resolveDestination("map"), null);
	assert.equal(resolveDestination("timeline"), null);
	assert.equal(resolveDestination(""), null);
});

test("only the graph and the map can show a framing", () => {
	// EntityCanvas recedes unmatched nodes; GeoView raises the points that
	// mention a term and GeoMap re-fits to them. Nothing else reads the cue, so
	// "framed 12 entities" said over the anomalies table would be a claim the
	// screen cannot keep.
	const framing = WORKBENCH_DESTINATIONS.map((d) => d.id).filter(
		(id) => resolveDestination(id).framesEntities,
	);
	assert.deepEqual(framing, ["graph", "geo"]);
});

test("the id list the model is offered is the table itself", () => {
	assert.deepEqual([...DESTINATION_IDS], WORKBENCH_DESTINATIONS.map((d) => d.id));
});

// ── The terms ───────────────────────────────────────────────────────────────

test("blank and whitespace-only terms are dropped, not looked up", () => {
	assert.deepEqual(normalizeHighlights(["Dali", "  ", "", " Patapsco "]).terms, ["Dali", "Patapsco"]);
});

test("duplicates collapse case-insensitively, keeping the first spelling", () => {
	// Two spellings of one name would each cost a round trip and each resolve to
	// the same node, then be framed twice.
	assert.deepEqual(normalizeHighlights(["Dali", "DALI", "dali"]).terms, ["Dali"]);
});

test("the cap keeps the front of the list and REPORTS what it dropped", () => {
	const many = Array.from({ length: HIGHLIGHT_LIMIT + 5 }, (_, i) => `entity-${i}`);
	const { terms, overflow } = normalizeHighlights(many);
	assert.equal(terms.length, HIGHLIGHT_LIMIT);
	assert.equal(terms[0], "entity-0");
	// A silent truncation is a tool telling the model it framed things it did not.
	assert.equal(overflow, 5);
});

test("overflow counts only what the cap dropped, not what dedupe removed", () => {
	const { terms, overflow } = normalizeHighlights(["Dali", "dali", "Dali "]);
	assert.deepEqual(terms, ["Dali"]);
	assert.equal(overflow, 0);
});

// ── What the graph said ─────────────────────────────────────────────────────

/** A lookup the graph answered. `checked` is explicit at every call site here
 *  because the whole point of the field is that "answered: no" and "never
 *  answered" must not be expressible by the same literal. */
const answered = (asked, name) => ({ asked, name, checked: true });
/** A lookup that never reached the graph. */
const unreachable = (asked) => ({ asked, name: null, checked: false });

test("a term the graph does not know becomes unresolved, never a framed entity", () => {
	const { resolved, unresolved } = collectHighlights([answered("Dali", "Dali"), answered("Atlantis", null)]);
	assert.deepEqual(resolved, [{ asked: "Dali", name: "Dali" }]);
	assert.deepEqual(unresolved, ["Atlantis"]);
});

test("the graph's name travels, not the model's word for it", () => {
	// The resolver folds curated aliases, so "cargo ship" comes back as "Dali".
	// Framing the model's word would light nothing while reporting success.
	const { resolved } = collectHighlights([answered("cargo ship", "Dali")]);
	assert.deepEqual(resolved, [{ asked: "cargo ship", name: "Dali" }]);
});

test("two aliases of one entity frame it once", () => {
	const { resolved } = collectHighlights([answered("the Dali", "Dali"), answered("cargo ship", "Dali")]);
	assert.equal(resolved.length, 1);
});

test("an empty name counts as absent, not as an entity called nothing", () => {
	const { resolved, unresolved } = collectHighlights([answered("x", "   ")]);
	assert.deepEqual(resolved, []);
	assert.deepEqual(unresolved, ["x"]);
});

// ── The graph answered, versus the graph was never asked ────────────────────
// This pair is the whole reason `checked` exists. Mutations: deleting the
// `!lookup.checked` branch in collectHighlights; deleting the `unchecked`
// paragraph in composeDirectViewReport; changing `checked: false` to route into
// `unresolved`.

test("a term whose lookup FAILED is unchecked, never unresolved", () => {
	const { resolved, unresolved, unchecked } = collectHighlights([
		answered("Dali", "Dali"),
		unreachable("Baltimore"),
	]);
	assert.deepEqual(resolved, [{ asked: "Dali", name: "Dali" }]);
	// The load-bearing assertion: a failed check must NOT become a claim that
	// this profile's graph has no Baltimore.
	assert.deepEqual(unresolved, []);
	assert.deepEqual(unchecked, ["Baltimore"]);
});

test("an unchecked term is not framed either — it is neither lit nor asserted absent", () => {
	const { resolved } = collectHighlights([unreachable("Dali")]);
	assert.deepEqual(resolved, [], "a term nobody verified must never reach the browser");
});

test("the report says an unchecked term is NOT known to be absent", () => {
	const text = composeDirectViewReport({
		destination: graph,
		resolved: [],
		unresolved: [],
		unchecked: ["Baltimore"],
		overflow: 0,
		focus: null,
	});
	assert.match(text, /Baltimore/);
	assert.match(text, /could not be reached/);
	// The sentence that stops the model telling the person their memory is empty.
	assert.match(text, /says nothing about whether those entities exist/);
	// And it must NOT reuse the absence wording.
	assert.doesNotMatch(text, /contains no entity matching: Baltimore/);
});

test("absent and unchecked terms are reported as two different findings", () => {
	const text = composeDirectViewReport({
		destination: graph,
		resolved: [],
		unresolved: ["Atlantis"],
		unchecked: ["Baltimore"],
		overflow: 0,
		focus: null,
	});
	const absentLine = text.split("\n").find((line) => line.includes("Atlantis"));
	const uncheckedLine = text.split("\n").find((line) => line.includes("Baltimore"));
	assert.ok(absentLine && uncheckedLine, "both findings must appear");
	assert.notEqual(absentLine, uncheckedLine, "they must not be merged into one sentence");
	assert.match(absentLine, /contains no entity matching/);
	assert.match(uncheckedLine, /could not be reached/);
});

// ── What the model is told ──────────────────────────────────────────────────

const graph = resolveDestination("graph");
const geo = resolveDestination("geo");
const anomalies = resolveDestination("anomalies");

test("the report names the terms that matched nothing", () => {
	const text = composeDirectViewReport({
		destination: graph,
		resolved: [{ asked: "Dali", name: "Dali" }],
		unresolved: ["Atlantis", "Shangri-La"],
		overflow: 0,
	});
	assert.match(text, /NOT framed/);
	assert.match(text, /Atlantis, Shangri-La/);
	// And it says what to do about it, or the model's next attempt is another guess.
	assert.match(text, /recall_memory/);
});

test("the report states the rename when the graph knew the entity by another name", () => {
	const text = composeDirectViewReport({
		destination: graph,
		resolved: [{ asked: "cargo ship", name: "Dali" }],
		unresolved: [],
		overflow: 0,
	});
	assert.match(text, /"cargo ship" → "Dali"/);
});

test("the report does not invent a rename when there was none", () => {
	const text = composeDirectViewReport({
		destination: graph,
		resolved: [{ asked: "Dali", name: "Dali" }],
		unresolved: [],
		overflow: 0,
	});
	assert.doesNotMatch(text, /Resolved from your wording/);
});

test("a surface with no cue channel is told so, with something framed on it", () => {
	// The precise invisible claim this exists to prevent: "framed 12 entities"
	// over a table that has no way to show them, repeated to the person as fact.
	const text = composeDirectViewReport({
		destination: anomalies,
		resolved: [{ asked: "Dali", name: "Dali" }],
		unresolved: [],
		overflow: 0,
	});
	assert.match(text, /no cue channel/);
});

test("the map is NOT warned about, because the map now consumes the cue", () => {
	const text = composeDirectViewReport({
		destination: geo,
		resolved: [{ asked: "Dali", name: "Dali" }],
		unresolved: [],
		overflow: 0,
	});
	assert.doesNotMatch(text, /no cue channel/);
});

test("framing with no destination warns that the person may be nowhere it shows", () => {
	const text = composeDirectViewReport({
		destination: null,
		resolved: [{ asked: "Dali", name: "Dali" }],
		unresolved: [],
		overflow: 0,
	});
	assert.match(text, /graph and the map only/);
});

test("a destination-only move carries no framing caveat to be confused by", () => {
	const text = composeDirectViewReport({
		destination: anomalies,
		resolved: [],
		unresolved: [],
		overflow: 0,
	});
	assert.doesNotMatch(text, /no cue channel/);
	assert.doesNotMatch(text, /Framing/);
});

test("dropped terms are reported in the text, not only in the count", () => {
	const text = composeDirectViewReport({
		destination: graph,
		resolved: [{ asked: "Dali", name: "Dali" }],
		unresolved: [],
		overflow: 3,
	});
	assert.match(text, /3 further terms were dropped/);
});

test("every report ends by telling the model this was a request, not a change", () => {
	// A model told the view moved will tell the person the view moved. The
	// authority ledger may have held the command as a Follow offer, and the seat
	// never learns which.
	for (const destination of [graph, geo, anomalies, null]) {
		const text = composeDirectViewReport({
			destination,
			resolved: [{ asked: "Dali", name: "Dali" }],
			unresolved: [],
			overflow: 0,
		});
		assert.match(text, /do not tell them the view has moved/);
	}
});

test("one entity is 'entity', several are 'entities'", () => {
	assert.match(
		composeDirectViewReport({ destination: graph, resolved: [{ asked: "a", name: "A" }], unresolved: [], overflow: 0 }),
		/Framing 1 entity:/,
	);
	assert.match(
		composeDirectViewReport({
			destination: graph,
			resolved: [
				{ asked: "a", name: "A" },
				{ asked: "b", name: "B" },
			],
			unresolved: [],
			overflow: 0,
		}),
		/Framing 2 entities:/,
	);
});
