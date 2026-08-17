/**
 * The verb the surface was missing, and the rule that makes it honest.
 *
 * Creation was the ONE mutation on this board that arrived anonymous: the model
 * could already make todos through the bridged MCP `add_todo`, which writes no
 * attribution at all, and it was left bridged precisely because there was no
 * native equivalent. So a board could not say which of its rows an assistant put
 * there.
 *
 * The link rule is the other half. The backend verifies that a linked memory
 * EXISTS; it cannot verify that anybody was ever shown it, and a "why does this
 * task exist" chain whose first link is a uuid the model guessed is worse than
 * no chain. So the seat restricts links to the memories surfaced this run —
 * the same set the citation contract already polices in an answer.
 *
 * Every test here was checked by deleting the line it covers and confirming it
 * went red. The mutation is named on each block.
 *
 * Run: npm run build && node --test eval/create-todo.test.mjs
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { composeCreateReport, resolveMemoryLinks } from "../dist/todo-tools.js";
// `memoryCitationKey` moved to memory-tools.ts when it gained a second consumer:
// it now also gates forget_memory, where refusing anything but the two spellings
// the model has actually seen is what keeps a model-supplied string out of a
// destructive request URL. The rule under test is unchanged.
import { memoryCitationKey } from "../dist/memory-tools.js";

// ── Reading an id the model is capable of producing ─────────────────────────
// Mutations: delete the bracketed branch; delete the bare-8 branch; drop
// `.toLowerCase()`; widen `{8}` to `+`.

test("the bracketed citation is the form the model writes in prose", () => {
	assert.equal(memoryCitationKey("[mem:9f3c1b20]"), "9f3c1b20");
});

test("the bare eight characters are the form it reads out of a listing", () => {
	assert.equal(memoryCitationKey("9f3c1b20"), "9f3c1b20");
});

test("case is normalised, because the two surfaces do not agree on it", () => {
	assert.equal(memoryCitationKey("[mem:9F3C1B20]"), "9f3c1b20");
	assert.equal(memoryCitationKey("9F3C1B20"), "9f3c1b20");
});

test("surrounding whitespace is forgiven", () => {
	assert.equal(memoryCitationKey("  [mem:9f3c1b20]  "), "9f3c1b20");
});

test("anything that is not the contract's shape is refused, not salvaged", () => {
	// A full uuid is refused ON PURPOSE. The model is never shown one, so an id
	// in that shape is something it constructed, and the backend would happily
	// verify a constructed uuid that happens to exist.
	for (const bad of [
		"9f3c1b20-0000-4000-8000-000000000001",
		"[mem:95%]",
		"[mem:9f3c1b2]",
		"[mem:9f3c1b201]",
		"zzzzzzzz",
		"",
		"mem:9f3c1b20",
	]) {
		assert.equal(memoryCitationKey(bad), null, bad);
	}
});

// ── Only what the model has actually been shown ─────────────────────────────
// Mutations: return the short id instead of consulting `known`; drop the
// `unknown.push`; delete the `seen` de-duplication.

const SHOWN = { "9f3c1b20": "9f3c1b20-0000-4000-8000-000000000001" };
const known = (shortId) => SHOWN[shortId] ?? null;

test("a shown memory resolves to its full uuid, which is what the backend verifies", () => {
	const { ids, unknown } = resolveMemoryLinks(["[mem:9f3c1b20]"], known);
	assert.deepEqual(ids, ["9f3c1b20-0000-4000-8000-000000000001"]);
	assert.deepEqual(unknown, []);
});

test("a memory the model was never shown is REFUSED, not linked", () => {
	// The load-bearing rule. The backend cannot catch this: `deadbeef` may name
	// a real memory in this profile, and linking it would put a motivation on
	// the todo that nobody ever read.
	const { ids, unknown } = resolveMemoryLinks(["[mem:deadbeef]"], known);
	assert.deepEqual(ids, []);
	assert.deepEqual(unknown, ["[mem:deadbeef]"]);
});

test("a refused link is named back in the model's own spelling", () => {
	// So the model can tell which of the four ids it sent was the problem.
	const { unknown } = resolveMemoryLinks(["[mem:9f3c1b20]", "not-an-id"], known);
	assert.deepEqual(unknown, ["not-an-id"]);
});

test("one good id among bad ones still links", () => {
	// Partial success reported honestly beats an all-or-nothing refusal that
	// discards work the model got right.
	const { ids, unknown } = resolveMemoryLinks(["[mem:deadbeef]", "9f3c1b20"], known);
	assert.equal(ids.length, 1);
	assert.equal(unknown.length, 1);
});

test("two spellings of one memory link it once", () => {
	const { ids } = resolveMemoryLinks(["[mem:9f3c1b20]", "9F3C1B20"], known);
	assert.deepEqual(ids, ["9f3c1b20-0000-4000-8000-000000000001"]);
});

// ── What the model is told it made ──────────────────────────────────────────
// Mutations: drop the short id from the first line; delete the unknown-links
// paragraph; delete the commentError branch; make the warning unconditional.

test("the report leads with the short id, which is the handle every other tool takes", () => {
	// Without it the model must call list_todos before it can touch the thing it
	// just created.
	const text = composeCreateReport({
		shortId: "BOLT-8",
		linked: 0,
		unknownLinks: [],
		commentError: null,
		author: "agent:anthropic/me",
	});
	assert.match(text, /^Created \[BOLT-8\]\./);
	assert.match(text, /agent:anthropic\/me/);
});

test("refused links are stated with the route to fixing them", () => {
	const text = composeCreateReport({
		shortId: "BOLT-8",
		linked: 1,
		unknownLinks: ["[mem:deadbeef]"],
		commentError: null,
		author: "a",
	});
	assert.match(text, /NOT linked/);
	assert.match(text, /\[mem:deadbeef\]/);
	assert.match(text, /recall the memory first/);
});

test("a creation whose signature failed says the todo is INDISTINGUISHABLE from the user's", () => {
	// The todo exists either way. What the model must not do is go on believing
	// the board records that it made this one.
	const text = composeCreateReport({
		shortId: "BOLT-8",
		linked: 0,
		unknownLinks: [],
		commentError: "backend error 503",
		author: "a",
	});
	assert.match(text, /WARNING/);
	assert.match(text, /backend error 503/);
	assert.match(text, /distinguishes it from a todo the user typed/);
});

test("a clean creation carries no warning", () => {
	const text = composeCreateReport({
		shortId: "BOLT-8",
		linked: 0,
		unknownLinks: [],
		commentError: null,
		author: "a",
	});
	assert.doesNotMatch(text, /WARNING/);
});

test("nothing linked and nothing refused says neither", () => {
	// Silence about links the model did not ask for. A line reading "Linked to 0
	// memories" on every creation is prompt spent on nothing.
	const text = composeCreateReport({
		shortId: "BOLT-8",
		linked: 0,
		unknownLinks: [],
		commentError: null,
		author: "a",
	});
	assert.doesNotMatch(text, /Linked to/);
	assert.doesNotMatch(text, /NOT linked/);
});

test("one memory is 'memory', several are 'memories'", () => {
	assert.match(
		composeCreateReport({ shortId: "B-1", linked: 1, unknownLinks: [], commentError: null, author: "a" }),
		/1 memory,/,
	);
	assert.match(
		composeCreateReport({ shortId: "B-1", linked: 3, unknownLinks: [], commentError: null, author: "a" }),
		/3 memories,/,
	);
});
