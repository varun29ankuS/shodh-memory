/**
 * The floor every tool description in this seat has to clear.
 *
 * These are not style tests. A tool description is prompt — it is loaded into
 * the model's context on every turn — and the part that goes missing is always
 * the same one: negative scope, the clause that says when this is the WRONG
 * tool. A description without it still reads perfectly well, which is exactly
 * why nothing catches it.
 *
 * Every test here was checked by deleting the line it covers and confirming it
 * went red. The mutations are listed against each block.
 *
 * Run: npm run build && npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { composeToolDescription, TOOL_DESCRIPTION_PARTS } from "../dist/tool-descriptions.js";

const whole = {
	does: "Does a thing.",
	useWhen: "Use it when a thing needs doing.",
	notFor: "Do not use it for the other thing; that is other_tool.",
	returns: "Returns what it did, and never what it did not.",
};

// ── Composition ─────────────────────────────────────────────────────────────
// Mutations: dropping `composed.push(value)` and returning a constant; joining
// with "" instead of " "; reordering TOOL_DESCRIPTION_PARTS.

test("the four parts compose in the order a reader decides in", () => {
	assert.equal(
		composeToolDescription("t", whole),
		"Does a thing. Use it when a thing needs doing. " +
			"Do not use it for the other thing; that is other_tool. " +
			"Returns what it did, and never what it did not.",
	);
});

test("every part reaches the composed description", () => {
	const composed = composeToolDescription("t", whole);
	for (const part of TOOL_DESCRIPTION_PARTS) {
		assert.ok(composed.includes(whole[part]), `${part} is missing from the composed description`);
	}
});

test("the parts are separated, not run together", () => {
	// The failure this pins is a join("") — which produces prose that reads
	// almost right and that no eyeball catches in a diff.
	assert.ok(composeToolDescription("t", whole).includes("thing. Use it"));
});

// ── The required-part rule ──────────────────────────────────────────────────
// Mutations: deleting the `value.length === 0` guard; deleting the throw;
// removing `notFor` from TOOL_DESCRIPTION_PARTS.

test("a description missing its negative scope is refused, not quietly composed", () => {
	// `empty` and `required` are asserted, not merely "it threw": with the
	// emptiness guard removed a blank part still trips the punctuation check,
	// and a test that accepted any throw would stay green through the deletion
	// it exists to catch.
	assert.throws(
		() => composeToolDescription("some_tool", { ...whole, notFor: "" }),
		/some_tool.*empty.*notFor.*required/s,
		"an empty notFor must fail as EMPTY — it is the part that is invisible when absent",
	);
});

test("every one of the four parts is required, not just the first", () => {
	for (const part of TOOL_DESCRIPTION_PARTS) {
		assert.throws(
			() => composeToolDescription("some_tool", { ...whole, [part]: "   " }),
			new RegExp(`empty \`${part}\``),
			`a blank \`${part}\` was not refused as empty`,
		);
	}
});

test("the refusal names the tool and the field, so the fix is one edit away", () => {
	try {
		composeToolDescription("claim_todo", { ...whole, returns: "" });
		assert.fail("expected a throw");
	} catch (error) {
		assert.match(error.message, /claim_todo/);
		assert.match(error.message, /returns/);
		// The recovery instruction: which fields exist at all.
		for (const part of TOOL_DESCRIPTION_PARTS) assert.match(error.message, new RegExp(part));
	}
});

// ── Sentence punctuation ────────────────────────────────────────────────────
// Mutations: deleting the SENTENCE_END test; widening the regex to /./.

test("a part that does not end a sentence is refused", () => {
	assert.throws(
		() => composeToolDescription("t", { ...whole, does: "Does a thing" }),
		/does.*punctuation/s,
	);
});

test("a closing quote or bracket after the stop still counts as an ending", () => {
	// Real descriptions end on a quoted example; rejecting those would push
	// authors into contorting the prose to satisfy the checker.
	assert.equal(
		typeof composeToolDescription("t", { ...whole, useWhen: 'Use it when the user says "do the thing."' }),
		"string",
	);
});

test("a question mark ends a sentence too", () => {
	assert.equal(typeof composeToolDescription("t", { ...whole, notFor: "Is it ever wrong? Yes, for X." }), "string");
});

// ── The floor the guidance actually names ───────────────────────────────────

test("four required parts put every description over the three-to-four sentence floor", () => {
	// Anthropic's define-tools guidance: "Aim for at least 3-4 sentences for
	// each tool description". This asserts the STRUCTURE delivers it, which is
	// what makes the floor hold for a tool nobody has written yet.
	const sentences = composeToolDescription("t", whole).match(/[.!?]["')\]]?(\s|$)/g) ?? [];
	assert.ok(sentences.length >= 4, `composed only ${sentences.length} sentences`);
});
