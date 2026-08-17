/**
 * The shape every tool description in this seat has to fill.
 *
 * WHY A COMPOSER AND NOT NINE HAND-WRITTEN STRINGS. A tool description is not
 * documentation — it is prompt, loaded into the model's context on every single
 * turn, and Anthropic's own tool-authoring guidance puts it first among all
 * levers: "Provide extremely detailed descriptions. This is by far the most
 * important factor in tool performance"
 * (https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools).
 * The same page names four things a description must carry, and the third is the
 * one everybody forgets: what the tool does, WHEN IT SHOULD BE USED AND WHEN IT
 * SHOULD NOT, what the parameters mean, and "any important caveats or
 * limitations, such as what information the tool does not return".
 *
 * The four fields below are those four things. Making them named, required
 * fields rather than a prose convention is the whole point: a convention is
 * something the tenth tool quietly skips, and the field this codebase would have
 * skipped is `notFor` — negative scope is the one clause that is invisible when
 * missing, because a description without it still reads perfectly well and
 * simply lets the model reach for the wrong tool.
 *
 * IT ALSO ENFORCES A FLOOR. The same guidance says "Aim for at least 3-4
 * sentences for each tool description". Four required parts, each ending in
 * sentence punctuation, makes that floor structural instead of aspirational.
 *
 * Pure, so the rule can be made to fail a test without a model, a backend or a
 * running seat.
 */

/**
 * One tool's description, in the four parts the model actually needs.
 *
 * They are ordered the way a reader decides: what is this, when do I want it,
 * when do I not, and what will I get back. The composed string is flowing prose
 * rather than labelled sections — the model reads it inside a system prompt
 * alongside eight others, and a wall of `WHAT: … WHEN: …` headings would spend
 * tokens on scaffolding that the sentence order already conveys.
 */
export interface ToolDescriptionParts {
	/** What the tool does, stated as an action on the world. */
	does: string;
	/** The situation that should make the model reach for it. */
	useWhen: string;
	/**
	 * The situation in which this is the WRONG tool — and, wherever one exists,
	 * the right one by name.
	 *
	 * NAMING THE ALTERNATIVE IS THE POINT. "Don't use this for X" leaves the
	 * model with a prohibition and no route; "that is what `comment_on_todo` is
	 * for" leaves it with a next action. The seat has nine tools with real
	 * adjacencies — remember vs. record_seat_learning, update vs. comment,
	 * direct vs. inspect — and every one of those pairs is a decision the model
	 * makes from these sentences alone.
	 */
	notFor: string;
	/**
	 * What comes back — including what does NOT.
	 *
	 * The negative half is not padding. A model that does not know a listing
	 * omits comments will conclude from an empty comment field that there are
	 * none, and state that to the user as fact.
	 */
	returns: string;
}

/** The four field names, in composition order. Exported so the invariant test
 *  and the composer cannot drift apart. */
export const TOOL_DESCRIPTION_PARTS = ["does", "useWhen", "notFor", "returns"] as const;

/** Sentence-ending punctuation. A part that does not end in one of these would
 *  run into the next part when joined, producing a sentence neither author
 *  wrote. */
const SENTENCE_END = /[.!?]["')\]]?$/;

/**
 * The four parts as one description, or an error naming the part at fault.
 *
 * IT THROWS RATHER THAN COERCING, and it throws at construction time — every
 * caller builds its tools when the conversation is created, so a description
 * missing its negative scope fails when the seat starts a conversation, in a
 * message naming the tool and the field. The alternative, quietly composing
 * three parts out of four, produces exactly the defect this file exists to
 * prevent and produces it invisibly.
 */
export function composeToolDescription(tool: string, parts: ToolDescriptionParts): string {
	const composed: string[] = [];
	for (const part of TOOL_DESCRIPTION_PARTS) {
		const value = parts[part].trim();
		if (value.length === 0) {
			throw new Error(
				`Tool "${tool}" has an empty \`${part}\` in its description. All four parts are required: ` +
					`${TOOL_DESCRIPTION_PARTS.join(", ")}.`,
			);
		}
		if (!SENTENCE_END.test(value)) {
			throw new Error(
				`Tool "${tool}" has a \`${part}\` that does not end in sentence punctuation: "${value.slice(-40)}". ` +
					"The parts are joined into prose, so an unterminated one runs into the next.",
			);
		}
		composed.push(value);
	}
	return composed.join(" ");
}
