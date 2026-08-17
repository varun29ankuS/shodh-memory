/**
 * The vocabulary the model uses to move the person's view.
 *
 * WHY A TOOL AND NOT AN INFERENCE. Until now the workbench moved only as a side
 * effect of `recall_memory`: the browser read the recall event and derived a cue
 * from the model's query string (front/ui/src/lib/view/commands.ts). That gives
 * the model exactly one thing it can say about the view — "I searched for this"
 * — and it can say it only by searching. It cannot open the map because an
 * answer turned out to be geographic, or the graph because a question was
 * relational, and it cannot say WHY it wants to. Everything below exists to make
 * those sentences sayable.
 *
 * THE MODEL PROPOSES. Nothing here applies anything. The tool emits a
 * `view_command` event; the authority ledger in the browser decides whether it
 * applies or waits as a Follow offer, and the person's own hand always outranks
 * it on a dimension they have touched this turn (front/ui/src/lib/view/
 * authority.ts). The seat cannot see that verdict and does not pretend to: what
 * it records is the ASK.
 *
 * THE TOOL MUST NOT LIE TO THE MODEL, because a model that is told the view
 * moved will tell the person the view moved. Two honesty rules follow, and they
 * are the reason this file talks to the backend at all:
 *
 *  - a destination that does not exist is an error naming the ones that do;
 *  - an entity this profile's graph does not contain is NEVER silently framed.
 *    Every term is resolved against the graph first, the resolved name travels
 *    (the resolver folds curated aliases, so "the cargo ship" can come back as
 *    "Dali", and framing the model's word would light nothing), and terms that
 *    resolve to nothing are named back to the model so it can try again.
 *
 * Everything above the tool definition is a pure function, so the rules can be
 * tested without a backend, a browser or a running seat.
 */

import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";
import type { ShodhBackend } from "./backend.js";
import type { SeatEvent } from "./events.js";

/**
 * The surfaces, transcribed from front/ui/src/components/layout/destinations.ts.
 *
 * DUPLICATED AS DATA, DELIBERATELY. `front/ui` and `seat` are separate packages
 * with separate tsconfigs and the front ships as one inlined index.html; there
 * is no import path between them, and inventing one to share ten strings would
 * couple a Node service's build to a browser bundle's. The duplication is made
 * safe by the failure it produces: a path that drifts from the router lands the
 * person on a 404, and the id list is the model's own menu, so a stale entry is
 * a tool that offers a door that is not there. Both are checked at the boundary
 * — `resolveDestination` is the only way in, and its test pins this table.
 *
 * `noun` is how the destination is spoken back to the model. It is not the rail
 * label: "Conversations" is a rail row, "the conversation" is what a sentence
 * about it says.
 */
export const WORKBENCH_DESTINATIONS = [
	{ id: "briefing", path: "/", noun: "the briefing" },
	{ id: "chat", path: "/chat", noun: "the conversation" },
	{ id: "recall", path: "/recall", noun: "recall" },
	{ id: "graph", path: "/graph", noun: "the graph" },
	{ id: "geo", path: "/geo", noun: "the map" },
	{ id: "anomalies", path: "/anomalies", noun: "anomalies" },
	{ id: "tasks", path: "/tasks", noun: "tasks" },
	{ id: "history", path: "/history", noun: "history" },
	{ id: "sources", path: "/sources", noun: "sources" },
	{ id: "providers", path: "/providers", noun: "providers" },
] as const;

export type DestinationId = (typeof WORKBENCH_DESTINATIONS)[number]["id"];

export interface Destination {
	id: string;
	path: string;
	noun: string;
	/**
	 * Whether framed entities visibly narrow THIS surface.
	 *
	 * Two surfaces read the cue and nothing else does: `EntityCanvas` recedes
	 * unmatched nodes and aims its camera at the matched set, and `GeoView`
	 * raises the points whose memory mentions a term while `GeoMap` re-fits to
	 * what was raised. Anomalies, tasks, sources, history, recall, providers and
	 * the briefing have no cue channel — the terms reach them and change
	 * nothing.
	 *
	 * Recorded here rather than left to prose because the tool's answer to the
	 * model depends on it. "Framed 12 entities" said over the anomalies table is
	 * the precise invisible claim this whole mechanism exists to prevent, and
	 * the model would repeat it to the person as fact.
	 */
	framesEntities: boolean;
}

const ENTITY_SURFACES: ReadonlySet<string> = new Set(["graph", "geo"]);

/** The ids, in menu order, for the tool schema and for error messages. */
export const DESTINATION_IDS: readonly string[] = WORKBENCH_DESTINATIONS.map((d) => d.id);

/**
 * A destination id, or null when the model named something that is not a
 * surface. Null is a value the caller must handle; there is no fallback
 * destination, because "I could not open that so I opened something else" is a
 * worse answer than an error.
 */
export function resolveDestination(id: string): Destination | null {
	const entry = WORKBENCH_DESTINATIONS.find((d) => d.id === id.trim().toLowerCase());
	if (!entry) return null;
	return { id: entry.id, path: entry.path, noun: entry.noun, framesEntities: ENTITY_SURFACES.has(entry.id) };
}

/**
 * How many terms a single command may frame.
 *
 * The same number the browser's cue derivation uses (`ENTITY_LIMIT` in
 * front/ui/src/lib/view/commands.ts) and for the same reason: past a couple of
 * dozen terms the matched set stops being a narrowing and lights most of the
 * graph. It also bounds the entity lookups this tool issues, which are one HTTP
 * round trip each.
 */
export const HIGHLIGHT_LIMIT = 24;

export interface NormalizedHighlights {
	terms: string[];
	/** How many terms were dropped by the cap. Reported, never swallowed. */
	overflow: number;
}

/**
 * Trim, drop blanks, de-duplicate case-insensitively, cap — order preserved.
 *
 * Order is kept because the model lists what it considers most relevant first,
 * so the cap keeps the front of the list rather than an arbitrary slice. The
 * de-duplication is case-insensitive and keeps the FIRST spelling: two spellings
 * of one name would otherwise each cost a round trip and each resolve to the
 * same node.
 */
export function normalizeHighlights(raw: readonly string[]): NormalizedHighlights {
	const seen = new Set<string>();
	const terms: string[] = [];
	let overflow = 0;
	for (const value of raw) {
		const term = value.trim();
		if (term.length === 0) continue;
		const key = term.toLowerCase();
		if (seen.has(key)) continue;
		seen.add(key);
		if (terms.length >= HIGHLIGHT_LIMIT) {
			overflow += 1;
			continue;
		}
		terms.push(term);
	}
	return { terms, overflow };
}

/** One term's fate: the word the model used, and the graph's name for it. */
export interface ResolvedTerm {
	asked: string;
	name: string;
}

export interface HighlightOutcome {
	resolved: ResolvedTerm[];
	unresolved: string[];
}

/**
 * Collapse lookup results into the set that will actually be framed.
 *
 * De-duplicated BY THE RESOLVED NAME, not by the asked term: two aliases of one
 * entity ("the Dali", "cargo ship") resolve to a single node, and framing it
 * twice would make a one-entity narrowing report as two.
 */
export function collectHighlights(
	lookups: readonly { asked: string; name: string | null }[],
): HighlightOutcome {
	const seen = new Set<string>();
	const resolved: ResolvedTerm[] = [];
	const unresolved: string[] = [];
	for (const lookup of lookups) {
		if (lookup.name === null || lookup.name.trim().length === 0) {
			unresolved.push(lookup.asked);
			continue;
		}
		const key = lookup.name.toLowerCase();
		if (seen.has(key)) continue;
		seen.add(key);
		resolved.push({ asked: lookup.asked, name: lookup.name });
	}
	return { resolved, unresolved };
}

function list(values: readonly string[]): string {
	return values.join(", ");
}

/**
 * What the model is told it did.
 *
 * Pure, and separated from the tool, because this text is the whole honesty
 * contract: it is what the model reads before it writes a sentence to the
 * person, and every claim in it has to be one the browser will actually honour.
 * Three things it must never omit — a term that named nothing, terms dropped by
 * the cap, and a destination that cannot show a narrowing.
 */
export function composeDirectViewReport(input: {
	destination: Destination | null;
	resolved: readonly ResolvedTerm[];
	unresolved: readonly string[];
	overflow: number;
}): string {
	const lines: string[] = [];
	const { destination, resolved, unresolved, overflow } = input;

	if (destination) lines.push(`Asked the workbench to open ${destination.noun}.`);

	if (resolved.length > 0) {
		const renamed = resolved.filter((term) => term.name.toLowerCase() !== term.asked.toLowerCase());
		const framed = `Framing ${resolved.length} ${resolved.length === 1 ? "entity" : "entities"}: ${list(
			resolved.map((term) => term.name),
		)}.`;
		lines.push(
			renamed.length > 0
				? `${framed} Resolved from your wording: ${list(renamed.map((t) => `"${t.asked}" → "${t.name}"`))}.`
				: framed,
		);
	}

	if (unresolved.length > 0) {
		lines.push(
			`NOT framed — this profile's graph contains no entity matching: ${list(unresolved)}. ` +
				"Use recall_memory first and take names from the results; the graph is the authority on what exists here.",
		);
	}

	if (overflow > 0) {
		lines.push(
			`${overflow} further ${overflow === 1 ? "term was" : "terms were"} dropped: a command frames at most ${HIGHLIGHT_LIMIT}.`,
		);
	}

	// Where the narrowing is actually visible. Stated whenever there is
	// something framed and the surface in question cannot show it — including
	// the case where no destination was named at all, because then the person is
	// wherever they already were and that may be none of the two.
	if (resolved.length > 0) {
		if (!destination) {
			lines.push("Framing narrows the graph and the map only; on any other surface it changes nothing on screen.");
		} else if (!destination.framesEntities) {
			lines.push(
				`${destination.noun} has no cue channel: the framing is recorded but nothing on that screen narrows. ` +
					"Open the graph or the map if the entities are the point.",
			);
		}
	}

	lines.push(
		"This is a request, not a change. If the person has already moved this part of the view during your turn, " +
			"the workbench holds it as a Follow they can accept or decline — so do not tell them the view has moved.",
	);

	return lines.join("\n");
}

export interface ViewToolContext {
	backend: ShodhBackend;
	/** The person's memory namespace — the graph the entities are checked against. */
	userId: string;
	emit(event: SeatEvent): void;
}

const directViewParameters = Type.Object({
	reason: Type.String({
		minLength: 8,
		maxLength: 400,
		description:
			"Why this view, in your own words, addressed to the user — it is shown to them verbatim beside the change. " +
			'State what the evidence is, not what you are doing: "these 12 memories cluster on the Malabar coast" is ' +
			'useful, "opening Geo" is not.',
	}),
	destination: Type.Optional(
		Type.Union(
			WORKBENCH_DESTINATIONS.map((destination) => Type.Literal(destination.id)),
			{
				description:
					"Which surface to open. briefing: what is in here and what changed. chat: the conversation. " +
					"recall: search over memory. graph: entities and how they relate. geo: where memory happened. " +
					"anomalies: what deviates from normal. tasks: recorded work. history: tool calls and retrievals. " +
					"sources: what wrote into this profile. providers: models and keys. " +
					"Omit to frame entities without moving the person.",
			},
		),
	),
	highlight: Type.Optional(
		Type.Array(Type.String({ minLength: 1, maxLength: 120 }), {
			maxItems: HIGHLIGHT_LIMIT,
			description:
				"Entity names to light and aim the camera at. They are checked against this profile's graph and any " +
				"that name nothing are reported back to you rather than framed. Visible on the graph and the map only.",
		}),
	),
});

export function createViewTools(context: ViewToolContext): AgentTool<any>[] {
	const directViewTool: AgentTool<typeof directViewParameters> = {
		name: "direct_view",
		label: "Direct the view",
		description:
			"Move the user's workbench to the surface that answers their question, and light the entities the answer " +
			"is about. Use it when the shape of the answer has a place — a relational answer belongs on the graph, a " +
			"geographic one on the map, a question about where something came from on sources. " +
			"Requires a reason, which is shown to the user. The user always outranks you: if they have moved that part " +
			"of the view during this turn, your request waits as an offer they can accept.",
		parameters: directViewParameters,
		execute: async (toolCallId, params) => {
			const reason = params.reason.trim();
			if (reason.length === 0) {
				throw new Error("A view change needs a reason: it is shown to the user beside the change.");
			}

			// Re-checked at runtime rather than trusted to the schema. The union
			// constrains a well-behaved caller, but the failure this guards is a
			// model emitting a plausible-looking id ("map", "timeline") — and the
			// answer to that has to name the real ones, or the model's next
			// attempt is another guess.
			let destination: Destination | null = null;
			if (params.destination !== undefined) {
				destination = resolveDestination(params.destination);
				if (!destination) {
					throw new Error(
						`"${params.destination}" is not a surface of this workbench. Valid destinations: ${list(DESTINATION_IDS)}.`,
					);
				}
			}

			const { terms, overflow } = normalizeHighlights(params.highlight ?? []);
			if (destination === null && terms.length === 0) {
				throw new Error(
					"Nothing to do: give a destination, or entities to highlight, or both. A reason alone moves nothing.",
				);
			}

			// One round trip per term, in parallel and bounded by HIGHLIGHT_LIMIT.
			// A transport failure propagates: a view command built on entity
			// checks that did not happen would frame unverified terms, which is
			// the one thing this tool promises not to do.
			const lookups = await Promise.all(
				terms.map(async (asked) => {
					const entity = await context.backend.findEntity(context.userId, asked);
					return { asked, name: entity?.name ?? null };
				}),
			);
			const { resolved, unresolved } = collectHighlights(lookups);

			const entities = resolved.map((term) => term.name);
			context.emit({
				type: "view_command",
				tool_call_id: toolCallId,
				reason,
				destination: destination?.path ?? null,
				entities,
				unresolved,
			});

			// Every term named something imaginary AND there was nowhere to go:
			// nothing was asked for, so the model must not be told a command was
			// issued. The event above is still emitted — it carries the empty
			// outcome and the unresolved terms, which is the durable record that
			// this was attempted and found nothing.
			const text = composeDirectViewReport({ destination, resolved, unresolved, overflow });
			return {
				content: [{ type: "text", text }],
				details: { destination: destination?.path ?? null, entities, unresolved },
			} satisfies AgentToolResult<{ destination: string | null; entities: string[]; unresolved: string[] }>;
		},
	};

	return [directViewTool];
}
