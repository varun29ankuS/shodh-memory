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
 * authority.ts). What this file records is still the ASK.
 *
 * IT NOW WAITS TO HEAR WHAT HAPPENED, and that is the difference between this
 * version and the one before it. The tool registers the ask with `ViewLink`
 * (view-link.ts), emits, and waits a bounded moment for the browser to report
 * the ledger's verdict back over `POST /v1/conversations/{id}/view-report`. The
 * model is then told the truth: moved, waiting as an offer, refused — or, when
 * nothing answered, that the verdict is NOT KNOWN. The one thing it is never
 * told is that a move happened because a move was requested.
 *
 * KNOWING IT WAS DECLINED IS NOT A ROUTE AROUND THE PERSON. The verdict text
 * says so explicitly, and there is nothing here that retries: the ledger is
 * unchanged, and an offer is the person's to accept.
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
import { composeToolDescription } from "./tool-descriptions.js";
import {
	composeVerdict,
	composeViewReport,
	VIEW_REPLY_TIMEOUT_MS,
	type ViewLink,
	type ViewOutcome,
} from "./view-link.js";

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
 * The other direction: a router path, as the surface it names.
 *
 * Needed because the browser reports where the person IS as a path, and a model
 * reading "/geo" has to know that means the map. Null when the path is not one
 * of the ten — a surface added to the router before it is added to this table —
 * and the caller then quotes the raw path rather than inventing a name for it.
 */
export function destinationForPath(path: string): Destination | null {
	const entry = WORKBENCH_DESTINATIONS.find((d) => d.path === path);
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

/**
 * One term's lookup, as it came back.
 *
 * `checked` IS THE FIELD THAT MATTERS. `name: null` means the graph answered and
 * has nothing by that name; `checked: false` means the graph never answered at
 * all. Collapsing the second into the first is the exact lie this file forbids —
 * "this profile's graph contains no entity matching X" is a claim about the
 * corpus, and a transport failure is not evidence for it. The model would repeat
 * it to the person as a fact about their data.
 */
export interface TermLookup {
	asked: string;
	name: string | null;
	/** False when the lookup itself failed, so `name` says nothing. */
	checked: boolean;
}

export interface HighlightOutcome {
	resolved: ResolvedTerm[];
	/** The graph answered, and holds nothing by that name. */
	unresolved: string[];
	/** The graph was never reached. Not framed, and NOT reported as absent. */
	unchecked: string[];
}

/**
 * Collapse lookup results into the set that will actually be framed.
 *
 * De-duplicated BY THE RESOLVED NAME, not by the asked term: two aliases of one
 * entity ("the Dali", "cargo ship") resolve to a single node, and framing it
 * twice would make a one-entity narrowing report as two.
 *
 * The unchecked bucket is deliberately NOT de-duplicated by name — there is no
 * name to de-duplicate by, only the word the model used.
 */
export function collectHighlights(lookups: readonly TermLookup[]): HighlightOutcome {
	const seen = new Set<string>();
	const resolved: ResolvedTerm[] = [];
	const unresolved: string[] = [];
	const unchecked: string[] = [];
	for (const lookup of lookups) {
		if (!lookup.checked) {
			unchecked.push(lookup.asked);
			continue;
		}
		if (lookup.name === null || lookup.name.trim().length === 0) {
			unresolved.push(lookup.asked);
			continue;
		}
		const key = lookup.name.toLowerCase();
		if (seen.has(key)) continue;
		seen.add(key);
		resolved.push({ asked: lookup.asked, name: lookup.name });
	}
	return { resolved, unresolved, unchecked };
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
	/** Terms whose graph lookup failed. Absent is treated as none. */
	unchecked?: readonly string[];
	overflow: number;
	/** The entity to open in the inspector, as the graph names it, or null. */
	focus: { id: string; name: string } | null;
}): string {
	const lines: string[] = [];
	const { destination, resolved, unresolved, overflow, focus } = input;
	const unchecked = input.unchecked ?? [];

	if (destination) lines.push(`Asked the workbench to open ${destination.noun}.`);
	if (focus) lines.push(`Asked it to open ${focus.name} in the inspector.`);

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

	// SAID SEPARATELY FROM `unresolved`, AND WORDED SO IT CANNOT BE READ AS ONE.
	// The paragraph above is a statement about the person's corpus; this one is a
	// statement about this seat's reach. A model that read a backend outage as
	// "the graph has no Dali" would tell the person their memory is missing
	// something that is sitting in it.
	if (unchecked.length > 0) {
		lines.push(
			`NOT framed, and NOT known to be absent — the graph could not be reached to check: ${list(unchecked)}. ` +
				"This says nothing about whether those entities exist in this profile. Retry the command once; if it " +
				"fails again the memory backend is down, so answer from what you already have and do not describe the view.",
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
	/** Which conversation's browser is being asked. The link is per-process and
	 *  keyed on this, so one tab's verdict can never answer another's ask. */
	conversationId: string;
	viewLink: ViewLink;
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
	focus: Type.Optional(
		Type.String({
			minLength: 1,
			maxLength: 120,
			description:
				"ONE entity name to open in the inspector — the detail panel showing its type and what it is connected " +
				"to. Use it when the answer is about a single thing; `highlight` is for a set. Checked against this " +
				"profile's graph like the highlights are, and reported back unopened if it names nothing.",
		}),
	),
});

/**
 * No parameters, and that is the guarantee.
 *
 * A read tool with a filter, a destination or an id would be a tool whose
 * arguments could ask for something; this one can only ask "what is there".
 * There is nothing in the request for a browser to misread as an instruction.
 */
const inspectViewParameters = Type.Object({});

export function createViewTools(context: ViewToolContext): AgentTool<any>[] {
	const directViewTool: AgentTool<typeof directViewParameters> = {
		name: "direct_view",
		label: "Direct the view",
		description: composeToolDescription("direct_view", {
			does:
				"Asks the user's workbench to open a surface, light the entities an answer is about, and open one of " +
				"them in the inspector — one request, however many of those three you name, because they are one intent.",
			useWhen:
				"Reach for it when the shape of your answer has a place: a relational answer belongs on the graph, a " +
				"geographic one on the map, a question about where something came from on sources. The reason you give " +
				"is shown to the user verbatim, so state the evidence — \"these 12 memories cluster on the Malabar " +
				"coast\" — rather than the action.",
			notFor:
				"Do not call it to re-issue a request the workbench refused or is still holding as an offer: the user " +
				"outranks you on any axis they have touched this turn, and asking twice is the same as not asking. Do " +
				"not call it to move someone who is already where you would send them — inspect_view tells you that " +
				"first, and it changes nothing. Do not call it merely to look active; a view that rearranges itself " +
				"without a finding behind it is noise.",
			returns:
				"Every entity name is checked against this profile's graph before anything is sent, and names that " +
				"match nothing come back to you unframed rather than being silently dropped. The result then states " +
				"what the workbench did on each axis — MOVED, ALREADY THERE, WAITING as an offer, REFUSED, or NOT " +
				"KNOWN when nothing answered — and only the first three permit you to tell the user the view moved. " +
				"It returns nothing about the memories or entities themselves.",
		}),
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
			const focusTerm = params.focus?.trim() ?? "";
			if (destination === null && terms.length === 0 && focusTerm.length === 0) {
				throw new Error(
					"Nothing to do: give a destination, entities to highlight, or one to focus. A reason alone moves nothing.",
				);
			}

			// One round trip per term, in parallel and bounded by HIGHLIGHT_LIMIT.
			//
			// EACH TERM SETTLES ON ITS OWN, so `Promise.all` here can no longer
			// reject. A failing lookup still must not frame its term — that rule
			// has not moved — but it used to reject the whole batch, discarding
			// the nineteen terms that HAD resolved and the destination with them,
			// and handing the model a raw transport message it could do nothing
			// with. A term whose check failed now travels as `unchecked`: not
			// framed, and explicitly not reported as absent, which is the
			// distinction one rejected promise used to destroy along with
			// everything else.
			const lookups: TermLookup[] = await Promise.all(
				terms.map(async (asked): Promise<TermLookup> => {
					try {
						const entity = await context.backend.findEntity(context.userId, asked);
						return { asked, name: entity?.name ?? null, checked: true };
					} catch {
						return { asked, name: null, checked: false };
					}
				}),
			);
			const { resolved, unresolved, unchecked } = collectHighlights(lookups);

			// THE FOCUS CARRIES A UUID, NOT A NAME, because the browser selects by
			// `UniverseStar.id` and that id is this node's `uuid`. Resolving it
			// here is the same rule the highlights follow — the graph's identity
			// travels, never the model's word — and it is the only way the
			// inspector can open on the thing that was meant.
			//
			// Its failure is caught for the same reason the highlights' are: an
			// inspector term the graph never answered about is unchecked, not
			// missing.
			let focus: { id: string; name: string } | null = null;
			let focusUnresolved: string | null = null;
			let focusUnchecked: string | null = null;
			if (focusTerm.length > 0) {
				try {
					const node = await context.backend.findEntity(context.userId, focusTerm);
					if (node && node.name.trim().length > 0) focus = { id: node.uuid, name: node.name };
					else focusUnresolved = focusTerm;
				} catch {
					focusUnchecked = focusTerm;
				}
			}

			const allUnresolved = focusUnresolved === null ? unresolved : [...unresolved, focusUnresolved];
			const allUnchecked = focusUnchecked === null ? unchecked : [...unchecked, focusUnchecked];

			// EVERY TERM FAILED AND THERE WAS NOWHERE TO GO. Nothing can be asked
			// for, and emitting a command that names nothing would put a row in the
			// trail claiming an act that never happened. The model is told what
			// went wrong instead, in the recoverable channel — this is the one
			// branch where a throw is more honest than a report.
			if (destination === null && resolved.length === 0 && focus === null && allUnchecked.length > 0) {
				throw new Error(
					`The graph could not be reached, so none of ${list(allUnchecked)} could be checked, and there is no ` +
						"destination to open without them. Nothing was requested. This is not a statement that those " +
						"entities are absent — retry once, and if it fails again answer without moving the view.",
				);
			}

			const entities = resolved.map((term) => term.name);

			// REGISTERED BEFORE THE EVENT LEAVES. The browser answers on a
			// separate HTTP request, and on a loopback bind that request can be
			// handled before this call returns — a waiter opened after the emit
			// would miss the verdict it exists to catch and report "not known"
			// over an answer that had already arrived.
			context.viewLink.open(context.conversationId, toolCallId);
			context.emit({
				type: "view_command",
				tool_call_id: toolCallId,
				reason,
				destination: destination?.path ?? null,
				entities,
				unresolved: allUnresolved,
				unchecked: allUnchecked,
				focus,
			});

			const outcomes = await context.viewLink.await(toolCallId);

			// Every term named something imaginary AND there was nowhere to go:
			// nothing was asked for, so the model must not be told a command was
			// issued. The event above is still emitted — it carries the empty
			// outcome and the unresolved terms, which is the durable record that
			// this was attempted and found nothing.
			const text = [
				composeDirectViewReport({
					destination,
					resolved,
					unresolved: allUnresolved,
					unchecked: allUnchecked,
					overflow,
					focus,
				}),
				composeVerdict(outcomes),
			].join("\n\n");
			return {
				content: [{ type: "text", text }],
				details: {
					destination: destination?.path ?? null,
					entities,
					unresolved: allUnresolved,
					unchecked: allUnchecked,
					focus,
					// Null, not an empty array: "the browser said nothing" and "the
					// browser said nothing changed" are different facts, and a
					// consumer of `details` must be able to tell them apart.
					outcomes,
				},
			} satisfies AgentToolResult<{
				destination: string | null;
				entities: string[];
				unresolved: string[];
				unchecked: string[];
				focus: { id: string; name: string } | null;
				outcomes: ViewOutcome[] | null;
			}>;
		},
	};

	/**
	 * Perception: what is the person actually looking at.
	 *
	 * WHY IT PROBES RATHER THAN READS A CACHE. The seat holds the last state the
	 * browser reported, but "last reported" is not "current" — the person may
	 * have navigated three times since, and a stale picture served as a fresh one
	 * is the same class of falsehood as an assumed verdict. So this asks, waits
	 * the same bounded moment `direct_view` waits, and only falls back to the
	 * cached snapshot WITH ITS AGE STATED when nothing answers.
	 *
	 * IT CANNOT CHANGE ANYTHING. It emits a probe, which the browser answers by
	 * reading its own store; there is no dimension in the request, no path, no
	 * entity, nothing the browser could apply even if it wanted to. That matters
	 * because a getter that could smuggle a change would let the model move the
	 * view without an ask ever appearing in the audit trail, which is exactly the
	 * hole the authority ledger was built to close.
	 */
	const inspectViewTool: AgentTool<typeof inspectViewParameters> = {
		name: "inspect_view",
		label: "Look at the workbench",
		description: composeToolDescription("inspect_view", {
			does:
				"Reports what the user is looking at right now: which surface is open, which memory profile, what is " +
				"narrowed and by whom, what is open in the inspector, which axes the user is holding this turn, and " +
				"which of your offers are still waiting on screen.",
			useWhen:
				"Call it before direct_view when it matters whether your request would apply immediately or would only " +
				"become an offer, and whenever you are about to send someone where they may already be. It takes no " +
				"arguments and changes nothing, so it costs you only the round trip.",
			notFor:
				"It cannot move anything — direct_view is the only tool that asks for a change, and this one has no " +
				"parameter through which a change could be smuggled. It is also not a way to read the user's data: it " +
				"says where they are and what is narrowed, never what any memory or entity contains.",
			returns:
				"A live reading when the workbench answers within two seconds. When it does not, you get the last " +
				"state the workbench reported WITH ITS AGE STATED, which is not the same thing — anything the user has " +
				"done since is missing from it — and when no browser has ever reported, the call fails rather than " +
				"guessing.",
		}),
		parameters: inspectViewParameters,
		execute: async () => {
			// Same order rule as the command path: the probe must be known before
			// the event carrying its id leaves.
			const probeId = context.viewLink.openProbe(context.conversationId);
			context.emit({ type: "view_probe", probe_id: probeId });
			const live = await context.viewLink.awaitProbe(probeId);

			// The cache is consulted ONLY when the probe went unanswered, and what
			// it yields is labelled with its age rather than presented as a
			// reading. A snapshot from before the person's last three navigations
			// is still useful — it is where they were — but only if it is offered
			// as that.
			const cached = live === null ? context.viewLink.lastSnapshot(context.conversationId) : null;
			const view = live ?? cached?.view ?? null;
			if (view === null) {
				throw new Error(
					`The workbench did not answer within ${VIEW_REPLY_TIMEOUT_MS / 1000}s and has never reported its ` +
						"state, so nothing here knows what is on screen. Either no browser is watching this conversation " +
						"or the tab is closed. Do not describe the view; answer from what you know and let the person look.",
				);
			}

			const ageMs = cached?.ageMs ?? null;
			const text = composeViewReport({
				view,
				ageMs,
				destinationNoun: destinationForPath(view.destination)?.noun ?? null,
			});
			return {
				content: [{ type: "text", text }],
				details: { view, live: live !== null, age_ms: ageMs },
			} satisfies AgentToolResult<{ view: typeof view; live: boolean; age_ms: number | null }>;
		},
	};

	return [directViewTool, inspectViewTool];
}
