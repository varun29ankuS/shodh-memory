/**
 * The browser's half of the view loop — the return path the seat never had.
 *
 * THE GAP THIS CLOSES. `direct_view` (view-tools.ts) emits a `view_command` and
 * stops. Whether that command moved the view, or waited as a Follow offer the
 * person may never accept, is decided by the authority ledger in the browser
 * (front/ui/src/stores/view.ts) — and until now that verdict reached nobody. The
 * model wrote its sentence to the person without knowing whether the view had
 * moved, and the audit trail could only say what was ASKED. Both are the same
 * defect: an open loop.
 *
 * THE TRANSPORT IS THE ONE THE BROWSER ALREADY USES. Everything the client
 * sends the seat is an HTTP request on the conversation resource (POST
 * .../messages, PATCH .../model); the SSE stream is one-way by construction and
 * cannot carry a reply. So the return path is one more route on that same
 * resource — `POST /v1/conversations/{id}/view-report` — rather than a second
 * channel. No socket, no dependency, no new lifecycle to get wrong.
 *
 * TWO USES, ONE MECHANISM. A report either volunteers an outcome (the browser
 * decided something) or answers a probe (the model asked what is on screen).
 * Both carry the current view state, because both are moments when the browser
 * knows it. `ViewLink` is the rendezvous: a waiter registered before the
 * question goes out, resolved by the report that answers it, and TIMED OUT
 * HONESTLY when no report arrives.
 *
 * THE UNKNOWN IS A VALUE, NOT A DEFAULT. A closed tab, a client that is not a
 * browser, a slow link: all of them end in "no verdict arrived", and every path
 * here returns null for that rather than assuming success. A tool that told the
 * model "applied" on a timeout would be teaching it to tell the person the view
 * moved when nothing did, which is the precise failure this whole mechanism
 * exists to prevent.
 *
 * Everything except the registry is a pure function, so the wire contract can be
 * tested without a server, a browser or a socket.
 */

/**
 * The independently-owned axes of the view — front/ui/src/lib/view/authority.ts
 * `VIEW_DIMENSIONS`, transcribed.
 *
 * DUPLICATED AS DATA for the same reason the destination table is (view-tools.ts):
 * `front/ui` and `seat` are separate packages with no import path between them.
 * The duplication is made safe by the boundary — a dimension the browser reports
 * that is not in this list is REJECTED with a 400 naming these, never coerced,
 * so drift surfaces as a loud failure on the first report rather than as a
 * silently mislabelled row in an audit trail.
 */
export const VIEW_DIMENSIONS = ["cue", "frame", "destination", "focus"] as const;
export type ViewDimension = (typeof VIEW_DIMENSIONS)[number];

/**
 * What became of one command, on one dimension.
 *
 * A CLOSED SET, AND EVERY MEMBER IS A TRANSITION THE BROWSER'S STORE ACTUALLY
 * PERFORMS — none of them is inferred by watching state change:
 *
 *  - `applied`    — the view moved now. The person is looking at it.
 *  - `already`    — the view was already there, so nothing moved and nothing
 *                   waits. Distinct from `applied` because no change occurred;
 *                   the same as `applied` in what the person can see.
 *  - `offered`    — the person held this axis, so the command waits as a Follow
 *                   they can accept. NOT a terminal state: a later report says
 *                   how it ended.
 *  - `followed`   — the person accepted the offer, and the view moved then.
 *  - `declined`   — the person refused it: they pressed Dismiss, or they took
 *                   that axis by hand, which is the same statement made with a
 *                   different gesture.
 *  - `expired`    — the turn ended with the offer unanswered. Nobody said no;
 *                   nobody said yes. Kept apart from `declined` because "the
 *                   person refused" and "the person never saw it" are different
 *                   facts about the same person.
 *  - `superseded` — the model itself issued a newer command for that dimension
 *                   while this one was still waiting.
 *
 * There is deliberately no `unknown` member. An outcome that never arrived
 * produces NO record at all — absence is how this system says "not known", and a
 * state value spelling it would invite a row asserting that the browser reported
 * something it did not.
 */
export const VIEW_OUTCOME_STATES = [
	"applied",
	"already",
	"offered",
	"followed",
	"declined",
	"expired",
	"superseded",
] as const;
export type ViewOutcomeState = (typeof VIEW_OUTCOME_STATES)[number];

/** States in which the view is showing what the command asked for. */
const HONOURED: ReadonlySet<string> = new Set<ViewOutcomeState>(["applied", "already", "followed"]);

/** States in which nothing further will happen — the ask is closed. */
const TERMINAL: ReadonlySet<string> = new Set<ViewOutcomeState>([
	"applied",
	"already",
	"followed",
	"declined",
	"expired",
	"superseded",
]);

export function isHonoured(state: ViewOutcomeState): boolean {
	return HONOURED.has(state);
}

export function isTerminal(state: ViewOutcomeState): boolean {
	return TERMINAL.has(state);
}

/** One dimension's fate, as the browser reports it. */
export interface ViewOutcome {
	/** The `direct_view` call this answers. */
	tool_call_id: string;
	dimension: ViewDimension;
	state: ViewOutcomeState;
}

/** The narrowing in force, and whose it is. */
export interface ViewCueState {
	text: string;
	entities: string[];
	/** "agent" only while the bus's own record still matches what is on screen. */
	author: "user" | "agent";
}

/** The single object open in the Inspector. */
export interface ViewFocusState {
	/** An entity uuid — `UniverseStar.id` (src/graph_memory.rs), which is
	 *  `EntityNode.uuid`, which is what `POST /api/graph/entity/find` returns. */
	id: string;
	/** The graph's name for it, or null when the browser has not loaded the
	 *  universe that would name it. Null is "this seat does not know", never a
	 *  guess. */
	name: string | null;
}

/**
 * What is on screen, as the browser sees it.
 *
 * WHAT IS DELIBERATELY ABSENT is as much the contract as what is here. No memory
 * text, no recall results, no conversation content, no provider or credential
 * state, no DOM and no pixels. A perception tool that returned the corpus would
 * be a retrieval tool wearing a getter's clothes — it would put content into the
 * model's context that nobody asked to retrieve and that no audit row describes
 * as a retrieval. This says WHERE the person is and WHAT IS NARROWED, which is
 * what a model needs to decide whether to move them, and nothing more.
 */
export interface ViewSnapshot {
	/** Router path, e.g. "/geo". */
	destination: string;
	/** The memory profile the workbench is pointed at, or null before one loads. */
	profile: string | null;
	cue: ViewCueState | null;
	focus: ViewFocusState | null;
	/** Axes the person holds for the rest of this turn. An agent command on one
	 *  of these becomes an offer instead of a move. */
	claimed: ViewDimension[];
	/** Follow offers waiting for an answer right now. */
	offers: { dimension: ViewDimension; reason: string }[];
}

/** One `POST /v1/conversations/{id}/view-report` body, validated. */
export interface ViewReport {
	/** The probe this answers, or null when the browser volunteered it. */
	probe_id: string | null;
	outcomes: ViewOutcome[];
	view: ViewSnapshot;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] | null {
	if (!Array.isArray(value)) return null;
	const out: string[] = [];
	for (const item of value) {
		if (typeof item !== "string") return null;
		out.push(item);
	}
	return out;
}

/** How many outcomes one report may carry. One `direct_view` produces at most
 *  one command per dimension, and a report batches at most one call's worth plus
 *  the offers it resolved; the cap exists so a malformed client cannot make the
 *  seat write an unbounded number of audit rows from a single request. */
export const MAX_OUTCOMES_PER_REPORT = 32;

/**
 * Read a report body, or say exactly what is wrong with it.
 *
 * STRICT, AND NOTHING IS COERCED. An unknown dimension or state is rejected with
 * the valid list rather than dropped or mapped to a neighbour: this body becomes
 * durable audit rows, and a row that says `declined` because the seat did not
 * recognise `dismissed` is a false statement about what a person did. The error
 * strings name the closed sets so a client (or a newer browser talking to an
 * older seat) is told precisely what this build accepts.
 */
export function parseViewReport(raw: unknown): { report: ViewReport } | { error: string } {
	if (!isRecord(raw)) return { error: "Body must be a JSON object" };

	let probeId: string | null = null;
	if (raw.probe_id !== undefined && raw.probe_id !== null) {
		if (typeof raw.probe_id !== "string" || raw.probe_id.length === 0) {
			return { error: "probe_id must be a non-empty string when present" };
		}
		probeId = raw.probe_id;
	}

	const outcomes: ViewOutcome[] = [];
	if (raw.outcomes !== undefined) {
		if (!Array.isArray(raw.outcomes)) return { error: "outcomes must be an array" };
		if (raw.outcomes.length > MAX_OUTCOMES_PER_REPORT) {
			return { error: `outcomes must hold at most ${MAX_OUTCOMES_PER_REPORT} entries` };
		}
		for (const item of raw.outcomes) {
			if (!isRecord(item)) return { error: "each outcome must be an object" };
			if (typeof item.tool_call_id !== "string" || item.tool_call_id.length === 0) {
				return { error: "each outcome needs a non-empty tool_call_id" };
			}
			if (!(VIEW_DIMENSIONS as readonly string[]).includes(item.dimension as string)) {
				return { error: `dimension must be one of: ${VIEW_DIMENSIONS.join(", ")}` };
			}
			if (!(VIEW_OUTCOME_STATES as readonly string[]).includes(item.state as string)) {
				return { error: `state must be one of: ${VIEW_OUTCOME_STATES.join(", ")}` };
			}
			outcomes.push({
				tool_call_id: item.tool_call_id,
				dimension: item.dimension as ViewDimension,
				state: item.state as ViewOutcomeState,
			});
		}
	}

	const view = raw.view;
	if (!isRecord(view)) return { error: "view is required and must be an object" };
	if (typeof view.destination !== "string" || view.destination.length === 0) {
		return { error: "view.destination must be a non-empty path" };
	}
	if (view.profile !== null && typeof view.profile !== "string") {
		return { error: "view.profile must be a string or null" };
	}

	let cue: ViewCueState | null = null;
	if (view.cue !== undefined && view.cue !== null) {
		if (!isRecord(view.cue)) return { error: "view.cue must be an object or null" };
		const entities = stringArray(view.cue.entities);
		if (typeof view.cue.text !== "string" || entities === null) {
			return { error: "view.cue needs a text string and an entities string array" };
		}
		if (view.cue.author !== "user" && view.cue.author !== "agent") {
			return { error: 'view.cue.author must be "user" or "agent"' };
		}
		cue = { text: view.cue.text, entities, author: view.cue.author };
	}

	let focus: ViewFocusState | null = null;
	if (view.focus !== undefined && view.focus !== null) {
		if (!isRecord(view.focus)) return { error: "view.focus must be an object or null" };
		if (typeof view.focus.id !== "string" || view.focus.id.length === 0) {
			return { error: "view.focus.id must be a non-empty string" };
		}
		if (view.focus.name !== null && typeof view.focus.name !== "string") {
			return { error: "view.focus.name must be a string or null" };
		}
		focus = { id: view.focus.id, name: view.focus.name };
	}

	const claimedRaw = stringArray(view.claimed);
	if (claimedRaw === null) return { error: "view.claimed must be an array of dimensions" };
	for (const dimension of claimedRaw) {
		if (!(VIEW_DIMENSIONS as readonly string[]).includes(dimension)) {
			return { error: `view.claimed may only hold: ${VIEW_DIMENSIONS.join(", ")}` };
		}
	}

	if (!Array.isArray(view.offers)) return { error: "view.offers must be an array" };
	const offers: { dimension: ViewDimension; reason: string }[] = [];
	for (const item of view.offers) {
		if (!isRecord(item)) return { error: "each offer must be an object" };
		if (!(VIEW_DIMENSIONS as readonly string[]).includes(item.dimension as string)) {
			return { error: `offer dimension must be one of: ${VIEW_DIMENSIONS.join(", ")}` };
		}
		if (typeof item.reason !== "string") return { error: "each offer needs a reason string" };
		offers.push({ dimension: item.dimension as ViewDimension, reason: item.reason });
	}

	return {
		report: {
			probe_id: probeId,
			outcomes,
			view: {
				destination: view.destination,
				profile: view.profile as string | null,
				cue,
				focus,
				claimed: claimedRaw as ViewDimension[],
				offers,
			},
		},
	};
}

/**
 * How long a tool waits for the browser to answer.
 *
 * The browser reports on the same tick it dispatches, so on a loopback bind the
 * round trip is a few milliseconds and this is two orders of magnitude of
 * headroom. It is a ceiling on a wrong answer, not a target: the cost of waiting
 * is a pause in one tool call, and the cost of NOT waiting is a model that
 * describes a view it has never seen.
 *
 * It is also the cost paid by every caller that is not a browser — an eval
 * harness, a curl session — and that is the correct trade. Those callers pay two
 * seconds per view tool call; a person gets a model that does not lie to them.
 */
export const VIEW_REPLY_TIMEOUT_MS = 2_000;

/** How long an ask stays answerable after it is issued. A Follow offer can sit
 *  on screen for as long as the turn lasts and be accepted well after the tool
 *  returned, so the ask must still validate then; ten minutes is far past any
 *  turn and still bounds the map. */
const ASK_RETENTION_MS = 600_000;

/** Ceiling on remembered asks, so a long session cannot grow this without
 *  bound. Oldest first — an ask that old is past {@link ASK_RETENTION_MS}
 *  anyway. */
const MAX_REMEMBERED_ASKS = 512;

interface Waiter<T> {
	resolve(value: T | null): void;
	timer: ReturnType<typeof setTimeout>;
}

interface Ask {
	conversationId: string;
	issuedAt: number;
	waiter: Waiter<ViewOutcome[]> | null;
	/**
	 * Outcomes that arrived before anyone was waiting for them.
	 *
	 * BUFFERED, NOT DROPPED. The caller registers the ask, emits, and then waits,
	 * and today nothing yields between those last two — so the report cannot
	 * overtake the waiter. That is a property of the current emit path, not of
	 * this class, and it would stop being true the moment emitting became
	 * asynchronous. A verdict discarded because it was two lines early would
	 * surface as an occasional "not known" over an answer that had arrived, which
	 * is the hardest possible version of this bug to find. So the answer is kept
	 * either way and `await` returns it immediately.
	 */
	delivered: ViewOutcome[] | null;
}

/**
 * The rendezvous between a tool call and the browser's answer.
 *
 * ONE PER SEAT PROCESS, shared by the HTTP route and every conversation's tools,
 * because the two ends of one question live in different objects: the tool that
 * asks runs inside `Conversation`, and the report that answers arrives on the
 * server's router.
 */
export class ViewLink {
	private readonly asks = new Map<string, Ask>();
	private readonly probes = new Map<string, { conversationId: string; waiter: Waiter<ViewSnapshot> | null }>();
	/** The last snapshot each conversation's browser reported, with its arrival
	 *  time. Never served as "current" — only ever as "this is how old it is". */
	private readonly snapshots = new Map<string, { view: ViewSnapshot; at: number }>();
	private readonly timeoutMs: number;
	private probeSeq = 0;

	constructor(timeoutMs: number = VIEW_REPLY_TIMEOUT_MS) {
		this.timeoutMs = timeoutMs;
	}

	/**
	 * Register an ask BEFORE its event is emitted.
	 *
	 * THE ORDER IS THE WHOLE POINT. On a loopback bind the browser's report can
	 * arrive before the emitting call returns, so a waiter registered after the
	 * emit would miss the answer it was created to catch and time out over a
	 * verdict that was sitting in the map. Callers must `open` first, `emit`
	 * second, `await` third.
	 */
	open(conversationId: string, toolCallId: string): void {
		this.evictStale();
		this.asks.set(toolCallId, { conversationId, issuedAt: Date.now(), waiter: null, delivered: null });
	}

	/** Whether this call was issued by this process and is still answerable. */
	knows(toolCallId: string): boolean {
		const ask = this.asks.get(toolCallId);
		if (!ask) return false;
		if (Date.now() - ask.issuedAt > ASK_RETENTION_MS) {
			this.asks.delete(toolCallId);
			return false;
		}
		return true;
	}

	/**
	 * The outcomes for one ask, or null when none arrived in time.
	 *
	 * Null is the honest unknown and the caller must render it as such. It covers
	 * every reason the browser did not answer — no browser, a closed tab, a
	 * client that is not a browser at all — and none of those is distinguishable
	 * from here, which is exactly why they collapse to one value instead of to a
	 * guess.
	 */
	await(toolCallId: string): Promise<ViewOutcome[] | null> {
		const ask = this.asks.get(toolCallId);
		if (!ask) return Promise.resolve(null);
		if (ask.delivered !== null) return Promise.resolve(ask.delivered);
		return new Promise((resolve) => {
			const timer = setTimeout(() => {
				if (ask.waiter?.timer === timer) ask.waiter = null;
				resolve(null);
			}, this.timeoutMs);
			// DELIBERATELY NOT UNREF'D. An unref'd timer does not hold the event
			// loop open, so on a quiet loop the process can exit — or the timer
			// simply never fire — leaving this promise pending forever and the
			// tool call hung. A settled "not known" two seconds late is the whole
			// contract; a promise that never settles is worse than either answer.
			ask.waiter = { resolve, timer };
		});
	}

	/** Issue a probe id and register its waiter. Same order rule as {@link open}. */
	openProbe(conversationId: string): string {
		const probeId = `probe-${Date.now().toString(36)}-${(this.probeSeq += 1).toString(36)}`;
		// The waiter is attached by `awaitProbe`. Registering the id first is the
		// same ordering rule as `open`: the probe must be known before the event
		// carrying it leaves, or the answer arrives at an empty map.
		this.probes.set(probeId, { conversationId, waiter: null });
		return probeId;
	}

	/** The snapshot answering a probe, or null when none arrived in time. */
	awaitProbe(probeId: string): Promise<ViewSnapshot | null> {
		const probe = this.probes.get(probeId);
		if (!probe) return Promise.resolve(null);
		return new Promise((resolve) => {
			const timer = setTimeout(() => {
				this.probes.delete(probeId);
				resolve(null);
			}, this.timeoutMs);
			// See the note in `await`: this timer is not unref'd either.
			probe.waiter = { resolve, timer };
		});
	}

	/**
	 * Deliver a report. Returns the outcomes whose ask this process issued.
	 *
	 * OUTCOMES FOR UNKNOWN ASKS ARE RETURNED TOO, and the caller decides — the
	 * seat can restart while a Follow offer is still on screen, and refusing that
	 * report would drop the one record of a person accepting an offer. The route
	 * checks the durable event store before rejecting; see server.ts.
	 */
	report(conversationId: string, report: ViewReport): void {
		this.snapshots.set(conversationId, { view: report.view, at: Date.now() });

		if (report.probe_id !== null) {
			const probe = this.probes.get(report.probe_id);
			// A probe answered from a different conversation is not this probe.
			// Ignored rather than trusted: the id is the correlation and the
			// conversation is the check on it.
			if (probe && probe.conversationId === conversationId) {
				this.probes.delete(report.probe_id);
				if (probe.waiter) {
					clearTimeout(probe.waiter.timer);
					probe.waiter.resolve(report.view);
				}
			}
		}

		// Grouped per ask, so a call whose three dimensions landed differently
		// resolves its waiter once, with all three.
		const byAsk = new Map<string, ViewOutcome[]>();
		for (const outcome of report.outcomes) {
			const list = byAsk.get(outcome.tool_call_id);
			if (list) list.push(outcome);
			else byAsk.set(outcome.tool_call_id, [outcome]);
		}
		for (const [toolCallId, outcomes] of byAsk) {
			const ask = this.asks.get(toolCallId);
			if (!ask || ask.conversationId !== conversationId) continue;
			// FIRST REPORT WINS. Two tabs on one conversation each dispatch and
			// each report, and their verdicts can differ; the tool answers with
			// the first that arrives rather than merging two people's views into
			// one sentence. Both reports still become audit rows — the trail
			// records both browsers, the model is told about one.
			if (ask.delivered !== null) continue;
			ask.delivered = outcomes;
			const waiter = ask.waiter;
			if (!waiter) continue;
			ask.waiter = null;
			clearTimeout(waiter.timer);
			waiter.resolve(outcomes);
		}
	}

	/** The last snapshot this conversation's browser sent, and its age in ms. */
	lastSnapshot(conversationId: string): { view: ViewSnapshot; ageMs: number } | null {
		const entry = this.snapshots.get(conversationId);
		if (!entry) return null;
		return { view: entry.view, ageMs: Date.now() - entry.at };
	}

	/** Drop everything belonging to a conversation that is going away. */
	forget(conversationId: string): void {
		this.snapshots.delete(conversationId);
		for (const [id, ask] of this.asks) if (ask.conversationId === conversationId) this.asks.delete(id);
		for (const [id, probe] of this.probes) {
			if (probe.conversationId !== conversationId) continue;
			if (probe.waiter) clearTimeout(probe.waiter.timer);
			this.probes.delete(id);
		}
	}

	private evictStale(): void {
		const now = Date.now();
		for (const [id, ask] of this.asks) {
			if (now - ask.issuedAt > ASK_RETENTION_MS) this.asks.delete(id);
		}
		// Map iteration is insertion-ordered, so this drops the oldest first.
		while (this.asks.size >= MAX_REMEMBERED_ASKS) {
			const oldest = this.asks.keys().next();
			if (oldest.done) break;
			this.asks.delete(oldest.value);
		}
	}
}

/* ------------------------------------------------------------------ *
 * WHAT THE MODEL IS TOLD
 * ------------------------------------------------------------------ */

/** How a dimension is spoken to the model. Not the internal name: "the camera"
 *  is what `frame` is, and a model reading "frame: applied" has to guess. */
const DIMENSION_NOUN: Record<ViewDimension, string> = {
	cue: "the narrowing",
	frame: "the camera",
	destination: "the destination",
	focus: "the opened entity",
};

/**
 * One dimension's outcome, in a sentence.
 *
 * WHY THE PERSON IS NAMED AS THE HOLDER on every non-applied state: the
 * authority rule grants a claim to nobody else. `decide` returns "offer" only
 * when the author is the agent and the dimension is in `claimed`, and `claimed`
 * is written only by the person's own hand, by their Dismiss, or by the surface
 * they were reading when the turn began. So "who holds it" is not a variable
 * this wire needs to carry — it is an invariant of the rule, and stating it as
 * one is more honest than a field that could only ever hold one value.
 */
export function describeOutcome(outcome: ViewOutcome): string {
	const noun = DIMENSION_NOUN[outcome.dimension];
	switch (outcome.state) {
		case "applied":
			return `${noun}: MOVED. The person is looking at it now.`;
		case "already":
			return `${noun}: ALREADY THERE. Nothing moved, because the view was already showing it.`;
		case "offered":
			return `${noun}: WAITING. The person holds this axis, so your request is on screen as a Follow they can accept. It has NOT been applied.`;
		case "followed":
			return `${noun}: ACCEPTED. The person took the offer and the view moved.`;
		case "declined":
			return `${noun}: REFUSED. The person dismissed the offer, or moved that axis themselves.`;
		case "expired":
			return `${noun}: UNANSWERED. The offer lapsed when the turn ended — the person neither accepted nor refused it.`;
		case "superseded":
			return `${noun}: REPLACED by a later request of your own, before the person answered this one.`;
	}
}

/**
 * The verdict paragraph appended to a `direct_view` result.
 *
 * `null` outcomes mean nothing arrived, and that is written out in full rather
 * than left as silence: silence is what the model reads as success, and the one
 * behaviour this feature exists to stop is a model reporting a move it cannot
 * see. The instruction is explicit because the failure is a sentence the model
 * writes to a person, not an internal state.
 */
export function composeVerdict(outcomes: readonly ViewOutcome[] | null): string {
	if (outcomes === null) {
		return (
			"VERDICT NOT KNOWN. The workbench did not answer within " +
			`${VIEW_REPLY_TIMEOUT_MS / 1000}s — the tab may be closed, or nothing may be watching this conversation. ` +
			"Do NOT tell the person the view moved, and do not repeat the request; say what you found and let them look."
		);
	}
	if (outcomes.length === 0) {
		return (
			// The causes are NO LONGER ENUMERATED here, and that is a correction.
			// This sentence used to say "every entity you named resolved to
			// nothing, or the view was already exactly as asked", which became
			// false the moment a term could also fail to be CHECKED — and it would
			// have been false in the direction that matters, telling the model the
			// person's graph lacks something nobody ever looked for. The paragraph
			// above this one already separates framed, absent and unchecked by
			// name; this line's job is only to say that no axis moved.
			"VERDICT: nothing was actionable. The workbench received the request and found no axis to change. " +
			"The report above says which of your terms were framed, which the graph does not hold, and which could " +
			"not be checked — read it there rather than assuming a cause."
		);
	}
	const lines = outcomes.map((outcome) => `- ${describeOutcome(outcome)}`);
	if (outcomes.some((outcome) => outcome.state === "offered")) {
		lines.push(
			"A waiting offer is the person's to accept. Do not re-issue the request to get around it — " +
				"they outrank you on any axis they are holding, and asking twice is the same as not asking.",
		);
	}
	return `VERDICT from the workbench:\n${lines.join("\n")}`;
}

/**
 * The view state, in the words a model can act on.
 *
 * `ageMs` is stated whenever the snapshot was not fetched for this call, because
 * a cached picture presented as current is the same class of lie as an assumed
 * verdict. A live probe answers with age zero and says so by not mentioning age
 * at all.
 */
export function composeViewReport(input: { view: ViewSnapshot; ageMs: number | null; destinationNoun: string | null }): string {
	const lines: string[] = [];
	const { view, ageMs, destinationNoun } = input;

	lines.push(
		`On screen: ${destinationNoun ?? view.destination}` +
			(destinationNoun ? ` (${view.destination})` : " — a path this build has no name for"),
	);
	lines.push(`Profile: ${view.profile ?? "none selected yet"}.`);

	if (view.cue) {
		const whose = view.cue.author === "agent" ? "set by you" : "the person's own";
		const entities = view.cue.entities.length > 0 ? ` — lighting ${view.cue.entities.join(", ")}` : "";
		lines.push(`Narrowed to "${view.cue.text}" (${whose})${entities}.`);
	} else {
		lines.push("Nothing is narrowed: the whole corpus is in view.");
	}

	if (view.focus) {
		lines.push(
			view.focus.name
				? `Open in the inspector: ${view.focus.name}.`
				: `Open in the inspector: entity ${view.focus.id} — the workbench has not loaded the graph that would name it.`,
		);
	}

	if (view.claimed.length > 0) {
		lines.push(
			`THE PERSON HOLDS ${view.claimed.map((d) => DIMENSION_NOUN[d]).join(", ")} for the rest of this turn. ` +
				"A direct_view touching those axes will WAIT as an offer rather than move anything — expect it.",
		);
	} else {
		lines.push("The person has not taken any axis this turn, so a direct_view would apply immediately.");
	}

	for (const offer of view.offers) {
		lines.push(`Waiting on screen: an offer to change ${DIMENSION_NOUN[offer.dimension]} — "${offer.reason}".`);
	}

	if (ageMs !== null) {
		lines.push(
			`This is the last state the workbench reported, ${Math.round(ageMs / 1000)}s ago, not a fresh reading — ` +
				"it did not answer this request, so anything the person has done since is not in it.",
		);
	}

	return lines.join("\n");
}
