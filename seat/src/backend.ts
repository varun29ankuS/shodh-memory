/**
 * Typed HTTP client for the shodh-memory Rust backend.
 *
 * Every shape below is transcribed from the actual handler source
 * (src/handlers/types.rs, src/handlers/recall.rs, src/handlers/remember.rs,
 * src/handlers/crud.rs, src/memory/types.rs) — field names match the serde
 * serialization exactly. Do not "fix" casing here without re-reading the Rust.
 */

/** src/memory/types.rs `ScoreAttribution` — present per memory when recall is called with debug:true. */
export interface ScoreAttribution {
	memory_id: string;
	rrf_base: number;
	graph_rrf: number;
	hybrid_rrf: number;
	hebbian_boost: number;
	attribute_boost: number;
	temporal_prefilter_boost: number;
	temporal_fact_boost: number;
	interference_adjustment: number;
	prospective_boost: number;
	fact_source_boost: number;
	ontological_boost: number;
	importance_factor: number;
	recency_factor: number;
	arousal_factor: number;
	credibility_factor: number;
	feedback_multiplier: number;
	quality_gate: number;
	final_score: number;
	sources: string[];
}

/** src/handlers/types.rs `RecallExperience` */
export interface RecallExperience {
	content: string;
	memory_type: string | null;
	tags: string[];
}

/** src/handlers/types.rs `RecallMemory` */
export interface RecallMemory {
	id: string;
	experience: RecallExperience;
	importance: number;
	created_at: string;
	score: number;
	tier: string;
	score_attribution?: ScoreAttribution;
}

/** src/handlers/types.rs `RecallFact` */
export interface RecallFact {
	id: string;
	fact: string;
	confidence: number;
	support_count: number;
	related_entities: string[];
}

/** src/handlers/types.rs `RecallTodo` */
export interface RecallTodo {
	id: string;
	short_id: string;
	content: string;
	status: string;
	priority: string;
	project: string | null;
	due_date: string | null;
	score: number;
}

/** src/handlers/types.rs `RecallLineageEdge` */
export interface RecallLineageEdge {
	from: string;
	to: string;
	relation: string;
	confidence: number;
}

/** src/handlers/types.rs `RecallReminder` */
export interface RecallReminder {
	id: string;
	content: string;
	keywords: string[];
	match_type: string;
	priority: number;
	created_at: string;
}

/** src/handlers/types.rs `RecallResponse` (vectors are skip_serializing_if empty → optional here). */
export interface RecallResponse {
	memories: RecallMemory[];
	count: number;
	retrieval_stats?: unknown;
	todos?: RecallTodo[];
	todo_count?: number;
	facts?: RecallFact[];
	fact_count?: number;
	triggered_reminders?: RecallReminder[];
	reminder_count?: number;
	lineage?: RecallLineageEdge[];
	lineage_count?: number;
}

/** Valid `mode` values, from src/handlers/recall.rs `parse_retrieval_mode`. */
export type RecallMode =
	| "hybrid"
	| "semantic"
	| "associative"
	| "temporal"
	| "causal"
	| "spatial"
	| "mission"
	| "action_outcome";

/** Valid memory types, from src/handlers/remember.rs `parse_experience_type`. */
export type MemoryType =
	| "observation"
	| "decision"
	| "learning"
	| "error"
	| "discovery"
	| "pattern"
	| "context"
	| "task"
	| "code_edit"
	| "file_access"
	| "search"
	| "command"
	| "conversation"
	| "intention";

/** src/handlers/recall.rs `reinforce_feedback` outcome strings. */
export type ReinforceOutcome = "helpful" | "misleading" | "neutral";

/** src/handlers/recall.rs `ReinforceFeedbackResponse` */
export interface ReinforceStats {
	memories_processed: number;
	associations_strengthened: number;
	importance_boosts: number;
	importance_decays: number;
}

/** src/handlers/remember.rs `RememberResponse` */
export interface RememberResponse {
	id: string;
	success: boolean;
}

/** src/memory/feedback.rs `ToolAction` — tool-aware feedback attribution input. */
export interface ToolAction {
	tool_name: string;
	inputs: Record<string, string>;
	success: boolean;
	output_snippet?: string;
}

/** src/handlers/recall.rs `ProactiveSurfacedMemory` (embedding is #[serde(skip)] — never sent). */
export interface ProactiveSurfacedMemory {
	id: string;
	content: string;
	memory_type: string;
	score: number;
	importance: number;
	created_at: string;
	tags: string[];
	tier: string;
	relevance_reason: string;
	matched_entities?: string[];
}

/** src/handlers/recall.rs `FeedbackProcessed` */
export interface FeedbackProcessed {
	memories_evaluated: number;
	reinforced: string[];
	weakened: string[];
}

/** src/handlers/recall.rs `ProactiveContextResponse` (fields the seat consumes; extras tolerated). */
export interface ProactiveContextResponse {
	memories: ProactiveSurfacedMemory[];
	memory_count: number;
	feedback_processed?: FeedbackProcessed;
	temporal_credits_applied?: number;
	latency_ms: number;
	ingested_memory_id?: string;
}

/**
 * Why a backend call failed, as a closed set of compile-time constants.
 *
 * The distinction this preserves cannot be recovered downstream: `request()`
 * flattens the original error into a message, and for every transport failure
 * except a timeout that message is exactly "fetch failed" — the code lives on
 * `.cause`, which the flattening discards. Classification therefore happens at
 * the throw site or not at all.
 *
 * Every member being a constant is what makes a value of this type safe to
 * publish from the UNAUTHENTICATED /healthz route (server.ts `route`); the
 * underlying message is not, because it quotes the backend's host and port.
 */
export type BackendFailureKind =
	| "timeout"
	| "refused"
	| "dns"
	| "tls"
	| "reset"
	| "unreachable"
	| "protocol"
	| "http"
	| "other";

/**
 * What /healthz may publish for a failed probe. Widening this to a bare string
 * is what would let a hostname or URL back into an unauthenticated response, so
 * the HTTP case is pinned to a status number rather than left interpolatable.
 */
export type HealthDetail = BackendFailureKind | `http-${number}` | "http-client-error";

/**
 * How an HTTP failure is named on the unauthenticated health route.
 *
 * 5xx is published with its status: "the backend is up and broken" is what an
 * operator needs, and a server-error code says nothing about this deployment's
 * secrets. 4xx is collapsed. A bare `http-401` would tell an anonymous caller
 * that the seat's own backend credential is being rejected, which is a fact
 * about our configuration rather than about the backend's liveness, and this
 * route answers before authorize().
 */
export function healthDetailForHttp(status: number): HealthDetail {
	return status >= 400 && status < 500 ? "http-client-error" : `http-${status}`;
}

/** TLS failures report cert codes that carry no ERR_TLS_/ERR_SSL_ prefix. */
const TLS_CERT_CODES: ReadonlySet<string> = new Set([
	"DEPTH_ZERO_SELF_SIGNED_CERT",
	"SELF_SIGNED_CERT_IN_CHAIN",
	"CERT_HAS_EXPIRED",
	"CERT_NOT_YET_VALID",
	"UNABLE_TO_VERIFY_LEAF_SIGNATURE",
	"ERR_TLS_CERT_ALTNAME_INVALID",
]);

/**
 * Walk an error and everything underneath it, outermost first.
 *
 * `fetch` wraps transport failures in `TypeError: fetch failed` and hangs the
 * real error off `.cause`; a dual-stack host (localhost resolving to both ::1
 * and 127.0.0.1) nests an AggregateError in that position with one entry per
 * address. The `seen` set guards against a self-referential `cause`.
 */
function* walkCauses(error: unknown): Generator<unknown> {
	const seen = new Set<unknown>();
	const queue: unknown[] = [error];
	while (queue.length > 0) {
		const current = queue.shift();
		if (current === null || current === undefined || seen.has(current)) continue;
		seen.add(current);
		yield current;
		if (current instanceof AggregateError) queue.push(...current.errors);
		if (current instanceof Error && current.cause !== undefined) queue.push(current.cause);
	}
}

/**
 * Classify a transport failure from the original error object.
 *
 * Verified against Node 26 in eval/backend-classify.test.mjs. Note the `code`
 * type guard: a timeout arrives as a DOMException whose `code` is the numeric
 * legacy value 23, not a string, so comparing it against the errno names would
 * silently match nothing.
 */
export function classifyTransportError(error: unknown): BackendFailureKind {
	for (const node of walkCauses(error)) {
		const candidate = node as { name?: unknown; code?: unknown };
		const name = typeof candidate.name === "string" ? candidate.name : "";
		const code = typeof candidate.code === "string" ? candidate.code : "";

		if (name === "TimeoutError" || name === "AbortError") return "timeout";
		if (TLS_CERT_CODES.has(code) || code.startsWith("ERR_TLS_") || code.startsWith("ERR_SSL_")) {
			return "tls";
		}

		switch (code) {
			case "ECONNREFUSED":
				return "refused";
			// Resolver behaviour differs by platform and by CI runner: a
			// sandboxed runner can report EAI_AGAIN or EAI_NONAME where a
			// desktop reports ENOTFOUND. They are one failure to an operator.
			case "ENOTFOUND":
			case "EAI_AGAIN":
			case "EAI_NONAME":
			case "EAI_FAIL":
			case "EAI_NODATA":
			case "ENODATA":
				return "dns";
			case "ECONNRESET":
			case "EPIPE":
			case "UND_ERR_SOCKET":
				return "reset";
			case "ETIMEDOUT":
			case "UND_ERR_CONNECT_TIMEOUT":
			case "UND_ERR_HEADERS_TIMEOUT":
			case "UND_ERR_BODY_TIMEOUT":
				return "timeout";
			case "EHOSTUNREACH":
			case "ENETUNREACH":
				return "unreachable";
		}
	}
	return "other";
}

export class ShodhBackendError extends Error {
	readonly status: number;
	readonly body: string;
	/**
	 * Why the call failed. Not settable directly — see the factories below.
	 *
	 * A required constructor parameter forces a throw site to make a CHOICE,
	 * but not a correct one: passing "other" satisfies the type checker while
	 * discarding the diagnosis. The factories remove the choice instead, so the
	 * only way to get a transport kind is to hand over the original error and
	 * let it be classified.
	 */
	readonly kind: BackendFailureKind;

	private constructor(message: string, status: number, body: string, kind: BackendFailureKind) {
		super(message);
		this.name = "ShodhBackendError";
		this.status = status;
		this.body = body;
		this.kind = kind;
	}

	/** The request never produced a response. The cause decides the kind. */
	static transport(message: string, cause: unknown): ShodhBackendError {
		return new ShodhBackendError(message, 0, "", classifyTransportError(cause));
	}

	/** The backend answered with a non-2xx status: reachable, and erroring. */
	static http(message: string, status: number, body: string): ShodhBackendError {
		return new ShodhBackendError(message, status, body, "http");
	}

	/** The backend answered, but not with JSON — a live endpoint, wrong protocol. */
	static protocol(message: string, status: number, body: string): ShodhBackendError {
		return new ShodhBackendError(message, status, body, "protocol");
	}
}

export interface RecallParams {
	userId: string;
	query: string;
	limit?: number;
	mode?: RecallMode;
	/** debug:true asks the pipeline for per-memory ScoreAttribution. */
	debug?: boolean;
}

export interface RememberParams {
	userId: string;
	content: string;
	memoryType?: MemoryType;
	tags?: string[];
	importance?: number;
}

export interface ProactiveContextParams {
	userId: string;
	context: string;
	maxResults?: number;
	semanticThreshold?: number;
	/**
	 * Required, no default: the backend defaults auto_ingest to TRUE
	 * (src/handlers/recall.rs:172-174) and silently ingests the previous
	 * response as memories. Callers must decide explicitly.
	 */
	autoIngest: boolean;
	/** Previous assistant message — triggers implicit-feedback processing of the pending surfaced set. */
	previousResponse?: string;
	/** The user message following that response (negative-keyword detection). */
	userFollowup?: string;
	/** Tool actions performed since the pending set was surfaced (tool-aware attribution). */
	toolActions?: ToolAction[];
}

export class ShodhBackend {
	private readonly apiUrl: string;
	private readonly apiKey: string;
	private readonly timeoutMs: number;

	constructor(apiUrl: string, apiKey: string, timeoutMs: number) {
		this.apiUrl = apiUrl;
		this.apiKey = apiKey;
		this.timeoutMs = timeoutMs;
	}

	private async request<T>(method: "GET" | "POST" | "DELETE", pathname: string, body?: unknown): Promise<T> {
		const url = `${this.apiUrl}${pathname}`;
		let response: Response;
		try {
			response = await fetch(url, {
				method,
				headers: {
					"Content-Type": "application/json",
					"X-API-Key": this.apiKey,
				},
				body: body === undefined ? undefined : JSON.stringify(body),
				signal: AbortSignal.timeout(this.timeoutMs),
			});
		} catch (error) {
			const reason = error instanceof Error ? error.message : String(error);
			throw ShodhBackendError.transport(
				`Backend unreachable (${method} ${pathname}): ${reason}`,
				error,
			);
		}
		const text = await response.text();
		if (!response.ok) {
			throw ShodhBackendError.http(
				`Backend error ${response.status} on ${method} ${pathname}`,
				response.status,
				text.slice(0, 2000),
			);
		}
		try {
			return JSON.parse(text) as T;
		} catch {
			// A proxy's HTML error page lands here. The backend answered, so this
			// is not an unreachable one — it is a live endpoint speaking the wrong
			// protocol, and conflating the two sends operators to the wrong place.
			throw ShodhBackendError.protocol(
				`Backend returned non-JSON on ${method} ${pathname}`,
				response.status,
				text.slice(0, 2000),
			);
		}
	}

	/** POST /api/recall — src/handlers/recall.rs `recall`, request shape src/handlers/types.rs `RecallRequest`. */
	recall(params: RecallParams): Promise<RecallResponse> {
		return this.request<RecallResponse>("POST", "/api/recall", {
			user_id: params.userId,
			query: params.query,
			limit: params.limit ?? 5,
			mode: params.mode ?? "hybrid",
			debug: params.debug ?? false,
		});
	}

	/** POST /api/remember — src/handlers/remember.rs `remember` / `RememberRequest`. */
	remember(params: RememberParams): Promise<RememberResponse> {
		return this.request<RememberResponse>("POST", "/api/remember", {
			user_id: params.userId,
			content: params.content,
			memory_type: params.memoryType,
			tags: params.tags ?? [],
			importance: params.importance,
		});
	}

	/**
	 * POST /api/proactive_context — src/handlers/recall.rs `proactive_context`.
	 *
	 * This is the ONLY handler that writes feedback momentum (the
	 * `feedback_multiplier` in ScoreAttribution): when `previous_response` is
	 * present it consumes the pending surfaced set from the PREVIOUS call,
	 * computes implicit-feedback signals, updates momentum, and internally
	 * applies reinforce_recall + Hebbian edge strengthening for helpful and
	 * misleading classifications (recall.rs:1670-1720). It then surfaces a new
	 * set and stores it as pending for the next call.
	 */
	proactiveContext(params: ProactiveContextParams): Promise<ProactiveContextResponse> {
		return this.request<ProactiveContextResponse>("POST", "/api/proactive_context", {
			user_id: params.userId,
			context: params.context,
			max_results: params.maxResults ?? 5,
			semantic_threshold: params.semanticThreshold ?? 0.6,
			auto_ingest: params.autoIngest,
			previous_response: params.previousResponse,
			user_followup: params.userFollowup,
			...(params.toolActions && params.toolActions.length > 0 ? { tool_actions: params.toolActions } : {}),
		});
	}

	/** POST /api/reinforce — src/handlers/recall.rs `reinforce_feedback` / `ReinforceFeedbackRequest`. */
	reinforce(userId: string, ids: string[], outcome: ReinforceOutcome): Promise<ReinforceStats> {
		return this.request<ReinforceStats>("POST", "/api/reinforce", {
			user_id: userId,
			ids,
			outcome,
		});
	}

	/** DELETE /api/memory/{id}?user_id=… — src/handlers/crud.rs `delete_memory`. */
	deleteMemory(userId: string, memoryId: string): Promise<unknown> {
		return this.request<unknown>(
			"DELETE",
			`/api/memory/${encodeURIComponent(memoryId)}?user_id=${encodeURIComponent(userId)}`,
		);
	}

	/** GET /health — src/handlers/health.rs `health`. */
	health(): Promise<{ status: string; version: string }> {
		return this.request<{ status: string; version: string }>("GET", "/health");
	}
}
