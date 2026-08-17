/**
 * Seat HTTP server (node:http, no framework).
 *
 * Endpoints:
 *   GET    /healthz
 *   GET    /v1/models[?refresh=1]
 *   GET    /v1/providers                          provider auth status (no secrets)
 *   PUT    /v1/providers/{id}/key                 { api_key } — stored server-side
 *   DELETE /v1/providers/{id}/key                 remove stored key (env auth remains)
 *   GET    /v1/mcp/servers                        MCP servers: status, transport, tool lists
 *   POST   /v1/mcp/servers/{name}/reconnect       re-run one server's connection
 *   GET    /v1/conversations[?user_id]            persisted session list
 *   POST   /v1/conversations                      { user_id, provider, model, system_prompt? }
 *   GET    /v1/conversations/{id}                 state + transcript + durable events
 *   PATCH  /v1/conversations/{id}                 { title } — rename
 *   DELETE /v1/conversations/{id}
 *   POST   /v1/conversations/{id}/messages        { text }  → SSE stream of SeatEvents
 *   POST   /v1/conversations/{id}/view-report     browser → seat: view verdicts + state
 *   PATCH  /v1/conversations/{id}/model           { provider, model }
 *   GET    /v1/learning/events[?limit&conversation_id]
 *   POST   /v1/learning/revert                    { event_id }
 *
 * Conversations are durable: metadata, transcript snapshots and every
 * non-delta SeatEvent are persisted per turn (store.ts), and a conversation
 * that is not live in memory is rehydrated from the store on its next
 * message. Live `Conversation` objects are a cache over that store.
 *
 * Auth: optional bearer token (mandatory for non-loopback binds, enforced at
 * config load). Provider credentials never appear in any response or event.
 */

import * as crypto from "node:crypto";
import * as http from "node:http";
import type { AuthInteraction, AuthPrompt } from "@earendil-works/pi-ai";
import type { ShodhBackend, HealthDetail } from "./backend.js";
// Value import, not type-only: handleHealth narrows with `instanceof`, which
// needs the runtime binding.
import { ShodhBackendError, healthDetailForHttp } from "./backend.js";
import type { SeatConfig } from "./config.js";
import {
	Conversation,
	ConversationBusyError,
	type ConversationDeps,
	type MemoryMechanisms,
	UnknownModelError,
} from "./conversation.js";
import {
	AUDIT_EVENT_TYPES,
	buildAuditRows,
	pairToolCalls,
	toCsv,
	toJsonl,
} from "./audit.js";
import type { SeatEvent } from "./events.js";
import { LedgerError, type LearningLedger } from "./ledger.js";
import { type McpHost, UnknownMcpServerError } from "./mcp.js";
import {
	type ModelRegistry,
	ProviderKeyUnsupportedError,
	UnknownProviderError,
} from "./models-registry.js";
import {
	deriveTitle,
	EMPTY_USAGE_TOTALS,
	isDurableEvent,
	type SeatStore,
	type StoredConversation,
	type StoredEvent,
	type UsageTotals,
} from "./store.js";
import { parseViewReport, type ViewLink } from "./view-link.js";

const MAX_BODY_BYTES = 1_048_576;
const SSE_HEARTBEAT_MS = 15_000;
/**
 * Event ceiling for one audit export. The whole body is built in memory before
 * the first byte is written (Content-Length is required for the browser to show
 * a download with a size), so this bounds peak memory. A window that hits the
 * ceiling is narrowed with `since`/`until`, and the `X-Audit-Rows` header lets
 * a caller see that it did.
 */
const MAX_AUDIT_EXPORT_EVENTS = 50_000;
/**
 * How far back a view report may reach to prove its ask was real, when the
 * in-memory registry has forgotten it (a restart). Generous, because the only
 * cost is one indexed read per unrecognised outcome, and the failure it prevents
 * is refusing to record a person accepting an offer.
 */
const MAX_VIEW_ASK_LOOKBACK = 2_000;

interface CreateConversationBody {
	user_id?: string;
	provider?: string;
	model?: string;
	system_prompt?: string;
	harness_learning?: boolean;
	/** Per-mechanism overrides (A/B evaluation arms); absent fields keep their ON defaults. */
	memory_mechanisms?: {
		guidance?: boolean;
		proactive_framing?: boolean;
		proactive_max?: number;
		recall_lineage?: boolean;
		verify_loop?: boolean;
		mcp_memory_tool_filter?: boolean;
	};
}

/** Map the wire's snake_case mechanism overrides onto MemoryMechanisms fields. */
function parseMechanisms(body: CreateConversationBody): Partial<MemoryMechanisms> | undefined {
	const wire = body.memory_mechanisms;
	if (!wire || typeof wire !== "object") return undefined;
	const overrides: Partial<MemoryMechanisms> = {};
	if (typeof wire.guidance === "boolean") overrides.guidance = wire.guidance;
	if (typeof wire.proactive_framing === "boolean") overrides.proactiveFraming = wire.proactive_framing;
	if (typeof wire.proactive_max === "number" && Number.isInteger(wire.proactive_max) && wire.proactive_max >= 1 && wire.proactive_max <= 10) {
		overrides.proactiveMax = wire.proactive_max;
	}
	if (typeof wire.recall_lineage === "boolean") overrides.recallLineage = wire.recall_lineage;
	if (typeof wire.verify_loop === "boolean") overrides.verifyLoop = wire.verify_loop;
	if (typeof wire.mcp_memory_tool_filter === "boolean") overrides.mcpMemoryToolFilter = wire.mcp_memory_tool_filter;
	return Object.keys(overrides).length > 0 ? overrides : undefined;
}

class HttpError extends Error {
	readonly status: number;

	constructor(status: number, message: string) {
		super(message);
		this.status = status;
	}
}

function readBody(request: http.IncomingMessage): Promise<string> {
	return new Promise((resolve, reject) => {
		let size = 0;
		const chunks: Buffer[] = [];
		request.on("data", (chunk: Buffer) => {
			size += chunk.length;
			if (size > MAX_BODY_BYTES) {
				reject(new HttpError(413, "Request body too large"));
				request.destroy();
				return;
			}
			chunks.push(chunk);
		});
		request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
		request.on("error", reject);
	});
}

function parseJson<T>(raw: string): T {
	if (!raw.trim()) throw new HttpError(400, "Empty request body");
	try {
		return JSON.parse(raw) as T;
	} catch {
		throw new HttpError(400, "Invalid JSON body");
	}
}

function sendJson(response: http.ServerResponse, status: number, payload: unknown): void {
	const body = JSON.stringify(payload);
	response.writeHead(status, {
		"Content-Type": "application/json; charset=utf-8",
		"Content-Length": Buffer.byteLength(body),
	});
	response.end(body);
}

export interface SeatServerDeps {
	config: SeatConfig;
	backend: ShodhBackend;
	registry: ModelRegistry;
	ledger: LearningLedger;
	mcpHost: McpHost;
	store: SeatStore;
	/** Shared with every conversation's view tools: the tools ask, this route
	 *  answers, and one instance is what makes those the same exchange. */
	viewLink: ViewLink;
}

/** Wire shape of one conversation in list/detail responses. */
function conversationSummary(stored: StoredConversation, live: Conversation | undefined): object {
	return {
		conversation_id: stored.conversation_id,
		user_id: stored.user_id,
		title: stored.title,
		model: live
			? live.model
			: { provider: stored.provider, id: stored.model_id, name: stored.model_name },
		created_at: stored.created_at,
		updated_at: stored.updated_at,
		turns: stored.turns,
		usage: stored.usage,
		busy: live?.isStreaming ?? false,
	};
}

/** Last assistant text in a persisted transcript, for re-arming the momentum
 *  leg after rehydration. Shape per pi's `AssistantMessage` (content blocks). */
function lastAssistantText(messages: unknown[]): string | undefined {
	for (let index = messages.length - 1; index >= 0; index -= 1) {
		const message = messages[index] as { role?: string; content?: unknown };
		if (message?.role !== "assistant" || !Array.isArray(message.content)) continue;
		const text = message.content
			.filter(
				(block): block is { type: "text"; text: string } =>
					typeof block === "object" &&
					block !== null &&
					(block as { type?: string }).type === "text" &&
					typeof (block as { text?: unknown }).text === "string",
			)
			.map((block) => block.text)
			.join("");
		if (text) return text;
	}
	return undefined;
}

export class SeatServer {
	private readonly deps: SeatServerDeps;
	private readonly conversations = new Map<string, Conversation>();
	/** In-flight browser-OAuth logins, one per provider. */
	private readonly oauthSessions = new Map<
		string,
		{
			controller: AbortController;
			prompts: Map<string, { resolve: (value: string) => void; reject: (reason: Error) => void }>;
		}
	>();
	private readonly server: http.Server;

	constructor(deps: SeatServerDeps) {
		this.deps = deps;
		this.server = http.createServer((request, response) => {
			this.route(request, response).catch((error) => {
				// Only HttpError messages are written for clients. Anything else is an
				// internal failure whose text (file paths, backend URLs) stays in the
				// server log — same rule the health route documents for its detail field.
				const isHttpError = error instanceof HttpError;
				const status = isHttpError ? error.status : 500;
				const message = isHttpError ? error.message : "Internal server error";
				if (!isHttpError) {
					console.error("[seat] unhandled route error:", error);
				}
				if (!response.headersSent) {
					sendJson(response, status, { error: message });
				} else {
					response.end();
				}
			});
		});
	}

	listen(): Promise<void> {
		return new Promise((resolve, reject) => {
			this.server.once("error", reject);
			this.server.listen(this.deps.config.port, this.deps.config.host, () => resolve());
		});
	}

	close(): Promise<void> {
		return new Promise((resolve) => {
			for (const conversation of this.conversations.values()) conversation.abort();
			this.server.close(() => {
				this.deps.store.close();
				resolve();
			});
		});
	}

	private authorize(request: http.IncomingMessage): void {
		const token = this.deps.config.authToken;
		if (!token) return;
		const header = request.headers.authorization ?? "";
		if (header !== `Bearer ${token}`) throw new HttpError(401, "Unauthorized");
	}

	/** Stored metadata, or 404. The store is the source of truth for existence. */
	private storedConversation(id: string): StoredConversation {
		const stored = this.deps.store.getConversation(id);
		if (!stored) throw new HttpError(404, `Unknown conversation: ${id}`);
		return stored;
	}

	/**
	 * The live agent for a conversation, rehydrating from the store when this
	 * process has not touched it yet. Rehydration needs the stored model to
	 * still resolve; when it does not (its provider's key was removed, a local
	 * endpoint is gone), the conversation stays readable via GET and the caller
	 * is told to switch models — a 409 with the remedy, not a dead session.
	 */
	private liveConversation(id: string): Conversation {
		const live = this.conversations.get(id);
		if (live) return live;

		const stored = this.storedConversation(id);
		const model = this.deps.registry.resolve(stored.provider, stored.model_id);
		if (!model) {
			throw new HttpError(
				409,
				`Model ${stored.provider}/${stored.model_id} is not available right now — ` +
					`switch this conversation's model (PATCH /v1/conversations/${id}/model) and retry`,
			);
		}
		const messages = this.deps.store.loadTranscript(id) ?? [];
		const conversation = new Conversation(this.conversationDeps(), {
			userId: stored.user_id,
			model,
			systemPrompt: stored.system_prompt ?? undefined,
			restore: {
				id: stored.conversation_id,
				createdAt: new Date(stored.created_at),
				turn: stored.turns,
				messages,
				lastAssistantText: lastAssistantText(messages),
			},
		});
		this.conversations.set(id, conversation);
		return conversation;
	}

	private conversationDeps(): ConversationDeps {
		return {
			backend: this.deps.backend,
			registry: this.deps.registry,
			ledger: this.deps.ledger,
			// Passed as a getter, not a snapshot: see ConversationDeps.mcpTools.
			mcpTools: () => this.deps.mcpHost.getTools(),
			viewLink: this.deps.viewLink,
		};
	}

	private async route(request: http.IncomingMessage, response: http.ServerResponse): Promise<void> {
		const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
		const method = request.method ?? "GET";
		const segments = url.pathname.split("/").filter(Boolean);

		if (method === "GET" && url.pathname === "/healthz") {
			await this.handleHealth(response);
			return;
		}

		this.authorize(request);

		if (method === "GET" && url.pathname === "/v1/models") {
			await this.handleModels(url, response);
			return;
		}
		if (method === "GET" && url.pathname === "/v1/providers") {
			sendJson(response, 200, { providers: await this.deps.registry.listProviders() });
			return;
		}
		if (segments[0] === "v1" && segments[1] === "providers" && segments[2] && segments[3] === "key" && segments.length === 4) {
			if (method === "PUT") {
				await this.handleProviderKeySet(decodeURIComponent(segments[2]), request, response);
				return;
			}
			if (method === "DELETE") {
				await this.handleProviderKeyClear(decodeURIComponent(segments[2]), response);
				return;
			}
		}
		if (segments[0] === "v1" && segments[1] === "providers" && segments[2] && segments[3] === "oauth") {
			const providerId = decodeURIComponent(segments[2]);
			if (method === "POST" && segments[4] === "start" && segments.length === 5) {
				await this.handleOAuthStart(providerId, request, response);
				return;
			}
			if (method === "POST" && segments[4] === "input" && segments.length === 5) {
				await this.handleOAuthInput(providerId, request, response);
				return;
			}
		}
		if (method === "GET" && url.pathname === "/v1/mcp/servers") {
			sendJson(response, 200, { servers: this.deps.mcpHost.listServers() });
			return;
		}
		if (
			segments[0] === "v1" &&
			segments[1] === "mcp" &&
			segments[2] === "servers" &&
			segments[3] &&
			segments[4] === "reconnect" &&
			segments.length === 5 &&
			method === "POST"
		) {
			await this.handleMcpReconnect(decodeURIComponent(segments[3]), response);
			return;
		}
		if (method === "GET" && url.pathname === "/v1/conversations") {
			const userId = url.searchParams.get("user_id") ?? undefined;
			const conversations = this.deps.store
				.listConversations(userId)
				.map((stored) => conversationSummary(stored, this.conversations.get(stored.conversation_id)));
			sendJson(response, 200, { conversations });
			return;
		}
		if (method === "POST" && url.pathname === "/v1/conversations") {
			await this.handleCreateConversation(request, response);
			return;
		}
		if (segments[0] === "v1" && segments[1] === "conversations" && segments[2]) {
			const conversationId = segments[2];
			if (method === "GET" && segments.length === 3) {
				const stored = this.storedConversation(conversationId);
				const live = this.conversations.get(conversationId);
				sendJson(response, 200, {
					...conversationSummary(stored, live),
					messages: live ? live.transcript() : (this.deps.store.loadTranscript(conversationId) ?? []),
					events: this.deps.store.listEvents(conversationId),
				});
				return;
			}
			if (method === "PATCH" && segments.length === 3) {
				await this.handleRename(conversationId, request, response);
				return;
			}
			if (method === "DELETE" && segments.length === 3) {
				this.storedConversation(conversationId);
				const live = this.conversations.get(conversationId);
				if (live?.isStreaming) throw new HttpError(409, "Conversation is busy — abort or wait, then delete");
				this.conversations.delete(conversationId);
				// The pending asks and the last known view belong to a conversation
				// that no longer exists; a verdict for one of them could only be
				// recorded against a row the cascade just removed.
				this.deps.viewLink.forget(conversationId);
				this.deps.store.deleteConversation(conversationId);
				sendJson(response, 200, { deleted: true });
				return;
			}
			if (method === "POST" && segments[3] === "messages" && segments.length === 4) {
				await this.handleMessage(this.liveConversation(conversationId), request, response);
				return;
			}
			if (method === "PATCH" && segments[3] === "model" && segments.length === 4) {
				await this.handleModelChange(conversationId, request, response);
				return;
			}
			if (method === "POST" && segments[3] === "view-report" && segments.length === 4) {
				await this.handleViewReport(conversationId, request, response);
				return;
			}
		}
		if (method === "GET" && url.pathname === "/v1/learning/events") {
			const limitParam = url.searchParams.get("limit");
			const limit = limitParam ? Number.parseInt(limitParam, 10) : 100;
			if (!Number.isFinite(limit) || limit <= 0 || limit > 1000) {
				throw new HttpError(400, "limit must be an integer in [1, 1000]");
			}
			const events = await this.deps.ledger.list({
				limit,
				conversationId: url.searchParams.get("conversation_id") ?? undefined,
			});
			sendJson(response, 200, { events });
			return;
		}
		if (method === "POST" && url.pathname === "/v1/learning/revert") {
			await this.handleRevert(request, response);
			return;
		}
		if (method === "GET" && url.pathname === "/v1/audit/tool-calls") {
			this.handleToolCalls(url, response);
			return;
		}
		if (method === "GET" && url.pathname === "/v1/audit/export") {
			await this.handleAuditExport(url, response);
			return;
		}

		throw new HttpError(404, `No route: ${method} ${url.pathname}`);
	}

	private async handleHealth(response: http.ServerResponse): Promise<void> {
		let backend: { ok: boolean; detail: HealthDetail | "healthy" | "ok" | "unexpected-status" };
		try {
			const health = await this.deps.backend.health();
			const known = health.status === "ok" || health.status === "healthy";
			// Normalised rather than echoed: `detail` is a closed vocabulary on the
			// failure path, and a field that is an enum in one branch and free text
			// in the other is a contract nobody can rely on.
			backend = { ok: known, detail: known ? (health.status as "ok" | "healthy") : "unexpected-status" };
		} catch (error) {
			// The probe's message quotes the backend host and port, so it stays in
			// this log and never reaches the response. What makes `kind` safe to
			// publish is that it is one of a fixed set of literals — not the trust
			// level of the caller, which on this route is none: route() answers
			// /healthz before authorize().
			console.error("[seat] backend health probe failed:", error);
			const failure = error instanceof ShodhBackendError ? error : null;
			backend = {
				ok: false,
				// An HTTP status means the backend answered — reachable, and erroring.
				// Collapsing that into "unreachable" sends operators to the network
				// when the problem is the service. Status codes carry no secrets.
				detail: failure?.kind === "http" ? healthDetailForHttp(failure.status) : (failure?.kind ?? "other"),
			};
		}
		sendJson(response, backend.ok ? 200 : 503, {
			seat: "ok",
			backend,
			conversations: this.conversations.size,
			// This route is deliberately UNAUTHENTICATED (see route()), so it
			// carries the liveness summary only. Endpoints and connection-error
			// text belong to GET /v1/mcp/servers, behind the bearer token: a
			// failure message can quote the URL it failed against, and that URL
			// is sometimes where the credential lives.
			mcp_servers: this.deps.mcpHost.healthSummary(),
		});
	}

	/**
	 * Re-run one MCP server's connection. The manual half of the MCP lifecycle
	 * — the host runs no retry loop (mcp.ts explains why), so this is how a
	 * server that died, or whose config was just fixed, comes back. Returns the
	 * server's post-attempt state, including the failure if it failed again.
	 */
	private async handleMcpReconnect(name: string, response: http.ServerResponse): Promise<void> {
		try {
			sendJson(response, 200, { server: await this.deps.mcpHost.reconnect(name) });
		} catch (error) {
			if (error instanceof UnknownMcpServerError) throw new HttpError(404, error.message);
			throw error;
		}
	}

	private async handleModels(url: URL, response: http.ServerResponse): Promise<void> {
		let localErrors: Record<string, string> = {};
		if (url.searchParams.get("refresh")) {
			localErrors = await this.deps.registry.refreshLocal();
		}
		const models = await this.deps.registry.listAvailable();
		sendJson(response, 200, { models, local_errors: localErrors });
	}

	private async handleCreateConversation(
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const body = parseJson<CreateConversationBody>(await readBody(request));
		if (!body.user_id || typeof body.user_id !== "string") throw new HttpError(400, "user_id is required");
		if (!body.provider || typeof body.provider !== "string") throw new HttpError(400, "provider is required");
		if (!body.model || typeof body.model !== "string") throw new HttpError(400, "model is required");

		const model = this.deps.registry.resolve(body.provider, body.model);
		if (!model) throw new HttpError(400, `Unknown model: ${body.provider}/${body.model}`);

		let conversation: Conversation;
		try {
			conversation = new Conversation(this.conversationDeps(), {
				userId: body.user_id,
				model,
				systemPrompt: typeof body.system_prompt === "string" ? body.system_prompt : undefined,
				// Default true; `harness_learning: false` exists for A/B evaluation
				// control arms only (see ConversationOptions.harnessLearning).
				harnessLearning: body.harness_learning !== false,
				memoryMechanisms: parseMechanisms(body),
			});
		} catch (error) {
			throw new HttpError(400, error instanceof Error ? error.message : String(error));
		}
		this.conversations.set(conversation.id, conversation);
		const stored = this.deps.store.createConversation({
			conversationId: conversation.id,
			userId: conversation.userId,
			provider: model.provider,
			modelId: model.id,
			modelName: model.name,
			systemPrompt: typeof body.system_prompt === "string" ? body.system_prompt : undefined,
			createdAt: conversation.createdAt,
		});
		sendJson(response, 201, {
			...conversationSummary(stored, conversation),
			harness_user_id: conversation.harnessUserId,
		});
	}

	private async handleRename(
		conversationId: string,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		this.storedConversation(conversationId);
		const body = parseJson<{ title?: string }>(await readBody(request));
		const title = typeof body.title === "string" ? body.title.trim() : "";
		if (!title) throw new HttpError(400, "title is required");
		if (title.length > 200) throw new HttpError(400, "title must be at most 200 characters");
		this.deps.store.renameConversation(conversationId, title);
		sendJson(response, 200, { conversation_id: conversationId, title });
	}

	private async handleProviderKeySet(
		providerId: string,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const body = parseJson<{ api_key?: string }>(await readBody(request));
		const apiKey = typeof body.api_key === "string" ? body.api_key.trim() : "";
		if (!apiKey) throw new HttpError(400, "api_key is required");
		try {
			sendJson(response, 200, { provider: await this.deps.registry.setApiKey(providerId, apiKey) });
		} catch (error) {
			if (error instanceof UnknownProviderError) throw new HttpError(404, error.message);
			if (error instanceof ProviderKeyUnsupportedError) throw new HttpError(400, error.message);
			throw error;
		}
	}

	private async handleProviderKeyClear(providerId: string, response: http.ServerResponse): Promise<void> {
		try {
			sendJson(response, 200, { provider: await this.deps.registry.clearCredential(providerId) });
		} catch (error) {
			if (error instanceof UnknownProviderError) throw new HttpError(404, error.message);
			throw error;
		}
	}

	/**
	 * Browser-OAuth bridge. pi's login flows are interaction-driven
	 * (`AuthInteraction.notify` for URLs/device codes/progress,
	 * `AuthInteraction.prompt` for pasted codes and selections —
	 * packages/ai/src/auth/types.ts); this streams those interactions to the
	 * client as SSE frames and feeds prompt answers back through
	 * POST .../oauth/input. One login at a time per provider; disconnecting
	 * the stream aborts the flow. The resulting credential lands in the same
	 * seat-owned store as API keys and never reaches the client.
	 */
	private async handleOAuthStart(
		providerId: string,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const provider = this.deps.registry.models.getProvider(providerId);
		if (!provider) throw new HttpError(404, `Unknown provider: ${providerId}`);
		if (!provider.auth.oauth) throw new HttpError(400, `${provider.name} has no OAuth flow`);
		if (this.oauthSessions.has(providerId)) {
			throw new HttpError(409, `An OAuth login for ${providerId} is already in progress`);
		}

		response.writeHead(200, {
			"Content-Type": "text/event-stream; charset=utf-8",
			"Cache-Control": "no-cache, no-transform",
			Connection: "keep-alive",
			"X-Accel-Buffering": "no",
		});
		response.write("retry: 5000\n\n");
		response.on("error", () => {});
		const write = (type: string, payload: unknown): void => {
			if (response.writableEnded || response.destroyed) return;
			try {
				response.write(`event: ${type}\ndata: ${JSON.stringify(payload)}\n\n`);
			} catch {
				// Disconnect handler aborts the flow.
			}
		};

		const controller = new AbortController();
		const session = {
			controller,
			prompts: new Map<string, { resolve: (value: string) => void; reject: (reason: Error) => void }>(),
		};
		this.oauthSessions.set(providerId, session);
		request.on("close", () => {
			if (!response.writableEnded) controller.abort();
			// Evict eagerly. pi's login can take time to settle after an abort
			// (it may be blocking on the localhost callback), and a session whose
			// client is gone must not 409-block the user's next attempt while it
			// winds down.
			if (this.oauthSessions.get(providerId) === session) {
				this.oauthSessions.delete(providerId);
			}
		});
		const heartbeat = setInterval(() => {
			if (!response.writableEnded) response.write(": ping\n\n");
		}, SSE_HEARTBEAT_MS);

		const interaction: AuthInteraction = {
			signal: controller.signal,
			notify: (event) => write("oauth_notify", event),
			prompt: (prompt: AuthPrompt) => {
				const promptId = crypto.randomUUID();
				write("oauth_prompt", {
					prompt_id: promptId,
					type: prompt.type,
					message: prompt.message,
					placeholder: "placeholder" in prompt ? prompt.placeholder : undefined,
					options: prompt.type === "select" ? prompt.options : undefined,
				});
				return new Promise<string>((resolve, reject) => {
					session.prompts.set(promptId, { resolve, reject });
					// Either the whole flow or this one prompt can be cancelled
					// out from under us (pi races manual-code prompts against
					// callback servers).
					const cancel = () => {
						if (session.prompts.delete(promptId)) {
							write("oauth_prompt_cancelled", { prompt_id: promptId });
							reject(new Error("Prompt cancelled"));
						}
					};
					controller.signal.addEventListener("abort", cancel, { once: true });
					prompt.signal?.addEventListener("abort", cancel, { once: true });
				});
			},
		};

		try {
			await this.deps.registry.models.login(providerId, "oauth", interaction);
			const info = (await this.deps.registry.listProviders()).find(
				(candidate) => candidate.id === providerId,
			);
			write("oauth_complete", { provider: info ?? null });
		} catch (error) {
			write("oauth_error", {
				message: error instanceof Error ? error.message : String(error),
			});
		} finally {
			clearInterval(heartbeat);
			for (const [, pending] of session.prompts) pending.reject(new Error("Login finished"));
			session.prompts.clear();
			// Only remove the registry entry if it is still OURS. A stale flow
			// settling late must never delete the fresh session that replaced it
			// after eager eviction — that would silently orphan a live login.
			if (this.oauthSessions.get(providerId) === session) {
				this.oauthSessions.delete(providerId);
			}
			if (!response.writableEnded) response.end();
		}
	}

	private async handleOAuthInput(
		providerId: string,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const body = parseJson<{ prompt_id?: string; value?: string }>(await readBody(request));
		if (!body.prompt_id || typeof body.value !== "string") {
			throw new HttpError(400, "prompt_id and value are required");
		}
		const session = this.oauthSessions.get(providerId);
		const pending = session?.prompts.get(body.prompt_id);
		if (!session || !pending) throw new HttpError(404, "No such pending prompt");
		session.prompts.delete(body.prompt_id);
		pending.resolve(body.value);
		sendJson(response, 200, { accepted: true });
	}

	private async handleMessage(
		conversation: Conversation,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const body = parseJson<{ text?: string }>(await readBody(request));
		if (!body.text || typeof body.text !== "string" || !body.text.trim()) {
			throw new HttpError(400, "text is required");
		}
		if (conversation.isStreaming) throw new HttpError(409, "Conversation is busy");

		response.writeHead(200, {
			"Content-Type": "text/event-stream; charset=utf-8",
			"Cache-Control": "no-cache, no-transform",
			Connection: "keep-alive",
			"X-Accel-Buffering": "no",
		});
		response.write("retry: 5000\n\n");

		const heartbeat = setInterval(() => {
			if (!response.writableEnded) response.write(": ping\n\n");
		}, SSE_HEARTBEAT_MS);

		// A destroyed response must never take the agent loop down: writes are
		// guarded and stream errors are absorbed (the disconnect handler below
		// aborts the run).
		response.on("error", () => {});
		const write = (event: SeatEvent): void => {
			if (response.writableEnded || response.destroyed) return;
			try {
				response.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
			} catch {
				// Socket torn down mid-write; the close handler aborts the run.
			}
		};

		// Tee: every non-delta event is captured for the store while it streams
		// to the client, so a reopened conversation can replay its evidence
		// surface — including a turn the client disconnected from.
		const durableEvents: StoredEvent[] = [];
		const usageDelta: UsageTotals = { ...EMPTY_USAGE_TOTALS };
		let currentTurn = conversation.turnCount + 1;
		const sink = (event: SeatEvent): void => {
			if (event.type === "turn_start") currentTurn = event.turn;
			if (isDurableEvent(event)) {
				durableEvents.push({ turn: currentTurn, ts: new Date().toISOString(), event });
			}
			if (event.type === "usage") {
				usageDelta.input += event.usage.input;
				usageDelta.output += event.usage.output;
				usageDelta.cache_read += event.usage.cacheRead;
				usageDelta.cache_write += event.usage.cacheWrite;
				usageDelta.reasoning += event.usage.reasoning ?? 0;
				usageDelta.total_tokens += event.usage.totalTokens;
				usageDelta.cost_total += event.usage.cost.total;
			}
			write(event);
		};

		let clientGone = false;
		request.on("close", () => {
			if (!response.writableEnded) {
				clientGone = true;
				conversation.abort();
			}
		});

		const hadTitle = this.deps.store.getConversation(conversation.id)?.title != null;
		try {
			await conversation.sendMessage(body.text, sink);
		} catch (error) {
			if (error instanceof ConversationBusyError) {
				sink({ type: "error", message: error.message });
			} else if (!clientGone) {
				sink({ type: "error", message: error instanceof Error ? error.message : String(error) });
			}
		} finally {
			// Persist whatever actually happened — including an aborted turn.
			// A store failure must not tear down the response; it is logged and
			// the live conversation remains authoritative for this process.
			try {
				this.deps.store.persistTurn({
					conversationId: conversation.id,
					messages: conversation.transcript(),
					turns: conversation.turnCount,
					usageDelta,
					events: durableEvents,
					titleCandidate: hadTitle ? undefined : deriveTitle(body.text),
				});
			} catch (persistError) {
				console.error(
					`[seat] failed to persist turn for conversation ${conversation.id}: ` +
						`${persistError instanceof Error ? persistError.message : String(persistError)}`,
				);
			}
			clearInterval(heartbeat);
			if (!response.writableEnded) response.end();
		}
	}

	/**
	 * The browser reporting what it did with a view command, and what is on
	 * screen — the return leg of the view loop (view-link.ts).
	 *
	 * WHY A ROUTE AND NOT THE STREAM. The SSE stream is one-way, and the reports
	 * that matter most arrive when there is no stream at all: a Follow offer sits
	 * on screen and is accepted after the turn has ended. So the answer comes back
	 * the way everything else the browser sends does — an HTTP request on the
	 * conversation resource.
	 *
	 * OUTCOMES ARE PERSISTED HERE, NOT STREAMED THROUGH A TURN'S SINK. A late
	 * outcome has no turn to ride, and giving early ones a different path from
	 * late ones would leave two writers for one kind of row, which is how a trail
	 * ends up with duplicates in one window and gaps in another.
	 *
	 * AN OUTCOME FOR AN ASK THIS SEAT NEVER ISSUED IS REFUSED. The registry is
	 * checked first and the durable event store second — the seat can restart
	 * while an offer is still on screen, and the persisted `view_command` is then
	 * the only evidence the ask was real. Only when neither knows the id is the
	 * report rejected, because at that point it is a claim about an act with no
	 * record, and writing it would put a verdict in the trail for a command that
	 * appears nowhere in it.
	 */
	private async handleViewReport(
		conversationId: string,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		const stored = this.storedConversation(conversationId);
		const parsed = parseViewReport(parseJson<unknown>(await readBody(request)));
		if ("error" in parsed) throw new HttpError(400, parsed.error);
		const report = parsed.report;

		// The turn a late outcome is filed under is the one in progress when it
		// arrived — where a reader scanning the timeline would look for it. What
		// joins it to the command it answers is the tool call id in its payload,
		// never its position.
		const live = this.conversations.get(conversationId);
		const turn = Math.max(1, live?.turnCount ?? stored.turns);
		const at = report.view.destination;

		const unrecognised: string[] = [];
		let recorded = 0;
		for (const outcome of report.outcomes) {
			if (!this.deps.viewLink.knows(outcome.tool_call_id) && !this.askWasIssued(conversationId, outcome.tool_call_id)) {
				unrecognised.push(outcome.tool_call_id);
				continue;
			}
			const written = this.deps.store.appendEvent(conversationId, {
				turn,
				ts: new Date().toISOString(),
				event: {
					type: "view_outcome",
					tool_call_id: outcome.tool_call_id,
					dimension: outcome.dimension,
					state: outcome.state,
					at,
				},
			});
			if (written) recorded += 1;
		}

		if (report.outcomes.length > 0 && recorded === 0 && unrecognised.length > 0) {
			throw new HttpError(
				404,
				`No view command with ${unrecognised.length === 1 ? "id" : "ids"} ${unrecognised.join(", ")} was issued for this conversation`,
			);
		}

		// Handed over AFTER the durable write, so a tool can never be told a
		// verdict the trail does not contain.
		this.deps.viewLink.report(conversationId, report);

		// `unrecognised` is returned rather than swallowed: a partially accepted
		// report is a client bug, and a 200 that hid it would let the browser go
		// on reporting into a void.
		sendJson(response, 200, { recorded, unrecognised });
	}

	/**
	 * Whether a `view_command` with this tool call id is in the durable store for
	 * this conversation.
	 *
	 * The registry's memory dies with the process; the store's does not. This is
	 * the only thing that lets a Follow offer accepted after a seat restart be
	 * recorded rather than rejected as a fabrication.
	 */
	private askWasIssued(conversationId: string, toolCallId: string): boolean {
		return this.deps.store
			.queryEvents({ conversationId, types: ["view_command"], limit: MAX_VIEW_ASK_LOOKBACK })
			.some((row) => row.event.type === "view_command" && row.event.tool_call_id === toolCallId);
	}

	/**
	 * Model swap by id, not by live object: the whole point of PATCHing the
	 * model may be that the stored one no longer resolves, so this must work
	 * without rehydrating the conversation under its old model.
	 */
	private async handleModelChange(
		conversationId: string,
		request: http.IncomingMessage,
		response: http.ServerResponse,
	): Promise<void> {
		this.storedConversation(conversationId);
		const body = parseJson<{ provider?: string; model?: string }>(await readBody(request));
		if (!body.provider || !body.model) throw new HttpError(400, "provider and model are required");

		const live = this.conversations.get(conversationId);
		try {
			if (live) {
				const model = live.setModel(body.provider, body.model);
				this.deps.store.setModel(conversationId, model.provider, model.id, model.name);
				sendJson(response, 200, { model });
				return;
			}
			const model = this.deps.registry.resolve(body.provider, body.model);
			if (!model) throw new UnknownModelError(body.provider, body.model);
			this.deps.store.setModel(conversationId, model.provider, model.id, model.name);
			sendJson(response, 200, { model: { provider: model.provider, id: model.id, name: model.name } });
		} catch (error) {
			if (error instanceof UnknownModelError) throw new HttpError(400, error.message);
			if (error instanceof ConversationBusyError) throw new HttpError(409, error.message);
			throw error;
		}
	}

	/**
	 * Shared query parsing for the audit routes. Rejects a malformed timestamp
	 * rather than coercing it: `since=yesterday` silently becoming "no lower
	 * bound" would hand a reviewer a wider window than they asked for while
	 * looking like it worked.
	 */
	private auditWindow(url: URL): { userId?: string; conversationId?: string; since?: string; until?: string } {
		const window: { userId?: string; conversationId?: string; since?: string; until?: string } = {};
		const userId = url.searchParams.get("user_id");
		if (userId) window.userId = userId;
		const conversationId = url.searchParams.get("conversation_id");
		if (conversationId) window.conversationId = conversationId;
		for (const bound of ["since", "until"] as const) {
			const raw = url.searchParams.get(bound);
			if (!raw) continue;
			const parsed = new Date(raw);
			if (Number.isNaN(parsed.getTime())) {
				throw new HttpError(400, `${bound} must be an ISO-8601 timestamp, got "${raw}"`);
			}
			window[bound] = parsed.toISOString();
		}
		return window;
	}

	/**
	 * Tool calls across every conversation, newest window first — the read the
	 * event store could not previously serve, because `listEvents` requires a
	 * conversation id and a reviewer asking "when was this tool used" does not
	 * have one.
	 */
	private handleToolCalls(url: URL, response: http.ServerResponse): void {
		const window = this.auditWindow(url);
		const limitParam = url.searchParams.get("limit");
		const limit = limitParam ? Number.parseInt(limitParam, 10) : 500;
		if (!Number.isFinite(limit) || limit <= 0 || limit > 10000) {
			throw new HttpError(400, "limit must be an integer in [1, 10000]");
		}
		const toolName = url.searchParams.get("tool_name") ?? undefined;
		// Two event rows per call, so the row budget is doubled to keep `limit`
		// meaning "calls" rather than "half as many calls as you asked for".
		const events = this.deps.store.queryEvents({
			...window,
			types: ["tool_call_start", "tool_call_end"],
			limit: limit * 2,
		});
		const all = pairToolCalls(events);
		const calls = toolName ? all.filter((call) => call.tool_name === toolName) : all;
		sendJson(response, 200, { tool_calls: calls.slice(0, limit) });
	}

	/**
	 * The audit trail as a file: ledger + tool calls + retrievals, one sorted
	 * stream, JSONL or CSV.
	 *
	 * Served as an attachment because the artefact is the point — this is what
	 * gets handed to a reviewer, and a trail that can only be read inside the
	 * UI that produced it is not evidence.
	 */
	private async handleAuditExport(url: URL, response: http.ServerResponse): Promise<void> {
		const format = url.searchParams.get("format") ?? "jsonl";
		if (format !== "jsonl" && format !== "csv") {
			throw new HttpError(400, `format must be "jsonl" or "csv", got "${format}"`);
		}
		const window = this.auditWindow(url);
		const [entries, events] = [
			await this.deps.ledger.query(window),
			this.deps.store.queryEvents({ ...window, types: AUDIT_EVENT_TYPES, limit: MAX_AUDIT_EXPORT_EVENTS }),
		];
		const rows = buildAuditRows({ entries, events });
		const body = format === "csv" ? toCsv(rows) : toJsonl(rows);
		const stamp = new Date().toISOString().replaceAll(":", "-").slice(0, 19);
		response.writeHead(200, {
			"Content-Type": format === "csv" ? "text/csv; charset=utf-8" : "application/x-ndjson; charset=utf-8",
			"Content-Length": Buffer.byteLength(body),
			"Content-Disposition": `attachment; filename="shodh-audit-${stamp}.${format}"`,
			// The count is in a header so a caller can detect the ceiling below
			// without parsing the body.
			"X-Audit-Rows": String(rows.length),
		});
		response.end(body);
	}

	private async handleRevert(request: http.IncomingMessage, response: http.ServerResponse): Promise<void> {
		const body = parseJson<{ event_id?: string }>(await readBody(request));
		if (!body.event_id || typeof body.event_id !== "string") throw new HttpError(400, "event_id is required");
		try {
			// This route is only reachable by a human driving the seat's HTTP
			// surface (the UI's Revert control); no seat loop calls it.
			const revertEntry = await this.deps.ledger.revert(body.event_id, this.deps.backend, "user");
			sendJson(response, 200, { revert: revertEntry });
		} catch (error) {
			if (error instanceof LedgerError) throw new HttpError(409, error.message);
			throw error;
		}
	}
}
