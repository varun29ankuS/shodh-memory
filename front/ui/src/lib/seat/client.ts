/**
 * The seat harness client. Every call goes to a same-origin `/seat/*` path:
 * the shodh-front Rust binary (and the vite dev proxy) strips the prefix and
 * injects the seat bearer token, so the browser holds no credential for the
 * seat, exactly as it holds no X-API-Key for the backend.
 *
 * Endpoints mirror seat/src/server.ts's route table; shapes are in types.ts.
 */

import { api, ApiError, NetworkError } from "@/lib/api/client";
import type {
  ConversationDetail,
  ConversationSummary,
  LedgerEntryView,
  McpServerInfo,
  ModelRef,
  OAuthFlowEvent,
  ProviderInfo,
  SeatEvent,
  SeatHealthResponse,
  SeatModelInfo,
  SeatReachability,
} from "./types";

export async function probeSeat(signal?: AbortSignal): Promise<SeatReachability> {
  // /healthz answers 200 with the backend up and 503 with it down — the seat
  // itself is reachable in both cases, so a 503 is parsed, not treated as
  // "seat offline".
  try {
    const health = await api.get<SeatHealthResponse>("/seat/healthz", signal);
    return { state: "online", backendOk: health.backend.ok, backendDetail: health.backend.detail };
  } catch (err) {
    if (err instanceof ApiError) {
      if (err.status === 503) {
        try {
          const health = JSON.parse(err.body) as SeatHealthResponse;
          return {
            state: "online",
            backendOk: health.backend.ok,
            backendDetail: health.backend.detail,
          };
        } catch {
          return { state: "offline", detail: `seat answered 503: ${err.body.slice(0, 120)}` };
        }
      }
      // 502 is the front proxy saying it could not reach the seat process.
      return { state: "offline", detail: `seat answered ${err.status}` };
    }
    if (err instanceof NetworkError) return { state: "offline", detail: err.message };
    throw err;
  }
}

export function listModels(refresh: boolean, signal?: AbortSignal) {
  return api.get<{ models: SeatModelInfo[]; local_errors: Record<string, string> }>(
    `/seat/v1/models${refresh ? "?refresh=1" : ""}`,
    signal,
  );
}

export function listProviders(signal?: AbortSignal) {
  return api.get<{ providers: ProviderInfo[] }>("/seat/v1/providers", signal);
}

export async function setProviderKey(providerId: string, apiKey: string): Promise<ProviderInfo> {
  const res = await fetch(`/seat/v1/providers/${encodeURIComponent(providerId)}/key`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return ((await res.json()) as { provider: ProviderInfo }).provider;
}

export async function clearProviderKey(providerId: string): Promise<ProviderInfo> {
  const res = await fetch(`/seat/v1/providers/${encodeURIComponent(providerId)}/key`, {
    method: "DELETE",
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return ((await res.json()) as { provider: ProviderInfo }).provider;
}

/** Every configured MCP server, connected or not — the failures are the point
 *  as much as the successes, so the seat never filters this list. */
export function listMcpServers(signal?: AbortSignal) {
  return api.get<{ servers: McpServerInfo[] }>("/seat/v1/mcp/servers", signal);
}

/** Re-run one server's connection. The seat runs no retry loop (seat/src/mcp.ts
 *  explains why), so this is the remedy for a server that dropped or whose
 *  configuration was just fixed. Resolves with its post-attempt state — which
 *  may still be a failure, carrying the new reason. */
export async function reconnectMcpServer(name: string): Promise<McpServerInfo> {
  const res = await fetch(`/seat/v1/mcp/servers/${encodeURIComponent(name)}/reconnect`, {
    method: "POST",
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return ((await res.json()) as { server: McpServerInfo }).server;
}

export function listConversations(userId: string, signal?: AbortSignal) {
  return api.get<{ conversations: ConversationSummary[] }>(
    `/seat/v1/conversations?user_id=${encodeURIComponent(userId)}`,
    signal,
  );
}

export function getConversation(id: string, signal?: AbortSignal) {
  return api.get<ConversationDetail>(`/seat/v1/conversations/${encodeURIComponent(id)}`, signal);
}

export function createConversation(body: {
  user_id: string;
  provider: string;
  model: string;
  system_prompt?: string;
}) {
  return api.post<ConversationSummary & { harness_user_id: string }>("/seat/v1/conversations", body);
}

export async function renameConversation(id: string, title: string): Promise<void> {
  const res = await fetch(`/seat/v1/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`/seat/v1/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
}

export async function changeModel(id: string, provider: string, model: string): Promise<ModelRef> {
  const res = await fetch(`/seat/v1/conversations/${encodeURIComponent(id)}/model`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, model }),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return ((await res.json()) as { model: ModelRef }).model;
}

export function listLedger(conversationId: string | undefined, limit: number, signal?: AbortSignal) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (conversationId) params.set("conversation_id", conversationId);
  return api.get<{ events: LedgerEntryView[] }>(`/seat/v1/learning/events?${params}`, signal);
}

export function revertLedgerEvent(eventId: string) {
  return api.post<{ revert: { id: string } }>("/seat/v1/learning/revert", { event_id: eventId });
}

/**
 * The audit trail as the seat serves it: the raw JSONL/CSV body of
 * `GET /v1/audit/export`.
 *
 * NOT `api.get`, and this is the whole reason it is written out. That helper
 * ends in `res.json()`, which throws on the second line of an NDJSON body —
 * the response here is a FILE, not a JSON document, and the point of reading it
 * as text is that the History screen renders exactly the bytes the download
 * hands a reviewer (features/history/derive.ts states why at length).
 *
 * Error handling mirrors `client.ts`'s `request` so this call is
 * indistinguishable from every other one at the call site: a 4xx/5xx is an
 * `ApiError` carrying the seat's own message, an unreachable seat is a
 * `NetworkError`, and an abort is rethrown untouched so react-query reads it as
 * a cancellation.
 */
export async function fetchAuditTrail(path: string, signal?: AbortSignal): Promise<string> {
  let res: Response;
  try {
    res = await fetch(path, { method: "GET", signal });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new NetworkError(cause);
  }
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return res.text();
}

/**
 * The filename the seat put on the response, or null when it did not reach us.
 *
 * IT USUALLY DOES NOT. The seat sends
 * `Content-Disposition: attachment; filename="shodh-audit-….jsonl"`, but the
 * shodh-front proxy forwards exactly two response headers — `content-type` and
 * `cache-control` (front/src/main.rs `forward`) — so in the shipped binary this
 * header is dropped. It survives the vite dev proxy, which forwards everything.
 * Preferring it where it exists keeps the saved name the SERVER'S name in the
 * one mode where the server can state it.
 *
 * Any value carrying a path separator is refused rather than sanitised: a
 * filename this client cannot vouch for should not be handed to the browser's
 * download path at all, and the caller has a name of its own to fall back to.
 *
 * Only the plain `filename=` parameter is read. RFC 5987's `filename*=` form
 * carries a charset and percent-encoding this would have to decode correctly to
 * use safely, the seat never emits it (server.ts writes an ASCII stamp), and
 * falling back to the caller's own name is the right answer for a form we do
 * not fully parse — so it is deliberately left to return null rather than
 * half-decoded.
 *
 * Exported for its test. It is a parser over a header this client does not
 * control, which is the shape of thing that fails silently and plausibly.
 */
export function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null;
  const match = /(^|;)\s*filename\s*=\s*(?:"([^"]*)"|([^;]*))/i.exec(header);
  const raw = (match?.[2] ?? match?.[3] ?? "").trim();
  if (!raw || raw.includes("/") || raw.includes("\\")) return null;
  return raw;
}

/**
 * Download one audit export, as the file itself.
 *
 * The body is returned as an untouched `Blob`. Nothing on this side re-encodes
 * it: the export's defining property is that two exports of the same window are
 * byte-identical and therefore diffable, and a client that rebuilt the file
 * from parsed rows would be a second, drifting definition of the artefact.
 */
export async function fetchAuditFile(
  path: string,
  fallbackFilename: string,
): Promise<{ blob: Blob; filename: string }> {
  let res: Response;
  try {
    res = await fetch(path, { method: "GET" });
  } catch (cause) {
    throw new NetworkError(cause);
  }
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
  return {
    blob: await res.blob(),
    filename: filenameFromDisposition(res.headers.get("Content-Disposition")) ?? fallbackFilename,
  };
}

/**
 * SSE over fetch, by hand: frames separated by a blank line, `event:`/`data:`
 * fields, comments (`: ping` heartbeats) and `retry:` ignored — the exact
 * grammar seat/src/server.ts emits. ~30 lines beats a dependency, and
 * `EventSource` cannot POST anyway.
 *
 * Resolves when the stream closes. Throws NetworkError if the connection
 * fails before or during the stream; frames already delivered stand.
 */
async function streamSse(
  path: string,
  body: unknown,
  onFrame: (eventName: string, data: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new NetworkError(cause);
  }
  if (!res.ok || !res.body) {
    throw new ApiError(res.status, await res.text().catch(() => ""));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (frame: string): void => {
    let eventName = "";
    let data = "";
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trimStart();
    }
    if (data) onFrame(eventName, data);
  };

  for (;;) {
    let chunk: ReadableStreamReadResult<Uint8Array>;
    try {
      chunk = await reader.read();
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
      throw new NetworkError(cause);
    }
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    for (;;) {
      const boundary = buffer.indexOf("\n\n");
      if (boundary === -1) break;
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatch(frame);
    }
  }
  if (buffer.trim()) dispatch(buffer);
}

/** POST a message and stream the SeatEvents back (payloads carry their own
 *  `type`, so the frame name is redundant here). */
export function streamMessage(
  conversationId: string,
  text: string,
  onEvent: (event: SeatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return streamSse(
    `/seat/v1/conversations/${encodeURIComponent(conversationId)}/messages`,
    { text },
    (_eventName, data) => onEvent(JSON.parse(data) as SeatEvent),
    signal,
  );
}

/**
 * Run a provider's browser-OAuth login through the seat bridge
 * (seat/src/server.ts handleOAuthStart) and surface each interaction.
 * The credential never reaches this client — only status frames do.
 */
export function startOAuthLogin(
  providerId: string,
  onEvent: (event: OAuthFlowEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return streamSse(
    `/seat/v1/providers/${encodeURIComponent(providerId)}/oauth/start`,
    {},
    (eventName, data) => {
      const payload = JSON.parse(data) as Record<string, unknown>;
      switch (eventName) {
        case "oauth_notify":
          onEvent({ kind: "notify", event: payload as never });
          break;
        case "oauth_prompt":
          onEvent({ kind: "prompt", ...(payload as object) } as OAuthFlowEvent);
          break;
        case "oauth_prompt_cancelled":
          onEvent({ kind: "prompt_cancelled", prompt_id: String(payload.prompt_id) });
          break;
        case "oauth_complete":
          onEvent({ kind: "complete", provider: (payload.provider ?? null) as never });
          break;
        case "oauth_error":
          onEvent({ kind: "error", message: String(payload.message ?? "Login failed") });
          break;
        default:
          break;
      }
    },
    signal,
  );
}

export async function sendOAuthInput(providerId: string, promptId: string, value: string): Promise<void> {
  const res = await fetch(`/seat/v1/providers/${encodeURIComponent(providerId)}/oauth/input`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt_id: promptId, value }),
  });
  if (!res.ok) throw new ApiError(res.status, await res.text().catch(() => ""));
}
