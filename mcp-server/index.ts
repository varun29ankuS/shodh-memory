#!/usr/bin/env node
/**
 * Shodh-Memory MCP Server
 *
 * Gives Claude persistent memory across sessions.
 * Connects to shodh-memory REST API running locally.
 *
 * Features:
 * - Semantic search with vector similarity
 * - Context summary for quick session bootstrapping
 * - Graceful network failure handling with retries
 * - Memory categorization by type and importance
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  type CallToolRequest,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
  ListResourceTemplatesRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as fs from "fs";
import * as crypto from "crypto";
import { fileURLToPath } from "url";
import { nextReconnectDelay, serializeAndValidateBody, shouldWarnInsecureApiUrl } from "./security-utils";
import { describeUserId } from "./index-helpers";
import { stripSystemNoise, getContent as _getContent, getType as _getType, formatSurfacedMemories as _formatSurfacedMemories, formatToolCallContent } from "./string-utils";
import { TokenTracker } from "./token-tracking";
import { resolvePackageVersion } from "./version";
import { renderContent, MEMORY_PREVIEW_MAX } from "./memory-format";
import { ShodhIpcClient, type WindowsIpcHelper } from "./ipc-client";
import { DrainController } from "./drain";
import { renderCommandsResource } from "./commands-resource";
import {
  decorateTool,
  isReadOnlyTool,
  TOOL_OUTPUT_SCHEMAS,
  type ToolDefinition,
} from "./tool-metadata";
import {
  shodhDataRoot,
  loadOrCreatePersistedApiKey,
  publishSharedApiKey,
  apiKeyFilePath,
} from "./api-key-store";
import {
  registerShim,
  unregisterShim,
  recordSpawnedServer,
  clearSpawnedServer,
  backendPidToReap,
} from "./backend-lifecycle";

const __filename = (typeof import.meta !== "undefined" && import.meta.url) ? fileURLToPath(import.meta.url) : "";
const __dirname = __filename ? path.dirname(__filename) : process.cwd();

const SERVER_VERSION = resolvePackageVersion(__dirname);

// Configuration
// Priority: SHODH_API_URL (full URL) > SHODH_HOST+SHODH_PORT (constructed) > localhost default
function resolveApiUrl(): string {
  if (process.env.SHODH_API_URL) return process.env.SHODH_API_URL;
  const host = process.env.SHODH_HOST;
  const port = process.env.SHODH_PORT;
  if (host) {
    const scheme = port === "443" ? "https" : "http";
    const portSuffix = (port && port !== "443" && port !== "80") ? `:${port}` : "";
    return `${scheme}://${host}${portSuffix}`;
  }
  if (port) return `http://127.0.0.1:${port}`;
  return "http://127.0.0.1:3030";
}
const API_URL = resolveApiUrl();
const WS_URL = API_URL.replace(/^http/, "ws") + "/api/stream";
const IPC_ENDPOINT = process.env.SHODH_IPC_ENDPOINT?.trim() || "";
const IPC_REQUIRED = /^(1|true|yes|on)$/i.test(process.env.SHODH_IPC_REQUIRED?.trim() || "");
if (IPC_REQUIRED && !IPC_ENDPOINT) {
  throw new Error("SHODH_IPC_REQUIRED requires SHODH_IPC_ENDPOINT for the TypeScript MCP client");
}
const IPC_WEBSOCKET_STREAM_ENABLED = Boolean(IPC_ENDPOINT)
  && process.env.SHODH_STREAM !== "false"
  && process.env.SHODH_STREAM_WEBSOCKET === "true";
const USER_ID = process.env.SHODH_USER_ID || "claude-code";

// Detect whether the server is local (safe for auto-generated keys)
function isLocalServer(): boolean {
  if (IPC_ENDPOINT) return true;
  try {
    const url = new URL(API_URL);
    const host = url.hostname;
    return host === "127.0.0.1" || host === "localhost" || host === "::1" || host === "0.0.0.0";
  } catch {
    return false;
  }
}

// Sandbox mode — used by Smithery to scan tools without a running backend
const SANDBOX_MODE = process.env.SMITHERY_SANDBOX === "true";

// API Key resolution order:
//   1. SHODH_API_KEY (explicit, preferred)
//   2. SHODH_DEV_API_KEY (matches what the server accepts in dev mode)
//   3. First key from SHODH_API_KEYS (matches server production config)
//   4. Shared persisted key at <data-root>/.api-key, auto-generated on first
//      use for local servers (passed to the server as SHODH_DEV_API_KEY).
//      Persisting it means concurrent shims and Claude Code hooks all use the
//      SAME key — previously each shim generated its own in-memory key and
//      whichever shim lost the spawn race got 401s for the whole session.
//   5. Error for remote servers
let API_KEY = "";
let apiKeySource = "";
// Path of the persisted shared key file, when one is in use (for diagnostics).
let apiKeyFile: string | null = null;
if (process.env.SHODH_API_KEY) {
  API_KEY = process.env.SHODH_API_KEY;
  apiKeySource = "SHODH_API_KEY";
} else if (process.env.SHODH_DEV_API_KEY) {
  API_KEY = process.env.SHODH_DEV_API_KEY;
  apiKeySource = "SHODH_DEV_API_KEY";
} else if (process.env.SHODH_API_KEYS?.split(",")[0]?.trim()) {
  API_KEY = process.env.SHODH_API_KEYS!.split(",")[0]!.trim();
  apiKeySource = "SHODH_API_KEYS";
} else if (SANDBOX_MODE) {
  API_KEY = "sandbox";
  apiKeySource = "sandbox";
}
if (!API_KEY) {
  if (isLocalServer()) {
    // Load (or generate once and persist) the shared local key — zero config.
    try {
      const persisted = loadOrCreatePersistedApiKey(shodhDataRoot(), () =>
        crypto.randomBytes(32).toString("hex")
      );
      API_KEY = persisted.key;
      apiKeyFile = persisted.file;
      apiKeySource = persisted.created ? "auto-generated (persisted)" : "persisted key file";
      if (persisted.created) {
        console.error(`[shodh-memory] No API key set — generated one and saved it to ${persisted.file}`);
        console.error("[shodh-memory] Hooks and other local MCP clients will share this key automatically.");
      } else {
        console.error(`[shodh-memory] API key loaded from ${persisted.file}.`);
      }
    } catch (err) {
      // Persistence failed (e.g. read-only data dir) — fall back to an
      // in-memory key so this shim still works, but say so loudly because
      // hooks and sibling shims will NOT share it.
      API_KEY = crypto.randomBytes(32).toString("hex");
      apiKeySource = "auto-generated";
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[shodh-memory] WARNING: could not persist auto-generated API key (${msg}).`);
      console.error("[shodh-memory] Hooks and other MCP clients will not share this key; set SHODH_API_KEY to fix.");
    }
  } else {
    console.error("ERROR: SHODH_API_KEY is required for remote servers.");
    console.error("");
    console.error("To fix, add to your MCP config (claude_desktop_config.json or mcp.json):");
    console.error(`  "env": { "SHODH_API_KEY": "your-api-key" }`);
    console.error("");
    console.error("Or set in your shell:");
    console.error("  export SHODH_API_KEY=your-api-key");
    process.exit(1);
  }
}
// Share an environment-supplied key with the hooks. An MCP `env` block reaches
// this shim but NOT Claude Code's hooks, so without this a user who configures
// SHODH_API_KEY in mcp.json gets a working shim and hooks that 401 on every
// capture.
//
// Two conditions, both needed. isLocalServer() checks the backend URL, which
// alone is not enough: a loopback tunnel or local proxy in front of a remote
// backend still looks local. So the key's own provenance is checked too, and
// SHODH_API_KEYS is excluded — it is the production-shaped variable (the server
// reads it as its full key list), and a production key must not be written to
// disk to satisfy a local convenience.
const KEY_SOURCE_IS_LOCAL_SHAPED =
  apiKeySource === "SHODH_API_KEY" || apiKeySource === "SHODH_DEV_API_KEY";
if (API_KEY && apiKeyFile === null && KEY_SOURCE_IS_LOCAL_SHAPED && isLocalServer()) {
  try {
    const published = publishSharedApiKey(shodhDataRoot(), API_KEY);
    apiKeyFile = apiKeyFilePath(shodhDataRoot());
    if (published) {
      console.error(`[shodh-memory] Shared this key with Claude Code hooks via ${apiKeyFile}.`);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[shodh-memory] WARNING: could not share the API key with hooks (${msg}).`);
    console.error("[shodh-memory] Hook-based memory capture will fail unless SHODH_API_KEY is also set in your shell.");
  }
}

// Log which source was used (without revealing the key itself).
// Persisted/auto-generated/sandbox sources already logged their own message above.
if (apiKeySource === "SHODH_DEV_API_KEY") {
  console.error("[shodh-memory] WARNING: API key loaded from SHODH_DEV_API_KEY — this is a development key. Use SHODH_API_KEY for production.");
} else if (apiKeySource === "SHODH_API_KEY" || apiKeySource === "SHODH_API_KEYS") {
  console.error(`[shodh-memory] API key loaded from ${apiKeySource}.`);
}
const IPC_CLIENT = IPC_ENDPOINT
  ? new ShodhIpcClient(IPC_ENDPOINT, API_KEY, getWindowsIpcHelper())
  : null;
const BACKEND_LOCATION = IPC_ENDPOINT || API_URL;
const RETRY_ATTEMPTS = 3;
const RETRY_DELAY_MS = 1000;
const REQUEST_TIMEOUT_MS = 10000;
// Write operations (POST/PUT/DELETE) get a longer timeout because the server
// persists data before post-processing (graph, lineage, temporal facts).
// Issue #109: 10s was too short, causing retries that created duplicate memories.
const WRITE_TIMEOUT_MS = 30000;

// -----------------------------------------------------------------------------
// In-flight drain on stdin EOF (issue #405)
// -----------------------------------------------------------------------------
// MCP hosts (e.g. Claude Desktop) close the shim's stdin on a thread switch but
// leave stdout open. A tool call mid-flight when stdin EOFs can still be
// answered over stdout — provided the process does not exit first. The drain
// controller keeps us alive until in-flight calls settle, bounded by a grace
// window, then delivers an error for anything still stuck so the caller never
// eats the host's ~4-minute call timeout.

// Shape any tool response can take (mirrors the handler's return union).
type CallToolResult = {
  content: { type: string; text: string }[];
  isError?: boolean;
  _meta?: unknown;
};

// Grace window for draining in-flight tool calls after stdin EOF. Derived from
// the request budget so a slow-but-progressing backend call is never truncated:
// a retried idempotent GET can take up to RETRY_ATTEMPTS * REQUEST_TIMEOUT_MS of
// timeouts, and a write up to WRITE_TIMEOUT_MS — 3*10s + 30s = 60s covers a
// handler that chains both. Still far below the host's ~240s call timeout, so
// the caller always gets a response (real, or an abandon error) well in time.
const DRAIN_GRACE_MS = RETRY_ATTEMPTS * REQUEST_TIMEOUT_MS + WRITE_TIMEOUT_MS;

// Result returned to a tool call still in flight when the grace window expires.
const DRAIN_ABANDON_RESULT: CallToolResult = {
  content: [
    {
      type: "text",
      text:
        "Error: the MCP session ended (host closed stdin) before this tool call finished, " +
        "and the shim's drain grace window elapsed while the backend was still working. " +
        "The request was abandoned during shutdown; the backend may have already applied the change. " +
        "Re-check state or retry in a new session.",
    },
  ],
  isError: true,
};

// True while stdout can still carry a response back to the host.
function isStdoutWritable(): boolean {
  const out = process.stdout;
  return Boolean(out) && out.writable !== false && !out.writableEnded && !out.destroyed;
}

// Constructed here; `gracefulShutdown` is a hoisted function declaration below.
const drain = new DrainController<CallToolResult>({
  graceMs: DRAIN_GRACE_MS,
  abandonResult: DRAIN_ABANDON_RESULT,
  isOutputWritable: isStdoutWritable,
  shutdown: (reason) => gracefulShutdown(reason),
});

// Warn if non-localhost URL uses HTTP (security risk)
if ((!IPC_ENDPOINT || IPC_WEBSOCKET_STREAM_ENABLED)
    && shouldWarnInsecureApiUrl(API_URL, process.env.SHODH_ALLOW_HTTP)) {
  console.error("[shodh-memory] WARNING: Using HTTP for a non-localhost server is insecure.");
  console.error("[shodh-memory] Set SHODH_API_URL to an https:// URL, or set SHODH_ALLOW_HTTP=true to suppress this warning.");
}

// Input validation limits
const MAX_CONTENT_LENGTH = 100_000; // 100KB max for content fields
const MAX_QUERY_LENGTH = 10_000;    // 10KB max for search queries
const MAX_LIMIT = 250;              // Max results per query

// =============================================================================
// TOKEN TRACKING - Context window awareness (SHO-115)
// =============================================================================

// Parse a numeric env var with validation. A malformed (NaN) or out-of-range
// value would otherwise silently break token tracking — e.g. a NaN budget makes
// `tokens / budget` NaN so alerts never fire; a 0 budget makes it Infinity so
// they always fire. Reject anything outside [min, max] and use the default.
function parseEnvNumber(
  raw: string | undefined,
  fallback: number,
  min: number,
  max: number,
): number {
  if (raw === undefined || raw.trim() === "") return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < min || n > max) {
    console.error(
      `[shodh-memory] Invalid numeric env value "${raw}" (expected ${min}..${max}) — using default ${fallback}.`,
    );
    return fallback;
  }
  return n;
}

// Token budget configuration (default 100k tokens, ~400k chars).
// Budget: a positive integer. Alert threshold: a fraction in (0, 1].
const TOKEN_BUDGET = parseEnvNumber(
  process.env.SHODH_TOKEN_BUDGET,
  100_000,
  1,
  Number.MAX_SAFE_INTEGER,
);
const ALERT_THRESHOLD = parseEnvNumber(process.env.SHODH_ALERT_THRESHOLD, 0.9, 0.01, 1);

// Content-aware token tracker (replaces naive len/4 with CJK/code/prose heuristic)
const tokenTracker = new TokenTracker(TOKEN_BUDGET, ALERT_THRESHOLD);

// Legacy aliases for compatibility with existing code that uses module globals
let sessionTokens = 0;
let sessionStartTime = Date.now();

function estimateTokens(text: string): number {
  return tokenTracker.estimateTokens(text);
}

function getTokenStatus(): { tokens: number; budget: number; percent: number; alert: string | null } {
  const percent = sessionTokens / TOKEN_BUDGET;
  return {
    tokens: sessionTokens,
    budget: TOKEN_BUDGET,
    percent: Math.round(percent * 100) / 100,
    alert: percent >= ALERT_THRESHOLD ? `context_${Math.round(ALERT_THRESHOLD * 100)}_percent` : null,
  };
}

// =============================================================================
// TOOL CALL TRACKING - For session digest
// =============================================================================

// Tracks tool invocation counts per session (reset on session reset)
const toolCallCounts: Map<string, number> = new Map();

// Reset session (call on new conversation or explicit clear)
function resetTokenSession(): void {
  sessionTokens = 0;
  sessionStartTime = Date.now();
  toolCallCounts.clear();
}

// Streaming ingestion settings
// Local IPC is request/response only. WebSocket streaming remains available as
// an explicit opt-in using SHODH_STREAM_WEBSOCKET=true and SHODH_API_URL.
let STREAM_ENABLED = IPC_ENDPOINT
  ? IPC_WEBSOCKET_STREAM_ENABLED
  : process.env.SHODH_STREAM !== "false";
const STREAM_MIN_CONTENT_LENGTH = 50; // minimum content length to stream

// Proactive surfacing settings
// When enabled, relevant memories are automatically surfaced with tool responses
const PROACTIVE_SURFACING = process.env.SHODH_PROACTIVE !== "false"; // enabled by default
const PROACTIVE_MIN_CONTEXT_LENGTH = 30; // minimum context length to trigger surfacing
const MAX_CONTEXT_LENGTH = 4000; // max chars sent to backend (MiniLM truncates at ~256 tokens anyway)

// Track last proactive_context response for implicit feedback loop.
// The backend uses this to evaluate whether surfaced memories were helpful.
// Guard against concurrent proactive_context calls corrupting feedback state.
let lastProactiveResponse: string = "";
let proactiveCallInFlight = false;

// =============================================================================
// STREAMING MEMORY INGESTION - Continuous background memory capture
// =============================================================================

let streamSocket: WebSocket | null = null;
let streamConnecting = false;
let streamReconnectTimer: ReturnType<typeof setTimeout> | null = null;
let streamReconnectDelay = 1000; // Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 60s
const STREAM_RECONNECT_MAX_DELAY = 60_000;

// Buffer for messages while reconnecting
const streamBuffer: string[] = [];
const MAX_BUFFER_SIZE = 100;
let streamHandshakeComplete = false;

// Connect to streaming endpoint
async function connectStream(): Promise<void> {
  if (!STREAM_ENABLED || streamConnecting || (streamSocket?.readyState === WebSocket.OPEN)) {
    return;
  }

  streamConnecting = true;
  streamHandshakeComplete = false;

  try {
    // Auth: Bun supports headers in WebSocket constructor, but Node.js does not.
    // Pass API key as query parameter for cross-runtime compatibility.
    // Server accepts both X-API-Key header and ?api_key= query parameter.
    const wsUrlWithAuth = WS_URL + (WS_URL.includes("?") ? "&" : "?") + "api_key=" + encodeURIComponent(API_KEY);

    // Also try passing header for Bun (ignored by Node.js WebSocket)
    streamSocket = new WebSocket(wsUrlWithAuth, {
      headers: {
        "X-API-Key": API_KEY
      }
    } as any);

    streamSocket.onopen = () => {
      streamConnecting = false;
      streamReconnectDelay = 1000; // Reset backoff on successful connection
      console.error("[Stream] WebSocket connected to", WS_URL);
      // Send handshake first - server expects StreamHandshake as first message
      const handshake = JSON.stringify({
        user_id: USER_ID,
        mode: "conversation",
        extraction_config: {
          checkpoint_interval_ms: 5000,
          max_buffer_size: 50,
          auto_dedupe: true,
          extract_entities: true,
        },
      });
      streamSocket?.send(handshake);
      console.error("[Stream] Sent handshake");
    };

    streamSocket.onmessage = (event) => {
      try {
        const response = JSON.parse(event.data as string);
        // Check for handshake ACK (server uses serde tag format: { "type": "ack", ... })
        if (response.type === "ack" && response.message_type === "handshake") {
          streamHandshakeComplete = true;
          console.error("[Stream] Handshake ACK received, streaming ready");
          // Now flush buffered messages
          const bufferedCount = streamBuffer.length;
          while (streamBuffer.length > 0) {
            const msg = streamBuffer.shift();
            if (msg && streamSocket?.readyState === WebSocket.OPEN) {
              streamSocket.send(msg);
            }
          }
          if (bufferedCount > 0) {
            console.error(`[Stream] Flushed ${bufferedCount} buffered messages`);
          }
        }
      } catch (e) {
        console.error("[Stream] Failed to parse incoming message:", e);
      }
    };

    streamSocket.onclose = (event) => {
      console.error("[Stream] WebSocket closed:", event.code, event.reason || "(no reason)");
      streamSocket = null;
      streamConnecting = false;
      streamHandshakeComplete = false;
      // Reconnect after delay with exponential backoff
      if (STREAM_ENABLED && !streamReconnectTimer) {
        const delay = streamReconnectDelay;
        streamReconnectDelay = nextReconnectDelay(streamReconnectDelay, STREAM_RECONNECT_MAX_DELAY);
        streamReconnectTimer = setTimeout(() => {
          streamReconnectTimer = null;
          console.error(`[Stream] Attempting reconnect (next delay: ${streamReconnectDelay}ms)...`);
          connectStream().catch((e) => console.error("[Stream] Reconnect failed:", e));
        }, delay);
      }
    };

    streamSocket.onerror = (error) => {
      console.error("[Stream] WebSocket error:", error);
      // Error handler - close will be called after
    };
  } catch (err) {
    console.error("[Stream] Failed to create WebSocket:", err);
    streamConnecting = false;
  }
}

// Stream a memory to the server (non-blocking)
function streamMemory(content: string, tags: string[] = [], source: string = "assistant", timestamp?: string): void {
  if (!STREAM_ENABLED || content.length < STREAM_MIN_CONTENT_LENGTH) return;

  // Server expects serde tag format: { "type": "content", ... }
  const message = JSON.stringify({
    type: "content",
    content: content.slice(0, 4000),
    source: source,
    timestamp: timestamp || new Date().toISOString(), // Use provided timestamp or current time
    tags: ["stream", ...tags],
    metadata: {},
  });

  if (streamSocket?.readyState === WebSocket.OPEN && streamHandshakeComplete) {
    streamSocket.send(message);
    console.error(`[Stream] Sent memory (${content.length} chars) with tags:`, tags);
  } else {
    // Buffer message with FIFO eviction and try to reconnect
    if (streamBuffer.length >= MAX_BUFFER_SIZE) {
      streamBuffer.shift();
      console.error(`[Stream] Buffer full, evicted oldest message (size: ${MAX_BUFFER_SIZE})`);
    }
    streamBuffer.push(message);
    console.error(`[Stream] Buffered memory (socket not ready, buffer size: ${streamBuffer.length})`);
    connectStream().catch((e) => console.error("[Stream] Reconnect failed:", e));
  }
}

// Flush buffered stream messages immediately (triggers extraction on server)
function streamFlush(): void {
  if (!STREAM_ENABLED) return;

  if (streamSocket?.readyState === WebSocket.OPEN && streamHandshakeComplete) {
    streamSocket.send(JSON.stringify({ type: "flush" }));
  }
}

// Initialize stream connection on server start
if (STREAM_ENABLED) {
  console.error("[Stream] Initializing WebSocket connection to", WS_URL);
  connectStream().catch((err) => {
    console.error("[Stream] Initial connection failed:", err);
  });
} else if (IPC_ENDPOINT) {
  console.error("[Stream] Disabled in IPC mode; set SHODH_STREAM_WEBSOCKET=true to opt in");
}

// Types matching the Rust API response structure
// Note: API returns memory_type in simplified responses, experience_type in legacy
interface Experience {
  content: string;
  memory_type?: string;
  experience_type?: string; // legacy alias
  tags?: string[];
}

interface Memory {
  id: string;
  experience?: Experience;
  content?: string; // flat format from simplified API
  memory_type?: string; // flat format from simplified API
  score?: number;
  created_at?: string;
  importance?: number;
  tier?: string;
}

interface ApiResponse<T> {
  data?: T;
  error?: string;
}

// Helper: Get content from memory (handles nested and flat structure)
function getContent(m: Memory): string {
  return m.content || m.experience?.content || '';
}

// Helper: Get memory type from memory (handles both formats)
function getType(m: Memory): string {
  return m.memory_type || m.experience?.memory_type || m.experience?.experience_type || 'Observation';
}

// =============================================================================
// STRUCTURED OUTPUT HELPERS
//
// These build the `structuredContent` payloads that accompany (never replace)
// the formatted text, matching the schemas declared in ./tool-metadata.ts.
// Undefined-valued keys are dropped so the payload carries only what the
// backend actually returned rather than a wall of nulls.
// =============================================================================

/** Drop undefined-valued keys so optional schema fields stay genuinely absent. */
function compact(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v !== undefined && v !== null) out[k] = v;
  }
  return out;
}

/**
 * Structured form of a memory, matching the MEMORY_ITEM schema fragment.
 *
 * `full` mirrors the tool's own full_content flag so the structured body is
 * exactly the body the text rendered — a consumer reading structuredContent
 * must not silently get a different amount of text than the human-readable
 * channel. `content_truncated` states which it is, so a consumer never has to
 * guess whether it holds the whole memory.
 */
function structuredMemory(m: Memory, full: boolean = false): Record<string, unknown> {
  const body = getContent(m);
  const truncated = !full && body.length > MEMORY_PREVIEW_MAX;
  return compact({
    id: m.id,
    content: truncated ? body.slice(0, MEMORY_PREVIEW_MAX) : body,
    content_truncated: truncated,
    memory_type: getType(m),
    tags: m.experience?.tags,
    score: m.score,
    importance: m.importance,
    created_at: m.created_at,
    tier: m.tier,
  });
}

/** Wire shape of a todo as returned by /api/todos/* — only the fields projected. */
interface TodoWire {
  id?: string;
  seq_num?: number;
  project_prefix?: string | null;
  project?: string | null;
  content?: string;
  status?: string;
  priority?: string;
  due_date?: string | null;
  created_at?: string;
  score?: number;
  similarity_score?: number | null;
}

/**
 * Structured form of a todo, matching the TODO_ITEM schema fragment.
 *
 * This is a PROJECTION, not a pass-through: the wire todo carries the full
 * embedding vector (hundreds of floats) and the entire comment thread, and
 * echoing those into structuredContent would dwarf the text channel it is
 * meant to complement.
 *
 * `short_id` mirrors `Todo::short_id()` in src/memory/types.rs exactly —
 * "{project_prefix|SHO}-{seq_num}" when seq_num > 0, otherwise "SHO-{first 4
 * chars of the uuid}".
 */
function structuredTodo(t: TodoWire): Record<string, unknown> {
  const shortId =
    t.seq_num && t.seq_num > 0
      ? `${t.project_prefix || "SHO"}-${t.seq_num}`
      : t.id
        ? `SHO-${t.id.slice(0, 4)}`
        : undefined;
  return compact({
    id: t.id,
    short_id: shortId,
    content: t.content,
    status: t.status,
    priority: t.priority,
    project: t.project ?? undefined,
    score: t.score ?? t.similarity_score ?? undefined,
    created_at: t.created_at,
    due_date: t.due_date ?? undefined,
  });
}

// Helper: Sleep for retry delays
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// PROACTIVE MEMORY SURFACING - Auto-surface relevant memories with responses
// =============================================================================

interface SurfacedMemory {
  content: string;
  memory_type: string;
  relevance_score: number;
}

async function backendRequest<T>(
  endpoint: string,
  method: string = "GET",
  body?: object,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<T> {
  if (IPC_CLIENT) {
    return IPC_CLIENT.request<T>(endpoint, method, body ?? null, timeoutMs);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      method,
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!response.ok) {
      const errorText = await response.text().catch(() => "Unknown error");
      throw new Error(`API error ${response.status}: ${errorText}`);
    }
    try {
      return await response.json() as T;
    } catch {
      throw new Error(`API returned invalid JSON (HTTP ${response.status})`);
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

// Surface relevant memories based on context (non-blocking, returns null on failure)
async function surfaceRelevant(context: string, maxResults: number = 3): Promise<SurfacedMemory[] | null> {
  if (!PROACTIVE_SURFACING || context.length < PROACTIVE_MIN_CONTEXT_LENGTH) {
    return null;
  }

  try {
    const result = await backendRequest<{ memories?: SurfacedMemory[] }>(
      "/api/relevant",
      "POST",
      {
        user_id: USER_ID,
        context: context.slice(0, 2000),
        config: {
          semantic_threshold: 0.65,
          max_results: maxResults,
        },
      },
      3000,
    );
    return result.memories || null;
  } catch (e) {
    console.error("[Proactive] Failed to surface memories:", e);
    return null;
  }
}

// Format surfaced memories for inclusion in tool response
function formatSurfacedMemories(memories: SurfacedMemory[]): string {
  if (!memories || memories.length === 0) return "";

  const formatted = memories
    .map((m, i) => `  ${i + 1}. [${((m.relevance_score ?? 0) * 100).toFixed(0)}%] ${renderContent(m.content, undefined, 80, false)}`)
    .join("\n");

  return `\n\n[Relevant memories surfaced]\n${formatted}`;
}

// Stream tool interactions automatically (non-blocking)
function streamToolCall(toolName: string, args: Record<string, unknown>, resultText: string): void {
  // Skip ingesting memory management tools to avoid noise
  if (["remember", "recall", "forget", "list_memories"].includes(toolName)) return;

  // Tools annotated readOnlyHint:true must not modify the store, and this path
  // fires on EVERY call (tool output is essentially always over the streaming
  // minimum), so without the gate every read tool would write a "Tool: X /
  // Result: ..." memory. That is also circular by this function's own stated
  // intent — recording what the store already knows back into the store.
  // Conversation capture stays the job of proactive_context and the hooks.
  if (isReadOnlyTool(toolName)) return;

  const argsStr = JSON.stringify(args, null, 2);
  const content = `Tool: ${toolName}\nInput: ${argsStr}\nResult: ${resultText.slice(0, 1000)}${resultText.length > 1000 ? "..." : ""}`;

  streamMemory(content, ["tool-call", toolName], "tool");
}

// =============================================================================
// LINEAGE / GRAPH / FACTS HELPERS
// =============================================================================

const FULL_UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Minimal slice of GET /api/memory/{id} (crud::MemoryWithHierarchy) used for
// id resolution and preview rendering.
interface MemoryLookup {
  id: string;
  experience?: { content?: string; experience_type?: string };
}

// Resolve a full UUID or 8+ char prefix to a concrete memory id.
//
// GET /api/memory/{id} resolves prefixes server-side (crud::resolve_memory →
// find_memory_by_prefix), but the lineage handlers do a bare uuid::parse_str
// (src/handlers/lineage.rs) and reject prefixes — so anything that feeds a
// memory id into a lineage endpoint must resolve it here first.
async function resolveMemoryId(idOrPrefix: string): Promise<string> {
  const trimmed = idOrPrefix.trim();
  if (FULL_UUID_RE.test(trimmed)) return trimmed;
  const memory = await apiCall<MemoryLookup>(
    `/api/memory/${encodeURIComponent(trimmed)}?user_id=${encodeURIComponent(USER_ID)}`,
    "GET",
  );
  return memory.id;
}

// Fetch short content previews for a bounded set of memory ids so causal-chain
// output shows what each node IS, not just a UUID. Parallel, failures
// tolerated: a deleted/unreadable memory simply renders as its id.
const PREVIEW_FETCH_CAP = 16;
const PREVIEW_CHARS = 90;

async function fetchMemoryPreviews(ids: string[]): Promise<Map<string, string>> {
  const unique = [...new Set(ids)].slice(0, PREVIEW_FETCH_CAP);
  const previews = new Map<string, string>();
  await Promise.all(
    unique.map(async (id) => {
      try {
        const m = await apiCall<MemoryLookup>(
          `/api/memory/${encodeURIComponent(id)}?user_id=${encodeURIComponent(USER_ID)}`,
          "GET",
        );
        const content = m.experience?.content || "";
        if (content) {
          previews.set(id, content.length > PREVIEW_CHARS ? content.slice(0, PREVIEW_CHARS) + "…" : content);
        }
      } catch {
        // Memory deleted since the edge was written — the id alone still identifies it.
      }
    }),
  );
  return previews;
}

// Prose rendering of a causal relation read from→to. The edge's `from` is
// always the earlier memory (cause/origin/evidence), `to` the later one — per
// the inference table in src/memory/lineage.rs infer_by_types: Error→Task =
// Caused, Task→Learning = ResolvedBy, Learning→Decision = InformedBy,
// Discovery→Task = TriggeredBy, Decision→Decision = SupersededBy. InformedBy
// therefore reads "from informed to" (evidence → the decision it informed),
// NOT the enum's to-perspective name. The one inversion is BranchedFrom:
// branch anchoring (mod.rs) writes from=pivot, to=origin, so "from branched
// from to" is literally correct there.
const CAUSAL_RELATION_PROSE: Record<string, string> = {
  Caused: "caused",
  ResolvedBy: "was resolved by",
  InformedBy: "informed",
  SupersededBy: "was superseded by",
  TriggeredBy: "triggered",
  BranchedFrom: "branched from",
  RelatedTo: "is related to",
};

// Wire shape of a lineage edge (src/memory/lineage.rs LineageEdge; MemoryId
// serializes as a bare UUID string).
interface LineageEdgeWire {
  id: string;
  from: string;
  to: string;
  relation: string;
  confidence: number;
  source: string; // "Inferred" | "Confirmed" | "Explicit"
  branch_id: string | null;
  created_at: string;
  reinforcement_count: number;
}

/**
 * Structured form of a lineage edge, matching the LINEAGE_EDGE schema fragment.
 *
 * The wire calls the identifier `id`; both the formatted text and
 * validate_causal_link's parameter call it `edge_id`, so the structured channel
 * follows the surface rather than the wire.
 */
function structuredLineageEdge(edge: LineageEdgeWire): Record<string, unknown> {
  return {
    edge_id: edge.id,
    from: edge.from,
    to: edge.to,
    relation: edge.relation,
    confidence: edge.confidence,
    source: edge.source,
    created_at: edge.created_at,
    reinforcement_count: edge.reinforcement_count,
  };
}

function formatLineageEdge(
  edge: LineageEdgeWire,
  previews: Map<string, string>,
  indent: string,
): string {
  const conf = (edge.confidence * 100).toFixed(0);
  const prose = CAUSAL_RELATION_PROSE[edge.relation] || edge.relation;
  let out = `${indent}${edge.from} ──${edge.relation}──▶ ${edge.to}\n`;
  out += `${indent}  (${conf}% confidence, ${edge.source} │ edge_id: ${edge.id})\n`;
  const fromPreview = previews.get(edge.from);
  const toPreview = previews.get(edge.to);
  // Prose reading only when both sides have content — a UUID mid-sentence
  // reads like a broken render; the arrow line above already carries the ids.
  if (fromPreview && toPreview) {
    out += `${indent}  "${fromPreview}" ${prose} "${toPreview}"\n`;
  }
  return out;
}

// RelationType (src/graph_memory.rs) serializes either as a bare string
// ("Causes", "Triggers", "CoOccurs", ...) or as { "Custom": "Enabled" } for
// schema-typed custom relations. Normalize both to a display string.
function formatRelationType(rt: unknown): string {
  if (typeof rt === "string") return rt;
  if (rt && typeof rt === "object" && "Custom" in (rt as Record<string, unknown>)) {
    return String((rt as Record<string, unknown>).Custom);
  }
  return String(rt);
}

// Robust API call with retries and timeout
async function apiCall<T>(
  endpoint: string,
  method: string = "GET",
  body?: object
): Promise<T> {
  let lastError: Error | null = null;

  // Only idempotent GET requests are retried. Retrying a write (POST/PUT/DELETE)
  // after a committed-but-unacknowledged response would create duplicate
  // memories — fail fast instead and let the caller decide whether to re-issue.
  const maxAttempts = method === "GET" ? RETRY_ATTEMPTS : 1;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const timeout = method === "GET" ? REQUEST_TIMEOUT_MS : WRITE_TIMEOUT_MS;

      if (body) {
        const bodyValidation = serializeAndValidateBody(body, MAX_CONTENT_LENGTH);
        if (!bodyValidation.ok) {
          throw new Error(bodyValidation.error);
        }
      }
      return await backendRequest<T>(endpoint, method, body, timeout);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on client errors (4xx) — parse status code explicitly
      const statusMatch = lastError.message.match(/API error (\d+)/);
      if (statusMatch && parseInt(statusMatch[1], 10) >= 400 && parseInt(statusMatch[1], 10) < 500) {
        throw lastError;
      }

      // Log retry attempt
      if (attempt < maxAttempts) {
        console.error(`Attempt ${attempt} failed: ${lastError.message}. Retrying in ${RETRY_DELAY_MS}ms...`);
        await sleep(RETRY_DELAY_MS * attempt); // Exponential backoff
      }
    }
  }

  // Provide helpful error message
  const errMsg = lastError?.message || 'Unknown error';
  if (errMsg.includes('ECONNREFUSED') || errMsg.includes('ENOENT') || errMsg.includes('fetch failed')) {
    throw new Error(
      `Cannot connect to shodh-memory server at ${BACKEND_LOCATION}. ` +
      `Start the server with: shodh-memory-server`
    );
  }
  throw new Error(`Failed after ${maxAttempts} attempt${maxAttempts === 1 ? "" : "s"}: ${errMsg}`);
}

// Check if server is available
async function isServerAvailable(): Promise<boolean> {
  try {
    await backendRequest("/health", "GET", undefined, 2000);
    return true;
  } catch {
    return false;
  }
}

// Create MCP server
const server = new Server(
  {
    name: "shodh-memory",
    version: SERVER_VERSION,
  },
  {
    capabilities: {
      tools: {},
      resources: {},
      prompts: {},
    },
  }
);

// Tool definitions. Behavioural annotations (readOnlyHint/destructiveHint/
// idempotentHint/openWorldHint, plus a display title) and output schemas are
// NOT written inline here — they live in ./tool-metadata.ts as a single source
// of truth, because the read-only set also gates ambient memory ingestion
// (see autoStreamContext / streamToolCall). decorateTool() merges them in and
// throws if a tool is missing an entry, so the two can never drift.
const TOOL_DEFINITIONS: ToolDefinition[] = [
      {
        name: "remember",
        description: "Store a memory for future recall. Use this to remember important information, decisions, user preferences, project context, or anything you want to recall later.",
        inputSchema: {
          type: "object",
          properties: {
            content: {
              type: "string",
              description: "The content to remember (observation, decision, learning, etc.)",
            },
            type: {
              type: "string",
              enum: ["Observation", "Decision", "Learning", "Error", "Discovery", "Pattern", "Context", "Task", "CodeEdit", "FileAccess", "Search", "Command", "Conversation"],
              description: "Type of memory",
              default: "Observation",
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Optional tags for categorization",
            },
            created_at: {
              type: "string",
              description: "Optional ISO 8601 timestamp for the memory (e.g., '2025-12-15T06:30:00Z'). If not provided, uses current time.",
            },
            // SHO-104: Richer context encoding
            emotional_valence: {
              type: "number",
              description: "Emotional valence: -1.0 (negative) to 1.0 (positive), 0.0 = neutral. E.g., bug found: -0.3, feature shipped: 0.7",
            },
            emotional_arousal: {
              type: "number",
              description: "Arousal level: 0.0 (calm) to 1.0 (highly aroused). E.g., routine task: 0.2, critical issue: 0.9",
            },
            emotion: {
              type: "string",
              description: "Dominant emotion label (e.g., 'joy', 'frustration', 'surprise')",
            },
            source_type: {
              type: "string",
              enum: ["user", "system", "api", "file", "web", "ai_generated", "inferred"],
              description: "Source type: where the information came from",
            },
            credibility: {
              type: "number",
              description: "Credibility score: 0.0 to 1.0 (1.0 = verified facts, 0.3 = inferred)",
            },
            episode_id: {
              type: "string",
              description: "Episode ID - groups memories into coherent episodes/conversations",
            },
            sequence_number: {
              type: "number",
              description: "Sequence number within episode (1, 2, 3...)",
            },
            preceding_memory_id: {
              type: "string",
              description: "ID of the preceding memory (for temporal chains)",
            },
            parent_id: {
              type: "string",
              description: "Parent memory ID for hierarchical organization. Creates memory trees (e.g., '71-research' -> 'algebraic' -> '21×27≡-1')",
            },
            importance: {
              type: "number",
              description: "Optional importance override (0.0-1.0). Bypasses auto-calculation. Use for memories where importance is known: Decision=0.8, Learning=0.7, Error=0.7, Discovery=0.6, Observation=0.3",
            },
            // Robotics context
            robot_id: {
              type: "string",
              description: "Robot/drone identifier for multi-robot systems",
            },
            mission_id: {
              type: "string",
              description: "Mission identifier for grouping experiences",
            },
            geo_location: {
              type: "array",
              items: { type: "number" },
              minItems: 3,
              maxItems: 3,
              description: "GPS coordinates [latitude, longitude, altitude] in WGS84",
            },
            local_position: {
              type: "array",
              items: { type: "number" },
              minItems: 3,
              maxItems: 3,
              description: "Local position [x, y, z] in meters (robot-local frame)",
            },
            heading: {
              type: "number",
              description: "Heading in degrees (0-360)",
            },
            action_type: {
              type: "string",
              description: "Action type name (e.g., 'navigate', 'grasp', 'dock')",
            },
            reward: {
              type: "number",
              description: "Reinforcement learning reward signal (-1.0 to 1.0)",
            },
            sensor_data: {
              type: "object",
              additionalProperties: { type: "number" },
              description: "Raw sensor readings (e.g., {battery: 72.5, temperature: 23.1})",
            },
            outcome_type: {
              type: "string",
              description: "Outcome type: success, failure, partial, aborted, timeout",
            },
            terrain_type: {
              type: "string",
              description: "Terrain type: indoor, outdoor, urban, rural, water, aerial",
            },
          },
          required: ["content"],
        },
      },
      {
        name: "recall",
        description: "Search memories AND todos using semantic similarity. Returns both relevant memories and matching todos. Use this to find past experiences, decisions, context, or pending work. Modes: 'semantic' (vector similarity), 'associative' (graph traversal), 'temporal' (time-based retrieval), 'hybrid' (combined), 'spatial' (geo-location based), 'mission' (mission context), 'action_outcome' (reward-based learning). Memory bodies are returned as previews (default 500 chars); when truncated the output carries an explicit marker with real lengths and a read_memory hint — a preview without a marker is complete. Pass full_content:true to get full bodies inline.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Natural language search query - searches both memories and todos",
            },
            limit: {
              type: "number",
              description: "Maximum number of memory results (default: 5). Todos limited to 5.",
              default: 5,
            },
            mode: {
              type: "string",
              enum: ["semantic", "associative", "temporal", "hybrid", "spatial", "mission", "action_outcome"],
              description: "Retrieval mode: 'semantic' for pure vector similarity, 'associative' for graph-based traversal (follows learned connections), 'temporal' for time-based retrieval, 'hybrid' for density-dependent combination (default), 'spatial' for geo-location based (REQUIRES geo_lat, geo_lon, geo_radius_meters; an index-only path that queries the geo index directly and skips semantic ranking), 'mission' for mission context (REQUIRES mission_id), 'action_outcome' for reward-based learning (uses reward_min/reward_max, defaults to positive rewards). NOTE: geo_lat/geo_lon/geo_radius_meters are NOT limited to 'spatial' mode — they compose with any mode here (hard radius filter over that mode's results, plus candidate injection so geo-relevant memories that are semantically silent for the query still surface); 'spatial' is only the dedicated index-only path.",
              default: "hybrid",
            },
            session_id: {
              type: "string",
              description: "Session ID for session-scoped retrieval. When provided, retrieves memories from that session's time window. Forces temporal mode.",
            },
            robot_id: {
              type: "string",
              description: "Filter by robot/drone identifier (for multi-robot systems)",
            },
            mission_id: {
              type: "string",
              description: "Filter by mission identifier",
            },
            geo_lat: {
              type: "number",
              description: "Geo filter: center latitude (-90 to 90). Requires geo_lon and geo_radius_meters (all three or none). Composes with ANY mode — applied as a hard radius filter over that mode's results, plus candidate injection so geo-relevant memories that never surface semantically still get considered. Only 'spatial' mode itself requires this triple; that mode is a separate index-only lookup that bypasses semantic ranking entirely.",
            },
            geo_lon: {
              type: "number",
              description: "Geo filter: center longitude (-180 to 180). Requires geo_lat and geo_radius_meters. Composes with any mode — see geo_lat.",
            },
            geo_radius_meters: {
              type: "number",
              description: "Geo filter: search radius in meters. Requires geo_lat and geo_lon. Composes with any mode — see geo_lat.",
            },
            action_type: {
              type: "string",
              description: "Filter by action type (e.g., 'navigate', 'grasp', 'dock')",
            },
            reward_min: {
              type: "number",
              description: "Filter by minimum reward value (-1.0 to 1.0)",
            },
            reward_max: {
              type: "number",
              description: "Filter by maximum reward value (-1.0 to 1.0)",
            },
            outcome_type: {
              type: "string",
              description: "Filter by outcome type: success, failure, partial, aborted, timeout",
            },
            failures_only: {
              type: "boolean",
              description: "If true, only return failure/error experiences",
            },
            terrain_type: {
              type: "string",
              description: "Filter by terrain type: indoor, outdoor, urban, rural, water, aerial",
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Filter by tags (any match)",
            },
            debug: {
              type: "boolean",
              description: "Enable retrieval diagnostics. Returns per-stage timing breakdown and per-memory score attribution showing exactly why each memory ranked where it did. Useful for debugging bad recalls.",
            },
            full_content: {
              type: "boolean",
              description: "Return complete memory bodies inline instead of previews. Increases token usage; prefer for small result sets. Default false — previews are capped and explicitly marked when truncated.",
              default: false,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "recall_by_tags",
        description: "Find memories by tags. Returns memories matching ANY of the provided tags. Useful for finding memories by category (e.g., 'tool:Edit', 'file:src/main.rs', 'source:hook', 'error', 'session-summary'). Memory bodies are returned as previews (default 500 chars); when truncated the output carries an explicit marker with real lengths and a read_memory hint — a preview without a marker is complete. Pass full_content:true to get full bodies inline.",
        inputSchema: {
          type: "object",
          properties: {
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Tags to search for (returns memories matching ANY of these tags)",
            },
            limit: {
              type: "number",
              description: "Maximum number of results (default: 50)",
              default: 50,
            },
            full_content: {
              type: "boolean",
              description: "Return complete memory bodies inline instead of previews. Increases token usage; prefer for small result sets. Default false — previews are capped and explicitly marked when truncated.",
              default: false,
            },
          },
          required: ["tags"],
        },
      },
      {
        name: "context_summary",
        description: "Get a condensed summary of recent learnings, decisions, and context. Use this at the start of a session to quickly understand what you've learned before.",
        inputSchema: {
          type: "object",
          properties: {
            include_decisions: {
              type: "boolean",
              description: "Include recent decisions (default: true)",
              default: true,
            },
            include_learnings: {
              type: "boolean",
              description: "Include recent learnings (default: true)",
              default: true,
            },
            include_context: {
              type: "boolean",
              description: "Include project context (default: true)",
              default: true,
            },
            max_items: {
              type: "number",
              description: "Maximum items per category (default: 5)",
              default: 5,
            },
          },
        },
      },
      {
        name: "list_memories",
        description: "List all stored memories",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Maximum number of results",
              default: 20,
            },
          },
        },
      },
      {
        name: "forget",
        description: "Delete a specific memory by ID",
        inputSchema: {
          type: "object",
          properties: {
            id: {
              type: "string",
              description: "The ID of the memory to delete",
            },
          },
          required: ["id"],
        },
      },
      {
        name: "memory_stats",
        description: "Get statistics about stored memories",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "verify_index",
        description: "Verify vector index integrity - diagnose orphaned memories that are stored but not searchable. Returns health status and count of orphaned memories.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "repair_index",
        description: "Repair vector index by re-indexing orphaned memories. Use this when verify_index shows unhealthy status. Returns count of repaired memories.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      // Backup & Restore tools
      {
        name: "backup_create",
        description: "Create a backup of all memories. Returns backup metadata including ID, size, and checksum. Backups are stored locally and can be restored later.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "backup_list",
        description: "List all available backups for this user. Returns backup history with IDs, timestamps, and sizes.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "backup_verify",
        description: "Verify backup integrity using SHA-256 checksum. Use to check if a backup is corrupted before restoring.",
        inputSchema: {
          type: "object",
          properties: {
            backup_id: {
              type: "number",
              description: "The backup ID to verify",
            },
          },
          required: ["backup_id"],
        },
      },
      {
        name: "backup_purge",
        description: "Purge old backups, keeping only the most recent N. Useful for managing disk space.",
        inputSchema: {
          type: "object",
          properties: {
            keep_count: {
              type: "number",
              description: "Number of backups to keep (default: 7)",
              default: 7,
            },
          },
        },
      },
      {
        name: "backup_restore",
        description: "Restore a previously created backup by ID. Destructive: replaces this user's memories, vector index and knowledge graph with the backup's contents. Todos, reminders and audit logs are NOT restored — they live in a store shared with other users, so restoring one user's copy would destroy everyone else's. The response names every store restored, and any that failed. Server restart is recommended after restore.",
        inputSchema: {
          type: "object",
          properties: {
            backup_id: {
              type: "number",
              description: "The backup ID to restore (from backup_list)",
            },
          },
          required: ["backup_id"],
        },
      },
      {
        name: "consolidation_report",
        description: "Get a report of what the memory system has been learning. Shows memory strengthening/decay events, edge formation, fact extraction, and maintenance cycles. Use this to understand how your memories are evolving.",
        inputSchema: {
          type: "object",
          properties: {
            since: {
              type: "string",
              description: "Start of report period (ISO 8601 format). Defaults to 24 hours ago.",
            },
            until: {
              type: "string",
              description: "End of report period (ISO 8601 format). Defaults to now.",
            },
          },
        },
      },
      {
        name: "proactive_context",
        description: "REQUIRED: Call this tool with EVERY user message to surface relevant memories and build conversation history. Pass the user's message as context. This enables: (1) retrieving memories relevant to what the user is asking, (2) building persistent memory of the conversation for future sessions. The system analyzes entities, semantic similarity, and recency to find contextually appropriate memories. Auto-ingest stores the context automatically. USAGE: Always call this FIRST when you receive a user message, passing their message as the context parameter. Surfaced memory bodies are previews (default 500 chars); when truncated the output carries an explicit marker with real lengths and a read_memory hint — a preview without a marker is complete. Pass full_content:true to get full bodies inline (increases token usage).",
        inputSchema: {
          type: "object",
          properties: {
            context: {
              type: "string",
              description: "The current conversation context or topic (e.g., recent messages, current task description)",
            },
            semantic_threshold: {
              type: "number",
              description: "Minimum semantic similarity (0.0-1.0) for memories to be surfaced (default: 0.65)",
              default: 0.65,
            },
            entity_match_weight: {
              type: "number",
              description: "Weight for entity matching in relevance scoring (0.0-1.0, default: 0.4)",
              default: 0.4,
            },
            recency_weight: {
              type: "number",
              description: "Weight for recency boost in relevance scoring (0.0-1.0, default: 0.2)",
              default: 0.2,
            },
            max_results: {
              type: "number",
              description: "Maximum number of memories to surface (default: 5)",
              default: 5,
            },
            memory_types: {
              type: "array",
              items: { type: "string" },
              description: "Filter to specific memory types (e.g., ['Decision', 'Learning', 'Context']). Empty means all types.",
            },
            auto_ingest: {
              type: "boolean",
              description: "Automatically store the context as a Conversation memory (default: true). Set to false to only surface memories without storing.",
              default: true,
            },
            tool_actions: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  tool_name: { type: "string", description: "Tool or actuator name (e.g., 'Edit', 'Bash', 'navigate', 'grasp')" },
                  inputs: { type: "object", additionalProperties: { type: "string" }, description: "Key-value input parameters" },
                  success: { type: "boolean", description: "Whether the action succeeded" },
                  output_snippet: { type: "string", description: "First 200 chars of output" },
                  reward: { type: "number", description: "Reward signal for robotics (-1.0 to 1.0)" },
                },
                required: ["tool_name", "success"],
              },
              description: "Tool/actuator actions performed since last proactive_context call. Used for causal feedback attribution.",
            },
            full_content: {
              type: "boolean",
              description: "Return complete memory bodies inline instead of previews. Increases token usage; prefer for small result sets. Default false — previews are capped and explicitly marked when truncated.",
              default: false,
            },
          },
          required: ["context"],
        },
      },
      {
        name: "token_status",
        description: "Get MCP pipeline token throughput for this session. Tracks tokens flowing through shodh memory tools only — NOT the AI context window. Use for internal diagnostics, not context window health.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "reset_token_session",
        description: "Reset the token counter for a new session. Call this when starting a new conversation or after context has been compressed/summarized.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "session_digest",
        description: "Get a consolidated digest of the current session: timestamps, token usage, memories created/recalled, tools used with counts, entities extracted, topic changes, and consolidation events. Use after context compression or at session milestones.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "session_history",
        description: "Show what you worked on across recent sessions. Returns session summaries with entities, memory stats, and timestamps. Use group_by_project to detect cross-session project continuity via entity overlap.",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Number of sessions to show (default: 10, max: 100)",
              default: 10,
            },
            group_by_project: {
              type: "boolean",
              description: "Detect and show project threads across sessions (default: false)",
              default: false,
            },
          },
        },
      },
      {
        name: "fact_narratives",
        description: "Get synthesized narratives from accumulated facts, clustered by topic with confidence levels and causal chains. Shows what the system has learned, organized into coherent themes.",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Maximum clusters to return (default: 20, max: 50)",
              default: 20,
            },
            entity_filter: {
              type: "string",
              description: "Filter to facts related to a specific entity/topic",
            },
          },
        },
      },
      {
        name: "purge_facts",
        description: "Delete facts matching a content pattern. Use dry_run=true to preview before deleting. Useful for cleaning up garbage facts (e.g., 'relates to' template noise).",
        inputSchema: {
          type: "object",
          properties: {
            pattern: {
              type: "string",
              description: "Substring to match in fact content (case-insensitive, min 3 chars)",
            },
            dry_run: {
              type: "boolean",
              description: "If true, count matches without deleting (default: false)",
              default: false,
            },
          },
          required: ["pattern"],
        },
      },
      // Prospective Memory / Reminders (SHO-116)
      {
        name: "set_reminder",
        description: "Set a reminder for the future. Triggers on time (at specific time or after duration) or context match (when keywords appear in conversation). Reminders will surface automatically when conditions are met.",
        inputSchema: {
          type: "object",
          properties: {
            content: {
              type: "string",
              description: "What to remember/remind about",
            },
            trigger_type: {
              type: "string",
              enum: ["time", "duration", "context"],
              description: "When to trigger: 'time' (at specific ISO timestamp), 'duration' (after N seconds), 'context' (when keywords match)",
            },
            trigger_at: {
              type: "string",
              description: "ISO 8601 timestamp for 'time' trigger (e.g., '2025-12-23T18:00:00Z')",
            },
            after_seconds: {
              type: "number",
              description: "Seconds from now for 'duration' trigger",
            },
            keywords: {
              type: "array",
              items: { type: "string" },
              description: "Keywords for 'context' trigger - reminder surfaces when any keyword appears",
            },
            priority: {
              type: "number",
              description: "Priority 1-5 (5 = highest, default: 3)",
              default: 3,
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Optional tags for categorization",
            },
            threshold: {
              type: "number",
              description: "Semantic similarity threshold for 'context' trigger (0.0-1.0, default: 0.7). Lower values match more broadly, higher values require closer semantic match.",
            },
          },
          required: ["content", "trigger_type"],
        },
      },
      {
        name: "list_reminders",
        description: "List all pending reminders. Use to check what reminders are scheduled.",
        inputSchema: {
          type: "object",
          properties: {
            status: {
              type: "string",
              enum: ["pending", "triggered", "dismissed", "all"],
              description: "Filter by status (default: pending)",
              default: "pending",
            },
          },
        },
      },
      {
        name: "dismiss_reminder",
        description: "Dismiss/acknowledge a triggered reminder. Call this after you've handled a reminder.",
        inputSchema: {
          type: "object",
          properties: {
            reminder_id: {
              type: "string",
              description: "ID of the reminder to dismiss",
            },
          },
          required: ["reminder_id"],
        },
      },
      // =================================================================
      // GTD Todo List Tools
      // =================================================================
      {
        name: "add_todo",
        description: "Add a task to your todo list. Supports GTD workflow with projects, contexts (@computer, @phone), priorities, due dates, and subtasks (via parent_id).",
        inputSchema: {
          type: "object",
          properties: {
            content: {
              type: "string",
              description: "What needs to be done",
            },
            status: {
              type: "string",
              enum: ["backlog", "todo", "in_progress", "blocked"],
              description: "Initial status (default: todo)",
              default: "todo",
            },
            priority: {
              type: "string",
              enum: ["urgent", "high", "medium", "low", "none"],
              description: "Priority level (default: medium)",
              default: "medium",
            },
            project: {
              type: "string",
              description: "Project name (created if doesn't exist)",
            },
            contexts: {
              type: "array",
              items: { type: "string" },
              description: "Contexts like @computer, @phone, @errands",
            },
            due_date: {
              type: "string",
              description: "Due date - ISO format or 'today', 'tomorrow', 'monday', etc.",
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Optional tags for categorization",
            },
            blocked_on: {
              type: "string",
              description: "Who/what you're waiting on (sets status to blocked)",
            },
            notes: {
              type: "string",
              description: "Additional notes",
            },
            recurrence: {
              type: "string",
              enum: ["daily", "weekly", "monthly"],
              description: "Recurrence pattern for repeating tasks",
            },
            parent_id: {
              type: "string",
              description: "Parent todo ID or short prefix (e.g. SHO-8) to create this as a subtask of that todo",
            },
            blocked_by: {
              type: "array",
              items: { type: "string" },
              description: "Todos this one depends on, as short keys (e.g. SHO-3) or UUIDs. This todo stays blocked until they are done. Unknown references are rejected.",
            },
            related_memory_ids: {
              type: "array",
              items: { type: "string" },
              description: "Memory UUIDs that motivated this task — the 'why does this exist' link back to the memories it came from. Verified to exist before linking.",
            },
          },
          required: ["content"],
        },
      },
      {
        name: "list_todos",
        description: "List or search todos. Supports semantic search via query parameter, or GTD-style filtering. Returns Linear-style formatted output grouped by status.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Semantic search query - when provided, uses vector similarity to find matching todos instead of listing all",
            },
            status: {
              type: "array",
              items: { type: "string", enum: ["backlog", "todo", "in_progress", "blocked", "done", "cancelled"] },
              description: "Filter by status(es)",
            },
            project: {
              type: "string",
              description: "Filter by project name",
            },
            context: {
              type: "string",
              description: "Filter by context (e.g., @computer)",
            },
            priority: {
              type: "string",
              enum: ["urgent", "high", "medium", "low"],
              description: "Filter by priority",
            },
            due: {
              type: "string",
              enum: ["today", "overdue", "this_week", "all"],
              description: "Filter by due date",
            },
            limit: {
              type: "number",
              description: "Maximum results (default: 50)",
              default: 50,
            },
            offset: {
              type: "number",
              description: "Skip first N items for pagination (default: 0)",
              default: 0,
            },
          },
        },
      },
      {
        name: "update_todo",
        description: "Update a todo's properties. Use short ID prefix (e.g., SHO-1a2b) or full ID.",
        inputSchema: {
          type: "object",
          properties: {
            todo_id: {
              type: "string",
              description: "Todo ID or short prefix",
            },
            content: {
              type: "string",
              description: "New content",
            },
            status: {
              type: "string",
              enum: ["backlog", "todo", "in_progress", "blocked", "done", "cancelled"],
              description: "New status",
            },
            priority: {
              type: "string",
              enum: ["urgent", "high", "medium", "low", "none"],
              description: "New priority",
            },
            project: {
              type: "string",
              description: "New project name",
            },
            contexts: {
              type: "array",
              items: { type: "string" },
              description: "New contexts",
            },
            due_date: {
              type: "string",
              description: "New due date",
            },
            blocked_on: {
              type: "string",
              description: "Who/what you're waiting on",
            },
            notes: {
              type: "string",
              description: "Additional notes",
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "New tags",
            },
            parent_id: {
              type: "string",
              description: "Parent todo ID or short prefix to make this a subtask. Pass empty string to remove parent.",
            },
            blocked_by: {
              type: "array",
              items: { type: "string" },
              description: "Replace this todo's dependencies with these todos, as short keys (e.g. SHO-3) or UUIDs. Pass an empty array to clear them. Unknown references are rejected.",
            },
            related_memory_ids: {
              type: "array",
              items: { type: "string" },
              description: "Replace the memory UUIDs this task traces back to. Pass an empty array to clear them. Verified to exist before linking.",
            },
          },
          required: ["todo_id"],
        },
      },
      {
        name: "complete_todo",
        description: "Mark a todo as complete. For recurring tasks, automatically creates the next occurrence.",
        inputSchema: {
          type: "object",
          properties: {
            todo_id: {
              type: "string",
              description: "Todo ID or short prefix",
            },
          },
          required: ["todo_id"],
        },
      },
      {
        name: "delete_todo",
        description: "Delete a todo permanently.",
        inputSchema: {
          type: "object",
          properties: {
            todo_id: {
              type: "string",
              description: "Todo ID or short prefix",
            },
          },
          required: ["todo_id"],
        },
      },
      {
        name: "reorder_todo",
        description: "Move a todo up or down within its status group. Use to prioritize tasks manually.",
        inputSchema: {
          type: "object",
          properties: {
            todo_id: {
              type: "string",
              description: "Todo ID or short prefix",
            },
            direction: {
              type: "string",
              enum: ["up", "down"],
              description: "Direction to move the todo",
            },
          },
          required: ["todo_id", "direction"],
        },
      },
      {
        name: "add_project",
        description: "Create a new project to group todos. Use parent to create a sub-project under another project.",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "Project name",
            },
            prefix: {
              type: "string",
              description: "Custom prefix for todo IDs (e.g., 'BOLT', 'MEM'). Auto-derived from name if not provided.",
            },
            description: {
              type: "string",
              description: "Project description",
            },
            parent: {
              type: "string",
              description: "Parent project name or ID to create a sub-project",
            },
          },
          required: ["name"],
        },
      },
      {
        name: "list_projects",
        description: "List all projects with todo counts and status breakdown.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "archive_project",
        description:
          "Archive a project. Archived projects are hidden by default but can be restored.",
        inputSchema: {
          type: "object",
          properties: {
            project: {
              type: "string",
              description: "Project name or ID to archive",
            },
          },
          required: ["project"],
        },
      },
      {
        name: "delete_project",
        description:
          "Permanently delete a project. Use delete_todos=true to also delete all todos in the project.",
        inputSchema: {
          type: "object",
          properties: {
            project: {
              type: "string",
              description: "Project name or ID to delete",
            },
            delete_todos: {
              type: "boolean",
              description: "Also delete all todos in this project (default: false)",
            },
          },
          required: ["project"],
        },
      },
      {
        name: "todo_stats",
        description: "Get statistics about your todos - counts by status, overdue items, etc.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "list_subtasks",
        description: "List subtasks of a parent todo. Use add_todo with parent_id to create subtasks.",
        inputSchema: {
          type: "object",
          properties: {
            parent_id: {
              type: "string",
              description: "Parent todo ID or short prefix",
            },
          },
          required: ["parent_id"],
        },
      },
      {
        name: "add_todo_comment",
        description: "Add a comment to a todo. Use to track progress, notes, or resolution details.",
        inputSchema: {
          type: "object",
          properties: {
            todo_id: {
              type: "string",
              description: "Todo ID or short prefix (e.g., 'BOLT-1', 'MEM-2')",
            },
            content: {
              type: "string",
              description: "Comment content (supports markdown)",
            },
            comment_type: {
              type: "string",
              enum: ["comment", "progress", "resolution", "activity"],
              description: "Type of comment: comment (default), progress (updates), resolution (fix details), activity (system)",
            },
          },
          required: ["todo_id", "content"],
        },
      },
      {
        name: "list_todo_comments",
        description: "List all comments and activity history for a specific todo.",
        inputSchema: {
          type: "object",
          properties: {
            todo_id: {
              type: "string",
              description: "Todo ID or short prefix (e.g., 'BOLT-1', 'MEM-2')",
            },
          },
          required: ["todo_id"],
        },
      },
      {
        name: "update_todo_comment",
        description: "Update an existing comment on a todo.",
        inputSchema: {
          type: "object",
          properties: {
            todo_id: {
              type: "string",
              description: "Todo ID or short prefix",
            },
            comment_id: {
              type: "string",
              description: "Comment ID (UUID)",
            },
            content: {
              type: "string",
              description: "New comment content",
            },
          },
          required: ["todo_id", "comment_id", "content"],
        },
      },
      {
        name: "delete_todo_comment",
        description: "Delete a comment from a todo.",
        inputSchema: {
          type: "object",
          properties: {
            todo_id: {
              type: "string",
              description: "Todo ID or short prefix",
            },
            comment_id: {
              type: "string",
              description: "Comment ID (UUID)",
            },
          },
          required: ["todo_id", "comment_id"],
        },
      },
      {
        name: "read_memory",
        description: "Read the FULL content of a specific memory by ID. Use this when you need to see the complete text of a memory that was truncated in search results.",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: {
              type: "string",
              description: "The memory ID (full UUID or short prefix like '5581cd02')",
            },
          },
          required: ["memory_id"],
        },
      },
      // =======================================================================
      // CAUSAL LINEAGE, KNOWLEDGE GRAPH, ANOMALIES, FACTS
      //
      // Abstraction choice: one tool per QUESTION an agent asks, not one per
      // API route. "Why did X happen / what did X cause" (trace_lineage),
      // "what causal structure exists" (list_causal_edges), "record a causal
      // link" (add_causal_link), "settle an inferred link" (validate_causal_link),
      // "what surrounds this entity" (explore_entity), "what entities exist"
      // (list_entities), "what is statistically unusual" (list_anomalies),
      // "what does the system know as distilled fact" (search_facts), and
      // "these memories helped/misled" (reinforce_memories). Mode parameters
      // are used only where the choice is a filter on the same question
      // (trace direction, facts query-vs-entity), never to fuse different
      // questions into one tool — an LLM discriminates between tools by name
      // and first sentence, and a grab-bag "lineage" tool with an operation
      // enum would make every causal question a two-step guess.
      //
      // Deliberately NOT exposed (operator/maintenance surface, not agent
      // affordances): graph clear/rebuild/canonicalize, tier-census/curvature/
      // universe (dashboard diagnostics), lineage branches (pivot bookkeeping
      // with no natural agent trigger), raw entity/relationship writes (the
      // graph is built by extraction; manual writes bypass provenance).
      // =======================================================================
      {
        name: "trace_lineage",
        description: "Trace the causal chain of a memory: what led to it (direction=backward, the default) or what it went on to cause (forward). Follows typed causal edges between memories (Caused, ResolvedBy, InformedBy, SupersededBy, TriggeredBy) and reports the root cause — the oldest ancestor — when tracing backward. Use this, not recall, to answer 'why did X happen' or 'what did X cause' for a memory whose ID you have (from recall/read_memory output). For exploring around a named entity instead of a memory, use explore_entity.",
        inputSchema: {
          type: "object",
          properties: {
            memory_id: {
              type: "string",
              description: "Memory to trace from (full UUID or 8+ char prefix from recall results)",
            },
            direction: {
              type: "string",
              enum: ["backward", "forward", "both"],
              description: "backward = find causes (default), forward = find effects, both = full chain",
              default: "backward",
            },
            max_depth: {
              type: "number",
              description: "Maximum hops to traverse (default: 10, max: 100)",
              default: 10,
            },
          },
          required: ["memory_id"],
        },
      },
      {
        name: "list_causal_edges",
        description: "Survey the causal lineage graph: totals by relation type and source (inferred/confirmed/explicit), average confidence, and the highest-confidence edges with their edge IDs. Use to see what causal structure exists across all memories, or to find edge IDs for validate_causal_link. For the chain around one specific memory, use trace_lineage instead.",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Maximum edges to list (default: 15)",
              default: 15,
            },
          },
        },
      },
      {
        name: "add_causal_link",
        description: "Record an explicit causal edge between two memories. from_memory_id is the cause/origin (the earlier event), to_memory_id is the effect (the later one); the relation reads from→to, e.g. relation=Caused means 'from caused to'. Use when you learn that one remembered event caused, resolved, informed, superseded, or triggered another. Explicit links carry full confidence and strengthen causal recall and root-cause tracing.",
        inputSchema: {
          type: "object",
          properties: {
            from_memory_id: {
              type: "string",
              description: "The cause/origin memory (full UUID or 8+ char prefix)",
            },
            to_memory_id: {
              type: "string",
              description: "The effect memory (full UUID or 8+ char prefix)",
            },
            relation: {
              type: "string",
              enum: ["Caused", "ResolvedBy", "InformedBy", "SupersededBy", "TriggeredBy", "BranchedFrom", "RelatedTo"],
              description: "Causal relation read from→to: Caused (error→task it spawned), ResolvedBy (task→fix that closed it), InformedBy (evidence→decision it informed: 'from informed to'), SupersededBy (old→replacement), TriggeredBy ('from triggered to'), BranchedFrom (pivot→the origin it branched from; the only relation whose from is the NEWER memory), RelatedTo (causal but untyped)",
            },
          },
          required: ["from_memory_id", "to_memory_id", "relation"],
        },
      },
      {
        name: "validate_causal_link",
        description: "Confirm or reject a causal edge by its edge ID (shown by trace_lineage and list_causal_edges). Confirming an inferred edge raises it to full confidence and strengthens the knowledge-graph connections between the two memories' entities; rejecting deletes the edge. Use when the user or the evidence settles whether an inferred causal link is real — confirmed chains make root-cause tracing trustworthy.",
        inputSchema: {
          type: "object",
          properties: {
            edge_id: {
              type: "string",
              description: "The lineage edge ID (edge_id field in trace_lineage / list_causal_edges output)",
            },
            verdict: {
              type: "string",
              enum: ["confirm", "reject"],
              description: "confirm = the causal link is real; reject = it is spurious (deletes the edge)",
            },
          },
          required: ["edge_id", "verdict"],
        },
      },
      {
        name: "explore_entity",
        description: "Walk the knowledge graph outward from a named entity: connected entities by hop distance, and the typed relationships between them (Causes, Triggers, DependsOn, custom types, plus co-occurrence), each with strength and source context. Use when you have an entity NAME (person, system, place, ship, ...) and want its connections — recall searches episodic text instead, and trace_lineage follows memory-to-memory causality instead. Name matching is fuzzy (case-insensitive, stems, substrings); the output states which entity actually matched. Use list_entities to browse what exists.",
        inputSchema: {
          type: "object",
          properties: {
            entity_name: {
              type: "string",
              description: "Entity name to start from (fuzzily matched; browse names via list_entities)",
            },
            max_depth: {
              type: "number",
              description: "Hops to traverse: 1 = direct neighbors (default), 2 = neighborhood, 3 = wide (can be large)",
              default: 1,
            },
          },
          required: ["entity_name"],
        },
      },
      {
        name: "list_entities",
        description: "List the entities in the knowledge graph, ranked by salience (learned importance), with their types and mention counts. Use to discover what people, systems, places, and concepts the graph tracks — or to find the exact name to pass to explore_entity. For memory contents use recall; this is the graph's cast of characters.",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Maximum entities to return (default: 30)",
              default: 30,
            },
          },
        },
      },
      {
        name: "list_anomalies",
        description: "Rank recent memories by statistical deviation from this user's own rolling baseline (novel entities, unusual entity co-occurrence, untyped-relation share). Each entry carries per-component z-scores and a deterministic explanation of why it deviates. Use to answer 'what has been unusual lately' or to spot weak signals worth investigating — this is deviation scoring against the corpus's own shape, not content search; there is no query. The baseline is the most recent 200 scored episodes (server default); at least 10 must exist before anything is scored.",
        inputSchema: {
          type: "object",
          properties: {
            limit: {
              type: "number",
              description: "Maximum entries, ranked by deviation (default: 10)",
              default: 10,
            },
            min_sigma: {
              type: "number",
              description: "|z| at or above which an entry is flagged as anomalous (default: 2.0)",
              default: 2.0,
            },
          },
        },
      },
      {
        name: "search_facts",
        description: "Search the distilled semantic facts the system has consolidated out of episodic memories — stable knowledge like preferences, capabilities, relationships, and procedures, each with a confidence score and supporting-memory count. Pass query for keyword search, entity for facts about one entity, or neither to list recent facts. Use for 'what does the system KNOW about X' — recall searches raw episodic memories instead, and fact_narratives returns topic-clustered summaries instead of individual facts.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Keyword search over fact statements",
            },
            entity: {
              type: "string",
              description: "Return facts related to this entity (takes precedence if both given)",
            },
            limit: {
              type: "number",
              description: "Maximum facts to return (default: 20)",
              default: 20,
            },
          },
        },
      },
      {
        name: "reinforce_memories",
        description: "Give Hebbian feedback on memories after using them: outcome 'helpful' boosts their importance and strengthens their associations so they surface more readily, 'misleading' decays them, 'neutral' just records the access. Call after completing a task with the memory IDs recall gave you — this is how retrieval learns from outcomes instead of only from similarity.",
        inputSchema: {
          type: "object",
          properties: {
            memory_ids: {
              type: "array",
              items: { type: "string" },
              description: "Memory IDs that were used (full UUIDs or 8+ char prefixes from recall output)",
            },
            outcome: {
              type: "string",
              enum: ["helpful", "misleading", "neutral"],
              description: "helpful = they contributed to success; misleading = they pointed the wrong way; neutral = merely accessed",
            },
          },
          required: ["memory_ids", "outcome"],
        },
      },
];

// List available tools, decorated with annotations and output schemas.
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOL_DEFINITIONS.map(decorateTool) };
});

// Auto-stream context from tool arguments (captures conversation intent)
function autoStreamContext(toolName: string, args: Record<string, unknown>): void {
  // Skip tools that already handle their own streaming or are meta/diagnostic
  if (["proactive_context", "streaming_status", "token_status", "reset_token_session"].includes(toolName)) return;

  // Tools annotated readOnlyHint:true must not modify the store. Streaming a
  // memory here would make that annotation false — a client auto-approving
  // "safe" tools would then silently write on every read. The annotation is the
  // contract; this gate is what makes it true.
  if (isReadOnlyTool(toolName)) return;

  // Extract meaningful context from tool arguments
  let context = "";
  if (args.query && typeof args.query === "string") {
    context = `Query: ${args.query}`;
  } else if (args.content && typeof args.content === "string") {
    context = args.content;
  } else if (args.context && typeof args.context === "string") {
    context = args.context;
  }

  // Stream if we have meaningful context
  if (context.length >= 20) {
    streamMemory(context, ["auto-context", toolName], "user");
  }
}

// Handle tool calls
const handleCallTool = async (request: CallToolRequest) => {
  const { name, arguments: args } = request.params;

  // Ensure streaming is connected (lazy reconnect on tool calls)
  if (STREAM_ENABLED && (!streamSocket || streamSocket.readyState !== WebSocket.OPEN)) {
    connectStream().catch(() => {});
  }

  // Track tool call counts for session digest
  toolCallCounts.set(name, (toolCallCounts.get(name) || 0) + 1);

  // Auto-capture context from tool arguments (non-blocking)
  autoStreamContext(name, args as Record<string, unknown>);

  // Check server availability first. If the health check fails, try to bring
  // the backend back (it may have crashed or been killed since we started)
  // instead of failing every subsequent tool call for the rest of the session.
  let serverUp = await isServerAvailable();
  if (!serverUp) {
    console.error("[shodh-memory] Backend health check failed — attempting to restart it...");
    serverUp = await recoverBackend();
  }
  if (!serverUp) {
    return {
      content: [
        {
          type: "text",
          text: `Memory server unavailable at ${BACKEND_LOCATION}, and automatic restart did not bring it back. Please ensure shodh-memory-server is running.\n\nTo start: shodh-memory-server`,
        },
      ],
      isError: true,
    };
  }

  // Result type for tool responses
  // Result type for tool responses.
  //
  // `structuredContent` is the machine-readable channel that accompanies the
  // formatted text; `content` is always populated as well (the MCP spec expects
  // it for backwards compatibility, and the formatted text reads better inline
  // for small results). Any tool declaring an outputSchema MUST set this on
  // every non-error return — the SDK client throws when an outputSchema is
  // declared and structuredContent is absent on a successful call.
  type ToolResult = {
    content: { type: string; text: string }[];
    isError?: boolean;
    structuredContent?: Record<string, unknown>;
  };

  // Inner function to execute tool logic - allows us to capture result for auto-ingest
  const executeTool = async (): Promise<ToolResult> => {
    switch (name) {
      case "remember": {
        const {
          content,
          type = "Observation",
          tags = [],
          created_at,
          // SHO-104: Richer context encoding fields
          emotional_valence,
          emotional_arousal,
          emotion,
          source_type,
          credibility,
          episode_id,
          sequence_number,
          preceding_memory_id,
          // Hierarchy
          parent_id,
          // Importance override
          importance,
          // Robotics context
          robot_id,
          mission_id,
          geo_location,
          local_position,
          heading,
          action_type,
          reward,
          sensor_data,
          outcome_type,
          terrain_type,
        } = args as {
          content: string;
          type?: string;
          tags?: string[];
          created_at?: string;
          emotional_valence?: number;
          emotional_arousal?: number;
          emotion?: string;
          source_type?: string;
          credibility?: number;
          episode_id?: string;
          sequence_number?: number;
          preceding_memory_id?: string;
          parent_id?: string;
          importance?: number;
          robot_id?: string;
          mission_id?: string;
          geo_location?: number[];
          local_position?: number[];
          heading?: number;
          action_type?: string;
          reward?: number;
          sensor_data?: Record<string, number>;
          outcome_type?: string;
          terrain_type?: string;
        };

        if (!content || content.length === 0) {
          return { content: [{ type: "text", text: "Error: 'content' is required and cannot be empty" }], isError: true };
        }
        if (content.length > MAX_CONTENT_LENGTH) {
          return { content: [{ type: "text", text: `Error: 'content' exceeds maximum length of ${MAX_CONTENT_LENGTH} characters` }], isError: true };
        }
        // Validate robotics fields
        if (geo_location && geo_location.length !== 3) {
          return { content: [{ type: "text", text: "Error: 'geo_location' must be exactly [latitude, longitude, altitude]" }], isError: true };
        }
        if (local_position && local_position.length !== 3) {
          return { content: [{ type: "text", text: "Error: 'local_position' must be exactly [x, y, z]" }], isError: true };
        }
        if (reward !== undefined && (reward < -1.0 || reward > 1.0)) {
          return { content: [{ type: "text", text: `Error: 'reward' must be between -1.0 and 1.0, got: ${reward}` }], isError: true };
        }
        if (heading !== undefined && (heading < 0 || heading > 360)) {
          return { content: [{ type: "text", text: `Error: 'heading' must be between 0 and 360 degrees, got: ${heading}` }], isError: true };
        }

        const result = await apiCall<{ id: string }>("/api/remember", "POST", {
          user_id: USER_ID,
          content,
          memory_type: type,
          tags,
          ...(created_at && { created_at }),
          // SHO-104: Pass richer context to API
          ...(emotional_valence !== undefined && { emotional_valence }),
          ...(emotional_arousal !== undefined && { emotional_arousal }),
          ...(emotion && { emotion }),
          ...(source_type && { source_type }),
          ...(credibility !== undefined && { credibility }),
          ...(episode_id && { episode_id }),
          ...(sequence_number !== undefined && { sequence_number }),
          ...(preceding_memory_id && { preceding_memory_id }),
          // Hierarchy
          ...(parent_id && { parent_id }),
          // Importance override
          ...(importance !== undefined && { importance }),
          // Robotics context
          ...(robot_id && { robot_id }),
          ...(mission_id && { mission_id }),
          ...(geo_location && geo_location.length === 3 && { geo_location }),
          ...(local_position && local_position.length === 3 && { local_position }),
          ...(heading !== undefined && { heading }),
          ...(action_type && { action_type }),
          ...(reward !== undefined && { reward }),
          ...(sensor_data && Object.keys(sensor_data).length > 0 && { sensor_data }),
          ...(outcome_type && { outcome_type }),
          ...(terrain_type && { terrain_type }),
        });

        // Format response with branded display
        let response = `🐘 Memory Stored\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `📝 ${renderContent(content, result.id, 60, false)}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Type: ${type}`;
        if (tags.length > 0) {
          response += ` │ Tags: ${tags.join(', ')}`;
        }
        response += `\nID: ${result.id}`;

        return {
          content: [{ type: "text", text: response }],
          structuredContent: { id: result.id, memory_type: type, tags },
        };
      }

      case "recall": {
        const {
          query, limit: rawLimit = 5, mode = "hybrid", session_id,
          robot_id, mission_id, geo_lat, geo_lon, geo_radius_meters,
          action_type, reward_min, reward_max, outcome_type, failures_only,
          terrain_type, tags, debug: debugMode, full_content = false,
        } = args as {
          query: string; limit?: number; mode?: string; session_id?: string;
          robot_id?: string; mission_id?: string;
          geo_lat?: number; geo_lon?: number; geo_radius_meters?: number;
          action_type?: string; reward_min?: number; reward_max?: number;
          outcome_type?: string; failures_only?: boolean;
          terrain_type?: string; tags?: string[]; debug?: boolean;
          full_content?: boolean;
        };

        if (!query || query.length === 0) {
          return { content: [{ type: "text", text: "Error: 'query' is required and cannot be empty" }], isError: true };
        }
        if (query.length > MAX_QUERY_LENGTH) {
          return { content: [{ type: "text", text: `Error: 'query' exceeds maximum length of ${MAX_QUERY_LENGTH} characters` }], isError: true };
        }
        const validModes = ["semantic", "associative", "temporal", "hybrid", "spatial", "mission", "action_outcome"];
        if (!validModes.includes(mode)) {
          return { content: [{ type: "text", text: `Error: 'mode' must be one of: ${validModes.join(", ")}` }], isError: true };
        }
        const limit = Math.max(1, Math.min(Math.floor(rawLimit), MAX_LIMIT));

        // Mode-specific required parameter validation
        if (mode === "spatial" && (geo_lat === undefined || geo_lon === undefined || geo_radius_meters === undefined)) {
          return { content: [{ type: "text", text: "Error: 'spatial' mode requires geo_lat, geo_lon, and geo_radius_meters" }], isError: true };
        }
        if (mode === "mission" && !mission_id) {
          return { content: [{ type: "text", text: "Error: 'mission' mode requires mission_id" }], isError: true };
        }
        if (reward_min !== undefined && reward_max !== undefined && reward_min > reward_max) {
          return { content: [{ type: "text", text: `Error: reward_min (${reward_min}) must be <= reward_max (${reward_max})` }], isError: true };
        }

        interface StageTiming {
          query_analysis_us: number;
          embedding_us: number;
          graph_expansion_us: number;
          vector_search_us: number;
          fusion_us: number;
          scoring_us: number;
          total_us: number;
        }

        interface ScoreAttribution {
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

        interface RetrievalStats {
          mode: string;
          semantic_candidates: number;
          graph_candidates: number;
          graph_density: number;
          graph_weight: number;
          semantic_weight: number;
          graph_hops: number;
          entities_activated: number;
          retrieval_time_us: number;
          stage_timings?: StageTiming;
          score_attributions?: ScoreAttribution[];
        }

        interface RecallTodo {
          id: string;
          short_id: string;
          content: string;
          status: string;
          priority: string;
          project?: string;
          score: number;
          created_at: string;
        }

        interface RecallLineageEdge {
          from: string;
          to: string;
          relation: string;
          confidence: number;
        }

        interface RecallResponse {
          memories: Memory[];
          count: number;
          retrieval_stats?: RetrievalStats;
          todos?: RecallTodo[];
          todo_count?: number;
          lineage?: RecallLineageEdge[];
          lineage_count?: number;
        }

        const result = await apiCall<RecallResponse>("/api/recall", "POST", {
          user_id: USER_ID,
          query,
          limit,
          mode,
          ...(session_id ? { session_id } : {}),
          ...(robot_id ? { robot_id } : {}),
          ...(mission_id ? { mission_id } : {}),
          ...(geo_lat !== undefined ? { geo_lat } : {}),
          ...(geo_lon !== undefined ? { geo_lon } : {}),
          ...(geo_radius_meters !== undefined ? { geo_radius_meters } : {}),
          ...(action_type ? { action_type } : {}),
          ...(reward_min !== undefined ? { reward_min } : {}),
          ...(reward_max !== undefined ? { reward_max } : {}),
          ...(outcome_type ? { outcome_type } : {}),
          ...(failures_only !== undefined ? { failures_only } : {}),
          ...(terrain_type ? { terrain_type } : {}),
          ...(tags && tags.length > 0 ? { tags } : {}),
          ...(debugMode ? { debug: true } : {}),
        });

        const memories = result.memories || [];
        const todos = result.todos || [];
        const stats = result.retrieval_stats;
        const lineage = result.lineage || [];

        // Structured payload for both the empty and non-empty paths. A tool
        // that declares an outputSchema must emit structuredContent on EVERY
        // successful return, including "nothing found" — the SDK client rejects
        // a schema'd success result that omits it.
        const recallStructured = (): Record<string, unknown> => ({
          query,
          mode,
          memories: memories.map((m) => structuredMemory(m, full_content)),
          todos: todos.map((t) =>
            compact({
              id: t.id,
              short_id: t.short_id,
              content: t.content,
              status: t.status,
              priority: t.priority,
              project: t.project,
              score: t.score,
              created_at: t.created_at,
            }),
          ),
          lineage: lineage.map((e) =>
            compact({ from: e.from, to: e.to, relation: e.relation, confidence: e.confidence }),
          ),
          memory_count: memories.length,
          todo_count: todos.length,
        });

        if (memories.length === 0 && todos.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: `🐘 No memories or todos found for: "${query}"\n   Mode: ${mode}`,
              },
            ],
            structuredContent: recallStructured(),
          };
        }

        // Build formatted response
        const totalCount = memories.length + todos.length;
        let response = `🐘 Recalled ${totalCount} Results`;
        if (memories.length > 0 && todos.length > 0) {
          response += ` (${memories.length} memories, ${todos.length} todos)`;
        }
        response += `\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Query: "${query.slice(0, 40)}${query.length > 40 ? '...' : ''}" │ Mode: ${mode}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        // Helper to format timestamp
        const formatTime = (ts: string | undefined): string => {
          if (!ts) return '';
          const d = new Date(ts);
          const now = new Date();
          const diffMs = now.getTime() - d.getTime();
          const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

          if (diffDays === 0) {
            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          } else if (diffDays === 1) {
            return 'Yesterday';
          } else if (diffDays < 7) {
            return `${diffDays}d ago`;
          } else {
            return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
          }
        };

        // Backend normalizes scores relative to top result (0-0.95 range).
        // Pass through directly — no client-side rescaling needed.
        const memoryDisplayScores = memories.map(m => m.score || 0);
        const todoDisplayScores = todos.map(t => t.score || 0);

        // Format memories
        if (memories.length > 0) {
          response += `📝 MEMORIES\n`;
          for (let i = 0; i < memories.length; i++) {
            const m = memories[i];
            const content = getContent(m);
            const displayScore = memoryDisplayScores[i];
            const score = (displayScore * 100).toFixed(0);
            const filled = Math.max(0, Math.min(10, Math.round(displayScore * 10)));
            const matchBar = '█'.repeat(filled) + '░'.repeat(10 - filled);
            const timeStr = formatTime(m.created_at);

            response += `• ${matchBar} ${score}% │ ${timeStr}\n`;
            response += `  ${renderContent(content, m.id, MEMORY_PREVIEW_MAX, full_content)}\n`;
            response += `  ┗━ ${getType(m)}${m.tier ? ` │ ${m.tier}` : ''} │ ${m.id}\n`;
            if (i < memories.length - 1) response += `\n`;
          }
        }

        // Format todos
        if (todos.length > 0) {
          if (memories.length > 0) response += `\n`;
          response += `✅ TODOS\n`;
          for (let i = 0; i < todos.length; i++) {
            const t = todos[i];
            const displayScore = todoDisplayScores[i];
            const score = (displayScore * 100).toFixed(0);
            const filled = Math.max(0, Math.min(10, Math.round(displayScore * 10)));
            const matchBar = '█'.repeat(filled) + '░'.repeat(10 - filled);
            const statusIcon = t.status === 'done' ? '✓' : t.status === 'in_progress' ? '▶' : t.status === 'blocked' ? '⊗' : '○';
            const timeStr = formatTime(t.created_at);

            response += `• ${matchBar} ${score}% │ ${timeStr}\n`;
            response += `  ${statusIcon} ${renderContent(t.content, undefined, 180, full_content)}\n`;
            response += `  ┗━ ${t.short_id} │ ${t.status} │ ${t.priority}`;
            if (t.project) response += ` │ ${t.project}`;
            response += `\n`;
            if (i < todos.length - 1) response += `\n`;
          }
        }

        // Build stats summary for associative/hybrid modes
        if (stats && (mode === "associative" || mode === "hybrid")) {
          response += `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `📊 Retrieval Stats\n`;
          const graphPct = (stats.graph_weight * 100).toFixed(0);
          const semPct = (stats.semantic_weight * 100).toFixed(0);
          response += `   Graph: ${graphPct}% │ Semantic: ${semPct}% │ Density: ${(stats.graph_density ?? 0).toFixed(2)}\n`;
          response += `   Candidates: ${stats.graph_candidates} graph + ${stats.semantic_candidates} semantic\n`;
          response += `   Entities: ${stats.entities_activated} │ Time: ${(stats.retrieval_time_us / 1000).toFixed(1)}ms`;
        }

        // Format lineage edges connecting recalled memories
        if (lineage.length > 0) {
          // Build short ID lookup from recalled memories
          const idShort = (id: string) => id;
          const idToContent = new Map<string, string>();
          for (const m of memories) {
            idToContent.set(m.id, getContent(m).slice(0, 40));
          }

          response += `\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `🔗 LINEAGE (${lineage.length} causal edge${lineage.length > 1 ? 's' : ''})\n`;
          for (const edge of lineage) {
            const fromLabel = idToContent.get(edge.from) || idShort(edge.from);
            const toLabel = idToContent.get(edge.to) || idShort(edge.to);
            const conf = (edge.confidence * 100).toFixed(0);
            response += `  ${idShort(edge.from)} ──${edge.relation}──▶ ${idShort(edge.to)}  (${conf}%)\n`;
            response += `    "${fromLabel}..." → "${toLabel}..."\n`;
          }
        }

        // Debug diagnostics: stage timings + per-memory score attribution
        if (debugMode && stats) {
          response += `\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `🔬 RETRIEVAL DIAGNOSTICS\n`;

          if (stats.stage_timings) {
            const t = stats.stage_timings;
            response += `\n⏱️ Stage Timings:\n`;
            response += `  Query analysis:  ${(t.query_analysis_us / 1000).toFixed(1)}ms\n`;
            response += `  Embedding:       ${(t.embedding_us / 1000).toFixed(1)}ms\n`;
            response += `  Graph expansion: ${(t.graph_expansion_us / 1000).toFixed(1)}ms\n`;
            response += `  Vector search:   ${(t.vector_search_us / 1000).toFixed(1)}ms\n`;
            response += `  RRF fusion:      ${(t.fusion_us / 1000).toFixed(1)}ms\n`;
            response += `  Scoring:         ${(t.scoring_us / 1000).toFixed(1)}ms\n`;
            response += `  ─────────────────────────\n`;
            response += `  Total:           ${(t.total_us / 1000).toFixed(1)}ms\n`;
          }

          if (stats.score_attributions && stats.score_attributions.length > 0) {
            response += `\n📊 Score Attribution (top ${Math.min(stats.score_attributions.length, 5)}):\n`;
            for (const attr of stats.score_attributions.slice(0, 5)) {
              const shortId = attr.memory_id.slice(0, 8);
              response += `\n  ${shortId}… (final: ${attr.final_score.toFixed(4)})\n`;
              response += `    Sources: ${attr.sources.join(", ")}\n`;
              response += `    RRF base: ${attr.rrf_base.toFixed(4)} │ Graph: ${attr.graph_rrf.toFixed(4)} │ Hybrid: ${attr.hybrid_rrf.toFixed(4)}\n`;

              // Show non-neutral boosts only (neutral = 1.0 for multiplicative, 0.0 for additive)
              const boosts: string[] = [];
              if (attr.hebbian_boost !== 1.0) boosts.push(`hebbian: ${attr.hebbian_boost.toFixed(3)}`);
              if (attr.attribute_boost !== 1.0) boosts.push(`attribute: ${attr.attribute_boost.toFixed(3)}`);
              if (attr.temporal_prefilter_boost !== 1.0) boosts.push(`temporal: ${attr.temporal_prefilter_boost.toFixed(3)}`);
              if (attr.temporal_fact_boost !== 1.0) boosts.push(`fact-temporal: ${attr.temporal_fact_boost.toFixed(3)}`);
              if (attr.interference_adjustment !== 1.0) boosts.push(`interference: ${attr.interference_adjustment.toFixed(3)}`);
              if (attr.prospective_boost !== 1.0) boosts.push(`prospective: ${attr.prospective_boost.toFixed(3)}`);
              if (attr.fact_source_boost !== 1.0) boosts.push(`fact-source: ${attr.fact_source_boost.toFixed(3)}`);
              if (attr.ontological_boost !== 1.0) boosts.push(`ontological: ${attr.ontological_boost.toFixed(3)}`);
              if (boosts.length > 0) {
                response += `    Boosts: ${boosts.join(" │ ")}\n`;
              }

              response += `    L5 factors: imp=${attr.importance_factor.toFixed(3)} rec=${attr.recency_factor.toFixed(3)} aro=${attr.arousal_factor.toFixed(3)} cred=${attr.credibility_factor.toFixed(3)} fb=${attr.feedback_multiplier.toFixed(3)}\n`;
              if (attr.quality_gate !== 1.0) {
                response += `    Quality gate: ${attr.quality_gate.toFixed(3)}\n`;
              }
            }
          }
        }

        return {
          content: [{ type: "text", text: response }],
          structuredContent: recallStructured(),
        };
      }

      case "recall_by_tags": {
        const { tags, limit: rawTagLimit = 50, full_content = false } = args as { tags: string[]; limit?: number; full_content?: boolean };

        if (!tags || tags.length === 0) {
          return {
            content: [{ type: "text", text: "Error: 'tags' is required and must contain at least one tag" }],
            isError: true,
          };
        }

        const tagLimit = Math.max(1, Math.min(Math.floor(rawTagLimit), MAX_LIMIT));
        const tagResult = await apiCall<{ memories: Memory[]; count: number }>("/api/recall/tags", "POST", {
          user_id: USER_ID,
          tags,
          limit: tagLimit,
        });

        const tagMemories = tagResult.memories || [];

        const tagStructured = (): Record<string, unknown> => ({
          tags,
          memories: tagMemories.map((m) => structuredMemory(m, full_content)),
          count: tagMemories.length,
        });

        if (tagMemories.length === 0) {
          return {
            content: [{ type: "text", text: `No memories found matching tags: ${tags.join(", ")}` }],
            structuredContent: tagStructured(),
          };
        }

        let tagResponse = `🏷️ Recall by Tags: ${tags.join(", ")}\n`;
        tagResponse += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        tagResponse += `Found ${tagMemories.length} memories\n\n`;

        for (let i = 0; i < tagMemories.length; i++) {
          const m = tagMemories[i];
          const content = getContent(m);
          const memTags = (m.experience?.tags || []).join(", ");
          tagResponse += `${String(i + 1).padStart(2)}. ${renderContent(content, m.id, MEMORY_PREVIEW_MAX, full_content)}\n`;
          tagResponse += `    ┗━ ${getType(m)} │ tags: [${memTags}] │ ${m.id}\n\n`;
        }

        return {
          content: [{ type: "text", text: tagResponse.trimEnd() }],
          structuredContent: tagStructured(),
        };
      }

      case "context_summary": {
        const {
          include_decisions = true,
          include_learnings = true,
          include_context = true,
          max_items = 5,
        } = args as {
          include_decisions?: boolean;
          include_learnings?: boolean;
          include_context?: boolean;
          max_items?: number;
        };

        // Fetch all memories
        const result = await apiCall<{ memories: Memory[] }>("/api/memories", "POST", {
          user_id: USER_ID,
        });

        const memories = result.memories || [];

        if (memories.length === 0) {
          let response = `🐘 Context Summary\n`;
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `No memories stored yet.\n`;
          response += `Start remembering to build context!`;
          return {
            content: [{ type: "text", text: response }],
          };
        }

        // Categorize memories
        const decisions: Memory[] = [];
        const learnings: Memory[] = [];
        const context: Memory[] = [];
        const patterns: Memory[] = [];
        const errors: Memory[] = [];

        for (const m of memories) {
          const type = getType(m);
          switch (type) {
            case 'Decision':
              decisions.push(m);
              break;
            case 'Learning':
              learnings.push(m);
              break;
            case 'Context':
              context.push(m);
              break;
            case 'Pattern':
              patterns.push(m);
              break;
            case 'Error':
              errors.push(m);
              break;
          }
        }

        // Build branded response
        let response = `🐘 Context Summary\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Total: ${memories.length} memories │ `;
        response += `📋 ${decisions.length} │ 💡 ${learnings.length} │ 📁 ${context.length} │ 🔄 ${patterns.length} │ ⚠️ ${errors.length}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        if (include_context && context.length > 0) {
          response += `📁 PROJECT CONTEXT\n`;
          for (const m of context.slice(0, max_items)) {
            response += `   • ${renderContent(getContent(m), m.id, 70, false)}\n`;
          }
          response += `\n`;
        }

        if (include_decisions && decisions.length > 0) {
          response += `📋 DECISIONS\n`;
          for (const m of decisions.slice(0, max_items)) {
            response += `   • ${renderContent(getContent(m), m.id, 70, false)}\n`;
          }
          response += `\n`;
        }

        if (include_learnings && learnings.length > 0) {
          response += `💡 LEARNINGS\n`;
          for (const m of learnings.slice(0, max_items)) {
            response += `   • ${renderContent(getContent(m), m.id, 70, false)}\n`;
          }
          response += `\n`;
        }

        if (patterns.length > 0) {
          response += `🔄 PATTERNS\n`;
          for (const m of patterns.slice(0, max_items)) {
            response += `   • ${renderContent(getContent(m), m.id, 70, false)}\n`;
          }
          response += `\n`;
        }

        if (errors.length > 0) {
          response += `⚠️ ERRORS TO AVOID\n`;
          for (const m of errors.slice(0, Math.min(3, max_items))) {
            response += `   • ${renderContent(getContent(m), m.id, 70, false)}\n`;
          }
        }

        if (decisions.length === 0 && learnings.length === 0 && context.length === 0) {
          response += `ℹ️  Tip: Use types like Decision, Learning, Context when remembering\n`;
          response += `   to build richer context summaries.`;
        }

        return {
          content: [{ type: "text", text: response.trimEnd() }],
        };
      }

      case "list_memories": {
        const { limit = 20 } = args as { limit?: number };

        const result = await apiCall<{ memories: Memory[] }>("/api/memories", "POST", {
          user_id: USER_ID,
        });

        const memories = (result.memories || []).slice(0, limit);

        const listStructured = (): Record<string, unknown> => ({
          memories: memories.map((m) => structuredMemory(m)),
          count: memories.length,
        });

        if (memories.length === 0) {
          let response = `🐘 Memory List\n`;
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `No memories stored yet.`;
          return {
            content: [{ type: "text", text: response }],
            structuredContent: listStructured(),
          };
        }

        // Group by type for summary
        const typeCounts: Record<string, number> = {};
        for (const m of memories) {
          const type = getType(m);
          typeCounts[type] = (typeCounts[type] || 0) + 1;
        }

        let response = `🐘 Memory List\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Showing ${memories.length} memories\n`;
        response += `Types: ${Object.entries(typeCounts).map(([t, c]) => `${t}(${c})`).join(' │ ')}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        for (let i = 0; i < memories.length; i++) {
          const m = memories[i];
          const content = getContent(m);
          const typeIcon = {
            'Decision': '📋',
            'Learning': '💡',
            'Context': '📁',
            'Pattern': '🔄',
            'Error': '⚠️',
            'Observation': '👁️',
            'Discovery': '🔍',
            'Task': '✅',
            'CodeEdit': '📝',
            'FileAccess': '📄',
            'Search': '🔎',
            'Command': '⚡',
            'Conversation': '💬',
          }[getType(m)] || '📦';

          response += `${String(i + 1).padStart(2)}. ${typeIcon} ${renderContent(content, m.id, MEMORY_PREVIEW_MAX, false)}\n`;
          response += `    ┗━ ${getType(m)}${m.tier ? ` │ ${m.tier}` : ''} │ ${m.id}\n`;
        }

        return {
          content: [{ type: "text", text: response.trimEnd() }],
          structuredContent: listStructured(),
        };
      }

      case "forget": {
        const { id } = args as { id: string };

        await apiCall(`/api/memory/${encodeURIComponent(id)}?user_id=${USER_ID}`, "DELETE");

        let response = `🐘 Memory Deleted\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `✓ Removed: ${id}`;

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "memory_stats": {
        // Wire shape of GET /api/users/{id}/stats. The tier counts and
        // total_retrievals are returned by the current backend but were missing
        // from this interface; they are declared here because the structured
        // payload passes them through.
        interface MemoryStats {
          total_memories: number;
          total_importance?: number;
          avg_importance?: number;
          average_importance?: number; // API uses this name
          graph_nodes: number;
          graph_edges: number;
          indexed_vectors?: number;
          vector_index_count?: number; // API uses this name
          working_memory_count?: number;
          session_memory_count?: number;
          long_term_memory_count?: number;
          total_retrievals?: number;
        }

        const result = await apiCall<MemoryStats>(`/api/users/${encodeURIComponent(USER_ID)}/stats`, "GET");

        // Handle both old and new field names for compatibility
        const indexedCount = result.vector_index_count ?? result.indexed_vectors ?? 0;
        const avgImportance = result.average_importance ?? result.avg_importance ?? 0;

        let response = `🐘 Memory Statistics\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Total Memories: ${result.total_memories || 0}\n`;
        response += `Graph: ${result.graph_nodes || 0} nodes │ ${result.graph_edges || 0} edges\n`;
        response += `Indexed Vectors: ${indexedCount}\n`;
        response += `Avg Importance: ${avgImportance.toFixed(2)}\n`;
        // These are buffer-occupancy counts, not a partition: a memory is
        // written to long-term storage immediately and also held in the working
        // buffer until evicted, so it is counted twice and the three numbers sum
        // to more than the total. Labelling them "working / session / long-term"
        // read as a tier breakdown that does not add up.
        response += `In working buffer: ${result.working_memory_count ?? 0} │ in session buffer: ${result.session_memory_count ?? 0} │ persisted: ${result.long_term_memory_count ?? 0}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        // No "By Type" section: GET /api/users/{id}/stats returns no type
        // histogram. list_memories computes one from the memories it lists.
        response += `\nFor a breakdown by memory type, call list_memories.\n`;

        return {
          content: [{ type: "text", text: response.trimEnd() }],
          structuredContent: compact({
            total_memories: result.total_memories ?? 0,
            working_memory_count: result.working_memory_count,
            session_memory_count: result.session_memory_count,
            long_term_memory_count: result.long_term_memory_count,
            vector_index_count: indexedCount,
            average_importance: avgImportance,
            total_retrievals: result.total_retrievals,
            graph_nodes: result.graph_nodes,
            graph_edges: result.graph_edges,
          }),
        };
      }

      case "verify_index": {
        interface IndexIntegrityReport {
          total_storage: number;
          total_indexed: number;
          orphaned_count: number;
          orphaned_ids: string[];
          is_healthy: boolean;
        }

        const result = await apiCall<IndexIntegrityReport>("/api/index/verify", "POST", {
          user_id: USER_ID,
        });

        const statusIcon = result.is_healthy ? "✓" : "⚠️";
        const healthText = result.is_healthy ? "HEALTHY" : "UNHEALTHY";

        let response = `🐘 Index Verification\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `${statusIcon} ${healthText}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Storage: ${result.total_storage} memories\n`;
        response += `Indexed: ${result.total_indexed} vectors\n`;
        response += `Orphaned: ${result.orphaned_count}\n`;

        if (result.orphaned_count > 0) {
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `⚠️ Run repair_index to fix orphaned memories`;
        }

        return {
          content: [{ type: "text", text: response }],
          structuredContent: {
            is_healthy: result.is_healthy,
            total_storage: result.total_storage,
            total_indexed: result.total_indexed,
            orphaned_count: result.orphaned_count,
            orphaned_ids: result.orphaned_ids || [],
          },
        };
      }

      case "repair_index": {
        interface RepairIndexResponse {
          success: boolean;
          total_storage: number;
          total_indexed: number;
          repaired: number;
          failed: number;
          is_healthy: boolean;
        }

        const result = await apiCall<RepairIndexResponse>("/api/index/repair", "POST", {
          user_id: USER_ID,
        });

        const statusIcon = result.is_healthy ? "✓" : "⚠️";
        const statusText = result.success ? "SUCCESS" : "PARTIAL";

        let response = `🐘 Index Repair\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `${statusIcon} ${statusText}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Storage: ${result.total_storage} memories\n`;
        response += `Indexed: ${result.total_indexed} vectors\n`;
        response += `Repaired: ${result.repaired}\n`;
        response += `Failed: ${result.failed}\n`;

        if (result.failed > 0) {
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `⚠️ ${result.failed} could not be repaired`;
        }

        return {
          content: [{ type: "text", text: response }],
        };
      }

      // =========================================================================
      // BACKUP & RESTORE TOOLS
      // =========================================================================

      case "backup_create": {
        interface BackupMetadata {
          backup_id: number;
          created_at: string;
          user_id: string;
          backup_type: string;
          size_bytes: number;
          checksum: string;
          memory_count: number;
          sequence_number: number;
          secondary_stores?: string[];
          secondary_size_bytes?: number;
        }

        interface BackupResponse {
          success: boolean;
          backup?: BackupMetadata;
          message: string;
        }

        const result = await apiCall<BackupResponse>("/api/backup/create", "POST", {
          user_id: USER_ID,
        });

        let response = `🐘 Backup Created\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        if (result.success && result.backup) {
          const b = result.backup;
          const sizeMB = (b.size_bytes / (1024 * 1024)).toFixed(2);
          response += `✓ Backup ID: ${b.backup_id}\n`;
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `Type: ${b.backup_type}\n`;
          response += `Memories: ${b.memory_count}\n`;
          response += `Size: ${sizeMB} MB\n`;
          if (b.secondary_stores && b.secondary_stores.length > 0) {
            const secSizeMB = ((b.secondary_size_bytes || 0) / (1024 * 1024)).toFixed(2);
            response += `Secondary stores: ${b.secondary_stores.length} (${secSizeMB} MB)\n`;
            response += `  Includes: ${b.secondary_stores.join(", ")}\n`;
          }
          response += `Checksum: ${b.checksum.slice(0, 16)}...\n`;
          response += `Created: ${new Date(b.created_at).toLocaleString()}\n`;
        } else {
          response += `✗ Failed: ${result.message || "Unknown backup creation error"}\n`;
        }

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "backup_list": {
        interface BackupMetadata {
          backup_id: number;
          created_at: string;
          user_id: string;
          backup_type: string;
          size_bytes: number;
          checksum: string;
          memory_count: number;
          sequence_number: number;
        }

        interface ListBackupsResponse {
          success: boolean;
          backups: BackupMetadata[];
          count: number;
        }

        const result = await apiCall<ListBackupsResponse>("/api/backups", "POST", {
          user_id: USER_ID,
        });

        let response = `🐘 Available Backups\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        if (result.backups.length === 0) {
          response += `No backups available.\n`;
          response += `Use backup_create to create your first backup.`;
        } else {
          response += `Found: ${result.count} backup(s)\n`;
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

          for (const b of result.backups) {
            const sizeMB = (b.size_bytes / (1024 * 1024)).toFixed(2);
            const date = new Date(b.created_at).toLocaleString();
            response += `📦 Backup #${b.backup_id}\n`;
            response += `   Type: ${b.backup_type} │ Memories: ${b.memory_count} │ Size: ${sizeMB} MB\n`;
            response += `   Created: ${date}\n\n`;
          }
        }

        return {
          content: [{ type: "text", text: response.trimEnd() }],
          structuredContent: {
            backups: (result.backups || []).map((b) =>
              compact({
                backup_id: b.backup_id,
                created_at: b.created_at,
                backup_type: b.backup_type,
                size_bytes: b.size_bytes,
                memory_count: b.memory_count,
                checksum: b.checksum,
                sequence_number: b.sequence_number,
              }),
            ),
            count: result.count ?? (result.backups || []).length,
          },
        };
      }

      case "backup_verify": {
        const { backup_id } = args as { backup_id: number };

        interface VerifyBackupResponse {
          success: boolean;
          is_valid: boolean;
          message: string;
        }

        const result = await apiCall<VerifyBackupResponse>("/api/backup/verify", "POST", {
          user_id: USER_ID,
          backup_id,
        });

        const statusIcon = result.is_valid ? "✓" : "✗";
        const statusText = result.is_valid ? "VALID" : "INVALID";

        let response = `🐘 Backup Verification\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `${statusIcon} Backup #${backup_id}: ${statusText}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += result.message || "No verification details provided";

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "backup_purge": {
        const { keep_count = 7 } = args as { keep_count?: number };

        interface PurgeBackupsResponse {
          success: boolean;
          purged_count: number;
        }

        const result = await apiCall<PurgeBackupsResponse>("/api/backups/purge", "POST", {
          user_id: USER_ID,
          keep_count,
        });

        let response = `🐘 Backup Purge\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        if (result.purged_count === 0) {
          response += `No backups purged (keeping ${keep_count}, none exceeded limit)`;
        } else {
          response += `✓ Purged ${result.purged_count} old backup(s)\n`;
          response += `Kept ${keep_count} most recent backup(s)`;
        }

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "backup_restore": {
        const { backup_id } = args as { backup_id: number };

        interface RestoreBackupResponse {
          success: boolean;
          message: string;
          restored_stores: string[];
          failed_stores?: string[];
        }

        const result = await apiCall<RestoreBackupResponse>("/api/backup/restore", "POST", {
          user_id: USER_ID,
          backup_id,
        });

        const failed = result.failed_stores ?? [];

        let response = `🔄 Backup Restore\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        if (result.success) {
          response += `✓ Backup #${backup_id} restored successfully\n`;
          if (result.restored_stores.length > 0) {
            response += `Restored stores: ${result.restored_stores.join(", ")}\n`;
          }
          response += `\n⚠️ ${result.message || "Restore completed with no additional details"}`;
        } else if (failed.length > 0) {
          // Partial restore: some stores came back, others were cleared and not
          // replaced. Reporting this as a plain success is how an operator ends
          // up believing data was recovered that was not.
          response += `⚠️ Backup #${backup_id} restored INCOMPLETELY\n`;
          if (result.restored_stores.length > 0) {
            response += `Restored stores: ${result.restored_stores.join(", ")}\n`;
          }
          response += `Failed stores: ${failed.join(", ")}\n`;
          response += `\n${result.message || "Some stores were not recovered"}`;
        } else {
          response += `✗ Restore failed: ${result.message || "Unknown restore error"}`;
        }

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "proactive_context": {
        const {
          context,
          semantic_threshold = 0.65,
          entity_match_weight = 0.4,
          recency_weight = 0.2,
          max_results = 5,
          memory_types = [],
          auto_ingest = true,
          tool_actions = [],
          full_content = false,
        } = args as {
          context: string;
          semantic_threshold?: number;
          entity_match_weight?: number;
          recency_weight?: number;
          max_results?: number;
          memory_types?: string[];
          auto_ingest?: boolean;
          tool_actions?: { tool_name: string; inputs?: Record<string, string>; success: boolean; output_snippet?: string; reward?: number }[];
          full_content?: boolean;
        };

        // --- Response types matching ProactiveContextResponse (Rust backend) ---

        interface ProactiveSurfacedMemory {
          id: string;
          content: string;
          memory_type: string;
          score: number;
          importance: number;
          created_at: string;
          tags: string[];
          relevance_reason: string;
          matched_entities: string[];
        }

        interface ReminderItem {
          id: string;
          content: string;
          trigger_type: string;
          status: string;
          due_at: string | null;
          created_at: string;
          triggered_at: string | null;
          dismissed_at: string | null;
          priority: number;
          tags: string[];
          overdue_seconds: number | null;
        }

        interface FeedbackProcessed {
          memories_evaluated: number;
          reinforced: string[];
          weakened: string[];
        }

        interface ProactiveTodoItem {
          id: string;
          short_id: string;
          content: string;
          status: string;
          priority: string;
          project: string | null;
          due_date: string | null;
          relevance_reason: string;
          similarity_score: number | null;
        }

        interface DetectedEntityInfo {
          name: string;
          entity_type: string;
        }

        interface ProactiveFact {
          id: string;
          fact: string;
          confidence: number;
          support_count: number;
          related_entities: string[];
        }

        interface ProactiveContextResponse {
          memories: ProactiveSurfacedMemory[];
          due_reminders: ReminderItem[];
          context_reminders: ReminderItem[];
          memory_count: number;
          reminder_count: number;
          ingested_memory_id: string | null;
          feedback_processed: FeedbackProcessed | null;
          relevant_todos: ProactiveTodoItem[];
          todo_count: number;
          relevant_facts: ProactiveFact[];
          latency_ms: number;
          detected_entities: DetectedEntityInfo[];
          temporal_credits_applied: number | null;
        }

        // Clean system scaffolding from context — AI clients pass full conversation
        // including <task-notification> XML which overwhelms BM25 and embedding.
        const cleanedContext = stripSystemNoise(context).slice(0, MAX_CONTEXT_LENGTH);
        if (cleanedContext.length < PROACTIVE_MIN_CONTEXT_LENGTH) {
          // Returns before the backend call, so nothing was surfaced and
          // nothing was ingested regardless of the auto_ingest argument.
          return {
            content: [{ type: "text", text: "No relevant memories surfaced (context too short after cleaning).\n\n[Latency: 0.0ms]" }],
            structuredContent: {
              memories: [],
              detected_entities: [],
              todos: [],
              facts: [],
              count: 0,
              // Returns before the backend call, so nothing could be ingested
              // whatever the argument said.
              ingest_requested: false,
            },
          };
        }

        // Guard: if another proactive_context call is in-flight, skip feedback
        // to avoid corrupted state from concurrent updates.
        const skipFeedback = proactiveCallInFlight;
        proactiveCallInFlight = true;

        // Single API call to the full proactive context pipeline:
        // feedback loop, coactivation, segmented ingest, semantic todos, context reminders.
        // Wrapped so a thrown apiCall resets the in-flight guard — otherwise the
        // flag would stay true forever and permanently disable the feedback loop.
        let result: ProactiveContextResponse;
        try {
          result = await apiCall<ProactiveContextResponse>("/api/proactive_context", "POST", {
            user_id: USER_ID,
            context: cleanedContext,
            max_results,
            semantic_threshold,
            entity_match_weight,
            recency_weight,
            memory_types,
            auto_ingest,
            // Implicit feedback: send previous response so backend can evaluate which memories helped.
            // Skipped if another proactive_context call was in-flight (prevents corrupted feedback).
            previous_response: skipFeedback ? undefined : (lastProactiveResponse || undefined),
            // user_followup means "the user's message AFTER the agent response"
            // (src/memory/feedback.rs feeds it to detect_negative_keywords
            // against the pending surfaced set). The pending set was created on
            // the PREVIOUS call, alongside lastProactiveResponse; the user's
            // reaction to that response is THIS message. This previously sent
            // the message from BEFORE the response — the original ask — so the
            // negative-keyword scan ran against the question instead of the
            // reaction, and corrections like "no, that's wrong" never
            // registered as negative feedback.
            user_followup: (skipFeedback || !lastProactiveResponse) ? undefined : (cleanedContext || undefined),
            // Tool-aware feedback attribution: causal signal from tool/actuator actions
            ...(tool_actions.length > 0 ? { tool_actions } : {}),
          });
        } catch (e) {
          proactiveCallInFlight = false;
          throw e;
        }

        const memories = result.memories || [];
        const entities = result.detected_entities || [];

        const facts = result.relevant_facts || [];

        // Shared by the "nothing surfaced" and the fully-populated returns.
        //
        // `ingest_requested` echoes the argument rather than claiming an
        // outcome. The backend spawns the ingest in a background task and waits
        // only 50ms to read the id back (handlers/recall.rs), so
        // `ingested_memory_id` is frequently absent for contexts that WERE
        // stored — deriving a boolean "auto_ingested" from it would report
        // false while a memory was being written.
        const proactiveStructured = (): Record<string, unknown> => ({
          memories: memories.map((m) => {
            const body = m.content ?? "";
            const truncated = !full_content && body.length > MEMORY_PREVIEW_MAX;
            return compact({
              id: m.id,
              content: truncated ? body.slice(0, MEMORY_PREVIEW_MAX) : body,
              content_truncated: truncated,
              memory_type: m.memory_type,
              score: m.score,
              importance: m.importance,
              tags: m.tags,
              relevance_reason: m.relevance_reason,
              matched_entities: m.matched_entities,
              created_at: m.created_at,
            });
          }),
          detected_entities: entities.map((e) => compact({ name: e.name, entity_type: e.entity_type })),
          todos: (result.relevant_todos || []).map((t) =>
            compact({
              id: t.id,
              short_id: t.short_id,
              content: t.content,
              status: t.status,
              priority: t.priority,
              project: t.project ?? undefined,
              score: t.similarity_score ?? undefined,
            }),
          ),
          facts: facts.map((f) =>
            compact({ id: f.id, fact: f.fact, confidence: f.confidence, support_count: f.support_count }),
          ),
          count: memories.length,
          ingest_requested: auto_ingest,
          ...(result.ingested_memory_id ? { ingested_memory_id: result.ingested_memory_id } : {}),
          ...(result.latency_ms !== undefined ? { latency_ms: result.latency_ms } : {}),
        });

        if (memories.length === 0 && result.reminder_count === 0 && result.todo_count === 0 && facts.length === 0) {
          const entityList = entities.length > 0
            ? `\n\nDetected entities: ${entities.map(e => `"${e.name}" (${e.entity_type})`).join(', ')}`
            : '';
          const feedbackNote = result.feedback_processed
            ? `\n[Feedback: ${result.feedback_processed.memories_evaluated} evaluated, ${result.feedback_processed.reinforced.length} reinforced, ${result.feedback_processed.weakened.length} weakened]`
            : '';
          const temporalNote = result.temporal_credits_applied
            ? `\n[Temporal credits: ${result.temporal_credits_applied} multi-turn signals applied]`
            : '';

          const emptyText = `No relevant memories surfaced for this context.${entityList}${feedbackNote}${temporalNote}\n\n[Latency: ${(result.latency_ms ?? 0).toFixed(1)}ms]`;
          lastProactiveResponse = emptyText;
          proactiveCallInFlight = false;

          return {
            content: [{ type: "text", text: emptyText }],
            structuredContent: proactiveStructured(),
          };
        }

        // Format detected entities summary
        const entitySummary = entities.length > 0
          ? `\n\nDetected entities: ${entities.map(e => `"${e.name}" (${e.entity_type})`).join(', ')}`
          : '';

        // Format reminders from unified response (due + context-triggered)
        let reminderBlock = "";
        {
          const allReminders = [...(result.due_reminders || []), ...(result.context_reminders || [])];
          const uniqueReminders = allReminders.filter((r, i, arr) =>
            arr.findIndex(x => x.id === r.id) === i
          );

          if (uniqueReminders.length > 0) {
            reminderBlock = `\n\n`;
            reminderBlock += `🐘━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🐘\n`;
            reminderBlock += `┃  SHODH MEMORY                    REMINDERS (${String(uniqueReminders.length).padStart(2)})  ┃\n`;
            reminderBlock += `┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n`;

            for (const r of uniqueReminders) {
              const icon = r.overdue_seconds && r.overdue_seconds > 0 ? "⏰" : "📌";
              const contentText = r.content.slice(0, 38);
              reminderBlock += `┃  ${icon} ${contentText.padEnd(44)} [${r.id}] ┃\n`;

              if (r.overdue_seconds && r.overdue_seconds > 0) {
                const mins = Math.round(r.overdue_seconds / 60);
                const overdueText = mins > 60
                  ? `⚠️  OVERDUE by ${Math.round(mins/60)}h ${mins % 60}m`
                  : `⚠️  OVERDUE by ${mins}m`;
                reminderBlock += `┃     ${overdueText.padEnd(47)} ┃\n`;
              } else if (r.due_at) {
                const dueText = `Due: ${new Date(r.due_at).toLocaleString()}`;
                reminderBlock += `┃     ${dueText.padEnd(47)} ┃\n`;
              }
            }

            reminderBlock += `┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n`;
            reminderBlock += `\n💡 Use dismiss_reminder with the [id] shown above`;
          }
        }

        // Format todos from unified response (semantic + in_progress)
        let todoBlock = "";
        {
          const todos = result.relevant_todos || [];
          if (todos.length > 0) {
            todoBlock = "\n\n📋 Relevant Todos:\n";
            for (const t of todos) {
              const statusIcon = t.status === "in_progress" ? "🔄" : t.status === "blocked" ? "🚫" : "☐";
              const proj = t.project ? ` [${t.project}]` : "";
              const due = t.due_date ? ` (due: ${t.due_date})` : "";
              todoBlock += `  ${statusIcon} ${t.priority} ${t.short_id}: ${renderContent(t.content, undefined, 60, full_content)}${proj}${due}\n`;
              todoBlock += `     ${t.relevance_reason}\n`;
            }
          }
        }

        // Format consolidated facts from knowledge graph
        let factsBlock = "";
        {
          const facts = (result.relevant_facts || [])
            .filter((f: ProactiveFact) => f.confidence >= 0.4);
          if (facts.length > 0) {
            factsBlock = "\n\n🐘 Known Facts:\n";
            for (const f of facts) {
              const conf = (f.confidence * 100).toFixed(0);
              const entities = f.related_entities.length > 0 ? ` [${f.related_entities.slice(0, 3).join(', ')}]` : '';
              const factText = f.fact.length > 120 ? f.fact.slice(0, 120) + '...' : f.fact;
              factsBlock += `  • (${conf}%) ${factText}${entities}\n`;
            }
          }
        }

        // Add temporal framing - helps AI reason about time
        const now = new Date();
        const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
        const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const temporalHeader = `📅 ${dayNames[now.getDay()]}, ${monthNames[now.getMonth()]} ${now.getDate()}, ${now.getFullYear()} at ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}\n\n`;

        // Format memories with relative timestamps for temporal reasoning
        const formattedWithTime = memories
          .map((m, i) => {
            const score = (m.score * 100).toFixed(0);
            const entityMatchStr = (m.matched_entities && m.matched_entities.length > 0)
              ? `\n   Matched: ${m.matched_entities.join(', ')}`
              : '';
            const tagsStr = (m.tags && m.tags.length > 0)
              ? `\n   Tags: ${m.tags.slice(0, 5).join(', ')}`
              : '';

            // Calculate relative time
            let timeStr = '';
            if (m.created_at) {
              const d = new Date(m.created_at);
              const diffMs = now.getTime() - d.getTime();
              const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
              if (diffDays === 0) {
                timeStr = ` (today at ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })})`;
              } else if (diffDays === 1) {
                timeStr = ` (yesterday)`;
              } else if (diffDays < 7) {
                timeStr = ` (${diffDays}d ago)`;
              } else {
                timeStr = ` (${d.toLocaleDateString([], { month: 'short', day: 'numeric' })})`;
              }
            }

            const importanceBar = m.importance >= 0.8 ? '🔴' : m.importance >= 0.5 ? '🟡' : '⚪';
            const preview = renderContent(m.content, m.id, MEMORY_PREVIEW_MAX, full_content);
            return `${i + 1}. ${importanceBar} [${score}%]${timeStr} ${preview}\n   ${m.memory_type}${m.tier ? ` | ${m.tier}` : ''} | ${m.relevance_reason}${entityMatchStr}${tagsStr}`;
          })
          .join("\n\n");

        // Feedback loop status
        const feedbackNote = result.feedback_processed
          ? `\n[Feedback loop: ${result.feedback_processed.memories_evaluated} evaluated, ${result.feedback_processed.reinforced.length} reinforced, ${result.feedback_processed.weakened.length} weakened]`
          : '';
        const temporalNote = result.temporal_credits_applied
          ? `\n[Temporal credits: ${result.temporal_credits_applied} multi-turn signals applied]`
          : '';

        // Ingestion confirmation
        const ingestNote = result.ingested_memory_id
          ? `\n[Context ingested: ${result.ingested_memory_id}]`
          : '';

        // Summary counts
        const summaryParts: string[] = [];
        if (memories.length > 0) summaryParts.push(`${memories.length} memories`);
        if (facts.length > 0) summaryParts.push(`${facts.length} facts`);
        if (result.todo_count > 0) summaryParts.push(`${result.todo_count} todos`);
        if (result.reminder_count > 0) summaryParts.push(`${result.reminder_count} reminders`);
        const summary = summaryParts.length > 0 ? `Surfaced ${summaryParts.join(', ')}` : 'No relevant context found';

        const responseText = `${temporalHeader}${summary}:\n\n${formattedWithTime}${entitySummary}${factsBlock}${reminderBlock}${todoBlock}${feedbackNote}${temporalNote}${ingestNote}\n\n[Latency: ${(result.latency_ms ?? 0).toFixed(1)}ms | Threshold: ${(semantic_threshold * 100).toFixed(0)}%]`;

        // Store clean semantic content for implicit feedback on next call.
        // Strip display formatting (emoji borders, latency markers, entity summaries)
        // that would add embedding noise and dilute the semantic signal.
        const cleanContent = memories
          .map((m: { content?: string }) => m.content || "")
          .filter((c: string) => c.length > 0)
          .join("\n");
        lastProactiveResponse = cleanContent || responseText;
        proactiveCallInFlight = false;

        return {
          content: [{ type: "text", text: responseText }],
          structuredContent: proactiveStructured(),
        };
      }


      case "token_status": {
        const status = getTokenStatus();
        const sessionDuration = Math.round((Date.now() - sessionStartTime) / 1000 / 60);
        const remaining = status.budget - status.tokens;
        const percentUsed = Math.round(status.percent * 100);

        // Visual progress bar
        const barLength = 20;
        const filledLength = Math.round(percentUsed / 100 * barLength);
        const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);

        let response = `🐘 MCP Pipeline Throughput\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `${bar} ${percentUsed}%\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Pipeline tokens: ${status.tokens.toLocaleString()}\n`;
        response += `Budget: ${status.budget.toLocaleString()}\n`;
        response += `Session: ${sessionDuration} min\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Note: Tracks memory tool I/O only, not AI context window.`;

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "reset_token_session": {
        const previousTokens = sessionTokens;
        resetTokenSession();

        // Signal context compression to backend session tracker
        apiCall("/api/sessions/context-compressed", "POST", {
          user_id: USER_ID,
          tokens_before: previousTokens,
          tokens_after: 0,
        }).catch(() => {});

        let response = `🐘 Token Session Reset\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Previous: ${previousTokens.toLocaleString()} tokens\n`;
        response += `Current: 0 tokens\n`;
        response += `Budget: ${TOKEN_BUDGET.toLocaleString()} tokens\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `✓ Counter cleared`;

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "session_digest": {
        // Fetch backend session digest
        const digestResult = await apiCall("/api/sessions/digest", "POST", {
          user_id: USER_ID,
          token_budget: TOKEN_BUDGET,
        });

        const now = new Date();
        const sessionStart = new Date(sessionStartTime);
        const durationMins = Math.round((now.getTime() - sessionStartTime) / 60000);

        let digestResponse = `📊 Session Digest\n`;
        digestResponse += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        digestResponse += `📅 ${now.toLocaleDateString()} | ⏱ ${sessionStart.toLocaleTimeString()} → ${now.toLocaleTimeString()} (${durationMins}m)\n\n`;

        // Token usage
        const tokenStatus = getTokenStatus();
        const pct = Math.round(tokenStatus.percent * 100);
        const barLen = 20;
        const filled = Math.round(pct / 100 * barLen);
        const bar = "█".repeat(filled) + "░".repeat(barLen - filled);
        digestResponse += `🔤 Tokens: ${tokenStatus.tokens.toLocaleString()} / ${tokenStatus.budget.toLocaleString()} (${pct}%)\n`;
        digestResponse += `   [${bar}]\n\n`;

        // Backend session data (memories, entities, etc.)
        if (digestResult && typeof digestResult === "object" && "digest" in digestResult && digestResult.digest) {
          const d = digestResult.digest as Record<string, unknown>;
          digestResponse += `🐘 Memory: ${d.memories_created ?? 0} created, ${d.memories_surfaced ?? 0} surfaced, ${d.memories_used ?? 0} used`;
          const hitRate = d.memory_hit_rate as number;
          if (hitRate > 0) digestResponse += ` (${Math.round(hitRate * 100)}% hit rate)`;
          digestResponse += `\n`;

          if ((d.todos_created as number) > 0 || (d.todos_completed as number) > 0) {
            digestResponse += `📋 Todos: ${d.todos_created ?? 0} created, ${d.todos_completed ?? 0} completed\n`;
          }
          if ((d.entity_count as number) > 0) {
            const entities = d.entities_extracted as string[];
            const preview = entities.slice(0, 8).join(", ");
            const more = entities.length > 8 ? ` +${entities.length - 8} more` : "";
            digestResponse += `🏷️ Entities: ${d.entity_count} extracted (${preview}${more})\n`;
          }
          if ((d.topic_changes as number) > 0) {
            digestResponse += `🔀 Topic changes: ${d.topic_changes}\n`;
          }
          if ((d.compressions as number) > 0) {
            digestResponse += `⟳ Context compressions: ${d.compressions}\n`;
          }
          if ((d.consolidation_events as number) > 0) {
            digestResponse += `⚙️ Consolidation events: ${d.consolidation_events}\n`;
          }
        }

        // MCP-side tool call counts
        if (toolCallCounts.size > 0) {
          digestResponse += `\n🔧 Tools Used:\n`;
          const sorted = [...toolCallCounts.entries()].sort((a, b) => b[1] - a[1]);
          for (const [tool, count] of sorted) {
            digestResponse += `   ${tool}: ${count}\n`;
          }
        }

        digestResponse += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;

        return {
          content: [{ type: "text", text: digestResponse }],
        };
      }

      case "session_history": {
        const { limit: histLimit = 10, group_by_project = false } = args as {
          limit?: number;
          group_by_project?: boolean;
        };

        const histResult = await apiCall("/api/sessions/history", "POST", {
          user_id: USER_ID,
          limit: histLimit,
          group_by_project,
        }) as {
          success?: boolean;
          sessions?: Array<{
            session_id?: string;
            content: string;
            entities: string[];
            started_at?: string;
            duration_secs?: number;
            memories_created?: number;
            created_at: string;
          }>;
          project_threads?: Array<{
            name: string;
            sessions: number[];
            shared_entities: string[];
            session_count: number;
          }>;
          total?: number;
        } | null;

        if (!histResult?.success || !histResult?.sessions?.length) {
          return {
            content: [{ type: "text", text: "No session history found. Sessions are recorded when Claude Code exits." }],
          };
        }

        let histResponse = `Session History (${histResult.total ?? histResult.sessions.length} sessions)\n`;
        histResponse += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        for (let i = 0; i < histResult.sessions.length; i++) {
          const s = histResult.sessions[i];
          const created = new Date(s.created_at);
          const dateStr = created.toLocaleDateString([], { month: "short", day: "numeric" });
          const timeStr = created.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
          const durMin = s.duration_secs ? Math.round(s.duration_secs / 60) : null;
          const durStr = durMin ? ` (${durMin}m)` : "";

          histResponse += `\n${i + 1}. ${dateStr}, ${timeStr}${durStr}\n`;

          if (s.entities.length > 0) {
            const preview = s.entities.slice(0, 10).join(", ");
            const more = s.entities.length > 10 ? ` +${s.entities.length - 10} more` : "";
            histResponse += `   Entities: ${preview}${more}\n`;
          }

          if (s.memories_created != null) {
            histResponse += `   Memories created: ${s.memories_created}\n`;
          }

          // Show content (first 200 chars if long)
          const contentPreview = s.content.length > 200 ? s.content.slice(0, 200) + "..." : s.content;
          histResponse += `   ${contentPreview}\n`;
        }

        // Project threads
        if (histResult.project_threads && histResult.project_threads.length > 0) {
          histResponse += `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          histResponse += `Active Projects (cross-session continuity):\n`;
          for (const thread of histResult.project_threads) {
            histResponse += `  • ${thread.name} — ${thread.session_count} sessions\n`;
            if (thread.shared_entities.length > 0) {
              histResponse += `    Shared: ${thread.shared_entities.slice(0, 8).join(", ")}\n`;
            }
          }
        }

        return {
          content: [{ type: "text", text: histResponse }],
        };
      }

      case "fact_narratives": {
        const { limit: narLimit = 20, entity_filter } = args as {
          limit?: number;
          entity_filter?: string;
        };

        const narResult = await apiCall("/api/facts/narratives", "POST", {
          user_id: USER_ID,
          limit: narLimit,
          entity_filter: entity_filter || null,
        }) as {
          success?: boolean;
          clusters?: Array<{
            topic: string;
            entities: string[];
            facts: Array<{ id: string; fact: string; confidence: number; support_count: number }>;
            narrative: string;
            avg_confidence: number;
            total_support: number;
            causal_chains: Array<{
              from_fact: string;
              to_fact: string;
              relation: string;
            }>;
          }>;
          total_facts?: number;
          total_clusters?: number;
        } | null;

        if (!narResult?.success || !narResult?.clusters?.length) {
          return {
            content: [{ type: "text", text: "No fact narratives found. Facts are extracted during memory consolidation." }],
          };
        }

        let narResponse = `Fact Narratives (${narResult.total_clusters ?? narResult.clusters.length} topics, ${narResult.total_facts ?? 0} facts)\n`;
        narResponse += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        for (const cluster of narResult.clusters) {
          narResponse += `\n${cluster.narrative || `Regarding ${cluster.topic}:`}\n`;
          narResponse += `  Confidence: ${Math.round((cluster.avg_confidence ?? 0) * 100)}% avg | Support: ${cluster.total_support ?? 0} memories\n`;

          if (cluster.causal_chains?.length > 0) {
            narResponse += `  Causal chains:\n`;
            for (const chain of cluster.causal_chains) {
              const arrow = chain.relation === "superseded_by" ? " ✕→ "
                : chain.relation === "resolved_by" ? " ✓→ "
                : " → ";
              const fromPreview = chain.from_fact.length > 60 ? chain.from_fact.slice(0, 60) + "..." : chain.from_fact;
              const toPreview = chain.to_fact.length > 60 ? chain.to_fact.slice(0, 60) + "..." : chain.to_fact;
              narResponse += `    ${fromPreview}${arrow}${toPreview}\n`;
            }
          }
        }

        return {
          content: [{ type: "text", text: narResponse }],
        };
      }

      case "purge_facts": {
        const { pattern, dry_run = false } = args as {
          pattern: string;
          dry_run?: boolean;
        };

        if (!pattern || pattern.length < 3) {
          return {
            content: [{ type: "text", text: "Pattern must be at least 3 characters." }],
          };
        }

        const purgeResult = await apiCall("/api/facts/purge", "POST", {
          user_id: USER_ID,
          pattern,
          dry_run,
        }) as {
          success?: boolean;
          deleted?: number;
          total_scanned?: number;
          dry_run?: boolean;
        } | null;

        if (!purgeResult?.success) {
          return {
            content: [{ type: "text", text: "Failed to purge facts. Server may be unavailable." }],
          };
        }

        const mode = purgeResult.dry_run ? "DRY RUN" : "PURGED";
        return {
          content: [{
            type: "text",
            text: `${mode}: ${purgeResult.deleted} of ${purgeResult.total_scanned} facts match "${pattern}"${purgeResult.dry_run ? "\nRe-run with dry_run=false to delete." : ""}`,
          }],
        };
      }

      case "consolidation_report": {
        const { since, until } = args as { since?: string; until?: string };

        interface ConsolidationStats {
          total_memories: number;
          memories_strengthened: number;
          memories_decayed: number;
          memories_at_risk: number;
          edges_formed: number;
          edges_strengthened: number;
          edges_potentiated: number;
          edges_pruned: number;
          facts_extracted: number;
          facts_reinforced: number;
          maintenance_cycles: number;
          total_maintenance_duration_ms: number;
        }

        interface MemoryChange {
          memory_id: string;
          content_preview: string;
          activation_before: number;
          activation_after: number;
          change_reason: string;
          at_risk: boolean;
          timestamp: string;
        }

        interface AssociationChange {
          from_memory_id: string;
          to_memory_id: string;
          strength_before: number | null;
          strength_after: number;
          co_activations: number | null;
          reason: string;
          timestamp: string;
        }

        interface ConsolidationReport {
          period: {
            start: string;
            end: string;
          };
          strengthened_memories: MemoryChange[];
          decayed_memories: MemoryChange[];
          formed_associations: AssociationChange[];
          strengthened_associations: AssociationChange[];
          potentiated_associations: AssociationChange[];
          pruned_associations: AssociationChange[];
          extracted_facts: unknown[];
          reinforced_facts: unknown[];
          statistics: ConsolidationStats;
        }

        const result = await apiCall<ConsolidationReport>("/api/consolidation/report", "POST", {
          user_id: USER_ID,
          since,
          until,
        });

        const stats = result.statistics;

        // Calculate event count
        const eventCount =
          result.strengthened_memories.length +
          result.decayed_memories.length +
          result.formed_associations.length +
          result.strengthened_associations.length +
          result.potentiated_associations.length +
          result.pruned_associations.length +
          result.extracted_facts.length +
          result.reinforced_facts.length;

        // Format dates
        const startDate = new Date(result.period.start).toLocaleString();
        const endDate = new Date(result.period.end).toLocaleString();

        let response = `🐘 Consolidation Report\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Period: ${startDate} → ${endDate}\n`;
        response += `Events: ${eventCount} │ Memories: ${stats.total_memories}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        // Memory changes
        if (stats.memories_strengthened > 0 || stats.memories_decayed > 0 || stats.memories_at_risk > 0) {
          response += `🐘 MEMORY DYNAMICS\n`;
          if (stats.memories_strengthened > 0) response += `   ↑ ${stats.memories_strengthened} strengthened\n`;
          if (stats.memories_decayed > 0) response += `   ↓ ${stats.memories_decayed} decayed\n`;
          if (stats.memories_at_risk > 0) response += `   ⚠️ ${stats.memories_at_risk} at risk\n`;
          response += `\n`;
        }

        // Edge changes (associations)
        if (stats.edges_formed > 0 || stats.edges_strengthened > 0 || stats.edges_potentiated > 0 || stats.edges_pruned > 0) {
          response += `🔗 ASSOCIATIONS (Hebbian)\n`;
          if (stats.edges_formed > 0) response += `   + ${stats.edges_formed} formed\n`;
          if (stats.edges_strengthened > 0) response += `   ↑ ${stats.edges_strengthened} strengthened\n`;
          if (stats.edges_potentiated > 0) response += `   ★ ${stats.edges_potentiated} permanent (LTP)\n`;
          if (stats.edges_pruned > 0) response += `   ✂ ${stats.edges_pruned} pruned\n`;
          response += `\n`;
        }

        // Fact changes
        if (stats.facts_extracted > 0 || stats.facts_reinforced > 0) {
          response += `📚 FACTS\n`;
          if (stats.facts_extracted > 0) response += `   + ${stats.facts_extracted} extracted\n`;
          if (stats.facts_reinforced > 0) response += `   ↑ ${stats.facts_reinforced} reinforced\n`;
          response += `\n`;
        }

        // Maintenance cycles
        if (stats.maintenance_cycles > 0) {
          const durationSec = (stats.total_maintenance_duration_ms / 1000).toFixed(2);
          response += `⚙️ MAINTENANCE: ${stats.maintenance_cycles} cycles (${durationSec}s)\n`;
        }

        // No activity message
        if (eventCount === 0) {
          response += `ℹ️ No consolidation activity in this period.\n`;
          response += `   Store and access memories to trigger learning.`;
        }

        return {
          content: [{ type: "text", text: response.trimEnd() }],
        };
      }

      // =================================================================
      // Prospective Memory / Reminders (SHO-116)
      // =================================================================

      case "set_reminder": {
        const { content, trigger_type, trigger_at, after_seconds, keywords, priority = 3, tags = [], threshold } = args as {
          content: string;
          trigger_type: "time" | "duration" | "context";
          trigger_at?: string;
          after_seconds?: number;
          keywords?: string[];
          priority?: number;
          tags?: string[];
          threshold?: number;
        };

        if (!content || content.length === 0) {
          return { content: [{ type: "text", text: "Error: 'content' is required and cannot be empty" }], isError: true };
        }
        if (content.length > MAX_CONTENT_LENGTH) {
          return { content: [{ type: "text", text: `Error: 'content' exceeds maximum length of ${MAX_CONTENT_LENGTH} characters` }], isError: true };
        }
        if (priority < 1 || priority > 5 || !Number.isFinite(priority)) {
          return { content: [{ type: "text", text: "Error: 'priority' must be between 1 and 5" }], isError: true };
        }

        // Build trigger object based on type
        let trigger: Record<string, unknown>;
        switch (trigger_type) {
          case "time":
            if (!trigger_at) {
              return {
                content: [{ type: "text", text: "Error: 'trigger_at' is required for time-based reminders" }],
                isError: true,
              };
            }
            trigger = { type: "time", at: trigger_at };
            break;
          case "duration":
            if (!after_seconds || after_seconds <= 0) {
              return {
                content: [{ type: "text", text: "Error: 'after_seconds' must be positive for duration-based reminders" }],
                isError: true,
              };
            }
            trigger = { type: "duration", after_seconds };
            break;
          case "context":
            if (!keywords || keywords.length === 0) {
              return {
                content: [{ type: "text", text: "Error: 'keywords' is required for context-based reminders" }],
                isError: true,
              };
            }
            const ctxThreshold = (threshold !== undefined && threshold >= 0.0 && threshold <= 1.0) ? threshold : 0.7;
            trigger = { type: "context", keywords, threshold: ctxThreshold };
            break;
          default:
            return {
              content: [{ type: "text", text: `Error: Invalid trigger_type: ${trigger_type}` }],
              isError: true,
            };
        }

        interface ReminderResponse {
          id: string;
          content: string;
          trigger_type: string;
          due_at: string | null;
          created_at: string;
        }

        const result = await apiCall<ReminderResponse>("/api/remind", "POST", {
          user_id: USER_ID,
          content,
          trigger,
          priority,
          tags,
        });

        let response = `🐘 Reminder Set\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `ID: ${result.id}\n`;
        response += `Content: ${content}\n`;
        response += `Trigger: ${trigger_type}`;
        if (trigger_type === "time" && result.due_at) {
          response += ` (${new Date(result.due_at).toLocaleString()})`;
        } else if (trigger_type === "duration" && after_seconds) {
          const mins = Math.round(after_seconds / 60);
          response += ` (in ${mins > 60 ? Math.round(mins/60) + 'h' : mins + 'm'})`;
        } else if (trigger_type === "context" && keywords) {
          response += ` (keywords: ${keywords.join(", ")})`;
        }
        response += `\n`;
        if (priority !== 3) {
          response += `Priority: ${'★'.repeat(priority)}${'☆'.repeat(5-priority)}\n`;
        }

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "list_reminders": {
        const { status = "pending" } = args as { status?: string };

        interface ReminderItem {
          id: string;
          content: string;
          trigger_type: string;
          status: string;
          due_at: string | null;
          created_at: string;
          priority: number;
          overdue_seconds: number | null;
        }

        interface ListRemindersResponse {
          reminders: ReminderItem[];
          count: number;
        }

        const result = await apiCall<ListRemindersResponse>("/api/reminders", "POST", {
          user_id: USER_ID,
          status: status === "all" ? null : status,
        });

        const remindersStructured = (): Record<string, unknown> => ({
          status_filter: status,
          reminders: (result.reminders || []).map((r) =>
            compact({
              id: r.id,
              content: r.content,
              status: r.status,
              trigger_type: r.trigger_type,
              due_at: r.due_at ?? undefined,
              created_at: r.created_at,
              priority: r.priority,
              overdue_seconds: r.overdue_seconds ?? undefined,
            }),
          ),
          count: result.count ?? (result.reminders || []).length,
        });

        if (result.count === 0) {
          return {
            content: [{ type: "text", text: `No ${status === "all" ? "" : status + " "}reminders found.` }],
            structuredContent: remindersStructured(),
          };
        }

        let response = `🐘 SHODH REMINDERS (${result.count})\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        for (const r of result.reminders) {
          const icon = r.overdue_seconds && r.overdue_seconds > 0 ? "⏰" : "📌";
          const statusBadge = r.status === "triggered" ? " [TRIGGERED]" : "";
          response += `${icon} ${r.content.slice(0, 50)}${r.content.length > 50 ? "..." : ""}${statusBadge}\n`;
          response += `   Type: ${r.trigger_type} | Priority: ${'★'.repeat(r.priority)} | ID: ${r.id}\n`;
          if (r.due_at) {
            response += `   Due: ${new Date(r.due_at).toLocaleString()}\n`;
          }
          if (r.overdue_seconds && r.overdue_seconds > 0) {
            const mins = Math.round(r.overdue_seconds / 60);
            response += `   ⚠️ Overdue by ${mins > 60 ? Math.round(mins/60) + 'h' : mins + 'm'}\n`;
          }
          response += `\n`;
        }

        return {
          content: [{ type: "text", text: response }],
          structuredContent: remindersStructured(),
        };
      }

      case "dismiss_reminder": {
        const { reminder_id } = args as { reminder_id: string };

        interface ActionResponse {
          success: boolean;
          message: string;
        }

        const result = await apiCall<ActionResponse>(`/api/reminders/${encodeURIComponent(reminder_id)}/dismiss`, "POST", {
          user_id: USER_ID,
        });

        return {
          content: [
            {
              type: "text",
              text: result.success
                ? `✓ Reminder dismissed: ${reminder_id}`
                : `⚠️ ${result.message || "No message returned"}`,
            },
          ],
        };
      }

      // =================================================================
      // GTD Todo List Handlers
      // =================================================================

      case "add_todo": {
        const {
          content: todoContent,
          status = "todo",
          priority = "medium",
          project,
          contexts = [],
          due_date,
          tags = [],
          blocked_on,
          notes,
          recurrence,
          parent_id,
          blocked_by,
          related_memory_ids,
        } = args as {
          content: string;
          status?: string;
          priority?: string;
          project?: string;
          contexts?: string[];
          due_date?: string;
          tags?: string[];
          blocked_on?: string;
          notes?: string;
          recurrence?: string;
          parent_id?: string;
          blocked_by?: string[];
          related_memory_ids?: string[];
        };

        if (!todoContent || todoContent.length === 0) {
          return { content: [{ type: "text", text: "Error: 'content' is required and cannot be empty" }], isError: true };
        }
        if (todoContent.length > MAX_CONTENT_LENGTH) {
          return { content: [{ type: "text", text: `Error: 'content' exceeds maximum length of ${MAX_CONTENT_LENGTH} characters` }], isError: true };
        }

        interface TodoResponse {
          success: boolean;
          todo: {
            id: string;
            content: string;
            status: string;
            priority: string;
            project_id?: string;
            due_date?: string;
          };
          formatted: string;
        }

        const result = await apiCall<TodoResponse>("/api/todos/add", "POST", {
          user_id: USER_ID,
          content: todoContent,
          status,
          priority,
          project,
          contexts,
          due_date,
          tags,
          blocked_on,
          notes,
          recurrence,
          parent_id,
          blocked_by,
          related_memory_ids,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
          structuredContent: structuredTodo(result.todo as TodoWire),
        };
      }

      case "list_todos": {
        const {
          query,
          status: statusFilter,
          project,
          context,
          priority,
          due,
          limit = 50,
          offset = 0,
        } = args as {
          query?: string;
          status?: string[];
          project?: string;
          context?: string;
          priority?: string;
          due?: string;
          limit?: number;
          offset?: number;
        };

        const clampedLimit = Math.max(1, Math.min(Math.floor(limit), MAX_LIMIT));
        const clampedOffset = Math.max(0, Math.floor(offset));

        interface ListTodosResponse {
          success: boolean;
          todos: TodoWire[];
          projects: unknown[];
          formatted: string;
          count: number;
        }

        const result = await apiCall<ListTodosResponse>("/api/todos/list", "POST", {
          user_id: USER_ID,
          query,
          status: statusFilter,
          project,
          context,
          priority,
          due,
          limit: clampedLimit,
          offset: clampedOffset,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
          structuredContent: {
            todos: (result.todos || []).map(structuredTodo),
            count: (result.todos || []).length,
            ...(result.count !== undefined ? { total: result.count } : {}),
          },
        };
      }

      case "update_todo": {
        const {
          todo_id,
          content: newContent,
          status,
          priority,
          project,
          contexts,
          due_date,
          blocked_on,
          notes,
          tags,
          parent_id,
          blocked_by,
          related_memory_ids,
        } = args as {
          todo_id: string;
          content?: string;
          status?: string;
          priority?: string;
          project?: string;
          contexts?: string[];
          due_date?: string;
          blocked_on?: string;
          notes?: string;
          tags?: string[];
          parent_id?: string;
          blocked_by?: string[];
          related_memory_ids?: string[];
        };

        interface UpdateTodoResponse {
          success: boolean;
          todo: unknown;
          formatted: string;
        }

        const result = await apiCall<UpdateTodoResponse>(`/api/todos/${encodeURIComponent(todo_id)}/update`, "POST", {
          user_id: USER_ID,
          content: newContent,
          status,
          priority,
          project,
          contexts,
          due_date,
          blocked_on,
          notes,
          tags,
          parent_id,
          blocked_by,
          related_memory_ids,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "complete_todo": {
        const { todo_id } = args as { todo_id: string };

        interface CompleteTodoResponse {
          success: boolean;
          todo: unknown;
          next_recurrence?: unknown;
          formatted: string;
        }

        const result = await apiCall<CompleteTodoResponse>(`/api/todos/${encodeURIComponent(todo_id)}/complete`, "POST", {
          user_id: USER_ID,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "delete_todo": {
        const { todo_id } = args as { todo_id: string };

        interface DeleteTodoResponse {
          success: boolean;
          formatted: string;
        }

        const result = await apiCall<DeleteTodoResponse>(`/api/todos/${encodeURIComponent(todo_id)}?user_id=${USER_ID}`, "DELETE");

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "reorder_todo": {
        const { todo_id, direction } = args as { todo_id: string; direction: string };

        interface ReorderTodoResponse {
          success: boolean;
          todo: unknown;
          formatted: string;
        }

        const result = await apiCall<ReorderTodoResponse>(`/api/todos/${encodeURIComponent(todo_id)}/reorder`, "POST", {
          user_id: USER_ID,
          direction,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "add_project": {
        const { name, prefix, description, parent } = args as { name: string; prefix?: string; description?: string; parent?: string };

        interface ProjectResponse {
          success: boolean;
          project: { id: string; name: string; prefix?: string };
          formatted: string;
        }

        const result = await apiCall<ProjectResponse>("/api/projects", "POST", {
          user_id: USER_ID,
          name,
          prefix,
          description,
          parent,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "list_projects": {
        // `projects` arrives as (Project, ProjectStats) tuples — see
        // handlers/todos.rs list_projects — hence the pair type.
        interface ProjectWire {
          id?: string;
          name?: string;
          prefix?: string | null;
          description?: string | null;
          status?: string;
          parent_id?: string | null;
        }
        interface ProjectStatsWire {
          total?: number;
          backlog?: number;
          todo?: number;
          in_progress?: number;
          blocked?: number;
          done?: number;
          cancelled?: number;
        }
        interface ListProjectsResponse {
          success: boolean;
          count?: number;
          projects: [ProjectWire, ProjectStatsWire][];
          formatted: string;
        }

        const result = await apiCall<ListProjectsResponse>("/api/projects/list", "POST", {
          user_id: USER_ID,
        });

        const projectPairs = result.projects || [];

        return {
          content: [{ type: "text", text: result.formatted }],
          structuredContent: {
            projects: projectPairs.map(([p, stats]) =>
              compact({
                id: p?.id,
                name: p?.name,
                prefix: p?.prefix ?? undefined,
                description: p?.description ?? undefined,
                status: p?.status,
                parent_id: p?.parent_id ?? undefined,
                stats: stats ? compact({ ...stats }) : undefined,
              }),
            ),
            count: result.count ?? projectPairs.length,
          },
        };
      }

      case "archive_project": {
        const { project } = args as { project: string };

        interface ProjectResponse {
          success: boolean;
          project: { id: string; name: string };
          formatted: string;
        }

        const result = await apiCall<ProjectResponse>(`/api/projects/${encodeURIComponent(project)}/update`, "POST", {
          user_id: USER_ID,
          status: "archived",
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "delete_project": {
        const { project, delete_todos } = args as { project: string; delete_todos?: boolean };

        interface ProjectResponse {
          success: boolean;
          project: { id: string; name: string };
          formatted: string;
        }

        const result = await apiCall<ProjectResponse>(`/api/projects/${encodeURIComponent(project)}/delete`, "POST", {
          user_id: USER_ID,
          delete_todos: delete_todos ?? false,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "todo_stats": {
        // Flat status counts (UserTodoStats), passed through as-is rather than
        // reshaped — a renaming layer would drift from the backend.
        interface TodoStatsResponse {
          stats: Record<string, number>;
          formatted: string;
        }

        const result = await apiCall<TodoStatsResponse>("/api/todos/stats", "POST", {
          user_id: USER_ID,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
          structuredContent: { total: 0, ...(result.stats || {}) },
        };
      }

      case "list_subtasks": {
        const { parent_id } = args as { parent_id: string };

        interface ListSubtasksResponse {
          success: boolean;
          todos: TodoWire[];
          formatted: string;
        }

        const result = await apiCall<ListSubtasksResponse>(
          `/api/todos/${parent_id}/subtasks?user_id=${USER_ID}`,
          "GET"
        );

        const subtasks = result.todos || [];

        return {
          content: [{ type: "text", text: result.formatted }],
          structuredContent: {
            parent_id,
            subtasks: subtasks.map(structuredTodo),
            count: subtasks.length,
          },
        };
      }

      case "add_todo_comment": {
        const { todo_id, content, comment_type } = args as {
          todo_id: string;
          content: string;
          comment_type?: string;
        };

        interface CommentResponse {
          success: boolean;
          comment: unknown;
          formatted: string;
        }

        const result = await apiCall<CommentResponse>(
          `/api/todos/${todo_id}/comments`,
          "POST",
          {
            user_id: USER_ID,
            content,
            comment_type,
          }
        );

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "list_todo_comments": {
        const { todo_id } = args as { todo_id: string };

        interface CommentWire {
          id?: string;
          content?: string;
          author?: string;
          comment_type?: string;
          created_at?: string;
          updated_at?: string | null;
        }
        interface CommentListResponse {
          success: boolean;
          count: number;
          comments: CommentWire[];
          formatted: string;
        }

        const result = await apiCall<CommentListResponse>(
          `/api/todos/${todo_id}/comments?user_id=${USER_ID}`,
          "GET"
        );

        const comments = result.comments || [];

        return {
          content: [{ type: "text", text: result.formatted }],
          structuredContent: {
            todo_id,
            comments: comments.map((c) =>
              compact({
                id: c.id,
                content: c.content,
                author: c.author,
                comment_type: c.comment_type,
                created_at: c.created_at,
              }),
            ),
            count: result.count ?? comments.length,
          },
        };
      }

      case "update_todo_comment": {
        const { todo_id, comment_id, content } = args as {
          todo_id: string;
          comment_id: string;
          content: string;
        };

        interface CommentResponse {
          success: boolean;
          comment: unknown;
          formatted: string;
        }

        const result = await apiCall<CommentResponse>(
          `/api/todos/${todo_id}/comments/${comment_id}/update`,
          "POST",
          {
            user_id: USER_ID,
            content,
          }
        );

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "delete_todo_comment": {
        const { todo_id, comment_id } = args as {
          todo_id: string;
          comment_id: string;
        };

        interface CommentResponse {
          success: boolean;
          formatted: string;
        }

        const result = await apiCall<CommentResponse>(
          `/api/todos/${todo_id}/comments/${comment_id}?user_id=${USER_ID}`,
          "DELETE"
        );

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "read_memory": {
        const memory_id = (args as any).memory_id || (args as any).id;

        if (!memory_id || typeof memory_id !== 'string' || memory_id.trim().length === 0) {
          return {
            content: [{ type: "text", text: "Error: 'memory_id' is required. Pass the full UUID or 8+ character prefix from recall results." }],
            isError: true,
          };
        }

        // Response includes hierarchy: parent_id in memory, children_ids/children_count
        interface MemoryWithHierarchy {
          id: string;
          experience: {
            content: string;
            experience_type: string;
            entities?: string[];
          };
          importance: number;
          created_at: string;
          tier?: string;
          parent_id?: string;
          children_ids: string[];
          children_count: number;
        }

        // Backend accepts both full UUIDs and 8+ char hex prefixes
        let memory: MemoryWithHierarchy | null = null;

        try {
          memory = await apiCall<MemoryWithHierarchy>(
            `/api/memory/${memory_id}?user_id=${encodeURIComponent(USER_ID)}`,
            "GET"
          );
        } catch (e) {
          console.error(`[Memory] Failed to fetch memory ${memory_id}:`, e);
        }

        if (!memory) {
          // Reported as a successful call rather than an error (unchanged
          // behaviour); the structured payload says so explicitly.
          return {
            content: [{ type: "text", text: `Memory not found: ${memory_id}` }],
            structuredContent: { found: false, id: memory_id },
          };
        }

        // Format full memory content with hierarchy info
        const tags = memory.experience.entities?.join(", ") || "none";
        const created = new Date(memory.created_at).toLocaleString();

        let response = `Memory: ${memory.id}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Type: ${memory.experience.experience_type} | Tags: ${tags}\n`;
        response += `Tier: ${memory.tier || 'Unknown'} | Created: ${created} | Importance: ${(memory.importance * 100).toFixed(0)}%\n`;

        // Hierarchy info
        if (memory.parent_id) {
          response += `Parent: ${memory.parent_id}\n`;
        }
        if (memory.children_count > 0) {
          response += `Children: ${memory.children_count} (${memory.children_ids.join(", ")})\n`;
        }

        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
        response += memory.experience.content;

        return {
          content: [{ type: "text", text: response }],
          structuredContent: compact({
            found: true,
            id: memory.id,
            // read_memory's whole purpose is the untruncated body, so the
            // structured channel carries it in full too.
            content: memory.experience.content,
            memory_type: memory.experience.experience_type,
            entities: memory.experience.entities,
            created_at: memory.created_at,
            importance: memory.importance,
            tier: memory.tier,
            parent_id: memory.parent_id,
            children_ids: memory.children_ids,
            children_count: memory.children_count,
          }),
        };
      }

      // =======================================================================
      // CAUSAL LINEAGE TOOLS
      // =======================================================================

      case "trace_lineage": {
        const { memory_id, direction = "backward", max_depth = 10 } = args as {
          memory_id: string;
          direction?: string;
          max_depth?: number;
        };

        if (!memory_id || memory_id.trim().length === 0) {
          return { content: [{ type: "text", text: "Error: 'memory_id' is required (full UUID or 8+ char prefix from recall results)" }], isError: true };
        }
        const validDirections = ["backward", "forward", "both"];
        if (!validDirections.includes(direction)) {
          return { content: [{ type: "text", text: `Error: 'direction' must be one of: ${validDirections.join(", ")}` }], isError: true };
        }
        const depth = Math.max(1, Math.min(Math.floor(max_depth), 100));

        // Lineage endpoints require a full UUID (bare uuid parse server-side);
        // resolve prefixes via GET /api/memory/{id} first.
        let resolvedId: string;
        try {
          resolvedId = await resolveMemoryId(memory_id);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          return { content: [{ type: "text", text: `Could not resolve memory '${memory_id}': ${msg}` }], isError: true };
        }

        interface LineageTraceResponse {
          root: string;
          direction: string;
          edges: LineageEdgeWire[];
          path: string[];
          depth: number;
        }

        // Root cause (oldest ancestor) only exists in the backward direction.
        const wantRootCause = direction === "backward" || direction === "both";
        const [trace, rootCause] = await Promise.all([
          apiCall<LineageTraceResponse>("/api/lineage/trace", "POST", {
            user_id: USER_ID,
            memory_id: resolvedId,
            direction,
            max_depth: depth,
          }),
          wantRootCause
            ? apiCall<{ memory_id: string; root_cause_id: string | null }>("/api/lineage/root-cause", "POST", {
                user_id: USER_ID,
                memory_id: resolvedId,
              }).catch(() => null)
            : Promise.resolve(null),
        ]);

        const edges = trace.edges || [];

        // The text render caps how many edges it prints; the structured payload
        // deliberately does not — a consumer parsing it should see the whole
        // traced chain, not the display subset.
        const traceStructured = (): Record<string, unknown> =>
          compact({
            memory_id: trace.root ?? resolvedId,
            direction,
            root_cause_id: rootCause?.root_cause_id ?? undefined,
            edges: edges.map(structuredLineageEdge),
            path: trace.path,
            depth_reached: trace.depth,
            edge_count: edges.length,
          });

        if (edges.length === 0) {
          const hint = direction === "both"
            ? "This memory has no causal edges at all yet."
            : `Try direction:"both" to look in the other direction.`;
          return {
            content: [{
              type: "text",
              text: `🔗 No causal edges found ${direction} from ${resolvedId} (depth ${depth}).\n${hint}\nEdges are inferred automatically at ingest from type + entity overlap + temporal order, or recorded explicitly with add_causal_link.`,
            }],
            structuredContent: traceStructured(),
          };
        }

        // Previews for everything on the path plus edge endpoints (bounded).
        const previewIds = [...(trace.path || []), ...edges.flatMap((e) => [e.from, e.to])];
        const previews = await fetchMemoryPreviews(previewIds);

        let response = `🔗 Causal Trace (${direction})\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Memory: ${trace.root}\n`;
        const rootPreview = previews.get(trace.root);
        if (rootPreview) response += `  "${rootPreview}"\n`;
        response += `Depth reached: ${trace.depth} │ Edges: ${edges.length}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        // A hub memory at depth 10 can pull in dozens of edges; cap the render
        // and say so rather than flooding the context window.
        const TRACE_DISPLAY_CAP = 20;
        response += `EDGES (from = cause, to = effect)\n`;
        for (const edge of edges.slice(0, TRACE_DISPLAY_CAP)) {
          response += formatLineageEdge(edge, previews, "  ");
          response += `\n`;
        }
        if (edges.length > TRACE_DISPLAY_CAP) {
          response += `  … ${edges.length - TRACE_DISPLAY_CAP} more edges (re-run with a smaller max_depth to focus)\n\n`;
        }

        if (rootCause?.root_cause_id) {
          response += `⏮ ROOT CAUSE: ${rootCause.root_cause_id}\n`;
          const rcPreview = previews.get(rootCause.root_cause_id);
          if (rcPreview) response += `  "${rcPreview}"\n`;
        } else if (wantRootCause) {
          response += `⏮ ROOT CAUSE: this memory is itself the start of its chain (no older ancestor).\n`;
        }

        response += `\nUse read_memory for full content, validate_causal_link (edge_id + confirm/reject) to curate an edge.`;

        return { content: [{ type: "text", text: response }], structuredContent: traceStructured() };
      }

      case "list_causal_edges": {
        const { limit: rawEdgeLimit = 15 } = args as { limit?: number };
        const edgeLimit = Math.max(1, Math.min(Math.floor(rawEdgeLimit), MAX_LIMIT));

        interface LineageStatsWire {
          total_edges: number;
          inferred_edges: number;
          confirmed_edges: number;
          explicit_edges: number;
          total_branches: number;
          active_branches: number;
          edges_by_relation: Record<string, number>;
          avg_confidence: number;
        }

        // The server pages in storage order, so ask for a generous page and
        // rank by confidence here — otherwise "top edges" would be a sorted
        // sample of an arbitrary prefix.
        const EDGE_FETCH_LIMIT = 500;
        const [edgesResult, stats] = await Promise.all([
          apiCall<{ edges: LineageEdgeWire[]; total: number }>("/api/lineage/edges", "POST", {
            user_id: USER_ID,
            limit: EDGE_FETCH_LIMIT,
          }),
          apiCall<LineageStatsWire>("/api/lineage/stats", "POST", { user_id: USER_ID }).catch(() => null),
        ]);

        const edges = edgesResult.edges || [];

        const edgesStructured = (): Record<string, unknown> =>
          compact({
            edges: edges.map(structuredLineageEdge),
            count: edges.length,
            total: stats?.total_edges,
            stats: stats ? { ...stats } : undefined,
          });

        if (edges.length === 0) {
          return {
            content: [{
              type: "text",
              text: `🔗 The causal lineage graph is empty.\nEdges are inferred automatically at ingest (memory type + entity overlap + temporal order) and can be added explicitly with add_causal_link.`,
            }],
            structuredContent: edgesStructured(),
          };
        }

        // Highest-confidence first: curated (confirmed/explicit, 1.0) edges
        // surface above low-confidence inferences.
        edges.sort((a, b) => b.confidence - a.confidence);
        const totalFetched = edges.length;
        edges.length = Math.min(edges.length, edgeLimit);

        let response = `🔗 Causal Lineage Graph\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        if (stats) {
          response += `Edges: ${stats.total_edges} total │ ${stats.inferred_edges} inferred │ ${stats.confirmed_edges} confirmed │ ${stats.explicit_edges} explicit\n`;
          response += `Avg confidence: ${(stats.avg_confidence * 100).toFixed(0)}%`;
          const byRelation = Object.entries(stats.edges_by_relation || {})
            .sort((a, b) => b[1] - a[1])
            .map(([rel, n]) => `${rel}(${n})`)
            .join(" │ ");
          if (byRelation) response += `\nBy relation: ${byRelation}`;
          response += `\n`;
        }
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        // Preview budget covers the first PREVIEW_FETCH_CAP/2 edges' endpoints.
        const previews = await fetchMemoryPreviews(edges.flatMap((e) => [e.from, e.to]));

        response += `TOP ${edges.length} OF ${stats?.total_edges ?? totalFetched} EDGES BY CONFIDENCE (from = cause, to = effect)\n`;
        for (const edge of edges) {
          response += formatLineageEdge(edge, previews, "  ");
          response += `\n`;
        }

        response += `Use trace_lineage(memory_id) for the chain around one memory; validate_causal_link(edge_id, verdict) to confirm or reject an inferred edge.`;

        // `edges` has been sorted and truncated to edgeLimit above, so the
        // structured payload carries exactly the edges the text lists.
        return { content: [{ type: "text", text: response }], structuredContent: edgesStructured() };
      }

      case "add_causal_link": {
        const { from_memory_id, to_memory_id, relation } = args as {
          from_memory_id: string;
          to_memory_id: string;
          relation: string;
        };

        if (!from_memory_id || !to_memory_id) {
          return { content: [{ type: "text", text: "Error: 'from_memory_id' (the cause) and 'to_memory_id' (the effect) are both required" }], isError: true };
        }
        // Exact variants accepted by the server (src/handlers/lineage.rs match).
        const validRelations = ["Caused", "ResolvedBy", "InformedBy", "SupersededBy", "TriggeredBy", "BranchedFrom", "RelatedTo"];
        if (!validRelations.includes(relation)) {
          return { content: [{ type: "text", text: `Error: 'relation' must be one of: ${validRelations.join(", ")}` }], isError: true };
        }

        let fromId: string;
        let toId: string;
        try {
          [fromId, toId] = await Promise.all([
            resolveMemoryId(from_memory_id),
            resolveMemoryId(to_memory_id),
          ]);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          return { content: [{ type: "text", text: `Could not resolve memory ids: ${msg}` }], isError: true };
        }
        if (fromId === toId) {
          return { content: [{ type: "text", text: "Error: from_memory_id and to_memory_id resolve to the same memory — a memory cannot cause itself" }], isError: true };
        }

        const edge = await apiCall<LineageEdgeWire>("/api/lineage/link", "POST", {
          user_id: USER_ID,
          from_memory_id: fromId,
          to_memory_id: toId,
          relation,
        });

        const previews = await fetchMemoryPreviews([edge.from, edge.to]);
        const prose = CAUSAL_RELATION_PROSE[edge.relation] || edge.relation;

        let response = `🔗 Causal Link Recorded\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `"${previews.get(edge.from) || edge.from}"\n`;
        response += `  ──${edge.relation} (${prose})──▶\n`;
        response += `"${previews.get(edge.to) || edge.to}"\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `edge_id: ${edge.id} │ confidence: ${(edge.confidence * 100).toFixed(0)}% │ source: ${edge.source}`;

        return { content: [{ type: "text", text: response }] };
      }

      case "validate_causal_link": {
        const { edge_id, verdict } = args as { edge_id: string; verdict: string };

        if (!edge_id || edge_id.trim().length === 0) {
          return { content: [{ type: "text", text: "Error: 'edge_id' is required (from trace_lineage or list_causal_edges output)" }], isError: true };
        }
        if (verdict !== "confirm" && verdict !== "reject") {
          return { content: [{ type: "text", text: "Error: 'verdict' must be 'confirm' or 'reject'" }], isError: true };
        }

        if (verdict === "confirm") {
          const result = await apiCall<{ confirmed: boolean; graph_edges_strengthened: number }>(
            "/api/lineage/confirm",
            "POST",
            { user_id: USER_ID, edge_id },
          );
          if (!result.confirmed) {
            return {
              content: [{ type: "text", text: `Edge ${edge_id} was not confirmed — it does not exist (check the edge_id against list_causal_edges).` }],
              isError: true,
            };
          }
          let response = `✓ Causal edge confirmed: ${edge_id}\n`;
          response += `Confidence raised to 100%; ${result.graph_edges_strengthened} knowledge-graph edge(s) between the memories' entities strengthened.`;
          return { content: [{ type: "text", text: response }] };
        }

        const result = await apiCall<{ rejected: boolean }>("/api/lineage/reject", "POST", {
          user_id: USER_ID,
          edge_id,
        });
        if (!result.rejected) {
          return {
            content: [{ type: "text", text: `Edge ${edge_id} was not rejected — it does not exist (it may already have been rejected).` }],
            isError: true,
          };
        }
        return { content: [{ type: "text", text: `✕ Causal edge rejected and deleted: ${edge_id}` }] };
      }

      // =======================================================================
      // KNOWLEDGE GRAPH TOOLS
      // =======================================================================

      case "explore_entity": {
        const { entity_name, max_depth = 1 } = args as { entity_name: string; max_depth?: number };

        if (!entity_name || entity_name.trim().length === 0) {
          return { content: [{ type: "text", text: "Error: 'entity_name' is required (use list_entities to discover names)" }], isError: true };
        }
        const depth = Math.max(1, Math.min(Math.floor(max_depth), 3));

        // Entity wire shape (src/graph_memory.rs EntityNode). name_embedding is
        // deliberately never rendered.
        interface EntityWire {
          uuid: string;
          name: string;
          labels: string[];
          mention_count: number;
          salience: number;
          fine_type: string | null;
          kb_id: string | null;
        }
        interface TraversalWire {
          entities: Array<{ entity: EntityWire; hop_distance: number; decay_factor: number }>;
          relationships: Array<{
            uuid: string;
            from_entity: string;
            to_entity: string;
            relation_type: unknown;
            strength: number;
            context: string;
            invalidated_at: string | null;
            tier?: string;
          }>;
        }

        let traversal: TraversalWire;
        try {
          traversal = await apiCall<TraversalWire>("/api/graph/traverse", "POST", {
            user_id: USER_ID,
            entity_name,
            max_depth: depth,
          });
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          if (msg.includes("API error 404")) {
            // Reported as a successful call rather than an error (unchanged
            // behaviour); `found: false` carries that in the structured channel.
            return {
              content: [{ type: "text", text: `Entity not found in the knowledge graph: "${entity_name}" (no exact, case-insensitive, stemmed, or substring match). Use list_entities to see what the graph knows.` }],
              structuredContent: {
                query: entity_name,
                found: false,
                max_depth: depth,
                entities: [],
                relationships: [],
                entity_count: 0,
                relationship_count: 0,
              },
            };
          }
          throw e;
        }

        const entities = traversal.entities || [];
        const liveRels = (traversal.relationships || []).filter((r) => !r.invalidated_at);
        const invalidatedCount = (traversal.relationships || []).length - liveRels.length;
        const nameById = new Map(entities.map((t) => [t.entity.uuid, t.entity.name]));

        const describeEntity = (t: { entity: EntityWire; hop_distance: number }): string => {
          const e = t.entity;
          const type = e.fine_type || e.labels.join("/") || "Concept";
          return `  [hop ${t.hop_distance}] ${e.name} (${type} │ mentions: ${e.mention_count} │ salience: ${e.salience.toFixed(2)})`;
        };

        // Name matching is fuzzy server-side (exact → case-insensitive →
        // stemmed → substring); say what actually matched so the agent never
        // reads a neighborhood as belonging to the name it typed.
        const origin = entities.find((t) => t.hop_distance === 0)?.entity;
        const matchedNote = origin && origin.name !== entity_name ? ` (matched entity: "${origin.name}")` : "";

        let response = `🕸 Entity Graph: ${entity_name}${matchedNote} (depth ${depth})\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Connected entities: ${entities.length} │ Relationships: ${liveRels.length}`;
        if (invalidatedCount > 0) response += ` (${invalidatedCount} invalidated hidden)`;
        response += `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        // Entities: all hops, salience-ranked within hop, capped for readability.
        const ENTITY_DISPLAY_CAP = 30;
        const sortedEntities = [...entities].sort(
          (a, b) => a.hop_distance - b.hop_distance || b.entity.salience - a.entity.salience,
        );
        response += `ENTITIES\n`;
        for (const t of sortedEntities.slice(0, ENTITY_DISPLAY_CAP)) {
          response += describeEntity(t) + `\n`;
        }
        if (sortedEntities.length > ENTITY_DISPLAY_CAP) {
          response += `  … ${sortedEntities.length - ENTITY_DISPLAY_CAP} more (lower salience)\n`;
        }

        // Typed relationships are the signal — show them all (capped), with
        // the sentence that attested them. Untyped co-occurrence is bulk;
        // summarize it.
        const GENERIC_TYPES = new Set(["CoOccurs", "RelatedTo"]);
        const typed = liveRels
          .filter((r) => !GENERIC_TYPES.has(formatRelationType(r.relation_type)))
          .sort((a, b) => b.strength - a.strength);
        const generic = liveRels.filter((r) => GENERIC_TYPES.has(formatRelationType(r.relation_type)));

        const TYPED_DISPLAY_CAP = 20;
        if (typed.length > 0) {
          response += `\nTYPED RELATIONSHIPS (strongest first)\n`;
          for (const r of typed.slice(0, TYPED_DISPLAY_CAP)) {
            const from = nameById.get(r.from_entity) || r.from_entity;
            const to = nameById.get(r.to_entity) || r.to_entity;
            response += `  ${from} ──${formatRelationType(r.relation_type)}──▶ ${to} (${(r.strength * 100).toFixed(0)}%)\n`;
            if (r.context) {
              const ctx = r.context.length > 100 ? r.context.slice(0, 100) + "…" : r.context;
              response += `    ⌞ "${ctx}"\n`;
            }
          }
          if (typed.length > TYPED_DISPLAY_CAP) {
            response += `  … ${typed.length - TYPED_DISPLAY_CAP} more typed relationships\n`;
          }
        } else {
          response += `\nNo typed relationships in this neighborhood — only co-occurrence so far. Typed edges (Causes, Triggers, DependsOn, …) appear as the relation extractor finds explicit statements.\n`;
        }

        if (generic.length > 0) {
          response += `\nPlus ${generic.length} co-occurrence edge(s) (entities that appear together without an extracted typed relation).\n`;
        }

        response += `\nUse explore_entity on a neighbor to keep walking, or recall to read the memories behind a relationship's context.`;

        // The text render caps entities at 30 and typed relationships at 20;
        // the structured payload carries the full traversal so a consumer is
        // not silently handed a display subset.
        return {
          content: [{ type: "text", text: response }],
          structuredContent: compact({
            query: entity_name,
            found: true,
            matched_entity: origin?.name,
            max_depth: depth,
            entities: sortedEntities.map((t) =>
              compact({
                id: t.entity.uuid,
                name: t.entity.name,
                entity_type: t.entity.fine_type || t.entity.labels.join("/") || "Concept",
                labels: t.entity.labels,
                salience: t.entity.salience,
                mention_count: t.entity.mention_count,
                kb_id: t.entity.kb_id ?? undefined,
                hop_distance: t.hop_distance,
              }),
            ),
            relationships: liveRels.map((r) => {
              const relType = formatRelationType(r.relation_type);
              return compact({
                id: r.uuid,
                from: nameById.get(r.from_entity) || r.from_entity,
                to: nameById.get(r.to_entity) || r.to_entity,
                from_id: r.from_entity,
                to_id: r.to_entity,
                relation_type: relType,
                strength: r.strength,
                context: r.context || undefined,
                typed: !GENERIC_TYPES.has(relType),
              });
            }),
            entity_count: entities.length,
            relationship_count: liveRels.length,
            invalidated_count: invalidatedCount,
          }),
        };
      }

      case "list_entities": {
        const { limit: rawEntityLimit = 30 } = args as { limit?: number };
        const entityLimit = Math.max(1, Math.min(Math.floor(rawEntityLimit), MAX_LIMIT));

        interface EntityListWire {
          entities: Array<{
            uuid: string;
            name: string;
            labels: string[];
            mention_count: number;
            salience: number;
            fine_type: string | null;
          }>;
          count: number;
        }

        // The server truncates BEFORE any ordering (get_all_entities → take(limit)),
        // so a small server-side limit would sample arbitrary storage order.
        // Request a generous page and rank by salience here.
        const SERVER_FETCH_LIMIT = 500;
        const result = await apiCall<EntityListWire>("/api/graph/entities/all", "POST", {
          user_id: USER_ID,
          limit: SERVER_FETCH_LIMIT,
        });

        const all = result.entities || [];
        if (all.length === 0) {
          return {
            content: [{ type: "text", text: `🕸 The knowledge graph has no entities yet.\nEntities are extracted automatically as memories are stored — remember something first.` }],
            structuredContent: { entities: [], count: 0, total: 0 },
          };
        }

        const ranked = [...all].sort((a, b) => b.salience - a.salience).slice(0, entityLimit);

        let response = `🕸 Knowledge Graph Entities (top ${ranked.length} of ${all.length}${all.length === SERVER_FETCH_LIMIT ? "+" : ""} by salience)\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        for (let i = 0; i < ranked.length; i++) {
          const e = ranked[i];
          const type = e.fine_type || e.labels.join("/") || "Concept";
          response += `${String(i + 1).padStart(3)}. ${e.name}  (${type} │ mentions: ${e.mention_count} │ salience: ${e.salience.toFixed(2)})\n`;
        }
        response += `\nUse explore_entity(entity_name) to walk any of these.`;

        return {
          content: [{ type: "text", text: response }],
          structuredContent: {
            entities: ranked.map((e) =>
              compact({
                id: e.uuid,
                name: e.name,
                entity_type: e.fine_type || e.labels.join("/") || "Concept",
                labels: e.labels,
                salience: e.salience,
                mention_count: e.mention_count,
              }),
            ),
            count: ranked.length,
            // Entities seen in this page. The server truncates at
            // SERVER_FETCH_LIMIT before ordering, so when `all` is exactly that
            // size the real graph may hold more — same caveat the text renders
            // as a trailing "+".
            total: all.length,
          },
        };
      }

      // =======================================================================
      // ANOMALY & FACTS TOOLS
      // =======================================================================

      case "list_anomalies": {
        const { limit: rawAnomalyLimit = 10, min_sigma = 2.0 } = args as { limit?: number; min_sigma?: number };
        const anomalyLimit = Math.max(1, Math.min(Math.floor(rawAnomalyLimit), MAX_LIMIT));
        const sigma = Math.max(0, min_sigma);

        interface AnomalyWire {
          anomalies: Array<{
            memory_id: string;
            created_at: string;
            content_preview: string;
            max_abs_z: number;
            flagged: boolean;
            explanation: string;
            entities: Array<{ id: string; name: string }>;
          }>;
          episodes_scored: number;
          baseline_window: number;
          min_sigma: number;
        }

        const result = await apiCall<AnomalyWire>("/api/anomalies", "POST", {
          user_id: USER_ID,
          limit: anomalyLimit,
          min_sigma: sigma,
        });

        const anomalies = result.anomalies || [];

        const anomaliesStructured = (): Record<string, unknown> => ({
          min_sigma: result.min_sigma,
          anomalies: anomalies.map((a) =>
            compact({
              memory_id: a.memory_id,
              content_preview: a.content_preview,
              max_abs_z: a.max_abs_z,
              flagged: a.flagged,
              explanation: a.explanation || undefined,
              entities: a.entities,
              created_at: a.created_at,
            }),
          ),
          count: anomalies.length,
          flagged_count: anomalies.filter((a) => a.flagged).length,
          episodes_scored: result.episodes_scored,
          baseline_window: result.baseline_window,
        });

        if (anomalies.length === 0) {
          // The endpoint returns an empty feed (not z-scores against noise)
          // below its minimum baseline of scored episodes.
          const reason = result.episodes_scored < 10
            ? `Only ${result.episodes_scored} scored episode(s) exist — deviation needs a baseline of at least 10 before z-scores mean anything.`
            : `${result.episodes_scored} episodes scored against a window of ${result.baseline_window}; none deviate from the baseline.`;
          return {
            content: [{ type: "text", text: `📈 No anomalies.\n${reason}` }],
            structuredContent: anomaliesStructured(),
          };
        }

        const flaggedCount = anomalies.filter((a) => a.flagged).length;

        let response = `📈 Anomaly Feed (deviation vs your own baseline)\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `${anomalies.length} entries ranked by |z| │ ${flaggedCount} flagged at ≥${result.min_sigma.toFixed(1)}σ │ baseline: ${result.episodes_scored} episodes\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        for (let i = 0; i < anomalies.length; i++) {
          const a = anomalies[i];
          const marker = a.flagged ? "🚩" : "  ";
          const when = new Date(a.created_at).toLocaleString();
          response += `${marker} ${i + 1}. [${a.max_abs_z.toFixed(1)}σ] ${when}\n`;
          response += `     "${a.content_preview}"\n`;
          if (a.explanation) response += `     why: ${a.explanation}\n`;
          if (a.entities.length > 0) {
            response += `     entities: ${a.entities.map((e) => e.name).join(", ")}\n`;
          }
          response += `     memory: ${a.memory_id}\n\n`;
        }

        response += `Use read_memory(memory_id) for full content, trace_lineage(memory_id) to see what an anomalous memory caused or was caused by.`;

        return {
          content: [{ type: "text", text: response.trimEnd() }],
          structuredContent: anomaliesStructured(),
        };
      }

      case "search_facts": {
        const { query, entity, limit: rawFactLimit = 20 } = args as {
          query?: string;
          entity?: string;
          limit?: number;
        };
        const factLimit = Math.max(1, Math.min(Math.floor(rawFactLimit), MAX_LIMIT));

        interface FactWire {
          id: string;
          fact: string;
          confidence: number;
          support_count: number;
          related_entities: string[];
          fact_type: string;
          created_at: string;
          invalidated_at: string | null;
        }

        // One question, three filters: entity → /by-entity, query → /search,
        // neither → /list. Entity takes precedence when both are given.
        let endpoint: string;
        let body: Record<string, unknown>;
        let heading: string;
        if (entity && entity.trim().length > 0) {
          endpoint = "/api/facts/by-entity";
          body = { user_id: USER_ID, entity: entity.trim(), limit: factLimit };
          heading = `facts about "${entity.trim()}"`;
        } else if (query && query.trim().length > 0) {
          endpoint = "/api/facts/search";
          body = { user_id: USER_ID, query: query.trim(), limit: factLimit };
          heading = `facts matching "${query.trim()}"`;
        } else {
          endpoint = "/api/facts/list";
          body = { user_id: USER_ID, limit: factLimit };
          heading = "recent facts";
        }

        const result = await apiCall<{ facts: FactWire[]; total: number }>(endpoint, "POST", body);
        const facts = result.facts || [];

        const factsStructured = (): Record<string, unknown> =>
          compact({
            query: query && query.trim().length > 0 ? query.trim() : undefined,
            entity: entity && entity.trim().length > 0 ? entity.trim() : undefined,
            facts: facts.map((f) =>
              compact({
                id: f.id,
                fact: f.fact,
                fact_type: f.fact_type,
                confidence: f.confidence,
                support_count: f.support_count,
                related_entities: f.related_entities,
                created_at: f.created_at,
                invalidated_at: f.invalidated_at ?? undefined,
              }),
            ),
            count: facts.length,
          });

        if (facts.length === 0) {
          return {
            content: [{
              type: "text",
              text: `📚 No ${heading}.\nSemantic facts distill out of episodic memories during consolidation — a young or recently imported corpus may have none yet. Try recall for the underlying episodic memories, or fact_narratives for topic clusters.`,
            }],
            structuredContent: factsStructured(),
          };
        }

        let response = `📚 Semantic Facts: ${facts.length} ${heading}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
        for (let i = 0; i < facts.length; i++) {
          const f = facts[i];
          const invalidated = f.invalidated_at ? " ⚠ INVALIDATED (superseded — retained for audit)" : "";
          response += `${String(i + 1).padStart(2)}. ${f.fact}${invalidated}\n`;
          response += `    ┗━ ${f.fact_type} │ confidence: ${(f.confidence * 100).toFixed(0)}% │ support: ${f.support_count} memories`;
          if (f.related_entities.length > 0) {
            response += ` │ entities: ${f.related_entities.slice(0, 5).join(", ")}`;
          }
          response += `\n\n`;
        }
        response += `Facts are confidence-scored distillations; use recall to read the episodic memories behind them.`;

        return {
          content: [{ type: "text", text: response.trimEnd() }],
          structuredContent: factsStructured(),
        };
      }

      // =======================================================================
      // HEBBIAN FEEDBACK
      // =======================================================================

      case "reinforce_memories": {
        const { memory_ids, outcome } = args as { memory_ids: string[]; outcome: string };

        if (!memory_ids || memory_ids.length === 0) {
          return { content: [{ type: "text", text: "Error: 'memory_ids' must contain at least one memory ID" }], isError: true };
        }
        const validOutcomes = ["helpful", "misleading", "neutral"];
        if (!validOutcomes.includes(outcome)) {
          return { content: [{ type: "text", text: `Error: 'outcome' must be one of: ${validOutcomes.join(", ")}` }], isError: true };
        }

        // The endpoint silently drops non-UUID ids; resolve prefixes here so a
        // short id from recall output reinforces instead of vanishing.
        const resolved: string[] = [];
        const unresolvable: string[] = [];
        await Promise.all(
          memory_ids.slice(0, 50).map(async (id) => {
            try {
              resolved.push(await resolveMemoryId(id));
            } catch {
              unresolvable.push(id);
            }
          }),
        );

        if (resolved.length === 0) {
          return {
            content: [{ type: "text", text: `Error: none of the provided ids resolved to memories: ${unresolvable.join(", ")}` }],
            isError: true,
          };
        }

        interface ReinforceWire {
          memories_processed: number;
          associations_strengthened: number;
          importance_boosts: number;
          importance_decays: number;
        }

        const result = await apiCall<ReinforceWire>("/api/reinforce", "POST", {
          user_id: USER_ID,
          ids: resolved,
          outcome,
        });

        const verb = outcome === "helpful" ? "boosted" : outcome === "misleading" ? "decayed" : "recorded";
        let response = `🧠 Hebbian Feedback (${outcome})\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Memories processed: ${result.memories_processed} (${verb})\n`;
        response += `Associations strengthened: ${result.associations_strengthened}\n`;
        response += `Importance boosts: ${result.importance_boosts} │ decays: ${result.importance_decays}`;
        if (unresolvable.length > 0) {
          response += `\n⚠ Not found (skipped): ${unresolvable.join(", ")}`;
        }

        return { content: [{ type: "text", text: response }] };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  };

  // Execute tool and stream result automatically
  try {
    const result = await executeTool();

    // Stream tool interaction to memory (non-blocking)
    const resultText = result.content.map(c => c.text).join('\n');
    streamToolCall(name, args as Record<string, unknown>, resultText);

    // Token tracking: count tokens in response (SHO-115)
    const responseTokens = estimateTokens(resultText);
    sessionTokens += responseTokens;
    const tokenStatus = getTokenStatus();

    // Proactive surfacing: append relevant memories to non-memory tool responses
    if (PROACTIVE_SURFACING && !["remember", "recall", "forget", "list_memories", "proactive_context", "context_summary", "memory_stats"].includes(name)) {
      // Extract context from tool args
      const contextParts: string[] = [];
      if (args && typeof args === "object") {
        for (const [key, value] of Object.entries(args)) {
          if (typeof value === "string" && value.length > 10) {
            contextParts.push(value);
          }
        }
      }
      const context = contextParts.join(" ").slice(0, 1000);

      if (context.length >= PROACTIVE_MIN_CONTEXT_LENGTH) {
        const surfaced = await surfaceRelevant(context, 3);
        if (surfaced && surfaced.length > 0 && result.content.length > 0) {
          const surfacedText = formatSurfacedMemories(surfaced);
          result.content[result.content.length - 1].text += surfacedText;
        }
      }
    }

    // Inject context window warning if >= threshold (SHO-115)
    if (tokenStatus.alert && result.content.length > 0) {
      const percentUsed = Math.round(tokenStatus.percent * 100);
      const warning = `⚠️ CONTEXT ALERT: ${percentUsed}% of token budget used (${tokenStatus.tokens.toLocaleString()}/${tokenStatus.budget.toLocaleString()}). Consider starting a new session or running consolidation.\n\n`;
      result.content[0].text = warning + result.content[0].text;
    }

    // A tool that declares an outputSchema must return structuredContent on
    // every successful path — the SDK client rejects the result otherwise, and
    // the error it raises names the schema, not the code path that skipped it.
    // Log the tool name here so the cause is obvious in the server's stderr.
    if (TOOL_OUTPUT_SCHEMAS[name] && !result.isError && !result.structuredContent) {
      console.error(
        `[shodh-memory] BUG: tool "${name}" declares an outputSchema but returned no structuredContent. ` +
          `Clients validating structured output will reject this result.`,
      );
    }

    // Add _meta with token status to response
    return {
      ...result,
      _meta: {
        token_status: tokenStatus,
      },
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);

    // Provide helpful error messages
    let helpText = '';
    if (message.includes('ECONNREFUSED') || message.includes('fetch failed')) {
      helpText = '\n\nThe memory server appears to be offline. Start it with:\n  cd shodh-memory && cargo run';
    } else if (message.includes('API error 401')) {
      helpText = '\n\nAuthentication failed. Check your SHODH_API_KEY.';
    } else if (message.includes('API error 404')) {
      // A 404 carrying a structured error code came from a real handler that
      // could not find the *thing* asked for (MEMORY_NOT_FOUND, TODO_NOT_FOUND,
      // ...). That is a recoverable, id-level condition, and telling the agent
      // to suspect a version mismatch sends it down the wrong path. Only an
      // unrouted 404 — no code in the body — means the endpoint is missing.
      if (!/"code"\s*:\s*"[^"]+"/.test(message)) {
        helpText = '\n\nEndpoint not found. The server may be running an older version.';
      }
    }

    return {
      content: [
        {
          type: "text",
          text: `Error: ${message}${helpText}`,
        },
      ],
      isError: true,
    };
  }
};

// Register the tool handler under drain tracking (issue #405): the in-flight
// count drives whether a stdin EOF shuts down immediately or drains first.
server.setRequestHandler(CallToolRequestSchema, (request) =>
  drain.track(() => handleCallTool(request)),
);

// List resources (static commands + dynamic memories)
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  // Static resources - always available, appear first in @ autocomplete
  const staticResources = [
    {
      uri: "shodh://commands",
      name: "Available Commands",
      mimeType: "text/markdown",
      description: "List all shodh-memory commands and their usage",
    },
    {
      uri: "shodh://summary",
      name: "Session Summary",
      mimeType: "text/plain",
      description: "Recent learnings, decisions, and context",
    },
    {
      uri: "shodh://todos",
      name: "Pending Work",
      mimeType: "text/plain",
      description: "Your todo list and incomplete tasks",
    },
    {
      uri: "shodh://stats",
      name: "Memory Stats",
      mimeType: "application/json",
      description: "Memory system statistics and health",
    },
  ];

  try {
    const result = await apiCall<{ memories: Memory[] }>("/api/memories", "POST", {
      user_id: USER_ID,
    });

    const memories = result.memories || [];
    const memoryResources = memories.slice(0, 30).map((m) => {
      const content = getContent(m);
      return {
        uri: `memory://${m.id}`,
        name: content.slice(0, 50) + (content.length > 50 ? "..." : ""),
        mimeType: "text/plain",
        description: `Type: ${getType(m)}`,
      };
    });

    return {
      resources: [...staticResources, ...memoryResources],
    };
  } catch (e) {
    console.error("[Resources] Failed to list memory resources:", e);
    return { resources: staticResources };
  }
});

// Read a specific resource (shodh:// or memory://)
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;

  try {
    // Handle shodh:// static resources
    if (uri.startsWith("shodh://")) {
      const resource = uri.replace("shodh://", "");

      // shodh://search/{query} — the query is a URI-encoded path segment, so it
      // cannot be matched by the fixed-name switch below.
      if (resource === "search" || resource.startsWith("search/")) {
        const raw = resource.slice("search".length).replace(/^\//, "");
        let query: string;
        try {
          query = decodeURIComponent(raw);
        } catch {
          // Malformed percent-encoding: use the segment as typed rather than
          // failing the read outright.
          query = raw;
        }
        query = query.trim();
        if (!query) {
          throw new Error(
            "Empty search query. Use shodh://search/{query}, e.g. shodh://search/bridge%20collapse",
          );
        }

        const result = await apiCall<{ memories: Memory[] }>("/api/recall", "POST", {
          user_id: USER_ID,
          query,
          mode: "hybrid",
          limit: 10,
        });

        const memories = result.memories || [];
        const lines = memories.length > 0
          ? memories.map((m, i) => {
              const tier = m.tier ? ` | ${m.tier}` : "";
              return `${i + 1}. ${getContent(m)}\n   ${getType(m)}${tier} | ${m.id}`;
            })
          : ["No memories found."];

        return {
          contents: [{
            uri,
            mimeType: "text/plain",
            text: `Search results for "${query}" (${memories.length})\n\n${lines.join("\n\n")}`,
          }],
        };
      }

      switch (resource) {
        case "commands": {
          return {
            contents: [{
              uri,
              mimeType: "text/markdown",
              text: renderCommandsResource(TOOL_DEFINITIONS, SHODH_PROMPTS),
            }],
          };
        }

        case "summary": {
          const result = await apiCall<{
            learnings: Memory[];
            decisions: Memory[];
            context: Memory[];
          }>("/api/context_summary", "POST", {
            user_id: USER_ID,
            include_learnings: true,
            include_decisions: true,
            include_context: true,
            max_items: 5,
          });

          const parts: string[] = ["Session Summary\n"];
          if (result.learnings?.length) {
            parts.push("\nRecent Learnings:");
            result.learnings.forEach((m) => parts.push(`- ${getContent(m)}`));
          }
          if (result.decisions?.length) {
            parts.push("\nRecent Decisions:");
            result.decisions.forEach((m) => parts.push(`- ${getContent(m)}`));
          }
          if (result.context?.length) {
            parts.push("\nCurrent Context:");
            result.context.forEach((m) => parts.push(`- ${getContent(m)}`));
          }

          return {
            contents: [{
              uri,
              mimeType: "text/plain",
              text: parts.length > 1 ? parts.join("\n") : "No recent memories.",
            }],
          };
        }

        case "todos": {
          const result = await apiCall<{
            todos: Array<{
              id: string;
              content: string;
              status: string;
              priority: string;
              project_prefix?: string;
            }>;
          }>("/api/todos", "POST", {
            user_id: USER_ID,
            status: ["backlog", "todo", "in_progress", "blocked"],
          });

          const todos = result.todos || [];
          if (todos.length === 0) {
            return {
              contents: [{ uri, mimeType: "text/plain", text: "No pending tasks." }],
            };
          }

          const byStatus: Record<string, typeof todos> = {};
          todos.forEach((t) => {
            if (!byStatus[t.status]) byStatus[t.status] = [];
            byStatus[t.status].push(t);
          });

          const parts: string[] = ["Pending Work\n"];
          ["in_progress", "blocked", "todo", "backlog"].forEach((status) => {
            if (byStatus[status]?.length) {
              parts.push(`\n${status.replace("_", " ").toUpperCase()}:`);
              byStatus[status].forEach((t) => {
                const priority = t.priority !== "medium" ? ` [${t.priority}]` : "";
                const project = t.project_prefix ? ` (${t.project_prefix})` : "";
                parts.push(`- ${t.content}${priority}${project}`);
              });
            }
          });

          return {
            contents: [{ uri, mimeType: "text/plain", text: parts.join("\n") }],
          };
        }

        case "stats": {
          const stats = await apiCall<{
            total_memories: number;
            working_memory_count: number;
            session_memory_count: number;
            long_term_memory_count: number;
            vector_index_count: number;
            average_importance: number;
            total_retrievals: number;
            graph_nodes: number;
            graph_edges: number;
          }>(`/api/users/${encodeURIComponent(USER_ID)}/stats`, "GET");

          return {
            contents: [{
              uri,
              mimeType: "application/json",
              text: JSON.stringify(stats, null, 2),
            }],
          };
        }

        default:
          throw new Error(`Unknown resource: ${resource}`);
      }
    }

    // Handle memory:// resources
    const memoryId = uri.replace("memory://", "");
    const result = await apiCall<{ memories: Memory[] }>("/api/memories", "POST", {
      user_id: USER_ID,
    });

    const memory = (result.memories || []).find((m) => m.id === memoryId);

    if (!memory) {
      throw new Error(`Memory not found: ${memoryId}`);
    }

    const content = getContent(memory);

    return {
      contents: [
        {
          uri,
          mimeType: "text/plain",
          text: `Content: ${content}\n\nType: ${getType(memory)}\nCreated: ${memory.created_at || "unknown"}\nID: ${memory.id}`,
        },
      ],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to read resource: ${message}`);
  }
});

// =============================================================================
// MCP PROMPTS - Discoverable commands via /mcp__shodh-memory__<name>
// =============================================================================

// Define available prompts (these become slash commands in Claude Code)
const SHODH_PROMPTS = [
  {
    name: "quick_recall",
    description: "Search your memories for relevant context",
    arguments: [
      {
        name: "query",
        description: "What to search for in memories",
        required: true,
      },
    ],
  },
  {
    name: "session_summary",
    description: "Get a summary of recent learnings, decisions, and context",
    arguments: [],
  },
  {
    name: "what_i_know",
    description: "Surface everything related to a topic",
    arguments: [
      {
        name: "topic",
        description: "The topic to explore",
        required: true,
      },
    ],
  },
  {
    name: "pending_work",
    description: "Show todos and incomplete tasks",
    arguments: [],
  },
  {
    name: "recent_memories",
    description: "Show recently created memories",
    arguments: [
      {
        name: "count",
        description: "Number of memories (default: 10)",
        required: false,
      },
    ],
  },
  {
    name: "memory_health",
    description: "Check memory system status and statistics",
    arguments: [],
  },
];

// List available prompts
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: SHODH_PROMPTS.map((p) => ({
      name: p.name,
      description: p.description,
      arguments: p.arguments,
    })),
  };
});

// Get prompt content
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const promptName = request.params.name;
  const args = request.params.arguments || {};

  try {
    switch (promptName) {
      case "quick_recall": {
        const query = args.query as string;
        if (!query) {
          return {
            messages: [
              {
                role: "user",
                content: { type: "text", text: "Please provide a search query." },
              },
            ],
          };
        }
        const result = await apiCall<{ memories: Memory[] }>("/api/recall", "POST", {
          user_id: USER_ID,
          query,
          mode: "hybrid",
          limit: 5,
        });
        const memories = result.memories || [];
        const memoryText = memories.length > 0
          ? memories.map((m) => `- ${getContent(m)} (${getType(m)}${m.tier ? ` | ${m.tier}` : ''})`).join("\n")
          : "No memories found.";
        return {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: `Here's what I found about "${query}":\n\n${memoryText}`,
              },
            },
          ],
        };
      }

      case "session_summary": {
        const result = await apiCall<{
          learnings: Memory[];
          decisions: Memory[];
          context: Memory[];
        }>("/api/context_summary", "POST", {
          user_id: USER_ID,
          include_learnings: true,
          include_decisions: true,
          include_context: true,
          max_items: 5,
        });

        const parts: string[] = [];
        if (result.learnings?.length) {
          parts.push("**Recent Learnings:**");
          result.learnings.forEach((m) => parts.push(`- ${getContent(m)}`));
        }
        if (result.decisions?.length) {
          parts.push("\n**Recent Decisions:**");
          result.decisions.forEach((m) => parts.push(`- ${getContent(m)}`));
        }
        if (result.context?.length) {
          parts.push("\n**Current Context:**");
          result.context.forEach((m) => parts.push(`- ${getContent(m)}`));
        }

        const summaryText = parts.length > 0 ? parts.join("\n") : "No recent memories.";
        return {
          messages: [
            {
              role: "user",
              content: { type: "text", text: `Session Summary:\n\n${summaryText}` },
            },
          ],
        };
      }

      case "what_i_know": {
        const topic = args.topic as string;
        if (!topic) {
          return {
            messages: [
              {
                role: "user",
                content: { type: "text", text: "Please specify a topic to explore." },
              },
            ],
          };
        }
        const result = await apiCall<{ memories: Memory[] }>("/api/recall", "POST", {
          user_id: USER_ID,
          query: topic,
          mode: "hybrid",
          limit: 10,
        });
        const memories = result.memories || [];
        const grouped: Record<string, Memory[]> = {};
        memories.forEach((m) => {
          const type = getType(m);
          if (!grouped[type]) grouped[type] = [];
          grouped[type].push(m);
        });

        const parts: string[] = [`Everything I know about "${topic}":\n`];
        Object.entries(grouped).forEach(([type, mems]) => {
          parts.push(`\n**${type}s:**`);
          mems.forEach((m) => parts.push(`- ${getContent(m)}`));
        });

        return {
          messages: [
            {
              role: "user",
              content: {
                type: "text",
                text: memories.length > 0 ? parts.join("\n") : `No memories found about "${topic}".`,
              },
            },
          ],
        };
      }

      case "pending_work": {
        const result = await apiCall<{
          todos: Array<{
            id: string;
            content: string;
            status: string;
            priority: string;
            project_prefix?: string;
          }>;
        }>("/api/todos", "POST", {
          user_id: USER_ID,
          status: ["backlog", "todo", "in_progress", "blocked"],
        });
        const todos = result.todos || [];
        if (todos.length === 0) {
          return {
            messages: [
              {
                role: "user",
                content: { type: "text", text: "No pending tasks. You're all caught up!" },
              },
            ],
          };
        }

        const byStatus: Record<string, typeof todos> = {};
        todos.forEach((t) => {
          if (!byStatus[t.status]) byStatus[t.status] = [];
          byStatus[t.status].push(t);
        });

        const parts: string[] = ["**Pending Work:**\n"];
        ["in_progress", "blocked", "todo", "backlog"].forEach((status) => {
          if (byStatus[status]?.length) {
            parts.push(`\n*${status.replace("_", " ").toUpperCase()}:*`);
            byStatus[status].forEach((t) => {
              const priority = t.priority !== "medium" ? ` [${t.priority}]` : "";
              const project = t.project_prefix ? ` (${t.project_prefix})` : "";
              parts.push(`- ${t.content}${priority}${project}`);
            });
          }
        });

        return {
          messages: [
            {
              role: "user",
              content: { type: "text", text: parts.join("\n") },
            },
          ],
        };
      }

      case "recent_memories": {
        // Guard against a non-numeric or non-positive `count` arg (parseInt → NaN
        // would serialize to JSON null and silently fall back to the server default).
        const countRaw = parseInt((args.count as string) || "10", 10);
        const count = Number.isFinite(countRaw) && countRaw > 0 ? countRaw : 10;
        const result = await apiCall<{ memories: Memory[] }>("/api/memories", "POST", {
          user_id: USER_ID,
          limit: count,
        });
        const memories = result.memories || [];
        if (memories.length === 0) {
          return {
            messages: [
              {
                role: "user",
                content: { type: "text", text: "No memories found." },
              },
            ],
          };
        }

        const parts: string[] = [`**${memories.length} Recent Memories:**\n`];
        memories.forEach((m) => {
          const content = getContent(m);
          const type = getType(m);
          const preview = renderContent(content, m.id, MEMORY_PREVIEW_MAX, false);
          parts.push(`- [${type}] ${preview}`);
        });

        return {
          messages: [
            {
              role: "user",
              content: { type: "text", text: parts.join("\n") },
            },
          ],
        };
      }

      case "memory_health": {
        // Only fields GET /api/users/{id}/stats actually returns. It has no
        // recency counters and no type histogram: the previous version read
        // memories_last_24h / memories_last_7d / memories_by_type, which do not
        // exist, and printed "Last 24h: 0 / Last 7 days: 0" as though measured.
        const statsResult = await apiCall<{
          total_memories: number;
          working_memory_count?: number;
          session_memory_count?: number;
          long_term_memory_count?: number;
          vector_index_count?: number;
          average_importance?: number;
          total_retrievals?: number;
          graph_nodes?: number;
          graph_edges?: number;
        }>(`/api/users/${encodeURIComponent(USER_ID)}/stats`, "GET");

        const verifyResult = await apiCall<{
          is_healthy: boolean;
          orphaned_count: number;
        }>("/api/index/verify", "POST", { user_id: USER_ID });

        const parts: string[] = ["**Memory System Health:**\n"];
        parts.push(`Total memories: ${statsResult.total_memories || 0}`);
        // Buffer occupancy, not a tier partition — see the note in memory_stats.
        parts.push(
          `In working buffer: ${statsResult.working_memory_count ?? 0} │ ` +
            `in session buffer: ${statsResult.session_memory_count ?? 0} │ ` +
            `persisted: ${statsResult.long_term_memory_count ?? 0}`,
        );
        parts.push(`Indexed vectors: ${statsResult.vector_index_count ?? 0}`);
        parts.push(`Graph: ${statsResult.graph_nodes ?? 0} nodes │ ${statsResult.graph_edges ?? 0} edges`);
        if (typeof statsResult.average_importance === "number") {
          parts.push(`Avg importance: ${statsResult.average_importance.toFixed(2)}`);
        }
        if (typeof statsResult.total_retrievals === "number") {
          parts.push(`Total retrievals: ${statsResult.total_retrievals}`);
        }
        parts.push(`\nIndex status: ${verifyResult.is_healthy ? "✓ Healthy" : "⚠ Needs repair"}`);
        if (verifyResult.orphaned_count > 0) {
          parts.push(`Orphaned entries: ${verifyResult.orphaned_count}`);
        }
        parts.push("\nFor a breakdown by memory type, call list_memories.");

        return {
          messages: [
            {
              role: "user",
              content: { type: "text", text: parts.join("\n") },
            },
          ],
        };
      }

      default:
        return {
          messages: [
            {
              role: "user",
              content: { type: "text", text: `Unknown prompt: ${promptName}` },
            },
          ],
        };
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      messages: [
        {
          role: "user",
          content: { type: "text", text: `Error: ${message}` },
        },
      ],
    };
  }
});

// =============================================================================
// RESOURCE TEMPLATES - Pattern-based resource access
// =============================================================================

server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => {
  return {
    resourceTemplates: [
      {
        uriTemplate: "memory://{id}",
        name: "Memory by ID",
        description: "Access a specific memory by its ID",
        mimeType: "text/plain",
      },
      {
        uriTemplate: "shodh://stats",
        name: "Memory Statistics",
        description: "Current memory system statistics",
        mimeType: "application/json",
      },
      {
        uriTemplate: "shodh://todos",
        name: "Todo List",
        description: "Your pending tasks and work items",
        mimeType: "text/plain",
      },
      {
        uriTemplate: "shodh://search/{query}",
        name: "Search Memories",
        description: "Search memories for a specific query",
        mimeType: "text/plain",
      },
    ],
  };
});

// =============================================================================
// AUTO-SPAWN SERVER - Automatically start backend if not running
// =============================================================================

// Disable auto-spawn with SHODH_NO_AUTO_SPAWN=true
const AUTO_SPAWN_ENABLED = process.env.SHODH_NO_AUTO_SPAWN !== "true";

let serverProcess: ChildProcess | null = null;

function getBinaryPath(): string | null {
  const platform = process.platform;
  const binDir = path.join(__dirname, "..", "bin");

  // Use wrapper script that sets up library paths for bundled ONNX Runtime
  let wrapperName: string;
  let fallbackName: string;
  if (platform === "win32") {
    wrapperName = "shodh-memory.bat";
    fallbackName = "shodh-memory-server.exe";
  } else {
    wrapperName = "shodh-memory";
    fallbackName = "shodh-memory-server";
  }

  // Prefer direct binary (avoids spawn EINVAL with .bat + detached on Windows)
  const binaryPath = path.join(binDir, fallbackName);
  if (fs.existsSync(binaryPath)) {
    return binaryPath;
  }

  // Fallback to wrapper script (includes ONNX Runtime setup)
  const wrapperPath = path.join(binDir, wrapperName);
  if (fs.existsSync(wrapperPath)) {
    return wrapperPath;
  }

  return null;
}

function getWindowsIpcHelper(): WindowsIpcHelper | undefined {
  if (process.platform !== "win32") return undefined;
  const command = getBinaryPath();
  return command ? { command, args: ["ipc-exchange"] } : undefined;
}

async function isServerRunning(): Promise<boolean> {
  return isServerAvailable();
}

async function waitForServer(maxAttempts: number = 30): Promise<boolean> {
  for (let i = 0; i < maxAttempts; i++) {
    if (await isServerRunning()) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

// Outcome of an authenticated probe. "rejected" means the server answered and
// refused the key; "inconclusive" means we never got an authoritative answer
// (timeout, connection reset, 5xx). Collapsing the two would let a slow-starting
// backend be reported as a bad key, sending users to delete a key file that was
// never the problem.
type KeyProbe = "accepted" | "rejected" | "inconclusive";

/** Probe an authenticated endpoint — /health is public, so it proves nothing about auth. */
async function probeApiKey(): Promise<KeyProbe> {
  try {
    await backendRequest("/api/users", "GET", undefined, 3000);
    return "accepted";
  } catch (err) {
    const status = /API error (\d+)/.exec(err instanceof Error ? err.message : String(err));
    if (status) {
      const code = parseInt(status[1], 10);
      // Any answered status other than 401/403 means the key got past auth.
      return code === 401 || code === 403 ? "rejected" : "accepted";
    }
    return "inconclusive";
  }
}

/** Report a rejected key with the fix that matches how this shim got its key. */
function reportKeyRejected(): void {
  console.error("[shodh-memory] ERROR: the server rejected our API key (401/403).");
  console.error("[shodh-memory] All memory operations will fail until this is fixed.");
  if (apiKeyFile) {
    console.error(`[shodh-memory] Fix: stop the server, delete ${apiKeyFile}, and reconnect,`);
    console.error("[shodh-memory] or set SHODH_API_KEY to the key the server was started with.");
  } else {
    console.error("[shodh-memory] Fix: set SHODH_API_KEY to the key the server was started with.");
  }
}

async function ensureServerRunning(): Promise<void> {
  // Check if already running
  if (await isServerRunning()) {
    console.error("[shodh-memory] Backend server already running at", BACKEND_LOCATION);
    // If our key wasn't explicitly configured, verify it works against the
    // running server — /health is unauthenticated, so reachability alone
    // proves nothing about auth.
    if (!process.env.SHODH_API_KEY && isLocalServer()) {
      if (await probeApiKey() === "rejected") {
        console.error("[shodh-memory] The server was started with a different API key.");
        reportKeyRejected();
      }
    }
    return;
  }

  if (!AUTO_SPAWN_ENABLED) {
    console.error("[shodh-memory] Server not running at", BACKEND_LOCATION);
    console.error("[shodh-memory] Auto-spawn disabled (SHODH_NO_AUTO_SPAWN=true).");
    console.error("[shodh-memory] Start the server manually:");
    console.error("[shodh-memory]   shodh-memory-server");
    console.error("[shodh-memory] Or with Docker:");
    console.error("[shodh-memory]   docker run -d -p 3030:3030 roshera/shodh-memory");
    return;
  }

  const binaryPath = getBinaryPath();
  if (!binaryPath) {
    console.error("[shodh-memory] Server binary not found. Please run: npx @shodh/memory-mcp");
    console.error("[shodh-memory] Or download from: https://github.com/varun29ankuS/shodh-memory/releases");
    return;
  }

  // Validate that the resolved binary is within the expected bin directory
  const expectedBinDir = fs.realpathSync(path.join(__dirname, "..", "bin"));
  const resolvedBinary = fs.realpathSync(binaryPath);
  if (!resolvedBinary.startsWith(expectedBinDir + path.sep) && resolvedBinary !== expectedBinDir) {
    console.error(`[shodh-memory] WARNING: Binary path resolves outside expected directory: ${resolvedBinary}`);
    console.error(`[shodh-memory] Expected: ${expectedBinDir}`);
    return;
  }

  console.error("[shodh-memory] Starting backend server...");

  // Build a clean environment for the server process.
  // Only pass through system env + server-relevant SHODH_ vars.
  // MCP-client-specific vars (SHODH_RATE_LIMIT, SHODH_TOKEN_BUDGET, etc.)
  // must NOT leak to the server — they have different semantics.
  const serverEnv: Record<string, string> = {};
  const SERVER_ENV_ALLOWLIST = new Set([
    "SHODH_HOST", "SHODH_PORT", "SHODH_MEMORY_PATH", "SHODH_ENV",
    "SHODH_API_KEYS", "SHODH_DEV_API_KEY", "SHODH_MAX_USERS",
    "SHODH_RATE_LIMIT", "SHODH_RATE_BURST", "SHODH_MAX_CONCURRENT",
    "SHODH_REQUEST_TIMEOUT", "SHODH_WRITE_MODE", "SHODH_OFFLINE",
    "SHODH_LAZY_LOAD", "SHODH_ONNX_THREADS", "SHODH_VECTOR_BACKEND",
    "SHODH_CORS_ORIGINS", "SHODH_CORS_MAX_AGE", "SHODH_CORS_CREDENTIALS",
    "SHODH_IPC_ENABLED", "SHODH_IPC_ENDPOINT", "SHODH_IPC_REQUIRED",
    "RUST_LOG",
  ]);
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    if (key.startsWith("SHODH_")) {
      // Only pass through env vars the server actually understands
      if (SERVER_ENV_ALLOWLIST.has(key)) {
        serverEnv[key] = value;
      }
    } else {
      // Pass through all non-SHODH env vars (PATH, HOME, etc.)
      serverEnv[key] = value;
    }
  }
  if (IPC_ENDPOINT) {
    serverEnv["SHODH_IPC_ENDPOINT"] = IPC_ENDPOINT;
  } else {
    delete serverEnv["SHODH_IPC_ENDPOINT"];
  }
  // Always pass the API key for auth
  serverEnv["SHODH_DEV_API_KEY"] = API_KEY;

  // Spawn the server process (.bat files need shell: true on Windows)
  const isBat = binaryPath.endsWith(".bat");
  serverProcess = spawn(binaryPath, [], {
    detached: true,
    stdio: "ignore",
    env: serverEnv,
    ...(isBat && { shell: true }),
  });

  serverProcess.unref();

  // Record the spawned backend's pid so whichever shim exits LAST can reap it.
  // Recorded before the health wait: even a slow-starting backend must be
  // reapable, and recordSpawnedServer never clobbers a live sibling's record.
  if (serverProcess.pid) {
    try {
      recordSpawnedServer(shodhDataRoot(), serverProcess.pid);
    } catch (err) {
      console.error("[shodh-memory] Warning: could not record backend pidfile:", err instanceof Error ? err.message : err);
    }
  }

  // Wait for server to become available
  console.error("[shodh-memory] Waiting for server to start...");
  const started = await waitForServer();

  if (started) {
    console.error("[shodh-memory] Backend server started successfully");
    // Validate auth against the freshly spawned server. /health is public, so
    // a healthy server can still reject our key (e.g. a concurrent shim's
    // spawn won the port with a different key, or a stale persisted key).
    if (await probeApiKey() === "rejected") {
      reportKeyRejected();
    }
  } else {
    console.error("[shodh-memory] Warning: Server may not have started properly");
  }
}

// -----------------------------------------------------------------------------
// Backend recovery — re-run ensureServerRunning when a health check fails
// -----------------------------------------------------------------------------
// Concurrent tool calls share one in-flight recovery attempt, and a failed
// attempt is not retried for a cooldown window so a dead backend doesn't cost
// every tool call a full spawn-and-wait cycle.
let recoveryInFlight: Promise<void> | null = null;
let lastFailedRecoveryAt = 0;
const RECOVERY_COOLDOWN_MS = 30_000;

async function recoverBackend(): Promise<boolean> {
  if (Date.now() - lastFailedRecoveryAt < RECOVERY_COOLDOWN_MS) return false;
  if (!recoveryInFlight) {
    recoveryInFlight = ensureServerRunning()
      .catch((err) => {
        console.error("[shodh-memory] Backend restart attempt failed:", err instanceof Error ? err.message : err);
      })
      .finally(() => {
        recoveryInFlight = null;
      });
  }
  await recoveryInFlight;
  const up = await isServerAvailable();
  if (!up) {
    lastFailedRecoveryAt = Date.now();
  }
  return up;
}

// True once this shim registered itself in the shared shim pidfile directory.
let shimRegistered = false;
// Guard so the exit path releases the shared backend exactly once
// (process.on("exit") and signal handlers can both invoke cleanup).
let backendReleased = false;

// Kill an auto-spawned backend by pid. On POSIX the backend was spawned
// detached (its own process group, pgid == pid), so kill the group first.
function killBackendPid(pid: number): void {
  if (process.platform !== "win32") {
    try {
      process.kill(-pid, "SIGTERM");
      return;
    } catch (e) {
      console.error("[Cleanup] Process group kill failed, falling back to direct kill:", e);
    }
  }
  try { process.kill(pid, "SIGTERM"); } catch (_) { /* already gone */ }
}

// Release this shim's reference to the shared backend. The backend is only
// killed when (a) it was auto-spawned by a shim (pidfile exists) and (b) no
// other live shim remains — a shim exiting mid-session must never take the
// backend away from siblings that are still using it.
function releaseSharedBackend(): void {
  if (backendReleased) return;
  backendReleased = true;

  try {
    if (shimRegistered) {
      unregisterShim(shodhDataRoot());
      shimRegistered = false;
    }
    const pidToReap = backendPidToReap(shodhDataRoot());
    if (pidToReap !== null) {
      console.error(`[shodh-memory] Last shim exiting — stopping auto-spawned backend (pid ${pidToReap})`);
      killBackendPid(pidToReap);
      clearSpawnedServer(shodhDataRoot());
    }
  } catch (err) {
    // Pidfile bookkeeping failed (e.g. data dir vanished). Fall back to the
    // legacy behaviour for our own child only, so we never leak a process we
    // spawned ourselves.
    console.error("[Cleanup] Backend refcount failed, falling back to own-child kill:", err instanceof Error ? err.message : err);
    if (serverProcess && !serverProcess.killed && serverProcess.pid) {
      killBackendPid(serverProcess.pid);
    }
  }
}

// Graceful shutdown helper — tears down ALL event loop references
function cleanupServer() {
  // 1. Stop WebSocket reconnect loop (prevents event loop from staying alive)
  if (streamReconnectTimer) {
    clearTimeout(streamReconnectTimer);
    streamReconnectTimer = null;
  }
  STREAM_ENABLED = false; // prevent further reconnect attempts

  // 2. Close WebSocket explicitly
  if (streamSocket) {
    try { streamSocket.close(); } catch (_) { /* ignore */ }
    streamSocket = null;
  }

  // 3. Release the shared backend (kills it only if we are the last live shim)
  releaseSharedBackend();
}

// Cleanup on exit
process.on("exit", cleanupServer);

// Handle signals for clean shutdown
process.on("SIGINT", () => {
  console.error("[shodh-memory] Received SIGINT, shutting down...");
  cleanupServer();
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.error("[shodh-memory] Received SIGTERM, shutting down...");
  cleanupServer();
  process.exit(0);
});

// Guard against multiple shutdown calls (end + close can both fire)
let shuttingDown = false;
function gracefulShutdown(reason: string, code: number = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.error(`[shodh-memory] ${reason}`);
  cleanupServer();
  // Brief grace period for any in-flight stdout writes to flush
  setTimeout(() => process.exit(code), 100);
}

// Detect MCP session end via stdin close (host closed pipe).
// This is the primary shutdown signal from MCP hosts like kiro-cli, Cursor, etc.
// stdin "end" fires when EOF is read (host closed write end of pipe).
// stdin "close" fires when the underlying resource is freed.
// Both are terminal — the host is gone, there's no session to evict.
// Issue #405: hosts close stdin (not stdout) on a thread switch. Route the EOF
// through the drain controller so any in-flight tool call finishes and its
// response is written to the still-open stdout before we exit.
process.stdin.on("end", () => drain.onStdinClose("stdin closed (MCP session ended), shutting down..."));
process.stdin.on("close", () => drain.onStdinClose("stdin pipe closed, shutting down..."));

// If stdout dies while we are mid-drain there is nothing left to deliver to —
// abandon in-flight calls and exit instead of waiting out the grace window.
// Outside a drain, a stdout error means the host is gone: preserve the prior
// uncaught-EPIPE behaviour (log + exit) rather than silently swallowing it.
process.stdout.on("error", (err) => {
  if (drain.isDraining) {
    drain.onOutputLost("stdout errored during drain, shutting down...");
  } else {
    console.error("[shodh-memory] stdout error:", err);
    gracefulShutdown("stdout errored, shutting down...", 1);
  }
});
process.stdout.on("close", () => drain.onOutputLost("stdout closed during drain, shutting down..."));

// Catch unhandled errors to ensure cleanup runs
process.on("uncaughtException", (err) => {
  console.error("[shodh-memory] Uncaught exception:", err);
  gracefulShutdown("Shutting down after uncaught exception", 1);
});

process.on("unhandledRejection", (reason) => {
  console.error("[shodh-memory] Unhandled rejection:", reason);
  gracefulShutdown("Shutting down after unhandled rejection", 1);
});

// Smithery sandbox export — allows tool scanning without a running backend
export function createSandboxServer() {
  process.env.SMITHERY_SANDBOX = "true";
  return server;
}

// Start server
async function main() {
  if (SANDBOX_MODE) return;

  // Register this shim as a live user of the shared backend BEFORE any spawn,
  // so a sibling shim exiting right now sees us and leaves the backend alive.
  try {
    registerShim(shodhDataRoot());
    shimRegistered = true;
  } catch (err) {
    console.error("[shodh-memory] Warning: could not register shim pidfile:", err instanceof Error ? err.message : err);
  }

  // Ensure backend is running
  await ensureServerRunning();

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`Shodh-Memory MCP server v${SERVER_VERSION} running`);
  console.error(`Connecting to: ${BACKEND_LOCATION}`);
  console.error(`User ID: ${describeUserId(USER_ID)}`);
  console.error(`Streaming: ${STREAM_ENABLED ? "enabled" : "disabled"}`);
  console.error(`Proactive surfacing: ${PROACTIVE_SURFACING ? "enabled" : "disabled (SHODH_PROACTIVE=false)"}`);
}

main().catch(console.error);
