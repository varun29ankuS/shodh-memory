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
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const API_URL = process.env.SHODH_API_URL || "http://127.0.0.1:3030";
const WS_URL = API_URL.replace(/^http/, "ws") + "/api/stream";
const USER_ID = process.env.SHODH_USER_ID || "claude-code";

// API Key - required for authentication
// Set via SHODH_API_KEY env var, or configure in MCP settings
const API_KEY = process.env.SHODH_API_KEY;
if (!API_KEY) {
  console.error("ERROR: SHODH_API_KEY environment variable not set.");
  console.error("");
  console.error("To fix, add to your MCP config (claude_desktop_config.json or mcp.json):");
  console.error(`  "env": { "SHODH_API_KEY": "your-api-key" }`);
  console.error("");
  console.error("Or set in your shell:");
  console.error("  export SHODH_API_KEY=your-api-key");
  console.error("");
  console.error("For local development, use the same key set in SHODH_DEV_API_KEY on the server.");
  process.exit(1);
}
const RETRY_ATTEMPTS = 3;
const RETRY_DELAY_MS = 1000;
const REQUEST_TIMEOUT_MS = 10000;

// =============================================================================
// TOKEN TRACKING - Context window awareness (SHO-115)
// =============================================================================

// Token budget configuration (default 100k tokens, ~400k chars)
const TOKEN_BUDGET = parseInt(process.env.SHODH_TOKEN_BUDGET || "100000", 10);
const ALERT_THRESHOLD = parseFloat(process.env.SHODH_ALERT_THRESHOLD || "0.9");

// Session token tracking
let sessionTokens = 0;
let sessionStartTime = Date.now();

// Simple token estimation: ~4 chars per token (rough approximation)
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// Get current token status
function getTokenStatus(): { tokens: number; budget: number; percent: number; alert: string | null } {
  const percent = sessionTokens / TOKEN_BUDGET;
  return {
    tokens: sessionTokens,
    budget: TOKEN_BUDGET,
    percent: Math.round(percent * 100) / 100,
    alert: percent >= ALERT_THRESHOLD ? `context_${Math.round(ALERT_THRESHOLD * 100)}_percent` : null,
  };
}

// Reset session (call on new conversation or explicit clear)
function resetTokenSession(): void {
  sessionTokens = 0;
  sessionStartTime = Date.now();
}

// Streaming ingestion settings
const STREAM_ENABLED = process.env.SHODH_STREAM !== "false"; // enabled by default
const STREAM_MIN_CONTENT_LENGTH = 50; // minimum content length to stream

// Proactive surfacing settings
// When enabled, relevant memories are automatically surfaced with tool responses
const PROACTIVE_SURFACING = process.env.SHODH_PROACTIVE !== "false"; // enabled by default
const PROACTIVE_MIN_CONTEXT_LENGTH = 30; // minimum context length to trigger surfacing

// =============================================================================
// STREAMING MEMORY INGESTION - Continuous background memory capture
// =============================================================================

let streamSocket: WebSocket | null = null;
let streamConnecting = false;
let streamReconnectTimer: ReturnType<typeof setTimeout> | null = null;

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
    // Note: /api/stream is public (no auth required), so no headers needed
    // Bun supports headers via: new WebSocket(url, { headers: {...} })
    streamSocket = new WebSocket(WS_URL);

    streamSocket.onopen = () => {
      streamConnecting = false;
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
      console.error("[Stream] Sent handshake for user:", USER_ID);
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
      } catch {
        // Ignore parse errors
      }
    };

    streamSocket.onclose = (event) => {
      console.error("[Stream] WebSocket closed:", event.code, event.reason || "(no reason)");
      streamSocket = null;
      streamConnecting = false;
      streamHandshakeComplete = false;
      // Reconnect after delay
      if (STREAM_ENABLED && !streamReconnectTimer) {
        streamReconnectTimer = setTimeout(() => {
          streamReconnectTimer = null;
          console.error("[Stream] Attempting reconnect...");
          connectStream().catch(() => {});
        }, 5000);
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
    // Buffer message and try to reconnect
    if (streamBuffer.length < MAX_BUFFER_SIZE) {
      streamBuffer.push(message);
      console.error(`[Stream] Buffered memory (socket not ready, buffer size: ${streamBuffer.length})`);
    }
    connectStream().catch(() => {});
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
console.error("[Stream] Initializing connection to", WS_URL);
connectStream().catch((err) => {
  console.error("[Stream] Initial connection failed:", err);
});

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

// Surface relevant memories based on context (non-blocking, returns null on failure)
async function surfaceRelevant(context: string, maxResults: number = 3): Promise<SurfacedMemory[] | null> {
  if (!PROACTIVE_SURFACING || context.length < PROACTIVE_MIN_CONTEXT_LENGTH) {
    return null;
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000); // 3s timeout for surfacing

    const response = await fetch(`${API_URL}/api/relevant`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify({
        user_id: USER_ID,
        context: context.slice(0, 2000),
        config: {
          semantic_threshold: 0.65,
          max_results: maxResults,
        },
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) return null;

    const result = await response.json() as { memories?: SurfacedMemory[] };
    return result.memories || null;
  } catch {
    return null;
  }
}

// Format surfaced memories for inclusion in tool response
function formatSurfacedMemories(memories: SurfacedMemory[]): string {
  if (!memories || memories.length === 0) return "";

  const formatted = memories
    .map((m, i) => `  ${i + 1}. [${(m.relevance_score * 100).toFixed(0)}%] ${m.content.slice(0, 80)}...`)
    .join("\n");

  return `\n\n[Relevant memories surfaced]\n${formatted}`;
}

// Stream tool interactions automatically (non-blocking)
function streamToolCall(toolName: string, args: Record<string, unknown>, resultText: string): void {
  // Skip ingesting memory management tools to avoid noise
  if (["remember", "recall", "forget", "list_memories"].includes(toolName)) return;

  const argsStr = JSON.stringify(args, null, 2);
  const content = `Tool: ${toolName}\nInput: ${argsStr}\nResult: ${resultText.slice(0, 1000)}${resultText.length > 1000 ? "..." : ""}`;

  streamMemory(content, ["tool-call", toolName], "tool");
}

// Robust API call with retries and timeout
async function apiCall<T>(
  endpoint: string,
  method: string = "GET",
  body?: object
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= RETRY_ATTEMPTS; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

      const options: RequestInit = {
        method,
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY,
        },
        signal: controller.signal,
      };

      if (body) {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(`${API_URL}${endpoint}`, options);
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        throw new Error(`API error ${response.status}: ${errorText}`);
      }

      return await response.json() as T;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on client errors (4xx)
      if (lastError.message.includes('API error 4')) {
        throw lastError;
      }

      // Log retry attempt
      if (attempt < RETRY_ATTEMPTS) {
        console.error(`Attempt ${attempt} failed: ${lastError.message}. Retrying in ${RETRY_DELAY_MS}ms...`);
        await sleep(RETRY_DELAY_MS * attempt); // Exponential backoff
      }
    }
  }

  throw new Error(`Failed after ${RETRY_ATTEMPTS} attempts: ${lastError?.message || 'Unknown error'}`);
}

// Check if server is available
async function isServerAvailable(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const response = await fetch(`${API_URL}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    return response.ok;
  } catch {
    return false;
  }
}

// Create MCP server
const server = new Server(
  {
    name: "shodh-memory",
    version: "0.1.61",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
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
          },
          required: ["content"],
        },
      },
      {
        name: "recall",
        description: "Search memories using different retrieval modes. Use this to find relevant past experiences, decisions, or context. Modes: 'semantic' (vector similarity), 'associative' (graph traversal - follows learned connections between memories), 'hybrid' (combines both with density-dependent weighting).",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Natural language search query",
            },
            limit: {
              type: "number",
              description: "Maximum number of results (default: 5)",
              default: 5,
            },
            mode: {
              type: "string",
              enum: ["semantic", "associative", "hybrid"],
              description: "Retrieval mode: 'semantic' for pure vector similarity, 'associative' for graph-based traversal (follows learned connections), 'hybrid' for density-dependent combination (default)",
              default: "hybrid",
            },
          },
          required: ["query"],
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
      {
        name: "recall_by_tags",
        description: "Search memories by tags. Returns memories matching ANY of the provided tags.",
        inputSchema: {
          type: "object",
          properties: {
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Tags to search for (returns memories matching ANY tag)",
            },
            limit: {
              type: "number",
              description: "Maximum number of results (default: 20)",
              default: 20,
            },
          },
          required: ["tags"],
        },
      },
      {
        name: "recall_by_date",
        description: "Search memories within a date range.",
        inputSchema: {
          type: "object",
          properties: {
            start: {
              type: "string",
              description: "Start date (ISO 8601 format, e.g., '2024-01-01T00:00:00Z')",
            },
            end: {
              type: "string",
              description: "End date (ISO 8601 format, e.g., '2024-12-31T23:59:59Z')",
            },
            limit: {
              type: "number",
              description: "Maximum number of results (default: 20)",
              default: 20,
            },
          },
          required: ["start", "end"],
        },
      },
      {
        name: "forget_by_tags",
        description: "Delete memories matching ANY of the provided tags.",
        inputSchema: {
          type: "object",
          properties: {
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Tags to match for deletion",
            },
          },
          required: ["tags"],
        },
      },
      {
        name: "forget_by_date",
        description: "Delete memories within a date range.",
        inputSchema: {
          type: "object",
          properties: {
            start: {
              type: "string",
              description: "Start date (ISO 8601 format)",
            },
            end: {
              type: "string",
              description: "End date (ISO 8601 format)",
            },
          },
          required: ["start", "end"],
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
        description: "REQUIRED: Call this tool with EVERY user message to surface relevant memories and build conversation history. Pass the user's message as context. This enables: (1) retrieving memories relevant to what the user is asking, (2) building persistent memory of the conversation for future sessions. The system analyzes entities, semantic similarity, and recency to find contextually appropriate memories. Auto-ingest stores the context automatically. USAGE: Always call this FIRST when you receive a user message, passing their message as the context parameter.",
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
          },
          required: ["context"],
        },
      },
      {
        name: "streaming_status",
        description: "Check the status of WebSocket streaming connection. Use this to diagnose if streaming memory ingestion is working.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "token_status",
        description: "Get current token usage status for this session. Returns tokens used, budget remaining, and percentage consumed. Use this to check context window health.",
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
          },
          required: ["content"],
        },
      },
      {
        name: "list_todos",
        description: "List todos with GTD filters. Returns Linear-style formatted output grouped by status.",
        inputSchema: {
          type: "object",
          properties: {
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
        description: "Create a new project to group todos.",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "Project name",
            },
            description: {
              type: "string",
              description: "Project description",
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
    ],
  };
});

// Auto-stream context from tool arguments (captures conversation intent)
function autoStreamContext(toolName: string, args: Record<string, unknown>): void {
  // Skip tools that already handle their own streaming or are meta/diagnostic
  if (["proactive_context", "streaming_status", "token_status", "reset_token_session"].includes(toolName)) return;

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
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  // Ensure streaming is connected (lazy reconnect on tool calls)
  if (STREAM_ENABLED && (!streamSocket || streamSocket.readyState !== WebSocket.OPEN)) {
    connectStream().catch(() => {});
  }

  // Auto-capture context from tool arguments (non-blocking)
  autoStreamContext(name, args as Record<string, unknown>);

  // Check server availability first
  const serverUp = await isServerAvailable();
  if (!serverUp) {
    return {
      content: [
        {
          type: "text",
          text: `Memory server unavailable at ${API_URL}. Please ensure shodh-memory-server is running.\n\nTo start: cd shodh-memory && cargo run`,
        },
      ],
      isError: true,
    };
  }

  // Result type for tool responses
  type ToolResult = { content: { type: string; text: string }[]; isError?: boolean };

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
        };

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
        });

        // Format response with branded display
        let response = `🐘 Memory Stored\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `📝 ${content.slice(0, 60)}${content.length > 60 ? '...' : ''}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Type: ${type}`;
        if (tags.length > 0) {
          response += ` │ Tags: ${tags.join(', ')}`;
        }
        response += `\nID: ${result.id.slice(0, 8)}...`;

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "recall": {
        const { query, limit = 5, mode = "hybrid" } = args as { query: string; limit?: number; mode?: string };

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
        }

        interface RecallResponse {
          memories: Memory[];
          count: number;
          retrieval_stats?: RetrievalStats;
        }

        const result = await apiCall<RecallResponse>("/api/recall", "POST", {
          user_id: USER_ID,
          query,
          limit,
          mode,
        });

        const memories = result.memories || [];
        const stats = result.retrieval_stats;

        if (memories.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: `🐘 No memories found for: "${query}"\n   Mode: ${mode}`,
              },
            ],
          };
        }

        // Build formatted response
        let response = `🐘 Recalled ${memories.length} Memories\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Query: "${query.slice(0, 40)}${query.length > 40 ? '...' : ''}" │ Mode: ${mode}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        for (let i = 0; i < memories.length; i++) {
          const m = memories[i];
          const content = getContent(m);
          const score = ((m.score || 0) * 100).toFixed(0);
          const matchBar = '█'.repeat(Math.round((m.score || 0) * 10)) + '░'.repeat(10 - Math.round((m.score || 0) * 10));

          response += `${i + 1}. ${matchBar} ${score}%\n`;
          response += `   ${content.slice(0, 70)}${content.length > 70 ? '...' : ''}\n`;
          response += `   ┗━ ${getType(m)} │ ${m.id.slice(0, 8)}...\n`;
          if (i < memories.length - 1) response += `\n`;
        }

        // Build stats summary for associative/hybrid modes
        if (stats && (mode === "associative" || mode === "hybrid")) {
          response += `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `📊 Retrieval Stats\n`;
          const graphPct = (stats.graph_weight * 100).toFixed(0);
          const semPct = (stats.semantic_weight * 100).toFixed(0);
          response += `   Graph: ${graphPct}% │ Semantic: ${semPct}% │ Density: ${stats.graph_density.toFixed(2)}\n`;
          response += `   Candidates: ${stats.graph_candidates} graph + ${stats.semantic_candidates} semantic\n`;
          response += `   Entities: ${stats.entities_activated} │ Time: ${(stats.retrieval_time_us / 1000).toFixed(1)}ms`;
        }

        return {
          content: [{ type: "text", text: response }],
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
            response += `   • ${getContent(m).slice(0, 70)}${getContent(m).length > 70 ? '...' : ''}\n`;
          }
          response += `\n`;
        }

        if (include_decisions && decisions.length > 0) {
          response += `📋 DECISIONS\n`;
          for (const m of decisions.slice(0, max_items)) {
            response += `   • ${getContent(m).slice(0, 70)}${getContent(m).length > 70 ? '...' : ''}\n`;
          }
          response += `\n`;
        }

        if (include_learnings && learnings.length > 0) {
          response += `💡 LEARNINGS\n`;
          for (const m of learnings.slice(0, max_items)) {
            response += `   • ${getContent(m).slice(0, 70)}${getContent(m).length > 70 ? '...' : ''}\n`;
          }
          response += `\n`;
        }

        if (patterns.length > 0) {
          response += `🔄 PATTERNS\n`;
          for (const m of patterns.slice(0, max_items)) {
            response += `   • ${getContent(m).slice(0, 70)}${getContent(m).length > 70 ? '...' : ''}\n`;
          }
          response += `\n`;
        }

        if (errors.length > 0) {
          response += `⚠️ ERRORS TO AVOID\n`;
          for (const m of errors.slice(0, Math.min(3, max_items))) {
            response += `   • ${getContent(m).slice(0, 70)}${getContent(m).length > 70 ? '...' : ''}\n`;
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

        if (memories.length === 0) {
          let response = `🐘 Memory List\n`;
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `No memories stored yet.`;
          return {
            content: [{ type: "text", text: response }],
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

          response += `${String(i + 1).padStart(2)}. ${typeIcon} ${content.slice(0, 60)}${content.length > 60 ? '...' : ''}\n`;
          response += `    ┗━ ${getType(m)} │ ${m.id.slice(0, 8)}...\n`;
        }

        return {
          content: [{ type: "text", text: response.trimEnd() }],
        };
      }

      case "forget": {
        const { id } = args as { id: string };

        await apiCall(`/api/memory/${id}?user_id=${USER_ID}`, "DELETE");

        let response = `🐘 Memory Deleted\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `✓ Removed: ${id.slice(0, 8)}...`;

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "memory_stats": {
        interface MemoryStats {
          total_memories: number;
          memory_types: Record<string, number>;
          total_importance: number;
          avg_importance: number;
          graph_nodes: number;
          graph_edges: number;
          indexed_vectors: number;
        }

        const result = await apiCall<MemoryStats>(`/api/users/${USER_ID}/stats`, "GET");

        let response = `🐘 Memory Statistics\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Total Memories: ${result.total_memories || 0}\n`;
        response += `Graph: ${result.graph_nodes || 0} nodes │ ${result.graph_edges || 0} edges\n`;
        response += `Indexed Vectors: ${result.indexed_vectors || 0}\n`;
        response += `Avg Importance: ${(result.avg_importance || 0).toFixed(2)}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        if (result.memory_types && Object.keys(result.memory_types).length > 0) {
          response += `\nBy Type:\n`;
          const typeIcons: Record<string, string> = {
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
          };
          for (const [type, count] of Object.entries(result.memory_types)) {
            const icon = typeIcons[type] || '📦';
            const bar = '█'.repeat(Math.min(20, Math.round((count as number) / (result.total_memories || 1) * 20)));
            response += `   ${icon} ${type.padEnd(12)} ${bar} ${count}\n`;
          }
        }

        return {
          content: [{ type: "text", text: response.trimEnd() }],
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

      case "recall_by_tags": {
        const { tags, limit = 20 } = args as { tags: string[]; limit?: number };

        const result = await apiCall<{ memories: Memory[]; count: number }>("/api/recall/tags", "POST", {
          user_id: USER_ID,
          tags,
          limit,
        });

        const memories = result.memories || [];

        if (memories.length === 0) {
          let response = `🐘 Recall by Tags\n`;
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `Tags: ${tags.map(t => `#${t}`).join(' ')}\n`;
          response += `No memories found.`;
          return {
            content: [{ type: "text", text: response }],
          };
        }

        let response = `🐘 Recall by Tags\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Tags: ${tags.map(t => `#${t}`).join(' ')} │ Found: ${memories.length}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        for (let i = 0; i < memories.length; i++) {
          const m = memories[i];
          const content = getContent(m);
          response += `${String(i + 1).padStart(2)}. ${content.slice(0, 60)}${content.length > 60 ? '...' : ''}\n`;
          response += `    ┗━ ${getType(m)} │ ${m.id.slice(0, 8)}...\n`;
        }

        return {
          content: [{ type: "text", text: response.trimEnd() }],
        };
      }

      case "recall_by_date": {
        const { start, end, limit = 20 } = args as { start: string; end: string; limit?: number };

        const result = await apiCall<{ memories: Memory[]; count: number }>("/api/recall/date", "POST", {
          user_id: USER_ID,
          start,
          end,
          limit,
        });

        const memories = result.memories || [];

        // Format date range for display
        const startDate = new Date(start).toLocaleDateString();
        const endDate = new Date(end).toLocaleDateString();

        if (memories.length === 0) {
          let response = `🐘 Recall by Date\n`;
          response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
          response += `Range: ${startDate} → ${endDate}\n`;
          response += `No memories found.`;
          return {
            content: [{ type: "text", text: response }],
          };
        }

        let response = `🐘 Recall by Date\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Range: ${startDate} → ${endDate} │ Found: ${memories.length}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

        for (let i = 0; i < memories.length; i++) {
          const m = memories[i];
          const content = getContent(m);
          const created = m.created_at ? new Date(m.created_at).toLocaleString() : 'unknown';
          response += `${String(i + 1).padStart(2)}. ${content.slice(0, 55)}${content.length > 55 ? '...' : ''}\n`;
          response += `    ┗━ ${getType(m)} │ ${created}\n`;
        }

        return {
          content: [{ type: "text", text: response.trimEnd() }],
        };
      }

      case "forget_by_tags": {
        const { tags } = args as { tags: string[] };

        const result = await apiCall<{ deleted_count: number }>("/api/forget/tags", "POST", {
          user_id: USER_ID,
          tags,
        });

        let response = `🐘 Forget by Tags\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Tags: ${tags.map(t => `#${t}`).join(' ')}\n`;
        response += `✓ Deleted: ${result.deleted_count} memories`;

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "forget_by_date": {
        const { start, end } = args as { start: string; end: string };

        const result = await apiCall<{ deleted_count: number }>("/api/forget/date", "POST", {
          user_id: USER_ID,
          start,
          end,
        });

        const startDate = new Date(start).toLocaleDateString();
        const endDate = new Date(end).toLocaleDateString();

        let response = `🐘 Forget by Date\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Range: ${startDate} → ${endDate}\n`;
        response += `✓ Deleted: ${result.deleted_count} memories`;

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
        } = args as {
          context: string;
          semantic_threshold?: number;
          entity_match_weight?: number;
          recency_weight?: number;
          max_results?: number;
          memory_types?: string[];
          auto_ingest?: boolean;
        };

        // Stream context to memory (non-blocking, uses WebSocket)
        if (auto_ingest && context.length > 100) {
          streamMemory(context.slice(0, 2000), ["proactive-context"]);
          // Flush immediately to ensure context is persisted (don't wait for checkpoint)
          streamFlush();
        }

        interface DetectedEntity {
          text: string;
          entity_type: string;
          confidence: number;
          matched_memories: number;
        }

        interface SurfacedMemory {
          id: string;
          content: string;
          memory_type: string;
          importance: number;
          relevance_score: number;
          relevance_reason: string;
          matched_entities: string[];
          semantic_similarity: number;
          created_at: string;
          tags: string[];
        }

        interface RelevanceResponse {
          memories: SurfacedMemory[];
          detected_entities: DetectedEntity[];
          latency_ms: number;
          config_used: {
            semantic_threshold: number;
            entity_match_weight: number;
            semantic_weight: number;
            recency_weight: number;
            max_results: number;
            recency_half_life_hours: number;
            memory_types: string[];
          };
        }

        const result = await apiCall<RelevanceResponse>("/api/relevant", "POST", {
          user_id: USER_ID,
          context,
          config: {
            semantic_threshold,
            entity_match_weight,
            semantic_weight: 1.0 - entity_match_weight - recency_weight,
            recency_weight,
            max_results,
            memory_types,
          },
        });

        const memories = result.memories || [];
        const entities = result.detected_entities || [];

        if (memories.length === 0) {
          // No relevant memories, but show detected entities if any
          if (entities.length > 0) {
            const entityList = entities
              .map(e => `  - "${e.text}" (${e.entity_type}, ${(e.confidence * 100).toFixed(0)}% confidence)`)
              .join('\n');
            return {
              content: [
                {
                  type: "text",
                  text: `No relevant memories surfaced for this context.\n\nDetected entities:\n${entityList}\n\n[Latency: ${result.latency_ms.toFixed(1)}ms]`,
                },
              ],
            };
          }
          return {
            content: [
              {
                type: "text",
                text: `No relevant memories surfaced for this context.\n\n[Latency: ${result.latency_ms.toFixed(1)}ms]`,
              },
            ],
          };
        }

        // Format surfaced memories
        const formatted = memories
          .map((m, i) => {
            const score = (m.relevance_score * 100).toFixed(0);
            const entityMatchStr = (m.matched_entities && m.matched_entities.length > 0)
              ? `\n   Entity matches: ${m.matched_entities.join(', ')}`
              : '';
            const semScore = (m.semantic_similarity * 100).toFixed(0);
            return `${i + 1}. [${score}% relevant] ${m.content.slice(0, 100)}${m.content.length > 100 ? '...' : ''}\n   Type: ${m.memory_type} | semantic=${semScore}% | reason: ${m.relevance_reason}${entityMatchStr}`;
          })
          .join("\n\n");

        // Format detected entities summary
        const entitySummary = entities.length > 0
          ? `\n\nDetected entities: ${entities.map(e => `"${e.text}" (${e.entity_type})`).join(', ')}`
          : '';

        // Check for due reminders and context-triggered reminders (SHO-116)
        let reminderBlock = "";
        try {
          // Check time-based due reminders
          interface ReminderItem {
            id: string;
            content: string;
            trigger_type: string;
            priority: number;
            overdue_seconds: number | null;
            due_at: string | null;
          }
          interface DueRemindersResponse {
            reminders: ReminderItem[];
            count: number;
          }

          const dueResult = await apiCall<DueRemindersResponse>("/api/reminders/due", "POST", {
            user_id: USER_ID,
            mark_triggered: false,  // Don't auto-trigger, let user dismiss manually
          });

          // Check context-triggered reminders
          const contextResult = await apiCall<DueRemindersResponse>("/api/reminders/context", "POST", {
            user_id: USER_ID,
            context,
            mark_triggered: false,  // Don't auto-trigger, let user dismiss manually
          });

          // Combine all triggered reminders
          const allReminders = [...dueResult.reminders, ...contextResult.reminders];
          const uniqueReminders = allReminders.filter((r, i, arr) =>
            arr.findIndex(x => x.id === r.id) === i
          );

          if (uniqueReminders.length > 0) {
            // 54 chars wide box
            reminderBlock = `\n\n`;
            reminderBlock += `🐘━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🧠\n`;
            reminderBlock += `┃  SHODH MEMORY                    REMINDERS (${String(uniqueReminders.length).padStart(2)})  ┃\n`;
            reminderBlock += `┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n`;

            for (const r of uniqueReminders) {
              const icon = r.overdue_seconds && r.overdue_seconds > 0 ? "⏰" : "📌";
              const contentText = r.content.slice(0, 38);
              reminderBlock += `┃  ${icon} ${contentText.padEnd(44)} [${r.id.slice(0,8)}] ┃\n`;

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
        } catch (e) {
          // Log reminder check failures for debugging
          console.error("[proactive_context] Reminder check failed:", e);
        }

        // Check for due todos (GTD integration)
        let todoBlock = "";
        try {
          interface DueTodoItem {
            id: string;
            content: string;
            status: string;
            priority: string;
            due_date: string | null;
            project_name: string | null;
          }
          interface DueTodosResponse {
            todos: DueTodoItem[];
            formatted: string;
            count: number;
          }

          const dueTodosResult = await apiCall<DueTodosResponse>("/api/todos/due", "POST", {
            user_id: USER_ID,
          });

          if (dueTodosResult.count > 0) {
            todoBlock = "\n\n" + dueTodosResult.formatted;
          }
        } catch (e) {
          // Log todo check failures for debugging
          console.error("[proactive_context] Todo check failed:", e);
        }

        return {
          content: [
            {
              type: "text",
              text: `Surfaced ${memories.length} relevant memories:\n\n${formatted}${entitySummary}${reminderBlock}${todoBlock}\n\n[Latency: ${result.latency_ms.toFixed(1)}ms | Threshold: ${(semantic_threshold * 100).toFixed(0)}%]`,
            },
          ],
        };
      }


      case "streaming_status": {
        const wsState = streamSocket?.readyState;
        const stateNames = ["CONNECTING", "OPEN", "CLOSING", "CLOSED"];
        const stateName = wsState !== undefined ? stateNames[wsState] || "UNKNOWN" : "NULL";

        const status = {
          enabled: STREAM_ENABLED,
          ws_url: WS_URL,
          socket_state: stateName,
          handshake_complete: streamHandshakeComplete,
          buffer_size: streamBuffer.length,
          connecting: streamConnecting,
          reconnect_pending: streamReconnectTimer !== null,
        };

        // Try to reconnect if not connected
        if (!streamSocket || streamSocket.readyState !== WebSocket.OPEN) {
          connectStream().catch(() => {});
        }

        const statusIcon = status.handshake_complete ? '✓' : '✗';
        const statusText = status.handshake_complete ? 'ACTIVE' : 'INACTIVE';

        let response = `🐘 Streaming Status\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `${statusIcon} ${statusText}\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Enabled: ${status.enabled ? 'Yes' : 'No'}\n`;
        response += `Socket: ${status.socket_state}\n`;
        response += `Handshake: ${status.handshake_complete ? 'Complete' : 'Pending'}\n`;
        response += `Buffer: ${status.buffer_size} messages\n`;
        if (status.connecting) response += `Connecting...\n`;
        if (status.reconnect_pending) response += `Reconnect pending\n`;

        return {
          content: [{ type: "text", text: response.trimEnd() }],
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

        let response = `🐘 Token Status\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `${bar} ${percentUsed}%\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Used: ${status.tokens.toLocaleString()} tokens\n`;
        response += `Budget: ${status.budget.toLocaleString()} tokens\n`;
        response += `Remaining: ${remaining.toLocaleString()} tokens\n`;
        response += `Session: ${sessionDuration} min\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += status.alert
          ? `⚠️ ALERT: ${percentUsed}% used - Consider new session`
          : `✓ Context window healthy`;

        return {
          content: [{ type: "text", text: response }],
        };
      }

      case "reset_token_session": {
        const previousTokens = sessionTokens;
        resetTokenSession();

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
          response += `🧠 MEMORY DYNAMICS\n`;
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
        const { content, trigger_type, trigger_at, after_seconds, keywords, priority = 3, tags = [] } = args as {
          content: string;
          trigger_type: "time" | "duration" | "context";
          trigger_at?: string;
          after_seconds?: number;
          keywords?: string[];
          priority?: number;
          tags?: string[];
        };

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
            trigger = { type: "context", keywords, threshold: 0.7 };
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
        response += `ID: ${result.id.slice(0, 8)}...\n`;
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

        if (result.count === 0) {
          return {
            content: [{ type: "text", text: `No ${status === "all" ? "" : status + " "}reminders found.` }],
          };
        }

        let response = `🐘 SHODH REMINDERS (${result.count})\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

        for (const r of result.reminders) {
          const icon = r.overdue_seconds && r.overdue_seconds > 0 ? "⏰" : "📌";
          const statusBadge = r.status === "triggered" ? " [TRIGGERED]" : "";
          response += `${icon} ${r.content.slice(0, 50)}${r.content.length > 50 ? "..." : ""}${statusBadge}\n`;
          response += `   Type: ${r.trigger_type} | Priority: ${'★'.repeat(r.priority)} | ID: ${r.id.slice(0, 8)}...\n`;
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
        };
      }

      case "dismiss_reminder": {
        const { reminder_id } = args as { reminder_id: string };

        interface ActionResponse {
          success: boolean;
          message: string;
        }

        const result = await apiCall<ActionResponse>(`/api/reminders/${reminder_id}/dismiss`, "POST", {
          user_id: USER_ID,
        });

        return {
          content: [
            {
              type: "text",
              text: result.success
                ? `✓ Reminder dismissed: ${reminder_id.slice(0, 8)}...`
                : `⚠️ ${result.message}`,
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
        };

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

        const result = await apiCall<TodoResponse>("/api/todos", "POST", {
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
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "list_todos": {
        const {
          status: statusFilter,
          project,
          context,
          priority,
          due,
          limit = 50,
        } = args as {
          status?: string[];
          project?: string;
          context?: string;
          priority?: string;
          due?: string;
          limit?: number;
        };

        interface ListTodosResponse {
          success: boolean;
          todos: unknown[];
          projects: unknown[];
          formatted: string;
          count: number;
        }

        const result = await apiCall<ListTodosResponse>("/api/todos/list", "POST", {
          user_id: USER_ID,
          status: statusFilter,
          project,
          context,
          priority,
          due,
          limit,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
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
        };

        interface UpdateTodoResponse {
          success: boolean;
          todo: unknown;
          formatted: string;
        }

        const result = await apiCall<UpdateTodoResponse>(`/api/todos/${todo_id}/update`, "POST", {
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

        const result = await apiCall<CompleteTodoResponse>(`/api/todos/${todo_id}/complete`, "POST", {
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

        const result = await apiCall<DeleteTodoResponse>(`/api/todos/${todo_id}?user_id=${USER_ID}`, "DELETE");

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

        const result = await apiCall<ReorderTodoResponse>(`/api/todos/${todo_id}/reorder`, "POST", {
          user_id: USER_ID,
          direction,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "add_project": {
        const { name, description } = args as { name: string; description?: string };

        interface ProjectResponse {
          success: boolean;
          project: { id: string; name: string };
          formatted: string;
        }

        const result = await apiCall<ProjectResponse>("/api/projects", "POST", {
          user_id: USER_ID,
          name,
          description,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "list_projects": {
        interface ListProjectsResponse {
          success: boolean;
          projects: unknown[];
          formatted: string;
        }

        const result = await apiCall<ListProjectsResponse>("/api/projects/list", "POST", {
          user_id: USER_ID,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
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
          status: "Archived",
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

        const result = await apiCall<ProjectResponse>(`/api/projects/${encodeURIComponent(project)}`, "DELETE", {
          user_id: USER_ID,
          delete_todos: delete_todos ?? false,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "todo_stats": {
        interface TodoStatsResponse {
          stats: unknown;
          formatted: string;
        }

        const result = await apiCall<TodoStatsResponse>("/api/todos/stats", "POST", {
          user_id: USER_ID,
        });

        return {
          content: [{ type: "text", text: result.formatted }],
        };
      }

      case "list_subtasks": {
        const { parent_id } = args as { parent_id: string };

        interface ListSubtasksResponse {
          success: boolean;
          todos: unknown[];
          formatted: string;
        }

        const result = await apiCall<ListSubtasksResponse>(
          `/api/todos/${parent_id}/subtasks?user_id=${USER_ID}`,
          "GET"
        );

        return {
          content: [{ type: "text", text: result.formatted }],
        };
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
        if (surfaced && surfaced.length > 0) {
          const surfacedText = formatSurfacedMemories(surfaced);
          // Append surfaced memories to result
          result.content[result.content.length - 1].text += surfacedText;
        }
      }
    }

    // Inject context window warning if >= threshold (SHO-115)
    if (tokenStatus.alert) {
      const percentUsed = Math.round(tokenStatus.percent * 100);
      const warning = `⚠️ CONTEXT ALERT: ${percentUsed}% of token budget used (${tokenStatus.tokens.toLocaleString()}/${tokenStatus.budget.toLocaleString()}). Consider starting a new session or running consolidation.\n\n`;
      result.content[0].text = warning + result.content[0].text;
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
      helpText = '\n\nEndpoint not found. The server may be running an older version.';
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
});

// List resources (memories as browsable resources)
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  try {
    const result = await apiCall<{ memories: Memory[] }>("/api/memories", "POST", {
      user_id: USER_ID,
    });

    const memories = result.memories || [];

    return {
      resources: memories.slice(0, 50).map((m) => {
        const content = getContent(m);
        return {
          uri: `memory://${m.id}`,
          name: content.slice(0, 50) + (content.length > 50 ? "..." : ""),
          mimeType: "text/plain",
          description: `Type: ${getType(m)}`,
        };
      }),
    };
  } catch {
    return { resources: [] };
  }
});

// Read a specific memory resource
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;
  const memoryId = uri.replace("memory://", "");

  try {
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
    throw new Error(`Failed to read memory: ${message}`);
  }
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

  // Try wrapper first (includes ONNX Runtime setup)
  const wrapperPath = path.join(binDir, wrapperName);
  if (fs.existsSync(wrapperPath)) {
    return wrapperPath;
  }

  // Fallback to direct binary (requires system ONNX Runtime)
  const binaryPath = path.join(binDir, fallbackName);
  if (fs.existsSync(binaryPath)) {
    return binaryPath;
  }

  return null;
}

async function isServerRunning(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);

    const response = await fetch(`${API_URL}/health`, {
      signal: controller.signal,
    });

    clearTimeout(timeout);
    return response.ok;
  } catch {
    return false;
  }
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

async function ensureServerRunning(): Promise<void> {
  // Check if already running
  if (await isServerRunning()) {
    console.error("[shodh-memory] Backend server already running at", API_URL);
    return;
  }

  if (!AUTO_SPAWN_ENABLED) {
    console.error("[shodh-memory] Auto-spawn disabled. Please start the server manually.");
    return;
  }

  const binaryPath = getBinaryPath();
  if (!binaryPath) {
    console.error("[shodh-memory] Server binary not found. Please run: npx @shodh/memory-mcp");
    console.error("[shodh-memory] Or download from: https://github.com/varun29ankuS/shodh-memory/releases");
    return;
  }

  console.error("[shodh-memory] Starting backend server...");

  // Spawn the server process
  serverProcess = spawn(binaryPath, [], {
    detached: true,
    stdio: "ignore",
    env: {
      ...process.env,
      SHODH_DEV_API_KEY: API_KEY, // Pass the API key to the server
    },
  });

  serverProcess.unref();

  // Wait for server to become available
  console.error("[shodh-memory] Waiting for server to start...");
  const started = await waitForServer();

  if (started) {
    console.error("[shodh-memory] Backend server started successfully");
  } else {
    console.error("[shodh-memory] Warning: Server may not have started properly");
  }
}

// Cleanup on exit
process.on("exit", () => {
  if (serverProcess && !serverProcess.killed) {
    serverProcess.kill();
  }
});

// Start server
async function main() {
  // Ensure backend is running
  await ensureServerRunning();

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Shodh-Memory MCP server v0.1.51 running");
  console.error(`Connecting to: ${API_URL}`);
  console.error(`User ID: ${USER_ID}`);
  console.error(`Streaming: ${STREAM_ENABLED ? "enabled" : "disabled"}`);
  console.error(`Proactive surfacing: ${PROACTIVE_SURFACING ? "enabled" : "disabled (SHODH_PROACTIVE=false)"}`);
}

main().catch(console.error);
