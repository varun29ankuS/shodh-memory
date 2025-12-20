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
    ],
  };
});

// Auto-stream context from tool arguments (captures conversation intent)
function autoStreamContext(toolName: string, args: Record<string, unknown>): void {
  // Skip tools that already handle their own streaming
  if (["proactive_context", "streaming_status"].includes(toolName)) return;

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

        return {
          content: [
            {
              type: "text",
              text: `Remembered: "${content.slice(0, 50)}${content.length > 50 ? '...' : ''}"\nMemory ID: ${result.id}`,
            },
          ],
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
                text: `No memories found for: "${query}" (mode: ${mode})`,
              },
            ],
          };
        }

        const formatted = memories
          .map((m, i) => {
            const content = getContent(m);
            const score = ((m.score || 0) * 100).toFixed(0);
            return `${i + 1}. [${score}% match] ${content}\n   Type: ${getType(m)} | ID: ${m.id.slice(0, 8)}...`;
          })
          .join("\n\n");

        // Build stats summary for associative/hybrid modes
        let statsText = "";
        if (stats && (mode === "associative" || mode === "hybrid")) {
          const graphPct = (stats.graph_weight * 100).toFixed(0);
          const semPct = (stats.semantic_weight * 100).toFixed(0);
          statsText = `\n\n[Stats: ${stats.mode} mode | graph=${graphPct}% semantic=${semPct}% | density=${stats.graph_density.toFixed(2)} | ${stats.graph_candidates} graph + ${stats.semantic_candidates} semantic candidates | ${stats.entities_activated} entities | ${(stats.retrieval_time_us / 1000).toFixed(1)}ms]`;
        }

        return {
          content: [
            {
              type: "text",
              text: `Found ${memories.length} relevant memories (${mode} mode):\n\n${formatted}${statsText}`,
            },
          ],
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
          return {
            content: [
              {
                type: "text",
                text: "No memories stored yet. Start remembering things to build context!",
              },
            ],
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

        // Build summary
        const sections: string[] = [];

        if (include_context && context.length > 0) {
          const items = context.slice(0, max_items).map(m => `  - ${getContent(m).slice(0, 100)}`);
          sections.push(`PROJECT CONTEXT:\n${items.join('\n')}`);
        }

        if (include_decisions && decisions.length > 0) {
          const items = decisions.slice(0, max_items).map(m => `  - ${getContent(m).slice(0, 100)}`);
          sections.push(`DECISIONS MADE:\n${items.join('\n')}`);
        }

        if (include_learnings && learnings.length > 0) {
          const items = learnings.slice(0, max_items).map(m => `  - ${getContent(m).slice(0, 100)}`);
          sections.push(`LEARNINGS:\n${items.join('\n')}`);
        }

        if (patterns.length > 0) {
          const items = patterns.slice(0, max_items).map(m => `  - ${getContent(m).slice(0, 100)}`);
          sections.push(`PATTERNS NOTICED:\n${items.join('\n')}`);
        }

        if (errors.length > 0) {
          const items = errors.slice(0, Math.min(3, max_items)).map(m => `  - ${getContent(m).slice(0, 100)}`);
          sections.push(`ERRORS TO AVOID:\n${items.join('\n')}`);
        }

        const summary = sections.length > 0
          ? sections.join('\n\n')
          : `${memories.length} memories stored, but none categorized as decisions, learnings, or context. Consider using those types when remembering.`;

        return {
          content: [
            {
              type: "text",
              text: `CONTEXT SUMMARY (${memories.length} total memories)\n${'='.repeat(40)}\n\n${summary}`,
            },
          ],
        };
      }

      case "list_memories": {
        const { limit = 20 } = args as { limit?: number };

        const result = await apiCall<{ memories: Memory[] }>("/api/memories", "POST", {
          user_id: USER_ID,
        });

        const memories = (result.memories || []).slice(0, limit);

        if (memories.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: "No memories stored yet.",
              },
            ],
          };
        }

        const formatted = memories
          .map((m, i) => {
            const content = getContent(m);
            return `${i + 1}. ${content.slice(0, 80)}${content.length > 80 ? '...' : ''}\n   Type: ${getType(m)} | ID: ${m.id.slice(0, 8)}...`;
          })
          .join("\n\n");

        return {
          content: [
            {
              type: "text",
              text: `${memories.length} memories:\n\n${formatted}`,
            },
          ],
        };
      }

      case "forget": {
        const { id } = args as { id: string };

        await apiCall(`/api/memory/${id}?user_id=${USER_ID}`, "DELETE");

        return {
          content: [
            {
              type: "text",
              text: `Deleted memory: ${id}`,
            },
          ],
        };
      }

      case "memory_stats": {
        const result = await apiCall<Record<string, unknown>>(`/api/users/${USER_ID}/stats`, "GET");

        return {
          content: [
            {
              type: "text",
              text: `Memory Statistics:\n${JSON.stringify(result, null, 2)}`,
            },
          ],
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

        const statusIcon = result.is_healthy ? "✓" : "⚠";
        const healthText = result.is_healthy ? "Healthy" : "Unhealthy - orphaned memories detected";

        let response = `Index Integrity Report\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Status: ${statusIcon} ${healthText}\n`;
        response += `Total in storage: ${result.total_storage}\n`;
        response += `Total indexed: ${result.total_indexed}\n`;
        response += `Orphaned count: ${result.orphaned_count}\n`;

        if (result.orphaned_count > 0) {
          response += `\nRecommendation: Run repair_index to fix orphaned memories.`;
        }

        return {
          content: [
            {
              type: "text",
              text: response,
            },
          ],
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

        const statusIcon = result.is_healthy ? "✓" : "⚠";

        let response = `Index Repair Results\n`;
        response += `━━━━━━━━━━━━━━━━━━━━━\n`;
        response += `Status: ${statusIcon} ${result.success ? "Success" : "Partial success"}\n`;
        response += `Total in storage: ${result.total_storage}\n`;
        response += `Total indexed: ${result.total_indexed}\n`;
        response += `Repaired: ${result.repaired}\n`;
        response += `Failed: ${result.failed}\n`;
        response += `Index healthy: ${result.is_healthy ? "Yes" : "No"}\n`;

        if (result.failed > 0) {
          response += `\nNote: ${result.failed} memories could not be repaired (embedding generation failed).`;
        }

        return {
          content: [
            {
              type: "text",
              text: response,
            },
          ],
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
          return {
            content: [
              {
                type: "text",
                text: `No memories found with tags: ${tags.join(", ")}`,
              },
            ],
          };
        }

        const formatted = memories
          .map((m, i) => {
            const content = getContent(m);
            return `${i + 1}. ${content.slice(0, 80)}${content.length > 80 ? '...' : ''}\n   Type: ${getType(m)} | ID: ${m.id.slice(0, 8)}...`;
          })
          .join("\n\n");

        return {
          content: [
            {
              type: "text",
              text: `Found ${memories.length} memories with tags [${tags.join(", ")}]:\n\n${formatted}`,
            },
          ],
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

        if (memories.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: `No memories found between ${start} and ${end}`,
              },
            ],
          };
        }

        const formatted = memories
          .map((m, i) => {
            const content = getContent(m);
            return `${i + 1}. ${content.slice(0, 80)}${content.length > 80 ? '...' : ''}\n   Type: ${getType(m)} | Created: ${m.created_at || 'unknown'}`;
          })
          .join("\n\n");

        return {
          content: [
            {
              type: "text",
              text: `Found ${memories.length} memories between ${start} and ${end}:\n\n${formatted}`,
            },
          ],
        };
      }

      case "forget_by_tags": {
        const { tags } = args as { tags: string[] };

        const result = await apiCall<{ deleted_count: number }>("/api/forget/tags", "POST", {
          user_id: USER_ID,
          tags,
        });

        return {
          content: [
            {
              type: "text",
              text: `Deleted ${result.deleted_count} memories with tags: ${tags.join(", ")}`,
            },
          ],
        };
      }

      case "forget_by_date": {
        const { start, end } = args as { start: string; end: string };

        const result = await apiCall<{ deleted_count: number }>("/api/forget/date", "POST", {
          user_id: USER_ID,
          start,
          end,
        });

        return {
          content: [
            {
              type: "text",
              text: `Deleted ${result.deleted_count} memories between ${start} and ${end}`,
            },
          ],
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

        return {
          content: [
            {
              type: "text",
              text: `Surfaced ${memories.length} relevant memories:\n\n${formatted}${entitySummary}\n\n[Latency: ${result.latency_ms.toFixed(1)}ms | Threshold: ${(semantic_threshold * 100).toFixed(0)}%]`,
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

        return {
          content: [
            {
              type: "text",
              text: `Streaming Status:\n\n` +
                `Enabled: ${status.enabled}\n` +
                `WebSocket URL: ${status.ws_url}\n` +
                `Socket State: ${status.socket_state}\n` +
                `Handshake Complete: ${status.handshake_complete}\n` +
                `Buffer Size: ${status.buffer_size}\n` +
                `Currently Connecting: ${status.connecting}\n` +
                `Reconnect Pending: ${status.reconnect_pending}\n\n` +
                (status.handshake_complete ? "✓ Streaming is ACTIVE" : "✗ Streaming is NOT ACTIVE - attempting reconnect..."),
            },
          ],
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
        const sections: string[] = [];

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

        // Summary header
        sections.push(`CONSOLIDATION REPORT (${eventCount} events)`);
        sections.push(`Period: ${result.period.start} to ${result.period.end}`);
        sections.push('='.repeat(50));

        // Memory changes
        if (stats.memories_strengthened > 0 || stats.memories_decayed > 0 || stats.memories_at_risk > 0) {
          const memoryLines: string[] = [];
          if (stats.memories_strengthened > 0) memoryLines.push(`  + ${stats.memories_strengthened} memories strengthened`);
          if (stats.memories_decayed > 0) memoryLines.push(`  - ${stats.memories_decayed} memories decayed`);
          if (stats.memories_at_risk > 0) memoryLines.push(`  ! ${stats.memories_at_risk} memories at risk of forgetting`);
          sections.push(`MEMORY CHANGES:\n${memoryLines.join('\n')}`);
        }

        // Edge changes (associations)
        if (stats.edges_formed > 0 || stats.edges_strengthened > 0 || stats.edges_potentiated > 0 || stats.edges_pruned > 0) {
          const edgeLines: string[] = [];
          if (stats.edges_formed > 0) edgeLines.push(`  + ${stats.edges_formed} new associations formed`);
          if (stats.edges_strengthened > 0) edgeLines.push(`  + ${stats.edges_strengthened} associations strengthened`);
          if (stats.edges_potentiated > 0) edgeLines.push(`  * ${stats.edges_potentiated} associations became permanent (LTP)`);
          if (stats.edges_pruned > 0) edgeLines.push(`  - ${stats.edges_pruned} weak associations pruned`);
          sections.push(`ASSOCIATIONS (Hebbian Learning):\n${edgeLines.join('\n')}`);
        }

        // Fact changes
        if (stats.facts_extracted > 0 || stats.facts_reinforced > 0) {
          const factLines: string[] = [];
          if (stats.facts_extracted > 0) factLines.push(`  + ${stats.facts_extracted} facts extracted`);
          if (stats.facts_reinforced > 0) factLines.push(`  + ${stats.facts_reinforced} facts reinforced`);
          sections.push(`FACTS:\n${factLines.join('\n')}`);
        }

        // Maintenance cycles
        if (stats.maintenance_cycles > 0) {
          const durationSec = (stats.total_maintenance_duration_ms / 1000).toFixed(2);
          sections.push(`MAINTENANCE: ${stats.maintenance_cycles} cycle(s) completed (${durationSec}s total)`);
        }

        // No activity message
        if (eventCount === 0) {
          sections.push('No consolidation activity in this period. Store and access memories to trigger learning.');
        }

        return {
          content: [
            {
              type: "text",
              text: sections.join('\n\n'),
            },
          ],
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

    return result;
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
