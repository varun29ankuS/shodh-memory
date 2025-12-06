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

// Configuration
const API_URL = process.env.SHODH_API_URL || "http://127.0.0.1:3030";
const API_KEY = process.env.SHODH_API_KEY || "shodh-dev-key-change-in-production";
const USER_ID = process.env.SHODH_USER_ID || "claude-code";
const RETRY_ATTEMPTS = 3;
const RETRY_DELAY_MS = 1000;
const REQUEST_TIMEOUT_MS = 10000;

// Types matching the Rust API response structure
interface Experience {
  content: string;
  experience_type?: string;
  tags?: string[];
}

interface Memory {
  id: string;
  experience: Experience;
  score?: number;
  created_at?: string;
  importance?: number;
  tier?: string;
}

interface ApiResponse<T> {
  data?: T;
  error?: string;
}

// Helper: Get content from memory (handles nested structure)
function getContent(m: Memory): string {
  return m.experience?.content || '';
}

// Helper: Get experience type from memory
function getType(m: Memory): string {
  return m.experience?.experience_type || 'Observation';
}

// Helper: Sleep for retry delays
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
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
    version: "0.1.2",
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
          },
          required: ["content"],
        },
      },
      {
        name: "recall",
        description: "Search memories by semantic similarity. Use this to find relevant past experiences, decisions, or context.",
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
            memory_id: {
              type: "string",
              description: "The ID of the memory to delete",
            },
          },
          required: ["memory_id"],
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
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

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

  try {
    switch (name) {
      case "remember": {
        const { content, type = "Observation", tags = [] } = args as {
          content: string;
          type?: string;
          tags?: string[];
        };

        const result = await apiCall<{ memory_id: string }>("/api/record", "POST", {
          user_id: USER_ID,
          experience: {
            content,
            experience_type: type,
            tags,
          },
        });

        return {
          content: [
            {
              type: "text",
              text: `Remembered: "${content.slice(0, 50)}${content.length > 50 ? '...' : ''}"\nMemory ID: ${result.memory_id}`,
            },
          ],
        };
      }

      case "recall": {
        const { query, limit = 5 } = args as { query: string; limit?: number };

        const result = await apiCall<{ memories: Memory[] }>("/api/retrieve", "POST", {
          user_id: USER_ID,
          query,
          limit,
        });

        const memories = result.memories || [];

        if (memories.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: `No memories found for: "${query}"`,
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

        return {
          content: [
            {
              type: "text",
              text: `Found ${memories.length} relevant memories:\n\n${formatted}`,
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
        const { memory_id } = args as { memory_id: string };

        await apiCall(`/api/memory/${memory_id}?user_id=${USER_ID}`, "DELETE");

        return {
          content: [
            {
              type: "text",
              text: `Deleted memory: ${memory_id}`,
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

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
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

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Shodh-Memory MCP server v0.1.2 running");
  console.error(`Connecting to: ${API_URL}`);
  console.error(`User ID: ${USER_ID}`);
}

main().catch(console.error);
