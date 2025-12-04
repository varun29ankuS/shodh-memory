#!/usr/bin/env node
/**
 * Shodh-Memory MCP Server
 *
 * Gives Claude Code persistent memory across sessions.
 * Connects to shodh-memory REST API running locally.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const API_URL = process.env.SHODH_API_URL || "http://127.0.0.1:3030";
const API_KEY = process.env.SHODH_API_KEY || "shodh-dev-key-change-in-production";
const USER_ID = process.env.SHODH_USER_ID || "claude-code";

interface Memory {
  memory_id: string;
  content: string;
  experience_type?: string;
  score?: number;
  created_at?: string;
}

async function apiCall(endpoint: string, method: string = "GET", body?: object): Promise<any> {
  const options: RequestInit = {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
    },
  };

  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${API_URL}${endpoint}`, options);

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// Create MCP server
const server = new Server(
  {
    name: "shodh-memory",
    version: "0.1.0",
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
              enum: ["observation", "decision", "learning", "action", "error", "user_preference", "project_context"],
              description: "Type of memory",
              default: "observation",
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

  try {
    switch (name) {
      case "remember": {
        const { content, type = "observation", tags = [] } = args as {
          content: string;
          type?: string;
          tags?: string[];
        };

        const result = await apiCall("/api/record", "POST", {
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

        const result = await apiCall("/api/retrieve", "POST", {
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
          .map((m: Memory, i: number) =>
            `${i + 1}. [${((m.score || 0) * 100).toFixed(0)}%] ${m.content}\n   ID: ${m.memory_id.slice(0, 8)}...`
          )
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

      case "list_memories": {
        const { limit = 20 } = args as { limit?: number };

        const result = await apiCall("/api/memories", "POST", {
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
          .map((m: Memory, i: number) =>
            `${i + 1}. ${m.content.slice(0, 80)}${m.content.length > 80 ? '...' : ''}\n   Type: ${m.experience_type || 'unknown'} | ID: ${m.memory_id.slice(0, 8)}...`
          )
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

        await apiCall(`/api/memory/${memory_id}`, "DELETE");

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
        const result = await apiCall(`/api/stats/${USER_ID}`, "GET");

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
    return {
      content: [
        {
          type: "text",
          text: `Error: ${message}`,
        },
      ],
      isError: true,
    };
  }
});

// List resources (memories as browsable resources)
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  try {
    const result = await apiCall("/api/memories", "POST", {
      user_id: USER_ID,
    });

    const memories = result.memories || [];

    return {
      resources: memories.slice(0, 50).map((m: Memory) => ({
        uri: `memory://${m.memory_id}`,
        name: m.content.slice(0, 50) + (m.content.length > 50 ? "..." : ""),
        mimeType: "text/plain",
        description: `Type: ${m.experience_type || "unknown"}`,
      })),
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
    const result = await apiCall("/api/memories", "POST", {
      user_id: USER_ID,
    });

    const memory = (result.memories || []).find((m: Memory) => m.memory_id === memoryId);

    if (!memory) {
      throw new Error(`Memory not found: ${memoryId}`);
    }

    return {
      contents: [
        {
          uri,
          mimeType: "text/plain",
          text: `Content: ${memory.content}\n\nType: ${memory.experience_type || "unknown"}\nCreated: ${memory.created_at || "unknown"}\nID: ${memory.memory_id}`,
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
  console.error("Shodh-Memory MCP server running");
}

main().catch(console.error);
