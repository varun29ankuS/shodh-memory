# Shodh-Memory MCP Server

Give Claude Code persistent memory using Shodh-Memory.

## Tools Available

| Tool | Description |
|------|-------------|
| `remember` | Store a memory (observation, decision, learning, etc.) |
| `recall` | Semantic search to find relevant memories |
| `list_memories` | List all stored memories |
| `forget` | Delete a specific memory |
| `memory_stats` | Get memory statistics |

## Setup

### 1. Start shodh-memory server

```bash
# Set environment variables
export ORT_DYLIB_PATH="/path/to/onnxruntime.dll"
export SHODH_MODEL_PATH="/path/to/minilm-l6"

# Run server
./shodh-memory-server
```

### 2. Install MCP server

```bash
cd mcp-server
bun install
bun run build
```

### 3. Add to Claude Code config

Edit `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "bun",
      "args": ["run", "/path/to/shodh-memory/mcp-server/dist/index.js"],
      "env": {
        "SHODH_API_URL": "http://127.0.0.1:3030",
        "SHODH_USER_ID": "claude-code"
      }
    }
  }
}
```

Or for production with npx:

```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["-y", "@shodh/memory-mcp"],
      "env": {
        "SHODH_API_URL": "http://127.0.0.1:3030"
      }
    }
  }
}
```

## Usage Examples

Once configured, Claude can use these commands:

```
"Remember that the user prefers TypeScript over JavaScript"

"Recall what I learned about the user's coding preferences"

"List all my memories"

"Show memory stats"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHODH_API_URL` | `http://127.0.0.1:3030` | Shodh-Memory server URL |
| `SHODH_API_KEY` | `shodh-dev-key...` | API key for authentication |
| `SHODH_USER_ID` | `claude-code` | User ID for memory isolation |
