# Shodh-Memory MCP Server

Persistent AI memory with semantic search. Store observations, decisions, learnings, and recall them across sessions.

## Features

- **Semantic Search**: Find memories by meaning, not just keywords
- **Memory Types**: Categorize as Observation, Decision, Learning, Error, Pattern, etc.
- **Persistent**: Memories survive across sessions and restarts
- **Fast**: Sub-millisecond retrieval with vector indexing

## Installation

Add to your MCP client config:

**Claude Desktop / Claude Code** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["-y", "@shodh/memory-mcp"]
    }
  }
}
```

Config file locations:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**For Cursor/other MCP clients**: Similar configuration with the npx command.

## Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with optional type and tags |
| `recall` | Semantic search to find relevant memories |
| `context_summary` | Get categorized context for session bootstrap |
| `list_memories` | List all stored memories |
| `forget` | Delete a specific memory by ID |
| `memory_stats` | Get statistics about stored memories |

## Usage Examples

```
"Remember that the user prefers Rust over Python for systems programming"
"Recall what I know about user's programming preferences"
"List my recent memories"
"Show memory stats"
```

## License

Apache-2.0
