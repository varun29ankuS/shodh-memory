<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/logo.png" width="120" alt="Shodh-Memory">
</p>

<h1 align="center">Shodh-Memory MCP Server</h1>

<p align="center">
  <strong>v0.1.61</strong> | Persistent cognitive memory for AI agents
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/v/@shodh/memory-mcp" alt="npm"></a>
  <a href="https://github.com/varun29ankuS/shodh-memory/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://www.shodh-rag.com/memory">Documentation</a> |
  <a href="https://github.com/varun29ankuS/shodh-memory">GitHub</a> |
  <a href="https://pypi.org/project/shodh-memory/">Python SDK</a> |
  <a href="https://crates.io/crates/shodh-memory">Rust Crate</a>
</p>

---

## Features

- **Cognitive Architecture**: 3-tier memory (working, session, long-term) based on Cowan's model
- **Hebbian Learning**: "Neurons that fire together wire together" - associations strengthen with use
- **Semantic Search**: Find memories by meaning using MiniLM-L6 embeddings
- **Knowledge Graph**: Entity extraction and relationship tracking
- **Memory Consolidation**: Automatic decay, replay, and strengthening
- **1-Click Install**: Auto-downloads native server binary for your platform
- **Offline-First**: All models bundled (~15MB), no internet required after install
- **Fast**: Sub-millisecond graph lookup, 30-50ms semantic search

## Installation

Add to your MCP client config:

**Claude Desktop / Claude Code** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["-y", "@shodh/memory-mcp"],
      "env": {
        "SHODH_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Config file locations:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**For Cursor/other MCP clients**: Similar configuration with the npx command.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SHODH_API_KEY` | **Required**. API key for authentication | - |
| `SHODH_API_URL` | Backend server URL | `http://127.0.0.1:3030` |
| `SHODH_USER_ID` | User ID for memory isolation | `claude-code` |
| `SHODH_NO_AUTO_SPAWN` | Set to `true` to disable auto-starting the backend | `false` |
| `SHODH_STREAM` | Enable/disable streaming ingestion | `true` |
| `SHODH_PROACTIVE` | Enable/disable proactive memory surfacing | `true` |

## MCP Tools (15 total)

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with optional type and tags |
| `recall` | Semantic search to find relevant memories |
| `proactive_context` | Auto-surface relevant memories for current context |
| `context_summary` | Get categorized context for session bootstrap |
| `list_memories` | List all stored memories |
| `forget` | Delete a specific memory by ID |
| `forget_by_tags` | Delete memories matching any of the specified tags |
| `forget_by_date` | Delete memories within a date range |
| `memory_stats` | Get statistics about stored memories |
| `recall_by_tags` | Find memories by tag |
| `recall_by_date` | Find memories within a date range |
| `verify_index` | Check vector index health |
| `repair_index` | Repair orphaned memories |
| `consolidation_report` | View memory consolidation activity |
| `streaming_status` | Check WebSocket streaming connection status |

## REST API (for Developers)

The server exposes a REST API at `http://127.0.0.1:3030`:

```javascript
// Store a memory
const res = await fetch("http://127.0.0.1:3030/api/remember", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
  },
  body: JSON.stringify({
    user_id: "my-app",
    content: "User prefers dark mode",
    memory_type: "Observation",
    tags: ["preferences", "ui"]
  })
});

// Semantic search
const results = await fetch("http://127.0.0.1:3030/api/recall", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key"
  },
  body: JSON.stringify({
    user_id: "my-app",
    query: "user preferences",
    limit: 5
  })
});
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/remember` | POST | Store a memory |
| `/api/recall` | POST | Semantic search |
| `/api/recall/tags` | POST | Search by tags |
| `/api/recall/date` | POST | Search by date range |
| `/api/memories` | POST | List all memories |
| `/api/memory/{id}` | GET/PUT/DELETE | CRUD operations |
| `/api/context_summary` | POST | Get context summary |
| `/api/relevant` | POST | Proactive context surfacing |
| `/api/batch_remember` | POST | Store multiple memories |
| `/api/upsert` | POST | Create or update by external_id |
| `/api/graph/{user_id}/stats` | GET | Knowledge graph statistics |
| `/api/consolidation/report` | POST | Memory consolidation report |
| `/api/index/verify` | POST | Verify index integrity |
| `/metrics` | GET | Prometheus metrics |

## Cognitive Features

### Hebbian Learning
Memories that are frequently accessed together form stronger associations. The system automatically:
- Forms edges between co-retrieved memories
- Strengthens connections with repeated co-activation
- Enables Long-Term Potentiation (LTP) for permanent associations

### Memory Consolidation
Background processes maintain memory health:
- **Decay**: Unused memories gradually lose activation
- **Replay**: High-value memories are periodically replayed
- **Pruning**: Weak associations are removed
- **Promotion**: Important memories move to long-term storage

### 3-Tier Architecture
Based on Cowan's working memory model:
1. **Working Memory**: Recent, highly active memories
2. **Session Memory**: Current session context
3. **Long-Term Memory**: Persistent storage with vector indexing

## How It Works

1. **Install**: `npx -y @shodh/memory-mcp` downloads the package
2. **Auto-spawn**: On first run, downloads the native server binary (~15MB)
3. **Connect**: MCP client connects to the server via stdio
4. **Ready**: Start using `remember` and `recall` tools

The backend server runs locally and stores all data on your machine. No cloud dependency.

## Usage Examples

```
"Remember that the user prefers Rust over Python for systems programming"
"Recall what I know about user's programming preferences"
"What context do you have about this project?"
"List my recent memories"
"Show me the consolidation report"
```

## Platform Support

| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux | x64 | Supported |
| macOS | x64 | Supported |
| macOS | ARM64 (M1/M2) | Supported |
| Windows | x64 | Supported |

## Related Packages

- **Python SDK**: `pip install shodh-memory` - Native Python bindings
- **Rust Crate**: `cargo add shodh-memory` - Use as a library

## Links

- [Documentation](https://www.shodh-rag.com/memory)
- [GitHub Repository](https://github.com/varun29ankuS/shodh-memory)
- [Issue Tracker](https://github.com/varun29ankuS/shodh-memory/issues)

## License

Apache-2.0
