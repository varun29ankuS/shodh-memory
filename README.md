<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/logo.png" width="120" alt="Shodh-Memory">
</p>

<h1 align="center">Shodh-Memory</h1>

<p align="center"><b>Your AI agent remembers what matters, forgets what doesn't, and gets smarter the more you use it.</b></p>

<p align="center">
  <a href="https://github.com/varun29ankuS/shodh-memory/actions"><img src="https://github.com/varun29ankuS/shodh-memory/workflows/CI/badge.svg" alt="build"></a>
  <a href="https://registry.modelcontextprotocol.io/v0/servers?search=shodh"><img src="https://img.shields.io/badge/MCP-Registry-green" alt="MCP Registry"></a>
  <a href="https://crates.io/crates/shodh-memory"><img src="https://img.shields.io/crates/v/shodh-memory.svg" alt="crates.io"></a>
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/v/@shodh/memory-mcp.svg?logo=npm" alt="npm"></a>
  <a href="https://pypi.org/project/shodh-memory/"><img src="https://img.shields.io/pypi/v/shodh-memory.svg" alt="PyPI"></a>
  <a href="https://hub.docker.com/r/varunshodh/shodh-memory"><img src="https://img.shields.io/docker/pulls/varunshodh/shodh-memory.svg?logo=docker" alt="Docker"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

---

AI agents forget everything between sessions. They repeat mistakes, lose context, and treat every conversation like the first one.

Shodh-Memory fixes this. It's persistent memory that actually learns — memories you use often become easier to find, old irrelevant context fades automatically, and recalling one thing brings back related things. No API keys. No cloud. No external databases. One binary.

## Why Not Just Use mem0 / Cognee / Zep?

| | **Shodh** | **mem0** | **Cognee** | **Zep** |
|---|---|---|---|---|
| LLM calls to store a memory | **0** | 2+ per add | 3+ per cognify | 2+ per episode |
| External services needed | **None** | OpenAI + vector DB | OpenAI + Neo4j + vector DB | OpenAI + Neo4j |
| Time to store a memory | **55ms** | ~20 seconds | seconds | seconds |
| Learns from usage | **Yes** (Hebbian) | No | No | No |
| Forgets irrelevant data | **Yes** (decay) | No | No | Temporal only |
| Runs fully offline | **Yes** | No | No | No |
| Binary size | **~17MB** | pip install + API keys | pip install + API keys + Neo4j | Cloud only |

Every other memory system delegates intelligence to LLM API calls — that's why they're slow, expensive, and can't work offline. Shodh uses algorithmic intelligence: local embeddings, mathematical decay, learned associations. No LLM in the loop.

## Get Started

### Claude Code / Cursor (MCP)

```bash
# 1. Start the server
docker run -d -p 3030:3030 -v shodh-data:/data varunshodh/shodh-memory

# 2. Add to Claude Code
claude mcp add shodh-memory -- npx -y @shodh/memory-mcp
```

That's it. Claude now has persistent memory across sessions.

<details>
<summary>Or use a binary instead of Docker</summary>

Download from [GitHub Releases](https://github.com/varun29ankuS/shodh-memory/releases):

```bash
curl -L https://github.com/varun29ankuS/shodh-memory/releases/latest/download/shodh-memory-linux-x64.tar.gz | tar -xz
./shodh-memory
```
</details>

<details>
<summary>Cursor / Claude Desktop config</summary>

```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["-y", "@shodh/memory-mcp"],
      "env": {
        "SHODH_API_KEY": "your-key"
      }
    }
  }
}
```

Generate a key: `openssl rand -hex 32`, set it on the server via `SHODH_DEV_API_KEY` env var.
</details>

### Python

```bash
pip install shodh-memory
```

```python
from shodh_memory import Memory

memory = Memory(storage_path="./my_data")
memory.remember("User prefers dark mode", memory_type="Decision")
results = memory.recall("user preferences", limit=5)
```

### Rust

```toml
[dependencies]
shodh-memory = "0.1"
```

```rust
use shodh_memory::{MemorySystem, MemoryConfig};

let memory = MemorySystem::new(MemoryConfig::default())?;
memory.remember("user-1", "User prefers dark mode", MemoryType::Decision, vec![])?;
let results = memory.recall("user-1", "user preferences", 5)?;
```

### Docker

```bash
docker run -d -p 3030:3030 -v shodh-data:/data varunshodh/shodh-memory
```

## What It Does

```
You use a memory often  →  it becomes easier to find (Hebbian learning)
You stop using a memory →  it fades over time (activation decay)
You recall one memory   →  related memories surface too (spreading activation)
A connection is used    →  it becomes permanent (long-term potentiation)
```

Under the hood, memories flow through three tiers:

```
Working Memory ──overflow──▶ Session Memory ──importance──▶ Long-Term Memory
   (100 items)                  (500 MB)                      (RocksDB)
```

This is based on [Cowan's working memory model](https://doi.org/10.1177/0963721409359277) and [Wixted's memory decay research](https://doi.org/10.1111/j.1467-9280.2004.00687.x). The neuroscience isn't a gimmick — it's why the system gets better with use instead of just accumulating data.

## Performance

| Operation | Latency |
|-----------|---------|
| Store memory | 55-60ms |
| Semantic search | 34-58ms |
| Tag search | ~1ms |
| Entity lookup | 763ns |
| Graph traversal (3-hop) | 30µs |

Single 17MB binary. No GPU required. Runs on a $5 VPS.

## TUI Dashboard

```bash
shodh-tui
```

<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/dashboard.jpg" width="700" alt="Shodh Dashboard">
</p>

<p align="center"><i>Real-time activity feed, memory tiers, and detailed inspection</i></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/graph-map.jpg" width="700" alt="Shodh Graph Map">
</p>

<p align="center"><i>Knowledge graph — entity connections strengthened through use</i></p>

## 47 MCP Tools

Full list of tools available to Claude, Cursor, and other MCP clients:

<details>
<summary>Memory</summary>

`remember` · `recall` · `proactive_context` · `context_summary` · `list_memories` · `read_memory` · `forget` · `reinforce`
</details>

<details>
<summary>Todos (GTD)</summary>

`add_todo` · `list_todos` · `update_todo` · `complete_todo` · `delete_todo` · `reorder_todo` · `list_subtasks` · `add_todo_comment` · `list_todo_comments` · `update_todo_comment` · `delete_todo_comment` · `todo_stats`
</details>

<details>
<summary>Projects</summary>

`add_project` · `list_projects` · `archive_project` · `delete_project`
</details>

<details>
<summary>Reminders</summary>

`set_reminder` · `list_reminders` · `dismiss_reminder`
</details>

<details>
<summary>System</summary>

`memory_stats` · `verify_index` · `repair_index` · `token_status` · `reset_token_session` · `consolidation_report` · `backup_create` · `backup_list` · `backup_verify` · `backup_restore` · `backup_purge`
</details>

## REST API

160+ endpoints on `http://localhost:3030`. All `/api/*` endpoints require `X-API-Key` header.

[Full API reference →](https://www.shodh-memory.com/docs/api)

<details>
<summary>Quick examples</summary>

```bash
# Store a memory
curl -X POST http://localhost:3030/api/remember \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"user_id": "user-1", "content": "User prefers dark mode", "memory_type": "Decision"}'

# Search memories
curl -X POST http://localhost:3030/api/recall \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"user_id": "user-1", "query": "user preferences", "limit": 5}'
```
</details>

## Platform Support

Linux x86_64 · Linux ARM64 · macOS Apple Silicon · macOS Intel · Windows x86_64

## Production Deployment

<details>
<summary>Environment variables</summary>

```bash
SHODH_ENV=production              # Production mode
SHODH_API_KEYS=key1,key2,key3     # Comma-separated API keys
SHODH_HOST=127.0.0.1              # Bind address (default: localhost)
SHODH_PORT=3030                   # Port (default: 3030)
SHODH_MEMORY_PATH=/var/lib/shodh  # Data directory
SHODH_REQUEST_TIMEOUT=60          # Request timeout in seconds
SHODH_MAX_CONCURRENT=200          # Max concurrent requests
SHODH_CORS_ORIGINS=https://app.example.com
```
</details>

<details>
<summary>Docker Compose with TLS</summary>

```yaml
services:
  shodh-memory:
    image: varunshodh/shodh-memory:latest
    environment:
      - SHODH_ENV=production
      - SHODH_HOST=0.0.0.0
      - SHODH_API_KEYS=${SHODH_API_KEYS}
    volumes:
      - shodh-data:/data
    networks:
      - internal

  caddy:
    image: caddy:latest
    ports:
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
    networks:
      - internal

volumes:
  shodh-data:

networks:
  internal:
```
</details>

<details>
<summary>Reverse proxy (Nginx / Caddy)</summary>

The server binds to `127.0.0.1` by default. For network deployments, place behind a reverse proxy:

```caddyfile
memory.example.com {
    reverse_proxy localhost:3030
}
```
</details>

## Community

| Project | Description | Author |
|---------|-------------|--------|
| [SHODH on Cloudflare](https://github.com/doobidoo/shodh-cloudflare) | Edge-native implementation on Cloudflare Workers | [@doobidoo](https://github.com/doobidoo) |

## References

[1] Cowan, N. (2010). The Magical Mystery Four. *Current Directions in Psychological Science*. [2] Magee & Grienberger (2020). Synaptic Plasticity Forms and Functions. *Annual Review of Neuroscience*. [3] Subramanya et al. (2019). DiskANN. *NeurIPS 2019*.

## License

Apache 2.0

---

<p align="center">
  <a href="https://registry.modelcontextprotocol.io/v0/servers?search=shodh">MCP Registry</a> · <a href="https://hub.docker.com/r/varunshodh/shodh-memory">Docker Hub</a> · <a href="https://pypi.org/project/shodh-memory/">PyPI</a> · <a href="https://www.npmjs.com/package/@shodh/memory-mcp">npm</a> · <a href="https://crates.io/crates/shodh-memory">crates.io</a> · <a href="https://www.shodh-memory.com">Docs</a>
</p>
