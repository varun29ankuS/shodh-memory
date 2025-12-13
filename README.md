<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/logo.png" width="120" alt="Shodh-Memory">
</p>

<h1 align="center">Shodh-Memory</h1>

<p align="center">
  <a href="https://github.com/varun29ankuS/shodh-memory/actions"><img src="https://github.com/varun29ankuS/shodh-memory/workflows/CI/badge.svg" alt="build"></a>
  <a href="https://registry.modelcontextprotocol.io/v0/servers?search=shodh"><img src="https://img.shields.io/badge/MCP-Registry-green" alt="MCP Registry"></a>
  <a href="https://crates.io/crates/shodh-memory"><img src="https://img.shields.io/crates/v/shodh-memory.svg" alt="crates.io"></a>
  <a href="https://crates.io/crates/shodh-memory"><img src="https://img.shields.io/crates/d/shodh-memory.svg?label=crates.io%20downloads" alt="crates.io Downloads"></a>
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/v/@shodh/memory-mcp.svg?logo=npm" alt="npm"></a>
  <a href="https://pypi.org/project/shodh-memory/"><img src="https://img.shields.io/pypi/v/shodh-memory.svg" alt="PyPI"></a>
  <a href="https://pepy.tech/project/shodh-memory"><img src="https://static.pepy.tech/badge/shodh-memory" alt="PyPI Downloads"></a>
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/dm/@shodh/memory-mcp.svg?label=npm%20downloads" alt="npm Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

---

<p align="center"><i>Persistent memory for AI agents. Single binary. Local-first. Runs offline.</i></p>

---

> **For AI Agents** — Claude, Cursor, GPT, LangChain, AutoGPT, robotic systems, or your custom agents.
> Give them memory that persists across sessions, learns from experience, and runs entirely on your hardware.

---

We built this because AI agents forget everything between sessions. They make the same mistakes, ask the same questions, lose context constantly.

Shodh-Memory fixes that. It's a cognitive memory system—Hebbian learning, activation decay, semantic consolidation—packed into a single ~15MB binary that runs offline. Deploy on cloud, edge devices, or air-gapped systems.

**How it works:**

Experiences flow through three tiers based on Cowan's working memory model [1]. New information enters capacity-limited working memory, overflows into session storage, and consolidates into long-term memory based on importance. When memories are retrieved together successfully, their connections strengthen—classic Hebbian learning [2]. After enough co-activations, those connections become permanent. Unused memories naturally fade.

```
Working Memory ──overflow──▶ Session Memory ──importance──▶ Long-Term Memory
   (100 items)                  (500 MB)                      (RocksDB)
```

### Architecture

**Storage & Retrieval**

- Vamana graph index for approximate nearest neighbor search [3]
- MiniLM-L6 embeddings (384-dim, 25MB) for semantic similarity
- TinyBERT NER (15MB) for entity extraction (Person, Organization, Location, Misc)
- RocksDB for durable persistence across restarts

**Cognitive Processing**

- *Named entity recognition* — TinyBERT extracts entities; boosts importance and enables graph relationships
- *Spreading activation retrieval* — queries activate related memories through semantic and graph connections [5]
- *Activation decay* — exponential decay A(t) = A₀ · e^(-λt) applied each maintenance cycle
- *Hebbian strengthening* — co-retrieved memories form graph edges; weight increases on co-activation
- *Long-term potentiation* — edges surviving threshold co-activations become permanent

**Semantic Consolidation**

- Episodic memories older than 7 days compress into semantic facts
- Entity extraction preserves key information during compression

### Use cases

**Local LLM memory** — Give Claude, GPT, or any local model persistent memory across sessions.

**Robotics & drones** — On-device experience accumulation without cloud round-trips.

**Edge AI** — Run on Jetson, Raspberry Pi, industrial PCs. Sub-millisecond retrieval, zero network dependency.

**Personal knowledge base** — Your own searchable memory. Decisions, learnings, discoveries—private and local.

### Compared to alternatives

| | Shodh-Memory | Mem0 | Cognee |
|---|---|---|---|
| **Deployment** | Single 8MB binary | Cloud API | Neo4j + Vector DB |
| **Offline** | 100% | No | Partial |
| **Learning** | Hebbian + decay + LTP | Vector similarity | Knowledge graphs |
| **Latency** | Sub-millisecond | Network-bound | Database-bound |
| **Best for** | Local-first, edge, privacy | Cloud scale | Enterprise ETL |

### Performance

Measured on Intel i7-1355U (10 cores, 1.7GHz), release build.

**API Latencies**

| Endpoint | Operation | Latency |
|----------|-----------|---------|
| `POST /api/remember` | Store memory (existing user) | **55-60ms** |
| `POST /api/recall` | Semantic search | **34-58ms** |
| `POST /api/recall/tags` | Tag-based search | **~1ms** |
| `GET /api/list` | List memories | **~1ms** |
| `GET /health` | Health check | **~1ms** |

**Knowledge Graph (Criterion benchmarks)**

| Operation | Latency |
|-----------|---------|
| Entity lookup | 763ns |
| Relationship query | 2.2µs |
| Hebbian strengthen | 5.7µs |
| Graph traversal (3-hop) | 30µs |

**Neural Models**

| Model | Operation | Latency |
|-------|-----------|---------|
| MiniLM-L6-v2 (25MB) | Embedding (384-dim) | 33ms |
| TinyBERT-NER (15MB) | Entity extraction | 15ms |

### Installation

**Claude Code / Claude Desktop / Cursor:**

Add to your MCP config file:
```json
{
  "mcpServers": {
    "shodh-memory": {
      "command": "npx",
      "args": ["-y", "@shodh/memory-mcp"],
      "env": {
        "SHODH_API_KEY": "your-api-key"
      }
    }
  }
}
```

Use the same API key you set for the server (see [Authentication](#authentication) below).

Config file locations:

| Editor | Config Path |
|--------|-------------|
| Claude Desktop (macOS) | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Desktop (Windows) | `%APPDATA%\Claude\claude_desktop_config.json` |
| Claude Desktop (Linux) | `~/.config/Claude/claude_desktop_config.json` |
| Cursor | `~/.cursor/mcp.json` |

**Python:**
```
pip install shodh-memory
```

**From source:**
```
cargo build --release
./target/release/shodh-memory-server
```

### Usage

**Python**

```python
from shodh_memory import Memory

# Option 1: Pass API key directly
memory = Memory(api_key="your-api-key", storage_path="./my_data")

# Option 2: Use environment variable (recommended)
# export SHODH_API_KEY=your-api-key
memory = Memory(storage_path="./my_data")

# Store
memory.remember("User prefers dark mode", memory_type="Decision")
memory.remember("JWT tokens expire after 24h", memory_type="Learning")

# Search
results = memory.recall("user preferences", limit=5)

# Get memory statistics
stats = memory.get_stats()
```

**REST API**

```bash
# Store
curl -X POST http://localhost:3030/api/remember \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"user_id": "agent-1", "content": "Deployment requires Docker 24+", "tags": ["deployment"]}'

# Search
curl -X POST http://localhost:3030/api/recall \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"user_id": "agent-1", "query": "deployment requirements", "limit": 5}'
```

### Memory types

Different types get different importance weights in the scoring model:

- **Decision** (+0.30) — choices, preferences, conclusions
- **Learning** (+0.25) — new knowledge, facts learned
- **Error** (+0.25) — mistakes, things to avoid
- **Discovery**, **Pattern** (+0.20) — findings, recurring behaviors
- **Task** (+0.15) — work items
- **Context**, **Observation** (+0.10) — general info


### API reference

**Python client** (API parity with REST)

| Method | What it does |
|--------|--------------|
| **Core Memory** ||
| `remember(content, memory_type, tags, ...)` | Store a memory |
| `recall(query, limit, mode, ...)` | Semantic search |
| `list_memories(limit, memory_type)` | List all memories |
| `get_memory(memory_id)` | Get single memory by ID |
| `get_stats()` | Memory statistics |
| **Forget Operations** ||
| `forget(memory_id)` | Delete single memory by ID |
| `forget_by_age(days)` | Delete memories older than N days |
| `forget_by_importance(threshold)` | Delete low-importance memories |
| `forget_by_pattern(regex)` | Delete memories matching pattern |
| `forget_by_tags(tags)` | Delete memories by tags |
| `forget_by_date(start, end)` | Delete memories in date range |
| `forget_all()` | Delete ALL memories (GDPR) |
| **Context & Introspection** ||
| `context_summary(max_items, ...)` | Categorized context for LLM bootstrap |
| `brain_state(longterm_limit)` | 3-tier memory visualization |
| `flush()` | Flush data to disk |

**REST endpoints**

All protected endpoints require `X-API-Key` header.

| Endpoint | Method | Description | Avg Latency |
|----------|--------|-------------|-------------|
| **Core Memory** ||||
| `/api/remember` | POST | Store memory (embedding + NER) | 55ms |
| `/api/recall` | POST | Semantic search | 45ms |
| `/api/recall/tags` | POST | Tag-based search (no embedding) | 1ms |
| `/api/recall/date` | POST | Date-range search | 5ms |
| `/api/list/{user_id}` | GET | List all memories | 1ms |
| `/api/context_summary` | POST | Categorized context for session bootstrap | 15ms |
| **Forget Operations** ||||
| `/api/forget/age` | POST | Delete memories older than threshold | 5ms |
| `/api/forget/importance` | POST | Delete low-importance memories | 5ms |
| `/api/forget/pattern` | POST | Delete memories matching regex | 10ms |
| `/api/forget/tags` | POST | Delete memories by tags | 5ms |
| `/api/forget/date` | POST | Delete memories in date range | 5ms |
| **Hebbian Learning** ||||
| `/api/recall/tracked` | POST | Search with Hebbian feedback tracking | 45ms |
| `/api/reinforce` | POST | Hebbian reinforcement feedback | 10ms |
| **Batch & Consolidation** ||||
| `/api/batch_remember` | POST | Store multiple memories | 55ms/item |
| `/api/consolidate` | POST | Trigger semantic consolidation | 250ms |
| **Introspection** ||||
| `/api/memory/{id}` | GET/PUT/DELETE | Single memory operations | 10ms |
| `/api/users/{id}/stats` | GET | User statistics | 10ms |
| `/api/graph/{id}/stats` | GET | Knowledge graph statistics | 10ms |
| `/api/brain/{user_id}` | GET | 3-tier state visualization | 50ms |
| `/api/search/advanced` | POST | Multi-filter search | 50ms |
| **Health & Metrics** ||||
| `/health` | GET | Health check (no auth) | <1ms |
| `/health/live` | GET | Kubernetes liveness (no auth) | <1ms |
| `/health/ready` | GET | Kubernetes readiness (no auth) | <1ms |
| `/metrics` | GET | Prometheus metrics (no auth) | <1ms |

**Authentication**

Server and clients must use the same API key. Set once and use everywhere.

```bash
# Development: Pick any key for local development
export SHODH_DEV_API_KEY="my-dev-key"   # Server uses this
export SHODH_API_KEY="my-dev-key"       # Clients use this (same value!)

# Then use with curl:
curl -H "X-API-Key: my-dev-key" ...

# Production: Set multiple keys (comma-separated)
export SHODH_API_KEYS="your-secure-key-1,your-secure-key-2"
export SHODH_ENV=production
```

For MCP integrations, add `SHODH_API_KEY` to the `env` section of your config (see Installation above).

### Configuration

All settings via environment variables. Create a `.env` file or export directly.

```bash
# Server
SHODH_PORT=3030                    # Default: 3030
SHODH_MEMORY_PATH=./data           # Default: ./shodh_memory_data
SHODH_ENV=production               # Set for production mode

# Authentication (Server)
SHODH_API_KEYS=key1,key2           # Required in production (comma-separated)
SHODH_DEV_API_KEY=my-dev-key       # For development (if SHODH_API_KEYS not set)

# Authentication (Clients - Python, MCP, REST)
SHODH_API_KEY=my-dev-key           # Must match server's accepted keys

# Cognitive Parameters
SHODH_MAINTENANCE_INTERVAL=300     # Decay cycle in seconds (default: 300)
SHODH_ACTIVATION_DECAY=0.95        # Decay factor per cycle (default: 0.95)

# Integrations (optional)
LINEAR_API_URL=https://api.linear.app/graphql
LINEAR_WEBHOOK_SECRET=your-secret
GITHUB_API_URL=https://api.github.com
GITHUB_WEBHOOK_SECRET=your-secret

# Logging
RUST_LOG=info                      # Options: error, warn, info, debug, trace
```

### Platform support

| Platform | Status | Use case |
|----------|--------|----------|
| Linux x86_64 | ✓ | Servers, workstations |
| macOS ARM64 | ✓ | Development (Apple Silicon) |
| Windows x86_64 | ✓ | Development, industrial PCs |
| Linux ARM64 | Coming soon | Jetson, Raspberry Pi, drones |

### References

[1] Cowan, N. (2010). The Magical Mystery Four: How is Working Memory Capacity Limited, and Why? *Current Directions in Psychological Science*, 19(1), 51-57. https://pmc.ncbi.nlm.nih.gov/articles/PMC4207727/

[2] Magee, J.C., & Grienberger, C. (2020). Synaptic Plasticity Forms and Functions. *Annual Review of Neuroscience*, 43, 95-117. https://pmc.ncbi.nlm.nih.gov/articles/PMC10410470/

[3] Subramanya, S.J., et al. (2019). DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node. *NeurIPS 2019*. https://papers.nips.cc/paper/9527-diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node

[4] Dudai, Y., Karni, A., & Born, J. (2015). The Consolidation and Transformation of Memory. *Neuron*, 88(1), 20-32. https://pmc.ncbi.nlm.nih.gov/articles/PMC4183265/

[5] Anderson, J.R. (1983). A Spreading Activation Theory of Memory. *Journal of Verbal Learning and Verbal Behavior*, 22(3), 261-295.

### License

Apache 2.0

---

[MCP Registry](https://registry.modelcontextprotocol.io/v0/servers?search=shodh) · [PyPI](https://pypi.org/project/shodh-memory/) · [npm](https://www.npmjs.com/package/@shodh/memory-mcp) · [GitHub](https://github.com/varun29ankuS/shodh-memory)
