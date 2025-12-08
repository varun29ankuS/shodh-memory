<p align="center">
  <img src="https://raw.githubusercontent.com/varun29ankuS/shodh-memory/main/assets/logo.png" width="120" alt="Shodh-Memory">
</p>

<h1 align="center">Shodh-Memory</h1>

<p align="center">
  <a href="https://github.com/varun29ankuS/shodh-memory/actions"><img src="https://github.com/varun29ankuS/shodh-memory/workflows/CI/badge.svg" alt="build"></a>
  <a href="https://registry.modelcontextprotocol.io/v0/servers/io.github.varun29ankuS/shodh-memory"><img src="https://img.shields.io/badge/MCP-Registry-green" alt="MCP Registry"></a>
  <a href="https://crates.io/crates/shodh-memory"><img src="https://img.shields.io/crates/v/shodh-memory.svg" alt="crates.io"></a>
  <a href="https://crates.io/crates/shodh-memory"><img src="https://img.shields.io/crates/d/shodh-memory.svg?label=crates.io%20downloads" alt="crates.io Downloads"></a>
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/v/@shodh/memory-mcp.svg?logo=npm" alt="npm"></a>
  <a href="https://pypi.org/project/shodh-memory/"><img src="https://img.shields.io/pypi/v/shodh-memory.svg" alt="PyPI"></a>
  <a href="https://pepy.tech/project/shodh-memory"><img src="https://static.pepy.tech/badge/shodh-memory" alt="PyPI Downloads"></a>
  <a href="https://www.npmjs.com/package/@shodh/memory-mcp"><img src="https://img.shields.io/npm/dm/@shodh/memory-mcp.svg?label=npm%20downloads" alt="npm Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

---

<p align="center"><i>Memory that learns. Single binary. Runs offline.</i></p>

---

We built this because AI agents forget everything between sessions. They make the same mistakes, ask the same questions, lose context constantly.

Shodh-Memory fixes that. It's a cognitive memory system—Hebbian learning, activation decay, semantic consolidation—packed into a single 8MB binary that runs offline.

**How it works:**

Experiences flow through three tiers based on Cowan's working memory model<br>
[1]. New information enters capacity-limited working memory, overflows into session storage, and consolidates into long-term memory based on importance. When memories are retrieved together successfully, their connections strengthen—classic Hebbian learning<br>
[2]. After enough co-activations, those connections become permanent. Unused memories naturally fade. The system learns what matters to *you*.

**What you get:**

Your decisions, errors, and patterns—searchable and private. No cloud. No API keys. Your memory, your machine.

```
Working Memory ──overflow──▶ Session Memory ──importance──▶ Long-Term Memory
   (100 items)                  (500 MB)                      (RocksDB)
```

### Architecture

**Storage & Retrieval**

- Vamana graph index for approximate nearest neighbor search [3]
- MiniLM-L6 embeddings (384-dim, 25MB) for semantic similarity
- TinyBERT NER (15MB) for named entity extraction (Person, Organization, Location, Misc)
- RocksDB for durable persistence across restarts
- User isolation — each agent gets independent memory space

**Cognitive Processing**

- *Named entity recognition* — TinyBERT extracts Person, Organization, Location, Misc entities on every memory store; entities boost importance and enable graph relationships
- *Activation decay* — exponential decay A(t) = A₀ · e^(-λt) applied each maintenance cycle (λ configurable)
- *Hebbian strengthening* — co-retrieved memories form graph edges; edge weight w increases as w' = w + α(1 - w) on each co-activation
- *Long-term potentiation* — edges surviving threshold co-activations (default: 5) become permanent, exempt from decay
- *Importance scoring* — composite score from memory type, content length, entity density, technical terms, access frequency

**Semantic Consolidation**

- Episodic memories older than 7 days compress into semantic facts
- Entity extraction preserves key information during compression
- Original experiences archived, compressed form used for retrieval

**Context Bootstrapping**

- `context_summary()` provides categorized session context on startup
- Returns decisions, learnings, patterns, errors — structured for LLM consumption
- `brain_state()` exposes full 3-tier visualization data

### Use cases

**Local LLM memory** — Give Claude, GPT, or any local model persistent memory across sessions. Remember user preferences, past decisions, learned patterns.

**Robotics & drones** — On-device experience accumulation. A robot that remembers which actions worked, which failed, without cloud round-trips.

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

Measured with cold TCP connections on Intel i7-1355U (10 cores, 1.7GHz), release build.

**Real-Time API Latencies**

| Endpoint | Operation | Latency | Notes |
|----------|-----------|---------|-------|
| `POST /api/remember` | Store memory (new user) | **227-250ms** | Embedding + RocksDB + index |
| `POST /api/remember` | Store memory (existing user) | **55-60ms** | Embedding + storage |
| `POST /api/recall` | Semantic search | **34-58ms** | Embedding + vector search |
| `POST /api/recall/tags` | Tag-based search | **~1ms** | No embedding needed |
| `GET /api/list` | List memories | **~1ms** | Direct DB read |
| `DELETE /api/forget` | Delete memory | **~1ms** | Direct DB delete |
| `GET /health` | Health check | **~1ms** | HTTP baseline |

**Latency Breakdown (Remember Operation)**

| Component | Time |
|-----------|------|
| MiniLM-L6-v2 embedding | ~33ms |
| TinyBERT NER extraction | ~15ms |
| User lookup/create | ~50-80ms |
| RocksDB write | ~20-30ms |
| Vamana index update | ~20-30ms |
| HTTP overhead | ~1ms |

**Server-Side Metrics (Prometheus)**

```
Embedding generation: 112ms average (3.24s / 29 calls)
Distribution:
  <50ms:  48% of calls
  <100ms: 52% of calls
  <250ms: 97% of calls
```

The system uses two neural models: MiniLM-L6-v2 (25MB, 6-layer Transformer) for semantic embeddings and TinyBERT-NER (15MB) for named entity extraction. Both run quantized INT8 inference via ONNX Runtime.

### Installation

**Claude Code / Claude Desktop:**

Add to your `claude_desktop_config.json`:
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

memory = Memory(user_id="my-agent")

# Store
memory.remember("User prefers dark mode", memory_type="Decision")
memory.remember("JWT tokens expire after 24h", memory_type="Learning")

# Search
results = memory.recall("user preferences", limit=5)

# Session bootstrap - get categorized context
summary = memory.context_summary()
# Returns: decisions, learnings, patterns, errors
```

**REST API**

```
# Store
curl -X POST http://localhost:3030/api/record \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "user_id": "agent-1",
    "experience": {
      "content": "Deployment requires Docker 24+",
      "experience_type": "Learning"
    }
  }'

# Search
curl -X POST http://localhost:3030/api/retrieve \
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

Importance also increases with: content length, entity density, technical terms, and access frequency.

### API reference

**Python client**

| Method | What it does |
|--------|--------------|
| `remember(content, memory_type, tags)` | Store a memory |
| `recall(query, limit)` | Semantic search |
| `context_summary()` | Categorized context for session start |
| `brain_state()` | 3-tier visualization data |
| `stats()` | Memory statistics |
| `delete(memory_id)` | Remove a memory |

**REST endpoints**

All protected endpoints require `X-API-Key` header.

| Endpoint | Method | Description | Avg Latency |
|----------|--------|-------------|-------------|
| **Core Memory** ||||
| `/api/remember` | POST | Store memory (embedding + NER) | 55ms |
| `/api/recall` | POST | Semantic search | 45ms |
| `/api/recall/tags` | POST | Tag-based search (no embedding) | 1ms |
| `/api/recall/entities` | POST | Entity-based retrieval (NER) | 10ms |
| `/api/list/{user_id}` | GET | List all memories | 1ms |
| `/api/context_summary` | POST | Categorized context for session bootstrap | 15ms |
| **Forget Operations** ||||
| `/api/forget/age` | POST | Delete memories older than threshold | 5ms |
| `/api/forget/importance` | POST | Delete low-importance memories | 5ms |
| `/api/forget/pattern` | POST | Delete memories matching regex | 10ms |
| `/api/forget/tags` | POST | Delete memories by tags | 5ms |
| `/api/forget/date` | POST | Delete memories in date range | 5ms |
| **Hebbian Learning** ||||
| `/api/retrieve/tracked` | POST | Search with feedback tracking | 45ms |
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

**Neural Model Latencies**

| Model | Operation | Avg Latency |
|-------|-----------|-------------|
| MiniLM-L6-v2 (25MB) | Embedding generation (384-dim) | 33ms |
| TinyBERT-NER (15MB) | Entity extraction | 15ms |

*Latencies measured on Intel i7-1355U (10 cores), release build, warm cache.*

**Authentication**

```bash
# Development mode (SHODH_API_KEYS not set)
curl -H "X-API-Key: sk-shodh-dev-4f8b2c1d9e3a7f5b6d2c8e4a1b9f7d3c" ...

# Production mode (required)
export SHODH_API_KEYS="your-secure-key-1,your-secure-key-2"
export SHODH_ENV=production
```

**Example: Store and retrieve with Hebbian feedback**

```bash
# 1. Store a memory
curl -X POST http://localhost:3030/api/remember \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"user_id": "agent-1", "content": "Docker requires port 8080", "tags": ["docker"]}'
# Response: {"id": "abc-123", "success": true}

# 2. Retrieve with tracking
curl -X POST http://localhost:3030/api/retrieve/tracked \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"user_id": "agent-1", "query": "port configuration", "limit": 5}'
# Response: {"tracking_id": "xyz-789", "memories": [...]}

# 3. Send feedback (strengthens associations)
curl -X POST http://localhost:3030/api/reinforce \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"user_id": "agent-1", "memory_ids": ["abc-123"], "outcome": "helpful"}'
# Response: {"memories_processed": 1, "associations_strengthened": 1}
```

### Configuration

```
SHODH_PORT=3030                    # Default: 3030
SHODH_MEMORY_PATH=./data           # Default: ./shodh_memory_data
SHODH_API_KEYS=key1,key2           # Required in production
SHODH_MAINTENANCE_INTERVAL=300     # Decay cycle (seconds)
SHODH_ACTIVATION_DECAY=0.95        # Decay factor per cycle
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

### License

Apache 2.0

---

[MCP Registry](https://registry.modelcontextprotocol.io/v0/servers/io.github.varun29ankuS/shodh-memory) · [PyPI](https://pypi.org/project/shodh-memory/) · [npm](https://www.npmjs.com/package/@shodh/memory-mcp) · [GitHub](https://github.com/varun29ankuS/shodh-memory)
