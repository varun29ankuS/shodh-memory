# Shodh-Memory Architecture

Cognitive memory infrastructure for AI agents, grounded in neuroscience research.

## Overview

Shodh-memory is not a vector database. It's a **cognitive memory system** that models how biological memory actually works—with decay, consolidation, interference, and learning that strengthens with use.

```
┌─────────────────────────────────────────────────────────────────┐
│                        COGNITIVE LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Hebbian   │  │   Memory    │  │   Spreading Activation  │  │
│  │  Learning   │  │   Replay    │  │   & Interference        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      3-TIER MEMORY STORE                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Working   │→ │   Session   │→ │      Long-Term          │  │
│  │   Memory    │  │   Memory    │  │      Memory             │  │
│  │  (7±2 items)│  │ (minutes)   │  │  (days to permanent)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      STORAGE & INDEX                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  RocksDB    │  │   Vamana    │  │    Knowledge Graph      │  │
│  │  (durable)  │  │   (HNSW)    │  │    (entities + edges)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Documents

| Document | Description |
|----------|-------------|
| [Neuroscience Foundations](./01-neuroscience-foundations.md) | The cognitive science research behind shodh-memory |
| [3-Tier Memory Model](./02-three-tier-memory.md) | Working, session, and long-term memory tiers |
| [Hebbian Learning](./03-hebbian-learning.md) | "Neurons that fire together, wire together" |
| [Knowledge Graph](./04-knowledge-graph.md) | Entity extraction, relationships, spreading activation |
| [Memory Consolidation](./05-memory-consolidation.md) | Replay, fact extraction, interference detection |
| [Decay & Forgetting](./06-decay-and-forgetting.md) | Hybrid exponential + power-law decay model |

## Why This Matters for AI Agents

Current AI memory solutions are retrieval systems—they find what you stored. Shodh-memory is a **learning system**—it strengthens connections that matter and lets irrelevant information fade.

### The Problem with RAG

Traditional RAG treats all memories equally:
- Store embedding → retrieve by similarity → done
- No learning from what's useful vs. what's noise
- No consolidation of repeated patterns into durable knowledge
- No decay of stale information

### How Shodh Differs

Shodh-memory models cognitive processes:

1. **Memories compete for attention** — Working memory holds 7±2 items, forcing prioritization
2. **Use strengthens memories** — Hebbian learning increases edge weights between co-accessed memories
3. **Patterns become facts** — Repeated associations consolidate into semantic knowledge
4. **Unused memories decay** — Hybrid decay model based on Ebbinghaus + Wixted research
5. **Interference is modeled** — New similar memories can interfere with old ones

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Graph lookup | <1μs | In-memory adjacency list |
| Semantic search | 34-58ms | Vamana HNSW, 384-dim embeddings |
| Memory store | 40-80ms | Includes embedding + indexing |
| Consolidation cycle | Background | Runs every 5 minutes |

## Single Binary Philosophy

Shodh-memory ships as a single ~15MB binary with:
- Embedded ONNX runtime for embeddings
- Bundled MiniLM-L6 model (22M params)
- No external dependencies
- No cloud connectivity required

This enables deployment on:
- Edge devices (Raspberry Pi, Jetson)
- Air-gapped environments
- Privacy-sensitive contexts
- Offline-first applications

## Integration Points

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Claude Code    │     │   Python SDK     │     │   REST API       │
│   (via hooks)    │     │   (PyO3)         │     │   (OpenAPI 3.1)  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      MCP Server           │
                    │   (Model Context Protocol)│
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Shodh-Memory Core      │
                    │         (Rust)            │
                    └───────────────────────────┘
```

## Next Steps

- [Quick Start Guide](../quickstart.md)
- [API Reference](../../specs/openapi.yaml)
- [Memory Schema](../../specs/schemas/memory.md)
