# Ambient Memory Priming for Autonomites

## Problem

Three structural issues prevent shodh-memory from serving autonomites as seamless cognitive infrastructure:

1. **Pull-based retrieval doesn't scale.** Agents must explicitly call MCP/API to get memories. This requires knowing what to ask for, adds latency per call, and creates a visible seam between "what I know" and "what I looked up." Memory should feel like training data — ambient, not queried.

2. **NER fails on domain-specific data.** The current graph is built from NER entity extraction. On ISAP security/OSINT data, NER classifies 100% of entities as `Other:MISC`. The knowledge graph — the system's most differentiated component — is built on noise. 178 entities, zero meaningful classifications, 1846 edges reinforcing co-occurrence of generic terms like "infrastructure" and "bulletin."

3. **Rigid templates waste tokens.** Structured memory schemas, entity ontologies, and constrained tool interfaces add friction without proportional value. The system should be minimal on ceremony, maximal on signal.

## Design

### Core Principle

**Firehose in, compile out.**

Everything flows into shodh-memory without curation. Decay, consolidation, and feedback filter quality over time. At retrieval time, the pipeline — not the agent — asks shodh-memory for relevant context and weaves it into the agent's prompt. The agent never sees a memory block. It just thinks better.

### 1. Encoding: Embeddings + Keywords, No NER

**Current path:**
```
text → NER entity extraction → entity nodes → graph edges → embeddings → store
```

**New path:**
```
text → embedding + YAKE keyword extraction → store
       (NER entity extraction removed from hot path)
       (keywords feed BM25 index for term matching)
```

Changes:
- Remove NER from the encoding pipeline. No entity ontology, no typed nodes.
- Retain YAKE keyword extraction — lightweight, domain-agnostic, feeds BM25 directly. We don't need to know "Thailand" is a Location; we just need to know it's a significant term for keyword matching.
- Compute embeddings for semantic search (unchanged).
- All autonomite output flows in automatically via pipeline hooks.
- `created_at` timestamp preserved from source data for accurate decay computation.
- Importance starts uniform; feedback and usage patterns differentiate over time.
- Consolidation handles deduplication and progressive distillation.

What this means: encoding latency drops (NER removed, YAKE already runs). No entity ontology to maintain. No domain-specific extraction rules. Embeddings capture semantic content. Keywords enable precise term matching via BM25. The graph learns structure from usage, not from a parser that doesn't understand the domain.

### 2. Graph: Emergent from Usage, Not Extraction

**Current graph construction:**
```
NER extracts "indonesia" (MISC) and "campaign" (MISC) from same text
  → creates entity nodes
  → creates edge between them
  → Hebbian learning strengthens edge on re-encounter
```

**New graph construction:**
```
Memory A and Memory B retrieved together for a useful context
  → co-retrieval recorded
  → Hebbian learning strengthens association
  → repeated co-retrieval promotes to higher tier

Memory C's embedding is close to Memory A in vector space
  → spreading activation flows between them
  → if C proves useful when A was the retrieval cue, edge strengthens
```

The graph emerges from two signals:
1. **Embedding proximity** — memories that are semantically close form natural neighborhoods
2. **Co-retrieval patterns** — memories that get surfaced together and prove useful form reinforced associations

All existing machinery stays:
- Hebbian learning (operates on co-retrieval edges instead of NER edges)
- Tier promotion (L1→L2→L3 based on edge strength thresholds)
- LTP / burst detection (same temporal dynamics)
- Spreading activation (traverses co-retrieval graph instead of entity graph)
- Decay curves (hybrid exponential→power-law, unchanged)

**Cluster labeling (optional, deferred):** For interpretability, embedding-space clusters can be lazily labeled by extracting the most representative terms from cluster members. This is a read-path convenience, not a write-path requirement.

### 3. Retrieval: Pipeline-Time Compilation

**Current flow:**
```
pipeline → launches autonomite → agent calls MCP recall → gets memories → works
```

**New flow:**
```
pipeline → calls compile_context(autonomite, task, workspace_state)
         → shodh-memory returns enrichment text
         → prompt_template.py weaves into prompt alongside playbook/CLAUDE.md
         → launches autonomite
         → agent works with enriched context (never calls recall)
```

#### New Endpoint: `POST /api/compile_context`

Request:
```json
{
  "user_id": "autonomites-pipeline",
  "autonomite": "chain-tracer",
  "task_description": "Trace redirect chains for newly discovered domains",
  "workspace_context": {
    "recent_files": ["artifacts/chain-tracer/run_2026-03-26.json"],
    "git_recent": "3 commits: added 12 domains, updated dork queries"
  },
  "max_tokens": 2000,
  "format": "prose"
}
```

Response:
```json
{
  "context": "Recent chain-tracer runs found that .ac.th domains...",
  "memory_ids": ["uuid1", "uuid2"],
  "confidence": 0.72
}
```

Key properties:
- Returns **pre-formatted text**, not JSON memory objects. Ready to splice into a prompt.
- `max_tokens` lets the pipeline control context budget.
- `format: "prose"` returns natural text; `format: "structured"` returns categorized bullet points.
- `memory_ids` returned for feedback attribution (pipeline reports success/failure, these memories get credit/blame).
- Retrieval uses the full hybrid pipeline: BM25 + vector search + spreading activation + cognitive boosting.
- The query is constructed internally from the structured metadata — better than anything the agent would cobble together mid-run.

#### Cross-Agent Priming

All autonomites write to the same user namespace (`autonomites-pipeline`). When chain-tracer's runs strengthen certain regions of embedding space, dork-builder's next compile_context naturally picks up those associations through spreading activation. No explicit cross-referencing needed.

### 4. Feedback Loop

```
compile_context returns memory_ids
  → pipeline includes them in prompt
  → autonomite runs
  → pipeline reports outcome to shodh-memory:
      POST /api/feedback
      {
        "memory_ids": ["uuid1", "uuid2"],
        "outcome": "success",
        "autonomite": "chain-tracer",
        "run_id": "run_2026-03-26T..."
      }
  → helpful memories get importance boost
  → co-surfaced memories get Hebbian edge strengthening
  → unhelpful memories decay faster
```

This closes the loop: memories that contribute to successful runs get reinforced. Memories that don't, fade. The system learns what's actually useful, not what co-occurs syntactically.

## Validation Plan

### Phase 1: Data Loading

Load existing ISAP workspace artifacts into shodh-memory with embeddings-only encoding:

- **Autonomite run artifacts** (~100 JSON files across 8 autonomites, real timestamps from March 21-26)
- **Git commit messages and diffs** (structured change history)
- **Raw Claude session JSON** (full reasoning chains from autonomite runs)

Use `created_at` parameter to preserve real timestamps. This gives 5 days of data spanning the consolidation phase crossover (3 days).

### Phase 2: Consolidation Stress Test

Run consolidation cycles to simulate the passage of time:

1. Load all data with real timestamps (current state: 0-5 days old)
2. Run consolidation → observe what gets replayed, what edges form
3. Backdate a copy of the data by 30 days → run consolidation → compare
4. Backdate by 90 days → run consolidation → see what survives the power-law tail

Goal: verify that high-signal memories (real findings, cross-autonomite connections) survive while noise (generic observations, routine status reports) decays.

### Phase 3: Retrieval Quality Test

For each autonomite, construct a realistic `compile_context` request and evaluate:

1. **Relevance** — are the returned memories actually useful for the task?
2. **Cross-agent transfer** — does chain-tracer context surface when relevant to dork-builder?
3. **Temporal awareness** — do recent findings rank higher than stale ones?
4. **Signal-to-noise** — compare quality vs current NER-graph retrieval on same queries

### Phase 4: Pipeline Integration Test

Wire `compile_context` into `prompt_template.py` for a single autonomite. Run it against the real ISAP pipeline and compare output quality with and without memory priming.

## What Changes

| Component | Change | Risk |
|-----------|--------|------|
| shodh-memory encoding | Remove NER from hot path, embeddings only | Low — NER results currently useless |
| shodh-memory graph | Co-retrieval based edges replace NER edges | Medium — new graph construction logic |
| shodh-memory API | New `/api/compile_context` endpoint | Low — additive |
| shodh-memory API | New `/api/feedback` endpoint (structured) | Low — additive |
| Pipeline prompt_template.py | Call compile_context during assembly | Low — additive |
| Pipeline orchestrator.py | Report run outcomes via feedback API | Low — additive |

## What Stays

- Hebbian learning mechanics (input changes, algorithm unchanged)
- Hybrid decay curves (exponential→power-law)
- Tier promotion (L1→L2→L3)
- Consolidation / replay system
- Vector search (Vamana HNSW)
- BM25 full-text search (Tantivy)
- Feedback attribution signals
- MCP tools (available for manual use, not primary interface)
- TUI dashboard

## What Gets Removed

- NER entity extraction from encoding hot path
- Fixed entity ontology (Person, Organization, Location, etc.)
- Entity-based graph construction
- Rigid memory type templates

## Design Decisions

1. **NER vs keywords** — NER is removed entirely, not kept as a fallback. Even when NER correctly identifies entities (e.g., "Thailand" as Location), the embedding already captures the same semantics. The typed label adds interpretability for human inspection but doesn't improve retrieval quality. YAKE keyword extraction is retained — it's domain-agnostic, fast, and directly useful for BM25 term matching without requiring an ontology. One path, no branching logic.

2. **Graph migration** — clear the existing MISC graph and rebuild from co-retrieval edges during validation. The current graph (178 all-MISC entities, 1846 noise edges) has no signal worth preserving.

## Open Questions

1. **Cluster labeling strategy** — how to make the emergent graph inspectable? Options: most-representative-terms from YAKE keywords within cluster, LLM-generated cluster summaries, or leave unlabeled and rely on the TUI universe visualization.

2. **Session JSON parsing** — how much of the raw Claude session data to ingest? Full reasoning chains are rich but verbose. May want to extract key decisions/observations rather than raw transcript.

3. **Context budget allocation** — how should `max_tokens` be distributed across memory types? Recent findings vs established patterns vs cross-agent insights?
