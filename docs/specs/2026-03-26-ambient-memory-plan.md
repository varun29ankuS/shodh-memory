# Ambient Memory Priming — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove NER from encoding, add compile_context endpoint for pipeline-time retrieval, load real session data to validate.

**Architecture:** Skip NER in remember handler (keep YAKE keywords for BM25), add `/api/compile_context` endpoint that returns pre-formatted text from hybrid retrieval, build a Python loader for Claude session JSON. The co-retrieval graph infrastructure already exists (`record_memory_coactivation`) — we just stop building the NER graph and let co-retrieval edges become the primary associative structure.

**Tech Stack:** Rust (axum, serde, tokio), Python 3 (loader script), existing shodh-memory infrastructure (RocksDB, Vamana, Tantivy, YAKE)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/handlers/remember.rs` | Modify | Skip NER extraction, keep YAKE |
| `src/handlers/state.rs` | Modify | Skip NER-based graph construction when no NER entities |
| `src/handlers/recall.rs` | Modify | Add `compile_context` handler |
| `src/handlers/router.rs` | Modify | Register new endpoints |
| `src/handlers/types.rs` | Modify | Add request/response types |
| `scripts/load_sessions.py` | Create | Session JSON → bulk remember API calls |
| `tests/compile_context_tests.rs` | Create | Integration tests for new endpoint |

---

### Task 1: Skip NER in Encoding Pipeline

**Files:**
- Modify: `src/handlers/remember.rs:304-358`
- Modify: `src/handlers/state.rs:1915-1935`

The NER extraction at remember.rs:314-331 runs in a `spawn_blocking` alongside YAKE. We replace the NER block with an immediate empty vec — YAKE continues to run for keyword/BM25 indexing.

- [ ] **Step 1: Replace NER extraction with empty vec in remember handler**

In `src/handlers/remember.rs`, replace the NER spawn_blocking block (lines 304-347) while keeping YAKE:

```rust
    // PERF: YAKE keyword extraction only — NER removed per ambient-memory-priming spec.
    // Co-retrieval graph (built during recall) replaces NER-based entity graph.
    // YAKE keywords feed BM25 index for precise term matching.
    let yake = state.get_keyword_extractor();
    let content_for_yake = req.content.clone();

    let yake_result = tokio::task::spawn_blocking(move || yake.extract_texts(&content_for_yake)).await;

    let ner_entities: Vec<NerEntityRecord> = Vec::new();
    let extracted_keywords = match yake_result {
        Ok(keywords) => keywords,
        Err(e) => {
            if e.is_panic() {
                tracing::error!("YAKE extraction task panicked: {:?}", e);
            } else {
                tracing::debug!("YAKE extraction task cancelled: {:?}", e);
            }
            Vec::new()
        }
    };
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /Volumes/chonk/projects/autonomites/shodh-memory && cargo check 2>&1 | tail -5`
Expected: compiles with no errors (warnings OK)

- [ ] **Step 3: Verify graph construction gracefully handles empty NER**

Read `src/handlers/state.rs:1915-1936` — the `process_experience_into_graph` function already has a fallback when `experience.ner_entities.is_empty()`. It falls through to regex-based extraction from `experience.entities` (which contains YAKE keywords + user tags). No code change needed — the existing fallback handles our case. Verify by reading the code.

- [ ] **Step 4: Test remember still works**

Run: `cargo test remember -- --nocapture 2>&1 | tail -20`
Expected: existing remember tests pass

- [ ] **Step 5: Manual smoke test against running server**

```bash
curl -s -X POST http://localhost:3033/api/remember \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d '{
    "user_id": "test-ambient",
    "content": "Chain-tracer found that Thai university domains (.ac.th) are being used as redirect infrastructure for gambling SEO campaigns targeting Indonesian users.",
    "memory_type": "observation"
  }' | python3 -m json.tool
```
Expected: 200 OK with memory_id, no NER errors in server logs

- [ ] **Step 6: Verify no NER entities in graph for test memory**

```bash
curl -s -X POST http://localhost:3033/api/graph/entities/all \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d '{"user_id": "test-ambient", "limit": 50}' | python3 -m json.tool
```
Expected: entities from YAKE keywords only (no PER/ORG/LOC typed entities), or entity count reduced vs NER path

- [ ] **Step 7: Commit**

```bash
git add src/handlers/remember.rs
git commit -m "feat: skip NER in encoding pipeline, retain YAKE for BM25 keywords

NER classified 100% of ISAP domain data as Other:MISC, producing a noise
graph. Remove NER from encoding hot path. YAKE keywords still feed BM25
for term matching. Co-retrieval graph (already built during recall) becomes
primary associative structure."
```

---

### Task 2: Add compile_context Endpoint

**Files:**
- Modify: `src/handlers/types.rs` (add request/response types)
- Modify: `src/handlers/recall.rs` (add handler)
- Modify: `src/handlers/router.rs` (register route)

- [ ] **Step 1: Add request/response types**

In `src/handlers/types.rs`, add:

```rust
/// Request for pipeline-time context compilation.
/// Returns pre-formatted text ready for prompt injection.
#[derive(Debug, Deserialize)]
pub struct CompileContextRequest {
    pub user_id: String,
    /// Name of the autonomite requesting context
    #[serde(default)]
    pub autonomite: Option<String>,
    /// What the autonomite is about to do
    pub task_description: String,
    /// Recent workspace state for additional retrieval signal
    #[serde(default)]
    pub workspace_context: Option<String>,
    /// Max approximate characters in returned context (default 4000)
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    /// Number of memories to retrieve before formatting (default 10)
    #[serde(default = "default_compile_limit")]
    pub limit: usize,
}

fn default_max_chars() -> usize { 4000 }
fn default_compile_limit() -> usize { 10 }

#[derive(Debug, Serialize)]
pub struct CompileContextResponse {
    /// Pre-formatted context text, ready for prompt injection
    pub context: String,
    /// Memory IDs used (for feedback attribution)
    pub memory_ids: Vec<String>,
    /// How many memories contributed
    pub memory_count: usize,
    /// Processing time in ms
    pub latency_ms: f64,
}
```

- [ ] **Step 2: Verify types compile**

Run: `cargo check 2>&1 | tail -5`

- [ ] **Step 3: Implement compile_context handler**

In `src/handlers/recall.rs`, add the handler. This builds a query from structured metadata, runs the existing hybrid retrieval pipeline, and formats results as text:

```rust
/// POST /api/compile_context - Pipeline-time context compilation
///
/// Takes structured metadata about what an autonomite is about to do,
/// retrieves relevant memories via hybrid search, and returns pre-formatted
/// text ready for prompt injection. The agent never sees a "memory block" —
/// it just gets a richer context.
pub async fn compile_context(
    State(state): State<AppState>,
    Json(req): Json<super::types::CompileContextRequest>,
) -> Result<Json<super::types::CompileContextResponse>, AppError> {
    let op_start = std::time::Instant::now();

    crate::handlers::validation::validate_user_id(&req.user_id)
        .map_validation_err("user_id")?;

    // Build a rich query from structured metadata
    let mut query_parts: Vec<String> = Vec::new();
    if let Some(ref autonomite) = req.autonomite {
        query_parts.push(format!("autonomite: {}", autonomite));
    }
    query_parts.push(req.task_description.clone());
    if let Some(ref workspace) = req.workspace_context {
        query_parts.push(workspace.clone());
    }
    let query_text = query_parts.join(". ");

    let memory = state
        .get_user_memory(&req.user_id)
        .map_err(AppError::Internal)?;

    // Use existing hybrid retrieval (BM25 + vector + spreading activation)
    let memory_query = crate::memory::Query {
        text: query_text,
        limit: req.limit,
        mode: crate::memory::RetrievalMode::Hybrid,
        ..Default::default()
    };

    let results = {
        let guard = memory.read().map_err(|e| {
            AppError::Internal(format!("Memory lock poisoned: {}", e))
        })?;
        guard.recall(&memory_query).unwrap_or_default()
    };

    // Format results as prose text, not JSON
    let mut context_parts: Vec<String> = Vec::new();
    let mut memory_ids: Vec<String> = Vec::new();
    let mut char_count = 0;

    for result in &results {
        let content = &result.content;
        if char_count + content.len() > req.max_chars {
            // Truncate last entry if needed to stay within budget
            let remaining = req.max_chars.saturating_sub(char_count);
            if remaining > 100 {
                let truncated: String = content.chars().take(remaining).collect();
                context_parts.push(truncated);
                memory_ids.push(result.id.to_string());
            }
            break;
        }
        context_parts.push(content.clone());
        memory_ids.push(result.id.to_string());
        char_count += content.len();
    }

    let context = if context_parts.is_empty() {
        String::new()
    } else {
        context_parts.join("\n\n---\n\n")
    };

    let memory_count = memory_ids.len();

    // Record co-activation for memories surfaced together
    if memory_ids.len() > 1 {
        let graph = state.get_user_graph(&req.user_id);
        if let Ok(graph) = graph {
            let uuids: Vec<uuid::Uuid> = memory_ids.iter()
                .filter_map(|id| uuid::Uuid::parse_str(id).ok())
                .collect();
            if let Ok(g) = graph.read() {
                let _ = g.record_memory_coactivation(&uuids);
            }
        }
    }

    let latency_ms = op_start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(super::types::CompileContextResponse {
        context,
        memory_ids,
        memory_count,
        latency_ms,
    }))
}
```

**Note:** The exact field names on `result` (content, id) and `Query` struct fields need to be verified against the actual types. Read `src/handlers/types.rs` for `RecallMemory` fields and `src/memory/mod.rs` for `Query` struct. Adapt field access accordingly.

- [ ] **Step 4: Register route**

In `src/handlers/router.rs`, add after the proactive context routes (~line 91):

```rust
        .route("/api/compile_context", post(recall::compile_context))
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check 2>&1 | tail -20`
Expected: compiles. Fix any type mismatches against actual struct definitions.

- [ ] **Step 6: Smoke test the endpoint**

```bash
curl -s -X POST http://localhost:3033/api/compile_context \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d '{
    "user_id": "autonomites-pipeline",
    "autonomite": "chain-tracer",
    "task_description": "Trace redirect chains for newly discovered Thai university domains",
    "limit": 5
  }' | python3 -m json.tool
```
Expected: 200 OK with `context` (text), `memory_ids` (array), `memory_count`, `latency_ms`

- [ ] **Step 7: Commit**

```bash
git add src/handlers/types.rs src/handlers/recall.rs src/handlers/router.rs
git commit -m "feat: add /api/compile_context for pipeline-time memory retrieval

Returns pre-formatted text ready for prompt injection instead of JSON
memory objects. Takes structured autonomite metadata (name, task, workspace
state) to build high-quality retrieval queries. Records co-activation
for memories surfaced together to build the co-retrieval graph."
```

---

### Task 3: Build Session JSON Loader

**Files:**
- Create: `scripts/load_sessions.py`

The loader parses Claude session JSON, extracts meaningful content per message (assistant reasoning, tool outputs, key decisions), and bulk-loads via the remember API with original timestamps.

- [ ] **Step 1: Create the loader script**

```python
#!/usr/bin/env python3
"""Load Claude session JSON into shodh-memory.

Extracts assistant reasoning and observations from session files,
loads them as memories with original timestamps preserved.

Usage:
    python3 scripts/load_sessions.py <session.json> [--api-url URL] [--api-key KEY] [--user-id ID] [--dry-run]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def extract_memories_from_session(session_path: str) -> list[dict]:
    """Extract meaningful memory units from a Claude session JSON file."""
    with open(session_path) as f:
        data = json.load(f)

    memories = []
    session_id = data.get("session", {}).get("id", "unknown")
    project_path = data.get("session", {}).get("projectPath", "")

    for msg in data.get("messages", []):
        if msg.get("type") != "assistant":
            continue

        timestamp = msg.get("timestamp")
        content_blocks = msg.get("content", [])
        if isinstance(content_blocks, str):
            # Simple text message
            if len(content_blocks.strip()) > 50:
                memories.append({
                    "content": content_blocks.strip(),
                    "created_at": timestamp,
                    "tags": ["session", session_id[:8]],
                })
            continue

        # Process content blocks
        text_parts = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "").strip()
                if text:
                    text_parts.append(text)

            elif block_type == "thinking":
                thinking = block.get("thinking", "").strip()
                if len(thinking) > 100:
                    # Extract key observations from thinking blocks
                    # These are the agent's reasoning — rich signal
                    # Chunk long thinking into ~1000 char segments
                    for i in range(0, len(thinking), 1500):
                        chunk = thinking[i:i + 1500].strip()
                        if len(chunk) > 100:
                            memories.append({
                                "content": chunk,
                                "created_at": timestamp,
                                "memory_type": "observation",
                                "tags": ["thinking", session_id[:8]],
                            })

        # Combine text blocks into a single memory if substantial
        combined_text = "\n".join(text_parts)
        if len(combined_text) > 50:
            memories.append({
                "content": combined_text,
                "created_at": timestamp,
                "memory_type": "observation",
                "tags": ["assistant-response", session_id[:8]],
            })

    return memories


def load_memories(memories: list[dict], api_url: str, api_key: str, user_id: str, dry_run: bool = False):
    """Load extracted memories into shodh-memory via API."""
    loaded = 0
    failed = 0

    for i, mem in enumerate(memories):
        payload = {
            "user_id": user_id,
            "content": mem["content"],
            "tags": mem.get("tags", []),
        }
        if mem.get("created_at"):
            payload["created_at"] = mem["created_at"]
        if mem.get("memory_type"):
            payload["memory_type"] = mem["memory_type"]

        if dry_run:
            preview = mem["content"][:80].replace("\n", " ")
            print(f"  [{i+1}/{len(memories)}] {preview}...")
            loaded += 1
            continue

        try:
            req = Request(
                f"{api_url}/api/remember",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": api_key,
                },
                method="POST",
            )
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                loaded += 1
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(memories)}] loaded ({result.get('id', '?')[:8]}...)")
        except HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"  [{i+1}/{len(memories)}] FAILED {e.code}: {body[:100]}", file=sys.stderr)
            failed += 1
        except Exception as e:
            print(f"  [{i+1}/{len(memories)}] FAILED: {e}", file=sys.stderr)
            failed += 1

        # Small delay to avoid overwhelming the server on bulk load
        if not dry_run and (i + 1) % 5 == 0:
            time.sleep(0.1)

    return loaded, failed


def main():
    parser = argparse.ArgumentParser(description="Load Claude sessions into shodh-memory")
    parser.add_argument("sessions", nargs="+", help="Session JSON files or directories")
    parser.add_argument("--api-url", default="http://localhost:3033", help="Shodh API URL")
    parser.add_argument("--api-key", default=None, help="API key (or set SHODH_API_KEY)")
    parser.add_argument("--user-id", default="autonomites-pipeline", help="User ID for memories")
    parser.add_argument("--dry-run", action="store_true", help="Preview without loading")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("SHODH_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: --api-key or SHODH_API_KEY required", file=sys.stderr)
        sys.exit(1)

    # Collect session files
    session_files = []
    for path in args.sessions:
        p = Path(path)
        if p.is_dir():
            session_files.extend(sorted(p.glob("session-*.json")))
            session_files.extend(sorted(p.glob("*.jsonl")))
        elif p.exists():
            session_files.append(p)
        else:
            print(f"Warning: {path} not found, skipping", file=sys.stderr)

    if not session_files:
        print("No session files found", file=sys.stderr)
        sys.exit(1)

    total_loaded = 0
    total_failed = 0

    for session_file in session_files:
        print(f"\nProcessing: {session_file.name}")
        try:
            memories = extract_memories_from_session(str(session_file))
            print(f"  Extracted {len(memories)} memory units")

            if memories:
                loaded, failed = load_memories(
                    memories, args.api_url, api_key or "", args.user_id, args.dry_run
                )
                total_loaded += loaded
                total_failed += failed
        except json.JSONDecodeError as e:
            print(f"  SKIP: invalid JSON ({e})", file=sys.stderr)
        except Exception as e:
            print(f"  SKIP: {e}", file=sys.stderr)

    mode = "previewed" if args.dry_run else "loaded"
    print(f"\nDone: {total_loaded} {mode}, {total_failed} failed across {len(session_files)} files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test dry-run against a real session file**

Find a session file and run:
```bash
# Find session files
ls /Volumes/chonk/projects/autonomites/session-*.json 2>/dev/null | head -3

# Dry run
python3 scripts/load_sessions.py /Volumes/chonk/projects/autonomites/session-*.json --dry-run --user-id test-loader | head -30
```
Expected: extracted memory count, preview of each memory unit

- [ ] **Step 3: Commit**

```bash
git add scripts/load_sessions.py
git commit -m "feat: add session JSON loader for bulk memory ingestion

Parses Claude session JSON files, extracts assistant reasoning and
thinking blocks as memory units, loads via /api/remember with original
timestamps. Supports dry-run mode and directory scanning."
```

---

### Task 4: Clear Graph, Load Data, Evaluate

This is the validation phase. No code changes — we're loading real data and testing retrieval quality.

- [ ] **Step 1: Clear the existing noise graph**

```bash
curl -s -X DELETE http://localhost:3033/api/graph/autonomites-pipeline/clear \
  -H "X-API-Key: $SHODH_API_KEY" | python3 -m json.tool
```

- [ ] **Step 2: Rebuild shodh-memory with NER changes and restart**

```bash
cd /Volumes/chonk/projects/autonomites/shodh-memory
cargo build --release 2>&1 | tail -5
# User restarts the server with new binary
```

- [ ] **Step 3: Load session data**

```bash
source /Volumes/chonk/projects/autonomites/.env
python3 scripts/load_sessions.py \
  /Volumes/chonk/projects/autonomites/session-*.json \
  --api-url http://localhost:3033 \
  --api-key "$SHODH_API_KEY" \
  --user-id autonomites-pipeline
```

- [ ] **Step 4: Check stats after loading**

```bash
curl -s http://localhost:3033/api/users/autonomites-pipeline/stats \
  -H "X-API-Key: $SHODH_API_KEY" | python3 -m json.tool
```
Expected: increased `total_memories`, `vector_index_count` matching, graph edges should be minimal (only from YAKE keywords in fallback path, not NER)

- [ ] **Step 5: Test compile_context with realistic queries**

```bash
# Test: chain-tracer context
curl -s -X POST http://localhost:3033/api/compile_context \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d '{
    "user_id": "autonomites-pipeline",
    "autonomite": "chain-tracer",
    "task_description": "Trace redirect chains for Thai university domains being used as gambling redirect infrastructure",
    "limit": 5,
    "max_chars": 3000
  }' | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Memories: {d[\"memory_count\"]}, Latency: {d[\"latency_ms\"]:.1f}ms\n\n{d[\"context\"][:500]}...')"

# Test: dork-builder context
curl -s -X POST http://localhost:3033/api/compile_context \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d '{
    "user_id": "autonomites-pipeline",
    "autonomite": "dork-builder",
    "task_description": "Build Google dork queries to discover compromised Indonesian education domains",
    "limit": 5,
    "max_chars": 3000
  }' | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Memories: {d[\"memory_count\"]}, Latency: {d[\"latency_ms\"]:.1f}ms\n\n{d[\"context\"][:500]}...')"
```

Evaluate: are the surfaced memories actually relevant to each autonomite's task? Does cross-agent context surface (chain-tracer findings appearing for dork-builder)?

- [ ] **Step 6: Run consolidation to build co-retrieval edges**

```bash
# The compile_context calls above recorded co-activations.
# Run consolidation to strengthen edges and replay high-value memories.
curl -s -X POST http://localhost:3033/api/consolidate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SHODH_API_KEY" \
  -d '{"user_id": "autonomites-pipeline"}' | python3 -m json.tool
```

- [ ] **Step 7: Check graph state after consolidation**

```bash
curl -s http://localhost:3033/api/graph/autonomites-pipeline/stats \
  -H "X-API-Key: $SHODH_API_KEY" | python3 -m json.tool
```
Expected: edges should now be co-retrieval edges (from memories surfaced together), not NER-extracted entity relationships

- [ ] **Step 8: Re-test compile_context after consolidation**

Run the same queries from Step 5 again. Compare: has the quality of surfaced memories improved now that co-retrieval edges have been strengthened?

---

### Task 5: Tune and Iterate

Based on Task 4 results, adjustments may include:

- [ ] **Tuning hybrid search weights** — if BM25 dominates, reduce `BM25_WEIGHT` in `src/memory/hybrid_search.rs`. If vector similarity dominates, adjust accordingly.

- [ ] **Adjusting YAKE keyword extraction** — if too many noise keywords, tune `MAX_KEYWORDS` or score threshold in `src/embeddings/keywords.rs`.

- [ ] **Adjusting compile_context formatting** — if the prose output is too raw, add section headers, relevance ordering, or per-memory source attribution.

- [ ] **Loading more data** — if initial results are promising, load git commit messages, run artifacts, or additional session files.

These are iterative — decisions depend on what Task 4 reveals.
