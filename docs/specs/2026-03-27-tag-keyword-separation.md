# Tag/Keyword Separation Spec

## Problem

User-provided tags and YAKE-extracted keywords are merged into one flat `Vec<String>` during remember (remember.rs:327, :871). No provenance is preserved. This causes:

- **False positive tag queries**: `recall_by_tags(["bulletin"])` returns memories where "bulletin" was a YAKE keyword extracted from text *about* bulletins, not memories the user explicitly tagged as bulletins.
- **No confidence distinction**: An explicit tag like `"dork-builder"` carries the same weight as a YAKE keyword like `"discovered"`. One is a deliberate classification; the other is a statistical extraction.
- **No independent decay**: YAKE keywords should be more volatile than explicit tags. A user who tags something `"critical"` means it. YAKE extracting `"critical"` from text is a guess.

### Current Data Flow

```
User tags: ["dork-builder", "bulletin"]
YAKE keywords: ["bulletin", "dns takeover", "go.id", "judicial", "slot-toto"]

Merged (deduped): ["dork-builder", "bulletin", "dns takeover", "go.id", "judicial", "slot-toto"]

Stored as:
  experience.tags = merged      # no provenance
  experience.entities = merged  # identical copy
  RocksDB index: tag:{each}:{uuid}
  BM25 index: tags field = space-joined string
```

`recall_by_tags(["bulletin"])` matches both the user's explicit tag AND the YAKE extraction. No way to distinguish.

## Proposed Design

### Schema Change

Split `Experience` into three fields:

```rust
pub struct Experience {
    pub content: String,
    pub tags: Vec<String>,           // User-provided only. Never auto-populated.
    pub keywords: Vec<String>,       // YAKE-extracted. Auto-populated during encoding.
    pub ner_entities: Vec<NerEntityRecord>,  // Structured NER (currently unused, keep for future)
    // ... rest unchanged
}
```

Remove the `entities` field (currently an alias for `tags`). Anything consuming `entities` switches to `tags` + `keywords` as appropriate.

### Storage Changes

**RocksDB indices** (storage.rs:1338-1380):

```
tag:{normalized}:{uuid}      # Explicit tags only
keyword:{normalized}:{uuid}  # YAKE keywords only
entity:{normalized}:{uuid}   # Can be removed or aliased to keyword
```

**BM25 index** (hybrid_search.rs):

Keep the existing `tags_field` and `entities_field` in Tantivy, but populate them separately:
- `tags_field` = user-provided tags (space-joined)
- `entities_field` = YAKE keywords (space-joined)

BM25 full-text search still queries both fields, so keyword matching isn't degraded. The separation only matters for structured tag queries.

### Remember Changes (remember.rs)

Replace the merge logic at lines 327 and 871:

```rust
// Before (flat merge):
let mut merged_entities: Vec<String> = req.tags.clone();
// ... add NER + YAKE into same vec

// After (separated):
let tags: Vec<String> = req.tags.clone();  // User-provided, untouched
let keywords: Vec<String> = extracted_keywords;  // YAKE only, capped to MAX_KEYWORDS

let experience = Experience {
    tags,
    keywords,
    // ...
};
```

No merging. No dedup across the two sets. If the user tags something `"infrastructure"` and YAKE also extracts `"infrastructure"`, both lists contain it — that's fine. The duplication reinforces rather than conflates.

### Retrieval Changes

**`recall_by_tags`** (recall.rs:2356): Searches only `tag:` prefix in RocksDB. YAKE keywords are invisible to tag queries. This is the critical fix — explicit tags are a reliable signal for structured queries like bulletin fetching.

**`recall` (hybrid search)**: No change needed. BM25 still searches both `tags_field` and `entities_field`. Vector search is content-based. Graph spreading activation can traverse both tag and keyword nodes.

**`compile_context`**: No change needed. Uses hybrid search which already queries across all signals.

### Migration

Existing memories have merged tags/keywords with no provenance. Two options:

**Option A: Re-encode on read (lazy).** When a memory is loaded from RocksDB, treat the existing `entities` field as `keywords` (since most values are YAKE-extracted). Only memories created after the change will have clean `tags`. Old memories gradually get re-encoded when updated.

**Option B: Batch re-index.** Run YAKE on all existing content, diff against stored entities to infer which were user-provided vs extracted. Heuristic: if a tag matches a YAKE keyword for that content, classify as keyword; otherwise classify as tag. Not perfect but better than nothing.

**Recommendation: Option A.** The old data is already noisy. New data will be clean. The bulletin false-positive problem is already mitigated by the `fetch_bulletins` filter. A full re-index isn't worth the complexity.

### API Contract

The `/api/remember` request body is unchanged — `tags` field still accepted. The response and `/api/recall` responses gain a `keywords` field:

```json
{
  "experience": {
    "content": "...",
    "tags": ["dork-builder", "bulletin"],
    "keywords": ["dns takeover", "go.id", "judicial", "slot-toto"]
  }
}
```

MCP tools (`remember`, `recall`, `recall_by_tags`) work unchanged. The MCP server can expose `keywords` in responses but doesn't need to accept them as input — they're always auto-extracted.

## Scope

### In scope
- Split `entities`/`tags` into `tags` + `keywords` in Experience struct
- Separate RocksDB index prefixes
- `recall_by_tags` only searches explicit tags
- BM25 still indexes both
- Lazy migration for existing data

### Out of scope (future)
- Keyword confidence scores (YAKE provides scores but they're discarded at extraction)
- Independent decay rates for keywords vs tags
- Keyword-specific boosting in hybrid search weighting
- Tag suggestion/auto-complete based on keyword frequency

## Files Affected

| File | Change |
|------|--------|
| `src/memory/types.rs` | Split Experience.entities → tags + keywords |
| `src/handlers/remember.rs` | Stop merging, populate separately |
| `src/memory/storage.rs` | New `keyword:` index prefix, update index CRUD |
| `src/memory/hybrid_search.rs` | Populate BM25 fields separately |
| `src/handlers/recall.rs` | recall_by_tags uses only `tag:` prefix |
| `src/handlers/types.rs` | Update response serialization |
| `src/python.rs` | Update Python bindings if used |
| `mcp-server/index.ts` | Expose keywords in responses |
