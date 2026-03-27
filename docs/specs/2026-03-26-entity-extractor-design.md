# Entity Extractor: Maximal Munch + NLP Replacement for NER

## Problem

NER was dismantled because it classified 100% of ISAP domain data as `Other:MISC`, producing a noise graph full of word-couple entities. The knowledge graph needs meaningful entities to function as an index — the entity-centric topology is architecturally sound, but requires quality extraction.

## Solution

Replace NER with a two-layer extraction pipeline: a maximal munch classifier for domain-specific identifiers, and a Rust-native NLP layer (POS tagging + dependency parsing) for general noun/verb extraction. The NLP layer provides compound noun phrases as entities and verbs as typed relationship labels, replacing the current `RelatedTo` edges.

## Architecture

### Extraction Pipeline

Two layers, running in sequence on each memory's content text:

```
Input text
  │
  ├─ Layer 1: Maximal Munch (domain-specific, high confidence)
  │   ├─ Extract candidates via structural heuristics
  │   ├─ Sort by length descending
  │   ├─ Classify each (domain, URL, operator_id, cluster_name, IP, etc.)
  │   └─ Mark consumed character spans
  │
  ├─ Layer 2: NLP Parse (general, fills gaps)
  │   ├─ POS tagging + dependency parsing
  │   ├─ Extract noun phrases (compound nouns grouped by dep tree)
  │   ├─ Filter: skip spans consumed by Layer 1
  │   ├─ Filter: proper nouns (NNP) = high value, common nouns = low value
  │   ├─ Filter: subject/object role > nested prepositional nouns
  │   └─ Extract verbs for relationship typing
  │
  └─ Merge
      ├─ Layer 1 wins on overlap
      ├─ Layer 2 fills remaining noun phrases
      ├─ Verb triples: subject --verb--> object
      └─ Output: Vec<ExtractedEntity> + Vec<ExtractedTriple>
```

### Layer 1: Maximal Munch Classifier

**Candidate extraction** via structural heuristics (not regex patterns competing — heuristics identify candidate spans, classification happens after):

| Heuristic | What it catches | Entity type |
|-----------|----------------|-------------|
| Contains 2+ dots, no spaces | Domain names | `Domain` |
| Starts with `http://` or `https://` | URLs | `Url` |
| ALL-CAPS with hyphens, 6+ chars | Operator IDs | `OperatorId` |
| Matches known prefix (CLUSTER-, CVE-) | Cluster names, CVEs | `ClusterName`, `Cve` |
| IP-shaped (`\d+\.\d+\.\d+\.\d+`) | IP addresses | `IpAddress` |
| Config-provided patterns | Autonomite-specific | From config |

**Algorithm:**
1. Run all heuristics, collect candidate spans with classification
2. Sort by span length descending
3. Iterate: for each candidate, if its span doesn't overlap any already-consumed span, accept it and mark the span consumed
4. Result: non-overlapping set of domain-specific entities, longest match wins

**Config-driven patterns:** Autonomites can declare additional extraction patterns. These are passed to shodh via the `extraction_config` field on the remember API (already exists in the batch endpoint), or loaded from a config file at server startup:

```yaml
extraction:
  patterns:
    - type: operator_id
      match: "[A-Z]{2,}-[A-Z]+"
    - type: cluster_name
      match: "CLUSTER-[A-Z0-9-]+"
```

Default patterns (domains, URLs, IPs) always run. Config patterns layer on top.

### Layer 2: NLP (POS + Dependency Parse)

**Rust-native.** Use a Rust POS tagger and dependency parser (evaluate `rust-bert`, `nlprule`, or a lighter alternative at implementation time).

**Noun phrase extraction:**
- Dependency parse groups compound nouns: "gambling SEO injection" → single entity, not three
- Proper nouns (NNP/NNPS) → high confidence entities
- Noun phrases in subject/object position → medium confidence
- Common nouns, lowercase, single word → filtered out (noise: "thing", "way", "lot")
- Skip any noun phrase overlapping a Layer 1 consumed span

**Verb extraction for relationship typing:**
- Subject-verb-object triples from dependency parse
- Verb becomes the relationship type: "blogs compromised with injection" → `blogs --Compromised--> injection`
- Falls back to `RelatedTo` only when no verb triple is available

**Noun value filtering:**

| Signal | Confidence | Action |
|--------|-----------|--------|
| Proper noun (NNP) | High (0.8) | Always include |
| Multi-word noun phrase | High (0.7) | Always include |
| Subject/object of main verb | Medium (0.6) | Include |
| Common noun, capitalized mid-sentence | Medium (0.5) | Include |
| Common noun, lowercase, single word | Low | Filter out |

### Output Format

The extractor produces two output types that feed into graph construction:

**Entities:**
```rust
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,       // "Domain", "OperatorId", "NounPhrase", "ProperNoun", etc.
    pub confidence: f32,           // 0.0-1.0
    pub start_char: usize,
    pub end_char: usize,
    pub source: ExtractionSource,  // MaximalMunch or NlpParse
}
```

**Triples (from verb extraction):**
```rust
pub struct ExtractedTriple {
    pub subject: String,
    pub verb: String,              // becomes RelationType
    pub object: String,
    pub confidence: f32,
}
```

Both map to the existing graph insertion pipeline. `ExtractedEntity` maps to `NerEntityRecord` (or replaces it). `ExtractedTriple` provides typed edges instead of generic `RelatedTo`.

## Integration with `process_experience_into_graph()`

### What changes

**Phase 1 (extraction, no lock) in `state.rs:1896-2184`:**

Current entity source priority (NER records → tags → all-caps → issue IDs → verbs) is replaced by:

1. Run maximal munch on content → `Vec<ExtractedEntity>`
2. Run NLP parse on content → `Vec<ExtractedEntity>` + `Vec<ExtractedTriple>`
3. Merge (Layer 1 wins on overlap)
4. Apply salience scoring and truncate to `max_entities_per_memory`

The existing multi-source enrichment logic (tags as entities, all-caps detection, issue ID matching, verb extraction) is subsumed by the new pipeline — tags become a maximal munch config pattern, all-caps is a heuristic, issue IDs are a config pattern, verbs come from NLP.

**Phase 2 (graph insertion, with lock) in `state.rs:2186-2270`:**

Mostly unchanged. The additions:

- `ExtractedTriple` creates typed edges: instead of all co-occurrence pairs being `RelatedTo`, triples create edges with the verb as `RelationType`. Add a `RelationType::Dynamic(String)` variant to hold verb-derived relationship types (e.g., `Dynamic("compromised")`, `Dynamic("redirects_to")`). Existing named variants (`WorksWith`, `PartOf`, etc.) stay for structured relationships.
- Entity dedup pipeline (exact → case-insensitive → stemmed → embedding merge) stays as-is.

### What stays the same

- `NerEntityRecord` format (or its replacement `ExtractedEntity`) flows through `Experience.ner_entities`
- Episodic node creation
- Co-occurrence edge creation for entities without verb triples
- Salience scoring and `max_entities_per_memory` cap
- 4-tier entity dedup in `add_entity()`

## Module Structure

```
src/
  extraction/
    mod.rs              — public API: extract(text, config) -> ExtractionResult
    maximal_munch.rs    — Layer 1: heuristic candidate extraction + longest-match resolution
    nlp.rs              — Layer 2: POS tagging, dependency parsing, noun/verb extraction
    types.rs            — ExtractedEntity, ExtractedTriple, ExtractionConfig, ExtractionResult
    config.rs           — Pattern loading from autonomite config
```

New module, not modifications to existing files (except the integration point in `state.rs`).

## Aho-Corasick Dictionary

The maximal munch layer maintains a dictionary of previously-seen entities for fast multi-pattern matching:

- **Built from:** All existing entity nodes in the graph (loaded at startup)
- **Updated when:** New entities are discovered by heuristics and confirmed by graph insertion
- **Used for:** O(n) text scanning via Aho-Corasick automaton — finds all known entities in a single pass
- **Complements heuristics:** Known entities are found by dictionary, new entities are found by heuristics, both feed into longest-match resolution

The `aho-corasick` crate is the standard Rust implementation (by the ripgrep author, highly optimized).

## Rust NLP Library Evaluation

Evaluate at implementation time. Candidates:

| Library | POS | Dep Parse | Size | Notes |
|---------|-----|-----------|------|-------|
| `nlprule` | Yes | No | ~50MB model | LanguageTool-based, rules not ML |
| `rust-bert` | Yes | Yes | ~500MB model | Full transformer, heavy |
| `lingua` | No | No | Small | Language detection only |
| Custom rule-based | Basic | No | Zero | Heuristic POS from capitalization + position |

If no Rust library provides adequate dependency parsing, a fallback option is a lightweight rule-based approach: split on verbs, extract noun phrases on either side, infer subject-verb-object from word order. Less accurate than a full parser but zero dependencies.

The implementation plan should include an evaluation step before committing to a library.

## What This Enables

- **Meaningful graph structure** from actual domain artifacts, not NER noise
- **Typed relationships** from verbs instead of generic `RelatedTo`
- **Compound noun entities** grouped by dependency parse, not split into word fragments
- **Config-driven extraction** per autonomite for domain-specific patterns
- **Growing dictionary** via Aho-Corasick for fast recognition of known entities
- **Foundation for knowledge compiler** M2: graph topology that reflects actual knowledge structure
