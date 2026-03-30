# Entity Extractor: Maximal Munch + NLP Replacement for NER

## Context

Shodh-memory is a persistent cognitive memory system for autonomous agents ("autonomites"). It stores experiences as episodic memories with vector embeddings (HNSW) and organizes them into a knowledge graph with Hebbian learning — entities are nodes, co-occurrence and typed relationships are edges, and edge weights strengthen with repeated activation.

The system is written in Rust (RocksDB storage, no external DB), exposes an HTTP API, and integrates with Claude/Cursor via an MCP server. Autonomites are domain-specific agents that run in pipelines — each has its own identity, scripts, and workspace. The first production deployment is ISAP, a project that hunts malicious SEO redirect chains across compromised web infrastructure.

The knowledge graph serves as a shared index across autonomite runs: entities discovered by one autonomite (domains, IPs, operator fingerprints) become retrievable context for others. Graph topology — which entities cluster together, which relationships recur — is the foundation for a planned knowledge compiler that generates LoRA adapters from structural patterns.

## Problem

NER was dismantled because it classified 100% of ISAP domain data as `Other:MISC`, producing a noise graph full of word-couple entities. The knowledge graph needs meaningful entities to function as an index — the entity-centric topology is architecturally sound, but requires quality extraction.

## Solution

Replace NER with a three-source extraction pipeline: a maximal munch classifier for domain-specific identifiers, a Rust-native NLP layer (POS tagging + NP chunking) for general noun/verb extraction, and a metadata pass-through for structured inputs (tags, issue IDs) that bypasses text parsing entirely. The NLP layer provides compound noun phrases as entities and lemmatized verb triples as evidence for relationship typing.

## Architecture

### Extraction Pipeline

Three sources, running in sequence on each memory (Source 1 completes before Source 2 begins; Source 2 receives Source 1's entity list as input for SVO slot-filling):

```
Input: content text + structured metadata (tags, issue IDs)
  │
  ├─ Source 0: Metadata Pass-Through (structured, highest confidence)
  │   ├─ Tags → entities with type from tag namespace (or "Tag")
  │   ├─ Issue IDs → entities with type "IssueId"
  │   ├─ No text parsing — these come from API fields, not content
  │   └─ Confidence: 1.0 (user-provided structured data)
  │
  ├─ Source 1: Maximal Munch (domain-specific, high confidence)
  │   ├─ Extract candidates via structural heuristics
  │   ├─ Sort by length descending
  │   ├─ Classify each (domain, URL, operator_id, cluster_name, IP, etc.)
  │   ├─ Containment extraction (e.g., URL emits derived Domain/IP)
  │   └─ Mark consumed character spans
  │
  ├─ Source 2: NLP Parse (general, fills gaps)
  │   ├─ POS tagging (Rust-native, no dependency parsing)
  │   ├─ NP chunking via regex over POS tags
  │   ├─ NP filter: skip NP spans consumed by Source 1 (don't re-extract)
  │   ├─ SVO uses Source 1 entities as pre-identified NPs for subject/object slots
  │   ├─ Filter: proper nouns (NNP) = high value
  │   ├─ Filter: two-tier stopword strategy
  │   ├─ Extract verbs, lemmatize, detect passive voice
  │   └─ SVO extraction from word order (verb-adjacent NPs, including Source 1 entities)
  │
  └─ Merge
      ├─ Source 0 always included (metadata entities never suppressed by text extraction)
      ├─ Source 1 containment rules govern overlap (see below)
      ├─ Source 2 fills remaining noun phrases
      ├─ Verb triples stored as evidence (see Verb Triple Policy)
      └─ Output: Vec<ExtractedEntity> + Vec<ExtractedTriple>
```

### Layer 1: Maximal Munch Classifier

**Candidate extraction** via structural heuristics (not regex patterns competing — heuristics identify candidate spans, classification happens after):

| Heuristic | What it catches | Entity type |
|-----------|----------------|-------------|
| Contains 2+ dots, no spaces, valid TLD (IANA list or known infra TLDs) | Domain names | `Domain` |
| Starts with `http://` or `https://` | URLs | `Url` |
| ALL-CAPS with hyphens, 6+ chars | Operator IDs | `OperatorId` |
| Matches known prefix (CLUSTER-, CVE-) | Cluster names, CVEs | `ClusterName`, `Cve` |
| IPv4: 4 dot-separated octets, each 0-255 | IP addresses | `IpAddress` |
| Config-provided patterns | Autonomite-specific | From config |

**Algorithm:**
1. Run all heuristics, collect candidate spans with classification
2. Run AC dictionary scan, collect matched spans
3. **Unify candidate pool:** both heuristic and AC candidates enter the same set before resolution. This prevents AC from consuming a span before a heuristic's longer match is considered (e.g., AC finds "example.com" but heuristic finds "malicious-example.com" — the longer match must win).
4. Sort unified pool by span length descending. **Tie-breaking** for equal-length overlapping candidates: heuristic matches win over AC dictionary matches (heuristics have structural classification; AC matches are recognition only). Among same-source ties, the candidate appearing earlier in the text wins.
5. Iterate: for each candidate, if its span doesn't overlap any already-consumed span, accept it and mark the span consumed
6. **Containment extraction:** for accepted entities that contain sub-entities, emit derived children (see containment rules)
7. Result: set of domain-specific entities with containment-derived children

**Containment rules by type:**

"Layer 1 wins on overlap" from the previous revision was too blunt — a URL span would suppress the domain inside it, and the domain is often the more useful graph node. Instead, specific types emit derived children:

| Parent type | Derived child | Rule |
|-------------|--------------|------|
| `Url` | `Domain` | Extract host from URL, emit as separate `Domain` entity |
| `Url` | `IpAddress` | If host is an IP, emit as `IpAddress` instead of `Domain` |
| `Url` | `Path` | Extract normalized path (strip query params by default, configurable). **Root paths (`/`) are suppressed** — they carry no semantic signal and would create a hub node connecting every root-URL domain. Only non-trivial paths (>= 2 path segments or a meaningful filename) emit `Path` entities. |

`Url` is an **intermediate extraction entity** — it exists during the extraction pipeline to trigger containment rules and produce derived children (`Domain`, `IpAddress`, `Path`), but the `Url` itself is NOT persisted as a graph node. The raw URL string is stored as an attribute on the episodic node for provenance. Only the derived children become graph entities. This avoids URL fragmentation (query param variants creating duplicate nodes) while preserving the containment extraction mechanism.
| `Domain` | `DomainLabel` | Split hostname on dots/hyphens, check if labels are dictionary words. Emit meaningful labels as low-confidence entities (e.g., `malicious-payments.example.com` → `DomainLabel("malicious-payments")`). This recovers semantic signals from hostnames without Source 2 re-entering consumed spans. |

Other overlaps collapse normally (longest match wins). The parent and derived children all enter the entity set — containment is additive, not exclusive.

**Consumed span scope:** Source 2 skips the **parent span** (the full extent matched by the heuristic or AC), not just the derived children's spans. A URL heuristic matching positions 0-50 means Source 2 will not attempt NP chunking anywhere in positions 0-50, even though the derived Domain might only occupy positions 7-25. This is intentional: the parent span is structural text (URL syntax), not natural language, so POS tagging it would produce garbage. Source 1 entities from within consumed spans ARE available to Source 2 as pre-identified NPs for SVO extraction (see verb extraction section).

**Why Source 2 does NOT re-enter consumed spans:** POS tagging a domain name or URL produces garbage — these are structural identifiers, not natural language. Semantic signals inside them (hostname labels, path components) are better recovered through Source 1 containment rules, which understand the structure.

**Config-driven patterns:** Autonomites can declare additional extraction patterns. These are passed to shodh via the `extraction_config` field on the remember API (already exists in the batch endpoint), or loaded from a config file at server startup:

```yaml
extraction:
  patterns:
    - type: operator_id
      match: "[A-Z]{2,}-[A-Z]+"
    - type: cluster_name
      match: "CLUSTER-[A-Z0-9-]+"
```

Default patterns (domains, URLs, IPs) always run. Config patterns layer on top. Custom config patterns do NOT support containment rules — they produce flat entities only. If a custom pattern's span overlaps a built-in pattern's span, standard longest-match resolution applies. Containment extraction is limited to the built-in types (`Url` → `Domain`/`IpAddress`/`Path`, `Domain` → `DomainLabel`) where the decomposition semantics are well-defined.

**Config versioning:** The effective extraction config (defaults + autonomite patterns + stopword list) AND the AC dictionary state (rebuild sequence number + entity count) are hashed together to produce a version stamp. This stamp is persisted on each ingested memory so that reprocessing is explainable — the same memory under different configs or different AC states may yield different entities. The version stamp is stored on the episodic node as `extraction_config_version: String`. Including the AC state is essential: the config alone does not determine Source 1 output because the AC dictionary is a continuously mutating input.

### Layer 2: NLP (POS Tagging + NP Chunking)

**Rust-native, no ML frameworks.** Use a Rust POS tagger (`nlprule` or an Averaged Perceptron tagger). Dependency parsing is explicitly out of scope — compound nouns are extracted via NP chunking, not parse trees.

**Noun phrase extraction via POS-tag chunking:**
- POS-tag the text, then apply a chunking grammar over the tag sequence
- Proper nouns (NNP/NNPS) → high confidence entities
- Skip any noun phrase overlapping a Source 1 consumed span

**Chunking grammar:**

The initial grammar `(JJ|NN)* NN` is too narrow for security/infrastructure prose. Expanded grammar to handle common shapes:

```
NP → (DT)? (JJ|VBN|VBG|NN|NNP|CD)* (NN|NNS|NNP|NNPS)
     # DT    = optional determiner (stripped from entity text)
     # JJ    = adjective: "malicious payload"
     # VBN   = past participle as modifier: "compromised server"
     # VBG   = gerund as modifier: "phishing campaign"
     # NN/NNP = compound nouns: "gambling SEO injection", "Windows Server"
     # CD    = cardinal number: "Windows Server 2022"
     # Hyphenated compounds: pre-tokenize "command-and-control" as single token
     # Normalize Unicode hyphens (U+2011, U+2013, U+2014) to ASCII hyphen before tokenization
```

Examples this captures that `(JJ|NN)* NN` would miss:
- "proof of concept exploit" → PP attachment via explicit linking rule: when two NP chunks are separated only by `of`, `for`, or `in`, merge into a single entity (e.g., `NP("proof") + "of" + NP("concept exploit")` → `"proof of concept exploit"`). This is a post-chunking merge pass, not a grammar extension.
- "command-and-control server" → pre-tokenize hyphenated compounds
- "Windows Server 2022 host" → NNP + NNP + CD + NN
- "compromised web server" → VBN + NN + NN

The grammar must be **benchmarked on ISAP-like text** before finalizing (see Testing section). Passing threshold: >= 0.7 recall on the gold dataset's annotated noun phrases (i.e., the grammar captures at least 70% of expected NPs). Precision floor: >= 0.5 (at least half of extracted NPs are in the gold set). These thresholds are initial — adjust based on how truncation and downstream graph quality are affected.

**Verb extraction — evidence-first, not schema-first:**

SVO extraction from heuristic word order is inherently low-precision. "Nearest preceding NP / nearest following NP" will misread coordination, subordinate clauses, sentence fragments, and dense security prose. Verb triples are therefore stored as **evidence**, not promoted directly to graph relation types.

- SVO extraction from word order: identify verb, take nearest preceding NP as subject and nearest following NP as object. **Source 1 entities are injected as pre-identified NPs** into the SVO search — consumed spans are skipped for NP chunking (don't re-extract what Source 1 found) but Source 1 entities ARE available to fill subject/object slots in verb triples. Without this, infrastructure identifiers (domains, IPs) could never participate in typed relationships like `malware --Compromised--> example.com`.
- **Lemmatize** verbs before storing: "compromised", "compromising", "compromises" → `compromise`
- **Passive voice detection:** if auxiliary + past participle, invert subject/object assignment. Patterns beyond simple "was + VBN": present continuous passive ("is being compromised"), modal passive ("can be compromised"), perfect passive ("has been compromised"). Match the full auxiliary chain: `(is|are|was|were|be|been|being|has been|have been|had been|can be|could be|should be|must be|will be) + VBN`.
- **Agentless passive:** When the logical subject is omitted ("The server was compromised."), produce a triple with `subject: None`. These triples are stored as evidence but are never promoted to typed edges — a directed edge requires both a source and target node. They remain useful for entity extraction (the object "server" still participates) and for future analysis.

**Triple confidence scoring:**

Triple confidence is NOT flat — it reflects structural quality of the extraction. Even though triples are stored as evidence, confidence determines promotion eligibility and future analysis value.

| Signal | Adjustment |
|--------|-----------|
| Base | 0.4 |
| Active voice (no auxiliary chain) | +0.3 |
| Both subject and object are high-confidence entities (>= 0.7) | +0.2 |
| Coordination detected (multiple NPs or verbs in clause) | -0.3 |
| Object attached via preposition (via, through, with, from) | -0.4 |
| Both direct and prepositional objects present | -0.2 (use direct object, discard prepositional) |
| Subject-object token distance > 8 | -0.2 |

Only triples with confidence >= 0.5 AND verb in canonical dictionary are eligible for promotion. This prevents structurally dubious extractions from creating typed edges even when the verb happens to match.

Specific failure modes this catches:
- "actor scanned ports and exploited the server" → coordination penalty prevents `exploit(actor, ports)`
- "malware deployed via server" → prepositional penalty prevents `deploy(malware, server)` with wrong directionality
- Long-range dependencies across clauses → distance penalty

**Verb triple promotion policy:**

Only verb triples that match the **canonical verb dictionary** are promoted to typed edges. Unmapped lemmas are stored as `ExtractedTriple` evidence on the episodic node but create `RelatedTo` co-occurrence edges, not `Dynamic(lemma)` edges.

```
Canonical verb dictionary (initial set):
  ["infect", "compromise", "breach", "exploit"] → RelationType::Compromised
  ["redirect", "forward", "proxy"]              → RelationType::RedirectsTo
  ["host", "serve", "run"]                      → RelationType::Hosts
  ["contain", "include", "embed"]               → RelationType::Contains
  ["block", "filter", "deny"]                   → RelationType::Blocks
```

This dictionary is intentionally small and curated. It grows only when precision is validated against real extractions — never automatically. `RelationType::Dynamic(String)` is removed from the design; unmapped verbs stay as evidence until the dictionary is extended to cover them.

**Edge directionality:** When a promoted triple creates a typed edge, the triple's subject maps to the edge source node and the object maps to the edge target node. The directed edge reads as `subject --RelationType--> object`. For example, `compromise(malware, server)` creates `malware --Compromised--> server`. Passive voice inversion (handled during extraction) ensures that "the server was compromised by malware" produces the same edge direction as "malware compromised the server" — by the time the triple reaches graph insertion, subject/object have already been normalized to active-voice order.

- Falls back to `RelatedTo` when no verb triple is available OR when the verb is not in the canonical dictionary

**Noun filtering — two-tier stopword strategy:**

Single-word lowercase nouns like "server", "pod", "database" are critical infrastructure anchors in ISAP data. But they're also extremely high-frequency — "server" alone would create a hub node connected to everything, which is topologically useless. Filtering by morphological shape (lowercase + single word) is too crude. Instead, use two tiers:

**Tier 1 — Global stopwords (always filtered as standalone entities):**

> thing, way, lot, result, case, kind, type, part, point, fact, issue, problem, question, reason, example, number, time, place, area, end, side, use, set, group, level

**Stopwords inside multi-word NPs:** A Tier 1 word embedded within a multi-word NP does NOT invalidate the NP. The NP as a whole is retained if it passes the multi-word NP filter. "use case" is retained as a multi-word NP even though "use" and "case" are individually in Tier 1. Tier 1 filtering applies only to standalone single-word entities, not to NP components.

**Tier 2 — Domain high-frequency nouns (conditionally filtered):**

> server, system, host, service, network, port, file, data, user, client, request, response, connection, process, node

These are meaningful only when modified or in context. A bare "server" is noise; "compromised server" is signal. Tier 2 nouns are included ONLY when:
- Part of a multi-word NP ("dns server", "web server")
- Modified by adjective or participle ("compromised server", "malicious host")
- Participating in a verb triple as subject or object
- Otherwise, filtered out as standalone entities

Stretch goal: derive both tiers via TF-IDF over the corpus (words with zero variance across documents → Tier 1; words with high frequency but low discriminative power → Tier 2).

| Signal | Boost | Action |
|--------|-------|--------|
| Proper noun (NNP) | +0.4 | Always include |
| Multi-word noun phrase | +0.3 | Always include |
| Subject/object of main verb | +0.2 | Include |
| Common noun, capitalized mid-sentence | +0.1 | Include |
| Common noun, not in stopword list | +0.0 | Include (base confidence only) |
| Common noun, in stopword list | — | Filter out |

**Confidence combination:** An entity's confidence is `base + sum(boosts)`, capped by type ceiling. Base confidence is source-dependent: Source 0 (metadata) = 1.0, Source 1 (maximal munch) = 0.9, Source 2 (NLP) = 0.4. Signals are additive — a proper noun (NNP) that is also the subject of a main verb gets `0.4 + 0.4 + 0.2 = 1.0`, then capped by type.

**Type-aware confidence caps:**

Even with boosts, NLP-extracted noun phrases should not outrank structural identifiers during truncation. ISAP's primary graph topology is built from infrastructure artifacts, not prose.

| Entity type | Max confidence |
|-------------|---------------|
| Domain, IpAddress, Url | 1.0 |
| OperatorId, ClusterName, Cve | 1.0 |
| Metadata (tags, issue IDs) | 1.0 |
| ProperNoun | 0.85 |
| NounPhrase (multi-word) | 0.75 |
| NounPhrase (single-word, Tier 2 qualified) | 0.6 |

This ensures that truncation always preserves infrastructure identifiers over NLP-derived entities when budget is tight.

### Output Format

The extractor produces two output types that feed into graph construction:

**Entities:**
```rust
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,       // "Domain", "OperatorId", "NounPhrase", "ProperNoun", etc.
    pub confidence: f32,           // 0.0-1.0
    pub span: Option<(usize, usize)>,  // (start_char, end_char) — None for metadata entities
    pub source: ExtractionSource,  // Metadata, MaximalMunch, or NlpParse
}
```

**Triples (from verb extraction — stored as evidence):**
```rust
pub struct ExtractedTriple {
    pub subject: Option<String>,   // None for agentless passive ("The server was compromised.")
    pub verb: String,              // lemmatized form
    pub object: String,
    pub confidence: f32,
    pub passive: bool,             // true if passive voice detected (subject/object already inverted)
    pub promoted: bool,            // true if verb matched canonical dictionary → typed edge created
}
```

`ExtractedEntity` maps to `NerEntityRecord` (or replaces it). `ExtractedTriple` is stored as evidence on the episodic node; only triples with `promoted: true` (verb matched canonical dictionary AND confidence >= 0.5) create typed edges. Unpromoted triples with confidence >= 0.3 create `RelatedTo` co-occurrence edges. Triples below 0.3 confidence are stored as evidence only — no edges created. This prevents structurally dubious extractions from polluting graph topology even as `RelatedTo`.

## Integration with `process_experience_into_graph()`

### What changes

**Phase 1 (extraction, no lock) in `state.rs:1896-2184`:**

Current entity source priority (NER records → tags → all-caps → issue IDs → verbs) is replaced by:

1. Metadata pass-through: tags → entities, issue IDs → entities (no text parsing, confidence 1.0)
2. Run maximal munch on content → `Vec<ExtractedEntity>` (with containment-derived children)
3. Run NLP parse on content → `Vec<ExtractedEntity>` + `Vec<ExtractedTriple>`
4. Merge: metadata always included, Source 1 containment rules on overlap, Source 2 fills gaps
5. Apply salience scoring and truncate to `max_entities_per_memory`

**Metadata truncation cap:** Metadata entities are high-confidence but not unlimited. A memory with 50 tags should not starve text-extracted entities. Metadata entities are capped at `max_metadata_entities` (default: 10) before merge. Within that cap, they are exempt from salience-based truncation. Text-extracted entities fill the remaining budget (`max_entities_per_memory - metadata_count`).

When metadata exceeds the cap, select by **namespace priority**: tags with structured namespaces (e.g., `campaign:*`, `cve:*`, `operator:*`) rank above unnamespaced tags (`note`, `todo`). Issue IDs are ranked above all tags (they are structured identifiers, not user-applied labels). Priority ordering within tags is configurable per autonomite. Default ranking: issue IDs first, then namespaced tags (alphabetical), then unnamespaced tags (alphabetical).

**Tags and issue IDs stay as structured metadata sources.** They do not go through maximal munch or NLP — they are higher-confidence than text parsing and often do not appear verbatim in content. Routing them through text extraction would be a regression.

**Cross-source dedup:** The same identifier can appear in multiple sources — a CVE might exist as a `cve:CVE-2021-44228` tag (Source 0), a `CVE-2021-44228` pattern match (Source 1), and an issue ID (Source 0). During merge, entities are deduplicated by normalized text (case-insensitive, stripped of namespace prefixes). The highest-confidence instance wins. This is handled by the existing 4-tier entity dedup in `add_entity()`, which already covers exact and case-insensitive matching.

**Phase 2 (graph insertion, with lock) in `state.rs:2186-2270`:**

Mostly unchanged. The additions:

- `ExtractedTriple` handling: triples whose verb matches the canonical dictionary create typed edges using the mapped `RelationType` variant. All other triples create `RelatedTo` co-occurrence edges and store the raw triple as evidence on the episodic node (available for future dictionary expansion and analysis). No `RelationType::Dynamic(String)` — the schema stays closed until precision is validated.
- Entity dedup pipeline (exact → case-insensitive → stemmed → embedding merge) stays as-is.

### What stays the same

- `NerEntityRecord` format (or its replacement `ExtractedEntity`) flows through `Experience.ner_entities`. **Backwards compatibility:** Existing RocksDB records use `NerEntityRecord`. The migration strategy is to support deserialization from both formats (serde `#[serde(untagged)]` or explicit version field) so that old records are readable without a full data migration. New writes use `ExtractedEntity`. A lazy migration converts records on read.
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
    nlp.rs              — Layer 2: POS tagging, NP chunking, verb lemmatization, SVO extraction
    types.rs            — ExtractedEntity, ExtractedTriple, ExtractionConfig, ExtractionResult
    config.rs           — Pattern loading from autonomite config, config versioning
    metadata.rs         — Source 0: tag/issue ID pass-through
    verb_dictionary.rs  — Canonical verb → RelationType mapping
```

New module, not modifications to existing files (except the integration point in `state.rs`).

**Module boundary:** The extraction module is stateless — it does not reach into `state.rs` or the graph. The AC dictionary and extraction config are passed in at construction time:

```rust
pub struct Extractor {
    ac_dict: Arc<AcDictionary>,   // shared, rebuilt in background
    config: ExtractionConfig,      // versioned, from autonomite config
    verb_dict: VerbDictionary,     // canonical verb → RelationType mapping
}
```

`state.rs` owns the `Extractor` instance and feeds it the AC dictionary. The extraction module never imports graph types directly.

## Aho-Corasick Dictionary

The maximal munch layer maintains a dictionary of previously-seen entities for fast multi-pattern matching:

- **Built from:** Existing entity nodes in the graph that pass admission gates (loaded at startup). Admission requires ALL of: (a) `source: MaximalMunch` or `source: Metadata` or `confidence >= 0.6`, (b) seen in >= 3 distinct memories, (c) not in the stopword list, (d) not a single lowercase token unless explicitly whitelisted. Note on gate (a) interaction with confidence caps: Source 2 NounPhrases have base 0.4, but a proper noun NP with subject boost reaches 0.4 + 0.4 + 0.2 = 1.0, capped at 0.75 — which passes the 0.6 gate. Single-word common nouns cap at 0.6 — borderline. The `source: MaximalMunch` disjunction ensures all Source 1 entities pass regardless of confidence math. This prevents re-importing `Other:MISC` noise and stops the reinforcement loop where a marginal NP enters the dictionary → gets matched everywhere → gains co-occurrence weight → appears "real" by frequency. If the graph is known to contain legacy junk, seed from an empty dictionary and let it grow organically.
- **Updated when:** New entities are discovered by heuristics and confirmed by graph insertion. New entities are tracked in a **staging counter** (entity text → distinct memory count) but do **not** participate in extraction matching until they graduate to the frozen AC automaton after meeting admission gates (>= 3 distinct memories). This prevents a single noisy extraction from influencing matching — the staging counter is bookkeeping only, not a lookup dictionary.
- **Decay:** Entities not seen in the last N ingestion cycles (configurable, default: 50) lose dictionary eligibility on the next AC rebuild. An "ingestion cycle" is one call to `process_experience_into_graph()` — whether triggered by a single HTTP request or a batch pipeline run. This prevents stale entities from accumulating indefinitely. Decay is checked at rebuild time, not during extraction.
- **Used for:** O(n) text scanning via Aho-Corasick automaton — finds all known entities in a single pass
- **Complements heuristics:** Known entities are found by dictionary, new entities are found by heuristics, both feed into longest-match resolution
- **AC-matched entity type and confidence:** When the AC dictionary matches an entity, the match inherits the `entity_type` and `source` from the entity's original extraction (stored in the dictionary alongside the text). A `NounPhrase` originally extracted by Source 2 that later enters the AC dictionary retains `entity_type: "NounPhrase"` and `source: NlpParse` — it does NOT inherit Source 1 base confidence (0.9). The AC dictionary is a recognition mechanism, not a re-classification mechanism. The entity's confidence is recalculated using its original type's rules and caps.

The `aho-corasick` crate is the standard Rust implementation (by the ripgrep author, highly optimized).

**Rebuild strategy — dual-layer dictionary:**

Building an AC automaton is fast for searching but expensive to construct. Rebuilding synchronously on every `process_experience_into_graph()` call would bottleneck batch ingestion. Instead:

1. **Frozen AC automaton:** Rebuilt asynchronously in the background on a schedule (every 5 minutes or every 1,000 new staged entities, whichever comes first). Only the frozen automaton participates in extraction matching.
2. **Staging counter:** A `HashMap<String, HashSet<MemoryId>>` tracking new entities and the distinct memories they appear in. This is bookkeeping — it does NOT participate in extraction matching. Entities in staging are invisible to Source 1 until they graduate.
3. **Graduation on rebuild:** When the AC automaton is rebuilt, entities in the staging counter that meet admission gates (>= 3 distinct memories, not in stopword list, etc.) are added to the new automaton. Entities that don't meet gates stay in staging for the next cycle. The rebuild must be fenced (e.g., `Arc::swap` on the automaton reference) to ensure extraction always sees a consistent snapshot.

## Rust NLP Library Evaluation

Dependency parsing is out of scope (see Layer 2). The requirement is POS tagging only — NP chunking and SVO extraction are built on top of POS tags via rules, not a parse tree.

| Library | POS | Size | Notes |
|---------|-----|------|-------|
| `nlprule` | Yes | ~50MB model | LanguageTool-based, rules not ML. Has lemmatization. |
| Averaged Perceptron (custom) | Yes | ~5MB model | Standard NLP approach, trainable on Penn Treebank, pure Rust |
| Custom rule-based | Basic | Zero | Heuristic POS from capitalization + position. No lemmatization. |

`rust-bert` is explicitly excluded — it wraps C++ libtorch, adds gigabytes to the binary, complicates cross-compilation, and introduces cold-start latency. Not justified when NP chunking achieves the same compound-noun extraction without a parse tree.

The implementation plan should include an evaluation step before committing to a library. Key criterion: must provide lemmatization (needed for verb edge canonicalization) or pair with a standalone lemmatizer.

## Confidence Calibration

The confidence scores in the noun filtering table (0.8, 0.7, 0.6, etc.) are initial estimates. Since Phase 1 uses salience scoring and `max_entities_per_memory` truncation, uncalibrated scores across sources will cause truncation to favor whichever source has inflated scores rather than actual extraction quality.

**Calibration plan:**
1. Initial implementation uses the estimated scores
2. After gold dataset is built (see Testing), measure precision per source/signal combination
3. Adjust scores so that truncation rank-orders entities by actual precision
4. Metadata entities (Source 0, confidence 1.0) are exempt from truncation and do not participate in calibration

## Open Questions

These require decisions before or during implementation:

1. **URLs as entities vs. decomposed artifacts:** ~~Should `Url` be a graph entity, or should URLs be decomposed and normalized into host/path/query artifacts?~~ **Resolved:** Raw URLs are stored as attributes on the episodic node, not as primary graph entities. The graph entities are the normalized decompositions: `Domain` (host), optionally `Path` (normalized, query params stripped by default). This prevents URL fragmentation where `example.com/a?x=1` and `example.com/a?x=2` create separate nodes for what is semantically the same endpoint. The raw URL is preserved for provenance but does not participate in graph topology.

2. **Verb triple vs. co-occurrence edge precedence:** ~~When both a promoted verb triple and a co-occurrence edge exist for the same entity pair, which one wins in retrieval and ranking?~~ **Resolved:** Promoted typed edge takes precedence. When a promoted triple creates a typed edge for an entity pair, the co-occurrence `RelatedTo` edge for that pair is suppressed (not created). This prevents duplicate edges with conflicting semantics. If the same pair appears in a later memory without a verb triple, a `RelatedTo` edge is created normally — suppression is per-memory, not permanent.

3. **Legacy graph migration:** The existing graph may contain `Other:MISC` entities and generic `RelatedTo` edges from the old NER pipeline. What is the migration plan? Options: (a) leave old data, new extractions layer on top; (b) reprocess all memories through the new pipeline; (c) garbage-collect low-confidence entities below a threshold. Decision deferred to implementation, but the AC dictionary seeding gate (confidence/provenance filter) provides a partial answer.

## Testing Requirements

A gold dataset from real ISAP memories is required **before** selecting the Rust NLP library and **before** finalizing the chunking grammar.

**Gold dataset:**
- 50-100 representative memories from the ISAP workspace
- Hand-annotated with expected entities and triples
- Must include adversarial cases (see below)

**Precision/recall measured separately for:**
- Source 1 (maximal munch) entities
- Source 2 (NLP) noun phrase entities
- Verb triples (both promoted and unpromoted)
- Metadata pass-through (should be trivially 1.0)

**Adversarial test cases:**
- Passive voice: "the server was compromised by the malware"
- Coordinated verbs: "the actor scanned and exploited the host"
- Coordinated objects: "actor scanned ports and exploited the server" (must not produce `exploit(actor, ports)`)
- Prepositional traps: "malware deployed via server" (directionality)
- Long-range SVO: verb and object separated by > 8 tokens
- Sentence fragments and bullet points (common in security reports)
- URLs containing domains and IPs as sub-spans
- URL deduplication: `example.com/a?x=1` vs `example.com/a?x=2` → same normalized path
- Domain label semantics: `malicious-payments.example.com` → recover "malicious-payments" as DomainLabel
- Tags/issue IDs absent from content text (metadata-only entities)
- Config-supplied patterns overlapping with heuristic patterns
- Version strings that look like IPs: `1.0.14.234`
- File paths with dots: `file.tar.gz`, `config.v2.yaml`
- AC vs heuristic conflict: AC matches "example.com" inside a longer heuristic match "malicious-example.com"
- Tier 2 nouns: bare "server" filtered, "compromised server" preserved
- Dictionary drift: entity at 0.6 confidence seen in only 1 memory must not enter AC dictionary
- Source 1 entities in verb triples: "malware compromised example.com" must produce triple with example.com as object
- Agentless passive: "The server was compromised." → triple with subject: None, not promoted
- Root URL path suppression: `http://example.com/` must NOT emit Path("/")
- Metadata-only entity with no content span: tag entity must serialize/deserialize without char spans
- Tier 1 stopword inside multi-word NP: "use case" retained, standalone "use" filtered
- AC dictionary match retains original entity type (NounPhrase stays NounPhrase, not reclassified)
- Equal-length candidate tie-breaking: heuristic wins over AC match
- CVE cross-source dedup: `cve:CVE-2021-44228` tag + `CVE-2021-44228` Source 1 match → single entity, not two
- PP-linked NP merge: "proof of concept exploit" → single entity, not "proof" + "concept exploit"

## What This Enables

- **Meaningful graph structure** from actual domain artifacts, not NER noise
- **Typed relationships** from validated verb triples, evidence-based promotion
- **Compound noun entities** grouped by NP chunking over POS tags, not split into word fragments
- **Config-driven extraction** per autonomite for domain-specific patterns
- **Growing dictionary** via Aho-Corasick for fast recognition of known entities (with provenance gating)
- **Structured metadata preserved** as first-class entities, not downgraded through text parsing
- **Foundation for knowledge compiler** M2: graph topology that reflects actual knowledge structure
