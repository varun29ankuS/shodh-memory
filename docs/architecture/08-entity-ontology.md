# The entity and relation ontology

Written 2026-08-19. Every claim is grounded in a `file:line` on
`feat/ontology-named-individual-axis`. Following the evidence rules of
`docs/graph-construction-audit.md`: doc comments in this codebase are frequently
stale, so nothing here rests on one — where a comment and its code disagree, the
contradiction is reported as a finding.

This document exists because the ontology is spread across four representations
and several assignment paths, and a reader (human or agent) landing on any one of
them cannot tell which is authoritative. That ambiguity has already produced
three separate defects, documented below.

---

## 1. There are four entity type representations, not one

```
GLiNER fine label (141)          src/entity_type/entity-type-schema.json
        │  entity_type::coarse_of                  src/entity_type/mod.rs:74
        ▼
   coarse id (18)
        │  EntityLabel::from_coarse_id             src/graph_memory.rs:317
        ▼
   EntityLabel (35 + Other)                        src/graph_memory.rs:200
        │  NerEntityType::from_coarse              src/embeddings/ner.rs:65
        ▼
   NerEntityType (4: PER/ORG/LOC/MISC)             src/embeddings/ner.rs:44
```

Each arrow loses information. The last one loses the most: `from_coarse` routes
person/title/role → `Person`, organization/team/norp → `Organization`,
location/gpe/facility/environment → `Location`, and **everything else** → `Misc`.
Twenty-plus distinct classes collapse into one bucket.

### Which is authoritative for what

| representation | authoritative for | notes |
|---|---|---|
| fine label (141) | what GLiNER predicted | order is load-bearing — see §5 |
| coarse id (18) | the schema's own rollup | runtime-loadable via PR #504 (open) |
| **`EntityLabel`** | **everything downstream of ingest** | the node stores this in `labels` |
| `NerEntityType` | legacy; a pre-GLiNER view | should not drive decisions — see §2 |

**Rule for new code: ask `EntityLabel`.** It is the richest representation that
survives into storage. `NerEntityType` exists because it predates the 141-class
schema and still has consumers; it is not a simplification you should reach for.

---

## 2. Three defects, one cause

Each of these came from asking `NerEntityType` a question `EntityLabel` could
answer better. They are listed because the pattern will recur.

**a. Proper-noun-ness.** `is_proper_noun` was `!matches!(entity_type, Misc)`, so
every titled work, product, project, repository, service, named event and law was
recorded as a **common noun**. Two consequences: the entity lost the proper-noun
salience boost (`graph_memory.rs:9198`), and it entered the **stemmed merge
index** (`graph_memory.rs:3892, 3932`) — the index the code deliberately keeps
proper nouns out of, per its own comment at `graph_memory.rs:3740`, "to prevent
'Paris' → 'pari' merging with 'Parison'".

**b. Salience.** Same predicate, same call site, same cause.

**c. Entity admission.** `state.rs` filter rules 7 and 8 apply an extra
confidence bar to `Misc` spans that are lowercase or shorter than five
characters. Since `Misc` contains `Repository`, `Service`, `Database`,
`Technology` and `Module`, a span typed `Repository` = `"serde"` must clear a bar
a `Person` never faces. This one **drops** entities rather than re-weighting
them, and a dropped entity is unrecoverable downstream.

### The single source of truth

`EntityLabel::denotes_named_individual()` — `src/graph_memory.rs`.

It is a **second, orthogonal axis** to `parent_labels()`. Subsumption answers
"what kind of thing is this" and puts each label in one place in a tree; this
answers "does this thing bear a name", which cuts *across* that tree. `Work`
rolls up to `Concept` and `Repository` to `Technology`, yet "Little Women" and
"shodh-memory" are both named individuals — no position in a subsumption
hierarchy expresses the property they share. This is the same reason Palantir's
ontology carries interfaces alongside object types.

`named_ness_is_orthogonal_to_the_subsumption_hierarchy` fails if this ever
becomes derivable from `parent_labels()` — i.e. if the axis turns out redundant,
you find out.

Borderline classes resolve to `true` on a stated asymmetry: a wrong `true` costs
a missed merge; a wrong `false` fuses two distinct named things into one node and
every traversal downstream inherits the error.

---

## 3. The relation ontology

36 variants (`src/graph_memory.rs:2019`). Two lexicons map language onto them,
and they are **deliberately not identical**:

| | function | direction |
|---|---|---|
| query | `verb_stem_to_relation_types` — `memory/query_parser.rs:4104` | 122 verb stems → 31 relation types |
| ingest | `predicate_from_cues` — `graph_memory.rs:2251` | phrase cues → relation |
| ingest | `infer_relation_type_for_pair` — `graph_memory.rs:2465` | label pair → relation |

**The invariant:** every relation the query side can route to must be producible
by some extractor. It was violated — `Prefers`, `Recommends`, `Implements` and
`Approves` were queryable and unmintable, so a query for "what did X recommend"
searched for an edge type nothing could create and returned empty, which is
indistinguishable from a genuine absence of evidence.

**The deliberate asymmetry:** `query_parser` maps `"like"` → `Prefers`. Correct
for a *query*; wrong as an *extraction* cue, because in conversational text
"like" is a simile, a filler and a discourse marker far more often than a
preference. `weak_conversational_verbs_are_not_extraction_cues` pins this so it
is not "fixed" by someone reading only one lexicon.

**Two traps when auditing this enum:**

1. Excluding `graph_memory.rs` as a "declaration surface" hides two producers
   that live in it (`predicate_from_cues`, `infer_relation_type_for_pair`). An
   audit that does so reports 14 dead relations; the real answer was 4.
2. `Precedes` reads as dead but is minted as `Custom("Precedes")` by CATENA and
   normalised on read by `RelationType::normalize` (`graph_memory.rs:2184-2186`). Check the `Custom(...)` back door
   before calling a variant unreachable.

### What the ontology actually produces

Declaring a type does not populate it. Measured on a live 61k-edge graph
(tier-stratified sample, so directional):

```
CoRetrieved 40.2% · CoOccurs 23.3% · RelatedTo 19.4%   ← 83% generic
Precedes 8.5% · Causes 4.4% · CreatedBy 2.5% · rest <1%
10 of 36 declared types appear at all
```

`spreading_weight`'s own doc comment predicts the consequence: traversal must
run "preferentially along meaning rather than adjacency, otherwise traversal over
a co-occurrence graph just rediscovers lexical co-occurrence (which BM25 already
has)". **That weighting is gated by `SHODH_GRAPH_PREDICATE_WEIGHTS`, default
off** (`memory/graph_retrieval.rs:283`) — so by default `Causes` (1.3) and
`CoOccurs` (0.5) spread identically and the relation ontology is invisible to
retrieval.

---

## 4. The taxonomy layer

`src/taxonomy/` supplies category knowledge the corpus never states, the way
`src/kb/` supplies identity. 49,170 lemmas from WordNet, vendored, no network,
no model. Regenerate with `scripts/build_taxonomy.py`.

Sense selection is the whole difficulty and both obvious rules fail. Walking
every sense invents categories (`share` → `way`, `work` → `check`); taking
WordNet's first sense resolves `turtle` to a **turtleneck sweater**. The asset
prefers the synset whose own name is the lemma, falling back to sense 1.

**Status: half-wired.** Ingest can mint `IsA` edges behind
`SHODH_TAXONOMY_EDGES` (default off), but **nothing on the query side consults
the taxonomy** — no path turns the word "animal" in a query into a lookup against
the `animal` node. Until that exists the layer cannot answer the query it was
built for, flag or no flag. Its `spreading_weight` of 0.9 is also inert while
predicate weights are off.

---

## 5. Constraints that will bite you

**GLiNER maps class index → fine label by schema order** against
`label_embeddings.bin`, a `(141, 384)` matrix computed offline by a MiniLM label
tower. Reordering or renaming within 141 is cheap; **adding a class requires
regenerating that asset** via `scripts/export_gliner_bi_edge.py`. Runtime-loadable
is not the same as freely extensible.

**`EntityNode` is serialized with postcard** (`src/serialization.rs:4`), which is
positional and non-self-describing. Removing a field shifts every field after it
and silently corrupts stored entities. `is_proper_noun` therefore remains a
stored field even though it is fully derivable; removing it is a migration, not a
refactor.

---

## 6. Rules for changing the ontology

1. **Ask `EntityLabel`, not `NerEntityType`.** If you find yourself matching on
   `Misc`, you are asking a four-way question of a thirty-five-way world.
2. **A relation the query side can route to must be producible.** Adding a
   variant to `RelationType` without a producer creates a query that returns
   empty and looks like an honest answer.
3. **New mechanisms land default-off, and the experiment gets finished.**
   Shipping a flag off is correct discipline for something unmeasured; leaving it
   unmeasured for months is how `predicate_weights` became a "lever-1 prototype"
   nobody can say was accepted or rejected. Record the number, or record that it
   is open and when.
4. **Numbers carry `n`.** See `src/recall_harness/uncertainty.rs`. Every gated
   metric on the 100-case gate currently sits inside its own 95% interval by 9×
   to 152×, so a delta smaller than the interval is a change on a fixed
   benchmark, not evidence about quality.
