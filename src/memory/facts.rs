//! Semantic Fact Storage
//!
//! Persistent storage for semantic facts extracted from episodic memories.
//! Facts represent durable knowledge distilled from multiple experiences.
//!
//! Storage schema:
//! - `facts:{user_id}:{fact_id}` - Primary fact storage
//! - `facts_by_entity:{user_id}:{entity}:{fact_id}` - Entity index for fast lookup
//! - `facts_by_type:{user_id}:{type}:{fact_id}` - Type index
//! - `facts_embedding:{user_id}:{fact_id}` - Pre-computed embedding vector (384-dim)

use anyhow::Result;
use rocksdb::{IteratorMode, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::compression::{FactType, SemanticFact};

/// Response for fact queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactQueryResponse {
    pub facts: Vec<SemanticFact>,
    pub total: usize,
}

/// Statistics about semantic facts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactStats {
    pub total_facts: usize,
    pub by_type: std::collections::HashMap<String, usize>,
    pub avg_confidence: f32,
    pub avg_support: f32,
}

/// Postcard defaults for the trailing `SemanticFact` fields added by the
/// invalidation increment, in declaration order: `invalidated_at: Option`
/// (None = `0x00`), `invalidated_by: Option` (None = `0x00`),
/// `contradicts: Vec` (empty = varint `0x00`).
///
/// Facts were previously decoded with plain `try_decode`, which has NO
/// tolerance for a record written with fewer fields than the struct now
/// declares — postcard is positional and carries no field presence, so adding a
/// trailing field without this would make **every existing fact in every live
/// store fail to decode**. `try_decode_compat` appends these one at a time.
/// Keep in sync with any new trailing field.
const FACT_DEFAULT_SUFFIX: &[u8] = &[0x00, 0x00, 0x00];

/// Decode a stored `SemanticFact`, tolerating records written before the
/// invalidation fields existed. The single choke point for every fact read.
fn decode_fact(data: &[u8]) -> Result<SemanticFact> {
    let (fact, _needs_migration) =
        crate::serialization::try_decode_compat::<SemanticFact>(data, FACT_DEFAULT_SUFFIX)?;
    Ok(fact)
}

/// What [`SemanticFactStore::ingest_candidate`] decided about one extracted fact.
///
/// Every variant corresponds to writes that have ALREADY happened. The caller's
/// only remaining jobs are counting, emitting introspection events, and wiring
/// newly active facts into the knowledge graph — never deciding.
#[derive(Debug, Clone)]
pub enum FactIngestOutcome {
    /// Matched an active fact AND contributed source memories it did not
    /// already have. The existing row was reinforced and rewritten.
    Reinforced {
        fact: SemanticFact,
        confidence_before: f32,
    },

    /// Matched an active fact but contributed NO new source evidence.
    /// Nothing was written — see the idempotence note on `ingest_candidate`.
    AlreadyAttested { fact_id: String },

    /// Matched an INVALIDATED fact. Recognised, ignored, left to decay on its
    /// base half-life. Nothing was written.
    MatchedSuperseded {
        fact_id: String,
        superseded_by: Option<String>,
    },

    /// Genuinely new, contradicting nothing. Stored active.
    Stored { fact: SemanticFact },

    /// Contradicted an active fact and WON. The loser has been invalidated and
    /// rewritten; the winner is stored active.
    Superseded {
        winner: SemanticFact,
        loser_id: String,
    },

    /// Contradicted a better-supported active fact and LOST. The winner's
    /// contradiction link has been written and the newcomer is stored ALREADY
    /// invalidated, so the disagreement is auditable instead of lost.
    Rejected {
        loser: SemanticFact,
        winner_id: String,
    },
}

impl FactIngestOutcome {
    /// The fact that was newly written AND is active — the only thing a caller
    /// may connect to the knowledge graph or announce as "extracted".
    ///
    /// Deliberately excludes `Rejected`: a newcomer that lost arbitration is
    /// stored for audit and must not be wired into the graph, or the losing
    /// claim would still be reachable by traversal.
    pub fn newly_active_fact(&self) -> Option<&SemanticFact> {
        match self {
            Self::Stored { fact } => Some(fact),
            Self::Superseded { winner, .. } => Some(winner),
            _ => None,
        }
    }

    /// The fact whose existing row was reinforced, with its prior confidence.
    pub fn reinforced_fact(&self) -> Option<(&SemanticFact, f32)> {
        match self {
            Self::Reinforced {
                fact,
                confidence_before,
            } => Some((fact, *confidence_before)),
            _ => None,
        }
    }
}

/// Storage for semantic facts with indexing
pub struct SemanticFactStore {
    db: Arc<DB>,
}

impl SemanticFactStore {
    /// Create a new fact store backed by RocksDB
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Get references to all RocksDB databases for backup
    pub fn databases(&self) -> Vec<(&str, &Arc<DB>)> {
        vec![("semantic_facts", &self.db)]
    }

    /// Store a semantic fact
    pub fn store(&self, user_id: &str, fact: &SemanticFact) -> Result<()> {
        // Primary storage
        let key = format!("facts:{}:{}", user_id, fact.id);
        let value = crate::serialization::encode(fact)?;
        self.db.put(key.as_bytes(), &value)?;

        // Entity index - index by each related entity
        for entity in &fact.related_entities {
            let entity_key = format!(
                "facts_by_entity:{}:{}:{}",
                user_id,
                entity.to_lowercase(),
                fact.id
            );
            self.db.put(entity_key.as_bytes(), fact.id.as_bytes())?;
        }

        // Type index
        let type_name = format!("{:?}", fact.fact_type);
        let type_key = format!("facts_by_type:{}:{}:{}", user_id, type_name, fact.id);
        self.db.put(type_key.as_bytes(), fact.id.as_bytes())?;

        Ok(())
    }

    /// Store multiple facts in a batch
    pub fn store_batch(&self, user_id: &str, facts: &[SemanticFact]) -> Result<usize> {
        let mut stored = 0;
        for fact in facts {
            if self.store(user_id, fact).is_ok() {
                stored += 1;
            }
        }
        Ok(stored)
    }

    /// Get a fact by ID
    pub fn get(&self, user_id: &str, fact_id: &str) -> Result<Option<SemanticFact>> {
        let key = format!("facts:{}:{}", user_id, fact_id);
        match self.db.get(key.as_bytes())? {
            Some(data) => {
                let fact = decode_fact(&data)?;
                Ok(Some(fact))
            }
            None => Ok(None),
        }
    }

    /// Update an existing fact (for reinforcement)
    ///
    /// Rebuilds entity and type indices in case content/entities changed.
    pub fn update(&self, user_id: &str, fact: &SemanticFact) -> Result<()> {
        // Clean up old indices if the fact existed (entities/type may have changed)
        if let Some(old_fact) = self.get(user_id, &fact.id)? {
            for entity in &old_fact.related_entities {
                let entity_key = format!(
                    "facts_by_entity:{}:{}:{}",
                    user_id,
                    entity.to_lowercase(),
                    fact.id
                );
                let _ = self.db.delete(entity_key.as_bytes());
            }
            let old_type_name = format!("{:?}", old_fact.fact_type);
            let old_type_key = format!("facts_by_type:{}:{}:{}", user_id, old_type_name, fact.id);
            let _ = self.db.delete(old_type_key.as_bytes());
        }

        // Re-store with fresh indices
        self.store(user_id, fact)
    }

    /// Delete a fact
    pub fn delete(&self, user_id: &str, fact_id: &str) -> Result<bool> {
        // Get fact first to clean up indices
        if let Some(fact) = self.get(user_id, fact_id)? {
            // Delete entity indices
            for entity in &fact.related_entities {
                let entity_key = format!(
                    "facts_by_entity:{}:{}:{}",
                    user_id,
                    entity.to_lowercase(),
                    fact_id
                );
                self.db.delete(entity_key.as_bytes())?;
            }

            // Delete type index
            let type_name = format!("{:?}", fact.fact_type);
            let type_key = format!("facts_by_type:{}:{}:{}", user_id, type_name, fact_id);
            self.db.delete(type_key.as_bytes())?;

            // Delete primary record
            let key = format!("facts:{}:{}", user_id, fact_id);
            self.db.delete(key.as_bytes())?;

            // Delete embedding if present
            let _ = self.delete_embedding(user_id, fact_id);

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all facts for a user
    pub fn list(&self, user_id: &str, limit: usize) -> Result<Vec<SemanticFact>> {
        let prefix = format!("facts:{}:", user_id);
        let mut facts = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            // Stop when we leave the prefix
            if !key_str.starts_with(&prefix) {
                break;
            }

            // Skip index keys (they contain extra colons)
            if key_str.matches(':').count() > 2 {
                continue;
            }

            if let Ok(fact) = decode_fact(&value) {
                facts.push(fact);
                if facts.len() >= limit {
                    break;
                }
            }
        }

        // Sort by confidence (highest first)
        facts.sort_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then_with(|| a.id.cmp(&b.id))
        });

        Ok(facts)
    }

    /// Find facts by related entity
    pub fn find_by_entity(
        &self,
        user_id: &str,
        entity: &str,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        let prefix = format!("facts_by_entity:{}:{}:", user_id, entity.to_lowercase());
        let mut facts = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let fact_id = String::from_utf8_lossy(&value);
            if seen_ids.insert(fact_id.to_string()) {
                if let Some(fact) = self.get(user_id, &fact_id)? {
                    facts.push(fact);
                    if facts.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(facts)
    }

    /// Find facts by type
    pub fn find_by_type(
        &self,
        user_id: &str,
        fact_type: FactType,
        limit: usize,
    ) -> Result<Vec<SemanticFact>> {
        let type_name = format!("{:?}", fact_type);
        let prefix = format!("facts_by_type:{}:{}:", user_id, type_name);
        let mut facts = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let fact_id = String::from_utf8_lossy(&value);
            if let Some(fact) = self.get(user_id, &fact_id)? {
                facts.push(fact);
                if facts.len() >= limit {
                    break;
                }
            }
        }

        Ok(facts)
    }

    /// Search facts by keyword in fact content
    pub fn search(&self, user_id: &str, query: &str, limit: usize) -> Result<Vec<SemanticFact>> {
        let query_lower = query.to_lowercase();
        let all_facts = self.list(user_id, 1000)?; // Get all facts

        let mut matching: Vec<SemanticFact> = all_facts
            .into_iter()
            .filter(|f| f.fact.to_lowercase().contains(&query_lower))
            .collect();

        matching.truncate(limit);
        Ok(matching)
    }

    /// Get statistics about stored facts
    pub fn stats(&self, user_id: &str) -> Result<FactStats> {
        let facts = self.list(user_id, 10000)?;

        if facts.is_empty() {
            return Ok(FactStats::default());
        }

        let mut by_type: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut total_confidence: f32 = 0.0;
        let mut total_support: usize = 0;

        for fact in &facts {
            let type_name = format!("{:?}", fact.fact_type);
            *by_type.entry(type_name).or_insert(0) += 1;
            total_confidence += fact.confidence;
            total_support += fact.support_count;
        }

        let count = facts.len();
        Ok(FactStats {
            total_facts: count,
            by_type,
            avg_confidence: total_confidence / count as f32,
            avg_support: total_support as f32 / count as f32,
        })
    }

    /// Find facts that should decay (no reinforcement for too long)
    pub fn find_decaying_facts(
        &self,
        user_id: &str,
        max_age_days: i64,
    ) -> Result<Vec<SemanticFact>> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days);
        let all_facts = self.list(user_id, 10000)?;

        let decaying: Vec<SemanticFact> = all_facts
            .into_iter()
            .filter(|f| f.last_reinforced < cutoff)
            .collect();

        Ok(decaying)
    }

    /// Check if a similar fact already exists (hybrid dedup)
    ///
    /// Multi-gate pipeline when embedding is provided:
    /// 1. Entity gate: at least 1 shared entity, OR both have zero entities
    /// 2. Polarity gate: same negation polarity (prevents merging contradictions)
    /// 3. Cosine gate: embedding similarity >= FACT_DEDUP_COSINE_THRESHOLD
    /// 4. Jaccard floor: word overlap >= FACT_DEDUP_JACCARD_FLOOR
    ///
    /// Falls back to pure Jaccard (0.70) if no embedding is provided.
    pub fn find_similar(
        &self,
        user_id: &str,
        fact_content: &str,
        fact_entities: &[String],
        new_embedding: Option<&[f32]>,
    ) -> Result<Option<SemanticFact>> {
        use crate::constants::{
            FACT_DEDUP_COSINE_THRESHOLD, FACT_DEDUP_JACCARD_FALLBACK, FACT_DEDUP_JACCARD_FLOOR,
        };
        use crate::similarity::cosine_similarity;

        let facts = self.list(user_id, 1000)?;
        let query_lower = fact_content.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();
        let new_polarity = detect_polarity(&query_lower);
        let new_entity_set: std::collections::HashSet<&str> =
            fact_entities.iter().map(|s| s.as_str()).collect();

        let use_hybrid = new_embedding.is_some();
        let mut best_match: Option<(f32, SemanticFact)> = None;

        for fact in facts {
            let fact_lower = fact.fact.to_lowercase();
            let fact_words: std::collections::HashSet<&str> =
                fact_lower.split_whitespace().collect();

            // Compute Jaccard (needed in both modes)
            let intersection = query_words.intersection(&fact_words).count();
            let union = query_words.union(&fact_words).count();
            let jaccard = if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            };

            if use_hybrid {
                // Gate 1: Entity overlap — at least 1 shared entity, or both empty
                let existing_entity_set: std::collections::HashSet<&str> =
                    fact.related_entities.iter().map(|s| s.as_str()).collect();
                let both_empty = new_entity_set.is_empty() && existing_entity_set.is_empty();
                let has_overlap = !new_entity_set.is_disjoint(&existing_entity_set);
                if !both_empty && !has_overlap {
                    continue;
                }

                // Gate 2: Polarity match — prevents merging contradictions
                let existing_polarity = detect_polarity(&fact_lower);
                if new_polarity != existing_polarity {
                    continue;
                }

                // Gate 3: Cosine similarity
                let new_emb = new_embedding.unwrap();
                match self.get_embedding(user_id, &fact.id) {
                    Ok(Some(existing_emb)) => {
                        let cosine = cosine_similarity(new_emb, &existing_emb);
                        if cosine < FACT_DEDUP_COSINE_THRESHOLD {
                            continue;
                        }

                        // Gate 4: Jaccard sanity floor
                        if jaccard < FACT_DEDUP_JACCARD_FLOOR {
                            continue;
                        }

                        // Passed all gates — rank by cosine
                        if best_match.as_ref().is_none_or(|(s, _)| cosine > *s) {
                            best_match = Some((cosine, fact));
                        }
                    }
                    _ => {
                        // No stored embedding — fall back to Jaccard-only for this candidate
                        if jaccard >= FACT_DEDUP_JACCARD_FALLBACK
                            && best_match.as_ref().is_none_or(|(s, _)| jaccard > *s)
                        {
                            best_match = Some((jaccard, fact));
                        }
                    }
                }
            } else {
                // Fallback: pure Jaccard (legacy behavior when embedder unavailable)
                if jaccard >= FACT_DEDUP_JACCARD_FALLBACK {
                    return Ok(Some(fact));
                }
            }
        }

        Ok(best_match.map(|(_, fact)| fact))
    }

    /// Find an ACTIVE fact that directly contradicts the incoming one.
    ///
    /// This is the other half of the polarity gate in [`Self::find_similar`].
    /// That gate asks "same polarity?" and refuses to merge when the answer is
    /// no — which was correct as dedup and disastrous as knowledge management:
    /// the negation was simply stored as a second row, so "four crew injured"
    /// and "corrected: no crew injured" coexisted, unlinked, each ratcheting its
    /// own confidence and extending its own half-life. Nothing ever arbitrated.
    ///
    /// The gates here are deliberately the SAME as `find_similar`'s — shared
    /// entity, cosine >= `FACT_DEDUP_COSINE_THRESHOLD`, Jaccard >=
    /// `FACT_DEDUP_JACCARD_FLOOR` — with exactly one inversion: polarity must
    /// DIFFER. Two facts that are near-identical in embedding and wording, about
    /// the same entities, but opposite in polarity, are a claim and its
    /// negation. Reusing the same thresholds means contradiction detection is
    /// exactly as conservative as dedup already is; it cannot fire on pairs the
    /// system would not otherwise have considered "the same fact".
    ///
    /// Requires an embedding. The Jaccard-only fallback used by dedup is not
    /// safe here: negation is often a one-word difference ("no", "not"), so a
    /// pure bag-of-words score cannot distinguish a contradiction from a
    /// paraphrase, and a false positive INVALIDATES a true fact. When no
    /// embedder is available this returns `None` and the two rows coexist as
    /// before — degraded, but never wrong.
    ///
    /// Already-invalidated facts are skipped: a superseded fact is not evidence,
    /// so it cannot win or lose another arbitration.
    pub fn find_contradiction(
        &self,
        user_id: &str,
        fact_content: &str,
        fact_entities: &[String],
        new_embedding: Option<&[f32]>,
    ) -> Result<Option<SemanticFact>> {
        use crate::constants::{FACT_DEDUP_COSINE_THRESHOLD, FACT_DEDUP_JACCARD_FLOOR};
        use crate::similarity::cosine_similarity;

        let Some(new_emb) = new_embedding else {
            return Ok(None);
        };

        let query_lower = fact_content.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();
        let new_polarity = detect_polarity(&query_lower);
        let new_entity_set: std::collections::HashSet<&str> =
            fact_entities.iter().map(|s| s.as_str()).collect();

        let mut best_match: Option<(f32, SemanticFact)> = None;

        for fact in self.list(user_id, 1000)? {
            if !fact.is_active() {
                continue;
            }

            let fact_lower = fact.fact.to_lowercase();

            // Polarity must DIFFER — this is the inversion.
            if detect_polarity(&fact_lower) == new_polarity {
                continue;
            }

            // Entity gate (identical to find_similar).
            let existing_entity_set: std::collections::HashSet<&str> =
                fact.related_entities.iter().map(|s| s.as_str()).collect();
            let both_empty = new_entity_set.is_empty() && existing_entity_set.is_empty();
            if !both_empty && new_entity_set.is_disjoint(&existing_entity_set) {
                continue;
            }

            // Jaccard floor (identical to find_similar).
            let fact_words: std::collections::HashSet<&str> =
                fact_lower.split_whitespace().collect();
            let intersection = query_words.intersection(&fact_words).count();
            let union = query_words.union(&fact_words).count();
            let jaccard = if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            };
            if jaccard < FACT_DEDUP_JACCARD_FLOOR {
                continue;
            }

            // Cosine gate (identical to find_similar). No stored embedding means
            // we cannot judge safely — skip rather than guess.
            let Ok(Some(existing_emb)) = self.get_embedding(user_id, &fact.id) else {
                continue;
            };
            let cosine = cosine_similarity(new_emb, &existing_emb);
            if cosine < FACT_DEDUP_COSINE_THRESHOLD {
                continue;
            }

            if best_match.as_ref().is_none_or(|(s, _)| cosine > *s) {
                best_match = Some((cosine, fact));
            }
        }

        Ok(best_match.map(|(_, fact)| fact))
    }

    /// Admit one freshly extracted fact into the store, applying THE
    /// arbitration policy.
    ///
    /// # Why this exists
    ///
    /// There were two ingest paths — the timer-driven `run_maintenance` cycle
    /// and the on-demand `distill_facts` endpoint — and they disagreed. Only
    /// maintenance called [`Self::find_contradiction`]; `distill_facts` pushed
    /// a contradicting fact straight onto the "genuinely new" pile, which
    /// re-created the exact defect the invalidation increment was built to fix:
    /// a claim and its negation coexisting as two unlinked active rows, each
    /// ratcheting its own confidence and extending its own half-life. Which
    /// policy applied to a given fact then depended on whether a human clicked
    /// "consolidate" or a timer fired — an arbitrary property of scheduling
    /// deciding what the system believes.
    ///
    /// Two arbitration policies in one system is the defect regardless of which
    /// one is better, so both callers now route through here. This is the ONLY
    /// place that decides which of two conflicting facts is active.
    ///
    /// # The policy, in order
    ///
    /// 1. **Dedup** ([`Self::find_similar`]): same polarity, shared entity,
    ///    cosine and Jaccard above threshold ⇒ the same claim restated.
    ///    - If the match is INVALIDATED, it absorbs its own re-derivation
    ///      without coming back to life. The wrong claim usually stays in the
    ///      corpus and is re-extracted on every cycle forever; matching it back
    ///      onto its dead row is what stops a correction and the claim it
    ///      corrected from trading places indefinitely.
    ///    - If the match is active, reinforce it — but only when this candidate
    ///      contributes source memories the stored fact did not already have.
    ///      See the idempotence note below.
    /// 2. **Contradiction** ([`Self::find_contradiction`]): identical gates with
    ///    polarity INVERTED ⇒ a claim and its negation. The NEWER fact wins
    ///    unless the existing one has strictly greater `support_count`.
    ///    Recency is the default because a contradiction arriving later is
    ///    usually a correction. Support overrides recency because one stray
    ///    extraction should not overturn a many-times-attested claim.
    ///    Confidence is deliberately NOT the tie-breaker: it is a ratchet that
    ///    rises with every re-derivation, so using it would make an older claim
    ///    progressively harder to correct — the "well-supported wrong fact is
    ///    immortal" failure. Ties go to the newcomer.
    /// 3. Otherwise the fact is new. Store it.
    ///
    /// The loser of an arbitration is INVALIDATED, never deleted: the
    /// correction keeps an auditable record of what it replaced, and
    /// `source_memories` on both rows keeps the trust chain back to the
    /// episodes intact.
    ///
    /// # Idempotence (what makes full re-scans safe)
    ///
    /// Both callers re-run extraction over the FULL eligible corpus on every
    /// cycle. There used to be an incremental watermark that filtered each
    /// cycle to memories it had not seen; it advanced over every memory the
    /// filter passed, including memories the consolidator's age gate had
    /// excluded, so every memory on a live store was seen once too young and
    /// then excluded forever — the zero-facts defect. It also confined
    /// `CONSOLIDATION_MIN_SUPPORT` corroboration to a single cycle's batch,
    /// so support could never accumulate across cycles. The watermark is
    /// gone; re-derivation of already-stored facts is instead made inert
    /// HERE, at the evidence level, which handles every re-derivation path
    /// rather than one scheduling instance of it.
    ///
    /// Re-processing is not "merely wasteful" without this guard. The
    /// reinforcement branch used to refresh `last_reinforced` and apply the
    /// confidence boost UNCONDITIONALLY, so a re-derived fact got a fresh
    /// half-life and a confidence bump on every cycle, from evidence that had
    /// already been counted — an immortal fact. `support_count` was already
    /// guarded against exactly this; the guard simply did not extend to the
    /// other two.
    ///
    /// Reinforcement is therefore idempotent with respect to evidence: no new
    /// source memories ⇒ [`FactIngestOutcome::AlreadyAttested`] and NO write
    /// at all. Any re-derivation from an already-counted source set is inert.
    ///
    /// `now` is passed in so a caller processing a batch stamps every decision
    /// with one consistent instant.
    pub fn ingest_candidate(
        &self,
        user_id: &str,
        candidate: &SemanticFact,
        embedding: Option<&[f32]>,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<FactIngestOutcome> {
        use crate::constants::FACT_REINFORCEMENT_BOOST;

        // ── 1. Dedup ────────────────────────────────────────────────────────
        if let Some(mut existing) = self.find_similar(
            user_id,
            &candidate.fact,
            &candidate.related_entities,
            embedding,
        )? {
            if !existing.is_active() {
                tracing::debug!(
                    fact_id = %existing.id,
                    superseded_by = ?existing.invalidated_by,
                    "Fact ingest: re-derivation of an invalidated fact, ignored"
                );
                return Ok(FactIngestOutcome::MatchedSuperseded {
                    fact_id: existing.id.clone(),
                    superseded_by: existing.invalidated_by.clone(),
                });
            }

            let mut new_sources_added = false;
            for src in &candidate.source_memories {
                if !existing.source_memories.contains(src) {
                    existing.source_memories.push(src.clone());
                    new_sources_added = true;
                }
            }

            if !new_sources_added {
                tracing::debug!(
                    fact_id = %existing.id,
                    "Fact ingest: re-derivation from already-counted evidence, no write"
                );
                return Ok(FactIngestOutcome::AlreadyAttested {
                    fact_id: existing.id.clone(),
                });
            }

            let confidence_before = existing.confidence;
            existing.support_count += 1;
            existing.last_reinforced = now;
            let boost = FACT_REINFORCEMENT_BOOST * (1.0 - existing.confidence);
            existing.confidence = (existing.confidence + boost).min(1.0);
            for entity in &candidate.related_entities {
                if !existing.related_entities.contains(entity) {
                    existing.related_entities.push(entity.clone());
                }
            }

            self.update(user_id, &existing)?;
            // Refresh the stored encoding so dedup and contradiction detection
            // always compare against the latest embedder, not the one that
            // happened to be loaded when the fact was first minted.
            if let Some(emb) = embedding {
                let _ = self.store_embedding(user_id, &existing.id, emb);
            }

            return Ok(FactIngestOutcome::Reinforced {
                fact: existing,
                confidence_before,
            });
        }

        // ── 2. Contradiction ────────────────────────────────────────────────
        let contradiction = self.find_contradiction(
            user_id,
            &candidate.fact,
            &candidate.related_entities,
            embedding,
        )?;

        if let Some(mut existing) = contradiction {
            // Ties go to the newcomer: `>=`, not `>`.
            let incoming_wins = candidate.support_count >= existing.support_count;

            if incoming_wins {
                let mut winner = candidate.clone();
                winner.link_contradiction(&existing.id);
                existing.invalidate(Some(&winner.id), now);

                // Invalidate the loser BEFORE storing the winner. A crash
                // between the two loses a claim; the reverse order would leave
                // two ACTIVE contradicting rows — the state this whole
                // mechanism exists to prevent.
                self.update(user_id, &existing)?;
                self.store(user_id, &winner)?;
                if let Some(emb) = embedding {
                    let _ = self.store_embedding(user_id, &winner.id, emb);
                }

                tracing::info!(
                    superseded = %existing.id,
                    winner = %winner.id,
                    "Fact contradiction: newer claim supersedes"
                );
                return Ok(FactIngestOutcome::Superseded {
                    winner,
                    loser_id: existing.id,
                });
            }

            let mut loser = candidate.clone();
            loser.invalidate(Some(&existing.id), now);
            existing.link_contradiction(&loser.id);

            self.update(user_id, &existing)?;
            self.store(user_id, &loser)?;
            if let Some(emb) = embedding {
                let _ = self.store_embedding(user_id, &loser.id, emb);
            }

            tracing::info!(
                rejected = %loser.id,
                winner = %existing.id,
                "Fact contradiction: better-supported claim holds"
            );
            return Ok(FactIngestOutcome::Rejected {
                loser,
                winner_id: existing.id,
            });
        }

        // ── 3. Genuinely new ────────────────────────────────────────────────
        let fact = candidate.clone();
        self.store(user_id, &fact)?;
        if let Some(emb) = embedding {
            let _ = self.store_embedding(user_id, &fact.id, emb);
        }
        Ok(FactIngestOutcome::Stored { fact })
    }

    // =========================================================================
    // EMBEDDING PERSISTENCE
    // =========================================================================

    /// Store pre-computed embedding vector for a fact
    ///
    /// Key format: `facts_embedding:{user_id}:{fact_id}` → bincode Vec<f32>
    /// Stored separately from SemanticFact struct for backward compatibility.
    pub fn store_embedding(&self, user_id: &str, fact_id: &str, embedding: &[f32]) -> Result<()> {
        let key = format!("facts_embedding:{user_id}:{fact_id}");
        let value = crate::serialization::encode(embedding)?;
        self.db.put(key.as_bytes(), &value)?;
        Ok(())
    }

    /// Get pre-computed embedding vector for a fact
    pub fn get_embedding(&self, user_id: &str, fact_id: &str) -> Result<Option<Vec<f32>>> {
        let key = format!("facts_embedding:{user_id}:{fact_id}");
        match self.db.get(key.as_bytes())? {
            Some(data) => {
                let (embedding, _) = crate::serialization::try_decode::<Vec<f32>>(&data)?;
                Ok(Some(embedding))
            }
            None => Ok(None),
        }
    }

    /// Delete embedding for a fact (called during fact deletion)
    pub fn delete_embedding(&self, user_id: &str, fact_id: &str) -> Result<()> {
        let key = format!("facts_embedding:{user_id}:{fact_id}");
        self.db.delete(key.as_bytes())?;
        Ok(())
    }

    /// List all unique user IDs that have facts
    pub fn list_users(&self, limit: usize) -> Result<Vec<String>> {
        let prefix = "facts:";
        let mut users = std::collections::HashSet::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(prefix) {
                break;
            }

            // Key format: facts:{user_id}:{fact_id}
            // Skip index keys (facts_by_entity, facts_by_type)
            if key_str.starts_with("facts_by_") {
                continue;
            }

            // Extract user_id from key
            let parts: Vec<&str> = key_str.splitn(3, ':').collect();
            if parts.len() >= 2 {
                users.insert(parts[1].to_string());
                if users.len() >= limit {
                    break;
                }
            }
        }

        Ok(users.into_iter().collect())
    }

    /// Purge duplicate facts — analogous to synaptic consolidation where
    /// redundant traces merge into a single strong engram (Tononi & Cirelli 2014,
    /// synaptic homeostasis hypothesis: sleep prunes redundant synapses while
    /// strengthening unique traces).
    ///
    /// Groups facts by normalized text, keeps highest-support version,
    /// merges source_memories from duplicates into the survivor.
    pub fn purge_duplicates(&self, user_id: &str) -> Result<usize> {
        let all_facts = self.list(user_id, 10_000)?;
        if all_facts.is_empty() {
            return Ok(0);
        }

        // Group by normalized text: lowercase, trimmed, collapsed whitespace
        let mut groups: std::collections::HashMap<String, Vec<SemanticFact>> =
            std::collections::HashMap::new();
        for fact in all_facts {
            let normalized: String = fact
                .fact
                .to_lowercase()
                .split_whitespace()
                .collect::<Vec<&str>>()
                .join(" ");
            groups.entry(normalized).or_default().push(fact);
        }

        let mut purged = 0;
        for (_key, mut group) in groups {
            if group.len() < 2 {
                continue;
            }

            // Keep the fact with highest support_count (strongest trace)
            group.sort_by(|a, b| b.support_count.cmp(&a.support_count));
            let mut survivor = group.remove(0);

            // Merge source_memories from duplicates into survivor
            let mut all_sources: std::collections::HashSet<crate::memory::types::MemoryId> =
                survivor.source_memories.iter().cloned().collect();
            for dup in &group {
                for src in &dup.source_memories {
                    all_sources.insert(src.clone());
                }
                survivor.support_count += dup.support_count;
            }
            survivor.source_memories = all_sources.into_iter().collect();
            survivor.last_reinforced = chrono::Utc::now();
            self.update(user_id, &survivor)?;

            // Delete duplicates
            for dup in &group {
                self.delete(user_id, &dup.id)?;
                purged += 1;
            }
        }

        Ok(purged)
    }

    /// Purge noise facts that predate the quality filter — analogous to
    /// retroactive interference cleanup: the brain's consolidation process
    /// revisits existing traces and prunes those that no longer pass the
    /// hippocampal quality gate (Frankland & Bontempi 2005).
    ///
    /// Runs each stored fact through `is_knowledge_worthy()` and deletes
    /// facts that fail (noise that was stored before the filter existed).
    pub fn purge_noise_facts(&self, user_id: &str) -> Result<usize> {
        use super::compression::SemanticConsolidator;

        let all_facts = self.list(user_id, 10_000)?;
        let mut purged = 0;

        for fact in &all_facts {
            if !SemanticConsolidator::is_knowledge_worthy(&fact.fact) {
                self.delete(user_id, &fact.id)?;
                purged += 1;
            }
        }

        Ok(purged)
    }
}

/// Detect negation polarity of a fact statement.
///
/// Returns `true` for positive polarity (even negation count, including 0),
/// `false` for negative polarity (odd negation count).
/// Handles double-negation: "not unlike" = positive.
///
/// `pub(crate)`: `SemanticConsolidator::group_candidates_by_similarity` uses
/// the SAME polarity notion to keep a claim and its negation in separate
/// clusters, so what counts as "the same claim restated" is defined once —
/// here — for both extraction-time clustering and store-time dedup.
pub(crate) fn detect_polarity(text_lower: &str) -> bool {
    use crate::constants::FACT_NEGATION_MARKERS;
    let words: Vec<&str> = text_lower.split_whitespace().collect();
    let negation_count = words
        .iter()
        .filter(|w| FACT_NEGATION_MARKERS.iter().any(|marker| *w == marker))
        .count();
    negation_count % 2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store() -> (SemanticFactStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());
        (SemanticFactStore::new(db), temp_dir)
    }

    fn create_test_fact(id: &str, content: &str) -> SemanticFact {
        SemanticFact {
            id: id.to_string(),
            fact: content.to_string(),
            confidence: 0.8,
            support_count: 3,
            source_memories: vec![],
            related_entities: vec!["rust".to_string(), "memory".to_string()],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now(),
            fact_type: FactType::Pattern,
            invalidated_at: None,
            invalidated_by: None,
            contradicts: Vec::new(),
        }
    }

    /// A `SemanticFact` exactly as serialized before the invalidation fields
    /// existed — every field through `fact_type`, nothing after. This is the
    /// shape of every fact already in a live RocksDB store.
    ///
    /// Facts used to decode with plain `try_decode`, which has no EOF
    /// tolerance, so adding trailing fields without `FACT_DEFAULT_SUFFIX` would
    /// have made every one of them undecodable — a total, silent loss of the
    /// semantic layer on upgrade. This test is the guard on that.
    #[derive(serde::Serialize)]
    struct LegacyFactThroughFactType {
        id: String,
        fact: String,
        confidence: f32,
        support_count: usize,
        source_memories: Vec<crate::memory::MemoryId>,
        related_entities: Vec<String>,
        created_at: chrono::DateTime<chrono::Utc>,
        last_reinforced: chrono::DateTime<chrono::Utc>,
        fact_type: FactType,
    }

    #[test]
    fn legacy_fact_without_invalidation_fields_still_decodes() {
        assert_eq!(
            FACT_DEFAULT_SUFFIX.len(),
            3,
            "one default per trailing field added by the invalidation increment"
        );

        let now = chrono::Utc::now();
        let legacy = LegacyFactThroughFactType {
            id: "legacy-1".to_string(),
            fact: "the reactor was scrammed at 04:12".to_string(),
            confidence: 0.77,
            support_count: 5,
            source_memories: vec![crate::memory::MemoryId(uuid::Uuid::new_v4())],
            related_entities: vec!["reactor".to_string()],
            created_at: now,
            last_reinforced: now,
            fact_type: FactType::Pattern,
        };

        let bytes = crate::serialization::encode(&legacy).unwrap();
        let decoded = decode_fact(&bytes).expect("legacy fact must still decode");

        // Everything that WAS on disk survives untouched.
        assert_eq!(decoded.id, "legacy-1");
        assert_eq!(decoded.fact, "the reactor was scrammed at 04:12");
        assert!((decoded.confidence - 0.77).abs() < 1e-6);
        assert_eq!(decoded.support_count, 5);
        assert_eq!(decoded.related_entities, vec!["reactor".to_string()]);
        assert_eq!(decoded.source_memories.len(), 1);

        // ...and the new fields backfill to "never contradicted".
        assert_eq!(decoded.invalidated_at, None);
        assert_eq!(decoded.invalidated_by, None);
        assert!(decoded.contradicts.is_empty());
        assert!(
            decoded.is_active(),
            "a pre-invalidation fact must read as active, not as suppressed"
        );
    }

    #[test]
    fn current_fact_round_trips_invalidation_state() {
        let (store, _dir) = create_test_store();
        let mut fact = create_test_fact("f-inv", "no crew were injured");
        let now = chrono::Utc::now();
        fact.invalidate(Some("f-winner"), now);

        store.store("u", &fact).unwrap();
        let back = store.get("u", "f-inv").unwrap().expect("stored");

        assert!(!back.is_active());
        assert_eq!(
            back.invalidated_at.map(|t| t.timestamp()),
            Some(now.timestamp())
        );
        assert_eq!(back.invalidated_by.as_deref(), Some("f-winner"));
        assert_eq!(back.contradicts, vec!["f-winner".to_string()]);
        // Provenance to source memories must survive invalidation — the trust
        // chain does not break just because the claim was superseded.
        assert_eq!(back.source_memories, fact.source_memories);
    }

    #[test]
    fn invalidate_is_idempotent_and_keeps_the_first_timestamp() {
        let mut fact = create_test_fact("f", "no crew were injured");
        let first = chrono::Utc::now();
        let later = first + chrono::Duration::days(3);

        fact.invalidate(Some("winner-a"), first);
        fact.invalidate(Some("winner-a"), later);
        assert_eq!(
            fact.invalidated_at.map(|t| t.timestamp()),
            Some(first.timestamp()),
            "the audit trail records when belief stopped, not when it was re-noticed"
        );
        assert_eq!(
            fact.contradicts.len(),
            1,
            "re-linking the same opponent must not duplicate"
        );
    }

    #[test]
    fn find_contradiction_requires_an_embedding() {
        // Negation is often a one-word difference, so a bag-of-words score
        // cannot tell a contradiction from a paraphrase — and a false positive
        // INVALIDATES a true fact. Without an embedder the answer is "no
        // contradiction", never a guess.
        let (store, _dir) = create_test_store();
        let claim = create_test_fact("c1", "four crew were injured in the incident");
        store.store("u", &claim).unwrap();

        let found = store
            .find_contradiction(
                "u",
                "no crew were injured in the incident",
                &["rust".into()],
                None,
            )
            .unwrap();
        assert!(found.is_none(), "no embedding ⇒ no arbitration");
    }

    #[test]
    fn find_contradiction_matches_opposite_polarity_and_skips_same_polarity() {
        let (store, _dir) = create_test_store();

        // Identical embeddings isolate the polarity gate: everything else the
        // real pipeline checks (entity overlap, cosine, Jaccard) is satisfied,
        // so only polarity can decide the outcome.
        let emb = vec![0.5_f32; 8];

        let claim = create_test_fact("c1", "four crew were injured in the incident");
        store.store("u", &claim).unwrap();
        store.store_embedding("u", "c1", &emb).unwrap();

        // Opposite polarity, same entities, near-identical wording ⇒ contradiction.
        let hit = store
            .find_contradiction(
                "u",
                "four crew were not injured in the incident",
                &["rust".into()],
                Some(&emb),
            )
            .unwrap();
        assert_eq!(
            hit.map(|f| f.id),
            Some("c1".to_string()),
            "a negated restatement of the same claim is a contradiction"
        );

        // Same polarity ⇒ that is dedup's job, not arbitration's.
        let miss = store
            .find_contradiction(
                "u",
                "four crew were injured in the incident today",
                &["rust".into()],
                Some(&emb),
            )
            .unwrap();
        assert!(miss.is_none(), "same polarity must never be arbitrated");
    }

    #[test]
    fn find_contradiction_ignores_already_invalidated_facts() {
        // A superseded claim is not evidence, so it can neither win nor lose a
        // new arbitration. This is what stops a corrected fact and its
        // correction from trading places on every consolidation cycle.
        let (store, _dir) = create_test_store();
        let emb = vec![0.5_f32; 8];

        let mut dead = create_test_fact("dead", "four crew were injured in the incident");
        dead.invalidate(Some("winner"), chrono::Utc::now());
        store.store("u", &dead).unwrap();
        store.store_embedding("u", "dead", &emb).unwrap();

        let found = store
            .find_contradiction(
                "u",
                "four crew were not injured in the incident",
                &["rust".into()],
                Some(&emb),
            )
            .unwrap();
        assert!(found.is_none(), "dead rows are out of the arbitration");
    }

    #[test]
    fn test_store_and_get() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Rust is a systems programming language");

        store.store("user-1", &fact).unwrap();
        let retrieved = store.get("user-1", "fact-1").unwrap();

        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().fact,
            "Rust is a systems programming language"
        );
    }

    #[test]
    fn test_find_by_entity() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Rust has efficient memory management");

        store.store("user-1", &fact).unwrap();
        let results = store.find_by_entity("user-1", "rust", 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "fact-1");
    }

    #[test]
    fn test_find_by_type() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Pattern detected in codebase");

        store.store("user-1", &fact).unwrap();
        let results = store.find_by_type("user-1", FactType::Pattern, 10).unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_delete() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Test fact");

        store.store("user-1", &fact).unwrap();
        assert!(store.get("user-1", "fact-1").unwrap().is_some());

        store.delete("user-1", "fact-1").unwrap();
        assert!(store.get("user-1", "fact-1").unwrap().is_none());

        // Entity index should also be cleaned up
        let by_entity = store.find_by_entity("user-1", "rust", 10).unwrap();
        assert!(by_entity.is_empty());
    }

    // =========================================================================
    // ARBITRATION — the single ingest policy
    // =========================================================================

    /// Identical embeddings on both sides isolate whichever gate is under test:
    /// entity overlap, cosine and Jaccard are all satisfied by construction, so
    /// only polarity, support and liveness can decide an outcome.
    const EMB: &[f32] = &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

    fn candidate(id: &str, text: &str, support: usize, sources: usize) -> SemanticFact {
        let mut f = create_test_fact(id, text);
        f.support_count = support;
        f.source_memories = (0..sources)
            .map(|_| crate::memory::MemoryId(uuid::Uuid::new_v4()))
            .collect();
        f
    }

    fn active_count(store: &SemanticFactStore, user: &str) -> usize {
        store
            .list(user, 100)
            .unwrap()
            .iter()
            .filter(|f| f.is_active())
            .count()
    }

    #[test]
    fn a_contradiction_ends_with_exactly_one_active_fact_the_newer_one() {
        let (store, _dir) = create_test_store();
        let user = "arb";

        let claim = candidate("claim", "four crew were injured in the incident", 1, 1);
        store
            .ingest_candidate(user, &claim, Some(EMB), chrono::Utc::now())
            .unwrap();

        // Equal support ⇒ ties go to the newcomer.
        let correction = candidate(
            "correction",
            "four crew were not injured in the incident",
            1,
            1,
        );
        let outcome = store
            .ingest_candidate(user, &correction, Some(EMB), chrono::Utc::now())
            .unwrap();

        match &outcome {
            FactIngestOutcome::Superseded { winner, loser_id } => {
                assert_eq!(winner.id, "correction");
                assert_eq!(loser_id, "claim");
            }
            other => panic!("expected the correction to supersede the claim, got {other:?}"),
        }

        assert_eq!(
            active_count(&store, user),
            1,
            "a claim and its negation must never both be active"
        );

        let settled_claim = store
            .get(user, "claim")
            .unwrap()
            .expect("retained for audit");
        assert!(!settled_claim.is_active());
        assert_eq!(settled_claim.invalidated_by.as_deref(), Some("correction"));
        assert!(settled_claim
            .contradicts
            .contains(&"correction".to_string()));
        assert!(
            !settled_claim.source_memories.is_empty(),
            "invalidation must not break the trust chain back to the episodes"
        );

        let settled_correction = store.get(user, "correction").unwrap().expect("stored");
        assert!(settled_correction.is_active());
        assert!(
            settled_correction
                .contradicts
                .contains(&"claim".to_string()),
            "the winner must remember what it displaced"
        );
    }

    #[test]
    fn a_better_supported_claim_holds_and_the_newcomer_is_stored_dead() {
        let (store, _dir) = create_test_store();
        let user = "arb";

        let established = candidate(
            "established",
            "four crew were injured in the incident",
            5,
            5,
        );
        store
            .ingest_candidate(user, &established, Some(EMB), chrono::Utc::now())
            .unwrap();

        let stray = candidate("stray", "four crew were not injured in the incident", 1, 1);
        let outcome = store
            .ingest_candidate(user, &stray, Some(EMB), chrono::Utc::now())
            .unwrap();

        match &outcome {
            FactIngestOutcome::Rejected { loser, winner_id } => {
                assert_eq!(loser.id, "stray");
                assert_eq!(winner_id, "established");
            }
            other => panic!("expected the better-supported claim to hold, got {other:?}"),
        }
        assert!(
            outcome.newly_active_fact().is_none(),
            "a rejected newcomer must never be handed to the graph — it would \
             stay reachable by traversal despite having lost"
        );

        assert_eq!(active_count(&store, user), 1);
        assert!(store.get(user, "established").unwrap().unwrap().is_active());
        assert!(
            !store.get(user, "stray").unwrap().unwrap().is_active(),
            "the loser is retained, invalidated — visible disagreement, not silence"
        );
    }

    #[test]
    fn a_repeatedly_re_extracted_wrong_claim_does_not_flip_flop() {
        // The wrong claim usually STAYS in the corpus, so it is re-extracted on
        // every cycle forever. If each re-derivation were treated as a fresh
        // fact it would meet the correction in `find_contradiction` again and,
        // on equal support, the newcomer rule would flip the verdict — the two
        // claims trading places indefinitely.
        let (store, _dir) = create_test_store();
        let user = "flipflop";

        let claim = candidate("claim", "four crew were injured in the incident", 1, 1);
        store
            .ingest_candidate(user, &claim, Some(EMB), chrono::Utc::now())
            .unwrap();
        let correction = candidate(
            "correction",
            "four crew were not injured in the incident",
            1,
            1,
        );
        store
            .ingest_candidate(user, &correction, Some(EMB), chrono::Utc::now())
            .unwrap();

        let dead_before = store.get(user, "claim").unwrap().unwrap();

        // Five more cycles re-extract the wrong claim verbatim, each with a NEW
        // fact id, exactly as the consolidator mints them.
        for cycle in 0..5 {
            let redo = candidate(
                &format!("redo-{cycle}"),
                "four crew were injured in the incident",
                1,
                1,
            );
            let outcome = store
                .ingest_candidate(user, &redo, Some(EMB), chrono::Utc::now())
                .unwrap();
            match &outcome {
                FactIngestOutcome::MatchedSuperseded { fact_id, .. } => {
                    assert_eq!(fact_id, "claim", "must land back on its own dead row");
                }
                other => panic!("cycle {cycle}: expected the dead row to absorb it, got {other:?}"),
            }
        }

        assert_eq!(
            active_count(&store, user),
            1,
            "after six re-derivations exactly one fact is still active"
        );
        assert!(
            store.get(user, "correction").unwrap().unwrap().is_active(),
            "the correction must still be the one believed"
        );

        let dead_after = store.get(user, "claim").unwrap().unwrap();
        assert_eq!(
            dead_after.last_reinforced, dead_before.last_reinforced,
            "re-deriving a dead fact must not extend its half-life"
        );
        assert_eq!(
            dead_after.support_count, dead_before.support_count,
            "a dead fact must not accrue support"
        );
        assert_eq!(
            store.list(user, 100).unwrap().len(),
            2,
            "re-derivations must be absorbed, not stored as new rows"
        );
    }

    #[test]
    fn reinforcement_is_idempotent_when_no_new_evidence_arrives() {
        // The fact-extraction watermark is persisted as `timestamp_millis()`,
        // truncated DOWN from a nanosecond-precision `created_at`, and the next
        // cycle filters with a strict `>`. So the newest memory of every cycle
        // compares as newer than the watermark it produced and is re-processed
        // forever. That was called "merely wasteful"; it was not. The
        // reinforcement branch refreshed `last_reinforced` and applied the
        // confidence boost unconditionally, manufacturing an immortal,
        // ever-more-confident fact out of a rounding error.
        let (store, _dir) = create_test_store();
        let user = "idem";

        let first = candidate("f1", "the reactor was scrammed at 04:12", 1, 1);
        store
            .ingest_candidate(user, &first, Some(EMB), chrono::Utc::now())
            .unwrap();
        let stored = store.get(user, "f1").unwrap().unwrap();

        // Re-derivation from EXACTLY the same source memories, as a watermark
        // re-processing produces.
        for cycle in 0..4 {
            let mut redo = candidate(
                &format!("redo-{cycle}"),
                "the reactor was scrammed at 04:12",
                1,
                0,
            );
            redo.source_memories = stored.source_memories.clone();
            let later = chrono::Utc::now() + chrono::Duration::days(cycle + 1);
            let outcome = store
                .ingest_candidate(user, &redo, Some(EMB), later)
                .unwrap();
            assert!(
                matches!(outcome, FactIngestOutcome::AlreadyAttested { .. }),
                "cycle {cycle}: already-counted evidence must be inert, got {outcome:?}"
            );
        }

        let after = store.get(user, "f1").unwrap().unwrap();
        assert_eq!(
            after.confidence, stored.confidence,
            "confidence must not ratchet on evidence already counted"
        );
        assert_eq!(
            after.last_reinforced, stored.last_reinforced,
            "a re-derivation from the same sources must not hand out a fresh half-life"
        );
        assert_eq!(after.support_count, stored.support_count);
    }

    #[test]
    fn genuinely_new_evidence_still_reinforces() {
        // The guard above must not turn into "reinforcement never happens".
        let (store, _dir) = create_test_store();
        let user = "reinforce";

        let first = candidate("f1", "the reactor was scrammed at 04:12", 1, 1);
        store
            .ingest_candidate(user, &first, Some(EMB), chrono::Utc::now())
            .unwrap();
        let before = store.get(user, "f1").unwrap().unwrap();

        // Same claim, a DIFFERENT episode attesting it.
        let corroboration = candidate("f2", "the reactor was scrammed at 04:12", 1, 1);
        let outcome = store
            .ingest_candidate(user, &corroboration, Some(EMB), chrono::Utc::now())
            .unwrap();

        let (fact, confidence_before) = outcome
            .reinforced_fact()
            .expect("new source evidence must reinforce");
        assert_eq!(fact.id, "f1", "reinforcement lands on the EXISTING row");
        assert_eq!(confidence_before, before.confidence);
        assert!(
            fact.confidence > before.confidence,
            "new evidence must raise confidence"
        );
        assert_eq!(fact.support_count, before.support_count + 1);
        assert_eq!(
            store.list(user, 100).unwrap().len(),
            1,
            "corroboration merges into one engram, it does not mint a second"
        );
    }

    #[test]
    fn test_stats() {
        let (store, _dir) = create_test_store();

        store
            .store("user-1", &create_test_fact("fact-1", "Fact one"))
            .unwrap();
        store
            .store("user-1", &create_test_fact("fact-2", "Fact two"))
            .unwrap();

        let stats = store.stats("user-1").unwrap();
        assert_eq!(stats.total_facts, 2);
        assert!(stats.avg_confidence > 0.0);
    }
}
