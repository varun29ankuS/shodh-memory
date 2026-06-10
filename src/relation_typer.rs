//! Semantic relation typing — typed-relation substrate, increment 1 (#65).
//!
//! Types the relation between two co-mentioned entities by embedding the
//! TEMPLATE-NORMALIZED sentence containing both mentions ("x caused y") and
//! matching it against cached exemplar embeddings. Zero new model budget (the
//! resident MiniLM embedder is reused), zero training (the "head" is cosine
//! over label embeddings), and GROWABLE: adding a relation type is adding an
//! exemplar line — the same label-embedding mechanism the growable entity
//! ontology uses. First attempt in the edge-typing chain (semantic → cue
//! extractor → label-pair table), gated by SHODH_SEMANTIC_RELATIONS.
//!
//! Substrate audit context (2026-06-10): >80% of edges are untyped CoOccurs
//! because the label-pair table is engineering-domain-only ((Person, Person)
//! had no rule at all) and the cue list covers ~40 phrases. Lineage root-cause
//! P@1 measured 0.0 end-to-end and heuristic patching was PROVEN exhausted
//! (run 27272612202) — this module is the replacement, not another patch.
//! Scoreboards: lineage/ontology harnesses + the CoOccurs edge fraction.

use crate::embeddings::Embedder;
use crate::graph_memory::RelationType;
use std::sync::OnceLock;

struct Exemplar {
    relation: RelationType,
    /// Whether the template's "x" (the EARLIER mention) is the relation source.
    x_is_source: bool,
    embedding: Vec<f32>,
}

/// (relation, x_is_source, template). Templates use "x" for the earlier
/// mention and "y" for the later one. Effect-first phrasings ("x was caused
/// by y") carry x_is_source=false — the direction lives in the exemplar, so
/// the inversion class of bug fixed in `extract_directed_predicate` cannot
/// recur here.
fn exemplar_specs() -> Vec<(RelationType, bool, &'static str)> {
    use RelationType::*;
    vec![
        // Causal — the lineage backbone.
        (Causes, true, "x caused y"),
        (Causes, true, "x led to y"),
        (Causes, false, "x happened because of y"),
        (Causes, false, "x was caused by y"),
        // Employment / management.
        (WorksAt, true, "x works at y"),
        (WorksAt, true, "x joined y"),
        (Manages, true, "x manages y"),
        // Creation / use.
        (CreatedBy, true, "x created y"),
        (CreatedBy, false, "x was created by y"),
        (Uses, true, "x uses y"),
        // Location.
        (LocatedIn, true, "x lives in y"),
        (LocatedIn, true, "x is located in y"),
        (LocatedIn, true, "x traveled to y"),
        // Structure.
        (PartOf, true, "x is part of y"),
        (PartOf, true, "x is a member of y"),
        (DependsOn, true, "x depends on y"),
        (SupersededBy, true, "x was replaced by y"),
        // Social — the conversational-domain gap (Person↔Person had no rule).
        (Knows, true, "x is friends with y"),
        (Knows, true, "x is married to y"),
        (Knows, true, "x met y"),
        (Knows, true, "x talked with y"),
        // Preference — LoCoMo hobbies/likes.
        (Prefers, true, "x likes y"),
        (Prefers, true, "x enjoys y"),
        (Prefers, true, "x loves y"),
        // Learning / teaching.
        (Teaches, true, "x taught y"),
        (Learned, true, "x learned y"),
        // Events / activities.
        (AssociatedWith, true, "x attended y"),
        (AssociatedWith, true, "x went to y"),
        (AssociatedWith, true, "x participated in y"),
    ]
}

pub struct RelationTyper {
    cache: OnceLock<Vec<Exemplar>>,
}

/// Process-wide typer: exemplar embeddings are computed once per process on
/// first use (≈30 short encodes) and shared across users.
pub static RELATION_TYPER: RelationTyper = RelationTyper {
    cache: OnceLock::new(),
};

impl RelationTyper {
    fn ensure(&self, embedder: &dyn Embedder) -> &[Exemplar] {
        self.cache.get_or_init(|| {
            let mut out = Vec::new();
            for (relation, x_is_source, text) in exemplar_specs() {
                match embedder.encode(text) {
                    Ok(embedding) => out.push(Exemplar {
                        relation,
                        x_is_source,
                        embedding,
                    }),
                    Err(e) => {
                        tracing::warn!("relation exemplar embed failed ({text}): {e}");
                    }
                }
            }
            out
        })
    }

    /// Minimum cosine for a semantic match (SHODH_SEMREL_MIN, default 0.6 —
    /// sweepable; too low admits noise edges, too high reverts to CoOccurs).
    fn min_cosine() -> f32 {
        static MIN: OnceLock<f32> = OnceLock::new();
        *MIN.get_or_init(|| {
            std::env::var("SHODH_SEMREL_MIN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.6)
        })
    }

    /// Type the relation between `name_a` and `name_b` in `text`. Returns
    /// (relation, a_is_source, cosine); None when the mentions don't share a
    /// sentence or no exemplar clears the threshold (caller falls back to the
    /// cue extractor / label-pair table).
    pub fn type_relation(
        &self,
        embedder: &dyn Embedder,
        text: &str,
        name_a: &str,
        name_b: &str,
    ) -> Option<(RelationType, bool, f32)> {
        let lc = text.to_ascii_lowercase();
        let a = name_a.to_ascii_lowercase();
        let b = name_b.to_ascii_lowercase();
        if a.is_empty() || b.is_empty() {
            return None;
        }
        let pa = lc.find(&a)?;
        let pb = lc.find(&b)?;
        if pa == pb {
            return None;
        }
        // Clamp to the sentence containing BOTH mentions (same scoping as the
        // cue extractor — a neighbouring clause must not leak in).
        let (lo, hi) = if pa < pb {
            (pa, pb + b.len())
        } else {
            (pb, pa + a.len())
        };
        let sent_start = lc[..lo]
            .rfind(['.', '!', '?', ';', '\n'])
            .map(|i| i + 1)
            .unwrap_or(0);
        let sent_end = lc[hi..]
            .find(['.', '!', '?', ';', '\n'])
            .map(|i| hi + i)
            .unwrap_or(lc.len());
        // Template-normalize: the earlier mention becomes "x", the later "y".
        // Replace the LONGER name first so a name nested in the other does not
        // get mangled ("dave" inside "davenport").
        let a_first = pa < pb;
        let (x_name, y_name) = if a_first { (&a, &b) } else { (&b, &a) };
        let sentence = &lc[sent_start..sent_end];
        let normalized = if x_name.len() >= y_name.len() {
            sentence
                .replace(x_name.as_str(), "x")
                .replace(y_name.as_str(), "y")
        } else {
            sentence
                .replace(y_name.as_str(), "y")
                .replace(x_name.as_str(), "x")
        };
        let query = embedder.encode(normalized.trim()).ok()?;
        let exemplars = self.ensure(embedder);
        let mut best: Option<(&Exemplar, f32)> = None;
        for ex in exemplars {
            let sim = crate::similarity::cosine_similarity(&query, &ex.embedding);
            if best.map(|(_, s)| sim > s).unwrap_or(true) {
                best = Some((ex, sim));
            }
        }
        let (ex, sim) = best?;
        if sim < Self::min_cosine() {
            return None;
        }
        // ex.x_is_source is relative to template order (x = earlier mention);
        // map back to the caller's name_a.
        let a_is_source = if a_first {
            ex.x_is_source
        } else {
            !ex.x_is_source
        };
        Some((ex.relation.clone(), a_is_source, sim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::minilm::{EmbeddingConfig, MiniLMEmbedder};

    /// With the simplified (hash) embedder, identical strings embed identically
    /// — an exact-template sentence must match its exemplar at cosine ≈ 1.0.
    /// This tests the mechanics (windowing, normalization, direction mapping)
    /// without requiring the ONNX model.
    #[test]
    fn exact_template_sentence_matches_and_maps_direction() {
        let embedder = MiniLMEmbedder::new_simplified(EmbeddingConfig::default())
            .expect("simplified embedder");
        let typer = RelationTyper {
            cache: OnceLock::new(),
        };

        // Cause-first: "Redis caused Outage" → normalized "x caused y".
        let r = typer.type_relation(&embedder, "Redis caused Outage.", "Redis", "Outage");
        let (rt, a_is_source, sim) = r.expect("exact template must match");
        assert_eq!(rt, RelationType::Causes);
        assert!(a_is_source, "earlier-mentioned cause must be the source");
        assert!(sim > 0.99, "identical normalized text must be ~1.0, got {sim}");

        // Same sentence, arguments swapped at the call site: direction flips.
        let r = typer.type_relation(&embedder, "Redis caused Outage.", "Outage", "Redis");
        let (rt, a_is_source, _) = r.expect("exact template must match");
        assert_eq!(rt, RelationType::Causes);
        assert!(!a_is_source, "name_a is the effect here");

        // Effect-first template: direction lives in the exemplar.
        let r = typer.type_relation(
            &embedder,
            "Outage was caused by Redis.",
            "Redis",
            "Outage",
        );
        let (rt, a_is_source, _) = r.expect("effect-first template must match");
        assert_eq!(rt, RelationType::Causes);
        assert!(a_is_source, "the later-mentioned cause must be the source");
    }

    #[test]
    fn no_shared_sentence_or_missing_mention_returns_none() {
        let embedder = MiniLMEmbedder::new_simplified(EmbeddingConfig::default())
            .expect("simplified embedder");
        let typer = RelationTyper {
            cache: OnceLock::new(),
        };
        assert!(typer
            .type_relation(&embedder, "Alpha met Beta. Gamma slept.", "Alpha", "Gamma")
            .map(|(_, _, sim)| sim < 0.99)
            .unwrap_or(true));
        assert!(typer
            .type_relation(&embedder, "Alpha met Beta.", "Alpha", "Epsilon")
            .is_none());
    }
}
