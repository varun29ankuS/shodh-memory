//! OpenIE-lite relation extraction over the POS tagger.
//!
//! Increment 1 of the Stanford OpenIE substrate (the trained biaffine dependency
//! parser is the eventual quality upgrade; `encode_tokens` is its foundation).
//! Where [`crate::graph_memory::predicate_from_cues`] only matches ~30 hardcoded
//! substrings, this types an entity pair by the relational VERB that connects
//! them — found via the existing POS tagger ([`extract_chunks`]) — so far more
//! sentence-level relations become TYPED edges instead of bare `CoOccurs`,
//! densifying the substrate the typed-graph companion re-rank is bottlenecked on.
//!
//! HIGH PRECISION BY DESIGN: only unambiguously relational verb forms map. A
//! mistyped edge is faithfully amplified by the `typed_only` graph companion path
//! (memory/mod.rs) and by the lineage walk, so this errs toward `None` rather than
//! guessing. Gated behind `SHODH_OPENIE` at the call site (default off).

use crate::graph_memory::RelationType;
use crate::memory::query_parser::{extract_chunks, PosTag};

/// Map a lowercased surface verb form to a typed relation, or `None` when the
/// verb is not unambiguously relational. Matches inflected surface forms directly
/// (not Porter stems) so the mapping is explicit and predictable. Ambiguous verbs
/// that flip meaning with a particle ("led the team" vs "led to X", "built on")
/// are deliberately excluded — the substring cue path handles the particle forms,
/// and this only fires when that path has already missed.
pub fn verb_to_relation(verb: &str) -> Option<RelationType> {
    use RelationType::*;
    Some(match verb {
        // management / leadership
        "manages" | "manage" | "managed" | "managing" | "oversees" | "oversee" | "oversaw"
        | "supervises" | "supervise" | "supervised" | "directs" | "directed" => Manages,
        // authorship / creation
        "created" | "creates" | "create" | "creating" | "built" | "builds" | "building"
        | "develops" | "developed" | "develop" | "designed" | "designs" | "design" | "authored"
        | "founded" | "established" | "launched" => CreatedBy,
        // tooling / usage
        "uses" | "used" | "using" | "utilizes" | "utilized" | "leverages" | "leveraged" => Uses,
        // dependency
        "depends" | "depend" | "depended" | "relies" | "relied" | "requires" | "required"
        | "require" | "needs" | "needed" => DependsOn,
        // causation (the substring cue path catches particle forms like "led to";
        // these are the bare causative verbs)
        "caused" | "causes" | "cause" | "triggered" | "triggers" | "trigger" | "resulted"
        | "results" | "produced" | "produces" | "generated" | "generates" | "sparked"
        | "prompted" => Triggers,
        // succession / replacement
        "replaced" | "replaces" | "replace" | "superseded" | "supersedes" | "deprecated"
        | "obsoleted" => SupersededBy,
        // employment
        "joined" | "joins" | "join" => WorksAt,
        _ => return None,
    })
}

/// Type the relation between two entities from the text BETWEEN their mentions:
/// POS-tag the span and return the relation of the first POS-confirmed relational
/// verb. The caller supplies the already-isolated between-span (subject side →
/// object side by surface order). Returns `None` when the span is empty, too long
/// (the mentions are too far apart to be one reliable clause), or contains no
/// mapped verb.
pub fn relation_from_between(between: &str) -> Option<RelationType> {
    let trimmed = between.trim();
    // A reliable single-clause relation keeps the two mentions close. Beyond this
    // the span spreads across clauses and the connecting verb is unreliable.
    if trimmed.is_empty() || trimmed.len() > 160 {
        return None;
    }
    let extraction = extract_chunks(between);
    for chunk in &extraction.chunks {
        for word in &chunk.words {
            if word.pos == PosTag::Verb {
                if let Some(rt) = verb_to_relation(&word.text.to_lowercase()) {
                    return Some(rt);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_relational_verbs() {
        assert_eq!(verb_to_relation("manages"), Some(RelationType::Manages));
        assert_eq!(verb_to_relation("created"), Some(RelationType::CreatedBy));
        assert_eq!(verb_to_relation("uses"), Some(RelationType::Uses));
        assert_eq!(verb_to_relation("requires"), Some(RelationType::DependsOn));
        assert_eq!(verb_to_relation("triggered"), Some(RelationType::Triggers));
    }

    #[test]
    fn rejects_non_relational_and_ambiguous() {
        assert_eq!(verb_to_relation("is"), None);
        assert_eq!(verb_to_relation("said"), None);
        assert_eq!(verb_to_relation("led"), None); // ambiguous: "led the team" vs "led to"
        assert_eq!(verb_to_relation(""), None);
    }

    #[test]
    fn between_span_guards() {
        assert_eq!(relation_from_between(""), None);
        assert_eq!(relation_from_between("   "), None);
        // too long → unreliable cross-clause span
        let long = " and then a lot of unrelated filler ".repeat(6);
        assert_eq!(relation_from_between(&long), None);
    }
}
