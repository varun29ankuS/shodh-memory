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

/// A clause-level relation: the salient content head of the cause/source clause
/// and of the effect/target clause, with the relation type. `head_a` is the
/// source (cause). Both heads are lowercased surface strings to be resolved
/// against graph nodes at the call site.
#[derive(Debug, Clone, PartialEq)]
pub struct ClauseTriple {
    pub head_a: String,
    pub head_b: String,
    pub relation: RelationType,
}

/// Causal signals and whether they are EFFECT-FIRST (the effect clause precedes
/// the signal, the cause follows it). Cause-first signals are the opposite
/// ("cause SIGNAL effect"). This is the CATENA direction fix without a parse: the
/// signal's class, not entity order, sets cause→effect.
const CAUSAL_SIGNALS: &[(&str, bool)] = &[
    // cause-first: "<cause> SIGNAL <effect>"
    ("led to", false),
    ("leads to", false),
    ("leading to", false),
    ("resulted in", false),
    ("results in", false),
    ("resulting in", false),
    ("gave rise to", false),
    ("brought about", false),
    ("so that", false),
    ("which caused", false),
    ("which triggered", false),
    // effect-first: "<effect> SIGNAL <cause>"
    ("because of", true),
    ("because", true),
    ("due to", true),
    ("owing to", true),
    ("thanks to", true),
    ("as a result of", true),
    ("caused by", true),
    ("triggered by", true),
    ("stemmed from", true),
    ("stems from", true),
];

/// Extract clause-level CAUSAL relations from text. For each sentence, find the
/// earliest causal signal, split into the two clauses around it, take the content
/// noun nearest the signal on each side as the clause head, and emit a directed
/// `Triggers` edge cause→effect (the signal class sets direction). This captures
/// the clause/event-level causation the entity-pair typer structurally misses (it
/// needs BOTH entities + a cue in one sentence; the census showed ~17.8 causal cues
/// per edge it catches). POS tagger only — no parser. High precision: fires only on
/// an explicit signal with a content noun on both sides.
pub fn extract_clause_triples(text: &str) -> Vec<ClauseTriple> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| matches!(c, '.' | '!' | '?' | '\n' | ';')) {
        let sentence = raw.trim();
        if sentence.len() < 8 {
            continue;
        }
        let lc = sentence.to_lowercase();
        // Earliest signal wins; on a positional tie the longer (more specific) one.
        let mut best: Option<(usize, usize, bool)> = None;
        for &(sig, effect_first) in CAUSAL_SIGNALS {
            if let Some(pos) = lc.find(sig) {
                let better = match best {
                    None => true,
                    Some((bp, bl, _)) => pos < bp || (pos == bp && sig.len() > bl),
                };
                if better {
                    best = Some((pos, sig.len(), effect_first));
                }
            }
        }
        let Some((pos, sig_len, effect_first)) = best else {
            continue;
        };
        // Slice the ORIGINAL sentence (preserves casing for proper-noun tagging);
        // `.get` guards against a non-char-boundary on non-ASCII text.
        let (Some(left), Some(right)) = (sentence.get(..pos), sentence.get(pos + sig_len..)) else {
            continue;
        };
        let left_head = last_noun(left);
        let right_head = first_noun(right);
        let (Some(lh), Some(rh)) = (left_head, right_head) else {
            continue;
        };
        if lh.eq_ignore_ascii_case(&rh) {
            continue;
        }
        // cause-first: left is the cause (source). effect-first: right is the cause.
        let (head_a, head_b) = if effect_first { (rh, lh) } else { (lh, rh) };
        out.push(ClauseTriple {
            head_a,
            head_b,
            relation: RelationType::Triggers,
        });
    }
    out
}

/// First content noun (≥3 chars) in a clause, lowercased — the head nearest a
/// following signal.
fn first_noun(clause: &str) -> Option<String> {
    let ext = extract_chunks(clause);
    for chunk in &ext.chunks {
        for w in &chunk.words {
            if matches!(w.pos, PosTag::Noun | PosTag::ProperNoun) && w.text.len() >= 3 {
                return Some(w.text.to_lowercase());
            }
        }
    }
    None
}

/// Last content noun (≥3 chars) in a clause, lowercased — the head nearest a
/// preceding signal.
fn last_noun(clause: &str) -> Option<String> {
    let ext = extract_chunks(clause);
    let mut last = None;
    for chunk in &ext.chunks {
        for w in &chunk.words {
            if matches!(w.pos, PosTag::Noun | PosTag::ProperNoun) && w.text.len() >= 3 {
                last = Some(w.text.to_lowercase());
            }
        }
    }
    last
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

    #[test]
    fn extracts_clause_causation() {
        // An explicit causal signal yields one directed Triggers triple between
        // two distinct heads. (POS tagging is imperfect, so we assert structure +
        // direction-class, not the exact head tokens.)
        let t = extract_clause_triples("The team missed the deadline because of the storm.");
        assert!(!t.is_empty(), "should extract a causal clause triple");
        assert_eq!(t[0].relation, RelationType::Triggers);
        assert_ne!(t[0].head_a, t[0].head_b);
        // No signal → no triple.
        assert!(extract_clause_triples("The team is large and the office is bright.").is_empty());
    }
}
