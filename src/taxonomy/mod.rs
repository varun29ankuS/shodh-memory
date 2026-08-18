//! Offline noun taxonomy: surface form → the categories it belongs to.
//!
//! [`crate::kb`] answers *"are these two names the same thing?"*. This module
//! answers the orthogonal question *"what kind of thing is this?"*, from the
//! same premise and with the same discipline: a vendored asset, no network, no
//! model, and an explicit refusal to guess.
//!
//! # Why the graph cannot derive this itself
//!
//! Every extractor in the pipeline — OpenIE, CATENA, GLiNER — recovers structure
//! **between entities the text mentions**. None of them can produce a fact the
//! corpus never states. Measured on the LoCoMo gate corpus, that is exactly what
//! the hardest queries need:
//!
//! ```text
//! query : "What animal do both Nate and Joanna like?"
//! gold  : "Nate: I'm drawn to turtles. They're unique ..."
//! ```
//!
//! `animal` occurs in 12 of 629 documents and in none of the gold ones;
//! `turtles` occurs in 20 and is what the answer actually says. The bridge
//! `turtle IsA animal` appears nowhere in the conversation because no two people
//! discussing pets ever bother to say it. Without a taxonomy the query term and
//! the answer term are simply unrelated tokens, and no amount of extraction
//! quality changes that.
//!
//! # Honest scope
//!
//! On that corpus this layer bridges **one** gold pair that stemming does not
//! already reach (76 pairs total; 15 are recovered by stemming, 34 remain
//! reachable only by dense semantics). It was built knowing that number. It
//! earns its place on vocabulary-rich corpora and on queries that ask by
//! category — not as a fix for the LoCoMo multi-hop wall, which is dominated by
//! pure paraphrase.
//!
//! # Sense selection
//!
//! Ambiguity is the whole difficulty, and the two obvious rules both fail.
//! Walking *every* sense produces junk (`share` reaches `way`; `work` reaches
//! `check`). Taking WordNet's *first* sense is worse for the motivating case:
//! `turtle`'s first noun sense is a turtleneck sweater, so the one bridge this
//! module exists to serve would resolve to `garment`.
//!
//! The asset is therefore built by preferring **the synset whose own name is the
//! lemma** — `turtle.n.02` is *named* turtle, whereas `turtleneck.n.01` merely
//! lists it as an alias — falling back to sense 1 when no synset is so named.
//! See `scripts/build_taxonomy.py`. Ancestors so general that they relate
//! everything to everything (`entity`, `abstraction`, `object`, …) are dropped
//! at build time rather than filtered here.

use std::collections::HashMap;
use std::sync::OnceLock;

/// `lemma\tancestor1|ancestor2|...`, nearest ancestor first.
///
/// `include_str!`d, so it lives in the binary's read-only data rather than
/// being read from disk at startup — the same arrangement as
/// [`crate::gazetteer`] and [`crate::kb`].
const HYPERNYMS_TSV: &str = include_str!("hypernyms.tsv");

static INDEX: OnceLock<HashMap<&'static str, Vec<&'static str>>> = OnceLock::new();

fn index() -> &'static HashMap<&'static str, Vec<&'static str>> {
    INDEX.get_or_init(|| {
        let mut map = HashMap::new();
        for line in HYPERNYMS_TSV.lines() {
            let mut parts = line.split('\t');
            let (Some(lemma), Some(ancestors)) = (parts.next(), parts.next()) else {
                continue;
            };
            if lemma.is_empty() || ancestors.is_empty() {
                continue;
            }
            map.insert(lemma, ancestors.split('|').collect());
        }
        map
    })
}

/// Strip a regular English plural so `turtles` reaches the `turtle` row.
///
/// Deliberately *not* the Porter stemmer used elsewhere in the crate: a stemmer
/// maps `turtles` to `turtl`, which is not a WordNet lemma and matches nothing.
/// What is needed here is lemmatisation, and for the plural case the regular
/// rules cover the overwhelming majority. Irregulars (`children`, `mice`) are
/// not handled — they simply miss, which costs a lookup rather than producing a
/// wrong category. Missing an edge is recoverable; asserting a false one is the
/// error that propagates into every traversal downstream.
fn singularise(word: &str) -> Option<String> {
    let w = word;
    if let Some(stem) = w.strip_suffix("ies") {
        if stem.len() >= 2 {
            return Some(format!("{stem}y"));
        }
    }
    for suffix in ["ses", "xes", "zes", "ches", "shes"] {
        if let Some(stem) = w.strip_suffix(suffix) {
            if stem.len() >= 2 {
                return Some(format!("{stem}{}", &suffix[..suffix.len() - 2]));
            }
        }
    }
    if let Some(stem) = w.strip_suffix('s') {
        if stem.len() >= 3 && !stem.ends_with('s') {
            return Some(stem.to_string());
        }
    }
    None
}

/// Categories `surface` belongs to, nearest first, or empty when unknown.
///
/// Case-insensitive; tries the surface as given, then its singular form.
/// Returns `Vec` rather than a slice because the singular lookup borrows from a
/// temporary; the vectors are short (a handful of ancestors) and the call sites
/// are ingest-time, not per-query-per-candidate.
pub fn hypernyms(surface: &str) -> Vec<&'static str> {
    let lower = surface.to_lowercase();
    let idx = index();
    if let Some(found) = idx.get(lower.as_str()) {
        return found.clone();
    }
    if let Some(singular) = singularise(&lower) {
        if let Some(found) = idx.get(singular.as_str()) {
            return found.clone();
        }
    }
    Vec::new()
}

/// Whether `surface` is a kind of `category`.
///
/// The direction matters and is easy to invert: this asks whether *`category`*
/// is an ancestor of *`surface`* — `is_a("turtle", "animal")` is true and
/// `is_a("animal", "turtle")` is false.
pub fn is_a(surface: &str, category: &str) -> bool {
    let category = category.to_lowercase();
    hypernyms(surface).iter().any(|a| *a == category)
}

/// Number of lemmas the vendored table covers. Exposed for diagnostics and to
/// let a test assert the asset was actually embedded.
pub fn lemma_count() -> usize {
    index().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_is_embedded_and_non_trivial() {
        assert!(
            lemma_count() > 10_000,
            "vendored hypernym table looks empty or truncated: {} lemmas",
            lemma_count()
        );
    }

    #[test]
    fn the_motivating_bridge_resolves() {
        // conv-42_q56 in the LoCoMo gate: "What animal do both Nate and Joanna
        // like?" against a gold that says "I'm drawn to turtles". This is the
        // one gold pair this layer bridges that stemming does not.
        assert!(
            is_a("turtles", "animal"),
            "turtles must reach animal; got {:?}",
            hypernyms("turtles")
        );
        assert!(is_a("turtle", "animal"));
    }

    #[test]
    fn prefers_the_sense_named_by_the_lemma() {
        // WordNet orders `turtle` with the turtleneck-sweater sense first. If
        // the build script ever regresses to sense-1 selection this fails, and
        // the module silently becomes useless for its motivating case.
        let anc = hypernyms("turtle");
        assert!(
            anc.contains(&"reptile"),
            "expected the reptile sense, got {anc:?}"
        );
        assert!(
            !anc.contains(&"garment"),
            "resolved to the turtleneck-sweater sense: {anc:?}"
        );
    }

    #[test]
    fn does_not_invent_the_spurious_bridges_a_naive_walk_produces() {
        // An all-senses closure links `share` to `way` and `work` to `check`,
        // both of which produced false bridges in the gate corpus.
        assert!(!is_a("share", "way"), "{:?}", hypernyms("share"));
        assert!(!is_a("work", "check"), "{:?}", hypernyms("work"));
    }

    #[test]
    fn direction_is_not_symmetric() {
        assert!(is_a("guitar", "musical_instrument"));
        assert!(!is_a("musical_instrument", "guitar"));
    }

    #[test]
    fn abstract_roots_are_not_emitted() {
        for lemma in ["turtle", "guitar", "screenplay"] {
            for root in ["entity", "abstraction", "physical_entity", "object"] {
                assert!(
                    !is_a(lemma, root),
                    "{lemma} should not carry the useless root {root}"
                );
            }
        }
    }

    #[test]
    fn unknown_surfaces_return_empty_rather_than_guessing() {
        assert!(hypernyms("zzzznotaword").is_empty());
        assert!(hypernyms("").is_empty());
    }

    #[test]
    fn singularise_handles_regular_plurals_only() {
        assert_eq!(singularise("turtles").as_deref(), Some("turtle"));
        assert_eq!(singularise("stories").as_deref(), Some("story"));
        assert_eq!(singularise("boxes").as_deref(), Some("box"));
        // Irregulars are deliberately unhandled — a miss, never a wrong answer.
        assert_eq!(singularise("mice"), None);
        // Too short to be a plural stem.
        assert_eq!(singularise("as"), None);
    }
}
