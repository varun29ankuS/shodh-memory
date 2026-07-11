//! Appositive & definite-description alias extraction — the LLM-free "free-label
//! engine" (ER Plan Task 3.1).
//!
//! Parser-driven: "Apple, the iPhone maker", "the Dali, a container ship",
//! "Alphabet, Google's parent" each contain an APPOSITIVE (dep=`appos`) whose noun
//! phrase co-refers with its syntactic head. Emitting `(appositive-surface →
//! head-surface)` alias pairs teaches the alias table synonymy that no string or
//! embedding matcher reaches ("iPhone maker" = Apple) — with **no KB and no LLM**.
//! Model-free over an already-parsed token list ([`crate::dep_parser`]).

use crate::dep_parser::{self, ParsedToken};

/// A coreferent pair discovered from an appositive: `alias` (the appositive/
/// description phrase) names the same entity as `canonical` (its anchor).
#[derive(Debug, Clone, PartialEq)]
pub struct AliasPair {
    /// The appositive/description surface, e.g. `iPhone maker`.
    pub alias: String,
    /// The anchor surface it co-refers with, e.g. `Apple`.
    pub canonical: String,
}

/// Determiner words stripped from the front of a noun phrase so the alias surface
/// matches how the entity is stored (`the Dali` → `Dali`, `a container ship` →
/// `container ship`).
const LEADING_DET: &[&str] = &["the", "a", "an", "its", "their", "his", "her", "our", "this", "that"];

/// Dependency labels that grow a noun phrase. Traversal follows ONLY these, so the
/// NP of an anchor never swallows its own appositive/relative-clause/conjunct
/// branch — "Apple" stays "Apple", not "Apple the iPhone maker".
const NP_DEPS: &[&str] = &[
    "det", "predet", "compound", "amod", "poss", "case", "nummod", "nmod", "punct",
    "flat", "prt",
];

/// Indices of the noun phrase headed by `root`: the subtree reached following only
/// nominal-modifier edges ([`NP_DEPS`]), root included.
fn np_indices(tokens: &[ParsedToken], root: usize) -> Vec<usize> {
    let mut out = vec![root];
    let mut changed = true;
    while changed {
        changed = false;
        for (i, t) in tokens.iter().enumerate() {
            if i != t.head
                && !out.contains(&i)
                && out.contains(&t.head)
                && NP_DEPS.contains(&t.dep.to_lowercase().as_str())
            {
                out.push(i);
                changed = true;
            }
        }
    }
    out.sort_unstable();
    out
}

/// Strip a leading determiner and surrounding punctuation/whitespace.
fn clean_np(s: &str) -> String {
    let s = s.trim().trim_matches(|c: char| c == ',' || c == ';' || c == '.').trim();
    let mut words: Vec<&str> = s.split_whitespace().collect();
    if let Some(first) = words.first() {
        if LEADING_DET.contains(&first.to_lowercase().as_str()) {
            words.remove(0);
        }
    }
    words.join(" ")
}

/// Surface form of the noun phrase headed by `root`: its dependency subtree taken
/// as a contiguous span, determiner-stripped. Guarded so a mis-parse can't make an
/// appositive swallow half the sentence.
fn np_surface(tokens: &[ParsedToken], root: usize) -> String {
    let sub = np_indices(tokens, root);
    let (lo, hi) = match (sub.iter().min(), sub.iter().max()) {
        (Some(&lo), Some(&hi)) => (lo, hi),
        _ => return String::new(),
    };
    // A noun phrase is tight; a span this wide is a parse artifact — fall back to
    // the head token alone rather than emit a sentence-long "alias".
    if hi - lo > 8 {
        return clean_np(&tokens[root].text);
    }
    let mut words = Vec::new();
    for tok in tokens.iter().take(hi + 1).skip(lo) {
        let tx = tok.text.as_str();
        if matches!(tx, "," | ";" | "." | ":" | "(" | ")") {
            continue;
        }
        words.push(tx);
    }
    clean_np(&words.join(" "))
}

/// A surface is a usable alias iff it has content and isn't a bare pronoun/stopword.
fn is_valid(np: &str) -> bool {
    let n = np.trim();
    if n.len() < 3 {
        return false;
    }
    !matches!(
        n.to_lowercase().as_str(),
        "it" | "he" | "she" | "they" | "them" | "one" | "who" | "which" | "that"
    )
}

/// Extract coreferent alias pairs from every appositive in a parsed sentence/text.
/// De-duplicated by `(alias, canonical)` lowercased. Pure — no model.
pub fn extract_appositive_aliases(tokens: &[ParsedToken]) -> Vec<AliasPair> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (i, t) in tokens.iter().enumerate() {
        if !t.dep.eq_ignore_ascii_case("appos") || t.head >= tokens.len() {
            continue;
        }
        let alias = np_surface(tokens, i);
        let canonical = np_surface(tokens, t.head);
        if !is_valid(&alias)
            || !is_valid(&canonical)
            || alias.eq_ignore_ascii_case(&canonical)
        {
            continue;
        }
        let key = (alias.to_lowercase(), canonical.to_lowercase());
        if seen.insert(key) {
            out.push(AliasPair { alias, canonical });
        }
    }
    out
}

/// Extract appositive aliases from raw text via the shared parser. `None` when the
/// parser model is unavailable.
pub fn extract_from_text(text: &str) -> Option<Vec<AliasPair>> {
    let tokens = dep_parser::parse(text)?;
    Some(extract_appositive_aliases(&tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk(i: usize, text: &str, head: usize, dep: &str, pos: &str) -> ParsedToken {
        ParsedToken {
            i,
            text: text.to_string(),
            head,
            dep: dep.to_string(),
            pos: pos.to_string(),
            tag: pos.to_string(),
            lemma: text.to_lowercase(),
        }
    }

    #[test]
    fn definite_description_becomes_alias() {
        // "Apple , the iPhone maker" → alias "iPhone maker" ≡ "Apple"
        // maker(4) is appos of Apple(0); "the iPhone maker" is maker's subtree.
        let toks = vec![
            tk(0, "Apple", 0, "ROOT", "PROPN"),
            tk(1, ",", 0, "punct", "PUNCT"),
            tk(2, "the", 4, "det", "DET"),
            tk(3, "iPhone", 4, "compound", "PROPN"),
            tk(4, "maker", 0, "appos", "NOUN"),
        ];
        let pairs = extract_appositive_aliases(&toks);
        assert!(
            pairs.iter().any(|p| p.alias == "iPhone maker" && p.canonical == "Apple"),
            "got {pairs:?}"
        );
    }

    #[test]
    fn indefinite_appositive_strips_determiner() {
        // "the Dali , a container ship" → alias "container ship" ≡ "Dali"
        let toks = vec![
            tk(0, "the", 1, "det", "DET"),
            tk(1, "Dali", 1, "ROOT", "PROPN"),
            tk(2, ",", 1, "punct", "PUNCT"),
            tk(3, "a", 5, "det", "DET"),
            tk(4, "container", 5, "compound", "NOUN"),
            tk(5, "ship", 1, "appos", "NOUN"),
        ];
        let pairs = extract_appositive_aliases(&toks);
        assert!(
            pairs
                .iter()
                .any(|p| p.alias == "container ship" && p.canonical == "Dali"),
            "got {pairs:?}"
        );
    }

    #[test]
    fn no_appositive_no_alias() {
        // "the ship sank" — no appos, no aliases.
        let toks = vec![
            tk(0, "the", 1, "det", "DET"),
            tk(1, "ship", 2, "nsubj", "NOUN"),
            tk(2, "sank", 2, "ROOT", "VERB"),
        ];
        assert!(extract_appositive_aliases(&toks).is_empty());
    }

    #[test]
    fn pronoun_appositive_rejected() {
        // Junk guard: a pronoun appositive is not a usable alias.
        let toks = vec![
            tk(0, "Apple", 0, "ROOT", "PROPN"),
            tk(1, ",", 0, "punct", "PUNCT"),
            tk(2, "it", 0, "appos", "PRON"),
        ];
        assert!(extract_appositive_aliases(&toks).is_empty());
    }
}
