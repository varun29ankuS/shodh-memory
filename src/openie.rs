//! Open information extraction (OpenIE) — the OPEN arm of the causal spine.
//!
//! Where GLiREL only finds relations you *name* and CATENA works event→event, this
//! lets the **text supply the predicate**: it reads each sentence's grammar
//! (subject → verb(+particle/prep) → object) off the dependency parse, so `carried`,
//! `drifted into`, `rammed`, `funded` all come out without ever being enumerated.
//! Passive voice is normalised to active (agent = head) so arrows read cause→effect.
//! Predicates are typed into families ([`crate::causal_vocab`]) — the growable
//! semantic layer, not a hand-list. Complementary to GLiREL/CATENA (measured
//! overlap ~3 of 130).
//!
//! Model-free over an already-parsed token list, so the SVO logic is unit-testable
//! without loading the parser; [`extract_triples`] wraps it with [`crate::dep_parser`].

use crate::causal_vocab::{self, Family};
use crate::dep_parser::{self, ParsedToken};

/// An open (subject, predicate, object) triple read from the grammar.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenTriple {
    /// Subject argument span (the cause/agent).
    pub subject: String,
    /// Predicate: head-verb lemma plus any particle/preposition.
    pub predicate: String,
    /// Object argument span (the effect/patient).
    pub object: String,
    /// `"active"` or `"passive"` (passive already normalised to agent→patient).
    pub voice: &'static str,
    /// Head-verb lemma (the family/relation key).
    pub head_verb: String,
    /// Coarse predicate family.
    pub family: Family,
    /// Canonical relation label (`Struck`, `Damaged`, `Causes`, `Created`, …).
    pub relation: &'static str,
    /// Whether the predicate carries causal structure.
    pub causal: bool,
    /// Abstract/social predicate that over-fires — the caller should require
    /// stronger evidence before minting a durable edge.
    pub low_precision: bool,
}

/// Children adjacency: `children[i]` = token indices whose head is `i` (excluding
/// self-roots). O(n) build, so subtree spans are cheap.
fn children_of(tokens: &[ParsedToken]) -> Vec<Vec<usize>> {
    let mut ch = vec![Vec::new(); tokens.len()];
    for t in tokens {
        if t.head != t.i && t.head < tokens.len() {
            ch[t.head].push(t.i);
        }
    }
    ch
}

/// The surface span of the subtree rooted at `root` (root + all descendants),
/// in token order. This is the full argument phrase (`the container ship`), not
/// just its head noun.
fn subtree_span(tokens: &[ParsedToken], children: &[Vec<usize>], root: usize) -> String {
    let mut idxs = Vec::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        idxs.push(n);
        for &c in &children[n] {
            stack.push(c);
        }
    }
    idxs.sort_unstable();
    idxs.iter()
        .map(|&i| tokens[i].text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn dep_is(tokens: &[ParsedToken], i: usize, dep: &str) -> bool {
    tokens[i].dep == dep
}

fn make_triple(
    subject: String,
    predicate: String,
    object: String,
    voice: &'static str,
) -> OpenTriple {
    let head_verb = predicate
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    let family = causal_vocab::family_of(&head_verb);
    OpenTriple {
        subject,
        object,
        voice,
        relation: causal_vocab::canonical_relation(&head_verb),
        causal: family.is_causal(),
        low_precision: causal_vocab::is_abstract_social(&head_verb),
        family,
        head_verb,
        predicate,
    }
}

/// Extract open triples from an already-parsed token list. Pure — no model.
pub fn triples_from_tokens(tokens: &[ParsedToken]) -> Vec<OpenTriple> {
    let children = children_of(tokens);
    let mut out = Vec::new();

    for v in tokens {
        if v.pos != "VERB" || causal_vocab::is_light_verb(&v.lemma.to_lowercase()) {
            continue;
        }
        let kids = &children[v.i];
        let particles: Vec<&str> = kids
            .iter()
            .filter(|&&c| dep_is(tokens, c, "prt"))
            .map(|&c| tokens[c].text.as_str())
            .collect();
        let base_pred = {
            let mut p = v.lemma.to_lowercase();
            for prt in &particles {
                p.push(' ');
                p.push_str(&prt.to_lowercase());
            }
            p
        };

        let nsubj: Vec<usize> = kids
            .iter()
            .copied()
            .filter(|&c| dep_is(tokens, c, "nsubj"))
            .collect();
        let nsubjpass: Vec<usize> = kids
            .iter()
            .copied()
            .filter(|&c| dep_is(tokens, c, "nsubjpass"))
            .collect();
        // Passive agent: `by <agent>` — dep "agent" whose pobj child is the agent.
        let agents: Vec<usize> = kids
            .iter()
            .copied()
            .filter(|&c| dep_is(tokens, c, "agent"))
            .flat_map(|c| {
                children[c]
                    .iter()
                    .copied()
                    .filter(|&g| dep_is(tokens, g, "pobj"))
            })
            .collect();
        let dobj: Vec<usize> = kids
            .iter()
            .copied()
            .filter(|&c| matches!(tokens[c].dep.as_str(), "dobj" | "dative" | "oprd" | "attr"))
            .collect();
        let preps: Vec<usize> = kids
            .iter()
            .copied()
            .filter(|&c| dep_is(tokens, c, "prep"))
            .collect();

        // PASSIVE: "<patient> was <verb> by <agent>" → agent --verb--> patient.
        if !nsubjpass.is_empty() && !agents.is_empty() {
            for &a in &agents {
                for &pat in &nsubjpass {
                    if a == pat {
                        continue;
                    }
                    out.push(make_triple(
                        subtree_span(tokens, &children, a),
                        base_pred.clone(),
                        subtree_span(tokens, &children, pat),
                        "passive",
                    ));
                }
            }
            continue;
        }

        // ACTIVE: subject --verb(+prep)--> object.
        for &s in &nsubj {
            let subj = subtree_span(tokens, &children, s);
            for &o in &dobj {
                if o == s {
                    continue;
                }
                out.push(make_triple(
                    subj.clone(),
                    base_pred.clone(),
                    subtree_span(tokens, &children, o),
                    "active",
                ));
            }
            for &pr in &preps {
                let pred_p = format!("{} {}", base_pred, tokens[pr].text.to_lowercase());
                for &g in &children[pr] {
                    if dep_is(tokens, g, "pobj") && g != s {
                        out.push(make_triple(
                            subj.clone(),
                            pred_p.clone(),
                            subtree_span(tokens, &children, g),
                            "active",
                        ));
                    }
                }
            }
        }
    }
    out
}

/// Extract open triples from raw text via the shared parser. `None` if the parser
/// model is unavailable.
pub fn extract_triples(text: &str) -> Option<Vec<OpenTriple>> {
    let tokens = dep_parser::parse(text)?;
    Some(triples_from_tokens(&tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk(i: usize, text: &str, head: usize, dep: &str, pos: &str, lemma: &str) -> ParsedToken {
        ParsedToken {
            i,
            text: text.to_string(),
            head,
            dep: dep.to_string(),
            pos: pos.to_string(),
            tag: pos.to_string(),
            lemma: lemma.to_string(),
        }
    }

    #[test]
    fn active_svo_with_span_arguments() {
        // "the container ship rammed the Key Bridge"
        //  0:the(det→2) 1:container(compound→2) 2:ship(nsubj→3) 3:rammed(ROOT)
        //  4:the(det→6) 5:Key(compound→6) 6:Bridge(dobj→3)
        let toks = vec![
            tk(0, "the", 2, "det", "DET", "the"),
            tk(1, "container", 2, "compound", "NOUN", "container"),
            tk(2, "ship", 3, "nsubj", "NOUN", "ship"),
            tk(3, "rammed", 3, "ROOT", "VERB", "ram"),
            tk(4, "the", 6, "det", "DET", "the"),
            tk(5, "Key", 6, "compound", "PROPN", "Key"),
            tk(6, "Bridge", 3, "dobj", "PROPN", "Bridge"),
        ];
        let tr = triples_from_tokens(&toks);
        assert_eq!(tr.len(), 1);
        let t = &tr[0];
        assert_eq!(t.subject, "the container ship");
        assert_eq!(t.object, "the Key Bridge");
        assert_eq!(t.predicate, "ram");
        assert_eq!(t.relation, "Struck"); // family-typed from the open predicate
        assert!(t.causal);
        assert_eq!(t.voice, "active");
    }

    #[test]
    fn passive_is_normalised_to_agent_first() {
        // "the bridge was destroyed by the ship" → ship --destroy--> bridge
        //  0:the 1:bridge(nsubjpass→3) 2:was(auxpass→3) 3:destroyed(ROOT)
        //  4:by(agent→3) 5:the 6:ship(pobj→4)
        let toks = vec![
            tk(0, "the", 1, "det", "DET", "the"),
            tk(1, "bridge", 3, "nsubjpass", "NOUN", "bridge"),
            tk(2, "was", 3, "auxpass", "AUX", "be"),
            tk(3, "destroyed", 3, "ROOT", "VERB", "destroy"),
            tk(4, "by", 3, "agent", "ADP", "by"),
            tk(5, "the", 6, "det", "DET", "the"),
            tk(6, "ship", 4, "pobj", "NOUN", "ship"),
        ];
        let tr = triples_from_tokens(&toks);
        assert_eq!(tr.len(), 1);
        assert_eq!(tr[0].subject, "the ship"); // agent → subject
        assert_eq!(tr[0].object, "the bridge"); // patient → object
        assert_eq!(tr[0].relation, "Damaged");
        assert_eq!(tr[0].voice, "passive");
    }

    #[test]
    fn prepositional_object_and_open_predicate() {
        // "the ship drifted into the pier" → ship --drift into--> pier (motion=causal)
        let toks = vec![
            tk(0, "the", 1, "det", "DET", "the"),
            tk(1, "ship", 2, "nsubj", "NOUN", "ship"),
            tk(2, "drifted", 2, "ROOT", "VERB", "drift"),
            tk(3, "into", 2, "prep", "ADP", "into"),
            tk(4, "the", 5, "det", "DET", "the"),
            tk(5, "pier", 3, "pobj", "NOUN", "pier"),
        ];
        let tr = triples_from_tokens(&toks);
        assert_eq!(tr.len(), 1);
        assert_eq!(tr[0].predicate, "drift into");
        assert_eq!(tr[0].object, "the pier");
        assert_eq!(tr[0].family, Family::Motion);
        assert!(tr[0].causal, "caused-motion is causal");
    }

    #[test]
    fn light_verb_makes_no_triple() {
        // "the bridge is old" → copula, no triple.
        let toks = vec![
            tk(0, "the", 1, "det", "DET", "the"),
            tk(1, "bridge", 2, "nsubj", "NOUN", "bridge"),
            tk(2, "is", 2, "ROOT", "AUX", "be"),
            tk(3, "old", 2, "acomp", "ADJ", "old"),
        ];
        assert!(triples_from_tokens(&toks).is_empty());
    }
}
