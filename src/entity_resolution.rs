//! Head-block entity resolver (entity-resolution roadmap, Phase 1.2).
//!
//! The knowledge graph's entity nodes are surface **mentions**, not entities:
//! `Dali`, `the Dali`, `container ship`, `cargo ship`, `vessel` are five nodes
//! for one ship; `Key Bridge`, `Francis Scott Key Bridge`, `the bridge` are three
//! for one bridge. This collapses mentions into canonical entities,
//! deterministically and LLM-free — the FROZEN floor beneath the learned matcher
//! (Phase 2) and self-improving layers (Phase 4). Nothing here learns at runtime.
//!
//! Algorithm (precision-first; validated in Python on the GDELT bridge graph,
//! 669 mentions → ~386 entities with no over-merge disasters):
//!
//! 1. **Parse** each mention with the dependency parser ([`crate::dep_parser`]):
//!    take its syntactic head lemma. `Port of Baltimore` → `port`; `Francis Scott
//!    Key Bridge` → `bridge`. A verb-headed fragment (`ship crashed`) descends to
//!    its subject noun (`ship`); if it is still verb-headed it is an action/event,
//!    routed OUT to the event layer, not treated as an entity.
//! 2. **Block by head lemma.** Mentions with different heads never merge — this is
//!    what stops the union-find chaining that sank the rule-based v1.
//! 3. **Union-find within a block**, merging a pair only when types overlap AND one
//!    of: modifiers are compatible (subset/bare), OR they share a *rare* modifier
//!    (document-frequency ≤ [`RARE_DF`]), OR they share a *discriminative* causal
//!    relation held by ≤ [`MAX_CAUSAL_SHARE`] entities (the Bhattacharya–Getoor
//!    lever: `container ship` ~ `cargo ship` because both *struck the bridge*).
//! 4. **Canonical** representative = the most proper / most-mentioned / longest
//!    surface form in the cluster.

use std::collections::{HashMap, HashSet};

use crate::dep_parser::{self, ParsedToken};

/// Modifiers rarer than this document-frequency count as a strong alias signal.
pub const RARE_DF: usize = 3;
/// A causal relation held by at most this many entities is discriminative enough
/// to license a merge (a relation everyone has carries no identity information).
pub const MAX_CAUSAL_SHARE: usize = 4;

/// Determiners stripped from the clean surface form and never treated as heads.
const DET: &[&str] = &[
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "its",
    "their",
    "'s",
    "\u{2019}s",
];
/// Head lemmas that are never entities (calendrical / generic scaffolding).
const JUNK_HEAD: &[&str] = &[
    "tuesday", "monday", "morning", "night", "day", "week", "time", "hour", "thing", "area", "way",
    "number", "people", "one", "early", "today",
];
/// Function words excluded from a mention's modifier set (DET ∪ prepositions/conj).
const STOP_EXTRA: &[&str] = &["of", "for", "in", "on", "at", "to", "and", "&"];
/// Causal relation labels whose fingerprints license cross-modifier merges.
const CAUSAL_REL: &[&str] = &[
    "Triggers",
    "Causes",
    "Struck",
    "Damaged",
    "Killed",
    "Closed",
    "Disrupted",
    "Halted",
    "Blocked",
    "Suspended",
];

fn is_det(w: &str) -> bool {
    DET.contains(&w)
}
fn is_stop(w: &str) -> bool {
    DET.contains(&w) || STOP_EXTRA.contains(&w)
}
fn is_junk_head(w: &str) -> bool {
    JUNK_HEAD.contains(&w)
}
fn is_causal(label: &str) -> bool {
    CAUSAL_REL.contains(&label)
}

/// A raw entity mention node from the graph.
#[derive(Debug, Clone)]
pub struct Mention {
    pub id: String,
    pub label: String,
    pub types: HashSet<String>,
    pub proper: bool,
    pub mention_count: u32,
}

/// A directed causal edge between two mention ids.
#[derive(Debug, Clone)]
pub struct CausalRel {
    pub source: String,
    pub target: String,
    pub label: String,
}

/// The parse of one mention: what the merge rule actually keys on. Public so the
/// parse logic is testable without re-running the resolver.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedMention {
    /// Determiner-stripped surface form, e.g. `Port of Baltimore`.
    pub clean: String,
    /// Head lemma (lowercased), e.g. `port`.
    pub head: String,
    /// Coarse POS of the head after subject-descent (`NOUN`/`PROPN`/`VERB`/...).
    pub head_pos: String,
    /// Content modifiers (lemmas) other than the head.
    pub mods: HashSet<String>,
}

impl ParsedMention {
    /// A mention is a valid entity iff it has a non-junk noun/proper-noun head.
    /// Verb-headed or junk-headed mentions are not entities.
    pub fn is_entity(&self) -> bool {
        !self.head.is_empty()
            && !is_junk_head(&self.head)
            && (self.head_pos == "NOUN" || self.head_pos == "PROPN")
    }
    /// A mention whose head is still a verb after subject-descent is an
    /// action/event (`collapsed`, `rescued`), routed to the event layer.
    pub fn is_event(&self) -> bool {
        self.head_pos == "VERB" || self.head_pos == "AUX"
    }
}

/// Outcome of resolving a set of mentions.
#[derive(Debug, Clone)]
pub struct Resolution {
    /// mention id → canonical surface label.
    pub canon: HashMap<String, String>,
    /// Clusters of mention ids, each an entity; largest first.
    pub clusters: Vec<Vec<String>>,
    /// Mention ids routed to the event layer (verb-headed).
    pub events: Vec<String>,
    /// Mention ids dropped as junk (junk head, empty, or unparseable).
    pub dropped: Vec<String>,
    pub num_input: usize,
}

impl Resolution {
    /// Number of canonical entities (clusters).
    pub fn num_entities(&self) -> usize {
        self.clusters.len()
    }
    /// Fraction by which the node count shrank: 1 − entities/input.
    pub fn reduction(&self) -> f32 {
        if self.num_input == 0 {
            return 0.0;
        }
        1.0 - (self.num_entities() as f32) / (self.num_input as f32)
    }
}

// ---------------------------------------------------------------------------
// Parsing (per-mention) — pure over parsed tokens, so it is testable model-free.
// ---------------------------------------------------------------------------

fn is_alpha_or_upper(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let all_alpha = text.chars().all(char::is_alphabetic);
    let is_upper = text.chars().any(char::is_alphabetic)
        && text
            .chars()
            .filter(|c| c.is_alphabetic())
            .all(char::is_uppercase);
    all_alpha || is_upper
}

/// Compute a [`ParsedMention`] from an already-parsed token list. Pure: the head
/// selection, verb→subject descent, and modifier extraction are all here, so
/// tests can exercise them by hand-building tokens with no model loaded.
pub fn parse_mention_tokens(tokens: &[ParsedToken]) -> Option<ParsedMention> {
    if tokens.is_empty() {
        return None;
    }
    // Candidate content tokens (spaCy `is_alpha or isupper`); the fallback head.
    let content: Vec<&ParsedToken> = tokens
        .iter()
        .filter(|t| is_alpha_or_upper(&t.text))
        .collect();
    if content.is_empty() {
        return None;
    }

    // Syntactic root: the token whose head is itself; else the last content token.
    let mut root: &ParsedToken = tokens
        .iter()
        .find(|t| t.is_root())
        .unwrap_or_else(|| content[content.len() - 1]);

    // Subject+verb fragment ("ship crashed"): the entity is the verb's SUBJECT
    // noun, not the verb. Descend to the first nominal subject/object child.
    if root.pos == "VERB" || root.pos == "AUX" {
        if let Some(subj) = tokens
            .iter()
            .filter(|c| c.head == root.i && c.i != root.i)
            .filter(|c| matches!(c.dep.as_str(), "nsubj" | "nsubjpass" | "dobj"))
            .find(|c| c.pos == "NOUN" || c.pos == "PROPN")
        {
            root = subj;
        }
    }

    let head = root.lemma.to_lowercase();
    let head_pos = root.pos.clone();

    // Modifiers: content tokens other than the head, not function words, that are
    // nominal/adjectival/numeric. Filter by POS on the token, THEN reduce to lemma
    // (mapping first would decouple the lemma from its token's POS).
    let mods: HashSet<String> = content
        .iter()
        .filter(|t| matches!(t.pos.as_str(), "NOUN" | "PROPN" | "ADJ" | "NUM" | "X"))
        .map(|t| t.lemma.to_lowercase())
        .filter(|l| *l != head && !is_stop(l))
        .collect();

    let clean = tokens
        .iter()
        .filter(|t| !is_det(&t.text.to_lowercase()))
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    Some(ParsedMention {
        clean,
        head,
        head_pos,
        mods,
    })
}

// ---------------------------------------------------------------------------
// Clustering (pure over parsed mentions) — model-free, unit-testable.
// ---------------------------------------------------------------------------

/// A parsed mention plus the identity metadata the canonical picker needs.
#[derive(Debug, Clone)]
pub struct Resolvable {
    pub id: String,
    pub orig: String,
    pub parsed: ParsedMention,
    pub types: HashSet<String>,
    pub proper: bool,
    pub mention_count: u32,
}

struct DisjointSet {
    parent: HashMap<String, String>,
}
impl DisjointSet {
    fn new<'a>(ids: impl Iterator<Item = &'a String>) -> Self {
        DisjointSet {
            parent: ids.map(|i| (i.clone(), i.clone())).collect(),
        }
    }
    fn find(&mut self, x: &str) -> String {
        let mut cur = x.to_string();
        while self.parent[&cur] != cur {
            let grand = self.parent[&self.parent[&cur]].clone();
            self.parent.insert(cur.clone(), grand.clone());
            cur = grand;
        }
        cur
    }
    fn union(&mut self, a: &str, b: &str) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent.insert(rb, ra);
        }
    }
}

fn mods_compatible(a: &HashSet<String>, b: &HashSet<String>) -> bool {
    a.is_empty() || b.is_empty() || a.is_subset(b) || b.is_subset(a)
}

fn rare_shared_mod(
    a: &HashSet<String>,
    b: &HashSet<String>,
    mod_df: &HashMap<String, usize>,
) -> bool {
    a.iter()
        .any(|m| b.contains(m) && mod_df.get(m).copied().unwrap_or(0) <= RARE_DF)
}

/// Discriminative causal fingerprint of an entity: `(label, direction, other-head)`
/// for each causal edge it participates in.
type Fingerprint = (String, &'static str, String);

/// Cluster already-parsed, already-validated entities. Pure and deterministic:
/// no model, no env, no I/O — the whole merge rule is exercised here.
pub fn cluster(entities: &[Resolvable], causal: &[CausalRel]) -> Resolution {
    let ids: Vec<String> = entities.iter().map(|e| e.id.clone()).collect();
    let by_id: HashMap<&str, &Resolvable> = entities.iter().map(|e| (e.id.as_str(), e)).collect();

    // Modifier document frequency across the valid set.
    let mut mod_df: HashMap<String, usize> = HashMap::new();
    for e in entities {
        for m in &e.parsed.mods {
            *mod_df.entry(m.clone()).or_insert(0) += 1;
        }
    }

    // Causal fingerprints, restricted to edges between valid entities.
    let mut fp: HashMap<String, HashSet<Fingerprint>> = HashMap::new();
    for rel in causal {
        if !is_causal(&rel.label) {
            continue;
        }
        let (Some(s), Some(t)) = (
            by_id.get(rel.source.as_str()),
            by_id.get(rel.target.as_str()),
        ) else {
            continue;
        };
        fp.entry(rel.source.clone()).or_default().insert((
            rel.label.clone(),
            "out",
            t.parsed.head.clone(),
        ));
        fp.entry(rel.target.clone()).or_default().insert((
            rel.label.clone(),
            "in",
            s.parsed.head.clone(),
        ));
    }
    let mut fp_holders: HashMap<Fingerprint, usize> = HashMap::new();
    for set in fp.values() {
        for f in set {
            *fp_holders.entry(f.clone()).or_insert(0) += 1;
        }
    }
    let empty_fp: HashSet<Fingerprint> = HashSet::new();
    let causal_shared = |a: &str, b: &str| -> bool {
        let fa = fp.get(a).unwrap_or(&empty_fp);
        let fb = fp.get(b).unwrap_or(&empty_fp);
        fa.intersection(fb)
            .any(|f| fp_holders.get(f).copied().unwrap_or(0) <= MAX_CAUSAL_SHARE)
    };

    // Block by head lemma; union-find within a block only.
    let mut by_head: HashMap<&str, Vec<&Resolvable>> = HashMap::new();
    for e in entities {
        by_head.entry(e.parsed.head.as_str()).or_default().push(e);
    }
    let mut ds = DisjointSet::new(ids.iter());
    for grp in by_head.values() {
        for x in 0..grp.len() {
            for y in (x + 1)..grp.len() {
                let (a, b) = (grp[x], grp[y]);
                if a.types.is_disjoint(&b.types) {
                    continue;
                }
                let compatible = mods_compatible(&a.parsed.mods, &b.parsed.mods)
                    || rare_shared_mod(&a.parsed.mods, &b.parsed.mods, &mod_df)
                    || causal_shared(&a.id, &b.id);
                if compatible {
                    ds.union(&a.id, &b.id);
                }
            }
        }
    }

    // Gather clusters.
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for id in &ids {
        let root = ds.find(id);
        groups.entry(root).or_default().push(id.clone());
    }
    let mut clusters: Vec<Vec<String>> = groups.into_values().collect();

    // Canonical label per cluster = most proper / most-mentioned / longest form.
    let mut canon: HashMap<String, String> = HashMap::new();
    for members in &mut clusters {
        members.sort(); // deterministic member order
        let rep = members
            .iter()
            .map(|id| by_id[id.as_str()])
            .max_by(|a, b| {
                (
                    a.proper,
                    a.mention_count,
                    a.orig.split_whitespace().count(),
                    a.orig.len(),
                )
                    .cmp(&(
                        b.proper,
                        b.mention_count,
                        b.orig.split_whitespace().count(),
                        b.orig.len(),
                    ))
            })
            .unwrap();
        for id in members.iter() {
            canon.insert(id.clone(), rep.orig.clone());
        }
    }
    // Largest clusters first; break ties on canonical label for determinism.
    clusters.sort_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then_with(|| canon.get(&a[0]).cmp(&canon.get(&b[0])))
    });

    Resolution {
        num_input: entities.len(),
        canon,
        clusters,
        events: Vec::new(),
        dropped: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Top-level resolve — needs the parser (model); returns None if unavailable.
// ---------------------------------------------------------------------------

/// Resolve a set of raw mentions into canonical entities. Requires the dependency
/// parser (model configured via `SHODH_SPACY_MODEL_PATH`); returns `None` when it
/// is unavailable so callers fall back to unresolved mentions.
pub fn resolve(mentions: &[Mention], causal: &[CausalRel]) -> Option<Resolution> {
    if !dep_parser::is_available() {
        return None;
    }
    let num_input = mentions.len();
    let mut entities: Vec<Resolvable> = Vec::new();
    let mut events: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();

    for m in mentions {
        let parsed = match dep_parser::parse(&m.label).and_then(|t| parse_mention_tokens(&t)) {
            Some(p) => p,
            None => {
                dropped.push(m.id.clone());
                continue;
            }
        };
        if parsed.is_entity() {
            entities.push(Resolvable {
                id: m.id.clone(),
                orig: m.label.clone(),
                parsed,
                types: m.types.clone(),
                proper: m.proper,
                mention_count: m.mention_count,
            });
        } else if parsed.is_event() {
            events.push(m.id.clone());
        } else {
            dropped.push(m.id.clone());
        }
    }

    let mut resolution = cluster(&entities, causal);
    resolution.num_input = num_input;
    resolution.events = events;
    resolution.dropped = dropped;
    Some(resolution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(i: usize, text: &str, head: usize, dep: &str, pos: &str, lemma: &str) -> ParsedToken {
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

    fn ent(
        id: &str,
        orig: &str,
        head: &str,
        mods: &[&str],
        types: &[&str],
        proper: bool,
        mc: u32,
    ) -> Resolvable {
        Resolvable {
            id: id.to_string(),
            orig: orig.to_string(),
            parsed: ParsedMention {
                clean: orig.to_string(),
                head: head.to_string(),
                head_pos: "PROPN".to_string(),
                mods: mods.iter().map(|s| s.to_string()).collect(),
            },
            types: types.iter().map(|s| s.to_string()).collect(),
            proper,
            mention_count: mc,
        }
    }

    #[test]
    fn parse_descends_verb_to_subject() {
        // "ship crashed": crashed=root VERB, ship=nsubj NOUN → head becomes ship.
        let toks = vec![
            tok(0, "ship", 1, "nsubj", "NOUN", "ship"),
            tok(1, "crashed", 1, "ROOT", "VERB", "crash"),
        ];
        let p = parse_mention_tokens(&toks).unwrap();
        assert_eq!(p.head, "ship");
        assert!(p.is_entity());
        assert!(!p.is_event());
    }

    #[test]
    fn parse_pure_verb_is_event_not_entity() {
        // "collapsed": lone verb root, no nominal subject → stays an event.
        let toks = vec![tok(0, "collapsed", 0, "ROOT", "VERB", "collapse")];
        let p = parse_mention_tokens(&toks).unwrap();
        assert_eq!(p.head, "collapse");
        assert!(p.is_event());
        assert!(!p.is_entity());
    }

    #[test]
    fn parse_prepositional_head_is_not_the_object() {
        // "Port of Baltimore": Port=root, of=prep, Baltimore=pobj → head "port".
        let toks = vec![
            tok(0, "Port", 0, "ROOT", "PROPN", "Port"),
            tok(1, "of", 0, "prep", "ADP", "of"),
            tok(2, "Baltimore", 1, "pobj", "PROPN", "Baltimore"),
        ];
        let p = parse_mention_tokens(&toks).unwrap();
        assert_eq!(p.head, "port");
        assert!(p.mods.contains("baltimore"));
    }

    #[test]
    fn junk_head_is_not_an_entity() {
        let toks = vec![tok(0, "Tuesday", 0, "ROOT", "PROPN", "Tuesday")];
        let p = parse_mention_tokens(&toks).unwrap();
        assert!(!p.is_entity(), "calendrical head must not be an entity");
    }

    #[test]
    fn merge_same_head_bare_and_modified() {
        // "Dali" (bare) merges with "container Dali"; distinct-head "Bridge" does not.
        let ents = vec![
            ent("d1", "Dali", "dali", &[], &["Vessel"], true, 40),
            ent(
                "d2",
                "container Dali",
                "dali",
                &["container"],
                &["Vessel"],
                false,
                3,
            ),
            ent(
                "b1",
                "Key Bridge",
                "bridge",
                &["key"],
                &["Structure"],
                true,
                50,
            ),
        ];
        let res = cluster(&ents, &[]);
        assert_eq!(res.num_entities(), 2, "two heads → two entities");
        // Dali cluster has 2 members, canonical is the proper "Dali".
        let dali_cluster = res
            .clusters
            .iter()
            .find(|c| c.contains(&"d1".to_string()))
            .unwrap();
        assert_eq!(dali_cluster.len(), 2);
        assert_eq!(res.canon["d2"], "Dali");
    }

    #[test]
    fn no_merge_when_types_disjoint() {
        // Same head, incompatible types → must NOT merge (over-merge guard).
        let ents = vec![
            ent("a", "Jordan", "jordan", &[], &["Person"], true, 5),
            ent("b", "Jordan", "jordan", &[], &["Location"], true, 5),
        ];
        let res = cluster(&ents, &[]);
        assert_eq!(res.num_entities(), 2, "disjoint types must not merge");
    }

    #[test]
    fn causal_shares_license_cross_modifier_merge() {
        // "container ship" and "cargo ship": same head, DIFFERENT incompatible
        // modifiers, but both Struck the same bridge (rare fingerprint) → merge.
        let ents = vec![
            ent(
                "s1",
                "container ship",
                "ship",
                &["container"],
                &["Vessel"],
                false,
                4,
            ),
            ent(
                "s2",
                "cargo ship",
                "ship",
                &["cargo"],
                &["Vessel"],
                false,
                4,
            ),
        ];
        let causal = vec![
            CausalRel {
                source: "s1".into(),
                target: "brg".into(),
                label: "Struck".into(),
            },
            CausalRel {
                source: "s2".into(),
                target: "brg".into(),
                label: "Struck".into(),
            },
        ];
        // Need "brg" present as a valid entity for the fingerprint's other-head.
        let mut with_bridge = ents.clone();
        with_bridge.push(ent(
            "brg",
            "the bridge",
            "bridge",
            &[],
            &["Structure"],
            false,
            10,
        ));
        let res = cluster(&with_bridge, &causal);
        let ship_cluster = res
            .clusters
            .iter()
            .find(|c| c.contains(&"s1".to_string()))
            .unwrap();
        assert!(
            ship_cluster.contains(&"s2".to_string()),
            "shared discriminative causal relation should merge distinct modifiers"
        );
    }

    #[test]
    fn common_causal_relation_does_not_merge() {
        // A relation held by MANY entities is not discriminative → no merge on it.
        let mut ents = vec![
            ent(
                "s1",
                "container ship",
                "ship",
                &["container"],
                &["Vessel"],
                false,
                4,
            ),
            ent(
                "s2",
                "cargo ship",
                "ship",
                &["cargo"],
                &["Vessel"],
                false,
                4,
            ),
        ];
        // Five holders all Trigger the same target head → fingerprint over-shared.
        let mut causal = Vec::new();
        for (k, id) in ["s1", "s2", "x1", "x2", "x3"].iter().enumerate() {
            ents.push(ent(
                &format!("x{k}"),
                "x",
                "ship",
                &["other"],
                &["Vessel"],
                false,
                1,
            ));
            causal.push(CausalRel {
                source: (*id).into(),
                target: "t".into(),
                label: "Triggers".into(),
            });
        }
        ents.push(ent("t", "target", "target", &[], &["Event"], false, 1));
        let res = cluster(&ents, &causal);
        let s1_cluster = res
            .clusters
            .iter()
            .find(|c| c.contains(&"s1".to_string()))
            .unwrap();
        assert!(
            !s1_cluster.contains(&"s2".to_string()),
            "an over-shared (non-discriminative) causal relation must not merge"
        );
    }
}
