//! Rule-based matching: a user-facing `Matcher` (token patterns with the
//! IN/NOT_IN/REGEX operators, numeric LENGTH comparisons, and the ?/*/+
//! quantifiers), a `PhraseMatcher` (exact attr-sequence matching), and an
//! `EntityRuler` (patterns -> non-overlapping entity spans). Operates over the
//! per-token attributes the pipeline already produces (see `Pipeline::match_tokens`).

use fancy_regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Per-token attributes the matcher reads.
#[derive(Clone)]
pub struct MatchToken {
    pub orth: String,
    pub lower: String,
    pub norm: String,
    pub shape: String,
    pub lemma: String,
    pub pos: String,
    pub tag: String,
    pub dep: String,
    pub ent_type: String,
    pub is_alpha: bool,
    pub is_digit: bool,
    pub is_punct: bool,
    pub is_space: bool,
    pub is_lower: bool,
    pub is_upper: bool,
    pub is_title: bool,
    pub like_num: bool,
    pub length: i64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Attr {
    Orth,
    Lower,
    Norm,
    Shape,
    Lemma,
    Pos,
    Tag,
    Dep,
    EntType,
    Length,
    IsAlpha,
    IsDigit,
    IsPunct,
    IsSpace,
    IsLower,
    IsUpper,
    IsTitle,
    LikeNum,
}

fn attr_of(key: &str) -> Option<Attr> {
    Some(match key {
        "ORTH" | "TEXT" => Attr::Orth,
        "LOWER" => Attr::Lower,
        "NORM" => Attr::Norm,
        "SHAPE" => Attr::Shape,
        "LEMMA" => Attr::Lemma,
        "POS" => Attr::Pos,
        "TAG" => Attr::Tag,
        "DEP" => Attr::Dep,
        "ENT_TYPE" => Attr::EntType,
        "LENGTH" => Attr::Length,
        "IS_ALPHA" => Attr::IsAlpha,
        "IS_DIGIT" => Attr::IsDigit,
        "IS_PUNCT" => Attr::IsPunct,
        "IS_SPACE" => Attr::IsSpace,
        "IS_LOWER" => Attr::IsLower,
        "IS_UPPER" => Attr::IsUpper,
        "IS_TITLE" => Attr::IsTitle,
        "LIKE_NUM" => Attr::LikeNum,
        _ => return None,
    })
}

fn str_attr<'a>(a: Attr, t: &'a MatchToken) -> Option<&'a str> {
    Some(match a {
        Attr::Orth => &t.orth,
        Attr::Lower => &t.lower,
        Attr::Norm => &t.norm,
        Attr::Shape => &t.shape,
        Attr::Lemma => &t.lemma,
        Attr::Pos => &t.pos,
        Attr::Tag => &t.tag,
        Attr::Dep => &t.dep,
        Attr::EntType => &t.ent_type,
        _ => return None,
    })
}

fn bool_attr(a: Attr, t: &MatchToken) -> Option<bool> {
    Some(match a {
        Attr::IsAlpha => t.is_alpha,
        Attr::IsDigit => t.is_digit,
        Attr::IsPunct => t.is_punct,
        Attr::IsSpace => t.is_space,
        Attr::IsLower => t.is_lower,
        Attr::IsUpper => t.is_upper,
        Attr::IsTitle => t.is_title,
        Attr::LikeNum => t.like_num,
        _ => return None,
    })
}

enum Cons {
    StrEq(String),
    StrIn(HashSet<String>),
    StrNotIn(HashSet<String>),
    Regex(Regex),
    Bool(bool),
    NumEq(i64),
    NumNe(i64),
    NumGe(i64),
    NumLe(i64),
    NumGt(i64),
    NumLt(i64),
    NumIn(HashSet<i64>),
    NumNotIn(HashSet<i64>),
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    One,
    Opt,  // ?
    Star, // *
    Plus, // +
}

struct TokenPat {
    cons: Vec<(Attr, Cons)>,
    op: Op,
    unmatchable: bool,
}

fn str_set(v: &Value) -> HashSet<String> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default()
}
fn num_set(v: &Value) -> HashSet<i64> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default()
}

fn parse_token(v: &Value) -> TokenPat {
    let mut tp = TokenPat { cons: vec![], op: Op::One, unmatchable: false };
    let Some(obj) = v.as_object() else {
        tp.unmatchable = true;
        return tp;
    };
    if obj.is_empty() {
        return tp; // {} matches any single token
    }
    for (key, val) in obj {
        if key == "OP" {
            tp.op = match val.as_str() {
                Some("?") => Op::Opt,
                Some("*") => Op::Star,
                Some("+") => Op::Plus,
                Some("!") => {
                    tp.unmatchable = true; // negation quantifier not supported
                    Op::One
                }
                _ => Op::One,
            };
            continue;
        }
        let Some(attr) = attr_of(key) else {
            tp.unmatchable = true;
            continue;
        };
        let is_bool = matches!(
            attr,
            Attr::IsAlpha | Attr::IsDigit | Attr::IsPunct | Attr::IsSpace
                | Attr::IsLower | Attr::IsUpper | Attr::IsTitle | Attr::LikeNum
        );
        let is_num = attr == Attr::Length;
        let cons = if is_bool {
            match val.as_bool() {
                Some(b) => Cons::Bool(b),
                None => {
                    tp.unmatchable = true;
                    continue;
                }
            }
        } else if is_num {
            if let Some(n) = val.as_i64() {
                Cons::NumEq(n)
            } else if let Some(o) = val.as_object() {
                match parse_num_op(o) {
                    Some(c) => c,
                    None => {
                        tp.unmatchable = true;
                        continue;
                    }
                }
            } else {
                tp.unmatchable = true;
                continue;
            }
        } else if let Some(s) = val.as_str() {
            Cons::StrEq(s.to_string())
        } else if let Some(o) = val.as_object() {
            if let Some(iv) = o.get("IN") {
                Cons::StrIn(str_set(iv))
            } else if let Some(nv) = o.get("NOT_IN") {
                Cons::StrNotIn(str_set(nv))
            } else if let Some(rx) = o.get("REGEX").and_then(|x| x.as_str()) {
                match Regex::new(rx) {
                    Ok(r) => Cons::Regex(r),
                    Err(_) => {
                        tp.unmatchable = true;
                        continue;
                    }
                }
            } else {
                tp.unmatchable = true;
                continue;
            }
        } else {
            tp.unmatchable = true;
            continue;
        };
        tp.cons.push((attr, cons));
    }
    tp
}

fn parse_num_op(o: &serde_json::Map<String, Value>) -> Option<Cons> {
    if let Some(iv) = o.get("IN") {
        return Some(Cons::NumIn(num_set(iv)));
    }
    if let Some(nv) = o.get("NOT_IN") {
        return Some(Cons::NumNotIn(num_set(nv)));
    }
    for (k, v) in o {
        let n = v.as_i64()?;
        return Some(match k.as_str() {
            "==" => Cons::NumEq(n),
            "!=" => Cons::NumNe(n),
            ">=" => Cons::NumGe(n),
            "<=" => Cons::NumLe(n),
            ">" => Cons::NumGt(n),
            "<" => Cons::NumLt(n),
            _ => return None,
        });
    }
    None
}

fn cons_matches(attr: Attr, c: &Cons, t: &MatchToken) -> bool {
    match c {
        Cons::Bool(b) => bool_attr(attr, t) == Some(*b),
        Cons::StrEq(s) => str_attr(attr, t) == Some(s.as_str()),
        Cons::StrIn(set) => str_attr(attr, t).is_some_and(|v| set.contains(v)),
        Cons::StrNotIn(set) => str_attr(attr, t).is_some_and(|v| !set.contains(v)),
        Cons::Regex(r) => str_attr(attr, t).is_some_and(|v| r.is_match(v).unwrap_or(false)),
        Cons::NumEq(n) => num_val(attr, t) == Some(*n),
        Cons::NumNe(n) => num_val(attr, t).is_some_and(|v| v != *n),
        Cons::NumGe(n) => num_val(attr, t).is_some_and(|v| v >= *n),
        Cons::NumLe(n) => num_val(attr, t).is_some_and(|v| v <= *n),
        Cons::NumGt(n) => num_val(attr, t).is_some_and(|v| v > *n),
        Cons::NumLt(n) => num_val(attr, t).is_some_and(|v| v < *n),
        Cons::NumIn(set) => num_val(attr, t).is_some_and(|v| set.contains(&v)),
        Cons::NumNotIn(set) => num_val(attr, t).is_some_and(|v| !set.contains(&v)),
    }
}

fn num_val(attr: Attr, t: &MatchToken) -> Option<i64> {
    match attr {
        Attr::Length => Some(t.length),
        _ => None,
    }
}

fn token_matches(tp: &TokenPat, t: &MatchToken) -> bool {
    if tp.unmatchable {
        return false;
    }
    tp.cons.iter().all(|(a, c)| cons_matches(*a, c, t))
}

/// A `Matcher`: named token-pattern rules. Returns all (key, start, end)
/// matches, like spaCy's default (non-greedy) Matcher.
pub struct Matcher {
    rules: Vec<(String, Vec<TokenPat>)>,
}

impl Default for Matcher {
    fn default() -> Self {
        Matcher::new()
    }
}

impl Matcher {
    pub fn new() -> Matcher {
        Matcher { rules: vec![] }
    }

    /// Add a rule under `key`. `pattern` is a JSON array of token dicts.
    pub fn add(&mut self, key: &str, pattern: &Value) {
        let pats: Vec<TokenPat> =
            pattern.as_array().map(|a| a.iter().map(parse_token).collect()).unwrap_or_default();
        if !pats.is_empty() {
            self.rules.push((key.to_string(), pats));
        }
    }

    /// All matches as (key, start_token, end_token_exclusive), sorted by
    /// (start, end, key).
    pub fn find(&self, toks: &[MatchToken]) -> Vec<(String, usize, usize)> {
        let mut out: Vec<(String, usize, usize)> = vec![];
        for (key, pats) in &self.rules {
            for start in 0..=toks.len() {
                for end in match_ends(pats, toks, start) {
                    if end > start {
                        out.push((key.clone(), start, end));
                    }
                }
            }
        }
        out.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
        out.dedup();
        out
    }
}

/// All end positions where `pats` fully matches starting at `start` (NFA over
/// the ?/*/+ quantifiers; `visited` keeps it linear and avoids blow-up).
fn match_ends(pats: &[TokenPat], toks: &[MatchToken], start: usize) -> Vec<usize> {
    let mut ends: HashSet<usize> = HashSet::new();
    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut stack = vec![(0usize, start)];
    while let Some((k, pos)) = stack.pop() {
        if !visited.insert((k, pos)) {
            continue;
        }
        if k == pats.len() {
            ends.insert(pos);
            continue;
        }
        let p = &pats[k];
        let m = pos < toks.len() && token_matches(p, &toks[pos]);
        match p.op {
            Op::One => {
                if m {
                    stack.push((k + 1, pos + 1));
                }
            }
            Op::Opt => {
                stack.push((k + 1, pos)); // zero
                if m {
                    stack.push((k + 1, pos + 1)); // one
                }
            }
            Op::Star => {
                stack.push((k + 1, pos)); // zero
                if m {
                    stack.push((k, pos + 1)); // one+, stay
                }
            }
            Op::Plus => {
                if m {
                    stack.push((k + 1, pos + 1)); // last one
                    stack.push((k, pos + 1)); // more
                }
            }
        }
    }
    ends.into_iter().collect()
}

/// A `PhraseMatcher`: exact contiguous match of an attribute sequence.
pub struct PhraseMatcher {
    attr: Attr,
    rules: Vec<(String, Vec<String>)>, // (key, attr-value sequence)
}

impl PhraseMatcher {
    /// `attr` is "ORTH"/"LOWER"/"NORM"/"LEMMA"/... (defaults to ORTH if unknown).
    pub fn new(attr: &str) -> PhraseMatcher {
        PhraseMatcher { attr: attr_of(attr).unwrap_or(Attr::Orth), rules: vec![] }
    }

    /// Add a phrase as the attribute-value sequence of its tokens (caller
    /// produces this via `Pipeline::phrase_key`).
    pub fn add(&mut self, key: &str, seq: Vec<String>) {
        if !seq.is_empty() {
            self.rules.push((key.to_string(), seq));
        }
    }

    pub fn find(&self, toks: &[MatchToken]) -> Vec<(String, usize, usize)> {
        let mut out = vec![];
        let vals: Vec<&str> = toks.iter().map(|t| str_attr(self.attr, t).unwrap_or("")).collect();
        for (key, seq) in &self.rules {
            let m = seq.len();
            if m == 0 || m > vals.len() {
                continue;
            }
            for start in 0..=vals.len() - m {
                if (0..m).all(|i| vals[start + i] == seq[i]) {
                    out.push((key.clone(), start, start + m));
                }
            }
        }
        out.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
        out.dedup();
        out
    }
}

/// spaCy `util.filter_spans`: keep longest spans first (ties: leftmost), drop
/// any that overlap an already-kept span, then return sorted by start.
pub fn filter_spans(mut spans: Vec<(String, usize, usize)>) -> Vec<(String, usize, usize)> {
    spans.sort_by(|a, b| {
        let la = a.2 - a.1;
        let lb = b.2 - b.1;
        lb.cmp(&la).then(a.1.cmp(&b.1)) // length desc, then start asc
    });
    let mut taken: HashSet<usize> = HashSet::new();
    let mut kept: Vec<(String, usize, usize)> = vec![];
    for s in spans {
        if (s.1..s.2).any(|i| taken.contains(&i)) {
            continue;
        }
        for i in s.1..s.2 {
            taken.insert(i);
        }
        kept.push(s);
    }
    kept.sort_by(|a, b| a.1.cmp(&b.1));
    kept
}

/// An `EntityRuler`: token-pattern and phrase rules labelled with entity types;
/// produces non-overlapping spans via `filter_spans`.
pub struct EntityRuler {
    matcher: Matcher,
    phrase: HashMap<Attr, PhraseMatcher>,
}

impl Default for EntityRuler {
    fn default() -> Self {
        EntityRuler::new()
    }
}

impl EntityRuler {
    pub fn new() -> EntityRuler {
        EntityRuler { matcher: Matcher::new(), phrase: HashMap::new() }
    }

    pub fn add_token_pattern(&mut self, label: &str, pattern: &Value) {
        self.matcher.add(label, pattern);
    }

    pub fn add_phrase(&mut self, label: &str, attr: &str, seq: Vec<String>) {
        let a = attr_of(attr).unwrap_or(Attr::Orth);
        self.phrase.entry(a).or_insert_with(|| PhraseMatcher::new(attr)).add(label, seq);
    }

    /// Entity spans (label, start, end), filtered to non-overlapping.
    pub fn find(&self, toks: &[MatchToken]) -> Vec<(String, usize, usize)> {
        let mut all = self.matcher.find(toks);
        for pm in self.phrase.values() {
            all.extend(pm.find(toks));
        }
        filter_spans(all)
    }
}
