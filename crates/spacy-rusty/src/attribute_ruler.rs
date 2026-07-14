//! attribute_ruler: maps fine-grained TAG (+ LOWER/ORTH/DEP/IS_SPACE rules,
//! including Matcher operators IN/NOT_IN/REGEX) to coarse POS, MORPH, TAG.
//! Patterns apply in manifest order (later overrides earlier), matching spaCy.

use crate::lexeme::is_space;
use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashSet;

#[derive(Default, Clone)]
pub struct Anno {
    pub text: String,
    pub tag: String,
    pub pos: String,
    pub morph: String,
    pub dep: String,
    pub lemma: String, // set by LEMMA patterns (irregulars/pronouns); "" = unset
}

enum Attr {
    Tag,
    Lower,
    Orth,
    Dep,
    Pos,
}

enum Constraint {
    Eq(String),
    In(HashSet<String>),
    NotIn(HashSet<String>),
    Regex(Regex),
}

struct TokenSpec {
    // string-attr constraints
    str_c: Vec<(Attr, Constraint)>,
    is_space: Option<bool>,
    /// a key/value we couldn't interpret -> this spec can never match
    unmatchable: bool,
}

struct Pattern {
    specs: Vec<TokenSpec>,
    index: i64,
    set_tag: Option<String>,
    set_pos: Option<String>,
    set_morph: Option<String>,
    set_lemma: Option<String>,
}

pub struct AttributeRuler {
    patterns: Vec<Pattern>,
}

fn attr_of(key: &str) -> Option<Attr> {
    match key {
        "TAG" => Some(Attr::Tag),
        "LOWER" => Some(Attr::Lower),
        "ORTH" | "TEXT" => Some(Attr::Orth),
        "DEP" => Some(Attr::Dep),
        "POS" => Some(Attr::Pos),
        _ => None,
    }
}

fn set_of(v: &Value) -> HashSet<String> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default()
}

fn parse_spec(v: &Value) -> TokenSpec {
    let mut spec = TokenSpec { str_c: vec![], is_space: None, unmatchable: false };
    let obj = match v.as_object() {
        Some(o) => o,
        None => {
            spec.unmatchable = true;
            return spec;
        }
    };
    for (key, val) in obj {
        if key == "IS_SPACE" {
            match val.as_bool() {
                Some(b) => spec.is_space = Some(b),
                None => spec.unmatchable = true,
            }
            continue;
        }
        let attr = match attr_of(key) {
            Some(a) => a,
            None => {
                spec.unmatchable = true; // unknown key -> can't match safely
                continue;
            }
        };
        let constraint = if let Some(s) = val.as_str() {
            Constraint::Eq(s.to_string())
        } else if let Some(o) = val.as_object() {
            if let Some(inv) = o.get("IN") {
                Constraint::In(set_of(inv))
            } else if let Some(ni) = o.get("NOT_IN") {
                Constraint::NotIn(set_of(ni))
            } else if let Some(rx) = o.get("REGEX").and_then(|x| x.as_str()) {
                match Regex::new(rx) {
                    Ok(r) => Constraint::Regex(r),
                    Err(_) => {
                        spec.unmatchable = true;
                        continue;
                    }
                }
            } else {
                spec.unmatchable = true;
                continue;
            }
        } else {
            spec.unmatchable = true;
            continue;
        };
        spec.str_c.push((attr, constraint));
    }
    spec
}

impl AttributeRuler {
    pub fn load(man: &Value) -> AttributeRuler {
        let mut patterns = Vec::new();
        if let Some(arr) = man["attribute_ruler"]["patterns"].as_array() {
            for p in arr {
                let attrs = &p["attrs"];
                let index = p["index"].as_i64().unwrap_or(0);
                let set_tag = attrs.get("TAG").and_then(|x| x.as_str()).map(|s| s.to_string());
                let set_pos = attrs.get("POS").and_then(|x| x.as_str()).map(|s| s.to_string());
                let set_morph = attrs.get("MORPH").and_then(|x| x.as_str()).map(|s| s.to_string());
                let set_lemma = attrs.get("LEMMA").and_then(|x| x.as_str()).map(|s| s.to_string());
                // A single AR entry may hold several alternative token-patterns
                // (e.g. [[gets], [got]]) sharing the same attrs — emit one
                // internal pattern per alternative.
                let Some(alts) = p["patterns"].as_array() else { continue };
                for alt in alts {
                    let specs: Vec<TokenSpec> = alt
                        .as_array()
                        .map(|toks| toks.iter().map(parse_spec).collect())
                        .unwrap_or_default();
                    if specs.is_empty() {
                        continue;
                    }
                    patterns.push(Pattern {
                        specs,
                        index,
                        set_tag: set_tag.clone(),
                        set_pos: set_pos.clone(),
                        set_morph: set_morph.clone(),
                        set_lemma: set_lemma.clone(),
                    });
                }
            }
        }
        AttributeRuler { patterns }
    }

    fn token_val(attr: &Attr, t: &Anno) -> String {
        match attr {
            Attr::Tag => t.tag.clone(),
            Attr::Lower => t.text.to_lowercase(),
            Attr::Orth => t.text.clone(),
            Attr::Dep => t.dep.clone(),
            Attr::Pos => t.pos.clone(),
        }
    }

    fn spec_matches(spec: &TokenSpec, t: &Anno) -> bool {
        if spec.unmatchable {
            return false;
        }
        if let Some(b) = spec.is_space {
            if is_space(&t.text) != b {
                return false;
            }
        }
        for (attr, c) in &spec.str_c {
            let tv = Self::token_val(attr, t);
            let ok = match c {
                Constraint::Eq(v) => &tv == v,
                Constraint::In(set) => set.contains(&tv),
                Constraint::NotIn(set) => !set.contains(&tv),
                Constraint::Regex(r) => r.is_match(&tv).unwrap_or(false),
            };
            if !ok {
                return false;
            }
        }
        true
    }

    pub fn apply(&self, toks: &mut [Anno]) {
        let n = toks.len();
        for p in &self.patterns {
            let plen = p.specs.len();
            if plen == 0 || plen > n {
                continue;
            }
            for start in 0..=(n - plen) {
                if (0..plen).all(|k| Self::spec_matches(&p.specs[k], &toks[start + k])) {
                    let idx = if p.index >= 0 {
                        start + p.index as usize
                    } else {
                        ((start + plen) as i64 + p.index) as usize
                    };
                    if idx < n {
                        if let Some(ref v) = p.set_tag {
                            toks[idx].tag = v.clone();
                        }
                        if let Some(ref v) = p.set_pos {
                            toks[idx].pos = v.clone();
                        }
                        if let Some(ref v) = p.set_morph {
                            toks[idx].morph = if v == "_" { String::new() } else { v.clone() };
                        }
                        if let Some(ref v) = p.set_lemma {
                            toks[idx].lemma = v.clone();
                        }
                    }
                }
            }
        }
    }
}
