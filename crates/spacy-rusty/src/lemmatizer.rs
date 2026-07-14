//! Rule-based lemmatizer (spaCy mode="rule"): per-POS suffix rules + word
//! exceptions + a valid-lemma index, with an English `is_base_form`
//! short-circuit over the morphological features. Faithful port of
//! spacy/pipeline/lemmatizer.py::rule_lemmatize and
//! spacy/lang/en/lemmatizer.py::is_base_form. Returns the first candidate lemma
//! (spaCy assigns token.lemma_ = lemmatize(token)[0]).

use crate::model::Bundle;
use std::collections::{HashMap, HashSet};

struct PosTables {
    rules: Vec<(String, String)>,        // (old_suffix, new_suffix)
    exc: HashMap<String, Vec<String>>,   // surface (lowercased) -> lemmas
    index: HashSet<String>,              // valid lemmas
}

pub struct Lemmatizer {
    tables: HashMap<String, PosTables>, // keyed by lowercase UPOS
}

impl Lemmatizer {
    pub fn load(b: &Bundle) -> Option<Lemmatizer> {
        let lem = b.manifest.get("lemmatizer")?;
        if lem.is_null() {
            return None;
        }
        let tbls = &lem["tables"];
        let (rules_t, exc_t, index_t) =
            (&tbls["lemma_rules"], &tbls["lemma_exc"], &tbls["lemma_index"]);

        let mut pos_keys: HashSet<String> = HashSet::new();
        for t in [rules_t, exc_t, index_t] {
            if let Some(o) = t.as_object() {
                pos_keys.extend(o.keys().cloned());
            }
        }
        let mut tables = HashMap::new();
        for pos in pos_keys {
            let rules = rules_t
                .get(&pos)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|p| {
                            let p = p.as_array()?;
                            let old = p.first()?.as_str()?.to_string();
                            let new = p.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string();
                            Some((old, new))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let exc = exc_t
                .get(&pos)
                .and_then(|v| v.as_object())
                .map(|o| {
                    o.iter()
                        .map(|(w, lemmas)| {
                            let ls = lemmas
                                .as_array()
                                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                                .unwrap_or_default();
                            (w.clone(), ls)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let index = index_t
                .get(&pos)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            tables.insert(pos, PosTables { rules, exc, index });
        }
        Some(Lemmatizer { tables })
    }

    /// Lemmatize one token from its surface text, final UPOS, and morph features.
    pub fn lemmatize(&self, text: &str, pos: &str, morph: &HashMap<String, String>) -> String {
        let univ_pos = pos.to_lowercase();
        if univ_pos.is_empty() || univ_pos == "eol" || univ_pos == "space" {
            return text.to_lowercase();
        }
        if is_base_form(&univ_pos, morph) {
            return text.to_lowercase();
        }
        let tbl = self.tables.get(&univ_pos);
        let has_any = tbl
            .map(|t| !t.index.is_empty() || !t.exc.is_empty() || !t.rules.is_empty())
            .unwrap_or(false);
        if !has_any {
            return if univ_pos == "propn" { text.to_string() } else { text.to_lowercase() };
        }
        let tbl = tbl.unwrap();
        let orig = text.to_string();
        let string = text.to_lowercase();
        let mut forms: Vec<String> = Vec::new();
        let mut oov: Vec<String> = Vec::new();
        for (old, new) in &tbl.rules {
            if string.ends_with(old.as_str()) {
                let form = format!("{}{}", &string[..string.len() - old.len()], new);
                if form.is_empty() {
                    // pass
                } else if tbl.index.contains(&form) {
                    forms.insert(0, form); // in-vocab -> front
                } else if !is_alpha(&form) {
                    forms.push(form); // non-alpha (e.g. punctuation) -> back
                } else {
                    oov.push(form);
                }
            }
        }
        dedupe_in_place(&mut forms);
        // exceptions take priority (inserted at the front)
        if let Some(exs) = tbl.exc.get(&string) {
            for form in exs {
                if !forms.contains(form) {
                    forms.insert(0, form.clone());
                }
            }
        }
        if forms.is_empty() {
            forms.extend(oov);
        }
        if forms.is_empty() {
            forms.push(orig);
        }
        forms.into_iter().next().unwrap()
    }
}

fn is_base_form(univ_pos: &str, m: &HashMap<String, String>) -> bool {
    let get = |k: &str| m.get(k).map(|s| s.as_str());
    if univ_pos == "noun" && get("Number") == Some("Sing") {
        return true;
    }
    if univ_pos == "verb" && get("VerbForm") == Some("Inf") {
        return true;
    }
    if univ_pos == "verb"
        && get("VerbForm") == Some("Fin")
        && get("Tense") == Some("Pres")
        && get("Number").is_none()
    {
        return true;
    }
    if univ_pos == "adj" && get("Degree") == Some("Pos") {
        return true;
    }
    if get("VerbForm") == Some("Inf") {
        return true;
    }
    if get("VerbForm") == Some("None") {
        return true; // spaCy literally compares to the string "None"
    }
    if get("Degree") == Some("Pos") {
        return true;
    }
    false
}

fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic())
}

fn dedupe_in_place(v: &mut Vec<String>) {
    let mut seen = HashSet::new();
    v.retain(|x| seen.insert(x.clone()));
}

/// Parse spaCy's `str(token.morph)` ("Feat=Val|Feat2=Val2", or "" / "_") into a map.
pub fn parse_morph(s: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    if s.is_empty() || s == "_" {
        return m;
    }
    for feat in s.split('|') {
        if let Some((k, v)) = feat.split_once('=') {
            m.insert(k.to_string(), v.to_string());
        }
    }
    m
}
