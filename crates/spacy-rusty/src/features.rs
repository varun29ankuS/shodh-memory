//! Compute the tok2vec feature keys for a token from text, including NORM
//! resolution (special-case override -> lexeme_norm lookup -> BASE_NORMS ->
//! lowercase), matching spaCy.

use crate::hash::string_id;
use crate::lexeme;
use serde_json::Value;
use std::collections::HashMap;

pub struct Features {
    pub symbols: HashMap<String, u64>,
    /// lexeme_norm lookups, keyed by string_id(orth)
    lexeme_norm: HashMap<u64, String>,
    /// BASE_NORMS, keyed by orth string
    base_norms: HashMap<String, String>,
}

impl Features {
    pub fn from_manifest(man: &Value) -> Features {
        let mut symbols = HashMap::new();
        if let Some(obj) = man["symbols"].as_object() {
            for (k, v) in obj {
                if let Some(n) = v.as_u64() {
                    symbols.insert(k.clone(), n);
                }
            }
        }
        let mut lexeme_norm = HashMap::new();
        if let Some(obj) = man["norm_exceptions"].as_object() {
            for (k, v) in obj {
                if let (Ok(h), Some(s)) = (k.parse::<u64>(), v.as_str()) {
                    lexeme_norm.insert(h, s.to_string());
                }
            }
        }
        let mut base_norms = HashMap::new();
        if let Some(obj) = man["base_norms"].as_object() {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    base_norms.insert(k.clone(), s.to_string());
                }
            }
        }
        Features { symbols, lexeme_norm, base_norms }
    }

    pub fn sid(&self, s: &str) -> u64 {
        string_id(s, &self.symbols)
    }

    pub fn norm(&self, text: &str, override_norm: Option<&str>) -> String {
        if let Some(n) = override_norm {
            return n.to_string();
        }
        if let Some(n) = self.lexeme_norm.get(&string_id(text, &self.symbols)) {
            return n.clone();
        }
        if let Some(n) = self.base_norms.get(text) {
            return n.clone();
        }
        text.to_lowercase()
    }

    /// The 6 feature keys [NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE].
    pub fn keys(&self, text: &str, norm_override: Option<&str>, trailing_space: bool) -> Vec<u64> {
        let norm = self.norm(text, norm_override);
        vec![
            self.sid(&norm),
            self.sid(&lexeme::prefix(text)),
            self.sid(&lexeme::suffix(text)),
            self.sid(&lexeme::shape(text)),
            if trailing_space { 1 } else { 0 },
            if lexeme::is_space(text) { 1 } else { 0 },
        ]
    }

    /// ORTH key for static-vector lookup.
    pub fn orth(&self, text: &str) -> u64 {
        self.sid(text)
    }
}
