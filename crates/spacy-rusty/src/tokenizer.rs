//! Rule-based tokenizer — a faithful port of spaCy's `Tokenizer` (the
//! `explain()` reference algorithm), its whitespace-token rule, and the
//! post-tokenization special-case matcher. Carries per-token NORM overrides
//! from special-case rules.
//!
//! Mirrors spacy/tokenizer.pyx + util.compile_{prefix,suffix,infix}_regex.
//! Prefix/suffix/infix patterns use lookbehind, so we use `fancy-regex`.

use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Label {
    Token,
    Prefix,
    Suffix,
    Infix,
    UrlMatch,
    TokenMatch,
    Special(usize),
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Label::Token => write!(f, "TOKEN"),
            Label::Prefix => write!(f, "PREFIX"),
            Label::Suffix => write!(f, "SUFFIX"),
            Label::Infix => write!(f, "INFIX"),
            Label::UrlMatch => write!(f, "URL_MATCH"),
            Label::TokenMatch => write!(f, "TOKEN_MATCH"),
            Label::Special(i) => write!(f, "SPECIAL-{}", i),
        }
    }
}

#[derive(Debug, Clone)]
struct Piece {
    label: Label,
    text: String,
    norm: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub text: String,
    /// char (code-point) offset into the source text, matching spaCy `token.idx`
    pub start: usize,
    pub end: usize,
    /// trailing single space (spaCy `token.whitespace_ == " "`)
    pub ws: bool,
    /// NORM override from a special-case rule, if any
    pub norm: Option<String>,
}

type Special = Vec<(String, Option<String>)>; // (ORTH, NORM?)

pub struct Tokenizer {
    prefix_re: Option<Regex>,
    suffix_re: Option<Regex>,
    infix_re: Option<Regex>,
    url_re: Option<Regex>,
    token_re: Option<Regex>,
    specials: HashMap<String, Special>,
    /// special matcher: first pattern token -> [(pattern token texts, replacement ORTHs)]
    matcher: HashMap<String, Vec<(Vec<String>, Vec<String>)>>,
}

fn compile(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|e| panic!("regex compile failed for {:?}: {}", pattern, e))
}

fn join_list(v: &Value, wrap: impl Fn(&str) -> String) -> Option<String> {
    let arr = v.as_array()?;
    let parts: Vec<String> = arr
        .iter()
        .filter_map(|p| p.as_str())
        .filter(|p| !p.trim().is_empty())
        .map(|p| wrap(p))
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("|"))
    }
}

impl Tokenizer {
    pub fn from_manifest(tok: &Value) -> Self {
        let prefix_re = join_list(&tok["prefixes"], |p| format!("^{}", p)).map(|s| compile(&s));
        let suffix_re = join_list(&tok["suffixes"], |p| format!("{}$", p)).map(|s| compile(&s));
        let infix_re = join_list(&tok["infixes"], |p| p.to_string()).map(|s| compile(&s));
        let url_re = tok["url_match"].as_str().map(maybe_strip_u).map(|s| compile(&s));
        let token_re = tok["token_match"].as_str().map(maybe_strip_u).map(|s| compile(&s));

        let mut specials = HashMap::new();
        if let Some(rules) = tok["rules"].as_object() {
            for (orth, analyses) in rules {
                let pieces: Special = analyses
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|d| {
                                d.get("ORTH").and_then(|x| x.as_str()).map(|o| {
                                    let n = d.get("NORM").and_then(|x| x.as_str()).map(|s| s.to_string());
                                    (o.to_string(), n)
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if !pieces.is_empty() {
                    specials.insert(orth.clone(), pieces);
                }
            }
        }

        let mut t = Tokenizer {
            prefix_re,
            suffix_re,
            infix_re,
            url_re,
            token_re,
            specials,
            matcher: HashMap::new(),
        };
        t.build_matcher();
        t
    }

    fn build_matcher(&mut self) {
        let mut matcher: HashMap<String, Vec<(Vec<String>, Vec<String>)>> = HashMap::new();
        let keys: Vec<String> = self.specials.keys().cloned().collect();
        for key in keys {
            let pattern: Vec<String> =
                self.tokenize_word(&key, false).into_iter().map(|p| p.text).collect();
            if pattern.is_empty() {
                continue;
            }
            let replacement: Vec<String> =
                self.specials[&key].iter().map(|(o, _)| o.clone()).collect();
            matcher.entry(pattern[0].clone()).or_default().push((pattern, replacement));
        }
        for bucket in matcher.values_mut() {
            bucket.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        }
        self.matcher = matcher;
    }

    fn prefix_len(&self, s: &str) -> Option<usize> {
        self.prefix_re.as_ref()?.find(s).ok().flatten().map(|m| m.end())
    }
    fn suffix_start(&self, s: &str) -> Option<usize> {
        self.suffix_re.as_ref()?.find(s).ok().flatten().map(|m| m.start())
    }
    fn is_token_match(&self, s: &str) -> bool {
        self.token_re.as_ref().map_or(false, |r| r.is_match(s).unwrap_or(false))
    }
    fn is_url_match(&self, s: &str) -> bool {
        self.url_re.as_ref().map_or(false, |r| r.is_match(s).unwrap_or(false))
    }

    fn infix_split(&self, s: &str) -> Option<Vec<Piece>> {
        let re = self.infix_re.as_ref()?;
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for m in re.find_iter(s).flatten() {
            ranges.push((m.start(), m.end()));
        }
        if ranges.is_empty() {
            return None;
        }
        let mut out = Vec::new();
        let mut offset = 0usize;
        for (st, en) in ranges {
            if offset == 0 && st == 0 {
                continue;
            }
            if st > offset {
                out.push(piece(Label::Token, &s[offset..st]));
            }
            if en > st {
                out.push(piece(Label::Infix, &s[st..en]));
            }
            offset = en;
        }
        if offset < s.len() {
            out.push(piece(Label::Token, &s[offset..]));
        }
        Some(out)
    }

    fn specials_pieces(&self, s: &str) -> Vec<Piece> {
        self.specials[s]
            .iter()
            .enumerate()
            .map(|(i, (orth, norm))| Piece {
                label: Label::Special(i + 1),
                text: orth.clone(),
                norm: norm.clone(),
            })
            .collect()
    }

    fn tokenize_word(&self, substring: &str, with_special: bool) -> Vec<Piece> {
        let mut tokens: Vec<Piece> = Vec::new();
        let mut suffixes: Vec<Piece> = Vec::new();
        let mut s = substring.to_string();

        while !s.is_empty() {
            if with_special && self.specials.contains_key(&s) {
                tokens.extend(self.specials_pieces(&s));
                s.clear();
                continue;
            }

            loop {
                let has_pre = self.prefix_len(&s);
                let has_suf = self.suffix_start(&s);
                if has_pre.is_none() && has_suf.is_none() {
                    break;
                }
                if self.is_token_match(&s) {
                    break;
                }
                if with_special && self.specials.contains_key(&s) {
                    break;
                }
                let mut progressed = false;
                if let Some(end) = self.prefix_len(&s) {
                    if end > 0 {
                        tokens.push(piece(Label::Prefix, &s[..end]));
                        s = s[end..].to_string();
                        progressed = true;
                        if with_special && self.specials.contains_key(&s) {
                            continue;
                        }
                    }
                }
                if let Some(start) = self.suffix_start(&s) {
                    if start < s.len() {
                        suffixes.push(piece(Label::Suffix, &s[start..]));
                        s = s[..start].to_string();
                        progressed = true;
                    }
                }
                if !progressed {
                    break;
                }
            }

            if self.is_token_match(&s) {
                tokens.push(piece(Label::TokenMatch, &s));
                s.clear();
            } else if self.is_url_match(&s) {
                tokens.push(piece(Label::UrlMatch, &s));
                s.clear();
            } else if with_special && self.specials.contains_key(&s) {
                tokens.extend(self.specials_pieces(&s));
                s.clear();
            } else if let Some(splits) = self.infix_split(&s) {
                tokens.extend(splits);
                s.clear();
            } else if !s.is_empty() {
                tokens.push(piece(Label::Token, &s));
                s.clear();
            }
        }

        tokens.extend(suffixes.into_iter().rev());
        tokens
    }

    fn apply_matcher(&self, pieces: Vec<(Label, String, usize)>) -> Vec<(Label, String, usize)> {
        let n = pieces.len();
        let mut out: Vec<(Label, String, usize)> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let mut hit: Option<&(Vec<String>, Vec<String>)> = None;
            if let Some(bucket) = self.matcher.get(&pieces[i].1) {
                for cand in bucket {
                    let l = cand.0.len();
                    if i + l <= n
                        && (1..l).all(|k| pieces[i + k].2 == pieces[i].2)
                        && (0..l).all(|k| pieces[i + k].1 == cand.0[k])
                    {
                        hit = Some(cand);
                        break;
                    }
                }
            }
            if let Some((pattern, replacement)) = hit {
                let sid = pieces[i].2;
                for (k, orth) in replacement.iter().enumerate() {
                    out.push((Label::Special(k + 1), orth.clone(), sid));
                }
                i += pattern.len();
            } else {
                out.push(pieces[i].clone());
                i += 1;
            }
        }
        out
    }

    /// Reproduce spaCy `tokenizer.explain(text)`.
    pub fn explain(&self, text: &str) -> Vec<(String, String)> {
        let mut pieces: Vec<(Label, String, usize)> = Vec::new();
        for (sid, substring) in text.split_whitespace().enumerate() {
            for p in self.tokenize_word(substring, true) {
                pieces.push((p.label, p.text, sid));
            }
        }
        self.apply_matcher(pieces).into_iter().map(|(l, t, _)| (l.to_string(), t)).collect()
    }

    /// Full tokenization into the Doc token list (incl. whitespace tokens),
    /// with char offsets, trailing-space flags, and NORM overrides.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut tokens: Vec<Token> = Vec::new();
        let mut i = 0;
        while i < n {
            let start = i;
            let is_ws = chars[i].is_whitespace();
            while i < n && chars[i].is_whitespace() == is_ws {
                i += 1;
            }
            if is_ws {
                let first = chars[start];
                if !tokens.is_empty() && first == ' ' {
                    tokens.last_mut().unwrap().ws = true;
                    if i - start > 1 {
                        let txt: String = chars[start + 1..i].iter().collect();
                        tokens.push(Token { text: txt, start: start + 1, end: i, ws: false, norm: None });
                    }
                } else {
                    let txt: String = chars[start..i].iter().collect();
                    tokens.push(Token { text: txt, start, end: i, ws: false, norm: None });
                }
            } else {
                let substr: String = chars[start..i].iter().collect();
                let mut off = start;
                for p in self.tokenize_word(&substr, true) {
                    let len = p.text.chars().count();
                    tokens.push(Token { text: p.text, start: off, end: off + len, ws: false, norm: p.norm });
                    off += len;
                }
            }
        }
        tokens
    }
}

fn piece(label: Label, text: &str) -> Piece {
    Piece { label, text: text.to_string(), norm: None }
}

fn maybe_strip_u(p: &str) -> String {
    p.strip_prefix("(?u)").unwrap_or(p).to_string()
}
