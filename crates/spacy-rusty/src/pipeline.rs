//! The full inference pipeline: text -> tokens + POS + NER, loadable from
//! in-memory bytes (for wasm/browser). Ties tokenizer, features, tok2vec,
//! tagger, attribute_ruler, and NER.

use crate::attribute_ruler::{Anno, AttributeRuler};
use crate::features::Features;
use crate::lemmatizer::{parse_morph, Lemmatizer};
use crate::lexeme;
use crate::lexeme::is_space;
use crate::matcher::MatchToken;
use crate::model::Bundle;
use crate::ner::Ner;
use crate::parser::Parser;
use crate::tagger::Tagger;
use crate::tok2vec::Tok2Vec;
use crate::tokenizer::Tokenizer;
use crate::vectors::Vectors;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Serialize, Clone)]
pub struct TokenOut {
    pub i: usize,
    pub text: String,
    pub idx: usize,
    pub ws: bool,
    pub tag: String,
    pub pos: String,
    pub morph: String,
    pub ent_iob: String,
    pub ent_type: String,
    pub head: usize,        // parser: head token index (== i for a root)
    pub dep: String,        // parser: dependency label ("ROOT" for roots, "" if no parser)
    pub is_sent_start: bool,
    pub lemma: String,      // rule lemmatizer
    pub has_vector: bool,   // token has a static word vector
    pub vector_norm: f32,   // L2 norm of the token's vector (0.0 if none)
}

#[derive(Serialize, Clone)]
pub struct EntOut {
    pub start: usize,
    pub end: usize,
    pub label: String,
    pub text: String,
}

#[derive(Serialize, Clone)]
pub struct SentOut {
    pub start: usize, // char offset
    pub end: usize,
    pub start_token: usize,
    pub end_token: usize, // exclusive
}

#[derive(Serialize, Clone)]
pub struct ChunkOut {
    pub start: usize, // char offset
    pub end: usize,
    pub start_token: usize,
    pub end_token: usize, // exclusive
    pub root: usize,      // root token index of the chunk
    pub text: String,
}

#[derive(Serialize, Clone)]
pub struct DocOut {
    pub text: String,
    pub tokens: Vec<TokenOut>,
    pub ents: Vec<EntOut>,
    pub sents: Vec<SentOut>,
    pub noun_chunks: Vec<ChunkOut>,
}

/// A token-range view over a `DocOut` (spaCy `Span`).
#[derive(Serialize, Clone)]
pub struct SpanView {
    pub start: usize,       // char offset
    pub end: usize,         // char offset (exclusive)
    pub start_token: usize,
    pub end_token: usize,   // exclusive
    pub root: usize,        // index of the span's syntactic root token
    pub text: String,
}

impl DocOut {
    fn chars(&self) -> Vec<char> {
        self.text.chars().collect()
    }

    /// Slice `[start_token, end_token)` into a `Span` (spaCy `doc[a:b]`).
    pub fn span(&self, start_token: usize, end_token: usize) -> SpanView {
        let n = self.tokens.len();
        let s = start_token.min(n);
        let e = end_token.min(n).max(s);
        let chars = self.chars();
        let (start, end) = if e > s {
            let cs = self.tokens[s].idx;
            let last = &self.tokens[e - 1];
            (cs, last.idx + last.text.chars().count())
        } else {
            (0, 0)
        };
        let root = self.span_root(s, e);
        SpanView {
            start,
            end,
            start_token: s,
            end_token: e,
            root,
            text: chars[start..end].iter().collect(),
        }
    }

    /// spaCy `Span.root`: the token with the shortest path to the sentence
    /// root. An in-span sentence root wins immediately (first one); otherwise
    /// argmin over tokens whose head lies outside the span; ties -> first.
    fn span_root(&self, start: usize, end: usize) -> usize {
        if end <= start {
            return start;
        }
        for i in start..end {
            if self.tokens[i].head == i {
                return i; // sentence root inside the span
            }
        }
        let n_tokens = self.tokens.len();
        // a token has children iff some other token's head points to it
        let mut has_child = vec![false; n_tokens];
        for i in 0..n_tokens {
            let h = self.tokens[i].head;
            if h != i {
                has_child[h] = true;
            }
        }
        let words_to_root = |start_i: usize| -> usize {
            // spaCy `_count_words_to_root`: a childless punct/space token is
            // penalised to sent_length-1 so it doesn't become a span root.
            let txt = &self.tokens[start_i].text;
            if !has_child[start_i]
                && (lexeme::is_punct(txt) || lexeme::is_space(txt))
            {
                return n_tokens - 1;
            }
            let mut cur = start_i;
            let mut n = 0usize;
            while self.tokens[cur].head != cur {
                cur = self.tokens[cur].head;
                n += 1;
                if n >= n_tokens {
                    break;
                }
            }
            n
        };
        let mut best = usize::MAX;
        let mut root = start;
        let mut found = false;
        for i in start..end {
            let h = self.tokens[i].head;
            if start <= h && h < end {
                continue; // head inside span
            }
            let w = words_to_root(i);
            if w < best {
                best = w;
                root = i;
                found = true;
            }
        }
        if found {
            root
        } else {
            start
        }
    }

    /// Serialize to spaCy's `Doc.to_json()` structure exactly: `text`, `ents`
    /// (char offsets + label), `sents` (char offsets), and `tokens`
    /// (id/start/end char offsets, tag/pos/morph/lemma/dep/head).
    pub fn to_spacy_json(&self) -> Value {
        let tokens: Vec<Value> = self
            .tokens
            .iter()
            .map(|t| {
                serde_json::json!({
                    "id": t.i,
                    "start": t.idx,
                    "end": t.idx + t.text.chars().count(),
                    "tag": t.tag,
                    "pos": t.pos,
                    "morph": t.morph,
                    "lemma": t.lemma,
                    "dep": t.dep,
                    "head": t.head,
                })
            })
            .collect();
        let ents: Vec<Value> = self
            .ents
            .iter()
            .map(|e| serde_json::json!({"start": e.start, "end": e.end, "label": e.label}))
            .collect();
        let sents: Vec<Value> =
            self.sents.iter().map(|s| serde_json::json!({"start": s.start, "end": s.end})).collect();
        serde_json::json!({"text": self.text, "ents": ents, "sents": sents, "tokens": tokens})
    }
}

/// spaCy's English `noun_chunks` syntax iterator over a dependency parse.
/// `heads[i]` is i's head (== i for a root), `deps[i]` the dependency label,
/// `pos[i]` the UPOS. Returns base-NP spans as (start_token, end_token, root).
fn noun_chunks(heads: &[usize], deps: &[String], pos: &[String]) -> Vec<(usize, usize, usize)> {
    let n = heads.len();
    // left_edge[i]: leftmost index in i's subtree (incl. i), via relaxation.
    let mut l: Vec<usize> = (0..n).collect();
    let mut guard = 0;
    loop {
        let mut changed = false;
        for i in 0..n {
            let h = heads[i];
            if l[i] < l[h] {
                l[h] = l[i];
                changed = true;
            }
        }
        guard += 1;
        if !changed || guard > n + 10 {
            break;
        }
    }
    const LABELS: &[&str] = &[
        "oprd", "nsubj", "dobj", "nsubjpass", "pcomp", "pobj", "dative", "appos", "attr", "ROOT",
    ];
    let is_np = |d: &str| LABELS.contains(&d);
    let mut out = Vec::new();
    let mut prev_end: i64 = -1;
    for i in 0..n {
        if pos[i] != "NOUN" && pos[i] != "PROPN" && pos[i] != "PRON" {
            continue;
        }
        if (l[i] as i64) <= prev_end {
            continue; // prevent nested/overlapping chunks
        }
        if is_np(&deps[i]) {
            prev_end = i as i64;
            out.push((l[i], i + 1, i));
        } else if deps[i] == "conj" {
            // walk up the conj chain to the head it coordinates with
            let mut head = heads[i];
            while deps[head] == "conj" && heads[head] < head {
                head = heads[head];
            }
            if is_np(&deps[head]) {
                prev_end = i as i64;
                out.push((l[i], i + 1, i));
            }
        }
    }
    out
}

pub struct Pipeline {
    tokenizer: Tokenizer,
    features: Features,
    tok2vec: Tok2Vec,
    tagger: Tagger,
    parser: Option<Parser>,
    ruler: AttributeRuler,
    lemmatizer: Option<Lemmatizer>,
    ner: Ner,
    vectors: Option<Arc<Vectors>>,
    uses_static: bool,
    pub model_name: String,
}

fn parse_row2word(json: Option<&str>) -> Vec<String> {
    json.and_then(|s| serde_json::from_str::<Vec<String>>(s).ok()).unwrap_or_default()
}

fn parse_key2row(json: Option<&str>) -> HashMap<u64, usize> {
    let mut m = HashMap::new();
    if let Some(s) = json {
        if let Ok(v) = serde_json::from_str::<Value>(s) {
            if let Some(obj) = v.as_object() {
                for (k, r) in obj {
                    if let (Ok(key), Some(row)) = (k.parse::<u64>(), r.as_u64()) {
                        m.insert(key, row as usize);
                    }
                }
            }
        }
    }
    m
}

impl Pipeline {
    pub fn from_bytes(manifest_json: &str, safetensors: &[u8], key2row_json: Option<&str>) -> Pipeline {
        Pipeline::from_bytes_full(manifest_json, safetensors, key2row_json, None)
    }

    /// As `from_bytes`, plus the row->word table that enables `most_similar` to
    /// return readable neighbor words (`vectors_row2word.json`).
    pub fn from_bytes_full(
        manifest_json: &str,
        safetensors: &[u8],
        key2row_json: Option<&str>,
        row2word_json: Option<&str>,
    ) -> Pipeline {
        let bundle = Bundle::from_bytes(manifest_json, safetensors);
        let key2row = parse_key2row(key2row_json);
        let row2word = parse_row2word(row2word_json);
        let cfg = &bundle.manifest["tok2vec"];
        let uses_static = cfg["include_static_vectors"].as_bool().unwrap_or(false);
        let model_name =
            bundle.manifest["model_name"].as_str().unwrap_or("unknown").to_string();
        let tokenizer = Tokenizer::from_manifest(&bundle.manifest["tokenizer"]);
        let features = Features::from_manifest(&bundle.manifest);
        // One shared vectors table (held once, used by tok2vec, NER, and the API).
        let vectors = Vectors::load(&bundle, key2row, row2word).map(Arc::new);
        let tok2vec = Tok2Vec::load(&bundle, cfg, "tok2vec", vectors.clone());
        let tagger = Tagger::load(&bundle);
        let parser = Parser::load(&bundle);
        let ruler = AttributeRuler::load(&bundle.manifest);
        let lemmatizer = Lemmatizer::load(&bundle);
        let ner = Ner::load(&bundle, vectors.clone());
        Pipeline {
            tokenizer, features, tok2vec, tagger, parser, ruler, lemmatizer, ner, vectors,
            uses_static, model_name,
        }
    }

    pub fn process(&self, text: &str) -> DocOut {
        let toks = self.tokenizer.tokenize(text);
        let n = toks.len();
        let feats6: Vec<Vec<u64>> =
            toks.iter().map(|t| self.features.keys(&t.text, t.norm.as_deref(), t.ws)).collect();
        let orths: Vec<u64> = toks.iter().map(|t| self.features.orth(&t.text)).collect();
        let orths_opt = if self.uses_static { Some(orths.as_slice()) } else { None };

        let vecs = self.tok2vec.forward(&feats6, orths_opt);
        let tags = self.tagger.predict(&vecs);
        // The parser shares the main tok2vec output (Tok2VecListener): feed it
        // the same vecs. It yields head/dep_ and sentence starts.
        let parse = self.parser.as_ref().map(|p| p.predict(&vecs));

        let mut annos: Vec<Anno> = toks
            .iter()
            .zip(tags.iter())
            .enumerate()
            .map(|(i, (t, tg))| Anno {
                text: t.text.clone(),
                tag: tg.clone(),
                // Populate dep_ BEFORE the attribute_ruler so its DEP-conditioned
                // patterns fire (matches spaCy's order parser -> attribute_ruler;
                // this is what closes the v1 pos_ gap vs the full pipeline).
                dep: parse.as_ref().map(|p| p[i].dep.clone()).unwrap_or_default(),
                ..Default::default()
            })
            .collect();
        self.ruler.apply(&mut annos);

        // Lemma: the attribute_ruler sets irregular/pronoun lemmas (LEMMA attr);
        // the rule lemmatizer (overwrite=false) fills only tokens it left unset,
        // using the FINAL pos_/morph.
        let lemmas: Vec<String> = (0..n)
            .map(|i| {
                if !annos[i].lemma.is_empty() {
                    annos[i].lemma.clone()
                } else {
                    match self.lemmatizer.as_ref() {
                        Some(l) => {
                            l.lemmatize(&annos[i].text, &annos[i].pos, &parse_morph(&annos[i].morph))
                        }
                        None => String::new(),
                    }
                }
            })
            .collect();

        let feats4: Vec<Vec<u64>> = feats6.iter().map(|f| f[..4].to_vec()).collect();
        let spaces: Vec<bool> = toks.iter().map(|t| is_space(&t.text)).collect();
        let ents_tok = self.ner.predict(&feats4, orths_opt, &spaces);

        let mut iob = vec!["O".to_string(); n];
        let mut etype = vec![String::new(); n];
        for (s, e, lab) in &ents_tok {
            for k in *s..*e {
                iob[k] = if k == *s { "B".into() } else { "I".into() };
                etype[k] = lab.clone();
            }
        }

        let chars: Vec<char> = text.chars().collect();
        let tokens: Vec<TokenOut> = toks
            .iter()
            .enumerate()
            .map(|(i, t)| TokenOut {
                i,
                text: t.text.clone(),
                idx: t.start,
                ws: t.ws,
                tag: annos[i].tag.clone(),
                pos: annos[i].pos.clone(),
                morph: annos[i].morph.clone(),
                ent_iob: iob[i].clone(),
                ent_type: etype[i].clone(),
                head: parse.as_ref().map(|p| p[i].head).unwrap_or(i),
                dep: annos[i].dep.clone(),
                is_sent_start: parse.as_ref().map(|p| p[i].is_sent_start).unwrap_or(i == 0),
                lemma: lemmas[i].clone(),
                has_vector: self.vectors.as_ref().map(|v| v.has_vector(orths[i])).unwrap_or(false),
                vector_norm: self
                    .vectors
                    .as_ref()
                    .map(|v| Vectors::norm(&v.get(orths[i])))
                    .unwrap_or(0.0),
            })
            .collect();
        let ents: Vec<EntOut> = ents_tok
            .iter()
            .map(|(s, e, lab)| {
                let cs = toks[*s].start;
                let ce = toks[*e - 1].end;
                EntOut {
                    start: cs,
                    end: ce,
                    label: lab.clone(),
                    text: chars[cs..ce].iter().collect(),
                }
            })
            .collect();

        // doc.sents: split at each is_sent_start token (token 0 always starts one)
        let sents: Vec<SentOut> = match parse.as_ref() {
            Some(p) if n > 0 => {
                let mut out = Vec::new();
                let mut start = 0usize;
                for i in 1..n {
                    if p[i].is_sent_start {
                        out.push(SentOut {
                            start: toks[start].start,
                            end: toks[i - 1].end,
                            start_token: start,
                            end_token: i,
                        });
                        start = i;
                    }
                }
                out.push(SentOut {
                    start: toks[start].start,
                    end: toks[n - 1].end,
                    start_token: start,
                    end_token: n,
                });
                out
            }
            _ => vec![],
        };

        // doc.noun_chunks: base noun phrases from the parse (needs final pos_/dep_)
        let noun_chunks: Vec<ChunkOut> = match parse.as_ref() {
            Some(p) if n > 0 => {
                let heads: Vec<usize> = (0..n).map(|i| p[i].head).collect();
                let deps: Vec<String> = (0..n).map(|i| annos[i].dep.clone()).collect();
                let poss: Vec<String> = (0..n).map(|i| annos[i].pos.clone()).collect();
                noun_chunks(&heads, &deps, &poss)
                    .into_iter()
                    .map(|(s, e, root)| {
                        let cs = toks[s].start;
                        let ce = toks[e - 1].end;
                        ChunkOut {
                            start: cs,
                            end: ce,
                            start_token: s,
                            end_token: e,
                            root,
                            text: chars[cs..ce].iter().collect(),
                        }
                    })
                    .collect()
            }
            _ => vec![],
        };

        DocOut { text: text.to_string(), tokens, ents, sents, noun_chunks }
    }

    /// Per-token attributes for the rule matcher (matcher.rs operates on these).
    pub fn match_tokens(&self, text: &str) -> Vec<MatchToken> {
        let doc = self.process(text);
        let toks = self.tokenizer.tokenize(text);
        doc.tokens
            .iter()
            .zip(toks.iter())
            .map(|(t, tk)| {
                let orth = t.text.clone();
                MatchToken {
                    lower: orth.to_lowercase(),
                    norm: self.features.norm(&orth, tk.norm.as_deref()),
                    shape: lexeme::shape(&orth),
                    lemma: t.lemma.clone(),
                    pos: t.pos.clone(),
                    tag: t.tag.clone(),
                    dep: t.dep.clone(),
                    ent_type: t.ent_type.clone(),
                    is_alpha: lexeme::is_alpha(&orth),
                    is_digit: lexeme::is_digit(&orth),
                    is_punct: lexeme::is_punct(&orth),
                    is_space: lexeme::is_space(&orth),
                    is_lower: lexeme::is_lower(&orth),
                    is_upper: lexeme::is_upper(&orth),
                    is_title: lexeme::is_title(&orth),
                    like_num: lexeme::like_num(&orth),
                    length: orth.chars().count() as i64,
                    orth,
                }
            })
            .collect()
    }

    /// Run a token-pattern Matcher: `patterns` is a JSON object
    /// `{ "KEY": [ [ {token}, ... ], ... ], ... }`. Returns (key, start, end).
    pub fn run_matcher(&self, text: &str, patterns: &Value) -> Vec<(String, usize, usize)> {
        let mut m = crate::matcher::Matcher::new();
        if let Some(obj) = patterns.as_object() {
            for (key, pats) in obj {
                if let Some(arr) = pats.as_array() {
                    for p in arr {
                        m.add(key, p);
                    }
                }
            }
        }
        m.find(&self.match_tokens(text))
    }

    /// Run a PhraseMatcher over `text` for the given `attr` and phrase strings.
    /// Returns (phrase, start, end) using the phrase string as the key.
    pub fn run_phrase(&self, text: &str, attr: &str, phrases: &[String]) -> Vec<(String, usize, usize)> {
        let mut pm = crate::matcher::PhraseMatcher::new(attr);
        for ph in phrases {
            pm.add(ph, self.phrase_key(ph, attr));
        }
        pm.find(&self.match_tokens(text))
    }

    /// Run an EntityRuler: `patterns` is a JSON array of `{ "label", "pattern" }`
    /// where pattern is a token list or a phrase string. Returns (label,start,end),
    /// non-overlapping (spaCy `filter_spans`).
    pub fn run_entity_ruler(&self, text: &str, patterns: &Value) -> Vec<(String, usize, usize)> {
        let mut ruler = crate::matcher::EntityRuler::new();
        if let Some(arr) = patterns.as_array() {
            for r in arr {
                let label = r["label"].as_str().unwrap_or("");
                match &r["pattern"] {
                    Value::String(s) => ruler.add_phrase(label, "ORTH", self.phrase_key(s, "ORTH")),
                    p @ Value::Array(_) => ruler.add_token_pattern(label, p),
                    _ => {}
                }
            }
        }
        ruler.find(&self.match_tokens(text))
    }

    /// The attribute-value sequence of `phrase`'s tokens, for a PhraseMatcher
    /// keyed on `attr` ("ORTH"/"LOWER"/"NORM"/"LEMMA"/...).
    pub fn phrase_key(&self, phrase: &str, attr: &str) -> Vec<String> {
        self.match_tokens(phrase)
            .iter()
            .map(|t| match attr {
                "LOWER" => t.lower.clone(),
                "NORM" => t.norm.clone(),
                "LEMMA" => t.lemma.clone(),
                "SHAPE" => t.shape.clone(),
                "POS" => t.pos.clone(),
                "TAG" => t.tag.clone(),
                _ => t.orth.clone(),
            })
            .collect()
    }

    /// ORTH keys for each token of `text` (the vector-lookup keys).
    fn token_orths(&self, text: &str) -> Vec<u64> {
        self.tokenizer.tokenize(text).iter().map(|t| self.features.orth(&t.text)).collect()
    }

    pub fn has_vectors(&self) -> bool {
        self.vectors.is_some()
    }

    pub fn vector_dim(&self) -> usize {
        self.vectors.as_ref().map(|v| v.dim).unwrap_or(0)
    }

    /// The static vector for a single word (zeros if OOV / no vectors).
    pub fn word_vector(&self, word: &str) -> Vec<f32> {
        match self.vectors.as_ref() {
            Some(v) => v.get(self.features.orth(word)),
            None => vec![],
        }
    }

    /// Mean of `text`'s token vectors (spaCy `Doc.vector`).
    pub fn doc_vector(&self, text: &str) -> Vec<f32> {
        match self.vectors.as_ref() {
            Some(v) => v.mean(&self.token_orths(text)),
            None => vec![],
        }
    }

    /// Mean of token vectors over `[start_token, end_token)` (spaCy `Span.vector`).
    pub fn span_vector(&self, text: &str, start: usize, end: usize) -> Vec<f32> {
        match self.vectors.as_ref() {
            Some(v) => {
                let orths = self.token_orths(text);
                let s = start.min(orths.len());
                let e = end.min(orths.len()).max(s);
                v.mean(&orths[s..e])
            }
            None => vec![],
        }
    }

    /// Cosine similarity of two texts' doc vectors (spaCy `Doc.similarity`).
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        if self.vectors.is_none() {
            return 0.0;
        }
        Vectors::cosine(&self.doc_vector(a), &self.doc_vector(b))
    }

    /// The `n` words most similar to `word` (spaCy `Vectors.most_similar`).
    pub fn most_similar(&self, word: &str, n: usize) -> Vec<(usize, String, f32)> {
        match self.vectors.as_ref() {
            Some(v) => v.most_similar(&v.get(self.features.orth(word)), n),
            None => vec![],
        }
    }
}
