//! wasm-bindgen bindings: load a model bundle from bytes and run inference,
//! returning a JS object (or JSON string).

use crate::pipeline::Pipeline;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SpacyModel {
    inner: Pipeline,
}

#[wasm_bindgen]
impl SpacyModel {
    /// Build from the bundle's model.json (string), model.safetensors (bytes),
    /// and optional vectors_key2row.json + vectors_row2word.json (strings, for
    /// `_md`/`_lg` static vectors; row2word enables readable `mostSimilar`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        manifest_json: String,
        safetensors: Vec<u8>,
        key2row_json: Option<String>,
        row2word_json: Option<String>,
    ) -> SpacyModel {
        SpacyModel {
            inner: Pipeline::from_bytes_full(
                &manifest_json,
                &safetensors,
                key2row_json.as_deref(),
                row2word_json.as_deref(),
            ),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.model_name.clone()
    }

    /// Process text -> a JS object { text, tokens[], ents[] }.
    pub fn process(&self, text: String) -> Result<JsValue, JsValue> {
        let doc = self.inner.process(&text);
        serde_wasm_bindgen::to_value(&doc).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Process text -> JSON string (handy for debugging).
    #[wasm_bindgen(js_name = processJson)]
    pub fn process_json(&self, text: String) -> String {
        serde_json::to_string(&self.inner.process(&text)).unwrap()
    }

    /// Whether this model has static word vectors (`_md`/`_lg`).
    #[wasm_bindgen(getter, js_name = hasVectors)]
    pub fn has_vectors(&self) -> bool {
        self.inner.has_vectors()
    }

    /// The static vector for a single word as a Float32Array (empty if no vectors).
    #[wasm_bindgen(js_name = wordVector)]
    pub fn word_vector(&self, word: String) -> Vec<f32> {
        self.inner.word_vector(&word)
    }

    /// The mean token-vector for `text` (spaCy `Doc.vector`) as a Float32Array.
    #[wasm_bindgen(js_name = docVector)]
    pub fn doc_vector(&self, text: String) -> Vec<f32> {
        self.inner.doc_vector(&text)
    }

    /// Mean token-vector over `[start, end)` tokens (spaCy `Span.vector`).
    #[wasm_bindgen(js_name = spanVector)]
    pub fn span_vector(&self, text: String, start: usize, end: usize) -> Vec<f32> {
        self.inner.span_vector(&text, start, end)
    }

    /// Cosine similarity of two texts' doc vectors (spaCy `Doc.similarity`).
    pub fn similarity(&self, a: String, b: String) -> f32 {
        self.inner.similarity(&a, &b)
    }

    /// The `n` words most similar to `word`: [{ word, row, score }, ...].
    #[wasm_bindgen(js_name = mostSimilar)]
    pub fn most_similar(&self, word: String, n: usize) -> Result<JsValue, JsValue> {
        #[derive(serde::Serialize)]
        struct Neighbor {
            word: String,
            row: usize,
            score: f32,
        }
        let out: Vec<Neighbor> = self
            .inner
            .most_similar(&word, n)
            .into_iter()
            .map(|(row, word, score)| Neighbor { word, row, score })
            .collect();
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Token-pattern Matcher. `patterns_json` is `{ "KEY": [[{token},...],...] }`.
    /// Returns [{ key, start, end }, ...].
    pub fn matcher(&self, text: String, patterns_json: String) -> Result<JsValue, JsValue> {
        let pats: serde_json::Value =
            serde_json::from_str(&patterns_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        spans_to_js(self.inner.run_matcher(&text, &pats), "key")
    }

    /// PhraseMatcher. `phrases_json` is a JSON array of strings; `attr` is e.g.
    /// "ORTH"/"LOWER". Returns [{ key, start, end }, ...] (key = the phrase).
    #[wasm_bindgen(js_name = phraseMatcher)]
    pub fn phrase_matcher(&self, text: String, attr: String, phrases_json: String) -> Result<JsValue, JsValue> {
        let phrases: Vec<String> =
            serde_json::from_str(&phrases_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        spans_to_js(self.inner.run_phrase(&text, &attr, &phrases), "key")
    }

    /// EntityRuler. `patterns_json` is `[{ "label", "pattern" }, ...]` where
    /// pattern is a token list or a phrase string. Returns [{ label, start, end }].
    #[wasm_bindgen(js_name = entityRuler)]
    pub fn entity_ruler(&self, text: String, patterns_json: String) -> Result<JsValue, JsValue> {
        let pats: serde_json::Value =
            serde_json::from_str(&patterns_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        spans_to_js(self.inner.run_entity_ruler(&text, &pats), "label")
    }
}

#[derive(serde::Serialize)]
struct MatchSpan {
    key: String,
    start: usize,
    end: usize,
}

fn spans_to_js(spans: Vec<(String, usize, usize)>, _field: &str) -> Result<JsValue, JsValue> {
    let out: Vec<MatchSpan> =
        spans.into_iter().map(|(key, start, end)| MatchSpan { key, start, end }).collect();
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}
