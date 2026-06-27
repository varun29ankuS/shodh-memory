//! GLiNER multitask NER via ONNX Runtime (zero-shot, schema-driven entity types).
//!
//! Faithful Rust port of the GLiNER multitask ONNX inference path
//! (`onnx-community/gliner-multitask-large-v0.5`, DeBERTa-v3-large backbone).
//! Entity types are supplied at call time (zero-shot), injected into the prompt
//! as `<<ENT>>`-marked labels ahead of the text, in a single sequence — the
//! true GLiNER architecture (not a separate-encoding approximation).
//!
//! Verified model facts (against the actual graph + tokenizer):
//! - ONNX inputs (all i64): `input_ids`, `attention_mask`, `words_mask`, `text_lengths`.
//! - ONNX output `logits`: shape `[3, batch, num_words, num_classes]` — BIO scoring
//!   (position 0 = Begin, 1 = Inside, 2 = Outside).
//! - Special tokens: `[CLS]`/start = 1, `[SEP]`/end = 2, `<<ENT>>` = 128002, `<<SEP>>` = 128003.
//!
//! Output offsets are CHARACTER offsets into the original text (matching the
//! Python reference), not byte offsets.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::Path;
use tokenizers::Tokenizer;

/// GLiNER special token ids (verified against the model's `added_tokens.json`).
const TOKEN_START: i64 = 1; // [CLS]
const TOKEN_END: i64 = 2; // [SEP]
const TOKEN_ENT: i64 = 128002; // <<ENT>>  (entity-type marker)
const TOKEN_SEP: i64 = 128003; // <<SEP>>  (separates labels from text)

/// Default detection threshold (sigmoid over BIO logits), matching the reference.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// A typed entity extracted by GLiNER. Offsets are character indices.
#[derive(Debug, Clone, PartialEq)]
pub struct GlinerEntity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

/// GLiNER multitask NER model (ONNX Runtime session + tokenizer).
pub struct GlinerNer {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl GlinerNer {
    /// Load from a directory containing the ONNX model + `tokenizer.json`.
    ///
    /// Tries `model_quantized.onnx`, then `model.onnx`, then `onnx/model.onnx`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let model_path = ["model_quantized.onnx", "model.onnx", "onnx/model.onnx"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| {
                anyhow::anyhow!("no GLiNER ONNX model found under {}", dir.display())
            })?;

        let tok_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", tok_path.display()))?;

        let num_threads = std::env::var("SHODH_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);

        let session = Session::builder()
            .context("ort session builder")?
            .with_intra_threads(num_threads)
            .context("set intra threads")?
            .with_inter_threads(1)
            .context("set inter threads")?
            .commit_from_file(&model_path)
            .with_context(|| format!("load GLiNER ONNX {}", model_path.display()))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Load from `SHODH_GLINER_MODEL_PATH` (a directory), if set.
    pub fn from_env() -> Result<Self> {
        let dir = std::env::var("SHODH_GLINER_MODEL_PATH")
            .context("SHODH_GLINER_MODEL_PATH not set")?;
        Self::from_dir(Path::new(&dir))
    }

    /// Extract typed entities for the supplied (zero-shot) entity types.
    pub fn extract(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<GlinerEntity>> {
        let words = split_words_with_char_spans(text);
        if words.is_empty() || entity_types.is_empty() {
            return Ok(Vec::new());
        }
        let chars: Vec<char> = text.chars().collect();

        let prompt = self.encode_ner_prompt(entity_types, &words)?;
        let seq = prompt.input_ids.len();

        let input_ids = Value::from_array((vec![1_i64, seq as i64], prompt.input_ids))
            .context("input_ids tensor")?;
        let attention_mask = Value::from_array((vec![1_i64, seq as i64], prompt.attention_mask))
            .context("attention_mask tensor")?;
        let words_mask = Value::from_array((vec![1_i64, seq as i64], prompt.words_mask))
            .context("words_mask tensor")?;
        let text_lengths =
            Value::from_array((vec![1_i64, 1_i64], vec![words.len() as i64])).context("text_lengths")?;

        let mut session = self
            .session
            .try_lock_for(std::time::Duration::from_secs(30))
            .ok_or_else(|| anyhow::anyhow!("GLiNER session lock timeout"))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => &input_ids,
                "attention_mask" => &attention_mask,
                "words_mask" => &words_mask,
                "text_lengths" => &text_lengths,
            ])
            .context("GLiNER inference failed")?;

        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("extract logits")?;

        self.decode_bio(&shape, logits, entity_types, &words, &chars, threshold)
    }

    /// Build the GLiNER prompt: `[CLS] <<ENT>>L0… <<ENT>>L1… <<SEP>> w0… w1… [SEP]`.
    fn encode_ner_prompt(
        &self,
        entity_types: &[&str],
        words: &[(String, usize, usize)],
    ) -> Result<EncodedPrompt> {
        let mut input_ids: Vec<i64> = vec![TOKEN_START];
        let mut words_mask: Vec<i64> = vec![0];

        for ty in entity_types {
            input_ids.push(TOKEN_ENT);
            words_mask.push(0);
            let enc = self
                .tokenizer
                .encode(*ty, false)
                .map_err(|e| anyhow::anyhow!("tokenize entity type '{ty}': {e}"))?;
            for &id in enc.get_ids() {
                input_ids.push(id as i64);
                words_mask.push(0);
            }
        }
        input_ids.push(TOKEN_SEP);
        words_mask.push(0);

        for (wi, (word, _s, _e)) in words.iter().enumerate() {
            let enc = self
                .tokenizer
                .encode(word.as_str(), false)
                .map_err(|err| anyhow::anyhow!("tokenize word '{word}': {err}"))?;
            for (j, &id) in enc.get_ids().iter().enumerate() {
                input_ids.push(id as i64);
                // words_mask is 1-indexed; only the FIRST subword of a word carries its index.
                words_mask.push(if j == 0 { (wi + 1) as i64 } else { 0 });
            }
        }
        input_ids.push(TOKEN_END);
        words_mask.push(0);

        let attention_mask = vec![1_i64; input_ids.len()];
        Ok(EncodedPrompt {
            input_ids,
            attention_mask,
            words_mask,
        })
    }

    /// Decode BIO logits `[3, 1, num_words, num_classes]` into entities.
    fn decode_bio(
        &self,
        shape: &[i64],
        logits: &[f32],
        entity_types: &[&str],
        words: &[(String, usize, usize)],
        chars: &[char],
        threshold: f32,
    ) -> Result<Vec<GlinerEntity>> {
        // Expected [3, 1, num_words, num_classes]; be defensive about the exact rank.
        if shape.len() != 4 || shape[0] != 3 {
            anyhow::bail!("unexpected GLiNER logits shape {:?} (want [3,1,W,C])", shape);
        }
        let num_words = shape[2] as usize;
        let num_classes = shape[3] as usize;
        if num_classes != entity_types.len() || num_words != words.len() {
            anyhow::bail!(
                "logits shape {:?} disagrees with words={} classes={}",
                shape,
                words.len(),
                entity_types.len()
            );
        }
        // Strides for [3, 1, W, C] (batch dim = 1).
        let bio_stride = num_words * num_classes;
        let at = |bio: usize, w: usize, c: usize| -> f32 {
            sigmoid(logits[bio * bio_stride + w * num_classes + c])
        };

        let mut out: Vec<GlinerEntity> = Vec::new();
        for c in 0..num_classes {
            let mut w = 0usize;
            while w < num_words {
                if at(0, w, c) >= threshold {
                    // Begin: extend while Inside continues.
                    let start_w = w;
                    let mut score_sum = at(0, w, c);
                    let mut count = 1usize;
                    let mut end_w = w;
                    let mut k = w + 1;
                    while k < num_words && at(1, k, c) >= threshold {
                        score_sum += at(1, k, c);
                        count += 1;
                        end_w = k;
                        k += 1;
                    }
                    let cs = words[start_w].1;
                    let ce = words[end_w].2;
                    out.push(GlinerEntity {
                        text: chars[cs..ce.min(chars.len())].iter().collect(),
                        label: entity_types[c].to_string(),
                        start: cs,
                        end: ce,
                        score: score_sum / count as f32,
                    });
                    w = end_w + 1;
                } else {
                    w += 1;
                }
            }
        }

        // Resolve overlaps: longest span wins at each start; drop exact duplicates.
        out.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
        out.dedup_by(|a, b| a.start == b.start && a.end == b.end);
        Ok(out)
    }
}

struct EncodedPrompt {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    words_mask: Vec<i64>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Split into `(word, char_start, char_end_exclusive)` on Unicode whitespace,
/// tracking CHARACTER offsets (matching the Python reference).
fn split_words_with_char_spans(text: &str) -> Vec<(String, usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
        out.push((chars[start..i].iter().collect(), start, i));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_spans_char_offsets() {
        let w = split_words_with_char_spans("Steve Jobs founded Apple in California.");
        assert_eq!(w[0], ("Steve".to_string(), 0, 5));
        assert_eq!(w[1], ("Jobs".to_string(), 6, 10));
        // "Steve Jobs" entity span = w[0].start .. w[1].end = [0,10]
        assert_eq!((w[0].1, w[1].2), (0, 10));
    }

    /// Parity test vs the Python GLiNER reference. Gated on the model being
    /// present (set SHODH_GLINER_MODEL_PATH); skipped otherwise so CI without
    /// the model still passes.
    #[test]
    fn parity_with_python_reference() {
        let Ok(model) = GlinerNer::from_env() else {
            eprintln!("SHODH_GLINER_MODEL_PATH not set — skipping GLiNER parity test");
            return;
        };
        let labels = [
            "person",
            "organization",
            "location",
            "aircraft",
            "company",
            "country",
            "government agency",
        ];
        let ents = model
            .extract("Steve Jobs founded Apple in California.", &labels, 0.5)
            .expect("extract");
        let find = |t: &str| ents.iter().find(|e| e.text == t).cloned();
        let jobs = find("Steve Jobs").expect("Steve Jobs not found");
        assert_eq!((jobs.label.as_str(), jobs.start, jobs.end), ("person", 0, 10));
        assert!((jobs.score - 0.996).abs() < 0.05, "score {}", jobs.score);
        let apple = find("Apple").expect("Apple not found");
        assert_eq!((apple.label.as_str(), apple.start, apple.end), ("company", 19, 24));
    }
}
