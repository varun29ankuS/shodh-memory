//! GLiREL zero-shot relation extraction via ONNX Runtime.
//!
//! Faithful Rust port of the GLiREL inference path (`jackboyla/glirel-large-v0`,
//! DeBERTa-v3-large backbone, projected hidden 768). Relation labels and entity
//! spans are supplied at call time (zero-shot). The model is split into two ONNX
//! graphs, matching the export pipeline:
//!
//! - **encoder** (`model.onnx`): DeBERTa-v3-large encoder + GLiREL projection.
//!   inputs `input_ids[1,S] i64`, `attention_mask[1,S] i64`
//!   → output `projected_hidden[1,S,768] f32`.
//! - **head** (`head.onnx`): the post-encoder relation head over WORD-level reps.
//!   inputs `word_rep[1,W,768] f32`, `word_mask[1,W] i64`,
//!   `rel_type_rep_raw[1,T,768] f32`, `span_idx[1,E,2] i64`,
//!   `relations_idx[1,P,2,2] i64`
//!   → output `logits[1,P,T] f32`.
//!
//! The glue between the two graphs is pure indexing (subword→word "first"
//! pooling, prompt split, `label_embed_strategy="both"` interleave, ordered
//! entity-pair generation). It reproduces `model.predict_relations` exactly
//! (verified: max score diff 7.8e-7, 0 mismatches against PyTorch).
//!
//! Verified model facts:
//! - Special token ids: `[REL]` = 128002, `[SEP]` = 2 (DeBERTa `</s>`), CLS = 1.
//! - The prompt is `[REL] L0 [REL] L1 … [REL] L_{T-1} [SEP] w0 w1 … w_{n-1}`,
//!   built as a PRE-SPLIT element list (each label is one element, even when
//!   multi-word) and subword-tokenized with `is_split_into_words` semantics
//!   (`add_special_tokens = true` wraps the whole sequence in CLS/SEP).
//! - First-subtoken pooling: each pre-split element keeps the embedding of its
//!   FIRST subword.
//! - `rel_type_rep_raw[t] = mean(word_emb[2t] ([REL]), word_emb[2t+1] (label))`.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::Path;
use tokenizers::Tokenizer;

/// GLiREL special tokens (verified against the model's tokenizer + `onnx_io.txt`).
const REL_TOKEN: &str = "[REL]";
const SEP_TOKEN: &str = "[SEP]";

/// Projected hidden dimension emitted by the encoder graph.
const HIDDEN: usize = 768;

/// Default emission threshold (sigmoid over the head logits), matching the reference.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// A predicted relation triple. `head_idx`/`tail_idx` index into the `entities`
/// slice passed to [`GlirelExtractor::extract_relations`].
#[derive(Debug, Clone, PartialEq)]
pub struct RelationTriple {
    pub head_idx: usize,
    pub tail_idx: usize,
    pub relation: String,
    pub score: f32,
}

/// GLiREL relation extractor (two ONNX Runtime sessions + tokenizer).
pub struct GlirelExtractor {
    encoder: Mutex<Session>,
    head: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl GlirelExtractor {
    /// Load from a directory containing `model.onnx`, `head.onnx`, and a
    /// `tokenizer.json` (looked up in the dir, then in `hf_tokenizer/`).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let encoder_path = ["model.onnx", "onnx/model.onnx"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!("no GLiREL encoder (model.onnx) under {}", dir.display()))?;

        let head_path = ["head.onnx", "onnx/head.onnx"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!("no GLiREL head (head.onnx) under {}", dir.display()))?;

        let tok_path = ["tokenizer.json", "hf_tokenizer/tokenizer.json"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!("no tokenizer.json under {}", dir.display()))?;

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", tok_path.display()))?;

        let num_threads = std::env::var("SHODH_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);

        let encoder = Session::builder()
            .context("ort session builder (encoder)")?
            .with_intra_threads(num_threads)
            .context("set intra threads (encoder)")?
            .with_inter_threads(1)
            .context("set inter threads (encoder)")?
            .commit_from_file(&encoder_path)
            .with_context(|| format!("load GLiREL encoder {}", encoder_path.display()))?;

        let head = Session::builder()
            .context("ort session builder (head)")?
            .with_intra_threads(num_threads)
            .context("set intra threads (head)")?
            .with_inter_threads(1)
            .context("set inter threads (head)")?
            .commit_from_file(&head_path)
            .with_context(|| format!("load GLiREL head {}", head_path.display()))?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            head: Mutex::new(head),
            tokenizer,
        })
    }

    /// Load from `SHODH_GLIREL_MODEL_PATH` (a directory), if set.
    pub fn from_env() -> Result<Self> {
        let dir =
            std::env::var("SHODH_GLIREL_MODEL_PATH").context("SHODH_GLIREL_MODEL_PATH not set")?;
        Self::from_dir(Path::new(&dir))
    }

    /// Extract relation triples among `entities` (zero-shot over `rel_labels`).
    ///
    /// `entities` are `(start_word, end_word, label)` with INCLUSIVE word indices
    /// into the words produced by tokenizing `text` with the GLiREL word regex
    /// (`\w+(?:[-_]\w+)*|\S`). The `label` is carried for the caller's benefit and
    /// does not affect scoring. A triple is emitted for every ordered entity pair
    /// `i != j` and label whose sigmoid score exceeds `threshold`.
    pub fn extract_relations(
        &self,
        text: &str,
        entities: &[(usize, usize, &str)],
        rel_labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<RelationTriple>> {
        let words = word_tokenize(text);
        let n = words.len();
        let t = rel_labels.len();
        let e = entities.len();
        if n == 0 || t == 0 || e < 2 {
            return Ok(Vec::new());
        }
        for &(s, en, _) in entities {
            if s > en || en >= n {
                anyhow::bail!(
                    "entity word span [{s},{en}] out of range for {n} words in {text:?}"
                );
            }
        }

        // --- 1. Build the pre-split prompt element list. ---
        //   [REL] L0 [REL] L1 ... [REL] L_{T-1} [SEP] w0 w1 ... w_{n-1}
        // prompt portion length = 2T + 1 (the trailing [SEP] is element index 2T).
        let prompt_len = 2 * t + 1;
        let mut elements: Vec<&str> = Vec::with_capacity(prompt_len + n);
        for label in rel_labels {
            elements.push(REL_TOKEN);
            elements.push(label);
        }
        elements.push(SEP_TOKEN);
        for w in &words {
            elements.push(w.as_str());
        }
        let num_elements = elements.len(); // = prompt_len + n

        // --- 2/3. Subword-tokenize the pre-split sequence (CLS/SEP auto-wrapped). ---
        let encoding = self
            .tokenizer
            .encode(elements, true)
            .map_err(|err| anyhow::anyhow!("GLiREL prompt tokenization failed: {err}"))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let word_ids = encoding.get_word_ids();
        let seq = input_ids.len();
        let attention_mask = vec![1_i64; seq];

        // --- 4. Encoder ONNX → projected_hidden[1,S,768]. ---
        let input_ids_t =
            Value::from_array((vec![1_i64, seq as i64], input_ids)).context("input_ids tensor")?;
        let attention_mask_t = Value::from_array((vec![1_i64, seq as i64], attention_mask))
            .context("attention_mask tensor")?;

        let proj_hidden: Vec<f32> = {
            let mut encoder = self
                .encoder
                .try_lock_for(std::time::Duration::from_secs(60))
                .ok_or_else(|| anyhow::anyhow!("GLiREL encoder lock timeout"))?;
            let outputs = encoder
                .run(ort::inputs![
                    "input_ids" => &input_ids_t,
                    "attention_mask" => &attention_mask_t,
                ])
                .context("GLiREL encoder inference failed")?;
            let (shape, data) = outputs["projected_hidden"]
                .try_extract_tensor::<f32>()
                .context("extract projected_hidden")?;
            if shape.len() != 3 || shape[2] as usize != HIDDEN {
                anyhow::bail!("unexpected projected_hidden shape {shape:?} (want [1,S,{HIDDEN}])");
            }
            if shape[1] as usize != seq {
                anyhow::bail!(
                    "projected_hidden seq {} disagrees with input seq {seq}",
                    shape[1]
                );
            }
            data.to_vec()
        };

        // --- 5. First-subtoken pooling: one row per pre-split element. ---
        // word_emb[el] = proj_hidden[first subword position whose word_id == el].
        let mut word_emb = vec![0.0f32; num_elements * HIDDEN];
        let mut filled = vec![false; num_elements];
        for (pos, wid) in word_ids.iter().enumerate() {
            let Some(el) = wid.map(|w| w as usize) else {
                continue; // CLS/SEP wrappers (word_id None) are skipped.
            };
            if el >= num_elements || filled[el] {
                continue; // only the FIRST subword of each element is kept.
            }
            let src = pos * HIDDEN;
            let dst = el * HIDDEN;
            word_emb[dst..dst + HIDDEN].copy_from_slice(&proj_hidden[src..src + HIDDEN]);
            filled[el] = true;
        }
        // Every prompt + word element must have been reached (else the tokenizer
        // split disagrees with the encoder graph — a hard error, not a silent drop).
        if let Some(missing) = filled.iter().position(|&f| !f) {
            anyhow::bail!(
                "GLiREL first-pooling missed element {missing} of {num_elements} \
                 (tokenizer/encoder element-alignment mismatch)"
            );
        }

        // --- 6. Slice word_rep / relation_rep, build rel_type_rep_raw. ---
        // word_rep = the n sentence words = elements[prompt_len .. prompt_len+n].
        let mut word_rep = vec![0.0f32; n * HIDDEN];
        let base = prompt_len * HIDDEN;
        word_rep.copy_from_slice(&word_emb[base..base + n * HIDDEN]);
        let word_mask = vec![1_i64; n];

        // relation_rep = elements[0 .. prompt_len-1] (drop the trailing [SEP]) = 2T rows.
        // rel_type_rep_raw[t] = mean(element[2t] ([REL]), element[2t+1] (label)).
        let mut rel_type_rep_raw = vec![0.0f32; t * HIDDEN];
        for ti in 0..t {
            let rel_row = (2 * ti) * HIDDEN; // [REL] element
            let lbl_row = (2 * ti + 1) * HIDDEN; // label element
            let dst = ti * HIDDEN;
            for k in 0..HIDDEN {
                rel_type_rep_raw[dst + k] = 0.5 * (word_emb[rel_row + k] + word_emb[lbl_row + k]);
            }
        }

        // --- 7. span_idx[1,E,2] (inclusive) + relations_idx[1,P,2,2] (ordered i!=j). ---
        let mut span_idx = Vec::with_capacity(e * 2);
        for &(s, en, _) in entities {
            span_idx.push(s as i64);
            span_idx.push(en as i64);
        }
        // All ordered pairs i != j, in (i outer, j inner) order — matching
        // generate_entity_pairs_indices. Each entry = [[h_s,h_e],[t_s,t_e]].
        let mut relations_idx = Vec::new();
        let mut pair_endpoints: Vec<(usize, usize)> = Vec::new();
        for i in 0..e {
            for j in 0..e {
                if i == j {
                    continue;
                }
                let (hs, he, _) = entities[i];
                let (ts, te, _) = entities[j];
                relations_idx.push(hs as i64);
                relations_idx.push(he as i64);
                relations_idx.push(ts as i64);
                relations_idx.push(te as i64);
                pair_endpoints.push((i, j));
            }
        }
        let p = pair_endpoints.len();

        // --- 8. Head ONNX → logits[1,P,T]. ---
        let word_rep_t = Value::from_array((vec![1_i64, n as i64, HIDDEN as i64], word_rep))
            .context("word_rep tensor")?;
        let word_mask_t =
            Value::from_array((vec![1_i64, n as i64], word_mask)).context("word_mask tensor")?;
        let rel_type_rep_t =
            Value::from_array((vec![1_i64, t as i64, HIDDEN as i64], rel_type_rep_raw))
                .context("rel_type_rep_raw tensor")?;
        let span_idx_t = Value::from_array((vec![1_i64, e as i64, 2_i64], span_idx))
            .context("span_idx tensor")?;
        let relations_idx_t =
            Value::from_array((vec![1_i64, p as i64, 2_i64, 2_i64], relations_idx))
                .context("relations_idx tensor")?;

        let (logits_shape, logits): (Vec<i64>, Vec<f32>) = {
            let mut head = self
                .head
                .try_lock_for(std::time::Duration::from_secs(60))
                .ok_or_else(|| anyhow::anyhow!("GLiREL head lock timeout"))?;
            let outputs = head
                .run(ort::inputs![
                    "word_rep" => &word_rep_t,
                    "word_mask" => &word_mask_t,
                    "rel_type_rep_raw" => &rel_type_rep_t,
                    "span_idx" => &span_idx_t,
                    "relations_idx" => &relations_idx_t,
                ])
                .context("GLiREL head inference failed")?;
            let (shape, data) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .context("extract logits")?;
            (shape.to_vec(), data.to_vec())
        };

        if logits_shape.len() != 3
            || logits_shape[1] as usize != p
            || logits_shape[2] as usize != t
        {
            anyhow::bail!("unexpected logits shape {logits_shape:?} (want [1,{p},{t}])");
        }

        // --- 9. sigmoid + emit. logits layout [1,P,T] is row-major over (p,c). ---
        let mut triples = Vec::new();
        for (pi, &(head_idx, tail_idx)) in pair_endpoints.iter().enumerate() {
            for (ci, label) in rel_labels.iter().enumerate() {
                let score = sigmoid(logits[pi * t + ci]);
                if score > threshold {
                    triples.push(RelationTriple {
                        head_idx,
                        tail_idx,
                        relation: (*label).to_string(),
                        score,
                    });
                }
            }
        }
        Ok(triples)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Word-tokenize with the GLiREL regex `\w+(?:[-_]\w+)*|\S`. Each match is one word.
fn word_tokenize(text: &str) -> Vec<String> {
    use regex::Regex;
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"\w+(?:[-_]\w+)*|\S").expect("GLiREL word regex is a valid constant pattern")
    });
    re.find_iter(text).map(|m| m.as_str().to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_tokenize_matches_reference() {
        let w = word_tokenize("HAL manufactures the Tejas aircraft for the Indian Air Force.");
        assert_eq!(
            w,
            vec![
                "HAL",
                "manufactures",
                "the",
                "Tejas",
                "aircraft",
                "for",
                "the",
                "Indian",
                "Air",
                "Force",
                "."
            ]
        );
        let w2 = word_tokenize("Apple acquired Beats, and Tim Cook leads Apple.");
        assert_eq!(
            w2,
            vec![
                "Apple", "acquired", "Beats", ",", "and", "Tim", "Cook", "leads", "Apple", "."
            ]
        );
    }

    /// Parity test vs the Python GLiREL reference. Gated on the model being
    /// present (set `SHODH_GLIREL_MODEL_PATH` to the build dir); skipped
    /// otherwise so CI without the 1.66GB model still passes.
    #[test]
    fn parity_with_python_reference() {
        let Ok(model) = GlirelExtractor::from_env() else {
            eprintln!("SHODH_GLIREL_MODEL_PATH not set — skipping GLiREL parity test");
            return;
        };

        // --- Case 1: HAL / Tejas / Indian Air Force ---
        // entity word spans (inclusive): HAL=[0,0], Tejas=[3,3], IAF=[7,9].
        let text1 = "HAL manufactures the Tejas aircraft for the Indian Air Force.";
        let entities1 = [
            (0usize, 0usize, "company"),
            (3, 3, "aircraft"),
            (7, 9, "government agency"),
        ];
        let labels1 = [
            "manufactures",
            "designed by",
            "operated by",
            "develops",
            "headquartered in",
            "subsidiary of",
            "supplies",
        ];
        let names1 = ["HAL", "Tejas", "Indian Air Force"];
        let triples1 = model
            .extract_relations(text1, &entities1, &labels1, 0.5)
            .expect("case1 extract");
        let got1: Vec<(&str, String, &str, f32)> = triples1
            .iter()
            .map(|r| (names1[r.head_idx], r.relation.clone(), names1[r.tail_idx], r.score))
            .collect();

        // Expected threshold-0.5 triples (head, relation, tail, score) from the reference.
        let expected1: &[(&str, &str, &str, f32)] = &[
            ("Tejas", "manufactures", "HAL", 0.88317),
            ("HAL", "manufactures", "Tejas", 0.882293),
            ("Indian Air Force", "manufactures", "HAL", 0.714816),
            ("HAL", "manufactures", "Indian Air Force", 0.679577),
            ("Indian Air Force", "manufactures", "Tejas", 0.584506),
            ("Tejas", "manufactures", "Indian Air Force", 0.569153),
        ];
        assert_parity("case1", &got1, expected1);

        // --- Case 2: Apple / Beats / Tim Cook / Apple#2 ---
        let text2 = "Apple acquired Beats, and Tim Cook leads Apple.";
        let entities2 = [
            (0usize, 0usize, "company"),
            (2, 2, "company"),
            (5, 6, "person"),
            (8, 8, "company"),
        ];
        let labels2 = [
            "acquired",
            "leads",
            "subsidiary of",
            "founded by",
            "works at",
            "competes with",
        ];
        // names disambiguate the two "Apple" entities by index, as in the reference.
        let names2 = ["Apple", "Beats", "Tim Cook", "Apple"];
        let triples2 = model
            .extract_relations(text2, &entities2, &labels2, 0.5)
            .expect("case2 extract");
        let got2: Vec<(&str, String, &str, f32)> = triples2
            .iter()
            .map(|r| (names2[r.head_idx], r.relation.clone(), names2[r.tail_idx], r.score))
            .collect();

        let expected2: &[(&str, &str, &str, f32)] = &[
            ("Apple", "acquired", "Beats", 0.846394),
            ("Apple", "acquired", "Beats", 0.843122),
            ("Beats", "acquired", "Apple", 0.83182),
            ("Beats", "acquired", "Apple", 0.830068),
            ("Apple", "leads", "Tim Cook", 0.702811),
            ("Apple", "leads", "Tim Cook", 0.693602),
            ("Apple", "acquired", "Apple", 0.631575),
            ("Apple", "acquired", "Apple", 0.630132),
            ("Beats", "leads", "Tim Cook", 0.608801),
            ("Tim Cook", "leads", "Apple", 0.578842),
            ("Tim Cook", "leads", "Apple", 0.574187),
        ];
        assert_parity("case2", &got2, expected2);
    }

    /// Assert that `got` triples match `expected` as multisets: every expected
    /// triple has a got triple with the same (head, relation, tail) and score
    /// within +/-0.05, and no spurious got triples remain unmatched.
    fn assert_parity(
        name: &str,
        got: &[(&str, String, &str, f32)],
        expected: &[(&str, &str, &str, f32)],
    ) {
        let mut used = vec![false; got.len()];
        for &(eh, er, et, es) in expected {
            let mut matched = false;
            for (gi, (gh, gr, gt, gs)) in got.iter().enumerate() {
                if !used[gi] && *gh == eh && gr == er && *gt == et && (gs - es).abs() <= 0.05 {
                    used[gi] = true;
                    matched = true;
                    break;
                }
            }
            assert!(
                matched,
                "[{name}] missing expected triple {eh:?} --{er}--> {et:?} (~{es:.4}); got = {got:?}"
            );
        }
        let spurious: Vec<_> = got
            .iter()
            .zip(used.iter())
            .filter(|(_, &u)| !u)
            .map(|(g, _)| g)
            .collect();
        assert!(
            spurious.is_empty(),
            "[{name}] spurious triples above threshold not in reference: {spurious:?}"
        );
        assert_eq!(
            got.len(),
            expected.len(),
            "[{name}] triple count {} != expected {}",
            got.len(),
            expected.len()
        );
    }
}
