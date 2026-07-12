//! GLiNER bi-encoder span typing (ONNX Runtime).
//!
//! Rust runtime for `knowledgator/gliner-bi-edge-v2.0`, a `BiEncoderSpanGLiNER`.
//! The shipped ONNX graph is the TEXT tower + span/label bilinear scorer; the
//! entity-type LABEL tower (MiniLM) was run ONCE offline, so its 141 fine-label
//! embeddings ship as `label_embeddings.bin` and are fed as the `labels_embeds`
//! input. Rust never runs the label tower — cheap, deterministic, on-device.
//!
//! This module replicates, in Rust, the exact 7-input construction and span
//! decode that `scripts/export_gliner_bi_edge.py` + gliner's `BiEncoderSpanProcessor`
//! / `SpanDecoder` perform in Python, so the output matches the Python parity probe
//! (fp32 span-F1 = 1.000 vs torch).
//!
//! # The 7 ONNX inputs (all built here, per text)
//! - `input_ids`      `[1, seq]`  i64 — subword ids of the whitespace-split words,
//!   tokenized pre-tokenized (`is_split_into_words`).
//! - `attention_mask` `[1, seq]`  i64 — all ones (single unpadded text).
//! - `words_mask`     `[1, seq]`  i64 — 1-based word index at the FIRST subword of
//!   each word, 0 for continuation subwords and special tokens
//!   (gliner `prepare_word_mask`, skip=0, token_level=False).
//! - `text_lengths`   `[1, 1]`    i64 — number of words (gliner `seq_length`).
//! - `span_idx`       `[1, W*Kw, 2]` i64 — every `(start, start+offset)` span for
//!   `start in 0..W`, `offset in 0..max_width` (gliner `prepare_span_idx`).
//! - `span_mask`      `[1, W*Kw]` bool — span end `<= W-1` (in-range), else false.
//! - `labels_embeds`  `[141, 384]` f32 — the precomputed fine-label embeddings.
//!
//! # Output / decode
//! `logits [1, W, max_width, 141]` (f32). Decode = gliner `SpanDecoder`:
//! sigmoid → keep `(start, width, class)` with `prob > threshold` and
//! `start+width+1 <= W`; greedy non-overlapping selection by descending score
//! (flat NER, single-label); map class → fine label (schema order) → coarse.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

use crate::graph_memory::EntityLabel;

/// A typed entity span produced by [`GlinerTyper::extract`].
#[derive(Debug, Clone)]
pub struct TypedSpan {
    /// Surface text of the span, sliced from the original input.
    pub text: String,
    /// GLiNER fine label (schema `fine[].label`, e.g. `"cargo ship"`, `"bridge"`).
    pub fine_label: String,
    /// Coarse rollup of `fine_label` (`crate::entity_type::coarse_of` →
    /// [`EntityLabel::from_coarse_id`]).
    pub coarse: EntityLabel,
    /// Sigmoid confidence of the span/label match (0.0–1.0).
    pub score: f32,
    /// Byte offset of the span start in the original text.
    pub start: usize,
    /// Byte offset of the span end (exclusive) in the original text.
    pub end: usize,
}

/// Configuration for the GLiNER bi-edge typer.
#[derive(Debug, Clone)]
pub struct GlinerConfig {
    /// Path to the fp32 ONNX graph (`model.onnx`).
    pub model_path: PathBuf,
    /// Path to the text-tower tokenizer (`tokenizer.json`).
    pub tokenizer_path: PathBuf,
    /// Path to the precomputed fine-label embeddings (`label_embeddings.bin`).
    pub label_embeddings_path: PathBuf,
    /// Minimum sigmoid probability to keep a span (parity probe used 0.3).
    pub threshold: f32,
    /// Maximum span width in words (gliner `config.max_width`; bi-edge = 12).
    pub max_width: usize,
    /// Maximum number of words per text (gliner `config.max_len`; bi-edge = 2048).
    pub max_len: usize,
}

impl Default for GlinerConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl GlinerConfig {
    /// Resolve configuration from the environment.
    ///
    /// Model directory: `SHODH_GLINER_MODEL_PATH`, else the first of a small set
    /// of conventional locations that actually contains `model.onnx`, else the
    /// local `./models/gliner-bi-edge`.
    pub fn from_env() -> Self {
        let base_path = std::env::var("SHODH_GLINER_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let candidates: Vec<Option<PathBuf>> = vec![
                    std::env::var("SHODH_PACKAGE_DIR")
                        .ok()
                        .map(|p| PathBuf::from(p).join("models/gliner-bi-edge")),
                    Some(PathBuf::from("./models/gliner-bi-edge")),
                    Some(PathBuf::from("../models/gliner-bi-edge")),
                    dirs::data_dir().map(|p| p.join("shodh-memory/models/gliner-bi-edge")),
                ];
                candidates
                    .into_iter()
                    .flatten()
                    .find(|p| p.join("model.onnx").exists())
                    .unwrap_or_else(|| PathBuf::from("./models/gliner-bi-edge"))
            });

        let threshold = std::env::var("SHODH_GLINER_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.3);

        Self {
            model_path: base_path.join("model.onnx"),
            tokenizer_path: base_path.join("tokenizer.json"),
            label_embeddings_path: base_path.join("label_embeddings.bin"),
            threshold,
            max_width: 12,
            max_len: 2048,
        }
    }

    /// True when every asset needed to run the typer is present on disk.
    pub fn assets_present(&self) -> bool {
        self.model_path.exists()
            && self.tokenizer_path.exists()
            && self.label_embeddings_path.exists()
    }
}

/// Lazily initialized ONNX session, tokenizer, and precomputed label embeddings.
struct LazyGlinerModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// `[num_labels * hidden]` row-major f32, fine-label (schema) order.
    label_embeds: Vec<f32>,
    /// Fine labels in schema order — index `c` is the class-`c` label.
    fine_labels: Vec<String>,
    num_labels: usize,
    hidden: usize,
}

impl LazyGlinerModel {
    fn new(config: &GlinerConfig) -> Result<Self> {
        // GUARD (upgrade-panic fix): every ort session creation must go through
        // the shared ORT_DYLIB_PATH guard, or a bare-name dylib load can pick up
        // an incompatible system onnxruntime and poison ort's global mutex.
        let offline_mode = std::env::var("SHODH_OFFLINE")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        super::minilm::pre_init_ort_runtime(offline_mode);

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let default_threads = 1;
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let default_threads = 2;

        let num_threads = std::env::var("SHODH_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_threads);

        tracing::info!(
            "Loading GLiNER bi-edge model from {:?} with {} threads",
            config.model_path,
            num_threads
        );

        let builder = Session::builder()
            .context("Failed to create GLiNER session builder")?
            .with_intra_threads(num_threads)
            .context("Failed to set GLiNER intra thread count")?
            .with_inter_threads(1)
            .context("Failed to set GLiNER inter thread count")?;

        // Disable thread pool spinning (Eigen spin-to-block deadlock on macOS
        // ARM64 heterogeneous cores). See microsoft/onnxruntime#10270, pykeio/ort#516.
        let builder = builder
            .with_intra_op_spinning(false)
            .context("Failed to disable GLiNER intra-op spinning")?
            .with_inter_op_spinning(false)
            .context("Failed to disable GLiNER inter-op spinning")?;

        let session = builder
            .commit_from_file(&config.model_path)
            .context("Failed to load GLiNER ONNX model")?;

        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load GLiNER tokenizer: {e}"))?;

        // Fine labels in schema order — the SAME order the export wrote
        // label_embeddings.bin rows in, so class index c ↔ fine_labels[c].
        let fine_labels: Vec<String> = crate::entity_type::fine_labels()
            .into_iter()
            .map(str::to_string)
            .collect();
        let num_labels = fine_labels.len();
        if num_labels == 0 {
            anyhow::bail!("entity-type schema exposed zero fine labels");
        }

        let (label_embeds, hidden) =
            load_label_embeddings(&config.label_embeddings_path, num_labels)?;

        tracing::info!(
            "GLiNER bi-edge loaded: {num_labels} labels × {hidden} dims, threshold {}",
            config.threshold
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            label_embeds,
            fine_labels,
            num_labels,
            hidden,
        })
    }
}

/// Read `label_embeddings.bin` (raw little-endian f32, `[num_labels, hidden]`
/// row-major). Returns the flat buffer and the inferred `hidden` size.
fn load_label_embeddings(path: &Path, num_labels: usize) -> Result<(Vec<f32>, usize)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read label embeddings from {path:?}"))?;
    if bytes.len() % 4 != 0 {
        anyhow::bail!(
            "label embeddings file {path:?} is not a whole number of f32 values ({} bytes)",
            bytes.len()
        );
    }
    let total = bytes.len() / 4;
    if total == 0 || total % num_labels != 0 {
        anyhow::bail!(
            "label embeddings file {path:?} has {total} f32 values, not divisible by \
             {num_labels} fine labels — schema/asset mismatch"
        );
    }
    let hidden = total / num_labels;
    let mut floats = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((floats, hidden))
}

/// GLiNER bi-encoder span typer.
///
/// Loads the fp32 text-tower ONNX + precomputed fine-label embeddings once (on
/// first [`extract`](Self::extract)) and types arbitrary text against the 141
/// fine labels of the entity-type schema.
pub struct GlinerTyper {
    config: GlinerConfig,
    lazy_model: OnceLock<Result<Arc<LazyGlinerModel>, String>>,
    /// Whether the required assets were present at construction.
    available: bool,
}

impl GlinerTyper {
    /// Create a typer from the given configuration. Never fails: if assets are
    /// missing the typer is created in an unavailable state and
    /// [`extract`](Self::extract) returns an empty vector.
    pub fn new(config: GlinerConfig) -> Self {
        let available = config.assets_present();
        if !available {
            tracing::warn!(
                "GLiNER bi-edge assets not found at {:?} — typer disabled (extract returns empty)",
                config.model_path
            );
        }
        Self {
            config,
            lazy_model: OnceLock::new(),
            available,
        }
    }

    /// Create a typer from environment configuration.
    pub fn from_env() -> Self {
        Self::new(GlinerConfig::from_env())
    }

    /// True when the model, tokenizer, and label embeddings are all present.
    pub fn is_available(&self) -> bool {
        self.available
    }

    fn ensure_model_loaded(&self) -> Result<&Arc<LazyGlinerModel>> {
        if !self.available {
            anyhow::bail!("GLiNER bi-edge assets not available");
        }
        let result = self.lazy_model.get_or_init(|| {
            LazyGlinerModel::new(&self.config)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });
        match result {
            Ok(model) => Ok(model),
            Err(e) => Err(anyhow::anyhow!("Failed to load GLiNER model: {e}")),
        }
    }

    /// Type `text` and return the non-overlapping fine-typed spans.
    ///
    /// Best-effort: returns an empty vector on empty input, missing assets, or
    /// any inference error (logged), so callers can treat the typer as an
    /// optional enrichment stage.
    pub fn extract(&self, text: &str) -> Vec<TypedSpan> {
        if text.trim().is_empty() {
            return Vec::new();
        }
        match self.extract_inner(text) {
            Ok(spans) => spans,
            Err(e) => {
                tracing::warn!("GLiNER typing failed: {e}");
                Vec::new()
            }
        }
    }

    fn extract_inner(&self, text: &str) -> Result<Vec<TypedSpan>> {
        let model = self.ensure_model_loaded()?;

        // 1. Word split (gliner WhitespaceTokenSplitter: `\w+(?:[-_]\w+)*|\S`).
        let mut words = whitespace_split(text);
        if words.is_empty() {
            return Ok(Vec::new());
        }
        if words.len() > self.config.max_len {
            words.truncate(self.config.max_len);
        }
        let num_words = words.len();
        let max_width = self.config.max_width;

        // 2. Tokenize the words as a pre-tokenized sequence (is_split_into_words),
        //    then build words_mask from the subword→word map.
        let word_refs: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        let encoding = model
            .tokenizer
            .encode(word_refs, true)
            .map_err(|e| anyhow::anyhow!("GLiNER tokenization failed: {e}"))?;

        let ids = encoding.get_ids();
        let word_ids = encoding.get_word_ids();
        let seq = ids.len();

        let input_ids: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
        let attention_mask: Vec<i64> = vec![1i64; seq];
        let words_mask = build_words_mask(word_ids);

        // 3. text_lengths = [[num_words]].
        let text_lengths: Vec<i64> = vec![num_words as i64];

        // 4/5. Enumerate spans and the in-range mask.
        let num_spans = num_words * max_width;
        let mut span_idx: Vec<i64> = Vec::with_capacity(num_spans * 2);
        let mut span_mask: Vec<bool> = Vec::with_capacity(num_spans);
        for start in 0..num_words {
            for offset in 0..max_width {
                let end = start + offset;
                span_idx.push(start as i64);
                span_idx.push(end as i64);
                // gliner: span_label != -1, where invalid = (end > num_words - 1).
                span_mask.push(end < num_words);
            }
        }

        // 6. labels_embeds (precomputed fine-label embeddings).
        let labels_embeds = model.label_embeds.clone();

        // Build tensors and run.
        let v_input_ids = Value::from_array((vec![1usize, seq], input_ids))
            .context("GLiNER input_ids tensor")?;
        let v_attention = Value::from_array((vec![1usize, seq], attention_mask))
            .context("GLiNER attention_mask tensor")?;
        let v_words_mask = Value::from_array((vec![1usize, seq], words_mask))
            .context("GLiNER words_mask tensor")?;
        let v_text_lengths = Value::from_array((vec![1usize, 1usize], text_lengths))
            .context("GLiNER text_lengths tensor")?;
        let v_span_idx = Value::from_array((vec![1usize, num_spans, 2usize], span_idx))
            .context("GLiNER span_idx tensor")?;
        let v_span_mask = Value::from_array((vec![1usize, num_spans], span_mask))
            .context("GLiNER span_mask tensor")?;
        let v_labels = Value::from_array((vec![model.num_labels, model.hidden], labels_embeds))
            .context("GLiNER labels_embeds tensor")?;

        let mut session = model
            .session
            .try_lock_for(std::time::Duration::from_secs(30))
            .ok_or_else(|| anyhow::anyhow!("GLiNER session lock timeout (30s)"))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => &v_input_ids,
                "attention_mask" => &v_attention,
                "words_mask" => &v_words_mask,
                "text_lengths" => &v_text_lengths,
                "span_idx" => &v_span_idx,
                "span_mask" => &v_span_mask,
                "labels_embeds" => &v_labels,
            ])
            .context("GLiNER inference failed")?;

        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract GLiNER logits")?;

        // logits: [batch=1, L(words), K(max_width), C(num_labels)].
        if shape.len() != 4 {
            anyhow::bail!("GLiNER logits rank {} != 4 (shape {:?})", shape.len(), &shape[..]);
        }
        let l_dim = shape[1] as usize;
        let k_dim = shape[2] as usize;
        let c_dim = shape[3] as usize;

        Ok(self.decode(text, &words, num_words, logits, l_dim, k_dim, c_dim, model))
    }

    /// gliner `SpanDecoder` decode: threshold on sigmoid probabilities, drop
    /// out-of-range spans, then greedy non-overlapping selection (flat NER).
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        text: &str,
        words: &[Word],
        num_words: usize,
        logits: &[f32],
        l_dim: usize,
        k_dim: usize,
        c_dim: usize,
        model: &LazyGlinerModel,
    ) -> Vec<TypedSpan> {
        let threshold = self.config.threshold;

        // Candidate (start, end_word_inclusive, class, score) above threshold.
        struct Cand {
            start: usize,
            end: usize,
            class: usize,
            score: f32,
        }
        let mut cands: Vec<Cand> = Vec::new();

        for s in 0..l_dim {
            for k in 0..k_dim {
                // Valid span: start + width + 1 <= num_words (== end <= num_words-1).
                if s + k + 1 > num_words {
                    continue;
                }
                let base = (s * k_dim + k) * c_dim;
                for c in 0..c_dim {
                    let p = sigmoid(logits[base + c]);
                    if p > threshold {
                        cands.push(Cand {
                            start: s,
                            end: s + k,
                            class: c,
                            score: p,
                        });
                    }
                }
            }
        }

        // greedy_search (flat NER, single-label): sort by descending score, keep a
        // candidate only if it does not overlap any already-selected span. Overlap
        // (gliner `has_overlapping`): identical (start,end) OR intervals intersect.
        cands.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut selected: Vec<Cand> = Vec::new();
        for cand in cands {
            let overlaps = selected.iter().any(|sel| {
                if sel.start == cand.start && sel.end == cand.end {
                    true // same span → single-label dedup
                } else {
                    !(cand.start > sel.end || sel.start > cand.end)
                }
            });
            if !overlaps {
                selected.push(cand);
            }
        }

        // Map to char offsets + labels, sorted by start (gliner returns by start).
        selected.sort_by_key(|c| c.start);
        let mut spans = Vec::with_capacity(selected.len());
        for cand in selected {
            if cand.start >= num_words || cand.end >= num_words {
                continue;
            }
            let fine_label = model
                .fine_labels
                .get(cand.class)
                .cloned()
                .unwrap_or_default();
            if fine_label.is_empty() {
                continue;
            }
            let coarse = crate::entity_type::coarse_of(&fine_label)
                .map(EntityLabel::from_coarse_id)
                .unwrap_or_else(|| EntityLabel::Other(fine_label.clone()));
            let start_byte = words[cand.start].start;
            let end_byte = words[cand.end].end;
            let surface = text.get(start_byte..end_byte).unwrap_or("").to_string();
            spans.push(TypedSpan {
                text: surface,
                fine_label,
                coarse,
                score: cand.score,
                start: start_byte,
                end: end_byte,
            });
        }
        spans
    }
}

/// One whitespace-split word with its byte offsets in the original text.
struct Word {
    text: String,
    start: usize,
    end: usize,
}

/// Regex matching gliner's `WhitespaceTokenSplitter`: `\w+(?:[-_]\w+)*|\S`.
static WORD_RE: OnceLock<regex::Regex> = OnceLock::new();

fn word_regex() -> &'static regex::Regex {
    WORD_RE.get_or_init(|| {
        regex::Regex::new(r"\w+(?:[-_]\w+)*|\S").expect("static GLiNER word regex is valid")
    })
}

/// Split `text` into words with byte offsets, matching gliner's whitespace splitter.
fn whitespace_split(text: &str) -> Vec<Word> {
    word_regex()
        .find_iter(text)
        .map(|m| Word {
            text: m.as_str().to_string(),
            start: m.start(),
            end: m.end(),
        })
        .collect()
}

/// Build gliner `words_mask` (skip_first_words=0, token_level=False): the 1-based
/// word index at the first subword of each word, 0 for continuation subwords and
/// special tokens.
fn build_words_mask(word_ids: &[Option<u32>]) -> Vec<i64> {
    let mut mask = Vec::with_capacity(word_ids.len());
    let mut prev: Option<u32> = None;
    let mut seen: i64 = 0;
    for &wid in word_ids {
        match wid {
            None => mask.push(0),
            Some(w) => {
                if Some(w) != prev {
                    seen += 1;
                    mask.push(seen);
                } else {
                    mask.push(0);
                }
            }
        }
        prev = wid;
    }
    mask
}

/// Numerically-stable logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn whitespace_split_matches_gliner_tokenizer() {
        let words = whitespace_split("The Dali lost power.");
        let toks: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(toks, vec!["The", "Dali", "lost", "power", "."]);
        // Offsets must slice back to the exact surface.
        for w in &words {
            assert_eq!(&"The Dali lost power."[w.start..w.end], w.text);
        }
    }

    #[test]
    fn whitespace_split_keeps_hyphenated_words() {
        let words = whitespace_split("state-of-the-art bi_encoder, ok");
        let toks: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(toks, vec!["state-of-the-art", "bi_encoder", ",", "ok"]);
    }

    #[test]
    fn words_mask_marks_first_subword_only() {
        // Two words: first splits into 2 subwords, second into 1, with CLS/SEP.
        // word_ids: [None, 0, 0, 1, None]
        let wm = build_words_mask(&[None, Some(0), Some(0), Some(1), None]);
        assert_eq!(wm, vec![0, 1, 0, 2, 0]);
    }

    #[test]
    fn sigmoid_is_monotonic_and_bounded() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(20.0) > 0.99);
        assert!(sigmoid(-20.0) < 0.01);
        assert!(sigmoid(1.0) > sigmoid(0.5));
    }

    /// PARITY HARNESS — replicates the Python bi-edge probe on the GDELT
    /// Baltimore-bridge diagnostic sentence. Requires the real fp32 assets; when
    /// they are absent (e.g. CI without the 149MB model) the test is a no-op with
    /// a clear message, matching the repo's model-dependent test convention.
    #[test]
    fn gliner_types_diagnostic_sentence() {
        let config = GlinerConfig::from_env();
        if !config.assets_present() {
            eprintln!(
                "SKIP gliner_types_diagnostic_sentence: assets not found at {:?}",
                config.model_path
            );
            return;
        }

        let typer = GlinerTyper::new(config);
        let text =
            "A cargo ship rammed the Francis Scott Key Bridge in Baltimore. The Dali lost power.";
        let spans = typer.extract(text);
        assert!(
            !spans.is_empty(),
            "GLiNER produced no spans on the diagnostic sentence"
        );

        // Look up the fine label the typer assigned to a given surface (case-insensitive).
        let label_for = |surface: &str| -> Option<String> {
            spans
                .iter()
                .find(|s| s.text.eq_ignore_ascii_case(surface))
                .map(|s| s.fine_label.clone())
        };

        // Baltimore → city (coarse gpe).
        let baltimore = label_for("Baltimore");
        assert_eq!(
            baltimore.as_deref(),
            Some("city"),
            "Baltimore should type as city, got {baltimore:?}; all spans: {:?}",
            debug_spans(&spans)
        );

        // Francis Scott Key Bridge → bridge (coarse facility).
        let bridge = label_for("Francis Scott Key Bridge");
        assert_eq!(
            bridge.as_deref(),
            Some("bridge"),
            "Francis Scott Key Bridge should type as bridge, got {bridge:?}; all spans: {:?}",
            debug_spans(&spans)
        );

        // Dali → the ship's name. Accept the vessel family (schema carries a
        // dedicated `cargo ship`; parity KEY_ENTITIES allows the family).
        let ship_family: HashSet<&str> =
            ["ship", "cargo ship", "warship", "vessel", "watercraft"]
                .into_iter()
                .collect();
        let dali = label_for("Dali");
        assert!(
            dali.as_deref().is_some_and(|l| ship_family.contains(l)),
            "Dali should type as a ship, got {dali:?}; all spans: {:?}",
            debug_spans(&spans)
        );

        // cargo ship → ship family (the schema's dedicated `cargo ship` is ideal).
        let cargo = label_for("cargo ship");
        assert!(
            cargo.as_deref().is_some_and(|l| ship_family.contains(l)),
            "cargo ship should type as a ship, got {cargo:?}; all spans: {:?}",
            debug_spans(&spans)
        );

        // Coarse rollup sanity: Baltimore's coarse is a real schema variant, not Other.
        if let Some(b) = spans.iter().find(|s| s.text.eq_ignore_ascii_case("Baltimore")) {
            assert!(
                !matches!(b.coarse, EntityLabel::Other(_)),
                "Baltimore coarse should roll up to a schema variant, got {:?}",
                b.coarse
            );
        }
    }

    fn debug_spans(spans: &[TypedSpan]) -> Vec<(String, String, f32)> {
        spans
            .iter()
            .map(|s| (s.text.clone(), s.fine_label.clone(), s.score))
            .collect()
    }

    /// PRODUCTION-TYPER GATE — the "everything is one bucket" symptom must be
    /// dead on real ingest. Types the 100 real GDELT Baltimore-bridge passages
    /// with the production `GlinerTyper` and asserts the coarse-label distribution
    /// is genuinely spread (no single coarse bucket dominates), and that
    /// "Baltimore" types as a geopolitical/location entity — not Technology or a
    /// generic Concept, which is exactly what the deleted bert-tiny + MISC-regex
    /// path produced.
    ///
    /// Model-gated (skip-with-log when the fp32 assets are absent), matching the
    /// repo's model-dependent test convention.
    #[test]
    fn gliner_spreads_coarse_types_on_gdelt_passages() {
        use std::collections::HashMap;

        let config = GlinerConfig::from_env();
        if !config.assets_present() {
            eprintln!(
                "SKIP gliner_spreads_coarse_types_on_gdelt_passages: assets not found at {:?}",
                config.model_path
            );
            return;
        }

        let passages_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("demos/gdelt-bridge/passages_100.jsonl");
        let raw = match std::fs::read_to_string(&passages_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "SKIP gliner_spreads_coarse_types_on_gdelt_passages: cannot read {passages_path:?}: {e}"
                );
                return;
            }
        };

        let typer = GlinerTyper::new(config);

        // Tally coarse EntityLabel buckets over every typed span in the corpus.
        let mut bucket_counts: HashMap<String, usize> = HashMap::new();
        let mut total_typed = 0usize;
        let mut baltimore_coarse: Option<EntityLabel> = None;

        for line in raw.lines().filter(|l| !l.trim().is_empty()) {
            let doc: serde_json::Value =
                serde_json::from_str(line).expect("passages_100.jsonl line is valid JSON");
            let Some(text) = doc.get("text").and_then(|t| t.as_str()) else {
                continue;
            };
            for span in typer.extract(text) {
                total_typed += 1;
                *bucket_counts
                    .entry(format!("{:?}", span.coarse))
                    .or_default() += 1;
                if baltimore_coarse.is_none() && span.text.eq_ignore_ascii_case("Baltimore") {
                    baltimore_coarse = Some(span.coarse.clone());
                }
            }
        }

        assert!(
            total_typed >= 100,
            "expected the 100 GDELT passages to yield many typed entities, got {total_typed}"
        );

        // Distribution report (surfaced on failure).
        let mut dist: Vec<(String, usize)> = bucket_counts.into_iter().collect();
        dist.sort_by(|a, b| b.1.cmp(&a.1));
        let report: Vec<String> = dist
            .iter()
            .map(|(k, c)| format!("{k}={c} ({:.1}%)", *c as f64 / total_typed as f64 * 100.0))
            .collect();
        eprintln!("GLiNER coarse distribution ({total_typed} typed spans): {report:?}");

        // (a) No single coarse bucket may dominate. The old bert-tiny + MISC-regex
        // path funnelled ~everything into one bucket; the fine-typed schema spreads
        // the mass across the 18 coarse classes. Measured top bucket on this corpus
        // is Facility at ~16% (a bridge-collapse story), comfortably under the 20%
        // gate — 18 distinct coarse buckets appear.
        let (top_label, top_count) = dist.first().cloned().expect("at least one bucket");
        let top_frac = top_count as f64 / total_typed as f64;
        assert!(
            top_frac <= 0.20,
            "coarse types must be spread, but '{top_label}' holds {:.1}% of {total_typed} typed \
             entities (> 20%). Full distribution: {report:?}",
            top_frac * 100.0
        );

        // (b) Baltimore is a city → gpe/location, never Technology/Concept.
        let baltimore = baltimore_coarse
            .expect("expected 'Baltimore' to be typed somewhere in the bridge corpus");
        assert!(
            matches!(baltimore, EntityLabel::Gpe | EntityLabel::Location),
            "Baltimore must type as Gpe/Location, got {baltimore:?}. Distribution: {report:?}"
        );
    }
}
