//! Padded-vs-dynamic sequence-length equivalence probe for the MiniLM embed pass.
//!
//! QUESTION UNDER TEST: `generate_embedding_onnx` pads every input to
//! `max_length` (256). Padded positions carry `attention_mask = 0` and the
//! Rust-side mean-pool already skips them — so IF the exported ONNX graph
//! masks correctly AND its kernels are shape-independent, shortening the
//! sequence must produce BIT-IDENTICAL embeddings. That is an assumption
//! about the export, not a fact. This probe settles it empirically before
//! any dynamic-padding change ships.
//!
//! Discovered context the probe accounts for:
//!   - tokenizer.json bakes in Fixed(128) padding + truncation at 128, so
//!     every encoding is exactly 128 ids (mask 0 on tokenizer pad). The Rust
//!     code then re-pads to 256 — so the current forward is 256 positions of
//!     which at most 128 are real. "True length" below = attention-mask sum.
//!   - Two DIFFERENT model files exist locally: the repo-adjacent
//!     `../models/minilm-l6/model_quantized.onnx` is mislabeled fp32 (what
//!     the local recall gate resolves to), while the auto-download cache has
//!     the real quint8_avx2 dynamic-quantized export (what fresh installs
//!     run). Both are probed.
//!
//! Protocol (per advisor review):
//!   0. Determinism control: same input, same shape, run twice — if not
//!      bit-stable, cross-shape comparison is unattributable noise.
//!   1. Primary verdict: raw `last_hidden_state` bytes at masked-in positions,
//!      256-pad vs {128, true-length, bucket-32}.
//!   2. Secondary: final pooled+normalized vectors through the exact pooling
//!      arithmetic copied from `minilm.rs` (same summation order).
//!   3. Latency: `session.run` steady-state per shape, plus first-call cost
//!      per new shape (the ORT re-planning cost that motivates bucketing).
//!
//! Deliberately `#[ignore]`d — needs local model files (like onnx_memory_probe):
//!   cargo test --test embed_pad_equivalence -- --ignored --nocapture

use ort::session::Session;
use ort::value::Value;
use shodh_memory::embeddings::minilm::{pre_init_ort_runtime, EmbeddingConfig};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Exact copy of the pooling + finalize arithmetic from
/// `MiniLMEmbedder::generate_embedding_onnx` / `finalize_pooled` for the
/// native-384 (MiniLM) path: masked mean-pool in seq order, NaN/Inf scrub,
/// L2 normalize. Kept byte-for-byte equivalent so any diff is attributable
/// to the ONNX outputs, not harness arithmetic.
fn pool_and_finalize(output_data: &[f32], attention: &[i64], hidden: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; hidden];
    let mut mask_sum = 0.0f32;
    for (seq_idx, &att) in attention.iter().enumerate() {
        if att == 1 {
            for (dim_idx, pooled_val) in pooled.iter_mut().enumerate() {
                let idx = seq_idx * hidden + dim_idx;
                *pooled_val += output_data[idx];
            }
            mask_sum += 1.0;
        }
    }
    if mask_sum > 0.0 {
        for val in &mut pooled {
            *val /= mask_sum;
        }
    }
    for val in pooled.iter_mut() {
        if val.is_nan() || val.is_infinite() {
            *val = 0.0;
        }
    }
    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON && !norm.is_nan() {
        for val in &mut pooled {
            *val /= norm;
        }
    }
    pooled
}

struct RunResult {
    /// Raw last_hidden_state, row-major [seq_len, hidden].
    hidden_state: Vec<f32>,
    /// Final pooled+normalized embedding via the minilm.rs arithmetic.
    embedding: Vec<f32>,
    /// session.run wall time.
    run_time: std::time::Duration,
}

/// Run one forward pass at a given padded length. Token/mask content beyond
/// `padded_len` is dropped (mirrors `take(max_length)` truncation).
fn run_at_length(
    session: &mut Session,
    tokens: &[u32],
    mask: &[u32],
    padded_len: usize,
    hidden: usize,
    wants_token_type: bool,
) -> RunResult {
    let n = tokens.len().min(padded_len);
    let mut input_ids = vec![0i64; padded_len];
    let mut attention = vec![0i64; padded_len];
    let token_type_ids = vec![0i64; padded_len];
    for i in 0..n {
        input_ids[i] = tokens[i] as i64;
        attention[i] = mask[i] as i64;
    }

    let input_ids_value = Value::from_array((vec![1, padded_len], input_ids)).unwrap();
    let attention_mask_value = Value::from_array((vec![1, padded_len], attention.clone())).unwrap();
    let token_type_ids_value = Value::from_array((vec![1, padded_len], token_type_ids)).unwrap();

    let start = Instant::now();
    let outputs = if wants_token_type {
        session
            .run(ort::inputs![
                "input_ids" => &input_ids_value,
                "attention_mask" => &attention_mask_value,
                "token_type_ids" => &token_type_ids_value,
            ])
            .unwrap()
    } else {
        session
            .run(ort::inputs![
                "input_ids" => &input_ids_value,
                "attention_mask" => &attention_mask_value,
            ])
            .unwrap()
    };
    let run_time = start.elapsed();

    let (_shape, output_data) = outputs[0].try_extract_tensor::<f32>().unwrap();
    RunResult {
        hidden_state: output_data.to_vec(),
        embedding: pool_and_finalize(output_data, &attention, hidden),
        run_time,
    }
}

/// Bitwise comparison of two f32 slices. Returns (n_differing, max_abs_diff).
fn bitwise_diff(a: &[f32], b: &[f32]) -> (usize, f32) {
    assert_eq!(a.len(), b.len());
    let mut n_diff = 0usize;
    let mut max_abs = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            n_diff += 1;
            let d = (x - y).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    (n_diff, max_abs)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn bucket32(len: usize, max_length: usize) -> usize {
    (len.div_ceil(32) * 32).min(max_length)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Aggregate diff stats for one candidate shape vs the 256-pad reference.
#[derive(Default)]
struct ShapeVerdict {
    texts_hidden_diff: usize,
    texts_embed_diff: usize,
    worst_hidden: f32,
    worst_embed: f32,
    worst_cos: f64,
    latencies_ms: Vec<f64>,
}

fn probe_model(model_path: &std::path::Path, tokenizer_path: &std::path::Path, max_length: usize) {
    let hidden = 384usize;
    println!("\n################ MODEL: {model_path:?} ################");

    let mut session = Session::builder()
        .unwrap()
        .with_intra_threads(2)
        .unwrap()
        .with_inter_threads(1)
        .unwrap()
        .with_intra_op_spinning(false)
        .unwrap()
        .with_inter_op_spinning(false)
        .unwrap()
        .commit_from_file(model_path)
        .unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let wants_token_type = session
        .inputs()
        .iter()
        .any(|i| i.name() == "token_type_ids");

    // ── Corpus: all 100 real locomo-gate queries + longer texts spanning the
    //    length range up to and beyond the tokenizer's 128 truncation. ────────
    let gate_cases = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/recall/locomo-gate-cases.jsonl"),
    )
    .expect("locomo-gate-cases.jsonl");
    let mut texts: Vec<String> = gate_cases
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["query"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    let base = "The quarterly report flagged unusual sensor drift near the northern \
                perimeter, and maintenance scheduled a full recalibration sweep. ";
    for reps in [1usize, 2, 3, 5, 8] {
        texts.push(base.repeat(reps));
    }

    // ── 0. Determinism control: same text, same shape, twice. ───────────────
    let enc = tokenizer.encode(texts[0].as_str(), true).unwrap();
    let (toks, msk) = (enc.get_ids().to_vec(), enc.get_attention_mask().to_vec());
    // warm up the 256 shape so the control measures steady state
    let _ = run_at_length(
        &mut session,
        &toks,
        &msk,
        max_length,
        hidden,
        wants_token_type,
    );
    let c1 = run_at_length(
        &mut session,
        &toks,
        &msk,
        max_length,
        hidden,
        wants_token_type,
    );
    let c2 = run_at_length(
        &mut session,
        &toks,
        &msk,
        max_length,
        hidden,
        wants_token_type,
    );
    let (nd_h, mx_h) = bitwise_diff(&c1.hidden_state, &c2.hidden_state);
    println!(
        "[control] same-shape repeat: hidden diffs = {nd_h}/{} (max {mx_h:e})",
        c1.hidden_state.len()
    );
    let same_shape_stable = nd_h == 0;

    // ── 1+2. Cross-shape comparison over the whole corpus. ──────────────────
    // Candidate shapes per text: encoding length (=128, tokenizer-padded),
    // true length (mask sum), bucket32(true length).
    let mut verdicts: [ShapeVerdict; 3] = Default::default();
    let shape_names = ["enc-len(128)", "true-len", "bucket-32"];
    for v in verdicts.iter_mut() {
        v.worst_cos = 1.0;
    }
    let mut lat_256 = Vec::new();
    let mut true_lens = Vec::new();
    let mut first_call_per_shape: Vec<(usize, f64)> = Vec::new();
    let mut seen_shapes = std::collections::HashSet::new();
    seen_shapes.insert(max_length);

    for (i, text) in texts.iter().enumerate() {
        let enc = tokenizer.encode(text.as_str(), true).unwrap();
        let toks = enc.get_ids().to_vec();
        let msk = enc.get_attention_mask().to_vec();
        let enc_len = toks.len().min(max_length);
        let true_len = msk.iter().filter(|&&m| m == 1).count().min(max_length);
        true_lens.push(true_len);
        let shapes = [enc_len, true_len, bucket32(true_len, max_length)];

        // Track first-call (re-planning) cost per new shape.
        for &len in &shapes {
            if seen_shapes.insert(len) {
                let r = run_at_length(&mut session, &toks, &msk, len, hidden, wants_token_type);
                first_call_per_shape.push((len, r.run_time.as_secs_f64() * 1e3));
            }
        }

        let r256 = run_at_length(
            &mut session,
            &toks,
            &msk,
            max_length,
            hidden,
            wants_token_type,
        );
        lat_256.push(r256.run_time.as_secs_f64() * 1e3);

        for (s, &len) in shapes.iter().enumerate() {
            let r = run_at_length(&mut session, &toks, &msk, len, hidden, wants_token_type);
            verdicts[s]
                .latencies_ms
                .push(r.run_time.as_secs_f64() * 1e3);
            // Primary: raw hidden state at true (masked-in) positions only.
            let valid = true_len * hidden;
            let (ndh, mxh) = bitwise_diff(&r256.hidden_state[..valid], &r.hidden_state[..valid]);
            // Secondary: final embeddings.
            let (nde, mxe) = bitwise_diff(&r256.embedding, &r.embedding);
            let cos = cosine(&r256.embedding, &r.embedding);
            if ndh > 0 {
                verdicts[s].texts_hidden_diff += 1;
                verdicts[s].worst_hidden = verdicts[s].worst_hidden.max(mxh);
            }
            if nde > 0 {
                verdicts[s].texts_embed_diff += 1;
                verdicts[s].worst_embed = verdicts[s].worst_embed.max(mxe);
                verdicts[s].worst_cos = verdicts[s].worst_cos.min(cos);
                if verdicts[s].texts_embed_diff <= 3 {
                    println!(
                        "  DIFF text[{i}] {}: len={len} true={true_len}: hidden {ndh}/{valid} \
                         (max {mxh:e}), embed {nde}/384 (max {mxe:e}, cos {cos:.9})",
                        shape_names[s]
                    );
                }
            }
        }
    }

    let n = texts.len();
    true_lens.sort_unstable();
    println!("\n=== VERDICT vs 256-pad over {n} texts (model {model_path:?}) ===");
    println!(
        "same-shape determinism control: {}",
        if same_shape_stable {
            "BIT-STABLE"
        } else {
            "NOT bit-stable — cross-shape identity unfalsifiable"
        }
    );
    println!(
        "true token lengths: min {} p50 {} max {}",
        true_lens[0],
        true_lens[n / 2],
        true_lens[n - 1]
    );
    println!(
        "pad-256 session.run p50 = {:.2} ms",
        median(lat_256.clone())
    );
    for (s, name) in shape_names.iter().enumerate() {
        let v = &verdicts[s];
        println!(
            "{name:>12}: hidden diffs {}/{n} texts (max {:e}) | embed diffs {}/{n} texts \
             (max {:e}, worst cos {:.9}) | p50 {:.2} ms",
            v.texts_hidden_diff,
            v.worst_hidden,
            v.texts_embed_diff,
            v.worst_embed,
            v.worst_cos,
            median(v.latencies_ms.clone()),
        );
    }
    println!(
        "first-call cost per NEW shape (ORT re-plan): {} distinct shapes",
        first_call_per_shape.len()
    );
    for (len, ms) in first_call_per_shape.iter() {
        println!("  len {len}: {ms:.2} ms");
    }
}

#[test]
#[ignore = "equivalence probe, needs local model files; run with --ignored --nocapture"]
fn padded_vs_dynamic_equivalence() {
    pre_init_ort_runtime(false);
    let config = EmbeddingConfig::from_env();
    assert!(
        config.model_path.exists(),
        "model not found at {:?}",
        config.model_path
    );

    // Model 1: whatever the production env resolution picks (on this machine:
    // ../models/minilm-l6/model_quantized.onnx — mislabeled fp32, used by the
    // local recall gate).
    probe_model(
        &config.model_path,
        &config.tokenizer_path,
        config.max_length,
    );

    // Model 2: the real quint8_avx2 dynamic-quantized export from the
    // auto-download cache (what fresh installs run). Skip silently if absent.
    if let Some(local) = dirs::data_local_dir() {
        let q = local.join("shodh-memory/models/minilm-l6");
        let qm = q.join("model_quantized.onnx");
        if qm.exists() && qm != config.model_path {
            probe_model(&qm, &q.join("tokenizer.json"), config.max_length);
        }
    }
}
