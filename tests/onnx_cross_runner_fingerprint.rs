//! Cross-runner model-output fingerprint.
//!
//! Four L1 gate runs of one tree (681b2d8d) produced three different per-case
//! result sets across `ubuntu-latest` runners while every run was internally
//! byte-stable over 5 repeats. The suspects are the three ONNX sessions
//! (MiniLM quint8, GLiNER fp32, cross-encoder quint8) and the AVX2 distance
//! kernels: all four select code paths by CPUID at runtime, and the runner
//! fleet mixes CPU SKUs.
//!
//! This probe runs a FIXED input set through every stage that touches floats
//! before a comparison and writes a JSONL fingerprint: sha256 over the raw
//! little-endian bytes of each output, plus the first values as hex bit
//! patterns so a diff shows magnitude and not just inequality. Run it on
//! several runners, diff the files, and the first stage whose hash differs is
//! the first divergent stage.
//!
//! `#[ignore]`d because it needs the real model files. The workflow
//! `.github/workflows/diag-cross-runner-fingerprint.yml` runs it as a matrix.

use sha2::{Digest, Sha256};
use shodh_memory::embeddings::cross_encoder::CrossEncoder;
use shodh_memory::embeddings::gliner::{GlinerConfig, GlinerTyper};
use shodh_memory::embeddings::minilm::{EmbeddingConfig, MiniLMEmbedder};
use shodh_memory::embeddings::Embedder;
use shodh_memory::vector_db::distance_inline::{dot_product_inline, euclidean_squared_inline};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

const CORPUS: &str = "tests/recall/corpora/locomo-gate.jsonl";
const CASES: &str = "tests/recall/locomo-gate-cases.jsonl";
const N_TEXTS: usize = 32;
const N_QUERIES: usize = 8;

fn sha256_f32(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    hex::encode(h.finalize())
}

fn sha256_str(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

fn sha256_file(p: &Path) -> String {
    match std::fs::read(p) {
        Ok(bytes) => hex::encode(Sha256::digest(&bytes)),
        Err(e) => format!("<unreadable: {e}>"),
    }
}

fn head_bits(values: &[f32], n: usize) -> Vec<String> {
    values
        .iter()
        .take(n)
        .map(|v| format!("{:08x}", v.to_bits()))
        .collect()
}

fn read_jsonl_field(path: &str, field: &str, n: usize) -> Vec<String> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("jsonl line");
            v[field]
                .as_str()
                .unwrap_or_else(|| panic!("{path}: missing string field {field}"))
                .to_string()
        })
        .take(n)
        .collect()
}

fn cpu_flags() -> String {
    let info = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let model = info
        .lines()
        .find(|l| l.starts_with("model name"))
        .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        .unwrap_or_else(|| "<unknown>".into());
    let flags = info
        .lines()
        .find(|l| l.starts_with("flags"))
        .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        .unwrap_or_default();
    let interesting = [
        "avx2",
        "fma",
        "avx512f",
        "avx512bw",
        "avx512_vnni",
        "avx_vnni",
        "amx_int8",
        "amx_bf16",
        "avx512_bf16",
        "avx512_fp16",
    ];
    let present: Vec<&str> = interesting
        .iter()
        .copied()
        .filter(|f| flags.split_whitespace().any(|x| x == *f))
        .collect();
    format!("model=\"{model}\" isa={}", present.join(","))
}

#[test]
#[ignore = "needs the MiniLM, GLiNER and cross-encoder model files; run via the diag workflow"]
fn cross_runner_fingerprint() {
    // Same determinism env the recall harness pins (pin_harness_threads):
    // single-threaded reductions so the only remaining variable is the ISA.
    // SAFETY: nothing else in this test process reads these concurrently.
    unsafe {
        for (k, v) in [("SHODH_ONNX_THREADS", "1"), ("RAYON_NUM_THREADS", "1")] {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
    }

    let out_path = std::env::var("SHODH_FINGERPRINT_OUT").unwrap_or_else(|_| "fingerprint.jsonl".into());
    let mut out = std::fs::File::create(&out_path).expect("create fingerprint output");
    let mut emit = |v: serde_json::Value| {
        let line = v.to_string();
        println!("FP {line}");
        writeln!(out, "{line}").expect("write fingerprint line");
    };

    let mut texts = read_jsonl_field(CORPUS, "content", N_TEXTS - 3);
    // Edge shapes: past the 256-token window (truncation path), a one-word
    // input (almost all padding), and non-ASCII (tokenizer normalisation).
    texts.push(
        "Nate said that the caterpillar became a butterfly and Joanna replied that the pottery class \
         was on Tuesday. "
            .repeat(12),
    );
    texts.push("Joanna".to_string());
    texts.push("Café résumé naïve — “smart quotes” and emoji 🦋 in the mix".to_string());
    let queries = read_jsonl_field(CASES, "query", N_QUERIES);
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    emit(serde_json::json!({
        "stage": "env",
        "cpu": cpu_flags(),
        "onnx_threads": std::env::var("SHODH_ONNX_THREADS").unwrap_or_default(),
        "inputs_sha256": sha256_str(&format!("{texts:?}{queries:?}")),
        "n_texts": texts.len(),
        "n_queries": queries.len(),
    }));

    // ---- MiniLM (quint8_avx2 dynamic-quantised export) ----
    let embed_cfg = EmbeddingConfig::from_env();
    let embedder = MiniLMEmbedder::new(embed_cfg.clone()).expect("load MiniLM");
    assert!(embedder.is_model_loaded(), "MiniLM must be the real ONNX model, not simplified");
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    let mut all = Sha256::new();
    for (i, t) in texts.iter().enumerate() {
        let e = embedder.encode(t).expect("encode");
        for v in &e {
            all.update(v.to_le_bytes());
        }
        emit(serde_json::json!({
            "stage": "minilm", "i": i, "dim": e.len(),
            "sha256": sha256_f32(&e), "head": head_bits(&e, 6),
        }));
        embeddings.push(e);
    }
    let mut qemb: Vec<Vec<f32>> = Vec::with_capacity(queries.len());
    for (i, q) in queries.iter().enumerate() {
        let e = embedder.encode_query(q).expect("encode query");
        for v in &e {
            all.update(v.to_le_bytes());
        }
        emit(serde_json::json!({
            "stage": "minilm_query", "i": i, "sha256": sha256_f32(&e), "head": head_bits(&e, 6),
        }));
        qemb.push(e);
    }
    emit(serde_json::json!({
        "stage": "minilm_all", "sha256": hex::encode(all.finalize()),
        "model_sha256": sha256_file(&embed_cfg.model_path),
        "model_path": embed_cfg.model_path.display().to_string(),
    }));

    // ---- Rust-side distance kernels (AVX2+FMA vs scalar by CPUID) ----
    let mut dots = Vec::new();
    let mut l2s = Vec::new();
    for q in &qemb {
        for e in &embeddings {
            dots.push(dot_product_inline(q, e));
            l2s.push(euclidean_squared_inline(q, e));
        }
    }
    emit(serde_json::json!({
        "stage": "distance", "dot_sha256": sha256_f32(&dots), "l2_sha256": sha256_f32(&l2s),
        "dot_head": head_bits(&dots, 6),
    }));

    // ---- GLiNER bi-edge (fp32 export) ----
    let gl_cfg = GlinerConfig::from_env();
    let typer = GlinerTyper::new(gl_cfg.clone());
    assert!(typer.is_available(), "GLiNER must be loadable — refusing to fingerprint the fallback NER");
    let mut gl_all = String::new();
    for (i, t) in texts.iter().enumerate() {
        let spans = typer.try_extract(t).expect("gliner extract");
        let mut line = String::new();
        for s in &spans {
            let _ = write!(
                line,
                "{}|{}|{:08x}|{}|{};",
                s.text,
                s.fine_label,
                s.score.to_bits(),
                s.start,
                s.end
            );
        }
        gl_all.push_str(&line);
        gl_all.push('\n');
        let scores: Vec<f32> = spans.iter().map(|s| s.score).collect();
        emit(serde_json::json!({
            "stage": "gliner", "i": i, "n_spans": spans.len(),
            "sha256": sha256_str(&line), "score_head": head_bits(&scores, 6),
            "spans_head": spans.iter().take(4).map(|s| format!("{}:{}", s.text, s.fine_label)).collect::<Vec<_>>(),
        }));
    }
    // Label competition at every span, top-3: the decode above keeps only the
    // argmax, so a runner that flips a near-tie label shows here first.
    let mut rank_all = String::new();
    for (i, t) in texts.iter().take(8).enumerate() {
        let ranked = typer.rank_labels(t, 3).expect("gliner rank_labels");
        let mut line = String::new();
        for r in &ranked {
            let _ = write!(line, "{}@{}-{}:", r.text, r.start, r.end);
            for (label, score) in &r.top {
                let _ = write!(line, "{label}={:08x},", score.to_bits());
            }
            line.push(';');
        }
        rank_all.push_str(&line);
        emit(serde_json::json!({
            "stage": "gliner_rank", "i": i, "sha256": sha256_str(&line),
            "head": line.chars().take(160).collect::<String>(),
        }));
    }
    emit(serde_json::json!({
        "stage": "gliner_all", "sha256": sha256_str(&gl_all), "rank_sha256": sha256_str(&rank_all),
        "model_sha256": sha256_file(&gl_cfg.model_path),
        "label_embeddings_sha256": sha256_file(&gl_cfg.label_embeddings_path),
    }));

    // ---- Cross-encoder (quint8_avx2 dynamic-quantised export) ----
    let ce_dir = CrossEncoder::model_dir();
    let ce = CrossEncoder::load(&ce_dir).expect("load cross-encoder");
    let mut ce_all = Sha256::new();
    for (i, q) in queries.iter().enumerate() {
        let scores = ce.score_pairs(q, &text_refs).expect("ce score_pairs");
        for v in &scores {
            ce_all.update(v.to_le_bytes());
        }
        emit(serde_json::json!({
            "stage": "ce", "i": i, "n": scores.len(),
            "sha256": sha256_f32(&scores), "head": head_bits(&scores, 8),
        }));
    }
    emit(serde_json::json!({
        "stage": "ce_all", "sha256": hex::encode(ce_all.finalize()),
        "model_dir": ce_dir.display().to_string(),
        "int8_model_sha256": sha256_file(&ce_dir.join("model_quint8_avx2.onnx")),
    }));

    // ---- Runtime identity: the same .so must be loaded on every runner ----
    let ort = std::env::var("ORT_DYLIB_PATH").unwrap_or_default();
    emit(serde_json::json!({
        "stage": "runtime",
        "ort_dylib_path": ort,
        "ort_sha256": if ort.is_empty() { "<unset>".to_string() } else { sha256_file(Path::new(&ort)) },
    }));
    eprintln!("fingerprint written to {out_path}");
}
