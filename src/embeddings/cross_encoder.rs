//! Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) over ONNX Runtime.
//!
//! A bi-encoder scores a query and a candidate SEPARATELY and compares the two
//! vectors, so the candidate representation cannot depend on the question being
//! asked. A cross-encoder concatenates them into one sequence and runs full
//! attention across the pair, so every candidate token can attend to every
//! query token. That interaction is the whole mechanism, and it is why the
//! cheaper alternative failed here: a linear adapter fitted over the same
//! frozen embeddings measured -8.3pp p@1 on holdout. Reprojection cannot
//! recover what was never jointly encoded.
//!
//! Measured offline on exported pools, n=1531 (scripts/rerank_pilot.py):
//!
//! | depth | recall@10 | p@1 | ndcg@10 | oracle headroom | ms/query |
//! |---|---|---|---|---|---|
//! | baseline | 0.5466 | 0.3207 | 0.4280 | -- | -- |
//! | 30 | 0.6250 | 0.5023 | 0.5576 | 43.3% | 516 |
//! | 100 | 0.6525 | 0.5069 | 0.5732 | 58.5% | 1285 |
//! | oracle | 0.7276 | 0.7949 | 0.7440 | ceiling | -- |
//!
//! Depth 30 keeps ~98% of the p@1 gain for 40% of the cost: depth buys
//! recall@10, not precision@1. Those timings are CPU torch via
//! sentence-transformers and are the number this session has to beat.
//!
//! Configuration:
//! - SHODH_CE_MODEL_PATH: directory holding model.onnx + tokenizer.json
//!   (default ./models/cross-encoder-ms-marco-minilm-l6)
//! - SHODH_ONNX_THREADS: shared with the other sessions, so the harness
//!   determinism pin covers this one too

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Pair sequence window. Pinned to the value the pilot measured with
/// (max_length=256); changing it invalidates the numbers above, because a
/// tighter window silently truncates the candidate and scores a different text
/// than the one that produced them.
pub const CE_TOKEN_WINDOW: usize = 256;

/// Default rerank depth. See the table above for why this is 30 and not 100.
pub const CE_DEFAULT_DEPTH: usize = 30;

/// Lock wait before giving up on a stuck inference, matching minilm.
const LOCK_TIMEOUT_SECS: u64 = 30;

/// `SHODH_CE_FP32=1` forces the full-precision export. Default is int8, which
/// is 2.8x faster and a quarter the size; this exists so an arm can price the
/// 3.6% of pair orderings the two disagree on.
fn fp32_requested() -> bool {
    std::env::var("SHODH_CE_FP32")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub struct CrossEncoder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// Standard BERT exports take token_type_ids; some do not. For a PAIR model
    /// this matters far more than it does for the embedder: the segment ids are
    /// what tell the model where the query stops and the candidate starts.
    /// Detected from the graph rather than assumed.
    has_token_type_ids: bool,
}

impl CrossEncoder {
    pub fn model_dir() -> PathBuf {
        std::env::var("SHODH_CE_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./models/cross-encoder-ms-marco-minilm-l6"))
    }

    /// Load the session. Returns Err when the model is absent so the caller can
    /// degrade to the unreranked ranking rather than failing the recall -- a
    /// missing optional model must never take a query down.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_variant(dir, !fp32_requested())
    }

    /// `prefer_int8` picks `model_quint8_avx2.onnx` when it is present.
    ///
    /// Measured on this machine at depth 30, debug build, 2 threads:
    /// fp32 525 ms/query (57 pairs/s, 91 MB), int8 **189 ms/query** (159
    /// pairs/s, **23 MB**) — 2.8x faster at a quarter the size, against a pilot
    /// torch baseline of 516 ms.
    ///
    /// Fidelity is measured, not assumed, by
    /// `int8_ranks_like_fp32_on_real_corpus_text`: top-1 agreement 3/3 and
    /// **pairwise concordance 0.9642** over 59,700 pairs of real corpus text.
    /// So int8 agrees with fp32 on 96.4% of orderings, NOT ~100% — the
    /// vector-side "int8 costs nothing" result does not transfer, because a
    /// bi-encoder's error is absorbed by a cosine over 384 dims while this
    /// model emits one logit whose small differences ARE the ranking. Both
    /// variants stay selectable so the end-to-end arms can price that 3.6%.
    pub fn load_variant(dir: &Path, prefer_int8: bool) -> Result<Self> {
        let int8_path = dir.join("model_quint8_avx2.onnx");
        let fp32_path = dir.join("model.onnx");
        let model_path = if prefer_int8 && int8_path.exists() {
            int8_path
        } else {
            fp32_path
        };
        let tokenizer_path = dir.join("tokenizer.json");
        if !model_path.exists() {
            anyhow::bail!("cross-encoder model not found at {}", model_path.display());
        }

        // Point ort at the pinned runtime before touching a session. The crate
        // is built with `load-dynamic`, so without this the loader takes the
        // first `onnxruntime` it finds on the system path -- on this machine a
        // 1.17.1 that ort 2.0-rc.11 refuses. The embedder does this in its own
        // loader, and a second session that skips it fails only at RUN time,
        // long after the code that forgot looks fine.
        crate::embeddings::minilm::pre_init_ort_runtime(false);

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let default_threads = 1;
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let default_threads = 2;
        let num_threads = std::env::var("SHODH_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_threads);

        // Same spinning settings as the other two sessions: Eigen spin-to-block
        // deadlocks on macOS heterogeneous P/E cores.
        let session = Session::builder()
            .context("cross-encoder: session builder")?
            .with_intra_threads(num_threads)
            .context("cross-encoder: intra threads")?
            .with_inter_threads(1)
            .context("cross-encoder: inter threads")?
            .with_intra_op_spinning(false)
            .context("cross-encoder: intra spinning")?
            .with_inter_op_spinning(false)
            .context("cross-encoder: inter spinning")?
            .commit_from_file(&model_path)
            .context("cross-encoder: load onnx")?;

        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("cross-encoder: load tokenizer: {e}"))?;
        // Truncate the PAIR longest-first, so a long candidate is clipped before
        // the query is. Losing query tokens would change the question being
        // asked; losing candidate tail only shortens the evidence.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: CE_TOKEN_WINDOW,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("cross-encoder: truncation: {e}"))?;
        // Pad to the longest member of each batch rather than always to the
        // window: at depth 30 most candidates are far shorter than 256, and a
        // fixed width would pay full price for every one.
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        tracing::info!(
            "cross-encoder loaded from {} ({} threads, token_type_ids={})",
            model_path.display(),
            num_threads,
            has_token_type_ids
        );
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            has_token_type_ids,
        })
    }

    /// Score the query against each candidate. Higher is more relevant.
    ///
    /// The raw logit is returned rather than a sigmoid of it. Ranking is all
    /// this is used for and sigmoid is monotonic, so the ORDER matches the
    /// pilot -- but the VALUES do not, so never compare a stored score against
    /// a pilot score.
    pub fn score_pairs(&self, query: &str, texts: &[&str]) -> Result<Vec<f32>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let pairs: Vec<(String, String)> = texts
            .iter()
            .map(|t| (query.to_string(), (*t).to_string()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| anyhow::anyhow!("cross-encoder: encode_batch: {e}"))?;

        let batch = encodings.len();
        let width = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);
        if width == 0 {
            return Ok(vec![0.0; batch]);
        }

        let mut input_ids = Vec::with_capacity(batch * width);
        let mut attention = Vec::with_capacity(batch * width);
        let mut type_ids = Vec::with_capacity(batch * width);
        for e in &encodings {
            // BatchLongest pads every member to one width, so a disagreement
            // here means the tokenizer config drifted.
            debug_assert_eq!(e.get_ids().len(), width);
            input_ids.extend(e.get_ids().iter().map(|&v| v as i64));
            attention.extend(e.get_attention_mask().iter().map(|&v| v as i64));
            type_ids.extend(e.get_type_ids().iter().map(|&v| v as i64));
        }

        let shape = vec![batch as i64, width as i64];
        let ids_v = Value::from_array((shape.clone(), input_ids))?;
        let mask_v = Value::from_array((shape.clone(), attention))?;
        let types_v = Value::from_array((shape, type_ids))?;

        let timeout = std::time::Duration::from_secs(LOCK_TIMEOUT_SECS);
        let mut session = self.session.try_lock_for(timeout).ok_or_else(|| {
            anyhow::anyhow!("cross-encoder: session lock timeout ({LOCK_TIMEOUT_SECS}s)")
        })?;

        let outputs = if self.has_token_type_ids {
            session.run(ort::inputs![
                "input_ids" => &ids_v,
                "attention_mask" => &mask_v,
                "token_type_ids" => &types_v,
            ])?
        } else {
            session.run(ort::inputs![
                "input_ids" => &ids_v,
                "attention_mask" => &mask_v,
            ])?
        };

        // A single-label reranker emits [batch, 1]; take the first logit of each
        // row so a [batch, N] export cannot silently interleave.
        let (out_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let per_row = if out_shape.len() >= 2 {
            (out_shape[1] as usize).max(1)
        } else {
            1
        };
        Ok((0..batch).map(|i| data[i * per_row]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs the model on disk:
    ///   cargo test --lib cross_encoder -- --ignored --nocapture
    #[test]
    #[ignore = "requires ./models/cross-encoder-ms-marco-minilm-l6"]
    fn relevant_passage_outscores_irrelevant_and_reports_throughput() {
        let ce = CrossEncoder::load(&CrossEncoder::model_dir()).expect("load cross-encoder");

        let query = "What did Melanie say about her daughter's piano recital?";
        let relevant = "Melanie mentioned her daughter played at the piano recital on \
                        Friday and that she was very proud of her.";
        let irrelevant = "The server migration to RocksDB completed overnight with no \
                          downtime reported.";

        let scores = ce
            .score_pairs(query, &[relevant, irrelevant])
            .expect("score pairs");
        assert_eq!(scores.len(), 2);

        // The assertion that keeps this module from being vacuous. A pair model
        // that is not actually attending across the pair -- segment ids wrong or
        // absent, query and candidate transposed, output row stride misread --
        // returns scores with no relationship to relevance, and every one of
        // those mistakes fails HERE rather than silently degrading a ranking
        // that still looks plausible.
        assert!(
            scores[0] > scores[1],
            "relevant passage must outscore irrelevant, got {scores:?}"
        );

        // Throughput at the depth this ships at. The pilot measured 58 pairs/s
        // (516 ms/query at depth 30) on CPU torch; that is the number to beat,
        // and it is the whole ship/no-ship question for this lever.
        let batch: Vec<&str> = std::iter::repeat(relevant).take(CE_DEFAULT_DEPTH).collect();
        let start = std::time::Instant::now();
        let out = ce.score_pairs(query, &batch).expect("score batch");
        let elapsed = start.elapsed().as_secs_f64();
        assert_eq!(out.len(), CE_DEFAULT_DEPTH);
        println!(
            "cross-encoder depth {}: {:.1} ms/query, {:.0} pairs/s  (pilot torch: 516 ms, 58 pairs/s)",
            CE_DEFAULT_DEPTH,
            elapsed * 1000.0,
            CE_DEFAULT_DEPTH as f64 / elapsed,
        );
    }
    /// int8 must RANK like fp32, or the pilot's fp32 numbers do not transfer.
    ///
    /// The vector-side finding that "int8 quantisation costs nothing" was
    /// measured on the EMBEDDER and does not carry here: a bi-encoder's error
    /// is absorbed by a cosine over 384 dims, while a cross-encoder emits one
    /// logit whose small absolute differences ARE the ranking. So this is
    /// measured rather than assumed.
    ///
    /// Agreement, not quality, is the right target: if int8 orders real corpus
    /// text the way fp32 does, it inherits fp32's measured 43.3% of oracle
    /// headroom. Quality is the pilot's job; fidelity is this test's.
    ///
    ///   cargo test --lib cross_encoder_int8 -- --ignored --nocapture
    #[test]
    #[ignore = "requires both fp32 and int8 models on disk"]
    fn int8_ranks_like_fp32_on_real_corpus_text() {
        use std::io::BufRead;

        let base = CrossEncoder::model_dir();
        let fp32 = CrossEncoder::load_variant(&base, false).expect("load fp32");
        let int8 = CrossEncoder::load_variant(&base, true).expect("load int8");

        // Real turns from the target corpus, not synthetic strings: quantisation
        // error depends on the activation distribution, and invented text does
        // not have the corpus's.
        let f = std::fs::File::open("tests/recall/corpora/locomo.jsonl").expect("corpus");
        let texts: Vec<String> = std::io::BufReader::new(f)
            .lines()
            .filter_map(|l| l.ok())
            .filter_map(|l| {
                serde_json::from_str::<serde_json::Value>(&l)
                    .ok()
                    .and_then(|v| v.get("content")?.as_str().map(str::to_string))
            })
            .take(200)
            .collect();
        assert!(texts.len() >= 100, "need corpus text, got {}", texts.len());
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let queries = [
            "What did Melanie say about her kids?",
            "Where did Caroline go on holiday?",
            "When did they talk about the piano recital?",
        ];

        let (mut top1_same, mut pairs_ok, mut pairs_total) = (0usize, 0usize, 0usize);
        for q in queries {
            let a = fp32.score_pairs(q, &refs).expect("fp32 scores");
            let b = int8.score_pairs(q, &refs).expect("int8 scores");

            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|(i, _)| i)
                    .unwrap()
            };
            if argmax(&a) == argmax(&b) {
                top1_same += 1;
            }

            // Pairwise concordance: the fraction of candidate pairs the two
            // models order the same way. This is what a ranker is FOR, and it
            // is insensitive to any monotone rescaling between the two.
            for i in 0..refs.len() {
                for j in (i + 1)..refs.len() {
                    pairs_total += 1;
                    if (a[i] > a[j]) == (b[i] > b[j]) {
                        pairs_ok += 1;
                    }
                }
            }
        }

        let concordance = pairs_ok as f64 / pairs_total as f64;
        println!(
            "int8 vs fp32: top-1 agreement {}/{}, pairwise concordance {:.4} over {} pairs",
            top1_same,
            queries.len(),
            concordance,
            pairs_total
        );

        // A cross-encoder whose ordering agrees with fp32 on ~99% of pairs is
        // carrying the same ranking signal. Set well below the observed value
        // so this is a REGRESSION gate on the quantised export, not a
        // restatement of one measurement.
        assert!(
            concordance > 0.95,
            "int8 must rank like fp32; concordance {concordance:.4}"
        );
    }
}
