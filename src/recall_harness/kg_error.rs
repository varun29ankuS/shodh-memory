//! KG error-detection benchmark (FB15k-237 + constrained noise injection).
//!
//! The standard protocol from the error-detection literature (CKRL, KGTtm,
//! CAGED): corrupt a fixed fraction of a clean knowledge graph's triples, rank
//! ALL triples by trustworthiness, and measure how well the injected errors
//! sink. Published baselines are learned KG-embedding models; every signal
//! here is DETERMINISTIC and LLM-free — the same statistics the memory graph
//! already stores (df/PMI, structural support, frequency priors) plus the
//! frozen MiniLM name-embedding. The point is to put a number on how far the
//! shodh signal family gets on a public benchmark, not to win a leaderboard
//! with a trained model.
//!
//! Fixture (built by `benchmarks/kg_error_inject.py`, source pinned):
//!   triples.tsv       head <TAB> relation <TAB> tail <TAB> label(1 clean, 0 injected)
//!   entity_names.tsv  mid <TAB> human-readable name
//!
//! Signals (higher = more trustworthy). The scorer never sees labels; the
//! graph statistics are computed over the NOISY graph, as they would be in
//! production where corrupted edges are already stored:
//! - `ppmi`      log2(co_excl · N / (df_h · df_t)) — independent co-occurrence
//!               support for the pair, EXCLUDING the scored triple itself
//!               (co_excl = 0 → no independent attestation → strongly negative).
//!               This is the memory graph's birth-PMI generalized to a support score.
//! - `struct`    Adamic-Adar over the entity co-mention graph — shared-neighbour
//!               support discounted by neighbour degree (bridge-aware).
//! - `slot`      relation-slot frequency prior: ln P(h | r, head-slot) +
//!               ln P(t | r, tail-slot). Deliberately weak against CONSTRAINED
//!               corruption (replacements are drawn from the same slot pool) —
//!               kept as the honest floor signal.
//! - `embed`     cosine(name embedding of h, name embedding of t) with frozen
//!               MiniLM — semantic coherence of the pair, no training.
//! - `ensemble`  mean of the per-signal rank fractions (rank-average fusion,
//!               deterministic tie-break by input order).
//!
//! Metrics: AUC (clean ranked above injected) and error-precision@K where
//! K = number of injected errors (the literature's standard cut).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgErrorReport {
    pub triples: usize,
    pub injected: usize,
    pub embedded_entities: usize,
    /// signal name -> (AUC, error-precision@K)
    pub signals: Vec<(String, f64, f64)>,
}

struct Triple {
    h: usize,
    r: usize,
    t: usize,
    clean: bool,
}

fn interner() -> (HashMap<String, usize>, Vec<String>) {
    (HashMap::new(), Vec::new())
}

fn intern(map: &mut HashMap<String, usize>, names: &mut Vec<String>, s: &str) -> usize {
    if let Some(&i) = map.get(s) {
        return i;
    }
    let i = names.len();
    map.insert(s.to_string(), i);
    names.push(s.to_string());
    i
}

/// AUC via rank-sum (Mann-Whitney U). `scores` parallel to `labels`
/// (true = clean/positive). Ties get average rank (dense enumeration).
fn auc(scores: &[f64], labels: &[bool]) -> f64 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    let mut rank = vec![0f64; scores.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            rank[k] = avg;
        }
        i = j + 1;
    }
    let pos = labels.iter().filter(|&&l| l).count() as f64;
    let neg = labels.len() as f64 - pos;
    if pos == 0.0 || neg == 0.0 {
        return 0.5;
    }
    let pos_rank_sum: f64 = rank
        .iter()
        .zip(labels)
        .filter(|(_, &l)| l)
        .map(|(r, _)| *r)
        .sum();
    (pos_rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg)
}

/// Error-precision@K, K = number of injected errors: of the K LOWEST-scored
/// triples, the fraction that are actually injected.
fn error_precision_at_k(scores: &[f64], labels: &[bool]) -> f64 {
    let k = labels.iter().filter(|&&l| !l).count();
    if k == 0 {
        return 0.0;
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    let hits = idx[..k].iter().filter(|&&i| !labels[i]).count();
    hits as f64 / k as f64
}

/// Rank fraction in [0,1] per score (higher score -> higher fraction),
/// deterministic tie-break by input order.
fn rank_fraction(scores: &[f64]) -> Vec<f64> {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));
    let mut frac = vec![0f64; n];
    for (pos, &i) in idx.iter().enumerate() {
        frac[i] = (pos + 1) as f64 / n as f64;
    }
    frac
}

pub fn run_kg_error(dir: &Path) -> Result<KgErrorReport> {
    let triples_raw = std::fs::read_to_string(dir.join("triples.tsv"))
        .with_context(|| format!("read {}/triples.tsv", dir.display()))?;
    let names_raw = std::fs::read_to_string(dir.join("entity_names.tsv"))
        .with_context(|| format!("read {}/entity_names.tsv", dir.display()))?;

    let (mut ent_map, mut ents) = interner();
    let (mut rel_map, mut rels) = interner();
    let mut triples: Vec<Triple> = Vec::new();
    for line in triples_raw.lines() {
        let mut parts = line.split('\t');
        let (Some(h), Some(r), Some(t), Some(label)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        triples.push(Triple {
            h: intern(&mut ent_map, &mut ents, h),
            r: intern(&mut rel_map, &mut rels, r),
            t: intern(&mut ent_map, &mut ents, t),
            clean: label.trim() == "1",
        });
    }
    anyhow::ensure!(!triples.is_empty(), "no triples parsed");
    let n = triples.len() as f64;
    let injected = triples.iter().filter(|t| !t.clean).count();
    let labels: Vec<bool> = triples.iter().map(|t| t.clean).collect();

    let mut display_name: HashMap<String, String> = HashMap::new();
    for line in names_raw.lines() {
        if let Some((mid, name)) = line.split_once('\t') {
            display_name.insert(mid.to_string(), name.to_string());
        }
    }

    // ---- statistics over the (noisy) graph ----
    let mut df = vec![0usize; ents.len()];
    let mut pair_co: HashMap<(usize, usize), usize> = HashMap::new();
    let mut slot_head: HashMap<(usize, usize), usize> = HashMap::new(); // (r,h)
    let mut slot_tail: HashMap<(usize, usize), usize> = HashMap::new(); // (r,t)
    let mut rel_count = vec![0usize; rels.len()];
    let mut neighbours: Vec<HashSet<usize>> = vec![HashSet::new(); ents.len()];
    for tr in &triples {
        df[tr.h] += 1;
        df[tr.t] += 1;
        let key = (tr.h.min(tr.t), tr.h.max(tr.t));
        *pair_co.entry(key).or_insert(0) += 1;
        *slot_head.entry((tr.r, tr.h)).or_insert(0) += 1;
        *slot_tail.entry((tr.r, tr.t)).or_insert(0) += 1;
        rel_count[tr.r] += 1;
        neighbours[tr.h].insert(tr.t);
        neighbours[tr.t].insert(tr.h);
    }

    // ppmi: independent co-occurrence support (exclude the scored triple).
    let ppmi: Vec<f64> = triples
        .iter()
        .map(|tr| {
            let key = (tr.h.min(tr.t), tr.h.max(tr.t));
            let co_excl = pair_co[&key] - 1;
            if co_excl == 0 {
                // No independent attestation: below any attested pair, ordered
                // by how hub-heavy the endpoints are (hubbier -> more suspicious).
                -((df[tr.h] * df[tr.t]) as f64).log2() - 1e3
            } else {
                ((co_excl as f64) * n / ((df[tr.h] * df[tr.t]) as f64)).log2()
            }
        })
        .collect();

    // struct: Adamic-Adar common-neighbour support (excluding the endpoints).
    let structural: Vec<f64> = triples
        .iter()
        .map(|tr| {
            let (a, b) = (&neighbours[tr.h], &neighbours[tr.t]);
            let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            small
                .iter()
                .filter(|z| **z != tr.h && **z != tr.t && large.contains(*z))
                .map(|&z| 1.0 / (neighbours[z].len().max(2) as f64).ln())
                .sum::<f64>()
        })
        .collect();

    // slot: relation-slot frequency prior.
    let slot: Vec<f64> = triples
        .iter()
        .map(|tr| {
            let rc = rel_count[tr.r].max(1) as f64;
            let ph = slot_head[&(tr.r, tr.h)] as f64 / rc;
            let pt = slot_tail[&(tr.r, tr.t)] as f64 / rc;
            ph.ln() + pt.ln()
        })
        .collect();

    // embed: MiniLM name-embedding cosine per unique entity, then per triple.
    let embedder = crate::embeddings::minilm::MiniLMEmbedder::new(
        crate::embeddings::minilm::EmbeddingConfig::default(),
    )
    .context("MiniLM init")?;
    use crate::embeddings::Embedder as _;
    let mut ent_emb: Vec<Vec<f32>> = Vec::with_capacity(ents.len());
    for mid in &ents {
        let text = display_name.get(mid).map(|s| s.as_str()).unwrap_or(mid);
        let v = embedder
            .encode(text)
            .with_context(|| format!("embed {mid}"))?;
        ent_emb.push(v);
    }
    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            (dot / (na * nb)) as f64
        }
    };
    let embed: Vec<f64> = triples
        .iter()
        .map(|tr| cosine(&ent_emb[tr.h], &ent_emb[tr.t]))
        .collect();

    // ensemble: mean rank fraction.
    let fracs = [
        rank_fraction(&ppmi),
        rank_fraction(&structural),
        rank_fraction(&slot),
        rank_fraction(&embed),
    ];
    let ensemble: Vec<f64> = (0..triples.len())
        .map(|i| fracs.iter().map(|f| f[i]).sum::<f64>() / fracs.len() as f64)
        .collect();

    let mut signals: Vec<(String, f64, f64)> = Vec::new();
    for (name, scores) in [
        ("ppmi", &ppmi),
        ("struct", &structural),
        ("slot", &slot),
        ("embed", &embed),
        ("ensemble", &ensemble),
    ] {
        signals.push((
            name.to_string(),
            auc(scores, &labels),
            error_precision_at_k(scores, &labels),
        ));
    }

    Ok(KgErrorReport {
        triples: triples.len(),
        injected,
        embedded_entities: ents.len(),
        signals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_perfect_and_random() {
        // clean scored above injected -> AUC 1.0
        let scores = [0.9, 0.8, 0.1, 0.2];
        let labels = [true, true, false, false];
        assert!((auc(&scores, &labels) - 1.0).abs() < 1e-9);
        // inverted -> 0.0
        let labels_inv = [false, false, true, true];
        assert!(auc(&scores, &labels_inv).abs() < 1e-9);
        // all tied -> 0.5
        let tied = [0.5, 0.5, 0.5, 0.5];
        assert!((auc(&tied, &labels) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn error_precision_counts_bottom_k() {
        // 2 injected; bottom-2 by score are exactly the injected -> 1.0
        let scores = [0.9, 0.8, 0.1, 0.2];
        let labels = [true, true, false, false];
        assert!((error_precision_at_k(&scores, &labels) - 1.0).abs() < 1e-9);
        // one of bottom-2 is clean -> 0.5
        let scores2 = [0.9, 0.15, 0.1, 0.8];
        assert!((error_precision_at_k(&scores2, &labels) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn rank_fraction_is_deterministic_under_ties() {
        let scores = [0.5, 0.5, 0.1];
        let f = rank_fraction(&scores);
        // tie broken by input order: index 0 before index 1.
        assert!(f[2] < f[0] && f[0] < f[1]);
    }
}
