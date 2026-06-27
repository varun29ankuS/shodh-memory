//! Anomaly / needle-in-haystack benchmark + detector interface.
//!
//! Instrument-first: this module defines a CONTROLLED substrate with *planted*
//! anomalies (known ground truth) so any anomaly detector can be measured —
//! precision@k, recall, and ROC-AUC against the plants — before we trust it.
//!
//! The substrate is a mixture-of-communities: clusters of semantically similar
//! items (embeddings near a community center) wired with intra-community graph
//! edges. Onto that we plant three kinds of needle:
//!   - `SemanticOutlier`   — embedding far from every cluster (vector says "odd")
//!   - `StructuralBridge`  — normal embedding, but graph edges span distant
//!                           communities (graph says "odd")
//!   - `CrossModalMismatch`— embedding from community B, graph edges into A
//!                           (the modalities DISAGREE — the comparative-fusion case)
//!
//! A detector implements [`AnomalyDetector`]: it returns one anomalousness score
//! per item. [`evaluate`] ranks by score and scores the ranking vs the plants.
//! A benchmark that cannot separate [`RandomDetector`] (AUC ≈ 0.5) from a real
//! detector is itself broken — the tests assert that separation.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Ground-truth anomaly kind planted into the substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnomalyKind {
    /// Embedding far from every community center; graph edges normal.
    SemanticOutlier,
    /// Embedding normal (in its community); graph edges span distant communities.
    StructuralBridge,
    /// Embedding from one community, graph edges into another — modalities disagree.
    CrossModalMismatch,
}

/// One item in the substrate. `planted` is the ground-truth label (None = normal).
#[derive(Debug, Clone)]
pub struct AnomalyItem {
    pub id: usize,
    /// Semantic vector (L2-normalized).
    pub embedding: Vec<f32>,
    /// Ground-truth community this item nominally belongs to.
    pub community: usize,
    /// Graph neighbor ids (undirected adjacency).
    pub neighbors: Vec<usize>,
    /// Ground-truth plant label; `None` for normal items.
    pub planted: Option<AnomalyKind>,
}

/// A generated substrate with planted needles.
#[derive(Debug, Clone)]
pub struct PlantedCorpus {
    pub items: Vec<AnomalyItem>,
    pub community_centers: Vec<Vec<f32>>,
    pub dim: usize,
}

impl PlantedCorpus {
    /// Indices of all planted (anomalous) items.
    pub fn planted_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, it)| it.planted.is_some())
            .map(|(i, _)| i)
            .collect()
    }

    /// Count of planted items.
    pub fn n_planted(&self) -> usize {
        self.items.iter().filter(|it| it.planted.is_some()).count()
    }
}

/// Parameters controlling substrate generation. Deterministic given `seed`.
#[derive(Debug, Clone)]
pub struct GenParams {
    pub n_normal: usize,
    pub communities: usize,
    pub dim: usize,
    /// Intra-community embedding noise (smaller = tighter clusters).
    pub noise: f32,
    /// Graph edges per normal item (to same-community members).
    pub degree: usize,
    pub n_semantic_outlier: usize,
    pub n_structural_bridge: usize,
    pub n_cross_modal: usize,
    pub seed: u64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            n_normal: 1000,
            communities: 20,
            dim: 384,
            noise: 0.15,
            degree: 8,
            n_semantic_outlier: 10,
            n_structural_bridge: 10,
            n_cross_modal: 10,
            seed: 42,
        }
    }
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= n;
    }
    v
}

fn rand_unit(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    unit((0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
}

/// Member embedding = community center + Gaussian-ish noise, renormalized.
fn near(center: &[f32], noise: f32, rng: &mut StdRng) -> Vec<f32> {
    unit(
        center
            .iter()
            .map(|&c| c + (rng.gen::<f32>() * 2.0 - 1.0) * noise)
            .collect(),
    )
}

/// Cosine similarity of two L2-normalized vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Generate a substrate with planted needles. Deterministic for a given seed.
pub fn generate(p: &GenParams) -> PlantedCorpus {
    let mut rng = StdRng::seed_from_u64(p.seed);
    let centers: Vec<Vec<f32>> = (0..p.communities).map(|_| rand_unit(&mut rng, p.dim)).collect();

    // Normal items: assigned round-robin to communities, embedding near center.
    let mut items: Vec<AnomalyItem> = Vec::with_capacity(p.n_normal);
    let mut by_community: Vec<Vec<usize>> = vec![Vec::new(); p.communities];
    for i in 0..p.n_normal {
        let c = i % p.communities;
        by_community[c].push(i);
        items.push(AnomalyItem {
            id: i,
            embedding: near(&centers[c], p.noise, &mut rng),
            community: c,
            neighbors: Vec::new(),
            planted: None,
        });
    }
    // Intra-community graph edges (each item → `degree` same-community members).
    for c in 0..p.communities {
        let members = by_community[c].clone();
        if members.len() < 2 {
            continue;
        }
        for &id in &members {
            for _ in 0..p.degree {
                let other = members[rng.gen_range(0..members.len())];
                if other != id && !items[id].neighbors.contains(&other) {
                    items[id].neighbors.push(other);
                    items[other].neighbors.push(id);
                }
            }
        }
    }

    let pick_member = |rng: &mut StdRng, by: &[Vec<usize>], c: usize| -> usize {
        let m = &by[c];
        m[rng.gen_range(0..m.len())]
    };

    // --- Plant SemanticOutlier: random embedding, but graph edges into a community ---
    for _ in 0..p.n_semantic_outlier {
        let host = rng.gen_range(0..p.communities);
        let id = items.len();
        let nbrs: Vec<usize> = (0..p.degree).map(|_| pick_member(&mut rng, &by_community, host)).collect();
        let emb = rand_unit(&mut rng, p.dim); // far from every center (high-dim random)
        for &nb in &nbrs {
            items[nb].neighbors.push(id);
        }
        items.push(AnomalyItem {
            id,
            embedding: emb,
            community: host,
            neighbors: nbrs,
            planted: Some(AnomalyKind::SemanticOutlier),
        });
    }

    // --- Plant StructuralBridge: normal embedding (community A), edges spanning many communities ---
    for _ in 0..p.n_structural_bridge {
        let home = rng.gen_range(0..p.communities);
        let id = items.len();
        let emb = near(&centers[home], p.noise, &mut rng);
        // edges into `degree` DISTINCT random communities (a broker that shouldn't exist)
        let nbrs: Vec<usize> = (0..p.degree)
            .map(|_| {
                let c = rng.gen_range(0..p.communities);
                pick_member(&mut rng, &by_community, c)
            })
            .collect();
        for &nb in &nbrs {
            items[nb].neighbors.push(id);
        }
        items.push(AnomalyItem {
            id,
            embedding: emb,
            community: home,
            neighbors: nbrs,
            planted: Some(AnomalyKind::StructuralBridge),
        });
    }

    // --- Plant CrossModalMismatch: embedding from community B, edges into community A ---
    for _ in 0..p.n_cross_modal {
        let graph_home = rng.gen_range(0..p.communities);
        let mut vec_home = rng.gen_range(0..p.communities);
        if vec_home == graph_home {
            vec_home = (vec_home + 1) % p.communities;
        }
        let id = items.len();
        let emb = near(&centers[vec_home], p.noise, &mut rng); // vector says vec_home
        let nbrs: Vec<usize> = (0..p.degree).map(|_| pick_member(&mut rng, &by_community, graph_home)).collect(); // graph says graph_home
        for &nb in &nbrs {
            items[nb].neighbors.push(id);
        }
        items.push(AnomalyItem {
            id,
            embedding: emb,
            community: graph_home,
            neighbors: nbrs,
            planted: Some(AnomalyKind::CrossModalMismatch),
        });
    }

    PlantedCorpus {
        items,
        community_centers: centers,
        dim: p.dim,
    }
}

/// A detector returns one anomalousness score per item (higher = more anomalous).
pub trait AnomalyDetector {
    fn name(&self) -> &str;
    /// One score per `corpus.items`, same order.
    fn score(&self, corpus: &PlantedCorpus) -> Vec<f32>;
}

/// Result of scoring a detector against the plants.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub detector: String,
    /// ROC-AUC of anomaly score vs planted/normal label (0.5 = random).
    pub auc: f32,
    /// Precision@k with k = number of planted items.
    pub precision_at_k: f32,
    /// Recall@k (same k); equals precision@k when k = n_planted.
    pub recall_at_k: f32,
    /// Per-kind recall@k.
    pub per_kind: Vec<(AnomalyKind, f32)>,
}

/// Evaluate a detector's scores against the planted ground truth.
pub fn evaluate(corpus: &PlantedCorpus, scores: &[f32], detector: &str) -> BenchResult {
    assert_eq!(scores.len(), corpus.items.len(), "score count must match items");
    let n = scores.len();
    let labels: Vec<bool> = corpus.items.iter().map(|it| it.planted.is_some()).collect();
    let n_pos = labels.iter().filter(|&&l| l).count();
    let n_neg = n - n_pos;

    // ROC-AUC via the Mann–Whitney rank-sum (ties get average ranks).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (scores[order[j]] - scores[order[i]]).abs() < f32::EPSILON {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0; // average of ranks (i+1..=j), 1-based
        for k in i..j {
            ranks[order[k]] = avg;
        }
        i = j;
    }
    let sum_pos_ranks: f64 = (0..n).filter(|&i| labels[i]).map(|i| ranks[i]).sum();
    let auc = if n_pos == 0 || n_neg == 0 {
        0.5
    } else {
        ((sum_pos_ranks - (n_pos as f64 * (n_pos as f64 + 1.0)) / 2.0) / (n_pos as f64 * n_neg as f64))
            as f32
    };

    // Top-k (k = n_planted) by descending score.
    let k = n_pos.max(1);
    let mut by_score: Vec<usize> = (0..n).collect();
    by_score.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    let topk: std::collections::HashSet<usize> = by_score.into_iter().take(k).collect();
    let hits = topk.iter().filter(|&&i| labels[i]).count();
    let precision_at_k = hits as f32 / k as f32;
    let recall_at_k = if n_pos == 0 { 0.0 } else { hits as f32 / n_pos as f32 };

    let mut per_kind = Vec::new();
    for kind in [
        AnomalyKind::SemanticOutlier,
        AnomalyKind::StructuralBridge,
        AnomalyKind::CrossModalMismatch,
    ] {
        let total = corpus.items.iter().filter(|it| it.planted == Some(kind)).count();
        if total == 0 {
            continue;
        }
        let found = topk
            .iter()
            .filter(|&&i| corpus.items[i].planted == Some(kind))
            .count();
        per_kind.push((kind, found as f32 / total as f32));
    }

    BenchResult {
        detector: detector.to_string(),
        auc,
        precision_at_k,
        recall_at_k,
        per_kind,
    }
}

/// Sanity baseline: random scores. Must score AUC ≈ 0.5 — the floor every real
/// detector has to beat for the benchmark to mean anything.
pub struct RandomDetector {
    pub seed: u64,
}
impl AnomalyDetector for RandomDetector {
    fn name(&self) -> &str {
        "random"
    }
    fn score(&self, corpus: &PlantedCorpus) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        (0..corpus.items.len()).map(|_| rng.gen::<f32>()).collect()
    }
}

/// First real detector — comparative (cross-modal) disagreement: how far an item's
/// own embedding is from the centroid of its GRAPH neighbors' embeddings. High when
/// the graph neighborhood and the semantic vector disagree (cross-modal mismatch,
/// semantic outlier) — the additive-fusion-blind signal.
pub struct CrossModalDisagreement;
impl AnomalyDetector for CrossModalDisagreement {
    fn name(&self) -> &str {
        "cross_modal_disagreement"
    }
    fn score(&self, corpus: &PlantedCorpus) -> Vec<f32> {
        corpus
            .items
            .iter()
            .map(|it| {
                if it.neighbors.is_empty() {
                    return 0.0;
                }
                let dim = corpus.dim;
                let mut centroid = vec![0.0f32; dim];
                for &nb in &it.neighbors {
                    for (c, x) in centroid.iter_mut().zip(&corpus.items[nb].embedding) {
                        *c += x;
                    }
                }
                let inv = 1.0 / it.neighbors.len() as f32;
                for c in &mut centroid {
                    *c *= inv;
                }
                // disagreement = 1 - cosine(own, neighbor-centroid). Centroid isn't unit;
                // normalize for a bounded, comparable score.
                let centroid = unit(centroid);
                1.0 - cosine(&it.embedding, &centroid)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substrate_has_expected_plants() {
        let p = GenParams::default();
        let corpus = generate(&p);
        assert_eq!(corpus.items.len(), p.n_normal + 30);
        assert_eq!(corpus.n_planted(), 30);
        // every planted item has neighbors (it's wired into the graph)
        for &i in &corpus.planted_indices() {
            assert!(!corpus.items[i].neighbors.is_empty());
        }
    }

    #[test]
    fn benchmark_separates_real_detector_from_random() {
        let corpus = generate(&GenParams::default());

        let rand_scores = RandomDetector { seed: 7 }.score(&corpus);
        let rand_res = evaluate(&corpus, &rand_scores, "random");

        let cm = CrossModalDisagreement;
        let cm_scores = cm.score(&corpus);
        let cm_res = evaluate(&corpus, &cm_scores, cm.name());

        println!(
            "ANOMALY_BENCH random:    AUC={:.3} P@k={:.3}",
            rand_res.auc, rand_res.precision_at_k
        );
        println!(
            "ANOMALY_BENCH crossmodal: AUC={:.3} P@k={:.3} per_kind={:?}",
            cm_res.auc, cm_res.precision_at_k, cm_res.per_kind
        );

        // The instrument must work: random sits at the 0.5 floor...
        assert!((rand_res.auc - 0.5).abs() < 0.12, "random AUC {} not ~0.5", rand_res.auc);
        // ...and a real detector must clearly beat it (proves the substrate is discriminable).
        assert!(
            cm_res.auc > rand_res.auc + 0.2,
            "cross-modal AUC {} did not beat random {}",
            cm_res.auc,
            rand_res.auc
        );
        // Cross-modal disagreement should catch the cross-modal + semantic plants well.
        let cross = cm_res
            .per_kind
            .iter()
            .find(|(k, _)| *k == AnomalyKind::CrossModalMismatch)
            .map(|(_, r)| *r)
            .unwrap_or(0.0);
        assert!(cross > 0.3, "cross-modal recall {} too low", cross);
    }
}
