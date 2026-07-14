//! Shared neural scoring for spaCy's transition-based parsers (NER + the
//! dependency parser). Both use the SAME machinery — reduce(t2v_width -> hidden)
//! -> PrecomputableAffine lower (nF state features, nP maxout pieces) -> maxout
//! -> upper Linear(-> n_classes) — and differ only in the *feature indices* and
//! the move grammar (which live in ner.rs / parser.rs). This module is just the
//! math, so the two systems share one verified implementation.

use crate::ml::{affine, dot};
use crate::model::Bundle;

pub struct Scorer {
    reduce_w: Vec<f32>,
    reduce_b: Vec<f32>,
    reduce_no: usize, // hidden width (64)
    reduce_ni: usize, // tok2vec width (96)
    lower_w: Vec<f32>,   // [nF, nO, nP, nI]
    lower_b: Vec<f32>,   // [nO, nP]
    lower_pad: Vec<f32>, // [1, nF, nO, nP] -> used as [nF, nO, nP]
    pub n_f: usize,
    n_o: usize,
    n_p: usize,
    upper_w: Vec<f32>, // [n_classes, nO]
    upper_b: Vec<f32>,
    pub n_classes: usize,
}

impl Scorer {
    /// Load `{prefix}.reduce/lower/upper` from the bundle, with dims read from
    /// `manifest[prefix].{reduce,lower,upper}`.
    pub fn load(b: &Bundle, prefix: &str) -> Scorer {
        let cfg = &b.manifest[prefix];
        let reduce = b.get(&format!("{}.reduce.W", prefix));
        let lower = b.get(&format!("{}.lower.W", prefix));
        let upper = b.get(&format!("{}.upper.W", prefix));
        Scorer {
            reduce_w: reduce.data.clone(),
            reduce_b: b.get(&format!("{}.reduce.b", prefix)).data.clone(),
            reduce_no: reduce.shape[0],
            reduce_ni: reduce.shape[1],
            lower_w: lower.data.clone(),
            lower_b: b.get(&format!("{}.lower.b", prefix)).data.clone(),
            lower_pad: b.get(&format!("{}.lower.pad", prefix)).data.clone(),
            n_f: cfg["lower"]["nF"].as_u64().unwrap() as usize,
            n_o: cfg["lower"]["nO"].as_u64().unwrap() as usize,
            n_p: cfg["lower"]["nP"].as_u64().unwrap() as usize,
            upper_w: upper.data.clone(),
            upper_b: b.get(&format!("{}.upper.b", prefix)).data.clone(),
            n_classes: upper.shape[0],
        }
    }

    /// Reduce each token vector (t2v_width -> hidden), then precompute the lower
    /// PrecomputableAffine output Yf[t], flattened as [f][o][p] (len nF*nO*nP).
    /// `vecs` are the tok2vec outputs feeding this parser (NER's own tok2vec, or
    /// the shared tok2vec for the dependency parser).
    pub fn precompute(&self, vecs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let fop = self.n_f * self.n_o * self.n_p;
        let ni = self.reduce_no;
        vecs.iter()
            .map(|v| {
                let x = affine(v, &self.reduce_w, &self.reduce_b, self.reduce_no, self.reduce_ni);
                let mut out = vec![0f32; fop];
                for f in 0..self.n_f {
                    for o in 0..self.n_o {
                        for p in 0..self.n_p {
                            let wbase = (((f * self.n_o + o) * self.n_p) + p) * ni;
                            out[f * self.n_o * self.n_p + o * self.n_p + p] =
                                dot(&self.lower_w[wbase..wbase + ni], &x) as f32;
                        }
                    }
                }
                out
            })
            .collect()
    }

    /// Score all classes for a state whose nF feature-token ids are `ids`
    /// (id < 0 => the lower `pad` row). `ids.len()` must equal `n_f`.
    pub fn score(&self, yf: &[Vec<f32>], ids: &[i64]) -> Vec<f32> {
        let op = self.n_o * self.n_p;
        let mut unmaxed = vec![0f32; op];
        for f in 0..self.n_f {
            let block: &[f32] = if ids[f] < 0 {
                &self.lower_pad[f * op..f * op + op]
            } else {
                &yf[ids[f] as usize][f * op..f * op + op]
            };
            for k in 0..op {
                unmaxed[k] += block[k];
            }
        }
        for k in 0..op {
            unmaxed[k] += self.lower_b[k];
        }
        // maxout over pieces -> hidden (n_o)
        let mut hidden = vec![0f32; self.n_o];
        for o in 0..self.n_o {
            let mut best = f32::NEG_INFINITY;
            for p in 0..self.n_p {
                let val = unmaxed[o * self.n_p + p];
                if val > best {
                    best = val;
                }
            }
            hidden[o] = best;
        }
        affine(&hidden, &self.upper_w, &self.upper_b, self.n_classes, self.n_o)
    }
}
