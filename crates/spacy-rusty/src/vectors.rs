//! Static word vectors + similarity (spaCy `Vocab.vectors`, default mode).
//!
//! Owns the `[n_rows * dim]` table and the ORTH-key -> row map. Shared (via
//! `Rc`) with the tok2vec static-vectors path so the table is held once. Exposes
//! the spaCy vector API: per-word/doc/span vectors, cosine similarity, and
//! `most_similar` (faithful to `Vectors.most_similar`).

use crate::model::Bundle;
use std::collections::HashMap;

pub struct Vectors {
    pub data: Vec<f32>, // [n_rows * dim], row-major
    pub dim: usize,
    pub n_rows: usize,
    pub key2row: HashMap<u64, usize>,
    row2word: Vec<String>,  // row -> representative word (for most_similar)
    filled: Vec<usize>,     // sorted unique rows referenced by key2row
}

impl Vectors {
    /// Build from the bundle's `vectors.data` tensor + the ORTH->row map and the
    /// row->word table. Returns `None` if this model has no static vectors.
    pub fn load(b: &Bundle, key2row: HashMap<u64, usize>, row2word: Vec<String>) -> Option<Vectors> {
        let t = b.tensors.get("vectors.data")?;
        let dim = t.shape[1];
        let n_rows = t.shape[0];
        let mut seen: Vec<usize> = key2row.values().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        Some(Vectors { data: t.data.clone(), dim, n_rows, key2row, row2word, filled: seen })
    }

    pub fn row(&self, orth: u64) -> Option<usize> {
        self.key2row.get(&orth).copied()
    }

    pub fn has_vector(&self, orth: u64) -> bool {
        self.key2row.contains_key(&orth)
    }

    /// The vector for an ORTH key (zeros if out-of-vocabulary).
    pub fn get(&self, orth: u64) -> Vec<f32> {
        match self.row(orth) {
            Some(r) => self.data[r * self.dim..(r + 1) * self.dim].to_vec(),
            None => vec![0f32; self.dim],
        }
    }

    /// Mean of the given ORTH keys' vectors (spaCy Doc/Span `.vector`: an average
    /// over *all* tokens, OOV contributing zeros). Empty -> zero vector.
    pub fn mean(&self, orths: &[u64]) -> Vec<f32> {
        let mut acc = vec![0f64; self.dim];
        if orths.is_empty() {
            return vec![0f32; self.dim];
        }
        for &o in orths {
            if let Some(r) = self.row(o) {
                let base = r * self.dim;
                for i in 0..self.dim {
                    acc[i] += self.data[base + i] as f64;
                }
            }
        }
        let n = orths.len() as f64;
        acc.iter().map(|&x| (x / n) as f32).collect()
    }

    /// L2 norm of a vector.
    pub fn norm(v: &[f32]) -> f32 {
        let s: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
        s.sqrt() as f32
    }

    /// Cosine similarity (spaCy `.similarity`: 0.0 if either vector has 0 norm).
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let na = Self::norm(a) as f64;
        let nb = Self::norm(b) as f64;
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64) * (y as f64)).sum();
        (dot / (na * nb)) as f32
    }

    /// The n most similar entries to `query`, by cosine, over the filled rows.
    /// Mirrors `Vectors.most_similar`: normalize rows + query, dot, top-n, round
    /// scores to 4 decimals and clip to [-1, 1]. Returns (row, word, score).
    pub fn most_similar(&self, query: &[f32], n: usize) -> Vec<(usize, String, f32)> {
        let qn = {
            let q = Self::norm(query) as f64;
            if q == 0.0 { 1.0 } else { q }
        };
        let mut scored: Vec<(usize, f32)> = self
            .filled
            .iter()
            .map(|&r| {
                let base = r * self.dim;
                let row = &self.data[base..base + self.dim];
                let rn = {
                    let v = Self::norm(row) as f64;
                    if v == 0.0 { 1.0 } else { v }
                };
                let dot: f64 =
                    query.iter().zip(row).map(|(&x, &y)| (x as f64) * (y as f64)).sum();
                let mut s = (dot / (qn * rn)) as f32;
                s = (s * 1e4).round() / 1e4; // spaCy rounds to 4 decimals
                s = s.clamp(-1.0, 1.0);
                (r, s)
            })
            .collect();
        // Top-n by score desc; tie-break by row for determinism. (spaCy's
        // argpartition leaves ties in arbitrary order, so only the *set* of
        // returned rows/scores is well-defined when scores tie.)
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0))
        });
        scored
            .into_iter()
            .take(n)
            .map(|(r, s)| {
                let w = self.row2word.get(r).cloned().unwrap_or_default();
                (r, w, s)
            })
            .collect()
    }
}
