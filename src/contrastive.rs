//! Contrastive projection adapter — ER Plan Task 4.2 (Sudowoodo-lite), GATED.
//!
//! A tiny ONLINE-learned linear projection over the frozen MiniLM entity embedding
//! that pulls coreferent surfaces together (`Dali` ≈ `container ship`) without
//! retraining the embedder. Self-supervised from the free-label engine's positive
//! pairs (appositive aliases, same causal-role) — no LLM, no hand labels. Stored as
//! a small `dim×dim` matrix in RocksDB, so it adapts per deployment with no
//! redeploy. Identity-initialised, so **untrained it is a no-op**.
//!
//! We adopt Sudowoodo's *technique* (contrastive self-supervision over augmented /
//! coreferent pairs), not its repo: the online update is a delta-rule contrastive
//! step — pull `W·a` toward a positive `b`, push it away from a negative `n` — which
//! ports to Rust as rank-1 matrix updates and is stable under a bounded learning
//! rate. GATED by `SHODH_CONTRASTIVE_ADAPTER` until it earns the default; drift is
//! bounded by consolidation decay of unreinforced adaptations.

/// A learned linear projection `W` over unit embeddings.
#[derive(Debug, Clone)]
pub struct ContrastiveAdapter {
    dim: usize,
    /// Row-major `dim×dim` projection. Identity at init.
    w: Vec<f32>,
    /// Online learning rate (bounded small for stability).
    lr: f32,
}

fn l2_normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Cosine similarity of two equal-length vectors (0.0 for empty/mismatched).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

impl ContrastiveAdapter {
    /// Identity projection (untrained no-op).
    pub fn identity(dim: usize) -> Self {
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        Self { dim, w, lr: 0.05 }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Project + L2-normalize `x` through `W`. Returns `x` unchanged (normalized) if
    /// dimensions mismatch, so a mis-sized embedding degrades gracefully.
    pub fn project(&self, x: &[f32]) -> Vec<f32> {
        if x.len() != self.dim {
            let mut v = x.to_vec();
            l2_normalize(&mut v);
            return v;
        }
        let mut out = vec![0.0f32; self.dim];
        for (r, o) in out.iter_mut().enumerate() {
            let row = &self.w[r * self.dim..(r + 1) * self.dim];
            *o = row.iter().zip(x).map(|(wij, xj)| wij * xj).sum();
        }
        l2_normalize(&mut out);
        out
    }

    /// One online contrastive step. Pull `W·a` toward the positive's CURRENT
    /// projection `W·b` (so a coreferent pair converges to a shared point rather than
    /// swapping); if a negative `n` is given, push `W·a` away from `W·n`. Delta-rule
    /// rank-1 updates on `W`, scaled by the learning rate — bounded and stable.
    /// No-op on dimension mismatch. Collapse is prevented in practice by negatives.
    pub fn learn(&mut self, a: &[f32], positive: &[f32], negative: Option<&[f32]>) {
        if a.len() != self.dim || positive.len() != self.dim {
            return;
        }
        let pa = self.project(a);
        // Pull W·a toward the positive's current projection: ΔW += lr·(W·b − W·a) ⊗ a
        let pb = self.project(positive);
        self.rank1_update(&pb, &pa, a, self.lr);
        // Push W·a away from the negative's current projection.
        if let Some(n) = negative {
            if n.len() == self.dim {
                let pn = self.project(n);
                self.rank1_update(&pn, &pa, a, -self.lr * 0.5);
            }
        }
    }

    /// `W += scale · (target − current) ⊗ a` — moves the projection of `a` toward
    /// `target` (or away, for negative scale).
    fn rank1_update(&mut self, target: &[f32], current: &[f32], a: &[f32], scale: f32) {
        for r in 0..self.dim {
            let delta_r = scale * (target[r] - current[r]);
            if delta_r == 0.0 {
                continue;
            }
            let row = &mut self.w[r * self.dim..(r + 1) * self.dim];
            for (wij, aj) in row.iter_mut().zip(a) {
                *wij += delta_r * aj;
            }
        }
    }

    /// Serialize `W` for persistence (RocksDB): `dim` (u32 LE) then row-major f32 LE.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.w.len() * 4);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        for x in &self.w {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    /// Reconstruct from `to_bytes`. `None` on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let dim = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
        let expect = 4 + dim * dim * 4;
        if dim == 0 || bytes.len() != expect {
            return None;
        }
        let mut w = Vec::with_capacity(dim * dim);
        for chunk in bytes[4..].chunks_exact(4) {
            w.push(f32::from_le_bytes(chunk.try_into().ok()?));
        }
        Some(Self { dim, w, lr: 0.05 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_a_noop() {
        let a = vec![0.3f32, 0.4, 0.5, 0.1];
        let adapter = ContrastiveAdapter::identity(4);
        let p = adapter.project(&a);
        let mut expect = a.clone();
        l2_normalize(&mut expect);
        for (x, y) in p.iter().zip(&expect) {
            assert!((x - y).abs() < 1e-5, "{p:?} vs {expect:?}");
        }
    }

    #[test]
    fn learning_pulls_a_positive_pair_closer() {
        // Two distinct unit vectors that should be treated as coreferent.
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let before = cosine(&a, &b);
        let mut adapter = ContrastiveAdapter::identity(4).with_lr(0.1);
        for _ in 0..50 {
            adapter.learn(&a, &b, None);
            adapter.learn(&b, &a, None); // symmetric
        }
        let after = cosine(&adapter.project(&a), &adapter.project(&b));
        assert!(
            after > before + 0.1,
            "contrastive learning should raise cos(a,b): {before} -> {after}"
        );
    }

    #[test]
    fn negative_pushes_apart() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let pos = vec![0.9f32, 0.1, 0.0, 0.0];
        let neg = vec![0.0f32, 0.0, 1.0, 0.0];
        let mut adapter = ContrastiveAdapter::identity(4).with_lr(0.1);
        let neg_before = cosine(&adapter.project(&a), &neg);
        for _ in 0..40 {
            adapter.learn(&a, &pos, Some(&neg));
        }
        let neg_after = cosine(&adapter.project(&a), &adapter.project(&neg));
        assert!(
            neg_after <= neg_before + 1e-3,
            "negative should not get closer: {neg_before} -> {neg_after}"
        );
    }

    #[test]
    fn roundtrips_through_bytes() {
        let mut adapter = ContrastiveAdapter::identity(6).with_lr(0.1);
        let a = vec![0.2f32, 0.1, 0.5, 0.3, 0.0, 0.6];
        let b = vec![0.5f32, 0.4, 0.1, 0.2, 0.3, 0.0];
        adapter.learn(&a, &b, None);
        let bytes = adapter.to_bytes();
        let restored = ContrastiveAdapter::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(restored.dim(), 6);
        for (x, y) in restored.project(&a).iter().zip(adapter.project(&a).iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
