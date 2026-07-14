//! tok2vec forward pass: MultiHashEmbed (concat of per-table gather-add
//! embeddings [+ optional static vectors], maxout-projection, layernorm)
//! followed by a MaxoutWindowEncoder (residual window-CNN blocks).

use crate::hash::hashembed_rows;
use crate::ml::{dot, layernorm, maxout_into};
use crate::model::Bundle;
use crate::vectors::Vectors;
use serde_json::Value;
use std::sync::Arc;

struct EncLayer {
    w: Vec<f32>,
    b: Vec<f32>,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
}

pub struct Tok2Vec {
    width: usize,
    n_tables: usize,
    rows: Vec<u32>,
    seeds: Vec<u32>,
    tables: Vec<Vec<f32>>, // each [nV * width]
    include_static: bool,
    static_w: Vec<f32>, // [width * vec_dim]
    vec_dim: usize,
    vectors: Option<Arc<Vectors>>, // shared static-vector table (ORTH key -> row)
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    proj_ln_g: Vec<f32>,
    proj_ln_b: Vec<f32>,
    proj_np: usize,
    concat: usize,
    enc: Vec<EncLayer>,
    enc_np: usize,
    window: usize,
}

fn vecf(b: &Bundle, key: &str) -> Vec<f32> {
    b.get(key).data.clone()
}

impl Tok2Vec {
    /// `cfg` is the manifest object for this tok2vec; `tp` is the tensor-key
    /// prefix (e.g. "tok2vec" or "ner.tok2vec"). `vectors` is the shared static-
    /// vector table (`None` if this tok2vec doesn't use static vectors).
    pub fn load(b: &Bundle, cfg: &Value, tp: &str, vectors: Option<Arc<Vectors>>) -> Tok2Vec {
        let width = cfg["width"].as_u64().unwrap() as usize;
        let n_tables = cfg["n_tables"].as_u64().unwrap() as usize;
        let rows: Vec<u32> =
            cfg["rows"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let seeds: Vec<u32> =
            cfg["seeds"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let tables: Vec<Vec<f32>> =
            (0..n_tables).map(|c| vecf(b, &format!("{}.embed.table{}.E", tp, c))).collect();
        let include_static = cfg["include_static_vectors"].as_bool().unwrap_or(false);
        let (static_w, vec_dim, vectors) = if include_static {
            let sw = vecf(b, &format!("{}.static.W", tp));
            let vd = vectors.as_ref().map(|v| v.dim).unwrap_or(0);
            (sw, vd, vectors)
        } else {
            (vec![], 0, None)
        };
        let concat = cfg["embed_proj"]["nI"].as_u64().unwrap() as usize;
        let proj_np = cfg["embed_proj"]["nP"].as_u64().unwrap() as usize;
        let enc_depth = cfg["encoder"]["depth"].as_u64().unwrap() as usize;
        let enc_np = cfg["encoder"]["nP"].as_u64().unwrap() as usize;
        let window = cfg["encoder"]["window"].as_u64().unwrap() as usize;
        let enc: Vec<EncLayer> = (0..enc_depth)
            .map(|k| EncLayer {
                w: vecf(b, &format!("{}.encoder.L{}.maxout.W", tp, k)),
                b: vecf(b, &format!("{}.encoder.L{}.maxout.b", tp, k)),
                ln_g: vecf(b, &format!("{}.encoder.L{}.ln.G", tp, k)),
                ln_b: vecf(b, &format!("{}.encoder.L{}.ln.b", tp, k)),
            })
            .collect();

        Tok2Vec {
            width,
            n_tables,
            rows,
            seeds,
            tables,
            include_static,
            static_w,
            vec_dim,
            vectors,
            proj_w: vecf(b, &format!("{}.embed.proj.W", tp)),
            proj_b: vecf(b, &format!("{}.embed.proj.b", tp)),
            proj_ln_g: vecf(b, &format!("{}.embed.proj.ln.G", tp)),
            proj_ln_b: vecf(b, &format!("{}.embed.proj.ln.b", tp)),
            proj_np,
            concat,
            enc,
            enc_np,
            window,
        }
    }

    /// Build the pre-projection embedding row (`concat`, length `self.concat`):
    /// the per-table gather-add segments followed by the optional static-vector
    /// projection. Same gather order / arithmetic as before.
    fn embed_concat_into(&self, feats: &[u64], orth: Option<u64>, concat: &mut [f32]) {
        let w = self.width;
        for c in 0..self.n_tables {
            let rows4 = hashembed_rows(feats[c], self.seeds[c], self.rows[c]);
            let seg = &mut concat[c * w..(c + 1) * w];
            seg.fill(0.0);
            let table = &self.tables[c];
            for &r in &rows4 {
                let base = (r as usize) * w;
                for i in 0..w {
                    seg[i] += table[base + i];
                }
            }
        }
        if self.include_static {
            let off = self.n_tables * w;
            let seg = &mut concat[off..off + w];
            // StaticVectors projection: V · Wᵀ, W is [width, vec_dim], no bias.
            // No vector row -> V is zero -> projection is zero (no bias to add).
            match orth.and_then(|key| {
                self.vectors.as_ref().and_then(|vt| vt.row(key).map(|r| (vt, r)))
            }) {
                Some((vt, row)) => {
                    let base = row * self.vec_dim;
                    let v = &vt.data[base..base + self.vec_dim];
                    for o in 0..w {
                        seg[o] = dot(&self.static_w[o * self.vec_dim..(o + 1) * self.vec_dim], v) as f32;
                    }
                }
                None => seg.fill(0.0),
            }
        }
    }

    /// Embed one token directly into `out` (length `width`), using `concat` as
    /// reusable scratch (length `self.concat`). No per-token allocation. The
    /// arithmetic — gather-add order, static projection, maxout, layernorm — is
    /// identical to the previous `Vec`-returning path.
    fn embed_into(&self, feats: &[u64], orth: Option<u64>, out: &mut [f32], concat: &mut [f32]) {
        let w = self.width;
        self.embed_concat_into(feats, orth, concat);
        maxout_into(concat, &self.proj_w, &self.proj_b, w, self.proj_np, self.concat, out);
        layernorm(out, &self.proj_ln_g, &self.proj_ln_b);
    }

    /// Embed-only output (MultiHashEmbed, before the window encoder).
    pub fn embed_seq(&self, feats: &[Vec<u64>], orths: Option<&[u64]>) -> Vec<Vec<f32>> {
        let mut concat = vec![0f32; self.concat];
        (0..feats.len())
            .map(|t| {
                let mut out = vec![0f32; self.width];
                self.embed_into(&feats[t], orths.map(|o| o[t]), &mut out, &mut concat);
                out
            })
            .collect()
    }

    /// Run the full tok2vec over a sequence. `feats[t]` are the n_tables hash
    /// keys; `orths[t]` is the ORTH key for static vectors (if used).
    pub fn forward(&self, feats: &[Vec<u64>], orths: Option<&[u64]>) -> Vec<Vec<f32>> {
        let w = self.width;
        let n = feats.len();

        // spaCy wraps the encoder in with_array(model, pad=receptive_field):
        // the sequence is zero-padded by `pad` rows at each end ONCE, the whole
        // residual stack runs over it (so the pad rows evolve and feed the real
        // boundary tokens' windows), then the padding is stripped.
        //
        // Flat ping-pong buffers (`cur`/`next`, each [len*w]) + reused window
        // and row scratch. The encoder is *token-stationary*: each token's
        // window (`x`, ~winlen f32) is reused across all output rows of the
        // maxout while it stays hot in registers/L1, and the layer's weight
        // matrix is small enough to stay resident in L2 across tokens. (A
        // weight-stationary batch variant was tried and measured slower — it
        // re-streams the windowed inputs once per output row.) Same math and
        // summation order as before → bit-for-bit identical.
        let pad = self.window * self.enc.len();
        let len = n + 2 * pad;
        let mut cur = vec![0f32; len * w];
        let mut concat = vec![0f32; self.concat];
        for t in 0..n {
            let s = (pad + t) * w;
            self.embed_into(&feats[t], orths.map(|o| o[t]), &mut cur[s..s + w], &mut concat);
        }
        let mut next = vec![0f32; len * w];
        let winlen = (2 * self.window + 1) * w;
        let mut win = vec![0f32; winlen];
        let mut row = vec![0f32; w];
        let win_i = self.window as isize;
        for layer in &self.enc {
            for t in 0..len {
                // expand_window(window): concat [t-window .. t+window]
                for (k, d) in (-win_i..=win_i).enumerate() {
                    let dst = &mut win[k * w..(k + 1) * w];
                    let idx = t as isize + d;
                    if idx < 0 || idx >= len as isize {
                        dst.fill(0.0);
                    } else {
                        let s = idx as usize * w;
                        dst.copy_from_slice(&cur[s..s + w]);
                    }
                }
                maxout_into(&win, &layer.w, &layer.b, w, self.enc_np, winlen, &mut row);
                layernorm(&mut row, &layer.ln_g, &layer.ln_b);
                let base = t * w;
                for i in 0..w {
                    next[base + i] = row[i] + cur[base + i]; // residual
                }
            }
            std::mem::swap(&mut cur, &mut next);
        }
        (0..n).map(|t| cur[(pad + t) * w..(pad + t + 1) * w].to_vec()).collect()
    }
}
