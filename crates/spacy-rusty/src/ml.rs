//! Small dense ops matching Thinc's inference math.
//!
//! The dot product accumulates in f32 across four independent lanes — the same
//! f32 arithmetic Thinc/blis (`sgemm`) uses, so this is *more* faithful to real
//! spaCy than our earlier f64 path, not less. Four lanes break the dependent
//! FMA chain so the compiler emits NEON/SSE `fmla` (latency-bound scalar ->
//! throughput-bound, ~2x). The fixed 4-way reassociation + f32 rounding leaves
//! every downstream argmax/threshold decision unchanged: tests/perf_identity.rs
//! verifies the full annotation set is byte-identical to the prior f64 runtime
//! across all 1000 held-out sentences for sm/md/lg, and golden_tok2vec still
//! matches spaCy's doc.tensor to <1e-4.

/// `acc + w*x`, lane-wise. With `relaxed-simd` this is `f32x4_relaxed_madd`
/// (a single fused multiply-add, the same op blis/Thinc `sgemm` uses — so it is
/// at least as faithful to real spaCy as the split mul+add); otherwise it is the
/// strict `add(mul(w,x), acc)` that reproduces the captured baseline bit-for-bit.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn madd(
    acc: core::arch::wasm32::v128,
    w: core::arch::wasm32::v128,
    x: core::arch::wasm32::v128,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    #[cfg(target_feature = "relaxed-simd")]
    {
        f32x4_relaxed_madd(w, x, acc)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        f32x4_add(acc, f32x4_mul(w, x))
    }
}

/// Dot product of a weight row and `x` (f32 accumulation, 4 lanes; widened to
/// f64 only at the return so callers' bias adds keep a little headroom).
///
/// On wasm with simd128 this uses one `f32x4` accumulator: lane `j` accumulates
/// exactly the same terms (w[j], w[j+4], …) the scalar lane `aj` would, and the
/// final reduction `((a0+a1)+(a2+a3))+tail` is identical — so every downstream
/// argmax/threshold is bit-for-bit unchanged (perf_identity.rs holds).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub(crate) fn dot(wrow: &[f32], x: &[f32]) -> f64 {
    use core::arch::wasm32::*;
    let n = wrow.len();
    let mut acc = f32x4_splat(0.0);
    let wp = wrow.as_ptr();
    let xp = x.as_ptr();
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: i+4<=n and both slices hold >= n elements; wasm v128 loads
        // are alignment-agnostic.
        unsafe {
            let wv = v128_load(wp.add(i) as *const v128);
            let xv = v128_load(xp.add(i) as *const v128);
            acc = madd(acc, wv, xv);
        }
        i += 4;
    }
    let a0 = f32x4_extract_lane::<0>(acc);
    let a1 = f32x4_extract_lane::<1>(acc);
    let a2 = f32x4_extract_lane::<2>(acc);
    let a3 = f32x4_extract_lane::<3>(acc);
    let mut tail = 0f32;
    while i < n {
        tail += wrow[i] * x[i];
        i += 1;
    }
    (((a0 + a1) + (a2 + a3)) + tail) as f64
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
pub(crate) fn dot(wrow: &[f32], x: &[f32]) -> f64 {
    let n = wrow.len();
    let (mut a0, mut a1, mut a2, mut a3) = (0f32, 0f32, 0f32, 0f32);
    let mut i = 0;
    while i + 4 <= n {
        a0 += wrow[i] * x[i];
        a1 += wrow[i + 1] * x[i + 1];
        a2 += wrow[i + 2] * x[i + 2];
        a3 += wrow[i + 3] * x[i + 3];
        i += 4;
    }
    let mut tail = 0f32;
    while i < n {
        tail += wrow[i] * x[i];
        i += 1;
    }
    (((a0 + a1) + (a2 + a3)) + tail) as f64
}

/// Four dot products (four weight rows `w0..w3`, all against the same `x`).
///
/// Register-blocked: one `x` load feeds all four rows and the four `f32x4`
/// accumulators form four independent dependency chains, hiding add latency.
/// Each row's lane layout and final reduction are identical to `dot`, so the
/// result of `dot4(...)[k]` equals `dot(wk, x)` bit-for-bit.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn dot4(w0: &[f32], w1: &[f32], w2: &[f32], w3: &[f32], x: &[f32]) -> [f64; 4] {
    use core::arch::wasm32::*;
    let n = x.len();
    let (mut a0, mut a1, mut a2, mut a3) =
        (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
    let (p0, p1, p2, p3, px) =
        (w0.as_ptr(), w1.as_ptr(), w2.as_ptr(), w3.as_ptr(), x.as_ptr());
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: i+4<=n and every slice holds >= n elements; wasm v128 loads
        // are alignment-agnostic.
        unsafe {
            let xv = v128_load(px.add(i) as *const v128);
            a0 = madd(a0, v128_load(p0.add(i) as *const v128), xv);
            a1 = madd(a1, v128_load(p1.add(i) as *const v128), xv);
            a2 = madd(a2, v128_load(p2.add(i) as *const v128), xv);
            a3 = madd(a3, v128_load(p3.add(i) as *const v128), xv);
        }
        i += 4;
    }
    #[inline]
    fn red(a: core::arch::wasm32::v128) -> f32 {
        use core::arch::wasm32::*;
        let l0 = f32x4_extract_lane::<0>(a);
        let l1 = f32x4_extract_lane::<1>(a);
        let l2 = f32x4_extract_lane::<2>(a);
        let l3 = f32x4_extract_lane::<3>(a);
        (l0 + l1) + (l2 + l3)
    }
    let (mut s0, mut s1, mut s2, mut s3) = (red(a0), red(a1), red(a2), red(a3));
    let (mut t0, mut t1, mut t2, mut t3) = (0f32, 0f32, 0f32, 0f32);
    while i < n {
        t0 += w0[i] * x[i];
        t1 += w1[i] * x[i];
        t2 += w2[i] * x[i];
        t3 += w3[i] * x[i];
        i += 1;
    }
    s0 += t0;
    s1 += t1;
    s2 += t2;
    s3 += t3;
    [s0 as f64, s1 as f64, s2 as f64, s3 as f64]
}

/// Eight dot products sharing one `x` — like `dot4` with twice the independent
/// accumulator chains, which hides `f32x4_add` latency on the wide window
/// matmul. `dot8(...)[k] == dot(wk, x)` bit-for-bit.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn dot8(w: [&[f32]; 8], x: &[f32]) -> [f64; 8] {
    use core::arch::wasm32::*;
    let n = x.len();
    let mut a = [f32x4_splat(0.0); 8];
    let p: [*const f32; 8] = [
        w[0].as_ptr(), w[1].as_ptr(), w[2].as_ptr(), w[3].as_ptr(),
        w[4].as_ptr(), w[5].as_ptr(), w[6].as_ptr(), w[7].as_ptr(),
    ];
    let px = x.as_ptr();
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: i+4<=n and every slice holds >= n elements; wasm v128 loads
        // are alignment-agnostic.
        unsafe {
            let xv = v128_load(px.add(i) as *const v128);
            a[0] = madd(a[0], v128_load(p[0].add(i) as *const v128), xv);
            a[1] = madd(a[1], v128_load(p[1].add(i) as *const v128), xv);
            a[2] = madd(a[2], v128_load(p[2].add(i) as *const v128), xv);
            a[3] = madd(a[3], v128_load(p[3].add(i) as *const v128), xv);
            a[4] = madd(a[4], v128_load(p[4].add(i) as *const v128), xv);
            a[5] = madd(a[5], v128_load(p[5].add(i) as *const v128), xv);
            a[6] = madd(a[6], v128_load(p[6].add(i) as *const v128), xv);
            a[7] = madd(a[7], v128_load(p[7].add(i) as *const v128), xv);
        }
        i += 4;
    }
    let red = |v: v128| -> f32 {
        let l0 = f32x4_extract_lane::<0>(v);
        let l1 = f32x4_extract_lane::<1>(v);
        let l2 = f32x4_extract_lane::<2>(v);
        let l3 = f32x4_extract_lane::<3>(v);
        (l0 + l1) + (l2 + l3)
    };
    let mut s = [red(a[0]), red(a[1]), red(a[2]), red(a[3]),
                 red(a[4]), red(a[5]), red(a[6]), red(a[7])];
    while i < n {
        for k in 0..8 {
            s[k] += w[k][i] * x[i];
        }
        i += 1;
    }
    [s[0] as f64, s[1] as f64, s[2] as f64, s[3] as f64,
     s[4] as f64, s[5] as f64, s[6] as f64, s[7] as f64]
}

/// Maxout into a caller-provided `out` buffer (no allocation). See `maxout`.
///
/// On wasm/simd128 each `o`-block's outputs share `x` loads across their weight
/// rows for a given piece (`dot8`/`dot4`), then reduce over pieces. Arithmetic
/// per (o,p) is identical to the scalar path → bit-for-bit output.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn maxout_into(x: &[f32], w: &[f32], b: &[f32], n_o: usize, n_p: usize, n_i: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), n_i);
    debug_assert_eq!(out.len(), n_o);
    let row = |o: usize, p: usize| -> &[f32] {
        let r = (o * n_p + p) * n_i;
        &w[r..r + n_i]
    };
    let mut o = 0;
    while o + 8 <= n_o {
        let mut m = [f64::NEG_INFINITY; 8];
        for p in 0..n_p {
            let d = dot8(
                [row(o, p), row(o + 1, p), row(o + 2, p), row(o + 3, p),
                 row(o + 4, p), row(o + 5, p), row(o + 6, p), row(o + 7, p)],
                x,
            );
            for k in 0..8 {
                let v = b[(o + k) * n_p + p] as f64 + d[k];
                if v > m[k] { m[k] = v; }
            }
        }
        for k in 0..8 {
            out[o + k] = m[k] as f32;
        }
        o += 8;
    }
    while o + 4 <= n_o {
        let (mut m0, mut m1, mut m2, mut m3) =
            (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in 0..n_p {
            let d = dot4(row(o, p), row(o + 1, p), row(o + 2, p), row(o + 3, p), x);
            let v0 = b[(o + 0) * n_p + p] as f64 + d[0];
            let v1 = b[(o + 1) * n_p + p] as f64 + d[1];
            let v2 = b[(o + 2) * n_p + p] as f64 + d[2];
            let v3 = b[(o + 3) * n_p + p] as f64 + d[3];
            if v0 > m0 { m0 = v0; }
            if v1 > m1 { m1 = v1; }
            if v2 > m2 { m2 = v2; }
            if v3 > m3 { m3 = v3; }
        }
        out[o + 0] = m0 as f32;
        out[o + 1] = m1 as f32;
        out[o + 2] = m2 as f32;
        out[o + 3] = m3 as f32;
        o += 4;
    }
    while o < n_o {
        let mut best = f64::NEG_INFINITY;
        for p in 0..n_p {
            let wbase = (o * n_p + p) * n_i;
            let acc = b[o * n_p + p] as f64 + dot(&w[wbase..wbase + n_i], x);
            if acc > best {
                best = acc;
            }
        }
        out[o] = best as f32;
        o += 1;
    }
}

/// Maxout into a caller-provided `out` buffer (no allocation). See `maxout`.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub fn maxout_into(x: &[f32], w: &[f32], b: &[f32], n_o: usize, n_p: usize, n_i: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), n_i);
    debug_assert_eq!(out.len(), n_o);
    for o in 0..n_o {
        let mut best = f64::NEG_INFINITY;
        for p in 0..n_p {
            let wbase = (o * n_p + p) * n_i;
            let acc = b[o * n_p + p] as f64 + dot(&w[wbase..wbase + n_i], x);
            if acc > best {
                best = acc;
            }
        }
        out[o] = best as f32;
    }
}

/// Maxout: W is [nO, nP, nI] row-major, b is [nO, nP]. Returns nO values,
/// each `max_p ( W[o,p,:]·x + b[o,p] )`.
pub fn maxout(x: &[f32], w: &[f32], b: &[f32], n_o: usize, n_p: usize, n_i: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_o];
    maxout_into(x, w, b, n_o, n_p, n_i, &mut out);
    out
}

/// Affine: W is [nO, nI] row-major, b is [nO]. Returns nO values `W·x + b`.
pub fn affine(x: &[f32], w: &[f32], b: &[f32], n_o: usize, n_i: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), n_i);
    let mut out = vec![0f32; n_o];
    for o in 0..n_o {
        out[o] = (b[o] as f64 + dot(&w[o * n_i..o * n_i + n_i], x)) as f32;
    }
    out
}

/// Linear (no bias): W is [nO, nI] row-major. Returns nO values `W·x`.
/// Numerically identical to `affine` with a zero bias, without allocating one.
pub fn linear(x: &[f32], w: &[f32], n_o: usize, n_i: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), n_i);
    let mut out = vec![0f32; n_o];
    for o in 0..n_o {
        out[o] = dot(&w[o * n_i..o * n_i + n_i], x) as f32;
    }
    out
}

/// Thinc LayerNorm (in place): mu/var over the row, var += 1e-8, then scale+shift.
pub fn layernorm(x: &mut [f32], g: &[f32], b: &[f32]) {
    let n = x.len();
    let mut mean = 0f64;
    for &v in x.iter() {
        mean += v as f64;
    }
    mean /= n as f64;
    let mut var = 0f64;
    for &v in x.iter() {
        let d = v as f64 - mean;
        var += d * d;
    }
    var = var / n as f64 + 1e-8;
    let inv = var.sqrt().recip();
    for i in 0..n {
        let xhat = (x[i] as f64 - mean) * inv;
        x[i] = (xhat as f32) * g[i] + b[i];
    }
}

/// Argmax over a slice.
pub fn argmax(x: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi
}
