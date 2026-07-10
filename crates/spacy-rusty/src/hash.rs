//! The two hash functions spaCy/Thinc use, ported bit-exactly.
//!
//! 1. `hash_string` — spaCy StringStore: MurmurHash64A over UTF-8 bytes, seed 1
//!    (spacy/strings.pyx -> murmurhash.mrmr.hash64). Turns an attr string into
//!    the u64 feature key.
//! 2. `hashembed_hashes` — Thinc HashEmbed: MurmurHash3_x86_128_uint64 over the
//!    u64 key (thinc/backends/numpy_ops.pyx). Pure u64 arithmetic, no byte reads
//!    -> endianness-independent. Returns 4 u32; each `% nV` indexes a row, and
//!    the 4 rows are gather-summed.

use std::collections::HashMap;

/// spaCy StringStore hash: MurmurHash64A(bytes, seed=1).
pub fn hash_string(s: &str) -> u64 {
    murmurhash64a(s.as_bytes(), 1)
}

/// spaCy `get_string_id`: registered symbols return their reserved integer id;
/// the empty string returns 0; everything else hashes via `hash_string`.
pub fn string_id(s: &str, symbols: &HashMap<String, u64>) -> u64 {
    if s.is_empty() {
        return 0;
    }
    if let Some(&id) = symbols.get(s) {
        return id;
    }
    hash_string(s)
}

fn murmurhash64a(data: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4a7935bd1e995;
    const R: u32 = 47;
    let len = data.len();
    let mut h: u64 = seed ^ (len as u64).wrapping_mul(M);

    let nblocks = len / 8;
    for i in 0..nblocks {
        let mut k = u64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap());
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }

    let tail = &data[nblocks * 8..];
    let rem = len & 7;
    // MurmurHash64A tail switch (fallthrough high->low byte, then one *M).
    if rem >= 7 {
        h ^= (tail[6] as u64) << 48;
    }
    if rem >= 6 {
        h ^= (tail[5] as u64) << 40;
    }
    if rem >= 5 {
        h ^= (tail[4] as u64) << 32;
    }
    if rem >= 4 {
        h ^= (tail[3] as u64) << 24;
    }
    if rem >= 3 {
        h ^= (tail[2] as u64) << 16;
    }
    if rem >= 2 {
        h ^= (tail[1] as u64) << 8;
    }
    if rem >= 1 {
        h ^= tail[0] as u64;
        h = h.wrapping_mul(M);
    }

    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

#[inline]
fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    k
}

/// Thinc HashEmbed hash: MurmurHash3_x86_128_uint64(val, seed) -> 4 u32.
pub fn hashembed_hashes(val: u64, seed: u32) -> [u32; 4] {
    let mut h1: u64 = val;
    h1 = h1.wrapping_mul(0x87c37b91114253d5);
    h1 = h1.rotate_left(31);
    h1 = h1.wrapping_mul(0x4cf5ad432745937f);
    h1 ^= seed as u64;
    h1 ^= 8;
    let mut h2: u64 = seed as u64;
    h2 ^= 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [
        (h1 & 0xffff_ffff) as u32,
        (h1 >> 32) as u32,
        (h2 & 0xffff_ffff) as u32,
        (h2 >> 32) as u32,
    ]
}

/// The 4 embedding row indices for a key in a table of `n_rows` rows.
pub fn hashembed_rows(val: u64, seed: u32, n_rows: u32) -> [u32; 4] {
    let h = hashembed_hashes(val, seed);
    [h[0] % n_rows, h[1] % n_rows, h[2] % n_rows, h[3] % n_rows]
}
