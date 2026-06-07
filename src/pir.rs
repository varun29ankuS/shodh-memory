//! Private Information Retrieval (encryption-v2 P7).
//!
//! Threat model **T3**: a hosted / multi-tenant service where the client fetches a
//! record WITHOUT the server learning which one. Near-term for shodh/substrate.
//! Scope: **exact-key private fetch only** — private semantic / vector recall
//! (private nearest-neighbour) is research-grade and explicitly out of scope
//! (design §9); hosted recall stays non-private until a practical scheme exists.
//!
//! This ships a real, tested **2-server information-theoretic PIR** (XOR scheme,
//! Chor–Goldreich–Kushilevitz–Sudan): trivial compute, *perfect* query privacy
//! provided the two servers do not collude. The DB is `N` fixed-size rows; the
//! client retrieves row `i` so that each server sees only a uniformly random
//! bitvector and learns nothing about `i`.
//!
//! The single-server **cPIR** alternative (SealPIR / Spiral-class) removes the
//! non-collusion assumption but requires a homomorphic-encryption dependency; it
//! is documented as future work and intentionally NOT implemented here (no
//! fake-shipped crypto).

use anyhow::{anyhow, Result};
use rand::rngs::OsRng;
use rand::RngCore;

/// A PIR query share for one server: one bit per DB row.
#[derive(Clone, Debug)]
pub struct QueryShare {
    bits: Vec<bool>,
}

impl QueryShare {
    pub fn len(&self) -> usize {
        self.bits.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }
}

/// Build the two query shares for retrieving row `index` of `n_rows`. Send share
/// 0 to server A and share 1 to server B; the servers must not collude.
pub fn build_query(index: usize, n_rows: usize) -> Result<(QueryShare, QueryShare)> {
    if index >= n_rows {
        return Err(anyhow!("index {index} out of range (n_rows {n_rows})"));
    }
    // Uniformly random bitvector for share A.
    let mut buf = vec![0u8; n_rows.div_ceil(8)];
    OsRng.fill_bytes(&mut buf);
    let mut a = vec![false; n_rows];
    for (j, bit) in a.iter_mut().enumerate() {
        *bit = (buf[j / 8] >> (j % 8)) & 1 == 1;
    }
    // Share B = A XOR e_index, so A XOR B = e_index (selects exactly row `index`).
    let mut b = a.clone();
    b[index] ^= true;
    Ok((QueryShare { bits: a }, QueryShare { bits: b }))
}

/// Server side: XOR together every row whose query bit is set. `db` is `N` rows,
/// each exactly `row_len` bytes (pad rows to a fixed size before serving).
pub fn answer(share: &QueryShare, db: &[Vec<u8>], row_len: usize) -> Result<Vec<u8>> {
    if share.bits.len() != db.len() {
        return Err(anyhow!(
            "query/db length mismatch: {} vs {}",
            share.bits.len(),
            db.len()
        ));
    }
    let mut acc = vec![0u8; row_len];
    for (bit, row) in share.bits.iter().zip(db.iter()) {
        if *bit {
            if row.len() != row_len {
                return Err(anyhow!("row length mismatch (rows must be padded equal)"));
            }
            for (a, r) in acc.iter_mut().zip(row.iter()) {
                *a ^= r;
            }
        }
    }
    Ok(acc)
}

/// Client side: XOR the two servers' answers to recover the requested row.
pub fn reconstruct(ans_a: &[u8], ans_b: &[u8]) -> Result<Vec<u8>> {
    if ans_a.len() != ans_b.len() {
        return Err(anyhow!("answer length mismatch"));
    }
    Ok(ans_a.iter().zip(ans_b).map(|(a, b)| a ^ b).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Vec<Vec<u8>> {
        vec![
            b"row-zero".to_vec(),
            b"row-one!".to_vec(),
            b"row-two!".to_vec(),
            b"row-3!!!".to_vec(),
            b"row-four".to_vec(),
        ]
    }

    #[test]
    fn two_server_pir_retrieves_correct_row() {
        let db = db();
        let row_len = 8;
        for i in 0..db.len() {
            let (qa, qb) = build_query(i, db.len()).unwrap();
            let aa = answer(&qa, &db, row_len).unwrap();
            let ab = answer(&qb, &db, row_len).unwrap();
            assert_eq!(reconstruct(&aa, &ab).unwrap(), db[i], "row {i}");
        }
    }

    #[test]
    fn shares_differ_only_at_target_bit() {
        let n = 7;
        let (qa, qb) = build_query(3, n).unwrap();
        let diffs: Vec<usize> = (0..n).filter(|&j| qa.bits[j] != qb.bits[j]).collect();
        // Exactly one differing position — and it's the target. Each share alone
        // is a uniform random bitvector (privacy).
        assert_eq!(diffs, vec![3]);
    }

    #[test]
    fn index_out_of_range_errors() {
        assert!(build_query(5, 5).is_err());
        assert!(build_query(0, 0).is_err());
    }

    #[test]
    fn length_mismatches_error() {
        let db = db();
        let (qa, _) = build_query(0, db.len()).unwrap();
        // wrong db length
        assert!(answer(&qa, &db[..3], 8).is_err());
        // wrong answer lengths
        assert!(reconstruct(&[0u8; 8], &[0u8; 4]).is_err());
    }
}
