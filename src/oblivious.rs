//! Oblivious-access seam (encryption-v2 P6).
//!
//! Access-pattern hiding — threat model **T2** (a host / co-tenant / hypervisor
//! observing *which* blocks the process reads and writes over time) — is NOT
//! provided by record encryption or index blinding alone. This module is the
//! seam the record + index stores route block access through, so a Path-ORAM
//! backend can slot in behind an `oblivious` feature later WITHOUT re-architecting
//! callers, plus the cheap sub-ORAM mitigations that *reduce* (not eliminate)
//! access-pattern leakage.
//!
//! Status (honest): the [`BlockStore`] trait, a direct in-memory impl, and the
//! [`pad_to_bucket`] mitigation ship now and are tested. Routing the live storage
//! layer onto `BlockStore`, the dummy-query / write-back-re-encryption helpers,
//! and a Path-ORAM impl are tracked follow-ups (design §8). Note: even under a
//! future `oblivious` mode, semantic/vector recall stays non-oblivious unless
//! routed through the oblivious layer — that limitation must be surfaced, not
//! silently claimed (design §11 B-6).

use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::RwLock;

/// Block-level access seam. The record + index stores route reads/writes through
/// this so an oblivious (Path-ORAM) backend can replace the direct one without
/// touching call sites.
pub trait BlockStore: Send + Sync {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn delete(&self, key: &[u8]) -> Result<()>;
    /// Sorted keys sharing `prefix`. A Path-ORAM impl serves these obliviously.
    fn keys_with_prefix(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>>;
}

/// Direct, NON-oblivious in-memory block store — reference impl and test double.
/// (The production direct impl wraps RocksDB; the oblivious impl is Path-ORAM.)
#[derive(Default)]
pub struct InMemoryBlockStore {
    map: RwLock<BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl BlockStore for InMemoryBlockStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.map.read().unwrap().get(key).cloned())
    }
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.map
            .write()
            .unwrap()
            .insert(key.to_vec(), value.to_vec());
        Ok(())
    }
    fn delete(&self, key: &[u8]) -> Result<()> {
        self.map.write().unwrap().remove(key);
        Ok(())
    }
    fn keys_with_prefix(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        Ok(self
            .map
            .read()
            .unwrap()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }
}

/// Cheap T2 mitigation: pad a result set up to the next multiple of `bucket` with
/// `None`s so the *count* of matches isn't revealed by the returned length.
/// Reduces (does not eliminate) volume leakage; pair with dummy queries and
/// write-back re-encryption. Real access-pattern hiding requires ORAM.
pub fn pad_to_bucket<T>(mut results: Vec<Option<T>>, bucket: usize) -> Vec<Option<T>> {
    if bucket == 0 {
        return results;
    }
    let rem = results.len() % bucket;
    if rem != 0 {
        results.extend((0..bucket - rem).map(|_| None));
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_block_store_basic() {
        let s = InMemoryBlockStore::default();
        s.put(b"tag:abc:1", b"1").unwrap();
        s.put(b"tag:abc:2", b"1").unwrap();
        s.put(b"entity:x:1", b"1").unwrap();
        assert_eq!(s.get(b"tag:abc:1").unwrap().as_deref(), Some(&b"1"[..]));
        assert_eq!(s.keys_with_prefix(b"tag:abc:").unwrap().len(), 2);
        s.delete(b"tag:abc:1").unwrap();
        assert!(s.get(b"tag:abc:1").unwrap().is_none());
        assert_eq!(s.keys_with_prefix(b"tag:abc:").unwrap().len(), 1);
    }

    #[test]
    fn pad_to_bucket_rounds_up_and_preserves_items() {
        let r = pad_to_bucket(vec![Some(1), Some(2), Some(3)], 4);
        assert_eq!(r.len(), 4);
        assert_eq!(r.iter().filter(|x| x.is_some()).count(), 3);

        // exact multiple: unchanged
        assert_eq!(pad_to_bucket(vec![Some(1), Some(2)], 2).len(), 2);
        // bucket 0: no-op
        assert_eq!(pad_to_bucket(vec![Some(1)], 0).len(), 1);
        // empty stays empty
        assert_eq!(pad_to_bucket::<i32>(vec![], 4).len(), 0);
    }
}
