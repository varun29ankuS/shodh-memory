//! Wire-level census of the memory decode path against a real on-disk store.
//!
//! Ignored by default: it needs a populated store, which CI does not have. It
//! exists because the two defects it measures were both invisible to synthetic
//! tests — a checksum laundered into a fallback decode, and a fallback decode
//! that fabricates a plausible memory, neither of which any unit test could
//! have seen without live bytes to look at.
//!
//! ```text
//! set SHODH_CENSUS_DIR=<path containing the profile directories>
//! cargo test --test decode_path_live_census -- --ignored --nocapture
//! ```
//!
//! `SHODH_CENSUS_PROFILES` (comma-separated) narrows the profile list.
//!
//! It opens each store with `DB::open_cf_for_read_only`, which takes no LOCK
//! and never writes, so it is safe against a store a live server has open. It
//! also never goes through `MemoryStorage::get`, which lazily rewrites anything
//! it decodes through a legacy path.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rocksdb::{IteratorMode, Options, ReadOptions, DB};
use shodh_memory::serialization::{
    read_sho_envelope, ShoEnvelope, SHO_VERSION_BINCODE2, SHO_VERSION_POSTCARD,
};

fn base_dir() -> Option<PathBuf> {
    std::env::var("SHODH_CENSUS_DIR")
        .or_else(|_| std::env::var("SHODH_SCRUB_DIR"))
        .ok()
        .map(PathBuf::from)
}

fn profiles() -> Vec<String> {
    match std::env::var("SHODH_CENSUS_PROFILES") {
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).collect(),
        Err(_) => ["claude-code", "claude", "defence-live", "gdelt-bridge"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    }
}

fn open_read_only(path: &Path) -> anyhow::Result<DB> {
    let opts = Options::default();
    let cfs = DB::list_cf(&opts, path)?;
    Ok(DB::open_cf_for_read_only(&opts, path, &cfs, false)?)
}

#[derive(Default)]
struct Census {
    /// 16-byte keys — the memory keyspace.
    memory_keys: u64,
    /// No SHO magic: a pre-cutover record, which must still decode.
    absent: u64,
    /// Magic present, too short to carry a checksum.
    truncated: u64,
    /// Magic present, complete envelope, CRC32 disagrees. THE population this
    /// change moves: previously handed to the fallback chain, now refused.
    checksum_mismatch: u64,
    /// Valid envelope, by version byte.
    valid: BTreeMap<u8, u64>,
    /// Decoded through the production decoder.
    decoded: u64,
    /// Refused by the production decoder.
    refused: u64,
    /// Decoded, but into a record whose id disagrees with its own key — a
    /// fabrication, by construction: the writer derives the key from the
    /// record, every fallback shape derives the id from the value.
    id_key_mismatch: u64,
    /// Records whose envelope is damaged AND which the production decoder
    /// still decoded. Must be zero: that is the defect this change removes.
    corrupt_but_decoded: u64,
    /// Sample of the disagreeing keys, for a human to go and look at.
    samples: Vec<String>,
    /// Sample of the records the decoder refused, with the reason.
    refusals: Vec<String>,
}

fn census_profile(db: &DB) -> Census {
    let mut c = Census::default();
    let mut opts = ReadOptions::default();
    opts.fill_cache(false);

    for item in db.iterator_opt(IteratorMode::Start, opts) {
        let (key, value) = match item {
            Ok(kv) => kv,
            Err(e) => panic!("read error during census: {e}"),
        };
        if key.len() != 16 {
            continue;
        }
        c.memory_keys += 1;

        let envelope = read_sho_envelope(&value);
        match envelope {
            ShoEnvelope::Absent => c.absent += 1,
            ShoEnvelope::Truncated { .. } => c.truncated += 1,
            ShoEnvelope::ChecksumMismatch { .. } => c.checksum_mismatch += 1,
            ShoEnvelope::Valid { version, .. } => *c.valid.entry(version).or_default() += 1,
        }

        match shodh_memory::memory::storage::deserialize_memory_for_migration(&value) {
            Ok(memory) => {
                c.decoded += 1;
                if envelope.is_corrupt() {
                    c.corrupt_but_decoded += 1;
                }
                if memory.id.0.as_bytes() != &key[..] {
                    c.id_key_mismatch += 1;
                    if c.samples.len() < 5 {
                        c.samples.push(format!(
                            "key {} decoded as id {} ({} bytes)",
                            hex::encode(&key),
                            memory.id.0,
                            value.len()
                        ));
                    }
                }
            }
            Err(e) => {
                c.refused += 1;
                if c.refusals.len() < 5 {
                    c.refusals.push(format!(
                        "key {} ({} bytes, first 8: {}): {e:#}",
                        hex::encode(&key),
                        value.len(),
                        hex::encode(&value[..value.len().min(8)])
                    ));
                }
            }
        }
    }
    c
}

#[test]
#[ignore = "requires a populated on-disk store; set SHODH_CENSUS_DIR"]
fn decode_path_census_over_live_profiles() {
    let Some(base) = base_dir() else {
        panic!("set SHODH_CENSUS_DIR to the directory holding the profile stores");
    };

    let mut looked_at = 0u64;
    for profile in profiles() {
        let path = base.join(&profile).join("storage");
        if !path.exists() {
            println!("== {profile}: no storage directory, skipping");
            continue;
        }
        let db = match open_read_only(&path) {
            Ok(db) => db,
            Err(e) => {
                println!("== {profile}: cannot open read-only: {e:#}");
                continue;
            }
        };
        let c = census_profile(&db);
        looked_at += c.memory_keys;

        println!("== {profile}");
        println!(
            "   memory keys {:>7} | envelope: v2 {:>7}  v1 {:>5}  other {:>4}  absent {:>5} \
             truncated {:>4}  crc_mismatch {:>4}",
            c.memory_keys,
            c.valid.get(&SHO_VERSION_POSTCARD).copied().unwrap_or(0),
            c.valid.get(&SHO_VERSION_BINCODE2).copied().unwrap_or(0),
            c.valid
                .iter()
                .filter(|(v, _)| **v != SHO_VERSION_POSTCARD && **v != SHO_VERSION_BINCODE2)
                .map(|(_, n)| *n)
                .sum::<u64>(),
            c.absent,
            c.truncated,
            c.checksum_mismatch,
        );
        println!(
            "   decoder     {:>7} decoded | {:>5} refused | {:>4} decoded but id != key",
            c.decoded, c.refused, c.id_key_mismatch
        );
        for s in &c.samples {
            println!("     - {s}");
        }
        for s in &c.refusals {
            println!("     ! {s}");
        }

        // Every record is classified exactly once: no key falls between the
        // envelope states, which is what made the old two-state reading able to
        // hide a whole class.
        let classified = c.absent
            + c.truncated
            + c.checksum_mismatch
            + c.valid.values().sum::<u64>();
        assert_eq!(
            classified, c.memory_keys,
            "{profile}: every memory key must land in exactly one envelope state"
        );
        assert_eq!(
            c.decoded + c.refused,
            c.memory_keys,
            "{profile}: every memory key must either decode or be refused"
        );

        // The fix, verified on real bytes rather than on a fixture: no record
        // whose checksum fails is decoded by the production decoder. Vacuous if
        // the store holds none — which is why the count is printed above.
        assert_eq!(
            c.corrupt_but_decoded, 0,
            "{profile}: {} records with a damaged envelope were still decoded — \
             a corrupt record reached a fallback decoder",
            c.corrupt_but_decoded
        );
    }

    assert!(
        looked_at > 0,
        "no memory records were examined — check SHODH_CENSUS_DIR"
    );
}
