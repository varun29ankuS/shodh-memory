//! Runs the integrity scrub against a real on-disk store, read-only.
//!
//! Ignored by default: it needs a populated store, which CI does not have. It
//! exists because the scrub's whole claim is about *live* data — the July
//! breakage was invisible precisely because every synthetic test passed.
//!
//! ```text
//! set SHODH_SCRUB_DIR=<path to shodh_memory_data>
//! cargo test --test integrity_live_scrub -- --ignored --nocapture
//! ```
//!
//! It opens each profile with `DB::open_cf_for_read_only`, which takes no LOCK
//! and never writes, so it is safe to run against a store a live server has
//! open. `SHODH_SCRUB_PROFILES` (comma-separated) narrows the profile list.

use std::path::{Path, PathBuf};

use rocksdb::{Options, DB};
use shodh_memory::integrity::{self, ClassCounts, ScrubBudget, Sweep};

fn data_dir() -> Option<PathBuf> {
    std::env::var("SHODH_SCRUB_DIR").ok().map(PathBuf::from)
}

fn profiles() -> Vec<String> {
    match std::env::var("SHODH_SCRUB_PROFILES") {
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).collect(),
        Err(_) => ["claude-code", "claude", "defence-live", "gdelt-bridge"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    }
}

/// Open a database read-only with every column family it actually declares.
///
/// Listing the CFs first rather than hardcoding them means a store whose CF set
/// has drifted still opens, instead of failing in a way that would be easy to
/// mistake for "no records".
fn open_read_only(path: &Path) -> anyhow::Result<DB> {
    let opts = Options::default();
    let cfs = DB::list_cf(&opts, path)?;
    let db = DB::open_cf_for_read_only(&opts, path, &cfs, false)?;
    Ok(db)
}

fn top_n(map: &std::collections::BTreeMap<String, u64>, n: usize) -> Vec<(String, u64)> {
    let mut v: Vec<_> = map.iter().map(|(k, c)| (k.clone(), *c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.truncate(n);
    v
}

fn print_counts(label: &str, c: &ClassCounts) {
    println!(
        "  {label:<12} keys_seen={:<7} scanned={:<7} clean={:<7} legacy={:<6} \
         implausible={:<6} undecodable={:<5} crc_mismatch={:<3} read_errors={} forgotten={}",
        c.keys_seen,
        c.scanned,
        c.clean,
        c.legacy,
        c.implausible,
        c.undecodable,
        c.checksum_mismatch,
        c.read_errors,
        c.forgotten
    );
    if !c.decode_paths.is_empty() {
        println!("    decode paths : {:?}", top_n(&c.decode_paths, 8));
    }
    if !c.checks_failed.is_empty() {
        println!("    checks fired : {:?}", top_n(&c.checks_failed, 12));
    }
    if !c.embedding_dims.is_empty() {
        println!("    embed dims   : {:?}", c.embedding_dims);
    }
}

#[test]
#[ignore = "requires a populated on-disk store; set SHODH_SCRUB_DIR"]
fn scrub_live_profiles() {
    let Some(base) = data_dir() else {
        panic!("set SHODH_SCRUB_DIR to the shodh_memory_data directory");
    };

    for profile in profiles() {
        let mem_path = base.join(&profile).join("storage");
        let graph_path = base.join(&profile).join("graph").join("graph");
        if !mem_path.exists() {
            println!("== {profile}: no storage directory, skipping");
            continue;
        }

        let mem_db = match open_read_only(&mem_path) {
            Ok(db) => db,
            Err(e) => {
                println!("== {profile}: could not open memory store read-only: {e}");
                continue;
            }
        };

        let mut sweep = Sweep::new(ScrubBudget {
            max_records: None,
            max_duration: None,
        });
        let started = std::time::Instant::now();
        let memories = integrity::scrub_memories(&mem_db, &mut sweep);
        let mem_ms = started.elapsed().as_millis();

        let mut graph_counts = ClassCounts::default();
        let mut graph_ms = 0u128;
        if graph_path.exists() {
            match open_read_only(&graph_path) {
                Ok(gdb) => {
                    if let Some(cf) = gdb.cf_handle("entities") {
                        let t = std::time::Instant::now();
                        graph_counts = integrity::scrub_graph_nodes(&gdb, cf, &mut sweep);
                        graph_ms = t.elapsed().as_millis();
                    } else {
                        println!("== {profile}: graph db has no 'entities' CF");
                    }
                }
                Err(e) => println!("== {profile}: could not open graph read-only: {e}"),
            }
        }

        println!("== {profile}  (memories {mem_ms}ms, graph {graph_ms}ms)");
        print_counts("memories", &memories);
        print_counts("graph_nodes", &graph_counts);

        // A sample of the evidence, so a defect population can be recognised
        // by its cohort rather than only by its count.
        for f in sweep.findings().iter().take(6) {
            println!(
                "    [{}] {} {:?} via {} created_at={:?} :: {}",
                f.record_class,
                &f.key[..16.min(f.key.len())],
                f.classification,
                f.decode_path,
                f.created_at.map(|t| t.to_rfc3339()),
                f.detail.chars().take(160).collect::<String>()
            );
        }
        println!(
            "    findings retained={} truncated={}",
            sweep.findings().len(),
            sweep.findings_truncated()
        );
        println!();
    }
}
