//! ONNX Runtime memory attribution probe — the #90 "confirm it's ONNX" experiment.
//!
//! Samples process RSS at each stage of embedder/NER lifecycle to attribute
//! the non-RocksDB share of the memory plateau (field data: ~650MB total,
//! ~268MB block cache, ~0.2MB memtables/readers → ~270MB unattributed).
//!
//! Arena behavior is ORT-internal and platform-independent, so a Windows
//! measurement transfers to the Linux field reports as a solid estimate.
//!
//! Deliberately `#[ignore]`d — this is a diagnostic, not a regression test:
//!   cargo test --test onnx_memory_probe -- --ignored --nocapture

use shodh_memory::embeddings::minilm::{EmbeddingConfig, MiniLMEmbedder};
use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::embeddings::Embedder;

/// Current process RSS in bytes (Windows: WorkingSet64 via PowerShell;
/// Linux: /proc/self/status VmRSS). Diagnostic-grade, not hot-path code.
fn rss_bytes() -> u64 {
    #[cfg(target_os = "windows")]
    {
        let pid = std::process::id();
        let out = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("(Get-Process -Id {pid}).WorkingSet64"),
            ])
            .output()
            .expect("powershell RSS sample failed");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .expect("RSS parse failed")
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap();
                return kb * 1024;
            }
        }
        panic!("VmRSS not found");
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        0
    }
}

fn mb(b: u64) -> f64 {
    b as f64 / 1_000_000.0
}

#[test]
#[ignore = "diagnostic probe, run explicitly with --ignored --nocapture"]
fn onnx_rss_attribution() {
    let mut last = rss_bytes();
    let mut report: Vec<(String, f64, f64)> = Vec::new();
    let mut stage = |name: &str, last: &mut u64| {
        let now = rss_bytes();
        let delta = now as i64 - *last as i64;
        report.push((name.to_string(), mb(now), delta as f64 / 1_000_000.0));
        *last = now;
    };

    stage("baseline (test harness only)", &mut last);

    // ── Embedder: init + first inference (session build + arena warm-up) ────
    let embedder = MiniLMEmbedder::new(EmbeddingConfig::default()).expect("embedder init");
    let _ = embedder
        .encode("The quick brown fox jumps over the lazy dog near the riverbank at dawn.")
        .expect("first encode");
    stage("embedder init + 1st embed", &mut last);

    // ── Embedder: 100 varied-length embeds (arena growth under load) ────────
    for i in 0..100 {
        let text = format!(
            "Patrol report {i}: convoy sighted moving along the northern access road, \
             transponder codes irregular, weather deteriorating; {} follow-up checks \
             requested by operations before the next rotation window closes.",
            i % 7
        );
        let _ = embedder.encode(&text).expect("batch embed");
    }
    stage("embedder +100 embeds", &mut last);

    // ── Embedder: second 100 (plateau check — arena reuse should be ~0Δ) ────
    for i in 0..100 {
        let text =
            format!("Second pass probe {i} with a moderately long sentence body to reuse arenas.");
        let _ = embedder.encode(&text).expect("reuse embed");
    }
    stage("embedder +100 more (reuse)", &mut last);

    // ── NER: init + first extraction ────────────────────────────────────────
    let ner = NeuralNer::new(NerConfig::default()).expect("NER init");
    let _ = ner
        .extract(
            "Sergeant Rao met Lieutenant Iyer at Checkpoint Delta near the Fuel Depot in Mumbai.",
        )
        .expect("first NER");
    stage("NER init + 1st extract", &mut last);

    // ── NER: 100 extractions ────────────────────────────────────────────────
    for i in 0..100 {
        let text = format!(
            "Report {i}: Anita Sharma from Larsen and Toubro visited the Chennai facility \
             with delegates from the Defence Research Organisation on Tuesday."
        );
        let _ = ner.extract(&text).expect("batch NER");
    }
    stage("NER +100 extracts", &mut last);

    println!("\n=== ONNX RSS attribution (decimal MB) ===");
    println!("{:<34} {:>10} {:>10}", "stage", "RSS", "Δ");
    for (name, rss, delta) in &report {
        println!("{name:<34} {rss:>10.1} {delta:>+10.1}");
    }
    let total: f64 = report.iter().skip(1).map(|(_, _, d)| d).sum();
    println!(
        "{:<34} {:>10} {:>+10.1}",
        "TOTAL ONNX-attributable", "", total
    );
}
