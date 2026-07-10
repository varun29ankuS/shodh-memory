//! Parity regression guard for the in-engine dependency parser (Task 1.1).
//!
//! The vendored `spacy-rusty` was validated at the binary level to reproduce
//! Python spaCy's `en_core_web_sm` heads exactly (`py_heads.tsv` ==
//! `rust_heads.tsv`, 669/669). This test re-proves that parity *through the
//! in-engine integration surface* — the `SHODH_SPACY_MODEL_PATH` bundle load and
//! the `dep_parser` wrapper — so a regression in loading, token mapping, or head
//! selection is caught here rather than downstream in entity resolution.
//!
//! The 15 MB model bundle is not committed. Set `SHODH_SPACY_MODEL_PATH` to a
//! directory containing `model.json` + `model.safetensors` to run it; without
//! the model the test skips (parity cannot be checked, but nothing regresses).

use shodh_memory::dep_parser;

const GOLDEN: &str = include_str!("fixtures/en_core_web_sm_heads_golden.tsv");

#[test]
fn in_engine_heads_match_python_spacy_golden() {
    if !dep_parser::is_available() {
        eprintln!(
            "SKIP in_engine_heads_match_python_spacy_golden: set SHODH_SPACY_MODEL_PATH \
             to an en_core_web_sm bundle (model.json + model.safetensors) to run parity."
        );
        return;
    }

    let mut checked = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for (lineno, line) in GOLDEN.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(
            cols.len() >= 3,
            "malformed golden row {} (expected mention\\troot\\tpos): {:?}",
            lineno + 1,
            line
        );
        let (mention, want_root, want_pos) = (cols[0], cols[1], cols[2]);

        let head = dep_parser::head_token(mention)
            .expect("parser available but returned no head");
        checked += 1;

        if head.text != want_root || head.pos != want_pos {
            mismatches.push(format!(
                "  {:?}: got ({:?}, {}) want ({:?}, {})",
                mention, head.text, head.pos, want_root, want_pos
            ));
        }
    }

    assert!(checked > 600, "golden set unexpectedly small: {checked} rows");
    assert!(
        mismatches.is_empty(),
        "{}/{} mentions diverged from Python spaCy golden heads:\n{}",
        mismatches.len(),
        checked,
        mismatches.join("\n")
    );
    eprintln!("dep_parser parity: {checked}/{checked} heads match Python spaCy golden");
}

/// The two canonical span-head cases from the plan's success criterion, asserted
/// explicitly so they are legible in the test output.
#[test]
fn canonical_span_heads() {
    if !dep_parser::is_available() {
        eprintln!("SKIP canonical_span_heads: SHODH_SPACY_MODEL_PATH unset");
        return;
    }
    // "Port of Baltimore" → head "Port" (PROPN), not "Baltimore".
    let port = dep_parser::head_token("Port of Baltimore").unwrap();
    assert_eq!(port.text, "Port", "Port of Baltimore head");

    // "ship crashed" → verb-headed fragment: the resolver uses pos == VERB to
    // reject/strip it. Assert the head is the verb so that signal is intact.
    let crashed = dep_parser::head_token("ship crashed").unwrap();
    assert_eq!(crashed.pos, "VERB", "ship crashed should be verb-headed");
}
