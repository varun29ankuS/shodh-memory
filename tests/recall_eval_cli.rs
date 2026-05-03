//! CLI surface tests for the `recall-eval` binary.
//!
//! These tests guard the command-line contract — flag names, defaults,
//! required arguments, and exit codes — without spinning up the full
//! `MemorySystem` (which loads ONNX models and is far too heavy for an
//! integration test).
//!
//! Heavyweight end-to-end coverage of the suite itself lives in the
//! `runner_executes_smoke_suite_and_produces_well_formed_report` unit test
//! inside `src/recall_harness/runner.rs`.
//!
//! Cargo populates `CARGO_BIN_EXE_recall-eval` automatically for any
//! integration test in this crate, so no `assert_cmd` dependency is needed.

use std::path::PathBuf;
use std::process::Command;

/// Locate the compiled `recall-eval` binary. Cargo guarantees this env var
/// for integration tests when the binary is declared in `Cargo.toml`.
fn recall_eval_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_recall-eval"))
}

/// `--help` must exit 0 and advertise every public flag. This is the
/// machine-checked contract for issue #266: CI scripts and humans both
/// rely on these flag names.
#[test]
fn help_lists_every_public_flag() {
    let out = Command::new(recall_eval_bin())
        .arg("--help")
        .output()
        .expect("spawning recall-eval --help");

    assert!(
        out.status.success(),
        "--help must exit 0, got {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in ["--suite", "--output", "--baseline", "--tolerance", "--storage"] {
        assert!(
            stdout.contains(flag),
            "--help output is missing {flag}\nfull stdout:\n{stdout}"
        );
    }
}

/// Missing the required `--output` flag must fail with a non-zero exit
/// (clap parse error). This guards against an accidental change of
/// `output` to `Option<PathBuf>` which would silently no-op the JSON
/// report and break the CI gate.
#[test]
fn missing_output_flag_is_a_parse_error() {
    let out = Command::new(recall_eval_bin())
        .output()
        .expect("spawning recall-eval with no args");

    assert!(
        !out.status.success(),
        "running with no --output must fail; got success exit"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--output"),
        "parse error must mention --output, got:\n{stderr}"
    );
}

/// An unknown `--suite` value must fail at parse time (before we spin up
/// any MemorySystem). Guards the `Suite` enum from accidentally being
/// loosened to a free-form string.
#[test]
fn unknown_suite_is_a_parse_error() {
    let out = Command::new(recall_eval_bin())
        .args(["--suite", "definitely-not-a-suite", "--output", "ignored.json"])
        .output()
        .expect("spawning recall-eval with bogus suite");

    assert!(
        !out.status.success(),
        "unknown --suite value must fail at parse time"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("suite") || stderr.contains("invalid value"),
        "stderr should explain the bad --suite value, got:\n{stderr}"
    );
}

/// Bogus `--tolerance` value (non-numeric) must fail at parse time.
/// Guards the `f64` parser on the tolerance flag — silently coercing it
/// to a default would let a typo accidentally bypass the CI gate.
#[test]
fn non_numeric_tolerance_is_a_parse_error() {
    let out = Command::new(recall_eval_bin())
        .args([
            "--output",
            "ignored.json",
            "--tolerance",
            "not-a-number",
        ])
        .output()
        .expect("spawning recall-eval with bogus tolerance");

    assert!(
        !out.status.success(),
        "non-numeric --tolerance must fail at parse time"
    );
}
