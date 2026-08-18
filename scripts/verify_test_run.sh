#!/usr/bin/env bash
#
# Run `cargo test --lib` and prove the run is evidence.
#
# A pass count is not evidence. Two failures this project has actually hit:
#
#   * Six git worktrees built the same crate into the same
#     `deps/shodh_memory-<hash>.exe`, so a run could execute a binary someone
#     else's branch had just linked. One run reported "0 passed; 1067 filtered
#     out" because the resident binary held a different branch's test names.
#     The summary line looked green-adjacent and the names were never checked.
#   * A mutation harness reported 19 survivors because its reporter flag did
#     not exist in the installed tool, so nothing ran at all.
#
# Both are the same bug: the run reported a number without confirming the
# number described the thing under test. So this script checks four things.
#
#   1. Which binary ran, by path and content hash. Two runs that disagree here
#      are not two runs of the same code.
#   2. Every sentinel test below is present in the executed output as
#      `test <name> ... ok`. Sentinels are chosen to be names that exist ONLY on
#      this branch — a stale or foreign binary cannot produce them.
#   3. passed + ignored + failed equals the number of tests the binary lists, so
#      nothing was silently filtered out.
#   4. The pass count is at or above the floor.
#
# Usage:  scripts/verify_test_run.sh [output-file]
# Exit:   0 only if all four hold and the suite itself passed.

set -uo pipefail

FLOOR="${SHODH_TEST_PASS_FLOOR:-1100}"
OUT="${1:-target/test-run-$(date +%Y%m%d-%H%M%S).log}"

# Names that exist only on this branch's test isolation work. If the binary that
# ran does not contain these, it is not the binary this branch built.
SENTINELS=(
  "test_support::tests::env_mutation_sites_are_accounted_for"
  "test_support::tests::the_env_mutation_scanner_counts_calls_and_ignores_comments"
  "test_support::tests::scoped_env_is_reentrant"
  "memory::readonly_recall_tests::default_recall_still_reinforces_what_it_read"
  "memory::graph_retrieval::tests::spreading_activation_readonly_gate_skips_hebbian_strengthening"
  "recall_harness::runner::tests::runner_executes_smoke_suite_and_produces_well_formed_report"
)

mkdir -p "$(dirname "$OUT")"

echo "== listing tests in the lib binary =="
LIST="${OUT}.list"
if ! cargo test --lib -- --list >"$LIST" 2>&1; then
  echo "FAIL: could not list tests; see $LIST" >&2
  exit 1
fi
LISTED=$(grep -c ': test$' "$LIST")
echo "listed: $LISTED"

if [ "$LISTED" -eq 0 ]; then
  echo "FAIL: the binary lists zero tests — nothing would have run." >&2
  exit 1
fi

echo "== running suite =="
START=$(date +%s)
cargo test --lib 2>&1 | tee "$OUT"
STATUS=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - START ))

# Which binary actually ran, and what is in it.
BIN=$(grep -o 'deps[/\\]shodh_memory-[0-9a-f]*\(\.exe\)\?' "$OUT" | head -1)
BINPATH=$(find target -type f -name "$(basename "$BIN")" 2>/dev/null | head -1)
if [ -n "$BINPATH" ]; then
  HASH=$( (sha256sum "$BINPATH" 2>/dev/null || shasum -a 256 "$BINPATH") | cut -d' ' -f1)
else
  HASH="unknown"
fi

PASSED=$(grep -o 'test result: .*[0-9]\+ passed' "$OUT" | grep -o '[0-9]\+ passed' | grep -o '^[0-9]\+' | head -1)
FAILED=$(grep -o '[0-9]\+ failed' "$OUT" | grep -o '^[0-9]\+' | head -1)
IGNORED=$(grep -o '[0-9]\+ ignored' "$OUT" | grep -o '^[0-9]\+' | head -1)
PASSED=${PASSED:-0}; FAILED=${FAILED:-0}; IGNORED=${IGNORED:-0}

echo
echo "================ RUN EVIDENCE ================"
echo "binary:   ${BINPATH:-$BIN}"
echo "sha256:   $HASH"
echo "listed:   $LISTED"
echo "passed:   $PASSED"
echo "failed:   $FAILED"
echo "ignored:  $IGNORED"
echo "elapsed:  ${ELAPSED}s"
echo "log:      $OUT"

RC=0

# 2. sentinel names must appear as executed-and-ok, not merely be listed.
for name in "${SENTINELS[@]}"; do
  if grep -qF "test ${name} ... ok" "$OUT"; then
    echo "sentinel OK   $name"
  else
    echo "sentinel MISS $name" >&2
    RC=1
  fi
done
if [ "$RC" -ne 0 ]; then
  echo "FAIL: a sentinel test did not execute. The binary that ran is not this branch's." >&2
fi

# 3. accounted-for: nothing silently filtered.
TOTAL=$(( PASSED + FAILED + IGNORED ))
if [ "$TOTAL" -ne "$LISTED" ]; then
  echo "FAIL: $TOTAL tests accounted for but $LISTED are in the binary — $(( LISTED - TOTAL )) never reported." >&2
  RC=1
fi

# 4. floor.
if [ "$PASSED" -lt "$FLOOR" ]; then
  echo "FAIL: $PASSED passed, floor is $FLOOR." >&2
  RC=1
fi

if [ "$STATUS" -ne 0 ] || [ "$FAILED" -ne 0 ]; then
  echo "FAIL: suite exited $STATUS with $FAILED failures." >&2
  RC=1
fi

if [ "$RC" -eq 0 ]; then
  echo "VERIFIED: $PASSED passed / $FAILED failed / $IGNORED ignored in ${ELAPSED}s, all sentinels executed."
fi
echo "=============================================="
exit "$RC"
