//! Per-leg pre-fusion score dump for the fusion-ablation study.
//!
//! Captures, for a single recall, the three raw retrieval legs BEFORE they are
//! fused — BM25 and dense-vector (from the hybrid component scores) and graph
//! (spreading activation) — each as a list of `(candidate, raw_score)`. An
//! offline A/B/C analysis then replays RRF vs calibrated score fusion on the
//! SAME candidate pool, so the question "does calibrated fusion un-bury the
//! graph-only gold?" is answered on real per-leg magnitudes, not a model.
//!
//! Implemented as a thread-local (same shape as [`super::gold_funnel`]) so the
//! recall pipeline needs no signature change and production callers pay nothing:
//! the thread-local is unset unless the recall harness arms it via [`begin`], so
//! every [`record_leg`] call is a cheap no-op. The harness arms it per query
//! only when `SHODH_FUSION_DUMP` is set, then drains it via [`take`].

use super::types::MemoryId;
use std::cell::RefCell;

thread_local! {
    static DUMP: RefCell<Option<DumpState>> = const { RefCell::new(None) };
}

#[derive(Default)]
struct DumpState {
    /// (leg name, that leg's raw `(candidate, score)` pairs, unsorted).
    legs: Vec<(String, Vec<(MemoryId, f32)>)>,
}

/// Arm the dump for the next recall. Clears any prior state.
pub fn begin() {
    DUMP.with(|c| *c.borrow_mut() = Some(DumpState::default()));
}

/// Disarm and return the recorded legs (None if the dump was never armed).
pub fn take() -> Option<Vec<(String, Vec<(MemoryId, f32)>)>> {
    DUMP.with(|c| c.borrow_mut().take().map(|s| s.legs))
}

/// True when the dump is armed — lets the pipeline skip building the leg
/// iterators when the study is not running.
pub fn is_armed() -> bool {
    DUMP.with(|c| c.borrow().is_some())
}

/// Record one leg's raw `(candidate, score)` pairs. No-op unless armed.
pub fn record_leg(leg: &str, items: impl Iterator<Item = (MemoryId, f32)>) {
    DUMP.with(|c| {
        let mut borrow = c.borrow_mut();
        let Some(state) = borrow.as_mut() else {
            return;
        };
        state.legs.push((leg.to_string(), items.collect()));
    });
}
