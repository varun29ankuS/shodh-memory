//! Per-stage gold-rank funnel diagnostic.
//!
//! Tracks, for a known set of gold memory ids, the best (lowest) rank of any gold id in the
//! candidate list at each pipeline stage boundary. This LOCATES where reachable gold is
//! dropped or demoted — turning "fusion buries it" from a hypothesis into a measured,
//! per-stage drop-off funnel.
//!
//! Implemented as a thread-local so the recall pipeline needs no signature change and
//! production callers pay nothing (the thread-local is unset → every record call is a cheap
//! no-op). The recall harness arms it per query via [`begin`], then drains it via [`take`].

use super::types::MemoryId;
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    static FUNNEL: RefCell<Option<FunnelState>> = const { RefCell::new(None) };
}

struct FunnelState {
    gold: HashSet<MemoryId>,
    /// (stage name, best gold rank at that stage; None if no gold present).
    stages: Vec<(String, Option<usize>)>,
    /// When armed by [`begin_capture`], every recorded stage's full ordered
    /// candidate list, in pipeline order, with the stage's score for each
    /// candidate where the stage has one. This is what lets a repeat
    /// divergence be placed on the leg it first entered rather than on the
    /// fused output it surfaced in, and be told apart as a reordering of the
    /// same scores or a change in the scores themselves.
    capture: Option<Vec<(String, Vec<(MemoryId, Option<f32>)>)>>,
}

/// Arm the funnel for the next recall with this query's gold ids. Clears any prior state.
pub fn begin(gold: HashSet<MemoryId>) {
    FUNNEL.with(|c| {
        *c.borrow_mut() = Some(FunnelState {
            gold,
            stages: Vec::new(),
            capture: None,
        });
    });
}

/// Arm the funnel to keep every stage's full ordered candidate list for the
/// next recall on this thread. No gold is tracked; the stage ranks read as
/// absent. Clears any prior state.
pub fn begin_capture() {
    FUNNEL.with(|c| {
        *c.borrow_mut() = Some(FunnelState {
            gold: HashSet::new(),
            stages: Vec::new(),
            capture: Some(Vec::new()),
        });
    });
}

/// Disarm and return the captured per-stage candidate lists, in the order the
/// pipeline recorded them (None if no capture was armed).
pub fn take_capture() -> Option<Vec<(String, Vec<(MemoryId, Option<f32>)>)>> {
    FUNNEL.with(|c| c.borrow_mut().take().and_then(|s| s.capture))
}

/// Disarm and return the recorded per-stage ranks (None if the funnel was never armed).
pub fn take() -> Option<Vec<(String, Option<usize>)>> {
    FUNNEL.with(|c| c.borrow_mut().take().map(|s| s.stages))
}

/// Record the best gold rank in `ids` (an ordered candidate list) for `stage`. No-op unless
/// the funnel is armed. `ids` must be in rank order (rank 0 = best).
pub fn record<'a>(stage: &str, ids: impl Iterator<Item = &'a MemoryId>) {
    record_inner(stage, ids.map(|id| (id, None)));
}

/// [`record`] for a stage that scores its candidates: keeps the score beside
/// each id when a capture is armed, so two repeats can be compared on values
/// and not only on order. `ids` must be in rank order (rank 0 = best).
pub fn record_scored<'a>(stage: &str, ids: impl Iterator<Item = (&'a MemoryId, f32)>) {
    record_inner(stage, ids.map(|(id, score)| (id, Some(score))));
}

fn record_inner<'a>(stage: &str, ids: impl Iterator<Item = (&'a MemoryId, Option<f32>)>) {
    FUNNEL.with(|c| {
        let mut borrow = c.borrow_mut();
        let Some(state) = borrow.as_mut() else {
            return;
        };
        let FunnelState {
            gold,
            stages,
            capture,
        } = state;
        let best: Option<usize> = if let Some(capture) = capture.as_mut() {
            let list: Vec<(MemoryId, Option<f32>)> =
                ids.map(|(id, score)| (id.clone(), score)).collect();
            let best = list.iter().position(|(id, _)| gold.contains(id));
            capture.push((stage.to_string(), list));
            best
        } else {
            let mut best = None;
            for (i, (id, _)) in ids.enumerate() {
                if gold.contains(id) {
                    best = Some(i);
                    break;
                }
            }
            best
        };
        stages.push((stage.to_string(), best));
    });
}

/// True when the funnel is armed — lets the pipeline skip building an iterator when not needed.
pub fn is_armed() -> bool {
    FUNNEL.with(|c| c.borrow().is_some())
}
