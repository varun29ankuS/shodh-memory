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
}

/// Arm the funnel for the next recall with this query's gold ids. Clears any prior state.
pub fn begin(gold: HashSet<MemoryId>) {
    FUNNEL.with(|c| {
        *c.borrow_mut() = Some(FunnelState {
            gold,
            stages: Vec::new(),
        });
    });
}

/// Disarm and return the recorded per-stage ranks (None if the funnel was never armed).
pub fn take() -> Option<Vec<(String, Option<usize>)>> {
    FUNNEL.with(|c| c.borrow_mut().take().map(|s| s.stages))
}

/// Record the best gold rank in `ids` (an ordered candidate list) for `stage`. No-op unless
/// the funnel is armed. `ids` must be in rank order (rank 0 = best).
pub fn record<'a>(stage: &str, ids: impl Iterator<Item = &'a MemoryId>) {
    FUNNEL.with(|c| {
        let mut borrow = c.borrow_mut();
        let Some(state) = borrow.as_mut() else {
            return;
        };
        let mut best: Option<usize> = None;
        for (i, id) in ids.enumerate() {
            if state.gold.contains(id) {
                best = Some(i);
                break;
            }
        }
        state.stages.push((stage.to_string(), best));
    });
}

/// True when the funnel is armed — lets the pipeline skip building an iterator when not needed.
pub fn is_armed() -> bool {
    FUNNEL.with(|c| c.borrow().is_some())
}
