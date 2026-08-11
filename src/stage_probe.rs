//! Per-stage recall latency probe (thread-local, opt-in, off by default).
//!
//! # Why this exists
//!
//! We know the end-to-end recall latency of the locomo-gate suite (p50 ≈ 87ms)
//! but not where the time goes. Without that breakdown three decisions are
//! blocked: whether hardware acceleration buys anything, whether the weekly
//! LongMemEval trend can be made to fit a CI runner's wall-clock budget, and
//! where performance work should go at all.
//!
//! # Why a thread-local and not a return value
//!
//! [`StageTiming`](crate::memory::types::StageTiming) already exists and is
//! populated at seven checkpoints inside `semantic_retrieve_inner`, but it only
//! rides out on the `recall_with_diagnostics` return value. That channel cannot
//! answer the question on its own, for two reasons:
//!
//! 1. **It measures a different function.** `MemorySystem::recall()` — the call
//!    the harness wall-clocks — wraps `recall_inner` with the companion re-rank,
//!    which can issue a SECOND, five-times-deeper `recall_inner`. A per-call
//!    return value attributes one invocation; only an accumulator sums them, and
//!    the sum is what has to reconcile against the wall clock.
//! 2. **It perturbs.** `recall_with_diagnostics` also builds the per-memory
//!    `ScoreAttribution` map, touched at ~15 sites across the scoring cascade.
//!    That is measurable work in the very stage we most want to measure.
//!
//! So the probe accumulates. The stage boundaries themselves are unchanged — the
//! same `Instant`s at the same seven checkpoints feed it, plus the sub-stage
//! splits (tokenize vs ONNX forward, BM25 vs the rest of fusion, RocksDB read vs
//! postcard decode) that live below the recall function and cannot be reached
//! without threading a parameter through the `Embedder` trait and the storage
//! layer.
//!
//! This is the same thread-local arm/drain pattern already used by
//! [`gold_funnel`](crate::memory::gold_funnel) and
//! [`fusion_features`](crate::memory::fusion_features): the harness arms it, the
//! production path populates it, the harness drains it. Unarmed — which is every
//! production call, and every harness call without `SHODH_STAGE_TIMING=1` — every
//! entry point below is a single thread-local `is_none` check and returns
//! immediately. No `Instant::now()` is taken, nothing is allocated, and no
//! branch downstream of the probe changes.
//!
//! # Threading
//!
//! Recall is synchronous on the caller's thread: the embedder is invoked
//! directly, the Layer-5 fetch loop is a sequential `for`, and the storage reads
//! happen inline. Work that fans out to a rayon pool is therefore NOT captured
//! here — it is captured by the enclosing stage timer, which is the correct
//! place for it. See the `stage_probe_ignores_other_threads` test.

use std::cell::RefCell;
use std::time::{Duration, Instant};

thread_local! {
    static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) };
}

/// Accumulated sub-stage durations for one armed region.
///
/// Every field is a SUM over the armed region, not a single observation: a query
/// that embeds twice (the polarity-sensitive path generates a second, negated-form
/// embedding) contributes two forward passes to `onnx_forward` and reports
/// `forwards == 2`. Without that count the embed distribution reads as
/// inexplicably bimodal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Probe {
    // --- top-level stages (mirrors `StageTiming`, summed across nested recalls) ---
    /// Layers 0.4-0.7: temporal analysis, attribute detection, fact lookups.
    pub query_analysis: Duration,
    /// Query embedding: tokenize + forward + pooling, including any cache lookup.
    pub embedding: Duration,
    /// Layers 1-2: episode coherence filter + graph spreading activation / PPR.
    pub graph_expansion: Duration,
    /// Layer 3: Vamana/SPANN ANN search.
    pub vector_search: Duration,
    /// Layer 4: BM25 + RRF fusion + the 4.45-4.9 boost stack.
    pub fusion: Duration,
    /// Layer 5: memory fetch + unified scoring + quality gate.
    pub scoring: Duration,
    /// Post-Layer-5: linguistic, competition, coactivation, hierarchy.
    pub post: Duration,
    /// Number of `semantic_retrieve_inner` invocations inside the armed region.
    /// `> 1` means the companion re-rank fired and issued a nested deep recall.
    pub inner_recalls: u32,

    // --- sub-stage splits ---
    /// Tokenizer only, excluding the forward pass.
    pub tokenize: Duration,
    /// `session.run` only — the ONNX forward pass.
    pub onnx_forward: Duration,
    /// Number of ONNX forward passes (0 on a query-cache hit, 2 on the polar path).
    pub forwards: u32,
    /// BM25/tantivy search, carved out of `fusion`.
    pub bm25: Duration,
    /// Layer-5 cold-path memory fetch, carved out of `scoring`.
    pub fetch: Duration,
    /// RocksDB `get` only, excluding deserialization. RECALL-WIDE, not a subset
    /// of `fetch`: measurement showed ~675 `MemoryStorage::get` calls per query
    /// against ~0.1ms of Layer-5 cold fetch, so almost all of them are issued by
    /// the graph traversal, not by the Layer-5 cascade. Attribute this to the
    /// whole of `recall()`; the top-level stage rows say which stage pays.
    pub storage_read: Duration,
    /// Deserialization of fetched memories. Recall-wide, same caveat as
    /// `storage_read`.
    pub storage_decode: Duration,
    /// Number of `MemoryStorage::get` calls in the armed region, from any stage.
    pub storage_reads: u32,
}

/// Arm the probe on the current thread, discarding any previous accumulation.
pub fn arm() {
    PROBE.with(|p| *p.borrow_mut() = Some(Probe::default()));
}

/// Drain and disarm. Returns `None` if the probe was never armed.
pub fn take() -> Option<Probe> {
    PROBE.with(|p| p.borrow_mut().take())
}

/// Whether the probe is armed on this thread.
///
/// Call sites in hot loops use this to skip taking an `Instant` at all.
#[inline]
pub fn is_armed() -> bool {
    PROBE.with(|p| p.borrow().is_some())
}

/// `Instant::now()`, but only when armed.
///
/// The idiom at every call site is:
/// ```ignore
/// let t = stage_probe::start();
/// let out = expensive();
/// stage_probe::record(t, |p, d| p.thing += d);
/// ```
/// Unarmed, `start()` returns `None` and `record` is a no-op, so the clock is
/// never read and the measured code is untouched.
#[inline]
pub fn start() -> Option<Instant> {
    if is_armed() {
        Some(Instant::now())
    } else {
        None
    }
}

/// Add the elapsed time since `t` to a field, if the probe is armed.
#[inline]
pub fn record(t: Option<Instant>, f: impl FnOnce(&mut Probe, Duration)) {
    if let Some(t) = t {
        let d = t.elapsed();
        with_probe(|p| f(p, d));
    }
}

/// Mutate the probe in place, if armed. Used for counters that have no duration.
#[inline]
pub fn with_probe(f: impl FnOnce(&mut Probe)) {
    PROBE.with(|p| {
        if let Some(probe) = p.borrow_mut().as_mut() {
            f(probe);
        }
    });
}

impl Probe {
    /// Sum of the seven top-level stages.
    ///
    /// This is what must reconcile against the harness's wall clock. A gap means
    /// time is being spent OUTSIDE the instrumented stages — in the `recall()`
    /// wrapper, in lock acquisition, or in a stage boundary that was drawn in the
    /// wrong place. Report the gap; do not hide it.
    pub fn stage_sum(&self) -> Duration {
        self.query_analysis
            + self.embedding
            + self.graph_expansion
            + self.vector_search
            + self.fusion
            + self.scoring
            + self.post
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unarmed is the production configuration: no clock read, no accumulation,
    /// and `take()` yields nothing for a caller to misreport as zero.
    #[test]
    fn unarmed_probe_records_nothing_and_takes_none() {
        // Ensure a clean slate — thread-locals persist across tests on a thread.
        let _ = take();
        assert!(!is_armed());
        assert!(start().is_none());
        record(start(), |p, d| p.embedding += d);
        with_probe(|p| p.forwards += 1);
        assert_eq!(take(), None);
    }

    #[test]
    fn armed_probe_accumulates_across_calls() {
        arm();
        assert!(is_armed());
        for _ in 0..3 {
            let t = start();
            assert!(t.is_some());
            std::thread::sleep(Duration::from_millis(2));
            record(t, |p, d| p.onnx_forward += d);
            with_probe(|p| p.forwards += 1);
        }
        let p = take().expect("armed probe must drain");
        assert_eq!(p.forwards, 3);
        // Three ~2ms sleeps must accumulate, not overwrite. Lower bound only —
        // sleep granularity makes any upper bound flaky on a shared machine.
        assert!(
            p.onnx_forward >= Duration::from_millis(6),
            "expected accumulation of 3x2ms, got {:?}",
            p.onnx_forward
        );
        assert!(!is_armed(), "take() must disarm");
    }

    #[test]
    fn stage_sum_adds_the_seven_top_level_stages_only() {
        let p = Probe {
            query_analysis: Duration::from_millis(1),
            embedding: Duration::from_millis(2),
            graph_expansion: Duration::from_millis(4),
            vector_search: Duration::from_millis(8),
            fusion: Duration::from_millis(16),
            scoring: Duration::from_millis(32),
            post: Duration::from_millis(64),
            // Sub-stages are carved OUT of the stages above; double-counting them
            // in the sum would break reconciliation against the wall clock.
            tokenize: Duration::from_millis(1000),
            onnx_forward: Duration::from_millis(1000),
            bm25: Duration::from_millis(1000),
            fetch: Duration::from_millis(1000),
            ..Default::default()
        };
        assert_eq!(p.stage_sum(), Duration::from_millis(127));
    }

    /// The probe is per-thread by construction. Work that fans out to another
    /// thread is invisible to it and must stay accounted for by the enclosing
    /// stage timer — this pins that contract so a future rayon-ised stage does
    /// not silently start under-reporting.
    #[test]
    fn stage_probe_ignores_other_threads() {
        arm();
        std::thread::spawn(|| {
            assert!(!is_armed(), "a fresh thread must not inherit the probe");
            with_probe(|p| p.forwards += 99);
        })
        .join()
        .expect("probe thread must not panic");
        let p = take().expect("armed probe must drain");
        assert_eq!(p.forwards, 0, "another thread must not mutate this probe");
    }
}
