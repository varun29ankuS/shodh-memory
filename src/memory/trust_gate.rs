//! The fitted per-query trust gate: the logistic model that decides, from the
//! shape of a query's candidate legs, how far to trust the vector leg over
//! BM25 in the flat adaptive fusion (`SHODH_FLAT_ADAPTIVE`, default on, with
//! `SHODH_ADAPT_FEATURE=fitted`).
//!
//! The coefficients are fitted offline by `scripts/fit_trust_gate.py` on the
//! feature export the recall harness writes (`SHODH_FUSION_FEATURE_EXPORT`),
//! standardised per feature on that export's mean and standard deviation.
//!
//! # Why the features are shaped the way they are
//!
//! The first fit (run 27267533447) used raw leg sizes: how many candidates
//! the vector, BM25, hybrid and graph legs returned. Those are the legs' caps,
//! not properties of the query, and when the caps moved (graph 10 → 30,
//! vector ~31 → ~90) every query sat 8 and 20 standard deviations outside the
//! fit on two features at once. The gate kept working only because the two
//! extrapolated logits (+6.2 and −6.9) happened to cancel; clamping either
//! alone collapsed `t` on every query.
//!
//! The features here are chosen so a cap change cannot move them:
//!
//! - peakedness enters as a log, so a 1→30 ratio and a 1→3 ratio are a
//!   difference of 2.3, not 27;
//! - the vector leg's confidence is the number of candidates above an
//!   absolute cosine ([`VEC_STRONG_COSINE`]), bounded by the cap but not set
//!   by it;
//! - each leg's concentration is the share of its top-10 scores within its
//!   top-30 ([`mass_ratio`]), which reads the same for a 90-wide and a
//!   300-wide leg;
//! - the graph leg contributes its peak activation only; its width is a cap.
//!
//! [`evaluate`] clamps each standardised feature to ±[`FIT_RANGE_SD`] before
//! it enters the logit, so a feature that drifts anyway contributes at most
//! its weight times the range, and it reports how many features were clamped
//! so the harness can show the drift per suite (`gate` block on the report).
//!
//! The per-query trace is a thread-local armed by the harness, like
//! [`super::gold_funnel`]: production callers pay one `RefCell` borrow.

use std::cell::RefCell;

/// Number of features the gate consumes; the order is [`FEATURE_NAMES`].
pub const FEATURE_COUNT: usize = 10;

/// Feature names in the order `evaluate` expects them. The fitter prints the
/// coefficient table in this order; keep them in step.
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "log_bm_peak",
    "log_vec_peak",
    "agreement_top10",
    "max_bm",
    "max_vec",
    "n_vec_strong",
    "vec_mass_10_of_30",
    "bm_mass_10_of_30",
    "graph_max_activation",
    "query_len",
];

/// A vector candidate at or above this cosine counts as strong evidence for
/// `n_vec_strong`. The first fit's mean `max_vec` was 0.597; a query whose
/// best candidate only just reaches the line has one strong candidate.
pub const VEC_STRONG_COSINE: f32 = 0.6;

/// Numerator and denominator depths of the concentration features.
pub const MASS_TOP: usize = 10;
pub const MASS_WITHIN: usize = 30;

/// (mean, standard deviation, weight) per standardised feature. Order must
/// match [`FEATURE_NAMES`]. Written by `scripts/fit_trust_gate.py`; the run
/// it came from is recorded beside it.
///
/// Fit: `scripts/fit_trust_gate.py` on the locomo feature export of 2026-10-24
/// (recall-eval at this branch, live models, SHODH_CE_RERANK=1). Training set:
/// the nine conversations other than conv-42, 1062 labelled queries; conv-42,
/// the conversation the locomo-gate suite is drawn from, held out (155 labelled
/// queries). L2 0.01. Holdout AUC 0.739, accuracy 0.652; train AUC 0.644.
pub const FIT: [(f32, f32, f32); FEATURE_COUNT] = [
    (0.892274, 0.420647, -0.109074),  // log_bm_peak
    (0.343218, 0.107176, 0.286599),   // log_vec_peak
    (0.219492, 0.17098, 0.0530612),   // agreement_top10
    (149.703, 61.5625, -0.201558),    // max_bm
    (0.628769, 0.0719277, 0.136404),  // max_vec
    (2.9049, 4.64375, 0.0716791),     // n_vec_strong
    (0.365606, 0.0110561, 0.143714),  // vec_mass_10_of_30
    (0.426865, 0.0617586, 0.0539207), // bm_mass_10_of_30
    (0.260756, 0.0820539, 0.0850745), // graph_max_activation
    (55.4576, 17.0671, 0.0148567),    // query_len
];

/// Intercept of the same fit.
pub const FIT_BIAS: f32 = 0.0533702;

/// How many standard deviations from its fit mean a feature may sit before
/// it is clamped and counted as outside the fit. Four covers every value a
/// fit of a thousand queries can have seen; beyond it the logit would be
/// extrapolation.
pub const FIT_RANGE_SD: f32 = 4.0;

/// Logit bound before the sigmoid; keeps `exp` finite on any input.
const LOGIT_CLAMP: f32 = 30.0;

/// Share of the top `top` scores within the top `within` of a leg sorted by
/// score descending. 1.0 for a leg of `top` or fewer candidates, or one whose
/// scores sum to nothing. Unchanged by any cap at or above `within`.
pub fn mass_ratio(sorted_desc: &[f32], top: usize, within: usize) -> f32 {
    let head: f32 = sorted_desc.iter().take(top).map(|s| s.max(0.0)).sum();
    let body: f32 = sorted_desc.iter().take(within).map(|s| s.max(0.0)).sum();
    if body <= 1e-6 {
        1.0
    } else {
        (head / body).clamp(0.0, 1.0)
    }
}

/// Number of scores at or above `floor`.
pub fn strong_count(scores: &[f32], floor: f32) -> usize {
    scores.iter().filter(|s| **s >= floor).count()
}

/// The gate's reading of one query.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct GateEval {
    /// Fitted probability that the vector leg ranks this query's gold above
    /// BM25; the fusion maps it to a vector-trust multiplier.
    pub t: f32,
    /// Each feature's distance from its fit mean, in fit standard deviations,
    /// before clamping.
    pub z: [f32; FEATURE_COUNT],
    /// Features whose `|z|` exceeded [`FIT_RANGE_SD`] and were clamped.
    pub out_of_fit: u8,
}

/// Standardise `feats` on the fit, clamp each to the fit range, and run the
/// logistic model.
pub fn evaluate(feats: &[f32; FEATURE_COUNT]) -> GateEval {
    let mut z = [0.0_f32; FEATURE_COUNT];
    let mut s = FIT_BIAS;
    let mut out_of_fit = 0u8;
    for (i, ((mu, sd, w), x)) in FIT.iter().zip(feats).enumerate() {
        let zi = if sd.abs() < 1e-9 { 0.0 } else { (x - mu) / sd };
        z[i] = zi;
        if zi.abs() > FIT_RANGE_SD {
            out_of_fit += 1;
        }
        s += w * zi.clamp(-FIT_RANGE_SD, FIT_RANGE_SD);
    }
    let t = 1.0 / (1.0 + (-s.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)).exp());
    GateEval { t, z, out_of_fit }
}

/// One query's gate inputs and reading, as the harness exports them.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct GateTrace {
    pub features: [f32; FEATURE_COUNT],
    pub eval: GateEval,
}

thread_local! {
    static TRACE: RefCell<Option<Option<GateTrace>>> = const { RefCell::new(None) };
}

/// Arm the trace for the next recall on this thread. Clears any prior state.
pub fn begin() {
    TRACE.with(|c| *c.borrow_mut() = Some(None));
}

/// Disarm and return the trace the last recall recorded. `None` if the trace
/// was never armed; `Some(None)` if it was armed but the fitted gate did not
/// run (another `SHODH_ADAPT_FEATURE`, or the adaptive path off).
pub fn take() -> Option<Option<GateTrace>> {
    TRACE.with(|c| c.borrow_mut().take())
}

/// Record the gate's inputs and reading for this recall. No-op unless armed.
pub fn record(feats: &[f32; FEATURE_COUNT], eval: &GateEval) {
    TRACE.with(|c| {
        if let Some(slot) = c.borrow_mut().as_mut() {
            *slot = Some(GateTrace {
                features: *feats,
                eval: *eval,
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at_fit_mean() -> [f32; FEATURE_COUNT] {
        let mut f = [0.0_f32; FEATURE_COUNT];
        for (i, (mu, _, _)) in FIT.iter().enumerate() {
            f[i] = *mu;
        }
        f
    }

    #[test]
    fn at_the_fit_mean_only_the_bias_speaks() {
        let e = evaluate(&at_fit_mean());
        let expected = 1.0 / (1.0 + (-FIT_BIAS).exp());
        assert!(
            (e.t - expected).abs() < 1e-6,
            "t={} expected {expected}",
            e.t
        );
        assert_eq!(e.out_of_fit, 0);
        assert!(e.z.iter().all(|z| z.abs() < 1e-6));
    }

    #[test]
    fn a_feature_far_outside_the_fit_is_clamped_and_counted() {
        let mut f = at_fit_mean();
        let (mu, sd, _) = FIT[5];
        f[5] = mu + 20.0 * sd;
        let e = evaluate(&f);
        assert_eq!(e.out_of_fit, 1);
        assert!(
            (e.z[5] - 20.0).abs() < 1e-4,
            "raw z is reported: {}",
            e.z[5]
        );
        let mut g = at_fit_mean();
        g[5] = mu + FIT_RANGE_SD * sd;
        let at_edge = evaluate(&g);
        assert!(
            (e.t - at_edge.t).abs() < 1e-6,
            "beyond the range the feature contributes exactly as at the range edge"
        );
    }

    #[test]
    fn the_logit_is_bounded_so_t_is_always_finite() {
        let mut f = at_fit_mean();
        for (i, (mu, sd, _)) in FIT.iter().enumerate() {
            f[i] = mu + 1e6 * sd;
        }
        let e = evaluate(&f);
        assert!(e.t.is_finite() && (0.0..=1.0).contains(&e.t));
    }

    #[test]
    fn mass_ratio_does_not_see_a_cap_at_or_beyond_its_window() {
        let leg: Vec<f32> = (0..300).map(|r| 1.0 / (1.0 + r as f32 * 0.05)).collect();
        let full = mass_ratio(&leg, MASS_TOP, MASS_WITHIN);
        let capped_90 = mass_ratio(&leg[..90], MASS_TOP, MASS_WITHIN);
        let capped_30 = mass_ratio(&leg[..30], MASS_TOP, MASS_WITHIN);
        assert_eq!(full, capped_90);
        assert_eq!(full, capped_30);
        assert!(full > 0.0 && full < 1.0, "{full}");
    }

    #[test]
    fn mass_ratio_reads_a_flat_leg_as_one_third_and_a_spike_as_one() {
        let flat = vec![0.5_f32; 60];
        assert!((mass_ratio(&flat, 10, 30) - 10.0 / 30.0).abs() < 1e-6);
        let mut spike = vec![0.0_f32; 60];
        spike[0] = 1.0;
        assert_eq!(mass_ratio(&spike, 10, 30), 1.0);
        assert_eq!(
            mass_ratio(&[], 10, 30),
            1.0,
            "an empty leg has no spread to read"
        );
    }

    #[test]
    fn strong_count_is_an_absolute_line_not_a_share() {
        let scores = [0.9_f32, 0.61, 0.6, 0.59, 0.1];
        assert_eq!(strong_count(&scores, VEC_STRONG_COSINE), 3);
        assert_eq!(strong_count(&scores[..2], VEC_STRONG_COSINE), 2);
    }

    #[test]
    fn trace_is_a_no_op_until_armed_and_drains_once() {
        let f = at_fit_mean();
        let e = evaluate(&f);
        record(&f, &e);
        assert_eq!(take(), None, "an unarmed trace records nothing");
        begin();
        assert_eq!(take(), Some(None), "armed but nothing recorded");
        begin();
        record(&f, &e);
        let got = take().flatten().expect("recorded trace");
        assert_eq!(got.features, f);
        assert_eq!(got.eval, e);
        assert_eq!(take(), None, "take disarms");
    }
}
