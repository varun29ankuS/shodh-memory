//! The fitted per-query trust gate: the logistic model that decides, from the
//! shape of a query's candidate legs, how far to trust the vector leg over
//! BM25 in the flat adaptive fusion (`SHODH_FLAT_ADAPTIVE`, default on, with
//! `SHODH_ADAPT_FEATURE=fitted`).
//!
//! The coefficients were fitted offline on the feature export the recall
//! harness writes (`SHODH_FUSION_FEATURE_EXPORT`), standardised per feature on
//! that export's mean and standard deviation. A logistic fit is only as good
//! as its inputs' distribution: a feature that drifts far from the mean it was
//! standardised on contributes a logit the fit never saw, and the gate's
//! output stops being the probability it was trained to be. [`evaluate`]
//! therefore reports, beside `t`, how many features sit outside
//! [`FIT_RANGE_SD`] standard deviations of their fit mean, and the harness
//! aggregates that over a suite so the drift is visible in the report rather
//! than in a ranking change nobody can attribute.
//!
//! Measured 2026-10-20 on the 100-case `locomo-gate` suite at 902daf8: two of
//! the eleven features (`n_graph`, `n_vec_pos`) sat 20 and 8 standard
//! deviations out on every query, because the leg caps had moved from 10 and
//! ~31 to 30 and ~90 since the fit. Their logits, +6.2 and −6.9, cancelled
//! closely enough that the gate still spread `t` over 0.002–0.749; any change
//! to either alone shifts every query.
//!
//! The per-query trace is a thread-local armed by the harness, like
//! [`super::gold_funnel`]: production callers pay one `RefCell` borrow.

use std::cell::RefCell;

/// Number of features the gate consumes; the order is [`FEATURE_NAMES`].
pub const FEATURE_COUNT: usize = 11;

/// Feature names in the order `evaluate` expects them. Matches the
/// `FusionFeatures` export field names where one exists.
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "bm_peak",
    "vec_peak",
    "agreement_top10",
    "max_bm",
    "max_vec",
    "n_bm_pos",
    "n_vec_pos",
    "n_hybrid",
    "n_graph",
    "graph_max_activation",
    "query_len",
];

/// (mean, standard deviation, weight) per standardised feature, from the
/// run-27267533447 fit. Order must match [`FEATURE_NAMES`].
pub const FIT: [(f32, f32, f32); FEATURE_COUNT] = [
    (2.77242, 1.87083, -0.301375),    // bm_peak
    (1.3841, 0.180661, -0.0212517),   // vec_peak
    (0.307389, 0.170755, -0.243719),  // agreement_top10
    (93.4129, 43.8602, -0.556471),    // max_bm
    (0.597371, 0.0823258, 0.236463),  // max_vec
    (106.897, 36.4931, 0.597537),     // n_bm_pos
    (31.4138, 7.54534, -0.881755),    // n_vec_pos
    (116.222, 38.3149, 0.582615),     // n_hybrid
    (9.90148, 0.987682, 0.304049),    // n_graph
    (0.358194, 0.0710801, 0.0264384), // graph_max_activation
    (54.1034, 16.0475, -0.571797),    // query_len
];

/// Intercept of the same fit.
pub const FIT_BIAS: f32 = -0.985886;

/// How many standard deviations from its fit mean a feature may sit before
/// it counts as outside the fit. Four covers every value a fit of a few
/// hundred queries can have seen; beyond it the logit is extrapolation.
pub const FIT_RANGE_SD: f32 = 4.0;

/// Logit bound before the sigmoid; keeps `exp` finite on any input.
const LOGIT_CLAMP: f32 = 30.0;

/// The gate's reading of one query.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct GateEval {
    /// Fitted probability that the vector leg ranks this query's gold above
    /// BM25; the fusion maps it to a vector-trust multiplier.
    pub t: f32,
    /// Each feature's distance from its fit mean, in fit standard deviations.
    pub z: [f32; FEATURE_COUNT],
    /// Features whose `|z|` exceeds [`FIT_RANGE_SD`].
    pub out_of_fit: u8,
}

/// Standardise `feats` on the fit and run the logistic model.
pub fn evaluate(feats: &[f32; FEATURE_COUNT]) -> GateEval {
    let mut z = [0.0_f32; FEATURE_COUNT];
    let mut s = FIT_BIAS;
    let mut out_of_fit = 0u8;
    for (i, ((mu, sd, w), x)) in FIT.iter().zip(feats).enumerate() {
        let zi = if sd.abs() < 1e-9 { 0.0 } else { (x - mu) / sd };
        z[i] = zi;
        s += w * zi;
        if zi.abs() > FIT_RANGE_SD {
            out_of_fit += 1;
        }
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
    fn a_feature_far_from_its_fit_mean_is_counted_not_hidden() {
        let mut f = at_fit_mean();
        let (mu, sd, _) = FIT[8]; // n_graph
        f[8] = mu + 20.0 * sd;
        let e = evaluate(&f);
        assert_eq!(e.out_of_fit, 1);
        assert!((e.z[8] - 20.0).abs() < 1e-4, "z={}", e.z[8]);
    }

    #[test]
    fn the_gate_suite_shape_at_902daf8_has_two_features_out_of_fit() {
        // The locomo-gate shape measured on 2026-10-20: graph leg 30 wide,
        // vector leg 90 wide, everything else near its fit mean. This is the
        // drift the report's out-of-fit fraction exists to surface.
        let mut f = at_fit_mean();
        f[6] = 90.0;
        f[8] = 30.0;
        let e = evaluate(&f);
        assert_eq!(e.out_of_fit, 2);
        assert!(e.z[6] > FIT_RANGE_SD && e.z[8] > FIT_RANGE_SD);
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
