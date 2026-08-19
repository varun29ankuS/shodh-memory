//! How much a retrieval number is allowed to mean.
//!
//! Every metric this harness reports is a mean over a finite set of questions.
//! Printed bare, `p@1 = 0.28` invites a reader — a reviewer, a future maintainer,
//! an investor's technical diligence — to treat it as a property of the system.
//! It is a property of the system *measured on 25 questions*, and those are very
//! different claims. This module exists so the second claim is the only one the
//! harness is able to make.
//!
//! # What the interval does and does not describe
//!
//! **Not run-to-run noise.** Retrieval here is deterministic: the harness hard
//! fails if rank lists diverge between repeats (RH-12), so repeating a run
//! reproduces every metric exactly. Rerunning tells you nothing new, which is
//! precisely why `--repeats` buys determinism and latency medians rather than
//! precision.
//!
//! **Generalisation to the population of questions.** The suite is a sample of
//! the questions users will actually ask. The interval answers: if this same
//! system met a different draw of questions from the same distribution, how far
//! could the number move? That is the question a reader is really asking, and
//! the one a point estimate silently answers "not at all".
//!
//! The practical consequence on the current gate: `multi_hop` carries **n = 25**,
//! so `p@1` moves in steps of 0.04 and its 95% interval is roughly ±0.18, while
//! the gate's 2% tolerance is 0.0056 absolute. The gate is therefore detecting
//! *changes on a fixed benchmark* — which is legitimate and deterministic — while
//! being read as *evidence about quality in general*, which at that sample size
//! it cannot support. Both readings are defensible; conflating them is not.
//!
//! # Why two estimators
//!
//! `p@1` is a proportion: each case contributes 0 or 1. The textbook normal
//! approximation `p ± z·√(p(1-p)/n)` is unreliable exactly where this harness
//! lives — small `n`, and proportions near 0 or 1, where it produces intervals
//! that run past the [0,1] boundary and under-cover badly. [`wilson_ci95`] is
//! used instead; it is well behaved at small `n` and never leaves [0,1].
//!
//! `ndcg@10`, `recall@10` and `mrr` are means of per-case scores anywhere in
//! [0,1], not proportions, so their spread depends on a per-case variance the
//! report does not store. Rather than invent one, [`bounded_mean_ci95`] applies
//! Popoviciu's inequality: any variable confined to [0,1] has variance at most
//! 1/4. That yields a rigorous *upper bound* on the interval — never
//! optimistic, occasionally wider than reality. An honest over-estimate of
//! uncertainty is the right failure direction for a number that will be quoted.

/// 1.96 — the two-sided 95% normal quantile.
const Z_95: f64 = 1.959_963_984_540_054;

/// Half-width of the 95% Wilson score interval for a proportion.
///
/// Returns the larger of the two arms so the value can be used as a symmetric
/// ± figure; the Wilson interval is asymmetric near the boundaries, and taking
/// the wider side keeps the reported uncertainty conservative.
///
/// `n == 0` yields 1.0: with no observations the metric is unconstrained across
/// its whole range, which is the honest answer rather than a divide-by-zero.
pub fn wilson_ci95(p: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let n_f = n as f64;
    let p = p.clamp(0.0, 1.0);
    let z2 = Z_95 * Z_95;
    let denom = 1.0 + z2 / n_f;
    let centre = (p + z2 / (2.0 * n_f)) / denom;
    let spread = (Z_95 / denom) * ((p * (1.0 - p) / n_f) + z2 / (4.0 * n_f * n_f)).sqrt();
    let lo = (centre - spread).max(0.0);
    let hi = (centre + spread).min(1.0);
    (p - lo).abs().max((hi - p).abs())
}

/// Half-width of a conservative 95% interval for the mean of per-case scores
/// confined to [0,1].
///
/// Uses Popoviciu's bound (variance ≤ 1/4), so this is an upper bound on the
/// true interval and never understates uncertainty. Independent of the observed
/// value, which is why it takes only `n`.
pub fn bounded_mean_ci95(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    (Z_95 * 0.5 / (n as f64).sqrt()).min(1.0)
}

/// The smallest change in a metric that `n` questions can distinguish from a
/// different draw of questions, at 95%.
///
/// A gate tolerance below this is not a stricter gate — it is a gate asserting
/// a distinction its sample cannot support. Use it to *label* a comparison, not
/// to widen a threshold: the fix for an under-resolved metric is more questions,
/// never a looser bound.
pub fn min_resolvable_delta(n: usize) -> f64 {
    bounded_mean_ci95(n)
}

/// How many questions would be needed for `absolute_delta` to sit at the edge of
/// the 95% interval.
///
/// Answers "what would it take to gate this honestly?" — and on the current
/// suite the answer is often larger than the benchmark, which is itself the
/// finding.
pub fn questions_needed_for(absolute_delta: f64) -> Option<usize> {
    if absolute_delta <= 0.0 {
        return None;
    }
    let n = (Z_95 * 0.5 / absolute_delta).powi(2).ceil();
    if n.is_finite() && n >= 1.0 {
        Some(n as usize)
    } else {
        None
    }
}

/// A metric with the uncertainty that belongs to it.
///
/// Constructed rather than assembled at each print site so no call path can
/// emit a value without its `n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    pub value: f64,
    pub n: usize,
    /// 95% half-width. Wilson for proportions, Popoviciu bound otherwise.
    pub ci95: f64,
}

impl Measured {
    /// For a proportion — every case contributed 0 or 1 (`p@1`).
    pub fn proportion(value: f64, n: usize) -> Self {
        Self {
            value,
            n,
            ci95: wilson_ci95(value, n),
        }
    }

    /// For a mean of per-case scores in [0,1] (`ndcg@10`, `recall@10`, `mrr`).
    pub fn bounded_mean(value: f64, n: usize) -> Self {
        Self {
            value,
            n,
            ci95: bounded_mean_ci95(n),
        }
    }

    /// `0.2800 ± 0.1764 (n=25)` — the only rendering, so a bare value cannot be
    /// printed by accident.
    pub fn render(&self) -> String {
        format!("{:.4} ± {:.4} (n={})", self.value, self.ci95, self.n)
    }

    /// Whether a change of `delta` against this measurement is larger than the
    /// sample can resolve. `false` does not mean "no change occurred" — on a
    /// deterministic benchmark a smaller change is still real *for these
    /// questions*; it means the change is not evidence about questions in
    /// general.
    pub fn resolves(&self, delta: f64) -> bool {
        delta.abs() >= self.ci95
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The number that started this: the gate blocked a PR on a 0.0056 tolerance
    /// against a metric whose 95% interval is ±0.18.
    #[test]
    fn multi_hop_p_at_1_on_the_current_gate_is_not_resolvable_at_two_percent() {
        let m = Measured::proportion(0.28, 25);
        let gate_tolerance = 0.28 * 0.02;
        assert!(
            m.ci95 > gate_tolerance * 10.0,
            "expected the interval to dwarf the tolerance; ci95={:.4} tolerance={:.4}",
            m.ci95,
            gate_tolerance
        );
        assert!(
            !m.resolves(gate_tolerance),
            "a 2% tolerance must not be reported as resolvable at n=25"
        );
        // One case flipping out of 25.
        assert!(
            !m.resolves(0.04),
            "a single-case flip is not evidence in general"
        );
    }

    /// Growing the sample is the fix, and the numbers have to show it.
    #[test]
    fn more_questions_shrink_the_interval() {
        let small = Measured::proportion(0.28, 25);
        let large = Measured::proportion(0.28, 281);
        assert!(
            large.ci95 < small.ci95 / 2.0,
            "{} vs {}",
            small.ci95,
            large.ci95
        );
    }

    /// Wilson must stay inside [0,1] where the normal approximation does not.
    /// At p=0 the naive formula gives a half-width of exactly 0, claiming perfect
    /// certainty from a handful of observations.
    #[test]
    fn wilson_is_sane_at_the_boundaries_where_the_normal_approximation_is_not() {
        let naive_at_zero = Z_95 * (0.0f64 * 1.0 / 10.0).sqrt();
        assert_eq!(naive_at_zero, 0.0, "the approximation this replaces");
        let w = wilson_ci95(0.0, 10);
        assert!(w > 0.0, "zero successes in 10 tries is not certainty");
        assert!(w <= 1.0);
        assert!(wilson_ci95(1.0, 10) > 0.0);
        for &(p, n) in &[(0.0, 1), (1.0, 1), (0.5, 3), (0.28, 25)] {
            let w = wilson_ci95(p, n);
            assert!((0.0..=1.0).contains(&w), "p={p} n={n} gave {w}");
            assert!(p - w >= -f64::EPSILON || p - w <= 1.0);
        }
    }

    /// No observations must not read as a confident zero.
    #[test]
    fn an_empty_sample_reports_maximum_uncertainty() {
        assert_eq!(wilson_ci95(0.0, 0), 1.0);
        assert_eq!(bounded_mean_ci95(0), 1.0);
    }

    /// The bound is conservative by construction — it must never be tighter than
    /// the proportion interval it may be compared against.
    #[test]
    fn the_bounded_mean_estimator_never_understates_uncertainty() {
        for n in [5usize, 25, 100, 281, 1531] {
            assert!(
                bounded_mean_ci95(n) >= wilson_ci95(0.5, n) - 1e-9,
                "n={n}: popoviciu bound must dominate the p=0.5 proportion interval"
            );
        }
    }

    /// The finding this module exists to make unmissable: a 2% relative tolerance
    /// on a retrieval metric needs a benchmark far larger than the field runs.
    #[test]
    fn a_two_percent_tolerance_needs_more_questions_than_the_suite_has() {
        let needed = questions_needed_for(0.28 * 0.02).expect("finite");
        assert!(
            needed > 1531,
            "expected the requirement to exceed the whole held-out suite, got {needed}"
        );
    }

    #[test]
    fn rendering_always_carries_n_and_the_interval() {
        let s = Measured::proportion(0.28, 25).render();
        assert!(s.contains("n=25"), "{s}");
        assert!(s.contains('±'), "{s}");
    }
}
