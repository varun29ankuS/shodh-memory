//! Hybrid Decay Model (SHO-103)
//!
//! Implements biologically-accurate memory decay based on neuroscience research.
//!
//! # The Problem with Pure Exponential Decay
//!
//! Traditional memory systems use exponential decay: `w(t) = w₀ × e^(-λt)`
//!
//! This produces a "cliff" effect where memories drop rapidly and then flatten:
//! - Day 1: 100% → 95%
//! - Day 7: 95% → 70%
//! - Day 30: 70% → 15% (steep cliff)
//!
//! # The Solution: Hybrid Decay
//!
//! Human memory follows a power-law for long-term retention, not exponential.
//!
//! This module implements a hybrid model:
//! - **Consolidation phase** (t < 3 days): Exponential decay
//!   - Fast filtering of noise and weak associations
//!   - Matches short-term synaptic depression
//! - **Long-term phase** (t ≥ 3 days): Power-law decay
//!   - Heavy tail preserves important memories longer
//!   - Matches empirical human forgetting curves
//!
//! ```text
//!         Exponential              Power-Law
//!         (consolidation)          (long-term retention)
//!
//! Strength │ ╲
//!     100% │  ╲
//!          │   ╲
//!      60% │    ╲___
//!          │        ╲____
//!      30% │             ╲________
//!          │                      ╲___________
//!       5% │─────────────────────────────────────────
//!          └────┬────────┬─────────────────────────► Time
//!               │        │
//!            t_cross   (days)
//!          (3 days)
//! ```
//!
//! # References
//!
//! - Wixted & Ebbesen (1991) "On the Form of Forgetting"
//! - Wixted (2004) "The psychology and neuroscience of forgetting"
//! - Anderson & Schooler (1991) "Reflections of the Environment in Memory"

use crate::constants::{
    DECAY_CROSSOVER_DAYS, DECAY_LAMBDA_CONSOLIDATION, POWERLAW_BETA, POWERLAW_BETA_POTENTIATED,
};

/// Calculates the hybrid decay factor for a given elapsed time.
///
/// Returns a value between 0.0 and 1.0 representing the retention ratio.
///
/// # Arguments
///
/// * `days_elapsed` - Time since last activation in days
/// * `potentiated` - Whether this is a potentiated/important memory (uses slower decay)
///
/// # Returns
///
/// Decay factor to multiply with original strength: `new_strength = old_strength * decay_factor`
///
/// # Example
///
/// ```ignore
/// let factor = hybrid_decay_factor(7.0, false);
/// let new_strength = old_strength * factor;
/// ```
#[inline]
pub fn hybrid_decay_factor(days_elapsed: f64, potentiated: bool) -> f32 {
    if days_elapsed <= 0.0 {
        return 1.0;
    }

    let beta = if potentiated {
        POWERLAW_BETA_POTENTIATED
    } else {
        POWERLAW_BETA
    };

    // Exponential rate for consolidation phase
    // Potentiated memories use slower exponential decay too
    let lambda = if potentiated {
        DECAY_LAMBDA_CONSOLIDATION * 0.5 // Half the rate for potentiated
    } else {
        DECAY_LAMBDA_CONSOLIDATION
    };

    if days_elapsed < DECAY_CROSSOVER_DAYS {
        // Consolidation phase: exponential decay
        // w(t) = w₀ × e^(-λt)
        (-lambda * days_elapsed).exp() as f32
    } else {
        // Long-term phase: power-law decay
        // First, calculate what value we'd have at crossover with exponential
        let value_at_crossover = (-lambda * DECAY_CROSSOVER_DAYS).exp();

        // Then apply power-law from crossover point
        // A(t) = A_cross × (t / t_cross)^(-β)
        let power_law_factor = (days_elapsed / DECAY_CROSSOVER_DAYS).powf(-beta);

        (value_at_crossover * power_law_factor) as f32
    }
}

/// Calculates the hybrid decay factor with custom parameters.
///
/// Use this for contexts that need different decay characteristics.
///
/// # Arguments
///
/// * `days_elapsed` - Time since last activation in days
/// * `crossover_days` - Days before switching from exponential to power-law
/// * `lambda` - Exponential decay rate for consolidation phase
/// * `beta` - Power-law exponent for long-term phase
///
/// # Example
///
/// ```ignore
/// // Faster decay for edge weights
/// let factor = hybrid_decay_factor_custom(days_elapsed, 1.0, 1.0, 0.6);
/// ```
#[inline]
pub fn hybrid_decay_factor_custom(
    days_elapsed: f64,
    crossover_days: f64,
    lambda: f64,
    beta: f64,
) -> f32 {
    if days_elapsed <= 0.0 {
        return 1.0;
    }

    if days_elapsed < crossover_days {
        // Consolidation phase: exponential decay
        (-lambda * days_elapsed).exp() as f32
    } else {
        // Long-term phase: power-law decay
        let value_at_crossover = (-lambda * crossover_days).exp();
        let power_law_factor = (days_elapsed / crossover_days).powf(-beta);
        (value_at_crossover * power_law_factor) as f32
    }
}

/// Calculates retention percentage for debugging/visualization.
///
/// Returns a human-readable percentage string showing retention at various time points.
#[allow(dead_code)]
pub fn retention_curve_debug(potentiated: bool) -> String {
    let days = [0.5, 1.0, 3.0, 7.0, 14.0, 30.0, 90.0, 365.0];
    let mode = if potentiated { "potentiated" } else { "normal" };

    let mut output = format!("Retention curve ({}):\n", mode);
    for d in days {
        let factor = hybrid_decay_factor(d, potentiated);
        output.push_str(&format!("  Day {:>5.1}: {:>6.2}%\n", d, factor * 100.0));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_decay_at_zero() {
        assert_eq!(hybrid_decay_factor(0.0, false), 1.0);
        assert_eq!(hybrid_decay_factor(-1.0, false), 1.0);
    }

    #[test]
    fn test_exponential_phase() {
        // During consolidation (< 3 days), should be exponential
        let factor_1day = hybrid_decay_factor(1.0, false);
        let factor_2day = hybrid_decay_factor(2.0, false);

        // Exponential property: ratio should be constant
        let ratio_1_to_2 = factor_2day / factor_1day;
        let expected_ratio = (-DECAY_LAMBDA_CONSOLIDATION).exp() as f32;

        assert!((ratio_1_to_2 - expected_ratio).abs() < 0.01);
    }

    #[test]
    fn test_powerlaw_phase() {
        // After crossover (> 3 days), should be power-law
        let factor_7day = hybrid_decay_factor(7.0, false);
        let factor_14day = hybrid_decay_factor(14.0, false);

        // Power-law property: doubling time should give 2^(-β) ratio
        let ratio = factor_14day / factor_7day;
        let expected_ratio = 2.0_f64.powf(-POWERLAW_BETA) as f32;

        assert!((ratio - expected_ratio).abs() < 0.02);
    }

    #[test]
    fn test_continuity_at_crossover() {
        // Values just before and after crossover should be close
        let just_before = hybrid_decay_factor(DECAY_CROSSOVER_DAYS - 0.001, false);
        let just_after = hybrid_decay_factor(DECAY_CROSSOVER_DAYS + 0.001, false);

        assert!((just_before - just_after).abs() < 0.01);
    }

    #[test]
    fn test_potentiated_decays_slower() {
        let normal = hybrid_decay_factor(30.0, false);
        let potentiated = hybrid_decay_factor(30.0, true);

        // Potentiated should retain more
        assert!(potentiated > normal);
    }

    #[test]
    fn test_heavy_tail_retention() {
        // Key property: power-law has heavy tail
        // At 365 days, we should still have meaningful retention
        let year_retention = hybrid_decay_factor(365.0, false);
        let year_retention_potentiated = hybrid_decay_factor(365.0, true);

        // Normal: should be > 1%
        assert!(year_retention > 0.01);
        // Potentiated: should be > 5%
        assert!(year_retention_potentiated > 0.05);
    }

    #[test]
    fn test_custom_parameters() {
        // Test custom function with aggressive decay
        let aggressive = hybrid_decay_factor_custom(7.0, 1.0, 1.5, 0.7);
        let normal = hybrid_decay_factor(7.0, false);

        assert!(aggressive < normal);
    }
}
