//! Single source of truth for retrieval-time ablation families.
//!
//! Every ablatable retrieval component names a FAMILY here and asks
//! [`is_enabled`]. One env var, one list, one place to look.
//!
//! Before this module there were two idioms and a gap. The semantic leg's boost
//! stack had an 18-family kill-switch parsed inline inside
//! `semantic_retrieve_inner`, while the graph leg's own scoring components —
//! lateral inhibition and Hebbian potentiation — had NO gate at all and could
//! therefore never be ablated. A component that cannot be switched off cannot be
//! measured, and an unmeasurable component accumulates in the pipeline on the
//! strength of its rationale rather than its effect. Both are now families like
//! any other.
//!
//! `SHODH_DISABLE_BOOSTS=<family,...>` disables the named families for the
//! process. The token `all` disables every family. Setting the variable at all —
//! even to the empty string — REPLACES [`DEFAULT_DISABLED`] rather than adding
//! to it, which is the escape hatch for running with nothing disabled.

use std::collections::HashSet;

/// Every ablatable retrieval component, by family name.
///
/// Semantic leg (`memory/mod.rs`, Layers 4.45 through 5.7): `attribute` through
/// `competition`.
///
/// Graph leg (`memory/graph_retrieval.rs`):
/// * `lateral_inhibition` — Step 6.5 winner-take-all, penalising a candidate
///   within `GRAPH_LATERAL_INHIBITION_THRESHOLD` cosine of a higher-ranked one.
/// * `quality_verbosity` — the raw content-LENGTH factor inside the Layer-5
///   quality gate, `(len/200).min(1.0)`, which multiplies a short correct answer
///   (a person name) below a longer wrong one (an org name). Disabling the family
///   keeps only the structural `elaboration` term. Previously reachable only via
///   `SHODH_FUSION_V2`, which also switched fusion to weighted-Borda, so the
///   verbosity factor could never be measured on its own.
/// * `graph_potentiation` — retrieval-driven edge strengthening
///   (`batch_strengthen_synapses`). Distinct from the `hebbian` family, which is
///   the semantic leg's L5 RANK boost; this is the WRITE that feeds
///   `ppr_edge_weight`. Also distinct from `SHODH_RECALL_READONLY`, which
///   suppresses it for measurement integrity — this family ablates the mechanism
///   while leaving other recall-path writes alone, so the two can be separated.
///
/// * `linguistic_resort` — whether the linguistic signal is applied as a SEPARATE
///   sort-only re-rank (enabled, current) or folded additively into the single
///   `.score` (disabled). Composes with the `linguistic` family, which controls
///   whether the signal applies at all. This one selects between two FORMS rather
///   than deleting a component; the alternative was a second config idiom, and one
///   documented mechanism beats two clean ones.
/// * `size_gated_final_sort` — the SIZE GATE on the final re-sort. Enabled
///   (current) means the sort runs only when `len > max_results`, so result-set
///   size decides whether the lexical re-rank or `.score` wins the final order.
///   Disabling removes the gate and always sorts by score.
///
/// Adding a name here is what makes a component measurable; adding one without a
/// matching [`is_enabled`] call at the component's site makes the family a lie,
/// so the two changes belong in the same commit.
pub const FAMILIES: [&str; 23] = [
    "attribute",
    "temporal_prefilter",
    "temporal_fact",
    "interference",
    "prospective",
    "fact_source",
    "ontological",
    "hebbian",
    "recency",
    "arousal",
    "credibility",
    "temporal_match",
    "feedback",
    "importance",
    "tag_penalty",
    "quality",
    "linguistic",
    "competition",
    "lateral_inhibition",
    "graph_potentiation",
    "quality_verbosity",
    "linguistic_resort",
    "size_gated_final_sort",
];

/// Families disabled when `SHODH_DISABLE_BOOSTS` is unset.
///
/// `hebbian` — the semantic leg's L5 rank boost — is off by default because the
/// L5 bisect (run 27251798933) measured it as a strict ordering saboteur: p@1
/// 0.4100 -> 0.4767 (+6.7pp) with recall@10 bit-identical. Its scores come from
/// graph co-activation, and edges strengthen on EVERY retrieval rather than on
/// outcome, so frequently co-retrieved hub memories climb within the top-10 and
/// displace gold at rank 1.
pub const DEFAULT_DISABLED: [&str; 1] = ["hebbian"];

/// Resolve the disabled-family set from the environment.
///
/// Unknown tokens are warned about and ignored, so a typo degrades to "nothing
/// extra disabled" rather than appearing to ablate a family that does not exist
/// and reporting the resulting null as a measurement.
pub fn disabled_families() -> HashSet<String> {
    let set: HashSet<String> = match std::env::var("SHODH_DISABLE_BOOSTS") {
        Ok(v) => v
            .split(',')
            .map(|t| t.trim().to_ascii_lowercase())
            .filter(|t| !t.is_empty())
            .collect(),
        Err(_) => DEFAULT_DISABLED.iter().map(|s| s.to_string()).collect(),
    };
    for tok in &set {
        if tok != "all" && !FAMILIES.contains(&tok.as_str()) {
            tracing::warn!("SHODH_DISABLE_BOOSTS: unknown ablation family '{tok}' (ignored)");
        }
    }
    set
}

/// Whether `family` is currently ablated.
pub fn is_disabled(family: &str) -> bool {
    debug_assert!(
        FAMILIES.contains(&family),
        "ablation family is not declared in ablation::FAMILIES"
    );
    let disabled = disabled_families();
    disabled.contains("all") || disabled.contains(family)
}

/// Whether `family` should run. The form call sites read most naturally.
pub fn is_enabled(family: &str) -> bool {
    !is_disabled(family)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialises tests that mutate `SHODH_DISABLE_BOOSTS`, which is
    /// process-global state.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        match value {
            Some(v) => std::env::set_var("SHODH_DISABLE_BOOSTS", v),
            None => std::env::remove_var("SHODH_DISABLE_BOOSTS"),
        }
        let out = f();
        std::env::remove_var("SHODH_DISABLE_BOOSTS");
        out
    }

    #[test]
    fn family_list_has_no_duplicates() {
        let unique: HashSet<&str> = FAMILIES.iter().copied().collect();
        assert_eq!(unique.len(), FAMILIES.len(), "duplicate family in FAMILIES");
    }

    #[test]
    fn every_default_disabled_family_is_declared() {
        for f in DEFAULT_DISABLED {
            assert!(
                FAMILIES.contains(&f),
                "default-disabled family is undeclared"
            );
        }
    }

    #[test]
    fn unset_env_disables_exactly_the_documented_default() {
        with_env(None, || {
            assert!(
                is_disabled("hebbian"),
                "L5 rank boost must stay off by default"
            );
            for f in FAMILIES.iter().filter(|f| !DEFAULT_DISABLED.contains(f)) {
                assert!(
                    is_enabled(f),
                    "family must be enabled when env is unset: {f}"
                );
            }
        });
    }

    #[test]
    fn newly_declared_families_ship_enabled() {
        // All three were ungated or unreachable before this module existed.
        // Declaring them must not change shipped behaviour, only make it
        // measurable - `quality_verbosity` in particular is a KNOWN-HARMFUL
        // factor that stays on until its removal is measured on its own.
        with_env(None, || {
            assert!(is_enabled("lateral_inhibition"));
            assert!(is_enabled("graph_potentiation"));
            assert!(is_enabled("quality_verbosity"));
        });
    }

    #[test]
    fn graph_potentiation_is_not_collateral_of_the_hebbian_default() {
        // `hebbian` (semantic L5 rank boost) is default-disabled. The graph-leg
        // WRITE is a different mechanism and must not switch off alongside it.
        with_env(None, || {
            assert!(is_disabled("hebbian"));
            assert!(is_enabled("graph_potentiation"));
        });
    }

    #[test]
    fn explicit_empty_value_replaces_the_default_rather_than_adding_to_it() {
        with_env(Some(""), || {
            assert!(
                is_enabled("hebbian"),
                "explicit empty must clear the default"
            );
        });
    }

    #[test]
    fn all_disables_every_declared_family() {
        with_env(Some("all"), || {
            for f in FAMILIES {
                assert!(is_disabled(f), "family survived 'all': {f}");
                // Assert through `is_enabled` too: that is the form call sites
                // use, and an `is_enabled` that can never return false would
                // leave every gate permanently open while `is_disabled` stayed
                // correct. Mutation-checked - without this the suite passes with
                // `is_enabled` hardcoded to true.
                assert!(!is_enabled(f), "is_enabled disagreed with is_disabled: {f}");
            }
        });
    }

    #[test]
    fn is_enabled_is_the_exact_negation_of_is_disabled() {
        for env in [
            None,
            Some(""),
            Some("all"),
            Some("lateral_inhibition,quality"),
        ] {
            with_env(env, || {
                for f in FAMILIES {
                    assert_ne!(
                        is_enabled(f),
                        is_disabled(f),
                        "is_enabled/is_disabled agreed for {f}"
                    );
                }
            });
        }
    }

    #[test]
    fn named_families_are_disabled_and_others_untouched() {
        with_env(Some("lateral_inhibition, quality"), || {
            assert!(is_disabled("lateral_inhibition"));
            assert!(!is_enabled("lateral_inhibition"));
            assert!(is_disabled("quality"));
            assert!(is_enabled("graph_potentiation"));
            assert!(is_enabled("recency"));
        });
    }

    #[test]
    fn unknown_token_does_not_disable_anything_real() {
        with_env(Some("latteral_inhibition"), || {
            for f in FAMILIES {
                assert!(is_enabled(f), "typo disabled a real family: {f}");
            }
        });
    }
}
