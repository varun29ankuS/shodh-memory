//! Fellegi-Sunter probabilistic entity matcher (ER Phase 2.1).
//!
//! A label-free learned matcher for entity mentions — the Splink method ported to
//! Rust with no ML runtime, just m/u probability tables and a log-weight sum.
//! Validated in Python (`demos/gdelt-bridge/splink_explore.py`): bridge F1 ≈ 0.78
//! with zero labels, interpretable Bayes factors.
//!
//! For a candidate pair of mentions, each of three comparisons is resolved to a
//! discrete agreement *level*:
//!   - `name`: Jaro-Winkler at thresholds [0.92, 0.75, 0.5] → 4 levels.
//!   - `head`: exact syntactic-head match → 2 levels.
//!   - `type`: exact entity-type match → 2 levels.
//!
//! Each level carries an `m` probability (P(level | the pair is a match)) and a
//! `u` probability (P(level | not a match)). The match weight is
//!   `w = log2(λ / (1-λ)) + Σ_c log2(m[c][level] / u[c][level])`
//! and the match probability is `2^w / (2^w + 1)`. `u` is estimated from the level
//! frequencies over all pairs (dominated by non-matches); `m` and the prior `λ`
//! are estimated by unsupervised expectation-maximisation. The result is a set of
//! inspectable Bayes factors — no black box.
//!
//! FROZEN structure; the m/u weights are the ONLINE-learned part (re-fittable
//! per domain, and updatable by streaming EM in Phase 4.1). Nothing here needs a
//! model: it consumes pre-extracted `(name, head, type)` records.

/// One mention as the matcher sees it. `name` should be caller-normalised
/// (lowercased, determiner-stripped); `head` is the dependency-head lemma; and
/// `entity_type` is the NER/GLiNER label.
#[derive(Debug, Clone)]
pub struct MatchRecord {
    pub name: String,
    pub head: String,
    pub entity_type: String,
}

const NAME_THRESHOLDS: [f64; 3] = [0.92, 0.75, 0.5];
const NAME_LEVELS: usize = 4; // ≥0.92, ≥0.75, ≥0.5, else
const BINARY_LEVELS: usize = 2; // equal, else
/// Probability floor so a never-observed level can't drive `log2` to ±∞.
const PROB_FLOOR: f64 = 1e-6;

/// Jaro-Winkler string similarity in [0,1]. Standard algorithm (Jaro similarity
/// with a Winkler common-prefix boost, prefix capped at 4, scaling 0.1).
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let j = jaro(a, b);
    if j == 0.0 {
        return 0.0;
    }
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let prefix = a
        .iter()
        .zip(b.iter())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count() as f64;
    j + prefix * 0.1 * (1.0 - j)
}

fn jaro(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (la, lb) = (a.len(), b.len());
    if la == 0 && lb == 0 {
        return 1.0;
    }
    if la == 0 || lb == 0 {
        return 0.0;
    }
    let max_dist = (la.max(lb) / 2).saturating_sub(1);
    let mut a_match = vec![false; la];
    let mut b_match = vec![false; lb];
    let mut matches = 0usize;
    for (i, &ca) in a.iter().enumerate() {
        let lo = i.saturating_sub(max_dist);
        let hi = (i + max_dist + 1).min(lb);
        for j in lo..hi {
            if !b_match[j] && b[j] == ca {
                a_match[i] = true;
                b_match[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    // Count transpositions.
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..la {
        if a_match[i] {
            while !b_match[k] {
                k += 1;
            }
            if a[i] != b[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }
    let m = matches as f64;
    let t = (transpositions / 2) as f64;
    (m / la as f64 + m / lb as f64 + (m - t) / m) / 3.0
}

fn name_level(a: &str, b: &str) -> usize {
    let s = jaro_winkler(a, b);
    for (lvl, &th) in NAME_THRESHOLDS.iter().enumerate() {
        if s >= th {
            return lvl;
        }
    }
    NAME_THRESHOLDS.len()
}

/// Exact-match level: 0 if equal and non-empty, else 1.
fn exact_level(a: &str, b: &str) -> usize {
    if !a.is_empty() && a == b {
        0
    } else {
        1
    }
}

/// The three agreement levels for a pair: (name, head, type).
fn levels(a: &MatchRecord, b: &MatchRecord) -> (usize, usize, usize) {
    (
        name_level(&a.name, &b.name),
        exact_level(&a.head, &b.head),
        exact_level(&a.entity_type, &b.entity_type),
    )
}

/// A learned Fellegi-Sunter model: per-comparison m/u probability tables and the
/// prior match rate λ.
#[derive(Debug, Clone)]
pub struct FellegiSunter {
    /// Prior P(a random pair is a match).
    pub lambda: f64,
    pub name_m: [f64; NAME_LEVELS],
    pub name_u: [f64; NAME_LEVELS],
    pub head_m: [f64; BINARY_LEVELS],
    pub head_u: [f64; BINARY_LEVELS],
    pub type_m: [f64; BINARY_LEVELS],
    pub type_u: [f64; BINARY_LEVELS],
}

#[inline]
fn bayes_weight(m: f64, u: f64) -> f64 {
    (m.max(PROB_FLOOR) / u.max(PROB_FLOOR)).log2()
}

impl FellegiSunter {
    /// Total match weight (log2 Bayes factor incl. the prior) for a pair.
    pub fn match_weight(&self, a: &MatchRecord, b: &MatchRecord) -> f64 {
        let (nl, hl, tl) = levels(a, b);
        let prior = (self.lambda.max(PROB_FLOOR) / (1.0 - self.lambda).max(PROB_FLOOR)).log2();
        prior
            + bayes_weight(self.name_m[nl], self.name_u[nl])
            + bayes_weight(self.head_m[hl], self.head_u[hl])
            + bayes_weight(self.type_m[tl], self.type_u[tl])
    }

    /// Posterior match probability in [0,1].
    pub fn match_probability(&self, a: &MatchRecord, b: &MatchRecord) -> f64 {
        let bf = 2f64.powf(self.match_weight(a, b));
        bf / (bf + 1.0)
    }

    /// Fit m/u and λ from unlabeled records. `u` is the level distribution over
    /// all pairs (dominated by non-matches); `m` and λ are estimated by EM. The
    /// EM is seeded to favour agreement so the two clusters (match / non-match)
    /// separate deterministically.
    pub fn fit(records: &[MatchRecord], em_iters: usize) -> FellegiSunter {
        // --- u: level frequencies over all candidate pairs ---
        let mut name_u = [0.0f64; NAME_LEVELS];
        let mut head_u = [0.0f64; BINARY_LEVELS];
        let mut type_u = [0.0f64; BINARY_LEVELS];
        let mut n_pairs = 0.0f64;
        for i in 0..records.len() {
            for j in (i + 1)..records.len() {
                let (nl, hl, tl) = levels(&records[i], &records[j]);
                name_u[nl] += 1.0;
                head_u[hl] += 1.0;
                type_u[tl] += 1.0;
                n_pairs += 1.0;
            }
        }
        normalize(&mut name_u, n_pairs);
        normalize(&mut head_u, n_pairs);
        normalize(&mut type_u, n_pairs);

        // --- m: seed favouring agreement (level 0), then EM refine ---
        let mut model = FellegiSunter {
            lambda: 0.1,
            name_m: [0.70, 0.18, 0.09, 0.03],
            name_u,
            head_m: [0.92, 0.08],
            head_u,
            type_m: [0.95, 0.05],
            type_u,
        };
        if n_pairs == 0.0 {
            return model;
        }

        for _ in 0..em_iters {
            let mut name_m_acc = [0.0f64; NAME_LEVELS];
            let mut head_m_acc = [0.0f64; BINARY_LEVELS];
            let mut type_m_acc = [0.0f64; BINARY_LEVELS];
            let mut gamma_sum = 0.0f64;

            // E-step: posterior match probability γ for each pair.
            for i in 0..records.len() {
                for j in (i + 1)..records.len() {
                    let (nl, hl, tl) = levels(&records[i], &records[j]);
                    let gamma = model.match_probability(&records[i], &records[j]);
                    name_m_acc[nl] += gamma;
                    head_m_acc[hl] += gamma;
                    type_m_acc[tl] += gamma;
                    gamma_sum += gamma;
                }
            }

            // M-step: m[level] = Σγ at that level / Σγ; λ = mean γ.
            if gamma_sum > 0.0 {
                for lvl in 0..NAME_LEVELS {
                    model.name_m[lvl] = name_m_acc[lvl] / gamma_sum;
                }
                for lvl in 0..BINARY_LEVELS {
                    model.head_m[lvl] = head_m_acc[lvl] / gamma_sum;
                    model.type_m[lvl] = type_m_acc[lvl] / gamma_sum;
                }
            }
            model.lambda = (gamma_sum / n_pairs).clamp(PROB_FLOOR, 1.0 - PROB_FLOOR);
        }
        model
    }
}

fn normalize(dist: &mut [f64], total: f64) {
    if total > 0.0 {
        for x in dist.iter_mut() {
            *x /= total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(name: &str, head: &str, ty: &str) -> MatchRecord {
        MatchRecord {
            name: name.to_string(),
            head: head.to_string(),
            entity_type: ty.to_string(),
        }
    }

    #[test]
    fn jaro_winkler_basics() {
        assert!((jaro_winkler("dali", "dali") - 1.0).abs() < 1e-9);
        assert_eq!(jaro_winkler("", ""), 1.0);
        assert_eq!(jaro_winkler("abc", ""), 0.0);
        // Winkler prefix boost: shared prefix scores higher than a mid-word match.
        assert!(jaro_winkler("martha", "marhta") > 0.9);
        assert!(jaro_winkler("dali", "dixie") < 0.7);
    }

    #[test]
    fn name_levels_bucket_by_threshold() {
        assert_eq!(name_level("container ship", "container ship"), 0); // identical → ≥0.92
        assert_eq!(name_level("dali", "zzzz"), 3); // no shared chars → below every threshold
                                                   // A near-miss lands in a middle bucket, never 0 and never the last level.
        let mid = name_level("dali", "dixie");
        assert!(
            (1..=2).contains(&mid),
            "near pair should bucket in the middle, got {mid}"
        );
    }

    #[test]
    fn em_separates_matches_from_nonmatches() {
        // Three ship variants (same head+type, similar names) + two distinct
        // entities. A label-free fit must score the variants high and the
        // cross-entity pairs low.
        let records = vec![
            rec("container ship", "ship", "Vessel"),
            rec("cargo ship", "ship", "Vessel"),
            rec("the ship", "ship", "Vessel"),
            rec("key bridge", "bridge", "Structure"),
            rec("baltimore", "baltimore", "Location"),
        ];
        let model = FellegiSunter::fit(&records, 20);

        let variant = model.match_probability(&records[0], &records[1]); // ship~ship
        let cross = model.match_probability(&records[0], &records[3]); // ship~bridge
        assert!(
            variant > 0.5,
            "ship variants should score as a match, got {variant}"
        );
        assert!(
            cross < variant,
            "distinct entities must score below variants ({cross} vs {variant})"
        );
        // Learned weights are inspectable Bayes factors: agreeing on head is
        // evidence for a match (m > u at level 0).
        assert!(
            model.head_m[0] > model.head_u[0],
            "head agreement should carry positive evidence"
        );
    }

    #[test]
    fn match_weight_prior_and_agreement_are_additive() {
        // A fully-agreeing pair must out-weight a fully-disagreeing pair.
        let model = FellegiSunter::fit(
            &[
                rec("apple", "apple", "Org"),
                rec("apple inc", "apple", "Org"),
                rec("nvidia", "nvidia", "Org"),
            ],
            15,
        );
        let agree =
            model.match_weight(&rec("apple", "apple", "Org"), &rec("apple", "apple", "Org"));
        let disagree =
            model.match_weight(&rec("apple", "apple", "Org"), &rec("zzz", "other", "Loc"));
        assert!(
            agree > disagree,
            "agreement weight {agree} must exceed disagreement {disagree}"
        );
    }
}
