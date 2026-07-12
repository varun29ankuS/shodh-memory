//! Fellegi-Sunter probabilistic entity matcher (ER Phase 2.1 + 2.2).
//!
//! A label-free learned matcher for entity mentions — the Splink method ported to
//! Rust with no ML runtime, just m/u probability tables and a log-weight sum.
//! Validated in Python (`demos/gdelt-bridge/splink_explore.py`): bridge F1 ≈ 0.78
//! with zero labels, interpretable Bayes factors.
//!
//! The model is a set of **comparisons**. For a candidate pair each comparison
//! resolves to a discrete agreement *level*; each level carries an `m` probability
//! (P(level | match)) and a `u` probability (P(level | non-match)). The match
//! weight is `w = log2(λ/(1-λ)) + Σ_c log2(m[c][level] / u[c][level])` and the
//! match probability is `2^w / (2^w + 1)`. `u` is estimated from the level
//! frequencies over all pairs (dominated by non-matches); `m` and the prior `λ`
//! are estimated by unsupervised EM.
//!
//! Default comparisons (2.1 → 2.2):
//!   - `name`: Jaro-Winkler at [0.92, 0.75, 0.5] → 4 levels.
//!   - `head`: exact syntactic-head match → 2 levels.
//!   - `type`: exact entity-type match → 2 levels.
//!   - `agent_role`: Jaccard overlap of the predicates the entity participates in,
//!     bucketed → 4 levels. **The learner auto-weights this per domain**: on a
//!     physical corpus where co-actors share roles it carries strong evidence; on
//!     a financial corpus where roles don't track identity, EM drives its weight
//!     toward zero — with no code change (measured in Python: +3.61 → +0.05).
//!   - `causal`: Jaccard overlap of RARE (high-IDF) causal fingerprints → 4 levels
//!     (the Bhattacharya-Getoor lever: a shared rare cause/effect = strong signal).
//!   - `embedding`: bucketed cosine of the name embeddings → 4 levels. The semantic
//!     axis — reaches synonymy character-level `name` misses (`vessel` ≈ `ship`).
//!
//! The comparisons span distinct axes (lexical, syntactic, ontological, behavioral,
//! causal, semantic) on purpose: Fellegi-Sunter assumes conditional independence,
//! so adding *correlated* features double-counts evidence. Beyond these, the next
//! gains are KB entity-linking (world-knowledge merges no feature reaches) and m/u
//! regularization (small-data stability) — not more comparisons.
//!
//! FROZEN structure; the m/u weights are the ONLINE-learned part (re-fittable per
//! domain; streaming-EM update is Phase 4.1). Model-free: consumes pre-extracted
//! records, no encoder/parser at match time.

use std::collections::HashSet;

/// One mention as the matcher sees it. `name` should be caller-normalised
/// (lowercased, determiner-stripped); `head` is the dependency-head lemma;
/// `entity_type` is the NER/GLiNER label; `agent_roles` are the predicate lemmas
/// the entity participates in (parser/edge-derived), used for role-overlap.
#[derive(Debug, Clone, Default)]
pub struct MatchRecord {
    pub name: String,
    pub head: String,
    pub entity_type: String,
    pub agent_roles: HashSet<String>,
    /// Discriminative (rare / high-IDF) causal fingerprints the entity holds,
    /// e.g. `"Struck>bridge"`. The caller pre-filters to rare relations (the
    /// Bhattacharya-Getoor lever); the comparison just measures overlap, so two
    /// mentions that share a rare cause/effect argue for being the same entity.
    pub rare_causal_fps: HashSet<String>,
    /// Pre-computed name embedding (e.g. MiniLM). The semantic axis: it reaches
    /// synonymy that character-level name similarity misses (`vessel` ≈ `ship`,
    /// `the span` ≈ `bridge`). `None` → no evidence, not disagreement.
    pub name_embedding: Option<Vec<f32>>,
}

const NAME_THRESHOLDS: [f64; 3] = [0.92, 0.75, 0.5];
const NAME_LEVELS: usize = 4;
const JACCARD_LEVELS: usize = 4; // ≥0.5, ≥0.2, >0, none
const EMB_THRESHOLDS: [f32; 3] = [0.85, 0.70, 0.55]; // cosine buckets for name embeddings
const EMB_LEVELS: usize = 4;
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

// --- level functions: (record, record) -> level index -----------------------

fn name_level(a: &MatchRecord, b: &MatchRecord) -> usize {
    let s = jaro_winkler(&a.name, &b.name);
    for (lvl, &th) in NAME_THRESHOLDS.iter().enumerate() {
        if s >= th {
            return lvl;
        }
    }
    NAME_THRESHOLDS.len()
}

fn head_level(a: &MatchRecord, b: &MatchRecord) -> usize {
    usize::from(a.head.is_empty() || a.head != b.head)
}

fn type_level(a: &MatchRecord, b: &MatchRecord) -> usize {
    usize::from(a.entity_type.is_empty() || a.entity_type != b.entity_type)
}

/// Bucket the Jaccard overlap of two string sets into 4 agreement levels.
/// Empty on either side → the lowest bucket (no evidence, not disagreement).
fn jaccard_level(a: &HashSet<String>, b: &HashSet<String>) -> usize {
    if a.is_empty() || b.is_empty() {
        return JACCARD_LEVELS - 1;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    let j = if union > 0.0 { inter / union } else { 0.0 };
    if j >= 0.5 {
        0
    } else if j >= 0.2 {
        1
    } else if j > 0.0 {
        2
    } else {
        3
    }
}

fn agent_role_level(a: &MatchRecord, b: &MatchRecord) -> usize {
    jaccard_level(&a.agent_roles, &b.agent_roles)
}

fn causal_level(a: &MatchRecord, b: &MatchRecord) -> usize {
    jaccard_level(&a.rare_causal_fps, &b.rare_causal_fps)
}

/// Cosine similarity of two equal-length vectors; 0.0 for empty/mismatched.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn embedding_level(a: &MatchRecord, b: &MatchRecord) -> usize {
    match (&a.name_embedding, &b.name_embedding) {
        (Some(x), Some(y)) => {
            let c = cosine(x, y);
            for (lvl, &th) in EMB_THRESHOLDS.iter().enumerate() {
                if c >= th {
                    return lvl;
                }
            }
            EMB_THRESHOLDS.len()
        }
        _ => EMB_LEVELS - 1, // no embedding → lowest bucket (no evidence)
    }
}

/// A single comparison: how to bucket a pair, plus the learned m/u per level.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub label: &'static str,
    pub m: Vec<f64>,
    pub u: Vec<f64>,
    level: fn(&MatchRecord, &MatchRecord) -> usize,
}

impl Comparison {
    fn new(
        label: &'static str,
        n_levels: usize,
        level: fn(&MatchRecord, &MatchRecord) -> usize,
    ) -> Self {
        // Seed m favouring agreement (level 0), u uniform; both refined in `fit`.
        let mut m = vec![0.0; n_levels];
        let mut u = vec![1.0 / n_levels as f64; n_levels];
        m[0] = 0.6;
        let rest = 0.4 / (n_levels.saturating_sub(1).max(1)) as f64;
        for (lvl, mv) in m.iter_mut().enumerate() {
            if lvl > 0 {
                *mv = rest;
            }
        }
        u[0] = u[0].max(PROB_FLOOR);
        Comparison { label, m, u, level }
    }

    #[inline]
    fn level_of(&self, a: &MatchRecord, b: &MatchRecord) -> usize {
        (self.level)(a, b)
    }
}

/// The default comparison set (name, head, type, agent_role).
pub fn default_comparisons() -> Vec<Comparison> {
    vec![
        Comparison::new("name", NAME_LEVELS, name_level),
        Comparison::new("head", 2, head_level),
        Comparison::new("type", 2, type_level),
        Comparison::new("agent_role", JACCARD_LEVELS, agent_role_level),
        Comparison::new("causal", JACCARD_LEVELS, causal_level),
        Comparison::new("embedding", EMB_LEVELS, embedding_level),
    ]
}

/// A learned Fellegi-Sunter model: the prior match rate λ and a set of
/// comparisons carrying inspectable m/u Bayes-factor tables.
#[derive(Debug, Clone)]
pub struct FellegiSunter {
    pub lambda: f64,
    pub comparisons: Vec<Comparison>,
}

#[inline]
fn bayes_weight(m: f64, u: f64) -> f64 {
    (m.max(PROB_FLOOR) / u.max(PROB_FLOOR)).log2()
}

impl FellegiSunter {
    /// Total match weight (log2 Bayes factor incl. the prior) for a pair.
    pub fn match_weight(&self, a: &MatchRecord, b: &MatchRecord) -> f64 {
        let prior = (self.lambda.max(PROB_FLOOR) / (1.0 - self.lambda).max(PROB_FLOOR)).log2();
        self.comparisons.iter().fold(prior, |w, c| {
            let lvl = c.level_of(a, b);
            w + bayes_weight(c.m[lvl], c.u[lvl])
        })
    }

    /// Posterior match probability in [0,1].
    pub fn match_probability(&self, a: &MatchRecord, b: &MatchRecord) -> f64 {
        let bf = 2f64.powf(self.match_weight(a, b));
        bf / (bf + 1.0)
    }

    /// The learned Bayes factor (m/u) at a comparison's top agreement level —
    /// the interpretable "how much does agreeing on this feature argue for a
    /// match" number. Returns 1.0 (no evidence) if the comparison is absent.
    pub fn top_bayes_factor(&self, label: &str) -> f64 {
        self.comparisons
            .iter()
            .find(|c| c.label == label)
            .map(|c| c.m[0].max(PROB_FLOOR) / c.u[0].max(PROB_FLOOR))
            .unwrap_or(1.0)
    }

    /// Fit with the default comparison set.
    pub fn fit(records: &[MatchRecord], em_iters: usize) -> FellegiSunter {
        Self::fit_with(records, default_comparisons(), em_iters)
    }

    /// Fit m/u and λ from unlabeled records over an explicit comparison set.
    /// `u` is the level distribution over all pairs; `m` and λ are estimated by
    /// EM, seeded to favour agreement so match / non-match separate.
    pub fn fit_with(
        records: &[MatchRecord],
        mut comparisons: Vec<Comparison>,
        em_iters: usize,
    ) -> FellegiSunter {
        // --- u: level frequencies over all candidate pairs ---
        let mut u_counts: Vec<Vec<f64>> =
            comparisons.iter().map(|c| vec![0.0; c.u.len()]).collect();
        let mut n_pairs = 0.0f64;
        for i in 0..records.len() {
            for j in (i + 1)..records.len() {
                for (ci, c) in comparisons.iter().enumerate() {
                    let lvl = c.level_of(&records[i], &records[j]);
                    u_counts[ci][lvl] += 1.0;
                }
                n_pairs += 1.0;
            }
        }
        for (ci, c) in comparisons.iter_mut().enumerate() {
            normalize(&mut u_counts[ci], n_pairs);
            c.u = u_counts[ci].clone();
        }

        let mut model = FellegiSunter {
            lambda: 0.1,
            comparisons,
        };
        if n_pairs == 0.0 {
            return model;
        }

        for _ in 0..em_iters {
            let mut m_acc: Vec<Vec<f64>> = model
                .comparisons
                .iter()
                .map(|c| vec![0.0; c.m.len()])
                .collect();
            let mut gamma_sum = 0.0f64;

            // E-step: posterior match probability γ per pair.
            for i in 0..records.len() {
                for j in (i + 1)..records.len() {
                    let gamma = model.match_probability(&records[i], &records[j]);
                    for (ci, c) in model.comparisons.iter().enumerate() {
                        let lvl = c.level_of(&records[i], &records[j]);
                        m_acc[ci][lvl] += gamma;
                    }
                    gamma_sum += gamma;
                }
            }

            // M-step: m[level] = Σγ at that level / Σγ; λ = mean γ.
            if gamma_sum > 0.0 {
                for (ci, c) in model.comparisons.iter_mut().enumerate() {
                    for (lvl, mv) in c.m.iter_mut().enumerate() {
                        *mv = m_acc[ci][lvl] / gamma_sum;
                    }
                }
                model.lambda = (gamma_sum / n_pairs).clamp(PROB_FLOOR, 1.0 - PROB_FLOOR);
            }
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

    fn rec(name: &str, head: &str, ty: &str, roles: &[&str]) -> MatchRecord {
        MatchRecord {
            name: name.to_string(),
            head: head.to_string(),
            entity_type: ty.to_string(),
            agent_roles: roles.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn rec_causal(name: &str, head: &str, ty: &str, fps: &[&str]) -> MatchRecord {
        MatchRecord {
            name: name.to_string(),
            head: head.to_string(),
            entity_type: ty.to_string(),
            rare_causal_fps: fps.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn rec_emb(name: &str, head: &str, ty: &str, emb: &[f32]) -> MatchRecord {
        MatchRecord {
            name: name.to_string(),
            head: head.to_string(),
            entity_type: ty.to_string(),
            name_embedding: Some(emb.to_vec()),
            ..Default::default()
        }
    }

    #[test]
    fn jaro_winkler_basics() {
        assert!((jaro_winkler("dali", "dali") - 1.0).abs() < 1e-9);
        assert_eq!(jaro_winkler("", ""), 1.0);
        assert_eq!(jaro_winkler("abc", ""), 0.0);
        assert!(jaro_winkler("martha", "marhta") > 0.9);
        assert!(jaro_winkler("dali", "dixie") < 0.7);
    }

    #[test]
    fn name_levels_bucket_by_threshold() {
        let a = rec("container ship", "ship", "Vessel", &[]);
        let b = rec("container ship", "ship", "Vessel", &[]);
        assert_eq!(name_level(&a, &b), 0);
        let far = rec("zzzz", "x", "x", &[]);
        assert_eq!(name_level(&a, &far), 3);
    }

    #[test]
    fn em_separates_matches_from_nonmatches() {
        let records = vec![
            rec("container ship", "ship", "Vessel", &["struck"]),
            rec("cargo ship", "ship", "Vessel", &["struck"]),
            rec("the ship", "ship", "Vessel", &["struck"]),
            rec("key bridge", "bridge", "Structure", &["collapsed"]),
            rec("baltimore", "baltimore", "Location", &[]),
        ];
        let model = FellegiSunter::fit(&records, 20);
        let variant = model.match_probability(&records[0], &records[1]);
        let cross = model.match_probability(&records[0], &records[3]);
        assert!(variant > 0.5, "ship variants should match, got {variant}");
        assert!(
            cross < variant,
            "distinct entities below variants ({cross} vs {variant})"
        );
    }

    #[test]
    fn match_weight_prior_and_agreement_are_additive() {
        let model = FellegiSunter::fit(
            &[
                rec("apple", "apple", "Org", &["released"]),
                rec("apple inc", "apple", "Org", &["released"]),
                rec("nvidia", "nvidia", "Org", &["announced"]),
            ],
            15,
        );
        let agree = model.match_weight(
            &rec("apple", "apple", "Org", &["released"]),
            &rec("apple", "apple", "Org", &["released"]),
        );
        let disagree = model.match_weight(
            &rec("apple", "apple", "Org", &["released"]),
            &rec("zzz", "other", "Loc", &["x"]),
        );
        assert!(
            agree > disagree,
            "agreement {agree} must exceed disagreement {disagree}"
        );
    }

    #[test]
    fn shared_rare_causal_fingerprint_is_evidence() {
        // The Bhattacharya-Getoor lever: mentions that share a RARE cause/effect
        // are likely the same entity. Three ship mentions all "Struck>bridge";
        // the bridge and river carry unrelated causal fingerprints.
        let records = vec![
            rec_causal("container ship", "ship", "Vessel", &["Struck>bridge"]),
            rec_causal("cargo ship", "ship", "Vessel", &["Struck>bridge"]),
            rec_causal("the vessel", "vessel", "Vessel", &["Struck>bridge"]),
            rec_causal(
                "key bridge",
                "bridge",
                "Structure",
                &["CollapsedInto>river"],
            ),
            rec_causal("patapsco", "river", "Location", &["Received>debris"]),
        ];
        let model = FellegiSunter::fit(&records, 25);
        assert!(
            model.top_bayes_factor("causal") > 1.0,
            "a shared rare causal fingerprint should be positive evidence, got {}",
            model.top_bayes_factor("causal")
        );
        let shares = model.match_probability(&records[0], &records[1]);
        let differs = model.match_probability(&records[0], &records[3]);
        assert!(
            shares > differs,
            "shared-cause pair {shares} should beat unrelated {differs}"
        );
    }

    #[test]
    fn cosine_is_sane() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn embedding_cosine_reaches_synonymy_name_misses() {
        // ship / vessel / boat: low string similarity, different heads, but
        // synonymous (high name-embedding cosine). The semantic axis is what ties
        // them — character-level `name` cannot.
        let records = vec![
            rec_emb("ship", "ship", "Vessel", &[1.0, 0.0, 0.0]),
            rec_emb("vessel", "vessel", "Vessel", &[0.96, 0.12, 0.0]),
            rec_emb("boat", "boat", "Vessel", &[0.9, 0.2, 0.0]),
            rec_emb("bridge", "bridge", "Structure", &[0.0, 0.0, 1.0]),
            rec_emb("river", "river", "Location", &[0.0, 1.0, 0.0]),
        ];
        // Premise: name similarity does NOT tie the synonyms at the top level.
        assert!(
            name_level(&records[0], &records[1]) > 0,
            "ship/vessel are not near-identical strings"
        );

        let model = FellegiSunter::fit(&records, 25);
        assert!(
            model.top_bayes_factor("embedding") > 1.0,
            "high name-embedding cosine should be positive evidence, got {}",
            model.top_bayes_factor("embedding")
        );
        let syn = model.match_probability(&records[0], &records[1]); // ship~vessel
        let unrel = model.match_probability(&records[0], &records[3]); // ship~bridge
        assert!(
            syn > unrel,
            "synonym pair {syn} should beat unrelated {unrel}"
        );
    }

    #[test]
    fn agent_role_weight_auto_adapts_per_domain() {
        // PHYSICAL: co-actors share roles → agent-role tracks identity. Matches
        // (ship variants) share {struck, approached}; the distinct bridge does not.
        let physical = vec![
            rec(
                "container ship",
                "ship",
                "Vessel",
                &["struck", "approached"],
            ),
            rec("cargo ship", "ship", "Vessel", &["struck", "approached"]),
            rec("the vessel", "vessel", "Vessel", &["struck", "approached"]),
            rec("key bridge", "bridge", "Structure", &["collapsed"]),
            rec("patapsco river", "river", "Location", &["flowed"]),
        ];
        // FINANCIAL: roles DON'T track identity. The true match (apple/apple inc)
        // shares no role; an unrelated firm shares apple's role → agent-role is
        // uninformative, so EM must not reward it.
        let financial = vec![
            rec("apple", "apple", "Org", &["released"]),
            rec("apple inc", "apple", "Org", &["sued"]),
            rec("nvidia", "nvidia", "Org", &["released"]),
            rec("microsoft", "microsoft", "Org", &["acquired"]),
            rec("intel", "intel", "Org", &["sued"]),
        ];

        let phys_model = FellegiSunter::fit(&physical, 25);
        let fin_model = FellegiSunter::fit(&financial, 25);

        let phys_role = phys_model.top_bayes_factor("agent_role");
        let fin_role = fin_model.top_bayes_factor("agent_role");

        // The SAME feature is weighted far higher where it is discriminative —
        // the learner auto-adapts per domain with no code change.
        assert!(
            phys_role > fin_role,
            "agent-role should carry more weight on the physical domain \
             (phys bf={phys_role:.2} vs fin bf={fin_role:.2})"
        );
        assert!(
            phys_role > 1.0,
            "agent-role should be positive evidence on the physical domain (bf={phys_role:.2})"
        );
    }
}
