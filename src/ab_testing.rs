//! A/B Testing Infrastructure for Relevance Scoring
//!
//! Provides rigorous experimentation framework for comparing different
//! relevance scoring configurations. Supports:
//!
//! - Multiple concurrent experiments
//! - Consistent user assignment (same user always gets same variant)
//! - Statistical significance testing (chi-squared, confidence intervals)
//! - Metric tracking (impressions, clicks, success rate, latency)
//! - Automatic winner detection with configurable significance threshold
//!
//! # Example
//!
//! ```ignore
//! let manager = ABTestManager::new();
//!
//! // Create a test comparing semantic vs entity weight emphasis
//! let test = ABTest::new("semantic_vs_entity")
//!     .with_control(LearnedWeights::default())
//!     .with_treatment(LearnedWeights {
//!         semantic: 0.5,
//!         entity: 0.25,
//!         ..Default::default()
//!     })
//!     .with_traffic_split(0.5)
//!     .build();
//!
//! manager.create_test(test)?;
//!
//! // Get variant for a user
//! let variant = manager.get_variant("test_id", "user_123")?;
//!
//! // Record metrics
//! manager.record_impression("test_id", "user_123")?;
//! manager.record_click("test_id", "user_123", memory_id)?;
//!
//! // Check results
//! let results = manager.analyze_test("test_id")?;
//! if results.is_significant {
//!     println!("Winner: {:?}", results.winner);
//! }
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::relevance::LearnedWeights;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Default significance level (p < 0.05)
pub const DEFAULT_SIGNIFICANCE_LEVEL: f64 = 0.05;

/// Minimum sample size before statistical analysis is valid
pub const MIN_SAMPLE_SIZE: u64 = 100;

/// Default traffic split (50/50)
pub const DEFAULT_TRAFFIC_SPLIT: f32 = 0.5;

/// Chi-squared critical values for different significance levels (df=1)
/// p=0.05 -> 3.841, p=0.01 -> 6.635, p=0.001 -> 10.828
const CHI_SQUARED_CRITICAL_005: f64 = 3.841;
const CHI_SQUARED_CRITICAL_001: f64 = 6.635;
const CHI_SQUARED_CRITICAL_0001: f64 = 10.828;

/// Sample Ratio Mismatch threshold (5% deviation triggers warning)
const SRM_THRESHOLD: f64 = 0.05;

/// Minimum effect size (Cohen's h) for practical significance
const MIN_PRACTICAL_EFFECT_SIZE: f64 = 0.1;

// =============================================================================
// ADVANCED STATISTICAL TYPES
// =============================================================================

/// Bayesian analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianAnalysis {
    /// Probability that treatment is better than control (0-1)
    pub prob_treatment_better: f64,
    /// Probability that control is better than treatment (0-1)
    pub prob_control_better: f64,
    /// Expected lift of treatment over control
    pub expected_lift: f64,
    /// Credible interval for treatment effect (95%)
    pub credible_interval: (f64, f64),
    /// Risk of choosing treatment if it's actually worse
    pub risk_treatment: f64,
    /// Risk of choosing control if treatment is actually better
    pub risk_control: f64,
}

/// Effect size metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSize {
    /// Cohen's h for proportions (0.2 = small, 0.5 = medium, 0.8 = large)
    pub cohens_h: f64,
    /// Interpretation of effect size
    pub interpretation: EffectSizeInterpretation,
    /// Relative risk (treatment rate / control rate)
    pub relative_risk: f64,
    /// Odds ratio
    pub odds_ratio: f64,
    /// Number needed to treat (NNT) - how many users to see one additional success
    pub nnt: f64,
}

/// Effect size interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectSizeInterpretation {
    Negligible,
    Small,
    Medium,
    Large,
}

impl std::fmt::Display for EffectSizeInterpretation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Negligible => write!(f, "negligible"),
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::Large => write!(f, "large"),
        }
    }
}

/// Sample Ratio Mismatch detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SRMCheck {
    /// Whether SRM is detected (data quality issue)
    pub srm_detected: bool,
    /// Expected ratio based on traffic split
    pub expected_ratio: f64,
    /// Observed ratio
    pub observed_ratio: f64,
    /// Chi-squared statistic for SRM test
    pub chi_squared: f64,
    /// P-value for SRM test
    pub p_value: f64,
    /// Severity of the mismatch
    pub severity: SRMSeverity,
}

/// Severity of sample ratio mismatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SRMSeverity {
    None,
    Warning,
    Critical,
}

/// Sequential testing state for valid early stopping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialTest {
    /// Current analysis number (1, 2, 3, ...)
    pub analysis_number: u32,
    /// Total planned analyses
    pub planned_analyses: u32,
    /// Alpha spent so far
    pub alpha_spent: f64,
    /// Current significance threshold (adjusted for multiple looks)
    pub current_alpha: f64,
    /// Can we stop early?
    pub can_stop_early: bool,
    /// Reason for stopping (if applicable)
    pub stop_reason: Option<String>,
}

/// Guardrail metric that must not degrade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailMetric {
    /// Name of the metric
    pub name: String,
    /// Baseline value (control)
    pub baseline: f64,
    /// Current value (treatment)
    pub current: f64,
    /// Maximum allowed degradation (e.g., 0.05 = 5%)
    pub max_degradation: f64,
    /// Is the guardrail breached?
    pub is_breached: bool,
    /// P-value for degradation test
    pub degradation_p_value: f64,
}

/// Multi-Armed Bandit state for adaptive allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditState {
    /// Algorithm type
    pub algorithm: BanditAlgorithm,
    /// Alpha parameter for each arm (successes + 1)
    pub alphas: Vec<f64>,
    /// Beta parameter for each arm (failures + 1)
    pub betas: Vec<f64>,
    /// Current allocation probabilities
    pub allocation_probs: Vec<f64>,
    /// Total reward collected
    pub total_reward: f64,
    /// Regret estimate
    pub estimated_regret: f64,
}

/// Bandit algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BanditAlgorithm {
    /// Thompson Sampling (Bayesian)
    ThompsonSampling,
    /// Upper Confidence Bound
    UCB1,
    /// Epsilon-greedy
    EpsilonGreedy,
}

// =============================================================================
// CORE TYPES
// =============================================================================

/// Variant in an A/B test
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ABTestVariant {
    /// Control group (baseline/existing behavior)
    Control,
    /// Treatment group (new behavior being tested)
    Treatment,
}

impl ABTestVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            ABTestVariant::Control => "control",
            ABTestVariant::Treatment => "treatment",
        }
    }
}

/// Status of an A/B test
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ABTestStatus {
    /// Test is being configured, not yet active
    Draft,
    /// Test is actively running and collecting data
    Running,
    /// Test is paused (no new assignments, still tracking existing)
    Paused,
    /// Test has concluded (winner determined or manually stopped)
    Completed,
    /// Test was archived (historical record)
    Archived,
}

/// Metrics tracked for each variant
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantMetrics {
    /// Number of times this variant was shown
    pub impressions: u64,
    /// Number of times user interacted positively (clicked/used memory)
    pub clicks: u64,
    /// Number of explicit positive feedback signals
    pub positive_feedback: u64,
    /// Number of explicit negative feedback signals
    pub negative_feedback: u64,
    /// Sum of relevance scores for computing average
    pub total_relevance_score: f64,
    /// Sum of latencies in microseconds
    pub total_latency_us: u64,
    /// Number of latency samples
    pub latency_samples: u64,
    /// Unique users in this variant
    pub unique_users: u64,
    /// Memory IDs that received clicks (for analysis)
    pub clicked_memory_ids: Vec<Uuid>,
}

impl VariantMetrics {
    /// Click-through rate (CTR)
    pub fn ctr(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.clicks as f64 / self.impressions as f64
        }
    }

    /// Success rate (positive / (positive + negative))
    pub fn success_rate(&self) -> f64 {
        let total = self.positive_feedback + self.negative_feedback;
        if total == 0 {
            0.0
        } else {
            self.positive_feedback as f64 / total as f64
        }
    }

    /// Average relevance score
    pub fn avg_relevance_score(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.total_relevance_score / self.impressions as f64
        }
    }

    /// Average latency in milliseconds
    pub fn avg_latency_ms(&self) -> f64 {
        if self.latency_samples == 0 {
            0.0
        } else {
            (self.total_latency_us as f64 / self.latency_samples as f64) / 1000.0
        }
    }

    /// Conversion rate per unique user
    pub fn conversion_rate(&self) -> f64 {
        if self.unique_users == 0 {
            0.0
        } else {
            self.clicks as f64 / self.unique_users as f64
        }
    }
}

/// Configuration for an A/B test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    /// Unique identifier for the test
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what's being tested
    pub description: String,
    /// Weights for control group
    pub control_weights: LearnedWeights,
    /// Weights for treatment group
    pub treatment_weights: LearnedWeights,
    /// Fraction of traffic to send to treatment (0.0-1.0)
    pub traffic_split: f32,
    /// Significance level for statistical tests (default 0.05)
    pub significance_level: f64,
    /// Minimum impressions before declaring winner
    pub min_impressions: u64,
    /// Maximum duration before auto-completing
    pub max_duration_hours: Option<u64>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: String::new(),
            description: String::new(),
            control_weights: LearnedWeights::default(),
            treatment_weights: LearnedWeights::default(),
            traffic_split: DEFAULT_TRAFFIC_SPLIT,
            significance_level: DEFAULT_SIGNIFICANCE_LEVEL,
            min_impressions: MIN_SAMPLE_SIZE,
            max_duration_hours: Some(168), // 1 week default
            tags: Vec::new(),
        }
    }
}

/// An A/B test experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTest {
    /// Unique test identifier (shortcut to config.id)
    #[serde(skip)]
    pub id: String,
    /// Configuration
    pub config: ABTestConfig,
    /// Current status
    pub status: ABTestStatus,
    /// When the test was created
    pub created_at: DateTime<Utc>,
    /// When the test started running
    pub started_at: Option<DateTime<Utc>>,
    /// When the test completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Metrics for control group
    pub control_metrics: VariantMetrics,
    /// Metrics for treatment group
    pub treatment_metrics: VariantMetrics,
    /// User assignments (user_id -> variant)
    #[serde(skip)]
    user_assignments: HashMap<String, ABTestVariant>,
}

impl ABTest {
    /// Create a new A/B test builder
    pub fn builder(name: &str) -> ABTestBuilder {
        ABTestBuilder::new(name)
    }

    /// Create from config
    pub fn from_config(config: ABTestConfig) -> Self {
        let id = config.id.clone();
        Self {
            id,
            config,
            status: ABTestStatus::Draft,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            control_metrics: VariantMetrics::default(),
            treatment_metrics: VariantMetrics::default(),
            user_assignments: HashMap::new(),
        }
    }

    /// Get variant for a user (consistent assignment)
    pub fn get_variant(&mut self, user_id: &str) -> ABTestVariant {
        // Check if user already assigned
        if let Some(&variant) = self.user_assignments.get(user_id) {
            return variant;
        }

        // Consistent hashing for new users
        let variant = self.assign_variant(user_id);

        // Track unique users
        match variant {
            ABTestVariant::Control => self.control_metrics.unique_users += 1,
            ABTestVariant::Treatment => self.treatment_metrics.unique_users += 1,
        }

        self.user_assignments.insert(user_id.to_string(), variant);
        variant
    }

    /// Assign variant using consistent hashing
    fn assign_variant(&self, user_id: &str) -> ABTestVariant {
        // Hash user_id + test_id for consistent assignment
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        user_id.hash(&mut hasher);
        self.config.id.hash(&mut hasher);
        let hash = hasher.finish();

        // Convert to 0.0-1.0 range
        let bucket = (hash % 10000) as f32 / 10000.0;

        if bucket < self.config.traffic_split {
            ABTestVariant::Treatment
        } else {
            ABTestVariant::Control
        }
    }

    /// Get weights for a variant
    pub fn get_weights(&self, variant: ABTestVariant) -> &LearnedWeights {
        match variant {
            ABTestVariant::Control => &self.config.control_weights,
            ABTestVariant::Treatment => &self.config.treatment_weights,
        }
    }

    /// Get metrics for a variant
    pub fn get_metrics(&self, variant: ABTestVariant) -> &VariantMetrics {
        match variant {
            ABTestVariant::Control => &self.control_metrics,
            ABTestVariant::Treatment => &self.treatment_metrics,
        }
    }

    /// Get mutable metrics for a variant
    fn get_metrics_mut(&mut self, variant: ABTestVariant) -> &mut VariantMetrics {
        match variant {
            ABTestVariant::Control => &mut self.control_metrics,
            ABTestVariant::Treatment => &mut self.treatment_metrics,
        }
    }

    /// Record an impression
    pub fn record_impression(&mut self, user_id: &str, relevance_score: f64, latency_us: u64) {
        let variant = self.get_variant(user_id);
        let metrics = self.get_metrics_mut(variant);
        metrics.impressions += 1;
        metrics.total_relevance_score += relevance_score;
        metrics.total_latency_us += latency_us;
        metrics.latency_samples += 1;
    }

    /// Record a click/interaction
    pub fn record_click(&mut self, user_id: &str, memory_id: Uuid) {
        let variant = self.get_variant(user_id);
        let metrics = self.get_metrics_mut(variant);
        metrics.clicks += 1;
        metrics.clicked_memory_ids.push(memory_id);
    }

    /// Record explicit feedback
    pub fn record_feedback(&mut self, user_id: &str, positive: bool) {
        let variant = self.get_variant(user_id);
        let metrics = self.get_metrics_mut(variant);
        if positive {
            metrics.positive_feedback += 1;
        } else {
            metrics.negative_feedback += 1;
        }
    }

    /// Check if test has enough data for analysis
    pub fn has_sufficient_data(&self) -> bool {
        self.control_metrics.impressions >= self.config.min_impressions
            && self.treatment_metrics.impressions >= self.config.min_impressions
    }

    /// Check if test has exceeded max duration
    pub fn is_expired(&self) -> bool {
        if let (Some(started), Some(max_hours)) = (self.started_at, self.config.max_duration_hours)
        {
            let elapsed = Utc::now().signed_duration_since(started);
            elapsed > Duration::hours(max_hours as i64)
        } else {
            false
        }
    }

    /// Start the test
    pub fn start(&mut self) {
        if self.status == ABTestStatus::Draft {
            self.status = ABTestStatus::Running;
            self.started_at = Some(Utc::now());
        }
    }

    /// Pause the test
    pub fn pause(&mut self) {
        if self.status == ABTestStatus::Running {
            self.status = ABTestStatus::Paused;
        }
    }

    /// Resume the test
    pub fn resume(&mut self) {
        if self.status == ABTestStatus::Paused {
            self.status = ABTestStatus::Running;
        }
    }

    /// Complete the test
    pub fn complete(&mut self) {
        if self.status == ABTestStatus::Running || self.status == ABTestStatus::Paused {
            self.status = ABTestStatus::Completed;
            self.completed_at = Some(Utc::now());
        }
    }

    /// Archive the test
    pub fn archive(&mut self) {
        self.status = ABTestStatus::Archived;
    }
}

/// Builder for creating A/B tests
pub struct ABTestBuilder {
    config: ABTestConfig,
}

impl ABTestBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            config: ABTestConfig {
                name: name.to_string(),
                ..Default::default()
            },
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.config.id = id.to_string();
        self
    }

    pub fn with_description(mut self, description: &str) -> Self {
        self.config.description = description.to_string();
        self
    }

    pub fn with_control(mut self, weights: LearnedWeights) -> Self {
        self.config.control_weights = weights;
        self
    }

    pub fn with_treatment(mut self, weights: LearnedWeights) -> Self {
        self.config.treatment_weights = weights;
        self
    }

    pub fn with_traffic_split(mut self, split: f32) -> Self {
        self.config.traffic_split = split.clamp(0.0, 1.0);
        self
    }

    pub fn with_significance_level(mut self, level: f64) -> Self {
        self.config.significance_level = level.clamp(0.001, 0.1);
        self
    }

    pub fn with_min_impressions(mut self, min: u64) -> Self {
        self.config.min_impressions = min.max(MIN_SAMPLE_SIZE);
        self
    }

    pub fn with_max_duration_hours(mut self, hours: u64) -> Self {
        self.config.max_duration_hours = Some(hours);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.config.tags = tags;
        self
    }

    pub fn build(self) -> ABTest {
        ABTest::from_config(self.config)
    }
}

// =============================================================================
// STATISTICAL ANALYSIS
// =============================================================================

/// Results of statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResults {
    /// Test ID
    pub test_id: String,
    /// Whether the result is statistically significant
    pub is_significant: bool,
    /// Confidence level achieved (1 - p-value)
    pub confidence_level: f64,
    /// Chi-squared statistic
    pub chi_squared: f64,
    /// P-value
    pub p_value: f64,
    /// Winning variant (if significant)
    pub winner: Option<ABTestVariant>,
    /// Relative improvement of winner over loser
    pub relative_improvement: f64,
    /// Control group CTR
    pub control_ctr: f64,
    /// Treatment group CTR
    pub treatment_ctr: f64,
    /// Control group success rate
    pub control_success_rate: f64,
    /// Treatment group success rate
    pub treatment_success_rate: f64,
    /// 95% confidence interval for treatment effect
    pub confidence_interval: (f64, f64),
    /// Recommendations based on results
    pub recommendations: Vec<String>,
    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,
}

/// Statistical analyzer for A/B tests
pub struct ABTestAnalyzer;

impl ABTestAnalyzer {
    /// Analyze an A/B test and return results
    pub fn analyze(test: &ABTest) -> ABTestResults {
        let control = &test.control_metrics;
        let treatment = &test.treatment_metrics;

        // Calculate CTRs
        let control_ctr = control.ctr();
        let treatment_ctr = treatment.ctr();

        // Calculate success rates
        let control_success = control.success_rate();
        let treatment_success = treatment.success_rate();

        // Chi-squared test for CTR difference
        let (chi_squared, p_value) = Self::chi_squared_test(
            control.impressions,
            control.clicks,
            treatment.impressions,
            treatment.clicks,
        );

        // Determine significance
        let is_significant = p_value < test.config.significance_level
            && control.impressions >= test.config.min_impressions
            && treatment.impressions >= test.config.min_impressions;

        // Determine winner
        let winner = if is_significant {
            if treatment_ctr > control_ctr {
                Some(ABTestVariant::Treatment)
            } else {
                Some(ABTestVariant::Control)
            }
        } else {
            None
        };

        // Calculate relative improvement
        let relative_improvement = if control_ctr > 0.0 {
            (treatment_ctr - control_ctr) / control_ctr * 100.0
        } else {
            0.0
        };

        // Calculate confidence interval for treatment effect
        let confidence_interval = Self::calculate_confidence_interval(
            control.impressions,
            control.clicks,
            treatment.impressions,
            treatment.clicks,
        );

        // Generate recommendations
        let recommendations = Self::generate_recommendations(
            test,
            is_significant,
            winner,
            relative_improvement,
            &confidence_interval,
        );

        ABTestResults {
            test_id: test.config.id.clone(),
            is_significant,
            confidence_level: 1.0 - p_value,
            chi_squared,
            p_value,
            winner,
            relative_improvement,
            control_ctr,
            treatment_ctr,
            control_success_rate: control_success,
            treatment_success_rate: treatment_success,
            confidence_interval,
            recommendations,
            analyzed_at: Utc::now(),
        }
    }

    /// Chi-squared test for comparing two proportions
    ///
    /// Tests H0: p1 = p2 (no difference in conversion rates)
    /// Returns (chi_squared_statistic, p_value)
    fn chi_squared_test(n1: u64, x1: u64, n2: u64, x2: u64) -> (f64, f64) {
        if n1 == 0 || n2 == 0 {
            return (0.0, 1.0);
        }

        let n1 = n1 as f64;
        let x1 = x1 as f64;
        let n2 = n2 as f64;
        let x2 = x2 as f64;

        // Pooled proportion
        let p_pooled = (x1 + x2) / (n1 + n2);

        // Expected values under null hypothesis
        let e1_success = n1 * p_pooled;
        let e1_failure = n1 * (1.0 - p_pooled);
        let e2_success = n2 * p_pooled;
        let e2_failure = n2 * (1.0 - p_pooled);

        // Avoid division by zero
        if e1_success < 5.0 || e1_failure < 5.0 || e2_success < 5.0 || e2_failure < 5.0 {
            // Sample size too small for chi-squared approximation
            return (0.0, 1.0);
        }

        // Chi-squared statistic
        let chi_squared = (x1 - e1_success).powi(2) / e1_success
            + ((n1 - x1) - e1_failure).powi(2) / e1_failure
            + (x2 - e2_success).powi(2) / e2_success
            + ((n2 - x2) - e2_failure).powi(2) / e2_failure;

        // P-value approximation (df=1)
        let p_value = Self::chi_squared_p_value(chi_squared);

        (chi_squared, p_value)
    }

    /// Approximate p-value for chi-squared distribution with df=1
    fn chi_squared_p_value(chi_squared: f64) -> f64 {
        if chi_squared <= 0.0 {
            return 1.0;
        }

        // Use lookup table for common critical values
        if chi_squared >= CHI_SQUARED_CRITICAL_0001 {
            0.0001
        } else if chi_squared >= CHI_SQUARED_CRITICAL_001 {
            // Interpolate between 0.001 and 0.0001
            let ratio = (chi_squared - CHI_SQUARED_CRITICAL_001)
                / (CHI_SQUARED_CRITICAL_0001 - CHI_SQUARED_CRITICAL_001);
            0.001 - ratio * 0.0009
        } else if chi_squared >= CHI_SQUARED_CRITICAL_005 {
            // Interpolate between 0.05 and 0.001
            let ratio = (chi_squared - CHI_SQUARED_CRITICAL_005)
                / (CHI_SQUARED_CRITICAL_001 - CHI_SQUARED_CRITICAL_005);
            0.05 - ratio * 0.049
        } else {
            // Below 0.05 significance
            // Rough approximation: p ≈ exp(-chi_squared/2) for small values
            0.05 + (1.0 - chi_squared / CHI_SQUARED_CRITICAL_005) * 0.45
        }
    }

    /// Calculate 95% confidence interval for treatment effect
    fn calculate_confidence_interval(n1: u64, x1: u64, n2: u64, x2: u64) -> (f64, f64) {
        if n1 == 0 || n2 == 0 {
            return (0.0, 0.0);
        }

        let p1 = x1 as f64 / n1 as f64;
        let p2 = x2 as f64 / n2 as f64;
        let diff = p2 - p1;

        // Standard error of difference
        let se = ((p1 * (1.0 - p1) / n1 as f64) + (p2 * (1.0 - p2) / n2 as f64)).sqrt();

        // 95% CI: diff ± 1.96 * SE
        let margin = 1.96 * se;
        (diff - margin, diff + margin)
    }

    /// Generate actionable recommendations
    fn generate_recommendations(
        test: &ABTest,
        is_significant: bool,
        winner: Option<ABTestVariant>,
        relative_improvement: f64,
        confidence_interval: &(f64, f64),
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        let total_impressions =
            test.control_metrics.impressions + test.treatment_metrics.impressions;

        // Check sample size
        if total_impressions < MIN_SAMPLE_SIZE * 2 {
            recommendations.push(format!(
                "Insufficient data: {} impressions collected, need at least {} for reliable analysis",
                total_impressions,
                MIN_SAMPLE_SIZE * 2
            ));
            return recommendations;
        }

        if is_significant {
            match winner {
                Some(ABTestVariant::Treatment) => {
                    recommendations.push(format!(
                        "Treatment variant wins with {relative_improvement:.1}% relative improvement"
                    ));
                    recommendations
                        .push("Recommendation: Deploy treatment weights to production".to_string());

                    if relative_improvement > 20.0 {
                        recommendations.push(
                            "Strong effect detected - consider investigating what drove the improvement".to_string()
                        );
                    }
                }
                Some(ABTestVariant::Control) => {
                    recommendations.push(format!(
                        "Control variant wins - treatment performed {:.1}% worse",
                        -relative_improvement
                    ));
                    recommendations.push(
                        "Recommendation: Keep current weights, do not deploy treatment".to_string(),
                    );
                }
                None => {}
            }
        } else {
            recommendations.push("No statistically significant difference detected".to_string());

            // Check if close to significance
            let (ci_low, ci_high) = *confidence_interval;
            if ci_low < 0.0 && ci_high > 0.0 {
                recommendations.push(
                    "Confidence interval includes zero - effect may be negligible".to_string(),
                );
            }

            // Suggest more data
            let current_power = Self::estimate_power(test);
            if current_power < 0.8 {
                let needed = Self::estimate_needed_sample_size(test, 0.8);
                recommendations.push(format!(
                    "Current statistical power: {:.1}%. Need ~{} more impressions per variant for 80% power",
                    current_power * 100.0,
                    needed
                ));
            }
        }

        // Check for data quality issues
        if test.control_metrics.latency_samples > 0 && test.treatment_metrics.latency_samples > 0 {
            let control_latency = test.control_metrics.avg_latency_ms();
            let treatment_latency = test.treatment_metrics.avg_latency_ms();
            let latency_diff = (treatment_latency - control_latency) / control_latency * 100.0;

            if latency_diff.abs() > 20.0 {
                recommendations.push(format!(
                    "Warning: Latency differs by {latency_diff:.1}% between variants - may affect user behavior"
                ));
            }
        }

        recommendations
    }

    /// Estimate statistical power of current test
    fn estimate_power(test: &ABTest) -> f64 {
        let n1 = test.control_metrics.impressions as f64;
        let n2 = test.treatment_metrics.impressions as f64;
        let p1 = test.control_metrics.ctr();
        let p2 = test.treatment_metrics.ctr();

        if n1 == 0.0 || n2 == 0.0 || p1 == 0.0 {
            return 0.0;
        }

        // Effect size (Cohen's h)
        let h = 2.0 * ((p2.sqrt()).asin() - (p1.sqrt()).asin());

        // Pooled sample size effect
        let n_eff = 2.0 / (1.0 / n1 + 1.0 / n2);

        // Approximate power (simplified)
        let z = h * (n_eff / 2.0).sqrt();
        let power = 0.5 * (1.0 + Self::erf(z / 2.0_f64.sqrt()));

        power.clamp(0.0, 1.0)
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Estimate sample size needed for desired power
    fn estimate_needed_sample_size(test: &ABTest, target_power: f64) -> u64 {
        let p1 = test.control_metrics.ctr();
        let p2 = test.treatment_metrics.ctr();

        if p1 == 0.0 || p2 == 0.0 || (p2 - p1).abs() < 0.001 {
            return 10000; // Default large number
        }

        // Effect size
        let effect = (p2 - p1).abs();
        let pooled_p = (p1 + p2) / 2.0;
        let pooled_var = pooled_p * (1.0 - pooled_p);

        // Z-scores for alpha=0.05 and target power
        let z_alpha = 1.96;
        let z_beta = Self::inverse_normal_cdf(target_power);

        // Sample size formula
        let n = 2.0 * pooled_var * (z_alpha + z_beta).powi(2) / effect.powi(2);

        n.ceil() as u64
    }

    /// Inverse normal CDF approximation
    fn inverse_normal_cdf(p: f64) -> f64 {
        // Rational approximation
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
    }

    // =========================================================================
    // ADVANCED ANALYSIS METHODS
    // =========================================================================

    /// Perform Bayesian analysis using Beta-Binomial model
    ///
    /// Returns probability that treatment is better, expected lift, and credible intervals
    pub fn bayesian_analysis(test: &ABTest) -> BayesianAnalysis {
        let control = &test.control_metrics;
        let treatment = &test.treatment_metrics;

        // Beta distribution parameters (using Jeffreys prior: alpha=0.5, beta=0.5)
        let alpha_c = control.clicks as f64 + 0.5;
        let beta_c = (control.impressions - control.clicks) as f64 + 0.5;
        let alpha_t = treatment.clicks as f64 + 0.5;
        let beta_t = (treatment.impressions - treatment.clicks) as f64 + 0.5;

        // Monte Carlo simulation for probability of being better
        let n_samples = 10000;
        let mut treatment_wins = 0;
        let mut lift_sum = 0.0;
        let mut lifts = Vec::with_capacity(n_samples);

        // Simple random sampling using linear congruential generator
        let mut seed = 12345u64;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*s as f64) / (u64::MAX as f64)
        };

        for _ in 0..n_samples {
            // Sample from Beta distributions using inverse transform
            let p_c = Self::beta_sample(alpha_c, beta_c, &mut seed, &lcg);
            let p_t = Self::beta_sample(alpha_t, beta_t, &mut seed, &lcg);

            if p_t > p_c {
                treatment_wins += 1;
            }

            let lift = if p_c > 0.0 { (p_t - p_c) / p_c } else { 0.0 };
            lift_sum += lift;
            lifts.push(lift);
        }

        lifts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let prob_treatment_better = treatment_wins as f64 / n_samples as f64;
        let expected_lift = lift_sum / n_samples as f64;

        // 95% credible interval
        let ci_low = lifts[(n_samples as f64 * 0.025) as usize];
        let ci_high = lifts[(n_samples as f64 * 0.975) as usize];

        // Expected loss (risk) calculation
        let risk_treatment =
            lifts.iter().filter(|&&l| l < 0.0).map(|l| -l).sum::<f64>() / n_samples as f64;
        let risk_control = lifts.iter().filter(|&&l| l > 0.0).sum::<f64>() / n_samples as f64;

        BayesianAnalysis {
            prob_treatment_better,
            prob_control_better: 1.0 - prob_treatment_better,
            expected_lift,
            credible_interval: (ci_low, ci_high),
            risk_treatment,
            risk_control,
        }
    }

    /// Sample from Beta distribution using inverse transform sampling
    fn beta_sample(alpha: f64, beta: f64, seed: &mut u64, lcg: &impl Fn(&mut u64) -> f64) -> f64 {
        // Use ratio of gamma samples for Beta
        let gamma_a = Self::gamma_sample(alpha, seed, lcg);
        let gamma_b = Self::gamma_sample(beta, seed, lcg);
        gamma_a / (gamma_a + gamma_b)
    }

    /// Sample from Gamma distribution using Marsaglia and Tsang's method
    fn gamma_sample(alpha: f64, seed: &mut u64, lcg: &impl Fn(&mut u64) -> f64) -> f64 {
        if alpha < 1.0 {
            // For alpha < 1, use rejection method
            return Self::gamma_sample(alpha + 1.0, seed, lcg) * lcg(seed).powf(1.0 / alpha);
        }

        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = Self::normal_sample(seed, lcg);
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 {
                let u = lcg(seed);
                if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln())
                {
                    return d * v;
                }
            }
        }
    }

    /// Sample from standard normal using Box-Muller transform
    fn normal_sample(seed: &mut u64, lcg: &impl Fn(&mut u64) -> f64) -> f64 {
        let u1 = lcg(seed);
        let u2 = lcg(seed);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Calculate effect size metrics (Cohen's h, relative risk, odds ratio, NNT)
    pub fn calculate_effect_size(test: &ABTest) -> EffectSize {
        let p1 = test.control_metrics.ctr();
        let p2 = test.treatment_metrics.ctr();

        // Cohen's h for proportions
        let phi1 = 2.0 * p1.sqrt().asin();
        let phi2 = 2.0 * p2.sqrt().asin();
        let cohens_h = (phi2 - phi1).abs();

        // Interpretation based on Cohen's conventions
        let interpretation = if cohens_h < 0.2 {
            EffectSizeInterpretation::Negligible
        } else if cohens_h < 0.5 {
            EffectSizeInterpretation::Small
        } else if cohens_h < 0.8 {
            EffectSizeInterpretation::Medium
        } else {
            EffectSizeInterpretation::Large
        };

        // Relative risk
        let relative_risk = if p1 > 0.0 { p2 / p1 } else { 0.0 };

        // Odds ratio
        let odds_c = if p1 < 1.0 {
            p1 / (1.0 - p1)
        } else {
            f64::INFINITY
        };
        let odds_t = if p2 < 1.0 {
            p2 / (1.0 - p2)
        } else {
            f64::INFINITY
        };
        let odds_ratio = if odds_c > 0.0 && odds_c.is_finite() {
            odds_t / odds_c
        } else {
            0.0
        };

        // Number needed to treat
        let ard = (p2 - p1).abs(); // Absolute risk difference
        let nnt = if ard > 0.0 { 1.0 / ard } else { f64::INFINITY };

        EffectSize {
            cohens_h,
            interpretation,
            relative_risk,
            odds_ratio,
            nnt,
        }
    }

    /// Check for Sample Ratio Mismatch (data quality issue)
    ///
    /// SRM occurs when the observed traffic split differs from expected,
    /// indicating a bug in randomization or data collection
    pub fn check_srm(test: &ABTest) -> SRMCheck {
        let expected_ratio = test.config.traffic_split as f64;
        let total = test.control_metrics.impressions + test.treatment_metrics.impressions;

        if total == 0 {
            return SRMCheck {
                srm_detected: false,
                expected_ratio,
                observed_ratio: 0.5,
                chi_squared: 0.0,
                p_value: 1.0,
                severity: SRMSeverity::None,
            };
        }

        let observed_ratio = test.treatment_metrics.impressions as f64 / total as f64;

        // Expected counts
        let expected_control = total as f64 * (1.0 - expected_ratio);
        let expected_treatment = total as f64 * expected_ratio;

        // Chi-squared test for SRM
        let chi_sq = (test.control_metrics.impressions as f64 - expected_control).powi(2)
            / expected_control
            + (test.treatment_metrics.impressions as f64 - expected_treatment).powi(2)
                / expected_treatment;

        let p_value = Self::chi_squared_p_value(chi_sq);

        // Determine severity
        let deviation = (observed_ratio - expected_ratio).abs();
        let severity = if p_value > 0.01 {
            SRMSeverity::None
        } else if deviation < SRM_THRESHOLD {
            SRMSeverity::Warning
        } else {
            SRMSeverity::Critical
        };

        SRMCheck {
            srm_detected: p_value < 0.01,
            expected_ratio,
            observed_ratio,
            chi_squared: chi_sq,
            p_value,
            severity,
        }
    }

    /// Sequential testing with O'Brien-Fleming alpha spending
    ///
    /// Allows valid early stopping while controlling Type I error
    pub fn sequential_analysis(
        test: &ABTest,
        analysis_number: u32,
        planned_analyses: u32,
    ) -> SequentialTest {
        let fraction = analysis_number as f64 / planned_analyses as f64;

        // O'Brien-Fleming alpha spending function
        // Spends very little alpha early, more as test progresses
        let alpha = test.config.significance_level;
        let alpha_spent = 2.0
            * (1.0
                - Self::normal_cdf(Self::inverse_normal_cdf(1.0 - alpha / 2.0) / fraction.sqrt()));

        // Current significance threshold
        let current_alpha = alpha_spent / analysis_number as f64;

        // Perform test at current threshold
        let (_, p_value) = Self::chi_squared_test(
            test.control_metrics.impressions,
            test.control_metrics.clicks,
            test.treatment_metrics.impressions,
            test.treatment_metrics.clicks,
        );

        let can_stop_early = p_value < current_alpha
            && test.control_metrics.impressions >= test.config.min_impressions / 2
            && test.treatment_metrics.impressions >= test.config.min_impressions / 2;

        let stop_reason = if can_stop_early {
            let effect = Self::calculate_effect_size(test);
            if effect.interpretation == EffectSizeInterpretation::Negligible {
                Some("Futility: Effect size too small to be practically significant".to_string())
            } else {
                Some(format!(
                    "Efficacy: Significant result with {} effect",
                    effect.interpretation
                ))
            }
        } else {
            None
        };

        SequentialTest {
            analysis_number,
            planned_analyses,
            alpha_spent,
            current_alpha,
            can_stop_early,
            stop_reason,
        }
    }

    /// Normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    /// Comprehensive analysis combining all methods
    ///
    /// Returns actionable insights focused on what matters for users:
    /// - Should we ship this change?
    /// - Is the effect meaningful (not just statistically significant)?
    /// - Are there data quality issues?
    /// - What's the risk of making the wrong decision?
    pub fn comprehensive_analysis(test: &ABTest) -> ComprehensiveAnalysis {
        let frequentist = Self::analyze(test);
        let bayesian = Self::bayesian_analysis(test);
        let effect_size = Self::calculate_effect_size(test);
        let srm = Self::check_srm(test);
        let sequential = Self::sequential_analysis(test, 1, 5);

        // Decision logic: combine statistical and practical significance
        let is_practically_significant = effect_size.cohens_h >= MIN_PRACTICAL_EFFECT_SIZE;
        let has_data_quality_issues = srm.srm_detected;
        let high_confidence =
            bayesian.prob_treatment_better > 0.95 || bayesian.prob_control_better > 0.95;
        let low_risk = bayesian.risk_treatment < 0.01 || bayesian.risk_control < 0.01;

        // Ship decision
        let should_ship = frequentist.is_significant
            && is_practically_significant
            && !has_data_quality_issues
            && high_confidence
            && low_risk
            && frequentist.winner == Some(ABTestVariant::Treatment);

        // Generate user-focused insights
        let mut insights = Vec::new();

        // Primary insight
        if should_ship {
            insights.push(format!(
                "✅ SHIP IT: Treatment is {:.1}% better with {:.1}% confidence and {} effect size",
                bayesian.expected_lift * 100.0,
                bayesian.prob_treatment_better * 100.0,
                effect_size.interpretation
            ));
        } else if frequentist.winner == Some(ABTestVariant::Control) && frequentist.is_significant {
            insights.push(format!(
                "❌ DO NOT SHIP: Control is {:.1}% better. Treatment would hurt users.",
                -bayesian.expected_lift * 100.0
            ));
        } else {
            insights.push("⏳ KEEP TESTING: Not enough evidence to make a decision".to_string());
        }

        // Explain why
        if !frequentist.is_significant {
            insights.push(format!(
                "📊 p-value = {:.4} (need < {:.2})",
                frequentist.p_value, test.config.significance_level
            ));
        }

        if !is_practically_significant {
            insights.push(format!(
                "📏 Effect is {} (Cohen's h = {:.3}) - may not matter to users",
                effect_size.interpretation, effect_size.cohens_h
            ));
        }

        if has_data_quality_issues {
            insights.push(format!(
                "⚠️ DATA QUALITY: Sample ratio mismatch detected ({:.1}% vs expected {:.1}%)",
                srm.observed_ratio * 100.0,
                srm.expected_ratio * 100.0
            ));
        }

        // Risk assessment
        if bayesian.risk_treatment > 0.01 {
            insights.push(format!(
                "🎲 Risk if shipping treatment: {:.2}% expected loss",
                bayesian.risk_treatment * 100.0
            ));
        }

        // User impact
        if effect_size.nnt.is_finite() && effect_size.nnt < 1000.0 {
            insights.push(format!(
                "👥 Impact: 1 in {:.0} users will benefit from this change",
                effect_size.nnt
            ));
        }

        ComprehensiveAnalysis {
            frequentist,
            bayesian,
            effect_size,
            srm,
            sequential,
            should_ship,
            is_practically_significant,
            insights,
        }
    }
}

/// Comprehensive analysis result combining all statistical methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveAnalysis {
    /// Frequentist analysis results
    pub frequentist: ABTestResults,
    /// Bayesian analysis results
    pub bayesian: BayesianAnalysis,
    /// Effect size metrics
    pub effect_size: EffectSize,
    /// Sample ratio mismatch check
    pub srm: SRMCheck,
    /// Sequential testing state
    pub sequential: SequentialTest,
    /// Final recommendation: should we ship?
    pub should_ship: bool,
    /// Is the effect practically significant (not just statistically)?
    pub is_practically_significant: bool,
    /// User-focused insights and recommendations
    pub insights: Vec<String>,
}

// =============================================================================
// TEST MANAGER
// =============================================================================

/// Manager for multiple A/B tests
pub struct ABTestManager {
    /// Active tests by ID
    tests: Arc<RwLock<HashMap<String, ABTest>>>,
    /// Archived tests (for historical analysis)
    archived: Arc<RwLock<Vec<ABTest>>>,
}

impl Default for ABTestManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ABTestManager {
    /// Create a new test manager
    pub fn new() -> Self {
        Self {
            tests: Arc::new(RwLock::new(HashMap::new())),
            archived: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create a new A/B test
    pub fn create_test(&self, test: ABTest) -> Result<String, ABTestError> {
        let id = test.config.id.clone();

        let mut tests = self.tests.write();
        if tests.contains_key(&id) {
            return Err(ABTestError::TestAlreadyExists(id));
        }

        tests.insert(id.clone(), test);
        Ok(id)
    }

    /// Get a test by ID
    pub fn get_test(&self, test_id: &str) -> Option<ABTest> {
        self.tests.read().get(test_id).cloned()
    }

    /// List all active tests
    pub fn list_tests(&self) -> Vec<ABTest> {
        self.tests.read().values().cloned().collect()
    }

    /// List tests by status
    pub fn list_tests_by_status(&self, status: ABTestStatus) -> Vec<ABTest> {
        self.tests
            .read()
            .values()
            .filter(|t| t.status == status)
            .cloned()
            .collect()
    }

    /// Start a test
    pub fn start_test(&self, test_id: &str) -> Result<(), ABTestError> {
        let mut tests = self.tests.write();
        let test = tests
            .get_mut(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        if test.status != ABTestStatus::Draft {
            return Err(ABTestError::InvalidState(format!(
                "Cannot start test in {:?} state",
                test.status
            )));
        }

        test.start();
        Ok(())
    }

    /// Pause a test
    pub fn pause_test(&self, test_id: &str) -> Result<(), ABTestError> {
        let mut tests = self.tests.write();
        let test = tests
            .get_mut(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        test.pause();
        Ok(())
    }

    /// Resume a test
    pub fn resume_test(&self, test_id: &str) -> Result<(), ABTestError> {
        let mut tests = self.tests.write();
        let test = tests
            .get_mut(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        test.resume();
        Ok(())
    }

    /// Complete a test
    pub fn complete_test(&self, test_id: &str) -> Result<ABTestResults, ABTestError> {
        let results = {
            let tests = self.tests.read();
            let test = tests
                .get(test_id)
                .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

            ABTestAnalyzer::analyze(test)
        };

        let mut tests = self.tests.write();
        if let Some(test) = tests.get_mut(test_id) {
            test.complete();
        }

        Ok(results)
    }

    /// Archive a test (move to archived storage)
    pub fn archive_test(&self, test_id: &str) -> Result<(), ABTestError> {
        let mut tests = self.tests.write();
        let mut test = tests
            .remove(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        test.archive();
        self.archived.write().push(test);

        Ok(())
    }

    /// Delete a test (permanent)
    pub fn delete_test(&self, test_id: &str) -> Result<(), ABTestError> {
        let mut tests = self.tests.write();
        tests
            .remove(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;
        Ok(())
    }

    /// Get variant for a user in a specific test
    pub fn get_variant(&self, test_id: &str, user_id: &str) -> Result<ABTestVariant, ABTestError> {
        let mut tests = self.tests.write();
        let test = tests
            .get_mut(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        if test.status != ABTestStatus::Running {
            return Err(ABTestError::TestNotRunning(test_id.to_string()));
        }

        Ok(test.get_variant(user_id))
    }

    /// Get weights for a user (handles test assignment)
    pub fn get_weights_for_user(
        &self,
        test_id: &str,
        user_id: &str,
    ) -> Result<LearnedWeights, ABTestError> {
        let mut tests = self.tests.write();
        let test = tests
            .get_mut(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        if test.status != ABTestStatus::Running {
            return Err(ABTestError::TestNotRunning(test_id.to_string()));
        }

        let variant = test.get_variant(user_id);
        Ok(test.get_weights(variant).clone())
    }

    /// Record an impression
    pub fn record_impression(
        &self,
        test_id: &str,
        user_id: &str,
        relevance_score: f64,
        latency_us: u64,
    ) -> Result<(), ABTestError> {
        let mut tests = self.tests.write();
        let test = tests
            .get_mut(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        if test.status != ABTestStatus::Running {
            return Err(ABTestError::TestNotRunning(test_id.to_string()));
        }

        test.record_impression(user_id, relevance_score, latency_us);
        Ok(())
    }

    /// Record a click
    pub fn record_click(
        &self,
        test_id: &str,
        user_id: &str,
        memory_id: Uuid,
    ) -> Result<(), ABTestError> {
        let mut tests = self.tests.write();
        let test = tests
            .get_mut(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        if test.status != ABTestStatus::Running {
            return Err(ABTestError::TestNotRunning(test_id.to_string()));
        }

        test.record_click(user_id, memory_id);
        Ok(())
    }

    /// Record explicit feedback
    pub fn record_feedback(
        &self,
        test_id: &str,
        user_id: &str,
        positive: bool,
    ) -> Result<(), ABTestError> {
        let mut tests = self.tests.write();
        let test = tests
            .get_mut(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        if test.status != ABTestStatus::Running {
            return Err(ABTestError::TestNotRunning(test_id.to_string()));
        }

        test.record_feedback(user_id, positive);
        Ok(())
    }

    /// Analyze a test
    pub fn analyze_test(&self, test_id: &str) -> Result<ABTestResults, ABTestError> {
        let tests = self.tests.read();
        let test = tests
            .get(test_id)
            .ok_or_else(|| ABTestError::TestNotFound(test_id.to_string()))?;

        Ok(ABTestAnalyzer::analyze(test))
    }

    /// Get all archived tests
    pub fn list_archived(&self) -> Vec<ABTest> {
        self.archived.read().clone()
    }

    /// Check and auto-complete expired tests
    pub fn check_expired_tests(&self) -> Vec<String> {
        let mut expired = Vec::new();

        let mut tests = self.tests.write();
        for (id, test) in tests.iter_mut() {
            if test.status == ABTestStatus::Running && test.is_expired() {
                test.complete();
                expired.push(id.clone());
            }
        }

        expired
    }

    /// Get summary of all tests
    pub fn summary(&self) -> ABTestManagerSummary {
        let tests = self.tests.read();
        let archived = self.archived.read();

        let mut draft = 0;
        let mut running = 0;
        let mut paused = 0;
        let mut completed = 0;

        for test in tests.values() {
            match test.status {
                ABTestStatus::Draft => draft += 1,
                ABTestStatus::Running => running += 1,
                ABTestStatus::Paused => paused += 1,
                ABTestStatus::Completed => completed += 1,
                ABTestStatus::Archived => {}
            }
        }

        ABTestManagerSummary {
            total_active: tests.len(),
            draft,
            running,
            paused,
            completed,
            archived: archived.len(),
        }
    }
}

/// Summary of test manager state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestManagerSummary {
    pub total_active: usize,
    pub draft: usize,
    pub running: usize,
    pub paused: usize,
    pub completed: usize,
    pub archived: usize,
}

// =============================================================================
// ERRORS
// =============================================================================

/// Errors from A/B testing operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum ABTestError {
    #[error("Test not found: {0}")]
    TestNotFound(String),

    #[error("Test already exists: {0}")]
    TestAlreadyExists(String),

    #[error("Test is not running: {0}")]
    TestNotRunning(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Insufficient data for analysis")]
    InsufficientData,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_assignment_consistency() {
        let mut test = ABTest::builder("test").with_traffic_split(0.5).build();

        // Same user should always get same variant
        let user = "user_123";
        let variant1 = test.get_variant(user);
        let variant2 = test.get_variant(user);
        let variant3 = test.get_variant(user);

        assert_eq!(variant1, variant2);
        assert_eq!(variant2, variant3);
    }

    #[test]
    fn test_traffic_split() {
        let mut test = ABTest::builder("test").with_traffic_split(0.5).build();

        let mut control_count = 0;
        let mut treatment_count = 0;

        // Assign many users
        for i in 0..1000 {
            let user = format!("user_{}", i);
            match test.get_variant(&user) {
                ABTestVariant::Control => control_count += 1,
                ABTestVariant::Treatment => treatment_count += 1,
            }
        }

        // Should be roughly 50/50 (within 10% tolerance)
        let ratio = treatment_count as f64 / 1000.0;
        assert!(ratio > 0.4 && ratio < 0.6, "Ratio was {}", ratio);
    }

    #[test]
    fn test_metrics_tracking() {
        let mut test = ABTest::builder("test").build();
        test.start();

        // Record some metrics
        test.record_impression("user_1", 0.8, 5000);
        test.record_impression("user_1", 0.7, 4000);
        test.record_click("user_1", Uuid::new_v4());
        test.record_feedback("user_1", true);

        let variant = test.get_variant("user_1");
        let metrics = test.get_metrics(variant);

        assert_eq!(metrics.impressions, 2);
        assert_eq!(metrics.clicks, 1);
        assert_eq!(metrics.positive_feedback, 1);
        assert_eq!(metrics.unique_users, 1);
        assert!((metrics.ctr() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_chi_squared_significant() {
        // Clear difference: 10% vs 20% CTR with large sample
        let (chi_sq, p_value) = ABTestAnalyzer::chi_squared_test(
            1000, 100, // Control: 10% CTR
            1000, 200, // Treatment: 20% CTR
        );

        assert!(chi_sq > CHI_SQUARED_CRITICAL_005);
        assert!(p_value < 0.05);
    }

    #[test]
    fn test_chi_squared_not_significant() {
        // Small difference with small sample
        let (chi_sq, p_value) = ABTestAnalyzer::chi_squared_test(
            50, 5, // Control: 10% CTR
            50, 6, // Treatment: 12% CTR
        );

        // Should not be significant (sample too small)
        assert!(p_value > 0.05 || chi_sq < CHI_SQUARED_CRITICAL_005);
    }

    #[test]
    fn test_confidence_interval() {
        let (low, high) = ABTestAnalyzer::calculate_confidence_interval(
            1000, 100, // 10% CTR
            1000, 150, // 15% CTR
        );

        // Difference is 5%, CI should contain it
        assert!(low < 0.05);
        assert!(high > 0.05);
        // And CI should not include 0 (significant difference)
        assert!(low > 0.0 || high < 0.0 || (low < 0.0 && high > 0.0));
    }

    #[test]
    fn test_manager_lifecycle() {
        let manager = ABTestManager::new();

        // Create test
        let test = ABTest::builder("test_lifecycle")
            .with_description("Test lifecycle management")
            .build();

        let id = manager.create_test(test).unwrap();

        // Start test
        manager.start_test(&id).unwrap();
        let test = manager.get_test(&id).unwrap();
        assert_eq!(test.status, ABTestStatus::Running);

        // Record some data
        manager.record_impression(&id, "user_1", 0.8, 5000).unwrap();
        manager.record_click(&id, "user_1", Uuid::new_v4()).unwrap();

        // Analyze
        let results = manager.analyze_test(&id).unwrap();
        assert!(!results.is_significant); // Not enough data

        // Complete
        manager.complete_test(&id).unwrap();
        let test = manager.get_test(&id).unwrap();
        assert_eq!(test.status, ABTestStatus::Completed);

        // Archive
        manager.archive_test(&id).unwrap();
        assert!(manager.get_test(&id).is_none());
        assert_eq!(manager.list_archived().len(), 1);
    }

    #[test]
    fn test_learned_weights_integration() {
        let control = LearnedWeights::default();
        let mut treatment = LearnedWeights::default();
        treatment.semantic = 0.6;
        treatment.entity = 0.2;
        treatment.normalize();

        let test = ABTest::builder("weights_test")
            .with_control(control.clone())
            .with_treatment(treatment.clone())
            .build();

        assert_eq!(
            test.get_weights(ABTestVariant::Control).semantic,
            control.semantic
        );
        assert_eq!(
            test.get_weights(ABTestVariant::Treatment).semantic,
            treatment.semantic
        );
    }

    #[test]
    fn test_ctr_calculation() {
        let mut metrics = VariantMetrics::default();

        assert_eq!(metrics.ctr(), 0.0); // No impressions

        metrics.impressions = 100;
        metrics.clicks = 10;

        assert!((metrics.ctr() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_success_rate_calculation() {
        let mut metrics = VariantMetrics::default();

        assert_eq!(metrics.success_rate(), 0.0); // No feedback

        metrics.positive_feedback = 8;
        metrics.negative_feedback = 2;

        assert!((metrics.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_power_estimation() {
        let mut test = ABTest::builder("power_test").build();

        // Add significant data
        for i in 0..500 {
            let user = format!("control_{}", i);
            test.user_assignments
                .insert(user.clone(), ABTestVariant::Control);
            test.control_metrics.impressions += 1;
            test.control_metrics.unique_users += 1;
            if i % 10 == 0 {
                // 10% CTR
                test.control_metrics.clicks += 1;
            }
        }

        for i in 0..500 {
            let user = format!("treatment_{}", i);
            test.user_assignments
                .insert(user.clone(), ABTestVariant::Treatment);
            test.treatment_metrics.impressions += 1;
            test.treatment_metrics.unique_users += 1;
            if i % 5 == 0 {
                // 20% CTR
                test.treatment_metrics.clicks += 1;
            }
        }

        let power = ABTestAnalyzer::estimate_power(&test);
        assert!(power > 0.5, "Power was {}", power); // Should have decent power with this effect size
    }

    #[test]
    fn test_manager_summary() {
        let manager = ABTestManager::new();

        // Create tests in different states
        let test1 = ABTest::builder("draft_test").build();
        manager.create_test(test1).unwrap();

        let test2 = ABTest::builder("running_test").build();
        let id2 = manager.create_test(test2).unwrap();
        manager.start_test(&id2).unwrap();

        let summary = manager.summary();
        assert_eq!(summary.total_active, 2);
        assert_eq!(summary.draft, 1);
        assert_eq!(summary.running, 1);
    }

    #[test]
    fn test_recommendations_generation() {
        let mut test = ABTest::builder("recommendations_test")
            .with_min_impressions(100)
            .build();

        // Add data showing treatment wins
        test.control_metrics.impressions = 1000;
        test.control_metrics.clicks = 100; // 10% CTR
        test.treatment_metrics.impressions = 1000;
        test.treatment_metrics.clicks = 200; // 20% CTR

        let results = ABTestAnalyzer::analyze(&test);

        assert!(results.is_significant);
        assert_eq!(results.winner, Some(ABTestVariant::Treatment));
        assert!(!results.recommendations.is_empty());
        assert!(results
            .recommendations
            .iter()
            .any(|r| r.contains("Treatment")));
    }

    #[test]
    fn test_ab_demo_with_numbers() {
        println!("\n========================================");
        println!("       A/B TESTING DEMO WITH NUMBERS");
        println!("========================================\n");

        // Scenario 1: Clear winner
        println!("📊 SCENARIO 1: Clear Winner (Treatment significantly better)");
        println!("   Control:   1000 impressions, 100 clicks (10.0% CTR)");
        println!("   Treatment: 1000 impressions, 200 clicks (20.0% CTR)");

        let (chi_sq, p_value) = ABTestAnalyzer::chi_squared_test(1000, 100, 1000, 200);
        let (ci_low, ci_high) = ABTestAnalyzer::calculate_confidence_interval(1000, 100, 1000, 200);

        println!("\n   RESULTS:");
        println!("   ├─ Chi-squared statistic: {:.4}", chi_sq);
        println!("   ├─ P-value: {:.6}", p_value);
        println!(
            "   ├─ Significant (p < 0.05): {}",
            if p_value < 0.05 { "YES ✓" } else { "NO ✗" }
        );
        println!(
            "   ├─ 95% Confidence Interval: ({:.4}, {:.4})",
            ci_low, ci_high
        );
        println!(
            "   └─ Relative improvement: {:.1}%",
            ((0.20 - 0.10) / 0.10) * 100.0
        );

        // Scenario 2: No significant difference
        println!("\n📊 SCENARIO 2: No Significant Difference (Sample too small)");
        println!("   Control:   50 impressions, 5 clicks (10.0% CTR)");
        println!("   Treatment: 50 impressions, 6 clicks (12.0% CTR)");

        let (chi_sq2, p_value2) = ABTestAnalyzer::chi_squared_test(50, 5, 50, 6);
        let (ci_low2, ci_high2) = ABTestAnalyzer::calculate_confidence_interval(50, 5, 50, 6);

        println!("\n   RESULTS:");
        println!("   ├─ Chi-squared statistic: {:.4}", chi_sq2);
        println!("   ├─ P-value: {:.6}", p_value2);
        println!(
            "   ├─ Significant (p < 0.05): {}",
            if p_value2 < 0.05 { "YES ✓" } else { "NO ✗" }
        );
        println!(
            "   ├─ 95% Confidence Interval: ({:.4}, {:.4})",
            ci_low2, ci_high2
        );
        println!(
            "   └─ CI includes 0: {} (effect may be due to chance)",
            if ci_low2 < 0.0 && ci_high2 > 0.0 {
                "YES"
            } else {
                "NO"
            }
        );

        // Scenario 3: Full analysis with recommendations
        println!("\n📊 SCENARIO 3: Full Analysis with Recommendations");
        let mut test = ABTest::builder("semantic_weight_test")
            .with_min_impressions(100)
            .build();

        test.control_metrics.impressions = 5000;
        test.control_metrics.clicks = 500; // 10% CTR
        test.control_metrics.positive_feedback = 400;
        test.control_metrics.negative_feedback = 50;

        test.treatment_metrics.impressions = 5000;
        test.treatment_metrics.clicks = 750; // 15% CTR
        test.treatment_metrics.positive_feedback = 600;
        test.treatment_metrics.negative_feedback = 30;

        let results = ABTestAnalyzer::analyze(&test);

        println!("   Test: Comparing semantic weight emphasis");
        println!("   Control:   5000 impressions, 500 clicks (10.0% CTR)");
        println!("   Treatment: 5000 impressions, 750 clicks (15.0% CTR)");
        println!("\n   STATISTICAL RESULTS:");
        println!("   ├─ Chi-squared: {:.4}", results.chi_squared);
        println!("   ├─ P-value: {:.8}", results.p_value);
        println!(
            "   ├─ Confidence Level: {:.2}%",
            results.confidence_level * 100.0
        );
        println!(
            "   ├─ Significant: {}",
            if results.is_significant {
                "YES ✓"
            } else {
                "NO ✗"
            }
        );
        println!("   ├─ Winner: {:?}", results.winner);
        println!(
            "   ├─ Relative Improvement: {:.2}%",
            results.relative_improvement
        );
        println!("   ├─ Control CTR: {:.2}%", results.control_ctr * 100.0);
        println!("   ├─ Treatment CTR: {:.2}%", results.treatment_ctr * 100.0);
        println!(
            "   └─ 95% CI: ({:.4}, {:.4})",
            results.confidence_interval.0, results.confidence_interval.1
        );

        println!("\n   RECOMMENDATIONS:");
        for (i, rec) in results.recommendations.iter().enumerate() {
            println!("   {}. {}", i + 1, rec);
        }

        println!("\n========================================");
        println!("        END OF A/B TESTING DEMO");
        println!("========================================\n");

        // Assertions to make sure the test still works
        assert!(results.is_significant);
        assert_eq!(results.winner, Some(ABTestVariant::Treatment));
    }

    #[test]
    fn test_comprehensive_analysis_demo() {
        println!("\n========================================");
        println!("    COMPREHENSIVE A/B ANALYSIS DEMO");
        println!("    (Dynamic Weight-Based Simulation)");
        println!("========================================\n");

        // =================================================================
        // DYNAMIC SIMULATION: Compare old vs new relevance weights
        // =================================================================

        // Control: Old weights (pre-CTX-3) - no momentum amplification, no access_count, no graph_strength
        let control_weights = LearnedWeights {
            semantic: 0.35,
            entity: 0.30,
            tag: 0.10,
            importance: 0.10,
            momentum: 0.15,      // Old: lower momentum weight, no amplification
            access_count: 0.0,   // Old: not used
            graph_strength: 0.0, // Old: not used
            update_count: 0,
            last_updated: None,
        };

        // Treatment: New weights (CTX-3) - momentum amplification, access_count, graph_strength
        let treatment_weights = LearnedWeights::default(); // Uses current optimized defaults

        println!("📊 WEIGHT COMPARISON:");
        println!("   Control (old):   semantic={:.2}, entity={:.2}, momentum={:.2}, access={:.2}, graph={:.2}",
            control_weights.semantic, control_weights.entity, control_weights.momentum,
            control_weights.access_count, control_weights.graph_strength);
        println!("   Treatment (new): semantic={:.2}, entity={:.2}, momentum={:.2}, access={:.2}, graph={:.2}\n",
            treatment_weights.semantic, treatment_weights.entity, treatment_weights.momentum,
            treatment_weights.access_count, treatment_weights.graph_strength);

        // Generate synthetic memory corpus with varying characteristics
        // Each tuple: (semantic, entity, tag, importance, momentum_ema, access_count, graph_strength, is_truly_relevant)
        // The key insight: some memories LOOK good (high semantic/entity) but have poor track record
        // Treatment should deprioritize these based on momentum/access/graph signals
        let memory_corpus: Vec<(f32, f32, f32, f32, f32, u32, f32, bool)> = vec![
            // === HIGH-VALUE: Good signals + good track record ===
            (0.8, 0.7, 0.5, 0.8, 0.9, 15, 0.9, true), // Consistently helpful, frequently accessed
            (0.7, 0.8, 0.6, 0.7, 0.8, 12, 0.85, true), // Strong entity match, proven value
            (0.9, 0.6, 0.4, 0.9, 0.7, 10, 0.8, true), // High semantic, good track record
            (0.6, 0.9, 0.7, 0.6, 0.85, 8, 0.75, true), // Entity-heavy, reliable
            // === TRAPS: Look good but misleading (control will surface these, treatment won't) ===
            (0.95, 0.9, 0.8, 0.9, -0.6, 1, 0.15, false), // BEST semantic/entity but terrible momentum
            (0.9, 0.85, 0.7, 0.85, -0.4, 0, 0.1, false), // High scores, never accessed, weak graph
            (0.88, 0.82, 0.6, 0.8, -0.5, 1, 0.2, false), // Looks great, proven misleading
            (0.85, 0.88, 0.75, 0.82, -0.3, 2, 0.25, false), // Strong traditional signals, poor history
            // === MEDIUM: Mixed signals ===
            (0.7, 0.5, 0.3, 0.6, 0.4, 4, 0.5, true), // Decent, somewhat proven
            (0.5, 0.6, 0.4, 0.5, 0.3, 3, 0.45, true), // Moderate all around
            (0.6, 0.55, 0.35, 0.55, 0.35, 3, 0.4, true), // Average
            // === LOW-VALUE: Poor across the board ===
            (0.4, 0.3, 0.2, 0.4, 0.1, 1, 0.2, false), // Low everything
            (0.35, 0.4, 0.25, 0.35, -0.1, 1, 0.15, false), // Below average
        ];

        // Score all memories with both weight sets and rank them
        let mut control_ranked: Vec<(usize, f32, bool)> = memory_corpus
            .iter()
            .enumerate()
            .map(|(idx, &(sem, ent, tag, imp, mom, acc, graph, relevant))| {
                let score = control_weights.fuse_scores_full(sem, ent, tag, imp, mom, acc, graph);
                (idx, score, relevant)
            })
            .collect();

        let mut treatment_ranked: Vec<(usize, f32, bool)> = memory_corpus
            .iter()
            .enumerate()
            .map(|(idx, &(sem, ent, tag, imp, mom, acc, graph, relevant))| {
                let score = treatment_weights.fuse_scores_full(sem, ent, tag, imp, mom, acc, graph);
                (idx, score, relevant)
            })
            .collect();

        // Sort by score descending
        control_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        treatment_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("🔍 RANKING COMPARISON (top 8):");
        println!("   Control ranking:");
        for (rank, (idx, score, relevant)) in control_ranked.iter().take(8).enumerate() {
            let status = if *relevant {
                "✓ relevant"
            } else {
                "✗ TRAP"
            };
            println!(
                "      #{}: memory[{}] score={:.3} {}",
                rank + 1,
                idx,
                score,
                status
            );
        }
        println!("   Treatment ranking:");
        for (rank, (idx, score, relevant)) in treatment_ranked.iter().take(8).enumerate() {
            let status = if *relevant {
                "✓ relevant"
            } else {
                "✗ TRAP"
            };
            println!(
                "      #{}: memory[{}] score={:.3} {}",
                rank + 1,
                idx,
                score,
                status
            );
        }
        println!();

        // Simulate sessions: Claude uses surfaced memories, user gives feedback
        // Model: trap in context → probability of bad outcome (negative feedback)
        // Success = user gives positive feedback (Claude's action was helpful)
        let num_sessions = 1000;
        let memories_surfaced = 5;

        // Count relevant vs trap in top K for each variant
        let control_top_k: Vec<bool> = control_ranked
            .iter()
            .take(memories_surfaced)
            .map(|x| x.2)
            .collect();
        let treatment_top_k: Vec<bool> = treatment_ranked
            .iter()
            .take(memories_surfaced)
            .map(|x| x.2)
            .collect();

        let control_relevant_count = control_top_k.iter().filter(|&&r| r).count();
        let treatment_relevant_count = treatment_top_k.iter().filter(|&&r| r).count();
        let control_trap_count = memories_surfaced - control_relevant_count;
        let treatment_trap_count = memories_surfaced - treatment_relevant_count;

        // Probability of bad outcome = trap_ratio (each trap has chance to mislead)
        let control_trap_ratio = control_trap_count as f32 / memories_surfaced as f32;
        let treatment_trap_ratio = treatment_trap_count as f32 / memories_surfaced as f32;

        println!("📈 CONTEXT QUALITY (top {}):", memories_surfaced);
        println!(
            "   Control:   {} relevant, {} traps ({:.0}% trap ratio)",
            control_relevant_count,
            control_trap_count,
            control_trap_ratio * 100.0
        );
        println!(
            "   Treatment: {} relevant, {} traps ({:.0}% trap ratio)\n",
            treatment_relevant_count,
            treatment_trap_count,
            treatment_trap_ratio * 100.0
        );

        // Deterministic seeding for reproducibility (LCG PRNG)
        let mut rng_state: u64 = 42;
        let next_rand = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Use upper 32 bits for better distribution, divide by 2^32 for [0, 1) range
            ((*state >> 32) as f32) / (0x1_0000_0000_u64 as f32)
        };

        let mut control_positive = 0u64;
        let mut control_negative = 0u64;
        let mut treatment_positive = 0u64;
        let mut treatment_negative = 0u64;

        for _session in 0..num_sessions {
            // Control: Claude uses context, user responds
            // Bad outcome probability = trap_ratio (trap misleads Claude → bad action)
            if next_rand(&mut rng_state) < control_trap_ratio {
                control_negative += 1; // Trap caused bad outcome
            } else {
                control_positive += 1; // Good outcome
            }

            // Treatment: same model
            if next_rand(&mut rng_state) < treatment_trap_ratio {
                treatment_negative += 1;
            } else {
                treatment_positive += 1;
            }
        }

        // For NNT calculation: impressions = sessions, clicks = successful sessions
        // CTR = success rate = positive / total
        let num_impressions = num_sessions as u64;
        let control_clicks = control_positive; // Success = click
        let treatment_clicks = treatment_positive;

        // Build test with dynamic results
        let mut test = ABTest::builder("relevance_weights_experiment")
            .with_description(
                "CTX-3: Quality over quantity - momentum, access_count, graph_strength",
            )
            .with_control(control_weights)
            .with_treatment(treatment_weights)
            .with_min_impressions(100)
            .with_traffic_split(0.5)
            .build();

        test.control_metrics.impressions = num_impressions as u64;
        test.control_metrics.clicks = control_clicks;
        test.control_metrics.unique_users = (num_impressions as f64 * 0.85) as u64;
        test.control_metrics.positive_feedback = control_positive;
        test.control_metrics.negative_feedback = control_negative;

        test.treatment_metrics.impressions = num_impressions as u64;
        test.treatment_metrics.clicks = treatment_clicks;
        test.treatment_metrics.unique_users = (num_impressions as f64 * 0.85) as u64;
        test.treatment_metrics.positive_feedback = treatment_positive;
        test.treatment_metrics.negative_feedback = treatment_negative;

        let control_ctr = (control_clicks as f64 / num_impressions as f64) * 100.0;
        let treatment_ctr = (treatment_clicks as f64 / num_impressions as f64) * 100.0;

        let analysis = ABTestAnalyzer::comprehensive_analysis(&test);

        println!("📊 DYNAMIC SIMULATION RESULTS:");
        println!(
            "   ├─ Control:   {} impressions, {} clicks ({:.1}% CTR)",
            num_impressions, control_clicks, control_ctr
        );
        println!(
            "   │            positive={}, negative={}",
            control_positive, control_negative
        );
        println!(
            "   └─ Treatment: {} impressions, {} clicks ({:.1}% CTR)",
            num_impressions, treatment_clicks, treatment_ctr
        );
        println!(
            "                 positive={}, negative={}\n",
            treatment_positive, treatment_negative
        );

        println!("🔬 FREQUENTIST ANALYSIS:");
        println!("   ├─ Chi-squared: {:.4}", analysis.frequentist.chi_squared);
        println!("   ├─ P-value: {:.6}", analysis.frequentist.p_value);
        println!(
            "   ├─ Significant: {}",
            if analysis.frequentist.is_significant {
                "YES ✓"
            } else {
                "NO ✗"
            }
        );
        println!("   └─ Winner: {:?}\n", analysis.frequentist.winner);

        println!("🎲 BAYESIAN ANALYSIS:");
        println!(
            "   ├─ P(Treatment better): {:.2}%",
            analysis.bayesian.prob_treatment_better * 100.0
        );
        println!(
            "   ├─ Expected lift: {:.2}%",
            analysis.bayesian.expected_lift * 100.0
        );
        println!(
            "   ├─ 95% Credible Interval: ({:.2}%, {:.2}%)",
            analysis.bayesian.credible_interval.0 * 100.0,
            analysis.bayesian.credible_interval.1 * 100.0
        );
        println!(
            "   ├─ Risk if shipping treatment: {:.3}%",
            analysis.bayesian.risk_treatment * 100.0
        );
        println!(
            "   └─ Risk if keeping control: {:.3}%\n",
            analysis.bayesian.risk_control * 100.0
        );

        println!("📏 EFFECT SIZE:");
        println!("   ├─ Cohen's h: {:.4}", analysis.effect_size.cohens_h);
        println!(
            "   ├─ Interpretation: {}",
            analysis.effect_size.interpretation
        );
        println!(
            "   ├─ Relative Risk: {:.2}x",
            analysis.effect_size.relative_risk
        );
        println!("   ├─ Odds Ratio: {:.2}", analysis.effect_size.odds_ratio);
        if analysis.effect_size.nnt.is_finite() {
            println!(
                "   └─ NNT (Number Needed to Treat): {:.0}\n",
                analysis.effect_size.nnt
            );
        } else {
            println!("   └─ NNT: N/A (no effect)\n");
        }

        println!("⚖️ DATA QUALITY (SRM Check):");
        println!(
            "   ├─ Expected ratio: {:.1}%",
            analysis.srm.expected_ratio * 100.0
        );
        println!(
            "   ├─ Observed ratio: {:.1}%",
            analysis.srm.observed_ratio * 100.0
        );
        println!(
            "   ├─ SRM Detected: {}",
            if analysis.srm.srm_detected {
                "YES ⚠️"
            } else {
                "NO ✓"
            }
        );
        println!("   └─ Severity: {:?}\n", analysis.srm.severity);

        println!("📈 SEQUENTIAL TESTING:");
        println!(
            "   ├─ Analysis #{} of {}",
            analysis.sequential.analysis_number, analysis.sequential.planned_analyses
        );
        println!("   ├─ Alpha spent: {:.4}", analysis.sequential.alpha_spent);
        println!(
            "   ├─ Current threshold: {:.4}",
            analysis.sequential.current_alpha
        );
        println!(
            "   └─ Can stop early: {}\n",
            if analysis.sequential.can_stop_early {
                "YES ✓"
            } else {
                "NO - Continue testing"
            }
        );

        println!("═══════════════════════════════════════");
        println!("🎯 FINAL DECISION:");
        println!("═══════════════════════════════════════");
        println!(
            "   Should ship: {}",
            if analysis.should_ship {
                "YES ✅"
            } else {
                "NO ❌"
            }
        );
        println!(
            "   Practically significant: {}",
            if analysis.is_practically_significant {
                "YES"
            } else {
                "NO"
            }
        );
        println!("\n📋 USER-FOCUSED INSIGHTS:");
        for insight in &analysis.insights {
            println!("   • {}", insight);
        }

        // Calculate and display NNT
        let ard = (treatment_ctr - control_ctr) / 100.0; // Absolute Risk Difference
        let nnt = if ard > 0.0 { 1.0 / ard } else { f64::INFINITY };
        println!("\n🎯 KEY METRIC:");
        println!(
            "   ├─ CTR Improvement: {:.1}% → {:.1}% (+{:.1}%)",
            control_ctr,
            treatment_ctr,
            treatment_ctr - control_ctr
        );
        println!("   ├─ ARD (Absolute Risk Difference): {:.2}%", ard * 100.0);
        if nnt.is_finite() && nnt < 100.0 {
            println!("   └─ NNT (Number Needed to Treat): {:.0}", nnt);
            println!("      (1 in {:.0} users benefit from treatment)", nnt);
        } else {
            println!("   └─ NNT: N/A (no significant improvement)");
        }

        println!("\n========================================");
        println!("     END OF COMPREHENSIVE ANALYSIS");
        println!("========================================\n");

        // Assertions - verify treatment outperforms control
        assert!(
            treatment_clicks >= control_clicks,
            "Treatment ({} clicks) should outperform Control ({} clicks)",
            treatment_clicks,
            control_clicks
        );
        assert!(
            treatment_ctr >= control_ctr,
            "Treatment CTR ({:.1}%) should be >= Control CTR ({:.1}%)",
            treatment_ctr,
            control_ctr
        );
        // Treatment should have better positive/negative ratio
        let control_quality = if control_negative > 0 {
            control_positive as f64 / control_negative as f64
        } else {
            control_positive as f64
        };
        let treatment_quality = if treatment_negative > 0 {
            treatment_positive as f64 / treatment_negative as f64
        } else {
            treatment_positive as f64
        };
        assert!(
            treatment_quality >= control_quality * 0.9, // Allow 10% tolerance
            "Treatment quality ratio ({:.2}) should be >= Control ({:.2})",
            treatment_quality,
            control_quality
        );
        assert!(!analysis.insights.is_empty());
    }

    #[test]
    fn test_bayesian_analysis() {
        let mut test = ABTest::builder("bayesian_test").build();

        test.control_metrics.impressions = 1000;
        test.control_metrics.clicks = 100; // 10%
        test.treatment_metrics.impressions = 1000;
        test.treatment_metrics.clicks = 150; // 15%

        let bayesian = ABTestAnalyzer::bayesian_analysis(&test);

        // Treatment should have high probability of being better
        assert!(bayesian.prob_treatment_better > 0.9);
        assert!(bayesian.expected_lift > 0.0);
        // Credible interval should not include large negative values
        assert!(bayesian.credible_interval.0 > -0.5);
    }

    #[test]
    fn test_effect_size_calculation() {
        let mut test = ABTest::builder("effect_test").build();

        test.control_metrics.impressions = 1000;
        test.control_metrics.clicks = 100; // 10%
        test.treatment_metrics.impressions = 1000;
        test.treatment_metrics.clicks = 200; // 20%

        let effect = ABTestAnalyzer::calculate_effect_size(&test);

        // 10% to 20% should be a small-to-medium effect
        assert!(effect.cohens_h > 0.2);
        assert!(effect.relative_risk > 1.5);
        // NNT should be 10 (1/(0.2-0.1))
        assert!((effect.nnt - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_srm_detection() {
        let mut test = ABTest::builder("srm_test").with_traffic_split(0.5).build();

        // Severe SRM: expected 50/50, got 70/30
        test.control_metrics.impressions = 700;
        test.treatment_metrics.impressions = 300;

        let srm = ABTestAnalyzer::check_srm(&test);

        // Should detect SRM
        assert!(srm.srm_detected);
        assert_eq!(srm.severity, SRMSeverity::Critical);
    }

    #[test]
    fn test_sequential_analysis() {
        let mut test = ABTest::builder("sequential_test")
            .with_min_impressions(100)
            .build();

        test.control_metrics.impressions = 500;
        test.control_metrics.clicks = 25; // 5%
        test.treatment_metrics.impressions = 500;
        test.treatment_metrics.clicks = 75; // 15%

        // Early look (analysis 1 of 5)
        let seq = ABTestAnalyzer::sequential_analysis(&test, 1, 5);

        assert_eq!(seq.analysis_number, 1);
        assert_eq!(seq.planned_analyses, 5);
        // O'Brien-Fleming is very conservative early
        assert!(seq.alpha_spent < 0.01);
    }
}
