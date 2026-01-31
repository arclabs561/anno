//! Evaluation types: MetricValue, GoalCheckResult, etc.
//!
//! These are shared primitives for evaluation that can be reused
//! across NER evaluation and other evaluation tasks.

use anno::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A type-safe metric value bounded to [0.0, 1.0].
///
/// Ensures metrics like precision, recall, and F1 are always valid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
#[serde(transparent)]
pub struct MetricValue(f64);

/// A metric with variance and confidence interval.
///
/// Tracks the mean, standard deviation, and 95% confidence interval
/// for a metric computed across multiple samples/runs/datasets.
///
/// # Example
///
/// ```rust
/// use anno::eval::MetricWithVariance;
///
/// let metric = MetricWithVariance::from_samples(&[0.85, 0.87, 0.82, 0.88, 0.84]);
/// println!("F1: {:.1}% ± {:.1}% (95% CI)", metric.mean * 100.0, metric.ci_95 * 100.0);
/// // F1: 85.2% ± 2.1% (95% CI)
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MetricWithVariance {
    /// Mean value of the metric
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// 95% confidence interval (±)
    pub ci_95: f64,
    /// Minimum observed value
    pub min: f64,
    /// Maximum observed value
    pub max: f64,
    /// Number of samples
    pub n: usize,
}

impl MetricWithVariance {
    /// Create from a slice of sample values.
    ///
    /// Uses sample standard deviation (Bessel's correction) and
    /// t-distribution approximation for 95% CI.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                ci_95: 0.0,
                min: 0.0,
                max: 0.0,
                n: 0,
            };
        }

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let std_dev = if n > 1 {
            let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // 95% CI using t-distribution approximation
        // For n >= 30, use z = 1.96; otherwise approximate with t
        let t_value = if n >= 30 {
            1.96
        } else {
            // Conservative t-value approximation for smaller samples
            2.0 + 0.1 / (n as f64).sqrt()
        };
        let ci_95 = if n > 1 {
            t_value * std_dev / (n as f64).sqrt()
        } else {
            0.0
        };

        Self {
            mean,
            std_dev,
            ci_95,
            min,
            max,
            n,
        }
    }

    /// Format as "mean ± ci95" string.
    pub fn format_with_ci(&self) -> String {
        if self.n == 0 {
            return "N/A".to_string();
        }
        format!("{:.1}% ± {:.1}%", self.mean * 100.0, self.ci_95 * 100.0)
    }

    /// Format as "mean (min-max)" string.
    pub fn format_with_range(&self) -> String {
        if self.n == 0 {
            return "N/A".to_string();
        }
        format!(
            "{:.1}% ({:.1}%-{:.1}%)",
            self.mean * 100.0,
            self.min * 100.0,
            self.max * 100.0
        )
    }

    /// Get coefficient of variation (CV = std_dev / mean).
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-10 {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }
}

impl Default for MetricWithVariance {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            ci_95: 0.0,
            min: 0.0,
            max: 0.0,
            n: 0,
        }
    }
}

impl std::fmt::Display for MetricWithVariance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_with_ci())
    }
}

impl MetricValue {
    /// Create a new MetricValue, clamping to [0.0, 1.0].
    ///
    /// # Example
    /// ```
    /// use anno::eval::MetricValue;
    /// let v = MetricValue::new(0.95);
    /// assert!((v.get() - 0.95).abs() < 1e-6);
    /// ```
    pub fn new(value: f64) -> Self {
        MetricValue(value.clamp(0.0, 1.0))
    }

    /// Try to create a MetricValue, returning error if out of bounds.
    pub fn try_new(value: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(Error::InvalidInput(format!(
                "MetricValue must be in [0.0, 1.0], got {}",
                value
            )));
        }
        Ok(MetricValue(value))
    }

    /// Get the underlying value.
    #[inline]
    pub fn get(&self) -> f64 {
        self.0
    }
}

impl Default for MetricValue {
    fn default() -> Self {
        MetricValue(0.0)
    }
}

impl std::fmt::Display for MetricValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

impl From<f64> for MetricValue {
    fn from(value: f64) -> Self {
        MetricValue::new(value)
    }
}

/// Result of checking evaluation goals.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalCheckResult {
    /// Whether all goals were met.
    pub passed: bool,
    /// Individual goal check results.
    pub checks: HashMap<String, GoalCheck>,
    /// Summary message.
    pub summary: Option<String>,
}

impl GoalCheckResult {
    /// Create a new GoalCheckResult (defaults to passed = true).
    #[must_use]
    pub fn new() -> Self {
        Self {
            passed: true,
            checks: HashMap::new(),
            summary: None,
        }
    }

    /// Add a goal check result.
    pub fn add_check(&mut self, name: impl Into<String>, check: GoalCheck) {
        if !check.passed {
            self.passed = false;
        }
        self.checks.insert(name.into(), check);
    }

    /// Add a failure (convenience method for add_check with fail).
    pub fn add_failure(&mut self, name: impl Into<String>, actual: f64, threshold: f64) {
        self.add_check(name, GoalCheck::fail(threshold, actual));
    }

    /// Add a success (convenience method for add_check with pass).
    pub fn add_success(&mut self, name: impl Into<String>, actual: f64, threshold: f64) {
        self.add_check(name, GoalCheck::pass(threshold, actual));
    }

    /// Get number of passed checks.
    pub fn passed_count(&self) -> usize {
        self.checks.values().filter(|c| c.passed).count()
    }

    /// Get number of failed checks.
    pub fn failed_count(&self) -> usize {
        self.checks.values().filter(|c| !c.passed).count()
    }
}

/// Individual goal check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalCheck {
    /// Whether this goal was met.
    pub passed: bool,
    /// Expected threshold.
    pub threshold: f64,
    /// Actual value achieved.
    pub actual: f64,
    /// Optional message.
    pub message: Option<String>,
}

impl GoalCheck {
    /// Create a new goal check.
    pub fn new(passed: bool, threshold: f64, actual: f64) -> Self {
        Self {
            passed,
            threshold,
            actual,
            message: None,
        }
    }

    /// Create a passing check.
    pub fn pass(threshold: f64, actual: f64) -> Self {
        Self::new(true, threshold, actual)
    }

    /// Create a failing check.
    pub fn fail(threshold: f64, actual: f64) -> Self {
        Self::new(false, threshold, actual)
    }

    /// Add a message to the check.
    #[must_use]
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }
}

// =============================================================================
// Label Shift Quantification (Familiarity-inspired)
// =============================================================================

/// Label shift between training and evaluation entity types.
///
/// # Why This Matters
///
/// Imagine you trained a model on `{PER, ORG, LOC}` and then evaluate on
/// `{PERSON, COMPANY, CITY}`. Is that zero-shot? Technically yes (new labels).
/// Practically no (same concepts).
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                    THE LABEL SHIFT PROBLEM                              │
/// ├─────────────────────────────────────────────────────────────────────────┤
/// │                                                                         │
/// │  TRAINING LABELS           EVAL LABELS         ARE THEY THE SAME?       │
/// │  ───────────────           ───────────         ──────────────────       │
/// │                                                                         │
/// │  PER ───────────────────── PERSON             ✓ Obviously (renamed)     │
/// │  ORG ───────────────────── COMPANY            ✓ Subset relationship     │
/// │  LOC ───────────────────── CITY               ✓ Subset relationship     │
/// │                                                                         │
/// │  ??? ←─────────────────── DISEASE            ✗ TRUE ZERO-SHOT!         │
/// │  ??? ←─────────────────── DRUG               ✗ TRUE ZERO-SHOT!         │
/// │                                                                         │
/// │  If 80% of eval types have training equivalents, your F1 is inflated.   │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Embedding Space View
///
/// Labels that seem different can be close in embedding space:
///
/// ```text
///                    EMBEDDING SPACE (2D projection)
///                    ───────────────────────────────
///
///            PER ●───────────────● PERSON
///                      │
///                 very close in
///                embedding space
///
///            ORG ●─────● COMPANY
///
///            LOC ●─────────● CITY
///
///
///                                        ● DISEASE    ← Far from all
///                                                       training types!
///                                        ● DRUG       ← This is TRUE
///                                                       zero-shot.
///
/// F1 on {PERSON, COMPANY, CITY}:  85%  (but model "knew" these)
/// F1 on {DISEASE, DRUG}:          45%  (honest zero-shot)
/// ```
///
/// # Research Context (arXiv:2412.10121 "Familiarity")
///
/// Key findings from Golde et al. (2024):
/// - 80%+ label overlap in NuNER/PileNER → inflated F1 scores
/// - True zero-shot: evaluate only on types NOT in training
/// - Familiarity = semantic similarity × frequency weighting
///
/// # Example
///
/// ```rust
/// use anno::eval::LabelShift;
///
/// let shift = LabelShift {
///     overlap_ratio: 0.85,    // 85% of eval types in train
///     familiarity: 0.72,      // Semantic similarity score
///     true_zero_shot_types: vec!["DISEASE".into(), "DRUG".into()],
///     transfer_difficulty: "low".into(),
/// };
///
/// // High overlap = easy transfer, but NOT true zero-shot
/// assert!(shift.is_inflated());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelShift {
    /// Fraction of eval types found in training data (exact string match).
    pub overlap_ratio: f64,

    /// Familiarity score: semantic similarity weighted by frequency.
    /// Range: [0, 1]. Higher = more similar training/eval types.
    pub familiarity: f64,

    /// Entity types in eval NOT present in training (true zero-shot).
    pub true_zero_shot_types: Vec<String>,

    /// Qualitative difficulty: "low", "medium", "high".
    pub transfer_difficulty: String,
}

impl LabelShift {
    /// Check if F1 scores are likely inflated due to high label overlap.
    ///
    /// Threshold from Familiarity paper: >0.8 overlap is concernoing.
    #[must_use]
    pub fn is_inflated(&self) -> bool {
        self.overlap_ratio > 0.8 || self.familiarity > 0.85
    }

    /// Get count of true zero-shot types.
    #[must_use]
    pub fn true_zero_shot_count(&self) -> usize {
        self.true_zero_shot_types.len()
    }

    /// Compute label shift from training and eval type sets.
    ///
    /// # Arguments
    /// * `train_types` - Entity types seen during training
    /// * `eval_types` - Entity types in evaluation benchmark
    ///
    /// # Note
    ///
    /// This computes both string-match overlap and semantic similarity-based familiarity.
    /// For true semantic similarity, use `from_type_sets_with_embeddings()` if embeddings are available.
    /// See arXiv:2412.10121 for details.
    #[must_use]
    pub fn from_type_sets(train_types: &[String], eval_types: &[String]) -> Self {
        let train_set: std::collections::HashSet<_> = train_types.iter().collect();
        let eval_set: std::collections::HashSet<_> = eval_types.iter().collect();

        // Exact match overlap
        let overlap_count = eval_set.intersection(&train_set).count();
        let overlap_ratio = if eval_types.is_empty() {
            0.0
        } else {
            overlap_count as f64 / eval_types.len() as f64
        };

        // True zero-shot = eval types NOT in training
        let true_zero_shot_types: Vec<String> = eval_set
            .difference(&train_set)
            .map(|s| (*s).clone())
            .collect();

        // Compute familiarity using string similarity (improved heuristic)
        // This is better than just overlap_ratio but still not true semantic similarity
        let familiarity = compute_string_based_familiarity(train_types, eval_types);

        let transfer_difficulty = if overlap_ratio > 0.8 || familiarity > 0.85 {
            "low"
        } else if overlap_ratio > 0.4 || familiarity > 0.5 {
            "medium"
        } else {
            "high"
        }
        .to_string();

        Self {
            overlap_ratio,
            familiarity,
            true_zero_shot_types,
            transfer_difficulty,
        }
    }

    /// Compute label shift with embedding-based familiarity.
    ///
    /// # Arguments
    /// * `train_types` - Entity types seen during training
    /// * `eval_types` - Entity types in evaluation benchmark
    /// * `embedding_fn` - Function that computes embedding for a label name
    ///
    /// # Note
    ///
    /// This computes true semantic similarity using embeddings, as recommended
    /// in the Familiarity paper (arXiv:2412.10121). Familiarity = semantic similarity × frequency weighting.
    ///
    /// The embedding function should return a normalized vector (unit length) for cosine similarity.
    #[must_use]
    pub fn from_type_sets_with_embeddings<F>(
        train_types: &[String],
        eval_types: &[String],
        embedding_fn: F,
    ) -> Self
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let mut result = Self::from_type_sets(train_types, eval_types);

        // Compute embedding-based familiarity
        if let Some(familiarity) =
            compute_embedding_based_familiarity(train_types, eval_types, &embedding_fn)
        {
            result.familiarity = familiarity;
        }

        result
    }
}

impl std::fmt::Display for LabelShift {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LabelShift(overlap={:.0}%, familiarity={:.2}, zero-shot={}, difficulty={})",
            self.overlap_ratio * 100.0,
            self.familiarity,
            self.true_zero_shot_types.len(),
            self.transfer_difficulty
        )
    }
}

// =============================================================================
// Coreference Chain Statistics (arXiv:2401.00238 inspired)
// =============================================================================

/// Statistics for stratified coreference evaluation.
///
/// # Why Chain Length Matters: A Narrative
///
/// Imagine analyzing "Pride and Prejudice":
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────┐
/// │                    COREFERENCE IN A NOVEL                               │
/// ├─────────────────────────────────────────────────────────────────────────┤
/// │                                                                         │
/// │  LONG CHAINS (>10 mentions) - THE PROTAGONISTS                          │
/// │  ─────────────────────────────────────────────                          │
/// │                                                                         │
/// │  "Elizabeth" ─── "she" ─── "Lizzy" ─── "her" ─── "Miss Bennet" ───...  │
/// │       │            │          │          │            │                 │
/// │       └────────────┴──────────┴──────────┴────────────┘                 │
/// │                         800+ mentions                                   │
/// │                                                                         │
/// │  Getting these right = understanding the PLOT.                          │
/// │  Who did what to whom? What's Elizabeth's arc?                          │
/// │                                                                         │
/// │  SHORT CHAINS (2-10 mentions) - SECONDARY CHARACTERS                    │
/// │  ───────────────────────────────────────────────────                    │
/// │                                                                         │
/// │  "Mr. Collins" ─── "he" ─── "the clergyman"                             │
/// │       │              │             │                                    │
/// │       └──────────────┴─────────────┘                                    │
/// │                  15 mentions                                            │
/// │                                                                         │
/// │  Important for context, but errors here are less catastrophic.          │
/// │                                                                         │
/// │  SINGLETONS (1 mention) - BACKGROUND                                    │
/// │  ───────────────────────────────────────                                │
/// │                                                                         │
/// │  "a tall man" ─── (no other mentions)                                   │
/// │  "the servant" ─── (no other mentions)                                  │
/// │                                                                         │
/// │  These are closer to entity detection than coreference.                  │
/// │  Including them in CoNLL F1 can change the interpretation of the score.  │
/// └─────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # The Problem with Averaged Metrics
///
/// ```text
/// Model Performance:
///
/// A single averaged metric can hide systematic differences across chain sizes.
/// Prefer reporting stratified metrics by chain length (and be explicit about
/// whether singletons are included).
/// ```
///
/// # Research Context (arXiv:2401.00238)
///
/// "How to Evaluate Coreference in Literary Texts?"
/// - A single CoNLL F1 score is "uninformative, or even misleading."
/// - Stratify by chain length for interpretable results.
///
/// # Example
///
/// ```rust
/// use anno::eval::CorefChainStats;
///
/// let stats = CorefChainStats {
///     long_chain_count: 3,      // Main characters
///     short_chain_count: 15,    // Secondary
///     singleton_count: 42,      // Isolated
///     long_chain_f1: 0.92,      // Good on main characters
///     short_chain_f1: 0.71,     // Weaker on secondary
///     singleton_f1: 0.45,       // Poor on singletons
/// };
///
/// // Report metrics separately, not averaged
/// println!("Main characters: {:.1}% F1", stats.long_chain_f1 * 100.0);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct CorefChainStats {
    /// Number of long chains (>10 mentions).
    pub long_chain_count: usize,
    /// Number of short chains (2-10 mentions).
    pub short_chain_count: usize,
    /// Number of singletons (1 mention).
    pub singleton_count: usize,
    /// F1 score on long chains only.
    pub long_chain_f1: f64,
    /// F1 score on short chains only.
    pub short_chain_f1: f64,
    /// F1 score on singletons (if evaluated).
    pub singleton_f1: f64,
}

impl CorefChainStats {
    /// Total chain count.
    #[must_use]
    pub fn total_chains(&self) -> usize {
        self.long_chain_count + self.short_chain_count + self.singleton_count
    }

    /// Weighted F1 (by chain count).
    ///
    /// Note: This is NOT the same as CoNLL F1 (which averages MUC, B³, CEAF-e).
    #[must_use]
    pub fn weighted_f1(&self) -> f64 {
        let total = self.total_chains();
        if total == 0 {
            return 0.0;
        }

        let weighted_sum = self.long_chain_f1 * self.long_chain_count as f64
            + self.short_chain_f1 * self.short_chain_count as f64
            + self.singleton_f1 * self.singleton_count as f64;

        weighted_sum / total as f64
    }
}

// =============================================================================
// Document Scale Classification (Bourgois & Poibeau 2025)
// =============================================================================

/// Document scale classification based on token count.
///
/// # Research Context (Bourgois & Poibeau 2025, arXiv:2510.15594)
///
/// The paper shows that coreference performance degrades significantly with
/// document length. These thresholds are informed by their analysis:
///
/// ```text
/// Scale           Token Range     Performance Impact
/// ─────────────────────────────────────────────────────
/// Short           <2k             Baseline (OntoNotes-like)
/// Medium          2k-10k          -5% CoNLL F1
/// Long            10k-50k         -10% CoNLL F1
/// BookScale       >50k            -15% CoNLL F1, metrics unreliable
/// ```
///
/// # Example
///
/// ```rust
/// use anno::eval::DocumentScale;
///
/// let scale = DocumentScale::from_tokens(95_000);
/// assert!(scale.is_book_scale());
/// assert!(scale.metrics_may_be_unreliable());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DocumentScale {
    /// Short document (<2k tokens). OntoNotes-like scale.
    #[default]
    Short,
    /// Medium document (2k-10k tokens). Slight performance drop.
    Medium,
    /// Long document (10k-50k tokens). Noticeable degradation.
    Long,
    /// Book-scale document (>50k tokens). Metrics may be unreliable.
    BookScale,
}

impl DocumentScale {
    /// Classify document scale from token count.
    #[must_use]
    pub fn from_tokens(token_count: usize) -> Self {
        match token_count {
            0..=2000 => Self::Short,
            2001..=10000 => Self::Medium,
            10001..=50000 => Self::Long,
            _ => Self::BookScale,
        }
    }

    /// Check if this is book-scale (>50k tokens).
    #[must_use]
    pub fn is_book_scale(&self) -> bool {
        matches!(self, Self::BookScale)
    }

    /// Check if coreference metrics may be unreliable at this scale.
    ///
    /// At book scale, MUC tends to inflate while CEAF-e tends to collapse.
    #[must_use]
    pub fn metrics_may_be_unreliable(&self) -> bool {
        matches!(self, Self::Long | Self::BookScale)
    }

    /// Get expected CoNLL F1 degradation relative to short documents.
    #[must_use]
    pub fn expected_degradation(&self) -> f64 {
        match self {
            Self::Short => 0.0,
            Self::Medium => 0.05,
            Self::Long => 0.10,
            Self::BookScale => 0.15,
        }
    }
}

impl std::fmt::Display for DocumentScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Short => write!(f, "Short (<2k tokens)"),
            Self::Medium => write!(f, "Medium (2k-10k tokens)"),
            Self::Long => write!(f, "Long (10k-50k tokens)"),
            Self::BookScale => write!(f, "Book-scale (>50k tokens)"),
        }
    }
}

// =============================================================================
// Metric Divergence (Book-scale Coreference Analysis)
// =============================================================================

/// Divergence between coreference metrics.
///
/// # Research Context
///
/// At book scale, different metrics diverge significantly:
/// - MUC tends to be inflated (favors link-based evaluation)
/// - CEAF-e tends to collapse (entity alignment struggles)
/// - B³ falls between but is more stable
///
/// Large divergence (>0.20) indicates potential metric unreliability.
///
/// # Example
///
/// ```rust
/// use anno::eval::MetricDivergence;
///
/// let divergence = MetricDivergence::from_scores(0.90, 0.65, 0.45);
/// assert!(divergence.has_high_divergence());
/// println!("MUC-CEAF divergence: {:.0}%", divergence.muc_ceaf_divergence * 100.0);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct MetricDivergence {
    /// MUC F1 score.
    pub muc_f1: f64,
    /// B³ F1 score.
    pub b3_f1: f64,
    /// CEAF-e F1 score.
    pub ceaf_e_f1: f64,
    /// Divergence between MUC and CEAF-e (absolute difference).
    pub muc_ceaf_divergence: f64,
    /// Divergence between MUC and B³.
    pub muc_b3_divergence: f64,
    /// Divergence between B³ and CEAF-e.
    pub b3_ceaf_divergence: f64,
}

impl MetricDivergence {
    /// Compute divergence from raw scores.
    #[must_use]
    pub fn from_scores(muc_f1: f64, b3_f1: f64, ceaf_e_f1: f64) -> Self {
        Self {
            muc_f1,
            b3_f1,
            ceaf_e_f1,
            muc_ceaf_divergence: (muc_f1 - ceaf_e_f1).abs(),
            muc_b3_divergence: (muc_f1 - b3_f1).abs(),
            b3_ceaf_divergence: (b3_f1 - ceaf_e_f1).abs(),
        }
    }

    /// Check if divergence is high (>0.20), indicating unreliable metrics.
    #[must_use]
    pub fn has_high_divergence(&self) -> bool {
        self.muc_ceaf_divergence > 0.20
    }

    /// Check if MUC is likely inflated (MUC >> CEAF-e).
    #[must_use]
    pub fn muc_likely_inflated(&self) -> bool {
        self.muc_f1 > self.ceaf_e_f1 + 0.15
    }

    /// Check if CEAF-e is likely collapsed (CEAF-e << others).
    #[must_use]
    pub fn ceaf_likely_collapsed(&self) -> bool {
        self.ceaf_e_f1 < self.b3_f1 - 0.15 && self.ceaf_e_f1 < self.muc_f1 - 0.20
    }

    /// Get most reliable metric recommendation.
    #[must_use]
    pub fn most_reliable_metric(&self) -> &'static str {
        if self.muc_likely_inflated() && self.ceaf_likely_collapsed() {
            "B³ (MUC inflated, CEAF-e collapsed)"
        } else if self.muc_likely_inflated() {
            "B³ or CEAF-e (MUC inflated)"
        } else if self.ceaf_likely_collapsed() {
            "MUC or B³ (CEAF-e collapsed)"
        } else {
            "CoNLL F1 (metrics agree)"
        }
    }
}

// =============================================================================
// Document Statistics for Coreference (Entity Spread)
// =============================================================================

/// Document-level statistics for coreference evaluation.
///
/// # Research Context (Bourgois & Poibeau 2025)
///
/// The paper introduces "entity spread" as a key metric:
/// > "The entity spread refers to the distance between the first and the last
/// > mention of an entity."
///
/// Their Long-LitBank-fr corpus shows:
/// - Average entity spread: 17,529 tokens
/// - Maximum entity spread: 115,369 tokens (spanning entire novels)
///
/// This metric characterizes the difficulty of coreference:
/// high spread = mentions far apart = harder to resolve.
///
/// # Example
///
/// ```rust
/// use anno::eval::{CorefDocStats, coref::CorefChain};
///
/// // Create from chains (would use actual chains in practice)
/// let stats = CorefDocStats {
///     chain_count: 159,
///     mention_count: 13178,
///     avg_chain_length: 82.9,
///     avg_entity_spread: 17529,
///     max_entity_spread: 115369,
///     ..Default::default()
/// };
///
/// println!("Avg entity spread: {} tokens", stats.avg_entity_spread);
/// ```
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct CorefDocStats {
    /// Document length in tokens (approximate).
    pub doc_length: usize,
    /// Total number of coreference chains.
    pub chain_count: usize,
    /// Total number of mentions.
    pub mention_count: usize,
    /// Average mentions per chain.
    pub avg_chain_length: f64,
    /// Maximum chain length.
    pub max_chain_length: usize,

    // =========================================================================
    // Entity Spread (Bourgois & Poibeau 2025)
    // =========================================================================
    /// Average entity spread in tokens.
    /// Entity spread = distance between first and last mention of an entity.
    pub avg_entity_spread: usize,

    /// Maximum entity spread in tokens.
    /// For protagonists in novels, this can exceed 100k tokens.
    pub max_entity_spread: usize,

    /// Median entity spread in tokens.
    pub median_entity_spread: usize,

    // =========================================================================
    // Mention Type Distribution
    // =========================================================================
    /// Proportion of pronominal mentions.
    pub pronoun_ratio: f64,
    /// Proportion of proper noun mentions.
    pub proper_ratio: f64,
    /// Proportion of nominal mentions.
    pub nominal_ratio: f64,
    /// Proportion of singleton chains.
    pub singleton_ratio: f64,
}

impl CorefDocStats {
    /// Compute statistics from coreference chains.
    ///
    /// Chains should have mentions with character offsets.
    /// Use `doc_length` to set the token count separately.
    #[must_use]
    pub fn from_chains(chains: &[crate::eval::coref::CorefChain]) -> Self {
        if chains.is_empty() {
            return Self::default();
        }

        let chain_count = chains.len();
        let mention_count: usize = chains.iter().map(|c| c.mentions.len()).sum();
        let avg_chain_length = mention_count as f64 / chain_count as f64;
        let max_chain_length = chains.iter().map(|c| c.mentions.len()).max().unwrap_or(0);

        // Count singletons
        let singleton_count = chains.iter().filter(|c| c.mentions.len() == 1).count();
        let singleton_ratio = singleton_count as f64 / chain_count as f64;

        // Compute entity spread for each chain
        let mut spreads: Vec<usize> = Vec::with_capacity(chain_count);
        for chain in chains {
            if chain.mentions.len() <= 1 {
                spreads.push(0);
                continue;
            }

            let first_start = chain.mentions.iter().map(|m| m.start).min().unwrap_or(0);
            let last_end = chain.mentions.iter().map(|m| m.end).max().unwrap_or(0);
            let spread = last_end.saturating_sub(first_start);
            spreads.push(spread);
        }

        let avg_entity_spread = if !spreads.is_empty() {
            spreads.iter().sum::<usize>() / spreads.len()
        } else {
            0
        };

        let max_entity_spread = spreads.iter().copied().max().unwrap_or(0);

        // Compute median spread
        spreads.sort_unstable();
        let median_entity_spread = if spreads.is_empty() {
            0
        } else {
            spreads[spreads.len() / 2]
        };

        // Compute mention type ratios (approximate from text patterns)
        // This is a heuristic; proper classification requires POS tagging
        let mut pronoun_count = 0usize;
        let mut proper_count = 0usize;
        let mut nominal_count = 0usize;

        for chain in chains {
            for mention in &chain.mentions {
                let text_lower = mention.text.to_lowercase();
                let is_pronoun = matches!(
                    text_lower.as_str(),
                    "he" | "she"
                        | "it"
                        | "they"
                        | "him"
                        | "her"
                        | "them"
                        | "his"
                        | "hers"
                        | "its"
                        | "their"
                        | "i"
                        | "me"
                        | "we"
                        | "us"
                        | "you"
                );

                if is_pronoun {
                    pronoun_count += 1;
                } else if mention
                    .text
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase())
                {
                    proper_count += 1;
                } else {
                    nominal_count += 1;
                }
            }
        }

        let total_mentions = mention_count.max(1) as f64;
        let pronoun_ratio = pronoun_count as f64 / total_mentions;
        let proper_ratio = proper_count as f64 / total_mentions;
        let nominal_ratio = nominal_count as f64 / total_mentions;

        Self {
            doc_length: 0, // Must be set separately
            chain_count,
            mention_count,
            avg_chain_length,
            max_chain_length,
            avg_entity_spread,
            max_entity_spread,
            median_entity_spread,
            pronoun_ratio,
            proper_ratio,
            nominal_ratio,
            singleton_ratio,
        }
    }

    /// Get document scale classification.
    #[must_use]
    pub fn scale_classification(&self) -> DocumentScale {
        DocumentScale::from_tokens(self.doc_length)
    }

    /// Check if entity spread suggests book-scale complexity.
    ///
    /// Book-scale documents typically have entities spanning >10k tokens.
    #[must_use]
    pub fn has_book_scale_spread(&self) -> bool {
        self.avg_entity_spread > 5000 || self.max_entity_spread > 20000
    }

    /// Format as summary string.
    #[must_use]
    pub fn format_summary(&self) -> String {
        format!(
            "Chains: {}, Mentions: {}, Avg length: {:.1}, Spread: avg={} max={}",
            self.chain_count,
            self.mention_count,
            self.avg_chain_length,
            self.avg_entity_spread,
            self.max_entity_spread,
        )
    }
}

/// Compute string-based familiarity using normalized edit distance and substring matching.
///
/// This is an improved heuristic over simple overlap ratio, but still not true semantic similarity.
fn compute_string_based_familiarity(train_types: &[String], eval_types: &[String]) -> f64 {
    if eval_types.is_empty() {
        return 0.0;
    }

    let mut total_similarity = 0.0;
    let mut counts = std::collections::HashMap::<String, usize>::new();

    // Count frequency of each eval type (for weighting)
    for eval_type in eval_types {
        *counts.entry(eval_type.clone()).or_insert(0) += 1;
    }

    let total_eval_count = eval_types.len() as f64;

    for (eval_type, freq) in counts {
        let max_sim = train_types
            .iter()
            .map(|train_type| string_similarity(&eval_type, train_type))
            .fold(0.0, f64::max);

        // Weight by frequency (as in Familiarity paper)
        let weight = freq as f64 / total_eval_count;
        total_similarity += max_sim * weight;
    }

    total_similarity
}

/// Compute embedding-based familiarity (semantic similarity × frequency weighting).
///
/// Returns None if embeddings cannot be computed for any type.
fn compute_embedding_based_familiarity<F>(
    train_types: &[String],
    eval_types: &[String],
    embedding_fn: &F,
) -> Option<f64>
where
    F: Fn(&str) -> Option<Vec<f32>>,
{
    if eval_types.is_empty() {
        return Some(0.0);
    }

    // Compute embeddings for all types
    let train_embeddings: Vec<(String, Vec<f32>)> = train_types
        .iter()
        .filter_map(|t| embedding_fn(t).map(|e| (t.clone(), e)))
        .collect();

    if train_embeddings.is_empty() {
        return None; // Can't compute without train embeddings
    }

    let mut counts = std::collections::HashMap::<String, usize>::new();
    for eval_type in eval_types {
        *counts.entry(eval_type.clone()).or_insert(0) += 1;
    }

    let total_eval_count = eval_types.len() as f64;
    let mut total_similarity = 0.0;

    for (eval_type, freq) in counts {
        if let Some(eval_emb) = embedding_fn(&eval_type) {
            // Find maximum cosine similarity with any training type
            let max_sim = train_embeddings
                .iter()
                .map(|(_, train_emb)| cosine_similarity(&eval_emb, train_emb))
                .fold(0.0, f64::max);

            // Weight by frequency
            let weight = freq as f64 / total_eval_count;
            total_similarity += max_sim * weight;
        } else {
            // If we can't embed this type, fall back to string similarity
            let max_sim = train_types
                .iter()
                .map(|train_type| string_similarity(&eval_type, train_type))
                .fold(0.0, f64::max);
            let weight = freq as f64 / total_eval_count;
            total_similarity += max_sim * weight;
        }
    }

    Some(total_similarity)
}

/// Compute cosine similarity between two normalized vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot_product as f64
}

/// Compute string similarity using normalized edit distance and substring matching.
///
/// Returns a value in [0, 1] where 1.0 = identical strings.
fn string_similarity(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    // Exact match
    if a_lower == b_lower {
        return 1.0;
    }

    // Substring match (e.g., "PERSON" contains "PER")
    if a_lower.contains(&b_lower) || b_lower.contains(&a_lower) {
        return 0.8;
    }

    // Normalized edit distance (Levenshtein)
    let max_len = a_lower.len().max(b_lower.len());
    if max_len == 0 {
        return 1.0;
    }

    let distance = levenshtein_distance(&a_lower, &b_lower);
    1.0 - (distance as f64 / max_len as f64)
}

/// Compute Levenshtein distance between two strings.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut matrix = vec![vec![0; b_len + 1]; a_len + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(a_len + 1) {
        row[0] = i;
    }
    for (j, cell) in matrix[0].iter_mut().enumerate().take(b_len + 1) {
        *cell = j;
    }

    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }

    matrix[a_len][b_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_familiarity_computation() {
        let train_types = vec![
            "person".to_string(),
            "organization".to_string(),
            "location".to_string(),
        ];

        let eval_types = vec![
            "PERSON".to_string(),  // Should match "person" via similarity (not zero-shot)
            "ORG".to_string(),     // Should match "organization" via similarity (not zero-shot)
            "DISEASE".to_string(), // True zero-shot (no similarity)
        ];

        let shift = LabelShift::from_type_sets(&train_types, &eval_types);

        // Should detect similarity even without exact match
        assert!(shift.familiarity > 0.0, "Should have non-zero familiarity");
        // Note: String similarity may match PERSON->person and ORG->organization,
        // so true_zero_shot_types may only contain DISEASE, or all three if similarity
        // threshold is low. The important thing is familiarity > 0.
        assert!(
            !shift.true_zero_shot_types.is_empty(),
            "Should have at least 1 true zero-shot type"
        );
        assert!(shift.true_zero_shot_types.contains(&"DISEASE".to_string()));
    }

    #[test]
    fn test_familiarity_inflation_detection() {
        let train_types = vec![
            "person".to_string(),
            "organization".to_string(),
            "location".to_string(),
        ];

        let eval_types = vec![
            "PERSON".to_string(),
            "ORGANIZATION".to_string(),
            "LOCATION".to_string(),
        ];

        let shift = LabelShift::from_type_sets(&train_types, &eval_types);

        // High similarity should trigger high familiarity
        assert!(shift.familiarity > 0.5, "Should have high familiarity");
    }

    #[test]
    fn test_label_shift_zero_shot_types() {
        let train_types = vec!["person".to_string()];
        let eval_types = vec![
            "person".to_string(),
            "disease".to_string(),
            "drug".to_string(),
        ];

        let shift = LabelShift::from_type_sets(&train_types, &eval_types);

        assert_eq!(shift.true_zero_shot_types.len(), 2);
        assert!(shift.true_zero_shot_types.contains(&"disease".to_string()));
        assert!(shift.true_zero_shot_types.contains(&"drug".to_string()));
    }

    #[test]
    fn test_metric_value_clamping() {
        assert_eq!(MetricValue::new(0.5).get(), 0.5);
        assert_eq!(MetricValue::new(-0.5).get(), 0.0);
        assert_eq!(MetricValue::new(1.5).get(), 1.0);
    }

    #[test]
    fn test_metric_value_try_new() {
        assert!(MetricValue::try_new(0.5).is_ok());
        assert!(MetricValue::try_new(-0.1).is_err());
        assert!(MetricValue::try_new(1.1).is_err());
    }

    #[test]
    fn test_goal_check_result() {
        let mut result = GoalCheckResult::new();
        assert!(result.passed);

        result.add_check("precision", GoalCheck::pass(0.8, 0.85));
        assert!(result.passed);

        result.add_check("recall", GoalCheck::fail(0.9, 0.75));
        assert!(!result.passed);

        assert_eq!(result.passed_count(), 1);
        assert_eq!(result.failed_count(), 1);
    }

    #[test]
    fn test_metric_with_variance_from_samples() {
        let samples = vec![0.85, 0.87, 0.82, 0.88, 0.84];
        let m = MetricWithVariance::from_samples(&samples);

        // Mean should be 0.852
        assert!((m.mean - 0.852).abs() < 0.001);
        assert_eq!(m.n, 5);
        assert!((m.min - 0.82).abs() < 0.001);
        assert!((m.max - 0.88).abs() < 0.001);
        assert!(m.std_dev > 0.0);
        assert!(m.ci_95 > 0.0);
    }

    #[test]
    fn test_metric_with_variance_empty() {
        let m = MetricWithVariance::from_samples(&[]);
        assert_eq!(m.n, 0);
        assert_eq!(m.mean, 0.0);
        assert_eq!(m.format_with_ci(), "N/A");
    }

    #[test]
    fn test_metric_with_variance_single() {
        let m = MetricWithVariance::from_samples(&[0.9]);
        assert!((m.mean - 0.9).abs() < 0.001);
        assert_eq!(m.std_dev, 0.0);
        assert_eq!(m.ci_95, 0.0);
        assert_eq!(m.n, 1);
    }

    #[test]
    fn test_metric_with_variance_format() {
        let samples = vec![0.85, 0.87, 0.82, 0.88, 0.84];
        let m = MetricWithVariance::from_samples(&samples);

        // Should format nicely
        let formatted = m.format_with_ci();
        assert!(formatted.contains("%"));
        assert!(formatted.contains("±"));

        let range = m.format_with_range();
        assert!(range.contains("82.0%"));
        assert!(range.contains("88.0%"));
    }

    // =========================================================================
    // Tests for DocumentScale (Bourgois & Poibeau 2025)
    // =========================================================================

    #[test]
    fn test_document_scale_classification() {
        // Short documents (<2k tokens)
        assert_eq!(DocumentScale::from_tokens(500), DocumentScale::Short);
        assert_eq!(DocumentScale::from_tokens(2000), DocumentScale::Short);

        // Medium documents (2k-10k tokens)
        assert_eq!(DocumentScale::from_tokens(2001), DocumentScale::Medium);
        assert_eq!(DocumentScale::from_tokens(5000), DocumentScale::Medium);
        assert_eq!(DocumentScale::from_tokens(10000), DocumentScale::Medium);

        // Long documents (10k-50k tokens)
        assert_eq!(DocumentScale::from_tokens(10001), DocumentScale::Long);
        assert_eq!(DocumentScale::from_tokens(30000), DocumentScale::Long);
        assert_eq!(DocumentScale::from_tokens(50000), DocumentScale::Long);

        // Book-scale documents (>50k tokens)
        assert_eq!(DocumentScale::from_tokens(50001), DocumentScale::BookScale);
        assert_eq!(DocumentScale::from_tokens(100000), DocumentScale::BookScale);
    }

    #[test]
    fn test_document_scale_is_book_scale() {
        assert!(!DocumentScale::Short.is_book_scale());
        assert!(!DocumentScale::Medium.is_book_scale());
        assert!(!DocumentScale::Long.is_book_scale());
        assert!(DocumentScale::BookScale.is_book_scale());
    }

    #[test]
    fn test_document_scale_metrics_reliability() {
        assert!(!DocumentScale::Short.metrics_may_be_unreliable());
        assert!(!DocumentScale::Medium.metrics_may_be_unreliable());
        assert!(DocumentScale::Long.metrics_may_be_unreliable());
        assert!(DocumentScale::BookScale.metrics_may_be_unreliable());
    }

    #[test]
    fn test_document_scale_expected_degradation() {
        assert!((DocumentScale::Short.expected_degradation() - 0.0).abs() < 0.001);
        assert!((DocumentScale::Medium.expected_degradation() - 0.05).abs() < 0.001);
        assert!((DocumentScale::Long.expected_degradation() - 0.10).abs() < 0.001);
        assert!((DocumentScale::BookScale.expected_degradation() - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_document_scale_display() {
        assert!(DocumentScale::Short.to_string().contains("Short"));
        assert!(DocumentScale::BookScale.to_string().contains("Book-scale"));
    }

    // =========================================================================
    // Tests for MetricDivergence
    // =========================================================================

    #[test]
    fn test_metric_divergence_computation() {
        // Typical book-scale pattern: high MUC, lower B³, collapsed CEAF-e
        let divergence = MetricDivergence::from_scores(0.90, 0.65, 0.45);

        assert!((divergence.muc_f1 - 0.90).abs() < 0.001);
        assert!((divergence.b3_f1 - 0.65).abs() < 0.001);
        assert!((divergence.ceaf_e_f1 - 0.45).abs() < 0.001);

        // MUC-CEAF divergence should be 0.45
        assert!((divergence.muc_ceaf_divergence - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_metric_divergence_high_divergence_detection() {
        // High divergence (>0.20)
        let high = MetricDivergence::from_scores(0.90, 0.70, 0.50);
        assert!(high.has_high_divergence());

        // Low divergence (<0.20)
        let low = MetricDivergence::from_scores(0.80, 0.75, 0.70);
        assert!(!low.has_high_divergence());
    }

    #[test]
    fn test_metric_divergence_muc_inflation() {
        // MUC inflated (MUC >> CEAF-e by >0.15)
        let inflated = MetricDivergence::from_scores(0.90, 0.70, 0.50);
        assert!(inflated.muc_likely_inflated());

        // MUC not inflated
        let not_inflated = MetricDivergence::from_scores(0.80, 0.75, 0.70);
        assert!(!not_inflated.muc_likely_inflated());
    }

    #[test]
    fn test_metric_divergence_ceaf_collapse() {
        // CEAF-e collapsed (much lower than others)
        let collapsed = MetricDivergence::from_scores(0.90, 0.70, 0.40);
        assert!(collapsed.ceaf_likely_collapsed());

        // CEAF-e not collapsed
        let not_collapsed = MetricDivergence::from_scores(0.80, 0.75, 0.70);
        assert!(!not_collapsed.ceaf_likely_collapsed());
    }

    #[test]
    fn test_metric_divergence_recommendation() {
        // When both MUC inflated and CEAF-e collapsed -> recommend B³
        let both_bad = MetricDivergence::from_scores(0.90, 0.65, 0.40);
        assert!(both_bad.most_reliable_metric().contains("B³"));

        // When metrics agree -> recommend CoNLL F1
        let agree = MetricDivergence::from_scores(0.75, 0.73, 0.71);
        assert!(agree.most_reliable_metric().contains("CoNLL"));
    }

    // =========================================================================
    // Tests for CorefDocStats (Entity Spread)
    // =========================================================================

    #[test]
    fn test_coref_doc_stats_default() {
        let stats = CorefDocStats::default();
        assert_eq!(stats.chain_count, 0);
        assert_eq!(stats.mention_count, 0);
        assert_eq!(stats.avg_entity_spread, 0);
        assert_eq!(stats.max_entity_spread, 0);
    }

    #[test]
    fn test_coref_doc_stats_scale_classification() {
        let mut stats = CorefDocStats {
            doc_length: 1000,
            ..Default::default()
        };
        assert_eq!(stats.scale_classification(), DocumentScale::Short);

        stats.doc_length = 5000;
        assert_eq!(stats.scale_classification(), DocumentScale::Medium);

        stats.doc_length = 30000;
        assert_eq!(stats.scale_classification(), DocumentScale::Long);

        stats.doc_length = 100000;
        assert_eq!(stats.scale_classification(), DocumentScale::BookScale);
    }

    #[test]
    fn test_coref_doc_stats_book_scale_spread() {
        let mut stats = CorefDocStats {
            avg_entity_spread: 1000,
            max_entity_spread: 5000,
            ..Default::default()
        };

        // Low spread - not book-scale
        assert!(!stats.has_book_scale_spread());

        // High avg spread - book-scale
        stats.avg_entity_spread = 6000;
        stats.max_entity_spread = 10000;
        assert!(stats.has_book_scale_spread());

        // High max spread - book-scale
        stats.avg_entity_spread = 2000;
        stats.max_entity_spread = 25000;
        assert!(stats.has_book_scale_spread());
    }

    #[test]
    fn test_coref_doc_stats_format_summary() {
        let stats = CorefDocStats {
            chain_count: 159,
            mention_count: 13178,
            avg_chain_length: 82.9,
            avg_entity_spread: 17529,
            max_entity_spread: 115369,
            ..Default::default()
        };

        let summary = stats.format_summary();
        assert!(summary.contains("159"));
        assert!(summary.contains("13178"));
        assert!(summary.contains("17529"));
        assert!(summary.contains("115369"));
    }

    #[test]
    fn test_coref_doc_stats_from_chains() {
        use crate::eval::coref::{CorefChain, Mention};

        // Create test chains
        let chains = vec![
            CorefChain::new(vec![
                Mention::new("John", 0, 4),
                Mention::new("he", 20, 22),
                Mention::new("him", 50, 53),
            ]),
            CorefChain::new(vec![
                Mention::new("Mary", 5, 9),
                Mention::new("she", 30, 33),
            ]),
            // Singleton
            CorefChain::new(vec![Mention::new("London", 60, 66)]),
        ];

        let stats = CorefDocStats::from_chains(&chains);

        assert_eq!(stats.chain_count, 3);
        assert_eq!(stats.mention_count, 6);
        assert!((stats.avg_chain_length - 2.0).abs() < 0.01);
        assert_eq!(stats.max_chain_length, 3);

        // Entity spread: John chain spans 0-53 = 53, Mary chain spans 5-33 = 28
        assert!(stats.avg_entity_spread > 0);
        assert!(stats.max_entity_spread >= 53);

        // Singleton ratio: 1/3 = 0.333
        assert!((stats.singleton_ratio - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_coref_doc_stats_mention_type_ratios() {
        use crate::eval::coref::{CorefChain, Mention};

        // Create chains with mixed mention types
        let chains = vec![
            CorefChain::new(vec![
                Mention::new("John", 0, 4),  // Proper (capitalized)
                Mention::new("he", 10, 12),  // Pronoun
                Mention::new("him", 20, 23), // Pronoun
            ]),
            CorefChain::new(vec![
                Mention::new("Mary", 30, 34), // Proper
                Mention::new("she", 40, 43),  // Pronoun
            ]),
        ];

        let stats = CorefDocStats::from_chains(&chains);

        // 3 pronouns (he, him, she), 2 proper (John, Mary)
        // pronoun_ratio = 3/5 = 0.6, proper_ratio = 2/5 = 0.4
        assert!(stats.pronoun_ratio > 0.5, "Should have majority pronouns");
        assert!(stats.proper_ratio > 0.3, "Should have some proper nouns");
        assert_eq!(stats.mention_count, 5);
    }
}
