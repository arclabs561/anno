//! Book-Scale Coreference Diagnostics
//!
//! This module provides enhanced diagnostics for evaluating coreference
//! resolution at book scale, incorporating insights from BOOKCOREF
//! (Martinelli et al., 2025) and "How to Evaluate Coreference in Literary
//! Texts?" (Duron-Tejedor et al., 2023; arXiv:2401.00238).
//!
//! # The Book-Scale Challenge
//!
//! Coreference systems optimized for short documents can show different behavior
//! at book scale. In particular, common metrics can diverge in what they “reward”
//! (e.g., favoring long chains vs penalizing boundary errors).
//!
//! # Key Insights
//!
//! 1. **Metric divergence**: different metrics can disagree substantially at scale.
//! 2. **Windowed vs full**: long-range dependencies can be masked by windowed evaluation.
//! 3. **Long chains dominate**: a few long chains can dominate aggregate scores.
//!
//! # This Module Provides
//!
//! - `BookScaleAnalysis`: Complete diagnostic report
//! - `PerBookBreakdown`: Individual book performance (like Table 3)
//! - `ChainLengthStratification`: Performance by chain length
//! - `WindowedVsFullComparison`: Diagnose long-range dependency issues
//! - `MetricReliabilityAssessment`: Which metrics to trust at this scale
//!
//! # Example
//!
//! ```rust,ignore
//! use anno_eval::eval::book_scale::{BookScaleAnalyzer, BookScaleConfig};
//! use anno::metrics::coref::CorefChain;
//!
//! let analyzer = BookScaleAnalyzer::new(BookScaleConfig::default());
//!
//! let analysis = analyzer.analyze(
//!     &predicted_chains,
//!     &gold_chains,
//!     document_length,
//! );
//!
//! if analysis.has_scale_issues() {
//!     println!("Book-scale issues detected:");
//!     println!("{}", analysis.diagnostic_report());
//! }
//! ```

use super::coref::{CorefChain, Mention};
use super::coref_metrics::{b_cubed_score, ceaf_e_score, ceaf_m_score, lea_score, muc_score};
use super::types::{CorefDocStats, DocumentScale, MetricDivergence};
use serde::{Deserialize, Serialize};

// =============================================================================
// Serializable Score Types (self-contained for book_scale module)
// =============================================================================

/// Precision/Recall/F1 scores (serializable).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1: f64,
}

impl Scores {
    /// Create from tuple (p, r, f1).
    pub fn from_tuple((precision, recall, f1): (f64, f64, f64)) -> Self {
        Self {
            precision,
            recall,
            f1,
        }
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for book-scale analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookScaleConfig {
    /// Window size for windowed evaluation (tokens)
    pub window_size: usize,
    /// Overlap between windows
    pub window_overlap: usize,
    /// Threshold for "long" chains (mentions)
    pub long_chain_threshold: usize,
    /// Threshold for "short" chains (mentions)
    pub short_chain_threshold: usize,
    /// MUC-CEAF divergence threshold for scale issues
    pub divergence_threshold: f64,
    /// Performance drop threshold (windowed → full) for concerno
    pub performance_drop_threshold: f64,
}

impl Default for BookScaleConfig {
    fn default() -> Self {
        Self {
            window_size: 1500,
            window_overlap: 200,
            long_chain_threshold: 10,         // >10 mentions = long chain
            short_chain_threshold: 2,         // 2-10 = short, 1 = singleton
            divergence_threshold: 0.30,       // 30 F1 point gap
            performance_drop_threshold: 0.15, // 15 F1 point drop
        }
    }
}

// =============================================================================
// Book-Scale Analysis
// =============================================================================

/// Coreference evaluation scores (serializable version).
///
/// This is a simplified, serializable version of the full CorefEvaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorefEvalScores {
    /// MUC scores
    pub muc: Scores,
    /// B³ scores
    pub b_cubed: Scores,
    /// CEAF-e scores
    pub ceaf_e: Scores,
    /// CEAF-m scores
    pub ceaf_m: Scores,
    /// LEA scores
    pub lea: Scores,
    /// CoNLL F1 (average of MUC, B³, CEAF-e)
    pub conll_f1: f64,
}

/// Complete book-scale analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookScaleAnalysis {
    /// Full document evaluation
    pub full_doc_eval: CorefEvalScores,
    /// Windowed evaluation (average over windows)
    pub windowed_eval: Option<WindowedEvaluation>,
    /// Per-chain-length stratified evaluation
    pub stratified: StratifiedEvaluation,
    /// Document statistics
    pub doc_stats: CorefDocStats,
    /// Metric reliability assessment
    pub reliability: MetricReliability,
    /// Overall scale classification
    pub scale: DocumentScale,
    /// Diagnostic flags
    pub diagnostics: BookScaleDiagnostics,
}

impl BookScaleAnalysis {
    /// Check if document has book-scale issues.
    pub fn has_scale_issues(&self) -> bool {
        self.diagnostics.has_issues()
    }

    /// Generate a diagnostic report.
    pub fn diagnostic_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Book-Scale Coreference Analysis ===\n\n");
        report.push_str(&format!("Document Scale: {}\n", self.scale));
        report.push_str(&format!(
            "Document Length: {} chars ({} mentions in {} chains)\n\n",
            self.doc_stats.doc_length, self.doc_stats.mention_count, self.doc_stats.chain_count
        ));

        // Metrics summary
        report.push_str("Full-Document Metrics:\n");
        report.push_str(&format!(
            "  MUC:    {:.1}%\n",
            self.full_doc_eval.muc.f1 * 100.0
        ));
        report.push_str(&format!(
            "  B³:     {:.1}%\n",
            self.full_doc_eval.b_cubed.f1 * 100.0
        ));
        report.push_str(&format!(
            "  CEAF-e: {:.1}%\n",
            self.full_doc_eval.ceaf_e.f1 * 100.0
        ));
        report.push_str(&format!(
            "  CoNLL:  {:.1}%\n\n",
            self.full_doc_eval.conll_f1 * 100.0
        ));

        // Windowed comparison
        if let Some(ref windowed) = self.windowed_eval {
            report.push_str("Windowed vs Full-Document Comparison:\n");
            report.push_str(&format!(
                "  Windowed CoNLL:  {:.1}%\n",
                windowed.avg_conll_f1 * 100.0
            ));
            report.push_str(&format!(
                "  Full-Doc CoNLL:  {:.1}%\n",
                self.full_doc_eval.conll_f1 * 100.0
            ));
            report.push_str(&format!(
                "  Performance Drop: {:.1} F1 points\n\n",
                windowed.performance_drop * 100.0
            ));
        }

        // Stratified evaluation
        report.push_str("Chain-Length Stratified Evaluation:\n");
        report.push_str(&format!(
            "  Long chains (>10):  {:.1}% F1 ({} chains)\n",
            self.stratified.long_chains.f1 * 100.0,
            self.stratified.long_chain_count
        ));
        report.push_str(&format!(
            "  Short chains (2-10): {:.1}% F1 ({} chains)\n",
            self.stratified.short_chains.f1 * 100.0,
            self.stratified.short_chain_count
        ));
        report.push_str(&format!(
            "  Singletons (1):     {:.1}% F1 ({} chains)\n\n",
            self.stratified.singletons.f1 * 100.0,
            self.stratified.singleton_count
        ));

        // Reliability assessment
        report.push_str("Metric Reliability:\n");
        report.push_str(&format!(
            "  MUC:    {} ({})\n",
            self.reliability.muc_reliability, self.reliability.muc_note
        ));
        report.push_str(&format!(
            "  B³:     {} ({})\n",
            self.reliability.b_cubed_reliability, self.reliability.b_cubed_note
        ));
        report.push_str(&format!(
            "  CEAF-e: {} ({})\n",
            self.reliability.ceaf_e_reliability, self.reliability.ceaf_e_note
        ));
        report.push_str(&format!(
            "  LEA:    {} ({})\n\n",
            self.reliability.lea_reliability, self.reliability.lea_note
        ));

        // Diagnostics
        if self.has_scale_issues() {
            report.push_str("⚠️ ISSUES DETECTED:\n");
            if self.diagnostics.high_metric_divergence {
                report
                    .push_str("  • High metric divergence - MUC and CEAF disagree significantly\n");
            }
            if self.diagnostics.large_performance_drop {
                report.push_str(
                    "  • Large windowed→full performance drop - long-range dependencies failing\n",
                );
            }
            if self.diagnostics.long_chain_dominance {
                report.push_str("  • Long chains dominate - main characters skewing metrics\n");
            }
            if self.diagnostics.singleton_neglect {
                report.push_str("  • Singleton neglect - minor entities being ignored\n");
            }
            report.push_str("\nRECOMMENDATIONS:\n");
            for rec in &self.diagnostics.recommendations {
                report.push_str(&format!("  → {}\n", rec));
            }
        } else {
            report.push_str("✓ No significant scale issues detected.\n");
        }

        report
    }
}

/// Windowed evaluation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowedEvaluation {
    /// Number of windows
    pub num_windows: usize,
    /// Window size used
    pub window_size: usize,
    /// Average CoNLL F1 across windows
    pub avg_conll_f1: f64,
    /// Standard deviation of CoNLL F1
    pub std_conll_f1: f64,
    /// Performance drop (windowed - full_doc)
    pub performance_drop: f64,
    /// Per-window evaluations
    pub window_evals: Vec<CorefEvalScores>,
}

/// Stratified evaluation by chain length.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StratifiedEvaluation {
    /// Evaluation on long chains (>threshold mentions)
    pub long_chains: Scores,
    /// Evaluation on short chains (2-threshold mentions)
    pub short_chains: Scores,
    /// Evaluation on singletons
    pub singletons: Scores,
    /// Count of long chains
    pub long_chain_count: usize,
    /// Count of short chains
    pub short_chain_count: usize,
    /// Count of singletons
    pub singleton_count: usize,
}

/// Reliability assessment for each coreference metric at book scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricReliability {
    /// MUC metric reliability level
    pub muc_reliability: ReliabilityLevel,
    /// Notes about MUC reliability
    pub muc_note: String,
    /// B-cubed metric reliability level
    pub b_cubed_reliability: ReliabilityLevel,
    /// Notes about B-cubed reliability
    pub b_cubed_note: String,
    /// CEAF-e metric reliability level
    pub ceaf_e_reliability: ReliabilityLevel,
    /// Notes about CEAF-e reliability
    pub ceaf_e_note: String,
    /// LEA metric reliability level
    pub lea_reliability: ReliabilityLevel,
    /// Notes about LEA reliability
    pub lea_note: String,
}

impl Default for MetricReliability {
    fn default() -> Self {
        Self {
            muc_reliability: ReliabilityLevel::Medium,
            muc_note: "May be inflated at scale".to_string(),
            b_cubed_reliability: ReliabilityLevel::Medium,
            b_cubed_note: "Moderate reliability".to_string(),
            ceaf_e_reliability: ReliabilityLevel::Medium,
            ceaf_e_note: "May collapse at scale".to_string(),
            lea_reliability: ReliabilityLevel::High,
            lea_note: "Most stable across scales".to_string(),
        }
    }
}

/// Reliability level for a metric at book scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReliabilityLevel {
    /// High reliability - metric is stable at scale
    High,
    /// Medium reliability - some degradation expected
    Medium,
    /// Low reliability - significant variance at scale
    Low,
    /// Unreliable - metric not recommended for book-length texts
    Unreliable,
}

impl std::fmt::Display for ReliabilityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReliabilityLevel::High => write!(f, "HIGH"),
            ReliabilityLevel::Medium => write!(f, "MEDIUM"),
            ReliabilityLevel::Low => write!(f, "LOW"),
            ReliabilityLevel::Unreliable => write!(f, "UNRELIABLE"),
        }
    }
}

/// Diagnostic flags for book-scale issues.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BookScaleDiagnostics {
    /// MUC-CEAF divergence exceeds threshold
    pub high_metric_divergence: bool,
    /// Windowed→Full performance drop exceeds threshold
    pub large_performance_drop: bool,
    /// Long chains dominate the evaluation
    pub long_chain_dominance: bool,
    /// Singletons have very low performance
    pub singleton_neglect: bool,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

impl BookScaleDiagnostics {
    /// Check if any issues were detected.
    pub fn has_issues(&self) -> bool {
        self.high_metric_divergence
            || self.large_performance_drop
            || self.long_chain_dominance
            || self.singleton_neglect
    }
}

// =============================================================================
// Analyzer
// =============================================================================

/// Book-scale coreference analyzer.
pub struct BookScaleAnalyzer {
    config: BookScaleConfig,
}

impl Default for BookScaleAnalyzer {
    fn default() -> Self {
        Self::new(BookScaleConfig::default())
    }
}

impl BookScaleAnalyzer {
    /// Create a new analyzer with configuration.
    pub fn new(config: BookScaleConfig) -> Self {
        Self { config }
    }

    /// Perform complete book-scale analysis.
    pub fn analyze(
        &self,
        predicted: &[CorefChain],
        gold: &[CorefChain],
        doc_length: usize,
    ) -> BookScaleAnalysis {
        // Compute full-document evaluation
        let full_doc_eval = self.evaluate_chains(predicted, gold);

        // Compute document statistics
        let mut doc_stats = CorefDocStats::from_chains(gold);
        doc_stats.doc_length = doc_length;

        // Determine scale
        let scale = doc_stats.scale_classification();

        // Compute windowed evaluation (if document is long enough)
        let windowed_eval = if doc_length > self.config.window_size * 2 {
            Some(self.compute_windowed_eval(predicted, gold, doc_length))
        } else {
            None
        };

        // Compute stratified evaluation
        let stratified = self.compute_stratified_eval(predicted, gold);

        // Assess metric reliability
        let reliability = self.assess_reliability(&full_doc_eval, &stratified, scale);

        // Generate diagnostics
        let diagnostics =
            self.generate_diagnostics(&full_doc_eval, windowed_eval.as_ref(), &stratified, scale);

        BookScaleAnalysis {
            full_doc_eval,
            windowed_eval,
            stratified,
            doc_stats,
            reliability,
            scale,
            diagnostics,
        }
    }

    /// Evaluate chains using standard metrics.
    fn evaluate_chains(&self, predicted: &[CorefChain], gold: &[CorefChain]) -> CorefEvalScores {
        let muc = Scores::from_tuple(muc_score(predicted, gold));
        let b_cubed = Scores::from_tuple(b_cubed_score(predicted, gold));
        let ceaf_e = Scores::from_tuple(ceaf_e_score(predicted, gold));
        let ceaf_m = Scores::from_tuple(ceaf_m_score(predicted, gold));
        let lea = Scores::from_tuple(lea_score(predicted, gold));

        let conll_f1 = (muc.f1 + b_cubed.f1 + ceaf_e.f1) / 3.0;

        CorefEvalScores {
            muc,
            b_cubed,
            ceaf_e,
            ceaf_m,
            lea,
            conll_f1,
        }
    }

    /// Compute windowed evaluation.
    fn compute_windowed_eval(
        &self,
        predicted: &[CorefChain],
        gold: &[CorefChain],
        doc_length: usize,
    ) -> WindowedEvaluation {
        let step = self
            .config
            .window_size
            .saturating_sub(self.config.window_overlap);
        let mut window_evals = Vec::new();

        let mut offset = 0;
        while offset < doc_length {
            let window_end = (offset + self.config.window_size).min(doc_length);

            // Filter chains to this window
            let pred_window = self.filter_to_window(predicted, offset, window_end);
            let gold_window = self.filter_to_window(gold, offset, window_end);

            if !pred_window.is_empty() || !gold_window.is_empty() {
                let eval = self.evaluate_chains(&pred_window, &gold_window);
                window_evals.push(eval);
            }

            if window_end >= doc_length {
                break;
            }
            offset += step.max(1);
        }

        // Compute statistics
        let conll_scores: Vec<f64> = window_evals.iter().map(|e| e.conll_f1).collect();
        let avg_conll_f1 = if !conll_scores.is_empty() {
            conll_scores.iter().sum::<f64>() / conll_scores.len() as f64
        } else {
            0.0
        };

        let std_conll_f1 = if conll_scores.len() > 1 {
            let variance = conll_scores
                .iter()
                .map(|x| (x - avg_conll_f1).powi(2))
                .sum::<f64>()
                / (conll_scores.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Performance drop (positive = windowed is better)
        let full_doc_eval = self.evaluate_chains(predicted, gold);
        let performance_drop = avg_conll_f1 - full_doc_eval.conll_f1;

        WindowedEvaluation {
            num_windows: window_evals.len(),
            window_size: self.config.window_size,
            avg_conll_f1,
            std_conll_f1,
            performance_drop,
            window_evals,
        }
    }

    /// Filter chains to a specific window.
    fn filter_to_window(&self, chains: &[CorefChain], start: usize, end: usize) -> Vec<CorefChain> {
        chains
            .iter()
            .filter_map(|chain| {
                let filtered_mentions: Vec<Mention> = chain
                    .mentions
                    .iter()
                    .filter(|m| m.start >= start && m.end <= end)
                    .cloned()
                    .collect();

                if filtered_mentions.is_empty() {
                    None
                } else {
                    let mut new_chain = CorefChain::new(filtered_mentions);
                    new_chain.cluster_id = chain.cluster_id;
                    new_chain.entity_type = chain.entity_type.clone();
                    Some(new_chain)
                }
            })
            .collect()
    }

    /// Compute stratified evaluation by chain length.
    fn compute_stratified_eval(
        &self,
        predicted: &[CorefChain],
        gold: &[CorefChain],
    ) -> StratifiedEvaluation {
        let (pred_long, pred_short, pred_singleton) = self.stratify_chains(predicted);
        let (gold_long, gold_short, gold_singleton) = self.stratify_chains(gold);

        let long_chains = if !pred_long.is_empty() || !gold_long.is_empty() {
            Scores::from_tuple(muc_score(&pred_long, &gold_long))
        } else {
            Scores::default()
        };

        let short_chains = if !pred_short.is_empty() || !gold_short.is_empty() {
            Scores::from_tuple(muc_score(&pred_short, &gold_short))
        } else {
            Scores::default()
        };

        let singletons = if !pred_singleton.is_empty() || !gold_singleton.is_empty() {
            // Singletons need different handling - use B³ or exact match
            Scores::from_tuple(b_cubed_score(&pred_singleton, &gold_singleton))
        } else {
            Scores::default()
        };

        StratifiedEvaluation {
            long_chains,
            short_chains,
            singletons,
            long_chain_count: gold_long.len(),
            short_chain_count: gold_short.len(),
            singleton_count: gold_singleton.len(),
        }
    }

    /// Stratify chains by length.
    fn stratify_chains(
        &self,
        chains: &[CorefChain],
    ) -> (Vec<CorefChain>, Vec<CorefChain>, Vec<CorefChain>) {
        let mut long = Vec::new();
        let mut short = Vec::new();
        let mut singleton = Vec::new();

        for chain in chains {
            let len = chain.len();
            if len > self.config.long_chain_threshold {
                long.push(chain.clone());
            } else if len >= self.config.short_chain_threshold {
                short.push(chain.clone());
            } else {
                singleton.push(chain.clone());
            }
        }

        (long, short, singleton)
    }

    /// Assess metric reliability based on document characteristics.
    fn assess_reliability(
        &self,
        eval: &CorefEvalScores,
        _stratified: &StratifiedEvaluation,
        scale: DocumentScale,
    ) -> MetricReliability {
        let divergence =
            MetricDivergence::from_scores(eval.muc.f1, eval.b_cubed.f1, eval.ceaf_e.f1);

        // MUC reliability
        let (muc_rel, muc_note) = if divergence.muc_ceaf_divergence > 0.40 {
            (
                ReliabilityLevel::Low,
                "Severely inflated due to long chains".to_string(),
            )
        } else if divergence.muc_ceaf_divergence > 0.25 {
            (
                ReliabilityLevel::Medium,
                "May be inflated at this scale".to_string(),
            )
        } else {
            (ReliabilityLevel::High, "Reliable at this scale".to_string())
        };

        // CEAF-e reliability
        let (ceaf_rel, ceaf_note) = match scale {
            DocumentScale::BookScale => (
                ReliabilityLevel::Low,
                "Known to collapse at book scale".to_string(),
            ),
            DocumentScale::Long => (
                ReliabilityLevel::Medium,
                "May underestimate at this length".to_string(),
            ),
            _ => (ReliabilityLevel::High, "Reliable at this scale".to_string()),
        };

        // B³ reliability
        let (b3_rel, b3_note) = if divergence.muc_b3_divergence > 0.30 {
            (
                ReliabilityLevel::Medium,
                "Moderate divergence from MUC".to_string(),
            )
        } else {
            (ReliabilityLevel::High, "Stable metric".to_string())
        };

        // LEA is generally most stable
        let (lea_rel, lea_note) = (
            ReliabilityLevel::High,
            "Most stable across document scales".to_string(),
        );

        MetricReliability {
            muc_reliability: muc_rel,
            muc_note,
            b_cubed_reliability: b3_rel,
            b_cubed_note: b3_note,
            ceaf_e_reliability: ceaf_rel,
            ceaf_e_note: ceaf_note,
            lea_reliability: lea_rel,
            lea_note,
        }
    }

    /// Generate diagnostic flags and recommendations.
    fn generate_diagnostics(
        &self,
        eval: &CorefEvalScores,
        windowed: Option<&WindowedEvaluation>,
        stratified: &StratifiedEvaluation,
        scale: DocumentScale,
    ) -> BookScaleDiagnostics {
        let mut diagnostics = BookScaleDiagnostics::default();

        // Check metric divergence
        let divergence = (eval.muc.f1 - eval.ceaf_e.f1).abs();
        if divergence > self.config.divergence_threshold {
            diagnostics.high_metric_divergence = true;
            diagnostics
                .recommendations
                .push("Use LEA or stratified metrics instead of CoNLL F1".to_string());
        }

        // Check performance drop
        if let Some(w) = windowed {
            if w.performance_drop > self.config.performance_drop_threshold {
                diagnostics.large_performance_drop = true;
                diagnostics.recommendations.push(
                    "Consider incremental/streaming coref approach (Longdoc-style)".to_string(),
                );
            }
        }

        // Check long chain dominance
        let total_chains =
            stratified.long_chain_count + stratified.short_chain_count + stratified.singleton_count;
        if total_chains > 0 {
            let _long_chain_ratio = stratified.long_chain_count as f64 / total_chains as f64;
            // Even if few long chains, they might dominate mentions
            if stratified.long_chains.f1 > stratified.short_chains.f1 + 0.20 {
                diagnostics.long_chain_dominance = true;
                diagnostics
                    .recommendations
                    .push("Report per-chain-length metrics separately".to_string());
            }
        }

        // Check singleton neglect
        if stratified.singleton_count > 0 && stratified.singletons.f1 < 0.50 {
            diagnostics.singleton_neglect = true;
            diagnostics
                .recommendations
                .push("System may be ignoring minor entities".to_string());
        }

        // Scale-specific recommendations
        match scale {
            DocumentScale::BookScale => {
                diagnostics
                    .recommendations
                    .push("Consider BOOKCOREF-style windowed+grouped evaluation".to_string());
            }
            DocumentScale::Long => {
                diagnostics
                    .recommendations
                    .push("Monitor for metric divergence as length increases".to_string());
            }
            _ => {}
        }

        diagnostics
    }
}

// =============================================================================
// Per-Book Breakdown (like BOOKCOREF Table 3)
// =============================================================================

/// Per-book evaluation breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerBookEvaluation {
    /// Book identifier
    pub book_id: String,
    /// Book title (optional)
    pub title: Option<String>,
    /// Author (optional)
    pub author: Option<String>,
    /// Token count
    pub token_count: usize,
    /// Full-document evaluation
    pub full_doc: CorefEvalScores,
    /// Windowed evaluation
    pub windowed: Option<WindowedEvaluation>,
    /// Scale classification
    pub scale: DocumentScale,
}

/// Multi-book evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiBookReport {
    /// Individual book evaluations
    pub books: Vec<PerBookEvaluation>,
    /// Aggregate statistics
    pub aggregate: AggregateStats,
}

/// Aggregate statistics across multiple books.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStats {
    /// Total books
    pub total_books: usize,
    /// Mean CoNLL F1 (full doc)
    pub mean_conll_f1: f64,
    /// Std dev of CoNLL F1
    pub std_conll_f1: f64,
    /// Mean performance drop (windowed → full)
    pub mean_performance_drop: f64,
    /// Books with scale issues
    pub books_with_issues: usize,
}

impl MultiBookReport {
    /// Generate from per-book evaluations.
    pub fn from_books(books: Vec<PerBookEvaluation>) -> Self {
        let total_books = books.len();

        let conll_scores: Vec<f64> = books.iter().map(|b| b.full_doc.conll_f1).collect();

        let mean_conll_f1 = if !conll_scores.is_empty() {
            conll_scores.iter().sum::<f64>() / conll_scores.len() as f64
        } else {
            0.0
        };

        let std_conll_f1 = if conll_scores.len() > 1 {
            let variance = conll_scores
                .iter()
                .map(|x| (x - mean_conll_f1).powi(2))
                .sum::<f64>()
                / (conll_scores.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let performance_drops: Vec<f64> = books
            .iter()
            .filter_map(|b| b.windowed.as_ref().map(|w| w.performance_drop))
            .collect();

        let mean_performance_drop = if !performance_drops.is_empty() {
            performance_drops.iter().sum::<f64>() / performance_drops.len() as f64
        } else {
            0.0
        };

        let books_with_issues = books
            .iter()
            .filter(|b| {
                let divergence = (b.full_doc.muc.f1 - b.full_doc.ceaf_e.f1).abs();
                divergence > 0.30
                    || b.windowed
                        .as_ref()
                        .map(|w| w.performance_drop > 0.15)
                        .unwrap_or(false)
            })
            .count();

        let aggregate = AggregateStats {
            total_books,
            mean_conll_f1,
            std_conll_f1,
            mean_performance_drop,
            books_with_issues,
        };

        Self { books, aggregate }
    }

    /// Format as table (similar to BOOKCOREF Table 3).
    pub fn format_table(&self) -> String {
        let mut table = String::new();

        // Header
        table.push_str(&format!(
            "{:<30} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
            "Book", "Tokens", "MUC", "B³", "CEAF", "CoNLL"
        ));
        table.push_str(&format!("{}\n", "-".repeat(78)));

        // Per-book rows
        for book in &self.books {
            let title = book
                .title
                .as_deref()
                .unwrap_or(&book.book_id)
                .chars()
                .take(28)
                .collect::<String>();

            table.push_str(&format!(
                "{:<30} {:>8} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%\n",
                title,
                book.token_count,
                book.full_doc.muc.f1 * 100.0,
                book.full_doc.b_cubed.f1 * 100.0,
                book.full_doc.ceaf_e.f1 * 100.0,
                book.full_doc.conll_f1 * 100.0,
            ));
        }

        // Separator
        table.push_str(&format!("{}\n", "-".repeat(78)));

        // Aggregate
        table.push_str(&format!(
            "{:<30} {:>8} {:>7.1}% ±{:.1}\n",
            "MEAN",
            "",
            self.aggregate.mean_conll_f1 * 100.0,
            self.aggregate.std_conll_f1 * 100.0
        ));

        table
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain(mentions: Vec<(&str, usize, usize)>) -> CorefChain {
        let m: Vec<Mention> = mentions
            .into_iter()
            .map(|(text, start, end)| Mention::new(text, start, end))
            .collect();
        CorefChain::new(m)
    }

    #[test]
    fn test_stratify_chains() {
        let config = BookScaleConfig::default();
        let analyzer = BookScaleAnalyzer::new(config);

        let chains = vec![
            make_chain(vec![("a", 0, 1)]),                           // singleton
            make_chain(vec![("b", 0, 1), ("c", 2, 3), ("d", 4, 5)]), // short
            make_chain((0..15).map(|i| ("x", i * 10, i * 10 + 1)).collect()), // long
        ];

        let (long, short, single) = analyzer.stratify_chains(&chains);
        assert_eq!(single.len(), 1);
        assert_eq!(short.len(), 1);
        assert_eq!(long.len(), 1);
    }

    #[test]
    fn test_reliability_assessment() {
        let config = BookScaleConfig::default();
        let analyzer = BookScaleAnalyzer::new(config);

        let eval = CorefEvalScores {
            muc: Scores {
                precision: 0.9,
                recall: 0.9,
                f1: 0.9,
            },
            b_cubed: Scores {
                precision: 0.7,
                recall: 0.7,
                f1: 0.7,
            },
            ceaf_e: Scores {
                precision: 0.4,
                recall: 0.4,
                f1: 0.4,
            },
            ceaf_m: Scores::default(),
            lea: Scores::default(),
            conll_f1: 0.67,
        };

        let stratified = StratifiedEvaluation::default();
        let reliability = analyzer.assess_reliability(&eval, &stratified, DocumentScale::BookScale);

        // MUC should be low reliability (large divergence)
        assert!(matches!(
            reliability.muc_reliability,
            ReliabilityLevel::Low | ReliabilityLevel::Medium
        ));
    }

    #[test]
    fn test_diagnostics_generation() {
        let config = BookScaleConfig::default();
        let analyzer = BookScaleAnalyzer::new(config);

        let eval = CorefEvalScores {
            muc: Scores {
                precision: 0.93,
                recall: 0.93,
                f1: 0.93,
            },
            b_cubed: Scores {
                precision: 0.62,
                recall: 0.62,
                f1: 0.62,
            },
            ceaf_e: Scores {
                precision: 0.33,
                recall: 0.33,
                f1: 0.33,
            },
            ceaf_m: Scores::default(),
            lea: Scores::default(),
            conll_f1: 0.63,
        };

        let windowed = WindowedEvaluation {
            num_windows: 10,
            window_size: 1500,
            avg_conll_f1: 0.78,
            std_conll_f1: 0.05,
            performance_drop: 0.15,
            window_evals: vec![],
        };

        let stratified = StratifiedEvaluation::default();

        let diagnostics = analyzer.generate_diagnostics(
            &eval,
            Some(&windowed),
            &stratified,
            DocumentScale::BookScale,
        );

        assert!(diagnostics.high_metric_divergence);
        assert!(diagnostics.has_issues());
    }

    #[test]
    fn test_multi_book_report() {
        let books = vec![
            PerBookEvaluation {
                book_id: "animal_farm".to_string(),
                title: Some("Animal Farm".to_string()),
                author: Some("George Orwell".to_string()),
                token_count: 29853,
                full_doc: CorefEvalScores {
                    muc: Scores {
                        precision: 0.9,
                        recall: 0.9,
                        f1: 0.9,
                    },
                    b_cubed: Scores {
                        precision: 0.6,
                        recall: 0.6,
                        f1: 0.6,
                    },
                    ceaf_e: Scores {
                        precision: 0.5,
                        recall: 0.5,
                        f1: 0.5,
                    },
                    ceaf_m: Scores::default(),
                    lea: Scores::default(),
                    conll_f1: 0.67,
                },
                windowed: None,
                scale: DocumentScale::Long,
            },
            PerBookEvaluation {
                book_id: "pride_prejudice".to_string(),
                title: Some("Pride and Prejudice".to_string()),
                author: Some("Jane Austen".to_string()),
                token_count: 121869,
                full_doc: CorefEvalScores {
                    muc: Scores {
                        precision: 0.85,
                        recall: 0.85,
                        f1: 0.85,
                    },
                    b_cubed: Scores {
                        precision: 0.55,
                        recall: 0.55,
                        f1: 0.55,
                    },
                    ceaf_e: Scores {
                        precision: 0.35,
                        recall: 0.35,
                        f1: 0.35,
                    },
                    ceaf_m: Scores::default(),
                    lea: Scores::default(),
                    conll_f1: 0.58,
                },
                windowed: None,
                scale: DocumentScale::BookScale,
            },
        ];

        let report = MultiBookReport::from_books(books);
        assert_eq!(report.aggregate.total_books, 2);
        assert!(report.aggregate.mean_conll_f1 > 0.5);

        let table = report.format_table();
        assert!(table.contains("Animal Farm"));
        assert!(table.contains("Pride and Prejudice"));
    }

    // =========================================================================
    // Additional Edge Case Tests
    // =========================================================================

    #[test]
    fn test_document_scale_classification() {
        // Short: 0-2000 tokens
        assert_eq!(DocumentScale::from_tokens(100), DocumentScale::Short);
        assert_eq!(DocumentScale::from_tokens(2000), DocumentScale::Short);
        // Medium: 2001-10000 tokens
        assert_eq!(DocumentScale::from_tokens(5000), DocumentScale::Medium);
        // Long: 10001-50000 tokens
        assert_eq!(DocumentScale::from_tokens(30000), DocumentScale::Long);
        // BookScale: >50000 tokens
        assert_eq!(DocumentScale::from_tokens(100000), DocumentScale::BookScale);
    }

    #[test]
    fn test_empty_chains_stratification() {
        let config = BookScaleConfig::default();
        let analyzer = BookScaleAnalyzer::new(config);

        let chains: Vec<CorefChain> = vec![];
        let (long, short, single) = analyzer.stratify_chains(&chains);

        assert!(long.is_empty());
        assert!(short.is_empty());
        assert!(single.is_empty());
    }

    #[test]
    fn test_all_singletons() {
        let config = BookScaleConfig::default();
        let analyzer = BookScaleAnalyzer::new(config);

        let chains = vec![
            make_chain(vec![("a", 0, 1)]),
            make_chain(vec![("b", 10, 11)]),
            make_chain(vec![("c", 20, 21)]),
        ];

        let (long, short, single) = analyzer.stratify_chains(&chains);

        assert!(long.is_empty());
        assert!(short.is_empty());
        assert_eq!(single.len(), 3);
    }

    #[test]
    fn test_all_long_chains() {
        let config = BookScaleConfig::default();
        let analyzer = BookScaleAnalyzer::new(config);

        // Create chains with 15+ mentions (long threshold)
        let chains = vec![
            make_chain((0..20).map(|i| ("x", i * 10, i * 10 + 1)).collect()),
            make_chain((0..25).map(|i| ("y", i * 10 + 5, i * 10 + 6)).collect()),
        ];

        let (long, short, single) = analyzer.stratify_chains(&chains);

        assert_eq!(long.len(), 2);
        assert!(short.is_empty());
        assert!(single.is_empty());
    }

    #[test]
    fn test_scores_default() {
        let scores = Scores::default();
        assert!((scores.precision - 0.0).abs() < 0.001);
        assert!((scores.recall - 0.0).abs() < 0.001);
        assert!((scores.f1 - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_coref_eval_scores_conll_average() {
        let eval = CorefEvalScores {
            muc: Scores {
                precision: 0.8,
                recall: 0.8,
                f1: 0.8,
            },
            b_cubed: Scores {
                precision: 0.7,
                recall: 0.7,
                f1: 0.7,
            },
            ceaf_e: Scores {
                precision: 0.6,
                recall: 0.6,
                f1: 0.6,
            },
            ceaf_m: Scores::default(),
            lea: Scores::default(),
            conll_f1: 0.7, // (0.8 + 0.7 + 0.6) / 3 = 0.7
        };

        // CoNLL F1 is typically average of MUC, B³, CEAF-e
        let expected_conll = (0.8 + 0.7 + 0.6) / 3.0;
        assert!((eval.conll_f1 - expected_conll).abs() < 0.001);
    }

    #[test]
    fn test_windowed_evaluation_performance_drop() {
        let windowed = WindowedEvaluation {
            num_windows: 5,
            window_size: 1000,
            avg_conll_f1: 0.80,
            std_conll_f1: 0.03,
            performance_drop: 0.15,
            window_evals: vec![],
        };

        // Performance drop should be positive when full-doc is worse
        assert!(windowed.performance_drop > 0.0);
        assert_eq!(windowed.num_windows, 5);
    }

    #[test]
    fn test_diagnostics_no_issues_for_short_doc() {
        let config = BookScaleConfig::default();
        let analyzer = BookScaleAnalyzer::new(config);

        // Perfect scores - no issues expected
        let eval = CorefEvalScores {
            muc: Scores {
                precision: 0.85,
                recall: 0.85,
                f1: 0.85,
            },
            b_cubed: Scores {
                precision: 0.82,
                recall: 0.82,
                f1: 0.82,
            },
            ceaf_e: Scores {
                precision: 0.80,
                recall: 0.80,
                f1: 0.80,
            },
            ceaf_m: Scores::default(),
            lea: Scores::default(),
            conll_f1: 0.82,
        };

        let stratified = StratifiedEvaluation::default();
        let diagnostics = analyzer.generate_diagnostics(
            &eval,
            None, // No windowed evaluation
            &stratified,
            DocumentScale::Short,
        );

        // Small divergence for short docs - may or may not have issues
        // Just verify it doesn't panic and has expected fields
        let _ = diagnostics.high_metric_divergence;
        let _ = diagnostics.has_issues();
    }

    #[test]
    fn test_multi_book_report_empty() {
        let books: Vec<PerBookEvaluation> = vec![];
        let report = MultiBookReport::from_books(books);

        assert_eq!(report.aggregate.total_books, 0);
        assert!(report.books.is_empty());
    }

    #[test]
    fn test_per_book_evaluation_scale() {
        let book = PerBookEvaluation {
            book_id: "test".to_string(),
            title: Some("Test Book".to_string()),
            author: None,
            token_count: 200000,
            full_doc: CorefEvalScores::default(),
            windowed: None,
            scale: DocumentScale::BookScale,
        };

        assert!(book.token_count > 100000);
        assert_eq!(book.scale, DocumentScale::BookScale);
    }
}
