//! NER and Coreference evaluation framework.
//!
//! # Overview
//!
//! This module provides comprehensive evaluation tools for:
//! - **Named Entity Recognition (NER)**: Standard metrics, error analysis, significance testing
//! - **Coreference Resolution**: MUC, B³, CEAF, LEA, BLANC, CoNLL F1
//!
//! # NER Evaluation
//!
//! ```rust,ignore
//! use anno::eval::{evaluate_ner_model, GoldEntity, ErrorAnalysis};
//! use anno::PatternNER;
//!
//! let model = PatternNER::new();
//! let test_cases = vec![
//!     ("Meeting on January 15".to_string(), vec![
//!         GoldEntity::new("January 15", anno::EntityType::Date, 11),
//!     ]),
//! ];
//!
//! let results = evaluate_ner_model(&model, &test_cases)?;
//! println!("F1: {:.1}%", results.f1 * 100.0);
//! ```
//!
//! # Coreference Evaluation
//!
//! ```rust,ignore
//! use anno::eval::{CorefChain, Mention, conll_f1, muc_score, b_cubed_score};
//!
//! let gold = vec![
//!     CorefChain::new(0, vec![Mention::new("John", 0, 4), Mention::new("he", 20, 22)]),
//! ];
//! let pred = gold.clone(); // Perfect match
//!
//! let (p, r, f1) = conll_f1(&gold, &pred);
//! assert!((f1 - 1.0).abs() < 0.001);
//! ```
//!
//! # Dataset Support
//!
//! | Dataset | Type | Size | Format |
//! |---------|------|------|--------|
//! | CoNLL-2003 | NER | ~22k sentences | BIO tags |
//! | WikiGold | NER | 145 docs | CoNLL |
//! | WNUT-17 | NER | ~5k tweets | CoNLL |
//! | MultiNERD | NER | ~50k examples | JSONL |
//! | GAP | Coref | 4.5k examples | TSV |
//! | PreCo | Coref | 12k docs | JSON |
//!
//! # Metrics
//!
//! **NER Metrics:**
//! - Precision, Recall, F1 (micro/macro)
//! - Per-entity-type breakdown
//! - Partial match (boundary overlap)
//! - Confidence threshold analysis
//!
//! **Coreference Metrics:**
//! - MUC (link-based)
//! - B³ (mention-based)
//! - CEAF-e/m (entity/mention alignment)
//! - LEA (link-based entity-aware)
//! - BLANC (rand-index based)
//! - CoNLL F1 (average of MUC, B³, CEAF-e)
//!
//! **Error Analysis:**
//! - Confusion matrix
//! - Error categorization (type, boundary, spurious, missed)
//! - Statistical significance testing (paired t-test)

use crate::EntityType;
use crate::{Error, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Evaluation Task Enum
// =============================================================================

/// Evaluation task type.
///
/// Clarifies what capability is being evaluated, since the same model
/// may support multiple tasks with different metrics.
///
/// # Example
///
/// ```rust
/// use anno::eval::{EvalTask, EvalMode};
///
/// let task = EvalTask::NER {
///     labels: vec!["PER", "ORG", "LOC"].into_iter().map(String::from).collect(),
///     mode: EvalMode::Strict,
/// };
///
/// match task {
///     EvalTask::NER { labels, mode } => {
///         println!("NER with {} entity types, {:?} mode", labels.len(), mode);
///     }
///     _ => {}
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvalTask {
    /// Named Entity Recognition: span extraction with type classification.
    ///
    /// Input: Text → Output: `Vec<Entity>`
    ///
    /// Metrics: Precision, Recall, F1 (Strict/Exact/Partial/Type modes)
    NER {
        /// Entity type labels expected (e.g., ["PER", "ORG", "LOC"])
        labels: Vec<String>,
        /// Evaluation mode (Strict, Exact, Partial, Type)
        mode: EvalMode,
    },

    /// Relation Extraction: entity pairs with relation types.
    ///
    /// Input: Text → Output: `Vec<(Entity, Relation, Entity)>`
    ///
    /// Metrics: Relation F1 (with/without entity correctness)
    RelationExtraction {
        /// Relation types expected (e.g., ["WORKS_AT", "BORN_IN"])
        relations: Vec<String>,
        /// Whether to require correct entity spans for relation credit
        require_entity_match: bool,
    },

    /// Coreference Resolution: mention clustering.
    ///
    /// Input: Text → Output: `Vec<CorefChain>`
    ///
    /// Metrics: MUC, B³, CEAF-e/m, LEA, BLANC, CoNLL F1
    Coreference {
        /// Which coreference metrics to compute
        metrics: Vec<CorefMetric>,
    },

    /// Discontinuous NER: non-contiguous span extraction.
    ///
    /// Input: Text → Output: `Vec<DiscontinuousEntity>`
    ///
    /// Metrics: Same as NER but with discontinuous span matching
    DiscontinuousNER {
        /// Entity type labels expected
        labels: Vec<String>,
    },

    /// Event Extraction: event triggers and arguments.
    ///
    /// Input: Text → Output: Events with trigger and argument spans
    ///
    /// Metrics: Trigger F1, Argument F1, Event F1
    EventExtraction {
        /// Event types (e.g., ["ATTACK", "MOVEMENT", "TRANSACTION"])
        event_types: Vec<String>,
        /// Argument roles (e.g., ["AGENT", "PATIENT", "LOCATION"])
        argument_roles: Vec<String>,
    },
}

impl Default for EvalTask {
    fn default() -> Self {
        EvalTask::NER {
            labels: vec![
                "PER".to_string(),
                "ORG".to_string(),
                "LOC".to_string(),
                "MISC".to_string(),
            ],
            mode: EvalMode::Strict,
        }
    }
}

/// Coreference evaluation metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorefMetric {
    /// MUC (link-based)
    MUC,
    /// B-cubed (mention-based)
    BCubed,
    /// CEAF entity-based
    CEAFe,
    /// CEAF mention-based
    CEAFm,
    /// LEA (link-based entity-aware)
    LEA,
    /// BLANC (rand-index based)
    BLANC,
    /// CoNLL F1 (average of MUC, B³, CEAF-e)
    CoNLL,
}

/// Evaluation mode for NER (re-exported from modes).
pub use modes::EvalMode;

// Submodules
pub mod active_learning;
pub mod analysis;
pub mod benchmark;
pub mod calibration;
pub mod coref;
pub mod coref_loader;
pub mod coref_metrics;
pub mod coref_resolver;
pub mod datasets;
pub mod dataset_comparison;
pub mod dataset_quality;
pub mod demographic_bias;
pub mod drift;
pub mod ensemble;
pub mod error_analysis;
pub mod evaluator;
pub mod few_shot;
pub mod gender_bias;
pub mod harness;
pub mod learning_curve;
pub mod length_bias;
pub mod loader;
pub mod long_tail;
pub mod metrics;
pub mod modes;
pub mod ood_detection;
pub mod prelude;
pub mod report;
pub mod robustness;
pub mod sampling;
pub mod synthetic;
pub mod synthetic_gen;
pub mod temporal_bias;
pub mod threshold_analysis;
pub mod types;
pub mod validation;

// Re-exports
#[allow(deprecated)]
pub use datasets::{GoldEntity, GroundTruthEntity};
pub use evaluator::*;
pub use harness::{
    BackendAggregateResult, BackendDatasetResult, BackendRegistry, DatasetStatsSummary,
    EvalConfig, EvalHarness, EvalResults,
};
pub use metrics::*;
pub use types::{CorefChainStats, GoalCheck, GoalCheckResult, LabelShift, MetricValue, MetricWithVariance};
pub use validation::*;

// Coreference re-exports
pub use coref::{CorefChain, CorefDocument, Mention, MentionType};
pub use coref_loader::{
    adversarial_coref_examples, synthetic_coref_dataset, CorefLoader, GapExample,
};
pub use coref_metrics::{
    b_cubed_score, blanc_score, ceaf_e_score, ceaf_m_score, compare_systems, conll_f1, lea_score,
    muc_score, AggregateCorefEvaluation, CorefEvaluation, CorefScores, SignificanceTest,
};

// Coreference resolution
pub use coref_resolver::{CorefConfig, CoreferenceResolver, SimpleCorefResolver};

// Gender bias evaluation
pub use gender_bias::{
    create_comprehensive_bias_templates, create_neopronoun_templates, create_winobias_templates,
    occupation_stereotype, GenderBiasEvaluator, GenderBiasResults, OccupationBiasMetrics,
    PronounGender, StereotypeType, WinoBiasExample,
};

// Demographic bias evaluation (ethnicity, region, script)
pub use demographic_bias::{
    create_diverse_location_dataset, create_diverse_name_dataset, DemographicBiasEvaluator,
    DemographicBiasResults, Ethnicity, Gender, LocationExample, LocationType, NameExample,
    NameFrequency, NameResult, Region, RegionalBiasResults, Script,
};

// Temporal bias evaluation (names by decade)
pub use temporal_bias::{
    create_temporal_name_dataset, Decade, TemporalBiasEvaluator, TemporalBiasResults,
    TemporalGender, TemporalNameExample,
};

// Entity length bias evaluation
pub use length_bias::{
    create_length_varied_dataset, EntityLengthEvaluator, LengthBiasResults, LengthBucket,
    LengthTestExample, WordCountBucket,
};

// Confidence calibration
pub use calibration::{
    calibration_grade, confidence_gap_grade, CalibrationEvaluator, CalibrationResults,
    ReliabilityBin, ThresholdMetrics,
};

// Robustness testing
pub use robustness::{
    robustness_grade, Perturbation, PerturbationMetrics, RobustnessEvaluator, RobustnessResults,
};

// Out-of-distribution detection
pub use ood_detection::{
    ood_rate_grade, OODAnalysisResults, OODConfig, OODDetector, OODStatus, VocabCoverageStats,
};

// Dataset quality metrics
pub use dataset_quality::{
    check_leakage, entity_imbalance_ratio, DatasetQualityAnalyzer, DifficultyMetrics,
    QualityReport, ReliabilityMetrics, ValidityMetrics,
};

// Learning curve analysis
pub use learning_curve::{
    suggested_train_sizes, CurveFitParams, DataPoint, LearningCurveAnalysis,
    LearningCurveAnalyzer, SampleEfficiencyMetrics,
};

// Ensemble disagreement analysis
pub use ensemble::{
    agreement_grade, kappa_interpretation, DisagreementDetail, EnsembleAnalysisResults,
    EnsembleAnalyzer, ModelPrediction, SingleExampleAnalysis,
};

// Dataset comparison (cross-domain analysis)
pub use dataset_comparison::{
    compare_datasets, compute_stats, estimate_difficulty, DatasetComparison, DatasetStats,
    Difficulty, DifficultyEstimate, LengthStats,
};

// Drift detection (production monitoring)
pub use drift::{
    ConfidenceDrift, DistributionDrift, DriftConfig, DriftDetector, DriftReport, DriftWindow,
    VocabularyDrift,
};

// Active learning for annotation selection
pub use active_learning::{
    estimate_budget, ActiveLearner, Candidate, SamplingStrategy, ScoreStats, SelectionResult,
};

// Error analysis and categorization
pub use error_analysis::{
    EntityInfo, ErrorAnalyzer, ErrorCategory, ErrorInstance, ErrorPattern, ErrorReport,
    PredictedEntity, TypeErrorStats,
};

// Threshold analysis (precision-recall curves)
pub use threshold_analysis::{
    format_threshold_table, interpret_curve, PredictionWithConfidence, ThresholdAnalyzer,
    ThresholdCurve, ThresholdPoint,
};

// Unified evaluation report (recommended entry point)
pub use report::{
    BiasSummary, CalibrationSummary, CoreMetrics, DataQualitySummary, DemographicBiasMetrics,
    ErrorSummary, EvalReport, GenderBiasMetrics, GoldEntity as ReportGoldEntity,
    LengthBiasMetrics, Priority, Recommendation, RecommendationCategory, ReportBuilder,
    TestCase, TypeMetrics as ReportTypeMetrics,
};

// Few-shot learning evaluation
pub use few_shot::{
    FewShotEvaluator, FewShotGold, FewShotPrediction, FewShotResults, FewShotTask,
    FewShotTaskResults, SupportExample, simulate_few_shot_task,
};

// Long-tail entity evaluation
pub use long_tail::{
    EntityFrequency, FrequencyBucket, FrequencySplit, LongTailAnalyzer, LongTailResults,
    TypePerformance, format_long_tail_results,
};

// Analysis re-exports
pub use analysis::{
    build_confusion_matrix, compare_ner_systems, ConfusionMatrix, ErrorAnalysis, ErrorType,
    NERError, NERSignificanceTest,
};

// Sampling re-exports
pub use sampling::{multi_seed_eval, stratified_sample, stratified_sample_ner};

/// Per-entity-type metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMetrics {
    /// Precision for this entity type.
    pub precision: f64,
    /// Recall for this entity type.
    pub recall: f64,
    /// F1 score for this entity type.
    pub f1: f64,
    /// Number of entities found by the model.
    pub found: usize,
    /// Number of entities expected (ground truth).
    pub expected: usize,
    /// Number of correctly identified entities.
    pub correct: usize,
}

/// NER evaluation results.
///
/// Contains both micro and macro F1 scores:
/// - **Micro F1**: Treats all entities as one pool (good for overall performance)
/// - **Macro F1**: Averages per-type scores (good for fairness across types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NEREvaluationResults {
    /// Overall precision (micro-averaged)
    pub precision: f64,
    /// Overall recall (micro-averaged)
    pub recall: f64,
    /// Overall F1 (micro-averaged) - treats all entities as one pool
    pub f1: f64,
    /// Macro F1 - average of per-type F1 scores (equal weight to each type)
    #[serde(default)]
    pub macro_f1: Option<f64>,
    /// Weighted F1 - per-type F1 weighted by support (entity count)
    #[serde(default)]
    pub weighted_f1: Option<f64>,
    /// Per-entity-type metrics
    pub per_type: HashMap<String, TypeMetrics>,
    /// Speed metrics
    pub tokens_per_second: f64,
    /// Total entities found by the model.
    pub found: usize,
    /// Total entities expected (ground truth).
    pub expected: usize,
    /// Additional metadata
    #[serde(default)]
    pub metadata: Option<EvaluationMetadata>,
}

/// Additional evaluation metadata for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationMetadata {
    /// Dataset name
    pub dataset_name: Option<String>,
    /// Dataset format (e.g., "CoNLL", "JSONL", "synthetic")
    pub dataset_format: Option<String>,
    /// Dataset version or checksum for integrity verification
    pub dataset_version: Option<String>,
    /// Number of test cases evaluated
    pub num_test_cases: usize,
    /// Total number of gold entities in the dataset
    pub total_gold_entities: Option<usize>,
    /// Evaluation timestamp (ISO 8601)
    pub timestamp: Option<String>,
    /// Model name/identifier
    pub model_info: Option<String>,
    /// Model version (if applicable)
    pub model_version: Option<String>,
    /// Matching mode used (e.g., "exact", "partial_0.5")
    pub matching_mode: Option<String>,
    /// anno version
    pub anno_version: Option<String>,
}

/// Convert EntityType to string label.
///
/// Used for evaluation metrics and dataset compatibility.
pub fn entity_type_to_string(et: &EntityType) -> String {
    et.as_label().to_string()
}

/// Entity type matching for evaluation.
///
/// Handles exact matches. Uses PartialEq implementation.
pub fn entity_type_matches(a: &EntityType, b: &EntityType) -> bool {
    a == b
}

/// Load CoNLL-2003 format dataset.
///
/// Format: Each line contains: word POS-tag chunk-tag NER-tag
/// Empty lines separate sentences.
/// NER tags: B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
pub fn load_conll2003<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Vec<GoldEntity>)>> {
    let content = std::fs::read_to_string(path.as_ref()).map_err(Error::Io)?;

    let mut test_cases: Vec<(String, Vec<GoldEntity>)> = Vec::new();
    let mut current_text = String::new();
    let mut current_entities: Vec<GoldEntity> = Vec::new();
    let mut char_offset = 0;

    for line in content.lines() {
        if line.trim().is_empty() {
            // End of sentence
            if !current_text.is_empty() {
                test_cases.push((current_text.clone(), current_entities.clone()));
            }
            current_text.clear();
            current_entities.clear();
            char_offset = 0;
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue; // Skip malformed lines
        }

        let word = parts[0];
        let ner_tag = parts[3];

        // Add word to text
        if !current_text.is_empty() {
            current_text.push(' ');
            char_offset += 1;
        }
        let word_start = char_offset;
        current_text.push_str(word);
        char_offset += word.len();
        let word_end = char_offset;

        // Parse NER tag
        if ner_tag != "O" {
            let (prefix, entity_type_str) = if let Some(dash_pos) = ner_tag.find('-') {
                (&ner_tag[..dash_pos], &ner_tag[dash_pos + 1..])
            } else {
                continue;
            };

            let entity_type = match entity_type_str {
                "PER" => EntityType::Person,
                "ORG" => EntityType::Organization,
                "LOC" => EntityType::Location,
                "MISC" => EntityType::Other("misc".to_string()),
                "DATE" => EntityType::Date,
                "MONEY" => EntityType::Money,
                "PERCENT" => EntityType::Percent,
                _ => continue,
            };

            if prefix == "B" {
                // Beginning of entity - start new entity
                current_entities.push(GoldEntity::with_span(
                    word,
                    entity_type,
                    word_start,
                    word_end,
                ));
            } else if prefix == "I" {
                // Inside entity - extend last entity if same type
                if let Some(last) = current_entities.last_mut() {
                    if entity_type_matches(&last.entity_type, &entity_type) {
                        // Extend entity
                        last.text.push(' ');
                        last.text.push_str(word);
                        last.end = word_end;
                    } else {
                        // Different type - start new entity
                        current_entities.push(GoldEntity::with_span(
                            word,
                            entity_type,
                            word_start,
                            word_end,
                        ));
                    }
                }
            }
        }
    }

    // Handle last sentence if file doesn't end with newline
    if !current_text.is_empty() {
        test_cases.push((current_text, current_entities));
    }

    // Validate all loaded entities
    for (text, entities) in &test_cases {
        let validation_result = validation::validate_ground_truth_entities(text, entities, false);
        if !validation_result.is_valid {
            return Err(Error::InvalidInput(format!(
                "Invalid entities in CoNLL dataset: {}",
                validation_result.errors.join("; ")
            )));
        }
    }

    Ok(test_cases)
}

/// Evaluate NER model on a dataset.
pub fn evaluate_ner_model(
    model: &dyn Model,
    test_cases: &[(String, Vec<GoldEntity>)],
) -> Result<NEREvaluationResults> {
    evaluate_ner_model_with_mapper(model, test_cases, None)
}

/// Evaluate NER model with optional type normalization.
///
/// # Arguments
/// * `model` - NER model to evaluate
/// * `test_cases` - Test cases with text and gold entities
/// * `type_mapper` - Optional TypeMapper to normalize domain-specific types
///
/// # Example
///
/// ```rust,ignore
/// use anno::{TypeMapper, PatternNER, Model};
/// use anno::eval::{evaluate_ner_model_with_mapper, GoldEntity};
///
/// // MIT Movie dataset - normalize ACTOR/DIRECTOR to Person
/// let mapper = TypeMapper::mit_movie();
/// let test_cases = vec![
///     ("Tom Hanks directed the movie".to_string(), vec![
///         GoldEntity::with_label("Tom Hanks", "ACTOR", 0),
///     ]),
/// ];
///
/// let model = PatternNER::new();
/// let results = evaluate_ner_model_with_mapper(&model, &test_cases, Some(&mapper));
/// ```
pub fn evaluate_ner_model_with_mapper(
    model: &dyn Model,
    test_cases: &[(String, Vec<GoldEntity>)],
    type_mapper: Option<&crate::TypeMapper>,
) -> Result<NEREvaluationResults> {
    let evaluator = evaluator::StandardNEREvaluator::new();

    if test_cases.is_empty() {
        return Ok(NEREvaluationResults {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            macro_f1: None,
            weighted_f1: None,
            per_type: HashMap::new(),
            tokens_per_second: 0.0,
            found: 0,
            expected: 0,
            metadata: Some(EvaluationMetadata {
                num_test_cases: 0,
                total_gold_entities: Some(0),
                timestamp: Some(chrono::Utc::now().to_rfc3339()),
                anno_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                ..Default::default()
            }),
        });
    }

    // Evaluate each test case
    let mut query_metrics = Vec::new();
    for (i, (text, ground_truth)) in test_cases.iter().enumerate() {
        let test_case_id = format!("test_case_{}", i);
        
        // Apply type normalization if mapper provided
        let normalized_truth: Vec<GoldEntity>;
        let truth_ref = if let Some(mapper) = type_mapper {
            normalized_truth = ground_truth
                .iter()
                .map(|e| GoldEntity {
                    text: e.text.clone(),
                    entity_type: mapper.normalize(e.entity_type.as_label()),
                    original_label: e.original_label.clone(), // Preserve original for debugging
                    start: e.start,
                    end: e.end,
                })
                .collect();
            &normalized_truth
        } else {
            ground_truth
        };
        
        let metrics =
            evaluator.evaluate_test_case(model, text, truth_ref, Some(&test_case_id))?;
        query_metrics.push(metrics);
    }

    // Aggregate metrics
    let aggregate = evaluator.aggregate(&query_metrics)?;

    // Compute macro F1
    let macro_f1 = if aggregate.per_type.is_empty() {
        None
    } else {
        let sum: f64 = aggregate.per_type.values().map(|m| m.f1).sum();
        Some(sum / aggregate.per_type.len() as f64)
    };

    // Compute weighted F1
    let weighted_f1 = if aggregate.per_type.is_empty() || aggregate.total_expected == 0 {
        None
    } else {
        let weighted_sum: f64 = aggregate
            .per_type
            .values()
            .map(|m| m.f1 * m.expected as f64)
            .sum();
        Some(weighted_sum / aggregate.total_expected as f64)
    };

    Ok(NEREvaluationResults {
        precision: aggregate.precision.get(),
        recall: aggregate.recall.get(),
        f1: aggregate.f1.get(),
        macro_f1,
        weighted_f1,
        per_type: aggregate.per_type,
        tokens_per_second: aggregate.tokens_per_second,
        found: aggregate.total_found,
        expected: aggregate.total_expected,
        metadata: Some(EvaluationMetadata {
            num_test_cases: aggregate.num_test_cases,
            total_gold_entities: Some(aggregate.total_expected),
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
            anno_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            ..Default::default()
        }),
    })
}

/// Compare multiple NER models on the same dataset.
pub fn compare_ner_models(
    models: &[(&str, &dyn Model)],
    test_cases: &[(String, Vec<GoldEntity>)],
) -> Result<HashMap<String, NEREvaluationResults>> {
    let mut results = HashMap::new();

    for (name, model) in models {
        log::info!("Evaluating {}...", name);
        let result = evaluate_ner_model(*model, test_cases)?;
        results.insert(name.to_string(), result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_to_string() {
        assert_eq!(entity_type_to_string(&EntityType::Person), "PER");
        assert_eq!(entity_type_to_string(&EntityType::Organization), "ORG");
        assert_eq!(entity_type_to_string(&EntityType::Location), "LOC");
    }

    #[test]
    fn test_entity_type_matches() {
        assert!(entity_type_matches(
            &EntityType::Person,
            &EntityType::Person
        ));
        assert!(!entity_type_matches(
            &EntityType::Person,
            &EntityType::Organization
        ));
    }
}
