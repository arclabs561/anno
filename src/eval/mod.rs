//! NER evaluation framework using standard datasets.
//!
//! Compares different NER backends:
//! - Rule-based NER (fallback, always available)
//! - GLiNER ONNX (zero-shot, state-of-the-art)
//! - Candle NER (BERT-based, fine-tuned)
//!
//! Supports multiple dataset formats:
//! - CoNLL-2003 (classic BIO tagging format)
//! - JSON/JSONL (modern format: OpenNER 1.0, MultiNERD, Wikiann)
//! - HuggingFace Datasets format
//!
//! Metrics:
//! - Precision, Recall, F1 (per entity type and overall)
//! - Exact match vs partial match
//! - Speed (tokens/second)
//! - Per-dataset statistics

use crate::EntityType;
use crate::{Error, Model, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// Submodules
pub mod datasets;
pub mod evaluator;
pub mod metrics;
pub mod synthetic;
pub mod types;
pub mod validation;

// Re-exports
#[allow(deprecated)]
pub use datasets::{GoldEntity, GroundTruthEntity};
pub use evaluator::*;
pub use metrics::*;
pub use types::{GoalCheck, GoalCheckResult, MetricValue};
pub use validation::*;

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
    match et {
        EntityType::Person => "PER".to_string(),
        EntityType::Organization => "ORG".to_string(),
        EntityType::Location => "LOC".to_string(),
        EntityType::Date => "DATE".to_string(),
        EntityType::Money => "MONEY".to_string(),
        EntityType::Percent => "PERCENT".to_string(),
        EntityType::Other(s) => s.clone(),
    }
}

/// Entity type matching for evaluation.
///
/// Handles exact matches and common variations.
pub fn entity_type_matches(a: &EntityType, b: &EntityType) -> bool {
    match (a, b) {
        (EntityType::Person, EntityType::Person) => true,
        (EntityType::Organization, EntityType::Organization) => true,
        (EntityType::Location, EntityType::Location) => true,
        (EntityType::Date, EntityType::Date) => true,
        (EntityType::Money, EntityType::Money) => true,
        (EntityType::Percent, EntityType::Percent) => true,
        (EntityType::Other(a_str), EntityType::Other(b_str)) => a_str == b_str,
        _ => false,
    }
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
        let metrics =
            evaluator.evaluate_test_case(model, text, ground_truth, Some(&test_case_id))?;
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
