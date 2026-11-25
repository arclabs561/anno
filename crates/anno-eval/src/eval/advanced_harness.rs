//! Advanced evaluation harness for specialized NER tasks.
//!
//! Extends the standard evaluation harness to support:
//! - Discontinuous NER
//! - Relation Extraction
//! - Visual NER
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno_eval::eval::advanced_harness::{AdvancedEvalHarness, AdvancedTaskResults};
//! use anno::W2NER;
//!
//! let mut harness = AdvancedEvalHarness::new();
//!
//! // Register a discontinuous NER model
//! harness.register_discontinuous("w2ner", Box::new(W2NER::new()));
//!
//! // Run evaluation on synthetic data
//! let results = harness.run_synthetic_discontinuous()?;
//!
//! // Generate report
//! println!("{}", results.summary());
//! ```

use anno::{DiscontinuousEntity, DiscontinuousNER, RelationExtractor, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::dataset::synthetic::{
    discontinuous::dataset as discontinuous_dataset, relations::dataset as relations_dataset,
};
use super::discontinuous::{DiscontinuousCounts, DiscontinuousEvalConfig, DiscontinuousNERMetrics};
use super::relation::{
    evaluate_relation_documents, RelationEvalConfig, RelationMetrics, RelationPrediction,
};
use super::visual::{
    evaluate_visual_ner, synthetic_visual_examples, VisualEvalConfig, VisualGold, VisualNERMetrics,
    VisualPrediction,
};

// =============================================================================
// RESULTS TYPES
// =============================================================================

/// Results from running an advanced evaluation task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTaskResults {
    /// Timestamp of the evaluation run.
    pub timestamp: String,
    /// Task type evaluated.
    pub task: String,
    /// Results per model.
    pub models: Vec<ModelResult>,
    /// Number of examples evaluated.
    pub num_examples: usize,
    /// Total entities/relations in gold.
    pub num_gold: usize,
}

impl AdvancedTaskResults {
    /// Get a summary string.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "=== {} Evaluation ({} examples) ===\n",
            self.task, self.num_examples
        );

        for model in &self.models {
            s.push_str(&format!(
                "\n{}: F1={:.1}%\n",
                model.name,
                model.primary_f1 * 100.0
            ));
        }

        s
    }
}

/// Results for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResult {
    /// Model name.
    pub name: String,
    /// Primary F1 score.
    pub primary_f1: f64,
    /// Additional metrics (task-specific).
    pub metrics: HashMap<String, f64>,
}

// =============================================================================
// DISCONTINUOUS NER EVALUATION
// =============================================================================

/// Run discontinuous NER evaluation on synthetic data.
///
/// Returns metrics for any model implementing `DiscontinuousNER`.
pub fn evaluate_discontinuous_synthetic<M: DiscontinuousNER>(
    model: &M,
    labels: &[&str],
    threshold: f32,
) -> Result<DiscontinuousNERMetrics> {
    let examples = discontinuous_dataset();
    let config = DiscontinuousEvalConfig::default();

    let mut counts = DiscontinuousCounts::default();
    for example in examples {
        let predicted = model.extract_discontinuous(&example.text, labels, threshold)?;
        counts.add_document(&example.entities, &predicted, &config);
    }
    Ok(counts.metrics())
}

/// Run discontinuous NER evaluation without a model (for testing metrics only).
pub fn evaluate_discontinuous_gold_vs_gold() -> DiscontinuousNERMetrics {
    let examples = discontinuous_dataset();
    let config = DiscontinuousEvalConfig::default();

    let mut counts = DiscontinuousCounts::default();
    for example in examples {
        let predicted: Vec<_> = example
            .entities
            .iter()
            .map(|g| DiscontinuousEntity {
                spans: g.spans.clone(),
                text: g.text.clone(),
                entity_type: g.entity_type.clone(),
                confidence: anno::Confidence::ONE,
            })
            .collect();
        counts.add_document(&example.entities, &predicted, &config);
    }
    counts.metrics()
}

// =============================================================================
// RELATION EXTRACTION EVALUATION
// =============================================================================

/// Run relation extraction evaluation on synthetic data.
pub fn evaluate_relations_synthetic<M: RelationExtractor>(
    model: &M,
    labels: &[&str],
    relations: &[&str],
    threshold: f32,
) -> Result<RelationMetrics> {
    let examples = relations_dataset();
    let config = RelationEvalConfig::default();

    let mut predictions_by_doc = Vec::with_capacity(examples.len());
    for example in &examples {
        let result = model.extract_with_relations(&example.text, labels, relations, threshold)?;
        let predicted: Vec<_> = result
            .relations
            .iter()
            .filter_map(|rel| RelationPrediction::from_triple_with_entities(rel, &result.entities))
            .collect();
        predictions_by_doc.push(predicted);
    }
    Ok(evaluate_relation_documents(
        examples
            .iter()
            .zip(&predictions_by_doc)
            .map(|(ex, pred)| (ex.relations.as_slice(), pred.as_slice())),
        &config,
    ))
}

/// Run relation extraction evaluation without a model (for testing metrics only).
pub fn evaluate_relations_gold_vs_gold() -> RelationMetrics {
    let examples = relations_dataset();
    let predictions_by_doc: Vec<Vec<_>> = examples
        .iter()
        .map(|ex| {
            ex.relations
                .iter()
                .map(|g| RelationPrediction {
                    head_span: g.head_span,
                    head_type: g.head_type.clone(),
                    tail_span: g.tail_span,
                    tail_type: g.tail_type.clone(),
                    relation_type: g.relation_type.clone(),
                    confidence: 1.0,
                })
                .collect()
        })
        .collect();
    evaluate_relation_documents(
        examples
            .iter()
            .zip(&predictions_by_doc)
            .map(|(ex, pred)| (ex.relations.as_slice(), pred.as_slice())),
        &RelationEvalConfig::default(),
    )
}

// =============================================================================
// VISUAL NER EVALUATION
// =============================================================================

/// Run visual NER evaluation on synthetic data.
pub fn evaluate_visual_gold_vs_gold() -> VisualNERMetrics {
    let examples = synthetic_visual_examples();
    let config = VisualEvalConfig::default();

    let gold: Vec<VisualGold> = examples
        .iter()
        .flat_map(|(_, entities)| entities.clone())
        .collect();

    // Perfect prediction = gold
    let pred: Vec<VisualPrediction> = gold
        .iter()
        .map(|g| VisualPrediction {
            text: g.text.clone(),
            entity_type: g.entity_type.clone(),
            bbox: g.bbox,
            confidence: 1.0,
        })
        .collect();

    evaluate_visual_ner(&gold, &pred, &config)
}

// =============================================================================
// DATASET STATISTICS
// =============================================================================

/// Get statistics about the synthetic advanced datasets.
pub fn synthetic_dataset_stats() -> SyntheticDatasetStats {
    let disc = discontinuous_dataset();
    let rel = relations_dataset();
    let vis = synthetic_visual_examples();

    SyntheticDatasetStats {
        discontinuous_examples: disc.len(),
        discontinuous_entities: disc.iter().map(|ex| ex.entities.len()).sum(),
        relation_examples: rel.len(),
        relations: rel.iter().map(|ex| ex.relations.len()).sum(),
        visual_examples: vis.len(),
        visual_entities: vis.iter().map(|(_, e)| e.len()).sum(),
    }
}

/// Statistics about synthetic advanced datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticDatasetStats {
    /// Number of discontinuous NER examples.
    pub discontinuous_examples: usize,
    /// Total discontinuous entities.
    pub discontinuous_entities: usize,
    /// Number of relation extraction examples.
    pub relation_examples: usize,
    /// Total relations.
    pub relations: usize,
    /// Number of visual NER examples.
    pub visual_examples: usize,
    /// Total visual entities.
    pub visual_entities: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discontinuous_synthetic_keeps_matches_in_their_document() {
        struct WrongDocument;
        impl DiscontinuousNER for WrongDocument {
            fn extract_discontinuous(
                &self,
                text: &str,
                _: &[&str],
                _: f32,
            ) -> Result<Vec<DiscontinuousEntity>> {
                if text == "Apple and Microsoft stocks rose sharply." {
                    // These LOC coordinates match "New York airports" in another example.
                    return Ok(vec![DiscontinuousEntity {
                        spans: vec![(0, 8), (25, 33)],
                        text: format!("{} {}", &text[..8], &text[25..33]),
                        entity_type: "LOC".into(),
                        confidence: anno::Confidence::ONE,
                    }]);
                }
                Ok(vec![])
            }
        }
        let m = evaluate_discontinuous_synthetic(&WrongDocument, &["LOC"], 0.5).unwrap();
        assert_eq!(m.num_predicted, 1);
        assert_eq!((m.exact_matches, m.boundary_matches), (0, 0));
        assert_eq!(
            (
                m.partial_span_precision,
                m.partial_span_recall,
                m.partial_span_f1
            ),
            (0.0, 0.0, 0.0)
        );
        let loc = &m.per_type["LOC"];
        assert!(loc.gold_count >= 1);
        assert_eq!(
            (
                loc.pred_count,
                loc.exact_matches,
                loc.exact_f1,
                loc.boundary_f1
            ),
            (1, 0, 0.0, 0.0)
        );
    }

    #[test]
    fn test_discontinuous_gold_vs_gold() {
        let metrics = evaluate_discontinuous_gold_vs_gold();
        assert!(
            (metrics.exact_f1 - 1.0).abs() < 0.001,
            "Perfect prediction should give F1=1.0, got {}",
            metrics.exact_f1
        );
    }

    #[test]
    fn test_relations_gold_vs_gold() {
        let metrics = evaluate_relations_gold_vs_gold();
        assert!(
            (metrics.strict_f1 - 1.0).abs() < 0.001,
            "Perfect prediction should give F1=1.0, got {}",
            metrics.strict_f1
        );
    }

    #[test]
    fn test_visual_gold_vs_gold() {
        let metrics = evaluate_visual_gold_vs_gold();
        assert!(
            (metrics.e2e_f1 - 1.0).abs() < 0.001,
            "Perfect prediction should give F1=1.0, got {}",
            metrics.e2e_f1
        );
    }

    #[test]
    fn test_synthetic_dataset_stats() {
        let stats = synthetic_dataset_stats();
        assert!(stats.discontinuous_examples > 0);
        assert!(stats.discontinuous_entities > 0);
        assert!(stats.relation_examples > 0);
        assert!(stats.relations > 0);
        assert!(stats.visual_examples > 0);
        assert!(stats.visual_entities > 0);
    }
}
