//! Advanced evaluators for specialized NER tasks.
//!
//! # Overview
//!
//! This module provides evaluators for tasks beyond standard NER:
//! - **Discontinuous NER**: Non-contiguous entity spans
//! - **Relation Extraction**: Entity pairs with relations
//! - **Zero-Shot NER**: Custom entity types at runtime
//! - **Event Extraction**: Triggers and arguments (future)
//!
//! # Architecture
//!
//! ```text
//! EvalTask (enum)
//!     │
//!     ├─► NER                 → StandardNEREvaluator
//!     ├─► DiscontinuousNER    → DiscontinuousEvaluator
//!     ├─► RelationExtraction  → RelationEvaluator
//!     ├─► Coreference         → CorefEvaluator
//!     └─► EventExtraction     → (future)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::eval::{AdvancedEvaluator, EvalTask};
//!
//! let evaluator = AdvancedEvaluator::for_task(&EvalTask::DiscontinuousNER {
//!     labels: vec!["LOC".into(), "PER".into()],
//! });
//! let results = evaluator.evaluate(&gold, &pred)?;
//! ```

use crate::{DiscontinuousEntity, Entity, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::discontinuous::{
    evaluate_discontinuous_ner, DiscontinuousEvalConfig, DiscontinuousGold, DiscontinuousNERMetrics,
};
use super::relation::{
    evaluate_relations, RelationEvalConfig, RelationGold, RelationMetrics, RelationPrediction,
};
use super::{EvalMode, EvalTask};

/// Unified evaluation results for any task type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalResults {
    /// Standard NER results
    NER {
        /// Precision score (0.0-1.0)
        precision: f64,
        /// Recall score (0.0-1.0)
        recall: f64,
        /// F1 score (0.0-1.0)
        f1: f64,
        /// Per-entity-type metrics breakdown
        per_type: HashMap<String, super::TypeMetrics>,
    },
    /// Discontinuous NER results
    Discontinuous(DiscontinuousNERMetrics),
    /// Relation extraction results
    Relation(RelationMetrics),
    /// Coreference results
    Coreference {
        /// CoNLL F1 (average of MUC, B³, CEAF)
        conll_f1: f64,
        /// MUC metric F1 score
        muc_f1: f64,
        /// B-cubed F1 score
        b3_f1: f64,
        /// CEAF F1 score
        ceaf_f1: f64,
    },
    /// Event extraction results (future)
    Event {
        /// Event trigger detection F1
        trigger_f1: f64,
        /// Argument extraction F1
        argument_f1: f64,
    },
}

impl EvalResults {
    /// Get the primary F1 score for this result.
    pub fn primary_f1(&self) -> f64 {
        match self {
            EvalResults::NER { f1, .. } => *f1,
            EvalResults::Discontinuous(m) => m.exact_f1,
            EvalResults::Relation(m) => m.strict_f1,
            EvalResults::Coreference { conll_f1, .. } => *conll_f1,
            EvalResults::Event { argument_f1, .. } => *argument_f1,
        }
    }

    /// Format as a summary string.
    pub fn summary(&self) -> String {
        match self {
            EvalResults::NER {
                precision,
                recall,
                f1,
                ..
            } => {
                format!(
                    "NER: P={:.1}% R={:.1}% F1={:.1}%",
                    precision * 100.0,
                    recall * 100.0,
                    f1 * 100.0
                )
            }
            EvalResults::Discontinuous(m) => {
                format!(
                    "DiscontinuousNER: ExactF1={:.1}% BoundaryF1={:.1}% PartialF1={:.1}%",
                    m.exact_f1 * 100.0,
                    m.entity_boundary_f1 * 100.0,
                    m.partial_span_f1 * 100.0
                )
            }
            EvalResults::Relation(m) => {
                format!(
                    "RelationExtraction: StrictF1={:.1}% BoundaryF1={:.1}%",
                    m.strict_f1 * 100.0,
                    m.boundary_f1 * 100.0
                )
            }
            EvalResults::Coreference {
                conll_f1,
                muc_f1,
                b3_f1,
                ceaf_f1,
            } => {
                format!(
                    "Coreference: CoNLL={:.1}% MUC={:.1}% B³={:.1}% CEAF={:.1}%",
                    conll_f1 * 100.0,
                    muc_f1 * 100.0,
                    b3_f1 * 100.0,
                    ceaf_f1 * 100.0
                )
            }
            EvalResults::Event {
                trigger_f1,
                argument_f1,
            } => {
                format!(
                    "EventExtraction: TriggerF1={:.1}% ArgumentF1={:.1}%",
                    trigger_f1 * 100.0,
                    argument_f1 * 100.0
                )
            }
        }
    }
}

/// Evaluator for discontinuous NER.
#[derive(Debug, Clone)]
pub struct DiscontinuousEvaluator {
    /// Evaluation configuration
    pub config: DiscontinuousEvalConfig,
    /// Expected entity types
    pub labels: Vec<String>,
}

impl DiscontinuousEvaluator {
    /// Create a new evaluator with default config.
    pub fn new(labels: Vec<String>) -> Self {
        Self {
            config: DiscontinuousEvalConfig::default(),
            labels,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(labels: Vec<String>, config: DiscontinuousEvalConfig) -> Self {
        Self { config, labels }
    }

    /// Evaluate predictions against gold standard.
    pub fn evaluate(
        &self,
        gold: &[DiscontinuousGold],
        pred: &[DiscontinuousEntity],
    ) -> DiscontinuousNERMetrics {
        evaluate_discontinuous_ner(gold, pred, &self.config)
    }

    /// Convert standard entities to discontinuous for comparison.
    pub fn entities_to_discontinuous(entities: &[Entity]) -> Vec<DiscontinuousEntity> {
        entities
            .iter()
            .map(|e| DiscontinuousEntity {
                spans: vec![(e.start, e.end)],
                text: e.text.clone(),
                entity_type: e.entity_type.as_label().to_string(),
                confidence: e.confidence as f32,
            })
            .collect()
    }
}

/// Evaluator for relation extraction.
#[derive(Debug, Clone)]
pub struct RelationEvaluator {
    /// Evaluation configuration
    pub config: RelationEvalConfig,
    /// Expected relation types
    pub relations: Vec<String>,
    /// Whether to require entity correctness for relation credit
    pub require_entity_match: bool,
}

impl RelationEvaluator {
    /// Create a new evaluator with default config.
    pub fn new(relations: Vec<String>, require_entity_match: bool) -> Self {
        Self {
            config: RelationEvalConfig::default(),
            relations,
            require_entity_match,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        relations: Vec<String>,
        require_entity_match: bool,
        config: RelationEvalConfig,
    ) -> Self {
        Self {
            config,
            relations,
            require_entity_match,
        }
    }

    /// Evaluate predictions against gold standard.
    pub fn evaluate(&self, gold: &[RelationGold], pred: &[RelationPrediction]) -> RelationMetrics {
        evaluate_relations(gold, pred, &self.config)
    }
}

/// Create an evaluator for a specific task.
pub fn evaluator_for_task(task: &EvalTask) -> Box<dyn TaskEvaluator> {
    match task {
        EvalTask::NER { labels, mode } => {
            Box::new(StandardNERTaskEvaluator::new(labels.clone(), *mode))
        }
        EvalTask::DiscontinuousNER { labels } => {
            Box::new(DiscontinuousTaskEvaluator::new(labels.clone()))
        }
        EvalTask::RelationExtraction {
            relations,
            require_entity_match,
        } => Box::new(RelationTaskEvaluator::new(
            relations.clone(),
            *require_entity_match,
        )),
        EvalTask::Coreference { metrics } => Box::new(CorefTaskEvaluator::new(metrics.clone())),
        EvalTask::EventExtraction {
            event_types,
            argument_roles,
        } => Box::new(EventTaskEvaluator::new(
            event_types.clone(),
            argument_roles.clone(),
        )),
    }
}

/// Trait for task-specific evaluation.
pub trait TaskEvaluator: Send + Sync {
    /// Get the task type.
    fn task(&self) -> &EvalTask;

    /// Get task name for display.
    fn name(&self) -> &str;

    /// Evaluate and return results.
    fn evaluate_generic(
        &self,
        gold: &dyn std::any::Any,
        pred: &dyn std::any::Any,
    ) -> Result<EvalResults>;
}

/// Standard NER task evaluator wrapper.
struct StandardNERTaskEvaluator {
    task: EvalTask,
}

impl StandardNERTaskEvaluator {
    fn new(labels: Vec<String>, mode: EvalMode) -> Self {
        Self {
            task: EvalTask::NER { labels, mode },
        }
    }
}

impl TaskEvaluator for StandardNERTaskEvaluator {
    fn task(&self) -> &EvalTask {
        &self.task
    }

    fn name(&self) -> &str {
        "NER"
    }

    fn evaluate_generic(
        &self,
        _gold: &dyn std::any::Any,
        _pred: &dyn std::any::Any,
    ) -> Result<EvalResults> {
        // Placeholder - actual implementation would use StandardNEREvaluator
        Ok(EvalResults::NER {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            per_type: HashMap::new(),
        })
    }
}

/// Discontinuous NER task evaluator wrapper.
struct DiscontinuousTaskEvaluator {
    task: EvalTask,
    evaluator: DiscontinuousEvaluator,
}

impl DiscontinuousTaskEvaluator {
    fn new(labels: Vec<String>) -> Self {
        Self {
            task: EvalTask::DiscontinuousNER {
                labels: labels.clone(),
            },
            evaluator: DiscontinuousEvaluator::new(labels),
        }
    }
}

impl TaskEvaluator for DiscontinuousTaskEvaluator {
    fn task(&self) -> &EvalTask {
        &self.task
    }

    fn name(&self) -> &str {
        "DiscontinuousNER"
    }

    fn evaluate_generic(
        &self,
        gold: &dyn std::any::Any,
        pred: &dyn std::any::Any,
    ) -> Result<EvalResults> {
        let gold = gold
            .downcast_ref::<Vec<DiscontinuousGold>>()
            .ok_or_else(|| crate::Error::InvalidInput("Expected Vec<DiscontinuousGold>".into()))?;
        let pred = pred
            .downcast_ref::<Vec<DiscontinuousEntity>>()
            .ok_or_else(|| {
                crate::Error::InvalidInput("Expected Vec<DiscontinuousEntity>".into())
            })?;

        let metrics = self.evaluator.evaluate(gold, pred);
        Ok(EvalResults::Discontinuous(metrics))
    }
}

/// Relation extraction task evaluator wrapper.
struct RelationTaskEvaluator {
    task: EvalTask,
    evaluator: RelationEvaluator,
}

impl RelationTaskEvaluator {
    fn new(relations: Vec<String>, require_entity_match: bool) -> Self {
        Self {
            task: EvalTask::RelationExtraction {
                relations: relations.clone(),
                require_entity_match,
            },
            evaluator: RelationEvaluator::new(relations, require_entity_match),
        }
    }
}

impl TaskEvaluator for RelationTaskEvaluator {
    fn task(&self) -> &EvalTask {
        &self.task
    }

    fn name(&self) -> &str {
        "RelationExtraction"
    }

    fn evaluate_generic(
        &self,
        gold: &dyn std::any::Any,
        pred: &dyn std::any::Any,
    ) -> Result<EvalResults> {
        let gold = gold
            .downcast_ref::<Vec<RelationGold>>()
            .ok_or_else(|| crate::Error::InvalidInput("Expected Vec<RelationGold>".into()))?;
        let pred = pred
            .downcast_ref::<Vec<RelationPrediction>>()
            .ok_or_else(|| crate::Error::InvalidInput("Expected Vec<RelationPrediction>".into()))?;

        let metrics = self.evaluator.evaluate(gold, pred);
        Ok(EvalResults::Relation(metrics))
    }
}

/// Coreference task evaluator wrapper.
struct CorefTaskEvaluator {
    task: EvalTask,
}

impl CorefTaskEvaluator {
    fn new(metrics: Vec<super::CorefMetric>) -> Self {
        Self {
            task: EvalTask::Coreference { metrics },
        }
    }
}

impl TaskEvaluator for CorefTaskEvaluator {
    fn task(&self) -> &EvalTask {
        &self.task
    }

    fn name(&self) -> &str {
        "Coreference"
    }

    fn evaluate_generic(
        &self,
        _gold: &dyn std::any::Any,
        _pred: &dyn std::any::Any,
    ) -> Result<EvalResults> {
        // Placeholder - use coref_metrics module
        Ok(EvalResults::Coreference {
            conll_f1: 0.0,
            muc_f1: 0.0,
            b3_f1: 0.0,
            ceaf_f1: 0.0,
        })
    }
}

/// Event extraction task evaluator wrapper.
struct EventTaskEvaluator {
    task: EvalTask,
}

impl EventTaskEvaluator {
    fn new(event_types: Vec<String>, argument_roles: Vec<String>) -> Self {
        Self {
            task: EvalTask::EventExtraction {
                event_types,
                argument_roles,
            },
        }
    }
}

impl TaskEvaluator for EventTaskEvaluator {
    fn task(&self) -> &EvalTask {
        &self.task
    }

    fn name(&self) -> &str {
        "EventExtraction"
    }

    fn evaluate_generic(
        &self,
        _gold: &dyn std::any::Any,
        _pred: &dyn std::any::Any,
    ) -> Result<EvalResults> {
        // Placeholder - future implementation
        Ok(EvalResults::Event {
            trigger_f1: 0.0,
            argument_f1: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discontinuous_evaluator() {
        let evaluator = DiscontinuousEvaluator::new(vec!["LOC".into()]);

        let gold = vec![DiscontinuousGold::new(
            vec![(0, 8), (25, 33)],
            "LOC",
            "New York airports",
        )];

        let pred = vec![DiscontinuousEntity {
            spans: vec![(0, 8), (25, 33)],
            text: "New York airports".to_string(),
            entity_type: "LOC".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluator.evaluate(&gold, &pred);
        assert!((metrics.exact_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_relation_evaluator() {
        let evaluator = RelationEvaluator::new(vec!["FOUNDED".into()], true);

        let gold = vec![RelationGold::new(
            (0, 10),
            "PER",
            "Steve Jobs",
            (20, 25),
            "ORG",
            "Apple",
            "FOUNDED",
        )];

        let pred = vec![RelationPrediction {
            head_span: (0, 10),
            head_type: "PER".to_string(),
            tail_span: (20, 25),
            tail_type: "ORG".to_string(),
            relation_type: "FOUNDED".to_string(),
            confidence: 0.9,
        }];

        let metrics = evaluator.evaluate(&gold, &pred);
        assert!((metrics.strict_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_evaluator_for_task() {
        let task = EvalTask::DiscontinuousNER {
            labels: vec!["LOC".into()],
        };

        let evaluator = evaluator_for_task(&task);
        assert_eq!(evaluator.name(), "DiscontinuousNER");
    }

    #[test]
    fn test_eval_results_summary() {
        let results = EvalResults::Discontinuous(DiscontinuousNERMetrics {
            exact_f1: 0.85,
            exact_precision: 0.9,
            exact_recall: 0.8,
            entity_boundary_f1: 0.9,
            entity_boundary_precision: 0.92,
            entity_boundary_recall: 0.88,
            partial_span_f1: 0.95,
            partial_span_precision: 0.96,
            partial_span_recall: 0.94,
            num_predicted: 10,
            num_gold: 10,
            exact_matches: 8,
            boundary_matches: 9,
            per_type: HashMap::new(),
        });

        let summary = results.summary();
        assert!(summary.contains("85.0%"));
    }
}
