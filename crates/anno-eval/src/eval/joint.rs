//! Joint entity analysis evaluation.
//!
//! Evaluates the three tasks in joint entity analysis:
//! - Named Entity Recognition (NER/semantic typing)
//! - Coreference Resolution
//! - Entity Linking
//!
//! Following Durrett & Klein (2014) evaluation protocol on ACE/OntoNotes.
//!
//! # Metrics
//!
//! | Task | Metrics |
//! |------|---------|
//! | NER | Precision, Recall, F1 (strict/partial) |
//! | Coref | MUC, B³, CEAFe, CoNLL F1 |
//! | Linking | Accuracy, Precision@k, NIL F1 |
//! | Joint | Weighted average, error analysis |
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::eval::joint::{JointEvaluator, JointGoldDocument};
//! use anno::joint::JointModel;
//!
//! let evaluator = JointEvaluator::new();
//! let results = evaluator.evaluate(&model, &gold_docs)?;
//! println!("NER F1: {:.2}", results.ner.f1);
//! println!("CoNLL F1: {:.2}", results.coref.conll_f1);
//! println!("Link Acc: {:.2}", results.linking.accuracy);
//! ```

use super::coref::CorefChain;
use super::coref_metrics::CorefEvaluation;
use super::ner_metrics::evaluate_entities;
use anno::joint::JointModel;
use anno::{Entity, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Gold Standard Types
// =============================================================================

/// Gold standard document for joint evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointGoldDocument {
    /// Document ID
    pub doc_id: String,
    /// Raw text
    pub text: String,
    /// Gold NER annotations
    pub entities: Vec<GoldEntity>,
    /// Gold coreference chains
    pub coref_chains: Vec<CorefChain>,
    /// Gold entity links (mention_idx -> KB_ID, None for NIL)
    pub links: Vec<GoldLink>,
}

/// Gold entity annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldEntity {
    /// Entity text
    pub text: String,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
    /// Entity type label
    pub entity_type: String,
}

/// Gold entity link annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldLink {
    /// Mention text
    pub mention_text: String,
    /// Start offset
    pub start: usize,
    /// End offset
    pub end: usize,
    /// Knowledge base ID (None for NIL)
    pub kb_id: Option<String>,
}

// =============================================================================
// Result Types
// =============================================================================

/// NER evaluation results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NERResults {
    /// Strict precision (exact match)
    pub precision: f64,
    /// Strict recall
    pub recall: f64,
    /// Strict F1
    pub f1: f64,
    /// Partial precision (overlap match)
    pub partial_precision: f64,
    /// Partial recall
    pub partial_recall: f64,
    /// Partial F1
    pub partial_f1: f64,
    /// Per-type metrics
    pub per_type: HashMap<String, TypeMetrics>,
}

/// Per-type metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeMetrics {
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1
    pub f1: f64,
    /// Gold count
    pub gold_count: usize,
    /// Predicted count
    pub pred_count: usize,
}

/// Entity linking results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinkingResults {
    /// Accuracy (correct predictions / total)
    pub accuracy: f64,
    /// Precision for non-NIL predictions
    pub precision: f64,
    /// Recall for non-NIL gold
    pub recall: f64,
    /// F1 for non-NIL
    pub f1: f64,
    /// NIL prediction F1
    pub nil_f1: f64,
    /// Mean reciprocal rank (if multiple candidates)
    pub mrr: f64,
}

/// Complete joint evaluation results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JointEvalResults {
    /// NER metrics
    pub ner: NERResults,
    /// Coreference metrics
    pub coref: CorefEvaluation,
    /// Entity linking metrics
    pub linking: LinkingResults,
    /// Number of documents evaluated
    pub num_documents: usize,
    /// Error analysis
    pub errors: JointErrorAnalysis,
}

impl JointEvalResults {
    /// Compute weighted joint F1 score.
    ///
    /// Following Durrett & Klein (2014), we average the three task F1s.
    #[must_use]
    pub fn joint_f1(&self) -> f64 {
        (self.ner.f1 + self.coref.conll_f1 + self.linking.f1) / 3.0
    }

    /// Summary string for display.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Joint F1: {:.1}% | NER: {:.1}% | CoNLL: {:.1}% | Link: {:.1}%",
            self.joint_f1() * 100.0,
            self.ner.f1 * 100.0,
            self.coref.conll_f1 * 100.0,
            self.linking.f1 * 100.0
        )
    }
}

/// Error analysis for joint model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JointErrorAnalysis {
    /// NER errors that cascaded to coref errors
    pub ner_coref_cascade: usize,
    /// NER errors that cascaded to linking errors
    pub ner_link_cascade: usize,
    /// Coref errors with correct NER
    pub coref_only_errors: usize,
    /// Link errors with correct NER
    pub link_only_errors: usize,
    /// Type-link inconsistencies caught
    pub type_link_conflicts: usize,
}

// =============================================================================
// Evaluator
// =============================================================================

/// Joint entity analysis evaluator.
#[derive(Debug, Clone)]
pub struct JointEvaluator {
    /// Whether to compute detailed per-type metrics
    pub per_type_metrics: bool,
    /// Whether to compute error cascade analysis
    pub error_analysis: bool,
    /// Minimum confidence threshold for predictions
    pub confidence_threshold: f64,
}

impl Default for JointEvaluator {
    fn default() -> Self {
        Self {
            per_type_metrics: true,
            error_analysis: true,
            confidence_threshold: 0.0,
        }
    }
}

impl JointEvaluator {
    /// Create a new joint evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluate joint model on gold documents.
    pub fn evaluate(
        &self,
        model: &JointModel,
        gold_docs: &[JointGoldDocument],
    ) -> crate::Result<JointEvalResults> {
        let mut all_gold_entities: Vec<Entity> = Vec::new();
        let mut all_pred_entities: Vec<Entity> = Vec::new();
        let mut all_gold_chains = Vec::new();
        let mut all_pred_chains = Vec::new();
        let mut all_gold_links = Vec::new();
        let mut all_pred_links = Vec::new();

        for doc in gold_docs {
            // Convert gold entities to format expected by model
            let input_entities: Vec<Entity> = doc
                .entities
                .iter()
                .map(|g| {
                    let entity_type = g
                        .entity_type
                        .parse()
                        .unwrap_or_else(|_| EntityType::Other(g.entity_type.clone()));
                    Entity::new(&g.text, entity_type, g.start, g.end, 1.0)
                })
                .collect();

            // Run joint model
            let result = model.analyze(&doc.text, &input_entities)?;

            // Collect NER predictions
            all_pred_entities.extend(result.entities.clone());

            // Collect gold entities
            for gold in &doc.entities {
                let entity_type = gold
                    .entity_type
                    .parse()
                    .unwrap_or_else(|_| EntityType::Other(gold.entity_type.clone()));
                all_gold_entities.push(Entity::new(
                    &gold.text,
                    entity_type,
                    gold.start,
                    gold.end,
                    1.0,
                ));
            }

            // Collect coref chains
            all_pred_chains.extend(result.chains.clone());
            all_gold_chains.extend(doc.coref_chains.clone());

            // Collect entity links
            for link in &result.links {
                all_pred_links.push(PredictedLink {
                    start: link.start,
                    end: link.end,
                    kb_id: link.kb_id.clone(),
                    confidence: link.confidence,
                });
            }
            for link in &doc.links {
                all_gold_links.push(link.clone());
            }
        }

        // Compute NER metrics
        let ner_eval = evaluate_entities(&all_gold_entities, &all_pred_entities);
        let ner_summary = ner_eval.summary();
        let ner = NERResults {
            precision: ner_summary.strict_precision,
            recall: ner_summary.strict_recall,
            f1: ner_summary.strict_f1,
            partial_precision: ner_summary.partial_precision,
            partial_recall: ner_summary.partial_recall,
            partial_f1: ner_summary.partial_f1,
            per_type: self.compute_per_type_ner(&all_gold_entities, &all_pred_entities),
        };

        // Compute coref metrics
        let coref = CorefEvaluation::compute(&all_pred_chains, &all_gold_chains);

        // Compute linking metrics
        let linking = self.compute_linking_metrics(&all_gold_links, &all_pred_links);

        // Error analysis
        let errors = if self.error_analysis {
            self.analyze_errors(
                &all_gold_entities,
                &all_pred_entities,
                &all_gold_links,
                &all_pred_links,
            )
        } else {
            JointErrorAnalysis::default()
        };

        Ok(JointEvalResults {
            ner,
            coref,
            linking,
            num_documents: gold_docs.len(),
            errors,
        })
    }

    fn compute_per_type_ner(
        &self,
        gold: &[Entity],
        pred: &[Entity],
    ) -> HashMap<String, TypeMetrics> {
        let mut per_type: HashMap<String, (usize, usize, usize)> = HashMap::new();

        // Count gold by type
        for g in gold {
            let type_label = g.entity_type.as_label().to_string();
            let entry = per_type.entry(type_label).or_insert((0, 0, 0));
            entry.0 += 1;
        }

        // Count predicted by type
        for p in pred {
            let type_label = p.entity_type.as_label().to_string();
            let entry = per_type.entry(type_label).or_insert((0, 0, 0));
            entry.1 += 1;
        }

        // Count correct matches
        for g in gold {
            let type_label = g.entity_type.as_label().to_string();
            let matched = pred
                .iter()
                .any(|p| p.start == g.start && p.end == g.end && p.entity_type == g.entity_type);
            if matched {
                let entry = per_type.entry(type_label).or_insert((0, 0, 0));
                entry.2 += 1;
            }
        }

        // Compute metrics
        per_type
            .into_iter()
            .map(|(type_name, (gold_count, pred_count, correct))| {
                let precision = if pred_count > 0 {
                    correct as f64 / pred_count as f64
                } else {
                    0.0
                };
                let recall = if gold_count > 0 {
                    correct as f64 / gold_count as f64
                } else {
                    0.0
                };
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };
                (
                    type_name,
                    TypeMetrics {
                        precision,
                        recall,
                        f1,
                        gold_count,
                        pred_count,
                    },
                )
            })
            .collect()
    }

    fn compute_linking_metrics(&self, gold: &[GoldLink], pred: &[PredictedLink]) -> LinkingResults {
        if gold.is_empty() {
            return LinkingResults::default();
        }

        let mut correct = 0;
        let mut nil_tp = 0;
        let mut nil_fp = 0;
        let mut nil_fn = 0;
        let mut non_nil_tp = 0;
        let mut non_nil_fp = 0;
        let mut non_nil_fn = 0;

        for g in gold {
            let matched_pred = pred.iter().find(|p| p.start == g.start && p.end == g.end);

            match (
                g.kb_id.as_ref(),
                matched_pred.and_then(|p| p.kb_id.as_ref()),
            ) {
                (Some(gold_kb), Some(pred_kb)) if gold_kb == pred_kb => {
                    correct += 1;
                    non_nil_tp += 1;
                }
                (Some(_), Some(_)) => {
                    // Wrong KB ID
                    non_nil_fn += 1;
                    non_nil_fp += 1;
                }
                (Some(_), None) => {
                    // Missed link
                    non_nil_fn += 1;
                    nil_fp += 1;
                }
                (None, Some(_)) => {
                    // False positive
                    nil_fn += 1;
                    non_nil_fp += 1;
                }
                (None, None) => {
                    // Correct NIL
                    correct += 1;
                    nil_tp += 1;
                }
            }
        }

        let accuracy = correct as f64 / gold.len() as f64;

        let non_nil_precision = if non_nil_tp + non_nil_fp > 0 {
            non_nil_tp as f64 / (non_nil_tp + non_nil_fp) as f64
        } else {
            0.0
        };
        let non_nil_recall = if non_nil_tp + non_nil_fn > 0 {
            non_nil_tp as f64 / (non_nil_tp + non_nil_fn) as f64
        } else {
            0.0
        };
        let f1 = if non_nil_precision + non_nil_recall > 0.0 {
            2.0 * non_nil_precision * non_nil_recall / (non_nil_precision + non_nil_recall)
        } else {
            0.0
        };

        let nil_precision = if nil_tp + nil_fp > 0 {
            nil_tp as f64 / (nil_tp + nil_fp) as f64
        } else {
            0.0
        };
        let nil_recall = if nil_tp + nil_fn > 0 {
            nil_tp as f64 / (nil_tp + nil_fn) as f64
        } else {
            0.0
        };
        let nil_f1 = if nil_precision + nil_recall > 0.0 {
            2.0 * nil_precision * nil_recall / (nil_precision + nil_recall)
        } else {
            0.0
        };

        LinkingResults {
            accuracy,
            precision: non_nil_precision,
            recall: non_nil_recall,
            f1,
            nil_f1,
            mrr: 0.0, // Would require candidate rankings
        }
    }

    fn analyze_errors(
        &self,
        gold_ner: &[Entity],
        pred_ner: &[Entity],
        gold_links: &[GoldLink],
        pred_links: &[PredictedLink],
    ) -> JointErrorAnalysis {
        let mut analysis = JointErrorAnalysis::default();

        // Find NER errors
        let ner_errors: Vec<_> = gold_ner
            .iter()
            .filter(|g| {
                !pred_ner
                    .iter()
                    .any(|p| p.start == g.start && p.end == g.end && p.entity_type == g.entity_type)
            })
            .collect();

        // Check if NER errors cascade to link errors
        for ner_err in &ner_errors {
            let has_link_gold = gold_links
                .iter()
                .any(|l| l.start == ner_err.start && l.end == ner_err.end);
            let has_link_pred = pred_links
                .iter()
                .any(|l| l.start == ner_err.start && l.end == ner_err.end);

            if has_link_gold && !has_link_pred {
                analysis.ner_link_cascade += 1;
            }
        }

        // Count link-only errors (correct NER, wrong link)
        for gold_link in gold_links {
            let ner_correct = pred_ner
                .iter()
                .any(|p| p.start == gold_link.start && p.end == gold_link.end);
            let link_correct = pred_links.iter().any(|p| {
                p.start == gold_link.start && p.end == gold_link.end && p.kb_id == gold_link.kb_id
            });

            if ner_correct && !link_correct {
                analysis.link_only_errors += 1;
            }
        }

        analysis
    }
}

/// Predicted link (internal).
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for construction; confidence useful for debugging
struct PredictedLink {
    start: usize,
    end: usize,
    kb_id: Option<String>,
    confidence: f64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use anno::joint::JointConfig;

    #[test]
    fn test_joint_evaluator_empty() {
        let evaluator = JointEvaluator::new();
        let model = JointModel::new(JointConfig::default()).unwrap();
        let results = evaluator.evaluate(&model, &[]).unwrap();

        assert_eq!(results.num_documents, 0);
    }

    #[test]
    fn test_linking_results_perfect() {
        let evaluator = JointEvaluator::new();

        let gold = vec![GoldLink {
            mention_text: "Paris".to_string(),
            start: 0,
            end: 5,
            kb_id: Some("Q90".to_string()),
        }];

        let pred = vec![PredictedLink {
            start: 0,
            end: 5,
            kb_id: Some("Q90".to_string()),
            confidence: 0.9,
        }];

        let results = evaluator.compute_linking_metrics(&gold, &pred);
        assert!((results.accuracy - 1.0).abs() < 0.001);
        assert!((results.f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_linking_results_nil() {
        let evaluator = JointEvaluator::new();

        let gold = vec![GoldLink {
            mention_text: "unknown".to_string(),
            start: 0,
            end: 7,
            kb_id: None, // NIL
        }];

        let pred = vec![PredictedLink {
            start: 0,
            end: 7,
            kb_id: None, // Correct NIL
            confidence: 0.8,
        }];

        let results = evaluator.compute_linking_metrics(&gold, &pred);
        assert!((results.accuracy - 1.0).abs() < 0.001);
        assert!((results.nil_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_joint_eval_results_summary() {
        let results = JointEvalResults {
            ner: NERResults {
                f1: 0.85,
                ..Default::default()
            },
            coref: CorefEvaluation {
                conll_f1: 0.72,
                ..Default::default()
            },
            linking: LinkingResults {
                f1: 0.68,
                ..Default::default()
            },
            ..Default::default()
        };

        let joint_f1 = results.joint_f1();
        // (0.85 + 0.72 + 0.68) / 3 = 0.75
        assert!((joint_f1 - 0.75).abs() < 0.001);

        let summary = results.summary();
        assert!(summary.contains("Joint F1: 75.0%"));
    }

    #[test]
    fn test_per_type_ner_metrics() {
        let evaluator = JointEvaluator::new();

        let gold = vec![
            Entity::new("Paris", EntityType::Location, 0, 5, 1.0),
            Entity::new("Obama", EntityType::Person, 10, 15, 1.0),
        ];

        let pred = vec![Entity::new("Paris", EntityType::Location, 0, 5, 0.9)];

        let per_type = evaluator.compute_per_type_ner(&gold, &pred);

        // LOC: 1 gold, 1 pred, 1 correct -> P=1, R=1, F1=1
        let loc = per_type.get("LOC").unwrap();
        assert!((loc.f1 - 1.0).abs() < 0.001);

        // PER: 1 gold, 0 pred, 0 correct -> P=0, R=0, F1=0
        let per = per_type.get("PER").unwrap();
        assert!((per.f1 - 0.0).abs() < 0.001);
    }
}
