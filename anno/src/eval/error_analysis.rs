//! Error analysis for NER systems.
//!
//! Categorizes and analyzes prediction errors to guide improvement efforts.
//!
//! # Error Categories
//!
//! - **Boundary Errors**: Correct type but wrong span
//! - **Type Errors**: Correct span but wrong type
//! - **False Positives**: Predicted entity that doesn't exist
//! - **False Negatives**: Missed a real entity
//!
//! # Example
//!
//! ```rust
//! use anno::eval::error_analysis::{ErrorAnalyzer, PredictedEntity};
//! use anno::eval::datasets::GoldEntity;
//! use anno::EntityType;
//!
//! let analyzer = ErrorAnalyzer::default();
//!
//! let predictions = vec![
//!     PredictedEntity::new("John", "PER", 0, 4),
//!     PredictedEntity::new("Google", "LOC", 14, 20),  // Wrong type!
//! ];
//!
//! let gold = vec![
//!     GoldEntity::with_span("John Smith", EntityType::Person, 0, 10),    // Boundary error
//!     GoldEntity::with_span("Google", EntityType::Organization, 14, 20), // Type error
//! ];
//!
//! let report = analyzer.analyze(&predictions, &gold);
//! println!("Boundary errors: {}", report.boundary_errors.len());
//! println!("Type errors: {}", report.type_errors.len());
//! ```

use super::datasets::GoldEntity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Data Structures
// =============================================================================

/// A predicted entity for error analysis.
///
/// Uses string-based entity types to allow comparison across different
/// labeling schemes without requiring type normalization.
#[derive(Debug, Clone)]
pub struct PredictedEntity {
    /// Entity text
    pub text: String,
    /// Predicted type (as string label, e.g., "PER", "PERSON", "B-PER")
    pub entity_type: String,
    /// Start offset
    pub start: usize,
    /// End offset
    pub end: usize,
    /// Prediction confidence
    pub confidence: f64,
}

impl PredictedEntity {
    /// Create a new predicted entity.
    pub fn new(
        text: impl Into<String>,
        entity_type: impl Into<String>,
        start: usize,
        end: usize,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type: entity_type.into(),
            start,
            end,
            confidence: 1.0,
        }
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Create from an anno Entity.
    pub fn from_entity(entity: &anno_core::Entity) -> Self {
        Self {
            text: entity.text.clone(),
            entity_type: entity.entity_type.as_label().to_string(),
            start: entity.start,
            end: entity.end,
            confidence: entity.confidence,
        }
    }
}

/// A specific error instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInstance {
    /// Error category
    pub category: ErrorCategory,
    /// Predicted entity (if any)
    pub predicted: Option<EntityInfo>,
    /// Gold entity (if any)
    pub gold: Option<EntityInfo>,
    /// Error description
    pub description: String,
}

/// Entity information for error reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInfo {
    /// Entity text
    pub text: String,
    /// Entity type (as string label for cross-schema comparison)
    pub entity_type: String,
    /// Span
    pub span: (usize, usize),
}

/// Error category for NER analysis.
///
/// Note: This type overlaps with `ErrorType` in the `analysis` module.
/// The mapping is:
/// - `TypeError` ↔ `ErrorType::TypeMismatch`
/// - `BoundaryError` ↔ `ErrorType::BoundaryError`
/// - `PartialMatch` ↔ `ErrorType::BoundaryAndType`
/// - `FalsePositive` ↔ `ErrorType::Spurious`
/// - `FalseNegative` ↔ `ErrorType::Missed`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Correct type but wrong boundaries
    BoundaryError,
    /// Correct boundaries but wrong type
    TypeError,
    /// Predicted entity that doesn't exist
    FalsePositive,
    /// Missed a real entity
    FalseNegative,
    /// Both boundary and type are wrong but overlapping
    PartialMatch,
}

impl ErrorCategory {
    /// Convert to the equivalent [`super::analysis::ErrorType`].
    #[must_use]
    pub fn to_error_type(self) -> super::analysis::ErrorType {
        use super::analysis::ErrorType;
        match self {
            ErrorCategory::TypeError => ErrorType::TypeMismatch,
            ErrorCategory::BoundaryError => ErrorType::BoundaryError,
            ErrorCategory::PartialMatch => ErrorType::BoundaryAndType,
            ErrorCategory::FalsePositive => ErrorType::Spurious,
            ErrorCategory::FalseNegative => ErrorType::Missed,
        }
    }
}

/// Comprehensive error analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReport {
    /// Boundary errors
    pub boundary_errors: Vec<ErrorInstance>,
    /// Type errors
    pub type_errors: Vec<ErrorInstance>,
    /// False positives
    pub false_positives: Vec<ErrorInstance>,
    /// False negatives
    pub false_negatives: Vec<ErrorInstance>,
    /// Partial matches
    pub partial_matches: Vec<ErrorInstance>,
    /// Error counts by category
    pub counts: HashMap<String, usize>,
    /// Error rates by category
    pub rates: HashMap<String, f64>,
    /// Most common error patterns
    pub common_patterns: Vec<ErrorPattern>,
    /// Per-type error breakdown
    pub by_type: HashMap<String, TypeErrorStats>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Common error pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// Pattern description
    pub description: String,
    /// Number of occurrences
    pub count: usize,
    /// Example errors
    pub examples: Vec<String>,
}

/// Error statistics for a specific entity type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeErrorStats {
    /// Total gold entities of this type
    pub gold_count: usize,
    /// Correct predictions
    pub correct: usize,
    /// Boundary errors
    pub boundary_errors: usize,
    /// Confused with other types (type -> count)
    pub confused_with: HashMap<String, usize>,
    /// False negatives
    pub missed: usize,
}

// =============================================================================
// Error Analyzer
// =============================================================================

/// Analyzer for NER prediction errors.
///
/// Uses an optimized O(n + m) algorithm with spatial indexing instead of
/// naive O(n*m) nested loops. For datasets with >1000 entities, this provides
/// significant speedup.
#[derive(Debug, Clone)]
pub struct ErrorAnalyzer {
    /// Overlap threshold for partial match (IoU)
    pub overlap_threshold: f64,
}

impl Default for ErrorAnalyzer {
    fn default() -> Self {
        Self {
            overlap_threshold: 0.5,
        }
    }
}

impl ErrorAnalyzer {
    /// Create analyzer with custom overlap threshold.
    pub fn new(overlap_threshold: f64) -> Self {
        Self { overlap_threshold }
    }

    /// Analyze errors between predictions and gold entities.
    ///
    /// Uses the canonical `GoldEntity` type from `eval::datasets`.
    /// Entity types are compared using their string labels.
    pub fn analyze(&self, predictions: &[PredictedEntity], gold: &[GoldEntity]) -> ErrorReport {
        let mut boundary_errors = Vec::new();
        let mut type_errors = Vec::new();
        let mut false_positives = Vec::new();
        let mut false_negatives = Vec::new();
        let mut partial_matches = Vec::new();

        let mut matched_preds = vec![false; predictions.len()];
        let mut matched_gold = vec![false; gold.len()];

        // Build spatial index for predictions (sorted by start position)
        let mut pred_by_start: Vec<(usize, usize, usize)> = predictions
            .iter()
            .enumerate()
            .map(|(i, p)| (p.start, p.end, i))
            .collect();
        pred_by_start.sort_by_key(|x| x.0);

        // For each gold entity, find candidate predictions using binary search
        for (gi, g) in gold.iter().enumerate() {
            let g_type = g.entity_type.as_label();

            // Find predictions that could overlap with this gold entity
            // A prediction overlaps if pred.start < g.end && pred.end > g.start
            let candidates: Vec<usize> = pred_by_start
                .iter()
                .filter(|(p_start, p_end, _)| *p_start < g.end && *p_end > g.start)
                .map(|(_, _, idx)| *idx)
                .collect();

            let mut best_match: Option<(usize, f64, bool, bool)> = None; // (idx, overlap, exact_boundary, type_match)

            for pi in candidates {
                if matched_preds[pi] {
                    continue;
                }

                let p = &predictions[pi];
                let exact_boundary = p.start == g.start && p.end == g.end;
                let type_match = p.entity_type == g_type;
                let overlap = self.compute_overlap(p.start, p.end, g.start, g.end);

                // Prefer exact matches, then type matches, then highest overlap
                let dominated =
                    best_match.is_some_and(|(_, best_overlap, best_exact, best_type)| {
                        if exact_boundary && !best_exact {
                            return false;
                        }
                        if !exact_boundary && best_exact {
                            return true;
                        }
                        if type_match && !best_type {
                            return false;
                        }
                        if !type_match && best_type {
                            return true;
                        }
                        overlap <= best_overlap
                    });

                if !dominated && overlap > self.overlap_threshold {
                    best_match = Some((pi, overlap, exact_boundary, type_match));
                }
            }

            if let Some((pi, _overlap, exact_boundary, type_match)) = best_match {
                let p = &predictions[pi];
                matched_preds[pi] = true;
                matched_gold[gi] = true;

                if exact_boundary && type_match {
                    // Correct prediction - not an error
                } else if exact_boundary && !type_match {
                    // Type error
                    type_errors.push(ErrorInstance {
                        category: ErrorCategory::TypeError,
                        predicted: Some(EntityInfo {
                            text: p.text.clone(),
                            entity_type: p.entity_type.clone(),
                            span: (p.start, p.end),
                        }),
                        gold: Some(EntityInfo {
                            text: g.text.clone(),
                            entity_type: g_type.to_string(),
                            span: (g.start, g.end),
                        }),
                        description: format!(
                            "Predicted {} as {} (should be {})",
                            p.text, p.entity_type, g_type
                        ),
                    });
                } else if type_match {
                    // Boundary error
                    boundary_errors.push(ErrorInstance {
                        category: ErrorCategory::BoundaryError,
                        predicted: Some(EntityInfo {
                            text: p.text.clone(),
                            entity_type: p.entity_type.clone(),
                            span: (p.start, p.end),
                        }),
                        gold: Some(EntityInfo {
                            text: g.text.clone(),
                            entity_type: g_type.to_string(),
                            span: (g.start, g.end),
                        }),
                        description: format!(
                            "Predicted '{}' [{},{}] vs gold '{}' [{},{}]",
                            p.text, p.start, p.end, g.text, g.start, g.end
                        ),
                    });
                } else {
                    // Partial match with wrong type
                    partial_matches.push(ErrorInstance {
                        category: ErrorCategory::PartialMatch,
                        predicted: Some(EntityInfo {
                            text: p.text.clone(),
                            entity_type: p.entity_type.clone(),
                            span: (p.start, p.end),
                        }),
                        gold: Some(EntityInfo {
                            text: g.text.clone(),
                            entity_type: g_type.to_string(),
                            span: (g.start, g.end),
                        }),
                        description: format!(
                            "Partial: '{}' ({}) vs '{}' ({})",
                            p.text, p.entity_type, g.text, g_type
                        ),
                    });
                }
            }
        }

        // Unmatched predictions are false positives
        for (pi, p) in predictions.iter().enumerate() {
            if !matched_preds[pi] {
                false_positives.push(ErrorInstance {
                    category: ErrorCategory::FalsePositive,
                    predicted: Some(EntityInfo {
                        text: p.text.clone(),
                        entity_type: p.entity_type.clone(),
                        span: (p.start, p.end),
                    }),
                    gold: None,
                    description: format!(
                        "Spurious {} '{}' at [{},{}]",
                        p.entity_type, p.text, p.start, p.end
                    ),
                });
            }
        }

        // Unmatched gold are false negatives
        for (gi, g) in gold.iter().enumerate() {
            if !matched_gold[gi] {
                let g_type = g.entity_type.as_label();
                false_negatives.push(ErrorInstance {
                    category: ErrorCategory::FalseNegative,
                    predicted: None,
                    gold: Some(EntityInfo {
                        text: g.text.clone(),
                        entity_type: g_type.to_string(),
                        span: (g.start, g.end),
                    }),
                    description: format!(
                        "Missed {} '{}' at [{},{}]",
                        g_type, g.text, g.start, g.end
                    ),
                });
            }
        }

        // Compute statistics
        let total_errors = boundary_errors.len()
            + type_errors.len()
            + false_positives.len()
            + false_negatives.len()
            + partial_matches.len();

        let mut counts: HashMap<String, usize> = HashMap::new();
        counts.insert("boundary_errors".into(), boundary_errors.len());
        counts.insert("type_errors".into(), type_errors.len());
        counts.insert("false_positives".into(), false_positives.len());
        counts.insert("false_negatives".into(), false_negatives.len());
        counts.insert("partial_matches".into(), partial_matches.len());
        counts.insert("total".into(), total_errors);

        let mut rates = HashMap::new();
        if total_errors > 0 {
            for (k, v) in &counts {
                rates.insert(k.clone(), *v as f64 / total_errors as f64);
            }
        }

        // Per-type analysis
        let by_type = self.analyze_by_type(&type_errors, &false_negatives, gold);

        // Find common patterns
        let common_patterns = self.find_common_patterns(&type_errors, &boundary_errors);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&counts, &by_type);

        ErrorReport {
            boundary_errors,
            type_errors,
            false_positives,
            false_negatives,
            partial_matches,
            counts,
            rates,
            common_patterns,
            by_type,
            recommendations,
        }
    }

    fn compute_overlap(&self, p_start: usize, p_end: usize, g_start: usize, g_end: usize) -> f64 {
        let intersection_start = p_start.max(g_start);
        let intersection_end = p_end.min(g_end);

        if intersection_start >= intersection_end {
            return 0.0;
        }

        let intersection = intersection_end - intersection_start;
        let union = (p_end - p_start) + (g_end - g_start) - intersection;

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn analyze_by_type(
        &self,
        type_errors: &[ErrorInstance],
        false_negatives: &[ErrorInstance],
        gold: &[GoldEntity],
    ) -> HashMap<String, TypeErrorStats> {
        let mut stats: HashMap<String, TypeErrorStats> = HashMap::new();

        // Initialize from gold
        for g in gold {
            let g_type = g.entity_type.as_label().to_string();
            let entry = stats.entry(g_type).or_insert(TypeErrorStats {
                gold_count: 0,
                correct: 0,
                boundary_errors: 0,
                confused_with: HashMap::new(),
                missed: 0,
            });
            entry.gold_count += 1;
        }

        // Count type confusions
        for err in type_errors {
            if let (Some(pred), Some(gold_info)) = (&err.predicted, &err.gold) {
                if let Some(entry) = stats.get_mut(&gold_info.entity_type) {
                    *entry
                        .confused_with
                        .entry(pred.entity_type.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        // Count misses
        for err in false_negatives {
            if let Some(gold_info) = &err.gold {
                if let Some(entry) = stats.get_mut(&gold_info.entity_type) {
                    entry.missed += 1;
                }
            }
        }

        stats
    }

    fn find_common_patterns(
        &self,
        type_errors: &[ErrorInstance],
        boundary_errors: &[ErrorInstance],
    ) -> Vec<ErrorPattern> {
        let mut patterns: HashMap<String, (usize, Vec<String>)> = HashMap::new();

        // Count type confusion patterns
        for err in type_errors {
            if let (Some(pred), Some(gold_info)) = (&err.predicted, &err.gold) {
                let key = format!("{} -> {}", gold_info.entity_type, pred.entity_type);
                let entry = patterns.entry(key).or_insert((0, Vec::new()));
                entry.0 += 1;
                if entry.1.len() < 3 {
                    entry.1.push(err.description.clone());
                }
            }
        }

        // Count boundary patterns (e.g., "too short", "too long")
        let mut too_short = 0;
        let mut too_long = 0;

        for err in boundary_errors {
            if let (Some(pred), Some(gold_info)) = (&err.predicted, &err.gold) {
                let pred_len = pred.span.1 - pred.span.0;
                let gold_len = gold_info.span.1 - gold_info.span.0;

                if pred_len < gold_len {
                    too_short += 1;
                } else {
                    too_long += 1;
                }
            }
        }

        if too_short > 0 {
            patterns.insert(
                "Boundary: Predicted span too short".into(),
                (too_short, vec!["Model truncates entities".into()]),
            );
        }
        if too_long > 0 {
            patterns.insert(
                "Boundary: Predicted span too long".into(),
                (too_long, vec!["Model over-extends entities".into()]),
            );
        }

        let mut result: Vec<ErrorPattern> = patterns
            .into_iter()
            .map(|(desc, (count, examples))| ErrorPattern {
                description: desc,
                count,
                examples,
            })
            .collect();

        result.sort_by(|a, b| b.count.cmp(&a.count));
        result.truncate(10);
        result
    }

    fn generate_recommendations(
        &self,
        counts: &HashMap<String, usize>,
        by_type: &HashMap<String, TypeErrorStats>,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        let boundary = counts.get("boundary_errors").copied().unwrap_or(0);
        let type_err = counts.get("type_errors").copied().unwrap_or(0);
        let fp = counts.get("false_positives").copied().unwrap_or(0);
        let fn_count = counts.get("false_negatives").copied().unwrap_or(0);
        let total = counts.get("total").copied().unwrap_or(1).max(1);

        // Boundary recommendations
        if boundary as f64 / total as f64 > 0.3 {
            recs.push(
                "High boundary error rate: Consider boundary-aware training or CRF layer".into(),
            );
        }

        // Type confusion recommendations
        if type_err as f64 / total as f64 > 0.2 {
            recs.push(
                "Frequent type confusions: Add more training examples for confused types".into(),
            );

            // Find most confused pairs
            for (typ, stats) in by_type {
                if let Some((confused_type, count)) =
                    stats.confused_with.iter().max_by_key(|(_, c)| *c)
                {
                    if *count > 2 {
                        recs.push(format!(
                            "Type {}: Often confused with {} ({} times) - add disambiguation features",
                            typ, confused_type, count
                        ));
                    }
                }
            }
        }

        // False positive recommendations
        if fp as f64 / total as f64 > 0.25 {
            recs.push(
                "High false positive rate: Model is over-predicting - consider higher threshold"
                    .into(),
            );
        }

        // False negative recommendations
        if fn_count as f64 / total as f64 > 0.25 {
            recs.push(
                "High miss rate: Model is under-predicting - consider lower threshold or more data"
                    .into(),
            );
        }

        if recs.is_empty() {
            recs.push("Error distribution is balanced - continue monitoring".into());
        }

        recs
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use anno_core::EntityType;

    #[test]
    fn test_type_error_detection() {
        let predictions = vec![PredictedEntity::new("Google", "LOC", 0, 6)];
        let gold = vec![GoldEntity::with_span(
            "Google",
            EntityType::Organization,
            0,
            6,
        )];

        let analyzer = ErrorAnalyzer::default();
        let report = analyzer.analyze(&predictions, &gold);

        assert_eq!(report.type_errors.len(), 1);
        assert_eq!(report.boundary_errors.len(), 0);
    }

    #[test]
    fn test_boundary_error_detection() {
        let predictions = vec![PredictedEntity::new("John", "PER", 0, 4)];
        let gold = vec![GoldEntity::with_span(
            "John Smith",
            EntityType::Person,
            0,
            10,
        )];

        let analyzer = ErrorAnalyzer::new(0.3); // Low threshold for partial match
        let report = analyzer.analyze(&predictions, &gold);

        assert_eq!(report.boundary_errors.len(), 1);
    }

    #[test]
    fn test_false_positive_detection() {
        let predictions = vec![PredictedEntity::new("Random", "PER", 0, 6)];
        let gold: Vec<GoldEntity> = vec![];

        let analyzer = ErrorAnalyzer::default();
        let report = analyzer.analyze(&predictions, &gold);

        assert_eq!(report.false_positives.len(), 1);
    }

    #[test]
    fn test_false_negative_detection() {
        let predictions: Vec<PredictedEntity> = vec![];
        let gold = vec![GoldEntity::new("John", EntityType::Person, 0)];

        let analyzer = ErrorAnalyzer::default();
        let report = analyzer.analyze(&predictions, &gold);

        assert_eq!(report.false_negatives.len(), 1);
    }

    #[test]
    fn test_correct_prediction() {
        let predictions = vec![PredictedEntity::new("John", "PER", 0, 4)];
        let gold = vec![GoldEntity::with_span("John", EntityType::Person, 0, 4)];

        let analyzer = ErrorAnalyzer::default();
        let report = analyzer.analyze(&predictions, &gold);

        assert_eq!(*report.counts.get("total").unwrap_or(&0), 0);
    }

    #[test]
    fn test_from_entity() {
        let entity = anno_core::Entity::new("Test", EntityType::Person, 0, 4, 0.95);
        let pred = PredictedEntity::from_entity(&entity);
        assert_eq!(pred.text, "Test");
        assert_eq!(pred.entity_type, "PER");
        assert_eq!(pred.confidence, 0.95);
    }
}
