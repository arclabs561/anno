//! Active learning utilities for NER annotation prioritization.
//!
//! These functions help decide which texts are most valuable to annotate next,
//! by scoring them according to model uncertainty. Texts where the model is least
//! confident produce the highest priority scores.
//!
//! # Example
//!
//! ```rust
//! use anno::{Entity, EntityType};
//! use anno::active::{annotation_priority, rank_for_annotation};
//!
//! let entities = vec![
//!     Entity::new("Alice", EntityType::Person, 0, 5, 0.6),
//!     Entity::new("IBM", EntityType::Organization, 10, 13, 0.9),
//! ];
//!
//! // Priority = max uncertainty = 1 - min(confidence)
//! let priority = annotation_priority(&entities);
//! assert!((priority - 0.4).abs() < 1e-10);
//!
//! // Rank a batch of texts
//! let batch: Vec<(&str, Vec<Entity>)> = vec![
//!     ("Alice met Bob.", entities),
//! ];
//! let ranked = rank_for_annotation(&batch);
//! assert_eq!(ranked[0].0, 0);
//! ```

use crate::{Entity, HierarchicalConfidence};

/// Per-span uncertainty breakdown.
///
/// Each field is `1 - confidence`, so higher values mean more uncertain.
pub struct SpanUncertainty {
    /// Overall uncertainty derived from the entity's top-level confidence score.
    pub overall: f64,
    /// Boundary uncertainty: how unsure the model is about exact span boundaries.
    pub boundary: f64,
    /// Type uncertainty: how unsure the model is about which entity type applies.
    pub type_score: f64,
}

/// Score a text's annotation priority.
///
/// Returns a value in `[0.0, 1.0]` where higher means more uncertain and therefore
/// more valuable to annotate. The score is the maximum per-entity uncertainty
/// (`1 - confidence`) across all entities in the text.
///
/// Returns `0.0` for texts with no entities (model is certain: nothing here).
pub fn annotation_priority(entities: &[Entity]) -> f64 {
    entities
        .iter()
        .map(|e| 1.0 - e.confidence.value())
        .fold(0.0_f64, f64::max)
}

/// Rank a batch of texts by annotation value, highest uncertainty first.
///
/// Returns `(original_index, priority_score)` pairs sorted descending by score.
/// Texts with equal priority preserve their original relative order (stable sort).
pub fn rank_for_annotation<S: AsRef<str>>(texts: &[(S, Vec<Entity>)]) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = texts
        .iter()
        .enumerate()
        .map(|(i, (_, entities))| (i, annotation_priority(entities)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// Per-span uncertainty breakdown for each entity in a slice.
///
/// When [`HierarchicalConfidence`] is present, `boundary` and `type_score` reflect
/// the model's uncertainty at those specific levels. Otherwise both fall back to `0.5`
/// (maximum uncertainty under no information).
pub fn span_uncertainties(entities: &[Entity]) -> Vec<(&Entity, SpanUncertainty)> {
    entities
        .iter()
        .map(|e| {
            let (boundary, type_score) = decompose_uncertainty(e.hierarchical_confidence.as_ref());
            (
                e,
                SpanUncertainty {
                    overall: 1.0 - e.confidence.value(),
                    boundary,
                    type_score,
                },
            )
        })
        .collect()
}

/// Extract boundary and type uncertainty from optional [`HierarchicalConfidence`].
fn decompose_uncertainty(hc: Option<&HierarchicalConfidence>) -> (f64, f64) {
    match hc {
        Some(h) => (1.0 - h.boundary.value(), 1.0 - h.type_score.value()),
        None => (0.5, 0.5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, EntityType, HierarchicalConfidence};

    #[test]
    fn annotation_priority_empty() {
        assert_eq!(annotation_priority(&[]), 0.0);
    }

    #[test]
    fn annotation_priority_single_entity() {
        let e = Entity::new("Alice", EntityType::Person, 0, 5, 0.8);
        let priority = annotation_priority(&[e]);
        assert!((priority - 0.2).abs() < 1e-10, "priority = {priority}");
    }

    #[test]
    fn annotation_priority_returns_max_uncertainty() {
        // Two entities: 0.9 confidence (uncertainty 0.1) and 0.6 confidence (uncertainty 0.4)
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("IBM", EntityType::Organization, 10, 13, 0.6),
        ];
        let priority = annotation_priority(&entities);
        assert!((priority - 0.4).abs() < 1e-10, "priority = {priority}");
    }

    #[test]
    fn annotation_priority_all_certain() {
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 1.0),
            Entity::new("Bob", EntityType::Person, 10, 13, 1.0),
        ];
        assert_eq!(annotation_priority(&entities), 0.0);
    }

    #[test]
    fn rank_for_annotation_orders_highest_uncertainty_first() {
        let batch = vec![
            (
                "high confidence text",
                vec![Entity::new("A", EntityType::Person, 0, 1, 0.95)],
            ),
            (
                "low confidence text",
                vec![Entity::new("B", EntityType::Person, 0, 1, 0.4)],
            ),
            (
                "medium confidence text",
                vec![Entity::new("C", EntityType::Person, 0, 1, 0.7)],
            ),
        ];
        let ranked = rank_for_annotation(&batch);
        // highest uncertainty (lowest confidence) should be first
        assert_eq!(ranked[0].0, 1, "low-confidence text should be first");
        assert_eq!(ranked[1].0, 2, "medium-confidence text should be second");
        assert_eq!(ranked[2].0, 0, "high-confidence text should be last");
    }

    #[test]
    fn rank_for_annotation_empty_batch() {
        let batch: Vec<(&str, Vec<Entity>)> = Vec::new();
        let ranked = rank_for_annotation(&batch);
        assert!(ranked.is_empty());
    }

    #[test]
    fn span_uncertainties_without_hierarchical() {
        let entities = vec![Entity::new("Alice", EntityType::Person, 0, 5, 0.7)];
        let result = span_uncertainties(&entities);
        assert_eq!(result.len(), 1);
        let (_, u) = &result[0];
        assert!((u.overall - 0.3).abs() < 1e-10, "overall = {}", u.overall);
        // No hierarchical confidence: boundary and type_score fall back to 0.5
        assert!((u.boundary - 0.5).abs() < 1e-10);
        assert!((u.type_score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn span_uncertainties_with_hierarchical() {
        let hc = HierarchicalConfidence::new(
            Confidence::new(0.9),
            Confidence::new(0.8),
            Confidence::new(0.6),
        );
        let mut e = Entity::new("Alice", EntityType::Person, 0, 5, 0.8);
        e.set_hierarchical_confidence(hc);
        let entities = vec![e];

        let result = span_uncertainties(&entities);
        let (_, u) = &result[0];
        // set_hierarchical_confidence updates e.confidence to the geometric mean of (0.9, 0.8, 0.6)
        let expected_combined = (0.9_f64 * 0.8 * 0.6).powf(1.0 / 3.0);
        let expected_overall = 1.0 - expected_combined;
        assert!(
            (u.overall - expected_overall).abs() < 1e-9,
            "overall = {}",
            u.overall
        );
        // boundary = 1 - 0.6 = 0.4
        assert!((u.boundary - 0.4).abs() < 1e-9, "boundary = {}", u.boundary);
        // type_score = 1 - 0.8 = 0.2
        assert!(
            (u.type_score - 0.2).abs() < 1e-9,
            "type_score = {}",
            u.type_score
        );
    }
}
