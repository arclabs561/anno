//! Hybrid NER - Combines pattern extraction with ML backends.
//!
//! # Design Rationale
//!
//! Rather than a complex ensemble with voting, this uses a simple **layered strategy**:
//!
//! 1. **PatternNER first** (fast, ~400ns): Extracts structured entities with 100% precision
//!    - DATE, TIME, MONEY, PERCENT, EMAIL, URL, PHONE
//!
//! 2. **ML backend second** (if available): Extracts semantic entities
//!    - PER, ORG, LOC (things patterns can't reliably detect)
//!
//! 3. **Merge**: Union with overlap resolution (pattern wins for structured types)
//!
//! This avoids:
//! - Voting complexity (no need to reconcile conflicting predictions)
//! - Latency multiplication (pattern is ~1000x faster than ML)
//! - Error propagation (each backend has clear responsibility)
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::{HybridNER, PatternNER};
//!
//! // Pattern-only (always available)
//! let ner = HybridNER::pattern_only();
//!
//! // With ML backend (when feature enabled)
//! #[cfg(feature = "onnx")]
//! let ner = HybridNER::with_ml(Box::new(BertNEROnnx::new()?));
//! ```

use crate::{Entity, EntityType, Model, PatternNER, Result};

/// Merge strategy for combining entities from multiple backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    /// Union: include all entities, pattern wins on overlap
    #[default]
    Union,
    /// Pattern priority: pattern entities always win overlaps
    PatternFirst,
    /// ML priority: ML entities always win overlaps
    MLFirst,
    /// Intersection: only entities both agree on (very conservative)
    Intersection,
}

/// Configuration for hybrid NER.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// How to merge entities from different backends
    pub merge_strategy: MergeStrategy,
    /// Minimum confidence threshold for ML entities (0.0-1.0)
    pub ml_confidence_threshold: f64,
    /// Skip ML backend for texts shorter than this (optimization)
    pub ml_min_text_length: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            merge_strategy: MergeStrategy::Union,
            ml_confidence_threshold: 0.5,
            ml_min_text_length: 10,
        }
    }
}

/// Hybrid NER combining PatternNER with optional ML backend.
///
/// # Architecture
///
/// ```text
/// Input Text
///     │
///     ├──► PatternNER ──► Structured entities (DATE, MONEY, EMAIL, etc.)
///     │         │
///     │         ▼
///     │    [Fast path: ~400ns]
///     │
///     └──► ML Backend ──► Semantic entities (PER, ORG, LOC)
///               │
///               ▼
///          [Slow path: ~50ms]
///               │
///               ▼
///         Merge Strategy
///               │
///               ▼
///         Final Entities
/// ```
pub struct HybridNER {
    pattern: PatternNER,
    ml_backend: Option<Box<dyn Model>>,
    config: HybridConfig,
}

impl HybridNER {
    /// Create hybrid NER with pattern extraction only.
    ///
    /// Use this when ML backends are unavailable or for latency-sensitive applications.
    #[must_use]
    pub fn pattern_only() -> Self {
        Self {
            pattern: PatternNER::new(),
            ml_backend: None,
            config: HybridConfig::default(),
        }
    }

    /// Create hybrid NER with an ML backend.
    ///
    /// The ML backend handles PER/ORG/LOC; patterns handle structured types.
    pub fn with_ml(ml_backend: Box<dyn Model>) -> Self {
        Self {
            pattern: PatternNER::new(),
            ml_backend: Some(ml_backend),
            config: HybridConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(ml_backend: Option<Box<dyn Model>>, config: HybridConfig) -> Self {
        Self {
            pattern: PatternNER::new(),
            ml_backend,
            config,
        }
    }

    /// Check if ML backend is available.
    #[must_use]
    pub fn has_ml_backend(&self) -> bool {
        self.ml_backend
            .as_ref()
            .is_some_and(|backend| backend.is_available())
    }

    /// Merge entities from pattern and ML backends.
    fn merge_entities(&self, pattern_entities: Vec<Entity>, ml_entities: Vec<Entity>) -> Vec<Entity> {
        match self.config.merge_strategy {
            MergeStrategy::Union | MergeStrategy::PatternFirst => {
                self.merge_union_pattern_priority(pattern_entities, ml_entities)
            }
            MergeStrategy::MLFirst => {
                self.merge_union_ml_priority(pattern_entities, ml_entities)
            }
            MergeStrategy::Intersection => {
                self.merge_intersection(pattern_entities, ml_entities)
            }
        }
    }

    /// Union merge with pattern entities winning overlaps.
    fn merge_union_pattern_priority(
        &self,
        pattern_entities: Vec<Entity>,
        ml_entities: Vec<Entity>,
    ) -> Vec<Entity> {
        let mut result = pattern_entities;

        // Add ML entities that don't overlap with pattern entities
        for ml_entity in ml_entities {
            if ml_entity.confidence < self.config.ml_confidence_threshold {
                continue;
            }

            let overlaps = result
                .iter()
                .any(|e| spans_overlap(e.start, e.end, ml_entity.start, ml_entity.end));

            if !overlaps {
                result.push(ml_entity);
            }
        }

        result.sort_by_key(|e| e.start);
        result
    }

    /// Union merge with ML entities winning overlaps.
    fn merge_union_ml_priority(
        &self,
        pattern_entities: Vec<Entity>,
        ml_entities: Vec<Entity>,
    ) -> Vec<Entity> {
        let mut result: Vec<Entity> = ml_entities
            .into_iter()
            .filter(|e| e.confidence >= self.config.ml_confidence_threshold)
            .collect();

        // Add pattern entities that don't overlap with ML entities
        for pattern_entity in pattern_entities {
            let overlaps = result
                .iter()
                .any(|e| spans_overlap(e.start, e.end, pattern_entity.start, pattern_entity.end));

            if !overlaps {
                result.push(pattern_entity);
            }
        }

        result.sort_by_key(|e| e.start);
        result
    }

    /// Intersection: only entities both backends found at same position.
    fn merge_intersection(
        &self,
        pattern_entities: Vec<Entity>,
        ml_entities: Vec<Entity>,
    ) -> Vec<Entity> {
        let mut result = Vec::new();

        for pattern_entity in &pattern_entities {
            for ml_entity in &ml_entities {
                // Check for significant overlap (IoU > 0.5)
                if spans_overlap(
                    pattern_entity.start,
                    pattern_entity.end,
                    ml_entity.start,
                    ml_entity.end,
                ) {
                    let iou = span_iou(
                        pattern_entity.start,
                        pattern_entity.end,
                        ml_entity.start,
                        ml_entity.end,
                    );
                    if iou > 0.5 {
                        // Use pattern entity (higher precision) but boost confidence
                        let mut merged = pattern_entity.clone();
                        merged.confidence = (pattern_entity.confidence + ml_entity.confidence) / 2.0;
                        result.push(merged);
                        break;
                    }
                }
            }
        }

        result.sort_by_key(|e| e.start);
        result
    }
}

impl Model for HybridNER {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        // Always run pattern extraction (fast)
        let pattern_entities = self.pattern.extract_entities(text, language)?;

        // Skip ML for short texts or if no backend
        if text.len() < self.config.ml_min_text_length || self.ml_backend.is_none() {
            return Ok(pattern_entities);
        }

        // Run ML backend if available
        let ml_entities = match &self.ml_backend {
            Some(backend) if backend.is_available() => {
                backend.extract_entities(text, language)?
            }
            _ => Vec::new(),
        };

        // Merge results
        Ok(self.merge_entities(pattern_entities, ml_entities))
    }

    fn supported_types(&self) -> Vec<EntityType> {
        let mut types = self.pattern.supported_types();

        if let Some(backend) = &self.ml_backend {
            for entity_type in backend.supported_types() {
                if !types.contains(&entity_type) {
                    types.push(entity_type);
                }
            }
        }

        types
    }

    fn is_available(&self) -> bool {
        true // Pattern extraction is always available
    }

    fn name(&self) -> &'static str {
        "hybrid"
    }

    fn description(&self) -> &'static str {
        "Hybrid NER (pattern + ML backend)"
    }
}

/// Check if two spans overlap.
#[inline]
fn spans_overlap(start1: usize, end1: usize, start2: usize, end2: usize) -> bool {
    start1 < end2 && start2 < end1
}

/// Calculate Intersection over Union for two spans.
#[inline]
fn span_iou(start1: usize, end1: usize, start2: usize, end2: usize) -> f64 {
    let intersection_start = start1.max(start2);
    let intersection_end = end1.min(end2);

    if intersection_start >= intersection_end {
        return 0.0;
    }

    let intersection = (intersection_end - intersection_start) as f64;
    let union = ((end1 - start1) + (end2 - start2)) as f64 - intersection;

    if union == 0.0 {
        1.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_only() {
        let ner = HybridNER::pattern_only();
        let entities = ner
            .extract_entities("Meeting on 2024-01-15 at $100", None)
            .unwrap();

        assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Money));
    }

    #[test]
    fn test_spans_overlap() {
        assert!(spans_overlap(0, 10, 5, 15)); // Overlapping
        assert!(!spans_overlap(0, 5, 5, 10)); // Adjacent (not overlapping)
        assert!(!spans_overlap(0, 5, 10, 15)); // Disjoint
        assert!(spans_overlap(0, 10, 0, 10)); // Identical
    }

    #[test]
    fn test_span_iou() {
        // Identical spans
        assert!((span_iou(0, 10, 0, 10) - 1.0).abs() < 0.001);

        // No overlap
        assert!((span_iou(0, 5, 10, 15) - 0.0).abs() < 0.001);

        // 50% overlap
        assert!((span_iou(0, 10, 5, 15) - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_merge_union_pattern_priority() {
        let ner = HybridNER::pattern_only();

        let pattern_entities = vec![Entity::new(
            "$100",
            EntityType::Money,
            0,
            4,
            0.95,
        )];

        let ml_entities = vec![
            Entity::new("$100", EntityType::Money, 0, 4, 0.8), // Overlaps - should be dropped
            Entity::new("John", EntityType::Person, 10, 14, 0.9), // No overlap - should be added
        ];

        let merged = ner.merge_union_pattern_priority(pattern_entities, ml_entities);

        assert_eq!(merged.len(), 2);
        assert!(merged.iter().any(|e| e.entity_type == EntityType::Money));
        assert!(merged.iter().any(|e| e.entity_type == EntityType::Person));
    }

    #[test]
    fn test_supported_types_combined() {
        let ner = HybridNER::pattern_only();
        let types = ner.supported_types();

        assert!(types.contains(&EntityType::Date));
        assert!(types.contains(&EntityType::Money));
        assert!(types.contains(&EntityType::Percent));
    }
}

