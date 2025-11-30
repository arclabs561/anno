//! Stacked NER - Composable extraction with principled conflict resolution.
//!
//! # The Core Idea
//!
//! Different NER backends are good at different things:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    BACKEND SPECIALIZATION                           │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  PatternNER                   HeuristicNER                        │
//! │  ───────────                  ──────────────                        │
//! │  Uses: Regex patterns         Uses: Capitalization, context         │
//! │  Good at: Structured data     Good at: Named entities               │
//! │                                                                     │
//! │    $100.00 ✓ (MONEY)            Dr. Smith ✓ (PERSON)                │
//! │    jan 15, 2024 ✓ (DATE)        Apple Inc. ✓ (ORG)                  │
//! │    test@mail.com ✓ (EMAIL)      New York ✓ (LOC)                    │
//! │                                                                     │
//! │    Dr. Smith ✗ (can't!)         $100.00 ✗ (no pattern!)             │
//! │                                                                     │
//! │  Precision: ~99%              Precision: ~70%                       │
//! │  (When it fires, it's right)  (Makes guesses based on heuristics)   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! `StackedNER` combines them to get the best of both worlds.
//!
//! # How Entities Flow Through Layers
//!
//! ```text
//! Input: "Email ceo@apple.com about Apple stock for $100"
//!
//!         │
//!         ▼
//! ┌───────────────────────────────────────────────────────────────────┐
//! │                     LAYER 1: PatternNER                           │
//! │                                                                   │
//! │   Scans for regex patterns:                                       │
//! │                                                                   │
//! │   "Email ceo@apple.com about Apple stock for $100"                │
//! │          └────EMAIL────┘                      └MONEY              │
//! │          (conf: 0.98)                        (conf: 0.95)         │
//! │                                                                   │
//! │   Output: [EMAIL: ceo@apple.com, MONEY: $100]                     │
//! └───────────────────────────────────────────────────────────────────┘
//!         │
//!         ▼
//! ┌───────────────────────────────────────────────────────────────────┐
//! │                   LAYER 2: HeuristicNER                         │
//! │                                                                   │
//! │   Scans for capitalized sequences + context:                      │
//! │                                                                   │
//! │   "Email ceo@apple.com about Apple stock for $100"                │
//! │                              └─ORG─┘                              │
//! │                             (conf: 0.65)                          │
//! │                                                                   │
//! │   Also found: "apple.com" as ORG (conf: 0.40) ← OVERLAP!          │
//! └───────────────────────────────────────────────────────────────────┘
//!         │
//!         ▼
//! ┌───────────────────────────────────────────────────────────────────┐
//! │                   CONFLICT RESOLUTION                             │
//! │                                                                   │
//! │   Conflict detected:                                              │
//! │     • EMAIL "ceo@apple.com" (0.98) from Layer 1                   │
//! │     • ORG "apple.com" (0.40) from Layer 2                         │
//! │                                                                   │
//! │   ┌─────────────────────────────────────────────────────────────┐ │
//! │   │ Strategy: HighestConf                                       │ │
//! │   │                                                             │ │
//! │   │   EMAIL (0.98) vs ORG (0.40)                                │ │
//! │   │                                                             │ │
//! │   │   Winner: EMAIL ✓                                           │ │
//! │   │   Discard: ORG ✗                                            │ │
//! │   └─────────────────────────────────────────────────────────────┘ │
//! └───────────────────────────────────────────────────────────────────┘
//!         │
//!         ▼
//! ┌───────────────────────────────────────────────────────────────────┐
//! │                      FINAL OUTPUT                                 │
//! │                                                                   │
//! │   [EMAIL: ceo@apple.com, ORG: Apple, MONEY: $100]                 │
//! │                                                                   │
//! │   Sorted by position in text.                                     │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Conflict Resolution Strategies
//!
//! ```text
//! When two entities overlap, how do we choose?
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ PRIORITY (default)                                              │
//! │ ────────────────────                                            │
//! │ First layer wins. Simple and predictable.                       │
//! │                                                                 │
//! │   Layer 1: [====EMAIL====]  ← Wins (came first)                 │
//! │   Layer 2:       [==ORG==]  ← Discarded                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ LONGEST_SPAN                                                    │
//! │ ─────────────                                                   │
//! │ Longer span wins. Prefers "New York City" over "New York".      │
//! │                                                                 │
//! │   Layer 1: [====EMAIL====]  ← Wins (14 chars)                   │
//! │   Layer 2:       [==ORG==]  ← Discarded (9 chars)               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ HIGHEST_CONF                                                    │
//! │ ─────────────                                                   │
//! │ Highest confidence wins. Trust the more certain prediction.     │
//! │                                                                 │
//! │   Layer 1: EMAIL (0.98)  ← Wins (higher confidence)             │
//! │   Layer 2: ORG (0.40)    ← Discarded                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ UNION                                                           │
//! │ ──────                                                          │
//! │ Keep both! Let downstream decide.                               │
//! │                                                                 │
//! │   Layer 1: [====EMAIL====]  ← Keep                              │
//! │   Layer 2:       [==ORG==]  ← Also keep                         │
//! │                                                                 │
//! │   Use when: Building a knowledge graph, need all hypotheses.    │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Examples
//!
//! Zero-config default (Pattern + Statistical):
//!
//! ```rust
//! use anno::{Model, StackedNER};
//!
//! let ner = StackedNER::default();
//! let entities = ner.extract_entities(
//!     "Dr. Smith charges $100/hr. Email: smith@test.com",
//!     None
//! ).unwrap();
//! ```
//!
//! Custom composition:
//!
//! ```rust
//! use anno::{Model, PatternNER, HeuristicNER, StackedNER};
//! use anno::backends::stacked::ConflictStrategy;
//!
//! let ner = StackedNER::builder()
//!     .layer(PatternNER::new())
//!     .layer(HeuristicNER::new())
//!     .strategy(ConflictStrategy::LongestSpan)
//!     .build();
//! ```
//!
//! Pattern-only (no heuristic):
//!
//! ```rust
//! use anno::{Model, StackedNER};
//!
//! let ner = StackedNER::pattern_only();
//! let entities = ner.extract_entities("Cost: $100", None).unwrap();
//! ```

use super::heuristic::HeuristicNER;
use super::pattern::PatternNER;
use crate::{Entity, EntityType, Model, Result};
use std::sync::Arc;

// =============================================================================
// Conflict Resolution
// =============================================================================

/// Strategy for resolving overlapping entity spans.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// First layer to claim a span wins. Simple and predictable.
    #[default]
    Priority,

    /// Longest span wins. Prefers "New York City" over "New York".
    LongestSpan,

    /// Highest confidence score wins.
    HighestConf,

    /// Keep all entities, even if they overlap.
    /// Useful when downstream processing handles disambiguation.
    Union,
}

impl ConflictStrategy {
    /// Resolve a conflict between two overlapping entities.
    fn resolve(&self, existing: &Entity, candidate: &Entity) -> Resolution {
        match self {
            ConflictStrategy::Priority => Resolution::KeepExisting,

            ConflictStrategy::LongestSpan => {
                let existing_len = existing.end - existing.start;
                let candidate_len = candidate.end - candidate.start;
                if candidate_len > existing_len {
                    Resolution::Replace
                } else {
                    Resolution::KeepExisting
                }
            }

            ConflictStrategy::HighestConf => {
                if candidate.confidence > existing.confidence {
                    Resolution::Replace
                } else {
                    Resolution::KeepExisting
                }
            }

            ConflictStrategy::Union => Resolution::KeepBoth,
        }
    }
}

#[derive(Debug)]
enum Resolution {
    KeepExisting,
    Replace,
    KeepBoth,
}

// =============================================================================
// StackedNER
// =============================================================================

/// Composable NER that combines multiple backends.
///
/// # Design
///
/// Different backends excel at different tasks:
///
/// | Backend | Best For | Trade-off |
/// |---------|----------|-----------|
/// | Pattern | Structured entities | Can't do named entities |
/// | Statistical | Named entities (no deps) | Lower accuracy |
/// | ML | Everything | Heavy dependencies |
///
/// `StackedNER` runs backends in order, merging results according to the
/// configured [`ConflictStrategy`].
///
/// # Default Configuration
///
/// `StackedNER::default()` creates a Pattern + Statistical configuration:
/// - Layer 1: `PatternNER` (dates, money, emails, etc.)
/// - Layer 2: `HeuristicNER` (person, org, location)
///
/// This provides solid NER coverage with zero ML dependencies.
///
/// # Example
///
/// ```rust
/// use anno::{Model, StackedNER};
///
/// // Default: Pattern + Statistical
/// let ner = StackedNER::default();
///
/// // Or build custom:
/// use anno::{PatternNER, HeuristicNER};
/// use anno::backends::stacked::ConflictStrategy;
///
/// let custom = StackedNER::builder()
///     .layer(PatternNER::new())
///     .layer(HeuristicNER::new())
///     .strategy(ConflictStrategy::LongestSpan)
///     .build();
/// ```
pub struct StackedNER {
    layers: Vec<Arc<dyn Model + Send + Sync>>,
    strategy: ConflictStrategy,
    name: String,
}

/// Builder for [`StackedNER`] with fluent configuration.
#[derive(Default)]
pub struct StackedNERBuilder {
    layers: Vec<Box<dyn Model + Send + Sync>>,
    strategy: ConflictStrategy,
}

impl StackedNERBuilder {
    /// Add a layer (order matters: earlier = higher priority).
    #[must_use]
    pub fn layer<M: Model + Send + Sync + 'static>(mut self, model: M) -> Self {
        self.layers.push(Box::new(model));
        self
    }

    /// Add a boxed layer.
    #[must_use]
    pub fn layer_boxed(mut self, model: Box<dyn Model + Send + Sync>) -> Self {
        self.layers.push(model);
        self
    }

    /// Set the conflict resolution strategy.
    #[must_use]
    pub fn strategy(mut self, strategy: ConflictStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Build the configured StackedNER.
    #[must_use]
    pub fn build(self) -> StackedNER {
        let name = if self.layers.is_empty() {
            "stacked(empty)".to_string()
        } else {
            format!(
                "stacked({})",
                self.layers
                    .iter()
                    .map(|l| l.name())
                    .collect::<Vec<_>>()
                    .join("+")
            )
        };

        StackedNER {
            layers: self.layers.into_iter().map(Arc::from).collect(),
            strategy: self.strategy,
            name,
        }
    }
}

impl StackedNER {
    /// Create default configuration: Pattern + Statistical layers.
    ///
    /// This provides zero-dependency NER with:
    /// - High-precision structured entity extraction (dates, money, etc.)
    /// - Heuristic named entity extraction (person, org, location)
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for custom configuration.
    #[must_use]
    pub fn builder() -> StackedNERBuilder {
        StackedNERBuilder::default()
    }

    /// Create with explicit layers and default priority strategy.
    #[must_use]
    pub fn with_layers(layers: Vec<Box<dyn Model + Send + Sync>>) -> Self {
        let mut builder = Self::builder().strategy(ConflictStrategy::Priority);
        for layer in layers {
            builder = builder.layer_boxed(layer);
        }
        builder.build()
    }

    /// Create with custom heuristic threshold.
    ///
    /// Higher threshold = fewer but higher confidence heuristic entities.
    /// Note: HeuristicNER does not currently support dynamic thresholding
    /// in constructor, so this method ignores the parameter for now but maintains API compat.
    #[must_use]
    pub fn with_heuristic_threshold(_threshold: f64) -> Self {
        Self::builder()
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .build()
    }

    /// Backwards compatibility alias.
    #[deprecated(since = "0.3.0", note = "Use with_heuristic_threshold instead")]
    #[must_use]
    pub fn with_statistical_threshold(threshold: f64) -> Self {
        Self::with_heuristic_threshold(threshold)
    }

    /// Pattern-only configuration (no heuristic layer).
    ///
    /// Extracts only structured entities: dates, times, money, percentages,
    /// emails, URLs, phone numbers.
    #[must_use]
    pub fn pattern_only() -> Self {
        Self::builder().layer(PatternNER::new()).build()
    }

    /// Heuristic-only configuration (no pattern layer).
    ///
    /// Extracts only named entities: person, organization, location.
    #[must_use]
    pub fn heuristic_only() -> Self {
        Self::builder().layer(HeuristicNER::new()).build()
    }

    /// Backwards compatibility alias.
    #[deprecated(since = "0.3.0", note = "Use heuristic_only instead")]
    #[must_use]
    pub fn statistical_only() -> Self {
        Self::heuristic_only()
    }

    /// Add an ML backend as highest priority.
    ///
    /// ML runs first, then Pattern fills structured gaps, then Heuristic.
    #[must_use]
    pub fn with_ml_first(ml_backend: Box<dyn Model + Send + Sync>) -> Self {
        Self::builder()
            .layer_boxed(ml_backend)
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .build()
    }

    /// Add an ML backend as fallback (lowest priority).
    ///
    /// Pattern runs first (high precision), then Heuristic, then ML.
    #[must_use]
    pub fn with_ml_fallback(ml_backend: Box<dyn Model + Send + Sync>) -> Self {
        Self::builder()
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .layer_boxed(ml_backend)
            .build()
    }

    /// Get the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer names in priority order.
    #[must_use]
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|l| l.name()).collect()
    }

    /// Get the conflict strategy.
    #[must_use]
    pub fn strategy(&self) -> ConflictStrategy {
        self.strategy
    }
}

impl Default for StackedNER {
    /// Default configuration: Pattern + Statistical layers.
    ///
    /// This provides zero-dependency NER with:
    /// - High-precision structured entity extraction (dates, money, etc.)
    /// - Heuristic named entity extraction (person, org, location)
    fn default() -> Self {
        Self::builder()
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .build()
    }
}

impl Model for StackedNER {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        let mut entities: Vec<Entity> = Vec::new();

        for layer in &self.layers {
            let layer_entities = layer.extract_entities(text, language)?;

            for candidate in layer_entities {
                // Find any overlapping existing entity
                let overlap_idx = entities
                    .iter()
                    .position(|e| !(candidate.end <= e.start || candidate.start >= e.end));

                match overlap_idx {
                    None => {
                        // No overlap - add directly
                        entities.push(candidate);
                    }
                    Some(idx) => {
                        // Resolve conflict
                        match self.strategy.resolve(&entities[idx], &candidate) {
                            Resolution::KeepExisting => {}
                            Resolution::Replace => {
                                entities[idx] = candidate;
                            }
                            Resolution::KeepBoth => {
                                entities.push(candidate);
                            }
                        }
                    }
                }
            }
        }

        // Sort by position
        entities.sort_by_key(|e| (e.start, e.end));

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        let mut types = Vec::new();
        for layer in &self.layers {
            types.extend(layer.supported_types());
        }
        types.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        types.dedup();
        types
    }

    fn is_available(&self) -> bool {
        self.layers.iter().any(|l| l.is_available())
    }

    fn name(&self) -> &'static str {
        // Leak for static lifetime - StackedNER instances are typically long-lived
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn description(&self) -> &'static str {
        "Stacked NER (multi-backend composition)"
    }
}

// =============================================================================
// Type Aliases for Backwards Compatibility
// =============================================================================

/// Alias for backwards compatibility.
#[deprecated(since = "0.2.0", note = "Use StackedNER instead")]
pub type LayeredNER = StackedNER;

/// Alias for backwards compatibility.
#[deprecated(since = "0.2.0", note = "Use StackedNER::default() instead")]
pub type TieredNER = StackedNER;

/// Alias for backwards compatibility.
#[deprecated(since = "0.2.0", note = "Use StackedNER instead")]
pub type CompositeNER = StackedNER;

// Capability markers: StackedNER combines pattern and heuristic extraction
impl crate::StructuredEntityCapable for StackedNER {}
impl crate::NamedEntityCapable for StackedNER {}

// =============================================================================
// BatchCapable and StreamingCapable Trait Implementations
// =============================================================================

impl crate::BatchCapable for StackedNER {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        texts
            .iter()
            .map(|text| self.extract_entities(text, language))
            .collect()
    }

    fn optimal_batch_size(&self) -> Option<usize> {
        Some(32) // Combination of pattern + heuristic
    }
}

impl crate::StreamingCapable for StackedNER {
    fn recommended_chunk_size(&self) -> usize {
        8_000 // Slightly smaller due to multi-layer processing
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(text: &str) -> Vec<Entity> {
        StackedNER::default().extract_entities(text, None).unwrap()
    }

    fn has_type(entities: &[Entity], ty: &EntityType) -> bool {
        entities.iter().any(|e| e.entity_type == *ty)
    }

    // =========================================================================
    // Default Configuration Tests
    // =========================================================================

    #[test]
    fn test_default_finds_patterns() {
        let e = extract("Cost: $100");
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn test_default_finds_heuristic() {
        let e = extract("Mr. Smith said hello");
        assert!(has_type(&e, &EntityType::Person));
    }

    #[test]
    fn test_default_finds_both() {
        let e = extract("Dr. Smith charges $200/hr");
        assert!(has_type(&e, &EntityType::Money));
        // May also find Person
    }

    #[test]
    fn test_no_overlaps() {
        let e = extract("Price is $100 from John at Google Inc.");
        for i in 0..e.len() {
            for j in (i + 1)..e.len() {
                let overlap = e[i].start < e[j].end && e[j].start < e[i].end;
                assert!(!overlap, "Overlap: {:?} and {:?}", e[i], e[j]);
            }
        }
    }

    #[test]
    fn test_sorted_output() {
        let e = extract("$100 for John in Paris on 2024-01-15");
        for i in 1..e.len() {
            assert!(e[i - 1].start <= e[i].start);
        }
    }

    // =========================================================================
    // Builder Tests
    // =========================================================================

    #[test]
    fn test_builder_empty() {
        let ner = StackedNER::builder().build();
        let e = ner.extract_entities("$100 for John", None).unwrap();
        assert!(e.is_empty()); // No layers = no entities
    }

    #[test]
    fn test_builder_single_layer() {
        let ner = StackedNER::builder().layer(PatternNER::new()).build();
        let e = ner.extract_entities("$100", None).unwrap();
        assert!(has_type(&e, &EntityType::Money));
    }

    #[test]
    fn test_builder_layer_names() {
        let ner = StackedNER::builder()
            .layer(PatternNER::new())
            .layer(HeuristicNER::new())
            .build();

        let names = ner.layer_names();
        assert!(names.contains(&"pattern"));
        assert!(names.contains(&"heuristic"));
    }

    #[test]
    fn test_builder_strategy() {
        let ner = StackedNER::builder()
            .layer(PatternNER::new())
            .strategy(ConflictStrategy::LongestSpan)
            .build();

        assert_eq!(ner.strategy(), ConflictStrategy::LongestSpan);
    }

    // =========================================================================
    // Convenience Constructor Tests
    // =========================================================================

    #[test]
    fn test_pattern_only() {
        let ner = StackedNER::pattern_only();
        let e = ner.extract_entities("$100 for Dr. Smith", None).unwrap();

        // Should find money
        assert!(has_type(&e, &EntityType::Money));
        // Should NOT find person (no heuristic layer)
        assert!(!has_type(&e, &EntityType::Person));
    }

    #[test]
    fn test_heuristic_only() {
        let ner = StackedNER::heuristic_only();
        // Use a name that HeuristicNER can detect (capitalized single word)
        let e = ner.extract_entities("$100 for John", None).unwrap();

        // HeuristicNER uses heuristics - may or may not find person
        // The key test is that it does NOT find money (no pattern layer)
        assert!(
            !has_type(&e, &EntityType::Money),
            "Should NOT find money without pattern layer: {:?}",
            e
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_statistical_only_deprecated_alias() {
        // Verify backwards compatibility
        let ner = StackedNER::statistical_only();
        let e = ner.extract_entities("John", None).unwrap();
        // Just verify it doesn't panic
        let _ = e;
    }

    // =========================================================================
    // Conflict Strategy Tests
    // =========================================================================

    #[test]
    fn test_strategy_default_is_priority() {
        let ner = StackedNER::default();
        assert_eq!(ner.strategy(), ConflictStrategy::Priority);
    }

    // =========================================================================
    // Mock Backend Tests for Conflict Resolution
    // =========================================================================

    use crate::MockModel;

    fn mock_model(name: &'static str, entities: Vec<Entity>) -> MockModel {
        MockModel::new(name).with_entities(entities)
    }

    fn mock_entity(text: &str, start: usize, ty: EntityType, conf: f64) -> Entity {
        Entity {
            text: text.to_string(),
            entity_type: ty,
            start,
            end: start + text.len(),
            confidence: conf,
            provenance: None,
            kb_id: None,
            canonical_id: None,
            normalized: None,
            hierarchical_confidence: None,
            visual_span: None,
            discontinuous_span: None,
            valid_from: None,
            valid_until: None,
            viewport: None,
        }
    }

    #[test]
    fn test_priority_first_wins() {
        let layer1 = mock_model(
            "l1",
            vec![mock_entity("New York", 0, EntityType::Location, 0.8)],
        );
        let layer2 = mock_model(
            "l2",
            vec![mock_entity("New York City", 0, EntityType::Location, 0.9)],
        );

        let ner = StackedNER::builder()
            .layer(layer1)
            .layer(layer2)
            .strategy(ConflictStrategy::Priority)
            .build();

        let e = ner.extract_entities("New York City", None).unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].text, "New York"); // First layer wins
    }

    #[test]
    fn test_longest_span_wins() {
        let layer1 = mock_model(
            "l1",
            vec![mock_entity("New York", 0, EntityType::Location, 0.8)],
        );
        let layer2 = mock_model(
            "l2",
            vec![mock_entity("New York City", 0, EntityType::Location, 0.7)],
        );

        let ner = StackedNER::builder()
            .layer(layer1)
            .layer(layer2)
            .strategy(ConflictStrategy::LongestSpan)
            .build();

        let e = ner.extract_entities("New York City", None).unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].text, "New York City"); // Longer wins
    }

    #[test]
    fn test_highest_conf_wins() {
        let layer1 = mock_model(
            "l1",
            vec![mock_entity("Apple", 0, EntityType::Organization, 0.6)],
        );
        let layer2 = mock_model(
            "l2",
            vec![mock_entity("Apple", 0, EntityType::Organization, 0.95)],
        );

        let ner = StackedNER::builder()
            .layer(layer1)
            .layer(layer2)
            .strategy(ConflictStrategy::HighestConf)
            .build();

        let e = ner.extract_entities("Apple Inc", None).unwrap();
        assert_eq!(e.len(), 1);
        assert!(e[0].confidence > 0.9);
    }

    #[test]
    fn test_union_keeps_all() {
        let layer1 = mock_model("l1", vec![mock_entity("John", 0, EntityType::Person, 0.8)]);
        let layer2 = mock_model("l2", vec![mock_entity("John", 0, EntityType::Person, 0.9)]);

        let ner = StackedNER::builder()
            .layer(layer1)
            .layer(layer2)
            .strategy(ConflictStrategy::Union)
            .build();

        let e = ner.extract_entities("John is here", None).unwrap();
        assert_eq!(e.len(), 2); // Both kept
    }

    #[test]
    fn test_non_overlapping_always_kept() {
        for strategy in [
            ConflictStrategy::Priority,
            ConflictStrategy::LongestSpan,
            ConflictStrategy::HighestConf,
        ] {
            let ner = StackedNER::builder()
                .layer(mock_model(
                    "l1",
                    vec![mock_entity("John", 0, EntityType::Person, 0.8)],
                ))
                .layer(mock_model(
                    "l2",
                    vec![mock_entity("Paris", 8, EntityType::Location, 0.9)],
                ))
                .strategy(strategy)
                .build();

            let e = ner.extract_entities("John in Paris", None).unwrap();
            assert_eq!(e.len(), 2, "Strategy {:?} should keep both", strategy);
        }
    }

    // =========================================================================
    // Complex Document Tests
    // =========================================================================

    #[test]
    fn test_press_release() {
        let text = r#"
            PRESS RELEASE - January 15, 2024
            
            Mr. John Smith, CEO of Acme Corporation, announced today that the company
            will invest $50 million in their San Francisco headquarters.
            
            Contact: press@acme.com or call (555) 123-4567
            
            The expansion is expected to increase revenue by 25%.
        "#;

        let e = extract(text);

        // Pattern entities
        assert!(has_type(&e, &EntityType::Date));
        assert!(has_type(&e, &EntityType::Money));
        assert!(has_type(&e, &EntityType::Email));
        assert!(has_type(&e, &EntityType::Phone));
        assert!(has_type(&e, &EntityType::Percent));
    }

    #[test]
    fn test_empty_text() {
        let e = extract("");
        assert!(e.is_empty());
    }

    #[test]
    fn test_no_entities() {
        let e = extract("the quick brown fox jumps over the lazy dog");
        assert!(e.is_empty());
    }

    #[test]
    fn test_supported_types() {
        let ner = StackedNER::default();
        let types = ner.supported_types();

        // Should include both pattern and heuristic types
        assert!(types.contains(&EntityType::Date));
        assert!(types.contains(&EntityType::Money));
        assert!(types.contains(&EntityType::Person));
        assert!(types.contains(&EntityType::Organization));
        assert!(types.contains(&EntityType::Location));
    }
}
