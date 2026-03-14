//! GLiNER2: Multi-task Information Extraction.
//!
//! GLiNER2 extends GLiNER to support:
//! - Named Entity Recognition (with label descriptions)
//! - Text Classification (single/multi-label)
//! - Hierarchical Structure Extraction
//! - Task Composition (multiple tasks in one pass)
//!
//! This backend is based on the GLiNER2 paper (arXiv:2507.18546). The details of
//! prompt formatting and the full task schema are paper-defined; this module
//! focuses on the inference integration and trait wiring used by `anno`.
//!
//! # Trait Integration
//!
//! GLiNER2 implements the standard `anno` traits:
//! - `Model` - Core entity extraction interface
//! - `ZeroShotNER` - Open-domain entity types
//! - `RelationExtractor` - Joint entity-relation extraction (via GLiREL)
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::{Model, ZeroShotNER, DEFAULT_GLINER2_MODEL};
//! use anno::backends::gliner2::{GLiNER2, TaskSchema};
//!
//! // Use the official Fastino Labs GLiNER2 model
//! let model = GLiNER2::from_pretrained(DEFAULT_GLINER2_MODEL)?;
//! // Or: GLiNER2::from_pretrained("fastino/gliner2-base-v1")?;
//!
//! // Standard Model trait
//! let entities = model.extract_entities("Apple announced iPhone 15", None)?;
//!
//! // Zero-shot with custom types
//! let types = &["company", "product", "event"];
//! let entities = model.extract_with_types(text, types, 0.5)?;
//!
//! // Multi-task extraction with schema
//! let schema = TaskSchema::new()
//!     .with_entities(&["person", "organization", "product"])
//!     .with_classification("sentiment", &["positive", "negative", "neutral"]);
//!
//! let result = model.extract_with_schema("Apple announced iPhone 15", &schema)?;
//! ```
//!
//! # Backends
//!
//! - **ONNX** (recommended): `cargo build --features onnx`
//! - **Candle** (native): `cargo build --features candle`

#[cfg(not(any(feature = "onnx", feature = "candle")))]
use crate::Error;
use crate::{Entity, EntityType, Language, Result};
use anno_core::EntityCategory;
pub(crate) mod relations;

use crate::backends::inference::{ExtractionWithRelations, RelationExtractor, ZeroShotNER};

/// Extract relations, using GLiREL (ONNX) when available, falling back to heuristics.
///
/// GLiREL is loaded lazily on first call and cached via `OnceLock`.
#[cfg(feature = "onnx")]
fn extract_relations_neural_or_heuristic(
    entities: &[Entity],
    text: &str,
    relation_types: &[&str],
    threshold: f32,
) -> Vec<crate::backends::inference::RelationTriple> {
    use std::sync::OnceLock;

    // Try to load GLiREL once. If it fails, fall back to heuristics permanently.
    static GLIREL: OnceLock<Option<crate::backends::glirel::GLiREL>> = OnceLock::new();

    let glirel = GLIREL.get_or_init(|| {
        let default_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
            .join("anno")
            .join("models")
            .join("glirel");

        match crate::backends::glirel::GLiREL::from_local(&default_dir) {
            Ok(model) => {
                log::info!("[GLiNER2] GLiREL model loaded for relation extraction");
                Some(model)
            }
            Err(e) => {
                log::debug!(
                    "[GLiNER2] GLiREL not available ({}), using heuristic relations",
                    e
                );
                None
            }
        }
    });

    if let Some(model) = glirel {
        match model.extract_relations(text, entities, relation_types, threshold) {
            Ok(rels) => return rels,
            Err(e) => {
                log::warn!(
                    "[GLiNER2] GLiREL inference failed ({}), falling back to heuristic",
                    e
                );
            }
        }
    }

    // Heuristic fallback
    relations::extract_relations_heuristic(entities, text, relation_types, threshold)
}

#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod schema;
#[cfg(feature = "candle")]
pub use candle::GLiNER2Candle;
#[cfg(feature = "onnx")]
pub use onnx::GLiNER2Onnx;
pub use schema::{
    ClassificationResult, ClassificationTask, EntityTask, ExtractedStructure, ExtractionResult,
    FieldType, LabelCache, StructureTask, StructureValue, TaskSchema,
};

// Stub implementations (no feature)
// =============================================================================

/// GLiNER2 stub (requires onnx or candle feature).
#[cfg(not(any(feature = "onnx", feature = "candle")))]
#[derive(Debug)]
pub struct GLiNER2 {
    _private: (),
}

#[cfg(not(any(feature = "onnx", feature = "candle")))]
impl GLiNER2 {
    /// Load model (requires feature).
    pub fn from_pretrained(_model_id: &str) -> Result<Self> {
        Err(Error::FeatureNotAvailable(
            "GLiNER2 requires 'onnx' or 'candle' feature. \
             Build with: cargo build --features candle"
                .to_string(),
        ))
    }

    /// Extract (requires feature).
    pub fn extract(&self, _text: &str, _schema: &TaskSchema) -> Result<ExtractionResult> {
        Err(Error::FeatureNotAvailable(
            "GLiNER2 requires features".to_string(),
        ))
    }
}

// =============================================================================
// Unified GLiNER2 type
// =============================================================================

/// GLiNER2 model - automatically selects best available backend.
#[cfg(feature = "candle")]
pub type GLiNER2 = GLiNER2Candle;

/// GLiNER2 model - ONNX backend (when candle not enabled).
#[cfg(all(feature = "onnx", not(feature = "candle")))]
pub type GLiNER2 = GLiNER2Onnx;

// =============================================================================
// Helper functions
// =============================================================================

/// Convert word span indices to character offsets.
pub(super) fn word_span_to_char_offsets(
    text: &str,
    words: &[&str],
    start_word: usize,
    end_word: usize,
) -> (usize, usize) {
    // Defensive: Validate bounds
    if words.is_empty()
        || start_word >= words.len()
        || end_word >= words.len()
        || start_word > end_word
    {
        // Return safe defaults: empty span at start of text
        return (0, 0);
    }

    // Track our search position in **bytes**.
    let mut byte_pos = 0;
    let mut start_byte = 0;
    let mut end_byte = text.len();
    let mut found_start = false;
    let mut found_end = false;

    for (i, word) in words.iter().enumerate() {
        if let Some(pos) = text.get(byte_pos..).and_then(|s| s.find(word)) {
            let abs_pos = byte_pos + pos;

            if i == start_word {
                start_byte = abs_pos;
                found_start = true;
            }
            if i == end_word {
                end_byte = abs_pos + word.len();
                found_end = true;
                // Early exit: we found both start and end
                break;
            }

            byte_pos = abs_pos + word.len();
        } else {
            // Word not found - this shouldn't happen in normal operation,
            // but if it does, we can't reliably compute offsets
            // Continue searching but mark that we may have incorrect results
        }
    }

    // If we didn't find the words, return safe defaults
    if !found_start || !found_end {
        // Return empty span to avoid incorrect entity extraction
        (0, 0)
    } else {
        // Convert byte offsets to character offsets (anno spans are char-based).
        crate::offset::bytes_to_chars(text, start_byte, end_byte)
    }
}

/// Map entity type string to EntityType.
///
/// Uses the canonical schema mapper for consistent semantics across all backends.
pub(super) fn map_entity_type(type_str: &str) -> EntityType {
    crate::schema::map_to_canonical(type_str, None)
}

// =============================================================================
// Model Trait Implementation (ONNX)
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::Model for GLiNER2Onnx {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        let schema = TaskSchema::new().with_entities(&[
            "person",
            "organization",
            "location",
            "date",
            "event",
        ]);

        let result = self.extract(text, &schema)?;
        Ok(result.entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Custom {
                name: "event".to_string(),
                category: EntityCategory::Creative,
            },
            EntityType::Custom {
                name: "product".to_string(),
                category: EntityCategory::Creative,
            },
            EntityType::custom("misc", EntityCategory::Misc),
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "GLiNER2-ONNX"
    }

    fn description(&self) -> &'static str {
        "Multi-task information extraction via GLiNER2 (ONNX backend)"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            relation_capable: true,
            dynamic_labels: true,
            ..Default::default()
        }
    }
}

#[cfg(feature = "onnx")]
impl crate::DynamicLabels for GLiNER2Onnx {
    fn extract_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        _language: Option<Language>,
    ) -> crate::Result<Vec<crate::Entity>> {
        <Self as ZeroShotNER>::extract_with_types(self, text, labels, 0.3)
    }
}

// =============================================================================
// Model Trait Implementation (Candle)
// =============================================================================

#[cfg(feature = "candle")]
impl crate::Model for GLiNER2Candle {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        let schema = TaskSchema::new().with_entities(&[
            "person",
            "organization",
            "location",
            "date",
            "event",
        ]);

        let result = self.extract(text, &schema)?;
        Ok(result.entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Custom {
                name: "event".to_string(),
                category: EntityCategory::Creative,
            },
            EntityType::Custom {
                name: "product".to_string(),
                category: EntityCategory::Creative,
            },
            EntityType::custom("misc", EntityCategory::Misc),
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "GLiNER2-Candle"
    }

    fn description(&self) -> &'static str {
        "Multi-task information extraction via GLiNER2 (native Rust/Candle)"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            relation_capable: true,
            dynamic_labels: true,
            ..Default::default()
        }
    }
}

#[cfg(feature = "candle")]
impl crate::DynamicLabels for GLiNER2Candle {
    fn extract_with_labels(
        &self,
        text: &str,
        labels: &[&str],
        _language: Option<Language>,
    ) -> crate::Result<Vec<crate::Entity>> {
        <Self as ZeroShotNER>::extract_with_types(self, text, labels, 0.3)
    }
}

// =============================================================================
// ZeroShotNER Trait Implementation
// =============================================================================

#[cfg(feature = "onnx")]
impl ZeroShotNER for GLiNER2Onnx {
    fn default_types(&self) -> &[&'static str] {
        &["person", "organization", "location", "date", "event"]
    }

    fn extract_with_types(
        &self,
        text: &str,
        types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        self.extract_ner(text, types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Use descriptions as entity types directly (GLiNER2 supports this)
        self.extract_ner(text, descriptions, threshold)
    }
}

#[cfg(feature = "candle")]
impl ZeroShotNER for GLiNER2Candle {
    fn default_types(&self) -> &[&'static str] {
        &["person", "organization", "location", "date", "event"]
    }

    fn extract_with_types(
        &self,
        text: &str,
        types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let type_strings: Vec<String> = types.iter().map(|s| s.to_string()).collect();
        self.extract_entities(text, &type_strings, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Use descriptions as entity types directly (GLiNER2 supports this)
        let type_strings: Vec<String> = descriptions.iter().map(|s| s.to_string()).collect();
        self.extract_entities(text, &type_strings, threshold)
    }
}

#[cfg(feature = "onnx")]
impl RelationExtractor for GLiNER2Onnx {
    fn extract_with_relations(
        &self,
        text: &str,
        types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        let entities = self.extract_ner(text, types, threshold)?;

        // Use GLiREL (neural) when available, fall back to heuristics
        let relations =
            extract_relations_neural_or_heuristic(&entities, text, relation_types, threshold);

        Ok(ExtractionWithRelations {
            entities,
            relations,
        })
    }
}

#[cfg(feature = "candle")]
impl RelationExtractor for GLiNER2Candle {
    fn extract_with_relations(
        &self,
        text: &str,
        types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        let type_strings: Vec<String> = types.iter().map(|s| s.to_string()).collect();
        let entities = self.extract_entities(text, &type_strings, threshold)?;

        // Use heuristic relations for Candle backend (GLiREL is ONNX-only)
        let relations =
            relations::extract_relations_heuristic(&entities, text, relation_types, threshold);

        Ok(ExtractionWithRelations {
            entities,
            relations,
        })
    }
}

// =============================================================================
// RelationCapable Trait Implementation (high-level public interface)
// =============================================================================

#[cfg(feature = "onnx")]
impl crate::RelationCapable for GLiNER2Onnx {
    fn extract_with_relations(
        &self,
        text: &str,
        _language: Option<Language>,
    ) -> Result<(Vec<Entity>, Vec<crate::Relation>)> {
        use crate::backends::inference::{DEFAULT_ENTITY_TYPES, DEFAULT_RELATION_TYPES};
        let result = <Self as RelationExtractor>::extract_with_relations(
            self,
            text,
            DEFAULT_ENTITY_TYPES,
            DEFAULT_RELATION_TYPES,
            0.3,
        )?;
        Ok(result.into_anno_relations())
    }
}

#[cfg(feature = "candle")]
impl crate::RelationCapable for GLiNER2Candle {
    fn extract_with_relations(
        &self,
        text: &str,
        _language: Option<Language>,
    ) -> Result<(Vec<Entity>, Vec<crate::Relation>)> {
        use crate::backends::inference::{DEFAULT_ENTITY_TYPES, DEFAULT_RELATION_TYPES};
        let result = <Self as RelationExtractor>::extract_with_relations(
            self,
            text,
            DEFAULT_ENTITY_TYPES,
            DEFAULT_RELATION_TYPES,
            0.3,
        )?;
        Ok(result.into_anno_relations())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(feature = "onnx", feature = "candle"))]
    fn test_relation_heuristic_unicode_safe_and_case_insensitive() {
        use crate::backends::inference::RelationTriple;
        use crate::offset::bytes_to_chars;

        let text = "Dr. 田中 is CEO of Apple Inc. in 東京. François works at OpenAI.";
        let span = |needle: &str| {
            let (b_start, _) = text
                .match_indices(needle)
                .next()
                .expect("needle should exist in test text");
            let b_end = b_start + needle.len();
            bytes_to_chars(text, b_start, b_end)
        };

        let (s, e) = span("田中");
        let e_tanaka = Entity::new("田中", EntityType::Person, s, e, 0.9);
        let (s, e) = span("Apple Inc.");
        let e_apple = Entity::new("Apple Inc.", EntityType::Organization, s, e, 0.9);
        let (s, e) = span("東京");
        let e_tokyo = Entity::new("東京", EntityType::Location, s, e, 0.9);
        let (s, e) = span("François");
        let e_francois = Entity::new("François", EntityType::Person, s, e, 0.9);
        let (s, e) = span("OpenAI");
        let e_openai = Entity::new("OpenAI", EntityType::Organization, s, e, 0.9);

        let entities = vec![e_tanaka, e_apple, e_tokyo, e_francois, e_openai];

        // Should not panic on Unicode text; should detect at least one trigger relation.
        let rels: Vec<RelationTriple> =
            relations::extract_relations_heuristic(&entities, text, &[], 0.0);
        assert!(
            rels.iter()
                .any(|r| r.relation_type == "CEO_OF" || r.relation_type == "WORKS_FOR"),
            "expected at least one trigger-based relation, got {:?}",
            rels
        );
    }

    #[test]
    fn test_task_schema_builder() {
        let schema = TaskSchema::new()
            .with_entities(&["person", "organization"])
            .with_classification("sentiment", &["positive", "negative"], false);

        assert!(schema.entities.is_some());
        assert_eq!(schema.entities.as_ref().unwrap().types.len(), 2);
        assert_eq!(schema.classifications.len(), 1);
    }

    #[test]
    fn test_structure_task_builder() {
        let task = StructureTask::new("product")
            .with_field("name", FieldType::String)
            .with_field_described("price", FieldType::String, "Product price in USD")
            .with_choice_field("category", &["electronics", "clothing"]);

        assert_eq!(task.fields.len(), 3);
        assert_eq!(task.fields[2].choices.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_word_span_to_char_offsets() {
        use crate::offset::TextSpan;

        let text = "John works at Apple";
        let words: Vec<&str> = text.split_whitespace().collect();

        let (start, end) = word_span_to_char_offsets(text, &words, 0, 0);
        assert_eq!(TextSpan::from_chars(text, start, end).extract(text), "John");

        let (start, end) = word_span_to_char_offsets(text, &words, 3, 3);
        assert_eq!(
            TextSpan::from_chars(text, start, end).extract(text),
            "Apple"
        );

        let (start, end) = word_span_to_char_offsets(text, &words, 0, 2);
        assert_eq!(
            TextSpan::from_chars(text, start, end).extract(text),
            "John works at"
        );
    }

    #[test]
    fn test_map_entity_type() {
        assert!(matches!(map_entity_type("person"), EntityType::Person));
        assert!(matches!(
            map_entity_type("ORGANIZATION"),
            EntityType::Organization
        ));
        assert!(matches!(map_entity_type("loc"), EntityType::Location));
        // Unknown types map to Custom/Other with the uppercase version (due to schema normalization)
        assert!(
            matches!(map_entity_type("custom_type"), EntityType::Custom { ref name, .. } if name == "CUSTOM_TYPE")
        );
        // Known special types map to Custom
        assert!(matches!(
            map_entity_type("product"),
            EntityType::Custom { .. }
        ));
        assert!(matches!(
            map_entity_type("event"),
            EntityType::Custom { .. }
        ));
    }

    // =========================================================================
    // word_span_to_char_offsets: edge cases
    // =========================================================================

    #[test]
    fn test_word_span_empty_words() {
        let text = "hello world";
        let words: Vec<&str> = vec![];
        let (s, e) = word_span_to_char_offsets(text, &words, 0, 0);
        assert_eq!((s, e), (0, 0), "empty words should return (0,0)");
    }

    #[test]
    fn test_word_span_start_gt_end() {
        let text = "a b c";
        let words: Vec<&str> = text.split_whitespace().collect();
        let (s, e) = word_span_to_char_offsets(text, &words, 2, 1);
        assert_eq!((s, e), (0, 0), "start > end should return (0,0)");
    }

    #[test]
    fn test_word_span_out_of_bounds() {
        let text = "a b c";
        let words: Vec<&str> = text.split_whitespace().collect();
        let (s, e) = word_span_to_char_offsets(text, &words, 0, 10);
        assert_eq!(
            (s, e),
            (0, 0),
            "end_word >= words.len() should return (0,0)"
        );
    }

    #[test]
    fn test_word_span_single_word_text() {
        use crate::offset::TextSpan;
        let text = "hello";
        let words: Vec<&str> = text.split_whitespace().collect();
        let (s, e) = word_span_to_char_offsets(text, &words, 0, 0);
        assert_eq!(TextSpan::from_chars(text, s, e).extract(text), "hello");
    }

    #[test]
    fn test_word_span_unicode_multibyte() {
        use crate::offset::TextSpan;
        // Each CJK char is 3 bytes but 1 char offset.
        let text = "田中 works at 東京タワー";
        let words: Vec<&str> = text.split_whitespace().collect();
        // words = ["田中", "works", "at", "東京タワー"]

        let (s, e) = word_span_to_char_offsets(text, &words, 0, 0);
        assert_eq!(TextSpan::from_chars(text, s, e).extract(text), "田中");

        let (s, e) = word_span_to_char_offsets(text, &words, 3, 3);
        assert_eq!(TextSpan::from_chars(text, s, e).extract(text), "東京タワー");

        // Multi-word span across the boundary
        let (s, e) = word_span_to_char_offsets(text, &words, 1, 2);
        assert_eq!(TextSpan::from_chars(text, s, e).extract(text), "works at");
    }

    #[test]
    fn test_word_span_emoji_text() {
        use crate::offset::TextSpan;
        let text = "I love 🎉 party";
        let words: Vec<&str> = text.split_whitespace().collect();
        // words = ["I", "love", "🎉", "party"]
        let (s, e) = word_span_to_char_offsets(text, &words, 2, 2);
        assert_eq!(TextSpan::from_chars(text, s, e).extract(text), "🎉");

        let (s, e) = word_span_to_char_offsets(text, &words, 3, 3);
        assert_eq!(TextSpan::from_chars(text, s, e).extract(text), "party");
    }

    // =========================================================================
    // map_entity_type: additional coverage
    // =========================================================================

    #[test]
    fn test_map_entity_type_case_variations() {
        // All case variants should resolve to the same canonical type
        assert!(matches!(map_entity_type("Person"), EntityType::Person));
        assert!(matches!(map_entity_type("PERSON"), EntityType::Person));
        assert!(matches!(map_entity_type("PER"), EntityType::Person));
        assert!(matches!(map_entity_type("location"), EntityType::Location));
        assert!(matches!(map_entity_type("Location"), EntityType::Location));
        assert!(matches!(map_entity_type("LOC"), EntityType::Location));
        // GPE maps to Custom (geopolitical entity), not Location
        assert!(matches!(map_entity_type("GPE"), EntityType::Custom { .. }));
        assert!(matches!(map_entity_type("ORG"), EntityType::Organization));
        assert!(matches!(map_entity_type("date"), EntityType::Date));
        assert!(matches!(map_entity_type("DATE"), EntityType::Date));
    }

    #[test]
    fn test_map_entity_type_empty_string() {
        // Empty string should not panic; falls through to Custom/Other
        let ty = map_entity_type("");
        assert!(matches!(ty, EntityType::Custom { .. }));
    }

    // =========================================================================
    // TaskSchema builder: deeper coverage
    // =========================================================================

    #[test]
    fn test_task_schema_empty() {
        let schema = TaskSchema::new();
        assert!(schema.entities.is_none());
        assert!(schema.classifications.is_empty());
        assert!(schema.structures.is_empty());
    }

    #[test]
    fn test_task_schema_with_entities_described() {
        let mut desc = std::collections::HashMap::new();
        desc.insert("person".to_string(), "A human being".to_string());
        desc.insert("org".to_string(), "An organization".to_string());

        let schema = TaskSchema::new().with_entities_described(desc);
        let ent = schema.entities.as_ref().unwrap();
        assert_eq!(ent.types.len(), 2);
        assert_eq!(ent.descriptions.len(), 2);
        assert!(ent.descriptions.contains_key("person"));
    }

    #[test]
    fn test_task_schema_multiple_classifications() {
        let schema = TaskSchema::new()
            .with_classification("sentiment", &["pos", "neg"], false)
            .with_classification("topic", &["tech", "sports", "politics"], true);

        assert_eq!(schema.classifications.len(), 2);
        assert_eq!(schema.classifications[0].name, "sentiment");
        assert!(!schema.classifications[0].multi_label);
        assert_eq!(schema.classifications[1].name, "topic");
        assert!(schema.classifications[1].multi_label);
        assert_eq!(schema.classifications[1].labels.len(), 3);
    }

    #[test]
    fn test_task_schema_full_pipeline() {
        // Build a schema using all builder methods in a chain
        let schema = TaskSchema::new()
            .with_entities(&["person", "org", "product"])
            .with_classification("sentiment", &["positive", "negative"], false)
            .with_structure(
                StructureTask::new("invoice")
                    .with_field("vendor", FieldType::String)
                    .with_field("items", FieldType::List)
                    .with_choice_field("currency", &["USD", "EUR", "GBP"]),
            );

        assert_eq!(schema.entities.as_ref().unwrap().types.len(), 3);
        assert_eq!(schema.classifications.len(), 1);
        assert_eq!(schema.structures.len(), 1);
        let st = &schema.structures[0];
        assert_eq!(st.name, "invoice");
        assert_eq!(st.fields.len(), 3);
        assert_eq!(st.fields[0].field_type, FieldType::String);
        assert_eq!(st.fields[1].field_type, FieldType::List);
        assert_eq!(st.fields[2].field_type, FieldType::Choice);
        assert_eq!(
            st.fields[2].choices.as_ref().unwrap(),
            &["USD", "EUR", "GBP"]
        );
    }
}
