//! Entity types and structures for NER.

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Entity type classification.
///
/// Standard NER types following CoNLL/OntoNotes conventions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Person name (PER)
    Person,
    /// Organization name (ORG)
    Organization,
    /// Location/Place (LOC)
    Location,
    /// Date or time expression (DATE)
    Date,
    /// Monetary value (MONEY)
    Money,
    /// Percentage (PERCENT)
    Percent,
    /// Other/Miscellaneous entity type
    Other(String),
}

impl EntityType {
    /// Convert to standard label string (CoNLL format).
    #[must_use]
    pub fn as_label(&self) -> &str {
        match self {
            EntityType::Person => "PER",
            EntityType::Organization => "ORG",
            EntityType::Location => "LOC",
            EntityType::Date => "DATE",
            EntityType::Money => "MONEY",
            EntityType::Percent => "PERCENT",
            EntityType::Other(s) => s.as_str(),
        }
    }

    /// Parse from standard label string.
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        match label.to_uppercase().as_str() {
            "PER" | "PERSON" | "B-PER" | "I-PER" => EntityType::Person,
            "ORG" | "ORGANIZATION" | "B-ORG" | "I-ORG" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" | "B-LOC" | "I-LOC" => EntityType::Location,
            "DATE" | "TIME" | "B-DATE" | "I-DATE" => EntityType::Date,
            "MONEY" | "CURRENCY" | "B-MONEY" | "I-MONEY" => EntityType::Money,
            "PERCENT" | "PERCENTAGE" | "B-PERCENT" | "I-PERCENT" => EntityType::Percent,
            other => EntityType::Other(other.to_string()),
        }
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_label())
    }
}

/// Extraction method used to identify an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ExtractionMethod {
    /// Regex pattern matching (high precision for structured data)
    Pattern,
    /// Machine learning model inference
    #[default]
    ML,
    /// Rule-based / gazetteer lookup
    Rule,
    /// Hybrid: multiple methods agreed
    Ensemble,
    /// Unknown or unspecified
    Unknown,
}

impl std::fmt::Display for ExtractionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractionMethod::Pattern => write!(f, "pattern"),
            ExtractionMethod::ML => write!(f, "ml"),
            ExtractionMethod::Rule => write!(f, "rule"),
            ExtractionMethod::Ensemble => write!(f, "ensemble"),
            ExtractionMethod::Unknown => write!(f, "unknown"),
        }
    }
}

/// Provenance information for an extracted entity.
///
/// Tracks where an entity came from for debugging, explainability,
/// and confidence calibration in hybrid/ensemble systems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Provenance {
    /// Name of the backend that produced this entity (e.g., "pattern", "bert-onnx")
    pub source: Cow<'static, str>,
    /// Extraction method used
    pub method: ExtractionMethod,
    /// Specific pattern/rule name (for pattern/rule-based extraction)
    pub pattern: Option<Cow<'static, str>>,
    /// Raw confidence from the source model (before any calibration)
    pub raw_confidence: Option<f64>,
}

impl Provenance {
    /// Create provenance for pattern-based extraction.
    #[must_use]
    pub fn pattern(pattern_name: &'static str) -> Self {
        Self {
            source: Cow::Borrowed("pattern"),
            method: ExtractionMethod::Pattern,
            pattern: Some(Cow::Borrowed(pattern_name)),
            raw_confidence: Some(1.0), // Patterns are deterministic
        }
    }

    /// Create provenance for ML-based extraction.
    #[must_use]
    pub fn ml(model_name: &'static str, confidence: f64) -> Self {
        Self {
            source: Cow::Borrowed(model_name),
            method: ExtractionMethod::ML,
            pattern: None,
            raw_confidence: Some(confidence),
        }
    }

    /// Create provenance for ensemble/hybrid extraction.
    #[must_use]
    pub fn ensemble(sources: &'static str) -> Self {
        Self {
            source: Cow::Borrowed(sources),
            method: ExtractionMethod::Ensemble,
            pattern: None,
            raw_confidence: None,
        }
    }
}

/// A recognized named entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text (surface form)
    pub text: String,
    /// Entity type classification
    pub entity_type: EntityType,
    /// Start position (byte offset in original text)
    pub start: usize,
    /// End position (byte offset, exclusive)
    pub end: usize,
    /// Confidence score (0.0-1.0, calibrated)
    pub confidence: f64,
    /// Provenance: which backend/method produced this entity
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

impl Entity {
    /// Create a new entity.
    #[must_use]
    pub fn new(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
        confidence: f64,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            start,
            end,
            confidence: confidence.clamp(0.0, 1.0),
            provenance: None,
        }
    }

    /// Create a new entity with provenance information.
    #[must_use]
    pub fn with_provenance(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
        confidence: f64,
        provenance: Provenance,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            start,
            end,
            confidence: confidence.clamp(0.0, 1.0),
            provenance: Some(provenance),
        }
    }

    /// Create an entity with default confidence (1.0).
    #[must_use]
    pub fn with_type(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
    ) -> Self {
        Self::new(text, entity_type, start, end, 1.0)
    }

    /// Get the extraction method, if known.
    #[must_use]
    pub fn method(&self) -> ExtractionMethod {
        self.provenance
            .as_ref()
            .map_or(ExtractionMethod::Unknown, |p| p.method)
    }

    /// Get the source backend name, if known.
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.provenance.as_ref().map(|p| p.source.as_ref())
    }

    /// Check if this entity overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Entity) -> bool {
        !(self.end <= other.start || other.end <= self.start)
    }

    /// Calculate overlap ratio (IoU) with another entity.
    #[must_use]
    pub fn overlap_ratio(&self, other: &Entity) -> f64 {
        let intersection_start = self.start.max(other.start);
        let intersection_end = self.end.min(other.end);

        if intersection_start >= intersection_end {
            return 0.0;
        }

        let intersection = (intersection_end - intersection_start) as f64;
        let union = ((self.end - self.start) + (other.end - other.start)
            - (intersection_end - intersection_start)) as f64;

        if union == 0.0 {
            return 1.0;
        }

        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_roundtrip() {
        let types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Money,
            EntityType::Percent,
        ];

        for t in types {
            let label = t.as_label();
            let parsed = EntityType::from_label(label);
            assert_eq!(t, parsed);
        }
    }

    #[test]
    fn test_entity_overlap() {
        let e1 = Entity::new("John", EntityType::Person, 0, 4, 0.9);
        let e2 = Entity::new("Smith", EntityType::Person, 5, 10, 0.9);
        let e3 = Entity::new("John Smith", EntityType::Person, 0, 10, 0.9);

        assert!(!e1.overlaps(&e2)); // No overlap
        assert!(e1.overlaps(&e3)); // e1 is contained in e3
        assert!(e3.overlaps(&e2)); // e3 contains e2
    }

    #[test]
    fn test_confidence_clamping() {
        let e1 = Entity::new("test", EntityType::Person, 0, 4, 1.5);
        assert!((e1.confidence - 1.0).abs() < f64::EPSILON);

        let e2 = Entity::new("test", EntityType::Person, 0, 4, -0.5);
        assert!(e2.confidence.abs() < f64::EPSILON);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn confidence_always_clamped(conf in -10.0f64..10.0) {
            let e = Entity::new("test", EntityType::Person, 0, 4, conf);
            prop_assert!(e.confidence >= 0.0);
            prop_assert!(e.confidence <= 1.0);
        }

        #[test]
        fn entity_type_roundtrip(label in "[A-Z]{3,10}") {
            let et = EntityType::from_label(&label);
            let back = EntityType::from_label(et.as_label());
            // Other types may round-trip to themselves or normalize
            prop_assert!(matches!(back, EntityType::Other(_)) || back == et);
        }

        #[test]
        fn overlap_is_symmetric(
            s1 in 0usize..100,
            len1 in 1usize..50,
            s2 in 0usize..100,
            len2 in 1usize..50,
        ) {
            let e1 = Entity::new("a", EntityType::Person, s1, s1 + len1, 1.0);
            let e2 = Entity::new("b", EntityType::Person, s2, s2 + len2, 1.0);
            prop_assert_eq!(e1.overlaps(&e2), e2.overlaps(&e1));
        }

        #[test]
        fn overlap_ratio_bounded(
            s1 in 0usize..100,
            len1 in 1usize..50,
            s2 in 0usize..100,
            len2 in 1usize..50,
        ) {
            let e1 = Entity::new("a", EntityType::Person, s1, s1 + len1, 1.0);
            let e2 = Entity::new("b", EntityType::Person, s2, s2 + len2, 1.0);
            let ratio = e1.overlap_ratio(&e2);
            prop_assert!(ratio >= 0.0);
            prop_assert!(ratio <= 1.0);
        }

        #[test]
        fn self_overlap_ratio_is_one(s in 0usize..100, len in 1usize..50) {
            let e = Entity::new("test", EntityType::Person, s, s + len, 1.0);
            let ratio = e.overlap_ratio(&e);
            prop_assert!((ratio - 1.0).abs() < 1e-10);
        }
    }
}
