//! Entity types and structures for NER.

use serde::{Deserialize, Serialize};

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
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
}

impl Entity {
    /// Create a new entity.
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
        }
    }

    /// Create an entity with default confidence (1.0).
    pub fn with_type(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
    ) -> Self {
        Self::new(text, entity_type, start, end, 1.0)
    }

    /// Check if this entity overlaps with another.
    pub fn overlaps(&self, other: &Entity) -> bool {
        !(self.end <= other.start || other.end <= self.start)
    }

    /// Calculate overlap ratio (IoU) with another entity.
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
