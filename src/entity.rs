//! Entity types and structures for NER.
//!
//! # Type Hierarchy
//!
//! Entities are organized into categories based on detection characteristics:
//!
//! - **Named entities** (require ML/context): Person, Organization, Location
//! - **Temporal entities** (structured patterns): Date, Time
//! - **Numeric entities** (structured patterns): Money, Percent, Quantity
//! - **Contact entities** (structured patterns): Email, Url, Phone
//!
//! # Design Principles
//!
//! 1. **First-class types** for common entities (no `Other("EMAIL")`)
//! 2. **Category queries** via `entity_type.category()`
//! 3. **Extensibility** via `Custom { name, category }` for domain-specific types
//! 4. **Normalization support** via `normalized` field on Entity

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

// ============================================================================
// Entity Category (OntoNotes-inspired)
// ============================================================================

/// Category of entity based on detection characteristics and semantics.
///
/// Based on OntoNotes 5.0 categories with extensions for structured data.
/// Determines whether ML is required or patterns suffice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityCategory {
    /// Named entities for people/groups (ML-required).
    /// Types: Person, NORP (nationalities/religious/political groups)
    Agent,
    /// Named entities for organizations/facilities (ML-required).
    /// Types: Organization, Facility
    Organization,
    /// Named entities for places (ML-required).
    /// Types: GPE (geo-political), Location (geographic)
    Place,
    /// Named entities for creative/conceptual (ML-required).
    /// Types: Event, Product, WorkOfArt, Law, Language
    Creative,
    /// Temporal entities (pattern-detectable).
    /// Types: Date, Time
    Temporal,
    /// Numeric entities (pattern-detectable).
    /// Types: Money, Percent, Quantity, Cardinal, Ordinal
    Numeric,
    /// Contact/identifier entities (pattern-detectable).
    /// Types: Email, Url, Phone
    Contact,
    /// Miscellaneous/unknown category
    Misc,
}

impl EntityCategory {
    /// Returns true if this category requires ML for detection.
    #[must_use]
    pub const fn requires_ml(&self) -> bool {
        matches!(
            self,
            EntityCategory::Agent
                | EntityCategory::Organization
                | EntityCategory::Place
                | EntityCategory::Creative
        )
    }

    /// Returns true if this category can be detected via patterns.
    #[must_use]
    pub const fn pattern_detectable(&self) -> bool {
        matches!(
            self,
            EntityCategory::Temporal | EntityCategory::Numeric | EntityCategory::Contact
        )
    }

    /// Returns OntoNotes-compatible category name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            EntityCategory::Agent => "agent",
            EntityCategory::Organization => "organization",
            EntityCategory::Place => "place",
            EntityCategory::Creative => "creative",
            EntityCategory::Temporal => "temporal",
            EntityCategory::Numeric => "numeric",
            EntityCategory::Contact => "contact",
            EntityCategory::Misc => "misc",
        }
    }
}

impl std::fmt::Display for EntityCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Entity Type
// ============================================================================

/// Entity type classification.
///
/// Organized into categories:
/// - **Named** (ML-required): Person, Organization, Location
/// - **Temporal** (pattern): Date, Time
/// - **Numeric** (pattern): Money, Percent, Quantity, Cardinal, Ordinal
/// - **Contact** (pattern): Email, Url, Phone
///
/// # Examples
///
/// ```rust
/// use anno::EntityType;
///
/// let ty = EntityType::Email;
/// assert!(ty.category().pattern_detectable());
/// assert!(!ty.category().requires_ml());
///
/// let ty = EntityType::Person;
/// assert!(ty.category().requires_ml());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    // === Named Entities (ML-required) ===
    /// Person name (PER) - requires ML/context
    Person,
    /// Organization name (ORG) - requires ML/context
    Organization,
    /// Location/Place (LOC/GPE) - requires ML/context
    Location,

    // === Temporal Entities (Pattern-detectable) ===
    /// Date expression (DATE) - pattern-detectable
    Date,
    /// Time expression (TIME) - pattern-detectable
    Time,

    // === Numeric Entities (Pattern-detectable) ===
    /// Monetary value (MONEY) - pattern-detectable
    Money,
    /// Percentage (PERCENT) - pattern-detectable
    Percent,
    /// Quantity with unit (QUANTITY) - pattern-detectable
    Quantity,
    /// Cardinal number (CARDINAL) - pattern-detectable
    Cardinal,
    /// Ordinal number (ORDINAL) - pattern-detectable
    Ordinal,

    // === Contact Entities (Pattern-detectable) ===
    /// Email address - pattern-detectable
    Email,
    /// URL/URI - pattern-detectable
    Url,
    /// Phone number - pattern-detectable
    Phone,

    // === Extensibility ===
    /// Domain-specific custom type with explicit category
    Custom {
        /// Type name (e.g., "DISEASE", "PRODUCT", "EVENT")
        name: String,
        /// Category for this custom type
        category: EntityCategory,
    },

    /// Legacy catch-all for unknown types (prefer Custom for new code)
    #[serde(rename = "Other")]
    Other(String),
}

impl EntityType {
    /// Get the category of this entity type.
    #[must_use]
    pub fn category(&self) -> EntityCategory {
        match self {
            // Agent entities (people/groups)
            EntityType::Person => EntityCategory::Agent,
            // Organization entities
            EntityType::Organization => EntityCategory::Organization,
            // Place entities (locations)
            EntityType::Location => EntityCategory::Place,
            // Temporal entities
            EntityType::Date | EntityType::Time => EntityCategory::Temporal,
            // Numeric entities
            EntityType::Money
            | EntityType::Percent
            | EntityType::Quantity
            | EntityType::Cardinal
            | EntityType::Ordinal => EntityCategory::Numeric,
            // Contact entities
            EntityType::Email | EntityType::Url | EntityType::Phone => EntityCategory::Contact,
            // Custom with explicit category
            EntityType::Custom { category, .. } => *category,
            // Legacy Other - assume misc
            EntityType::Other(_) => EntityCategory::Misc,
        }
    }

    /// Returns true if this entity type requires ML for detection.
    #[must_use]
    pub fn requires_ml(&self) -> bool {
        self.category().requires_ml()
    }

    /// Returns true if this entity type can be detected via patterns.
    #[must_use]
    pub fn pattern_detectable(&self) -> bool {
        self.category().pattern_detectable()
    }

    /// Convert to standard label string (CoNLL/OntoNotes format).
    #[must_use]
    pub fn as_label(&self) -> &str {
        match self {
            EntityType::Person => "PER",
            EntityType::Organization => "ORG",
            EntityType::Location => "LOC",
            EntityType::Date => "DATE",
            EntityType::Time => "TIME",
            EntityType::Money => "MONEY",
            EntityType::Percent => "PERCENT",
            EntityType::Quantity => "QUANTITY",
            EntityType::Cardinal => "CARDINAL",
            EntityType::Ordinal => "ORDINAL",
            EntityType::Email => "EMAIL",
            EntityType::Url => "URL",
            EntityType::Phone => "PHONE",
            EntityType::Custom { name, .. } => name.as_str(),
            EntityType::Other(s) => s.as_str(),
        }
    }

    /// Parse from standard label string.
    ///
    /// Handles various formats: CoNLL (PER), OntoNotes (PERSON), BIO (B-PER).
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        // Strip BIO prefix if present
        let label = label
            .strip_prefix("B-")
            .or_else(|| label.strip_prefix("I-"))
            .or_else(|| label.strip_prefix("E-"))
            .or_else(|| label.strip_prefix("S-"))
            .unwrap_or(label);

        match label.to_uppercase().as_str() {
            // Named entities
            "PER" | "PERSON" => EntityType::Person,
            "ORG" | "ORGANIZATION" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" => EntityType::Location,
            // Temporal
            "DATE" => EntityType::Date,
            "TIME" => EntityType::Time,
            // Numeric
            "MONEY" | "CURRENCY" => EntityType::Money,
            "PERCENT" | "PERCENTAGE" => EntityType::Percent,
            "QUANTITY" => EntityType::Quantity,
            "CARDINAL" => EntityType::Cardinal,
            "ORDINAL" => EntityType::Ordinal,
            // Contact
            "EMAIL" => EntityType::Email,
            "URL" | "URI" => EntityType::Url,
            "PHONE" | "TELEPHONE" => EntityType::Phone,
            // Unknown -> Other
            other => EntityType::Other(other.to_string()),
        }
    }

    /// Create a custom domain-specific entity type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use anno::{EntityType, EntityCategory};
    ///
    /// // Medical entity - custom domain-specific type
    /// let disease = EntityType::custom("DISEASE", EntityCategory::Agent);
    /// assert!(disease.requires_ml());
    ///
    /// // ID patterns - can be detected via patterns
    /// let product_id = EntityType::custom("PRODUCT_ID", EntityCategory::Misc);
    /// ```
    #[must_use]
    pub fn custom(name: impl Into<String>, category: EntityCategory) -> Self {
        EntityType::Custom {
            name: name.into(),
            category,
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
///
/// # Entity Structure
///
/// ```text
/// "Contact John at john@example.com on Jan 15"
///          ^^^^    ^^^^^^^^^^^^^^^^    ^^^^^^
///          PER     EMAIL               DATE
///          |       |                   |
///          Named   Contact             Temporal
///          (ML)    (Pattern)           (Pattern)
/// ```
///
/// # Normalization
///
/// Entities can have a normalized form for downstream processing:
/// - Dates: "Jan 15" → "2024-01-15" (ISO 8601)
/// - Money: "$1.5M" → "1500000 USD"
/// - Locations: "NYC" → "New York City"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text (surface form as it appears in source)
    pub text: String,
    /// Entity type classification
    pub entity_type: EntityType,
    /// Start position (byte offset in original text)
    pub start: usize,
    /// End position (byte offset, exclusive)
    pub end: usize,
    /// Confidence score (0.0-1.0, calibrated)
    pub confidence: f64,
    /// Normalized/canonical form (e.g., "Jan 15" → "2024-01-15")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized: Option<String>,
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
            normalized: None,
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
            normalized: None,
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

    /// Set the normalized form for this entity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use anno::{Entity, EntityType};
    ///
    /// let mut entity = Entity::new("Jan 15", EntityType::Date, 0, 6, 0.95);
    /// entity.set_normalized("2024-01-15");
    /// assert_eq!(entity.normalized.as_deref(), Some("2024-01-15"));
    /// ```
    pub fn set_normalized(&mut self, normalized: impl Into<String>) {
        self.normalized = Some(normalized.into());
    }

    /// Get the normalized form, or the original text if not normalized.
    #[must_use]
    pub fn normalized_or_text(&self) -> &str {
        self.normalized.as_deref().unwrap_or(&self.text)
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

    /// Get the entity category.
    #[must_use]
    pub fn category(&self) -> EntityCategory {
        self.entity_type.category()
    }

    /// Returns true if this entity was detected via patterns (not ML).
    #[must_use]
    pub fn is_structured(&self) -> bool {
        self.entity_type.pattern_detectable()
    }

    /// Returns true if this entity required ML for detection.
    #[must_use]
    pub fn is_named(&self) -> bool {
        self.entity_type.requires_ml()
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

    #[test]
    fn test_entity_categories() {
        // Agent/Org/Place entities require ML
        assert_eq!(EntityType::Person.category(), EntityCategory::Agent);
        assert_eq!(EntityType::Organization.category(), EntityCategory::Organization);
        assert_eq!(EntityType::Location.category(), EntityCategory::Place);
        assert!(EntityType::Person.requires_ml());
        assert!(!EntityType::Person.pattern_detectable());

        // Temporal entities are pattern-detectable
        assert_eq!(EntityType::Date.category(), EntityCategory::Temporal);
        assert_eq!(EntityType::Time.category(), EntityCategory::Temporal);
        assert!(EntityType::Date.pattern_detectable());
        assert!(!EntityType::Date.requires_ml());

        // Numeric entities are pattern-detectable
        assert_eq!(EntityType::Money.category(), EntityCategory::Numeric);
        assert_eq!(EntityType::Percent.category(), EntityCategory::Numeric);
        assert!(EntityType::Money.pattern_detectable());

        // Contact entities are pattern-detectable
        assert_eq!(EntityType::Email.category(), EntityCategory::Contact);
        assert_eq!(EntityType::Url.category(), EntityCategory::Contact);
        assert_eq!(EntityType::Phone.category(), EntityCategory::Contact);
        assert!(EntityType::Email.pattern_detectable());
    }

    #[test]
    fn test_new_types_roundtrip() {
        let types = [
            EntityType::Time,
            EntityType::Email,
            EntityType::Url,
            EntityType::Phone,
            EntityType::Quantity,
            EntityType::Cardinal,
            EntityType::Ordinal,
        ];

        for t in types {
            let label = t.as_label();
            let parsed = EntityType::from_label(label);
            assert_eq!(t, parsed, "Roundtrip failed for {}", label);
        }
    }

    #[test]
    fn test_custom_entity_type() {
        let disease = EntityType::custom("DISEASE", EntityCategory::Agent);
        assert_eq!(disease.as_label(), "DISEASE");
        assert!(disease.requires_ml());

        let product_id = EntityType::custom("PRODUCT_ID", EntityCategory::Misc);
        assert_eq!(product_id.as_label(), "PRODUCT_ID");
        assert!(!product_id.requires_ml());
        assert!(!product_id.pattern_detectable());
    }

    #[test]
    fn test_entity_normalization() {
        let mut e = Entity::new("Jan 15", EntityType::Date, 0, 6, 0.95);
        assert!(e.normalized.is_none());
        assert_eq!(e.normalized_or_text(), "Jan 15");

        e.set_normalized("2024-01-15");
        assert_eq!(e.normalized.as_deref(), Some("2024-01-15"));
        assert_eq!(e.normalized_or_text(), "2024-01-15");
    }

    #[test]
    fn test_entity_helpers() {
        let named = Entity::new("John", EntityType::Person, 0, 4, 0.9);
        assert!(named.is_named());
        assert!(!named.is_structured());
        assert_eq!(named.category(), EntityCategory::Agent);

        let structured = Entity::new("$100", EntityType::Money, 0, 4, 0.95);
        assert!(!structured.is_named());
        assert!(structured.is_structured());
        assert_eq!(structured.category(), EntityCategory::Numeric);
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
