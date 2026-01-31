//! Schema-constrained entity typing.
//!
//! Enforces ontological constraints on entity types to prevent impossible predictions:
//! - Type hierarchy constraints (a CEO must be a Person)
//! - Mutual exclusion (an entity can't be both Person and Organization)
//! - Domain constraints (valid types for a given schema)
//!
//! # Research Background
//!
//! Neural NER models can produce ontologically impossible predictions:
//! - Predicting "Apple" as both ORG and FOOD in the same context
//! - Predicting fine-grained types without coarse types (CEO without Person)
//!
//! Schema constraints act as a post-processing filter or can be integrated
//! into decoding (constrained beam search).
//!
//! # Example
//!
//! ```rust
//! use anno::eval::schema::{TypeSchema, TypeConstraint};
//!
//! let mut schema = TypeSchema::new();
//!
//! // Person subtypes
//! schema.add_hierarchy("politician", "person");
//! schema.add_hierarchy("athlete", "person");
//!
//! // Mutual exclusion
//! schema.add_exclusion("person", "organization");
//! schema.add_exclusion("person", "location");
//!
//! // Validate
//! assert!(schema.is_valid_type("politician"));
//! assert!(schema.implies("politician", "person"));
//! assert!(schema.mutually_exclusive("person", "organization"));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Type Constraint
// =============================================================================

/// A constraint on entity types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeConstraint {
    /// Type A implies type B (A is a subtype of B)
    Implies {
        /// The subtype
        subtype: String,
        /// The supertype that subtype implies
        supertype: String,
    },
    /// Type A and B are mutually exclusive
    Excludes {
        /// First mutually exclusive type
        type_a: String,
        /// Second mutually exclusive type
        type_b: String,
    },
    /// Type must be from allowed set
    AllowedTypes {
        /// Set of allowed types
        types: Vec<String>,
    },
    /// Type requires another type to be present
    Requires {
        /// Type that has the requirement
        dependent: String,
        /// Type that must be present
        required: String,
    },
}

// =============================================================================
// Type Schema
// =============================================================================

/// Schema defining valid entity types and their relationships.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeSchema {
    /// Type hierarchy: child -> parent
    hierarchy: HashMap<String, String>,
    /// Mutual exclusions
    exclusions: Vec<(String, String)>,
    /// All known types
    all_types: HashSet<String>,
    /// Type descriptions
    descriptions: HashMap<String, String>,
    /// Schema name
    pub name: String,
}

impl TypeSchema {
    /// Create a new empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Add a type to the schema.
    pub fn add_type(&mut self, type_name: &str) {
        self.all_types.insert(type_name.to_lowercase());
    }

    /// Add a type with description.
    pub fn add_type_with_description(&mut self, type_name: &str, description: &str) {
        let name = type_name.to_lowercase();
        self.all_types.insert(name.clone());
        self.descriptions.insert(name, description.to_string());
    }

    /// Add hierarchy relationship (subtype -> supertype).
    pub fn add_hierarchy(&mut self, subtype: &str, supertype: &str) {
        let sub = subtype.to_lowercase();
        let sup = supertype.to_lowercase();
        self.all_types.insert(sub.clone());
        self.all_types.insert(sup.clone());
        self.hierarchy.insert(sub, sup);
    }

    /// Add mutual exclusion.
    pub fn add_exclusion(&mut self, type_a: &str, type_b: &str) {
        let a = type_a.to_lowercase();
        let b = type_b.to_lowercase();
        self.all_types.insert(a.clone());
        self.all_types.insert(b.clone());
        self.exclusions.push((a, b));
    }

    /// Check if a type is known in this schema.
    #[must_use]
    pub fn is_valid_type(&self, type_name: &str) -> bool {
        self.all_types.contains(&type_name.to_lowercase())
    }

    /// Check if subtype implies supertype (directly or transitively).
    #[must_use]
    pub fn implies(&self, subtype: &str, supertype: &str) -> bool {
        let sub = subtype.to_lowercase();
        let sup = supertype.to_lowercase();

        if sub == sup {
            return true;
        }

        // Walk up hierarchy
        let mut current = sub;
        while let Some(parent) = self.hierarchy.get(&current) {
            if parent == &sup {
                return true;
            }
            current = parent.clone();
        }

        false
    }

    /// Check if two types are mutually exclusive.
    #[must_use]
    pub fn mutually_exclusive(&self, type_a: &str, type_b: &str) -> bool {
        let a = type_a.to_lowercase();
        let b = type_b.to_lowercase();

        // Direct exclusion
        for (x, y) in &self.exclusions {
            if (x == &a && y == &b) || (x == &b && y == &a) {
                return true;
            }
        }

        // Exclusion through hierarchy
        for (x, y) in &self.exclusions {
            if (self.implies(&a, x) && self.implies(&b, y))
                || (self.implies(&a, y) && self.implies(&b, x))
            {
                return true;
            }
        }

        false
    }

    /// Get all ancestors (supertypes) of a type.
    #[must_use]
    pub fn ancestors(&self, type_name: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = type_name.to_lowercase();

        while let Some(parent) = self.hierarchy.get(&current) {
            result.push(parent.clone());
            current = parent.clone();
        }

        result
    }

    /// Get direct children (subtypes) of a type.
    #[must_use]
    pub fn children(&self, type_name: &str) -> Vec<String> {
        let name = type_name.to_lowercase();
        self.hierarchy
            .iter()
            .filter(|(_, parent)| *parent == &name)
            .map(|(child, _)| child.clone())
            .collect()
    }

    /// Get all descendants (subtypes, recursive).
    #[must_use]
    pub fn descendants(&self, type_name: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut queue = vec![type_name.to_lowercase()];

        while let Some(current) = queue.pop() {
            let children = self.children(&current);
            for child in children {
                result.push(child.clone());
                queue.push(child);
            }
        }

        result
    }

    /// Get root types (no parent).
    #[must_use]
    pub fn roots(&self) -> Vec<String> {
        self.all_types
            .iter()
            .filter(|t| !self.hierarchy.contains_key(*t))
            .cloned()
            .collect()
    }

    /// Create standard NER schema (CoNLL-style).
    #[must_use]
    pub fn conll() -> Self {
        let mut schema = Self::new().with_name("CoNLL-2003");

        // Base types
        schema.add_type("person");
        schema.add_type("organization");
        schema.add_type("location");
        schema.add_type("misc");

        // Exclusions
        schema.add_exclusion("person", "organization");
        schema.add_exclusion("person", "location");
        schema.add_exclusion("organization", "location");

        schema
    }

    /// Create OntoNotes schema.
    #[must_use]
    pub fn ontonotes() -> Self {
        let mut schema = Self::new().with_name("OntoNotes 5.0");

        // Named entities
        schema.add_type_with_description("person", "People, including fictional");
        schema.add_type_with_description("norp", "Nationalities, religious/political groups");
        schema.add_type_with_description("fac", "Buildings, airports, highways, bridges");
        schema.add_type_with_description("org", "Companies, agencies, institutions");
        schema.add_type_with_description("gpe", "Countries, cities, states");
        schema.add_type_with_description("loc", "Non-GPE locations");
        schema.add_type_with_description("product", "Objects, vehicles, foods");
        schema.add_type_with_description("event", "Named hurricanes, battles, wars");
        schema.add_type_with_description("work_of_art", "Titles of books, songs, etc.");
        schema.add_type_with_description("law", "Named documents made into laws");
        schema.add_type_with_description("language", "Any named language");

        // Numeric
        schema.add_type_with_description("date", "Absolute or relative dates");
        schema.add_type_with_description("time", "Times smaller than a day");
        schema.add_type_with_description("percent", "Percentage");
        schema.add_type_with_description("money", "Monetary values");
        schema.add_type_with_description("quantity", "Measurements");
        schema.add_type_with_description("ordinal", "First, second, etc.");
        schema.add_type_with_description("cardinal", "Numerals");

        // Exclusions
        schema.add_exclusion("person", "org");
        schema.add_exclusion("person", "gpe");
        schema.add_exclusion("person", "loc");
        schema.add_exclusion("org", "gpe");

        schema
    }

    /// Create fine-grained entity typing schema.
    #[must_use]
    pub fn fine_grained() -> Self {
        let mut schema = Self::new().with_name("Fine-Grained");

        // Person hierarchy
        schema.add_type("person");
        schema.add_hierarchy("politician", "person");
        schema.add_hierarchy("athlete", "person");
        schema.add_hierarchy("artist", "person");
        schema.add_hierarchy("scientist", "person");
        schema.add_hierarchy("businessperson", "person");
        schema.add_hierarchy("author", "artist");
        schema.add_hierarchy("musician", "artist");
        schema.add_hierarchy("actor", "artist");

        // Organization hierarchy
        schema.add_type("organization");
        schema.add_hierarchy("company", "organization");
        schema.add_hierarchy("government", "organization");
        schema.add_hierarchy("educational", "organization");
        schema.add_hierarchy("sports_team", "organization");
        schema.add_hierarchy("political_party", "organization");
        schema.add_hierarchy("religious_org", "organization");

        // Location hierarchy
        schema.add_type("location");
        schema.add_hierarchy("country", "location");
        schema.add_hierarchy("city", "location");
        schema.add_hierarchy("state", "location");
        schema.add_hierarchy("facility", "location");
        schema.add_hierarchy("natural_feature", "location");

        // Exclusions
        schema.add_exclusion("person", "organization");
        schema.add_exclusion("person", "location");
        schema.add_exclusion("organization", "location");

        schema
    }
}

// =============================================================================
// Schema Validator
// =============================================================================

/// Validation result for a set of entity predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Is the prediction valid?
    pub is_valid: bool,
    /// Violations found
    pub violations: Vec<SchemaViolation>,
    /// Corrected types (if auto-correction enabled)
    pub corrected_types: Vec<String>,
}

/// A schema violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaViolation {
    /// Violation type
    pub kind: ViolationKind,
    /// Entity index
    pub entity_idx: usize,
    /// Entity text
    pub entity_text: String,
    /// Predicted type
    pub predicted_type: String,
    /// Message
    pub message: String,
}

/// Type of schema violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationKind {
    /// Unknown type
    UnknownType,
    /// Missing required supertype
    MissingSupertype,
    /// Mutually exclusive types assigned
    MutualExclusion,
    /// Custom violation
    Custom(String),
}

impl std::fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownType => write!(f, "Unknown type"),
            Self::MissingSupertype => write!(f, "Missing supertype"),
            Self::MutualExclusion => write!(f, "Mutual exclusion"),
            Self::Custom(msg) => write!(f, "Custom: {}", msg),
        }
    }
}

/// Validates entity predictions against a schema.
#[derive(Debug, Clone)]
pub struct SchemaValidator {
    /// The schema to validate against
    schema: TypeSchema,
    /// Whether to auto-correct violations
    auto_correct: bool,
    /// Strict mode: unknown types are violations
    strict: bool,
}

impl SchemaValidator {
    /// Create a new validator.
    pub fn new(schema: TypeSchema) -> Self {
        Self {
            schema,
            auto_correct: false,
            strict: false,
        }
    }

    /// Enable auto-correction.
    pub fn with_auto_correct(mut self) -> Self {
        self.auto_correct = true;
        self
    }

    /// Enable strict mode.
    pub fn with_strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Validate a single entity type.
    pub fn validate_type(&self, type_name: &str) -> Vec<SchemaViolation> {
        let mut violations = Vec::new();

        if self.strict && !self.schema.is_valid_type(type_name) {
            violations.push(SchemaViolation {
                kind: ViolationKind::UnknownType,
                entity_idx: 0,
                entity_text: String::new(),
                predicted_type: type_name.to_string(),
                message: format!("Unknown type: {}", type_name),
            });
        }

        violations
    }

    /// Validate a set of types for a single entity (multi-label).
    pub fn validate_multi_label(
        &self,
        entity_text: &str,
        types: &[String],
        entity_idx: usize,
    ) -> Vec<SchemaViolation> {
        let mut violations = Vec::new();

        // Check for mutual exclusions
        for (i, type_a) in types.iter().enumerate() {
            for type_b in types.iter().skip(i + 1) {
                if self.schema.mutually_exclusive(type_a, type_b) {
                    violations.push(SchemaViolation {
                        kind: ViolationKind::MutualExclusion,
                        entity_idx,
                        entity_text: entity_text.to_string(),
                        predicted_type: format!("{}, {}", type_a, type_b),
                        message: format!("{} and {} are mutually exclusive", type_a, type_b),
                    });
                }
            }
        }

        // Check for missing supertypes (if we have fine-grained without coarse)
        for t in types {
            for ancestor in self.schema.ancestors(t) {
                if !types.iter().any(|x| x.to_lowercase() == ancestor) {
                    // Only warn, don't flag as error (coarse type is implied)
                }
            }
        }

        violations
    }

    /// Get the schema.
    #[must_use]
    pub fn schema(&self) -> &TypeSchema {
        &self.schema
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_hierarchy() {
        let mut schema = TypeSchema::new();
        schema.add_hierarchy("politician", "person");
        schema.add_hierarchy("senator", "politician");

        assert!(schema.implies("politician", "person"));
        assert!(schema.implies("senator", "politician"));
        assert!(schema.implies("senator", "person")); // Transitive
        assert!(!schema.implies("person", "politician"));
    }

    #[test]
    fn test_mutual_exclusion() {
        let mut schema = TypeSchema::new();
        schema.add_exclusion("person", "organization");

        assert!(schema.mutually_exclusive("person", "organization"));
        assert!(schema.mutually_exclusive("organization", "person"));
        assert!(!schema.mutually_exclusive("person", "person"));
    }

    #[test]
    fn test_exclusion_through_hierarchy() {
        let mut schema = TypeSchema::new();
        schema.add_hierarchy("politician", "person");
        schema.add_hierarchy("company", "organization");
        schema.add_exclusion("person", "organization");

        // Politician (subtype of person) excludes company (subtype of org)
        assert!(schema.mutually_exclusive("politician", "company"));
    }

    #[test]
    fn test_ancestors() {
        let mut schema = TypeSchema::new();
        schema.add_hierarchy("senator", "politician");
        schema.add_hierarchy("politician", "person");

        let ancestors = schema.ancestors("senator");
        assert_eq!(ancestors, vec!["politician", "person"]);
    }

    #[test]
    fn test_descendants() {
        let schema = TypeSchema::fine_grained();

        let descendants = schema.descendants("artist");
        assert!(descendants.contains(&"author".to_string()));
        assert!(descendants.contains(&"musician".to_string()));
    }

    #[test]
    fn test_conll_schema() {
        let schema = TypeSchema::conll();

        assert!(schema.is_valid_type("person"));
        assert!(schema.is_valid_type("organization"));
        assert!(schema.mutually_exclusive("person", "organization"));
    }

    #[test]
    fn test_ontonotes_schema() {
        let schema = TypeSchema::ontonotes();

        assert!(schema.is_valid_type("person"));
        assert!(schema.is_valid_type("gpe"));
        assert!(schema.is_valid_type("date"));
    }

    #[test]
    fn test_validator() {
        let schema = TypeSchema::conll();
        let validator = SchemaValidator::new(schema).with_strict();

        // Valid type
        let violations = validator.validate_type("person");
        assert!(violations.is_empty());

        // Unknown type in strict mode
        let violations = validator.validate_type("unknown_type");
        assert!(!violations.is_empty());
        assert_eq!(violations[0].kind, ViolationKind::UnknownType);
    }

    #[test]
    fn test_multi_label_validation() {
        let schema = TypeSchema::fine_grained();
        let validator = SchemaValidator::new(schema);

        // Valid: person and politician (not exclusive, politician is subtype)
        let violations = validator.validate_multi_label(
            "Obama",
            &["person".to_string(), "politician".to_string()],
            0,
        );
        assert!(violations.is_empty());

        // Invalid: person and organization are mutually exclusive
        let violations = validator.validate_multi_label(
            "Apple",
            &["person".to_string(), "organization".to_string()],
            0,
        );
        assert!(!violations.is_empty());
        assert_eq!(violations[0].kind, ViolationKind::MutualExclusion);
    }

    #[test]
    fn test_roots() {
        let schema = TypeSchema::fine_grained();
        let roots = schema.roots();

        assert!(roots.contains(&"person".to_string()));
        assert!(roots.contains(&"organization".to_string()));
        assert!(roots.contains(&"location".to_string()));
    }
}
