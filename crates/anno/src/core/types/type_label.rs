//! Unified type label for entity classification.
//!
//! This module provides [`TypeLabel`], a unified representation for entity types
//! that bridges the gap between:
//!
//! - **Core types**: The canonical [`EntityType`] enum with known categories
//! - **Custom types**: Arbitrary string labels from domain-specific schemas
//!
//! # Motivation
//!
//! Different parts of the pipeline use different representations:
//!
//! | Component | Previous Type | Issue |
//! |-----------|---------------|-------|
//! | `Entity` | `EntityType` enum | No escape hatch for custom types |
//! | `Track.entity_type` | `Option<String>` | Loses type safety |
//! | `Identity.entity_type` | `Option<String>` | Inconsistent with Entity |
//! | `Signal.label` | `String` | Pure string, no typing |
//!
//! `TypeLabel` unifies these by supporting both core and custom types:
//!
//! ```rust
//! use anno::{TypeLabel, EntityType};
//!
//! // From a known type
//! let person = TypeLabel::Core(EntityType::Person);
//!
//! // From a custom domain type
//! let protein = TypeLabel::custom("PROTEIN");
//!
//! // String conversion
//! let from_str: TypeLabel = "Person".parse().unwrap();
//! assert_eq!(from_str, TypeLabel::Core(EntityType::Person));
//!
//! let custom_str: TypeLabel = "UNKNOWN_TYPE_XYZ".parse().unwrap();
//! assert!(matches!(custom_str, TypeLabel::Custom(_)));
//! ```

use super::super::entity::{EntityCategory, EntityType};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

/// A unified type label supporting both core and custom entity types.
///
/// Text labels use the categories assigned by [`EntityType::from_label`].
/// A caller-supplied [`EntityType`] retains its exact spelling and category.
/// The wire format uses a string whenever it preserves the label, or the
/// tagged `Custom { name, category }` form for explicit overrides.
#[derive(Debug, Clone)]
pub enum TypeLabel {
    /// An explicit taxonomy type, including custom types with a supplied category.
    Core(EntityType),
    /// A domain label whose category can be inferred from its spelling.
    Custom(String),
}

impl TypeLabel {
    /// Parse a label, recognizing built-in entity types and preserving domain spellings.
    #[must_use]
    pub fn custom(label: impl Into<String>) -> Self {
        Self::from(label.into())
    }

    /// Get the string representation of this label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Core(et) => et.as_label(),
            Self::Custom(s) => s,
        }
    }

    /// Check if this is a core entity type.
    #[must_use]
    pub const fn is_core(&self) -> bool {
        self.as_core().is_some()
    }

    /// Check if this is a domain-specific type.
    #[must_use]
    pub const fn is_custom(&self) -> bool {
        !self.is_core()
    }

    /// Borrow the underlying built-in entity type, when present.
    #[must_use]
    pub const fn as_core(&self) -> Option<&EntityType> {
        match self {
            Self::Core(EntityType::Custom { .. }) | Self::Custom(_) => None,
            Self::Core(et) => Some(et),
        }
    }

    /// Return the category declared by the taxonomy or supplied by the caller.
    #[must_use]
    pub fn category(&self) -> EntityCategory {
        match self {
            Self::Core(et) => et.category(),
            Self::Custom(s) => EntityType::from_label(s).category(),
        }
    }

    /// Recover the entity type, retaining custom spelling and category.
    #[must_use]
    pub fn to_entity_type(&self) -> EntityType {
        match self {
            Self::Core(et) => et.clone(),
            Self::Custom(s) => EntityType::custom(s, self.category()),
        }
    }

    /// Use the supplied category for a domain label. Built-in categories are fixed.
    #[must_use]
    pub fn to_entity_type_with_category(&self, category: EntityCategory) -> EntityType {
        match self.as_core() {
            Some(et) => et.clone(),
            None => EntityType::custom(self.as_str(), category),
        }
    }
}

// The two public variants can express the same domain type. Equality is
// semantic so direct construction agrees with conversion and deserialization.
impl PartialEq for TypeLabel {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Core(left), Self::Core(right)) => left == right,
            (Self::Custom(left), Self::Custom(right)) => left == right,
            (Self::Core(EntityType::Custom { name, category }), Self::Custom(label))
            | (Self::Custom(label), Self::Core(EntityType::Custom { name, category })) => {
                name == label && *category == EntityType::from_label(label).category()
            }
            _ => false,
        }
    }
}

impl Eq for TypeLabel {}

impl Hash for TypeLabel {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Category overrides with the same name can share a hash bucket.
        self.is_core().hash(state);
        self.as_str().hash(state);
    }
}

impl Default for TypeLabel {
    fn default() -> Self {
        Self::Custom("OTHER".to_string())
    }
}

impl fmt::Display for TypeLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<EntityType> for TypeLabel {
    fn from(et: EntityType) -> Self {
        Self::Core(et)
    }
}

impl From<&str> for TypeLabel {
    fn from(s: &str) -> Self {
        match EntityType::from_label(s) {
            EntityType::Custom { .. } => Self::Custom(s.to_owned()),
            et => Self::Core(et),
        }
    }
}

impl From<String> for TypeLabel {
    fn from(s: String) -> Self {
        let et = EntityType::from_label(&s);
        match et {
            EntityType::Custom { .. } => Self::Custom(s),
            _ => Self::Core(et),
        }
    }
}

impl From<Option<String>> for TypeLabel {
    fn from(opt: Option<String>) -> Self {
        opt.map_or(Self::default(), Self::from)
    }
}

impl FromStr for TypeLabel {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from(s))
    }
}

impl Serialize for TypeLabel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if Self::from(self.as_str()) == *self {
            serializer.serialize_str(self.as_str())
        } else {
            self.to_entity_type().serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for TypeLabel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Label(String),
            Explicit(EntityType),
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Label(s) => Self::from(s),
            Wire::Explicit(et) => Self::from(et),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_type() {
        let label = TypeLabel::Core(EntityType::Person);
        assert!(label.is_core());
        assert!(!label.is_custom());
        // as_label() returns CoNLL format ("PER"), not human-readable ("Person")
        assert_eq!(label.as_str(), "PER");
        assert_eq!(label.to_string(), "PER");
    }

    #[test]
    fn test_custom_type() {
        let label = TypeLabel::custom("UNKNOWN_TYPE_XYZ");
        assert!(!label.is_core());
        assert!(label.is_custom());
        assert_eq!(label.as_str(), "UNKNOWN_TYPE_XYZ");
    }

    #[test]
    fn test_from_str_core() {
        // "Person" parses to EntityType::Person via from_label
        let label: TypeLabel = "Person".parse().unwrap();
        assert!(label.is_core());
        assert_eq!(label.as_core(), Some(&EntityType::Person));
    }

    #[test]
    fn test_from_str_custom() {
        // Unknown types become Custom (not Core with Other)
        // This preserves the original string rather than wrapping in Other
        let label: TypeLabel = "UNKNOWN_TYPE_XYZ".parse().unwrap();
        assert!(label.is_custom());
        assert_eq!(label.as_str(), "UNKNOWN_TYPE_XYZ");
    }

    #[test]
    fn test_serde_roundtrip() {
        let labels = vec![
            TypeLabel::Core(EntityType::Person),
            TypeLabel::Core(EntityType::Organization),
            TypeLabel::custom("UNKNOWN_TYPE_XYZ"),
            TypeLabel::custom("PROTEIN"),
        ];

        for label in labels {
            let json = serde_json::to_string(&label).unwrap();
            let parsed: TypeLabel = serde_json::from_str(&json).unwrap();
            assert_eq!(label, parsed);
        }
    }

    #[test]
    fn custom_categories_survive_all_construction_paths() {
        let group = EntityType::from_label("GROUP");
        let typed = TypeLabel::from(group.clone());
        let parsed = TypeLabel::from("GROUP");
        assert!(typed.is_custom());
        assert_eq!(typed.as_core(), None);
        assert_eq!(parsed.to_entity_type(), group);
        assert_eq!(typed, parsed);
        assert_eq!(
            std::collections::HashSet::from([typed.clone(), parsed]).len(),
            1
        );
        assert_eq!(
            serde_json::to_value(&typed).unwrap(),
            serde_json::json!("GROUP")
        );

        for custom in [
            EntityType::custom("DISEASE", EntityCategory::Agent),
            EntityType::custom("PER", EntityCategory::Agent),
            EntityType::custom("caseSensitive", EntityCategory::Creative),
        ] {
            let label = TypeLabel::from(custom.clone());
            assert!(label.is_custom());
            let json = serde_json::to_value(&label).unwrap();
            assert!(json.get("Custom").is_some());
            let saved: TypeLabel = serde_json::from_value(json).unwrap();
            assert_eq!(saved, label);
            assert_eq!(saved.to_entity_type(), custom);
        }
    }

    #[test]
    fn test_to_entity_type() {
        let core = TypeLabel::Core(EntityType::Location);
        assert_eq!(core.to_entity_type(), EntityType::Location);

        // Custom types become EntityType::Custom with Misc category
        let custom = TypeLabel::custom("UNKNOWN_TYPE_XYZ");
        let et = custom.to_entity_type();
        assert!(matches!(et, EntityType::Custom { ref name, .. } if name == "UNKNOWN_TYPE_XYZ"));
    }
}
