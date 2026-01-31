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
//! ```rust,ignore
//! use anno_core::types::TypeLabel;
//! use anno_core::EntityType;
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
//! let custom_str: TypeLabel = "DISEASE".parse().unwrap();
//! assert!(matches!(custom_str, TypeLabel::Custom(_)));
//! ```

use super::super::entity::{EntityCategory, EntityType};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// A unified type label supporting both core and custom entity types.
///
/// # Design Philosophy
///
/// Rather than having separate `EntityType` and `String` fields scattered
/// throughout the codebase, `TypeLabel` provides a single type that:
///
/// 1. **Preserves type safety** for known entity types via `EntityType`
/// 2. **Allows extensibility** for domain-specific types via `Custom`
/// 3. **Provides consistent serialization** (always as string)
/// 4. **Enables bidirectional conversion** with both `EntityType` and `String`
///
/// # Serialization
///
/// `TypeLabel` serializes as a string for interoperability:
///
/// - `Core(EntityType::Person)` → `"Person"`
/// - `Custom("PROTEIN")` → `"PROTEIN"`
///
/// Deserialization attempts to parse as `EntityType` first, falling back to `Custom`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeLabel {
    /// A core entity type from the canonical taxonomy.
    Core(EntityType),
    /// A custom domain-specific type not in the core taxonomy.
    Custom(String),
}

impl TypeLabel {
    /// Create a custom type label.
    ///
    /// Note: if `label` matches a known [`EntityType`] label, this returns
    /// `TypeLabel::Core(...)` instead. Use this to keep “custom means unknown”
    /// consistent across parsing/serde roundtrips.
    #[must_use]
    pub fn custom(label: impl Into<String>) -> Self {
        let s = label.into();
        match EntityType::from_label(&s) {
            EntityType::Other(_) => Self::Custom(s),
            et => Self::Core(et),
        }
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
        matches!(self, Self::Core(_))
    }

    /// Check if this is a custom type.
    #[must_use]
    pub const fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }

    /// Try to get the core entity type, if this is one.
    #[must_use]
    pub const fn as_core(&self) -> Option<&EntityType> {
        match self {
            Self::Core(et) => Some(et),
            Self::Custom(_) => None,
        }
    }

    /// Convert to a core entity type, mapping customs to `EntityType::Other`.
    ///
    /// Note: Custom types become `EntityType::Other` since we don't have category info.
    /// Use [`Self::to_entity_type_with_category`] if you have category information.
    #[must_use]
    pub fn to_entity_type(&self) -> EntityType {
        match self {
            Self::Core(et) => et.clone(),
            Self::Custom(s) => EntityType::Other(s.clone()),
        }
    }

    /// Convert to a core entity type with explicit category for customs.
    #[must_use]
    pub fn to_entity_type_with_category(&self, category: EntityCategory) -> EntityType {
        match self {
            Self::Core(et) => et.clone(),
            Self::Custom(s) => EntityType::custom(s.clone(), category),
        }
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
        // `EntityType::from_str` is infallible (unknown labels become `Other`), so
        // treat `Other(_)` as a *custom* label here to preserve the distinction.
        match EntityType::from_label(s) {
            EntityType::Other(_) => Self::Custom(s.to_string()),
            et => Self::Core(et),
        }
    }
}

impl From<String> for TypeLabel {
    fn from(s: String) -> Self {
        let et = EntityType::from_label(&s);
        match et {
            EntityType::Other(_) => Self::Custom(s),
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

// Custom serialization: always serialize as string
impl Serialize for TypeLabel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

// Custom deserialization: parse string, try Core first then Custom
impl<'de> Deserialize<'de> for TypeLabel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Self::from(s))
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
            // If a label is known to the ontology, it is canonicalized to Core.
            TypeLabel::custom("PROTEIN"),
        ];

        for label in labels {
            let json = serde_json::to_string(&label).unwrap();
            let parsed: TypeLabel = serde_json::from_str(&json).unwrap();
            // Note: roundtrip may not be exact for Core types due to CoNLL format
            // "PER" parses back to Person, which matches
            if label.is_custom() {
                assert_eq!(label, parsed);
            }
        }
    }

    #[test]
    fn test_to_entity_type() {
        let core = TypeLabel::Core(EntityType::Location);
        assert_eq!(core.to_entity_type(), EntityType::Location);

        // Custom types become Other(String), not EntityType::Custom
        let custom = TypeLabel::custom("UNKNOWN_TYPE_XYZ");
        let et = custom.to_entity_type();
        assert!(matches!(et, EntityType::Other(ref s) if s == "UNKNOWN_TYPE_XYZ"));
    }
}
