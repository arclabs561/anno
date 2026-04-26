//! JSON Schema generation for anno types.
//!
//! When the `schema` feature is enabled, this module provides JSON Schema
//! generation for core types via `schemars`.
//!
//! # Usage
//!
//! Add the `schema` feature to your dependency:
//!
//! ```toml
//! [dependencies]
//! anno = { version = "0.9", features = ["schema"] }
//! ```
//!
//! Then generate schemas:
//!
//! ```rust,ignore
//! use crate::core::schema::generate_entity_schema;
//!
//! let schema = generate_entity_schema();
//! println!("{}", serde_json::to_string_pretty(&schema).unwrap());
//! ```
//!
//! # Generated Schemas
//!
//! The following types have JSON Schema support:
//!
//! - `Entity` - Core entity with type, span, confidence
//! - `EntityType` - Entity type classification
//! - `EntityCategory` - High-level entity categorization
//! - `Span` - Text or visual span
//! - `Relation` - Entity relationship
//! - `GroundedDocument` - Document with entities and signals
//!
//! # Interoperability
//!
//! The generated schemas can be used with:
//!
//! - TypeScript/JavaScript (via `json-schema-to-typescript`)
//! - Python (via `datamodel-code-generator`)
//! - Other languages with JSON Schema tooling
//!
//! # Example: Generate All Schemas
//!
//! ```rust,ignore
//! use crate::schema;
//! use std::fs;
//!
//! let schemas = vec![
//!     ("entity.json", schema::generate_entity_schema()),
//!     ("entity_type.json", schema::generate_entity_type_schema()),
//!     ("grounded_document.json", schema::generate_grounded_document_schema()),
//! ];
//!
//! for (filename, schema) in schemas {
//!     let json = serde_json::to_string_pretty(&schema).unwrap();
//!     fs::write(filename, json).unwrap();
//! }
//! ```

#[cfg(feature = "schema")]
use crate::Confidence;
#[cfg(feature = "schema")]
use schemars::{schema_for, JsonSchema};

#[cfg(feature = "schema")]
use serde_json::Value;

// Re-export schemars for downstream use
#[cfg(feature = "schema")]
pub use schemars;

/// Generate JSON Schema for `Entity`.
#[cfg(feature = "schema")]
pub fn generate_entity_schema() -> Value {
    serde_json::to_value(schema_for!(schema_types::SchemaEntity))
        .expect("Entity schema should be valid JSON")
}

/// Generate JSON Schema for `EntityType`.
#[cfg(feature = "schema")]
pub fn generate_entity_type_schema() -> Value {
    serde_json::to_value(schema_for!(schema_types::SchemaEntityType))
        .expect("EntityType schema should be valid JSON")
}

/// Generate JSON Schema for `GroundedDocument`.
#[cfg(feature = "schema")]
pub fn generate_grounded_document_schema() -> Value {
    serde_json::to_value(schema_for!(schema_types::SchemaGroundedDocument))
        .expect("GroundedDocument schema should be valid JSON")
}

/// Generate JSON Schema for `Span`.
#[cfg(feature = "schema")]
pub fn generate_span_schema() -> Value {
    serde_json::to_value(schema_for!(schema_types::SchemaSpan))
        .expect("Span schema should be valid JSON")
}

/// Generate JSON Schema for `Relation`.
#[cfg(feature = "schema")]
pub fn generate_relation_schema() -> Value {
    serde_json::to_value(schema_for!(schema_types::SchemaRelation))
        .expect("Relation schema should be valid JSON")
}

/// Generate all schemas as a map of name -> schema.
#[cfg(feature = "schema")]
pub fn generate_all_schemas() -> std::collections::HashMap<&'static str, Value> {
    let mut schemas = std::collections::HashMap::new();
    schemas.insert("entity", generate_entity_schema());
    schemas.insert("entity_type", generate_entity_type_schema());
    schemas.insert("span", generate_span_schema());
    schemas.insert("relation", generate_relation_schema());
    schemas.insert("grounded_document", generate_grounded_document_schema());
    schemas
}

// =============================================================================
// Schema-enabled wrapper types
// =============================================================================
//
// These types mirror the main types but with JsonSchema derive.
// They're used only for schema generation, not at runtime.

#[cfg(feature = "schema")]
mod schema_types {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Schema-enabled version of EntityCategory.
    #[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum SchemaEntityCategory {
        Agent,
        Organization,
        Place,
        Creative,
        Temporal,
        Numeric,
        Contact,
        Relation,
        Misc,
    }

    /// Schema-enabled version of EntityType.
    #[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum SchemaEntityType {
        Person,
        Organization,
        Location,
        Date,
        Time,
        Money,
        Percent,
        Quantity,
        Cardinal,
        Ordinal,
        Email,
        Url,
        Phone,
        Custom {
            name: String,
            category: SchemaEntityCategory,
        },
        Other(String),
    }

    /// Schema-enabled version of Span.
    #[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
    pub struct SchemaSpan {
        /// Start offset (character or visual coordinate).
        pub start: i64,
        /// End offset (character or visual coordinate).
        pub end: i64,
        /// Optional width for visual spans.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub width: Option<f64>,
        /// Optional height for visual spans.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub height: Option<f64>,
    }

    /// Schema-enabled version of Entity.
    #[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
    pub struct SchemaEntity {
        /// Entity surface text.
        pub text: String,
        /// Entity type classification.
        pub entity_type: SchemaEntityType,
        /// Start character offset.
        pub start: usize,
        /// End character offset (exclusive).
        pub end: usize,
        /// Confidence score (0.0-1.0).
        pub confidence: Confidence,
        /// Normalized/canonical form.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub normalized: Option<String>,
        /// External knowledge base ID (e.g., Wikidata Q-ID).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub kb_id: Option<String>,
        /// Local coreference cluster ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub canonical_id: Option<u64>,
    }

    /// Schema-enabled version of Relation.
    #[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
    pub struct SchemaRelation {
        /// Source entity index.
        pub source_idx: usize,
        /// Target entity index.
        pub target_idx: usize,
        /// Relation type (e.g., "CEO_OF", "LOCATED_IN").
        pub relation_type: String,
        /// Confidence score (0.0-1.0).
        pub confidence: Confidence,
        /// Trigger text span (e.g., "works at").
        #[serde(skip_serializing_if = "Option::is_none")]
        pub trigger: Option<SchemaSpan>,
    }

    /// Schema-enabled version of GroundedDocument.
    #[derive(Debug, Clone, JsonSchema, Serialize, Deserialize)]
    pub struct SchemaGroundedDocument {
        /// Document ID.
        pub id: String,
        /// Document text content.
        pub text: String,
        /// Extracted entities.
        pub entities: Vec<SchemaEntity>,
        /// Extracted relations.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub relations: Vec<SchemaRelation>,
        /// Document language (ISO 639-1).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub language: Option<String>,
        /// Source URL or path.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub source: Option<String>,
    }
}

#[cfg(feature = "schema")]
pub use schema_types::*;

#[cfg(test)]
#[cfg(feature = "schema")]
mod tests {
    use super::*;

    #[test]
    fn test_entity_schema_generation() {
        let schema = generate_entity_schema();
        assert!(schema.is_object());
        let obj = schema.as_object().unwrap();
        assert!(
            obj.contains_key("$schema") || obj.contains_key("title") || obj.contains_key("type")
        );
    }

    #[test]
    fn test_all_schemas_generation() {
        let schemas = generate_all_schemas();
        assert!(schemas.contains_key("entity"));
        assert!(schemas.contains_key("entity_type"));
        assert!(schemas.contains_key("grounded_document"));
    }
}
