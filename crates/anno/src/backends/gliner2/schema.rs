#![allow(unused_imports)]
//! GLiNER2 shared schema types: task definition, extraction results, caches.
//!
//! These are feature-agnostic — imported by both the ONNX and Candle backends.

#[cfg(feature = "candle")]
use std::sync::RwLock;
use crate::{Entity, EntityType, Error, Result};
use anno_core::EntityCategory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::backends::inference::{ExtractionWithRelations, RelationExtractor, ZeroShotNER};

// =============================================================================
// Special Token IDs (gliner-multitask-large-v0.5 vocabulary)
// Valid tokens: [MASK]=128000, [FLERT]=128001, <<ENT>>=128002, <<SEP>>=128003
// Note: [P], [C], [L] markers don't exist in this model - DO NOT USE 128004+
// =============================================================================

/// <<ENT>> token - entity type marker (class_token_index in config)
#[cfg(feature = "onnx")]
pub(super) const TOKEN_ENT: u32 = 128002;
/// <<SEP>> separator token
#[cfg(feature = "onnx")]
pub(super) const TOKEN_SEP: u32 = 128003;
/// Start token [CLS]
#[cfg(feature = "onnx")]
pub(super) const TOKEN_START: u32 = 1;
/// End token [SEP]
#[cfg(feature = "onnx")]
pub(super) const TOKEN_END: u32 = 2;

/// Default max span width
pub(super) const MAX_SPAN_WIDTH: usize = 12;
/// Max count for structure instances (0-19)
#[cfg(feature = "candle")]
pub(super) const MAX_COUNT: usize = 20;

// =============================================================================
// Label Embedding Cache
// =============================================================================

/// Cache for label embeddings to avoid recomputation
#[derive(Debug, Default)]
pub struct LabelCache {
    #[cfg(feature = "candle")]
    cache: RwLock<HashMap<String, Vec<f32>>>,
    #[cfg(not(feature = "candle"))]
    _phantom: std::marker::PhantomData<()>,
}

#[cfg(feature = "candle")]
impl LabelCache {
    pub(super) fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub(super) fn get(&self, label: &str) -> Option<Vec<f32>> {
        self.cache.read().unwrap_or_else(|e| e.into_inner()).get(label).cloned()
    }

    pub(super) fn insert(&self, label: String, embedding: Vec<f32>) {
        self.cache.write().unwrap_or_else(|e| e.into_inner()).insert(label, embedding);
    }
}

// =============================================================================
// Task Schema
// =============================================================================

/// Schema defining what to extract.
///
/// Use builder methods to construct complex schemas:
///
/// ```rust,ignore
/// let schema = TaskSchema::new()
///     .with_entities(&["person", "organization"])
///     .with_classification("sentiment", &["positive", "negative"], false)
///     .with_structure(
///         StructureTask::new("product")
///             .with_field("name", FieldType::String)
///             .with_field("price", FieldType::String)
///     );
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskSchema {
    /// Entity types to extract
    pub entities: Option<EntityTask>,
    /// Classification tasks
    pub classifications: Vec<ClassificationTask>,
    /// Structure extraction tasks
    pub structures: Vec<StructureTask>,
}

impl TaskSchema {
    /// Create empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add entity types to extract.
    pub fn with_entities(mut self, types: &[&str]) -> Self {
        self.entities = Some(EntityTask {
            types: types.iter().map(|s| s.to_string()).collect(),
            descriptions: HashMap::new(),
        });
        self
    }

    /// Add entity types with descriptions for better zero-shot.
    pub fn with_entities_described(mut self, types_with_desc: HashMap<String, String>) -> Self {
        let types: Vec<String> = types_with_desc.keys().cloned().collect();
        self.entities = Some(EntityTask {
            types,
            descriptions: types_with_desc,
        });
        self
    }

    /// Add a classification task.
    pub fn with_classification(mut self, name: &str, labels: &[&str], multi_label: bool) -> Self {
        self.classifications.push(ClassificationTask {
            name: name.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            multi_label,
            descriptions: HashMap::new(),
        });
        self
    }

    /// Add a structure extraction task.
    pub fn with_structure(mut self, task: StructureTask) -> Self {
        self.structures.push(task);
        self
    }
}

/// Entity extraction task configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityTask {
    /// Entity type labels
    pub types: Vec<String>,
    /// Optional descriptions for each type
    pub descriptions: HashMap<String, String>,
}

/// Classification task configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationTask {
    /// Task name (e.g., "sentiment")
    pub name: String,
    /// Class labels
    pub labels: Vec<String>,
    /// Whether multiple labels can be selected
    pub multi_label: bool,
    /// Optional descriptions for labels
    pub descriptions: HashMap<String, String>,
}

/// Hierarchical structure extraction task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StructureTask {
    /// Structure type name (parent entity)
    pub name: String,
    /// Child fields to extract
    pub fields: Vec<StructureField>,
}

impl StructureTask {
    /// Create new structure task.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fields: Vec::new(),
        }
    }

    /// Structure type (always equal to `name`).
    #[must_use]
    pub fn structure_type(&self) -> &str {
        &self.name
    }

    /// Add a field to extract.
    pub fn with_field(mut self, name: &str, field_type: FieldType) -> Self {
        self.fields.push(StructureField {
            name: name.to_string(),
            field_type,
            description: None,
            choices: None,
        });
        self
    }

    /// Add a field with description.
    pub fn with_field_described(
        mut self,
        name: &str,
        field_type: FieldType,
        description: &str,
    ) -> Self {
        self.fields.push(StructureField {
            name: name.to_string(),
            field_type,
            description: Some(description.to_string()),
            choices: None,
        });
        self
    }

    /// Add a choice field with constrained options.
    pub fn with_choice_field(mut self, name: &str, choices: &[&str]) -> Self {
        self.fields.push(StructureField {
            name: name.to_string(),
            field_type: FieldType::Choice,
            description: None,
            choices: Some(choices.iter().map(|s| s.to_string()).collect()),
        });
        self
    }
}

/// Structure field configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: FieldType,
    /// Optional description
    pub description: Option<String>,
    /// For Choice type: allowed values
    pub choices: Option<Vec<String>>,
}

/// Field type for structure extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// Single string value
    String,
    /// List of values
    List,
    /// Choice from constrained options
    Choice,
}

// =============================================================================
// Extraction Results
// =============================================================================

/// Combined extraction result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Extracted entities
    pub entities: Vec<Entity>,
    /// Classification results by task name
    pub classifications: HashMap<String, ClassificationResult>,
    /// Extracted structures
    pub structures: Vec<ExtractedStructure>,
}

/// Classification result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Selected label(s)
    pub labels: Vec<String>,
    /// Score for each label
    pub scores: HashMap<String, f32>,
}

/// Extracted structure instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractedStructure {
    /// Structure type
    pub structure_type: String,
    /// Extracted field values
    pub fields: HashMap<String, StructureValue>,
}

/// Value for a structure field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructureValue {
    /// Single value
    Single(String),
    /// List of values
    List(Vec<String>),
}

// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- constants ----

    #[test]
    fn max_span_width_is_12() {
        assert_eq!(MAX_SPAN_WIDTH, 12);
    }

    // ---- LabelCache (non-candle path) ----

    #[test]
    fn label_cache_default() {
        let _cache = LabelCache::default();
        // Constructing without panic is sufficient for the phantom-data path.
    }

    // ---- TaskSchema builder ----

    #[test]
    fn task_schema_default_is_empty() {
        let schema = TaskSchema::new();
        assert!(schema.entities.is_none());
        assert!(schema.classifications.is_empty());
        assert!(schema.structures.is_empty());
    }

    #[test]
    fn task_schema_with_entities() {
        let schema = TaskSchema::new().with_entities(&["person", "org"]);
        let ent = schema.entities.as_ref().unwrap();
        assert_eq!(ent.types, vec!["person", "org"]);
        assert!(ent.descriptions.is_empty());
    }

    #[test]
    fn task_schema_with_entities_described() {
        let mut descs = HashMap::new();
        descs.insert("person".to_string(), "a human".to_string());
        descs.insert("org".to_string(), "an organization".to_string());
        let schema = TaskSchema::new().with_entities_described(descs);
        let ent = schema.entities.as_ref().unwrap();
        assert_eq!(ent.types.len(), 2);
        assert_eq!(ent.descriptions.len(), 2);
        assert_eq!(ent.descriptions["person"], "a human");
    }

    #[test]
    fn task_schema_with_classification() {
        let schema =
            TaskSchema::new().with_classification("sentiment", &["positive", "negative"], true);
        assert_eq!(schema.classifications.len(), 1);
        let cls = &schema.classifications[0];
        assert_eq!(cls.name, "sentiment");
        assert_eq!(cls.labels, vec!["positive", "negative"]);
        assert!(cls.multi_label);
        assert!(cls.descriptions.is_empty());
    }

    #[test]
    fn task_schema_with_structure() {
        let st = StructureTask::new("product")
            .with_field("name", FieldType::String)
            .with_field("price", FieldType::String);
        let schema = TaskSchema::new().with_structure(st);
        assert_eq!(schema.structures.len(), 1);
        assert_eq!(schema.structures[0].name, "product");
        assert_eq!(schema.structures[0].fields.len(), 2);
    }

    #[test]
    fn task_schema_chained_builder() {
        let schema = TaskSchema::new()
            .with_entities(&["person"])
            .with_classification("topic", &["a", "b"], false)
            .with_structure(StructureTask::new("item").with_field("f", FieldType::List));
        assert!(schema.entities.is_some());
        assert_eq!(schema.classifications.len(), 1);
        assert_eq!(schema.structures.len(), 1);
    }

    // ---- StructureTask ----

    #[test]
    fn structure_task_new() {
        let st = StructureTask::new("invoice");
        assert_eq!(st.name, "invoice");
        assert_eq!(st.structure_type(), "invoice");
        assert!(st.fields.is_empty());
    }

    #[test]
    fn structure_task_with_field_described() {
        let st = StructureTask::new("t").with_field_described(
            "amount",
            FieldType::String,
            "total amount",
        );
        assert_eq!(st.fields.len(), 1);
        assert_eq!(st.fields[0].name, "amount");
        assert_eq!(st.fields[0].description.as_deref(), Some("total amount"));
    }

    #[test]
    fn structure_task_with_choice_field() {
        let st = StructureTask::new("t").with_choice_field("color", &["red", "blue"]);
        assert_eq!(st.fields[0].field_type, FieldType::Choice);
        assert_eq!(
            st.fields[0].choices.as_ref().unwrap(),
            &vec!["red".to_string(), "blue".to_string()]
        );
    }

    // ---- FieldType ----

    #[test]
    fn field_type_equality() {
        assert_eq!(FieldType::String, FieldType::String);
        assert_ne!(FieldType::String, FieldType::List);
        assert_ne!(FieldType::List, FieldType::Choice);
    }

    // ---- ExtractionResult / StructureValue ----

    #[test]
    fn extraction_result_default() {
        let r = ExtractionResult::default();
        assert!(r.entities.is_empty());
        assert!(r.classifications.is_empty());
        assert!(r.structures.is_empty());
    }

    #[test]
    fn structure_value_variants() {
        let single = StructureValue::Single("hello".into());
        let list = StructureValue::List(vec!["a".into(), "b".into()]);
        // Smoke-test debug formatting (ensures Debug derive works).
        let _ = format!("{single:?}");
        let _ = format!("{list:?}");
    }
}
