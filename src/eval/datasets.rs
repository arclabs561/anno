//! Dataset loading for NER evaluation.
#![allow(missing_docs)] // Internal evaluation types
//!
//! Supports multiple dataset formats:
//! - CoNLL-2003 (classic BIO tagging format)
//! - JSON/JSONL (modern format used by OpenNER, MultiNERD, Wikiann)
//! - HuggingFace Datasets format
//!
//! Modern datasets (2024-2025):
//! - OpenNER 1.0: 52 languages, standardized JSON
//! - MultiNERD: Multi-ontology, JSON format
//! - Wikiann: 282 languages, JSONL format

use crate::EntityType;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

// =============================================================================
// Gold Entity Types
// =============================================================================

/// Gold standard entity annotation.
///
/// This is the canonical entity type for all NER evaluation.
/// Use this instead of creating local entity structs.
///
/// # Example
/// ```rust
/// use anno::eval::GoldEntity;
/// use anno::EntityType;
///
/// let entity = GoldEntity::new("John Doe", EntityType::Person, 0);
/// assert_eq!(entity.end, 8);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldEntity {
    /// The entity text (surface form)
    pub text: String,
    /// Normalized entity type
    pub entity_type: EntityType,
    /// Original label from dataset (e.g., "B-PER", "ACTOR")
    pub original_label: String,
    /// Character offset (start)
    pub start: usize,
    /// Character offset (end, exclusive)
    pub end: usize,
}

impl GoldEntity {
    /// Create a new gold entity with computed end offset.
    ///
    /// Note: Uses character count (not byte count) for Unicode correctness.
    #[must_use]
    pub fn new(text: impl Into<String>, entity_type: EntityType, start: usize) -> Self {
        let text = text.into();
        // Use char count for Unicode correctness
        let end = start + text.chars().count();
        Self {
            text,
            entity_type,
            original_label: String::new(),
            start,
            end,
        }
    }

    /// Create with explicit span (start, end).
    pub fn with_span(
        text: impl Into<String>,
        entity_type: EntityType,
        start: usize,
        end: usize,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            original_label: String::new(),
            start,
            end,
        }
    }

    /// Create with explicit original label.
    ///
    /// Note: Uses character count (not byte count) for Unicode correctness.
    pub fn with_label(
        text: impl Into<String>,
        entity_type: EntityType,
        original_label: impl Into<String>,
        start: usize,
    ) -> Self {
        let text = text.into();
        let end = start + text.chars().count();
        Self {
            text,
            entity_type,
            original_label: original_label.into(),
            start,
            end,
        }
    }

    /// Create with all fields explicit.
    pub fn full(
        text: impl Into<String>,
        entity_type: EntityType,
        original_label: impl Into<String>,
        start: usize,
        end: usize,
    ) -> Self {
        Self {
            text: text.into(),
            entity_type,
            original_label: original_label.into(),
            start,
            end,
        }
    }

    /// Check if this entity overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if spans match exactly.
    pub fn span_matches(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }

    /// Check if this is an exact match (span + type).
    pub fn exact_matches(&self, other: &Self) -> bool {
        self.span_matches(other) && self.entity_type == other.entity_type
    }
}

/// Backwards-compatible alias for GoldEntity.
///
/// Deprecated: Use `GoldEntity` directly.
#[deprecated(since = "0.1.0", note = "Use GoldEntity instead")]
pub type GroundTruthEntity = GoldEntity;

/// JSON format for NER datasets (OpenNER, MultiNERD style).
///
/// Format:
/// ```json
/// {
///   "text": "John Smith works at Acme Corp.",
///   "entities": [
///     {"text": "John Smith", "label": "PER", "start": 0, "end": 10},
///     {"text": "Acme Corp", "label": "ORG", "start": 20, "end": 29}
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSONNERExample {
    pub text: String,
    pub entities: Vec<JSONEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSONEntity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub confidence: Option<f64>,
}

/// JSONL format (one JSON object per line).
///
/// Used by Wikiann and many modern datasets.
pub type JSONLNERExample = JSONNERExample;

/// Load JSON format NER dataset.
///
/// Supports:
/// - Single JSON file with array of examples
/// - JSONL file (one JSON object per line)
pub fn load_json_ner_dataset<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Vec<GoldEntity>)>> {
    let content = std::fs::read_to_string(path.as_ref()).map_err(Error::Io)?;

    let mut test_cases = Vec::new();

    // Try JSONL first (one object per line)
    let is_jsonl = content.lines().count() > 1
        && content
            .lines()
            .all(|line| line.trim().starts_with('{') && line.trim().ends_with('}'));

    if is_jsonl {
        // JSONL format
        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let example: JSONLNERExample = serde_json::from_str(line).map_err(|e| {
                Error::Parse(format!(
                    "Failed to parse JSONL line {}: {}",
                    line_num + 1,
                    e
                ))
            })?;

            let entities: Vec<GoldEntity> = example
                .entities
                .into_iter()
                .map(|e| {
                    let entity_type = map_label_to_entity_type(&e.label);
                    GoldEntity::full(e.text, entity_type, &e.label, e.start, e.end)
                })
                .collect();

            // Validate entities against text
            let validation = crate::eval::validation::validate_ground_truth_entities(
                &example.text,
                &entities,
                false, // Warnings for overlaps, not errors
            );
            if !validation.is_valid {
                return Err(Error::InvalidInput(format!(
                    "Invalid entities in dataset: {}",
                    validation.errors.join("; ")
                )));
            }

            test_cases.push((example.text, entities));
        }
    } else {
        // Single JSON file (array of examples)
        let examples: Vec<JSONNERExample> = serde_json::from_str(&content)
            .map_err(|e| Error::Parse(format!("Failed to parse JSON: {}", e)))?;

        for example in examples {
            let entities: Vec<GoldEntity> = example
                .entities
                .into_iter()
                .map(|e| {
                    let entity_type = map_label_to_entity_type(&e.label);
                    GoldEntity::full(e.text, entity_type, &e.label, e.start, e.end)
                })
                .collect();

            // Validate entities against text
            let validation = crate::eval::validation::validate_ground_truth_entities(
                &example.text,
                &entities,
                false, // Warnings for overlaps, not errors
            );
            if !validation.is_valid {
                return Err(Error::InvalidInput(format!(
                    "Invalid entities in dataset: {}",
                    validation.errors.join("; ")
                )));
            }

            test_cases.push((example.text, entities));
        }
    }

    Ok(test_cases)
}

/// Load HuggingFace Datasets format.
///
/// HuggingFace datasets are typically stored as JSON/JSONL with specific structure.
/// This function handles common HuggingFace NER dataset formats.
pub fn load_hf_ner_dataset<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Vec<GoldEntity>)>> {
    // HuggingFace datasets are often JSONL or JSON arrays
    // Try JSONL first, then fall back to JSON array
    load_json_ner_dataset(path)
}

/// Map label string to EntityType.
///
/// Handles various label formats:
/// - CoNLL: "PER", "ORG", "LOC", "MISC"
/// - OpenNER: Standardized labels
/// - MultiNERD: Extended labels (PER, ORG, LOC, ANIM, BIO, CEL, etc.)
/// - Wikiann: "PER", "ORG", "LOC", "MISC"
fn map_label_to_entity_type(label: &str) -> EntityType {
    let label_upper = label.to_uppercase();

    match label_upper.as_str() {
        "PER" | "PERSON" | "PERS" => EntityType::Person,
        "ORG" | "ORGANIZATION" | "ORGANISATION" => EntityType::Organization,
        "LOC" | "LOCATION" | "GPE" | "GEO" => EntityType::Location,
        "DATE" | "TIME" => EntityType::Date,
        "MONEY" | "CURRENCY" => EntityType::Money,
        "PERCENT" | "PERCENTAGE" => EntityType::Percent,
        "MISC" | "MISCELLANEOUS" => EntityType::Other("misc".to_string()),
        // MultiNERD extended types
        "ANIM" | "ANIMAL" => EntityType::Other("animal".to_string()),
        "BIO" | "BIOLOGICAL" => EntityType::Other("biological".to_string()),
        "CEL" | "CELESTIAL" => EntityType::Other("celestial".to_string()),
        "EVE" | "EVENT" => EntityType::Other("event".to_string()),
        "PROD" | "PRODUCT" => EntityType::Other("product".to_string()),
        "SPO" | "SPORT" => EntityType::Other("sport".to_string()),
        // Default: preserve original label
        _ => EntityType::Other(label.to_string()),
    }
}

/// Auto-detect dataset format and load.
///
/// Tries to detect format based on file extension and content:
/// - `.conll`, `.conll2003` → CoNLL-2003 format
/// - `.json`, `.jsonl` → JSON/JSONL format
/// - `.txt` → Try CoNLL first, then JSON
pub fn load_ner_dataset<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Vec<GoldEntity>)>> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "conll" | "conll2003" | "txt" => {
            // Try CoNLL format first
            load_conll_2003_dataset_internal(path).or_else(|_| {
                // If CoNLL fails, try JSON
                load_json_ner_dataset(path)
            })
        }
        "json" | "jsonl" => load_json_ner_dataset(path),
        _ => {
            // Try CoNLL first (most common), then JSON
            load_conll_2003_dataset_internal(path).or_else(|_| load_json_ner_dataset(path))
        }
    }
}

/// Internal CoNLL-2003 loader (used by datasets module).
fn load_conll_2003_dataset_internal<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<(String, Vec<GoldEntity>)>> {
    // Re-export the public function
    crate::eval::load_conll2003(path)
}

/// Dataset metadata for tracking dataset information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: String,
    pub format: String,
    pub language: Option<String>,
    pub entity_types: Vec<String>,
    pub num_examples: usize,
    pub source: Option<String>,
    pub year: Option<u32>,
}

/// Extract dataset metadata from loaded examples.
pub fn extract_dataset_metadata(
    examples: &[(String, Vec<GoldEntity>)],
    name: &str,
) -> DatasetMetadata {
    let mut entity_types = std::collections::HashSet::new();
    for (_, entities) in examples {
        for entity in entities {
            let type_str = crate::eval::entity_type_to_string(&entity.entity_type);
            entity_types.insert(type_str);
        }
    }

    DatasetMetadata {
        name: name.to_string(),
        format: "auto-detected".to_string(),
        language: None,
        entity_types: entity_types.into_iter().collect(),
        num_examples: examples.len(),
        source: None,
        year: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_json_ner_dataset() {
        let json_content = r#"[
            {
                "text": "John Smith works at Acme Corp.",
                "entities": [
                    {"text": "John Smith", "label": "PER", "start": 0, "end": 10},
                    {"text": "Acme Corp", "label": "ORG", "start": 20, "end": 29}
                ]
            }
        ]"#;

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_ner.json");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(json_content.as_bytes()).unwrap();
        file.flush().unwrap();

        let result = load_json_ner_dataset(&file_path).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "John Smith works at Acme Corp.");
        assert_eq!(result[0].1.len(), 2);

        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_load_jsonl_ner_dataset() {
        let jsonl_content = r#"{"text": "John Smith works.", "entities": [{"text": "John Smith", "label": "PER", "start": 0, "end": 10}]}
{"text": "Acme Corp is hiring.", "entities": [{"text": "Acme Corp", "label": "ORG", "start": 0, "end": 9}]}
"#;

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_ner.jsonl");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(jsonl_content.as_bytes()).unwrap();
        file.flush().unwrap();

        let result = load_json_ner_dataset(&file_path).unwrap();
        assert_eq!(result.len(), 2);

        std::fs::remove_file(&file_path).ok();
    }

    #[test]
    fn test_map_label_to_entity_type() {
        assert!(matches!(
            map_label_to_entity_type("PER"),
            EntityType::Person
        ));
        assert!(matches!(
            map_label_to_entity_type("ORG"),
            EntityType::Organization
        ));
        assert!(matches!(
            map_label_to_entity_type("LOC"),
            EntityType::Location
        ));
        assert!(matches!(
            map_label_to_entity_type("MISC"),
            EntityType::Other(_)
        ));
        assert!(matches!(
            map_label_to_entity_type("ANIM"),
            EntityType::Other(_)
        ));
    }

    #[test]
    fn test_load_ner_dataset_auto_detect() {
        // Test JSON detection
        let json_content = r#"[{"text": "Test", "entities": []}]"#;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_auto.json");
        let mut file = File::create(&file_path).unwrap();
        file.write_all(json_content.as_bytes()).unwrap();
        file.flush().unwrap();

        let result = load_ner_dataset(&file_path);
        assert!(result.is_ok());

        std::fs::remove_file(&file_path).ok();
    }
}
