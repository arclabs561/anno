//! Validation utilities for NER evaluation.
//!
//! Provides validation for:
//! - Ground truth entity spans (bounds checking, non-overlapping)
//! - Entity type consistency
//! - Text bounds validation
//! - Overlapping entity detection

use super::datasets::GoldEntity;
use crate::EntityType;
use crate::{Error, Result};

/// Validation result for ground truth entities.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors found
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a new validation result.
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error.
    pub fn add_error(&mut self, error: String) {
        self.is_valid = false;
        self.errors.push(error);
    }

    /// Add a warning.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Convert to Result, returning error if validation failed.
    pub fn into_result(self) -> Result<()> {
        if self.is_valid {
            Ok(())
        } else {
            Err(Error::InvalidInput(format!(
                "Validation failed: {}",
                self.errors.join("; ")
            )))
        }
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate ground truth entities against text.
///
/// Checks:
/// - Entity spans are within text bounds
/// - Entity spans are non-empty (start < end)
/// - Entity text matches the span in the text
/// - Entities don't overlap (optional, can be warning)
///
/// # Arguments
/// * `text` - The text being annotated
/// * `entities` - Ground truth entities to validate
/// * `strict` - If true, overlapping entities are errors; if false, warnings
///
/// # Returns
/// Validation result with errors and warnings
pub fn validate_ground_truth_entities(
    text: &str,
    entities: &[GoldEntity],
    strict: bool,
) -> ValidationResult {
    let mut result = ValidationResult::new();
    // Use character count for Unicode correctness (matches GoldEntity offsets)
    let text_char_len = text.chars().count();

    // Check each entity
    for (i, entity) in entities.iter().enumerate() {
        // Check for whitespace-only entities
        if entity.text.trim().is_empty() {
            result.add_warning(format!(
                "Entity {}: text is empty or whitespace-only: '{}'",
                i, entity.text
            ));
        }

        // Check bounds (using character count, not byte count)
        if entity.start >= text_char_len {
            result.add_error(format!(
                "Entity {}: start position {} out of bounds (text length: {} chars)",
                i, entity.start, text_char_len
            ));
            continue;
        }

        if entity.end > text_char_len {
            result.add_error(format!(
                "Entity {}: end position {} out of bounds (text length: {} chars)",
                i, entity.end, text_char_len
            ));
            continue;
        }

        // Check non-empty span
        if entity.start >= entity.end {
            result.add_error(format!(
                "Entity {}: invalid span (start {} >= end {})",
                i, entity.start, entity.end
            ));
            continue;
        }

        // Check text matches span (using character offsets, not byte offsets)
        let span_text: String = text
            .chars()
            .skip(entity.start)
            .take(entity.end - entity.start)
            .collect();
        if span_text != entity.text {
            result.add_warning(format!(
                "Entity {}: text mismatch. Expected '{}', found '{}'",
                i, entity.text, span_text
            ));
        }
    }

    // Check for overlapping entities
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let e1 = &entities[i];
            let e2 = &entities[j];

            // Check if spans overlap
            let overlap = (e1.start < e2.end) && (e2.start < e1.end);
            if overlap {
                let msg = format!(
                    "Entities {} and {} overlap: [{}, {}) and [{}, {})",
                    i, j, e1.start, e1.end, e2.start, e2.end
                );
                if strict {
                    result.add_error(msg);
                } else {
                    result.add_warning(msg);
                }
            }
        }
    }

    result
}

/// Validate entity type consistency across test cases.
///
/// Checks that entity types are used consistently (e.g., same type name
/// refers to same EntityType variant).
pub fn validate_entity_type_consistency(
    test_cases: &[(String, Vec<GoldEntity>)],
) -> ValidationResult {
    let mut result = ValidationResult::new();
    let mut type_map: std::collections::HashMap<String, EntityType> =
        std::collections::HashMap::new();

    for (case_idx, (_text, entities)) in test_cases.iter().enumerate() {
        for entity in entities {
            let type_str = crate::eval::entity_type_to_string(&entity.entity_type);
            if let Some(existing_type) = type_map.get(&type_str) {
                // Check if types match
                if !crate::eval::entity_type_matches(existing_type, &entity.entity_type) {
                    result.add_warning(format!(
                        "Test case {}: Entity type '{}' inconsistent with previous usage",
                        case_idx, type_str
                    ));
                }
            } else {
                type_map.insert(type_str, entity.entity_type.clone());
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_bounds() {
        let text = "Hello world";
        let entities = vec![GoldEntity {
            text: "Hello".to_string(),
            entity_type: EntityType::Person,
            original_label: "PER".to_string(),
            start: 0,
            end: 5,
        }];

        let result = validate_ground_truth_entities(text, &entities, false);
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_out_of_bounds() {
        let text = "Hello";
        let entities = vec![GoldEntity {
            text: "world".to_string(),
            entity_type: EntityType::Person,
            original_label: "PER".to_string(),
            start: 10,
            end: 15,
        }];

        let result = validate_ground_truth_entities(text, &entities, false);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validate_overlapping() {
        let text = "Hello world";
        let entities = vec![
            GoldEntity {
                text: "Hello".to_string(),
                entity_type: EntityType::Person,
                original_label: "PER".to_string(),
                start: 0,
                end: 5,
            },
            GoldEntity {
                text: "lo wo".to_string(),
                entity_type: EntityType::Person,
                original_label: "PER".to_string(),
                start: 3,
                end: 8,
            },
        ];

        let result = validate_ground_truth_entities(text, &entities, false);
        assert!(result.is_valid); // Warnings don't fail validation
        assert!(!result.warnings.is_empty());

        let result_strict = validate_ground_truth_entities(text, &entities, true);
        assert!(!result_strict.is_valid); // Errors fail validation
    }
}
