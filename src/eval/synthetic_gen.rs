//! Auto-generated synthetic test data.
//!
//! Unlike `synthetic.rs` which requires hand-coded offsets, this module
//! generates test data from templates with automatically computed offsets.
//!
//! # Example
//!
//! ```rust
//! use anno::eval::synthetic_gen::{Template, generate_test_cases};
//!
//! let templates = vec![
//!     Template::new("Meeting on {DATE} at {TIME}"),
//!     Template::new("Contact: {EMAIL}"),
//! ];
//!
//! let cases = generate_test_cases(&templates);
//! // Offsets are computed automatically from template positions
//! ```

use crate::eval::report::{SimpleGoldEntity, TestCase};
use std::collections::HashMap;

/// A template for generating test cases with automatic offset computation.
#[derive(Debug, Clone)]
pub struct Template {
    /// Template string with {TYPE} placeholders
    pattern: String,
    /// Custom entity values per type (optional)
    custom_values: HashMap<String, Vec<String>>,
}

impl Template {
    /// Create a new template from a pattern string.
    ///
    /// Placeholders are written as {TYPE}, e.g., {DATE}, {EMAIL}, {PERSON}.
    pub fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            custom_values: HashMap::new(),
        }
    }

    /// Add custom values for a specific entity type.
    pub fn with_values(mut self, entity_type: &str, values: Vec<&str>) -> Self {
        self.custom_values.insert(
            entity_type.to_string(),
            values.into_iter().map(String::from).collect(),
        );
        self
    }
}

/// Default entity values by type.
fn default_values(entity_type: &str) -> Vec<String> {
    match entity_type {
        "DATE" => vec![
            "January 15, 2024".into(),
            "March 3rd".into(),
            "2024-01-01".into(),
            "December 25".into(),
        ],
        "TIME" => vec![
            "3:00 PM".into(),
            "14:30".into(),
            "noon".into(),
            "9 AM".into(),
        ],
        "EMAIL" => vec![
            "user@example.com".into(),
            "test.email@domain.org".into(),
            "hello@world.io".into(),
        ],
        "MONEY" | "CURRENCY" => vec![
            "$1,234.56".into(),
            "€500".into(),
            "$99.99".into(),
            "£1,000".into(),
        ],
        "PHONE" => vec![
            "555-1234".into(),
            "(555) 123-4567".into(),
            "+1-800-555-0123".into(),
        ],
        "URL" => vec![
            "https://example.com".into(),
            "http://test.org/page".into(),
            "www.domain.io".into(),
        ],
        "PERSON" | "PER" => vec![
            "John Smith".into(),
            "María García".into(),
            "李明".into(),
            "Dr. Jane Doe".into(),
        ],
        "ORGANIZATION" | "ORG" => vec![
            "Google".into(),
            "Microsoft Corporation".into(),
            "United Nations".into(),
        ],
        "LOCATION" | "LOC" | "GPE" => vec![
            "New York".into(),
            "Tokyo".into(),
            "London, UK".into(),
        ],
        _ => vec![format!("[{}]", entity_type)],
    }
}

/// Generate test cases from templates.
///
/// For each template, generates one test case per entity value combination.
/// Offsets are computed automatically from the template structure.
pub fn generate_test_cases(templates: &[Template]) -> Vec<TestCase> {
    let mut cases = Vec::new();

    for template in templates {
        // Parse placeholders
        let placeholders = parse_placeholders(&template.pattern);
        if placeholders.is_empty() {
            // No placeholders, just create a case with no entities
            cases.push(TestCase {
                text: template.pattern.clone(),
                gold_entities: vec![],
            });
            continue;
        }

        // Get values for each placeholder type
        let mut type_values: Vec<(&str, Vec<String>)> = Vec::new();
        for (entity_type, _, _) in &placeholders {
            let values = template
                .custom_values
                .get(*entity_type)
                .cloned()
                .unwrap_or_else(|| default_values(entity_type));
            type_values.push((entity_type, values));
        }

        // Generate cases (for simplicity, just use first value of each type)
        let mut text = template.pattern.clone();
        let mut entities = Vec::new();
        let mut offset_adjustment: i64 = 0;

        for ((entity_type, placeholder_start, placeholder_end), (_type, values)) in
            placeholders.iter().zip(type_values.iter())
        {
            if values.is_empty() {
                continue;
            }
            let value = &values[0];
            
            // Compute adjusted positions
            let adjusted_start = (*placeholder_start as i64 + offset_adjustment) as usize;
            let placeholder_len = placeholder_end - placeholder_start;
            let value_len = value.len();
            
            // Replace placeholder with value
            let before = &text[..adjusted_start];
            let after = &text[adjusted_start + placeholder_len..];
            text = format!("{}{}{}", before, value, after);
            
            // Record entity
            entities.push(SimpleGoldEntity {
                text: value.clone(),
                entity_type: entity_type.to_string(),
                start: adjusted_start,
                end: adjusted_start + value_len,
            });
            
            // Update offset for next placeholder
            offset_adjustment += value_len as i64 - placeholder_len as i64;
        }

        cases.push(TestCase {
            text,
            gold_entities: entities,
        });
    }

    cases
}

/// Parse {TYPE} placeholders from a template string.
/// Returns (type, start, end) for each placeholder.
fn parse_placeholders(pattern: &str) -> Vec<(&str, usize, usize)> {
    let mut results = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = pattern.chars().collect();
    let _bytes = pattern.as_bytes(); // Reserved for potential byte-level parsing
    
    while i < chars.len() {
        if chars[i] == '{' {
            // Find closing brace
            let start_byte = pattern.char_indices().nth(i).map(|(b, _)| b).unwrap_or(0);
            let mut j = i + 1;
            while j < chars.len() && chars[j] != '}' {
                j += 1;
            }
            if j < chars.len() {
                let end_byte = pattern.char_indices().nth(j + 1).map(|(b, _)| b).unwrap_or(pattern.len());
                let type_start = start_byte + 1;
                let type_end = pattern.char_indices().nth(j).map(|(b, _)| b).unwrap_or(pattern.len());
                let entity_type = &pattern[type_start..type_end];
                results.push((entity_type, start_byte, end_byte));
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }

    results
}

/// Generate a comprehensive test set from built-in templates.
pub fn standard_test_set() -> Vec<TestCase> {
    let templates = vec![
        // Date/Time patterns
        Template::new("Meeting scheduled for {DATE} at {TIME}"),
        Template::new("Deadline: {DATE}"),
        Template::new("Call at {TIME}"),
        
        // Contact patterns
        Template::new("Email: {EMAIL}"),
        Template::new("Contact {EMAIL} for more info"),
        Template::new("Phone: {PHONE}"),
        
        // Financial patterns
        Template::new("Total: {MONEY}"),
        Template::new("Budget approved for {MONEY}"),
        Template::new("Invoice amount: {MONEY} due {DATE}"),
        
        // URL patterns
        Template::new("Visit {URL} for details"),
        Template::new("Link: {URL}"),
        
        // Named entities (for ML models)
        Template::new("{PERSON} works at {ORG}"),
        Template::new("CEO of {ORG}"),
        Template::new("Located in {LOC}"),
    ];

    generate_test_cases(&templates)
}

/// Generate test cases targeting specific entity types.
pub fn test_set_for_types(types: &[&str]) -> Vec<TestCase> {
    let mut templates = Vec::new();
    
    for entity_type in types {
        let pattern = format!("Test {{{}}}", entity_type);
        templates.push(Template::new(&pattern));
    }
    
    generate_test_cases(&templates)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_placeholders() {
        let placeholders = parse_placeholders("Meeting on {DATE} at {TIME}");
        assert_eq!(placeholders.len(), 2);
        assert_eq!(placeholders[0].0, "DATE");
        assert_eq!(placeholders[1].0, "TIME");
    }

    #[test]
    fn test_generate_simple_case() {
        let templates = vec![Template::new("Email: {EMAIL}")];
        let cases = generate_test_cases(&templates);
        
        assert_eq!(cases.len(), 1);
        assert!(cases[0].text.contains("@"));
        assert_eq!(cases[0].gold_entities.len(), 1);
        assert_eq!(cases[0].gold_entities[0].entity_type, "EMAIL");
    }

    #[test]
    fn test_offset_computation() {
        let templates = vec![Template::new("Date: {DATE}")];
        let cases = generate_test_cases(&templates);
        
        let case = &cases[0];
        let entity = &case.gold_entities[0];
        
        // Verify the offset is correct
        let extracted = &case.text[entity.start..entity.end];
        assert_eq!(extracted, entity.text);
    }

    #[test]
    fn test_multiple_placeholders() {
        let templates = vec![Template::new("{DATE} at {TIME}")];
        let cases = generate_test_cases(&templates);
        
        let case = &cases[0];
        assert_eq!(case.gold_entities.len(), 2);
        
        // Verify both offsets are correct
        for entity in &case.gold_entities {
            let extracted = &case.text[entity.start..entity.end];
            assert_eq!(extracted, entity.text, "Offset mismatch for {}", entity.entity_type);
        }
    }

    #[test]
    fn test_standard_test_set() {
        let cases = standard_test_set();
        assert!(!cases.is_empty());
        
        // Verify all offsets are valid
        for case in &cases {
            for entity in &case.gold_entities {
                assert!(entity.end <= case.text.len(), 
                    "Entity end {} exceeds text length {} in '{}'",
                    entity.end, case.text.len(), case.text);
                let extracted = &case.text[entity.start..entity.end];
                assert_eq!(extracted, entity.text,
                    "Offset mismatch: expected '{}', got '{}' in '{}'",
                    entity.text, extracted, case.text);
            }
        }
    }

    #[test]
    fn test_custom_values() {
        let template = Template::new("Name: {PERSON}")
            .with_values("PERSON", vec!["Alice", "Bob"]);
        
        let cases = generate_test_cases(&[template]);
        assert!(cases[0].text.contains("Alice")); // Uses first value
    }
}

