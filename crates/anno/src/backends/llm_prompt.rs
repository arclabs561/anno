//! Code-based prompt generation for LLM NER.
//!
//! Implements CodeNER-style prompting (arXiv:2507.20423) that frames NER
//! as a coding task, exploiting LLMs' superior code understanding.
//!
//! # Key Insight
//!
//! LLMs trained on code understand:
//! - Structured scope boundaries (like entity spans)
//! - Type annotations (like entity types)
//! - Sequential processing (like BIO tagging)
//!
//! By embedding NER instructions as code, we get better results than
//! natural language prompts.
//!
//! # Example
//!
//! ```ignore
//! use anno::backends::llm_prompt::{CodeNERPrompt, BIOSchema};
//! use anno::EntityType;
//!
//! let schema = BIOSchema::new(&[
//!     EntityType::Person,
//!     EntityType::Organization,
//!     EntityType::Location,
//! ]);
//!
//! let prompt = CodeNERPrompt::new(schema)
//!     .with_chain_of_thought(true);
//!
//! let rendered = prompt.render("Lynn Conway worked at IBM.");
//! // Send `rendered` to your LLM API
//! ```
//!
//! # References
//!
//! - CodeNER: Code Prompting for Named Entity Recognition (arXiv:2507.20423)

use crate::EntityType;
use std::collections::HashMap;

/// BIO tagging schema for NER.
///
/// Defines the entity types and their descriptions for prompting.
#[derive(Debug, Clone)]
pub struct BIOSchema {
    /// Entity types to extract
    pub entity_types: Vec<EntityType>,
    /// Human-readable descriptions for each type
    pub descriptions: HashMap<EntityType, String>,
}

impl BIOSchema {
    /// Create a new BIO schema with default descriptions.
    #[must_use]
    pub fn new(entity_types: &[EntityType]) -> Self {
        let mut descriptions = HashMap::new();

        for et in entity_types {
            // `EntityType` carries `#[non_exhaustive]`. Within the defining crate
            // (`anno`) the attribute has no effect on match exhaustiveness, so the
            // wildcard arm is unreachable here -- but it preserves resilience for
            // any future variant added in a single edit, and external callers
            // (other crates matching on `anno::EntityType`) still need the
            // wildcard pattern.
            #[allow(unreachable_patterns)]
            let desc = match et {
                EntityType::Person => "Person names (individuals, fictional characters)",
                EntityType::Organization => "Organizations (companies, institutions, groups)",
                EntityType::Location => "Locations (cities, countries, addresses, landmarks)",
                EntityType::Date => "Temporal expressions (dates, times, durations)",
                EntityType::Time => "Time expressions (clock times, periods)",
                EntityType::Money => "Monetary values (prices, amounts, currencies)",
                EntityType::Percent => "Percentage values",
                EntityType::Email => "Email addresses",
                EntityType::Phone => "Phone numbers",
                EntityType::Url => "Web URLs",
                EntityType::Quantity => "Quantities (measurements, counts)",
                EntityType::Cardinal => "Cardinal numbers",
                EntityType::Ordinal => "Ordinal numbers (1st, 2nd, etc.)",
                EntityType::Custom { name, .. } => name.as_str(),
                _ => "Named entities",
            };
            descriptions.insert(et.clone(), desc.to_string());
        }

        Self {
            entity_types: entity_types.to_vec(),
            descriptions,
        }
    }

    /// Set a custom description for an entity type.
    #[must_use]
    #[allow(dead_code)]
    pub fn with_description(mut self, entity_type: EntityType, description: &str) -> Self {
        self.descriptions
            .insert(entity_type, description.to_string());
        self
    }

    /// Render the schema as a code docstring.
    fn render_docstring(&self) -> String {
        let mut lines = vec![
            "    \"\"\"".to_string(),
            "    Extract named entities from text using BIO tagging.".to_string(),
            "    ".to_string(),
            "    BIO Schema:".to_string(),
            "    - B-{TYPE}: Beginning of entity of TYPE".to_string(),
            "    - I-{TYPE}: Inside (continuation) of entity".to_string(),
            "    - O: Outside any entity".to_string(),
            "    ".to_string(),
            "    Entity Types:".to_string(),
        ];

        for et in &self.entity_types {
            let label = et.as_label();
            let desc = self.descriptions.get(et).map_or("", |s| s.as_str());
            lines.push(format!("    - {}: {}", label, desc));
        }

        lines.push("    ".to_string());
        lines.push(
            "    Returns: List of entities with text, type, start, end positions.".to_string(),
        );
        lines.push("    \"\"\"".to_string());

        lines.join("\n")
    }
}

/// Code-based NER prompt generator.
///
/// Implements CodeNER-style prompting where NER is framed as a
/// coding task with BIO schema instructions.
#[derive(Debug, Clone)]
pub struct CodeNERPrompt {
    /// BIO schema definition
    schema: BIOSchema,
    /// Enable chain-of-thought reasoning
    use_cot: bool,
    /// System message prefix
    system_prefix: Option<String>,
}

impl CodeNERPrompt {
    /// Create a new code NER prompt with the given schema.
    #[must_use]
    pub fn new(schema: BIOSchema) -> Self {
        Self {
            schema,
            use_cot: false,
            system_prefix: None,
        }
    }

    /// Enable chain-of-thought reasoning.
    #[must_use]
    pub fn with_chain_of_thought(mut self, enabled: bool) -> Self {
        self.use_cot = enabled;
        self
    }

    /// Set a custom system message prefix.
    #[must_use]
    pub fn with_system_prefix(mut self, prefix: &str) -> Self {
        self.system_prefix = Some(prefix.to_string());
        self
    }

    /// Render the system message.
    #[must_use]
    pub fn render_system(&self) -> String {
        let prefix = self.system_prefix.as_deref().unwrap_or(
            "You are an expert NER system. Extract entities precisely using BIO tagging.",
        );

        format!(
            "{}\n\nRespond ONLY with valid JSON array of entities. No explanation.",
            prefix
        )
    }

    /// Render the user prompt for the given input text.
    #[must_use]
    pub fn render(&self, input_text: &str) -> String {
        // Function signature with schema
        let mut parts = vec![
            "```python".to_string(),
            "def extract_entities(text: str) -> list[dict]:".to_string(),
            self.schema.render_docstring(),
            "    pass".to_string(),
            "```".to_string(),
            String::new(),
        ];

        // Chain-of-thought instruction
        if self.use_cot {
            parts.push("# Instructions:".to_string());
            parts.push("1. First, identify potential entity spans in the text".to_string());
            parts.push("2. For each span, determine the most appropriate entity type".to_string());
            parts.push("3. Verify the start and end positions are correct".to_string());
            parts.push("4. Return the final JSON array".to_string());
            parts.push("".to_string());
        }

        // Input
        parts.push("# Task:".to_string());
        parts.push(format!("Input: \"{}\"", input_text));
        parts.push("Output:".to_string());

        parts.join("\n")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bio_schema_creation() {
        let schema = BIOSchema::new(&[EntityType::Person, EntityType::Organization]);

        assert_eq!(schema.entity_types.len(), 2);
        assert!(schema.descriptions.contains_key(&EntityType::Person));
    }

    #[test]
    fn test_prompt_rendering() {
        let schema = BIOSchema::new(&[EntityType::Person, EntityType::Location]);
        let prompt = CodeNERPrompt::new(schema);

        let rendered = prompt.render("John went to Paris.");

        assert!(rendered.contains("extract_entities"));
        assert!(rendered.contains("BIO Schema"));
        assert!(rendered.contains("PER"));
        assert!(rendered.contains("LOC"));
        assert!(rendered.contains("John went to Paris"));
    }

    #[test]
    fn test_chain_of_thought() {
        let schema = BIOSchema::new(&[EntityType::Person]);
        let prompt = CodeNERPrompt::new(schema).with_chain_of_thought(true);

        let rendered = prompt.render("Test.");

        assert!(rendered.contains("Instructions"));
        assert!(rendered.contains("identify potential entity spans"));
    }
}
