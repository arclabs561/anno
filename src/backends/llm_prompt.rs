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
//! ```rust
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
//!     .with_demonstrations(vec![
//!         ("Steve Jobs founded Apple.", vec![
//!             ("Steve Jobs", "PER", 0, 10),
//!             ("Apple", "ORG", 19, 24),
//!         ]),
//!     ])
//!     .with_chain_of_thought(true);
//!
//! let rendered = prompt.render("Elon Musk leads Tesla.");
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
                EntityType::Other(_) => "Miscellaneous named entities",
                EntityType::Custom { name, .. } => name.as_str(),
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
    pub fn with_description(mut self, entity_type: EntityType, description: &str) -> Self {
        self.descriptions.insert(entity_type, description.to_string());
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
        lines.push("    Returns: List of entities with text, type, start, end positions.".to_string());
        lines.push("    \"\"\"".to_string());

        lines.join("\n")
    }
}

/// Demonstration example for few-shot prompting.
#[derive(Debug, Clone)]
pub struct Demonstration {
    /// Input text
    pub text: String,
    /// Extracted entities: (text, type_label, start, end)
    pub entities: Vec<(String, String, usize, usize)>,
}

impl Demonstration {
    /// Create a new demonstration.
    #[must_use]
    pub fn new(text: &str, entities: Vec<(&str, &str, usize, usize)>) -> Self {
        Self {
            text: text.to_string(),
            entities: entities
                .into_iter()
                .map(|(t, ty, s, e)| (t.to_string(), ty.to_string(), s, e))
                .collect(),
        }
    }

    /// Render as JSON output.
    fn render_output(&self) -> String {
        if self.entities.is_empty() {
            return "[]".to_string();
        }

        let items: Vec<String> = self
            .entities
            .iter()
            .map(|(text, ty, start, end)| {
                format!(
                    r#"    {{"text": "{}", "type": "{}", "start": {}, "end": {}}}"#,
                    text, ty, start, end
                )
            })
            .collect();

        format!("[\n{}\n]", items.join(",\n"))
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
    /// Few-shot demonstrations
    demonstrations: Vec<Demonstration>,
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
            demonstrations: vec![],
            use_cot: false,
            system_prefix: None,
        }
    }

    /// Add few-shot demonstrations.
    #[must_use]
    pub fn with_demonstrations(
        mut self,
        demos: Vec<(&str, Vec<(&str, &str, usize, usize)>)>,
    ) -> Self {
        self.demonstrations = demos
            .into_iter()
            .map(|(text, entities)| Demonstration::new(text, entities))
            .collect();
        self
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
        let mut parts = vec![];

        // Function signature with schema
        parts.push("```python".to_string());
        parts.push("def extract_entities(text: str) -> list[dict]:".to_string());
        parts.push(self.schema.render_docstring());
        parts.push("    pass".to_string());
        parts.push("```".to_string());
        parts.push("".to_string());

        // Demonstrations
        if !self.demonstrations.is_empty() {
            parts.push("# Examples:".to_string());
            for (i, demo) in self.demonstrations.iter().enumerate() {
                parts.push(format!("\n## Example {}:", i + 1));
                parts.push(format!("Input: \"{}\"", demo.text));
                parts.push(format!("Output: {}", demo.render_output()));
            }
            parts.push("".to_string());
        }

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

    /// Get the expected JSON output format description.
    #[must_use]
    pub fn output_format(&self) -> &'static str {
        r#"[{"text": "entity_text", "type": "TYPE", "start": 0, "end": 10}, ...]"#
    }
}

/// Parse LLM response into entities.
///
/// Attempts to extract a JSON array of entities from the LLM output,
/// handling common formatting issues.
pub fn parse_llm_response(response: &str) -> Result<Vec<ParsedEntity>, ParseError> {
    // Try to find JSON array in response
    let json_str = extract_json_array(response)?;

    // Parse JSON
    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&json_str).map_err(|e| ParseError::InvalidJson(e.to_string()))?;

    // Convert to entities
    let mut entities = Vec::new();
    for (i, item) in parsed.iter().enumerate() {
        let text = item
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ParseError::MissingField(i, "text"))?
            .to_string();

        let entity_type = item
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ParseError::MissingField(i, "type"))?
            .to_string();

        let start = item
            .get("start")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ParseError::MissingField(i, "start"))? as usize;

        let end = item
            .get("end")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ParseError::MissingField(i, "end"))? as usize;

        let confidence = item.get("confidence").and_then(|v| v.as_f64());

        entities.push(ParsedEntity {
            text,
            entity_type,
            start,
            end,
            confidence,
        });
    }

    Ok(entities)
}

/// Extract JSON array from potentially messy LLM output.
fn extract_json_array(text: &str) -> Result<String, ParseError> {
    // Try direct parse first
    if text.trim().starts_with('[') {
        if let Some(end) = text.rfind(']') {
            let start = text.find('[').unwrap();
            return Ok(text[start..=end].to_string());
        }
    }

    // Look for ```json block
    if let Some(start) = text.find("```json") {
        let start = start + 7;
        if let Some(end) = text[start..].find("```") {
            let json = text[start..start + end].trim();
            if json.starts_with('[') {
                return Ok(json.to_string());
            }
        }
    }

    // Look for any [ ] pair
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                return Ok(text[start..=end].to_string());
            }
        }
    }

    Err(ParseError::NoJsonFound)
}

/// Parsed entity from LLM response.
#[derive(Debug, Clone)]
pub struct ParsedEntity {
    /// Entity text
    pub text: String,
    /// Entity type label
    pub entity_type: String,
    /// Start position in input
    pub start: usize,
    /// End position in input
    pub end: usize,
    /// Optional confidence score
    pub confidence: Option<f64>,
}

impl ParsedEntity {
    /// Convert to `Entity` with the given entity type mapping.
    pub fn to_entity(&self, type_map: &HashMap<String, EntityType>) -> Option<crate::Entity> {
        let entity_type = type_map
            .get(&self.entity_type)
            .or_else(|| type_map.get(&self.entity_type.to_uppercase()))
            .cloned()?;

        Some(crate::Entity::new(
            &self.text,
            entity_type,
            self.start,
            self.end,
            self.confidence.unwrap_or(0.8),
        ))
    }
}

/// Error during LLM response parsing.
#[derive(Debug)]
pub enum ParseError {
    /// No JSON array found in response
    NoJsonFound,
    /// Invalid JSON syntax
    InvalidJson(String),
    /// Missing required field
    MissingField(usize, &'static str),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoJsonFound => write!(f, "No JSON array found in LLM response"),
            Self::InvalidJson(e) => write!(f, "Invalid JSON: {}", e),
            Self::MissingField(i, field) => {
                write!(f, "Entity {} missing required field: {}", i, field)
            }
        }
    }
}

impl std::error::Error for ParseError {}

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
    fn test_prompt_with_demonstrations() {
        let schema = BIOSchema::new(&[EntityType::Person]);
        let prompt = CodeNERPrompt::new(schema).with_demonstrations(vec![(
            "Steve Jobs worked at Apple.",
            vec![("Steve Jobs", "PER", 0, 10)],
        )]);

        let rendered = prompt.render("Test input.");

        assert!(rendered.contains("Example 1"));
        assert!(rendered.contains("Steve Jobs"));
    }

    #[test]
    fn test_parse_clean_json() {
        let response = r#"[{"text": "John", "type": "PER", "start": 0, "end": 4}]"#;
        let entities = parse_llm_response(response).unwrap();

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "John");
        assert_eq!(entities[0].entity_type, "PER");
    }

    #[test]
    fn test_parse_json_with_markdown() {
        let response = r#"
Here are the entities:

```json
[{"text": "Paris", "type": "LOC", "start": 10, "end": 15}]
```

That's all!
"#;
        let entities = parse_llm_response(response).unwrap();

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "Paris");
    }

    #[test]
    fn test_parse_empty_response() {
        let response = "[]";
        let entities = parse_llm_response(response).unwrap();

        assert!(entities.is_empty());
    }

    #[test]
    fn test_parse_no_json() {
        let response = "I couldn't find any entities.";
        let result = parse_llm_response(response);

        assert!(matches!(result, Err(ParseError::NoJsonFound)));
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

