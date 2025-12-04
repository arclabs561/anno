//! UniversalNER: LLM-based Zero-Shot NER
//!
//! UniversalNER uses instruction-tuned LLMs (LLaMA-based) for open NER,
//! supporting 45+ entity types without retraining.
//!
//! # Architecture
//!
//! UniversalNER is fundamentally different from transformer-based NER:
//! - **LLM-based**: Uses large language models (LLaMA) with instruction tuning
//! - **Prompt-based**: Extracts entities via natural language prompts
//! - **Very flexible**: Supports any entity type via prompt engineering
//! - **Expensive**: Slower and more costly than transformer models
//!
//! # Research
//!
//! - **Paper**: [UniversalNER](https://universal-ner.github.io)
//! - **Performance**: Competitive with ChatGPT on NER tasks
//! - **Capabilities**: 45 entity types, unlimited via prompts
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::universal_ner::UniversalNER;
//!
//! let model = UniversalNER::new()?;
//! let entities = model.extract_entities(
//!     "Steve Jobs founded Apple in 1976.",
//!     &["person", "organization", "date"]
//! )?;
//! ```
//!
//! # Implementation Status
//!
//! This is a placeholder implementation. Full UniversalNER requires:
//! - LLM inference infrastructure (e.g., llama.cpp, vLLM)
//! - Instruction-tuned LLaMA models
//! - Prompt engineering for entity extraction
//!
//! For now, this backend provides the interface but uses heuristics as a fallback.

use crate::backends::inference::ZeroShotNER;
use crate::{Entity, EntityType, Model, Result};

/// UniversalNER backend for LLM-based zero-shot NER.
///
/// Currently a placeholder implementation. Full LLM integration pending.
pub struct UniversalNER {
    /// Whether LLM backend is available
    llm_available: bool,
}

impl UniversalNER {
    /// Create a new UniversalNER instance.
    ///
    /// # Note
    /// Currently returns a placeholder. Full LLM integration will be added later.
    pub fn new() -> Result<Self> {
        // Placeholder: Check for LLM availability (not yet implemented)
        Ok(Self {
            llm_available: false, // LLM backend not yet implemented
        })
    }

    /// Extract entities using LLM-based prompt engineering.
    ///
    /// This is a placeholder that will use LLM inference when available.
    #[allow(unused_variables)]
    fn extract_with_llm(&self, text: &str, _entity_types: &[&str]) -> Result<Vec<Entity>> {
        // Placeholder: Use simple heuristics for now
        // Full implementation would:
        // 1. Construct prompt: "Extract entities of type [types] from: [text]"
        // 2. Call LLM (llama.cpp, vLLM, etc.)
        // 3. Parse JSON response with entities
        // 4. Return structured entities

        // For now, return empty (LLM not available)
        Ok(vec![])
    }
}

impl Model for UniversalNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if !self.llm_available {
            // Return empty instead of error to allow evaluation to skip gracefully
            // Evaluation system will mark as "skipped" based on is_available()
            return Ok(vec![]);
        }

        // Placeholder: would call LLM here
        self.extract_with_llm(text, &["person", "organization", "location"])
    }

    fn supported_types(&self) -> Vec<EntityType> {
        // UniversalNER supports any entity type (zero-shot via LLM)
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
        ]
    }

    fn is_available(&self) -> bool {
        self.llm_available
    }

    fn name(&self) -> &'static str {
        "universal_ner"
    }

    fn description(&self) -> &'static str {
        "UniversalNER LLM-based zero-shot NER (placeholder - LLM integration pending)"
    }
}

impl ZeroShotNER for UniversalNER {
    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        _threshold: f32,
    ) -> Result<Vec<Entity>> {
        if !self.llm_available {
            // Return empty instead of error to allow evaluation to skip gracefully
            return Ok(vec![]);
        }
        // Placeholder: would call LLM here
        self.extract_with_llm(text, entity_types)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        _threshold: f32,
    ) -> Result<Vec<Entity>> {
        // For UniversalNER, descriptions are treated as entity types
        // (LLM can handle natural language descriptions)
        self.extract_with_types(text, descriptions, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_ner_creation() {
        let model = UniversalNER::new().unwrap();
        assert!(!model.is_available()); // LLM not yet implemented
        assert_eq!(model.name(), "universal_ner");
    }

    #[test]
    fn test_universal_ner_error_when_unavailable() {
        let model = UniversalNER::new().unwrap();
        // When LLM is unavailable, extract_entities returns empty vec (not error)
        // to allow evaluation system to skip gracefully based on is_available()
        assert!(!model.is_available());
        let result = model.extract_entities("Test text", None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
