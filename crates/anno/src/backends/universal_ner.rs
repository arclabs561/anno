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
//! This backend is LLM-backed and requires:
//! - A supported API provider (OpenAI / Anthropic / OpenRouter)
//! - An API key in the environment (loaded from `.env` if present)
//! - The `llm` feature for HTTP calls (`ureq`)
//!
//! Behavior is **explicit**:
//! - If unavailable, `extract_*` returns `FeatureNotAvailable` (no silent empty fallback).
//!
//! # Environment Variables
//!
//! Automatically loads from `.env` if present. Supported keys:
//! - `OPENAI_API_KEY` - OpenAI API
//! - `OPENROUTER_API_KEY` - OpenRouter API  
//! - `GEMINI_API_KEY` - Google Gemini API
//! - `ANTHROPIC_API_KEY` - Anthropic API
//! - `UNIVERSAL_NER_API_KEY` - Dedicated UniversalNER key

use crate::backends::inference::ZeroShotNER;
use crate::offset::TextSpan;
use crate::{Entity, EntityType, Model, Result};

/// UniversalNER backend for LLM-based zero-shot NER.
///
/// Automatically loads API keys from `.env` if present.
/// Returns explicit errors when unavailable - use `is_available()` to check.
pub struct UniversalNER {
    /// Whether LLM backend is available
    llm_available: bool,
}

impl UniversalNER {
    /// Create a new UniversalNER instance.
    ///
    /// Opportunistically loads `.env` file to check for API keys.
    /// Check `is_available()` before use. Returns an explicit error when unavailable.
    pub fn new() -> Result<Self> {
        // Load .env if present (idempotent)
        crate::env::load_dotenv();

        // LLM availability depends on:
        // - compile-time feature (`llm`) for HTTP support
        // - runtime configuration (API key)
        let universal_key = std::env::var("UNIVERSAL_NER_API_KEY")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
        let llm_available =
            cfg!(feature = "llm") && (crate::env::has_llm_api_key() || universal_key);

        Ok(Self { llm_available })
    }

    /// Extract entities using LLM-based prompt engineering.
    ///
    /// Calls OpenAI-compatible API with structured NER prompt.
    /// Requires `llm` feature for HTTP client (ureq).
    #[cfg(feature = "llm")]
    fn extract_with_llm(&self, text: &str, entity_types: &[&str]) -> Result<Vec<Entity>> {
        let (api_key, provider) = crate::env::llm_api_key().ok_or_else(|| {
            crate::Error::FeatureNotAvailable(
                "No LLM API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or similar.".into(),
            )
        })?;

        let types_str = entity_types.join(", ");
        let prompt = format!(
            r#"Extract named entities from the following text. Return ONLY a JSON array of objects with "text", "type", "start", "end" fields.

Entity types to extract: {types_str}

Text: "{text}"

Example output: [{{"text": "John Smith", "type": "person", "start": 0, "end": 10}}]

Return ONLY the JSON array, no other text:"#
        );

        let (url, model, auth_header) = match provider {
            "openai" => (
                "https://api.openai.com/v1/chat/completions",
                "gpt-4o-mini",
                format!("Bearer {}", api_key),
            ),
            "anthropic" => (
                "https://api.anthropic.com/v1/messages",
                "claude-3-haiku-20240307",
                api_key.clone(),
            ),
            "openrouter" => (
                "https://openrouter.ai/api/v1/chat/completions",
                "openai/gpt-4o-mini",
                format!("Bearer {}", api_key),
            ),
            other => {
                return Err(crate::Error::FeatureNotAvailable(format!(
                    "UniversalNER provider '{}' is not supported by this build. Supported: openai, anthropic, openrouter.",
                    other
                )));
            }
        };

        let response = if provider == "anthropic" {
            // Anthropic uses different API format
            let body = serde_json::json!({
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            });
            ureq::post(url)
                .set("x-api-key", &auth_header)
                .set("anthropic-version", "2023-06-01")
                .set("content-type", "application/json")
                .send_json(body)
        } else {
            // OpenAI-compatible format
            let body = serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            });
            ureq::post(url)
                .set("Authorization", &auth_header)
                .set("content-type", "application/json")
                .send_json(body)
        };

        let response =
            response.map_err(|e| crate::Error::Inference(format!("LLM API error: {}", e)))?;
        let json: serde_json::Value = response
            .into_json()
            .map_err(|e| crate::Error::Parse(format!("LLM response parse error: {}", e)))?;

        // Extract content from response
        let content = if provider == "anthropic" {
            json["content"][0]["text"].as_str().unwrap_or("[]")
        } else {
            json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("[]")
        };

        // Parse JSON array of entities
        self.parse_llm_response(content, text)
    }

    /// Fallback when eval-advanced feature is not enabled
    #[cfg(not(feature = "llm"))]
    fn extract_with_llm(&self, _text: &str, _entity_types: &[&str]) -> Result<Vec<Entity>> {
        Err(crate::Error::FeatureNotAvailable(
            "UniversalNER requires the 'llm' feature to make HTTP requests (ureq). Rebuild with --features llm (or eval-advanced) and provide an API key via .env."
                .into(),
        ))
    }

    /// Parse LLM response into entities.
    ///
    /// This is **pure** (no HTTP) and therefore always compiled so we can unit test it
    /// without network access.
    #[allow(dead_code)] // Used by `extract_with_llm` (when enabled) and unit tests.
    fn parse_llm_response(&self, content: &str, original_text: &str) -> Result<Vec<Entity>> {
        // Try to extract JSON array from response. Some providers wrap responses in
        // markdown/code fences or include extra explanation text.
        let json_str = content.trim();
        let json_str = json_str
            .strip_prefix("```json")
            .or_else(|| json_str.strip_prefix("```JSON"))
            .or_else(|| json_str.strip_prefix("```"))
            .unwrap_or(json_str)
            .trim();
        let json_str = json_str.strip_suffix("```").unwrap_or(json_str).trim();

        let json_str = if json_str.starts_with('[') {
            json_str.to_string()
        } else if let Some(start) = json_str.find('[') {
            if let Some(end) = json_str.rfind(']') {
                json_str[start..=end].to_string()
            } else {
                return Err(crate::Error::Parse(format!(
                    "UniversalNER LLM response did not contain a complete JSON array. Response begins: {:?}",
                    json_str.chars().take(200).collect::<String>()
                )));
            }
        } else {
            return Err(crate::Error::Parse(format!(
                "UniversalNER LLM response did not contain a JSON array. Response begins: {:?}",
                json_str.chars().take(200).collect::<String>()
            )));
        };

        let items: Vec<serde_json::Value> = serde_json::from_str(&json_str).map_err(|e| {
            crate::Error::Parse(format!(
                "UniversalNER failed to parse JSON array from LLM response: {}. Extracted JSON begins: {:?}",
                e,
                json_str.chars().take(200).collect::<String>()
            ))
        })?;

        let mut entities = Vec::new();
        for item in items {
            let text = item["text"].as_str().unwrap_or("");
            let type_str = item["type"].as_str().unwrap_or("misc");
            // Treat provided offsets as **character offsets** hints (LLMs are often wrong).
            let hint_start = item["start"].as_u64().unwrap_or(0) as usize;
            let hint_end = item["end"].as_u64().unwrap_or(0) as usize;

            if text.is_empty() || hint_end <= hint_start {
                continue;
            }

            // Prefer exact substring matches in the original text; choose the occurrence that
            // best matches the hint offsets. This avoids the "first occurrence" bug when the
            // same surface form appears multiple times.
            let mut occurrences: Vec<(usize, usize)> = Vec::new();
            for (start_byte, _) in original_text.match_indices(text) {
                let span = TextSpan::from_bytes(original_text, start_byte, start_byte + text.len());
                occurrences.push((span.char_start, span.char_end));
            }

            let (actual_start, actual_end) = if !occurrences.is_empty() {
                *occurrences
                    .iter()
                    .min_by_key(|(s, e)| {
                        let ds = (*s as isize - hint_start as isize).unsigned_abs();
                        let de = (*e as isize - hint_end as isize).unsigned_abs();
                        (ds + de, *s, *e)
                    })
                    .expect("non-empty occurrences")
            } else {
                // Fallback: accept hint offsets only if they round-trip to the claimed text.
                let char_count = original_text.chars().count();
                if hint_end <= char_count {
                    let extracted = TextSpan::from_chars(original_text, hint_start, hint_end)
                        .extract(original_text);
                    if extracted == text {
                        (hint_start, hint_end)
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            };

            let entity_type = match type_str.to_lowercase().as_str() {
                "person" | "per" => EntityType::Person,
                "organization" | "org" => EntityType::Organization,
                "location" | "loc" | "gpe" => EntityType::Location,
                "date" | "time" => EntityType::Date,
                "money" | "currency" => EntityType::Money,
                _ => EntityType::Other(type_str.to_string()),
            };

            let mut entity = Entity::new(
                text.to_string(),
                entity_type,
                actual_start,
                actual_end,
                0.9, // LLM-based, high confidence
            );
            entity.provenance = Some(crate::Provenance::ml("universal_ner", entity.confidence));
            entities.push(entity);
        }

        Ok(entities)
    }
}

impl Model for UniversalNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if !self.llm_available {
            return Err(crate::Error::FeatureNotAvailable(
                "UniversalNER requires an LLM API key. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, or UNIVERSAL_NER_API_KEY (loaded from .env if present)."
                    .into(),
            ));
        }

        self.extract_with_llm(text, &["person", "organization", "location"])
    }

    fn supported_types(&self) -> Vec<EntityType> {
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
        "UniversalNER: LLM-based zero-shot NER (requires `llm` feature + API key)"
    }
}

impl ZeroShotNER for UniversalNER {
    fn default_types(&self) -> &[&'static str] {
        &["person", "organization", "location"]
    }

    fn extract_with_types(
        &self,
        text: &str,
        entity_types: &[&str],
        _threshold: f32,
    ) -> Result<Vec<Entity>> {
        if !self.llm_available {
            return Err(crate::Error::FeatureNotAvailable(
                "UniversalNER requires an LLM API key. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, or UNIVERSAL_NER_API_KEY (loaded from .env if present)."
                    .into(),
            ));
        }
        self.extract_with_llm(text, entity_types)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        // For UniversalNER, descriptions are treated as entity types
        self.extract_with_types(text, descriptions, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn test_universal_ner_creation() {
        let model = UniversalNER::new().unwrap();
        assert_eq!(model.name(), "universal_ner");
    }

    #[test]
    fn test_universal_ner_availability_reflects_api_key() {
        // Env vars are global; serialize to avoid interference with other tests.
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Override any `.env` values (dotenv only sets if unset).
        for k in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "UNIVERSAL_NER_API_KEY",
        ] {
            std::env::set_var(k, "");
        }

        let model = UniversalNER::new().unwrap();
        assert!(
            !model.is_available(),
            "Empty keys must not count as available"
        );

        std::env::set_var("UNIVERSAL_NER_API_KEY", "dummy");
        let model2 = UniversalNER::new().unwrap();
        assert_eq!(model2.is_available(), cfg!(feature = "llm"));
    }

    #[test]
    fn test_universal_ner_errors_without_llm() {
        let model = UniversalNER::new().unwrap();
        if !model.is_available() {
            // Without LLM, should return explicit error (not silent empty).
            let result = model.extract_entities("Steve Jobs founded Apple", None);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_parse_llm_response_handles_code_fences_and_multiscript() {
        let model = UniversalNER::new().unwrap();
        let text = "李明 met Müller in الرياض. 😀";
        let response = r#"```json
[
  {"text":"李明","type":"person","start":0,"end":2},
  {"text":"Müller","type":"person","start":7,"end":13},
  {"text":"الرياض","type":"location","start":17,"end":23},
  {"text":"😀","type":"misc","start":25,"end":26}
]
```"#;
        let ents = model.parse_llm_response(response, text).expect("parse");
        assert!(!ents.is_empty());

        for e in ents {
            let extracted = TextSpan::from_chars(text, e.start, e.end).extract(text);
            assert_eq!(extracted, e.text, "entity span should round-trip");
        }
    }

    #[test]
    fn test_parse_llm_response_repeated_surface_form_uses_hint_offsets() {
        let model = UniversalNER::new().unwrap();
        let text = "Apple met Apple in Apple Park.";
        // Intentionally provide multiple occurrences with different hint offsets.
        let response = r#"[{"text":"Apple","type":"org","start":0,"end":5},{"text":"Apple","type":"org","start":10,"end":15},{"text":"Apple","type":"org","start":19,"end":24}]"#;
        let ents = model.parse_llm_response(response, text).expect("parse");

        let apples: Vec<_> = ents.into_iter().filter(|e| e.text == "Apple").collect();
        assert_eq!(apples.len(), 3);
        let mut starts: Vec<usize> = apples.iter().map(|e| e.start).collect();
        starts.sort_unstable();
        starts.dedup();
        assert_eq!(
            starts.len(),
            3,
            "each Apple should map to a distinct occurrence"
        );
    }
}
