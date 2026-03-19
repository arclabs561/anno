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
//! - A supported API provider (OpenRouter recommended, or Anthropic / Groq / Gemini / Ollama)
//! - An API key in the environment (loaded from `.env` if present), or a local Ollama instance
//! - The `llm` feature for HTTP calls (`ureq`)
//!
//! Behavior is **explicit**:
//! - If unavailable, `extract_*` returns `FeatureNotAvailable` (no silent empty fallback).
//!
//! # Environment Variables
//!
//! Automatically loads from `.env` if present. Supported keys (checked in order):
//! - `OPENROUTER_API_KEY` - OpenRouter API (recommended: unified gateway for all models)
//! - `GROQ_API_KEY` - Groq API (ultra-fast inference for open models)
//! - `ANTHROPIC_API_KEY` - Anthropic API
//! - `GEMINI_API_KEY` - Google Gemini API
//! - `OLLAMA_HOST` - Ollama server URL (default: `http://localhost:11434`; no key needed)
//! - `UNIVERSAL_NER_API_KEY` - Dedicated UniversalNER key

use std::collections::HashMap;

#[cfg(any(feature = "llm", test))]
use crate::sync::lock;
use crate::sync::Mutex;

#[cfg(feature = "llm")]
use crate::backends::chunking::{extract_chunked_parallel, ChunkConfig};
use crate::backends::inference::ZeroShotNER;
use crate::backends::llm_prompt::{BIOSchema, CodeNERPrompt};
use crate::offset::TextSpan;
use crate::{Entity, EntityType, Language, Model, Result};

/// Simple LRU-ish response cache keyed on (text_hash, types_hash, model).
/// Avoids duplicate API calls for the same input.
#[derive(Debug, Default)]
#[cfg_attr(not(feature = "llm"), allow(dead_code))]
struct ResponseCache {
    entries: HashMap<u64, Vec<Entity>>,
    /// Insertion order for eviction (oldest first).
    order: Vec<u64>,
    capacity: usize,
}

#[cfg_attr(not(feature = "llm"), allow(dead_code))]
impl ResponseCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            capacity,
        }
    }

    fn get(&self, key: u64) -> Option<&Vec<Entity>> {
        self.entries.get(&key)
    }

    fn insert(&mut self, key: u64, entities: Vec<Entity>) {
        if self.entries.len() >= self.capacity {
            // Evict oldest
            if let Some(oldest) = self.order.first().copied() {
                self.entries.remove(&oldest);
                self.order.remove(0);
            }
        }
        self.entries.insert(key, entities);
        self.order.push(key);
    }
}

/// Compute a simple hash for cache keys.
#[cfg(feature = "llm")]
fn cache_key(text: &str, types: &[&str], model: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    types.hash(&mut hasher);
    model.hash(&mut hasher);
    hasher.finish()
}

/// Prompting strategy for LLM-based NER.
#[derive(Debug, Clone, Default)]
pub enum PromptStrategy {
    /// Simple JSON extraction prompt (default, reliable across models).
    #[default]
    Simple,
    /// Compact output format using offset tuples and single-char type keys.
    /// Saves ~60% output tokens vs Simple: `[[0,12,"P"],[16,21,"L"]]` instead
    /// of verbose JSON objects with repeated field names and full text strings.
    Compact,
    /// CodeNER-style prompt that frames NER as a coding task with BIO schema.
    /// Can yield higher F1 on models with strong code understanding.
    CodeNER {
        /// Enable chain-of-thought reasoning (slower but sometimes more accurate).
        chain_of_thought: bool,
    },
}

/// UniversalNER backend for LLM-based zero-shot NER.
///
/// Automatically loads API keys from `.env` if present.
/// Returns explicit errors when unavailable - use `is_available()` to check.
pub struct UniversalNER {
    /// Whether LLM backend is available
    llm_available: bool,
    /// Optional LLM configuration (model, endpoint, etc.).
    /// Only read when `llm` feature is enabled (for HTTP calls).
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    config: Option<crate::backends::llm_client::LlmConfig>,
    /// Prompting strategy.
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    prompt_strategy: PromptStrategy,
    /// Optional domain context to improve NER quality.
    /// Per NER4all findings, domain context yields +5% recall over generic prompts.
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    domain_context: Option<String>,
    /// Response cache to avoid duplicate API calls.
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    cache: Mutex<ResponseCache>,
    /// Whether to run self-verification (re-query LLM to confirm entities).
    /// Per GPT-NER (NAACL 2025), reduces hallucination at the cost of 2x latency.
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    self_verify: bool,
    /// Maximum characters per LLM chunk. Texts exceeding this are split at
    /// sentence boundaries, processed in parallel, and entities coalesced.
    /// Default 4000 (safe for most LLM context windows with prompt overhead).
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    max_chunk_chars: usize,
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
        // - runtime configuration (API key or local Ollama)
        let universal_key = std::env::var("UNIVERSAL_NER_API_KEY")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
        let llm_available =
            cfg!(feature = "llm") && (crate::env::has_llm_api_key() || universal_key);

        Ok(Self {
            llm_available,
            config: None,
            prompt_strategy: PromptStrategy::default(),
            domain_context: None,
            cache: Mutex::new(ResponseCache::new(256)),
            self_verify: false,
            max_chunk_chars: 4000,
        })
    }

    /// Create with a specific LLM configuration.
    pub fn with_config(config: crate::backends::llm_client::LlmConfig) -> Result<Self> {
        crate::env::load_dotenv();
        let universal_key = std::env::var("UNIVERSAL_NER_API_KEY")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
        let llm_available =
            cfg!(feature = "llm") && (crate::env::has_llm_api_key() || universal_key);
        Ok(Self {
            llm_available,
            config: Some(config),
            prompt_strategy: PromptStrategy::default(),
            domain_context: None,
            cache: Mutex::new(ResponseCache::new(256)),
            self_verify: false,
            max_chunk_chars: 4000,
        })
    }

    /// Set the prompting strategy.
    pub fn prompt_strategy(mut self, strategy: PromptStrategy) -> Self {
        self.prompt_strategy = strategy;
        self
    }

    /// Set domain context to improve NER quality.
    ///
    /// Per NER4all (arXiv:2502.04351), providing specific domain context
    /// improves recall by ~5% over generic prompts in zero-shot settings.
    pub fn domain_context(mut self, context: &str) -> Self {
        self.domain_context = Some(context.to_string());
        self
    }

    /// Enable self-verification: re-query the LLM to confirm extracted entities.
    ///
    /// Per GPT-NER (NAACL 2025 Findings), this reduces hallucination where models
    /// over-confidently label null inputs as entities. Doubles latency/cost.
    pub fn self_verify(mut self, enabled: bool) -> Self {
        self.self_verify = enabled;
        self
    }

    /// Set max characters per chunk for large documents.
    ///
    /// Texts exceeding this limit are split at sentence boundaries, processed
    /// in parallel threads, and entities coalesced with overlap dedup.
    /// Default: 4000 chars (safe for most LLM context windows).
    pub fn max_chunk_chars(mut self, chars: usize) -> Self {
        self.max_chunk_chars = chars;
        self
    }

    /// Build system + user messages based on the current prompt strategy.
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    fn build_prompt(&self, text: &str, entity_types: &[&str]) -> (String, String) {
        match &self.prompt_strategy {
            PromptStrategy::Simple => {
                let types_str = entity_types.join(", ");
                let mut system = "You are a precise named entity recognition system. Extract entities and return ONLY a JSON array. No explanation.".to_string();

                if let Some(ctx) = &self.domain_context {
                    system.push_str("\n\n## Domain context\n");
                    system.push_str(ctx);
                }

                let user = format!(
                    r#"Extract named entities from the following text.

## Entity types to extract
{types_str}

## Output format
Return a JSON array of objects, each with "text", "type", "start" (character offset), "end" (character offset) fields.

## Text
"{text}"

## Example
[{{"text": "John Smith", "type": "person", "start": 0, "end": 10}}]

Return ONLY the JSON array:"#
                );
                (system, user)
            }
            PromptStrategy::Compact => {
                // Build type legend: single-char keys to minimize output tokens
                let legend: Vec<(String, &str)> = entity_types
                    .iter()
                    .map(|t| {
                        let key = t.chars().next().unwrap_or('X').to_uppercase().to_string();
                        (key, *t)
                    })
                    .collect();

                // Deduplicate keys by appending index if collision
                let mut seen = std::collections::HashSet::new();
                let legend: Vec<(String, &str)> = legend
                    .into_iter()
                    .enumerate()
                    .map(|(i, (mut k, t))| {
                        if !seen.insert(k.clone()) {
                            k = format!("{}{}", k, i);
                        }
                        (k, t)
                    })
                    .collect();

                let legend_str = legend
                    .iter()
                    .map(|(k, t)| format!("{}={}", k, t))
                    .collect::<Vec<_>>()
                    .join(", ");

                let mut system = "NER. Return JSON array of [start,end,key] tuples where start/end are 0-based character offsets (not byte or word). No text field, no explanation.".to_string();

                if let Some(ctx) = &self.domain_context {
                    system.push_str("\nDomain: ");
                    system.push_str(ctx);
                }

                // Show a worked example with exact character counting
                let user = format!(
                    r#"Legend: {legend_str}
Text: "{text}"
Example for "John met Ada in Rome.": [[0,4,"P"],[9,12,"P"],[16,20,"L"]]
Output:"#
                );
                (system, user)
            }
            PromptStrategy::CodeNER { chain_of_thought } => {
                let et_list: Vec<EntityType> = entity_types
                    .iter()
                    .map(|t| match t.to_lowercase().as_str() {
                        "person" | "per" => EntityType::Person,
                        "organization" | "org" => EntityType::Organization,
                        "location" | "loc" | "gpe" => EntityType::Location,
                        "date" | "time" => EntityType::Date,
                        "money" | "currency" => EntityType::Money,
                        other => EntityType::custom(other, anno_core::EntityCategory::Misc),
                    })
                    .collect();

                let schema = BIOSchema::new(&et_list);
                let mut prompt =
                    CodeNERPrompt::new(schema).with_chain_of_thought(*chain_of_thought);

                if let Some(ctx) = &self.domain_context {
                    prompt = prompt.with_system_prefix(&format!(
                        "You are an expert NER system. Extract entities precisely using BIO tagging.\n\n## Domain context\n{}",
                        ctx
                    ));
                }

                (prompt.render_system(), prompt.render(text))
            }
        }
    }

    /// Extract entities using LLM-based prompt engineering.
    ///
    /// Calls OpenAI-compatible API with structured NER prompt.
    /// Supports OpenRouter (recommended), Anthropic, Groq, Gemini, and Ollama providers.
    /// Requires `llm` feature for HTTP client (ureq).
    ///
    /// For texts exceeding `max_chunk_chars`, splits at sentence boundaries,
    /// processes chunks in parallel threads, and coalesces entities with
    /// overlap dedup.
    #[cfg(feature = "llm")]
    fn extract_with_llm(&self, text: &str, entity_types: &[&str]) -> Result<Vec<Entity>> {
        let char_count = text.chars().count();

        // If text fits in one chunk, process directly.
        if char_count <= self.max_chunk_chars {
            return self.extract_chunk(text, 0, entity_types);
        }

        // Use the shared parallel chunking infrastructure from streaming.rs.
        // Splits at sentence boundaries, processes chunks in parallel,
        // coalesces with exact-span + same-type overlap dedup.
        let config = ChunkConfig {
            chunk_size: self.max_chunk_chars,
            overlap: 200, // 200-char overlap to catch boundary entities
            respect_sentences: true,
            buffer_size: 1000,
        };

        extract_chunked_parallel(text, &config, |chunk_text, char_offset| {
            self.extract_chunk(chunk_text, char_offset, entity_types)
        })
    }

    /// Extract entities from a single chunk of text.
    ///
    /// `char_offset` is the chunk's starting character offset in the original
    /// document; entity offsets in the response are adjusted accordingly.
    #[cfg(feature = "llm")]
    fn extract_chunk(
        &self,
        chunk_text: &str,
        char_offset: usize,
        entity_types: &[&str],
    ) -> Result<Vec<Entity>> {
        let (api_key, provider) = crate::env::llm_api_key().ok_or_else(|| {
            crate::Error::FeatureNotAvailable(
                "No LLM API key found. Set OPENROUTER_API_KEY (recommended), GROQ_API_KEY, ANTHROPIC_API_KEY, or run Ollama locally.".into(),
            )
        })?;

        let default_model = |fallback: &str| -> String {
            self.config
                .as_ref()
                .map_or_else(|| fallback.to_string(), |c| c.model.clone())
        };

        // Determine model name for cache key
        let model_for_cache = match provider {
            "openrouter" => default_model("google/gemini-2.5-flash-lite"),
            "anthropic" => default_model("claude-haiku-4-5-20251001"),
            "groq" => default_model("llama-3.3-70b-versatile"),
            "ollama" => default_model("llama3.2:3b"),
            _ => default_model("google/gemini-2.5-flash-lite"),
        };

        // Check cache first (keyed on chunk text, not full document)
        let key = cache_key(chunk_text, entity_types, &model_for_cache);
        {
            let cache = lock(&self.cache);
            if let Some(cached) = cache.get(key) {
                // Adjust offsets from chunk-local to document-global
                let adjusted: Vec<Entity> = cached
                    .iter()
                    .map(|e| {
                        let mut e = e.clone();
                        if char_offset > 0 {
                            e.set_start(e.start() + char_offset);
                            e.set_end(e.end() + char_offset);
                        }
                        e
                    })
                    .collect();
                return Ok(adjusted);
            }
        }

        let (system_msg, user_msg) = self.build_prompt(chunk_text, entity_types);

        let max_tokens = self.config.as_ref().map_or(1024, |c| c.max_tokens);

        let (url, model, headers): (String, String, Vec<(&str, String)>) = match provider {
            "openrouter" => (
                "https://openrouter.ai/api/v1/chat/completions".to_string(),
                default_model("google/gemini-2.5-flash-lite"),
                vec![
                    ("Authorization", format!("Bearer {}", api_key)),
                    (
                        "HTTP-Referer",
                        "https://github.com/anno-rs/anno".to_string(),
                    ),
                    ("X-Title", "Anno NER".to_string()),
                ],
            ),
            "anthropic" => (
                "https://api.anthropic.com/v1/messages".to_string(),
                default_model("claude-haiku-4-5-20251001"),
                vec![
                    ("x-api-key", api_key.clone()),
                    ("anthropic-version", "2023-06-01".to_string()),
                ],
            ),
            "groq" => (
                "https://api.groq.com/openai/v1/chat/completions".to_string(),
                default_model("llama-3.3-70b-versatile"),
                vec![("Authorization", format!("Bearer {}", api_key))],
            ),
            "ollama" => {
                let host = std::env::var("OLLAMA_HOST")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string());
                (
                    format!("{}/v1/chat/completions", host),
                    default_model("llama3.2:3b"),
                    vec![("Authorization", "Bearer ollama".to_string())],
                )
            }
            "gemini" => (
                // Route Gemini through OpenRouter-compatible format
                "https://openrouter.ai/api/v1/chat/completions".to_string(),
                "google/gemini-2.5-flash-lite".to_string(),
                vec![("Authorization", format!("Bearer {}", api_key))],
            ),
            other => {
                return Err(crate::Error::FeatureNotAvailable(format!(
                    "LLM provider '{}' is not supported. Use OPENROUTER_API_KEY for access to all models.",
                    other
                )));
            }
        };

        // All providers use OpenAI-compatible format except direct Anthropic API
        let body = if provider == "anthropic" {
            serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": format!("{}\n\n{}", system_msg, user_msg)}
                ]
            })
        } else {
            serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens
            })
        };

        let mut req = ureq::post(&url);
        req = req.set("content-type", "application/json");
        for (key, value) in &headers {
            req = req.set(key, value);
        }

        let response = req
            .send_json(body)
            .map_err(|e| crate::Error::Inference(format!("LLM API error: {}", e)))?;

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

        let mut entities = if matches!(self.prompt_strategy, PromptStrategy::Compact) {
            self.parse_compact_response(content, chunk_text, entity_types)?
        } else {
            self.parse_llm_response(content, chunk_text)?
        };

        // Self-verification: re-query LLM to confirm each entity (GPT-NER strategy).
        // Reduces hallucination at the cost of additional API calls.
        if self.self_verify && !entities.is_empty() {
            entities =
                self.verify_entities(&url, &model, &headers, provider, chunk_text, entities)?;
        }

        // Cache the chunk-local result (before offset adjustment)
        lock(&self.cache).insert(key, entities.clone());

        // Adjust offsets from chunk-local to document-global
        if char_offset > 0 {
            for entity in &mut entities {
                entity.set_start(entity.start() + char_offset);
                entity.set_end(entity.end() + char_offset);
            }
        }

        Ok(entities)
    }

    /// Self-verification: ask the LLM to confirm each extracted entity.
    ///
    /// Per GPT-NER (NAACL 2025 Findings), LLMs tend to over-confidently
    /// label null inputs as entities. Verification filters hallucinations.
    #[cfg(feature = "llm")]
    fn verify_entities(
        &self,
        url: &str,
        model: &str,
        headers: &[(&str, String)],
        provider: &str,
        text: &str,
        entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        let entity_list: Vec<String> = entities
            .iter()
            .map(|e| format!("- \"{}\" ({})", e.text, e.entity_type.as_label()))
            .collect();

        let verify_prompt = format!(
            r#"Verify these named entities extracted from the text below.
For each entity, respond with "yes" if it is a valid entity of the stated type, or "no" if it is not.

## Text
"{text}"

## Entities to verify
{entities}

Respond with a JSON array of booleans in the same order. Example: [true, false, true]
Return ONLY the JSON array:"#,
            text = text,
            entities = entity_list.join("\n"),
        );

        let max_tokens = self.config.as_ref().map_or(256, |c| c.max_tokens.min(256));

        let body = if provider == "anthropic" {
            serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": verify_prompt}]
            })
        } else {
            serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an entity verification system. Respond ONLY with a JSON boolean array."},
                    {"role": "user", "content": verify_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens
            })
        };

        let mut req = ureq::post(url);
        req = req.set("content-type", "application/json");
        for (key, value) in headers {
            req = req.set(key, value);
        }

        let response = match req.send_json(body) {
            Ok(r) => r,
            Err(_) => return Ok(entities), // Verification failure -> keep originals
        };

        let json: serde_json::Value = match response.into_json() {
            Ok(j) => j,
            Err(_) => return Ok(entities),
        };

        let content = if provider == "anthropic" {
            json["content"][0]["text"].as_str().unwrap_or("[]")
        } else {
            json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("[]")
        };

        // Parse boolean array
        let trimmed = content.trim();
        let trimmed = trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed)
            .trim();
        let trimmed = trimmed.strip_suffix("```").unwrap_or(trimmed).trim();

        if let Ok(verdicts) = serde_json::from_str::<Vec<bool>>(trimmed) {
            Ok(entities
                .into_iter()
                .zip(verdicts)
                .filter_map(|(e, keep)| if keep { Some(e) } else { None })
                .collect())
        } else {
            // Could not parse verification response -> keep all entities
            Ok(entities)
        }
    }

    /// Fallback when `llm` feature is not enabled.
    #[cfg(not(feature = "llm"))]
    fn extract_with_llm(&self, _text: &str, _entity_types: &[&str]) -> Result<Vec<Entity>> {
        Err(crate::Error::FeatureNotAvailable(
            "UniversalNER requires the 'llm' feature to make HTTP requests (ureq). Rebuild with --features llm and set OPENROUTER_API_KEY (or run Ollama locally)."
                .into(),
        ))
    }

    /// Parse LLM response into entities.
    ///
    /// Parse compact format: `[[start, end, "key"], ...]`.
    ///
    /// No text field -- we extract from original_text using offsets.
    /// Type key is mapped back via the entity_types legend.
    #[allow(dead_code)]
    fn parse_compact_response(
        &self,
        content: &str,
        original_text: &str,
        entity_types: &[&str],
    ) -> Result<Vec<Entity>> {
        // Build key -> type mapping (same logic as build_prompt Compact)
        let mut key_to_type: Vec<(String, &str)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (i, t) in entity_types.iter().enumerate() {
            let mut k = t.chars().next().unwrap_or('X').to_uppercase().to_string();
            if !seen.insert(k.clone()) {
                k = format!("{}{}", k, i);
            }
            key_to_type.push((k, t));
        }

        let json_str = content.trim();
        let json_str = json_str
            .strip_prefix("```json")
            .or_else(|| json_str.strip_prefix("```JSON"))
            .or_else(|| json_str.strip_prefix("```"))
            .unwrap_or(json_str)
            .trim();
        let json_str = json_str.strip_suffix("```").unwrap_or(json_str).trim();

        // Find the array
        let json_str = if json_str.starts_with('[') {
            json_str.to_string()
        } else if let Some(start) = json_str.find('[') {
            if let Some(end) = json_str.rfind(']') {
                json_str[start..=end].to_string()
            } else {
                return Ok(Vec::new());
            }
        } else {
            return Ok(Vec::new());
        };

        let items: Vec<serde_json::Value> = serde_json::from_str(&json_str)
            .map_err(|e| crate::Error::Parse(format!("Compact NER parse error: {}", e)))?;

        let char_count = original_text.chars().count();
        let mut entities = Vec::new();

        for item in &items {
            let arr = match item.as_array() {
                Some(a) if a.len() >= 3 => a,
                _ => continue,
            };

            let start = match arr[0].as_u64() {
                Some(v) => v as usize,
                None => continue,
            };
            let end = match arr[1].as_u64() {
                Some(v) => v as usize,
                None => continue,
            };
            let key = match arr[2].as_str() {
                Some(k) => k,
                None => continue,
            };

            if start >= end || end > char_count {
                continue;
            }

            // Snap offsets to word boundaries if they land mid-word.
            // LLMs often get offsets slightly wrong; snapping recovers valid entities.
            let chars: Vec<char> = original_text.chars().collect();
            let mut adj_start = start;
            let mut adj_end = end;

            // Snap start leftward to word boundary (max 3 chars)
            while adj_start > 0
                && adj_start > start.saturating_sub(3)
                && !chars[adj_start.saturating_sub(1)].is_whitespace()
                && !chars[adj_start.saturating_sub(1)].is_ascii_punctuation()
            {
                adj_start -= 1;
            }
            // Snap end rightward to word boundary (max 3 chars)
            while adj_end < chars.len()
                && adj_end < end + 3
                && !chars[adj_end].is_whitespace()
                && !chars[adj_end].is_ascii_punctuation()
            {
                adj_end += 1;
            }

            let text_span: String = chars[adj_start..adj_end].iter().collect();
            let (start, end) = (adj_start, adj_end);

            if text_span.trim().is_empty() {
                continue;
            }

            // Map key back to entity type
            let type_str = key_to_type
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, t)| *t)
                .unwrap_or("misc");

            let entity_type = match type_str.to_lowercase().as_str() {
                "person" | "per" => EntityType::Person,
                "organization" | "org" => EntityType::Organization,
                "location" | "loc" | "gpe" => EntityType::Location,
                "date" | "time" => EntityType::Date,
                "money" | "currency" => EntityType::Money,
                other => EntityType::custom(other, anno_core::EntityCategory::Misc),
            };

            let mut entity = Entity::new(text_span, entity_type, start, end, 0.9);
            entity.provenance = Some(crate::Provenance::ml("universal_ner", entity.confidence));
            entities.push(entity);
        }

        Ok(entities)
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
                // SAFETY: guarded by !is_empty() above
                *occurrences
                    .iter()
                    .min_by_key(|(s, e)| {
                        let ds = (*s as isize - hint_start as isize).unsigned_abs();
                        let de = (*e as isize - hint_end as isize).unsigned_abs();
                        (ds + de, *s, *e)
                    })
                    .unwrap()
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
                _ => EntityType::custom(type_str, anno_core::EntityCategory::Misc),
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
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        if !self.llm_available {
            return Err(crate::Error::FeatureNotAvailable(
                "UniversalNER requires an LLM provider. Set OPENROUTER_API_KEY (recommended), GROQ_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, UNIVERSAL_NER_API_KEY, or run Ollama locally."
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

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            zero_shot: true,
            ..Default::default()
        }
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
                "UniversalNER requires an LLM provider. Set OPENROUTER_API_KEY (recommended), GROQ_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, UNIVERSAL_NER_API_KEY, or run Ollama locally."
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
    use std::sync::OnceLock;

    #[test]
    fn test_universal_ner_creation() {
        let model = UniversalNER::new().unwrap();
        assert_eq!(model.name(), "universal_ner");
    }

    #[test]
    fn test_universal_ner_availability_reflects_api_key() {
        // Env vars are global; serialize to avoid interference with other tests.
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = lock(ENV_LOCK.get_or_init(|| Mutex::new(())));

        // Override any `.env` values (dotenv only sets if unset).
        for k in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "GROQ_API_KEY",
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
            let extracted = TextSpan::from_chars(text, e.start(), e.end()).extract(text);
            assert_eq!(extracted, e.text, "entity span should round-trip");
        }
    }

    #[test]
    fn test_universal_ner_supported_types() {
        let model = UniversalNER::new().unwrap();
        let types = model.supported_types();
        assert!(types.contains(&EntityType::Person));
        assert!(types.contains(&EntityType::Organization));
        assert!(types.contains(&EntityType::Location));
        assert_eq!(types.len(), 3);
    }

    #[test]
    fn test_universal_ner_description() {
        let model = UniversalNER::new().unwrap();
        let desc = model.description();
        assert!(!desc.is_empty());
        assert!(
            desc.contains("UniversalNER"),
            "description should mention UniversalNER, got: {desc}"
        );
    }

    #[test]
    fn test_universal_ner_capabilities() {
        let model = UniversalNER::new().unwrap();
        let caps = model.capabilities();
        assert!(
            caps.zero_shot,
            "UniversalNER should have zero_shot capability"
        );
    }

    #[test]
    fn test_parse_llm_response_malformed_json() {
        let model = UniversalNER::new().unwrap();
        let text = "Hello world";

        // Completely invalid JSON.
        let result = model.parse_llm_response("this is not json", text);
        assert!(result.is_err(), "malformed JSON should return an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Parse"),
            "error should be a Parse variant: {msg}"
        );

        // JSON object instead of array.
        let result = model.parse_llm_response(r#"{"text": "Hello"}"#, text);
        assert!(
            result.is_err(),
            "JSON object (not array) should return an error"
        );

        // Incomplete array (no closing bracket).
        let result = model.parse_llm_response(r#"[{"text": "Hello""#, text);
        assert!(
            result.is_err(),
            "incomplete JSON array should return an error"
        );
    }

    #[test]
    fn test_parse_llm_response_empty_entity_list() {
        let model = UniversalNER::new().unwrap();
        let text = "No entities here at all.";
        let entities = model.parse_llm_response("[]", text).unwrap();
        assert!(
            entities.is_empty(),
            "empty JSON array should produce no entities"
        );
    }

    #[test]
    fn test_parse_llm_response_code_fence_variants() {
        let model = UniversalNER::new().unwrap();
        let text = "Alice met Bob.";

        // ```json ... ``` wrapping
        let fenced =
            "```json\n[{\"text\":\"Alice\",\"type\":\"person\",\"start\":0,\"end\":5}]\n```";
        let ents = model.parse_llm_response(fenced, text).unwrap();
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].text, "Alice");

        // ```JSON ... ``` wrapping (uppercase)
        let fenced_upper =
            "```JSON\n[{\"text\":\"Bob\",\"type\":\"person\",\"start\":10,\"end\":13}]\n```";
        let ents = model.parse_llm_response(fenced_upper, text).unwrap();
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].text, "Bob");

        // Plain ``` ... ``` wrapping
        let plain_fence =
            "```\n[{\"text\":\"Alice\",\"type\":\"person\",\"start\":0,\"end\":5}]\n```";
        let ents = model.parse_llm_response(plain_fence, text).unwrap();
        assert_eq!(ents.len(), 1);
    }

    #[test]
    fn test_parse_llm_response_with_preamble_text() {
        // Some LLMs add explanatory text before the JSON.
        let model = UniversalNER::new().unwrap();
        let text = "Alice met Bob in Paris.";
        let response = "Here are the entities I found:\n[{\"text\":\"Alice\",\"type\":\"person\",\"start\":0,\"end\":5},{\"text\":\"Paris\",\"type\":\"location\",\"start\":17,\"end\":22}]";
        let ents = model.parse_llm_response(response, text).unwrap();
        assert_eq!(ents.len(), 2);
    }

    #[test]
    fn test_parse_llm_response_offset_validation() {
        let model = UniversalNER::new().unwrap();
        let text = "Hello World";

        // Entity with end <= start should be skipped.
        let response = r#"[{"text":"Hello","type":"person","start":5,"end":3}]"#;
        let ents = model.parse_llm_response(response, text).unwrap();
        assert!(
            ents.is_empty(),
            "entity with end <= start should be filtered out"
        );

        // Entity with empty text should be skipped.
        let response = r#"[{"text":"","type":"person","start":0,"end":5}]"#;
        let ents = model.parse_llm_response(response, text).unwrap();
        assert!(
            ents.is_empty(),
            "entity with empty text should be filtered out"
        );
    }

    #[test]
    fn test_parse_llm_response_out_of_bounds_offsets() {
        let model = UniversalNER::new().unwrap();
        let text = "Short"; // 5 chars

        // Hint offsets way beyond text length, and entity text not found in original.
        let response = r#"[{"text":"Nonexistent","type":"person","start":100,"end":111}]"#;
        let ents = model.parse_llm_response(response, text).unwrap();
        assert!(
            ents.is_empty(),
            "out-of-bounds entity not found in text should be skipped"
        );
    }

    #[test]
    fn test_parse_llm_response_entity_type_mapping() {
        let model = UniversalNER::new().unwrap();
        let text = "Alice Bob Paris $100 Monday Acme";

        let response = r#"[
            {"text":"Alice","type":"PER","start":0,"end":5},
            {"text":"Bob","type":"person","start":6,"end":9},
            {"text":"Paris","type":"LOC","start":10,"end":15},
            {"text":"$100","type":"money","start":16,"end":20},
            {"text":"Monday","type":"date","start":21,"end":27},
            {"text":"Acme","type":"ORG","start":28,"end":32}
        ]"#;
        let ents = model.parse_llm_response(response, text).unwrap();

        let types: Vec<_> = ents.iter().map(|e| &e.entity_type).collect();
        // PER -> Person, person -> Person
        assert!(matches!(types[0], EntityType::Person));
        assert!(matches!(types[1], EntityType::Person));
        // LOC -> Location
        assert!(matches!(types[2], EntityType::Location));
        // money -> Money
        assert!(matches!(types[3], EntityType::Money));
        // date -> Date
        assert!(matches!(types[4], EntityType::Date));
        // ORG -> Organization
        assert!(matches!(types[5], EntityType::Organization));
    }

    #[test]
    fn test_parse_llm_response_provenance_is_ml() {
        let model = UniversalNER::new().unwrap();
        let text = "Alice met Bob.";
        let response = r#"[{"text":"Alice","type":"person","start":0,"end":5}]"#;
        let ents = model.parse_llm_response(response, text).unwrap();
        assert_eq!(ents.len(), 1);

        let prov = ents[0]
            .provenance
            .as_ref()
            .expect("universal_ner entities should have provenance");
        assert_eq!(prov.source, "universal_ner");
        assert!(
            matches!(prov.method, crate::ExtractionMethod::Neural),
            "provenance method should be Neural (ML), got: {:?}",
            prov.method
        );
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
        let mut starts: Vec<usize> = apples.iter().map(|e| e.start()).collect();
        starts.sort_unstable();
        starts.dedup();
        assert_eq!(
            starts.len(),
            3,
            "each Apple should map to a distinct occurrence"
        );
    }

    // ---- New tests for configurable model, prompt strategy, domain context ----

    #[test]
    fn test_with_config_sets_model() {
        let config = crate::backends::llm_client::LlmConfig::haiku();
        let model = UniversalNER::with_config(config).unwrap();
        assert_eq!(model.name(), "universal_ner");
        assert!(model.config.is_some());
        assert_eq!(
            model.config.as_ref().unwrap().model,
            "anthropic/claude-haiku-4.5"
        );
    }

    #[test]
    fn test_prompt_strategy_default_is_simple() {
        let model = UniversalNER::new().unwrap();
        assert!(matches!(model.prompt_strategy, PromptStrategy::Simple));
    }

    #[test]
    fn test_prompt_strategy_codener() {
        let model = UniversalNER::new()
            .unwrap()
            .prompt_strategy(PromptStrategy::CodeNER {
                chain_of_thought: true,
            });
        assert!(matches!(
            model.prompt_strategy,
            PromptStrategy::CodeNER {
                chain_of_thought: true
            }
        ));
    }

    #[test]
    fn test_build_prompt_simple() {
        let model = UniversalNER::new().unwrap();
        let (system, user) = model.build_prompt("Alice met Bob.", &["person", "location"]);
        assert!(system.contains("named entity recognition"));
        assert!(user.contains("person, location"));
        assert!(user.contains("Alice met Bob."));
        // Simple strategy should not contain BIO schema code
        assert!(!user.contains("extract_entities"));
    }

    #[test]
    fn test_build_prompt_codener() {
        let model = UniversalNER::new()
            .unwrap()
            .prompt_strategy(PromptStrategy::CodeNER {
                chain_of_thought: false,
            });
        let (system, user) = model.build_prompt("Alice met Bob.", &["person", "location"]);
        assert!(system.contains("NER"));
        // CodeNER strategy renders a Python-style code prompt
        assert!(user.contains("extract_entities"));
        assert!(user.contains("BIO Schema"));
    }

    #[test]
    fn test_build_prompt_codener_with_cot() {
        let model = UniversalNER::new()
            .unwrap()
            .prompt_strategy(PromptStrategy::CodeNER {
                chain_of_thought: true,
            });
        let (_system, user) = model.build_prompt("Test.", &["person"]);
        assert!(user.contains("identify potential entity spans"));
    }

    #[test]
    fn test_domain_context_injected_simple() {
        let model = UniversalNER::new()
            .unwrap()
            .domain_context("This is 19th-century German diplomatic correspondence.");
        let (system, _user) = model.build_prompt("Bismarck met Kaiser.", &["person"]);
        assert!(
            system.contains("19th-century German diplomatic"),
            "domain context should appear in system message"
        );
    }

    #[test]
    fn test_domain_context_injected_codener() {
        let model = UniversalNER::new()
            .unwrap()
            .prompt_strategy(PromptStrategy::CodeNER {
                chain_of_thought: false,
            })
            .domain_context("Biomedical research papers.");
        let (system, _user) = model.build_prompt("BRCA1 inhibits p53.", &["gene", "protein"]);
        assert!(
            system.contains("Biomedical research"),
            "domain context should appear in CodeNER system message"
        );
    }

    // ---- Compact prompt strategy tests ----

    #[test]
    fn test_build_prompt_compact() {
        let model = UniversalNER::new()
            .unwrap()
            .prompt_strategy(PromptStrategy::Compact);
        let (system, user) = model.build_prompt("Alice met Bob.", &["person", "location"]);
        assert!(system.contains("NER"));
        assert!(user.contains("Legend:"));
        assert!(user.contains("P=person"));
        assert!(user.contains("L=location"));
        // Compact prompt should be shorter than Simple
        assert!(
            user.len() < 200,
            "compact prompt should be concise: {} chars",
            user.len()
        );
    }

    #[test]
    fn test_parse_compact_response() {
        let model = UniversalNER::new().unwrap();
        let text = "Alice met Bob in Paris.";
        let response = r#"[[0,5,"P"],[10,13,"P"],[17,22,"L"]]"#;
        let ents = model
            .parse_compact_response(response, text, &["person", "location"])
            .expect("parse compact");
        assert_eq!(ents.len(), 3);
        assert_eq!(ents[0].text, "Alice");
        assert!(matches!(ents[0].entity_type, EntityType::Person));
        assert_eq!(ents[1].text, "Bob");
        assert!(matches!(ents[1].entity_type, EntityType::Person));
        assert_eq!(ents[2].text, "Paris");
        assert!(matches!(ents[2].entity_type, EntityType::Location));
    }

    #[test]
    fn test_parse_compact_response_with_fences() {
        let model = UniversalNER::new().unwrap();
        let text = "Alice in Zurich.";
        let response = "```json\n[[0,5,\"P\"],[9,15,\"L\"]]\n```";
        let ents = model
            .parse_compact_response(response, text, &["person", "location"])
            .expect("parse compact with fences");
        assert_eq!(ents.len(), 2);
    }

    #[test]
    fn test_max_chunk_chars_builder() {
        let model = UniversalNER::new().unwrap().max_chunk_chars(2000);
        assert_eq!(model.max_chunk_chars, 2000);
    }

    #[test]
    fn test_default_max_chunk_chars() {
        let model = UniversalNER::new().unwrap();
        assert_eq!(model.max_chunk_chars, 4000);
    }

    #[test]
    fn test_parse_compact_response_filters_invalid() {
        let model = UniversalNER::new().unwrap();
        let text = "Short"; // 5 chars
        let response = r#"[[0,5,"P"],[10,20,"L"],[3,2,"P"]]"#;
        let ents = model
            .parse_compact_response(response, text, &["person", "location"])
            .expect("parse compact filters");
        // Only first is valid (second out of bounds, third start>=end)
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].text, "Short");
    }

    // Overlap-aware dedup tests are in streaming::tests (shared infra).
}
