//! LLM client abstraction for NER and coreference tasks.
//!
//! This module provides a trait-based abstraction over LLM providers
//! (OpenAI, Anthropic, Ollama, etc.) for use in NER-related tasks like
//! entity verification and mention disambiguation.
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::llm_client::{LlmConfig, LlmProvider, LlmRequest, LlmResponse};
//!
//! // Implement LlmProvider for your preferred backend
//! struct MockProvider;
//!
//! impl LlmProvider for MockProvider {
//!     fn complete(&self, request: LlmRequest) -> Result<LlmResponse, String> {
//!         Ok(LlmResponse {
//!             text: "Yes".to_string(),
//!             tokens_used: 5,
//!         })
//!     }
//!
//!     fn name(&self) -> &str {
//!         "mock"
//!     }
//! }
//! ```
//!
//! # Feature Flags
//!
//! This module is always available but specific provider implementations
//! require the `llm` feature flag. All models accessed via OpenRouter
//! (openrouter.ai) or local Ollama.

use serde::{Deserialize, Serialize};

/// Configuration for LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Model identifier (e.g., "google/gemini-2.5-flash-lite", "anthropic/claude-haiku-4.5", "deepseek/deepseek-v3.2")
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// System prompt (optional)
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// API endpoint override (optional)
    #[serde(default)]
    pub endpoint: Option<String>,
    /// API key (optional, can be set via env var)
    #[serde(default, skip_serializing)]
    pub api_key: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "google/gemini-2.5-flash-lite".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: None,
            api_key: None,
        }
    }
}

impl LlmConfig {
    /// Preset: Google Gemini 2.5 Flash Lite via OpenRouter ($0.10/$0.40 per M tokens).
    /// Fast and cheap, good for bulk NER. Default model.
    pub fn gemini_flash_lite() -> Self {
        Self {
            model: "google/gemini-2.5-flash-lite".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: None,
            api_key: None,
        }
    }

    /// Preset: Google Gemini 2.5 Flash via OpenRouter ($0.30/$2.50 per M tokens).
    /// Strong quality, higher cost than lite variant.
    pub fn gemini_flash() -> Self {
        Self {
            model: "google/gemini-2.5-flash".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: None,
            api_key: None,
        }
    }

    /// Preset: Anthropic Claude Haiku 4.5 via OpenRouter ($1.00/$5.00 per M tokens).
    /// Highest structured output quality for NER.
    pub fn haiku() -> Self {
        Self {
            model: "anthropic/claude-haiku-4.5".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: None,
            api_key: None,
        }
    }

    /// Preset: DeepSeek V3.2 via OpenRouter ($0.25/$0.40 per M tokens).
    /// Best open-source quality, cheapest output tokens.
    pub fn deepseek() -> Self {
        Self {
            model: "deepseek/deepseek-v3.2".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: None,
            api_key: None,
        }
    }

    /// Preset: Llama 3.3 70B via OpenRouter ($0.10/$0.32 per M tokens).
    /// Also available via Groq for ultra-fast inference (set GROQ_API_KEY).
    pub fn llama3() -> Self {
        Self {
            model: "meta-llama/llama-3.3-70b-instruct".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: None,
            api_key: None,
        }
    }

    /// Preset: Llama 4 Scout via OpenRouter ($0.08/$0.30 per M tokens).
    /// Newest Llama, strong quality for the price.
    pub fn llama4() -> Self {
        Self {
            model: "meta-llama/llama-4-scout".to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: None,
            api_key: None,
        }
    }

    /// Preset: Groq direct API (ultra-fast inference for open models).
    /// Uses GROQ_API_KEY. Model should be a Groq-hosted model ID.
    pub fn groq(model: &str) -> Self {
        Self {
            model: model.to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: Some("https://api.groq.com/openai/v1/chat/completions".to_string()),
            api_key: None,
        }
    }

    /// Preset: Local Ollama model.
    pub fn ollama(model: &str) -> Self {
        Self {
            model: model.to_string(),
            max_tokens: 1024,
            temperature: 0.0,
            system_prompt: None,
            endpoint: Some("http://localhost:11434/v1/chat/completions".to_string()),
            api_key: Some("ollama".to_string()),
        }
    }

    /// Create config for a specific model.
    pub fn with_model(model: &str) -> Self {
        Self {
            model: model.to_string(),
            ..Default::default()
        }
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set system prompt.
    pub fn system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }
}

/// A request to an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    /// The prompt text
    pub prompt: String,
    /// Configuration for this request
    pub config: LlmConfig,
    /// Optional context/chat history
    #[serde(default)]
    pub context: Vec<Message>,
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role: "system", "user", "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

/// Response from an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Generated text
    pub text: String,
    /// Tokens used (input + output)
    pub tokens_used: usize,
}

/// Trait for LLM provider implementations.
///
/// Implement this trait to integrate with different LLM backends.
pub trait LlmProvider: Send + Sync {
    /// Generate a completion for the given request.
    fn complete(&self, request: LlmRequest) -> Result<LlmResponse, String>;

    /// Provider name for logging/debugging.
    fn name(&self) -> &str;

    /// Check if the provider is available and configured.
    fn is_available(&self) -> bool {
        true
    }
}

/// A no-op provider that always returns a default response.
///
/// Useful for testing and as a fallback when no LLM is configured.
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::llm_client::{MockProvider, LlmProvider};
///
/// // Create with default empty response
/// let provider = MockProvider::default();
///
/// // Or create with custom response
/// let provider = MockProvider::new("Yes");
/// ```
#[derive(Debug, Clone, Default)]
pub struct MockProvider {
    response: String,
}

impl MockProvider {
    /// Create a new mock provider with a fixed response.
    pub fn new(response: &str) -> Self {
        Self {
            response: response.to_string(),
        }
    }
}

impl LlmProvider for MockProvider {
    fn complete(&self, _request: LlmRequest) -> Result<LlmResponse, String> {
        Ok(LlmResponse {
            text: self.response.clone(),
            tokens_used: 0,
        })
    }

    fn name(&self) -> &str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_provider() {
        let provider = MockProvider::new("Yes");
        let request = LlmRequest {
            prompt: "Is 'he' a reference to John?".to_string(),
            config: LlmConfig::default(),
            context: vec![],
        };
        let response = provider.complete(request).unwrap();
        assert_eq!(response.text, "Yes");
    }

    #[test]
    fn test_llm_config_builder() {
        let config = LlmConfig::with_model("google/gemini-2.5-flash")
            .max_tokens(10)
            .temperature(0.0)
            .system_prompt("You are an NER assistant.");

        assert_eq!(config.model, "google/gemini-2.5-flash");
        assert_eq!(config.max_tokens, 10);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(
            config.system_prompt,
            Some("You are an NER assistant.".to_string())
        );
    }

    #[test]
    fn test_preset_gemini_flash() {
        let config = LlmConfig::gemini_flash();
        assert_eq!(config.model, "google/gemini-2.5-flash");
        assert_eq!(config.max_tokens, 1024);
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn test_preset_haiku() {
        let config = LlmConfig::haiku();
        assert_eq!(config.model, "anthropic/claude-haiku-4.5");
    }

    #[test]
    fn test_preset_deepseek() {
        let config = LlmConfig::deepseek();
        assert_eq!(config.model, "deepseek/deepseek-v3.2");
    }

    #[test]
    fn test_preset_ollama() {
        let config = LlmConfig::ollama("llama3.2:3b");
        assert_eq!(config.model, "llama3.2:3b");
        assert!(config.endpoint.as_ref().unwrap().contains("localhost"));
        assert_eq!(config.api_key.as_deref(), Some("ollama"));
    }

    #[test]
    fn test_default_is_gemini_flash_lite() {
        let config = LlmConfig::default();
        assert_eq!(config.model, "google/gemini-2.5-flash-lite");
        assert_eq!(config.max_tokens, 1024);
    }

    #[test]
    fn test_preset_gemini_flash_lite() {
        let config = LlmConfig::gemini_flash_lite();
        assert_eq!(config.model, "google/gemini-2.5-flash-lite");
    }
}
