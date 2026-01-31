//! LLM client abstraction for NER and coreference tasks.
//!
//! This module provides a trait-based abstraction over LLM providers
//! (OpenAI, Anthropic, Ollama, etc.) for use in NER-related tasks like
//! entity verification and mention disambiguation.
//!
//! # Example
//!
//! ```rust,no_run
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
//! may require feature flags:
//!
//! - `llm-openai`: OpenAI API client
//! - `llm-ollama`: Local Ollama client
//! - `llm-anthropic`: Anthropic Claude client

use serde::{Deserialize, Serialize};

/// Configuration for LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Model identifier (e.g., "gpt-4", "qwen2.5:7b", "claude-3-sonnet")
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
            model: "gpt-4o-mini".to_string(),
            max_tokens: 100,
            temperature: 0.0,
            system_prompt: None,
            endpoint: None,
            api_key: None,
        }
    }
}

impl LlmConfig {
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
/// ```rust
/// use anno::backends::llm_client::{MockProvider, LlmProvider};
///
/// // Create with default empty response
/// let provider = MockProvider::default();
///
/// // Or create with custom response
/// let provider = MockProvider::new("Yes");
/// ```
#[derive(Debug, Clone)]
#[derive(Default)]
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
        let config = LlmConfig::with_model("qwen2.5:7b")
            .max_tokens(10)
            .temperature(0.0)
            .system_prompt("You are an NER assistant.");

        assert_eq!(config.model, "qwen2.5:7b");
        assert_eq!(config.max_tokens, 10);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(
            config.system_prompt,
            Some("You are an NER assistant.".to_string())
        );
    }
}
