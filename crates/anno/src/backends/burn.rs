//! Burn ML Framework Integration for NER
//!
//! Burn is a flexible, portable deep learning framework for Rust that supports:
//! - Training (unlike Candle/ONNX which are inference-only)
//! - Multiple backends: NdArray (pure Rust), Tch (PyTorch), Wgpu (WebGPU)
//! - ONNX import via `burn-import`
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    BurnNER Architecture                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Input: "Steve Jobs founded Apple"                          │
//! │                        │                                     │
//! │  ┌────────────────────▼────────────────────┐                │
//! │  │         Tokenizer (HuggingFace)          │                │
//! │  └────────────────────┬────────────────────┘                │
//! │                        │                                     │
//! │  ┌────────────────────▼────────────────────┐                │
//! │  │      Encoder (BERT via Burn)             │                │
//! │  │      Backend: NdArray/Wgpu/Tch           │                │
//! │  └────────────────────┬────────────────────┘                │
//! │                        │                                     │
//! │  ┌────────────────────▼────────────────────┐                │
//! │  │     Classification Head (Linear)         │                │
//! │  └────────────────────┬────────────────────┘                │
//! │                        │                                     │
//! │  Output: B-PER I-PER O B-ORG                                │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - `burn` - Enable NdArray backend (pure Rust, portable)
//! - `burn-gpu` - Enable WebGPU backend (GPU acceleration)
//! - `burn-torch` - Enable PyTorch backend (libtorch required)
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::burn::{BurnNER, BurnConfig, BurnBackendType};
//! use anno::Model;
//!
//! // Use NdArray backend (default, pure Rust)
//! let ner = BurnNER::new()?;
//!
//! // Use WebGPU backend for GPU acceleration
//! let config = BurnConfig::new().with_wgpu();
//! let ner = BurnNER::with_config(config)?;
//!
//! let entities = ner.extract_entities("Marie Curie won the Nobel Prize", None)?;
//! ```
//!
//! # Model Loading
//!
//! Burn models can be loaded from:
//! 1. Burn's native format (`.mpk` files)
//! 2. ONNX models via `burn-import`
//! 3. Converted from PyTorch/HuggingFace models
//!
//! For HuggingFace models, use `BurnNER::from_pretrained()` which handles
//! the conversion automatically.

use crate::{Entity, EntityType, Model, Result};

// =============================================================================
// Backend Configuration
// =============================================================================

/// Burn backend types.
///
/// Each backend has different tradeoffs:
/// - **NdArray**: Pure Rust, no dependencies, portable (typically slower than GPU backends)
/// - **Wgpu**: WebGPU, works on most GPUs, browser-compatible
/// - **Tch**: PyTorch backend, fastest but requires libtorch
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BurnBackendType {
    /// Pure Rust ndarray backend - no external dependencies
    #[default]
    NdArray,
    /// PyTorch backend via tch - requires libtorch
    Tch,
    /// WebGPU backend - portable GPU acceleration
    Wgpu,
}

impl std::fmt::Display for BurnBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BurnBackendType::NdArray => write!(f, "ndarray"),
            BurnBackendType::Tch => write!(f, "tch"),
            BurnBackendType::Wgpu => write!(f, "wgpu"),
        }
    }
}

/// Burn device types.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BurnDevice {
    /// CPU execution.
    #[default]
    Cpu,
    /// CUDA device by ordinal (e.g., `Cuda(0)` for the first CUDA device).
    Cuda(usize),
    /// Apple Metal GPU.
    Metal,
    /// Vulkan device (where supported by the selected backend).
    Vulkan,
}

/// Configuration for Burn-based models.
#[derive(Debug, Clone, Default)]
pub struct BurnConfig {
    /// Backend type
    pub backend: BurnBackendType,
    /// Device
    pub device: BurnDevice,
    /// Model ID (HuggingFace or local path)
    pub model_id: Option<String>,
    /// Confidence threshold
    pub threshold: f64,
}

impl BurnConfig {
    /// Create default config (ndarray on CPU).
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: BurnBackendType::NdArray,
            device: BurnDevice::Cpu,
            model_id: None,
            threshold: 0.5,
        }
    }

    /// Use ndarray backend (pure Rust, portable).
    #[must_use]
    pub fn with_ndarray(mut self) -> Self {
        self.backend = BurnBackendType::NdArray;
        self
    }

    /// Use tch (PyTorch) backend.
    #[must_use]
    pub fn with_tch(mut self) -> Self {
        self.backend = BurnBackendType::Tch;
        self
    }

    /// Use wgpu (WebGPU) backend.
    #[must_use]
    pub fn with_wgpu(mut self) -> Self {
        self.backend = BurnBackendType::Wgpu;
        self
    }

    /// Set model ID.
    #[must_use]
    pub fn with_model(mut self, model_id: &str) -> Self {
        self.model_id = Some(model_id.to_string());
        self
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

// =============================================================================
// Burn NER Implementation
// =============================================================================

/// Standard CoNLL-style NER labels.
const CONLL_LABELS: &[&str] = &[
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC",
];

/// Burn-powered NER model.
///
/// This provides token classification NER using the Burn ML framework.
///
/// Current status:
/// - The **Burn integration is scaffolding only** (model import + inference are not wired up yet).
/// - Calls to [`Model::extract_entities`] return a clear error instead of silently falling back to
///   heuristics.
///
/// # Backends
///
/// | Backend | Feature | Dependencies |
/// |---------|---------|--------------|
/// | NdArray | `burn` | None |
/// | Wgpu | `burn-gpu` | WebGPU runtime |
/// | Tch | `burn-torch` | libtorch |
#[derive(Debug, Clone)]
pub struct BurnNER {
    config: BurnConfig,
    model_name: String,
    id2label: Vec<String>,
}

impl Default for BurnNER {
    fn default() -> Self {
        Self {
            config: BurnConfig::default(),
            model_name: "burn-ner".to_string(),
            id2label: CONLL_LABELS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl BurnNER {
    /// Create with default configuration.
    ///
    /// Uses NdArray backend (pure Rust) by default.
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }

    /// Create with specific configuration.
    pub fn with_config(config: BurnConfig) -> Result<Self> {
        Ok(Self {
            config,
            ..Self::default()
        })
    }

    /// Load a pre-trained model from HuggingFace.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model ID (e.g., "dslim/bert-base-NER")
    ///
    /// # Note
    /// This only records the model id in the config today.
    ///
    /// Burn model import + inference is not wired up yet; attempting to run inference will return
    /// an explicit error (no silent heuristic fallback).
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        #[cfg(feature = "burn")]
        {
            log::info!("[BurnNER] Loading model: {}", model_id);
            // FUTURE WORK: Implement HuggingFace model loading via burn-import
            //
            // Implementation steps:
            // 1. Use burn-import crate to convert PyTorch/safetensors to Burn format
            // 2. Define BERT architecture in Burn (BertEmbedding, BertEncoder, BertPooler)
            // 3. Add token classification head (Linear layer over hidden states)
            // 4. Load converted weights into Burn model
            // 5. Handle tokenizer (use tokenizers crate with HF tokenizer.json)
            //
            // References:
            // - burn-import: https://github.com/tracel-ai/burn/tree/main/crates/burn-import
            // - Burn examples: https://github.com/tracel-ai/burn/tree/main/examples
            //
            // NOTE: We do not silently fall back to heuristics here anymore.
        }

        Ok(Self {
            config: BurnConfig::new().with_model(model_id),
            model_name: model_id.to_string(),
            id2label: CONLL_LABELS.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Get the current backend type.
    #[must_use]
    pub fn backend(&self) -> BurnBackendType {
        self.config.backend
    }

    /// Check if the Burn feature is enabled.
    #[must_use]
    pub fn is_burn_enabled() -> bool {
        cfg!(feature = "burn")
    }

    /// Get model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Extract entities using Burn inference.
    #[cfg(feature = "burn")]
    fn extract_with_burn(&self, text: &str) -> Result<Vec<Entity>> {
        let _ = text;
        Err(crate::Error::FeatureNotAvailable(
            "BurnNER inference is not implemented yet. \
Use `BertNEROnnx` (`--features onnx`) or `GLiNERCandle` (`--features candle`) for ML inference. \
If you want a zero-dependency baseline, use `HeuristicNER`/`StackedNER`."
                .to_string(),
        ))
    }

    /// Map label string to EntityType.
    fn label_to_entity_type(label: &str) -> EntityType {
        let tag = label
            .strip_prefix("B-")
            .or_else(|| label.strip_prefix("I-"))
            .unwrap_or(label);

        match tag.to_uppercase().as_str() {
            "PER" | "PERSON" => EntityType::Person,
            "ORG" | "ORGANIZATION" => EntityType::Organization,
            "LOC" | "LOCATION" | "GPE" => EntityType::Location,
            "DATE" => EntityType::Date,
            "TIME" => EntityType::Time,
            "MONEY" => EntityType::Money,
            "PERCENT" => EntityType::Percent,
            other => EntityType::Other(other.to_string()),
        }
    }
}

impl Model for BurnNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        if self.config.model_id.is_none() {
            return Err(crate::Error::ModelInit(
                "BurnNER has no model configured. Call `BurnNER::from_pretrained(...)` or \
`BurnNER::with_config(BurnConfig::new().with_model(...))`. Note: model import/inference is \
not implemented yet; see the error from `extract_entities` once a model is configured."
                    .to_string(),
            ));
        }

        self.extract_with_burn(text)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.id2label
            .iter()
            .filter(|l| l.starts_with("B-"))
            .map(|l| Self::label_to_entity_type(l))
            .collect()
    }

    fn is_available(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "burn_ner"
    }

    fn description(&self) -> &'static str {
        "BurnNER (planned): Burn scaffolding; model import + inference not implemented yet"
    }
}

// Implement marker traits
impl crate::NamedEntityCapable for BurnNER {}

// =============================================================================
// GPU Capability
// =============================================================================

impl crate::GpuCapable for BurnNER {
    fn is_gpu_active(&self) -> bool {
        matches!(
            self.config.backend,
            BurnBackendType::Wgpu | BurnBackendType::Tch
        ) && !matches!(self.config.device, BurnDevice::Cpu)
    }

    fn device(&self) -> &str {
        match (&self.config.backend, &self.config.device) {
            (BurnBackendType::Wgpu, BurnDevice::Vulkan) => "vulkan",
            (BurnBackendType::Wgpu, _) => "wgpu",
            (BurnBackendType::Tch, BurnDevice::Cuda(_)) => "cuda",
            (BurnBackendType::Tch, BurnDevice::Metal) => "metal",
            _ => "cpu",
        }
    }
}

// =============================================================================
// Batch Capability
// =============================================================================

impl crate::BatchCapable for BurnNER {
    fn optimal_batch_size(&self) -> Option<usize> {
        match self.config.backend {
            BurnBackendType::NdArray => Some(4),
            BurnBackendType::Wgpu => Some(16),
            BurnBackendType::Tch => Some(32),
        }
    }
}

// =============================================================================
// Type Alias for Backwards Compatibility
// =============================================================================

/// Alias for backwards compatibility.
pub type BurnPoweredNER = BurnNER;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchCapable, GpuCapable, Model};

    #[test]
    fn test_burn_config_defaults() {
        let config = BurnConfig::new();
        assert_eq!(config.backend, BurnBackendType::NdArray);
        assert_eq!(config.device, BurnDevice::Cpu);
        assert_eq!(config.threshold, 0.5);
    }

    #[test]
    fn test_burn_config_builder() {
        let config = BurnConfig::new()
            .with_wgpu()
            .with_model("dslim/bert-base-NER")
            .with_threshold(0.7);

        assert_eq!(config.backend, BurnBackendType::Wgpu);
        assert_eq!(config.model_id, Some("dslim/bert-base-NER".to_string()));
        assert!((config.threshold - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_burn_ner_creation() {
        let ner = BurnNER::new().unwrap();
        assert_eq!(ner.name(), "burn_ner");
        assert!(!ner.is_available());
        assert_eq!(ner.backend(), BurnBackendType::NdArray);
    }

    #[test]
    fn test_burn_ner_empty_input() {
        let ner = BurnNER::new().unwrap();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_burn_ner_requires_model_configuration() {
        let ner = BurnNER::new().unwrap();
        let err = ner
            .extract_entities("Dr. John Smith works at Google in California", None)
            .unwrap_err();
        assert!(
            matches!(err, crate::Error::ModelInit(_)),
            "expected ModelInit error, got: {:?}",
            err
        );
    }

    #[test]
    fn test_burn_ner_multilingual_error_is_unicode_safe() {
        // Ensure we don't panic on multi-byte text even when returning an error.
        let ner = BurnNER::new().unwrap();
        let err = ner
            .extract_entities("習近平在北京會見了普京。", None)
            .unwrap_err();
        assert!(matches!(err, crate::Error::ModelInit(_)));
    }

    #[test]
    fn test_burn_ner_configured_model_still_errors_clearly() {
        let ner = BurnNER::from_pretrained("dslim/bert-base-NER").unwrap();
        let err = ner
            .extract_entities("Marie Curie won the Nobel Prize in Paris.", None)
            .unwrap_err();
        assert!(
            matches!(err, crate::Error::FeatureNotAvailable(_)),
            "expected FeatureNotAvailable error, got: {:?}",
            err
        );
    }

    #[test]
    fn test_label_to_entity_type() {
        assert_eq!(BurnNER::label_to_entity_type("B-PER"), EntityType::Person);
        assert_eq!(
            BurnNER::label_to_entity_type("I-ORG"),
            EntityType::Organization
        );
        assert_eq!(BurnNER::label_to_entity_type("B-LOC"), EntityType::Location);
        assert_eq!(
            BurnNER::label_to_entity_type("B-MISC"),
            EntityType::Other("MISC".to_string())
        );
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", BurnBackendType::NdArray), "ndarray");
        assert_eq!(format!("{}", BurnBackendType::Tch), "tch");
        assert_eq!(format!("{}", BurnBackendType::Wgpu), "wgpu");
    }

    #[test]
    fn test_gpu_capable() {
        let ner = BurnNER::new().unwrap();
        assert!(!ner.is_gpu_active()); // NdArray on CPU is not GPU

        let config = BurnConfig::new().with_wgpu();
        let ner = BurnNER::with_config(config).unwrap();
        // Still not active because device is Cpu by default
        assert!(!ner.is_gpu_active());
    }

    #[test]
    fn test_batch_capable() {
        let ner = BurnNER::new().unwrap();
        assert_eq!(ner.optimal_batch_size(), Some(4)); // NdArray

        let config = BurnConfig::new().with_wgpu();
        let ner = BurnNER::with_config(config).unwrap();
        assert_eq!(ner.optimal_batch_size(), Some(16)); // Wgpu
    }

    #[test]
    fn test_feature_check() {
        // Should compile regardless of feature
        let _enabled = BurnNER::is_burn_enabled();
    }
}
