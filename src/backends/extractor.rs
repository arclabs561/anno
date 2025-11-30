//! Unified NER extractor with fallback support.
//!
//! Provides a single entry point for NER that:
//! - Tries ML backends first (BERT ONNX, GLiNER, Candle)
//! - Falls back to pattern extraction
//! - Supports hybrid mode (ML + patterns combined)
//!
//! # Backend Selection Priority
//!
//! | Backend | Feature | CoNLL-03 F1 | CrossNER F1 | Speed | Notes |
//! |---------|---------|-------------|-------------|-------|-------|
//! | GLiNER v2.1 | `onnx` | ~90%* | ~61% | ~100ms | Zero-shot, SOTA |
//! | BERT ONNX | `onnx` | ~86% | N/A | ~50ms | Fixed 4 types |
//! | Candle BERT | `candle` | ~74% | N/A | ~50ms | Rust-native |
//! | PatternNER | always | N/A | N/A | ~400ns | Structured only |
//!
//! *CoNLL F1 from original GLiNER paper; CrossNER from arxiv:2507.18546
//!
//! # Design Philosophy (from hop)
//!
//! - **ML-first**: Use best available ML model
//! - **Graceful degradation**: Falls back to patterns if ML unavailable
//! - **Hybrid mode**: Best of both worlds (ML for context, patterns for structure)
//! - **Clean adapters**: Each backend wrapped to implement common trait

use crate::{Entity, EntityType, Model, PatternNER, Result};
use std::sync::Arc;

/// Default models for each backend.
pub mod defaults {
    /// BERT ONNX model (protectai, reliable).
    pub const BERT_ONNX: &str = "protectai/bert-base-NER-onnx";

    /// GLiNER small model (~50M params, fastest).
    pub const GLINER_SMALL: &str = "onnx-community/gliner_small-v2.1";

    /// GLiNER medium model (~110M params, balanced) - default.
    pub const GLINER_MEDIUM: &str = "onnx-community/gliner_medium-v2.1";

    /// GLiNER large model (~340M params, most accurate).
    pub const GLINER_LARGE: &str = "onnx-community/gliner_large-v2.1";

    /// GLiNER multitask model (relation extraction too).
    pub const GLINER_MULTITASK: &str = "onnx-community/gliner-multitask-large-v0.5";

    /// Candle BERT model.
    pub const CANDLE_BERT: &str = "dslim/bert-base-NER";
}

/// Standard entity types for NER.
pub const STANDARD_ENTITY_TYPES: &[&str] = &[
    "person",
    "organization",
    "location",
    "date",
    "money",
    "percent",
];

/// Backend type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BackendType {
    /// GLiNER via gline-rs (zero-shot, SOTA)
    GLiNER,
    /// BERT ONNX (reliable default)
    BertOnnx,
    /// Candle (Rust-native)
    Candle,
    /// Pattern-based only
    Pattern,
}

impl BackendType {
    /// Get human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            BackendType::GLiNER => "gliner",
            BackendType::BertOnnx => "bert-onnx",
            BackendType::Candle => "candle",
            BackendType::Pattern => "pattern",
        }
    }

    /// Check if this backend requires network for model download.
    #[must_use]
    pub fn requires_network(&self) -> bool {
        !matches!(self, BackendType::Pattern)
    }

    /// Check if this backend supports zero-shot NER.
    #[must_use]
    pub fn supports_zero_shot(&self) -> bool {
        matches!(self, BackendType::GLiNER)
    }
}

/// NER extractor with fallback support.
///
/// This is the recommended way to use NER in anno. It handles:
/// - Backend selection based on available features
/// - Graceful fallback when ML models fail
/// - Hybrid mode combining ML and patterns
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::extractor::NERExtractor;
///
/// // Automatic selection (best available)
/// let extractor = NERExtractor::best_available();
///
/// // Explicit backend
/// let extractor = NERExtractor::with_bert_onnx("protectai/bert-base-NER-onnx")?;
///
/// // Extract entities
/// let entities = extractor.extract("Apple announced new iPhone in Cupertino.", None)?;
/// ```
pub struct NERExtractor {
    /// Primary ML backend (optional)
    primary: Option<Arc<dyn Model>>,
    /// Fallback backend (always PatternNER)
    fallback: Arc<PatternNER>,
    /// Backend type identifier
    backend_type: BackendType,
}

impl NERExtractor {
    /// Create with explicit primary and fallback.
    pub fn new(primary: Option<Arc<dyn Model>>, backend_type: BackendType) -> Self {
        Self {
            primary,
            fallback: Arc::new(PatternNER::new()),
            backend_type,
        }
    }

    /// Create with pattern-based backend only.
    ///
    /// Fast (~400ns) but limited to structured entities:
    /// DATE, TIME, MONEY, PERCENT, EMAIL, URL, PHONE
    #[must_use]
    pub fn pattern_only() -> Self {
        Self {
            primary: None,
            fallback: Arc::new(PatternNER::new()),
            backend_type: BackendType::Pattern,
        }
    }

    /// Create the best available NER extractor.
    ///
    /// Tries backends in priority order:
    /// 1. GLiNER (if `onnx` feature enabled) - zero-shot, best accuracy (~90% CoNLL F1)
    /// 2. BERT ONNX (if `onnx` feature enabled) - reliable, fixed types (~86% F1)
    /// 3. Candle (if `candle` feature enabled) - Rust-native (~74% F1)
    /// 4. PatternNER (always) - structured entities only
    #[must_use]
    pub fn best_available() -> Self {
        // Try GLiNER first (best accuracy, zero-shot)
        #[cfg(feature = "onnx")]
        {
            if let Ok(extractor) = Self::with_gliner(defaults::GLINER_SMALL) {
                log::info!("[NER] Using GLiNER Small (~90% F1, zero-shot)");
                return extractor;
            }
            log::warn!("[NER] GLiNER init failed, trying BERT ONNX");

            // Fallback to BERT ONNX (reliable)
            if let Ok(extractor) = Self::with_bert_onnx(defaults::BERT_ONNX) {
                log::info!("[NER] Using BERT ONNX (~86% F1)");
                return extractor;
            }
            log::warn!("[NER] BERT ONNX init failed, trying Candle");
        }

        // Try Candle (Rust-native)
        #[cfg(feature = "candle")]
        {
            if let Ok(extractor) = Self::with_candle(defaults::CANDLE_BERT) {
                log::info!("[NER] Using Candle (~74% F1)");
                return extractor;
            }
            log::warn!("[NER] Candle init failed, falling back to patterns");
        }

        // Ultimate fallback: patterns only
        log::info!("[NER] Using PatternNER (structured entities only)");
        Self::pattern_only()
    }

    /// Create the fastest available NER extractor.
    ///
    /// Prioritizes speed over accuracy:
    /// 1. GLiNER small (if `onnx` feature) - fast zero-shot
    /// 2. PatternNER (always) - ~400ns per call
    #[must_use]
    pub fn fast() -> Self {
        #[cfg(feature = "onnx")]
        {
            if let Ok(extractor) = Self::with_gliner(defaults::GLINER_SMALL) {
                log::info!("[NER] Using GLiNER Small (fast mode)");
                return extractor;
            }
        }
        log::info!("[NER] Using PatternNER (structured entities only)");
        Self::pattern_only()
    }

    /// Create the highest quality NER extractor.
    ///
    /// Prioritizes accuracy over speed:
    /// 1. GLiNER large (if `onnx` feature) - highest accuracy
    /// 2. GLiNER medium (if `onnx` feature) - fallback
    /// 3. BERT ONNX (if `onnx` feature) - reliable
    /// 4. PatternNER (always)
    #[must_use]
    pub fn best_quality() -> Self {
        #[cfg(feature = "onnx")]
        {
            if let Ok(extractor) = Self::with_gliner(defaults::GLINER_LARGE) {
                log::info!("[NER] Using GLiNER Large (best quality)");
                return extractor;
            }
            if let Ok(extractor) = Self::with_gliner(defaults::GLINER_MEDIUM) {
                log::info!("[NER] Using GLiNER Medium");
                return extractor;
            }
            if let Ok(extractor) = Self::with_bert_onnx(defaults::BERT_ONNX) {
                log::info!("[NER] Using BERT ONNX");
                return extractor;
            }
        }
        log::info!("[NER] Using PatternNER (structured entities only)");
        Self::pattern_only()
    }

    /// Create with BERT ONNX backend.
    ///
    /// Uses standard BERT models fine-tuned for NER with BIO tagging.
    /// Reliable and widely tested, but limited to fixed entity types.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model identifier (e.g., "protectai/bert-base-NER-onnx")
    #[cfg(feature = "onnx")]
    pub fn with_bert_onnx(model_name: &str) -> Result<Self> {
        let bert = crate::backends::BertNEROnnx::new(model_name)?;
        Ok(Self {
            primary: Some(Arc::new(bert)),
            fallback: Arc::new(PatternNER::new()),
            backend_type: BackendType::BertOnnx,
        })
    }

    /// Stub for when onnx feature is disabled.
    #[cfg(not(feature = "onnx"))]
    pub fn with_bert_onnx(_model_name: &str) -> Result<Self> {
        Ok(Self::pattern_only())
    }

    /// Create with GLiNER backend (zero-shot NER).
    ///
    /// GLiNER is the **recommended backend** for best accuracy on named entities.
    /// It supports zero-shot NER (any entity type without retraining).
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model identifier (e.g., "onnx-community/gliner_small-v2.1")
    #[cfg(feature = "onnx")]
    pub fn with_gliner(model_name: &str) -> Result<Self> {
        let gliner = crate::backends::GLiNEROnnx::new(model_name)?;
        Ok(Self {
            primary: Some(Arc::new(gliner)),
            fallback: Arc::new(PatternNER::new()),
            backend_type: BackendType::GLiNER,
        })
    }

    /// Stub for when onnx feature is disabled.
    #[cfg(not(feature = "onnx"))]
    pub fn with_gliner(_model_name: &str) -> Result<Self> {
        Ok(Self::pattern_only())
    }

    /// Create with Candle backend (Rust-native transformers).
    ///
    /// Uses Candle ML framework to run transformer-based NER models.
    ///
    /// # Arguments
    /// * `model_name` - HuggingFace model identifier (e.g., "dslim/bert-base-NER")
    #[cfg(feature = "candle")]
    pub fn with_candle(model_name: &str) -> Result<Self> {
        let candle = crate::backends::CandleNER::new(model_name)?;
        Ok(Self {
            primary: Some(Arc::new(candle)),
            fallback: Arc::new(PatternNER::new()),
            backend_type: BackendType::Candle,
        })
    }

    /// Stub for when candle feature is disabled.
    #[cfg(not(feature = "candle"))]
    pub fn with_candle(_model_name: &str) -> Result<Self> {
        Ok(Self::pattern_only())
    }

    /// Extract entities with automatic fallback.
    ///
    /// Tries primary ML backend first, falls back to patterns if it fails.
    pub fn extract(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        // Try primary backend first
        if let Some(ref primary) = self.primary {
            if primary.is_available() {
                match primary.extract_entities(text, language) {
                    Ok(entities) if !entities.is_empty() => return Ok(entities),
                    Ok(_) => {
                        log::debug!("[NER] Primary returned empty, using fallback");
                    }
                    Err(e) => {
                        log::debug!("[NER] Primary failed ({}), using fallback", e);
                    }
                }
            }
        }

        // Fallback to patterns
        self.fallback.extract_entities(text, language)
    }

    /// Extract entities using hybrid strategy.
    ///
    /// Combines ML model (for semantic entities) with patterns (for structured entities):
    /// - ML: Person, Organization, Location (context-dependent)
    /// - Patterns: Date, Money, Percent, Email, URL, Phone (format-based)
    ///
    /// This gets best of both worlds:
    /// - High F1 on ambiguous entities (via ML)
    /// - 100% precision on pattern entities (via patterns)
    pub fn extract_hybrid(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Step 1: Get ML entities (context-dependent types)
        if let Some(ref primary) = self.primary {
            if primary.is_available() {
                if let Ok(ml_entities) = primary.extract_entities(text, language) {
                    // Keep only semantic entities from ML
                    entities.extend(
                        ml_entities
                            .into_iter()
                            .filter(|e| e.entity_type.requires_ml()),
                    );
                }
            }
        }

        // Step 2: Get pattern entities (structured types)
        if let Ok(pattern_entities) = self.fallback.extract_entities(text, language) {
            // Add pattern entities that don't overlap with ML
            for pe in pattern_entities {
                let overlaps = entities.iter().any(|e| {
                    // Check for span overlap: NOT (e ends before pe starts OR pe ends before e starts)
                    !(e.end <= pe.start || pe.end <= e.start)
                });
                if !overlaps {
                    entities.push(pe);
                }
            }
        }

        // Sort by position
        entities.sort_by_key(|e| e.start);

        Ok(entities)
    }

    /// Get the active backend type.
    #[must_use]
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    /// Get the name of the active backend.
    #[must_use]
    pub fn active_backend_name(&self) -> &'static str {
        if let Some(ref primary) = self.primary {
            if primary.is_available() {
                return primary.name();
            }
        }
        self.fallback.name()
    }

    /// Check if ML backend is available.
    #[must_use]
    pub fn has_ml_backend(&self) -> bool {
        self.primary.as_ref().is_some_and(|p| p.is_available())
    }

    /// Check if this extractor supports zero-shot NER.
    #[must_use]
    pub fn supports_zero_shot(&self) -> bool {
        self.backend_type.supports_zero_shot()
    }
}

// Make NERExtractor implement Model for compatibility
impl Model for NERExtractor {
    fn extract_entities(&self, text: &str, language: Option<&str>) -> Result<Vec<Entity>> {
        self.extract(text, language)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        if let Some(ref primary) = self.primary {
            if primary.is_available() {
                return primary.supported_types();
            }
        }
        self.fallback.supported_types()
    }

    fn is_available(&self) -> bool {
        true // Always available (has pattern fallback)
    }

    fn name(&self) -> &'static str {
        self.active_backend_name()
    }

    fn description(&self) -> &'static str {
        match self.backend_type {
            BackendType::GLiNER => "GLiNER zero-shot NER via gline-rs",
            BackendType::BertOnnx => "BERT NER via ONNX Runtime",
            BackendType::Candle => "BERT NER via Candle (Rust-native)",
            BackendType::Pattern => "Pattern-based NER (structured entities only)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_only() {
        let extractor = NERExtractor::pattern_only();
        assert_eq!(extractor.backend_type(), BackendType::Pattern);
        assert!(!extractor.has_ml_backend());
        assert!(!extractor.supports_zero_shot());
    }

    #[test]
    fn test_best_available_always_works() {
        // best_available should never panic, always falls back to patterns
        let extractor = NERExtractor::best_available();
        assert!(extractor.is_available());

        // Should extract pattern entities
        let text = "Meeting on 2024-01-15 cost $100.";
        let entities = extractor.extract(text, None).unwrap();
        let has_date = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Date));
        let has_money = entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Money));
        assert!(has_date || has_money, "Should find pattern entities");
    }

    #[test]
    fn test_backend_type_properties() {
        assert!(BackendType::GLiNER.requires_network());
        assert!(BackendType::BertOnnx.requires_network());
        assert!(BackendType::Candle.requires_network());
        assert!(!BackendType::Pattern.requires_network());

        assert!(BackendType::GLiNER.supports_zero_shot());
        assert!(!BackendType::BertOnnx.supports_zero_shot());
        assert!(!BackendType::Candle.supports_zero_shot());
        assert!(!BackendType::Pattern.supports_zero_shot());
    }

    #[test]
    fn test_extract_hybrid() {
        let extractor = NERExtractor::pattern_only();
        let text = "Meeting at 3:30 PM cost $50.";
        let entities = extractor.extract_hybrid(text, None).unwrap();

        // Should find pattern entities even without ML
        assert!(!entities.is_empty());
    }
}
