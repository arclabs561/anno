//! Backend Factory for Runtime Backend Creation
//!
//! This module provides a factory pattern for creating backend instances
//! from string names, enabling dynamic backend selection for evaluation.
//!
//! # Design Philosophy
//!
//! - **Feature-aware**: Only creates backends when features are enabled
//! - **Graceful degradation**: Returns errors for unavailable backends
//! - **Model defaults**: Uses sensible default models for each backend
//! - **Trait-based**: Returns trait objects for polymorphic usage

use crate::{Model, Result};

/// Factory for creating backend instances from names.
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend instance from a name.
    ///
    /// # Supported Backends
    ///
    /// ## Always Available
    /// - `pattern` / `RegexNER` - Pattern-based NER
    /// - `heuristic` / `HeuristicNER` - Heuristic NER
    /// - `stacked` / `StackedNER` - Stacked NER
    ///
    /// ## ONNX Feature Required
    /// - `bert_onnx` / `BertNEROnnx` - BERT ONNX NER
    /// - `gliner_onnx` / `GLiNEROnnx` - GLiNER ONNX (zero-shot)
    /// - `nuner` / `NuNER` - NuNER (zero-shot, token-based)
    /// - `w2ner` / `W2NER` - W2NER (discontinuous NER)
    /// - `gliner2` / `GLiNER2Onnx` - GLiNER2 multi-task
    ///
    /// ## Candle Feature Required
    /// - `candle_ner` / `CandleNER` - Candle BERT NER
    /// - `gliner_candle` / `GLiNERCandle` - GLiNER Candle (zero-shot)
    /// - `gliner2_candle` / `GLiNER2Candle` - GLiNER2 Candle
    ///
    /// ## Coreference
    /// - `coref_resolver` / `SimpleCorefResolver` - Simple coreference resolver
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use anno::eval::backend_factory::BackendFactory;
    ///
    /// let backend = BackendFactory::create("pattern")?;
    /// let entities = backend.extract_entities("Meeting on Jan 15", None)?;
    /// ```
    pub fn create(backend_name: &str) -> Result<Box<dyn Model>> {
        match backend_name.to_lowercase().as_str() {
            // Always available backends
            "pattern" | "patternner" | "regex" | "regexner" => Ok(Box::new(crate::RegexNER::new())),
            "heuristic" | "heuristicner" => Ok(Box::new(crate::HeuristicNER::new())),
            "stacked" | "stackedner" => Ok(Box::new(crate::StackedNER::default())),

            // ONNX backends
            #[cfg(feature = "onnx")]
            "bert_onnx" | "bertneronnx" => {
                use crate::backends::onnx::BertNEROnnx;
                use crate::DEFAULT_BERT_ONNX_MODEL;
                BertNEROnnx::new(DEFAULT_BERT_ONNX_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "Failed to create BertNEROnnx: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "bert_onnx" | "bertneronnx" => Err(crate::Error::FeatureNotAvailable(
                "BertNEROnnx requires 'onnx' feature".to_string(),
            )),

            #[cfg(feature = "onnx")]
            "gliner_onnx" | "glineronnx" => {
                use crate::backends::gliner_onnx::GLiNEROnnx;
                use crate::DEFAULT_GLINER_MODEL;
                GLiNEROnnx::new(DEFAULT_GLINER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "Failed to create GLiNEROnnx: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "gliner_onnx" | "glineronnx" => Err(crate::Error::FeatureNotAvailable(
                "GLiNEROnnx requires 'onnx' feature".to_string(),
            )),

            #[cfg(feature = "onnx")]
            "nuner" | "nunerzero" => {
                use crate::backends::nuner::NuNER;
                use crate::DEFAULT_NUNER_MODEL;
                NuNER::from_pretrained(DEFAULT_NUNER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!("Failed to create NuNER: {}", e))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "nuner" | "nunerzero" => Err(crate::Error::FeatureNotAvailable(
                "NuNER requires 'onnx' feature".to_string(),
            )),

            #[cfg(feature = "onnx")]
            "w2ner" => {
                use crate::backends::w2ner::W2NER;
                use crate::DEFAULT_W2NER_MODEL;
                W2NER::from_pretrained(DEFAULT_W2NER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!("Failed to create W2NER: {}", e))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "w2ner" => Err(crate::Error::FeatureNotAvailable(
                "W2NER requires 'onnx' feature".to_string(),
            )),

            #[cfg(feature = "onnx")]
            "gliner2" | "gliner2onnx" => {
                use crate::backends::gliner2::GLiNER2Onnx;
                use crate::DEFAULT_GLINER2_MODEL;
                GLiNER2Onnx::from_pretrained(DEFAULT_GLINER2_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "Failed to create GLiNER2Onnx: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "gliner2" | "gliner2onnx" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER2Onnx requires 'onnx' feature".to_string(),
            )),

            // Candle backends
            #[cfg(feature = "candle")]
            "candle_ner" | "candlener" => {
                use crate::backends::candle::CandleNER;
                use crate::DEFAULT_CANDLE_MODEL;
                CandleNER::from_pretrained(DEFAULT_CANDLE_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "Failed to create CandleNER: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "candle"))]
            "candle_ner" | "candlener" => Err(crate::Error::FeatureNotAvailable(
                "CandleNER requires 'candle' feature".to_string(),
            )),

            #[cfg(feature = "candle")]
            "gliner_candle" | "glinercandle" => {
                use crate::backends::gliner_candle::GLiNERCandle;
                use crate::DEFAULT_GLINER_CANDLE_MODEL;
                GLiNERCandle::from_pretrained(DEFAULT_GLINER_CANDLE_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "Failed to create GLiNERCandle: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "candle"))]
            "gliner_candle" | "glinercandle" => Err(crate::Error::FeatureNotAvailable(
                "GLiNERCandle requires 'candle' feature".to_string(),
            )),

            #[cfg(all(feature = "candle", feature = "onnx"))]
            "gliner2_candle" | "gliner2candle" => {
                use crate::backends::gliner2::GLiNER2Candle;
                use crate::DEFAULT_GLINER2_MODEL;
                GLiNER2Candle::from_pretrained(DEFAULT_GLINER2_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "Failed to create GLiNER2Candle: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(all(feature = "candle", feature = "onnx")))]
            "gliner2_candle" | "gliner2candle" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER2Candle requires both 'candle' and 'onnx' features".to_string(),
            )),

            // TPLinker (always available - placeholder implementation)
            "tplinker" | "tplink" => {
                use crate::backends::tplinker::TPLinker;
                Ok(Box::new(TPLinker::new()?) as Box<dyn Model>)
            }

            // Poly-Encoder GLiNER (requires onnx)
            #[cfg(feature = "onnx")]
            "gliner_poly" | "gliner-poly" | "poly_gliner" => {
                use crate::backends::gliner_poly::GLiNERPoly;
                // Use default GLiNER model for now (poly-encoder models pending)
                Ok(
                    Box::new(GLiNERPoly::new("onnx-community/gliner_small-v2.1")?)
                        as Box<dyn Model>,
                )
            }

            // DeBERTa-v3 NER (requires onnx)
            #[cfg(feature = "onnx")]
            "deberta_v3" | "deberta-v3" | "deberta" => {
                use crate::backends::deberta_v3::DeBERTaV3NER;
                Ok(Box::new(DeBERTaV3NER::new("microsoft/deberta-v3-base")?) as Box<dyn Model>)
            }

            // ALBERT NER (requires onnx)
            #[cfg(feature = "onnx")]
            "albert" | "albert_ner" => {
                use crate::backends::albert::ALBERTNER;
                Ok(Box::new(ALBERTNER::new("albert-base-v2")?) as Box<dyn Model>)
            }

            // UniversalNER (placeholder - LLM integration pending)
            "universal_ner" | "universal-ner" | "universalner" => {
                use crate::backends::universal_ner::UniversalNER;
                Ok(Box::new(UniversalNER::new()?) as Box<dyn Model>)
            }

            // Unknown backend
            _ => Err(crate::Error::InvalidInput(format!(
                "Unknown backend: '{}'. Available: pattern, heuristic, stacked, tplinker{}",
                backend_name,
                if cfg!(feature = "onnx") {
                    ", bert_onnx, gliner_onnx, nuner, w2ner, gliner2"
                } else {
                    ""
                }
            ))),
        }
    }

    /// List all available backends (based on enabled features).
    #[must_use]
    pub fn available_backends() -> Vec<&'static str> {
        #[allow(unused_mut)] // mut needed for extend/push calls
        let mut backends = vec![
            "pattern",
            "heuristic",
            "stacked",
            "tplinker",
            "universal_ner",
        ];

        #[cfg(feature = "onnx")]
        {
            backends.extend(&[
                "bert_onnx",
                "gliner_onnx",
                "nuner",
                "w2ner",
                "gliner2",
                "gliner_poly",
                "deberta_v3",
                "albert",
            ]);
        }

        #[cfg(feature = "candle")]
        {
            backends.extend(&["candle_ner", "gliner_candle"]);
        }

        #[cfg(all(feature = "candle", feature = "onnx"))]
        {
            backends.push("gliner2_candle");
        }

        backends
    }

    /// Check if a backend is available (feature-enabled).
    #[must_use]
    pub fn is_available(backend_name: &str) -> bool {
        Self::available_backends().contains(&backend_name.to_lowercase().as_str())
    }
}

/// Helper to create a coreference resolver from a name.
///
/// Note: Coreference resolvers don't implement `Model`, so this is separate.
pub fn create_coref_resolver(
    name: &str,
) -> Result<Box<dyn crate::eval::coref_resolver::CoreferenceResolver>> {
    match name.to_lowercase().as_str() {
        "coref_resolver" | "simplecorefresolver" | "simple" => {
            use crate::eval::coref_resolver::{CorefConfig, SimpleCorefResolver};
            Ok(Box::new(SimpleCorefResolver::new(CorefConfig::default())))
        }
        "box" | "box_coref" | "boxcorefresolver" => {
            use crate::backends::box_embeddings::BoxCorefConfig;
            use crate::eval::coref_resolver::BoxCorefResolver;
            Ok(Box::new(BoxCorefResolver::new(BoxCorefConfig::default())))
        }
        _ => Err(crate::Error::InvalidInput(format!(
            "Unknown coreference resolver: '{}'. Available: coref_resolver, box",
            name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_backend() {
        let backend = BackendFactory::create("pattern");
        assert!(backend.is_ok());
    }

    #[test]
    fn test_heuristic_backend() {
        let backend = BackendFactory::create("heuristic");
        assert!(backend.is_ok());
    }

    #[test]
    fn test_stacked_backend() {
        let backend = BackendFactory::create("stacked");
        assert!(backend.is_ok());
    }

    #[test]
    fn test_unknown_backend() {
        let backend = BackendFactory::create("nonexistent");
        assert!(backend.is_err());
    }

    #[test]
    fn test_available_backends() {
        let backends = BackendFactory::available_backends();
        assert!(backends.contains(&"pattern"));
        assert!(backends.contains(&"heuristic"));
        assert!(backends.contains(&"stacked"));
    }
}
