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

use anno::{Model, Result};

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
            "pattern" | "patternner" | "regex" | "regexner" => Ok(Box::new(anno::RegexNER::new())),
            "heuristic" | "heuristicner" => Ok(Box::new(anno::HeuristicNER::new())),
            "stacked" | "stackedner" => Ok(Box::new(anno::StackedNER::default())),
            "crf" | "crfner" => Ok(Box::new(anno::backends::crf::CrfNER::new())),
            "hmm" | "hmmner" => Ok(Box::new(anno::backends::hmm::HmmNER::new())),
            "ensemble" | "ensemblener" => {
                use anno::backends::ensemble::EnsembleNER;
                Ok(Box::new(EnsembleNER::default()) as Box<dyn Model>)
            }
            "heuristic_crf" | "heuristic-crf" | "heuristiccrfner"
            | "bilstm_crf" | "bilstm-crf" | "bilstmcrf" | "bilstmcrfner" => {
                use anno::backends::heuristic_crf::HeuristicCrfNER;
                Ok(Box::new(HeuristicCrfNER::new()) as Box<dyn Model>)
            }

            // ONNX backends
            #[cfg(feature = "onnx")]
            "bert_onnx" | "bertneronnx" => {
                use anno::backends::onnx::BertNEROnnx;
                use crate::DEFAULT_BERT_ONNX_MODEL;
                BertNEROnnx::new(DEFAULT_BERT_ONNX_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        anno::Error::FeatureNotAvailable(format!(
                            "Failed to create BertNEROnnx: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "bert_onnx" | "bertneronnx" => Err(anno::Error::FeatureNotAvailable(
                "BertNEROnnx requires 'onnx' feature".to_string(),
            )),

            #[cfg(feature = "onnx")]
            "gliner" => {
                // First-class alias: prefer ONNX when available.
                use anno::backends::gliner_onnx::GLiNEROnnx;
                use crate::DEFAULT_GLINER_MODEL;
                GLiNEROnnx::new(DEFAULT_GLINER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        anno::Error::FeatureNotAvailable(format!(
                            "Failed to create GLiNER (onnx): {}",
                            e
                        ))
                    })
            }
            #[cfg(all(not(feature = "onnx"), feature = "candle"))]
            "gliner" => {
                // Fallback alias: Candle implementation when ONNX isn't enabled.
                use anno::backends::gliner_candle::GLiNERCandle;
                use crate::DEFAULT_GLINER_CANDLE_MODEL;
                GLiNERCandle::from_pretrained(DEFAULT_GLINER_CANDLE_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        anno::Error::FeatureNotAvailable(format!(
                            "Failed to create GLiNER (candle): {}",
                            e
                        ))
                    })
            }
            #[cfg(all(not(feature = "onnx"), not(feature = "candle")))]
            "gliner" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER requires 'onnx' (preferred) or 'candle' feature".to_string(),
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

            // B2NER (COLING 2025, unified NER training on 54 datasets)
            // Note: only LLM-scale models (7B/20B) available on HuggingFace.
            // Requires LLM backend, not ONNX. Pending encoder-scale release.
            "b2ner" => Err(crate::Error::FeatureNotAvailable(
                "B2NER only has LLM-scale models (7B/20B) on HuggingFace. \
                 Encoder-scale ONNX weights pending release."
                    .to_string(),
            )),

            // GLiNER PII Edge (60+ PII categories, zero-shot)
            #[cfg(feature = "onnx")]
            "gliner_pii" | "pii_ml" => {
                use crate::backends::gliner_onnx::GLiNEROnnx;
                GLiNEROnnx::new(anno::models::GLINER_PII)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "GLiNER PII Edge model unavailable: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "gliner_pii" | "pii_ml" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER PII requires 'onnx' feature".to_string(),
            )),

            // GLiNER-RelEx (joint NER + relation extraction, zero-shot)
            #[cfg(feature = "onnx")]
            "gliner_relex" | "relex" => {
                use crate::backends::gliner_onnx::GLiNEROnnx;
                GLiNEROnnx::new(anno::models::GLINER_RELEX)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "GLiNER-RelEx model unavailable: {}\n\n\
                             Export: uv run scripts/export_glirel_onnx.py",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "gliner_relex" | "relex" => Err(crate::Error::FeatureNotAvailable(
                "GLiNER-RelEx requires 'onnx' feature".to_string(),
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
            "nuner_4k" | "nunerzero4k" => {
                use crate::backends::nuner::NuNER;
                NuNER::from_pretrained("numind/NuNER_Zero-4k")
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "Failed to create NuNER 4k: {}",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "nuner_4k" | "nunerzero4k" => Err(crate::Error::FeatureNotAvailable(
                "NuNER 4k requires 'onnx' feature".to_string(),
            )),

            #[cfg(feature = "onnx")]
            "w2ner" => {
                use crate::backends::w2ner::W2NER;
                use crate::DEFAULT_W2NER_MODEL;
                // Allow override via environment variable for custom/exported models
                let model_path = std::env::var("W2NER_MODEL_PATH")
                    .unwrap_or_else(|_| DEFAULT_W2NER_MODEL.to_string());
                W2NER::from_pretrained(&model_path)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::FeatureNotAvailable(format!(
                            "W2NER model unavailable: {}\n\n\
                             Options:\n\
                             1. Set W2NER_MODEL_PATH to a local model directory\n\
                             2. Export your own: uv run scripts/export_w2ner_to_onnx.py\n\
                             3. For HF models, set HF_TOKEN and request access",
                            e
                        ))
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
                            "CandleNER model unavailable: {}",
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
                            "GLiNERCandle model unavailable: {}",
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

            // TPLinker (ONNX neural with `onnx` feature, heuristic fallback otherwise)
            "tplinker" | "tplink" => {
                use anno::backends::tplinker::TPLinker;
                Ok(Box::new(TPLinker::new()?) as Box<dyn Model>)
            }

            // Poly-Encoder GLiNER (requires onnx)
            #[cfg(feature = "onnx")]
            "gliner_poly" | "gliner-poly" | "poly_gliner" => {
                use anno::backends::gliner_poly::GLiNERPoly;
                use anno::DEFAULT_GLINER_POLY_MODEL;
                GLiNERPoly::new(DEFAULT_GLINER_POLY_MODEL)
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| crate::Error::model_init(e.to_string()))
            }
            #[cfg(not(feature = "onnx"))]
            "gliner_poly" | "gliner-poly" | "poly_gliner" => Err(crate::Error::FeatureNotAvailable(
                "GLiNERPoly requires 'onnx' feature".to_string(),
            )),

            // DeBERTa-v3 NER (requires onnx)
            #[cfg(feature = "onnx")]
            "deberta_v3" | "deberta-v3" | "deberta" => {
                use crate::backends::deberta_v3::DeBERTaV3NER;
                // Require an explicit local/exported ONNX model path.
                let Ok(model_path) = std::env::var("DEBERTA_MODEL_PATH") else {
                    return Err(crate::Error::FeatureNotAvailable(
                        "DeBERTa-v3 backend requires a local ONNX export. Set DEBERTA_MODEL_PATH (e.g. after running `uv run scripts/export_deberta_ner_to_onnx.py`)."
                            .to_string(),
                    ));
                };
                DeBERTaV3NER::new(&model_path)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::Retrieval(format!(
                            "DeBERTa-v3 model unavailable: {}\n\n\
                             Options:\n\
                             1. Export your own: uv run scripts/export_deberta_ner_to_onnx.py\n\
                             2. Set DEBERTA_MODEL_PATH to a local model directory\n\
                             3. Use --model bert-onnx or --model candle-ner instead",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "deberta_v3" | "deberta-v3" | "deberta" => Err(crate::Error::FeatureNotAvailable(
                "DeBERTa-v3 NER requires 'onnx' feature".to_string(),
            )),

            // ALBERT NER (requires onnx)
            #[cfg(feature = "onnx")]
            "albert" | "albert_ner" => {
                use crate::backends::albert::ALBERTNER;
                // Require an explicit local/exported ONNX model path.
                let Ok(model_path) = std::env::var("ALBERT_MODEL_PATH") else {
                    return Err(crate::Error::FeatureNotAvailable(
                        "ALBERT backend requires a local ONNX export. Set ALBERT_MODEL_PATH to a local model directory containing ONNX weights."
                            .to_string(),
                    ));
                };
                ALBERTNER::new(&model_path)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| {
                        crate::Error::Retrieval(format!(
                            "ALBERT model unavailable: {}\n\n\
                             Options:\n\
                             1. Export your own ONNX model\n\
                             2. Set ALBERT_MODEL_PATH to a local model directory\n\
                             3. Use --model bert-onnx or --model candle-ner instead",
                            e
                        ))
                    })
            }
            #[cfg(not(feature = "onnx"))]
            "albert" | "albert_ner" => Err(crate::Error::FeatureNotAvailable(
                "ALBERT NER requires 'onnx' feature".to_string(),
            )),

            // UniversalNER (LLM-backed zero-shot, requires `llm` feature + API key)
            "universal_ner" | "universal-ner" | "universalner" => {
                use anno::backends::universal_ner::UniversalNER;
                let m = UniversalNER::new()?;
                if !m.is_available() {
                    return Err(crate::Error::FeatureNotAvailable(
                        "UniversalNER requires the `llm` feature and a non-empty API key. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, or UNIVERSAL_NER_API_KEY."
                            .to_string(),
                    ));
                }
                Ok(Box::new(m) as Box<dyn Model>)
            }

            // Unknown backend
            _ => Err(crate::Error::InvalidInput(format!(
                "Unknown backend: '{}'. Available: pattern, heuristic, stacked, crf, hmm, ensemble, heuristic_crf, tplinker{}",
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
            "crf",
            "hmm",
            "ensemble",
            "heuristic_crf",
            "tplinker",
        ];

        // UniversalNER requires the optional `llm` feature plus a non-empty API key.
        // If either is missing, treat it as unavailable to avoid “Feature not available”
        // failures in the matrix harness.
        if cfg!(feature = "llm") {
            anno::env::load_dotenv();
            if anno::env::has_llm_api_key() || std::env::var("UNIVERSAL_NER_API_KEY").is_ok() {
                backends.push("universal_ner");
            }
        }

        #[cfg(feature = "onnx")]
        {
            backends.extend(&[
                "bert_onnx",
                "gliner",
                "gliner_onnx",
                "nuner",
                "nuner_4k",
                "b2ner",
                "w2ner",
                "gliner2",
                "gliner_pii",
                "gliner_relex",
                "gliner_poly",
            ]);

            // Optional backends that require explicit local ONNX exports.
            if std::env::var("DEBERTA_MODEL_PATH").is_ok() {
                backends.push("deberta_v3");
            }
            if std::env::var("ALBERT_MODEL_PATH").is_ok() {
                backends.push("albert");
            }
        }

        #[cfg(feature = "candle")]
        {
            backends.extend(&["candle_ner", "gliner_candle"]);
            // `gliner` is also available as an alias when candle is enabled
            // (and onnx is not required).
            if !cfg!(feature = "onnx") {
                backends.push("gliner");
            }
        }

        #[cfg(all(feature = "candle", feature = "onnx"))]
        {
            backends.push("gliner2_candle");
        }

        backends
    }

    /// List all available coreference resolvers.
    ///
    /// Coreference resolvers are *not* `Model`s, so they are kept separate from
    /// [`Self::available_backends`]. They are used by `TaskEvaluator` for coref-family tasks.
    #[must_use]
    pub fn available_coref_resolvers() -> Vec<&'static str> {
        vec!["coref_resolver", "mention_ranking"]
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
        "mention_ranking" | "mention-ranking" | "mentionranking" => {
            use anno::backends::coref::mention_ranking::MentionRankingCoref;
            Ok(Box::new(MentionRankingCoref::new()))
        }
        _ => Err(crate::Error::InvalidInput(format!(
            "Unknown coreference resolver: '{}'. Available: coref_resolver, mention_ranking",
            name
        ))),
    }
}

/// Create a text-based coreference backend (CorefBackend trait).
///
/// Unlike `create_coref_resolver` which takes pre-extracted entities,
/// `CorefBackend` operates on raw text and returns mention clusters directly.
/// This is the interface used by neural coref models (FCoref, MentionRanking).
pub fn create_coref_backend(
    name: &str,
) -> Result<Box<dyn anno::CorefBackend>> {
    match name.to_lowercase().as_str() {
        "mention_ranking" | "mention-ranking" | "mentionranking" => {
            use anno::backends::coref::mention_ranking::MentionRankingCoref;
            Ok(Box::new(MentionRankingCoref::new()))
        }
        #[cfg(feature = "onnx")]
        "fcoref" | "f-coref" | "fastcoref" => {
            use anno::backends::coref::fcoref::FCoref;
            let model_path = std::env::var("FCOREF_MODEL_PATH").ok();
            let fcoref = if let Some(path) = model_path {
                FCoref::from_path(&path)?
            } else {
                FCoref::from_pretrained("biu-nlp/f-coref")?
            };
            Ok(Box::new(fcoref))
        }
        #[cfg(not(feature = "onnx"))]
        "fcoref" | "f-coref" | "fastcoref" => Err(crate::Error::FeatureNotAvailable(
            "FCoref requires 'onnx' feature. Export: uv run scripts/export_fcoref.py".to_string(),
        )),
        _ => Err(crate::Error::InvalidInput(format!(
            "Unknown coref backend: '{}'. Available: mention_ranking, fcoref",
            name
        ))),
    }
}

/// List available coref backends (text-based CorefBackend).
pub fn available_coref_backends() -> Vec<&'static str> {
    let mut backends = vec!["mention_ranking"];
    #[cfg(feature = "onnx")]
    {
        backends.push("fcoref");
    }
    backends
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

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_backend_factory_pattern_returns_regex_only() {
        let model = BackendFactory::create("pattern").unwrap();
        println!("Model name: {}", model.name());
        assert_eq!(model.name(), "regex", "pattern should return RegexNER");

        let entities = model
            .extract_entities("John Smith went to Paris", None)
            .unwrap();
        println!("Entities: {:?}", entities);

        // Should NOT have PER or LOC
        for e in &entities {
            assert!(
                !matches!(e.entity_type, crate::EntityType::Person),
                "Unexpected Person entity: {:?}",
                e
            );
            assert!(
                !matches!(e.entity_type, crate::EntityType::Location),
                "Unexpected Location entity: {:?}",
                e
            );
        }
    }
}
