//! CLI argument parsing and structure definitions

use clap::{Parser, Subcommand, ValueEnum};

use crate::cli::commands;

/// Information extraction CLI (NER + coreference).
#[derive(Parser)]
#[command(name = "anno")]
#[command(
    author,
    version,
    about = "Information extraction CLI (NER + coreference)",
    long_about = r#"
anno - Information extraction from text

EXAMPLES:
  anno extract --text "Lynn Conway worked at IBM and Xerox PARC in California."
  anno extract --model gliner --extract-types "DRUG,SYMPTOM" \
    --text "Aspirin can treat headaches and reduce fever."
  anno extract --extract-relations --relation-types "FOUNDED,WORKS_FOR" \
    --text "Steve Jobs founded Apple in 1976."
  anno debug --coref -t "Sophie Wilson designed the ARM processor. She revolutionized computing."
  anno batch --dir ./docs --output ./results --format json
  anno models download ...
  anno info

OFFLINE:
  anno models download ...          # prefetch weights
  ANNO_NO_DOWNLOADS=1 anno ...      # cached-only mode

Run `anno help <command>` for details on any subcommand.
"#
)]
#[command(propagate_version = true)]
pub struct Cli {
    /// The subcommand to execute.
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Color output mode
    #[arg(long, global = true, default_value = "auto", value_name = "WHEN")]
    pub color: ColorMode,

    /// Text to extract entities from (shorthand for `anno extract`)
    #[arg(trailing_var_arg = true)]
    pub text: Vec<String>,
}

/// When to colorize output.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum ColorMode {
    /// Colorize when stdout is a terminal
    #[default]
    Auto,
    /// Always colorize
    Always,
    /// Never colorize
    Never,
}

/// Available CLI commands.
#[derive(Subcommand)]
pub enum Commands {
    /// Extract entities from text
    #[command(visible_alias = "x")]
    Extract(crate::cli::commands::ExtractArgs),

    /// Generate HTML debug visualization
    #[command(visible_alias = "d")]
    Debug(commands::DebugArgs),

    /// Evaluate predictions against gold annotations
    #[command(visible_alias = "e")]
    Eval(commands::EvalArgs),

    /// Validate JSONL annotation files
    #[command(visible_alias = "v")]
    Validate(commands::ValidateArgs),

    /// Deep analysis with multiple models
    #[command(visible_alias = "a")]
    #[command(hide = true)]
    Analyze(commands::AnalyzeArgs),

    /// Work with NER datasets
    #[command(visible_alias = "ds")]
    #[command(hide = true)]
    Dataset(commands::DatasetArgs),

    /// Comprehensive evaluation across all task-dataset-backend combinations
    #[command(visible_alias = "bench")]
    #[cfg(feature = "eval")]
    #[command(hide = true)]
    Benchmark(commands::BenchmarkArgs),

    /// Inspect sampler history from the randomized matrix sampler harness
    #[cfg(feature = "eval")]
    #[command(hide = true)]
    #[command(name = "sampler", visible_alias = "muxer")]
    Muxer(commands::MuxerArgs),

    /// Show model and version info
    #[command(visible_alias = "i")]
    Info,

    /// List and compare available models
    Models(commands::ModelsArgs),

    /// Cross-document entity coalescing: cluster entities across multiple documents
    #[command(visible_alias = "coalesce")]
    #[cfg(feature = "eval")]
    #[command(hide = true)]
    CrossDoc(commands::CrossDocArgs),

    /// Enhance entities with additional metadata
    #[command(hide = true)]
    Enhance(commands::EnhanceArgs),

    /// Full processing pipeline
    #[command(visible_alias = "p")]
    #[command(hide = true)]
    Pipeline(commands::PipelineArgs),

    /// Query and filter entities/clusters
    #[command(visible_alias = "q")]
    #[command(hide = true)]
    Query(commands::QueryArgs),

    /// Compare documents, models, or clusters
    #[command(hide = true)]
    Compare(commands::CompareArgs),

    /// Manage cache for extraction results
    #[command(hide = true)]
    Cache(commands::CacheArgs),

    /// Query evaluation history
    #[cfg(feature = "eval")]
    #[command(hide = true)]
    History(commands::HistoryArgs),

    /// Manage configuration files for workflows
    #[command(hide = true)]
    Config(commands::ConfigArgs),

    /// Batch process multiple documents efficiently
    #[command(visible_alias = "b")]
    #[command(hide = true)]
    Batch(commands::BatchArgs),

    /// Privacy: detect and redact PII
    #[command(visible_alias = "priv")]
    #[command(hide = true)]
    Privacy(commands::PrivacyArgs),

    /// Watch directory for incremental processing
    #[command(visible_alias = "w")]
    #[command(hide = true)]
    Watch(commands::WatchArgs),

    /// Domain shift detection
    #[command(hide = true)]
    Domain(commands::DomainArgs),

    /// Explain why entities were extracted
    #[command(hide = true)]
    Explain(commands::ExplainArgs),

    /// Analyze singleton coreference clusters
    #[command(hide = true)]
    Singleton(commands::SingletonArgs),

    /// Export annotations to different formats (brat, CoNLL, JSONL, etc.)
    #[command(visible_alias = "ex")]
    #[command(hide = true)]
    Export(commands::ExportArgs),

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

/// Model backend selection
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum ModelBackend {
    // === Always Available (Zero Dependencies) ===
    /// Regex matching only (dates, emails, etc.)
    #[value(alias = "regex")]
    Pattern,
    /// Heuristic NER (persons, orgs, locs via capitalization + context)
    #[value(alias = "statistical")]
    Heuristic,
    /// Minimal heuristic (low complexity experiment)
    Minimal,
    /// Automatic (Language-detected routing)
    Auto,
    /// Stacked: Pattern + Heuristic (default)
    #[default]
    Stacked,
    /// CRF sequence labeling (classical ML)
    Crf,
    /// Hidden Markov Model (classical statistical)
    Hmm,
    /// Ensemble: weighted voting across multiple backends
    Ensemble,
    /// CRF with heuristic emission features (gazetteer, word shape)
    #[value(alias = "heuristic-crf", alias = "bilstm-crf", alias = "bilstm_crf")]
    HeuristicCrf,
    /// TPLinker: joint entity-relation extraction
    #[value(alias = "tplink")]
    Tplinker,
    /// Universal NER: LLM-based zero-shot (requires API key)
    #[value(alias = "universal-ner", hide = true)]
    UniversalNer,

    // === ONNX Feature Required ===
    /// GLiNER via ONNX (enabled by default; disable with --no-default-features)
    #[cfg(feature = "onnx")]
    Gliner,
    /// GLiNER2 multi-task (NER + classification + structure)
    #[cfg(feature = "onnx")]
    Gliner2,
    /// NuNER
    #[cfg(feature = "onnx")]
    Nuner,
    /// W2NER for nested entities
    #[cfg(feature = "onnx")]
    W2ner,
    /// BERT ONNX for CoNLL-style NER
    #[cfg(feature = "onnx")]
    #[value(alias = "bert")]
    BertOnnx,
    /// DeBERTa-v3 NER
    #[cfg(feature = "onnx")]
    #[value(alias = "deberta")]
    DebertaV3,
    /// ALBERT NER
    #[cfg(feature = "onnx")]
    Albert,
    /// GLiNER Poly-Encoder
    #[cfg(feature = "onnx")]
    #[value(alias = "gliner-poly")]
    #[value(hide = true)]
    GlinerPoly,

    // === Candle Feature Required ===
    /// GLiNER via Candle (requires --features candle)
    #[cfg(feature = "candle")]
    GlinerCandle,
    /// Candle BERT NER (requires --features candle)
    #[cfg(feature = "candle")]
    CandleNer,
}

impl ModelBackend {
    /// Create a model instance from this backend type.
    pub fn create_model(self) -> Result<Box<dyn anno::Model>, String> {
        // Explicitly reject backends that are present for forward-compat wiring but do not
        // implement inference today. (They remain parseable by name, but are hidden from help.)
        #[cfg(feature = "onnx")]
        if matches!(self, Self::GlinerPoly) {
            return Err(
                "GLiNER Poly (`gliner-poly`) is scaffolding only and does not implement inference yet. \
Use `--model gliner` instead."
                    .to_string(),
            );
        }

        #[cfg(feature = "eval")]
        {
            use anno_eval::eval::backend_factory::BackendFactory;
            let factory_name = match self {
                // Always available
                Self::Pattern => "pattern",
                Self::Heuristic => "heuristic",
                Self::Minimal => "heuristic",
                Self::Auto => "stacked",
                Self::Stacked => "stacked",
                Self::Crf => "crf",
                Self::Hmm => "hmm",
                Self::Ensemble => "ensemble",
                Self::HeuristicCrf => "heuristic_crf",
                Self::Tplinker => "tplinker",
                Self::UniversalNer => "universal_ner",
                // ONNX
                #[cfg(feature = "onnx")]
                Self::Gliner => "gliner_onnx",
                #[cfg(feature = "onnx")]
                Self::Gliner2 => "gliner2",
                #[cfg(feature = "onnx")]
                Self::Nuner => "nuner",
                #[cfg(feature = "onnx")]
                Self::W2ner => "w2ner",
                #[cfg(feature = "onnx")]
                Self::BertOnnx => "bert_onnx",
                #[cfg(feature = "onnx")]
                Self::DebertaV3 => "deberta_v3",
                #[cfg(feature = "onnx")]
                Self::Albert => "albert",
                #[cfg(feature = "onnx")]
                Self::GlinerPoly => "gliner_poly", // rejected above; kept for exhaustiveness
                // Candle
                #[cfg(feature = "candle")]
                Self::GlinerCandle => "gliner_candle",
                #[cfg(feature = "candle")]
                Self::CandleNer => "candle_ner",
            };
            BackendFactory::create(factory_name)
                .map_err(|e| format!("Failed to create model '{}': {}\n  Tip: Run 'anno models list' to see available backends.", self.name(), e))
        }
        #[cfg(not(feature = "eval"))]
        {
            use anno::{HeuristicNER, RegexNER, StackedNER};
            match self {
                // Always available
                Self::Pattern => Ok(Box::new(RegexNER::new())),
                Self::Heuristic => Ok(Box::new(HeuristicNER::new())),
                Self::Minimal => Ok(Box::new(HeuristicNER::new())),
                Self::Auto => Ok(Box::new(StackedNER::default())),
                Self::Stacked => Ok(Box::new(StackedNER::default())),
                Self::Crf => Ok(Box::new(anno::backends::crf::CrfNER::new())),
                Self::Hmm => Ok(Box::new(anno::backends::hmm::HmmNER::new())),
                Self::Ensemble => Ok(Box::new(anno::backends::ensemble::EnsembleNER::default())),
                Self::HeuristicCrf => Ok(Box::new(anno::backends::heuristic_crf::HeuristicCrfNER::new())),
                Self::Tplinker => anno::backends::tplinker::TPLinker::new()
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| format!("Failed to create TPLinker: {}\n  Tip: Use 'anno models info tplinker' to check model status.", e)),
                Self::UniversalNer => anno::backends::universal_ner::UniversalNER::new()
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| format!("Failed to create UniversalNER: {}\n  Tip: Check API key with OPENROUTER_API_KEY or use --model gliner for offline NER.", e)),
                // ONNX
                #[cfg(feature = "onnx")]
                Self::Gliner => anno::GLiNEROnnx::new(anno::DEFAULT_GLINER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| format!("Failed to load GLiNER: {}\n  Tip: Use 'anno models info gliner' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::Gliner2 => anno::backends::gliner2::GLiNER2Onnx::from_pretrained(anno::DEFAULT_GLINER2_MODEL)
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| format!("Failed to load GLiNER2: {}\n  Tip: Use 'anno models info gliner2' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::Nuner => anno::backends::nuner::NuNER::from_pretrained(anno::DEFAULT_NUNER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| format!("Failed to load NuNER: {}\n  Tip: Use 'anno models info nuner' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::W2ner => {
                    // Allow override via environment variable for custom/exported models
                    let model_path = std::env::var("W2NER_MODEL_PATH")
                        .unwrap_or_else(|_| anno::DEFAULT_W2NER_MODEL.to_string());
                    anno::backends::w2ner::W2NER::from_pretrained(&model_path)
                        .map(|m| Box::new(m) as Box<dyn anno::Model>)
                        .map_err(|e| format!(
                            "W2NER model unavailable: {}\n\n\
                             Options:\n\
                             1. Set W2NER_MODEL_PATH to a local model directory\n\
                             2. Export your own model: uv run scripts/export_w2ner_to_onnx.py\n\
                             3. For HuggingFace models, set HF_TOKEN and request model access\n\n\
                             Alternatives:\n\
                             - Use --model gliner for zero-shot NER\n\
                             - Use --model gliner2 for nested entity support",
                            e
                        ))
                }
                #[cfg(feature = "onnx")]
                Self::BertOnnx => anno::backends::onnx::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL)
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| format!("Failed to load BERT ONNX: {}\n  Tip: Use 'anno models info bert-onnx' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::DebertaV3 => {
                    // Support custom export via environment variable
                    if let Ok(model_path) = std::env::var("DEBERTA_MODEL_PATH") {
                        anno::backends::deberta_v3::DeBERTaV3NER::new(&model_path)
                            .map(|m| Box::new(m) as Box<dyn anno::Model>)
                            .map_err(|e| format!("DeBERTa-v3 failed to load from {}: {}", model_path, e))
                    } else {
                        Err("DeBERTa-v3 requires custom ONNX export.\n\
                             Export: uv run scripts/export_deberta_ner_to_onnx.py\n\
                             Then: DEBERTA_MODEL_PATH=/path/to/model anno extract --model deberta-v3\n\n\
                             Ready alternatives: --model candle-ner, --model bert-onnx".to_string())
                    }
                }
                #[cfg(feature = "onnx")]
                Self::Albert => {
                    // Support custom export via environment variable
                    if let Ok(model_path) = std::env::var("ALBERT_MODEL_PATH") {
                        anno::backends::albert::ALBERTNER::new(&model_path)
                            .map(|m| Box::new(m) as Box<dyn anno::Model>)
                            .map_err(|e| format!("ALBERT failed to load from {}: {}", model_path, e))
                    } else {
                        Err("ALBERT requires custom ONNX export.\n\
                             Ready alternatives: --model candle-ner, --model bert-onnx".to_string())
                    }
                }
                #[cfg(feature = "onnx")]
                Self::GlinerPoly => unreachable!("rejected above"),
                // Candle
                #[cfg(feature = "candle")]
                Self::GlinerCandle => anno::backends::gliner_candle::GLiNERCandle::from_pretrained(anno::DEFAULT_GLINER_CANDLE_MODEL)
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| format!(
                        "GLiNER-Candle model unavailable: {}\n\n\
                         GLiNER Candle is experimental and has compatibility issues with\n\
                         most GLiNER models due to non-standard weight formats.\n\n\
                         Recommended: Use --model gliner (ONNX version) instead.\n\
                         It works with all GLiNER models and provides better performance.",
                        e
                    )),
                #[cfg(feature = "candle")]
                Self::CandleNer => anno::backends::candle::CandleNER::from_pretrained(anno::DEFAULT_CANDLE_MODEL)
                    .map(|m| Box::new(m) as Box<dyn anno::Model>)
                    .map_err(|e| format!(
                        "CandleNER model unavailable: {}\n\n\
                         The model may lack tokenizer.json or safetensors files.\n\n\
                         Alternatives:\n\
                         - Use --model bert-onnx (ONNX version, more compatible)\n\
                         - Use --model heuristic for pattern-based extraction",
                        e
                    )),
                // Note: `Burn` is rejected above.
            }
        }
    }

    /// Create a `RelationCapable` model instance, if this backend supports joint entity+relation
    /// extraction.  Returns `None` for all other backends (callers should fall back to
    /// `create_model()` + co-occurrence edges).
    pub fn try_create_relation_model(
        self,
    ) -> Option<Result<Box<dyn anno::RelationCapable>, String>> {
        match self {
            Self::Tplinker => Some(
                anno::backends::tplinker::TPLinker::new()
                    .map(|m| Box::new(m) as Box<dyn anno::RelationCapable>)
                    .map_err(|e| format!("Failed to create TPLinker: {}", e)),
            ),
            #[cfg(feature = "onnx")]
            Self::Gliner2 => Some(
                anno::backends::gliner2::GLiNER2Onnx::from_pretrained(anno::DEFAULT_GLINER2_MODEL)
                    .map(|m| Box::new(m) as Box<dyn anno::RelationCapable>)
                    .map_err(|e| {
                        format!(
                            "Failed to load GLiNER2 for relation extraction: {}\n  \
                             Tip: Use 'anno models info gliner2' to check model status.",
                            e
                        )
                    }),
            ),
            _ => None,
        }
    }

    /// Get the canonical string name for this backend.
    pub fn name(self) -> &'static str {
        match self {
            // Always available
            Self::Pattern => "pattern",
            Self::Heuristic => "heuristic",
            Self::Minimal => "minimal",
            Self::Auto => "auto",
            Self::Stacked => "stacked",
            Self::Crf => "crf",
            Self::Hmm => "hmm",
            Self::Ensemble => "ensemble",
            Self::HeuristicCrf => "heuristic-crf",
            Self::Tplinker => "tplinker",
            Self::UniversalNer => "universal-ner",
            // ONNX
            #[cfg(feature = "onnx")]
            Self::Gliner => "gliner",
            #[cfg(feature = "onnx")]
            Self::Gliner2 => "gliner2",
            #[cfg(feature = "onnx")]
            Self::Nuner => "nuner",
            #[cfg(feature = "onnx")]
            Self::W2ner => "w2ner",
            #[cfg(feature = "onnx")]
            Self::BertOnnx => "bert-onnx",
            #[cfg(feature = "onnx")]
            Self::DebertaV3 => "deberta-v3",
            #[cfg(feature = "onnx")]
            Self::Albert => "albert",
            #[cfg(feature = "onnx")]
            Self::GlinerPoly => "gliner-poly",
            // Candle
            #[cfg(feature = "candle")]
            Self::GlinerCandle => "gliner-candle",
            #[cfg(feature = "candle")]
            Self::CandleNer => "candle-ner",
        }
    }
}

/// Unified output format selection for all commands
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable colored output (default)
    #[default]
    Human,
    /// JSON object with provenance and entities: `{ "provenance": ..., "entities": [...] }`
    Json,
    /// JSON lines: first line is provenance, then one entity per line
    Jsonl,
    /// Tab-separated values
    Tsv,
    /// Inline annotations in text
    Inline,
    /// Full GroundedDocument as JSON (for pipeline integration)
    Grounded,
    /// HTML report (for debug/eval commands)
    Html,
    /// Tree structure (for cross-doc command)
    Tree,
    /// Summary statistics only (for cross-doc command)
    Summary,
}

/// Evaluation task type
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum EvalTask {
    /// Named Entity Recognition
    #[default]
    Ner,
    /// Coreference Resolution
    Coref,
    /// Relation Extraction
    Relation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_backend_names() {
        assert_eq!(ModelBackend::Pattern.name(), "pattern");
        assert_eq!(ModelBackend::Heuristic.name(), "heuristic");
        assert_eq!(ModelBackend::Stacked.name(), "stacked");
    }

    #[test]
    fn test_model_backend_default_is_stacked() {
        assert!(matches!(ModelBackend::default(), ModelBackend::Stacked));
    }

    /// Verify stacked default uses ML backends when available, not just naive pattern/heuristic.
    #[cfg(feature = "onnx")]
    #[test]
    fn test_stacked_default_has_ml_backend_when_available() {
        use anno::StackedNER;
        let ner = StackedNER::default();
        let stats = ner.stats();

        // With onnx AND model available: 3 layers (ML + regex + heuristic)
        // With onnx but offline/no model: 2 layers (regex + heuristic)
        if stats.layer_count == 3 {
            let has_ml = stats.layer_names.iter().any(|name| {
                let n = name.to_lowercase();
                n.contains("bert") || n.contains("gliner")
            });
            assert!(
                has_ml,
                "Default stacked with 3 layers should include ML backend. Layers: {:?}",
                stats.layer_names
            );
        }
        // 2 layers is acceptable fallback when ML models aren't available
    }

    #[test]
    fn test_output_format_default_is_human() {
        assert!(matches!(OutputFormat::default(), OutputFormat::Human));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_zero_shot_models_exist() {
        // Verify zero-shot capable models are available when onnx feature is enabled
        let gliner_name = ModelBackend::Gliner.name();
        assert_eq!(gliner_name, "gliner");

        let gliner2_name = ModelBackend::Gliner2.name();
        assert_eq!(gliner2_name, "gliner2");
    }
}
