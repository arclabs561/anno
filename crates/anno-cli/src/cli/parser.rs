//! CLI argument parsing and structure definitions

use clap::{Parser, Subcommand, ValueEnum};

use crate::cli::commands;

/// Information Extraction CLI - NER, Coreference, Relations, Entity Linking
#[derive(Parser)]
#[command(name = "anno")]
#[command(
    author,
    version,
    about = "Information Extraction CLI - NER, Coreference, Relations, Entity Linking",
    long_about = r#"
anno - A unified information extraction toolkit

CAPABILITIES:
  • Named Entity Recognition (NER) - detect persons, orgs, locations, etc.
  • Coreference Resolution - link mentions to same entity ("She" → "Marie Curie")
  • Relation Extraction - extract (head, relation, tail) triples
  • Entity Linking - connect entities to knowledge bases (when configured)
  • Event Extraction - discourse-level event extraction

SIGNAL → TRACK → IDENTITY HIERARCHY:
  Level 1 (Signal)   : Raw detections/mentions with spans
  Level 2 (Track)    : Within-document coreference chains
  Level 3 (Identity) : Cross-document KB-linked entities

BACKENDS:
  • pattern    - High-precision patterns (dates, money, emails)
  • heuristic (alias: statistical) - Capitalization + context heuristics
  • stacked    - Best available stack (uses available feature-gated backends when enabled)
  • extra backends via feature flags (e.g., Candle); see `anno models list`

EXAMPLES:
  anno extract "Marie Curie was born in Paris."
  anno debug --coref --link-kb -t "Barack Obama met Angela Merkel. He discussed NATO."
  anno eval -t "..." -g "Marie Curie:PER:0:11"
  anno cross-doc ./docs --threshold 0.6
  anno coalesce ./docs --threshold 0.6  # alias for cross-doc
  anno tier --input graph.json --method leiden --levels 3
  anno info
"#
)]
#[command(propagate_version = true)]
pub struct Cli {
    /// The subcommand to execute.
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Text to extract entities from (shorthand for `anno extract`)
    #[arg(trailing_var_arg = true)]
    pub text: Vec<String>,
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
    Analyze(commands::AnalyzeArgs),

    /// Work with NER datasets
    #[command(visible_alias = "ds")]
    Dataset(commands::DatasetArgs),

    /// Comprehensive evaluation across all task-dataset-backend combinations
    #[command(visible_alias = "bench")]
    #[cfg(feature = "eval-advanced")]
    Benchmark(commands::BenchmarkArgs),

    /// Inspect muxer history from the randomized matrix harness
    #[cfg(feature = "eval-advanced")]
    Muxer(commands::MuxerArgs),

    /// Show model and version info
    #[command(visible_alias = "i")]
    Info,

    /// List and compare available models
    Models(commands::ModelsArgs),

    /// Cross-document entity coalescing: cluster entities across multiple documents
    #[command(visible_alias = "coalesce")]
    #[cfg(feature = "eval-advanced")]
    CrossDoc(commands::CrossDocArgs),

    /// Hierarchical clustering: reveal tier of abstraction
    #[cfg(feature = "eval-advanced")]
    Tier(commands::TierArgs),

    /// Enhance entities with additional metadata
    Enhance(commands::EnhanceArgs),

    /// Full processing pipeline
    #[command(visible_alias = "p")]
    Pipeline(commands::PipelineArgs),

    /// Query and filter entities/clusters
    #[command(visible_alias = "q")]
    Query(commands::QueryArgs),

    /// Compare documents, models, or clusters
    Compare(commands::CompareArgs),

    /// Manage cache for extraction results
    Cache(commands::CacheArgs),

    /// Query evaluation history
    #[cfg(feature = "eval")]
    History(commands::HistoryArgs),

    /// Manage configuration files for workflows
    Config(commands::ConfigArgs),

    /// Batch process multiple documents efficiently
    #[command(visible_alias = "b")]
    Batch(commands::BatchArgs),

    /// Joint NER + Coreference + Entity Linking analysis
    #[command(visible_alias = "j")]
    Joint(commands::JointArgs),

    /// Privacy: detect and redact PII
    #[command(visible_alias = "priv")]
    Privacy(commands::PrivacyArgs),

    /// Watch directory for incremental processing
    #[command(visible_alias = "w")]
    Watch(commands::WatchArgs),

    /// Domain shift detection
    Domain(commands::DomainArgs),

    /// Explain why entities were extracted
    Explain(commands::ExplainArgs),

    /// Analyze singleton coreference clusters
    Singleton(commands::SingletonArgs),

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
    /// BiLSTM + CRF (neural baseline, heuristic weights)
    #[value(alias = "bilstm-crf")]
    BiLstmCrf,
    /// TPLinker: joint entity-relation extraction
    #[value(alias = "tplink")]
    Tplinker,
    /// Universal NER: LLM-based zero-shot (requires API key)
    #[value(alias = "universal-ner")]
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
    GlinerPoly,

    // === Candle Feature Required ===
    /// GLiNER via Candle (requires --features candle)
    #[cfg(feature = "candle")]
    GlinerCandle,
    /// Candle BERT NER (requires --features candle)
    #[cfg(feature = "candle")]
    CandleNer,

    // === Burn Feature Required ===
    /// Burn ML framework NER (requires --features burn)
    #[cfg(feature = "burn")]
    #[value(alias = "burn-ner")]
    Burn,
}

impl ModelBackend {
    /// Create a model instance from this backend type.
    pub fn create_model(self) -> Result<Box<dyn anno::Model>, String> {
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
                Self::BiLstmCrf => "bilstm_crf",
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
                Self::GlinerPoly => "gliner_poly",
                // Candle
                #[cfg(feature = "candle")]
                Self::GlinerCandle => "gliner_candle",
                #[cfg(feature = "candle")]
                Self::CandleNer => "candle_ner",
                // Burn
                #[cfg(feature = "burn")]
                Self::Burn => "burn",
            };
            BackendFactory::create(factory_name)
                .map_err(|e| format!("Failed to create model '{}': {}", self.name(), e))
        }
        #[cfg(not(feature = "eval"))]
        {
            use crate::{AutoNER, HeuristicNER, RegexNER, StackedNER};
            match self {
                // Always available
                Self::Pattern => Ok(Box::new(RegexNER::new())),
                Self::Heuristic => Ok(Box::new(HeuristicNER::new())),
                Self::Minimal => Ok(Box::new(HeuristicNER::new())),
                Self::Auto => Ok(Box::new(AutoNER::new())),
                Self::Stacked => Ok(Box::new(StackedNER::default())),
                Self::Crf => Ok(Box::new(crate::backends::crf::CrfNER::new())),
                Self::Hmm => Ok(Box::new(crate::backends::hmm::HmmNER::new())),
                Self::Ensemble => Ok(Box::new(crate::backends::ensemble::EnsembleNER::default())),
                Self::BiLstmCrf => Ok(Box::new(crate::backends::bilstm_crf::BiLstmCrfNER::new())),
                Self::Tplinker => crate::backends::tplinker::TPLinker::new()
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!("Failed to create TPLinker: {}", e)),
                Self::UniversalNer => crate::backends::universal_ner::UniversalNER::new()
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!("Failed to create UniversalNER: {}", e)),
                // ONNX
                #[cfg(feature = "onnx")]
                Self::Gliner => crate::GLiNEROnnx::new(crate::DEFAULT_GLINER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!("Failed to load GLiNER: {}\n  Tip: Use 'anno models info gliner' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::Gliner2 => crate::backends::gliner2::GLiNER2Onnx::from_pretrained(crate::DEFAULT_GLINER2_MODEL)
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!("Failed to load GLiNER2: {}\n  Tip: Use 'anno models info gliner2' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::Nuner => crate::backends::nuner::NuNER::from_pretrained(crate::DEFAULT_NUNER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!("Failed to load NuNER: {}\n  Tip: Use 'anno models info nuner' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::W2ner => {
                    // Allow override via environment variable for custom/exported models
                    let model_path = std::env::var("W2NER_MODEL_PATH")
                        .unwrap_or_else(|_| crate::DEFAULT_W2NER_MODEL.to_string());
                    crate::backends::w2ner::W2NER::from_pretrained(&model_path)
                        .map(|m| Box::new(m) as Box<dyn crate::Model>)
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
                Self::BertOnnx => crate::backends::onnx::BertNEROnnx::new(crate::DEFAULT_BERT_ONNX_MODEL)
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!("Failed to load BERT ONNX: {}", e)),
                #[cfg(feature = "onnx")]
                Self::DebertaV3 => {
                    // Support custom export via environment variable
                    if let Ok(model_path) = std::env::var("DEBERTA_MODEL_PATH") {
                        crate::backends::deberta_v3::DeBERTaV3NER::new(&model_path)
                            .map(|m| Box::new(m) as Box<dyn crate::Model>)
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
                        crate::backends::albert::ALBERTNER::new(&model_path)
                            .map(|m| Box::new(m) as Box<dyn crate::Model>)
                            .map_err(|e| format!("ALBERT failed to load from {}: {}", model_path, e))
                    } else {
                        Err("ALBERT requires custom ONNX export.\n\
                             Ready alternatives: --model candle-ner, --model bert-onnx".to_string())
                    }
                }
                #[cfg(feature = "onnx")]
                Self::GlinerPoly => crate::backends::gliner_poly::GLiNERPoly::new("onnx-community/gliner_small-v2.1")
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!("Failed to load GLiNER Poly: {}", e)),
                // Candle
                #[cfg(feature = "candle")]
                Self::GlinerCandle => crate::backends::gliner_candle::GLiNERCandle::from_pretrained(crate::DEFAULT_GLINER_CANDLE_MODEL)
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!(
                        "GLiNER-Candle model unavailable: {}\n\n\
                         GLiNER Candle is experimental and has compatibility issues with\n\
                         most GLiNER models due to non-standard weight formats.\n\n\
                         Recommended: Use --model gliner (ONNX version) instead.\n\
                         It works with all GLiNER models and provides better performance.",
                        e
                    )),
                #[cfg(feature = "candle")]
                Self::CandleNer => crate::backends::candle::CandleNER::from_pretrained(crate::DEFAULT_CANDLE_MODEL)
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!(
                        "CandleNER model unavailable: {}\n\n\
                         The model may lack tokenizer.json or safetensors files.\n\n\
                         Alternatives:\n\
                         - Use --model bert-onnx (ONNX version, more compatible)\n\
                         - Use --model heuristic for pattern-based extraction",
                        e
                    )),
                // Burn
                #[cfg(feature = "burn")]
                Self::Burn => crate::backends::burn::BurnNER::new()
                    .map(|m| Box::new(m) as Box<dyn crate::Model>)
                    .map_err(|e| format!(
                        "BurnNER unavailable: {}\n\n\
                         BurnNER is experimental. For production use:\n\
                         - Use --model candle-ner for pure Rust inference\n\
                         - Use --model bert-onnx for ONNX inference",
                        e
                    ))
            }
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
            Self::BiLstmCrf => "bilstm-crf",
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
            // Burn
            #[cfg(feature = "burn")]
            Self::Burn => "burn",
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
}
