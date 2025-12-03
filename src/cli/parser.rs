//! CLI argument parsing and structure definitions

use clap::{Parser, Subcommand, ValueEnum};

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
  • Entity Linking - connect entities to knowledge bases (Wikidata)
  • Event Extraction - discourse-level events and shell nouns

SIGNAL → TRACK → IDENTITY HIERARCHY:
  Level 1 (Signal)   : Raw detections/mentions with spans
  Level 2 (Track)    : Within-document coreference chains  
  Level 3 (Identity) : Cross-document KB-linked entities

BACKENDS:
  • pattern    - High-precision patterns (dates, money, emails)
  • heuristic (alias: statistical) - Capitalization + context heuristics
  • gliner     - Zero-shot NER via ONNX (any entity type)
  • w2ner      - Nested/discontinuous entities

EXAMPLES:
  anno extract "Marie Curie won the Nobel Prize."
  anno debug --coref --link-kb -t "Barack Obama met Angela Merkel. He discussed NATO."
  anno eval -t "..." -g "Marie Curie:PER:0:11"
  anno info
"#
)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Text to extract entities from (shorthand for `anno extract`)
    #[arg(trailing_var_arg = true)]
    pub text: Vec<String>,
}

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

    /// Show model and version info
    #[command(visible_alias = "i")]
    Info,

    /// List and compare available models
    Models(commands::ModelsArgs),

    /// Cross-document coreference: cluster entities across multiple documents
    #[cfg(feature = "eval-advanced")]
    CrossDoc(commands::CrossDocArgs),

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

    /// Manage configuration files for workflows
    Config(commands::ConfigArgs),

    /// Batch process multiple documents efficiently
    #[command(visible_alias = "b")]
    Batch(commands::BatchArgs),

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
    /// Regex matching only (dates, emails, etc.)
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
    /// GLiNER via ONNX (requires --features onnx)
    #[cfg(feature = "onnx")]
    Gliner,
    /// GLiNER2 multi-task (NER + classification + structure, requires --features onnx)
    #[cfg(feature = "onnx")]
    Gliner2,
    /// NuNER (requires --features onnx)
    #[cfg(feature = "onnx")]
    Nuner,
    /// W2NER for nested entities (requires --features onnx)
    #[cfg(feature = "onnx")]
    W2ner,
    /// GLiNER via Candle (requires --features candle)
    #[cfg(feature = "candle")]
    GlinerCandle,
}

impl ModelBackend {
    pub fn create_model(self) -> Result<Box<dyn anno::Model>, String> {
        #[cfg(feature = "eval")]
        {
            use anno::eval::backend_factory::BackendFactory;
            let factory_name = match self {
                Self::Pattern => "pattern",
                Self::Heuristic => "heuristic",
                Self::Minimal => "heuristic",
                Self::Auto => "stacked",
                Self::Stacked => "stacked",
                #[cfg(feature = "onnx")]
                Self::Gliner => "gliner_onnx",
                #[cfg(feature = "onnx")]
                Self::Gliner2 => "gliner2",
                #[cfg(feature = "onnx")]
                Self::Nuner => "nuner",
                #[cfg(feature = "onnx")]
                Self::W2ner => "w2ner",
                #[cfg(feature = "candle")]
                Self::GlinerCandle => "gliner_candle",
            };
            return BackendFactory::create(factory_name)
                .map_err(|e| format!("Failed to create model '{}': {}", self.name(), e));
        }
        #[cfg(not(feature = "eval"))]
        {
            // Note: These imports will work when used from the binary
            // For now, this is a placeholder - the actual implementation
            // remains in src/bin/anno.rs until full refactoring is complete
            match self {
                Self::Pattern => Ok(Box::new(RegexNER::new())),
                Self::Heuristic => Ok(Box::new(HeuristicNER::new())),
                Self::Minimal => Ok(Box::new(HeuristicNER::new())),
                Self::Auto => Ok(Box::new(AutoNER::new())),
                Self::Stacked => Ok(Box::new(StackedNER::default())),
                #[cfg(feature = "onnx")]
                Self::Gliner => anno::GLiNEROnnx::new(anno::DEFAULT_GLINER_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| format!("Failed to load GLiNER: {}\n  Tip: Use 'anno models info gliner' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::Gliner2 => anno::backends::gliner2::GLiNER2Onnx::from_pretrained(anno::DEFAULT_GLINER2_MODEL)
                    .map(|m| Box::new(m) as Box<dyn Model>)
                    .map_err(|e| format!("Failed to load GLiNER2: {}\n  Tip: Use 'anno models info gliner2' to check model status.", e)),
                #[cfg(feature = "onnx")]
                Self::Nuner => Err("NuNER not yet implemented in CLI.\n  Tip: Use 'anno models list' to see available models.".to_string()),
                #[cfg(feature = "onnx")]
                Self::W2ner => Err("W2NER not yet implemented in CLI.\n  Tip: Use 'anno models list' to see available models.".to_string()),
                #[cfg(feature = "candle")]
                Self::GlinerCandle => Err("GLiNER Candle not yet implemented in CLI.\n  Tip: Use 'anno models list' to see available models.".to_string()),
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pattern => "pattern",
            Self::Heuristic => "heuristic",
            Self::Minimal => "minimal",
            Self::Auto => "auto",
            Self::Stacked => "stacked",
            #[cfg(feature = "onnx")]
            Self::Gliner => "gliner",
            #[cfg(feature = "onnx")]
            Self::Gliner2 => "gliner2",
            #[cfg(feature = "onnx")]
            Self::Nuner => "nuner",
            #[cfg(feature = "onnx")]
            Self::W2ner => "w2ner",
            #[cfg(feature = "candle")]
            Self::GlinerCandle => "gliner-candle",
        }
    }
}

/// Unified output format selection for all commands
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable colored output (default)
    #[default]
    Human,
    /// JSON array of entities
    Json,
    /// JSON lines (one object per line)
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

