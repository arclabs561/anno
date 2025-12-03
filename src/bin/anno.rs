//! anno - Information Extraction CLI
//!
//! A unified toolkit for named entity recognition, coreference resolution,
//! relation extraction, and entity linking.
//!
//! # Capabilities
//!
//! - **NER**: Named Entity Recognition (persons, organizations, locations, etc.)
//! - **Coreference**: Link mentions to the same entity ("She" → "Marie Curie")  
//! - **Relations**: Extract (head, relation, tail) triples
//! - **Entity Linking**: Connect entities to knowledge bases (Wikidata)
//! - **Events**: Discourse-level event extraction
//!
//! # Signal → Track → Identity Hierarchy
//!
//! ```text
//! Level 1 (Signal)   : Raw detections with spans  
//! Level 2 (Track)    : Within-document coreference chains
//! Level 3 (Identity) : Cross-document KB-linked entities
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Basic NER extraction
//! anno extract "Marie Curie won the Nobel Prize."
//!
//! # Debug with coreference and KB linking
//! anno debug --coref --link-kb -t "Barack Obama met Angela Merkel. He praised her."
//!
//! # Evaluate against gold annotations
//! anno eval -t "..." -g "Marie Curie:PER:0:11"
//!
//! # Validate annotation files
//! anno validate file.jsonl
//!
//! # Show available models and features
//! anno info
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::process::ExitCode;
use std::time::Instant;

#[cfg(feature = "eval")]
use glob::glob;

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use is_terminal::IsTerminal;

#[cfg(feature = "eval")]
use anno::eval::backend_factory::BackendFactory;
use anno::graph::{GraphDocument, GraphExportFormat};
use anno::grounded::{
    render_document_html, render_eval_html, EvalComparison, EvalMatch, GroundedDocument, Identity,
    IdentitySource, Location, Modality, Quantifier, Signal, SignalValidationError,
};
use anno::ingest::{CompositeResolver, DocumentPreprocessor};
use anno::{AutoNER, Entity, HeuristicNER, Model, RegexNER, StackedNER};

#[cfg(feature = "eval")]
use anno::eval::cdcr::{CDCRConfig, CDCRResolver, Document};

#[cfg(feature = "onnx")]
// GLiNER exports available when onnx feature is enabled
#[allow(unused_imports)]
use anno::{DEFAULT_GLINER2_MODEL, DEFAULT_GLINER_MODEL};

// ============================================================================
// CLI Structure
// ============================================================================

/// Information Extraction CLI - NER, Coreference, Relations, Entity Linking
///
/// UX/DESIGN NOTES:
/// - See hack/CLI_UX_CRITIQUE.md for comprehensive UX analysis
/// - Key issues: inconsistent input methods, model discoverability, output format handling
/// - TODO: Standardize input patterns, add `anno models` command, improve error messages
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
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Text to extract entities from (shorthand for `anno extract`)
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract entities from text
    #[command(visible_alias = "x")]
    Extract(ExtractArgs),

    /// Generate HTML debug visualization
    #[command(visible_alias = "d")]
    Debug(DebugArgs),

    /// Evaluate predictions against gold annotations
    #[command(visible_alias = "e")]
    Eval(EvalArgs),

    /// Validate JSONL annotation files
    #[command(visible_alias = "v")]
    Validate(ValidateArgs),

    /// Deep analysis with multiple models
    #[command(visible_alias = "a")]
    Analyze(AnalyzeArgs),

    /// Work with NER datasets
    #[command(visible_alias = "ds")]
    Dataset(DatasetArgs),

    /// Comprehensive evaluation across all task-dataset-backend combinations
    #[command(visible_alias = "bench")]
    #[cfg(feature = "eval-advanced")]
    Benchmark(BenchmarkArgs),

    /// Show model and version info
    #[command(visible_alias = "i")]
    Info,

    /// List and compare available models
    ///
    /// Shows which models are available in this build, their capabilities,
    /// and how to enable feature-gated models.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # List all models with availability status
    /// anno models list
    ///
    /// # Show details for a specific model
    /// anno models info gliner
    ///
    /// # Compare available models side-by-side
    /// anno models compare
    /// ```
    #[command(visible_alias = "m")]
    Models(ModelsArgs),

    /// Cross-document coreference: cluster entities across multiple documents
    ///
    /// Reads all text files from a directory, extracts entities, and clusters
    /// them across documents. Outputs clusters in JSON or tree format.
    ///
    /// # Relationship to Other Commands
    ///
    /// The CLI follows the Signal → Track → Identity hierarchy:
    /// - `extract`: Level 1 (Signal) - Raw entity extraction from single document
    /// - `debug`: Level 1 + 2 (Signal → Track) - Adds within-document coreference
    /// - `debug --link-kb`: Level 1 + 2 + 3 (Signal → Track → Identity) - Adds KB linking
    /// - `cross-doc`: Cross-document clustering (currently operates on raw entities, not full hierarchy)
    ///
    /// # Future Enhancements
    ///
    /// - Pipeline integration: Accept pre-processed GroundedDocument JSON from `extract`/`debug`
    /// - Hierarchy awareness: Use Tracks/Identities from single-doc processing in cross-doc clustering
    /// - Export/import: Save extract results, load into cross-doc for incremental processing
    /// - Query/filter: Filter clusters by entity type, document set, confidence threshold
    /// - Comparison mode: Show entity differences across documents, not just clusters
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Cluster entities from all .txt files in a directory
    /// anno cross-doc /path/to/documents --format json
    ///
    /// # Output as tree structure
    /// anno cross-doc /path/to/documents --format tree
    ///
    /// # Use specific model and similarity threshold
    /// anno cross-doc /path/to/documents --model gliner --threshold 0.7
    /// ```
    #[command(visible_alias = "cd")]
    #[cfg(feature = "eval-advanced")]
    CrossDoc(CrossDocArgs),

    /// Enhance existing GroundedDocument with additional processing layers
    ///
    /// Takes a GroundedDocument (from `extract --export` or `debug --export`) and
    /// adds additional processing layers (coreference, KB linking) incrementally.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Start with extraction
    /// anno extract "text" --export doc.json
    ///
    /// # Add coreference
    /// anno enhance doc.json --coref --export doc-with-coref.json
    ///
    /// # Add KB linking
    /// anno enhance doc-with-coref.json --link-kb --export doc-full.json
    /// ```
    #[command(visible_alias = "en")]
    Enhance(EnhanceArgs),

    /// Unified pipeline: extract → enhance → cross-doc in one command
    ///
    /// Orchestrates the full processing pipeline from raw text to cross-document
    /// clusters. Each step can be enabled/disabled independently.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Full pipeline with multiple texts
    /// anno pipeline "text1" "text2" --coref --link-kb --cross-doc
    ///
    /// # Process directory
    /// anno pipeline --dir ./docs --coref --cross-doc --output clusters.json
    ///
    /// # With files
    /// anno pipeline --files doc1.txt doc2.txt --coref
    /// ```
    #[command(visible_alias = "p")]
    Pipeline(PipelineArgs),

    /// Query and filter entities/clusters from GroundedDocuments or cross-doc results
    ///
    /// Provides a unified query interface for exploring extraction results.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Query single document
    /// anno query doc.json --type PER --min-confidence 0.8
    ///
    /// # Find specific entity
    /// anno query clusters.json --entity "Apple Inc" --format tree
    ///
    /// # Filter clusters
    /// anno query clusters.json --filter "type=ORG AND confidence>0.7"
    /// ```
    #[command(visible_alias = "q")]
    Query(QueryArgs),

    /// Compare documents, models, or clusters
    ///
    /// Shows differences and similarities between extraction results.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Compare two documents
    /// anno compare doc1.json doc2.json --format diff
    ///
    /// # Compare models on same text
    /// anno compare-models "text" --models stacked gliner
    /// ```
    Compare(CompareArgs),

    /// Manage cache for extraction results
    ///
    /// The CLI automatically caches extraction results. This command manages
    /// the cache manually.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # List cached results
    /// anno cache list
    ///
    /// # Clear cache
    /// anno cache clear
    ///
    /// # Invalidate specific model
    /// anno cache invalidate --model gliner
    /// ```
    Cache(CacheArgs),

    /// Manage configuration files for workflows
    ///
    /// Save and load common workflow configurations.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Create config from current settings
    /// anno config save my-workflow --model gliner --coref --link-kb
    ///
    /// # Use config
    /// anno pipeline --dir ./docs --config my-workflow
    /// ```
    Config(ConfigArgs),

    /// Batch process multiple documents efficiently
    ///
    /// Process multiple documents with parallel support and progress tracking.
    ///
    /// # Examples
    ///
    /// ```bash
    /// # Process directory with parallel workers
    /// anno batch --dir ./docs --coref --link-kb --parallel 4 --progress
    ///
    /// # Stream from stdin
    /// cat docs.jsonl | anno batch --stdin --coref
    /// ```
    #[command(visible_alias = "b")]
    Batch(BatchArgs),

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

// ============================================================================
// Shared Types
// ============================================================================

/// Model backend selection
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum ModelBackend {
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
    fn create_model(self) -> Result<Box<dyn Model>, String> {
        // Use BackendFactory for consistent backend creation when available
        #[cfg(feature = "eval")]
        {
            use anno::eval::backend_factory::BackendFactory;
            // Map backend enum to factory name
            let factory_name = match self {
                Self::Pattern => "pattern",
                Self::Heuristic => "heuristic",
                Self::Minimal => "heuristic", // Minimal uses heuristic
                Self::Auto => "stacked",      // Auto uses stacked
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
        // Fallback to original implementation when eval feature not available
        #[cfg(not(feature = "eval"))]
        match self {
            Self::Pattern => Ok(Box::new(RegexNER::new())),
            Self::Heuristic => Ok(Box::new(HeuristicNER::new())),
            // Minimal was merged into HeuristicNER
            Self::Minimal => Ok(Box::new(HeuristicNER::new())),
            Self::Auto => {
                // AutoNER just routes to default (StackedNER), doesn't combine models
                Ok(Box::new(AutoNER::new()))
            }
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

    fn name(self) -> &'static str {
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
enum OutputFormat {
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
enum EvalTask {
    /// Named Entity Recognition
    #[default]
    Ner,
    /// Coreference Resolution
    Coref,
    /// Relation Extraction
    Relation,
}

// ============================================================================
// Command Arguments
// ============================================================================

#[derive(Parser)]
struct ExtractArgs {
    /// Input text to process
    /// NOTE: Inconsistent with other commands - some support positional args, some don't
    /// TODO: Standardize input handling across all commands (see CLI_UX_CRITIQUE.md)
    #[arg(short, long)]
    text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    /// Model backend to use
    /// TODO: Add `anno models list` command for discoverability
    /// TODO: Better error messages when model requires feature flags (see CLI_UX_CRITIQUE.md)
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Filter to specific entity types (repeatable)
    #[arg(short, long = "label", value_name = "TYPE")]
    labels: Vec<String>,

    /// Output format
    #[arg(long, default_value = "human")]
    format: OutputFormat,

    /// Export GroundedDocument JSON to file (for pipeline integration)
    ///
    /// Saves the full GroundedDocument as JSON, which can be imported by
    /// `cross-doc --import` for cross-document coreference processing.
    /// Example: `anno extract --export doc.json "text"` → `anno cross-doc --import doc.json`
    ///
    /// Export format options:
    /// - "full" (default): Complete GroundedDocument with all metadata
    /// - "signals": Only signals (entities) without tracks/identities
    /// - "minimal": Just text and signals, minimal metadata
    #[arg(long, value_name = "PATH")]
    export: Option<String>,

    /// Export to graph format (neo4j, networkx, jsonld)
    ///
    /// Exports entities to graph format for RAG applications.
    /// Example: `anno extract "text" --export-graph neo4j`
    #[arg(long, value_name = "FORMAT")]
    export_graph: Option<String>,

    /// URL to fetch content from (requires eval-advanced feature)
    ///
    /// Fetches content from HTTP/HTTPS URLs and extracts text.
    /// Example: `anno extract --url https://example.com/article`
    #[arg(long, value_name = "URL")]
    url: Option<String>,

    /// Clean and normalize text before extraction
    ///
    /// Enables whitespace normalization and basic text cleaning.
    #[arg(long)]
    clean: bool,

    /// Normalize Unicode (basic normalization)
    #[arg(long)]
    normalize: bool,

    /// Detect and record language
    #[arg(long)]
    detect_lang: bool,

    /// Export format when using --export (full, signals, minimal)
    ///
    /// Controls what gets exported:
    /// - full: Complete GroundedDocument (signals, tracks, identities, all metadata)
    /// - signals: Only signals/entities (for lightweight exports)
    /// - minimal: Text + signals only (for maximum compatibility)
    #[arg(long, default_value = "full", value_name = "FORMAT")]
    export_format: String,

    /// Detect negated entities ("not John Smith")
    #[arg(long)]
    negation: bool,

    /// Detect quantified entities ("every employee")
    #[arg(long)]
    quantifiers: bool,

    /// Show context around entities and detailed information
    ///
    /// When enabled, shows:
    /// - Context windows around each entity
    /// - Detailed confidence scores
    /// - Additional metadata
    #[arg(short, long)]
    verbose: bool,

    /// Minimal output (suppress warnings and non-essential messages)
    ///
    /// When enabled:
    /// - Only shows essential results
    /// - Suppresses validation warnings
    /// - Reduces progress messages
    #[arg(short, long)]
    quiet: bool,

    /// Positional text argument
    #[arg(trailing_var_arg = true)]
    positional: Vec<String>,
}

#[derive(Parser)]
struct DebugArgs {
    /// Input text to process
    #[arg(short, long)]
    text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    /// Positional text arguments (alternative to --text)
    #[arg(value_name = "TEXT")]
    positional: Vec<String>,

    /// URL to fetch content from (requires eval-advanced feature)
    #[arg(long, value_name = "URL")]
    url: Option<String>,

    /// Clean whitespace (normalize spaces, line breaks)
    #[arg(long)]
    clean: bool,

    /// Normalize Unicode (basic normalization)
    #[arg(long)]
    normalize: bool,

    /// Detect and record language
    #[arg(long)]
    detect_lang: bool,

    /// Export to graph format (neo4j, networkx, jsonld)
    #[arg(long, value_name = "FORMAT")]
    export_graph: Option<String>,

    /// Model backend to use
    /// TODO: Add `anno models list` command for discoverability
    /// TODO: Better error messages when model requires feature flags (see CLI_UX_CRITIQUE.md)
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Output as HTML (default: text)
    ///
    /// Note: For HTML output, use `--format html` instead of `--html` flag.
    /// This flag is kept for backward compatibility but will be deprecated.
    #[arg(long)]
    html: bool,

    /// Export GroundedDocument JSON to file (for pipeline integration)
    ///
    /// Saves the full GroundedDocument as JSON, which can be imported by
    /// `cross-doc --import` for cross-document coreference processing.
    /// Example: `anno debug --export doc.json "text"` → `anno cross-doc --import doc.json`
    ///
    /// Export format options:
    /// - "full" (default): Complete GroundedDocument with all metadata
    /// - "signals": Only signals (entities) without tracks/identities
    /// - "minimal": Just text and signals, minimal metadata
    #[arg(long, value_name = "PATH")]
    export: Option<String>,

    /// Export format when using --export (full, signals, minimal)
    ///
    /// Controls what gets exported:
    /// - full: Complete GroundedDocument (signals, tracks, identities, all metadata)
    /// - signals: Only signals/entities (for lightweight exports)
    /// - minimal: Text + signals only (for maximum compatibility)
    #[arg(long, default_value = "full", value_name = "FORMAT")]
    export_format: String,

    /// Write output to file (default: stdout)
    #[arg(short, long, value_name = "PATH")]
    output: Option<String>,

    /// Run coreference resolution to form tracks
    #[arg(long)]
    coref: bool,

    /// Link tracks to KB identities (creates placeholder Wikidata IDs)
    #[arg(long)]
    link_kb: bool,

    /// Suppress status messages
    #[arg(short, long)]
    quiet: bool,

    /// Verbose output (show preprocessing metadata, etc.)
    #[arg(long)]
    verbose: bool,
}

#[derive(Parser)]
struct EvalArgs {
    /// Input text to process
    #[arg(short, long)]
    text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    /// Model backend to use
    /// TODO: Add `anno models list` command for discoverability
    /// TODO: Better error messages when model requires feature flags (see CLI_UX_CRITIQUE.md)
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Gold annotation: "text:label:start:end" (repeatable)
    #[arg(short, long = "gold", value_name = "SPEC")]
    gold_specs: Vec<String>,

    /// Load gold annotations from JSONL file
    #[arg(long, value_name = "PATH")]
    gold_file: Option<String>,

    /// Write HTML report to file
    #[arg(short, long, value_name = "PATH")]
    output: Option<String>,

    /// Output format (overrides default text output)
    ///
    /// Note: For JSON/HTML output, use `--format json` or `--format html` instead.
    /// These flags are kept for backward compatibility but will be deprecated.
    #[arg(long)]
    json: bool,

    /// Output format (overrides default text output)
    ///
    /// Note: For HTML output, use `--format html` instead.
    /// This flag is kept for backward compatibility but will be deprecated.
    #[arg(long)]
    html: bool,

    /// Show detailed match information and statistics
    ///
    /// When enabled, shows:
    /// - Per-entity match details
    /// - Precision/recall breakdowns
    /// - Detailed comparison statistics
    #[arg(short, long)]
    verbose: bool,

    /// Minimal output (suppress warnings and non-essential messages)
    ///
    /// When enabled:
    /// - Only shows final metrics
    /// - Suppresses validation warnings
    /// - Reduces progress messages
    #[arg(short, long)]
    quiet: bool,

    /// Positional text argument
    #[arg(trailing_var_arg = true)]
    positional: Vec<String>,
}

#[derive(Parser)]
struct ValidateArgs {
    /// JSONL files to validate
    #[arg(required = true)]
    files: Vec<String>,
}

#[derive(Parser)]
struct AnalyzeArgs {
    /// Input text to process
    #[arg(short, long)]
    text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    /// Positional text argument
    #[arg(trailing_var_arg = true)]
    positional: Vec<String>,
}

#[derive(Parser)]
struct ModelsArgs {
    /// Action to perform
    #[command(subcommand)]
    action: ModelsAction,
}

#[derive(Subcommand)]
enum ModelsAction {
    /// List all available models with status
    #[command(visible_alias = "ls")]
    List,

    /// Show detailed information about a model
    #[command(visible_alias = "i")]
    Info {
        /// Model name to get info for
        #[arg(value_name = "MODEL")]
        model: String,
    },

    /// Compare available models side-by-side
    #[command(visible_alias = "c")]
    Compare,
}

#[derive(Parser)]
struct DatasetArgs {
    /// Action to perform
    #[command(subcommand)]
    action: DatasetAction,
}

#[derive(Subcommand)]
enum DatasetAction {
    /// List available datasets
    #[command(visible_alias = "ls")]
    List,

    /// Show dataset statistics
    #[command(visible_alias = "i")]
    Info {
        /// Dataset name
        #[arg(short, long)]
        dataset: String,
    },

    /// Evaluate model on dataset
    #[command(visible_alias = "e")]
    Eval {
        /// Dataset name
        #[arg(short, long, default_value = "synthetic")]
        dataset: String,

        /// Model backend to use
        #[arg(short, long, default_value = "stacked")]
        model: ModelBackend,

        /// Task type: ner, coref, or relation
        #[arg(short, long, default_value = "ner")]
        task: EvalTask,
    },
}

/// Enhance existing GroundedDocument with additional processing
#[derive(Parser)]
struct EnhanceArgs {
    /// Input GroundedDocument JSON file (or "-" for stdin)
    #[arg(value_name = "FILE")]
    input: String,

    /// Run coreference resolution to form tracks
    #[arg(long)]
    coref: bool,

    /// Link tracks to KB identities
    #[arg(long)]
    link_kb: bool,

    /// Export enhanced document to file
    #[arg(short, long, value_name = "PATH")]
    export: Option<String>,

    /// Export format (full, signals, minimal)
    #[arg(long, default_value = "full", value_name = "FORMAT")]
    export_format: String,

    /// Output format for display
    #[arg(long, default_value = "human")]
    format: OutputFormat,

    /// Suppress status messages
    #[arg(short, long)]
    quiet: bool,

    /// Export to graph format (neo4j, networkx, jsonld)
    #[arg(long, value_name = "FORMAT")]
    export_graph: Option<String>,
}

/// Unified pipeline command
#[derive(Parser)]
struct PipelineArgs {
    /// Input text(s) to process (positional)
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,

    /// Read input from file(s)
    #[arg(short, long, value_name = "PATH")]
    files: Vec<String>,

    /// Process directory of text files
    #[arg(short, long, value_name = "DIR")]
    dir: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Run coreference resolution
    #[arg(long)]
    coref: bool,

    /// Link tracks to KB identities
    #[arg(long)]
    link_kb: bool,

    /// Run cross-document clustering
    #[arg(long)]
    cross_doc: bool,

    /// Similarity threshold for cross-doc clustering
    #[arg(long, default_value = "0.6")]
    threshold: f64,

    /// Output format
    #[arg(long, default_value = "human")]
    format: OutputFormat,

    /// Export results to file
    #[arg(short, long, value_name = "PATH")]
    output: Option<String>,

    /// Show progress
    #[arg(long)]
    progress: bool,

    /// Suppress status messages
    #[arg(short, long)]
    quiet: bool,
}

/// Query and filter entities/clusters
#[derive(Parser)]
struct QueryArgs {
    /// Input file (GroundedDocument JSON or cross-doc clusters JSON)
    #[arg(value_name = "FILE")]
    input: String,

    /// Filter by entity type
    #[arg(short, long, value_name = "TYPE")]
    r#type: Option<String>,

    /// Find specific entity by name
    #[arg(short, long, value_name = "TEXT")]
    entity: Option<String>,

    /// Minimum confidence threshold
    #[arg(long, value_name = "FLOAT")]
    min_confidence: Option<f64>,

    /// Filter expression (e.g., "type=ORG AND confidence>0.7")
    #[arg(short, long, value_name = "EXPR")]
    filter: Option<String>,

    /// Start offset for range queries (character position)
    #[arg(long, value_name = "OFFSET")]
    start_offset: Option<usize>,

    /// End offset for range queries (character position)
    #[arg(long, value_name = "OFFSET")]
    end_offset: Option<usize>,

    /// Filter for negated signals only
    #[arg(long)]
    negated: bool,

    /// Filter for signals with quantifiers
    #[arg(long)]
    quantified: bool,

    /// Filter for untracked signals (not in any track)
    #[arg(long)]
    untracked: bool,

    /// Filter for signals linked to identities (via tracks)
    #[arg(long)]
    linked: bool,

    /// Filter for signals not linked to identities
    #[arg(long)]
    unlinked: bool,

    /// Output format
    #[arg(long, default_value = "human")]
    format: OutputFormat,

    /// Output file
    #[arg(short, long, value_name = "PATH")]
    output: Option<String>,
}

/// Compare documents, models, or clusters
#[derive(Parser)]
struct CompareArgs {
    /// First input file
    #[arg(value_name = "FILE1")]
    file1: String,

    /// Second input file (or text for compare-models)
    #[arg(value_name = "FILE2")]
    file2: Option<String>,

    /// Compare models on same text (use file1 as text)
    #[arg(long)]
    models: bool,

    /// Models to compare (when --models is used)
    #[arg(long, value_delimiter = ',', value_name = "MODEL")]
    model_list: Vec<String>,

    /// Output format (diff, table, summary)
    #[arg(long, default_value = "diff")]
    format: String,

    /// Output file
    #[arg(short, long, value_name = "PATH")]
    output: Option<String>,
}

/// Cache management
#[derive(Parser)]
struct CacheArgs {
    /// Action to perform
    #[command(subcommand)]
    action: CacheAction,
}

#[derive(Subcommand)]
enum CacheAction {
    /// List cached results
    #[command(visible_alias = "ls")]
    List,

    /// Clear all cache
    Clear,

    /// Show cache statistics
    Stats,

    /// Invalidate cache entries
    Invalidate {
        /// Invalidate entries for specific model
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,

        /// Invalidate entries for specific file
        #[arg(long, value_name = "FILE")]
        file: Option<String>,
    },
}

/// Configuration management
#[derive(Parser)]
struct ConfigArgs {
    /// Action to perform
    #[command(subcommand)]
    action: ConfigAction,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Save current settings as config
    Save {
        /// Config name
        #[arg(value_name = "NAME")]
        name: String,

        /// Model to save in config
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,

        /// Include coreference in config
        #[arg(long)]
        coref: bool,

        /// Include KB linking in config
        #[arg(long)]
        link_kb: bool,

        /// Threshold for cross-doc
        #[arg(long, value_name = "FLOAT")]
        threshold: Option<f64>,
    },

    /// List saved configs
    #[command(visible_alias = "ls")]
    List,

    /// Show config details
    Show {
        /// Config name
        #[arg(value_name = "NAME")]
        name: String,
    },

    /// Delete config
    Delete {
        /// Config name
        #[arg(value_name = "NAME")]
        name: String,
    },
}

/// Batch processing
#[derive(Parser)]
struct BatchArgs {
    /// Process directory of files
    #[arg(short, long, value_name = "DIR")]
    dir: Option<String>,

    /// Read from stdin (JSONL format)
    #[arg(long)]
    stdin: bool,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Run coreference resolution
    #[arg(long)]
    coref: bool,

    /// Link tracks to KB identities
    #[arg(long)]
    link_kb: bool,

    /// Number of parallel workers
    #[arg(short, long, default_value = "1")]
    parallel: usize,

    /// Show progress bar
    #[arg(long)]
    progress: bool,

    /// Enable caching
    #[arg(long)]
    cache: bool,

    /// Output directory for results
    #[arg(short, long, value_name = "DIR")]
    output: Option<String>,

    /// Output format
    #[arg(long, default_value = "grounded")]
    format: OutputFormat,

    /// Suppress status messages
    #[arg(short, long)]
    quiet: bool,
}

#[cfg(feature = "eval-advanced")]
#[derive(Parser)]
struct CrossDocArgs {
    /// Directory containing text files to process (optional if --import is used)
    #[arg(value_name = "DIR")]
    directory: Option<String>,

    /// Model backend to use for entity extraction
    #[arg(short, long, default_value = "stacked")]
    model: ModelBackend,

    /// Similarity threshold for clustering (0.0-1.0)
    #[arg(short, long, default_value = "0.6")]
    threshold: f64,

    /// Require entity type match for clustering
    #[arg(long)]
    require_type_match: bool,

    /// Output format
    #[arg(short, long, default_value = "json")]
    format: OutputFormat,

    /// Import pre-processed GroundedDocument JSON file(s) instead of processing directory
    ///
    /// When enabled, reads GroundedDocument JSON file(s) (from `extract --export` or
    /// `debug --export`) instead of processing raw text files from the directory.
    /// This enables pipeline integration: extract → cross-doc
    ///
    /// Can specify multiple files:
    /// - `--import file1.json --import file2.json` (multiple files)
    /// - `--import "*.json"` (glob pattern)
    /// - `--import "-"` (read from stdin as JSONL, one GroundedDocument per line)
    ///
    /// Example: `anno extract --export doc.json "text"` → `anno cross-doc --import doc.json`
    #[arg(long, value_name = "PATH")]
    import: Vec<String>,

    /// Read input from stdin (JSONL format, one GroundedDocument per line)
    ///
    /// When enabled, reads GroundedDocument JSON objects from stdin in JSONL format.
    /// Each line should be a valid GroundedDocument JSON object.
    /// Useful for streaming pipelines: `cat docs.jsonl | anno cross-doc --stdin`
    #[arg(long)]
    stdin: bool,

    /// File extensions to process (comma-separated)
    #[arg(long, default_value = "txt,md")]
    extensions: String,

    /// Recursively search subdirectories
    #[arg(short, long)]
    recursive: bool,

    /// Minimum cluster size to include in output
    #[arg(long, default_value = "1")]
    min_cluster_size: usize,

    /// Filter to only cross-document clusters (appears in 2+ docs)
    #[arg(long)]
    cross_doc_only: bool,

    /// Filter by entity type (repeatable, e.g., --type PER --type ORG)
    #[arg(long = "type", value_name = "TYPE")]
    entity_types: Vec<String>,

    /// Maximum number of clusters to output (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_clusters: usize,

    /// Output file path (if not specified, prints to stdout)
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// Show progress and detailed cluster information
    ///
    /// When enabled, shows:
    /// - Processing progress for each document
    /// - Detailed cluster information
    /// - Extended context in tree output
    #[arg(short, long)]
    verbose: bool,
}

// CrossDocFormat removed - now using unified OutputFormat

#[derive(Parser)]
struct BenchmarkArgs {
    /// Tasks to evaluate (comma-separated: ner,coref,relation). Default: all
    #[arg(short, long, value_delimiter = ',')]
    tasks: Option<Vec<String>>,

    /// Datasets to use (comma-separated). Default: all suitable datasets
    #[arg(short, long, value_delimiter = ',')]
    datasets: Option<Vec<String>>,

    /// Backends to test (comma-separated). Default: all compatible backends
    #[arg(short, long, value_delimiter = ',')]
    backends: Option<Vec<String>>,

    /// Maximum examples per dataset (for quick testing)
    #[arg(short, long)]
    max_examples: Option<usize>,

    /// Random seed for sampling (for reproducibility and varied testing)
    #[arg(long)]
    seed: Option<u64>,

    /// Only use cached datasets (skip downloads)
    #[arg(long)]
    cached_only: bool,

    /// Output file for markdown report (default: stdout)
    #[arg(short, long)]
    output: Option<String>,
}

#[cfg(feature = "eval")]
use anno::eval::loader::DatasetId;

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result: Result<(), String> = match cli.command {
        Some(Commands::Extract(args)) => cmd_extract(args),
        Some(Commands::Debug(args)) => cmd_debug(args),
        Some(Commands::Eval(args)) => cmd_eval(args),
        Some(Commands::Validate(args)) => cmd_validate(args),
        Some(Commands::Analyze(args)) => cmd_analyze(args),
        Some(Commands::Dataset(args)) => cmd_dataset(args),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::Benchmark(args)) => cmd_benchmark(args),
        Some(Commands::Info) => cmd_info(),
        Some(Commands::Models(args)) => cmd_models(args),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::CrossDoc(args)) => cmd_crossdoc(args),
        Some(Commands::Enhance(args)) => cmd_enhance(args),
        Some(Commands::Pipeline(args)) => cmd_pipeline(args),
        Some(Commands::Query(args)) => cmd_query(args),
        Some(Commands::Compare(args)) => cmd_compare(args),
        Some(Commands::Cache(args)) => cmd_cache(args),
        Some(Commands::Config(args)) => cmd_config(args),
        Some(Commands::Batch(args)) => cmd_batch(args),
        Some(Commands::Completions { shell }) => {
            generate(shell, &mut Cli::command(), "anno", &mut io::stdout());
            Ok(())
        }
        None => {
            // No subcommand: treat positional args as text to extract
            if cli.text.is_empty() {
                eprintln!("No input provided. Run `anno --help` for usage.");
                return ExitCode::FAILURE;
            }
            let text = cli.text.join(" ");
            cmd_extract(ExtractArgs {
                url: None,
                clean: false,
                normalize: false,
                detect_lang: false,
                export_graph: None,
                text: Some(text),
                file: None,
                model: ModelBackend::default(),
                labels: vec![],
                format: OutputFormat::default(),
                export: None,
                export_format: "full".to_string(),
                negation: false,
                quantifiers: false,
                verbose: false,
                quiet: false,
                positional: vec![],
            })
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{} {}", color("31", "error:"), e);
            ExitCode::FAILURE
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Find similar model names using simple string similarity
fn find_similar_models(query: &str, candidates: &[&str]) -> Vec<String> {
    let query_lower = query.to_lowercase();
    let mut matches: Vec<(f64, &str)> = candidates
        .iter()
        .filter_map(|&candidate| {
            let candidate_lower = candidate.to_lowercase();
            // Check if query is a prefix of candidate or vice versa
            if candidate_lower.starts_with(&query_lower)
                || query_lower.starts_with(&candidate_lower)
            {
                Some((0.9, candidate))
            } else if candidate_lower.contains(&query_lower)
                || query_lower.contains(&candidate_lower)
            {
                Some((0.7, candidate))
            } else {
                // Simple Levenshtein-like check (first char match)
                if candidate_lower.chars().next() == query_lower.chars().next() {
                    Some((0.5, candidate))
                } else {
                    None
                }
            }
        })
        .collect();

    matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    matches
        .into_iter()
        .take(3)
        .map(|(_, name)| name.to_string())
        .collect()
}

// ============================================================================
// Commands
// ============================================================================

fn cmd_extract(args: ExtractArgs) -> Result<(), String> {
    // Level 1 (Signal): Raw entity extraction from single document
    // This is the foundation for all other commands:
    // - `debug` adds Level 2 (Track) via coreference resolution
    // - `debug --link-kb` adds Level 3 (Identity) via KB linking
    // - `cross-doc` clusters Level 1 entities across multiple documents

    // Resolve input: URL, file, text, or stdin
    let mut raw_text = if let Some(url) = &args.url {
        #[cfg(feature = "eval-advanced")]
        {
            let resolver = CompositeResolver::new();
            let resolved = resolver
                .resolve(url)
                .map_err(|e| format!("Failed to fetch URL {}: {}", url, e))?;
            resolved.text
        }
        #[cfg(not(feature = "eval-advanced"))]
        {
            return Err("URL resolution requires 'eval-advanced' feature. Enable with: cargo build --features eval-advanced".to_string());
        }
    } else {
        get_input_text(&args.text, args.file.as_deref(), &args.positional)?
    };

    // Preprocess text if requested
    if args.clean || args.normalize || args.detect_lang {
        let preprocessor = DocumentPreprocessor {
            clean_whitespace: args.clean,
            normalize_unicode: args.normalize,
            detect_language: args.detect_lang,
            chunk_size: None,
        };
        let prepared = preprocessor.prepare(&raw_text);
        raw_text = prepared.text;
        if args.verbose && !prepared.metadata.is_empty() {
            eprintln!("Preprocessing metadata: {:?}", prepared.metadata);
        }
    }

    let text = raw_text;
    let model = args.model.create_model()?;

    let start = Instant::now();
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;
    let elapsed = start.elapsed();

    // Filter by labels if specified
    let entities: Vec<_> = if args.labels.is_empty() {
        entities
    } else {
        entities
            .into_iter()
            .filter(|e| {
                args.labels
                    .iter()
                    .any(|l| e.entity_type.as_label().eq_ignore_ascii_case(l))
            })
            .collect()
    };

    // Build grounded document with validation using library method
    let mut doc = GroundedDocument::new("extract", &text);
    let mut validation_errors: Vec<SignalValidationError> = Vec::new();

    for e in &entities {
        let mut signal = Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        )
        .with_modality(Modality::Symbolic);

        // Detect negation
        if args.negation && is_negated(&text, e.start) {
            signal = signal.negated();
        }

        // Detect quantification
        if args.quantifiers {
            if let Some(q) = detect_quantifier(&text, e.start) {
                signal = signal.with_quantifier(q);
            }
        }

        // Use library validation method for consistent error handling
        match doc.add_signal_validated(signal) {
            Ok(_) => {
                // Signal added successfully
            }
            Err(err) => {
                validation_errors.push(err);
            }
        }
    }

    // Report validation errors
    if !validation_errors.is_empty() && !args.quiet {
        eprintln!(
            "{} {} validation errors:",
            color("33", "warning:"),
            validation_errors.len()
        );
        for err in &validation_errors {
            eprintln!("  - {}", err);
        }
    }

    // Output
    match args.format {
        OutputFormat::Json => {
            let output: Vec<_> = doc
                .signals()
                .iter()
                .map(|s| {
                    let (start, end) = s.text_offsets().unwrap_or((0, 0));
                    serde_json::json!({
                        "text": s.surface(),
                        "type": s.label(),
                        "start": start,
                        "end": end,
                        "confidence": s.confidence,
                        "negated": s.negated,
                        "quantifier": s.quantifier.map(|q| format!("{:?}", q)),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
        }
        OutputFormat::Jsonl => {
            for s in doc.signals() {
                let (start, end) = s.text_offsets().unwrap_or((0, 0));
                let obj = serde_json::json!({
                    "text": s.surface(),
                    "type": s.label(),
                    "start": start,
                    "end": end,
                    "confidence": s.confidence,
                });
                println!("{}", obj);
            }
        }
        OutputFormat::Tsv => {
            println!("start\tend\ttype\tconfidence\tnegated\ttext");
            for s in doc.signals() {
                let (start, end) = s.text_offsets().unwrap_or((0, 0));
                println!(
                    "{}\t{}\t{}\t{:.2}\t{}\t{}",
                    start,
                    end,
                    s.label(),
                    s.confidence,
                    s.negated,
                    s.surface()
                );
            }
        }
        OutputFormat::Grounded => {
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        }
        OutputFormat::Html => {
            return Err(
                "HTML format not supported for extract command. Use 'debug --format html' instead."
                    .to_string(),
            );
        }
        OutputFormat::Tree | OutputFormat::Summary => {
            return Err(
                "Tree/Summary formats are only available for cross-doc command.".to_string(),
            );
        }
        OutputFormat::Inline => {
            print_annotated_signals(&text, doc.signals());
        }
        OutputFormat::Human => {
            if args.quiet {
                for s in doc.signals() {
                    let (start, end) = s.text_offsets().unwrap_or((0, 0));
                    let neg = if s.negated { " [NEG]" } else { "" };
                    let quant = s
                        .quantifier
                        .map(|q| format!(" [{:?}]", q))
                        .unwrap_or_default();
                    println!(
                        "[{},{})\t{}\t{}{}{}",
                        start,
                        end,
                        s.label(),
                        s.surface(),
                        neg,
                        quant
                    );
                }
            } else {
                // Use doc.stats() for consistent statistics
                let stats = doc.stats();
                println!();
                println!(
                    "{} extracted {} entities in {:.1}ms (model: {}, avg confidence: {:.2}, tracks: {}, identities: {})",
                    color("32", "ok:"),
                    stats.signal_count,
                    elapsed.as_secs_f64() * 1000.0,
                    args.model.name(),
                    stats.avg_confidence,
                    stats.track_count,
                    stats.identity_count
                );
                println!();

                if doc.signals().is_empty() {
                    println!("  (no entities found)");
                } else {
                    print_signals(&doc, &text, !args.quiet);
                }
                println!();
                print_annotated_signals(&text, doc.signals());
            }
        }
    }

    // Export to file if requested
    if let Some(export_path) = args.export {
        let export_data = match args.export_format.as_str() {
            "full" => serde_json::to_value(&doc)
                .map_err(|e| format!("Failed to serialize GroundedDocument: {}", e))?,
            "signals" => {
                let signals: Vec<_> = doc.signals().iter().cloned().collect();
                serde_json::json!({
                    "id": doc.id,
                    "text": doc.text,
                    "signals": signals
                })
            }
            "minimal" => {
                let signals: Vec<_> = doc
                    .signals()
                    .iter()
                    .map(|s| {
                        let (start, end) = s.text_offsets().unwrap_or((0, 0));
                        serde_json::json!({
                            "surface": s.surface(),
                            "label": s.label(),
                            "start": start,
                            "end": end,
                            "confidence": s.confidence
                        })
                    })
                    .collect();
                serde_json::json!({
                    "id": doc.id,
                    "text": doc.text,
                    "signals": signals
                })
            }
            _ => {
                return Err(format!(
                    "Invalid export format '{}'. Use: full, signals, or minimal",
                    args.export_format
                ));
            }
        };

        let json = serde_json::to_string_pretty(&export_data)
            .map_err(|e| format!("Failed to serialize export data: {}", e))?;

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&export_path).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create directory for export file '{}': {}",
                        export_path, e
                    )
                })?;
            }
        }

        fs::write(&export_path, json)
            .map_err(|e| format!("Failed to write export file '{}': {}", export_path, e))?;
        if !args.quiet {
            eprintln!(
                "{} Exported {} format to {}",
                color("32", "✓"),
                args.export_format,
                export_path
            );
        }
    }

    // Export to graph format if requested
    if let Some(graph_format_str) = args.export_graph {
        let graph_format = match graph_format_str.to_lowercase().as_str() {
            "neo4j" | "cypher" => GraphExportFormat::Cypher,
            "networkx" | "nx" => GraphExportFormat::NetworkXJson,
            "jsonld" | "json-ld" => GraphExportFormat::JsonLd,
            _ => {
                return Err(format!(
                    "Invalid graph format '{}'. Use: neo4j, networkx, or jsonld",
                    graph_format_str
                ));
            }
        };

        let graph = GraphDocument::from_grounded_document(&doc);
        let graph_output = graph.export(graph_format);

        // Output graph to stdout (always print to stdout for graph export)
        // Note: If user wants to save to file, they can use shell redirection: --export-graph neo4j > output.cypher
        if !args.quiet {
            eprintln!(
                "{} Exported graph ({} nodes, {} edges) in {} format",
                color("32", "✓"),
                graph.node_count(),
                graph.edge_count(),
                graph_format_str
            );
        }
        println!("{}", graph_output);
    }

    Ok(())
}

fn cmd_debug(args: DebugArgs) -> Result<(), String> {
    // Level 1 + 2 (Signal → Track): Entity extraction + within-document coreference
    // With --link-kb: Level 1 + 2 + 3 (Signal → Track → Identity): Adds KB linking
    // This builds the full hierarchy that could be used by cross-doc for better clustering

    // Resolve input: URL, file, text, or stdin
    let mut raw_text = if let Some(url) = &args.url {
        #[cfg(feature = "eval-advanced")]
        {
            let resolver = CompositeResolver::new();
            let resolved = resolver
                .resolve(url)
                .map_err(|e| format!("Failed to fetch URL {}: {}", url, e))?;
            resolved.text
        }
        #[cfg(not(feature = "eval-advanced"))]
        {
            return Err("URL resolution requires 'eval-advanced' feature. Enable with: cargo build --features eval-advanced".to_string());
        }
    } else {
        get_input_text(&args.text, args.file.as_deref(), &args.positional)?
    };

    // Preprocess text if requested
    if args.clean || args.normalize || args.detect_lang {
        let preprocessor = DocumentPreprocessor {
            clean_whitespace: args.clean,
            normalize_unicode: args.normalize,
            detect_language: args.detect_lang,
            chunk_size: None,
        };
        let prepared = preprocessor.prepare(&raw_text);
        raw_text = prepared.text;
        if args.verbose && !prepared.metadata.is_empty() {
            eprintln!("Preprocessing metadata: {:?}", prepared.metadata);
        }
    }

    let text = raw_text;
    let model = args.model.create_model()?;

    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    // Build grounded document with validated signals
    // Always use actual offsets from model - don't re-find text (which would always find first occurrence)
    let mut doc = GroundedDocument::new("debug", &text);
    let mut signal_ids: Vec<u64> = Vec::new();

    for e in &entities {
        let signal = Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        );
        let id = doc.add_signal(signal);
        signal_ids.push(id);
    }

    // Run coreference resolution if requested
    if args.coref {
        resolve_coreference(&mut doc, &text, &signal_ids);
    }

    // Link tracks to KB identities if requested
    if args.link_kb {
        link_tracks_to_kb(&mut doc);
    }

    // Export to file if requested
    if let Some(export_path) = args.export {
        let export_data = match args.export_format.as_str() {
            "full" => serde_json::to_value(&doc)
                .map_err(|e| format!("Failed to serialize GroundedDocument: {}", e))?,
            "signals" => {
                let signals: Vec<_> = doc.signals().iter().cloned().collect();
                serde_json::json!({
                    "id": doc.id,
                    "text": doc.text,
                    "signals": signals
                })
            }
            "minimal" => {
                let signals: Vec<_> = doc
                    .signals()
                    .iter()
                    .map(|s| {
                        let (start, end) = s.text_offsets().unwrap_or((0, 0));
                        serde_json::json!({
                            "surface": s.surface(),
                            "label": s.label(),
                            "start": start,
                            "end": end,
                            "confidence": s.confidence
                        })
                    })
                    .collect();
                serde_json::json!({
                    "id": doc.id,
                    "text": doc.text,
                    "signals": signals
                })
            }
            _ => {
                return Err(format!(
                    "Invalid export format '{}'. Use: full, signals, or minimal",
                    args.export_format
                ));
            }
        };

        let json = serde_json::to_string_pretty(&export_data)
            .map_err(|e| format!("Failed to serialize export data: {}", e))?;

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&export_path).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create directory for export file '{}': {}",
                        export_path, e
                    )
                })?;
            }
        }

        fs::write(&export_path, json)
            .map_err(|e| format!("Failed to write export file '{}': {}", export_path, e))?;
        if !args.quiet {
            eprintln!(
                "{} Exported {} format to {}",
                color("32", "✓"),
                args.export_format,
                export_path
            );
        }
    }

    // Build spatial index and validate
    let index = doc.build_text_index();
    let errors = doc.validate();

    if !errors.is_empty() && !args.quiet {
        eprintln!(
            "{} {} validation errors:",
            color("33", "warning:"),
            errors.len()
        );
        for e in &errors {
            eprintln!("  - {}", e);
        }
    }

    // Show stats
    if !args.quiet {
        let stats = doc.stats();
        println!();
        println!("{}", color("1;36", "Document Analysis"));
        println!("  Text length: {} chars", text.len());
        println!("  Signals: {}", stats.signal_count);
        println!("  Tracks: {}", stats.track_count);
        println!("  Identities: {}", stats.identity_count);
        println!("  Spatial index nodes: {}", index.len());
        println!(
            "  Validation: {}",
            if errors.is_empty() {
                color("32", "valid")
            } else {
                color("31", &format!("{} errors", errors.len()))
            }
        );
        println!();
    }

    // Output format
    if args.html
        || args
            .output
            .as_ref()
            .map(|p| p.ends_with(".html"))
            .unwrap_or(false)
    {
        // Generate HTML
        let html = render_document_html(&doc);

        if let Some(path) = &args.output {
            fs::write(path, &html).map_err(|e| format!("Failed to write {}: {}", path, e))?;
            if !args.quiet {
                println!("{} HTML written to: {}", color("32", "ok:"), path);
            }
        } else {
            println!("{}", html);
        }
    } else {
        // Text output (default)
        if doc.signals().is_empty() {
            println!("  (no entities found)");
        } else {
            print_signals(&doc, &text, false);
        }
        println!();
        print_annotated_signals(&text, doc.signals());

        // Show tracks if coref was run
        if args.coref {
            let tracks: Vec<_> = doc.tracks().collect();
            if !tracks.is_empty() {
                println!();
                println!("{}", color("1;36", "Coreference Tracks"));
                for track in tracks {
                    let entity_type = track.entity_type.as_deref().unwrap_or("-");
                    let signals: Vec<String> = track
                        .signals
                        .iter()
                        .filter_map(|s| doc.get_signal(s.signal_id))
                        .map(|s| format!("\"{}\"", s.surface()))
                        .collect();
                    println!(
                        "  T{}: {} [{}] ({})",
                        track.id,
                        track.canonical_surface,
                        entity_type,
                        signals.join(", ")
                    );
                }
            }
        }

        // Show identities if KB linking was run
        if args.link_kb {
            let identities: Vec<_> = doc.identities().collect();
            if !identities.is_empty() {
                println!();
                println!("{}", color("1;36", "KB-Linked Identities"));
                for identity in identities {
                    let kb_id = identity.kb_id.as_deref().unwrap_or("-");
                    println!(
                        "  I{}: {} ({})",
                        identity.id, identity.canonical_name, kb_id
                    );
                }
            }
        }

        // Note: Text output always goes to stdout
        // Use --html --output file.html for HTML file output
    }

    Ok(())
}

fn cmd_eval(args: EvalArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &args.positional)?;

    // Load gold from file or args
    let gold = if let Some(gold_file) = &args.gold_file {
        load_gold_from_file(gold_file)?
    } else if !args.gold_specs.is_empty() {
        args.gold_specs
            .iter()
            .filter_map(|s| parse_gold_spec(s))
            .collect()
    } else {
        return Err(
            "No gold annotations. Use -g 'text:label:start:end' or --gold-file path.jsonl"
                .to_string(),
        );
    };

    if gold.is_empty() {
        return Err("No valid gold annotations found".to_string());
    }

    let model = args.model.create_model()?;

    let start = Instant::now();
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;
    let elapsed = start.elapsed();

    // Build signals
    let gold_signals: Vec<Signal<Location>> = gold
        .iter()
        .enumerate()
        .map(|(i, g)| {
            Signal::new(
                i as u64,
                Location::text(g.start, g.end),
                &g.text,
                &g.label,
                1.0,
            )
        })
        .collect();

    let pred_signals: Vec<Signal<Location>> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            Signal::new(
                i as u64,
                Location::text(e.start, e.end),
                &e.text,
                e.entity_type.as_label(),
                e.confidence as f32,
            )
        })
        .collect();

    let cmp = EvalComparison::compare(&text, gold_signals, pred_signals);

    // Detailed analysis with eval feature
    #[cfg(feature = "eval")]
    let detailed_analysis = {
        use anno::eval::analysis::ErrorAnalysis;
        use anno::eval::GoldEntity;
        use anno::EntityType;

        let gold_entities: Vec<GoldEntity> = gold
            .iter()
            .map(|g| GoldEntity {
                text: g.text.clone(),
                entity_type: EntityType::Other(g.label.clone()),
                original_label: g.label.clone(),
                start: g.start,
                end: g.end,
            })
            .collect();

        Some(ErrorAnalysis::analyze(&text, &entities, &gold_entities))
    };
    #[cfg(not(feature = "eval"))]
    let _detailed_analysis: Option<()> = None;

    // Output
    if args.json {
        let mut output = serde_json::json!({
            "model": args.model.name(),
            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
            "gold_count": cmp.gold.len(),
            "predicted_count": cmp.predicted.len(),
            "correct": cmp.correct_count(),
            "errors": cmp.error_count(),
            "precision": cmp.precision(),
            "recall": cmp.recall(),
            "f1": cmp.f1(),
        });

        let matches: Vec<_> = cmp
            .matches
            .iter()
            .map(|m| match m {
                EvalMatch::Correct { gold_id, pred_id } => serde_json::json!({
                    "type": "correct",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                }),
                EvalMatch::TypeMismatch {
                    gold_id,
                    pred_id,
                    gold_label,
                    pred_label,
                } => serde_json::json!({
                    "type": "type_mismatch",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                }),
                EvalMatch::BoundaryError {
                    gold_id,
                    pred_id,
                    iou,
                } => serde_json::json!({
                    "type": "boundary_error",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                    "iou": iou,
                }),
                EvalMatch::Spurious { pred_id } => serde_json::json!({
                    "type": "false_positive",
                    "pred_id": pred_id,
                }),
                EvalMatch::Missed { gold_id } => serde_json::json!({
                    "type": "false_negative",
                    "gold_id": gold_id,
                }),
            })
            .collect();
        output["matches"] = serde_json::Value::Array(matches);

        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
    } else if args.html {
        let html = render_eval_html(&cmp);
        if let Some(path) = &args.output {
            fs::write(path, &html).map_err(|e| format!("Write failed: {}", e))?;
            if !args.quiet {
                println!("{} HTML written to: {}", color("32", "ok:"), path);
            }
        } else {
            println!("{}", html);
        }
    } else {
        // Human readable
        println!();
        println!(
            "{}",
            color(
                "1;36",
                "======================================================================="
            )
        );
        println!(
            "  {}  model={}  time={:.1}ms",
            color("1;36", "EVALUATION"),
            args.model.name(),
            elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "  gold={}  pred={}  correct={}  errors={}",
            cmp.gold.len(),
            cmp.predicted.len(),
            cmp.correct_count(),
            cmp.error_count()
        );
        println!(
            "{}",
            color(
                "1;36",
                "======================================================================="
            )
        );
        println!();

        let p = cmp.precision() * 100.0;
        let r = cmp.recall() * 100.0;
        let f1 = cmp.f1() * 100.0;

        println!("  Precision: {}%", metric_colored(p));
        println!("  Recall:    {}%", metric_colored(r));
        println!("  F1:        {}%", metric_colored(f1));
        println!();

        print_matches(&cmp, args.verbose);

        #[cfg(feature = "eval")]
        if let Some(analysis) = detailed_analysis {
            println!();
            println!("{}:", color("1;33", "Error Breakdown"));
            for (err_type, count) in &analysis.counts {
                println!("  {:?}: {}", err_type, count);
            }
        }

        println!();
    }

    Ok(())
}

fn cmd_validate(args: ValidateArgs) -> Result<(), String> {
    let mut total_errors = 0;
    let mut total_warnings = 0;
    let mut total_entries = 0;

    for file in &args.files {
        let content =
            fs::read_to_string(file).map_err(|e| format!("Failed to read {}: {}", file, e))?;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            total_entries += 1;

            let entry: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: Invalid JSON: {}", file, line_num + 1, e))?;

            let text = entry["text"]
                .as_str()
                .ok_or_else(|| format!("{}:{}: Missing 'text' field", file, line_num + 1))?;

            let entities = entry["entities"]
                .as_array()
                .ok_or_else(|| format!("{}:{}: Missing 'entities' array", file, line_num + 1))?;

            let mut doc = GroundedDocument::new(format!("{}:{}", file, line_num + 1), text);

            for (i, ent) in entities.iter().enumerate() {
                // Check for missing required fields
                let start = match ent["start"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        eprintln!(
                            "{} {}:{}:entity[{}]: missing 'start' field",
                            color("33", "warn"),
                            file,
                            line_num + 1,
                            i
                        );
                        total_warnings += 1;
                        0
                    }
                };
                let end = match ent["end"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        eprintln!(
                            "{} {}:{}:entity[{}]: missing 'end' field",
                            color("33", "warn"),
                            file,
                            line_num + 1,
                            i
                        );
                        total_warnings += 1;
                        0
                    }
                };
                let ent_text = ent["text"].as_str().unwrap_or("");
                let ent_type = ent["type"]
                    .as_str()
                    .or(ent["label"].as_str())
                    .unwrap_or("UNK");

                let signal = Signal::new(
                    i as u64,
                    Location::text(start, end),
                    ent_text,
                    ent_type,
                    1.0,
                );

                if let Some(err) = signal.validate_against(text) {
                    match err {
                        SignalValidationError::OutOfBounds { .. }
                        | SignalValidationError::InvalidSpan { .. } => {
                            eprintln!(
                                "{} {}:{}:entity[{}]: {}",
                                color("31", "error"),
                                file,
                                line_num + 1,
                                i,
                                err
                            );
                            total_errors += 1;
                        }
                        SignalValidationError::TextMismatch { .. } => {
                            eprintln!(
                                "{} {}:{}:entity[{}]: {}",
                                color("33", "warn"),
                                file,
                                line_num + 1,
                                i,
                                err
                            );
                            total_warnings += 1;
                        }
                    }
                }

                doc.add_signal(signal);
            }
        }
    }

    println!();
    println!(
        "Validated {} entries in {} file(s)",
        total_entries,
        args.files.len()
    );
    if total_errors > 0 {
        println!("{} {} errors", color("31", "x"), total_errors);
    }
    if total_warnings > 0 {
        println!("{} {} warnings", color("33", "!"), total_warnings);
    }
    if total_errors == 0 && total_warnings == 0 {
        println!("{} All valid", color("32", "ok:"));
    }

    if total_errors > 0 {
        return Err(format!("{} validation errors", total_errors));
    }

    Ok(())
}

fn cmd_analyze(args: AnalyzeArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &args.positional)?;

    println!();
    println!(
        "{}",
        color(
            "1;36",
            "======================================================================="
        )
    );
    println!("  {}", color("1;36", "DEEP ANALYSIS"));
    println!(
        "{}",
        color(
            "1;36",
            "======================================================================="
        )
    );
    println!();

    let backends = [
        ModelBackend::Pattern,
        ModelBackend::Heuristic,
        ModelBackend::Stacked,
    ];

    let mut all_results: HashMap<String, Vec<Entity>> = HashMap::new();

    for backend in &backends {
        let model = backend.create_model()?;
        let start = Instant::now();
        let entities = model.extract_entities(&text, None).unwrap_or_default();
        let elapsed = start.elapsed();

        println!("{}:", color("1;33", backend.name()));
        println!(
            "  {} entities in {:.1}ms",
            entities.len(),
            elapsed.as_secs_f64() * 1000.0
        );

        if !entities.is_empty() {
            let mut by_type: HashMap<String, usize> = HashMap::new();
            for e in &entities {
                *by_type
                    .entry(e.entity_type.as_label().to_string())
                    .or_default() += 1;
            }
            for (t, c) in &by_type {
                println!("    {}: {}", t, c);
            }
        }
        println!();

        all_results.insert(backend.name().to_string(), entities);
    }

    // Find disagreements
    println!("{}:", color("1;33", "Model Agreement"));

    let stacked = all_results.get("stacked").cloned().unwrap_or_default();
    let pattern = all_results.get("pattern").cloned().unwrap_or_default();
    let heuristic = all_results.get("heuristic").cloned().unwrap_or_default();

    let mut all_found: Vec<&Entity> = Vec::new();
    let mut only_stacked: Vec<&Entity> = Vec::new();

    for e in &stacked {
        let in_pattern = pattern.iter().any(|p| p.start == e.start && p.end == e.end);
        let in_heuristic = heuristic
            .iter()
            .any(|s| s.start == e.start && s.end == e.end);

        if in_pattern || in_heuristic {
            all_found.push(e);
        } else {
            only_stacked.push(e);
        }
    }

    // Count entities unique to each model
    let pattern_only_count = pattern
        .iter()
        .filter(|p| !stacked.iter().any(|s| s.start == p.start && s.end == p.end))
        .count();
    let heuristic_only_count = heuristic
        .iter()
        .filter(|h| !stacked.iter().any(|s| s.start == h.start && s.end == h.end))
        .count();

    println!(
        "  Agreed (in stacked from pattern/heuristic): {} entities",
        all_found.len()
    );
    println!(
        "  Pattern-only (not in stacked): {} entities",
        pattern_only_count
    );
    println!(
        "  Heuristic-only (not in stacked): {} entities",
        heuristic_only_count
    );
    println!(
        "  Stacked-only (novel combinations): {} entities",
        only_stacked.len()
    );
    println!();

    // Show annotated text
    println!("{}:", color("1;33", "Annotated Text"));
    print_annotated_entities(&text, &stacked);
    println!();

    Ok(())
}

fn cmd_dataset(args: DatasetArgs) -> Result<(), String> {
    match args.action {
        DatasetAction::List => {
            println!();
            println!("{}", color("1;36", "Available Datasets"));
            println!();

            #[cfg(feature = "eval-advanced")]
            {
                println!("  Downloadable (with --features eval-advanced):");
                println!("    - wikigold    : WikiGold NER corpus");
                println!("    - wnut17      : WNUT 2017 emerging entities");
                println!("    - conll2003   : CoNLL 2003 (requires manual download)");
            }

            println!();
            println!("  Synthetic (always available):");
            println!("    - synthetic   : Generated test cases");
            println!("    - robustness  : Adversarial perturbations");
            println!();
        }
        DatasetAction::Info { dataset } => {
            println!("Dataset: {}", dataset);
            // TODO: Show stats about the dataset
        }
        DatasetAction::Eval {
            dataset,
            model,
            task,
        } => {
            #[cfg(feature = "eval")]
            {
                let m = model.create_model()?;

                let (name, test_cases) = if dataset == "synthetic" {
                    (
                        "synthetic".to_string(),
                        vec![
                            (
                                "Marie Curie won the Nobel Prize.".to_string(),
                                vec![
                                    anno::eval::GoldEntity {
                                        text: "Marie Curie".to_string(),
                                        original_label: "PER".to_string(),
                                        entity_type: anno::EntityType::Person,
                                        start: 0,
                                        end: 11,
                                    },
                                    anno::eval::GoldEntity {
                                        text: "Nobel Prize".to_string(),
                                        original_label: "MISC".to_string(),
                                        entity_type: anno::EntityType::Other("MISC".to_string()),
                                        start: 20,
                                        end: 31,
                                    },
                                ],
                            ),
                            (
                                "Apple Inc. is based in California.".to_string(),
                                vec![
                                    anno::eval::GoldEntity {
                                        text: "Apple Inc.".to_string(),
                                        original_label: "ORG".to_string(),
                                        entity_type: anno::EntityType::Organization,
                                        start: 0,
                                        end: 10,
                                    },
                                    anno::eval::GoldEntity {
                                        text: "California".to_string(),
                                        original_label: "LOC".to_string(),
                                        entity_type: anno::EntityType::Location,
                                        start: 24,
                                        end: 34,
                                    },
                                ],
                            ),
                            (
                                "Contact john@example.com for help.".to_string(),
                                vec![anno::eval::GoldEntity {
                                    text: "john@example.com".to_string(),
                                    original_label: "EMAIL".to_string(),
                                    entity_type: anno::EntityType::Other("EMAIL".to_string()),
                                    start: 8,
                                    end: 24,
                                }],
                            ),
                        ],
                    )
                } else {
                    // Parse dataset ID
                    let dataset_id: DatasetId = dataset
                        .parse::<DatasetId>()
                        .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                    #[cfg(not(feature = "eval-advanced"))]
                    {
                        let _ = dataset_id; // Suppress unused warning
                        return Err(
                            "Loading real datasets requires --features eval-advanced".to_string()
                        );
                    }

                    #[cfg(feature = "eval-advanced")]
                    {
                        use anno::eval::loader::DatasetLoader;

                        let loader = DatasetLoader::new()
                            .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                        println!(
                            "Loading {} (may download if not cached)...",
                            dataset_id.name()
                        );
                        let ds = loader
                            .load_or_download(dataset_id)
                            .map_err(|e| format!("Failed to load dataset: {}", e))?;

                        // Only warn if evaluating NER on non-NER dataset (not for coref/relation tasks)
                        if matches!(task, EvalTask::Ner)
                            && (dataset_id.is_coreference() || dataset_id.is_relation_extraction())
                        {
                            println!("{} Warning: Evaluating NER on non-NER dataset. Results may be empty.", color("33", "!"));
                        }

                        (ds.stats().name, ds.to_test_cases())
                    }
                };

                // Route to appropriate evaluation based on task
                match task {
                    EvalTask::Ner => {
                        #[cfg(feature = "eval-advanced")]
                        let type_mapper: Option<anno::TypeMapper> = if dataset != "synthetic" {
                            dataset
                                .parse::<DatasetId>()
                                .ok()
                                .and_then(|id| id.type_mapper())
                        } else {
                            None
                        };
                        #[cfg(not(feature = "eval-advanced"))]
                        let type_mapper: Option<anno::TypeMapper> = None;

                        println!();
                        println!("Evaluating {} on {} dataset (NER)...", model.name(), name);
                        if type_mapper.is_some() {
                            println!(
                                "  {} Using type mapping for domain-specific dataset",
                                color("33", "!")
                            );
                        }
                        println!("  Sentences: {}", test_cases.len());
                        println!();

                        let mut total_gold = 0;
                        let mut total_pred = 0;
                        let mut total_correct = 0;

                        let start_time = Instant::now();

                        // Validate gold annotations before evaluation (warn but continue)
                        #[cfg(feature = "eval-advanced")]
                        {
                            use anno::eval::validation::validate_ground_truth_entities;
                            for (text, gold) in &test_cases {
                                let validation = validate_ground_truth_entities(text, gold, false);
                                if !validation.is_valid {
                                    eprintln!(
                                        "{} Invalid gold annotations: {}",
                                        color("33", "warning:"),
                                        validation.errors.join("; ")
                                    );
                                }
                                // Note: Warnings are typically non-critical (e.g., overlapping entities)
                                // Only show first few warnings to avoid spam
                                if !validation.warnings.is_empty() && validation.warnings.len() <= 3
                                {
                                    for warning in validation.warnings.iter().take(3) {
                                        eprintln!("{} {}", color("33", "warning:"), warning);
                                    }
                                }
                            }
                        }

                        for (text, gold) in &test_cases {
                            let entities = m.extract_entities(text, None).unwrap_or_default();

                            total_gold += gold.len();
                            total_pred += entities.len();

                            // Track which predictions have been matched to prevent double-counting
                            let mut matched_pred = vec![false; entities.len()];

                            for gold_entity in gold {
                                // Apply type mapping if available
                                let gold_type = if let Some(ref mapper) = type_mapper {
                                    mapper.normalize(&gold_entity.original_label)
                                } else {
                                    anno::EntityType::from_label(&gold_entity.original_label)
                                };

                                // Match: exact span + type match (with flexible type matching)
                                // Find first unmatched prediction that matches
                                let matched = entities.iter().enumerate().any(|(i, e)| {
                                    if matched_pred[i] {
                                        return false; // Already matched
                                    }

                                    let span_match =
                                        e.start == gold_entity.start && e.end == gold_entity.end;
                                    if !span_match {
                                        return false;
                                    }

                                    // Type match with flexible matching
                                    let pred_type_str = e.entity_type.as_label();
                                    let gold_type_str = gold_type.as_label();

                                    // Exact match or flexible match
                                    let type_matches = pred_type_str == gold_type_str
                                        || types_match_flexible(pred_type_str, gold_type_str);

                                    if type_matches {
                                        matched_pred[i] = true; // Mark as matched
                                        return true;
                                    }

                                    false
                                });

                                if matched {
                                    total_correct += 1;
                                }
                            }
                        }

                        let elapsed = start_time.elapsed();

                        let p = if total_pred > 0 {
                            total_correct as f64 / total_pred as f64
                        } else {
                            0.0
                        };
                        let r = if total_gold > 0 {
                            total_correct as f64 / total_gold as f64
                        } else {
                            0.0
                        };
                        let f1 = if p + r > 0.0 {
                            2.0 * p * r / (p + r)
                        } else {
                            0.0
                        };

                        println!("Results:");
                        println!(
                            "  Gold: {}  Predicted: {}  Correct: {}",
                            total_gold, total_pred, total_correct
                        );
                        println!(
                            "  P: {:.1}%  R: {:.1}%  F1: {:.1}%",
                            p * 100.0,
                            r * 100.0,
                            f1 * 100.0
                        );
                        let ms_per_sent = if !test_cases.is_empty() {
                            elapsed.as_secs_f64() * 1000.0 / test_cases.len() as f64
                        } else {
                            0.0
                        };
                        println!(
                            "  Time: {:.1}s ({:.1}ms/sent)",
                            elapsed.as_secs_f64(),
                            ms_per_sent
                        );
                        println!();
                    }
                    EvalTask::Coref => {
                        #[cfg(not(feature = "eval-advanced"))]
                        {
                            return Err("Coreference evaluation requires --features eval-advanced"
                                .to_string());
                        }
                        #[cfg(feature = "eval-advanced")]
                        {
                            use anno::eval::coref_resolver::SimpleCorefResolver;
                            use anno::eval::loader::DatasetLoader;

                            if dataset == "synthetic" {
                                return Err("Coreference evaluation requires a real dataset (e.g., gap, preco, litbank)".to_string());
                            }

                            let dataset_id: DatasetId = dataset
                                .parse::<DatasetId>()
                                .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                            if !dataset_id.is_coreference() {
                                return Err(format!("Dataset '{}' is not a coreference dataset. Use: gap, preco, or litbank", dataset));
                            }

                            let loader = DatasetLoader::new()
                                .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                            println!();
                            println!(
                                "Evaluating coreference resolution on {} dataset...",
                                dataset_id.name()
                            );
                            println!("Loading dataset (may download if not cached)...");

                            let gold_docs =
                                loader.load_or_download_coref(dataset_id).map_err(|e| {
                                    format!("Failed to load coreference dataset: {}", e)
                                })?;

                            println!("  Documents: {}", gold_docs.len());
                            println!();

                            let resolver = SimpleCorefResolver::default();
                            let mut all_pred_chains: Vec<Vec<anno::eval::coref::CorefChain>> =
                                Vec::new();
                            let mut all_gold_chains: Vec<&[anno::eval::coref::CorefChain]> =
                                Vec::new();
                            let start_time = Instant::now();

                            for doc in &gold_docs {
                                let text = doc.text.as_str();
                                all_gold_chains.push(&doc.chains);

                                // Extract entities using NER
                                let entities = m.extract_entities(text, None).unwrap_or_default();

                                // Resolve coreference
                                let pred_chains = resolver.resolve_to_chains(&entities);
                                all_pred_chains.push(pred_chains);
                            }

                            let elapsed = start_time.elapsed();

                            // Build document pairs
                            let document_pairs: Vec<_> = all_pred_chains
                                .iter()
                                .zip(all_gold_chains.iter())
                                .map(|(pred, gold)| (pred.as_slice(), *gold))
                                .collect();

                            // Compute aggregate metrics
                            let results =
                                anno::eval::coref_metrics::AggregateCorefEvaluation::compute(
                                    &document_pairs,
                                );

                            println!("Results:");
                            println!("  CoNLL F1: {:.3}", results.mean.conll_f1);
                            println!(
                                "  MUC: P={:.3} R={:.3} F1={:.3}",
                                results.mean.muc.precision,
                                results.mean.muc.recall,
                                results.mean.muc.f1
                            );
                            println!(
                                "  B³: P={:.3} R={:.3} F1={:.3}",
                                results.mean.b_cubed.precision,
                                results.mean.b_cubed.recall,
                                results.mean.b_cubed.f1
                            );
                            println!(
                                "  CEAF-e: P={:.3} R={:.3} F1={:.3}",
                                results.mean.ceaf_e.precision,
                                results.mean.ceaf_e.recall,
                                results.mean.ceaf_e.f1
                            );
                            println!(
                                "  LEA: P={:.3} R={:.3} F1={:.3}",
                                results.mean.lea.precision,
                                results.mean.lea.recall,
                                results.mean.lea.f1
                            );
                            println!(
                                "  BLANC: P={:.3} R={:.3} F1={:.3}",
                                results.mean.blanc.precision,
                                results.mean.blanc.recall,
                                results.mean.blanc.f1
                            );
                            println!("  Documents: {}", results.num_documents);
                            println!("  Time: {:.1}s", elapsed.as_secs_f64());
                            println!();
                        }
                    }
                    EvalTask::Relation => {
                        #[cfg(not(feature = "eval-advanced"))]
                        {
                            return Err(
                                "Relation extraction evaluation requires --features eval-advanced"
                                    .to_string(),
                            );
                        }
                        #[cfg(feature = "eval-advanced")]
                        {
                            use anno::backends::inference::RelationExtractor;
                            use anno::eval::loader::DatasetLoader;
                            use anno::eval::relation::{
                                evaluate_relations, RelationEvalConfig, RelationPrediction,
                            };

                            if dataset == "synthetic" {
                                return Err("Relation extraction evaluation requires a real dataset (e.g., docred, retacred)".to_string());
                            }

                            let dataset_id: DatasetId = dataset
                                .parse::<DatasetId>()
                                .map_err(|e| format!("Invalid dataset '{}': {}", dataset, e))?;

                            if !dataset_id.is_relation_extraction() {
                                return Err(format!("Dataset '{}' is not a relation extraction dataset. Use: docred or retacred", dataset));
                            }

                            let loader = DatasetLoader::new()
                                .map_err(|e| format!("Failed to init dataset loader: {}", e))?;

                            println!();
                            println!(
                                "Evaluating relation extraction on {} dataset...",
                                dataset_id.name()
                            );
                            println!("Loading dataset (may download if not cached)...");

                            let gold_docs = loader
                                .load_or_download_relation(dataset_id)
                                .map_err(|e| format!("Failed to load relation dataset: {}", e))?;

                            println!("  Documents: {}", gold_docs.len());
                            println!();

                            // Try to use RelationExtractor if available (e.g., GLiNER2)
                            // Otherwise fall back to entity-pair heuristic

                            // Collect entity types and relation types from gold data
                            let mut entity_types = std::collections::HashSet::new();
                            let mut relation_types = std::collections::HashSet::new();
                            for doc in &gold_docs {
                                for rel in &doc.relations {
                                    entity_types.insert(rel.head_type.clone());
                                    entity_types.insert(rel.tail_type.clone());
                                    relation_types.insert(rel.relation_type.clone());
                                }
                            }

                            let entity_types_vec: Vec<&str> =
                                entity_types.iter().map(|s| s.as_str()).collect();
                            let relation_types_vec: Vec<&str> =
                                relation_types.iter().map(|s| s.as_str()).collect();

                            println!("  Entity types: {}", entity_types_vec.join(", "));
                            println!(
                                "  Relation types: {} ({} total)",
                                relation_types_vec.len(),
                                relation_types_vec
                                    .iter()
                                    .take(5)
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            );
                            println!();

                            // Check if we can use GLiNER2 for relation extraction
                            let use_relation_extractor: Option<Box<dyn RelationExtractor>> = {
                                #[cfg(feature = "onnx")]
                                {
                                    // Try to create GLiNER2 multitask model for relation extraction
                                    if let Ok(gliner2) =
                                        anno::backends::gliner2::GLiNER2Onnx::from_pretrained(
                                            "onnx-community/gliner-multitask-large-v0.5",
                                        )
                                    {
                                        Some(Box::new(gliner2) as Box<dyn RelationExtractor>)
                                    } else {
                                        None
                                    }
                                }
                                #[cfg(not(feature = "onnx"))]
                                {
                                    None
                                }
                            };

                            let mut all_gold = Vec::new();
                            let mut all_pred = Vec::new();
                            let start_time = Instant::now();

                            if let Some(ref rel_extractor) = use_relation_extractor {
                                println!("{} Using GLiNER2 RelationExtractor (heuristic-based regex matching)", color("32", "✓"));
                                println!("  Note: This uses regex matching on text, not a neural relation model.",);
                                println!();

                                for doc in &gold_docs {
                                    let text = doc.text.as_str();
                                    all_gold.extend(doc.relations.clone());

                                    // Use RelationExtractor
                                    match rel_extractor.extract_with_relations(
                                        text,
                                        &entity_types_vec,
                                        &relation_types_vec,
                                        0.5,
                                    ) {
                                        Ok(result) => {
                                            // Convert RelationTriples to RelationPredictions
                                            for triple in &result.relations {
                                                if let Some(pred) =
                                                    RelationPrediction::from_triple_with_entities(
                                                        triple,
                                                        &result.entities,
                                                    )
                                                {
                                                    all_pred.push(pred);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!(
                                                "{} Relation extraction failed: {}",
                                                color("33", "warning:"),
                                                e
                                            );
                                            // Fall back to entity-pair heuristic for this document
                                            let entities =
                                                m.extract_entities(text, None).unwrap_or_default();
                                            all_pred.extend(create_entity_pair_relations(
                                                &entities,
                                                text,
                                                &relation_types_vec,
                                            ));
                                        }
                                    }
                                }
                            } else {
                                println!("{} Using entity-pair heuristic (GLiNER2 multitask not available)", color("33", "!"));
                                println!();

                                for doc in &gold_docs {
                                    let text = doc.text.as_str();
                                    all_gold.extend(doc.relations.clone());

                                    // Extract entities using NER
                                    let entities =
                                        m.extract_entities(text, None).unwrap_or_default();

                                    // Create relation predictions from entity pairs
                                    all_pred.extend(create_entity_pair_relations(
                                        &entities,
                                        text,
                                        &relation_types_vec,
                                    ));
                                }
                            }

                            let elapsed = start_time.elapsed();

                            // Evaluate relations
                            // Note: require_entity_type_match=false because entity types may differ
                            // (e.g., gold uses "person" but pred uses "Person", or "PER" vs "PERSON")
                            let config = RelationEvalConfig {
                                overlap_threshold: 0.5,
                                require_entity_type_match: false, // More lenient for evaluation
                                directed_relations: true,
                            };
                            let metrics = evaluate_relations(&all_gold, &all_pred, &config);

                            // Output results (human-readable by default)
                            println!();
                            println!("{}", color("1;36", "======================================================================="));
                            println!(
                                "  {}  model={}  time={:.1}s",
                                color("1;36", "RELATION EXTRACTION EVALUATION"),
                                model.name(),
                                elapsed.as_secs_f64()
                            );
                            println!("{}", color("1;36", "======================================================================="));
                            println!();
                            println!("{}", metrics.to_string_human(false)); // verbose=false for now
                            println!();
                        }
                    }
                }
            }
            #[cfg(not(feature = "eval"))]
            {
                let _ = (dataset, model, task);
                return Err("Dataset evaluation requires --features eval".to_string());
            }
        }
    }

    Ok(())
}

#[cfg(feature = "eval-advanced")]
fn cmd_benchmark(args: BenchmarkArgs) -> Result<(), String> {
    use anno::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
    use anno::eval::task_mapping::Task;
    use std::fs;

    println!("=== Comprehensive Task-Dataset-Backend Evaluation ===\n");

    // Parse tasks
    let tasks = if let Some(task_strs) = args.tasks {
        let mut parsed = Vec::new();
        for t in task_strs {
            match t.to_lowercase().as_str() {
                "ner" | "ner_task" => parsed.push(Task::NER),
                "coref" | "coreference" | "intradoc_coref" => parsed.push(Task::IntraDocCoref),
                "relation" | "relation_extraction" => parsed.push(Task::RelationExtraction),
                other => {
                    return Err(format!(
                        "Unknown task: {}. Use: ner, coref, relation",
                        other
                    ));
                }
            }
        }
        parsed
    } else {
        Task::all().to_vec()
    };

    // Parse datasets
    let datasets = if let Some(dataset_strs) = args.datasets {
        let mut parsed = Vec::new();
        for d in dataset_strs {
            let dataset_id: DatasetId = d
                .parse()
                .map_err(|e| format!("Invalid dataset '{}': {}", d, e))?;
            parsed.push(dataset_id);
        }
        parsed
    } else {
        vec![] // Empty = use all suitable datasets
    };

    // Parse backends
    let backends = args.backends.unwrap_or_default();

    // Create evaluator
    let evaluator =
        TaskEvaluator::new().map_err(|e| format!("Failed to create evaluator: {}", e))?;

    // Configure evaluation using builder pattern
    use anno::eval::config_builder::TaskEvalConfigBuilder;
    let mut builder = TaskEvalConfigBuilder::new()
        .with_tasks(tasks)
        .with_datasets(datasets)
        .with_backends(backends)
        .require_cached(args.cached_only)
        .with_confidence_intervals(true)
        .with_familiarity(true);

    // Set max_examples (None means "all examples", 0 also means "all examples")
    if let Some(max) = args.max_examples {
        if max > 0 {
            builder = builder.with_max_examples(max);
        }
        // If max == 0, don't set it (None = unlimited)
    }

    // Only set seed if provided (default is 42 in builder)
    if let Some(seed) = args.seed {
        builder = builder.with_seed(seed);
    }

    let config = builder.build();

    println!("Running comprehensive evaluation...");
    println!("Tasks: {:?}", config.tasks);
    if !config.datasets.is_empty() {
        println!("Datasets: {:?}", config.datasets);
    } else {
        println!("Datasets: all suitable datasets");
    }
    if !config.backends.is_empty() {
        println!("Backends: {:?}", config.backends);
    } else {
        println!("Backends: all compatible backends");
    }
    if let Some(max) = config.max_examples {
        println!("Max examples per dataset: {}", max);
    }
    if let Some(seed) = config.seed {
        println!("Random seed: {}", seed);
    }
    println!();

    // Run evaluation
    let results = evaluator
        .evaluate_all(config)
        .map_err(|e| format!("Evaluation failed: {}", e))?;

    // Print summary
    println!("=== Evaluation Summary ===");
    println!("Total combinations: {}", results.summary.total_combinations);
    println!("Successful: {}", results.summary.successful);
    println!(
        "Skipped (feature not available): {}",
        results.summary.skipped
    );
    println!("Failed (actual errors): {}", results.summary.failed);
    println!("\nTasks evaluated: {}", results.summary.tasks.len());
    println!("Datasets used: {}", results.summary.datasets.len());
    println!("Backends tested: {}", results.summary.backends.len());
    println!();

    // Generate markdown report
    let report = results.to_markdown();

    // Output report
    if let Some(output_path) = &args.output {
        fs::write(output_path, &report)
            .map_err(|e| format!("Failed to write report to {}: {}", output_path, e))?;
        println!("Report saved to: {}", output_path);
    } else {
        println!("=== Markdown Report ===");
        println!("{}", report);
    }

    Ok(())
}

/// Create relation predictions from entity pairs using heuristics.
///
/// # Bugs Fixed:
/// - Character vs byte offset: Now uses character offsets consistently
/// - Bounds validation: Validates entity spans are within text bounds
/// - Distance limit: Configurable (default 200 chars) to catch cross-sentence relations
#[cfg(feature = "eval-advanced")]
fn create_entity_pair_relations(
    entities: &[Entity],
    text: &str,
    relation_types: &[&str],
) -> Vec<anno::eval::relation::RelationPrediction> {
    use anno::eval::relation::RelationPrediction;

    let text_char_len = text.chars().count();
    let max_distance = 200; // Increased from 100 to catch cross-sentence relations

    let mut pred_relations = Vec::new();

    // Validate entities first to avoid panics
    let valid_entities: Vec<&Entity> = entities
        .iter()
        .filter(|e| e.start < e.end && e.end <= text_char_len && e.start < text_char_len)
        .collect();

    // Limit to avoid O(n²) explosion with many entities
    // Only consider pairs from first 50 entities to keep it tractable
    let max_entities = 50.min(valid_entities.len());

    for i in 0..max_entities {
        for j in (i + 1)..max_entities {
            let head = valid_entities[i];
            let tail = valid_entities[j];

            // Calculate distance using character offsets
            let distance = if tail.start >= head.end {
                tail.start - head.end
            } else if head.start >= tail.end {
                head.start - tail.end
            } else {
                // Overlapping entities - skip (they can't have a relation)
                continue;
            };

            if distance > max_distance {
                continue;
            }

            // Extract text between entities using character offsets (not byte offsets)
            let between_text = if head.end <= tail.start {
                text.chars()
                    .skip(head.end)
                    .take(tail.start - head.end)
                    .collect::<String>()
            } else {
                text.chars()
                    .skip(tail.end)
                    .take(head.start - tail.end)
                    .collect::<String>()
            };

            // Simple regex matching for common relations
            let between_lower = between_text.to_lowercase();
            let rel_type = if between_lower.contains("founded") || between_lower.contains("founder")
            {
                "FOUNDED"
            } else if between_lower.contains("works for")
                || between_lower.contains("employee")
                || between_lower.contains("employed")
            {
                "WORKS_FOR"
            } else if between_lower.contains("located in")
                || between_lower.contains("based in")
                || between_lower.contains("in ")
            {
                "LOCATED_IN"
            } else if between_lower.contains("born in") {
                "BORN_IN"
            } else {
                // Use first relation type from gold data as fallback, or "RELATED"
                relation_types.first().copied().unwrap_or("RELATED")
            };

            pred_relations.push(RelationPrediction {
                head_span: (head.start, head.end),
                head_type: head.entity_type.as_label().to_string(),
                tail_span: (tail.start, tail.end),
                tail_type: tail.entity_type.as_label().to_string(),
                relation_type: rel_type.to_string(),
                confidence: 0.5,
            });
        }
    }

    pred_relations
}

fn cmd_info() -> Result<(), String> {
    println!();
    println!("{}", color("1;36", "anno"));
    println!("  Information Extraction: NER + Coreference + Relations + Entity Linking");
    println!();
    println!("{}:", color("1;33", "Version"));
    println!("  {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("{}:", color("1;33", "Available Models (this build)"));

    // Use the actual available_backends() function to show real availability
    let backends = anno::available_backends();
    for (name, available) in backends {
        let status = if available {
            color("32", "✓")
        } else {
            color("90", "✗")
        };
        let note = if available {
            ""
        } else {
            " (requires feature flag)"
        };
        println!("  {} {} {}", status, name, note);
    }
    println!();

    let model = StackedNER::default();
    println!("{}:", color("1;33", "Supported Entity Types (stacked)"));
    for t in model.supported_types() {
        let color_code = type_color(t.as_label());
        println!("  {} {}", color(color_code, "*"), t.as_label());
    }
    println!();

    println!("{}:", color("1;33", "Enabled Features"));
    let mut features: Vec<&str> = Vec::new();
    #[cfg(feature = "onnx")]
    features.push("onnx");
    #[cfg(feature = "candle")]
    features.push("candle");
    #[cfg(feature = "eval")]
    features.push("eval");
    #[cfg(feature = "eval-bias")]
    features.push("eval-bias");
    #[cfg(feature = "eval-advanced")]
    features.push("eval-advanced");
    #[cfg(feature = "discourse")]
    features.push("discourse");
    if features.is_empty() {
        println!("  (default features only)");
    } else {
        println!("  {}", features.join(", "));
    }
    println!();

    Ok(())
}

fn cmd_models(args: ModelsArgs) -> Result<(), String> {
    match args.action {
        ModelsAction::List => {
            println!();
            println!("{}", color("1;36", "Available Models"));
            println!();

            let backends = anno::available_backends();
            for (name, available) in backends {
                let status = if available {
                    color("32", "✓ Available")
                } else {
                    color("90", "✗ Not available")
                };
                let note = if available {
                    ""
                } else {
                    " (requires feature flag - see anno info)"
                };
                println!("  {} {}{}", status, name, note);
            }
            println!();
            println!(
                "Use 'anno models info <MODEL>' for detailed information about a specific model."
            );
            println!();
        }
        ModelsAction::Info { model } => {
            println!();
            println!("{}: {}", color("1;36", "Model Information"), model);
            println!();

            let backends = anno::available_backends();
            // Try to find model by exact name or common aliases
            let model_lower = model.to_lowercase();
            let found = backends.iter().find(|(n, _)| {
                n.eq_ignore_ascii_case(&model)
                    || (model_lower == "stacked" && n.eq_ignore_ascii_case("StackedNER"))
                    || (model_lower == "pattern" && n.eq_ignore_ascii_case("RegexNER"))
                    || (model_lower == "heuristic" && n.eq_ignore_ascii_case("HeuristicNER"))
                    || (model_lower == "gliner" && n.eq_ignore_ascii_case("GLiNEROnnx"))
                    || (model_lower == "bert" && n.eq_ignore_ascii_case("BertNEROnnx"))
            });

            let (name, available) = if let Some((n, a)) = found {
                (*n, *a)
            } else {
                // Model not found - provide helpful suggestions
                let backends_list: Vec<&str> = backends.iter().map(|(n, _)| *n).collect();
                let suggestions = find_similar_models(&model, &backends_list);
                let mut err_msg = format!("Model '{}' not found.", model);
                if !suggestions.is_empty() {
                    err_msg.push_str(&format!("\n  Did you mean: {}?", suggestions.join(", ")));
                }
                err_msg.push_str("\n  Use 'anno models list' to see all available models.");
                return Err(err_msg);
            };

            if !available {
                println!(
                    "  {} This model is not available in this build.",
                    color("31", "Error:")
                );
                println!();
                println!("  To enable this model:");
                match model.to_lowercase().as_str() {
                    "glineronnx" | "gliner" | "nuner" | "w2ner" | "bertneronnx" => {
                        println!("    cargo build --features onnx");
                    }
                    "candlener" | "glinercandle" => {
                        println!("    cargo build --features candle");
                    }
                    _ => {
                        println!("    Check the model name and required features.");
                    }
                }
                println!();
                return Ok(());
            }

            // Show model details
            // Normalize name for matching (handle both full names and aliases)
            let name_lower_str = if name == "StackedNER" {
                "stacked"
            } else if name == "RegexNER" {
                "pattern"
            } else if name == "HeuristicNER" {
                "heuristic"
            } else if name == "GLiNEROnnx" {
                "gliner"
            } else if name == "BertNEROnnx" {
                "bert"
            } else {
                &name.to_lowercase()
            };

            match name_lower_str {
                "pattern" | "regexner" => {
                    println!("  Type: Pattern-based NER");
                    println!("  Speed: ~400ns per entity");
                    println!("  Accuracy: ~95% on structured entities");
                    println!("  Entity Types: DATE, TIME, MONEY, EMAIL, URL, PHONE");
                    println!("  Use Case: Fast structured data extraction");
                }
                "heuristic" | "heuristicner" => {
                    println!("  Type: Heuristic-based NER");
                    println!("  Speed: ~50μs per entity");
                    println!("  Accuracy: ~65% F1 on CoNLL-2003");
                    println!("  Entity Types: PER, ORG, LOC");
                    println!("  Use Case: Quick baseline, no dependencies");
                }
                "stacked" | "stackedner" => {
                    println!("  Type: Composable layered extraction");
                    println!("  Speed: ~100μs per entity");
                    println!("  Accuracy: Varies by composition");
                    println!("  Entity Types: All (combines Pattern + Heuristic)");
                    println!("  Use Case: Default, combines patterns + heuristics");
                }
                "gliner" | "glineronnx" => {
                    println!("  Type: Zero-shot NER (bi-encoder)");
                    println!("  Speed: ~100ms per entity");
                    println!("  Accuracy: ~92% F1 on CoNLL-2003, ~60% on CrossNER");
                    println!("  Entity Types: Any (zero-shot, custom types)");
                    println!("  Use Case: Custom entity types without retraining");
                    println!("  Feature: Requires 'onnx' feature flag");
                }
                "gliner2" => {
                    println!("  Type: Multi-task (NER + classification + relations)");
                    println!("  Speed: ~130ms per entity");
                    println!("  Accuracy: ~92% F1 on NER, supports classification");
                    println!("  Entity Types: Any (zero-shot) + text classification");
                    println!("  Use Case: Joint NER and text classification");
                    println!("  Feature: Requires 'onnx' feature flag");
                }
                "nuner" => {
                    println!("  Type: Zero-shot NER (token-based)");
                    println!("  Speed: ~100ms per entity");
                    println!("  Accuracy: ~86% F1 on CoNLL-2003");
                    println!("  Entity Types: Any (zero-shot)");
                    println!("  Use Case: Alternative zero-shot approach");
                    println!("  Feature: Requires 'onnx' feature flag");
                }
                "w2ner" => {
                    println!("  Type: Nested/discontinuous NER");
                    println!("  Speed: ~150ms per entity");
                    println!("  Accuracy: ~85% F1 on CoNLL-2003");
                    println!("  Entity Types: Fixed (PER, ORG, LOC, MISC)");
                    println!("  Use Case: Overlapping or non-contiguous entities");
                    println!("  Feature: Requires 'onnx' feature flag");
                }
                "bertneronnx" => {
                    println!("  Type: High-quality NER (fixed types)");
                    println!("  Speed: ~50ms per entity");
                    println!("  Accuracy: ~86% F1 on CoNLL-2003");
                    println!("  Entity Types: PER, ORG, LOC, MISC");
                    println!("  Use Case: Standard 4-type NER");
                    println!("  Feature: Requires 'onnx' feature flag");
                }
                "candlener" => {
                    println!("  Type: Pure Rust BERT NER");
                    println!("  Speed: Varies (CPU/GPU)");
                    println!("  Accuracy: ~86% F1 on CoNLL-2003");
                    println!("  Entity Types: PER, ORG, LOC, MISC");
                    println!("  Use Case: Rust-native, no ONNX dependency");
                    println!("  Feature: Requires 'candle' feature flag");
                }
                _ => {
                    println!("  Type: Unknown");
                    println!("  Use 'anno models list' to see all available models.");
                }
            }
            println!();
        }
        ModelsAction::Compare => {
            println!();
            println!("{}", color("1;36", "Model Comparison"));
            println!();

            let backends = anno::available_backends();
            let available: Vec<_> = backends
                .into_iter()
                .filter(|(_, avail)| *avail)
                .map(|(name, _)| name)
                .collect();

            if available.is_empty() {
                println!("  No models available. Build with feature flags to enable models.");
                println!();
                return Ok(());
            }

            println!(
                "  {:<20} {:<15} {:<15} {:<30}",
                "Model", "Speed", "Accuracy", "Use Case"
            );
            println!("  {}", "-".repeat(80));

            for name in &available {
                let (speed, accuracy, use_case) = match name.to_lowercase().as_str() {
                    "pattern" | "regexner" => ("~400ns", "~95%", "Structured entities"),
                    "heuristic" | "heuristicner" => ("~50μs", "~65% F1", "Quick baseline"),
                    "stacked" | "stackedner" => ("~100μs", "Varies", "Default (composable)"),
                    "gliner" | "glineronnx" => ("~100ms", "~92% F1", "Zero-shot NER"),
                    "gliner2" => ("~130ms", "~92% F1", "Multi-task (NER+classify)"),
                    "nuner" => ("~100ms", "~86% F1", "Zero-shot (token-based)"),
                    "w2ner" => ("~150ms", "~85% F1", "Nested entities"),
                    "bertneronnx" => ("~50ms", "~86% F1", "Standard 4-type NER"),
                    "candlener" => ("Varies", "~86% F1", "Rust-native"),
                    _ => ("Unknown", "Unknown", "Unknown"),
                };
                println!(
                    "  {:<20} {:<15} {:<15} {:<30}",
                    name, speed, accuracy, use_case
                );
            }
            println!();
        }
    }

    Ok(())
}

/// Enhance existing GroundedDocument with additional processing layers
fn cmd_enhance(args: EnhanceArgs) -> Result<(), String> {
    // Load GroundedDocument from file or stdin
    let json_content = if args.input == "-" {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("Failed to read stdin: {}", e))?;
        buf
    } else {
        fs::read_to_string(&args.input)
            .map_err(|e| format!("Failed to read {}: {}", args.input, e))?
    };

    let mut doc: GroundedDocument = serde_json::from_str(&json_content)
        .map_err(|e| format!("Failed to parse GroundedDocument JSON: {}", e))?;

    // Collect signal IDs for coreference
    let signal_ids: Vec<u64> = doc.signals().iter().map(|s| s.id).collect();

    // Apply enhancements
    if args.coref {
        let text = doc.text.clone();
        resolve_coreference(&mut doc, &text, &signal_ids);
        log_success("Applied coreference resolution", args.quiet);
    }

    if args.link_kb {
        link_tracks_to_kb(&mut doc);
        log_success("Applied KB linking", args.quiet);
    }

    // Export if requested
    if let Some(export_path) = args.export {
        let export_data = match args.export_format.as_str() {
            "full" => serde_json::to_value(&doc)
                .map_err(|e| format!("Failed to serialize GroundedDocument: {}", e))?,
            "signals" => {
                let signals: Vec<_> = doc.signals().iter().cloned().collect();
                serde_json::json!({
                    "id": doc.id,
                    "text": doc.text,
                    "signals": signals
                })
            }
            "minimal" => {
                let signals: Vec<_> = doc
                    .signals()
                    .iter()
                    .map(|s| {
                        let (start, end) = s.text_offsets().unwrap_or((0, 0));
                        serde_json::json!({
                            "surface": s.surface(),
                            "label": s.label(),
                            "start": start,
                            "end": end,
                            "confidence": s.confidence
                        })
                    })
                    .collect();
                serde_json::json!({
                    "id": doc.id,
                    "text": doc.text,
                    "signals": signals
                })
            }
            _ => {
                return Err(format!(
                    "Invalid export format '{}'. Use: full, signals, or minimal",
                    args.export_format
                ));
            }
        };

        let json = serde_json::to_string_pretty(&export_data)
            .map_err(|e| format!("Failed to serialize export data: {}", e))?;

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&export_path).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "Failed to create directory for export file '{}': {}",
                        export_path, e
                    )
                })?;
            }
        }

        fs::write(&export_path, json)
            .map_err(|e| format!("Failed to write export file '{}': {}", export_path, e))?;
        if !args.quiet {
            eprintln!(
                "{} Exported {} format to {}",
                color("32", "✓"),
                args.export_format,
                export_path
            );
        }
    }

    // Output based on format
    match args.format {
        OutputFormat::Grounded | OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
        }
        OutputFormat::Human => {
            if !args.quiet {
                let stats = doc.stats();
                println!();
                println!("{}", color("1;36", "Enhanced Document"));
                println!("  Signals: {}", stats.signal_count);
                println!("  Tracks: {}", stats.track_count);
                println!("  Identities: {}", stats.identity_count);
                println!();
            }
            print_signals(&doc, &doc.text, false);
        }
        _ => {
            return Err(format!(
                "Format {:?} not supported for enhance command",
                args.format
            ));
        }
    }

    // Export to graph format if requested
    if let Some(graph_format_str) = args.export_graph {
        let graph_format = match graph_format_str.to_lowercase().as_str() {
            "neo4j" | "cypher" => GraphExportFormat::Cypher,
            "networkx" | "nx" => GraphExportFormat::NetworkXJson,
            "jsonld" | "json-ld" => GraphExportFormat::JsonLd,
            _ => {
                return Err(format!(
                    "Invalid graph format '{}'. Use: neo4j, networkx, or jsonld",
                    graph_format_str
                ));
            }
        };

        let graph = GraphDocument::from_grounded_document(&doc);
        let graph_output = graph.export(graph_format);

        // Output graph to stdout (always print to stdout for graph export)
        // Note: If user wants to save to file, they can use shell redirection: --export-graph neo4j > output.cypher
        if !args.quiet {
            eprintln!(
                "{} Exported graph ({} nodes, {} edges) in {} format",
                color("32", "✓"),
                graph.node_count(),
                graph.edge_count(),
                graph_format_str
            );
        }
        println!("{}", graph_output);
    }

    Ok(())
}

/// Unified pipeline command
fn cmd_pipeline(args: PipelineArgs) -> Result<(), String> {
    // Collect input texts
    let mut texts: Vec<(String, String)> = Vec::new(); // (id, text)

    if !args.text.is_empty() {
        for (idx, text) in args.text.iter().enumerate() {
            texts.push((format!("text{}", idx + 1), text.clone()));
        }
    }

    for file_path in &args.files {
        let text = fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read {}: {}", file_path, e))?;
        let doc_id = std::path::Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| file_path.clone());
        texts.push((doc_id, text));
    }

    if let Some(dir) = &args.dir {
        let dir_path = std::path::Path::new(dir);
        let entries = fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory {}: {}", dir, e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "txt" || ext == "md" {
                        let text = fs::read_to_string(&path)
                            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
                        let doc_id = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("doc{}", texts.len()));
                        texts.push((doc_id, text));
                    }
                }
            }
        }
    }

    if texts.is_empty() {
        return Err("No input provided. Use --text, --files, or --dir".to_string());
    }

    // Process each document
    let model = args.model.create_model()?;
    let mut documents: Vec<GroundedDocument> = Vec::new();

    #[cfg(all(feature = "cli", feature = "eval"))]
    let pb = if args.progress && !args.quiet {
        use indicatif::{ProgressBar, ProgressStyle};
        let pb = ProgressBar::new(texts.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    #[cfg(not(all(feature = "cli", feature = "eval")))]
    let pb: Option<()> = None;

    for (doc_id, text) in &texts {
        #[cfg(all(feature = "cli", feature = "eval"))]
        if let Some(ref pb) = pb {
            pb.set_message(format!("Processing {}", doc_id));
        }

        // Extract entities
        let entities = model
            .extract_entities(text, None)
            .map_err(|e| format!("Extraction failed for {}: {}", doc_id, e))?;

        // Build GroundedDocument
        let mut doc = GroundedDocument::new(doc_id, text);
        let mut signal_ids: Vec<u64> = Vec::new();

        for e in &entities {
            let signal = Signal::new(
                0,
                Location::text(e.start, e.end),
                &e.text,
                e.entity_type.as_label(),
                e.confidence as f32,
            );
            let id = doc.add_signal(signal);
            signal_ids.push(id);
        }

        // Apply enhancements
        if args.coref {
            resolve_coreference(&mut doc, text, &signal_ids);
        }

        if args.link_kb {
            link_tracks_to_kb(&mut doc);
        }

        documents.push(doc);

        #[cfg(all(feature = "cli", feature = "eval"))]
        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    #[cfg(all(feature = "cli", feature = "eval"))]
    if let Some(ref pb) = pb {
        pb.finish_with_message("Processing complete");
    }

    // Cross-document clustering if requested
    if args.cross_doc {
        #[cfg(feature = "eval-advanced")]
        {
            use anno::eval::cdcr::{CDCRConfig, CDCRResolver, Document};

            // Convert GroundedDocuments to CDCR Documents
            let cdcr_docs: Vec<Document> = documents
                .iter()
                .map(|doc| {
                    let entities: Vec<_> = doc
                        .signals()
                        .iter()
                        .map(|s| {
                            let (start, end) = s.text_offsets().unwrap_or((0, 0));
                            use anno::EntityType;
                            Entity::new(
                                s.surface(),
                                EntityType::from_label(s.label()),
                                start,
                                end,
                                s.confidence as f64,
                            )
                        })
                        .collect();
                    Document::new(&doc.id, &doc.text).with_entities(entities)
                })
                .collect();

            let config = CDCRConfig {
                min_similarity: args.threshold,
                require_type_match: false,
                ..Default::default()
            };
            let resolver = CDCRResolver::with_config(config);
            let clusters = resolver.resolve(&cdcr_docs);

            // Output clusters
            match args.format {
                OutputFormat::Json | OutputFormat::Grounded => {
                    let output = serde_json::to_string_pretty(&clusters)
                        .map_err(|e| format!("Failed to serialize clusters: {}", e))?;
                    if let Some(output_path) = &args.output {
                        fs::write(output_path, output)
                            .map_err(|e| format!("Failed to write output: {}", e))?;
                    } else {
                        println!("{}", output);
                    }
                }
                OutputFormat::Tree => {
                    // Build doc_index for looking up entity text
                    let doc_index: std::collections::HashMap<_, _> =
                        cdcr_docs.iter().map(|doc| (doc.id.clone(), doc)).collect();

                    // Tree format output
                    for cluster in &clusters {
                        println!("Cluster {}: {}", cluster.id, cluster.canonical_name);
                        for (doc_id, entity_idx) in &cluster.mentions {
                            // Get entity text from document if available
                            let mention_text = doc_index
                                .get(doc_id.as_str())
                                .and_then(|doc| doc.entities.get(*entity_idx))
                                .map(|e| e.text.clone())
                                .unwrap_or_else(|| format!("entity_{}", entity_idx));
                            println!("  - {} (doc: {})", mention_text, doc_id);
                        }
                        println!();
                    }
                }
                _ => {
                    // Human-readable summary
                    println!();
                    println!(
                        "{} Cross-document clusters: {}",
                        color("1;36", "Found"),
                        clusters.len()
                    );
                    for cluster in &clusters {
                        println!(
                            "  {}: {} mentions across {} documents",
                            cluster.canonical_name,
                            cluster.mentions.len(),
                            cluster.doc_count()
                        );
                    }
                }
            }
        }

        #[cfg(not(feature = "eval-advanced"))]
        {
            return Err("Cross-document clustering requires 'eval-advanced' feature".to_string());
        }
    } else {
        // Output individual documents
        match args.format {
            OutputFormat::Json | OutputFormat::Grounded => {
                let output = serde_json::to_string_pretty(&documents)
                    .map_err(|e| format!("Failed to serialize documents: {}", e))?;
                if let Some(output_path) = &args.output {
                    fs::write(output_path, output)
                        .map_err(|e| format!("Failed to write output: {}", e))?;
                } else {
                    println!("{}", output);
                }
            }
            _ => {
                // Human-readable output
                for doc in &documents {
                    println!();
                    println!("{}", color("1;36", &format!("Document: {}", doc.id)));
                    print_signals(doc, &doc.text, false);
                }
            }
        }
    }

    Ok(())
}

/// Query and filter entities/clusters
fn cmd_query(args: QueryArgs) -> Result<(), String> {
    // Load input file
    let json_content = if args.input == "-" {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format_error("read stdin", &e.to_string()))?;
        buf
    } else {
        read_input_file(&args.input)?
    };

    // Try to parse as GroundedDocument first, then as cross-doc clusters
    if let Ok(doc) = parse_grounded_document(&json_content) {
        // Query single document - use GroundedDocument helper methods where applicable
        let mut signals: Vec<Signal<Location>> = if let Some(ref filter_type) = args.r#type {
            // Use signals_with_label helper for type filtering (returns Vec<&Signal>, clone to Vec<Signal>)
            doc.signals_with_label(filter_type)
                .into_iter()
                .cloned()
                .collect()
        } else {
            doc.signals().iter().cloned().collect()
        };

        // Apply range filters using spatial index if both offsets provided
        if let (Some(start), Some(end)) = (args.start_offset, args.end_offset) {
            signals = doc
                .query_signals_in_range_indexed(start, end)
                .into_iter()
                .cloned()
                .collect();
        }

        // Apply additional filters
        if let Some(min_conf) = args.min_confidence {
            // Filter by confidence (could use confident_signals, but already have collection)
            signals.retain(|s| s.confidence >= min_conf as f32);
        }

        if let Some(ref entity_text) = args.entity {
            // Filter by entity name
            signals.retain(|s| {
                s.surface()
                    .to_lowercase()
                    .contains(&entity_text.to_lowercase())
            });
        }

        // Apply signal property filters
        if args.negated {
            signals.retain(|s| s.negated);
        }

        if args.quantified {
            signals.retain(|s| s.quantifier.is_some());
        }

        // Apply relationship filters (require checking track/identity membership)
        if args.untracked {
            signals.retain(|s| doc.track_for_signal(s.id).is_none());
        }

        if args.linked {
            signals.retain(|s| doc.identity_for_signal(s.id).is_some());
        }

        if args.unlinked {
            signals.retain(|s| doc.identity_for_signal(s.id).is_none());
        }

        // Output results
        match args.format {
            OutputFormat::Json | OutputFormat::Grounded => {
                let output = serde_json::to_string_pretty(&signals)
                    .map_err(|e| format!("Failed to serialize: {}", e))?;
                if let Some(output_path) = &args.output {
                    fs::write(output_path, output)
                        .map_err(|e| format!("Failed to write output: {}", e))?;
                } else {
                    println!("{}", output);
                }
            }
            _ => {
                println!("Found {} entities:", signals.len());
                for s in &signals {
                    let (start, end) = s.text_offsets().unwrap_or((0, 0));
                    println!(
                        "  [{}:{}] {} ({}) - {:.2}",
                        start,
                        end,
                        s.surface(),
                        s.label(),
                        s.confidence
                    );
                }
            }
        }
    } else if let Ok(_clusters) =
        serde_json::from_str::<Vec<anno::eval::cdcr::CrossDocCluster>>(&json_content)
            .map_err(|e| format_error("parse cross-doc clusters JSON", &e.to_string()))
    {
        // Query cross-doc clusters
        #[cfg(feature = "eval-advanced")]
        {
            let mut filtered: Vec<_> = clusters.iter().collect();

            // Apply filters
            if let Some(ref filter_type) = args.r#type {
                filtered.retain(|c| {
                    c.entity_type
                        .as_ref()
                        .map(|t| t.as_label().eq_ignore_ascii_case(filter_type))
                        .unwrap_or(false)
                });
            }

            if let Some(ref entity_text) = args.entity {
                filtered.retain(|c| {
                    c.canonical_name
                        .to_lowercase()
                        .contains(&entity_text.to_lowercase())
                });
            }

            // Output results
            // Build doc_index for looking up entity text (if documents available)
            // Note: doc_index is built earlier in the function, reuse it if available
            match args.format {
                OutputFormat::Tree => {
                    for cluster in &filtered {
                        println!("Cluster {}: {}", cluster.id, cluster.canonical_name);
                        for (doc_id, entity_idx) in &cluster.mentions {
                            // For tree format, just show doc_id and entity index
                            // Full entity text lookup would require doc_index which may not be in scope
                            println!("  - entity[{}] (doc: {})", entity_idx, doc_id);
                        }
                        println!();
                    }
                }
                OutputFormat::Json | OutputFormat::Grounded => {
                    let output = serde_json::to_string_pretty(&filtered)
                        .map_err(|e| format!("Failed to serialize: {}", e))?;
                    if let Some(output_path) = &args.output {
                        fs::write(output_path, output)
                            .map_err(|e| format!("Failed to write output: {}", e))?;
                    } else {
                        println!("{}", output);
                    }
                }
                _ => {
                    println!("Found {} clusters:", filtered.len());
                    for cluster in &filtered {
                        println!(
                            "  {}: {} mentions across {} documents",
                            cluster.canonical_name,
                            cluster.mentions.len(),
                            cluster.doc_count()
                        );
                    }
                }
            }
        }

        #[cfg(not(feature = "eval-advanced"))]
        {
            return Err("Cross-doc cluster querying requires 'eval-advanced' feature".to_string());
        }
    } else {
        return Err("Failed to parse input as GroundedDocument or cross-doc clusters".to_string());
    }

    Ok(())
}

/// Compare documents, models, or clusters
fn cmd_compare(args: CompareArgs) -> Result<(), String> {
    if args.models {
        // Compare models on same text
        let text = fs::read_to_string(&args.file1)
            .map_err(|e| format!("Failed to read {}: {}", args.file1, e))?;

        if args.model_list.is_empty() {
            return Err("--models requires --model-list with model names".to_string());
        }

        let mut results: Vec<(String, Vec<Entity>)> = Vec::new();

        for model_name in &args.model_list {
            let backend = match model_name.as_str() {
                "pattern" => ModelBackend::Pattern,
                "heuristic" => ModelBackend::Heuristic,
                "stacked" => ModelBackend::Stacked,
                #[cfg(feature = "onnx")]
                "gliner" => ModelBackend::Gliner,
                _ => {
                    return Err(format!("Unknown model: {}", model_name));
                }
            };

            let model = backend.create_model()?;
            let entities = model
                .extract_entities(&text, None)
                .map_err(|e| format!("Model {} failed: {}", model_name, e))?;
            results.push((model_name.clone(), entities));
        }

        // Output comparison
        match args.format.as_str() {
            "table" => {
                println!("\nModel Comparison:");
                println!("{:<15} {:<10}", "Model", "Entities");
                println!("{}", "-".repeat(25));
                for (name, entities) in &results {
                    println!("{:<15} {:<10}", name, entities.len());
                }
            }
            _ => {
                for (name, entities) in &results {
                    println!("\n{} ({} entities):", name, entities.len());
                    for e in entities {
                        println!("  - {} ({})", e.text, e.entity_type.as_label());
                    }
                }
            }
        }
    } else {
        // Compare two documents
        let file2 = args
            .file2
            .ok_or("Second file required for document comparison")?;

        let json1 = fs::read_to_string(&args.file1)
            .map_err(|e| format!("Failed to read {}: {}", args.file1, e))?;
        let json2 =
            fs::read_to_string(&file2).map_err(|e| format!("Failed to read {}: {}", file2, e))?;

        let doc1: GroundedDocument = serde_json::from_str(&json1)
            .map_err(|e| format!("Failed to parse {}: {}", args.file1, e))?;
        let doc2: GroundedDocument = serde_json::from_str(&json2)
            .map_err(|e| format!("Failed to parse {}: {}", file2, e))?;

        let sig1: std::collections::HashSet<String> = doc1
            .signals()
            .iter()
            .map(|s| format!("{}:{}:{}", s.surface(), s.label(), s.confidence))
            .collect();
        let sig2: std::collections::HashSet<String> = doc2
            .signals()
            .iter()
            .map(|s| format!("{}:{}:{}", s.surface(), s.label(), s.confidence))
            .collect();

        let only_in_1: Vec<_> = sig1.difference(&sig2).collect();
        let only_in_2: Vec<_> = sig2.difference(&sig1).collect();
        let in_both: Vec<_> = sig1.intersection(&sig2).collect();

        match args.format.as_str() {
            "diff" => {
                println!("\nComparison: {} vs {}", args.file1, file2);
                println!("\nOnly in {}: {}", args.file1, only_in_1.len());
                for s in &only_in_1 {
                    println!("  + {}", s);
                }
                println!("\nOnly in {}: {}", file2, only_in_2.len());
                for s in &only_in_2 {
                    println!("  - {}", s);
                }
                println!("\nIn both: {}", in_both.len());
            }
            "summary" => {
                println!("\nComparison Summary:");
                println!("  {}: {} entities", args.file1, doc1.signals().len());
                println!("  {}: {} entities", file2, doc2.signals().len());
                println!("  Common: {}", in_both.len());
                println!("  Only in {}: {}", args.file1, only_in_1.len());
                println!("  Only in {}: {}", file2, only_in_2.len());
            }
            _ => {
                println!("Unknown format: {}. Use 'diff' or 'summary'", args.format);
            }
        }
    }

    Ok(())
}

/// Cache management
fn cmd_cache(args: CacheArgs) -> Result<(), String> {
    let cache_dir = get_cache_dir()?;

    match args.action {
        CacheAction::List => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist: {}", cache_dir.display());
                return Ok(());
            }

            let entries = fs::read_dir(&cache_dir)
                .map_err(|e| format!("Failed to read cache directory: {}", e))?;

            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .collect();
            files.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());

            println!("Cached results ({} files):", files.len());
            for entry in files {
                if let Ok(metadata) = entry.metadata() {
                    let size = metadata.len();
                    let modified = if let Ok(modified_time) = metadata.modified() {
                        if let Ok(duration) = modified_time.duration_since(std::time::UNIX_EPOCH) {
                            if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp(
                                duration.as_secs() as i64,
                                0,
                            ) {
                                dt.format("%Y-%m-%d %H:%M:%S").to_string()
                            } else {
                                "unknown".to_string()
                            }
                        } else {
                            "unknown".to_string()
                        }
                    } else {
                        "unknown".to_string()
                    };

                    println!(
                        "  {} ({}) - {}",
                        entry.file_name().to_string_lossy(),
                        format_size(size),
                        modified
                    );
                }
            }
        }
        CacheAction::Clear => {
            if cache_dir.exists() {
                fs::remove_dir_all(&cache_dir)
                    .map_err(|e| format!("Failed to clear cache: {}", e))?;
                println!("{} Cache cleared", color("32", "✓"));
            } else {
                println!("Cache directory does not exist");
            }
        }
        CacheAction::Stats => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
                return Ok(());
            }

            let entries = fs::read_dir(&cache_dir)
                .map_err(|e| format!("Failed to read cache directory: {}", e))?;

            let mut total_size = 0u64;
            let mut count = 0usize;

            for entry in entries {
                if let Ok(entry) = entry {
                    if let Ok(metadata) = entry.metadata() {
                        total_size += metadata.len();
                        count += 1;
                    }
                }
            }

            println!("Cache Statistics:");
            println!("  Files: {}", count);
            println!("  Total size: {}", format_size(total_size));
        }
        CacheAction::Invalidate { model, file } => {
            if !cache_dir.exists() {
                println!("Cache directory does not exist");
                return Ok(());
            }

            let entries = fs::read_dir(&cache_dir)
                .map_err(|e| format!("Failed to read cache directory: {}", e))?;

            let mut removed = 0usize;

            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                    let should_remove = if let Some(ref m) = model {
                        name.starts_with(&format!("{}-", m))
                    } else if let Some(ref f) = file {
                        name.contains(f)
                    } else {
                        false
                    };

                    if should_remove {
                        if fs::remove_file(&path).is_ok() {
                            removed += 1;
                        }
                    }
                }
            }

            println!("{} Removed {} cache entries", color("32", "✓"), removed);
        }
    }

    Ok(())
}

/// Configuration management
fn cmd_config(args: ConfigArgs) -> Result<(), String> {
    let config_dir = get_config_dir()?;

    match args.action {
        ConfigAction::Save {
            name,
            model,
            coref,
            link_kb,
            threshold,
        } => {
            #[cfg(all(feature = "cli", feature = "eval"))]
            {
                use toml::Value;

                let mut config = toml::map::Map::new();

                if let Some(ref m) = model {
                    config.insert("model".to_string(), Value::String(m.clone()));
                }
                if coref {
                    config.insert("coref".to_string(), Value::Boolean(true));
                }
                if link_kb {
                    config.insert("link_kb".to_string(), Value::Boolean(true));
                }
                if let Some(t) = threshold {
                    config.insert("threshold".to_string(), Value::Float(t));
                }

                let toml_string = toml::to_string(&config)
                    .map_err(|e| format!("Failed to serialize config: {}", e))?;

                let config_file = config_dir.join(format!("{}.toml", name));
                fs::write(&config_file, toml_string)
                    .map_err(|e| format!("Failed to write config: {}", e))?;

                println!("{} Saved config: {}", color("32", "✓"), name);
            }

            #[cfg(not(all(feature = "cli", feature = "eval")))]
            {
                return Err("Config management requires 'cli' and 'eval' features".to_string());
            }
        }
        ConfigAction::List => {
            if !config_dir.exists() {
                println!("No configs found");
                return Ok(());
            }

            let entries = fs::read_dir(&config_dir)
                .map_err(|e| format!("Failed to read config directory: {}", e))?;

            let mut configs: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext == "toml")
                        .unwrap_or(false)
                })
                .map(|e| {
                    e.path()
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                })
                .collect();

            configs.sort();

            if configs.is_empty() {
                println!("No configs found");
            } else {
                println!("Saved configs:");
                for config in configs {
                    println!("  {}", config);
                }
            }
        }
        ConfigAction::Show { name } => {
            let config_file = config_dir.join(format!("{}.toml", name));
            if !config_file.exists() {
                return Err(format!("Config '{}' not found", name));
            }

            let content = fs::read_to_string(&config_file)
                .map_err(|e| format!("Failed to read config: {}", e))?;
            println!("Config: {}", name);
            println!("{}", content);
        }
        ConfigAction::Delete { name } => {
            let config_file = config_dir.join(format!("{}.toml", name));
            if !config_file.exists() {
                return Err(format!("Config '{}' not found", name));
            }

            fs::remove_file(&config_file).map_err(|e| format!("Failed to delete config: {}", e))?;
            println!("{} Deleted config: {}", color("32", "✓"), name);
        }
    }

    Ok(())
}

/// Batch processing
fn cmd_batch(args: BatchArgs) -> Result<(), String> {
    // Similar to pipeline but optimized for batch processing
    // Implementation would be similar to pipeline but with better parallelization
    // For now, delegate to pipeline command
    cmd_pipeline(PipelineArgs {
        text: vec![],
        files: vec![],
        dir: args.dir,
        model: args.model,
        coref: args.coref,
        link_kb: args.link_kb,
        cross_doc: false, // Batch doesn't do cross-doc by default
        threshold: 0.6,
        format: args.format,
        output: args.output,
        progress: args.progress,
        quiet: args.quiet,
    })
}

// Helper functions for cache and config
fn get_cache_dir() -> Result<std::path::PathBuf, String> {
    #[cfg(feature = "eval")]
    {
        use dirs::cache_dir;
        if let Some(mut cache) = cache_dir() {
            cache.push("anno");
            fs::create_dir_all(&cache)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
            Ok(cache)
        } else {
            // Fallback to current directory
            Ok(std::path::PathBuf::from(".anno-cache"))
        }
    }
    #[cfg(not(feature = "eval"))]
    {
        Ok(std::path::PathBuf::from(".anno-cache"))
    }
}

fn get_config_dir() -> Result<std::path::PathBuf, String> {
    #[cfg(feature = "eval")]
    {
        use dirs::config_dir;
        if let Some(mut config) = config_dir() {
            config.push("anno");
            fs::create_dir_all(&config)
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
            Ok(config)
        } else {
            // Fallback to current directory
            Ok(std::path::PathBuf::from(".anno-config"))
        }
    }
    #[cfg(not(feature = "eval"))]
    {
        Ok(std::path::PathBuf::from(".anno-config"))
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[cfg(feature = "eval-advanced")]
fn cmd_crossdoc(args: CrossDocArgs) -> Result<(), String> {
    #[cfg(not(feature = "eval"))]
    return Err("Cross-document coreference requires 'eval' feature. Build with: cargo build --features eval".to_string());

    #[cfg(feature = "eval")]
    {
        use std::path::{Path, PathBuf};

        // Create model
        let model = args.model.create_model()?;

        if args.verbose && args.directory.is_some() {
            eprintln!("Scanning directory: {}", args.directory.as_ref().unwrap());
        }

        // Collect text files
        let extensions: Vec<&str> = args.extensions.split(',').map(|s| s.trim()).collect();
        let mut files = Vec::new();

        fn collect_files(
            dir: &Path,
            extensions: &[&str],
            recursive: bool,
            files: &mut Vec<PathBuf>,
        ) -> Result<(), String> {
            let entries = fs::read_dir(dir)
                .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

            for entry in entries {
                let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
                let path = entry.path();

                if path.is_dir() && recursive {
                    collect_files(&path, extensions, recursive, files)?;
                } else if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext_str = ext.to_string_lossy().to_lowercase();
                        if extensions.iter().any(|&e| e == ext_str) {
                            files.push(path);
                        }
                    }
                }
            }
            Ok(())
        }

        // Check if import mode is enabled - use Corpus for better architecture
        let mut doc_paths: HashMap<String, String> = HashMap::new(); // doc_id -> file_path
        let mut use_corpus = false;
        let mut corpus = Corpus::new();
        let mut clusters_from_corpus: Option<Vec<CrossDocCluster>> = None;

        // Helper function to convert Identity to CrossDocCluster with proper mention extraction
        fn identity_to_cluster(identity: &Identity, corpus: &Corpus) -> CrossDocCluster {
            use crate::eval::cdcr::CrossDocCluster;
            use crate::EntityType;

            let mut cluster = CrossDocCluster::new(identity.id, &identity.canonical_name);
            cluster.kb_id = identity.kb_id.clone();
            cluster.confidence = identity.confidence as f64;
            if let Some(ref entity_type) = identity.entity_type {
                cluster.entity_type = Some(EntityType::from_label(entity_type));
            }

            // Extract mentions from TrackRefs if available
            if let Some(IdentitySource::CrossDocCoref { ref track_refs }) = &identity.source {
                let mut doc_set = std::collections::HashSet::new();
                for track_ref in track_refs {
                    // Get the track and extract signal IDs
                    if let Some(doc) = corpus.get_document(&track_ref.doc_id) {
                        if let Some(track) = doc.get_track(track_ref.track_id) {
                            // For each signal in the track, we need to find its entity index
                            // Since we're converting from GroundedDocument, we need to map signals to entities
                            // For now, use signal positions as entity indices (approximation)
                            for (pos, signal_ref) in track.signals.iter().enumerate() {
                                if let Some(signal) = doc.get_signal(signal_ref.signal_id) {
                                    // Find entity index by matching signal text and position
                                    // This is approximate - in a perfect world, we'd track the mapping
                                    let entity_idx = pos; // Use position as approximation
                                    cluster
                                        .mentions
                                        .push((track_ref.doc_id.clone(), entity_idx));
                                    doc_set.insert(track_ref.doc_id.clone());
                                }
                            }
                        }
                    }
                }
                cluster.documents = doc_set.into_iter().collect();
            }

            cluster
        }

        // Legacy helper for CDCR Document conversion (used when not using Corpus)
        fn load_grounded_doc_legacy(
            doc: &GroundedDocument,
            source_path: &str,
        ) -> (Document, usize) {
            // Prefer tracks if available (Level 2), otherwise use signals (Level 1)
            let tracks_vec: Vec<_> = doc.tracks().collect();
            let entities: Vec<_> = if !tracks_vec.is_empty() {
                // Use tracks: each track represents a within-doc coreference chain
                // Extract canonical mention from each track
                tracks_vec
                    .iter()
                    .filter_map(|track| {
                        // Get the first signal in the track as the canonical mention
                        let signal_ids: Vec<_> =
                            track.signals.iter().map(|sr| sr.signal_id).collect();
                        signal_ids
                            .first()
                            .and_then(|signal_id| {
                                doc.get_signal(*signal_id).map(|signal| {
                                    let (start, end) = signal.text_offsets().unwrap_or((0, 0));
                                    use anno::EntityType;
                                    Entity::new(
                                        signal.surface(),
                                        EntityType::from_label(signal.label()),
                                        start,
                                        end,
                                        signal.confidence as f64,
                                    )
                                })
                            })
                            .or_else(|| {
                                // Fallback: create entity from track canonical
                                use anno::EntityType;
                                Some(Entity::new(
                                    &track.canonical_surface,
                                    track
                                        .entity_type
                                        .as_ref()
                                        .map(|t| EntityType::from_label(t))
                                        .unwrap_or(EntityType::Other("UNKNOWN".into())),
                                    0,
                                    0,
                                    track.cluster_confidence as f64,
                                ))
                            })
                    })
                    .collect()
            } else {
                // Fallback to signals
                doc.signals()
                    .iter()
                    .map(|s| {
                        let (start, end) = s.text_offsets().unwrap_or((0, 0));
                        use anno::EntityType;
                        Entity::new(
                            s.surface(),
                            EntityType::from_label(s.label()),
                            start,
                            end,
                            s.confidence as f64,
                        )
                    })
                    .collect()
            };

            let entity_count = entities.len();
            let cdcr_doc = Document::new(&doc.id, &doc.text).with_entities(entities);
            (cdcr_doc, entity_count)
        }

        if !args.import.is_empty() || args.stdin {
            // Import mode: use Corpus for proper inter-doc coref with GroundedDocuments
            use_corpus = true;
            let mut import_files = Vec::new();

            if args.stdin {
                // Read from stdin (JSONL format)
                if args.verbose {
                    eprintln!("Reading GroundedDocuments from stdin (JSONL format)...");
                }
                let stdin = io::stdin();
                let reader = stdin.lock();
                for (line_num, line) in reader.lines().enumerate() {
                    let line = line.map_err(|e| {
                        format!("Failed to read stdin line {}: {}", line_num + 1, e)
                    })?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let mut doc: GroundedDocument = serde_json::from_str(&line).map_err(|e| {
                        format!("Failed to parse stdin line {}: {}", line_num + 1, e)
                    })?;
                    // Ensure tracks exist - if not, create them from signals for better clustering
                    if doc.tracks.is_empty() && !doc.signals.is_empty() {
                        // Could run within-doc coref here, but for now just use signals
                        // The Corpus will cluster based on signals if no tracks
                    }
                    corpus.add_document(doc);
                    doc_paths.insert(
                        corpus
                            .get_document(&format!("stdin:{}", line_num + 1))
                            .map(|d| d.id.clone())
                            .unwrap_or_else(|| format!("stdin:{}", line_num + 1)),
                        format!("stdin:{}", line_num + 1),
                    );
                    if args.verbose {
                        let stats = corpus
                            .get_document(&format!("stdin:{}", line_num + 1))
                            .map(|d| d.stats())
                            .unwrap_or_default();
                        eprintln!(
                            "  Imported {} signals, {} tracks from stdin line {}",
                            stats.signal_count,
                            stats.track_count,
                            line_num + 1
                        );
                    }
                }
            } else {
                // Collect files from import paths (support glob patterns)
                for import_pattern in &args.import {
                    if import_pattern == "-" {
                        // Special case: read from stdin
                        let stdin = io::stdin();
                        let reader = stdin.lock();
                        for (line_num, line) in reader.lines().enumerate() {
                            let line = line.map_err(|e| {
                                format!("Failed to read stdin line {}: {}", line_num + 1, e)
                            })?;
                            if line.trim().is_empty() {
                                continue;
                            }
                            let mut doc: GroundedDocument =
                                serde_json::from_str(&line).map_err(|e| {
                                    format!("Failed to parse stdin line {}: {}", line_num + 1, e)
                                })?;
                            let doc_id = doc.id.clone();
                            corpus.add_document(doc);
                            doc_paths.insert(doc_id.clone(), format!("stdin:{}", line_num + 1));
                            if args.verbose {
                                if let Some(d) = corpus.get_document(&doc_id) {
                                    let stats = d.stats();
                                    eprintln!(
                                        "  Imported {} signals, {} tracks from stdin line {}",
                                        stats.signal_count,
                                        stats.track_count,
                                        line_num + 1
                                    );
                                }
                            }
                        }
                    } else if import_pattern.contains('*')
                        || import_pattern.contains('?')
                        || import_pattern.contains('[')
                    {
                        // Glob pattern
                        if args.verbose {
                            eprintln!("Expanding glob pattern: {}", import_pattern);
                        }
                        let matches = glob(import_pattern).map_err(|e| {
                            format!("Invalid glob pattern '{}': {}", import_pattern, e)
                        })?;
                        for entry in matches {
                            match entry {
                                Ok(path) => {
                                    if path.is_file() {
                                        import_files.push(path);
                                    }
                                }
                                Err(e) => {
                                    if args.verbose {
                                        eprintln!("  Warning: glob match error: {}", e);
                                    }
                                }
                            }
                        }
                    } else {
                        // Regular file path
                        let path = Path::new(import_pattern);
                        if path.exists() && path.is_file() {
                            import_files.push(path.to_path_buf());
                        } else {
                            return Err(format!("Import file not found: {}", import_pattern));
                        }
                    }
                }

                // Load all collected files
                if args.verbose && !import_files.is_empty() {
                    eprintln!(
                        "Importing {} GroundedDocument file(s)...",
                        import_files.len()
                    );
                }

                for (idx, file_path) in import_files.iter().enumerate() {
                    if args.verbose {
                        eprint!(
                            "\r  Loading {}/{}: {}...",
                            idx + 1,
                            import_files.len(),
                            file_path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("?")
                        );
                        use std::io::Write;
                        io::stderr().flush().ok();
                    }

                    let json_content = fs::read_to_string(file_path).map_err(|e| {
                        format!(
                            "Failed to read import file '{}': {}",
                            file_path.display(),
                            e
                        )
                    })?;

                    let doc: GroundedDocument =
                        serde_json::from_str(&json_content).map_err(|e| {
                            format!(
                                "Failed to parse GroundedDocument JSON from '{}': {}",
                                file_path.display(),
                                e
                            )
                        })?;

                    let (cdcr_doc, entity_count) =
                        load_grounded_doc(&doc, &file_path.display().to_string());
                    documents.push(cdcr_doc);
                    doc_paths.insert(doc.id.clone(), file_path.display().to_string());

                    if args.verbose {
                        eprintln!(
                            "\r  Loaded {} entities from {}",
                            entity_count,
                            file_path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("?")
                        );
                    }
                }
            }

            let doc_count = corpus.documents().count();
            if doc_count == 0 {
                return Err(
                    "No GroundedDocuments imported. Check import paths or stdin input.".to_string(),
                );
            }

            if args.verbose {
                let total_signals: usize = corpus.documents().map(|d| d.stats().signal_count).sum();
                let total_tracks: usize = corpus.documents().map(|d| d.stats().track_count).sum();
                eprintln!(
                    "Imported {} documents with {} signals, {} tracks",
                    doc_count, total_signals, total_tracks
                );
            }

            // Use Corpus for inter-doc coref resolution (much cleaner than CDCR conversion)
            if args.verbose {
                eprintln!(
                    "Resolving inter-document coreference (threshold: {}, require_type_match: {})...",
                    args.threshold,
                    args.require_type_match
                );
            }

            let identity_ids =
                corpus.resolve_inter_doc_coref(args.threshold as f32, args.require_type_match);

            if args.verbose {
                eprintln!(
                    "Created {} identities from inter-doc coref",
                    identity_ids.len()
                );
            }

            // Convert identities to CrossDocCluster for output compatibility
            // Extract mentions from TrackRefs in Identity
            let mut clusters: Vec<CrossDocCluster> = Vec::new();
            for &id in &identity_ids {
                if let Some(identity) = corpus.identities.get(&id) {
                    let mut cluster = identity.to_cross_doc_cluster();

                    // Populate mentions from TrackRefs
                    if let Some(IdentitySource::CrossDocCoref { ref track_refs }) = &identity.source
                    {
                        let mut doc_set = std::collections::HashSet::new();
                        for track_ref in track_refs {
                            if let Some(doc) = corpus.get_document(&track_ref.doc_id) {
                                if let Some(track) = doc.get_track(track_ref.track_id) {
                                    // Add each signal in the track as a mention
                                    // Use signal position as entity index (approximation)
                                    for (pos, signal_ref) in track.signals.iter().enumerate() {
                                        cluster.mentions.push((track_ref.doc_id.clone(), pos));
                                        doc_set.insert(track_ref.doc_id.clone());
                                    }
                                }
                            }
                        }
                        cluster.documents = doc_set.into_iter().collect();
                    }

                    // Apply filters
                    if cluster.len() >= args.min_cluster_size
                        && (!args.cross_doc_only || cluster.doc_count() > 1)
                        && (args.entity_types.is_empty()
                            || cluster
                                .entity_type
                                .as_ref()
                                .map(|et| {
                                    let type_label = et.as_label().to_uppercase();
                                    args.entity_types
                                        .iter()
                                        .any(|t| t.to_uppercase() == type_label)
                                })
                                .unwrap_or(false))
                    {
                        clusters.push(cluster);
                    }
                }
            }

            // Sort by importance
            clusters.sort_by(|a, b| {
                b.doc_count()
                    .cmp(&a.doc_count())
                    .then_with(|| b.len().cmp(&a.len()))
                    .then_with(|| b.canonical_name.cmp(&a.canonical_name))
            });

            // Limit output
            if args.max_clusters > 0 {
                clusters.truncate(args.max_clusters);
            }

            clusters_from_corpus = Some(clusters);
        } else {
            // Normal mode: extract from text files, use CDCRResolver (legacy path)
            let mut documents = Vec::new();
            // Normal mode: extract entities from text files
            // Directory is required in normal mode
            let dir = if let Some(ref dir_str) = args.directory {
                Path::new(dir_str)
            } else {
                return Err("Directory is required when --import is not used. Use: anno cross-doc <DIR> or anno cross-doc --import <FILE>".to_string());
            };

            collect_files(dir, &extensions, args.recursive, &mut files)?;

            if files.is_empty() {
                return Err(format!(
                    "No files found with extensions: {}",
                    args.extensions
                ));
            }

            if args.verbose {
                eprintln!("Found {} files", files.len());
                eprintln!("Extracting entities...");
            }

            // NOTE: Currently operates on raw entities (Level 1: Signal)
            // With --import, can use Level 2 (Tracks) and Level 3 (Identities) from pre-processed docs
            let total_files = files.len();
            for (idx, file_path) in files.iter().enumerate() {
                if args.verbose {
                    eprint!(
                        "\r  Processing {}/{}: {}...",
                        idx + 1,
                        total_files,
                        file_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("?")
                    );
                    use std::io::Write;
                    io::stderr().flush().ok();
                }

                let text = fs::read_to_string(file_path)
                    .map_err(|e| format!("Failed to read {}: {}", file_path.display(), e))?;

                let doc_id = file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("doc{}", idx));

                // Store file path for later display
                doc_paths.insert(doc_id.clone(), file_path.display().to_string());

                let entities = model
                    .extract_entities(&text, None)
                    .map_err(|e| format!("Failed to extract entities from {}: {}", doc_id, e))?;

                // Build GroundedDocument and optionally run coreference for better clustering
                // This enables using tracks (Level 2) instead of just raw signals (Level 1)
                let mut grounded_doc = GroundedDocument::new(&doc_id, &text);
                let mut signal_ids: Vec<u64> = Vec::new();

                for e in &entities {
                    let signal = Signal::new(
                        0,
                        Location::text(e.start, e.end),
                        &e.text,
                        e.entity_type.as_label(),
                        e.confidence as f32,
                    );
                    let id = grounded_doc.add_signal(signal);
                    signal_ids.push(id);
                }

                // If we have tracks from import, use them; otherwise use signals
                // For now, convert to CDCR Document using signals (can be enhanced to use tracks)
                documents.push(Document::new(&doc_id, &text).with_entities(entities));
            }

            if args.verbose {
                eprintln!("\r  Processed {} files successfully", total_files);
            }
        }

        if args.verbose {
            let total_entities: usize = documents.iter().map(|d| d.entities.len()).sum();
            eprintln!(
                "Clustering {} entities across {} documents...",
                total_entities,
                documents.len()
            );
        }

        // Configure and run cross-doc coref using CDCRResolver (for raw text files)
        let config = CDCRConfig {
            min_similarity: args.threshold,
            require_type_match: args.require_type_match,
            use_lsh: documents.len() > 100, // Use LSH for large document sets
            ..Default::default()
        };

        let resolver = CDCRResolver::with_config(config);
        let mut clusters = resolver.resolve(&documents);

        // Filter clusters
        let mut filtered_clusters: Vec<_> = clusters
            .into_iter()
            .filter(|c| {
                // Minimum size filter
                if c.len() < args.min_cluster_size {
                    return false;
                }
                // Cross-doc only filter
                if args.cross_doc_only && c.doc_count() <= 1 {
                    return false;
                }
                // Entity type filter
                if !args.entity_types.is_empty() {
                    if let Some(ref entity_type) = c.entity_type {
                        let type_label = entity_type.as_label().to_uppercase();
                        if !args
                            .entity_types
                            .iter()
                            .any(|t| t.to_uppercase() == type_label)
                        {
                            return false;
                        }
                    } else {
                        return false; // Skip clusters without type if filtering by type
                    }
                }
                true
            })
            .collect();

        // Sort by importance
        filtered_clusters.sort_by(|a, b| {
            b.doc_count()
                .cmp(&a.doc_count())
                .then_with(|| b.len().cmp(&a.len()))
                .then_with(|| b.canonical_name.cmp(&a.canonical_name))
        });

        // Limit output
        let clusters: Vec<_> = if args.max_clusters > 0 {
            filtered_clusters
                .into_iter()
                .take(args.max_clusters)
                .collect()
        } else {
            filtered_clusters
        };

        // Use clusters from Corpus if available, otherwise use CDCR clusters
        let final_clusters: Vec<CrossDocCluster> =
            if let Some(corpus_clusters) = clusters_from_corpus {
                corpus_clusters
            } else {
                filtered_clusters
            };

        // Prepare output
        let output_text = match args.format {
            OutputFormat::Json => {
                // Enhanced JSON with metadata
                let mut output = serde_json::Map::new();
                let doc_count = if use_corpus {
                    corpus.documents().count()
                } else {
                    documents.len()
                };
                let total_entities = if use_corpus {
                    corpus
                        .documents()
                        .map(|d| d.stats().signal_count)
                        .sum::<usize>()
                } else {
                    documents.iter().map(|d| d.entities.len()).sum::<usize>()
                };
                output.insert("metadata".to_string(), serde_json::json!({
                "documents_processed": doc_count,
                "total_entities": total_entities,
                "clusters_found": final_clusters.len(),
                "cross_document_clusters": final_clusters.iter().filter(|c| c.doc_count() > 1).count(),
                "threshold": args.threshold,
                "require_type_match": args.require_type_match,
                "filters": {
                    "min_cluster_size": args.min_cluster_size,
                    "cross_doc_only": args.cross_doc_only,
                    "entity_types": args.entity_types,
                    "max_clusters": args.max_clusters,
                }
            }));
                output.insert(
                    "clusters".to_string(),
                    serde_json::to_value(&final_clusters)
                        .map_err(|e| format!("Failed to serialize clusters: {}", e))?,
                );
                serde_json::to_string_pretty(&output)
                    .map_err(|e| format!("Failed to serialize output: {}", e))?
            }
            OutputFormat::Jsonl => {
                let mut lines = Vec::new();
                for cluster in &final_clusters {
                    let json = serde_json::to_string(cluster)
                        .map_err(|e| format!("Failed to serialize cluster: {}", e))?;
                    lines.push(json);
                }
                lines.join("\n")
            }
            OutputFormat::Tree => {
                // Build document index for O(1) lookups (only needed for CDCR path)
                let doc_index: HashMap<&str, &Document> = if !use_corpus {
                    documents.iter().map(|d| (d.id.as_str(), d)).collect()
                } else {
                    HashMap::new() // Not needed for Corpus path
                };

                let mut output = String::new();
                // Sort clusters by importance (doc count, then mention count)
                let mut sorted_clusters: Vec<_> = final_clusters.iter().collect();
                sorted_clusters.sort_by(|a, b| {
                    b.doc_count()
                        .cmp(&a.doc_count())
                        .then_with(|| b.len().cmp(&a.len()))
                        .then_with(|| b.canonical_name.cmp(&a.canonical_name))
                });

                // Simplified header - less visual noise
                output.push_str(&format!(
                    "{}\n",
                    color("1;36", "Cross-Document Entity Clusters")
                ));
                output.push_str("\n");

                // Summary header
                let doc_count = if use_corpus {
                    corpus.documents().count()
                } else {
                    documents.len()
                };
                let total_entities = if use_corpus {
                    corpus
                        .documents()
                        .map(|d| d.stats().signal_count)
                        .sum::<usize>()
                } else {
                    documents.iter().map(|d| d.entities.len()).sum::<usize>()
                };
                let cross_doc_clusters =
                    final_clusters.iter().filter(|c| c.doc_count() > 1).count();
                let singleton_clusters = final_clusters.len() - cross_doc_clusters;

                output.push_str(&format!("{}\n", color("1;33", "Summary")));
                output.push_str(&format!("  Documents: {}\n", doc_count));
                output.push_str(&format!("  Entities: {}\n", total_entities));
                output.push_str(&format!(
                    "  Clusters: {} ({} cross-doc, {} singleton)\n",
                    final_clusters.len(),
                    color("32", &cross_doc_clusters.to_string()),
                    singleton_clusters
                ));
                if !args.entity_types.is_empty() {
                    output.push_str(&format!(
                        "  Filtered by: {}\n",
                        args.entity_types.join(", ")
                    ));
                }
                output.push_str("\n");

                // Entity type breakdown
                let mut type_counts: HashMap<String, usize> = HashMap::new();
                for cluster in &final_clusters {
                    if let Some(ref entity_type) = cluster.entity_type {
                        *type_counts
                            .entry(entity_type.as_label().to_string())
                            .or_insert(0) += 1;
                    }
                }
                if !type_counts.is_empty() {
                    output.push_str(&format!("{}\n", color("1;33", "Entity Types")));
                    let mut type_vec: Vec<_> = type_counts.iter().collect();
                    type_vec.sort_by(|a, b| b.1.cmp(a.1));
                    for (etype, count) in type_vec {
                        output.push_str(&format!("  {}: {}\n", etype, count));
                    }
                    output.push_str("\n");
                }

                output.push_str(&format!("{}\n", color("1;36", "Clusters")));
                output.push_str("\n");

                // Determine display limit
                let display_limit = if args.max_clusters > 0 {
                    args.max_clusters
                } else if !args.verbose {
                    50 // Default limit for non-verbose
                } else {
                    sorted_clusters.len() // No limit in verbose mode
                };

                for cluster in sorted_clusters.iter().take(display_limit) {
                    let is_cross_doc = cluster.doc_count() > 1;
                    let prefix = if is_cross_doc {
                        color("32", "●")
                    } else {
                        color("90", "○")
                    };

                    // Cluster header: prefix + name + type
                    let mut header = format!("{} {}", prefix, color("1", &cluster.canonical_name));
                    if let Some(ref entity_type) = cluster.entity_type {
                        header.push_str(&format!(" ({})", entity_type.as_label()));
                    }
                    if is_cross_doc {
                        header.push_str(&format!(" {}", color("32", "[cross-doc]")));
                    }
                    output.push_str(&format!("{}\n", header));

                    // Metadata line
                    let mut meta_parts = Vec::new();
                    meta_parts.push(format!("{} mentions", cluster.len()));
                    meta_parts.push(format!(
                        "{} doc{}",
                        cluster.doc_count(),
                        if cluster.doc_count() == 1 { "" } else { "s" }
                    ));
                    if cluster.confidence < 1.0 {
                        meta_parts.push(format!("conf: {:.2}", cluster.confidence));
                    }
                    output.push_str(&format!("  {}\n", meta_parts.join(" • ")));

                    if let Some(ref kb_id) = cluster.kb_id {
                        output.push_str(&format!("  KB: {}\n", color("36", kb_id)));
                    }

                    // Show documents with paths (truncate if too many)
                    if !cluster.documents.is_empty() {
                        let max_docs_to_show = if args.verbose { 20 } else { 5 };
                        let doc_list: Vec<String> = cluster
                            .documents
                            .iter()
                            .take(max_docs_to_show)
                            .map(|doc_id| {
                                let path = doc_paths
                                    .get(doc_id)
                                    .map(|p| format!("{} ({})", doc_id, p))
                                    .unwrap_or_else(|| doc_id.clone());
                                color("36", &path)
                            })
                            .collect();
                        let doc_count = cluster.documents.len();
                        if doc_count > max_docs_to_show {
                            output.push_str(&format!(
                                "  Docs: {} (and {} more)\n",
                                doc_list.join(", "),
                                doc_count - max_docs_to_show
                            ));
                        } else {
                            output.push_str(&format!("  Docs: {}\n", doc_list.join(", ")));
                        }
                    }

                    // Show mentions - always show sample, verbose adds context
                    if !cluster.mentions.is_empty() {
                        let sample_size = if args.verbose {
                            cluster.mentions.len()
                        } else {
                            cluster.mentions.len().min(3)
                        };

                        for (doc_id, entity_idx) in cluster.mentions.iter().take(sample_size) {
                            if let Some(doc) = doc_index.get(doc_id.as_str()) {
                                if let Some(entity) = doc.entities.get(*entity_idx) {
                                    if args.verbose {
                                        // Extract context safely (50 chars before/after)
                                        let context_window = 50;

                                        // Find character boundaries for entity
                                        let entity_start_char = doc
                                            .text
                                            .char_indices()
                                            .position(|(byte_idx, _)| byte_idx >= entity.start)
                                            .unwrap_or(0);
                                        let entity_end_char = doc
                                            .text
                                            .char_indices()
                                            .position(|(byte_idx, _)| byte_idx >= entity.end)
                                            .unwrap_or(doc.text.chars().count());

                                        // Calculate context character range
                                        let context_start_char =
                                            entity_start_char.saturating_sub(context_window);
                                        let context_end_char = (entity_end_char + context_window)
                                            .min(doc.text.chars().count());

                                        // Convert back to byte positions
                                        let safe_start = doc
                                            .text
                                            .char_indices()
                                            .nth(context_start_char)
                                            .map(|(byte_idx, _)| byte_idx)
                                            .unwrap_or(0);
                                        let safe_end = doc
                                            .text
                                            .char_indices()
                                            .nth(context_end_char)
                                            .map(|(byte_idx, _)| byte_idx)
                                            .unwrap_or(doc.text.len());

                                        let context = &doc.text[safe_start..safe_end];

                                        // Ensure entity positions are at character boundaries
                                        let entity_start_byte = doc
                                            .text
                                            .char_indices()
                                            .find(|&(byte_idx, _)| byte_idx >= entity.start)
                                            .map(|(byte_idx, _)| byte_idx)
                                            .unwrap_or(entity.start);
                                        let entity_end_byte = doc
                                            .text
                                            .char_indices()
                                            .find(|&(byte_idx, _)| byte_idx >= entity.end)
                                            .map(|(byte_idx, _)| byte_idx)
                                            .unwrap_or(entity.end);
                                        let entity_text =
                                            &doc.text[entity_start_byte..entity_end_byte];

                                        // Calculate offsets within the safe context window (in bytes)
                                        let before_len = entity.start.saturating_sub(safe_start);
                                        let after_start =
                                            (entity.end - safe_start).min(context.len());

                                        // Find character boundaries for safe slicing
                                        let before = if safe_start < entity.start && before_len > 0
                                        {
                                            // Find the character boundary closest to before_len
                                            let target_byte = before_len.min(context.len());
                                            let char_boundary = context
                                                .char_indices()
                                                .find(|&(byte_idx, _)| byte_idx >= target_byte)
                                                .map(|(byte_idx, _)| byte_idx)
                                                .unwrap_or(context.len());
                                            &context[..char_boundary.min(context.len())]
                                        } else {
                                            ""
                                        };

                                        let after = if after_start < context.len() {
                                            // Find the character boundary at after_start
                                            let char_boundary = context
                                                .char_indices()
                                                .find(|&(byte_idx, _)| byte_idx >= after_start)
                                                .map(|(byte_idx, _)| byte_idx)
                                                .unwrap_or(context.len());
                                            &context[char_boundary.min(context.len())..]
                                        } else {
                                            ""
                                        };

                                        let before_marker =
                                            if safe_start < entity.start { "..." } else { "" };
                                        let after_marker =
                                            if entity.end < safe_end { "..." } else { "" };

                                        output.push_str(&format!(
                                            "    {} {}: {}{}[{}]{}{}\n",
                                            color("90", "•"),
                                            color("36", doc_id),
                                            before_marker,
                                            before,
                                            color("1;32", entity_text),
                                            after,
                                            after_marker
                                        ));
                                    } else {
                                        // Non-verbose: just show entity text
                                        output.push_str(&format!(
                                            "    {} {}: \"{}\"\n",
                                            color("90", "•"),
                                            color("36", doc_id),
                                            entity.text
                                        ));
                                    }
                                }
                            }
                        }

                        if cluster.mentions.len() > sample_size {
                            output.push_str(&format!(
                                "    {} ... and {} more\n",
                                color("90", "•"),
                                cluster.mentions.len() - sample_size
                            ));
                        }
                    }

                    output.push_str("\n");
                }

                // Show limit message if applicable
                if sorted_clusters.len() > display_limit {
                    let more_count = sorted_clusters.len() - display_limit;
                    let message = format!(
                        "... {} more cluster{} (use --max-clusters {} or --verbose to see all)",
                        more_count,
                        if more_count == 1 { "" } else { "s" },
                        sorted_clusters.len()
                    );
                    output.push_str(&format!("{}\n", color("90", &message)));
                }
                output
            }
            OutputFormat::Summary => {
                let total_entities: usize = documents.iter().map(|d| d.entities.len()).sum();
                let cross_doc_clusters = clusters.iter().filter(|c| c.doc_count() > 1).count();
                let singleton_clusters = clusters.len() - cross_doc_clusters;
                let avg_cluster_size = if clusters.is_empty() {
                    0.0
                } else {
                    clusters.iter().map(|c| c.len()).sum::<usize>() as f64 / clusters.len() as f64
                };
                let max_cluster_size = clusters.iter().map(|c| c.len()).max().unwrap_or(0);
                let max_doc_count = clusters.iter().map(|c| c.doc_count()).max().unwrap_or(0);

                // Entity type distribution
                use std::collections::HashMap;
                let mut type_counts: HashMap<String, usize> = HashMap::new();
                for cluster in &clusters {
                    if let Some(ref entity_type) = cluster.entity_type {
                        *type_counts
                            .entry(entity_type.as_label().to_string())
                            .or_insert(0) += 1;
                    }
                }

                let mut output = String::new();
                output.push_str(&format!(
                    "{}\n",
                    color(
                        "1;36",
                        "═══════════════════════════════════════════════════════════"
                    )
                ));
                output.push_str(&format!(
                    "{}\n",
                    color("1;36", "  Cross-Document Coreference Summary")
                ));
                output.push_str(&format!(
                    "{}\n",
                    color(
                        "1;36",
                        "═══════════════════════════════════════════════════════════"
                    )
                ));
                output.push_str("\n");
                output.push_str(&format!("{}\n", color("1;33", "Document Statistics:")));
                output.push_str(&format!("  Documents processed: {}\n", documents.len()));
                output.push_str(&format!("  Total entities extracted: {}\n", total_entities));
                output.push_str(&format!(
                    "  Average entities per document: {:.1}\n",
                    if documents.is_empty() {
                        0.0
                    } else {
                        total_entities as f64 / documents.len() as f64
                    }
                ));
                output.push_str("\n");
                output.push_str(&format!("{}\n", color("1;33", "Cluster Statistics:")));
                output.push_str(&format!("  Total clusters: {}\n", clusters.len()));
                output.push_str(&format!(
                    "  Cross-document clusters: {} ({:.1}%)\n",
                    cross_doc_clusters,
                    if clusters.is_empty() {
                        0.0
                    } else {
                        cross_doc_clusters as f64 / clusters.len() as f64 * 100.0
                    }
                ));
                output.push_str(&format!("  Singleton clusters: {}\n", singleton_clusters));
                output.push_str(&format!(
                    "  Average cluster size: {:.2} mentions\n",
                    avg_cluster_size
                ));
                output.push_str(&format!(
                    "  Largest cluster: {} mentions\n",
                    max_cluster_size
                ));
                output.push_str(&format!(
                    "  Most documents per cluster: {}\n",
                    max_doc_count
                ));
                output.push_str("\n");
                if !type_counts.is_empty() {
                    output.push_str(&format!("{}\n", color("1;33", "Entity Type Distribution:")));
                    let mut type_vec: Vec<_> = type_counts.iter().collect();
                    type_vec.sort_by(|a, b| b.1.cmp(a.1));
                    for (etype, count) in type_vec {
                        let percentage = if clusters.is_empty() {
                            0.0
                        } else {
                            *count as f64 / clusters.len() as f64 * 100.0
                        };
                        output.push_str(&format!("  {}: {} ({:.1}%)\n", etype, count, percentage));
                    }
                }
                output
            }
            OutputFormat::Human
            | OutputFormat::Tsv
            | OutputFormat::Inline
            | OutputFormat::Grounded
            | OutputFormat::Html => {
                return Err(format!("Format '{}' not supported for cross-doc command. Use: json, jsonl, tree, or summary.", 
                match args.format {
                    OutputFormat::Human => "human",
                    OutputFormat::Tsv => "tsv",
                    OutputFormat::Inline => "inline",
                    OutputFormat::Grounded => "grounded",
                    OutputFormat::Html => "html",
                    _ => unreachable!(),
                }
            ));
            }
        };

        // Write output to file or stdout
        if let Some(output_path) = args.output {
            fs::write(&output_path, &output_text)
                .map_err(|e| format!("Failed to write output to {}: {}", output_path, e))?;
            if args.verbose {
                eprintln!("Output written to: {}", output_path);
            }
        } else {
            print!("{}", output_text);
        }

        Ok(())
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn get_input_text(
    text: &Option<String>,
    file: Option<&str>,
    positional: &[String],
) -> Result<String, String> {
    // Check explicit text arg
    if let Some(t) = text {
        return Ok(t.clone());
    }

    // Check file arg
    if let Some(f) = file {
        return read_input_file(f);
    }

    // Check positional args
    if !positional.is_empty() {
        return Ok(positional.join(" "));
    }

    // Try stdin
    if !io::stdin().is_terminal() {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format_error("read stdin", &e.to_string()))?;
        if !buf.is_empty() {
            return Ok(buf);
        }
    }

    Err("No input text provided. Use -t 'text' or -f file or pipe via stdin".to_string())
}

/// Read a file with consistent error handling
fn read_input_file(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format_error("read file", &format!("{}: {}", path, e)))
}

/// Parse a GroundedDocument from JSON with consistent error handling
fn parse_grounded_document(json: &str) -> Result<GroundedDocument, String> {
    serde_json::from_str(json)
        .map_err(|e| format_error("parse GroundedDocument JSON", &e.to_string()))
}

/// Write output to file or stdout with consistent error handling
fn write_output(content: &str, path: Option<&str>) -> Result<(), String> {
    if let Some(output_path) = path {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(output_path).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| {
                    format_error("create directory", &format!("{}: {}", parent.display(), e))
                })?;
            }
        }
        fs::write(output_path, content)
            .map_err(|e| format_error("write output", &format!("{}: {}", output_path, e)))?;
    } else {
        print!("{}", content);
    }
    Ok(())
}

/// Format error message consistently
fn format_error(operation: &str, details: &str) -> String {
    format!("Failed to {}: {}", operation, details)
}

/// Log info message (respects quiet flag)
fn log_info(msg: &str, quiet: bool) {
    if !quiet {
        eprintln!("{}", msg);
    }
}

/// Log verbose message (only if verbose enabled)
fn log_verbose(msg: &str, verbose: bool) {
    if verbose {
        eprintln!("{}", msg);
    }
}

/// Log success message with color (respects quiet flag)
fn log_success(msg: &str, quiet: bool) {
    if !quiet {
        eprintln!("{} {}", color("32", "✓"), msg);
    }
}

#[derive(Debug, Clone)]
struct GoldSpec {
    text: String,
    label: String,
    start: usize,
    end: usize,
}

/// Parse gold spec with format: "text:label:start:end"
/// Uses rsplit to handle text containing colons (like URLs)
fn parse_gold_spec(s: &str) -> Option<GoldSpec> {
    // Split from right to handle colons in text
    let parts: Vec<&str> = s.rsplitn(4, ':').collect();
    if parts.len() < 4 {
        return None;
    }

    let end: usize = parts[0].parse().ok()?;
    let start: usize = parts[1].parse().ok()?;
    let label = parts[2].to_string();
    let text = parts[3].to_string();

    Some(GoldSpec {
        text,
        label,
        start,
        end,
    })
}

fn load_gold_from_file(path: &str) -> Result<Vec<GoldSpec>, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let mut gold = Vec::new();
    let mut warnings = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let entry: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("Invalid JSON in gold file at line {}: {}", line_num + 1, e))?;

        if let Some(entities) = entry["entities"].as_array() {
            for (i, ent) in entities.iter().enumerate() {
                // Validate required fields are present
                let start = match ent["start"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        warnings.push(format!(
                            "{}:{}: entity[{}] missing 'start' field, defaulting to 0",
                            path,
                            line_num + 1,
                            i
                        ));
                        0
                    }
                };
                let end = match ent["end"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        warnings.push(format!(
                            "{}:{}: entity[{}] missing 'end' field, defaulting to 0",
                            path,
                            line_num + 1,
                            i
                        ));
                        0
                    }
                };

                gold.push(GoldSpec {
                    text: ent["text"].as_str().unwrap_or("").to_string(),
                    label: ent["type"]
                        .as_str()
                        .or(ent["label"].as_str())
                        .unwrap_or("UNK")
                        .to_string(),
                    start,
                    end,
                });
            }
        }
    }

    // Report warnings to stderr
    for warning in &warnings {
        eprintln!("{} {}", color("33", "warning:"), warning);
    }

    Ok(gold)
}

/// Detect if entity at position is negated
fn is_negated(text: &str, entity_start: usize) -> bool {
    let prefix: String = text.chars().take(entity_start).collect();
    let words: Vec<&str> = prefix.split_whitespace().collect();
    let last_words: Vec<&str> = words.iter().rev().take(3).copied().collect();

    const NEGATION_WORDS: &[&str] = &[
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "without",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "couldn't",
        "shouldn't",
    ];

    for word in &last_words {
        if NEGATION_WORDS.contains(&word.to_lowercase().as_str()) {
            return true;
        }
    }

    false
}

/// Detect quantifier before entity
fn detect_quantifier(text: &str, entity_start: usize) -> Option<Quantifier> {
    let prefix: String = text.chars().take(entity_start).collect();
    let words: Vec<&str> = prefix.split_whitespace().collect();

    words
        .last()
        .and_then(|word| match word.to_lowercase().as_str() {
            "every" | "all" | "each" | "any" => Some(Quantifier::Universal),
            "some" | "certain" | "a" | "an" => Some(Quantifier::Existential),
            "no" | "none" => Some(Quantifier::None),
            "the" | "this" | "that" | "these" | "those" => Some(Quantifier::Definite),
            _ => None,
        })
}

/// Flexible type matching for evaluation (handles PER/PERSON, LOC/LOCATION, etc.)
fn types_match_flexible(pred: &str, gold: &str) -> bool {
    let pred = pred.to_uppercase();
    let gold = gold.to_uppercase();

    if pred == gold {
        return true;
    }

    // Allow common mappings
    match (pred.as_str(), gold.as_str()) {
        // Person
        ("PERSON", "PER") | ("PER", "PERSON") => true,
        // Location
        ("LOCATION", "LOC") | ("LOC", "LOCATION") | ("LOCATION", "GPE") | ("GPE", "LOCATION") => {
            true
        }
        // Organization
        ("ORGANIZATION", "ORG") | ("ORG", "ORGANIZATION") => true,
        // Date/Time
        ("DATE", "YEAR") | ("YEAR", "DATE") | ("DATE", "HOURS") => true,
        _ => false,
    }
}

fn color(code: &str, text: &str) -> String {
    if io::stdout().is_terminal() {
        format!("\x1b[{}m{}\x1b[0m", code, text)
    } else {
        text.to_string()
    }
}

/// Resolve coreference by grouping signals into tracks.
///
/// This uses a simple rule-based approach:
/// 1. Named entities of the same type that overlap in their canonical form
/// 2. Pronouns detected in text and linked to nearest compatible antecedent
fn resolve_coreference(doc: &mut GroundedDocument, text: &str, signal_ids: &[u64]) {
    // Pronouns by gender
    let male_pronouns = ["he", "him", "his"];
    let female_pronouns = ["she", "her", "hers"];
    let neutral_pronouns = ["they", "them", "their", "theirs"]; // Can refer to any
    let org_pronouns = ["it", "its"];

    // First, detect pronouns in text that weren't found by NER
    // (pronoun_id, pronoun_type: "male", "female", "org", "any")
    let mut pronoun_signals: Vec<(u64, &str)> = Vec::new();

    // Build byte-to-char offset mapping for proper conversion
    let byte_to_char: Vec<usize> = text.char_indices().map(|(byte_idx, _)| byte_idx).collect();
    let char_count = text.chars().count();

    // Helper to convert byte offset to char offset
    let byte_to_char_offset = |byte_offset: usize| -> usize {
        byte_to_char
            .iter()
            .position(|&b| b == byte_offset)
            .unwrap_or_else(|| {
                // If exact match not found, it's at the end or between chars
                if byte_offset >= text.len() {
                    char_count
                } else {
                    // Find the char that contains this byte
                    byte_to_char
                        .iter()
                        .take_while(|&&b| b < byte_offset)
                        .count()
                }
            })
    };

    // Find pronouns in text and add them as signals
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();

    for (pronouns, ptype) in [
        (&male_pronouns[..], "male"),
        (&female_pronouns[..], "female"),
        (&org_pronouns[..], "org"),
        (&neutral_pronouns[..], "any"),
    ] {
        for &pronoun in pronouns {
            // Find all occurrences (byte offsets from find())
            let mut byte_start = 0;
            while let Some(pos) = text_lower[byte_start..].find(pronoun) {
                let abs_byte_start = byte_start + pos;
                let abs_byte_end = abs_byte_start + pronoun.len();

                // Convert to character offsets for Location::Text
                let char_start = byte_to_char_offset(abs_byte_start);
                let char_end = byte_to_char_offset(abs_byte_end);

                // Check word boundaries using character indices
                let is_word_start = char_start == 0
                    || !chars
                        .get(char_start.saturating_sub(1))
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);
                let is_word_end = char_end >= char_count
                    || !chars
                        .get(char_end)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);

                if is_word_start && is_word_end {
                    // Check if this position already has a signal (using char offsets)
                    let already_exists = doc.signals().iter().any(|s| {
                        if let Location::Text {
                            start: s_start,
                            end: s_end,
                        } = &s.location
                        {
                            *s_start == char_start && *s_end == char_end
                        } else {
                            false
                        }
                    });

                    if !already_exists {
                        // Add pronoun as a signal (use byte slice for surface text)
                        let surface = &text[abs_byte_start..abs_byte_end];
                        let signal = Signal::new(
                            0,
                            Location::text(char_start, char_end), // Use char offsets
                            surface,
                            "PRON", // Special label for pronouns
                            0.9,
                        );
                        let sig_id = doc.add_signal(signal);
                        pronoun_signals.push((sig_id, ptype));
                    }
                }

                // Advance by one byte to find next occurrence
                byte_start = abs_byte_start + 1;
            }
        }
    }

    // Group NER signals by type
    let mut per_signals: Vec<u64> = Vec::new();
    let mut org_signals: Vec<u64> = Vec::new();
    let mut loc_signals: Vec<u64> = Vec::new();

    for &sig_id in signal_ids {
        if let Some(sig) = doc.get_signal(sig_id) {
            let label_lower = sig.label.to_lowercase();
            match label_lower.as_str() {
                "per" | "person" => per_signals.push(sig_id),
                "org" | "organization" => org_signals.push(sig_id),
                "loc" | "location" | "gpe" => loc_signals.push(sig_id),
                _ => {}
            }
        }
    }

    // Create tracks from named entities (group same-type entities by canonical form)
    let mut track_assignments: HashMap<u64, u64> = HashMap::new(); // signal_id -> track_id

    // For each entity type, create tracks by grouping similar surface forms
    for signals in [&per_signals, &org_signals, &loc_signals] {
        if signals.is_empty() {
            continue;
        }

        // Simple grouping: each unique entity gets its own track
        let mut canonical_groups: HashMap<String, Vec<u64>> = HashMap::new();

        for &sig_id in signals {
            if let Some(sig) = doc.get_signal(sig_id) {
                // Use lowercase canonical form for grouping
                let canonical = normalize_entity_name(&sig.surface);
                canonical_groups.entry(canonical).or_default().push(sig_id);
            }
        }

        // Create a track for each group
        for (canonical, group_signals) in canonical_groups {
            let track_id = doc.create_track_from_signals(&canonical, &group_signals);
            if let Some(tid) = track_id {
                for &sig_id in &group_signals {
                    track_assignments.insert(sig_id, tid);
                }
            }
        }
    }

    // Link pronouns to nearest compatible antecedent's track
    for (pronoun_id, pronoun_type) in &pronoun_signals {
        let pronoun_sig = match doc.get_signal(*pronoun_id) {
            Some(s) => s.clone(),
            None => continue,
        };

        let pronoun_start = match &pronoun_sig.location {
            Location::Text { start, .. } => *start,
            _ => continue,
        };

        // For person pronouns, we need to filter by gender compatibility
        let compatible_signals: Vec<u64> = match *pronoun_type {
            "male" => {
                // Filter to likely male names
                per_signals
                    .iter()
                    .filter(|&&id| {
                        doc.get_signal(id)
                            .map(|s| is_likely_male(&s.surface))
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            }
            "female" => {
                // Filter to likely female names
                per_signals
                    .iter()
                    .filter(|&&id| {
                        doc.get_signal(id)
                            .map(|s| is_likely_female(&s.surface))
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            }
            "org" => org_signals.clone(),
            "any" => per_signals
                .iter()
                .chain(org_signals.iter())
                .cloned()
                .collect(),
            _ => continue,
        };

        let mut nearest: Option<(u64, usize)> = None; // (signal_id, distance)

        for sig_id in &compatible_signals {
            if let Some(sig) = doc.get_signal(*sig_id) {
                if let Location::Text { end, .. } = &sig.location {
                    if *end < pronoun_start {
                        let distance = pronoun_start - end;
                        if nearest.map_or(true, |(_, prev_dist)| distance < prev_dist) {
                            nearest = Some((*sig_id, distance));
                        }
                    }
                }
            }
        }

        // Add pronoun to the track of its antecedent
        if let Some((antecedent_id, _)) = nearest {
            if let Some(&track_id) = track_assignments.get(&antecedent_id) {
                // Get position (number of signals already in track)
                let position = doc
                    .get_track(track_id)
                    .map(|t| t.signals.len() as u32)
                    .unwrap_or(0);
                // Add pronoun signal to this track (updates index)
                if doc.add_signal_to_track(*pronoun_id, track_id, position) {
                    track_assignments.insert(*pronoun_id, track_id);
                }
            }
        }
    }
}

/// Simple heuristic for determining if a name is likely male.
fn is_likely_male(name: &str) -> bool {
    // Get first name (first word)
    let first_name = name.split_whitespace().next().unwrap_or("").to_lowercase();

    // Common male first names
    let male_names = [
        "james", "john", "robert", "michael", "william", "david", "richard", "joseph", "thomas",
        "charles", "barack", "donald", "joe", "george", "bill", "vladimir", "emmanuel", "boris",
        "xi", "narendra", "justin", "elon", "jeff", "mark", "steve", "tim", "satya", "sundar",
        "albert", "isaac", "stephen", "neil", "peter", "paul", "matthew", "andrew", "philip",
        "simon",
    ];

    male_names.contains(&first_name.as_str())
}

/// Simple heuristic for determining if a name is likely female.
fn is_likely_female(name: &str) -> bool {
    // Get first name (first word)
    let first_name = name.split_whitespace().next().unwrap_or("").to_lowercase();

    // Common female first names
    let female_names = [
        "mary",
        "patricia",
        "jennifer",
        "linda",
        "elizabeth",
        "angela",
        "marie",
        "susan",
        "margaret",
        "dorothy",
        "hillary",
        "nancy",
        "kamala",
        "michelle",
        "melania",
        "jill",
        "theresa",
        "ursula",
        "christine",
        "sanna",
        "jacinda",
        "oprah",
        "beyonce",
        "taylor",
        "sheryl",
        "marissa",
        "susan",
        "ginni",
        "diana",
        "catherine",
        "anne",
        "victoria",
        "queen",
        "jane",
        "sarah",
    ];

    female_names.contains(&first_name.as_str())
}

/// Link tracks to KB identities.
///
/// Creates placeholder Wikidata-style identities for each track.
/// In a production system, this would query a real KB like Wikidata.
fn link_tracks_to_kb(doc: &mut GroundedDocument) {
    // Well-known entities with Wikidata IDs
    let known_entities: HashMap<&str, (&str, &str)> = [
        (
            "barack obama",
            ("Q76", "44th President of the United States"),
        ),
        ("angela merkel", ("Q567", "Chancellor of Germany 2005-2021")),
        ("berlin", ("Q64", "Capital of Germany")),
        ("nato", ("Q7184", "North Atlantic Treaty Organization")),
        (
            "donald trump",
            ("Q22686", "45th President of the United States"),
        ),
        (
            "joe biden",
            ("Q6279", "46th President of the United States"),
        ),
        ("vladimir putin", ("Q7747", "President of Russia")),
        ("emmanuel macron", ("Q3052772", "President of France")),
        ("elon musk", ("Q317521", "CEO of Tesla and SpaceX")),
        ("marie curie", ("Q7186", "Physicist and chemist")),
        ("albert einstein", ("Q937", "Theoretical physicist")),
        ("new york", ("Q60", "City in New York State")),
        ("london", ("Q84", "Capital of the United Kingdom")),
        ("paris", ("Q90", "Capital of France")),
        ("google", ("Q95", "American technology company")),
        ("apple", ("Q312", "American technology company")),
        ("microsoft", ("Q2283", "American technology company")),
        ("united nations", ("Q1065", "International organization")),
        ("european union", ("Q458", "Political and economic union")),
    ]
    .into_iter()
    .collect();

    // Collect track IDs first to avoid borrow issues
    let track_ids: Vec<u64> = doc.tracks().map(|t| t.id).collect();

    for track_id in track_ids {
        let (canonical, entity_type) = {
            let track = match doc.get_track(track_id) {
                Some(t) => t,
                None => continue,
            };
            (track.canonical_surface.clone(), track.entity_type.clone())
        };

        let canonical_lower = canonical.to_lowercase();

        // Look up in known entities
        if let Some(&(qid, description)) = known_entities.get(canonical_lower.as_str()) {
            // Create identity from KB
            let mut identity = Identity::from_kb(
                0, // Will be assigned by add_identity
                &canonical, "wikidata", qid,
            );
            identity.aliases.push(description.to_string());
            if let Some(etype) = &entity_type {
                identity.entity_type = Some(etype.clone());
            }

            let identity_id = doc.add_identity(identity);
            doc.link_track_to_identity(track_id, identity_id);
        } else {
            // Create placeholder identity without KB link
            let identity = Identity::new(0, &canonical);
            let identity_id = doc.add_identity(identity);
            doc.link_track_to_identity(track_id, identity_id);
        }
    }
}

/// Normalize an entity name for grouping (lowercase, trim)
fn normalize_entity_name(name: &str) -> String {
    name.to_lowercase().trim().to_string()
}

fn type_color(typ: &str) -> &'static str {
    match typ.to_lowercase().as_str() {
        "person" | "per" => "1;34",
        "organization" | "org" => "1;32",
        "location" | "loc" | "gpe" => "1;33",
        "date" | "time" => "1;35",
        "money" | "percent" => "1;36",
        "email" | "url" | "phone" => "36",
        _ => "1;37",
    }
}

fn metric_colored(value: f64) -> String {
    let code = if value >= 90.0 {
        "1;32"
    } else if value >= 70.0 {
        "1;33"
    } else if value >= 50.0 {
        "33"
    } else {
        "1;31"
    };
    color(code, &format!("{:5.1}", value))
}

fn confidence_bar(conf: f32) -> String {
    // Clamp to valid range to prevent underflow if conf > 1.0
    let filled = ((conf * 10.0).round() as usize).min(10);
    let empty = 10 - filled;
    let code = if conf >= 0.9 {
        "32"
    } else if conf >= 0.7 {
        "33"
    } else {
        "31"
    };
    format!(
        "{}{} {:3.0}%",
        color(code, &"#".repeat(filled)),
        color("90", &".".repeat(empty)),
        conf * 100.0
    )
}

fn print_signals(doc: &GroundedDocument, text: &str, verbose: bool) {
    let mut by_type: HashMap<String, Vec<&Signal<Location>>> = HashMap::new();
    for s in doc.signals() {
        by_type.entry(s.label().to_string()).or_default().push(s);
    }

    for (typ, signals) in &by_type {
        let col = type_color(typ);
        println!("  {} ({}):", color(col, typ), signals.len());
        for s in signals {
            let (start, end) = s.text_offsets().unwrap_or((0, 0));
            let neg = if s.negated {
                color("31", " [NEG]")
            } else {
                String::new()
            };
            let quant = s
                .quantifier
                .map(|q| color("35", &format!(" [{:?}]", q)))
                .unwrap_or_default();

            println!(
                "    [{:3},{:3}) {} \"{}\"{}{}",
                start,
                end,
                confidence_bar(s.confidence),
                s.surface(),
                neg,
                quant
            );

            if verbose {
                let ctx_start = start.saturating_sub(15);
                let ctx_end = (end + 15).min(text.chars().count());
                let before: String = text
                    .chars()
                    .skip(ctx_start)
                    .take(start - ctx_start)
                    .collect();
                let entity: String = text.chars().skip(start).take(end - start).collect();
                let after: String = text.chars().skip(end).take(ctx_end - end).collect();
                println!(
                    "           {}{}{}{}{}",
                    color("90", "..."),
                    color("90", &before),
                    color("1;33", &entity),
                    color("90", &after),
                    color("90", "...")
                );
            }
        }
    }
}

fn print_annotated_entities(text: &str, entities: &[Entity]) {
    let mut sorted: Vec<&Entity> = entities.iter().collect();
    sorted.sort_by_key(|e| e.start);

    let chars: Vec<char> = text.chars().collect();
    let char_len = chars.len();
    let mut result = String::new();
    let mut last_end = 0;

    for e in sorted {
        if e.start >= char_len || e.end > char_len || e.start >= e.end {
            continue;
        }
        if e.start < last_end {
            continue;
        }

        if e.start > last_end {
            let before: String = chars[last_end..e.start].iter().collect();
            result.push_str(&before);
        }

        let col = type_color(e.entity_type.as_label());
        let entity_text: String = chars[e.start..e.end].iter().collect();
        result.push_str(&color(
            col,
            &format!("[{}: {}]", e.entity_type.as_label(), entity_text),
        ));
        last_end = e.end;
    }

    if last_end < char_len {
        let after: String = chars[last_end..].iter().collect();
        result.push_str(&after);
    }

    println!();
    for line in result.lines() {
        println!("  {}", line);
    }
}

fn print_annotated_signals(text: &str, signals: &[Signal<Location>]) {
    let mut sorted: Vec<&Signal<Location>> = signals.iter().collect();
    sorted.sort_by_key(|s| s.text_offsets().map(|(start, _)| start).unwrap_or(0));

    let chars: Vec<char> = text.chars().collect();
    let char_len = chars.len();
    let mut result = String::new();
    let mut last_end = 0;

    for s in sorted {
        let (start, end) = match s.text_offsets() {
            Some((start, end)) => (start, end),
            None => continue,
        };

        if start >= char_len || end > char_len || start >= end {
            continue;
        }
        if start < last_end {
            continue;
        }

        if start > last_end {
            let before: String = chars[last_end..start].iter().collect();
            result.push_str(&before);
        }

        let col = type_color(s.label());
        let entity_text: String = chars[start..end].iter().collect();
        result.push_str(&color(col, &format!("[{}: {}]", s.label(), entity_text)));
        last_end = end;
    }

    if last_end < char_len {
        let after: String = chars[last_end..].iter().collect();
        result.push_str(&after);
    }

    println!();
    for line in result.lines() {
        println!("  {}", line);
    }
}

fn print_matches(cmp: &EvalComparison, _verbose: bool) {
    for m in &cmp.matches {
        match m {
            EvalMatch::Correct { gold_id, .. } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("32", "+"),
                    color("32", "correct"),
                    g.map(|s| s.label.as_str()).unwrap_or("?"),
                    g.map(|s| s.surface()).unwrap_or("?")
                );
            }
            EvalMatch::TypeMismatch {
                gold_id,
                gold_label,
                pred_label,
                ..
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  {} {}: \"{}\" ({} -> {})",
                    color("33", "!"),
                    color("33", "type mismatch"),
                    g.map(|s| s.surface()).unwrap_or("?"),
                    gold_label,
                    pred_label
                );
            }
            EvalMatch::BoundaryError {
                gold_id,
                pred_id,
                iou,
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                println!(
                    "  {} {}: gold=\"{}\" pred=\"{}\" (IoU={:.2})",
                    color("33", "!"),
                    color("33", "boundary"),
                    g.map(|s| s.surface()).unwrap_or("?"),
                    p.map(|s| s.surface()).unwrap_or("?"),
                    iou
                );
            }
            EvalMatch::Spurious { pred_id } => {
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("31", "x"),
                    color("31", "false positive"),
                    p.map(|s| s.label.as_str()).unwrap_or("?"),
                    p.map(|s| s.surface()).unwrap_or("?")
                );
            }
            EvalMatch::Missed { gold_id } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("31", "x"),
                    color("31", "false negative"),
                    g.map(|s| s.label.as_str()).unwrap_or("?"),
                    g.map(|s| s.surface()).unwrap_or("?")
                );
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gold_spec_simple() {
        let spec =
            parse_gold_spec("Marie Curie:PER:0:11").expect("Test gold spec should parse correctly");
        assert_eq!(spec.text, "Marie Curie");
        assert_eq!(spec.label, "PER");
        assert_eq!(spec.start, 0);
        assert_eq!(spec.end, 11);
    }

    #[test]
    fn test_parse_gold_spec_with_colon_in_text() {
        // URL containing colons
        let spec = parse_gold_spec("https://example.com:URL:0:19")
            .expect("Test gold spec should parse correctly");
        assert_eq!(spec.text, "https://example.com");
        assert_eq!(spec.label, "URL");
        assert_eq!(spec.start, 0);
        assert_eq!(spec.end, 19);
    }

    #[test]
    fn test_parse_gold_spec_invalid() {
        assert!(parse_gold_spec("invalid").is_none());
        assert!(parse_gold_spec("text:label").is_none());
        assert!(parse_gold_spec("text:label:notanumber:10").is_none());
    }

    #[test]
    fn test_is_negated() {
        assert!(is_negated("He is not a doctor", 10));
        assert!(is_negated("Never trust John", 12));
        assert!(!is_negated("Trust John", 6));
    }

    #[test]
    fn test_detect_quantifier() {
        assert_eq!(
            detect_quantifier("every employee", 6),
            Some(Quantifier::Universal)
        );
        assert_eq!(
            detect_quantifier("some people", 5),
            Some(Quantifier::Existential)
        );
        assert_eq!(
            detect_quantifier("the manager", 4),
            Some(Quantifier::Definite)
        );
        assert_eq!(detect_quantifier("John Smith", 0), None);
    }

    #[test]
    fn test_model_backend_names() {
        assert_eq!(ModelBackend::Pattern.name(), "pattern");
        assert_eq!(ModelBackend::Heuristic.name(), "heuristic");
        assert_eq!(ModelBackend::Stacked.name(), "stacked");
    }

    #[test]
    fn test_confidence_bar_normal() {
        // Normal cases
        let bar = confidence_bar(0.5);
        assert!(bar.contains("50%"));

        let bar = confidence_bar(1.0);
        assert!(bar.contains("100%"));

        let bar = confidence_bar(0.0);
        assert!(bar.contains("0%"));
    }

    #[test]
    fn test_confidence_bar_clamping() {
        // Edge case: confidence slightly over 1.0 should not panic
        let bar = confidence_bar(1.01);
        assert!(bar.contains("101%")); // Display shows actual value
                                       // But the bar itself should be clamped to 10 filled chars (not panic)

        // Edge case: confidence at exactly 1.0
        let bar = confidence_bar(1.0);
        assert!(bar.contains("100%"));
    }

    #[test]
    fn test_is_negated_unicode() {
        // Test with Unicode text (character offsets, not byte offsets)
        // "café" has 4 chars but 5 bytes (é is 2 bytes in UTF-8)
        assert!(!is_negated("café John", 5)); // "John" starts at char 5
        assert!(is_negated("not café John", 9)); // "not" is in the prefix
    }

    #[test]
    fn test_detect_quantifier_unicode() {
        // Test with Unicode text
        // "every café employee" - "employee" starts at char index 11
        assert_eq!(
            detect_quantifier("every café employee", 11),
            None // "café" is not a quantifier
        );
        // "every employee" still works
        assert_eq!(
            detect_quantifier("every employee", 6),
            Some(Quantifier::Universal)
        );
    }

    #[test]
    fn test_normalize_entity_name() {
        assert_eq!(normalize_entity_name("  John Smith  "), "john smith");
        assert_eq!(normalize_entity_name("MARIE CURIE"), "marie curie");
        assert_eq!(normalize_entity_name("Test"), "test");
    }

    #[test]
    fn test_is_likely_male() {
        assert!(is_likely_male("John Smith"));
        assert!(is_likely_male("Barack Obama"));
        assert!(!is_likely_male("Marie Curie"));
        assert!(!is_likely_male("Unknown Person"));
    }

    #[test]
    fn test_is_likely_female() {
        assert!(is_likely_female("Marie Curie"));
        assert!(is_likely_female("Hillary Clinton"));
        assert!(!is_likely_female("John Smith"));
        assert!(!is_likely_female("Unknown Person"));
    }

    #[test]
    fn test_type_color() {
        assert_eq!(type_color("PER"), "1;34");
        assert_eq!(type_color("person"), "1;34");
        assert_eq!(type_color("ORG"), "1;32");
        assert_eq!(type_color("LOC"), "1;33");
        assert_eq!(type_color("UNKNOWN"), "1;37");
    }

    #[test]
    fn test_metric_colored() {
        // High score (>= 90)
        let result = metric_colored(95.0);
        assert!(result.contains("95.0"));

        // Medium score (>= 70)
        let result = metric_colored(75.0);
        assert!(result.contains("75.0"));

        // Low score (< 50)
        let result = metric_colored(30.0);
        assert!(result.contains("30.0"));
    }

    #[test]
    fn test_color_function() {
        // When not in a terminal, color() should return plain text
        // This test verifies the function doesn't panic
        let result = color("32", "test");
        assert!(result.contains("test"));
    }
}
