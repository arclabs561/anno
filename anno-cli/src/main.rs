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
//! Level 3 (Identity) : Cross-document entity coalescing and KB linking
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

#[cfg(any(feature = "eval", feature = "eval-advanced"))]
use glob::glob;

use clap::{CommandFactory, Parser, ValueEnum};
use is_terminal::IsTerminal;

#[cfg(feature = "eval")]
use anno::eval::backend_factory::BackendFactory;
use anno::graph::{GraphDocument, GraphExportFormat};
use anno::grounded::{
    render_document_html, render_eval_html, EvalComparison, EvalMatch, GroundedDocument, Identity,
    Location, Modality, Quantifier, Signal, SignalValidationError,
};
use anno::ingest::DocumentPreprocessor;
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
// Use CLI module's Cli and Commands
use anno::cli::parser::OutputFormat;
// Import Args types directly from commands module (they're re-exported)
use anno::cli::commands::{CompareArgs, EnhanceArgs, ModelsArgs, QueryArgs};
// Import action enums from their specific modules
use anno::cli::commands::models::ModelsAction;

// ============================================================================
// Shared Types (Legacy - most moved to cli module)
// ============================================================================
// ModelBackend, OutputFormat, EvalTask are now in src/cli/parser.rs

/// Model backend selection (legacy - use anno::cli::parser::ModelBackend)
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
#[allow(dead_code)]
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

// OutputFormat and EvalTask are now in src/cli/parser.rs

// ============================================================================
// Command Arguments
// ============================================================================

// All Args structs moved to cli module - see src/cli/commands/*.rs

// ============================================================================
// Legacy Command Handlers (to be extracted)
// ============================================================================
// All Args structs are now in src/cli/commands/*.rs - these are just the handlers

#[cfg(feature = "eval")]
use anno::eval::loader::DatasetId;
#[cfg(feature = "eval-advanced")]
use anno::ingest::url_resolver::CompositeResolver;

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> ExitCode {
    use anno::cli::commands::*;
    use anno::cli::output::color;
    use anno::cli::parser::{Cli, Commands, ModelBackend, OutputFormat};
    use clap_complete::generate;

    let cli = Cli::parse();

    let result: Result<(), String> = match cli.command {
        Some(Commands::Extract(args)) => extract::run(args),
        Some(Commands::Debug(args)) => debug::run(args),
        Some(Commands::Eval(args)) => eval::run(args),
        Some(Commands::Validate(args)) => validate::run(args),
        Some(Commands::Analyze(args)) => analyze::run(args),
        Some(Commands::Dataset(args)) => dataset::run(args),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::Benchmark(args)) => benchmark::run(args),
        Some(Commands::Info) => info::run(),
        Some(Commands::Models(args)) => models::run(args),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::CrossDoc(args)) => crossdoc::run(args),
        #[cfg(feature = "eval-advanced")]
        Some(Commands::Strata(args)) => strata::run(args),
        Some(Commands::Enhance(args)) => enhance::run(args),
        Some(Commands::Pipeline(args)) => pipeline::run(args),
        Some(Commands::Query(args)) => query::run(args),
        Some(Commands::Compare(args)) => compare::run(args),
        Some(Commands::Cache(args)) => cache::run(args),
        Some(Commands::Config(args)) => config::run(args),
        Some(Commands::Batch(args)) => batch::run(args),
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
            extract::run(anno::cli::commands::ExtractArgs {
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

fn cmd_extract(args: anno::cli::commands::ExtractArgs) -> Result<(), String> {
    // Level 1 (Signal): Raw entity extraction from single document
    // This is the foundation for all other commands:
    // - `debug` adds Level 2 (Track) via coreference resolution
    // - `debug --link-kb` adds Level 3 (Identity) via KB linking
    // - `crossdoc`/`coalesce` clusters Level 1 entities across multiple documents

    // Resolve input: URL, file, text, or stdin
    let mut raw_text = if let Some(url) = &args.url {
        #[cfg(feature = "eval-advanced")]
        {
            use anno::ingest::UrlResolver;
            let resolver = CompositeResolver::new();
            let resolved = resolver
                .resolve(url)
                .map_err(|e| format!("Failed to fetch URL {}: {}", url, e))?;
            resolved.text
        }
        #[cfg(not(feature = "eval-advanced"))]
        {
            #[allow(unused_variables)]
            let _url = url;
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
                "Tree/Summary formats are only available for crossdoc/coalesce command."
                    .to_string(),
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

fn cmd_debug(args: anno::cli::commands::DebugArgs) -> Result<(), String> {
    // Level 1 + 2 (Signal → Track): Entity extraction + within-document coreference
    // With --link-kb: Level 1 + 2 + 3 (Signal → Track → Identity): Adds KB linking
    // This builds the full hierarchy that could be used by coalescing for better clustering

    // Resolve input: URL, file, text, or stdin
    let mut raw_text = if let Some(url) = &args.url {
        #[cfg(feature = "eval-advanced")]
        {
            use anno::ingest::UrlResolver;
            let resolver = CompositeResolver::new();
            let resolved = resolver
                .resolve(url)
                .map_err(|e| format!("Failed to fetch URL {}: {}", url, e))?;
            resolved.text
        }
        #[cfg(not(feature = "eval-advanced"))]
        {
            #[allow(unused_variables)]
            let _url = url;
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

fn cmd_eval(args: anno::cli::commands::EvalArgs) -> Result<(), String> {
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

fn cmd_validate(args: anno::cli::commands::ValidateArgs) -> Result<(), String> {
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

fn cmd_analyze(args: anno::cli::commands::AnalyzeArgs) -> Result<(), String> {
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

// cmd_dataset moved to src/cli/commands/dataset.rs

// cmd_benchmark moved to src/cli/commands/benchmark.rs

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

// cmd_enhance moved to src/cli/commands/enhance.rs
fn _cmd_enhance_legacy(_args: EnhanceArgs) -> Result<(), String> {
    // This function is a legacy stub - functionality moved to src/cli/commands/enhance.rs
    Err("This function should not be called. Use enhance::run() instead.".to_string())
}

// cmd_pipeline moved to src/cli/commands/pipeline.rs

// cmd_query moved to src/cli/commands/query.rs
fn _cmd_query_legacy(_args: QueryArgs) -> Result<(), String> {
    // This function is a legacy stub - functionality moved to src/cli/commands/query.rs
    Err("This function should not be called. Use query::run() instead.".to_string())
}
fn _cmd_compare_legacy(_args: CompareArgs) -> Result<(), String> {
    // This function is a legacy stub - functionality moved to src/cli/commands/compare.rs
    Err("This function should not be called. Use compare::run() instead.".to_string())
}

// ============================================================================
// Cache management
// cmd_cache moved to src/cli/commands/cache.rs

/// Configuration management
// cmd_config moved to src/cli/commands/config.rs

/// Batch processing
// cmd_batch moved to src/cli/commands/batch.rs

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
