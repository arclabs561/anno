//! Extract command - Level 1 (Signal): Raw entity extraction

use clap::Parser;
use std::fs;
use std::time::Instant;

use super::super::output::{color, log_info, print_annotated_signals, print_signals};
use super::super::parser::{ModelBackend, OutputFormat};
use super::super::utils::{detect_quantifier, get_input_text, is_negated};

use crate::graph::{GraphDocument, GraphExportFormat}; // Re-exported from anno-core
use crate::grounded::{GroundedDocument, Location, Modality, Signal, SignalValidationError}; // Re-exported from anno-core
use crate::ingest::DocumentPreprocessor;

/// Extract entities from text
#[derive(Parser, Debug)]
pub struct ExtractArgs {
    /// Input text to process
    #[arg(short, long)]
    pub text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    pub file: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Filter to specific entity types (repeatable)
    #[arg(short, long = "label", value_name = "TYPE")]
    pub labels: Vec<String>,

    /// Output format
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Export GroundedDocument JSON to file
    #[arg(long, value_name = "PATH")]
    pub export: Option<String>,

    /// Export to graph format (neo4j, networkx, jsonld)
    #[arg(long, value_name = "FORMAT")]
    pub export_graph: Option<String>,

    /// URL to fetch content from (requires eval-advanced feature)
    #[arg(long, value_name = "URL")]
    pub url: Option<String>,

    /// Clean and normalize text before extraction
    #[arg(long)]
    pub clean: bool,

    /// Normalize Unicode
    #[arg(long)]
    pub normalize: bool,

    /// Detect and record language
    #[arg(long)]
    pub detect_lang: bool,

    /// Export format when using --export (full, signals, minimal)
    #[arg(long, default_value = "full", value_name = "FORMAT")]
    pub export_format: String,

    /// Detect negated entities
    #[arg(long)]
    pub negation: bool,

    /// Detect quantified entities
    #[arg(long)]
    pub quantifiers: bool,

    /// Show context around entities
    #[arg(short, long)]
    pub verbose: bool,

    /// Minimal output
    #[arg(short, long)]
    pub quiet: bool,

    /// Positional text argument
    #[arg(trailing_var_arg = true)]
    pub positional: Vec<String>,
}

pub fn run(args: ExtractArgs) -> Result<(), String> {
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
            log_info(
                &format!("Preprocessing metadata: {:?}", prepared.metadata),
                args.quiet,
            );
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
