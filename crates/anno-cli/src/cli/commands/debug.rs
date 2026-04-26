//! Debug command - Generate HTML debug visualization

use clap::Parser;
use std::fs;

use super::super::output::{color, print_signals};
use super::super::parser::ModelBackend;
use super::super::utils::{get_input_text, link_tracks_to_kb, resolve_coreference};

#[cfg(feature = "eval")]
use crate::cli::ingest::{CompositeResolver, UrlResolver};
use anno::core::grounded::{render_document_html, GroundedDocument, Location, Signal, SignalId};
#[cfg(feature = "graph")]
use anno::graph::{GraphDocument, GraphExportFormat};
use anno::ingest::DocumentPreprocessor;

/// Generate HTML debug visualization
#[derive(Parser, Debug)]
pub struct DebugArgs {
    /// Input text to process
    #[arg(short, long)]
    pub text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    pub file: Option<String>,

    /// Positional text arguments (alternative to --text)
    #[arg(value_name = "TEXT")]
    pub positional: Vec<String>,

    /// URL to fetch content from (requires `eval` feature)
    #[arg(long, value_name = "URL")]
    pub url: Option<String>,

    /// Clean whitespace (normalize spaces, line breaks)
    #[arg(long)]
    pub clean: bool,

    /// Normalize Unicode (basic normalization)
    #[arg(long)]
    pub normalize: bool,

    /// Detect and record language
    #[arg(long)]
    pub detect_lang: bool,

    /// Export to graph format (neo4j, networkx, jsonld)
    #[arg(long, value_name = "FORMAT")]
    pub export_graph: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Output as HTML (default: text)
    #[arg(long)]
    pub html: bool,

    /// Export GroundedDocument JSON to file (for pipeline integration)
    #[arg(long, value_name = "PATH")]
    pub export: Option<String>,

    /// Export format when using --export (full, signals, minimal)
    #[arg(long, default_value = "full", value_name = "FORMAT")]
    pub export_format: String,

    /// Write output to file (default: stdout)
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,

    /// Run coreference resolution to form tracks
    #[arg(long)]
    pub coref: bool,

    /// Attach demo KB-style IDs to tracks (offline; no network).
    #[arg(long)]
    pub link_kb: bool,

    /// Suppress status messages
    #[arg(short, long)]
    pub quiet: bool,

    /// Verbose output (repeat for more detail: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

/// Execute the debug command.
pub fn run(args: DebugArgs) -> Result<(), String> {
    // Level 1 + 2 (Signal → Track): Entity extraction + within-document coreference
    // With --link-kb: attach demo identities to tracks (debug-only, offline).

    // Pick a stable document id so the HTML report is self-describing.
    // (Otherwise every report is doc_id="debug", which is annoying when comparing runs.)
    let doc_id = if let Some(url) = &args.url {
        url.clone()
    } else if let Some(path) = &args.file {
        path.clone()
    } else {
        "debug".to_string()
    };

    // Resolve input: URL, file, text, or stdin
    let mut raw_text = if let Some(url) = &args.url {
        #[cfg(feature = "eval")]
        {
            let resolver = CompositeResolver::new();
            let resolved = resolver
                .resolve(url)
                .map_err(|e| format!("Failed to fetch URL {}: {}", url, e))?;
            resolved.text
        }
        #[cfg(not(feature = "eval"))]
        {
            #[allow(unused_variables)]
            let _url = url;
            return Err(
                "URL resolution requires 'eval' feature. Enable with: cargo build -p anno-cli --features eval"
                    .to_string(),
            );
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
        if args.verbose >= 1 && !prepared.metadata.is_empty() {
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
    let mut doc = GroundedDocument::new(doc_id, &text);
    let mut signal_ids: Vec<SignalId> = Vec::new();

    for e in &entities {
        let id = doc.add_signal(Signal::from(e));
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
                let signals: Vec<_> = doc.signals().to_vec();
                serde_json::json!({
                    "id": doc.id(),
                    "text": doc.text(),
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
                    "id": doc.id(),
                    "text": doc.text(),
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
    let _index = doc.build_text_index();
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
        // Text output (default) - use dense format with verbose levels
        if doc.signals().is_empty() {
            println!("(no entities)");
        } else {
            // Use verbose level from args, but ensure tracks/identities are shown if coref/link_kb was run
            let effective_verbose = if args.coref || args.link_kb {
                args.verbose.max(2) // At least level 2 if coref or KB linking was run
            } else {
                args.verbose
            };
            print_signals(&doc, &text, effective_verbose);
        }
    }

    // Export to graph format if requested (always prints to stdout).
    if let Some(graph_format_str) = args.export_graph.as_deref() {
        #[cfg(not(feature = "graph"))]
        {
            let _ = graph_format_str;
            return Err("Graph export requires the 'graph' feature to be enabled.".to_string());
        }

        #[cfg(feature = "graph")]
        {
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

            let graph = anno::graph::grounded_to_graph_document(&doc);
            let graph_output = graph.export(graph_format);

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
    }

    Ok(())
}
