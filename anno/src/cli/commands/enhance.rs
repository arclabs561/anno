//! Enhance command - Enhance existing GroundedDocument with additional processing

use clap::Parser;
use std::fs;
use std::io::{self, Read};

use super::super::output::{color, print_signals};
use super::super::parser::OutputFormat;
use super::super::utils::{link_tracks_to_kb, log_success, resolve_coreference};
use anno_core::GroundedDocument;
use anno_core::{GraphDocument, GraphExportFormat};

/// Enhance existing GroundedDocument with additional processing
#[derive(Parser, Debug)]
pub struct EnhanceArgs {
    /// Input GroundedDocument JSON file (or "-" for stdin)
    #[arg(value_name = "FILE")]
    pub input: String,

    /// Run coreference resolution to form tracks
    #[arg(long)]
    pub coref: bool,

    /// Link tracks to KB identities
    #[arg(long)]
    pub link_kb: bool,

    /// Export enhanced document to file
    #[arg(short, long, value_name = "PATH")]
    pub export: Option<String>,

    /// Export format (full, signals, minimal)
    #[arg(long, default_value = "full", value_name = "FORMAT")]
    pub export_format: String,

    /// Output format for display
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Suppress status messages
    #[arg(short, long)]
    pub quiet: bool,

    /// Export to graph format (neo4j, networkx, jsonld)
    #[arg(long, value_name = "FORMAT")]
    pub export_graph: Option<String>,
}

pub fn run(args: EnhanceArgs) -> Result<(), String> {
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
