//! Query command - Query and filter entities/clusters

use clap::Parser;
use std::fs;
use std::io::{self, Read};

use super::super::parser::OutputFormat;
use super::super::utils::{format_error, parse_grounded_document, read_input_file};
use anno_core::{Location, Signal};

/// Query and filter entities/clusters
#[derive(Parser, Debug)]
pub struct QueryArgs {
    /// Input file (GroundedDocument JSON or cross-doc clusters JSON)
    #[arg(value_name = "FILE")]
    pub input: String,

    /// Filter by entity type
    #[arg(short, long, value_name = "TYPE")]
    pub r#type: Option<String>,

    /// Find specific entity by name
    #[arg(short, long, value_name = "TEXT")]
    pub entity: Option<String>,

    /// Minimum confidence threshold
    #[arg(long, value_name = "FLOAT")]
    pub min_confidence: Option<f64>,

    /// Filter expression (e.g., "type=ORG AND confidence>0.7")
    #[arg(short, long, value_name = "EXPR")]
    pub filter: Option<String>,

    /// Start offset for range queries (character position)
    #[arg(long, value_name = "OFFSET")]
    pub start_offset: Option<usize>,

    /// End offset for range queries (character position)
    #[arg(long, value_name = "OFFSET")]
    pub end_offset: Option<usize>,

    /// Filter for negated signals only
    #[arg(long)]
    pub negated: bool,

    /// Filter for signals with quantifiers
    #[arg(long)]
    pub quantified: bool,

    /// Filter for untracked signals (not in any track)
    #[arg(long)]
    pub untracked: bool,

    /// Filter for signals linked to identities (via tracks)
    #[arg(long)]
    pub linked: bool,

    /// Filter for signals not linked to identities
    #[arg(long)]
    pub unlinked: bool,

    /// Output format
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Output file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,
}

pub fn run(args: QueryArgs) -> Result<(), String> {
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
    } else if let Ok(clusters) =
        serde_json::from_str::<Vec<crate::eval::cdcr::CrossDocCluster>>(&json_content)
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
            match args.format {
                OutputFormat::Tree => {
                    for cluster in &filtered {
                        println!("Cluster {}: {}", cluster.id, cluster.canonical_name);
                        for (doc_id, entity_idx) in &cluster.mentions {
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
            let _clusters = clusters; // Suppress unused variable warning
            return Err("Cross-doc cluster querying requires 'eval-advanced' feature".to_string());
        }
    } else {
        return Err("Failed to parse input as GroundedDocument or cross-doc clusters".to_string());
    }

    Ok(())
}
