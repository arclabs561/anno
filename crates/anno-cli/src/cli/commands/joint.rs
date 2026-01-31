//! Joint command - Joint entity analysis (NER + Coreference + Linking)
//!
//! Implements Durrett & Klein (2014): "A Joint Model for Entity Analysis"
//! Uses belief propagation for inference over a factor graph.
//!
//! # Examples
//!
//! ```bash
//! # Basic usage
//! anno joint "Barack Obama visited France. The president met with Macron."
//!
//! # From file
//! anno joint --file document.txt
//!
//! # Lower threshold to detect more entities
//! anno joint -t "..." --threshold 0.3
//!
//! # JSON output for pipeline integration
//! anno joint -t "..." --format json
//!
//! # Disable specific factors for ablation
//! anno joint -t "..." --no-coref-ner
//! ```

use super::super::output::{color, print_annotated_entities};
use super::super::parser::{ModelBackend, OutputFormat};
use super::super::utils::get_input_text;
use anno::joint::{JointConfig, JointModel};
use anno::Model;
use anno_core::Entity;
use clap::Parser;
use std::time::Instant;

/// Joint entity analysis (NER + Coreference + Linking)
///
/// Combines named entity recognition, coreference resolution, and entity linking
/// in a single factor graph model using belief propagation for inference.
///
/// Based on: Durrett & Klein (2014) "A Joint Model for Entity Analysis"
#[derive(Parser, Debug)]
#[command(after_help = "EXAMPLES:\n  \
        anno joint \"Barack Obama visited France.\"\n  \
        anno joint -f article.txt --format json\n  \
        anno joint -t \"...\" --threshold 0.3 --no-coref-link")]
pub struct JointArgs {
    /// Input text to process
    #[arg(short, long)]
    pub text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    pub file: Option<String>,

    /// Positional text argument
    /// Positional text input
    pub positional: Vec<String>,

    /// Model backend for initial NER
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Confidence threshold for NER (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub threshold: f64,

    /// Maximum belief propagation iterations
    #[arg(long, default_value = "5")]
    pub max_iterations: usize,

    /// Disable Link+NER cross-task factors
    #[arg(long, help = "Ablation: remove Wikipedia semantics from NER")]
    pub no_link_ner: bool,

    /// Disable Coref+NER cross-task factors
    #[arg(long, help = "Ablation: remove type consistency across coreference")]
    pub no_coref_ner: bool,

    /// Disable Coref+Link cross-task factors
    #[arg(long, help = "Ablation: remove link relatedness across coreference")]
    pub no_coref_link: bool,

    /// Output format
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Export full result to JSON file
    #[arg(long, value_name = "PATH")]
    pub export: Option<String>,

    /// Verbose output (repeat for more detail: -v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Minimal output (suppress status messages)
    #[arg(short, long)]
    pub quiet: bool,
}

/// Run joint entity analysis
pub fn run(args: JointArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &args.positional)?;

    // Create NER model for initial entity extraction
    let ner_model: Box<dyn Model> = args.model.create_model()?;

    // Extract initial entities
    let start_ner = Instant::now();
    let entities: Vec<Entity> = ner_model
        .extract_entities(&text, None)
        .map_err(|e| format!("NER extraction failed: {}", e))?
        .into_iter()
        .filter(|e| e.confidence >= args.threshold)
        .collect();
    let ner_elapsed = start_ner.elapsed();

    // Create joint model config
    let config = JointConfig {
        enable_link_ner: !args.no_link_ner,
        enable_coref_ner: !args.no_coref_ner,
        enable_coref_link: !args.no_coref_link,
        max_iterations: args.max_iterations,
        ..Default::default()
    };

    // Create and run joint model
    let joint_model =
        JointModel::new(config).map_err(|e| format!("Failed to create joint model: {}", e))?;

    let start_joint = Instant::now();
    let result = joint_model
        .analyze(&text, &entities)
        .map_err(|e| format!("Joint analysis failed: {}", e))?;
    let joint_elapsed = start_joint.elapsed();

    // Handle different output formats
    match args.format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            return run_json_output(&text, &result, &args);
        }
        OutputFormat::Grounded => {
            // Export full JointResult as JSON
            let json = serde_json::to_string_pretty(&result)
                .map_err(|e| format!("Failed to serialize result: {}", e))?;
            println!("{}", json);
            return Ok(());
        }
        _ => {} // Continue to human-readable output
    }

    // Human-readable output
    if !args.quiet {
        println!();
        println!(
            "{}",
            color(
                "1;36",
                "═══════════════════════════════════════════════════════════════════════"
            )
        );
        println!("  {}", color("1;36", "JOINT ENTITY ANALYSIS"));
        println!(
            "  {}",
            color("0;36", "Durrett & Klein (2014) Factor Graph Model")
        );
        println!(
            "{}",
            color(
                "1;36",
                "═══════════════════════════════════════════════════════════════════════"
            )
        );
        println!();
    }

    // Print NER results
    if !args.quiet {
        println!(
            "{}  {} entities in {:.1}ms",
            color("1;33", "NER:"),
            entities.len(),
            ner_elapsed.as_secs_f64() * 1000.0
        );

        if args.verbose > 0 {
            println!("      Model: {}", args.model.name());
            println!("      Threshold: {}", args.threshold);
        }
        println!();
    }

    // Print joint results
    if !args.quiet {
        // Count entity types for summary
        let mut type_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for e in &result.entities {
            *type_counts
                .entry(e.entity_type.as_label().to_string())
                .or_insert(0) += 1;
        }

        println!(
            "{}  {} entities, {} chains, {} links in {:.1}ms",
            color("1;33", "Joint:"),
            result.entities.len(),
            result.chains.len(),
            result.links.len(),
            joint_elapsed.as_secs_f64() * 1000.0
        );

        // Show type breakdown
        if !type_counts.is_empty() {
            let type_summary: Vec<String> = type_counts
                .iter()
                .map(|(t, c)| format!("{}: {}", t, c))
                .collect();
            println!("      Types: {}", type_summary.join(", "));
        }

        if args.verbose > 0 {
            println!("      BP Iterations: {}", args.max_iterations);
            println!(
                "      Factors: Link+NER={}, Coref+NER={}, Coref+Link={}",
                if args.no_link_ner { "off" } else { "on" },
                if args.no_coref_ner { "off" } else { "on" },
                if args.no_coref_link { "off" } else { "on" }
            );
        }
        println!();
    }

    // Print coreference chains
    if !result.chains.is_empty() {
        println!("{}", color("1;33", "Coreference Chains:"));
        for chain in &result.chains {
            let mentions: Vec<String> = chain
                .mentions
                .iter()
                .map(|m| format!("\"{}\"", m.text))
                .collect();
            println!(
                "  Chain {}: {}",
                chain.cluster_id.unwrap_or(anno_core::CanonicalId::ZERO),
                mentions.join(" ↔ ")
            );
        }
        println!();
    }

    // Print entity links
    if !result.links.is_empty() {
        println!("{}", color("1;33", "Entity Links:"));
        for link in &result.links {
            let kb_id = link.kb_id.as_deref().unwrap_or("NIL");
            println!(
                "  \"{}\" → {} ({:.2})",
                link.mention_text, kb_id, link.confidence
            );
        }
        println!();
    }

    // Print annotated text
    println!("{}", color("1;33", "Annotated Text:"));
    println!();
    print_annotated_entities(&text, &result.entities);
    println!();

    // Verbose: show factor graph config
    if args.verbose > 1 && !args.quiet {
        println!("{}", color("0;36", "Factor Graph Configuration:"));
        println!(
            "  Link+NER:   {} (Wikipedia semantics → NER types)",
            if args.no_link_ner {
                color("0;31", "disabled")
            } else {
                color("0;32", "enabled")
            }
        );
        println!(
            "  Coref+NER:  {} (type consistency across mentions)",
            if args.no_coref_ner {
                color("0;31", "disabled")
            } else {
                color("0;32", "enabled")
            }
        );
        println!(
            "  Coref+Link: {} (link relatedness across mentions)",
            if args.no_coref_link {
                color("0;31", "disabled")
            } else {
                color("0;32", "enabled")
            }
        );
        println!();
    }

    // Export to file if requested
    if let Some(export_path) = &args.export {
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| format!("Failed to serialize result: {}", e))?;
        std::fs::write(export_path, json)
            .map_err(|e| format!("Failed to write to {}: {}", export_path, e))?;
        if !args.quiet {
            println!(
                "{}  Exported to {}",
                color("0;32", "✓"),
                color("0;36", export_path)
            );
        }
    }

    Ok(())
}

fn run_json_output(
    text: &str,
    result: &anno::joint::JointResult,
    _args: &JointArgs,
) -> Result<(), String> {
    // Build JSON output matching other commands' structure
    let output = serde_json::json!({
        "text": text,
        "entities": result.entities.iter().map(|e| {
            serde_json::json!({
                "text": e.text,
                "type": e.entity_type.as_label(),
                "start": e.start,
                "end": e.end,
                "confidence": e.confidence
            })
        }).collect::<Vec<_>>(),
        "chains": result.chains.iter().map(|c| {
            serde_json::json!({
                "id": c.cluster_id,
                "mentions": c.mentions.iter().map(|m| {
                    serde_json::json!({
                        "text": m.text,
                        "start": m.start,
                        "end": m.end
                    })
                }).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>(),
        "links": result.links.iter().map(|l| {
            serde_json::json!({
                "mention": l.mention_text,
                "start": l.start,
                "end": l.end,
                "kb_id": l.kb_id,
                "confidence": l.confidence
            })
        }).collect::<Vec<_>>()
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|e| format!("Failed to serialize output: {}", e))?
    );
    Ok(())
}
