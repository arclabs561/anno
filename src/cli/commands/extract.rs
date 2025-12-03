//! Extract command - Level 1 (Signal): Raw entity extraction

use clap::Parser;
use std::time::Instant;

use super::super::parser::{ModelBackend, OutputFormat};
use super::super::output::{log_info, write_output};
use super::super::utils::{get_input_text, read_input_file};

use anno::ingest::{CompositeResolver, DocumentPreprocessor};
use anno::{Model, Entity};

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

pub fn cmd_extract(args: ExtractArgs) -> Result<(), String> {
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
            log_info(&format!("Preprocessing metadata: {:?}", prepared.metadata), args.quiet);
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
    let mut entities = entities;
    if !args.labels.is_empty() {
        let label_set: std::collections::HashSet<String> = args.labels.iter().cloned().collect();
        entities.retain(|e| label_set.contains(&e.entity_type.to_string()));
    }

    // Format output based on format type
    let output = match args.format {
        OutputFormat::Human => {
            if entities.is_empty() {
                "No entities found.\n".to_string()
            } else {
                let mut result = format!("Found {} entities in {:.2}ms:\n\n", entities.len(), elapsed.as_secs_f64() * 1000.0);
                for entity in &entities {
                    result.push_str(&format!("  {}: {} ({}-{})\n", 
                        entity.entity_type, 
                        entity.text, 
                        entity.start, 
                        entity.end));
                }
                result
            }
        }
        OutputFormat::Json => {
            serde_json::to_string_pretty(&entities)
                .map_err(|e| format!("Failed to serialize entities: {}", e))?
        }
        OutputFormat::Jsonl => {
            entities.iter()
                .map(|e| serde_json::to_string(e))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to serialize entities: {}", e))?
                .join("\n") + "\n"
        }
        OutputFormat::Tsv => {
            let mut result = "text\ttype\tstart\tend\tconfidence\n".to_string();
            for entity in &entities {
                result.push_str(&format!("{}\t{}\t{}\t{}\t{}\n",
                    entity.text,
                    entity.entity_type,
                    entity.start,
                    entity.end,
                    entity.confidence));
            }
            result
        }
        _ => return Err(format!("Format {:?} not yet implemented for extract command", args.format)),
    };

    write_output(&output, None)?;

    if args.verbose {
        log_info(&format!("Extraction completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0), args.quiet);
    }

    Ok(())
}

