//! Pipeline command - Unified pipeline command

use clap::Parser;
use std::fs;

use super::super::output::{color, print_signals};
use super::super::parser::{ModelBackend, OutputFormat};
use super::super::utils::{link_tracks_to_kb, resolve_coreference};
use anno_core::{GroundedDocument, Location, Signal};

/// Unified pipeline command
#[derive(Parser, Debug)]
pub struct PipelineArgs {
    /// Input text(s) to process (positional)
    #[arg(trailing_var_arg = true)]
    pub text: Vec<String>,

    /// Read input from file(s)
    #[arg(short, long, value_name = "PATH")]
    pub files: Vec<String>,

    /// Process directory of text files
    #[arg(short, long, value_name = "DIR")]
    pub dir: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Run coreference resolution
    #[arg(long)]
    pub coref: bool,

    /// Link tracks to KB identities
    #[arg(long)]
    pub link_kb: bool,

    /// Run cross-document clustering
    #[arg(long)]
    pub cross_doc: bool,

    /// Similarity threshold for cross-doc clustering
    #[arg(long, default_value = "0.6")]
    pub threshold: f64,

    /// Output format
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Export results to file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,

    /// Show progress
    #[arg(long)]
    pub progress: bool,

    /// Suppress status messages
    #[arg(short, long)]
    pub quiet: bool,
}

pub fn run(args: PipelineArgs) -> Result<(), String> {
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
            use crate::eval::cdcr::{CDCRConfig, CDCRResolver, Document};

            // Convert GroundedDocuments to CDCR Documents
            let cdcr_docs: Vec<Document> = documents
                .iter()
                .map(|doc| {
                    let entities: Vec<_> = doc
                        .signals()
                        .iter()
                        .map(|s| {
                            let (start, end) = s.text_offsets().unwrap_or((0, 0));
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
