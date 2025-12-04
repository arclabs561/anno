//! Compare command - Compare documents, models, or clusters

use clap::Parser;
use std::fs;

use super::super::parser::ModelBackend;
use crate::Entity;
use anno_core::GroundedDocument;

/// Compare documents, models, or clusters
#[derive(Parser, Debug)]
pub struct CompareArgs {
    /// First input file
    #[arg(value_name = "FILE1")]
    pub file1: String,

    /// Second input file (or text for compare-models)
    #[arg(value_name = "FILE2")]
    pub file2: Option<String>,

    /// Compare models on same text (use file1 as text)
    #[arg(long)]
    pub models: bool,

    /// Models to compare (when --models is used)
    #[arg(long, value_delimiter = ',', value_name = "MODEL")]
    pub model_list: Vec<String>,

    /// Output format (diff, table, summary)
    #[arg(long, default_value = "diff")]
    pub format: String,

    /// Output file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,
}

pub fn run(args: CompareArgs) -> Result<(), String> {
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
