//! Import command - Import annotations from brat/CoNLL/JSONL for training/evaluation
//!
//! Supports importing from:
//! - brat standoff format (.ann files)
//! - CoNLL format (IOB/BIO tagging)
//! - JSONL format

use clap::Parser;
use std::fs;
use std::path::PathBuf;

use super::super::output::color;
use super::export::ExportFormat;

/// Import annotations from different formats
#[derive(Parser, Debug)]
pub struct ImportArgs {
    /// Input file or directory
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output file (JSONL format)
    #[arg(short, long, value_name = "PATH")]
    pub output: PathBuf,

    /// Import format
    #[arg(short, long, default_value = "brat")]
    pub format: ExportFormat,

    /// Include text file content in output
    #[arg(long)]
    pub include_text: bool,

    /// Quiet mode
    #[arg(short, long)]
    pub quiet: bool,
}

/// Imported annotation from external format.
#[derive(Debug, Clone)]
pub struct ImportedAnnotation {
    /// Entity text
    pub text: String,
    /// Entity type label
    pub entity_type: String,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
    /// Source annotation system
    pub source: String,
    /// Optional confidence score
    pub confidence: Option<f64>,
}

/// Run the import command.
pub fn run(args: ImportArgs) -> Result<(), String> {
    // Validate input
    if !args.input.exists() {
        return Err(format!("Input not found: {:?}", args.input));
    }

    // Collect files to process
    let files: Vec<PathBuf> = if args.input.is_file() {
        vec![args.input.clone()]
    } else {
        let ext = match args.format {
            ExportFormat::Brat => "ann",
            ExportFormat::Conll => "conll",
            ExportFormat::Jsonl => "jsonl",
            ExportFormat::NTriples => "nt",
            ExportFormat::JsonLd => "jsonld",
        };
        fs::read_dir(&args.input)
            .map_err(|e| format!("Failed to read directory: {}", e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == ext))
            .collect()
    };

    if files.is_empty() {
        return Err("No annotation files found in input".into());
    }

    if !args.quiet {
        eprintln!(
            "{} Importing {} files from {:?} format",
            color("32", "[import]"),
            files.len(),
            args.format
        );
    }

    let mut all_annotations = Vec::new();
    let mut success_count = 0;
    let mut error_count = 0;

    for file in &files {
        match import_file(file, args.format, args.include_text) {
            Ok(annotations) => {
                let count = annotations.len();
                all_annotations.extend(annotations);
                success_count += 1;
                if !args.quiet {
                    eprintln!(
                        "  {} {:?} ({} annotations)",
                        color("32", "✓"),
                        file.file_name().unwrap_or_default(),
                        count
                    );
                }
            }
            Err(e) => {
                error_count += 1;
                if !args.quiet {
                    eprintln!(
                        "  {} {:?}: {}",
                        color("31", "✗"),
                        file.file_name().unwrap_or_default(),
                        e
                    );
                }
            }
        }
    }

    // Write output
    let output_content: String = all_annotations
        .iter()
        .map(|a| {
            serde_json::json!({
                "text": a.text,
                "type": a.entity_type,
                "start": a.start,
                "end": a.end,
                "source": a.source,
                "confidence": a.confidence,
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");

    fs::write(&args.output, output_content)
        .map_err(|e| format!("Failed to write output: {}", e))?;

    if !args.quiet {
        eprintln!();
        eprintln!(
            "{} Imported {} annotations from {} files to {:?}",
            color("32", "[done]"),
            all_annotations.len(),
            success_count,
            args.output
        );
    }

    if error_count > 0 && success_count == 0 {
        Err("All imports failed".into())
    } else {
        Ok(())
    }
}

fn import_file(
    input: &PathBuf,
    format: ExportFormat,
    include_text: bool,
) -> Result<Vec<ImportedAnnotation>, String> {
    match format {
        ExportFormat::Brat => import_brat(input, include_text),
        ExportFormat::Conll => import_conll(input),
        ExportFormat::Jsonl => import_jsonl(input),
        ExportFormat::NTriples => {
            // TODO: Parse N-Triples and extract entity annotations
            Err("Import from N-Triples format not yet supported. Use jsonl or brat.".into())
        }
        ExportFormat::JsonLd => {
            // TODO: Parse JSON-LD and extract entity annotations
            Err("Import from JSON-LD format not yet supported. Use jsonl or brat.".into())
        }
    }
}

/// Import from brat standoff format
fn import_brat(input: &PathBuf, include_text: bool) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    // Try to read corresponding .txt file for entity text
    let txt_path = input.with_extension("txt");
    let txt_content = if include_text && txt_path.exists() {
        Some(fs::read_to_string(&txt_path).ok())
    } else {
        None
    };
    let txt_content = txt_content.flatten();

    let mut annotations = Vec::new();
    let mut confidences: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    // First pass: collect confidences from attributes
    for line in content.lines() {
        if line.starts_with('A') {
            // Attribute line: A1	Confidence T1 0.85
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 && parts[1].starts_with("Confidence") {
                let attr_parts: Vec<&str> = parts[1].split_whitespace().collect();
                if attr_parts.len() >= 3 {
                    let tid = attr_parts[1];
                    if let Ok(conf) = attr_parts[2].parse::<f64>() {
                        confidences.insert(tid.to_string(), conf);
                    }
                }
            }
        }
    }

    // Second pass: parse entity annotations
    for line in content.lines() {
        if line.starts_with('T') {
            // Entity line: T1	Type Start End	Text
            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            if parts.len() >= 3 {
                let tid = parts[0];
                let type_span: Vec<&str> = parts[1].split_whitespace().collect();
                if type_span.len() >= 3 {
                    let entity_type = type_span[0].to_string();
                    let start: usize = type_span[1].parse().map_err(|_| "Invalid start offset")?;
                    let end: usize = type_span[2].parse().map_err(|_| "Invalid end offset")?;

                    // Get text from annotation or from txt file
                    let text = if parts.len() > 2 && !parts[2].is_empty() {
                        parts[2].to_string()
                    } else if let Some(ref txt) = txt_content {
                        txt.chars().skip(start).take(end - start).collect()
                    } else {
                        format!("[{}:{}]", start, end)
                    };

                    annotations.push(ImportedAnnotation {
                        text,
                        entity_type,
                        start,
                        end,
                        source: input.to_string_lossy().to_string(),
                        confidence: confidences.get(tid).copied(),
                    });
                }
            }
        }
    }

    Ok(annotations)
}

/// Import from CoNLL IOB format
fn import_conll(input: &PathBuf) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let mut annotations = Vec::new();
    let mut current_entity: Option<(String, String, usize)> = None; // (type, text, start)
    let mut char_idx = 0;

    for line in content.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let word = parts[0];
            let tag = parts[1];

            let word_len = word.len();

            if tag.starts_with("B-") {
                // End previous entity if any
                if let Some((entity_type, text, start)) = current_entity.take() {
                    annotations.push(ImportedAnnotation {
                        text,
                        entity_type,
                        start,
                        end: char_idx,
                        source: input.to_string_lossy().to_string(),
                        confidence: None,
                    });
                }
                // Start new entity
                let entity_type = tag
                    .strip_prefix("B-")
                    .expect("tag.starts_with('B-') checked above")
                    .to_string();
                current_entity = Some((entity_type, word.to_string(), char_idx));
            } else if tag.starts_with("I-") && current_entity.is_some() {
                // Continue entity
                if let Some((_, ref mut text, _)) = current_entity {
                    text.push(' ');
                    text.push_str(word);
                }
            } else {
                // End entity
                if let Some((entity_type, text, start)) = current_entity.take() {
                    annotations.push(ImportedAnnotation {
                        text,
                        entity_type,
                        start,
                        end: char_idx,
                        source: input.to_string_lossy().to_string(),
                        confidence: None,
                    });
                }
            }

            char_idx += word_len + 1; // +1 for space
        }
    }

    // End final entity if any
    if let Some((entity_type, text, start)) = current_entity {
        annotations.push(ImportedAnnotation {
            text,
            entity_type,
            start,
            end: char_idx,
            source: input.to_string_lossy().to_string(),
            confidence: None,
        });
    }

    Ok(annotations)
}

/// Import from JSONL format
fn import_jsonl(input: &PathBuf) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let mut annotations = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("Invalid JSON: {}", e))?;

        annotations.push(ImportedAnnotation {
            text: obj["text"].as_str().unwrap_or("").to_string(),
            entity_type: obj["type"].as_str().unwrap_or("").to_string(),
            start: obj["start"].as_u64().unwrap_or(0) as usize,
            end: obj["end"].as_u64().unwrap_or(0) as usize,
            source: obj["source"]
                .as_str()
                .unwrap_or(&input.to_string_lossy())
                .to_string(),
            confidence: obj["confidence"].as_f64(),
        });
    }

    Ok(annotations)
}
