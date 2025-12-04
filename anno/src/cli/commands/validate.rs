//! Validate command - Validate JSONL annotation files

use super::super::output::color;
use crate::grounded::{GroundedDocument, Location, Signal, SignalValidationError}; // Re-exported from anno-core
use clap::Parser;
use std::fs;

/// Validate JSONL annotation files
#[derive(Parser, Debug)]
pub struct ValidateArgs {
    /// JSONL files to validate
    #[arg(required = true)]
    pub files: Vec<String>,
}

pub fn run(args: ValidateArgs) -> Result<(), String> {
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
