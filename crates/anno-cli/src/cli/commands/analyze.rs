//! Analyze command - Deep analysis with multiple models

use super::super::output::{color, print_annotated_entities};
use super::super::parser::ModelBackend;
use super::super::utils::get_input_text;
use anno::Entity;
use clap::Parser;
use std::collections::HashMap;
use std::time::Instant;

/// Deep analysis with multiple models
#[derive(Parser, Debug)]
pub struct AnalyzeArgs {
    /// Input text to process
    #[arg(short, long)]
    pub text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    pub file: Option<String>,

    /// Positional text argument
    /// Positional text input
    pub positional: Vec<String>,
}

/// Execute the analyze command.
pub fn run(args: AnalyzeArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &args.positional)?;

    println!();
    println!(
        "{}",
        color(
            "1;36",
            "======================================================================="
        )
    );
    println!("  {}", color("1;36", "DEEP ANALYSIS"));
    println!(
        "{}",
        color(
            "1;36",
            "======================================================================="
        )
    );
    println!();

    let mut backends = vec![
        ModelBackend::Pattern,
        ModelBackend::Heuristic,
        ModelBackend::Stacked,
    ];
    #[cfg(feature = "onnx")]
    {
        backends.push(ModelBackend::BertOnnx);
        backends.push(ModelBackend::Nuner);
    }

    let mut all_results: HashMap<String, Vec<Entity>> = HashMap::new();

    for backend in &backends {
        let model = backend.create_model()?;
        let start = Instant::now();
        let entities = model.extract_entities(&text, None).unwrap_or_default();
        let elapsed = start.elapsed();

        println!("{}:", color("1;33", backend.name()));
        println!(
            "  {} entities in {:.1}ms",
            entities.len(),
            elapsed.as_secs_f64() * 1000.0
        );

        if !entities.is_empty() {
            let mut by_type: HashMap<String, usize> = HashMap::new();
            for e in &entities {
                *by_type
                    .entry(e.entity_type.as_label().to_string())
                    .or_default() += 1;
            }
            for (t, c) in &by_type {
                println!("    {}: {}", t, c);
            }
        }
        println!();

        all_results.insert(backend.name().to_string(), entities);
    }

    // Find agreement/disagreement across all backends
    println!("{}:", color("1;33", "Model Agreement"));

    // Use stacked as the reference model for comparison
    let stacked = all_results.get("stacked").cloned().unwrap_or_default();

    // Count how many other backends agree with each stacked entity
    let mut agreed = 0usize;
    let mut stacked_only = 0usize;
    for e in &stacked {
        let confirming = all_results
            .iter()
            .filter(|(name, _)| name.as_str() != "stacked")
            .filter(|(_, entities)| {
                entities
                    .iter()
                    .any(|o| o.start() == e.start() && o.end() == e.end())
            })
            .count();
        if confirming > 0 {
            agreed += 1;
        } else {
            stacked_only += 1;
        }
    }

    // Count entities unique to each non-stacked backend
    for backend in &backends {
        let name = backend.name();
        if name == "stacked" {
            continue;
        }
        let entities = match all_results.get(name) {
            Some(e) => e,
            None => continue,
        };
        let unique_count = entities
            .iter()
            .filter(|e| {
                !stacked
                    .iter()
                    .any(|s| s.start() == e.start() && s.end() == e.end())
            })
            .count();
        if unique_count > 0 {
            println!(
                "  {}-only (not in stacked): {} entities",
                name, unique_count
            );
        }
    }

    println!("  Confirmed by 2+ backends: {} entities", agreed);
    println!("  Stacked-only: {} entities", stacked_only);
    println!();

    // Show annotated text
    println!("{}:", color("1;33", "Annotated Text"));
    print_annotated_entities(&text, &stacked);
    println!();

    Ok(())
}
