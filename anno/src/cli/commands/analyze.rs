//! Analyze command - Deep analysis with multiple models

use super::super::output::{color, print_annotated_entities};
use super::super::parser::ModelBackend;
use super::super::utils::get_input_text;
use anno_core::Entity;
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
    #[arg(trailing_var_arg = true)]
    pub positional: Vec<String>,
}

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

    let backends = [
        ModelBackend::Pattern,
        ModelBackend::Heuristic,
        ModelBackend::Stacked,
    ];

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

    // Find disagreements
    println!("{}:", color("1;33", "Model Agreement"));

    let stacked = all_results.get("stacked").cloned().unwrap_or_default();
    let pattern = all_results.get("pattern").cloned().unwrap_or_default();
    let heuristic = all_results.get("heuristic").cloned().unwrap_or_default();

    let mut all_found: Vec<&Entity> = Vec::new();
    let mut only_stacked: Vec<&Entity> = Vec::new();

    for e in &stacked {
        let in_pattern = pattern.iter().any(|p| p.start == e.start && p.end == e.end);
        let in_heuristic = heuristic
            .iter()
            .any(|s| s.start == e.start && s.end == e.end);

        if in_pattern || in_heuristic {
            all_found.push(e);
        } else {
            only_stacked.push(e);
        }
    }

    // Count entities unique to each model
    let pattern_only_count = pattern
        .iter()
        .filter(|p| !stacked.iter().any(|s| s.start == p.start && s.end == p.end))
        .count();
    let heuristic_only_count = heuristic
        .iter()
        .filter(|h| !stacked.iter().any(|s| s.start == h.start && s.end == h.end))
        .count();

    println!(
        "  Agreed (in stacked from pattern/heuristic): {} entities",
        all_found.len()
    );
    println!(
        "  Pattern-only (not in stacked): {} entities",
        pattern_only_count
    );
    println!(
        "  Heuristic-only (not in stacked): {} entities",
        heuristic_only_count
    );
    println!(
        "  Stacked-only (novel combinations): {} entities",
        only_stacked.len()
    );
    println!();

    // Show annotated text
    println!("{}:", color("1;33", "Annotated Text"));
    print_annotated_entities(&text, &stacked);
    println!();

    Ok(())
}
