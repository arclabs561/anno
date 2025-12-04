//! Eval command - Evaluate predictions against gold annotations

use clap::Parser;
use std::fs;
use std::time::Instant;

#[cfg(feature = "eval")]
use super::super::output::print_matches;
use super::super::output::{color, metric_colored};
use super::super::parser::ModelBackend;
use super::super::utils::{get_input_text, load_gold_from_file, parse_gold_spec};

use crate::grounded::{
    render_eval_html, EvalComparison, EvalMatch, Location, Signal,
};

/// Evaluate predictions against gold annotations
#[derive(Parser, Debug)]
pub struct EvalArgs {
    /// Input text to process
    #[arg(short, long)]
    pub text: Option<String>,

    /// Read input from file
    #[arg(short, long, value_name = "PATH")]
    pub file: Option<String>,

    /// Model backend to use
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Gold annotation: "text:label:start:end" (repeatable)
    #[arg(short, long = "gold", value_name = "SPEC")]
    pub gold_specs: Vec<String>,

    /// Load gold annotations from JSONL file
    #[arg(long, value_name = "PATH")]
    pub gold_file: Option<String>,

    /// Write HTML report to file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,

    /// Output format (overrides default text output)
    #[arg(long)]
    pub json: bool,

    /// Output format (overrides default text output)
    #[arg(long)]
    pub html: bool,

    /// Show detailed match information and statistics
    #[arg(short, long)]
    pub verbose: bool,

    /// Minimal output (suppress warnings and non-essential messages)
    #[arg(short, long)]
    pub quiet: bool,

    /// Positional text argument
    #[arg(trailing_var_arg = true)]
    pub positional: Vec<String>,
}

pub fn run(args: EvalArgs) -> Result<(), String> {
    let text = get_input_text(&args.text, args.file.as_deref(), &args.positional)?;

    // Load gold from file or args
    let gold = if let Some(gold_file) = &args.gold_file {
        load_gold_from_file(gold_file)?
    } else if !args.gold_specs.is_empty() {
        args.gold_specs
            .iter()
            .filter_map(|s| parse_gold_spec(s))
            .collect()
    } else {
        return Err(
            "No gold annotations. Use -g 'text:label:start:end' or --gold-file path.jsonl"
                .to_string(),
        );
    };

    if gold.is_empty() {
        return Err("No valid gold annotations found".to_string());
    }

    let model = args.model.create_model()?;

    let start = Instant::now();
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;
    let elapsed = start.elapsed();

    // Build signals
    let gold_signals: Vec<Signal<Location>> = gold
        .iter()
        .enumerate()
        .map(|(i, g)| {
            Signal::new(
                i as u64,
                Location::text(g.start, g.end),
                &g.text,
                &g.label,
                1.0,
            )
        })
        .collect();

    let pred_signals: Vec<Signal<Location>> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            Signal::new(
                i as u64,
                Location::text(e.start, e.end),
                &e.text,
                e.entity_type.as_label(),
                e.confidence as f32,
            )
        })
        .collect();

    let cmp = EvalComparison::compare(&text, gold_signals, pred_signals);

    // Detailed analysis with eval feature
    #[cfg(feature = "eval")]
    let detailed_analysis = {
        use crate::eval::analysis::ErrorAnalysis;
        use crate::eval::GoldEntity;
        use anno_core::EntityType;

        let gold_entities: Vec<GoldEntity> = gold
            .iter()
            .map(|g| GoldEntity {
                text: g.text.clone(),
                entity_type: EntityType::Other(g.label.clone()),
                original_label: g.label.clone(),
                start: g.start,
                end: g.end,
            })
            .collect();

        Some(ErrorAnalysis::analyze(&text, &entities, &gold_entities))
    };
    #[cfg(not(feature = "eval"))]
    let _detailed_analysis: Option<()> = None;

    // Output
    if args.json {
        let mut output = serde_json::json!({
            "model": args.model.name(),
            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
            "gold_count": cmp.gold.len(),
            "predicted_count": cmp.predicted.len(),
            "correct": cmp.correct_count(),
            "errors": cmp.error_count(),
            "precision": cmp.precision(),
            "recall": cmp.recall(),
            "f1": cmp.f1(),
        });

        let matches: Vec<_> = cmp
            .matches
            .iter()
            .map(|m| match m {
                EvalMatch::Correct { gold_id, pred_id } => serde_json::json!({
                    "type": "correct",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                }),
                EvalMatch::TypeMismatch {
                    gold_id,
                    pred_id,
                    gold_label,
                    pred_label,
                } => serde_json::json!({
                    "type": "type_mismatch",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                }),
                EvalMatch::BoundaryError {
                    gold_id,
                    pred_id,
                    iou,
                } => serde_json::json!({
                    "type": "boundary_error",
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                    "iou": iou,
                }),
                EvalMatch::Spurious { pred_id } => serde_json::json!({
                    "type": "false_positive",
                    "pred_id": pred_id,
                }),
                EvalMatch::Missed { gold_id } => serde_json::json!({
                    "type": "false_negative",
                    "gold_id": gold_id,
                }),
            })
            .collect();
        output["matches"] = serde_json::Value::Array(matches);

        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
    } else if args.html {
        let html = render_eval_html(&cmp);
        if let Some(path) = &args.output {
            fs::write(path, &html).map_err(|e| format!("Write failed: {}", e))?;
            if !args.quiet {
                println!("{} HTML written to: {}", color("32", "ok:"), path);
            }
        } else {
            println!("{}", html);
        }
    } else {
        // Human readable
        println!();
        println!(
            "{}",
            color(
                "1;36",
                "======================================================================="
            )
        );
        println!(
            "  {}  model={}  time={:.1}ms",
            color("1;36", "EVALUATION"),
            args.model.name(),
            elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "  gold={}  pred={}  correct={}  errors={}",
            cmp.gold.len(),
            cmp.predicted.len(),
            cmp.correct_count(),
            cmp.error_count()
        );
        println!(
            "{}",
            color(
                "1;36",
                "======================================================================="
            )
        );
        println!();

        let p = cmp.precision() * 100.0;
        let r = cmp.recall() * 100.0;
        let f1 = cmp.f1() * 100.0;

        println!("  Precision: {}%", metric_colored(p));
        println!("  Recall:    {}%", metric_colored(r));
        println!("  F1:        {}%", metric_colored(f1));
        println!();

        #[cfg(feature = "eval")]
        print_matches(&cmp, args.verbose);

        #[cfg(feature = "eval")]
        if let Some(analysis) = detailed_analysis {
            println!();
            println!("{}:", color("1;33", "Error Breakdown"));
            for (err_type, count) in &analysis.counts {
                println!("  {:?}: {}", err_type, count);
            }
        }

        println!();
    }

    Ok(())
}
