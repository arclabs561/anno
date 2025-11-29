//! NER Debug Tool
//!
//! Interactive debugging for NER evaluation and visualization.
//!
//! # Usage
//!
//! ```bash
//! # Debug arbitrary text with real model
//! cargo run --example debug_ner -- --text "Marie Curie won the Nobel Prize."
//!
//! # Compare against gold annotations  
//! cargo run --example debug_ner -- --text "Marie Curie won." --gold "Marie Curie:PER:0:11"
//!
//! # Output HTML report
//! cargo run --example debug_ner -- --text "..." --output debug.html
//!
//! # Verbose mode with confidence breakdown
//! cargo run --example debug_ner -- --text "..." --verbose
//! ```

use anno::grounded::{
    render_document_html, render_eval_html, EvalComparison, EvalMatch, GroundedDocument, Location,
    Signal,
};
use anno::{Model, StackedNER};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut text = String::new();
    let mut gold_specs: Vec<String> = Vec::new();
    let mut output_path = None;
    let mut verbose = false;
    let mut show_help = false;
    let mut labels: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--text" | "-t" => {
                if i + 1 < args.len() {
                    text = args[i + 1].clone();
                    i += 1;
                }
            }
            "--gold" | "-g" => {
                if i + 1 < args.len() {
                    gold_specs.push(args[i + 1].clone());
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--label" | "-l" => {
                if i + 1 < args.len() {
                    labels.push(args[i + 1].clone());
                    i += 1;
                }
            }
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => show_help = true,
            _ => {
                // Treat as text if no flag
                if !args[i].starts_with('-') && text.is_empty() {
                    text = args[i].clone();
                }
            }
        }
        i += 1;
    }

    if show_help || text.is_empty() {
        print_help();
        return;
    }

    // Parse gold annotations: "text:label:start:end"
    let gold: Vec<GoldSpec> = gold_specs
        .iter()
        .filter_map(|s| parse_gold_spec(s))
        .collect();

    println!(
        "\x1b[1;36m╔══════════════════════════════════════════════════════════════════╗\x1b[0m"
    );
    println!("\x1b[1;36m║\x1b[0m  \x1b[1mNER Debug Tool\x1b[0m                                                  \x1b[1;36m║\x1b[0m");
    println!(
        "\x1b[1;36m╚══════════════════════════════════════════════════════════════════╝\x1b[0m"
    );
    println!();

    // Show input
    println!("\x1b[1;33mInput\x1b[0m ({} chars):", text.len());
    if text.len() > 100 {
        println!("  \x1b[90m\"{}\"\x1b[0m...", &text[..100]);
    } else {
        println!("  \x1b[90m\"{}\"\x1b[0m", text);
    }
    println!();

    // Initialize model
    let model = StackedNER::default();
    println!(
        "\x1b[1;33mModel\x1b[0m: {} ({})",
        model.name(),
        model.description()
    );
    println!(
        "  Supported types: {:?}",
        model
            .supported_types()
            .iter()
            .map(|t| t.as_label())
            .collect::<Vec<_>>()
    );
    println!();

    // Extract entities
    let start = std::time::Instant::now();
    let entities = match model.extract_entities(&text, None) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("\x1b[1;31mError:\x1b[0m {}", err);
            return;
        }
    };
    let elapsed = start.elapsed();

    // Filter by labels if specified
    let entities: Vec<_> = if labels.is_empty() {
        entities
    } else {
        entities
            .into_iter()
            .filter(|e| {
                labels
                    .iter()
                    .any(|l| e.entity_type.as_label().eq_ignore_ascii_case(l))
            })
            .collect()
    };

    println!(
        "\x1b[1;32mExtracted {} entities\x1b[0m in {:.2}ms",
        entities.len(),
        elapsed.as_secs_f64() * 1000.0
    );
    println!();

    // Build grounded document
    let mut doc = GroundedDocument::new("debug", &text);
    for e in &entities {
        let signal = Signal::new(
            0,
            Location::text(e.start, e.end),
            &e.text,
            e.entity_type.as_label(),
            e.confidence as f32,
        );
        doc.add_signal(signal);
    }

    // Validate
    let validation_errors = doc.validate();
    if !validation_errors.is_empty() {
        println!("\x1b[1;31m⚠ Validation Errors:\x1b[0m");
        for err in &validation_errors {
            println!("  - {}", err);
        }
        println!();
    }

    // Display entities
    println!("\x1b[1;33mEntities:\x1b[0m");
    if entities.is_empty() {
        println!("  (none found)");
    } else {
        // Group by type
        let mut by_type: std::collections::HashMap<String, Vec<_>> =
            std::collections::HashMap::new();
        for e in &entities {
            by_type
                .entry(e.entity_type.as_label().to_string())
                .or_default()
                .push(e);
        }

        for (typ, ents) in by_type.iter() {
            let color = type_color(typ);
            println!("  \x1b[{}m{}\x1b[0m ({}):", color, typ, ents.len());
            for e in ents {
                let conf_bar = confidence_bar(e.confidence);
                println!(
                    "    [{:3},{:3}) {} \x1b[90m\"{}\"\x1b[0m",
                    e.start, e.end, conf_bar, e.text
                );

                if verbose {
                    // Show context
                    let ctx_start = e.start.saturating_sub(15);
                    let ctx_end = (e.end + 15).min(text.len());
                    let before: String = text
                        .chars()
                        .skip(ctx_start)
                        .take(e.start - ctx_start)
                        .collect();
                    let entity: String = text.chars().skip(e.start).take(e.end - e.start).collect();
                    let after: String = text.chars().skip(e.end).take(ctx_end - e.end).collect();
                    println!(
                        "           \x1b[90m...{}\x1b[1;33m{}\x1b[0m\x1b[90m{}...\x1b[0m",
                        before, entity, after
                    );
                }
            }
        }
    }
    println!();

    // If gold annotations provided, run comparison
    if !gold.is_empty() {
        run_comparison(&text, &entities, &gold, verbose);
    }

    // Show annotated text
    println!("\x1b[1;33mAnnotated Text:\x1b[0m");
    print_annotated_text(&text, &entities);
    println!();

    // Output HTML if requested
    if let Some(path) = output_path {
        let html = if gold.is_empty() {
            render_document_html(&doc)
        } else {
            let gold_signals = gold_to_signals(&gold);
            let pred_signals = entities_to_signals(&entities);
            let cmp = EvalComparison::compare(&text, gold_signals, pred_signals);
            render_eval_html(&cmp)
        };

        fs::write(&path, &html).expect("Failed to write HTML");
        println!("\x1b[1;32m✓\x1b[0m HTML written to: \x1b[4m{}\x1b[0m", path);
    }
}

fn print_help() {
    eprintln!("\x1b[1mNER Debug Tool\x1b[0m - Visualize and debug entity extraction");
    eprintln!();
    eprintln!("\x1b[1;33mUSAGE:\x1b[0m");
    eprintln!("  debug_ner --text \"Your text here\"");
    eprintln!("  debug_ner \"Your text here\"           # text as positional arg");
    eprintln!();
    eprintln!("\x1b[1;33mOPTIONS:\x1b[0m");
    eprintln!("  -t, --text TEXT       Input text to process");
    eprintln!("  -g, --gold SPEC       Gold annotation: \"text:label:start:end\" (repeatable)");
    eprintln!("  -l, --label TYPE      Filter to specific entity type (repeatable)");
    eprintln!("  -o, --output PATH     Output HTML report to file");
    eprintln!("  -v, --verbose         Show detailed context and confidence");
    eprintln!("  -h, --help            Show this help");
    eprintln!();
    eprintln!("\x1b[1;33mEXAMPLES:\x1b[0m");
    eprintln!("  \x1b[90m# Extract all entities\x1b[0m");
    eprintln!("  debug_ner \"Marie Curie won the Nobel Prize in 1903.\"");
    eprintln!();
    eprintln!("  \x1b[90m# Filter to persons only\x1b[0m");
    eprintln!("  debug_ner \"Marie Curie worked with Pierre.\" -l Person");
    eprintln!();
    eprintln!("  \x1b[90m# Compare against gold annotations\x1b[0m");
    eprintln!("  debug_ner -t \"Marie Curie won the Nobel Prize.\" \\");
    eprintln!("            -g \"Marie Curie:Person:0:11\" \\");
    eprintln!("            -g \"Nobel Prize:Misc:20:31\"");
    eprintln!();
    eprintln!("  \x1b[90m# Generate HTML report\x1b[0m");
    eprintln!("  debug_ner -t \"...\" -o report.html -v");
}

#[derive(Debug, Clone)]
struct GoldSpec {
    text: String,
    label: String,
    start: usize,
    end: usize,
}

fn parse_gold_spec(s: &str) -> Option<GoldSpec> {
    // Format: "text:label:start:end"
    let parts: Vec<&str> = s.rsplitn(3, ':').collect();
    if parts.len() < 3 {
        eprintln!(
            "\x1b[33mWarning:\x1b[0m Invalid gold spec '{}' (expected 'text:label:start:end')",
            s
        );
        return None;
    }

    let end: usize = parts[0].parse().ok()?;
    let start: usize = parts[1].parse().ok()?;
    let rest: Vec<&str> = parts[2].rsplitn(2, ':').collect();

    if rest.len() < 2 {
        eprintln!(
            "\x1b[33mWarning:\x1b[0m Invalid gold spec '{}' (expected 'text:label:start:end')",
            s
        );
        return None;
    }

    let label = rest[0].to_string();
    let text = rest[1].to_string();

    Some(GoldSpec {
        text,
        label,
        start,
        end,
    })
}

fn run_comparison(text: &str, entities: &[anno::Entity], gold: &[GoldSpec], verbose: bool) {
    let gold_signals = gold_to_signals(gold);
    let pred_signals = entities_to_signals(entities);

    let cmp = EvalComparison::compare(text, gold_signals, pred_signals);

    println!(
        "\x1b[1;36m═══════════════════════════════════════════════════════════════════\x1b[0m"
    );
    println!("\x1b[1;36m  EVALUATION\x1b[0m");
    println!(
        "\x1b[1;36m═══════════════════════════════════════════════════════════════════\x1b[0m"
    );
    println!();

    // Metrics
    let p = cmp.precision() * 100.0;
    let r = cmp.recall() * 100.0;
    let f1 = cmp.f1() * 100.0;

    println!("  \x1b[1mGold:\x1b[0m {}  \x1b[1mPredicted:\x1b[0m {}  \x1b[1mCorrect:\x1b[0m {}  \x1b[1mErrors:\x1b[0m {}",
             cmp.gold.len(), cmp.predicted.len(), cmp.correct_count(), cmp.error_count());
    println!();

    // Color-coded metrics
    let p_color = metric_color(p);
    let r_color = metric_color(r);
    let f1_color = metric_color(f1);

    println!(
        "  \x1b[1mPrecision:\x1b[0m \x1b[{}m{:5.1}%\x1b[0m",
        p_color, p
    );
    println!(
        "  \x1b[1mRecall:\x1b[0m    \x1b[{}m{:5.1}%\x1b[0m",
        r_color, r
    );
    println!(
        "  \x1b[1mF1:\x1b[0m        \x1b[{}m{:5.1}%\x1b[0m",
        f1_color, f1
    );
    println!();

    // Match details
    println!("\x1b[1;33mMatch Details:\x1b[0m");
    for m in &cmp.matches {
        match m {
            EvalMatch::Correct { gold_id, .. } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  \x1b[32m✓\x1b[0m \x1b[32mCorrect\x1b[0m: [{}] \"{}\"",
                    g.map(|s| s.label.as_str()).unwrap_or("?"),
                    g.map(|s| s.surface()).unwrap_or("?")
                );
            }
            EvalMatch::TypeMismatch {
                gold_id,
                gold_label,
                pred_label,
                ..
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  \x1b[33m⚠\x1b[0m \x1b[33mType mismatch\x1b[0m: \"{}\" ({} → {})",
                    g.map(|s| s.surface()).unwrap_or("?"),
                    gold_label,
                    pred_label
                );
            }
            EvalMatch::BoundaryError {
                gold_id,
                pred_id,
                iou,
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                println!("  \x1b[33m⚠\x1b[0m \x1b[33mBoundary error\x1b[0m: gold=\"{}\" pred=\"{}\" (IoU={:.2})",
                         g.map(|s| s.surface()).unwrap_or("?"),
                         p.map(|s| s.surface()).unwrap_or("?"),
                         iou);

                if verbose {
                    if let (Some(g), Some(p)) = (g, p) {
                        if let (Some((gs, ge)), Some((ps, pe))) =
                            (g.text_offsets(), p.text_offsets())
                        {
                            println!("       gold=[{},{}) pred=[{},{})", gs, ge, ps, pe);
                        }
                    }
                }
            }
            EvalMatch::Spurious { pred_id } => {
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                println!(
                    "  \x1b[31m✗\x1b[0m \x1b[31mFalse positive\x1b[0m: [{}] \"{}\"",
                    p.map(|s| s.label.as_str()).unwrap_or("?"),
                    p.map(|s| s.surface()).unwrap_or("?")
                );
            }
            EvalMatch::Missed { gold_id } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  \x1b[31m✗\x1b[0m \x1b[31mFalse negative\x1b[0m: [{}] \"{}\"",
                    g.map(|s| s.label.as_str()).unwrap_or("?"),
                    g.map(|s| s.surface()).unwrap_or("?")
                );
            }
        }
    }
    println!();
}

fn gold_to_signals(gold: &[GoldSpec]) -> Vec<Signal<Location>> {
    gold.iter()
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
        .collect()
}

fn entities_to_signals(entities: &[anno::Entity]) -> Vec<Signal<Location>> {
    entities
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
        .collect()
}

fn print_annotated_text(text: &str, entities: &[anno::Entity]) {
    // Sort by start position
    let mut sorted: Vec<_> = entities.iter().collect();
    sorted.sort_by_key(|e| e.start);

    let mut result = String::new();
    let mut last_end = 0;
    let chars: Vec<char> = text.chars().collect();

    for e in sorted {
        // Add text before entity
        if e.start > last_end {
            let before: String = chars[last_end..e.start].iter().collect();
            result.push_str(&before);
        }

        // Skip overlapping
        if e.start < last_end {
            continue;
        }

        // Add highlighted entity
        let color = type_color(e.entity_type.as_label());
        let entity_text: String = chars[e.start..e.end].iter().collect();
        result.push_str(&format!(
            "\x1b[{}m[{}: {}]\x1b[0m",
            color,
            e.entity_type.as_label(),
            entity_text
        ));
        last_end = e.end;
    }

    // Add remaining text
    if last_end < chars.len() {
        let after: String = chars[last_end..].iter().collect();
        result.push_str(&after);
    }

    // Wrap long lines
    for line in result.lines() {
        if line.len() > 100 {
            // Simple wrapping
            let mut pos = 0;
            while pos < line.len() {
                let end = (pos + 100).min(line.len());
                println!("  {}", &line[pos..end]);
                pos = end;
            }
        } else {
            println!("  {}", line);
        }
    }
}

fn type_color(typ: &str) -> &'static str {
    match typ.to_lowercase().as_str() {
        "person" | "per" => "1;34",           // Bold blue
        "organization" | "org" => "1;32",     // Bold green
        "location" | "loc" | "gpe" => "1;33", // Bold yellow
        "date" | "time" => "1;35",            // Bold magenta
        "money" | "percent" => "1;36",        // Bold cyan
        "email" | "url" | "phone" => "36",    // Cyan
        _ => "1;37",                          // Bold white
    }
}

fn metric_color(value: f64) -> &'static str {
    if value >= 90.0 {
        "1;32" // Bold green
    } else if value >= 70.0 {
        "1;33" // Bold yellow
    } else if value >= 50.0 {
        "33" // Yellow
    } else {
        "1;31" // Bold red
    }
}

fn confidence_bar(conf: f64) -> String {
    let filled = (conf * 10.0).round() as usize;
    let empty = 10 - filled;

    let color = if conf >= 0.9 {
        "32" // Green
    } else if conf >= 0.7 {
        "33" // Yellow
    } else {
        "31" // Red
    };

    format!(
        "\x1b[{}m{}\x1b[90m{}\x1b[0m {:.0}%",
        color,
        "█".repeat(filled),
        "░".repeat(empty),
        conf * 100.0
    )
}
