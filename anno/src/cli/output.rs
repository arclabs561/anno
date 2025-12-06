//! Output formatting utilities for CLI commands

use is_terminal::IsTerminal;
use std::collections::HashMap;
use std::io::{self, Write};

use anno_core::Entity;
use anno_core::{GroundedDocument, Location, Signal};

#[cfg(feature = "eval")]
use crate::grounded::{EvalComparison, EvalMatch}; // Re-exported from anno-core

/// Log info message (respects quiet flag)
pub fn log_info(msg: &str, quiet: bool) {
    if !quiet {
        eprintln!("{}", msg);
    }
}

/// Write output to file or stdout
pub fn write_output(content: &str, path: Option<&str>) -> Result<(), String> {
    if let Some(path) = path {
        std::fs::write(path, content).map_err(|e| format!("Failed to write to {}: {}", path, e))?;
    } else {
        print!("{}", content);
        io::stdout()
            .flush()
            .map_err(|e| format!("Failed to flush stdout: {}", e))?;
    }
    Ok(())
}

/// Format file size in human-readable format
pub fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// Colorize text with ANSI escape codes (only if stdout is a terminal)
pub fn color(code: &str, text: &str) -> String {
    if io::stdout().is_terminal() {
        format!("\x1b[{}m{}\x1b[0m", code, text)
    } else {
        text.to_string()
    }
}

/// Get color code for entity type
pub fn type_color(typ: &str) -> &'static str {
    match typ.to_lowercase().as_str() {
        "person" | "per" => "1;34",
        "organization" | "org" => "1;32",
        "location" | "loc" | "gpe" => "1;33",
        "date" | "time" => "1;35",
        "money" | "percent" => "1;36",
        "email" | "url" | "phone" => "36",
        _ => "1;37",
    }
}

/// Format metric value with color based on threshold
pub fn metric_colored(value: f64) -> String {
    let code = if value >= 90.0 {
        "1;32"
    } else if value >= 70.0 {
        "1;33"
    } else if value >= 50.0 {
        "33"
    } else {
        "1;31"
    };
    color(code, &format!("{:5.1}", value))
}

/// Create confidence bar visualization
pub fn confidence_bar(conf: f32) -> String {
    // Clamp to valid range to prevent underflow if conf > 1.0
    let filled = ((conf * 10.0).round() as usize).min(10);
    let empty = 10 - filled;
    let code = if conf >= 0.9 {
        "32"
    } else if conf >= 0.7 {
        "33"
    } else {
        "31"
    };
    format!(
        "{}{}",
        color(code, &"#".repeat(filled)),
        color("90", &".".repeat(empty))
    )
}

/// Print signals grouped by type
pub fn print_signals(doc: &GroundedDocument, text: &str, verbose: bool) {
    let mut by_type: HashMap<String, Vec<&Signal<Location>>> = HashMap::new();
    for s in doc.signals() {
        by_type.entry(s.label().to_string()).or_default().push(s);
    }

    for (typ, signals) in &by_type {
        let col = type_color(typ);
        println!("  {} ({}):", color(col, typ), signals.len());
        for s in signals {
            let (start, end) = s.text_offsets().unwrap_or((0, 0));
            let neg = if s.negated {
                color("31", " [NEG]")
            } else {
                String::new()
            };
            let quant = s
                .quantifier
                .map(|q| color("35", &format!(" [{:?}]", q)))
                .unwrap_or_default();

            // Show confidence bar only (percentage is redundant)
            println!(
                "    [{:3},{:3}) {} \"{}\"{}{}",
                start,
                end,
                confidence_bar(s.confidence),
                s.surface(),
                neg,
                quant
            );

            if verbose {
                let ctx_start = start.saturating_sub(15);
                let ctx_end = (end + 15).min(text.chars().count());
                let before: String = text
                    .chars()
                    .skip(ctx_start)
                    .take(start - ctx_start)
                    .collect();
                let entity: String = text.chars().skip(start).take(end - start).collect();
                let after: String = text.chars().skip(end).take(ctx_end - end).collect();
                println!(
                    "           {}{}{}{}{}",
                    color("90", "..."),
                    color("90", &before),
                    color("1;33", &entity),
                    color("90", &after),
                    color("90", "...")
                );
            }
        }
    }
}

/// Print annotated entities inline with text
pub fn print_annotated_entities(text: &str, entities: &[Entity]) {
    let mut sorted: Vec<&Entity> = entities.iter().collect();
    sorted.sort_by_key(|e| e.start);

    let chars: Vec<char> = text.chars().collect();
    let char_len = chars.len();
    let mut result = String::new();
    let mut last_end = 0;

    for e in sorted {
        if e.start >= char_len || e.end > char_len || e.start >= e.end {
            continue;
        }
        if e.start < last_end {
            continue;
        }

        if e.start > last_end {
            let before: String = chars[last_end..e.start].iter().collect();
            result.push_str(&before);
        }

        let col = type_color(e.entity_type.as_label());
        let entity_text: String = chars[e.start..e.end].iter().collect();
        result.push_str(&color(
            col,
            &format!("[{}: {}]", e.entity_type.as_label(), entity_text),
        ));
        last_end = e.end;
    }

    if last_end < char_len {
        let after: String = chars[last_end..].iter().collect();
        result.push_str(&after);
    }

    println!();
    for line in result.lines() {
        println!("  {}", line);
    }
}

/// Print annotated signals inline with text
pub fn print_annotated_signals(text: &str, signals: &[Signal<Location>]) {
    let mut sorted: Vec<&Signal<Location>> = signals.iter().collect();
    sorted.sort_by_key(|s| s.text_offsets().map(|(start, _)| start).unwrap_or(0));

    let chars: Vec<char> = text.chars().collect();
    let char_len = chars.len();
    let mut result = String::new();
    let mut last_end = 0;

    for s in sorted {
        let (start, end) = match s.text_offsets() {
            Some((start, end)) => (start, end),
            None => continue,
        };

        if start >= char_len || end > char_len || start >= end {
            continue;
        }
        if start < last_end {
            continue;
        }

        if start > last_end {
            let before: String = chars[last_end..start].iter().collect();
            result.push_str(&before);
        }

        let col = type_color(s.label());
        let entity_text: String = chars[start..end].iter().collect();
        result.push_str(&color(col, &format!("[{}: {}]", s.label(), entity_text)));
        last_end = end;
    }

    if last_end < char_len {
        let after: String = chars[last_end..].iter().collect();
        result.push_str(&after);
    }

    println!();
    for line in result.lines() {
        println!("  {}", line);
    }
}

/// Print evaluation matches with color coding
#[cfg(feature = "eval")]
pub fn print_matches(cmp: &EvalComparison, _verbose: bool) {
    for m in &cmp.matches {
        match m {
            EvalMatch::Correct { gold_id, .. } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("32", "+"),
                    color("32", "correct"),
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
                    "  {} {}: \"{}\" ({} -> {})",
                    color("33", "!"),
                    color("33", "type mismatch"),
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
                println!(
                    "  {} {}: gold=\"{}\" pred=\"{}\" (IoU={:.2})",
                    color("33", "!"),
                    color("33", "boundary"),
                    g.map(|s| s.surface()).unwrap_or("?"),
                    p.map(|s| s.surface()).unwrap_or("?"),
                    iou
                );
            }
            EvalMatch::Spurious { pred_id } => {
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("31", "x"),
                    color("31", "false positive"),
                    p.map(|s| s.label.as_str()).unwrap_or("?"),
                    p.map(|s| s.surface()).unwrap_or("?")
                );
            }
            EvalMatch::Missed { gold_id } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                println!(
                    "  {} {}: [{}] \"{}\"",
                    color("31", "x"),
                    color("31", "false negative"),
                    g.map(|s| s.label.as_str()).unwrap_or("?"),
                    g.map(|s| s.surface()).unwrap_or("?")
                );
            }
        }
    }
}
