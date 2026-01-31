//! Output formatting utilities for CLI commands

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};

use anno_core::Entity;
use anno_core::{GroundedDocument, Location, Signal};

#[cfg(feature = "eval")]
use anno_core::core::grounded::{EvalComparison, EvalMatch};

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

/// Print document extraction results with hierarchical verbose levels.
///
/// Verbosity levels follow CLI best practices (inspired by `iw -vvv`):
/// - **Level 0 (default)**: Dense, expert-friendly - entity counts and spans only
/// - **Level 1 (-v)**: Add confidence scores and context snippets
/// - **Level 2 (-vv)**: Add tracks (within-doc coreference), basic statistics
/// - **Level 3 (-vvv)**: Add identities (KB links), full metadata, timing, annotated text
///
/// Each level is a strict superset: higher levels include all information from lower levels.
pub fn print_signals(doc: &GroundedDocument, text: &str, verbose_level: u8) {
    let mut by_type: HashMap<String, Vec<&Signal<Location>>> = HashMap::new();
    for s in doc.signals() {
        by_type.entry(s.label().to_string()).or_default().push(s);
    }

    let text_len = text.chars().count();

    if by_type.is_empty() {
        if verbose_level == 0 {
            println!(
                "(no entities found - try -v for debugging or --model gliner for zero-shot NER)"
            );
        } else {
            println!("(no entities found)");
        }
        return;
    }

    // Deterministic ordering:
    // - Groups are ordered by their first occurrence in the text (min start offset).
    // - Within each group, entities are ordered by (start, end, surface).
    //
    // This avoids HashMap iteration non-determinism and makes docs/tests reproducible.
    #[derive(Debug)]
    struct TypeGroup<'a> {
        label: String,
        signals: Vec<&'a Signal<Location>>,
        min_start: usize,
    }

    let mut groups: Vec<TypeGroup<'_>> = by_type
        .into_iter()
        .map(|(label, mut signals)| {
            signals.sort_by(|a, b| {
                let (a_start, a_end) = a.text_offsets().unwrap_or((usize::MAX, usize::MAX));
                let (b_start, b_end) = b.text_offsets().unwrap_or((usize::MAX, usize::MAX));
                a_start
                    .cmp(&b_start)
                    .then_with(|| a_end.cmp(&b_end))
                    .then_with(|| a.surface().cmp(b.surface()))
            });

            let min_start = signals
                .iter()
                .filter_map(|s| s.text_offsets().map(|(start, _)| start))
                .min()
                .unwrap_or(usize::MAX);

            TypeGroup {
                label,
                signals,
                min_start,
            }
        })
        .collect();

    groups.sort_by(|a, b| {
        a.min_start
            .cmp(&b.min_start)
            .then_with(|| a.label.cmp(&b.label))
    });

    // Level 0: Entity-focused (no spans - they're implementation details)
    if verbose_level == 0 {
        for g in &groups {
            let col = type_color(&g.label);
            let entities: Vec<String> = g
                .signals
                .iter()
                .map(|s| format!("\"{}\"", s.surface()))
                .collect();
            println!(
                "{}:{} {}",
                color(col, &g.label),
                g.signals.len(),
                entities.join(" ")
            );
        }
        return;
    }

    // Level 1+: More detailed output
    for g in &groups {
        let col = type_color(&g.label);
        println!("{}:{}", color(col, &g.label), g.signals.len());
        for s in &g.signals {
            let (start, end) = s.text_offsets().unwrap_or((0, 0));

            // Level 1: Entity text with confidence (no spans)
            let conf_str = format!("({:.2})", s.confidence);
            let neg = if s.negated {
                color("31", " [NEG]")
            } else {
                String::new()
            };
            let quant = s
                .quantifier
                .map(|q| color("35", &format!(" [{:?}]", q)))
                .unwrap_or_default();

            print!("  \"{}\" {}", s.surface(), color("90", &conf_str));
            if !neg.is_empty() || !quant.is_empty() {
                print!("{}{}", neg, quant);
            }
            println!();

            // Level 1+: Context snippets (shows surrounding text)
            // Use 30 chars for better context (was 15, too short)
            let ctx_start = start.saturating_sub(30);
            let ctx_end = (end + 30).min(text_len);
            let before: String = text
                .chars()
                .skip(ctx_start)
                .take(start.saturating_sub(ctx_start))
                .collect();
            let entity: String = text.chars().skip(start).take(end - start).collect();
            let after: String = text.chars().skip(end).take(ctx_end - end).collect();
            println!(
                "    {}{}{}{}{}",
                if ctx_start > 0 {
                    color("90", "...")
                } else {
                    String::new()
                },
                color("90", &before),
                color("1;33", &entity),
                color("90", &after),
                if ctx_end < text_len {
                    color("90", "...")
                } else {
                    String::new()
                }
            );
        }
    }

    // Level 2+: Tracks (within-document coreference chains)
    let tracks: Vec<_> = doc.tracks().collect();
    if verbose_level >= 2 && !tracks.is_empty() {
        println!();
        println!("{}:", color("1;36", "Coreference"));
        for track in &tracks {
            let track_type = track.entity_type.as_deref().unwrap_or("-");
            // Show entity text, not signal IDs (more useful for humans)
            let mentions: Vec<String> = track
                .signals
                .iter()
                .filter_map(|s| doc.get_signal(s.signal_id))
                .map(|sig| format!("\"{}\"", sig.surface()))
                .collect();
            let identity_link = track
                .identity_id
                .map(|id| format!(" -> I{}", id))
                .unwrap_or_default();
            let _cluster_conf = if verbose_level >= 3 {
                format!(" (conf:{:.2})", track.cluster_confidence)
            } else {
                String::new()
            };
            // Only show tracks with multiple mentions (actual coreference)
            // Single mentions are not interesting - they're just the entity itself
            if mentions.len() > 1 {
                println!(
                    "  \"{}\" [{}] → {}",
                    track.canonical_surface,
                    track_type,
                    mentions.join(" ")
                );
            }
            if !identity_link.is_empty() {
                println!("    {}", identity_link);
            }
        }
    }

    // Level 2+: Basic statistics
    if verbose_level >= 2 {
        let stats = doc.stats();
        println!();
        println!(
            "{}: {} entities, {} tracks, {} identities, avg confidence {:.2}",
            color("90", "stats"),
            stats.signal_count,
            stats.track_count,
            stats.identity_count,
            stats.avg_confidence
        );
    }

    // Level 3+: Identities (KB-linked entities), full metadata, annotated text
    if verbose_level >= 3 {
        let identities: Vec<_> = doc.identities().collect();
        if !identities.is_empty() {
            println!();
            println!("{}:", color("1;35", "Identities"));
            for identity in &identities {
                let kb_info =
                    if let (Some(kb_name), Some(kb_id)) = (&identity.kb_name, &identity.kb_id) {
                        format!(" [{}/{}]", kb_name, kb_id)
                    } else {
                        String::new()
                    };
                let aliases = if !identity.aliases.is_empty() {
                    format!(" aliases: {}", identity.aliases.join(", "))
                } else {
                    String::new()
                };
                let desc = identity
                    .description
                    .as_deref()
                    .map(|d| format!(" desc: \"{}\"", d))
                    .unwrap_or_default();
                println!(
                    "  I{}: \"{}\"{}{}{}",
                    identity.id, identity.canonical_name, kb_info, aliases, desc
                );
            }
        }

        // Annotated text (full document with entity highlights)
        println!();
        println!("{}:", color("1;37", "Annotated text"));
        print_annotated_signals(text, doc.signals());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_bar_normal() {
        // Normal cases - function returns a visual bar
        let bar = confidence_bar(0.5);
        assert!(bar.contains('#')); // 50% should have some filled chars
        assert!(bar.contains('.')); // and some empty chars

        let bar = confidence_bar(1.0);
        assert!(bar.contains('#')); // 100% should be fully filled

        let bar = confidence_bar(0.0);
        assert!(bar.contains('.')); // 0% should be mostly empty
    }

    #[test]
    fn test_confidence_bar_clamping() {
        // Edge case: confidence slightly over 1.0 should not panic
        let bar = confidence_bar(1.01);
        assert!(bar.contains('#')); // Should have filled chars

        // Edge case: confidence at exactly 1.0
        let bar = confidence_bar(1.0);
        assert!(bar.contains('#')); // 100% should be fully filled
    }

    #[test]
    fn test_type_color() {
        assert_eq!(type_color("PER"), "1;34");
        assert_eq!(type_color("person"), "1;34");
        assert_eq!(type_color("ORG"), "1;32");
        assert_eq!(type_color("LOC"), "1;33");
        assert_eq!(type_color("UNKNOWN"), "1;37");
    }

    #[test]
    fn test_metric_colored() {
        // High score (>= 90)
        let result = metric_colored(95.0);
        assert!(result.contains("95.0"));

        // Medium score (>= 70)
        let result = metric_colored(75.0);
        assert!(result.contains("75.0"));

        // Low score (< 50)
        let result = metric_colored(30.0);
        assert!(result.contains("30.0"));
    }

    #[test]
    fn test_color_function() {
        // When not in a terminal, color() should return plain text
        let result = color("32", "test");
        assert!(result.contains("test"));
    }
}
