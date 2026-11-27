//! Render measured extraction results as standalone SVG or self-contained HTML.
//!
//! Use [`overlap`] to measure shared characters in grounded documents, [`errors`]
//! to examine entity evaluation outcomes, and [`Tradeoff`] to compare operating
//! points across cohorts. Each view can render an offline report from a saved input.
//!
//! ```
//! use anno::viz::Tradeoff;
//!
//! let study = br#"{"schema":"anno.tradeoff/v1","title":"Threshold study",
//!     "unit":"sentences","series":[{"label":"Dev","observations":10,"gold":8,
//!     "points":[{"label":"Stock","correct":4,"unmatched":1}]}]}"#;
//! let svg = Tradeoff::from_json(study.as_slice())?.render_svg()?;
//! assert!(svg.contains("50.0%"));
//! # Ok::<(), String>(())
//! ```

pub mod errors;
pub mod overlap;
mod tradeoff;

pub use tradeoff::{OperatingPoint, Tradeoff, TradeoffSeries, TRADEOFF_SCHEMA};

use crate::core::grounded::html_escape as escape;
use std::fmt::Write;
use unicode_width::UnicodeWidthChar;

fn check_label(value: &str, what: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{what} must contain a label"));
    }
    if value.chars().any(|c| {
        !matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..='\u{10FFFF}')
    }) {
        return Err(format!("{what} contains a character XML cannot represent"));
    }
    Ok(())
}

fn html_document(title: &str, svg: &str, table: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{}</title>
<style>body{{max-width:76rem;margin:1rem auto;padding:0 1rem;color:#183541;background:white;font:1rem/1.45 "DejaVu Sans",sans-serif}}svg{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:1.5rem 0;font-variant-numeric:tabular-nums}}caption{{text-align:left;font-size:1.25rem;font-weight:700;margin-bottom:.5rem}}td,th{{padding:.45rem .6rem;text-align:left;vertical-align:top;border-bottom:1px solid #d5e0e2}}thead th{{color:#456371}}.scroll{{overflow-x:auto}}</style>
</head><body><main>{svg}<div class="scroll">{table}</div></main></body></html>"#,
        escape(title)
    )
}

/// Group digits without changing integer precision.
fn integer(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::new();
    for (i, c) in digits.bytes().enumerate() {
        if i != 0 && (digits.len() - i).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(char::from(c));
    }
    grouped
}

/// Wrap at word boundaries, splitting an overlong word at scalar boundaries.
/// Approximate display cells for proportional DejaVu Sans. Wide uppercase
/// letters get extra room and combining marks stay with their base character.
fn wrap(value: &str, columns: usize) -> Vec<String> {
    // Work in quarter cells to accommodate the wider advances of the bold
    // headings and full-width glyphs without wrapping ordinary titles early.
    let capacity = columns * 4;
    let cells = |c: char| match c.width() {
        None | Some(0) => 0,
        Some(2) => 8,
        _ => match c {
            'M' | 'W' | 'm' | '%' | '@' => 8,
            'w' => 7,
            'i' | 'j' | 'l' | 'I' | '!' | '.' | ',' | ':' | ';' | '|' => 3,
            'A'..='Z' => 6,
            c if c.is_ascii() => 5,
            _ => 8,
        },
    };
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut used = 0;
    for word in value.split_whitespace() {
        let length: usize = word.chars().map(cells).sum();
        if used != 0 {
            if used + 2 + length > capacity {
                lines.push(std::mem::take(&mut line));
                used = 0;
            } else {
                line.push(' ');
                used += 2;
            }
        }
        for c in word.chars() {
            let width = cells(c);
            if used != 0 && used + width > capacity {
                lines.push(std::mem::take(&mut line));
                used = 0;
            }
            line.push(c);
            used += width;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

fn lines(svg: &mut String, lines: &[String], x: i32, y: i32, leading: i32, class: &str) {
    for (row, line) in lines.iter().enumerate() {
        let y = y + row as i32 * leading;
        writeln!(
            svg,
            r#"<text x="{x}" y="{y}" class="{class}">{}</text>"#,
            escape(line)
        )
        .unwrap();
    }
}
