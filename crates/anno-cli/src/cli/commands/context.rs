//! Context window export command - export entities with surrounding text for human review
//!
//! Provides surrounding text context for each entity, essential for:
//! - Human review and annotation
//! - Debugging extraction errors
//! - Training data preparation
//! - Quality assurance workflows

use clap::{Parser, ValueEnum};
use std::fs;
use std::path::PathBuf;

use super::super::output::color;
use super::super::parser::ModelBackend;

/// Export entities with context windows
#[derive(Parser, Debug)]
pub struct ContextArgs {
    /// Input file or text
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Input text directly
    #[arg(short, long)]
    pub text: Option<String>,

    /// Model backend
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Context window size (characters before/after)
    #[arg(long, default_value = "50")]
    pub window: usize,

    /// Include full sentence context
    #[arg(long)]
    pub full_sentence: bool,

    /// Output format
    #[arg(long, default_value = "human")]
    pub format: ContextFormat,

    /// Output file (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Filter by entity type
    #[arg(long)]
    pub entity_type: Option<String>,

    /// Quiet mode
    #[arg(short, long)]
    pub quiet: bool,
}

/// Context output format
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum ContextFormat {
    /// Human-readable with highlighting
    #[default]
    Human,
    /// JSON with structured context
    Json,
    /// TSV for spreadsheets
    Tsv,
    /// Markdown for documentation
    Markdown,
    /// brat-style standoff format
    Brat,
}

/// Entity with surrounding context for human review.
#[derive(Debug, Clone)]
pub struct EntityContext {
    /// Entity surface text
    pub text: String,
    /// Entity type label
    pub entity_type: String,
    /// Start CHARACTER offset (not byte offset!)
    pub start: usize,
    /// End CHARACTER offset (exclusive, not byte offset!)
    pub end: usize,
    /// Extraction confidence
    pub confidence: f32,
    /// Text before entity (context window)
    pub left_context: String,
    /// Text after entity (context window)
    pub right_context: String,
    /// Full sentence containing entity (if available)
    pub sentence: Option<String>,
    /// Start CHARACTER offset of sentence in original text
    pub sentence_start: Option<usize>,
}

/// Run the context analysis command.
pub fn run(args: ContextArgs) -> Result<(), String> {
    // Get input text
    let text = if let Some(path) = &args.input {
        fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?
    } else if let Some(t) = &args.text {
        t.clone()
    } else {
        return Err("No input provided. Use --input or --text".into());
    };

    // Create model and extract entities
    let model = args.model.create_model()?;
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    // Filter if requested
    let entities: Vec<_> = if let Some(ref type_filter) = args.entity_type {
        entities
            .into_iter()
            .filter(|e| e.entity_type.as_label().eq_ignore_ascii_case(type_filter))
            .collect()
    } else {
        entities
    };

    // Build context for each entity
    let contexts: Vec<EntityContext> = entities
        .iter()
        .map(|e| build_context(e, &text, args.window, args.full_sentence))
        .collect();

    // Format output
    let output_str = match args.format {
        ContextFormat::Human => format_human(&contexts, args.quiet),
        ContextFormat::Json => format_json(&contexts),
        ContextFormat::Tsv => format_tsv(&contexts),
        ContextFormat::Markdown => format_markdown(&contexts),
        ContextFormat::Brat => format_brat(&contexts),
    };

    // Output
    if let Some(path) = &args.output {
        fs::write(path, &output_str).map_err(|e| format!("Failed to write file: {}", e))?;
        if !args.quiet {
            eprintln!(
                "{} Exported {} entities with context to {:?}",
                color("32", "✓"),
                contexts.len(),
                path
            );
        }
    } else {
        println!("{}", output_str);
    }

    Ok(())
}

fn build_context(
    entity: &anno_core::Entity,
    text: &str,
    window: usize,
    full_sentence: bool,
) -> EntityContext {
    let start = entity.start; // CHARACTER offset
    let end = entity.end; // CHARACTER offset
    let char_count = text.chars().count();

    // Character-based context window (entity offsets are CHARACTER offsets)
    let left_start = start.saturating_sub(window);
    let right_end = (end + window).min(char_count);

    // Extract context using character iteration (not byte slicing!)
    let left_context: String = text
        .chars()
        .skip(left_start)
        .take(start.saturating_sub(left_start))
        .collect();
    let right_context: String = text
        .chars()
        .skip(end)
        .take(right_end.saturating_sub(end))
        .collect();

    // Sentence context (if requested)
    let (sentence, sentence_start) = if full_sentence {
        find_sentence(text, start, end)
    } else {
        (None, None)
    };

    EntityContext {
        text: entity.text.clone(),
        entity_type: entity.entity_type.as_label().to_string(),
        start,
        end,
        confidence: entity.confidence as f32,
        left_context,
        right_context,
        sentence,
        sentence_start,
    }
}

fn find_sentence(text: &str, start: usize, end: usize) -> (Option<String>, Option<usize>) {
    // Note: start and end are CHARACTER offsets, not byte offsets
    let sentence_terminators = ['.', '!', '?'];
    let chars: Vec<char> = text.chars().collect();
    let char_count = chars.len();

    // Find start of sentence (scan backward from entity start)
    let mut sentence_start = 0;
    for i in (0..start.min(char_count)).rev() {
        if sentence_terminators.contains(&chars[i]) {
            sentence_start = i + 1;
            break;
        }
    }

    // Skip leading whitespace
    while sentence_start < start.min(char_count) && chars[sentence_start].is_whitespace() {
        sentence_start += 1;
    }

    // Find end of sentence (scan forward from entity end)
    let sentence_end = chars[end.min(char_count)..char_count]
        .iter()
        .position(|&c| sentence_terminators.contains(&c))
        .map(|pos| end.min(char_count) + pos + 1)
        .unwrap_or(char_count);

    // Extract sentence using character indices
    let sentence: String = chars[sentence_start..sentence_end].iter().collect();
    (Some(sentence), Some(sentence_start))
}

fn format_human(contexts: &[EntityContext], quiet: bool) -> String {
    if quiet {
        return contexts
            .iter()
            .map(|c| format!("{}\t{}\t{}", c.entity_type, c.text, c.start))
            .collect::<Vec<_>>()
            .join("\n");
    }

    let mut output = String::new();
    output.push_str(&format!("{}\n\n", color("1;36", "Entity Context Export")));

    for (i, ctx) in contexts.iter().enumerate() {
        output.push_str(&format!(
            "{}: {} \"{}\"\n",
            color("1;33", &format!("Entity {}", i + 1)),
            color("36", &ctx.entity_type),
            ctx.text
        ));
        output.push_str(&format!(
            "  Span: {}:{} ({:.0}% conf)\n",
            ctx.start,
            ctx.end,
            ctx.confidence * 100.0
        ));

        // Context with highlighting
        output.push_str(&format!(
            "  Context: ...{}{}{}...\n",
            color("90", &ctx.left_context),
            color("1;33", &ctx.text),
            color("90", &ctx.right_context)
        ));

        if let Some(ref sentence) = ctx.sentence {
            // Highlight entity in sentence using character offsets
            let sentence_chars: Vec<char> = sentence.chars().collect();
            let relative_start = ctx.start.saturating_sub(ctx.sentence_start.unwrap_or(0));
            let relative_end = relative_start + ctx.text.chars().count();

            let before: String = sentence_chars.iter().take(relative_start).collect();
            let entity_text = &ctx.text;
            let after: String = sentence_chars.iter().skip(relative_end).collect();
            output.push_str(&format!(
                "  Sentence: {}{}{}\n",
                before,
                color("1;33", entity_text),
                after
            ));
        }
        output.push('\n');
    }

    output.push_str(&format!("─── {} entities exported\n", contexts.len()));

    output
}

fn format_json(contexts: &[EntityContext]) -> String {
    let json_contexts: Vec<_> = contexts
        .iter()
        .map(|c| {
            serde_json::json!({
                "entity": {
                    "text": c.text,
                    "type": c.entity_type,
                    "start": c.start,
                    "end": c.end,
                    "confidence": (c.confidence * 100.0).round() / 100.0,
                },
                "context": {
                    "left": c.left_context,
                    "right": c.right_context,
                    "sentence": c.sentence,
                    "sentence_start": c.sentence_start,
                }
            })
        })
        .collect();

    serde_json::to_string_pretty(&serde_json::json!({
        "entities": json_contexts,
        "count": contexts.len(),
    }))
    .unwrap_or_default()
}

fn format_tsv(contexts: &[EntityContext]) -> String {
    let mut output =
        String::from("text\ttype\tstart\tend\tconf\tleft_context\tright_context\tsentence\n");
    for c in contexts {
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\n",
            c.text,
            c.entity_type,
            c.start,
            c.end,
            c.confidence,
            c.left_context.replace(['\t', '\n'], " "),
            c.right_context.replace(['\t', '\n'], " "),
            c.sentence
                .as_deref()
                .unwrap_or("")
                .replace(['\t', '\n'], " ")
        ));
    }
    output
}

fn format_markdown(contexts: &[EntityContext]) -> String {
    let mut output = String::from("# Entity Context Export\n\n");

    output.push_str("| Entity | Type | Span | Confidence | Context |\n");
    output.push_str("|--------|------|------|------------|--------|\n");

    for c in contexts {
        let context_preview = format!(
            "...{}**{}**{}...",
            &c.left_context[c.left_context.len().saturating_sub(20)..],
            c.text,
            &c.right_context[..c.right_context.len().min(20)]
        );
        output.push_str(&format!(
            "| {} | {} | {}:{} | {:.0}% | {} |\n",
            c.text,
            c.entity_type,
            c.start,
            c.end,
            c.confidence * 100.0,
            context_preview.replace('|', "\\|")
        ));
    }

    output.push_str(&format!("\n*{} entities exported*\n", contexts.len()));

    output
}

fn format_brat(contexts: &[EntityContext]) -> String {
    let mut output = String::new();

    // brat standoff format: T1\tType Start End\tText
    for (i, c) in contexts.iter().enumerate() {
        output.push_str(&format!(
            "T{}\t{} {} {}\t{}\n",
            i + 1,
            c.entity_type,
            c.start,
            c.end,
            c.text
        ));
    }

    output
}
