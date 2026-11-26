//! Inspect reported entity provenance, confidence, context, and character offsets.

use clap::Parser;

use super::super::output::color;
use super::super::parser::ModelBackend;

/// Explain an entity extraction decision
#[derive(Parser, Debug)]
#[command(group(clap::ArgGroup::new("input_source").args(["text", "positional"])))]
pub struct ExplainArgs {
    /// Entity ID from extract JSON (xxh3:...)
    #[arg(short, long)]
    pub entity_id: Option<String>,

    /// Text containing the entity
    #[arg(short, long)]
    pub text: Option<String>,

    /// Entity span in Unicode characters, from inclusive start to exclusive end
    #[arg(short, long, value_name = "START:END")]
    pub span: Option<String>,

    /// Model backend to use for explanation
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Show other returned entities overlapping this span
    #[arg(long)]
    pub show_all: bool,

    /// Positional text input
    pub positional: Vec<String>,
}

/// Run the explain command.
pub fn run(args: ExplainArgs) -> Result<(), String> {
    // Get text from args
    let text = if let Some(t) = args.text {
        t
    } else if !args.positional.is_empty() {
        args.positional.join(" ")
    } else {
        return Err("No text provided. Use --text or provide text as positional argument.".into());
    };

    // Parse span if provided
    let span = if let Some(s) = &args.span {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err("Span must be in format START:END (e.g., 0:11)".into());
        }
        let start: usize = parts[0].parse().map_err(|_| "Invalid start offset")?;
        let end: usize = parts[1].parse().map_err(|_| "Invalid end offset")?;
        if start >= end || end > text.chars().count() {
            return Err("Span must be a nonempty range within the input text".into());
        }
        Some((start, end))
    } else {
        None
    };

    // Create model and extract entities
    let model = args.model.create_model()?;
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    if entities.is_empty() && args.entity_id.is_none() {
        println!("No entities found in text.");
        return Ok(());
    }

    // Filter to specific span if requested
    let prefix = super::extract::entity_id_prefix(&text);
    let entities_to_explain: Vec<_> = entities
        .iter()
        .filter(|e| {
            span.is_none_or(|pair| pair == (e.start(), e.end()))
                && args.entity_id.as_ref().is_none_or(|id| {
                    *id == super::extract::compute_entity_id(
                        &prefix,
                        &e.text,
                        e.entity_type.as_label(),
                        e.start(),
                        e.end(),
                    )
                })
        })
        .collect();
    if entities_to_explain.is_empty() {
        if let Some(id) = args.entity_id {
            return Err(format!("No entity matches {id} in this extraction"));
        }
        println!("No entities match the specified span.");
        return Ok(());
    }

    // Print explanations
    for (idx, entity) in entities_to_explain.iter().enumerate() {
        if idx > 0 {
            println!();
        }

        let source = entity
            .provenance
            .as_ref()
            .map(|p| p.source.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        println!("{}", color("1;36", &format!("Entity: \"{}\"", entity.text)));
        println!();

        // Type decision
        println!("{}:", color("1;33", "Type Decision"));
        println!(
            "  {} ({:.0}%)",
            color("32", entity.entity_type.as_label()),
            entity.confidence * 100.0
        );
        println!();

        // Source backend
        println!("{}:", color("1;33", "Source Backend"));
        println!("  {}", source);
        println!();

        if let Some(provenance) = &entity.provenance {
            println!("  method: {:?}", provenance.method);
            if let Some(pattern) = &provenance.pattern {
                println!("  rule: {pattern}");
            }
            if let Some(version) = &provenance.model_version {
                println!("  version: {version}");
            }
        }

        // Context
        let ctx_start = entity.start().saturating_sub(30);
        let ctx_end = (entity.end() + 30).min(text.chars().count());
        let before: String = text
            .chars()
            .skip(ctx_start)
            .take(entity.start() - ctx_start)
            .collect();
        let entity_text: String = text
            .chars()
            .skip(entity.start())
            .take(entity.end() - entity.start())
            .collect();
        let after: String = text
            .chars()
            .skip(entity.end())
            .take(ctx_end - entity.end())
            .collect();

        println!("{}:", color("1;33", "Context"));
        println!(
            "  {}{}{}{}{}",
            if ctx_start > 0 { "..." } else { "" },
            color("90", &before),
            color("1;33", &entity_text),
            color("90", &after),
            if ctx_end < text.chars().count() {
                "..."
            } else {
                ""
            }
        );
        println!();

        // Span info
        println!("{}:", color("1;33", "Span"));
        println!("  start: {} (character offset)", entity.start());
        println!("  end: {} (exclusive, character offset)", entity.end());
        println!("  length: {} chars", entity.end() - entity.start());

        // If showing all candidates, show other potential extractions
        if args.show_all && entities.len() > 1 {
            println!();
            println!("{}:", color("1;33", "Overlapping entities"));
            for other in &entities {
                if std::ptr::eq(other, *entity) {
                    continue;
                }
                // Check for overlap
                let overlaps = !(other.end() <= entity.start() || other.start() >= entity.end());
                if overlaps {
                    let other_source = other
                        .provenance
                        .as_ref()
                        .map(|p| p.source.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    println!(
                        "  {} \"{}\" ({:.0}%) from {}",
                        other.entity_type.as_label(),
                        other.text,
                        other.confidence * 100.0,
                        other_source
                    );
                }
            }
        }
    }

    Ok(())
}
