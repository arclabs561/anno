//! Explain command - Show why an entity was classified (debugging, auditing, trust)
//!
//! This command provides introspection into entity extraction decisions,
//! useful for debugging, auditing, and building user trust.
//!
//! # Explanation Hierarchy (see docs/notes/design/cli/OUTPUT_DESIGN.md - dense output sketch)
//!
//! "Why did you extract this?" has multiple valid answers at different levels:
//!
//! - **Statistical**: P(PERSON|features) = 0.97, top features with weights
//! - **Linguistic**: Syntax position, context words, gazetteer matches
//! - **Semantic**: Compatible predicates, selectional restrictions
//! - **Provenance**: Similar training examples, model attention patterns
//! - **Counterfactual**: "If 'Dr.' removed, conf drops 0.97→0.89"
//!
//! Future: Add `--level statistical|linguistic|semantic|all` flag.

use clap::Parser;

use super::super::output::color;
use super::super::parser::ModelBackend;

/// Explain an entity extraction decision
#[derive(Parser, Debug)]
pub struct ExplainArgs {
    /// Entity ID to explain (e.g., e:6c926597)
    #[arg(short, long)]
    pub entity_id: Option<String>,

    /// Text containing the entity
    #[arg(short, long)]
    pub text: Option<String>,

    /// Entity span (start:end) to explain
    #[arg(short, long, value_name = "START:END")]
    pub span: Option<String>,

    /// Model backend to use for explanation
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Show all candidate extractions (not just winners)
    #[arg(long)]
    pub show_all: bool,

    /// Positional text argument
    /// Positional text input
    pub positional: Vec<String>,
}

/// Feature contribution to an entity decision
#[derive(Debug, Clone)]
pub struct FeatureContribution {
    /// Feature name
    pub name: String,
    /// Feature value
    pub value: String,
    /// Feature weight in decision
    pub weight: f64,
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
        Some((start, end))
    } else {
        None
    };

    // Create model and extract entities
    let model = args.model.create_model()?;
    let entities = model
        .extract_entities(&text, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    if entities.is_empty() {
        println!("No entities found in text.");
        return Ok(());
    }

    // Filter to specific span if requested
    let entities_to_explain: Vec<_> = if let Some((start, end)) = span {
        entities
            .iter()
            .filter(|e| e.start() == start && e.end() == end)
            .collect()
    } else {
        entities.iter().collect()
    };

    if entities_to_explain.is_empty() {
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

        // Features/evidence (simulate based on entity characteristics)
        println!("{}:", color("1;33", "Features"));
        let features = analyze_features(&text, entity);
        for feat in &features {
            let sign = if feat.weight > 0.0 { "+" } else { "" };
            println!(
                "  {} = {} ({}{:.2})",
                color("90", &feat.name),
                feat.value,
                sign,
                feat.weight
            );
        }
        println!();

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
            println!("{}:", color("1;33", "Other Candidates"));
            for other in &entities {
                if other.start() == entity.start() && other.end() == entity.end() {
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
                        "  {} \"{}\" ({:.0}%) from {} - {}",
                        other.entity_type.as_label(),
                        other.text,
                        other.confidence * 100.0,
                        other_source,
                        color("31", "conflict resolved")
                    );
                }
            }
        }
    }

    Ok(())
}

/// Analyze features that contributed to entity classification
fn analyze_features(text: &str, entity: &anno::Entity) -> Vec<FeatureContribution> {
    let mut features = Vec::new();
    let entity_text = &entity.text;

    // Capitalization
    let first_char = entity_text.chars().next();
    if let Some(c) = first_char {
        if c.is_uppercase() {
            features.push(FeatureContribution {
                name: "capitalization".into(),
                value: "TitleCase".into(),
                weight: 0.15,
            });
        }
    }

    // All caps
    if entity_text
        .chars()
        .all(|c| !c.is_alphabetic() || c.is_uppercase())
        && entity_text.len() > 1
    {
        features.push(FeatureContribution {
            name: "all_caps".into(),
            value: "true".into(),
            weight: 0.10,
        });
    }

    // Contains period (likely abbreviation or title)
    if entity_text.contains('.') {
        features.push(FeatureContribution {
            name: "contains_period".into(),
            value: "true".into(),
            weight: 0.05,
        });
    }

    // Word count
    let word_count = entity_text.split_whitespace().count();
    features.push(FeatureContribution {
        name: "word_count".into(),
        value: word_count.to_string(),
        weight: if word_count > 1 { 0.05 } else { 0.0 },
    });

    // Left context analysis
    let left_ctx: String = text.chars().take(entity.start()).collect();
    let left_words: Vec<&str> = left_ctx.split_whitespace().rev().take(3).collect();

    // Title detection
    let titles = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sir", "Lord", "Lady"];
    for title in &titles {
        if left_words.iter().any(|w| w.ends_with(title)) {
            features.push(FeatureContribution {
                name: "context_left".into(),
                value: format!("preceded by '{}'", title),
                weight: 0.20,
            });
            break;
        }
    }

    // Verb context for persons
    let right_ctx: String = text.chars().skip(entity.end()).take(50).collect();
    let person_verbs = ["said", "says", "told", "announced", "declared", "stated"];
    for verb in &person_verbs {
        if right_ctx.to_lowercase().starts_with(&format!(" {}", verb))
            || right_ctx.to_lowercase().starts_with(&format!(", {}", verb))
        {
            features.push(FeatureContribution {
                name: "context_right".into(),
                value: format!("followed by '{}'", verb),
                weight: 0.15,
            });
            break;
        }
    }

    // Organization indicators
    let org_suffixes = [
        "Inc.",
        "Corp.",
        "LLC",
        "Ltd.",
        "Co.",
        "Company",
        "Corporation",
    ];
    for suffix in &org_suffixes {
        if entity_text.ends_with(suffix) {
            features.push(FeatureContribution {
                name: "org_suffix".into(),
                value: format!("ends with '{}'", suffix),
                weight: 0.25,
            });
            break;
        }
    }

    // Location indicators
    let loc_preps = ["in", "at", "from", "to", "near"];
    for prep in &loc_preps {
        if left_words.iter().any(|w| w.to_lowercase() == *prep) {
            features.push(FeatureContribution {
                name: "location_preposition".into(),
                value: format!("preceded by '{}'", prep),
                weight: 0.18,
            });
            break;
        }
    }

    // Pattern match (for structured types)
    if entity.entity_type.as_label() == "EMAIL" {
        features.push(FeatureContribution {
            name: "pattern_match".into(),
            value: "email_regex".into(),
            weight: 1.0,
        });
    } else if entity.entity_type.as_label() == "DATE" {
        features.push(FeatureContribution {
            name: "pattern_match".into(),
            value: "date_regex".into(),
            weight: 1.0,
        });
    } else if entity.entity_type.as_label() == "MONEY" {
        features.push(FeatureContribution {
            name: "pattern_match".into(),
            value: "money_regex".into(),
            weight: 1.0,
        });
    }

    // Sort by weight descending
    features.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    features
}
