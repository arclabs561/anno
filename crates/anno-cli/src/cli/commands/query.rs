//! Query command - Query and filter entities/clusters

use clap::Parser;
use std::fs;
use std::io::{self, Read};

use super::super::parser::OutputFormat;
use super::super::utils::{format_error, parse_grounded_document, read_input_file};
use anno::{Location, Signal};

/// Query and filter entities/clusters
#[derive(Parser, Debug)]
pub struct QueryArgs {
    /// Input file (GroundedDocument JSON or cross-doc clusters JSON)
    #[arg(value_name = "FILE")]
    pub input: String,

    /// Filter by entity type
    #[arg(short, long, value_name = "TYPE")]
    pub r#type: Option<String>,

    /// Find specific entity by name
    #[arg(short, long, value_name = "TEXT")]
    pub entity: Option<String>,

    /// Minimum confidence threshold
    #[arg(long, value_name = "FLOAT")]
    pub min_confidence: Option<f64>,

    /// Filter expression (e.g., "type=ORG AND confidence>0.7")
    #[arg(short, long, value_name = "EXPR")]
    pub filter: Option<String>,

    /// Start offset for range queries (character position)
    #[arg(long, value_name = "OFFSET")]
    pub start_offset: Option<usize>,

    /// End offset for range queries (character position)
    #[arg(long, value_name = "OFFSET")]
    pub end_offset: Option<usize>,

    /// Filter for negated signals only
    #[arg(long)]
    pub negated: bool,

    /// Filter for signals with quantifiers
    #[arg(long)]
    pub quantified: bool,

    /// Filter for untracked signals (not in any track)
    #[arg(long)]
    pub untracked: bool,

    /// Filter for signals linked to identities (via tracks)
    #[arg(long)]
    pub linked: bool,

    /// Filter for signals not linked to identities
    #[arg(long)]
    pub unlinked: bool,

    /// Output format
    #[arg(long, default_value = "human")]
    pub format: OutputFormat,

    /// Output file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<String>,
}

#[derive(Default)]
struct FilterExpression {
    entity_type: Option<String>,
    confidence_greater_than: Option<f64>,
}

fn parse_filter_expression(expr: &str) -> Result<FilterExpression, String> {
    let mut filter = FilterExpression::default();
    for clause in expr.split(" AND ") {
        let clause = clause.trim();
        if let Some(entity_type) = clause.strip_prefix("type=") {
            if entity_type.is_empty()
                || entity_type.chars().any(char::is_whitespace)
                || entity_type.contains(['=', '<', '>'])
            {
                return Err(format!("invalid type value in filter: {entity_type:?}"));
            }
            if filter.entity_type.is_some() {
                return Err("filter may contain only one type clause".to_string());
            }
            filter.entity_type = Some(entity_type.to_string());
        } else if let Some(confidence) = clause.strip_prefix("confidence>") {
            if filter.confidence_greater_than.is_some() {
                return Err("filter may contain only one confidence clause".to_string());
            }
            let confidence = confidence
                .parse::<f64>()
                .map_err(|_| format!("invalid confidence threshold in filter: {confidence}"))?;
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err(format!(
                    "confidence threshold must be finite and within 0..=1: {confidence}"
                ));
            }
            filter.confidence_greater_than = Some(confidence);
        } else {
            return Err(format!(
                "unsupported filter clause '{clause}'; use type=TYPE and confidence>VALUE joined by AND"
            ));
        }
    }
    Ok(filter)
}

/// Execute the query command.
pub fn run(args: QueryArgs) -> Result<(), String> {
    let expression = args
        .filter
        .as_deref()
        .map(parse_filter_expression)
        .transpose()?
        .unwrap_or_default();
    // Load input file
    let json_content = if args.input == "-" {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format_error("read stdin", &e.to_string()))?;
        buf
    } else {
        read_input_file(&args.input)?
    };

    // Try to parse as GroundedDocument first, then as cross-doc clusters
    if let Ok(doc) = parse_grounded_document(&json_content) {
        // Query single document - use GroundedDocument helper methods where applicable
        let mut signals: Vec<Signal<Location>> = if let Some(ref filter_type) = args.r#type {
            // Use signals_with_label helper for type filtering (returns Vec<&Signal>, clone to Vec<Signal>)
            doc.signals_with_label(filter_type)
                .into_iter()
                .cloned()
                .collect()
        } else {
            doc.signals().to_vec()
        };

        // Apply range filters using spatial index if both offsets provided
        if let (Some(start), Some(end)) = (args.start_offset, args.end_offset) {
            signals = doc
                .query_signals_in_range_indexed(start, end)
                .into_iter()
                .cloned()
                .collect();
        }

        // Apply additional filters
        if let Some(min_conf) = args.min_confidence {
            // Filter by confidence (could use confident_signals, but already have collection)
            signals.retain(|s| s.confidence >= min_conf);
        }

        if let Some(ref filter_type) = expression.entity_type {
            signals.retain(|s| s.label().eq_ignore_ascii_case(filter_type));
        }

        if let Some(confidence) = expression.confidence_greater_than {
            signals.retain(|s| s.confidence > confidence);
        }

        if let Some(ref entity_text) = args.entity {
            // Filter by entity name
            signals.retain(|s| {
                s.surface()
                    .to_lowercase()
                    .contains(&entity_text.to_lowercase())
            });
        }

        // Apply signal property filters
        if args.negated {
            signals.retain(|s| s.negated);
        }

        if args.quantified {
            signals.retain(|s| s.quantifier.is_some());
        }

        // Apply relationship filters (require checking track/identity membership)
        if args.untracked {
            signals.retain(|s| doc.track_for_signal(s.id).is_none());
        }

        if args.linked {
            signals.retain(|s| doc.identity_for_signal(s.id).is_some());
        }

        if args.unlinked {
            signals.retain(|s| doc.identity_for_signal(s.id).is_none());
        }

        // Output results
        match args.format {
            OutputFormat::Json | OutputFormat::Grounded => {
                let output = serde_json::to_string_pretty(&signals)
                    .map_err(|e| format!("Failed to serialize: {}", e))?;
                if let Some(output_path) = &args.output {
                    fs::write(output_path, output)
                        .map_err(|e| format!("Failed to write output: {}", e))?;
                } else {
                    println!("{}", output);
                }
            }
            _ => {
                println!("Found {} entities:", signals.len());
                for s in &signals {
                    let (start, end) = s.text_offsets().unwrap_or((0, 0));
                    println!(
                        "  [{}:{}] {} ({}) - {:.2}",
                        start,
                        end,
                        s.surface(),
                        s.label(),
                        s.confidence
                    );
                }
            }
        }
    } else {
        // Try to parse as cross-doc clusters (requires cross-doc support).
        #[cfg(feature = "eval")]
        {
            if let Ok(clusters) =
                serde_json::from_str::<Vec<anno_eval::cdcr::CrossDocCluster>>(&json_content)
                    .map_err(|e| format_error("parse cross-doc clusters JSON", &e.to_string()))
            {
                let mut filtered: Vec<_> = clusters.iter().collect();

                // Apply filters
                if let Some(ref filter_type) = args.r#type {
                    filtered.retain(|c| {
                        c.entity_type
                            .as_ref()
                            .map(|t| t.as_label().eq_ignore_ascii_case(filter_type))
                            .unwrap_or(false)
                    });
                }

                if let Some(ref filter_type) = expression.entity_type {
                    filtered.retain(|c| {
                        c.entity_type
                            .as_ref()
                            .map(|t| t.as_label().eq_ignore_ascii_case(filter_type))
                            .unwrap_or(false)
                    });
                }

                if expression.confidence_greater_than.is_some() {
                    return Err(
                        "confidence filters are only supported for GroundedDocument input"
                            .to_string(),
                    );
                }

                if let Some(ref entity_text) = args.entity {
                    filtered.retain(|c| {
                        c.canonical_name
                            .to_lowercase()
                            .contains(&entity_text.to_lowercase())
                    });
                }

                // Output results
                match args.format {
                    OutputFormat::Tree => {
                        for cluster in &filtered {
                            println!("Cluster {}: {}", cluster.id, cluster.canonical_name);
                            for (doc_id, entity_idx) in &cluster.mentions {
                                println!("  - entity[{}] (doc: {})", entity_idx, doc_id);
                            }
                            println!();
                        }
                    }
                    OutputFormat::Json | OutputFormat::Grounded => {
                        let output = serde_json::to_string_pretty(&filtered)
                            .map_err(|e| format!("Failed to serialize: {}", e))?;
                        if let Some(output_path) = &args.output {
                            fs::write(output_path, output)
                                .map_err(|e| format!("Failed to write output: {}", e))?;
                        } else {
                            println!("{}", output);
                        }
                    }
                    _ => {
                        println!("Found {} clusters:", filtered.len());
                        for cluster in &filtered {
                            println!(
                                "  {}: {} mentions across {} documents",
                                cluster.canonical_name,
                                cluster.mentions.len(),
                                cluster.doc_count()
                            );
                        }
                    }
                }
                return Ok(());
            }
        }

        // If we get here with eval feature but failed to parse as clusters, error out
        return Err("Failed to parse input as GroundedDocument".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documented_filter_expression_maps_to_existing_predicates() {
        let filter = parse_filter_expression("type=ORG AND confidence>0.7").unwrap();
        assert_eq!(filter.entity_type.as_deref(), Some("ORG"));
        assert_eq!(filter.confidence_greater_than, Some(0.7));
    }

    #[test]
    fn filter_expression_rejects_silent_noop_clauses() {
        assert!(parse_filter_expression("surface~Alice").is_err());
        assert!(parse_filter_expression("type=ORG AND type=PER").is_err());
        assert!(parse_filter_expression("confidence>NaN").is_err());
        assert!(parse_filter_expression("confidence>1.1").is_err());
        assert!(parse_filter_expression("type>ORG").is_err());
    }

    #[test]
    fn query_filter_expression_uses_strict_confidence_and_writes_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.json");
        let output = dir.path().join("output.json");
        let entities = vec![
            anno::Entity::new("Equal", anno::EntityType::Organization, 0, 5, 0.7),
            anno::Entity::new("Included", anno::EntityType::Organization, 6, 14, 0.9),
            anno::Entity::new("WrongType", anno::EntityType::Person, 15, 24, 0.99),
        ];
        let document = anno::GroundedDocument::from_entity_signals(
            "test",
            "Equal Included WrongType",
            &entities,
        );
        fs::write(&input, serde_json::to_string(&document).unwrap()).unwrap();

        run(QueryArgs {
            input: input.to_string_lossy().into_owned(),
            r#type: None,
            entity: None,
            min_confidence: None,
            filter: Some("type=ORG AND confidence>0.7".to_string()),
            start_offset: None,
            end_offset: None,
            negated: false,
            quantified: false,
            untracked: false,
            linked: false,
            unlinked: false,
            format: OutputFormat::Json,
            output: Some(output.to_string_lossy().into_owned()),
        })
        .unwrap();

        let output: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output).unwrap()).unwrap();
        let signals = output.as_array().unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0]["surface"], "Included");
    }
}
