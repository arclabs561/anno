//! Query command - Query and filter entities/clusters

mod edition;
mod memberships;
mod predecessors;
mod publication;
mod relations;

use clap::{Parser, Subcommand};
use publication::CatalogRecord;
use std::fs;
use std::io::{self, Read};

use super::super::parser::OutputFormat;
use super::super::utils::{format_error, parse_grounded_document, read_input_file};
use anno::{Location, Signal};

/// Query and filter entities/clusters
#[derive(Subcommand, Debug)]
pub enum QueryCommand {
    /// Publish or compare complete revisions of one grounded document.
    Publication(publication::PublicationArgs),
    /// Stage, publish or read a coherent catalog edition in SQLite.
    Edition(edition::EditionArgs),
    /// Publish derived track memberships for a complete corpus or compare two corpus editions.
    Memberships(memberships::MembershipArgs),
    /// Find possible predecessors for one track in an earlier corpus edition.
    Predecessors(predecessors::PredecessorArgs),
    /// Read saved mention and concept relations with current track assignments.
    Relations(relations::RelationsArgs),
}

#[derive(Parser, Debug)]
#[command(subcommand_negates_reqs = true, args_conflicts_with_subcommands = true)]
pub struct QueryArgs {
    #[command(subcommand)]
    pub command: Option<QueryCommand>,
    /// Input file (GroundedDocument JSON or cross-doc clusters JSON)
    #[arg(value_name = "FILE", required = true)]
    pub input: Option<String>,

    /// Select occurrences or clusters bound to this exact catalog and accession.
    #[arg(long, num_args = 2, value_names = ["CATALOG", "ACCESSION"], conflicts_with_all = ["untracked", "unlinked"])]
    pub catalog_record: Vec<String>,

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
    if let Some(command) = args.command {
        return match command {
            QueryCommand::Publication(args) => publication::run(args),
            QueryCommand::Edition(args) => edition::run(args),
            QueryCommand::Memberships(args) => memberships::run(args),
            QueryCommand::Predecessors(args) => predecessors::run(args),
            QueryCommand::Relations(args) => relations::run(args),
        };
    }
    let catalog = CatalogRecord::from_args(&args.catalog_record)?;
    let input = args.input.as_deref().ok_or("missing query input")?;
    let expression = args
        .filter
        .as_deref()
        .map(parse_filter_expression)
        .transpose()?
        .unwrap_or_default();
    // Load input file
    let json_content = if input == "-" {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format_error("read stdin", &e.to_string()))?;
        buf
    } else {
        read_input_file(input)?
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

        if let Some(catalog) = &catalog {
            signals.retain(|signal| {
                doc.identity_for_signal(signal.id)
                    .and_then(anno::Identity::qualified_kb_ref)
                    == Some(catalog.as_ref())
            });
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
        #[cfg(feature = "eval")]
        return query_clusters(&json_content, &args, &expression, catalog.as_ref());
        #[cfg(not(feature = "eval"))]
        return Err("Failed to parse input as GroundedDocument. Cross-document queries require the eval feature.".into());
    }

    Ok(())
}

/// Query the JSON report emitted by cross-doc, or a bare cluster array from older callers.
#[cfg(feature = "eval")]
fn query_clusters(
    json: &str,
    args: &QueryArgs,
    expression: &FilterExpression,
    catalog: Option<&CatalogRecord>,
) -> Result<(), String> {
    use anno_eval::cdcr::CrossDocCluster;
    use std::io::Write;

    #[derive(serde::Deserialize)]
    struct Report {
        clusters: Vec<CrossDocCluster>,
    }

    let mut clusters: Vec<CrossDocCluster> = if json.trim_start().starts_with('[') {
        serde_json::from_str(json)
    } else {
        serde_json::from_str::<Report>(json).map(|report| report.clusters)
    }
    .map_err(|e| format!("Expected GroundedDocument, cross-doc report or cluster array: {e}"))?;

    if expression.confidence_greater_than.is_some() || args.min_confidence.is_some() {
        return Err("confidence filters are only supported for GroundedDocument input".into());
    }
    if args.start_offset.is_some()
        || args.end_offset.is_some()
        || args.negated
        || args.quantified
        || args.linked
        || args.unlinked
        || args.untracked
    {
        return Err(
            "Offset, signal-property and track-membership filters require GroundedDocument input"
                .into(),
        );
    }
    for entity_type in [args.r#type.as_ref(), expression.entity_type.as_ref()]
        .into_iter()
        .flatten()
    {
        clusters.retain(|cluster| {
            cluster
                .entity_type
                .as_ref()
                .is_some_and(|ty| ty.as_label().eq_ignore_ascii_case(entity_type))
        });
    }
    if let Some(entity) = &args.entity {
        let name = entity.to_lowercase();
        clusters.retain(|cluster| cluster.canonical_name.to_lowercase().contains(&name));
    }
    if let Some(record) = catalog {
        clusters.retain(|cluster| {
            cluster.kb_name.as_deref().zip(cluster.kb_id.as_deref()) == Some(record.as_ref())
        });
    }

    let output = match args.format {
        OutputFormat::Json => {
            let mut json = serde_json::to_string_pretty(&clusters)
                .map_err(|e| format!("Failed to serialize clusters: {e}"))?;
            json.push('\n');
            json
        }
        OutputFormat::Tree => {
            let mut output = String::new();
            for cluster in clusters {
                output.push_str(&format!(
                    "Cluster {}: {}\n",
                    cluster.id, cluster.canonical_name
                ));
                for (doc_id, entity_idx) in cluster.mentions {
                    output.push_str(&format!("  - entity[{entity_idx}] (doc: {doc_id})\n"));
                }
                output.push('\n');
            }
            output
        }
        OutputFormat::Human | OutputFormat::Summary => {
            let mut output = format!("Found {} clusters:\n", clusters.len());
            for cluster in clusters {
                output.push_str(&format!(
                    "  {}: {} mentions across {} documents\n",
                    cluster.canonical_name,
                    cluster.mentions.len(),
                    cluster.doc_count()
                ));
            }
            output
        }
        other => {
            let formats = "human, summary, tree or json";
            return Err(format!(
                "Format {other:?} is unsupported for cluster queries. Use {formats}."
            ));
        }
    };
    if let Some(path) = &args.output {
        fs::write(path, output)
    } else {
        io::stdout().lock().write_all(output.as_bytes())
    }
    .map_err(|e| format_error("write query output", &e.to_string()))
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
            command: None,
            input: Some(input.to_string_lossy().into_owned()),
            catalog_record: Vec::new(),
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
