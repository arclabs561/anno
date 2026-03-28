//! Export command - Export annotations to various formats
//!
//! Supports exporting to:
//! - brat standoff format (.ann files)
//! - CoNLL format (IOB/BIO tagging)
//! - JSONL (one entity per line)
//! - N-Triples (RDF triples for knowledge graphs)
//! - JSON-LD (linked data for knowledge graphs)
//! - graph-ntriples (N-Triples via the lattix graph substrate; stable triple rendering)
//! - Graph CSV (node + co-occurrence edge tables for graph DB import)
//!
//! When the selected backend supports relation extraction (e.g. `tplinker`), graph-oriented formats
//! (graph-ntriples, ntriples, jsonld, graph-csv) emit typed semantic triples instead of falling
//! back to co-occurrence edges.

use clap::{Parser, ValueEnum};
use std::fs;
use std::path::{Path, PathBuf};

use super::super::output::color;
use super::super::parser::ModelBackend;
use super::super::utils::parse_grounded_document;

/// Export annotations to different formats
#[derive(Parser, Debug)]
pub struct ExportArgs {
    /// Input file or directory
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output directory
    #[arg(short, long, value_name = "DIR")]
    pub output: PathBuf,

    /// Export format
    #[arg(short, long, default_value = "brat")]
    pub format: ExportFormat,

    /// Model backend to use for extraction.
    /// Use `--model tplinker` to enable joint entity+relation extraction (actual semantic
    /// triples rather than co-occurrence edges) for graph-oriented formats.
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Overwrite existing files
    #[arg(long)]
    pub overwrite: bool,

    /// Include confidence scores in output
    #[arg(long)]
    pub include_confidence: bool,

    /// Base URI prefix for RDF/KG namespaces (ntriples, jsonld, graph-ntriples).
    /// Use a real namespace for interoperable output, e.g.:
    ///   --base-uri https://www.gutenberg.org/ebooks/   (Project Gutenberg texts)
    ///   --base-uri https://dbpedia.org/resource/        (DBpedia-aligned)
    ///   --base-uri urn:anno:                            (default, stable URN prefix)
    #[arg(long, default_value = "urn:anno:")]
    pub base_uri: String,

    /// Quiet mode
    #[arg(short, long)]
    pub quiet: bool,
}

/// Export format
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum ExportFormat {
    /// brat standoff format (.ann files)
    #[default]
    Brat,
    /// CoNLL format (IOB tagging)
    Conll,
    /// JSONL (one entity per line)
    Jsonl,
    /// N-Triples (RDF format for knowledge graphs)
    #[value(name = "ntriples")]
    NTriples,
    /// JSON-LD (linked data format for knowledge graphs)
    #[value(name = "jsonld")]
    JsonLd,
    /// N-Triples via the lattix graph substrate (stable triple rendering; requires `--features graph`).
    /// With a relation-capable model (e.g. `--model tplinker`), emits real semantic triples.
    #[cfg(feature = "graph")]
    #[value(name = "graph-ntriples")]
    GraphNTriples,
    /// Graph CSV: node table + co-occurrence (or semantic) edge table for graph DB import.
    #[value(name = "graph-csv", alias = "kuzu")]
    GraphCsv,
}

// =============================================================================
// Extraction result
// =============================================================================

struct Extracted {
    entities: Vec<anno_core::Entity>,
    /// Populated when the backend supports relation extraction; empty otherwise.
    relations: Vec<anno_core::Relation>,
}

impl Extracted {
    fn entities_only(entities: Vec<anno_core::Entity>) -> Self {
        Self {
            entities,
            relations: Vec::new(),
        }
    }
}

/// Run the export command.
pub fn run(args: ExportArgs) -> Result<(), String> {
    // Validate input
    if !args.input.exists() {
        return Err(format!("Input not found: {:?}", args.input));
    }

    // Create output directory
    if !args.output.exists() {
        fs::create_dir_all(&args.output)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    // Choose extraction path: relation-capable model vs plain model.
    let relation_model = args.model.try_create_relation_model().transpose()?;
    let plain_model = if relation_model.is_none() {
        Some(args.model.create_model()?)
    } else {
        None
    };

    // Collect files to process
    let files: Vec<PathBuf> = if args.input.is_file() {
        vec![args.input.clone()]
    } else {
        fs::read_dir(&args.input)
            .map_err(|e| format!("Failed to read directory: {}", e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .is_some_and(|e| e == "txt" || e == "json" || e == "jsonl")
            })
            .collect()
    };

    if files.is_empty() {
        return Err("No .txt/.json/.jsonl files found in input".into());
    }

    if !args.quiet {
        let mode = if relation_model.is_some() {
            " (relation-capable)"
        } else {
            ""
        };
        eprintln!(
            "{} Exporting {} files to {:?} format{}",
            color("32", "[export]"),
            files.len(),
            args.format,
            mode,
        );
    }

    let mut success_count = 0;
    let mut error_count = 0;

    for file in &files {
        let is_json = file
            .extension()
            .is_some_and(|e| e == "json" || e == "jsonl");

        // Extract — JSON files are parsed as GroundedDocument; text files run extraction.
        let extracted = if is_json {
            let json_content =
                fs::read_to_string(file).map_err(|e| format!("Failed to read file: {}", e))?;
            let doc = parse_grounded_document(&json_content)?;
            let entities: Vec<anno_core::Entity> = doc
                .signals()
                .iter()
                .filter_map(|s| {
                    let (start, end) = s.text_offsets()?;
                    Some(anno_core::Entity::new(
                        s.surface(),
                        s.label.to_entity_type(),
                        start,
                        end,
                        s.confidence.value(),
                    ))
                })
                .collect();
            Ok((doc.text().to_owned(), Extracted::entities_only(entities)))
        } else if let Some(ref rm) = relation_model {
            let content =
                fs::read_to_string(file).map_err(|e| format!("Failed to read file: {}", e))?;
            #[allow(deprecated)]
            let (entities, relations) = rm
                .extract_with_relations(&content, None)
                .map_err(|e| format!("Extraction failed: {}", e))?;
            Ok((
                content,
                Extracted {
                    entities,
                    relations,
                },
            ))
        } else if let Some(ref m) = plain_model {
            let content =
                fs::read_to_string(file).map_err(|e| format!("Failed to read file: {}", e))?;
            let entities = m
                .extract_entities(&content, None)
                .map_err(|e| format!("Extraction failed: {}", e))?;
            Ok((content, Extracted::entities_only(entities)))
        } else {
            Err("No model available".to_string())
        };

        match extracted {
            Err(e) => {
                error_count += 1;
                if !args.quiet {
                    eprintln!(
                        "  {} {:?}: {}",
                        color("31", "✗"),
                        file.file_name().unwrap_or_default(),
                        e
                    );
                }
            }
            Ok((content, ext)) => {
                let entity_count = ext.entities.len();
                let rel_count = ext.relations.len();
                match export_file(ExportFileOpts {
                    input: file,
                    output_dir: &args.output,
                    content: &content,
                    ext,
                    format: args.format,
                    include_confidence: args.include_confidence,
                    overwrite: args.overwrite,
                    base_uri: &args.base_uri,
                }) {
                    Ok(()) => {
                        success_count += 1;
                        if !args.quiet {
                            let rel_suffix = if rel_count > 0 {
                                format!(", {} relations", rel_count)
                            } else {
                                String::new()
                            };
                            eprintln!(
                                "  {} {:?} ({} entities{})",
                                color("32", "✓"),
                                file.file_name().unwrap_or_default(),
                                entity_count,
                                rel_suffix,
                            );
                        }
                    }
                    Err(e) => {
                        error_count += 1;
                        if !args.quiet {
                            eprintln!(
                                "  {} {:?}: {}",
                                color("31", "✗"),
                                file.file_name().unwrap_or_default(),
                                e
                            );
                        }
                    }
                }
            }
        }
    }

    if !args.quiet {
        eprintln!();
        eprintln!(
            "{} Exported {} files ({} failed)",
            color("32", "[done]"),
            success_count,
            error_count
        );
    }

    if error_count > 0 && success_count == 0 {
        Err("All exports failed".into())
    } else {
        Ok(())
    }
}

struct ExportFileOpts<'a> {
    input: &'a Path,
    output_dir: &'a Path,
    content: &'a str,
    ext: Extracted,
    format: ExportFormat,
    include_confidence: bool,
    overwrite: bool,
    base_uri: &'a str,
}

fn export_file(opts: ExportFileOpts<'_>) -> Result<(), String> {
    let ExportFileOpts {
        input,
        output_dir,
        content,
        ext,
        format,
        include_confidence,
        overwrite,
        base_uri,
    } = opts;
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();

    // Graph CSV writes two CSVs (nodes + edges); handle before the single-path logic below.
    if matches!(format, ExportFormat::GraphCsv) {
        let nodes_path = output_dir.join(format!("{}-nodes.csv", stem));
        let edges_path = output_dir.join(format!("{}-edges.csv", stem));

        for path in [&nodes_path, &edges_path] {
            if path.exists() && !overwrite {
                return Err(format!(
                    "Output file already exists: {:?} (use --overwrite)",
                    path
                ));
            }
        }

        let (nodes_csv, edges_csv) = anno::export::to_graph_csv(
            &ext.entities,
            &ext.relations,
            &input.to_string_lossy(),
            include_confidence,
        );
        fs::write(&nodes_path, nodes_csv)
            .map_err(|e| format!("Failed to write nodes CSV: {}", e))?;
        fs::write(&edges_path, edges_csv)
            .map_err(|e| format!("Failed to write edges CSV: {}", e))?;

        return Ok(());
    }

    // Determine output filename for single-file formats
    let output_path = match format {
        ExportFormat::Brat => output_dir.join(format!("{}.ann", stem)),
        ExportFormat::Conll => output_dir.join(format!("{}.conll", stem)),
        ExportFormat::Jsonl => output_dir.join(format!("{}.jsonl", stem)),
        ExportFormat::NTriples => output_dir.join(format!("{}.nt", stem)),
        ExportFormat::JsonLd => output_dir.join(format!("{}.jsonld", stem)),
        #[cfg(feature = "graph")]
        ExportFormat::GraphNTriples => output_dir.join(format!("{}.nt", stem)),
        // GraphCsv handled above (writes two files); guard against future variants.
        _ => {
            return Err(format!(
                "Unsupported single-file export format: {:?}",
                format
            ))
        }
    };

    if output_path.exists() && !overwrite {
        return Err(format!(
            "Output file already exists: {:?} (use --overwrite)",
            output_path
        ));
    }

    let source_label = input.to_string_lossy();
    let output_content = match format {
        ExportFormat::Brat => anno::export::to_brat(content, &ext.entities, include_confidence),
        ExportFormat::Conll => anno::export::to_conll(content, &ext.entities),
        ExportFormat::Jsonl => {
            anno::export::to_jsonl(&ext.entities, &source_label, include_confidence)
        }
        ExportFormat::NTriples => {
            anno::export::to_ntriples(&ext.entities, &ext.relations, &source_label, base_uri)
        }
        ExportFormat::JsonLd => anno::export::to_jsonld(
            &ext.entities,
            &ext.relations,
            &source_label,
            include_confidence,
            base_uri,
        ),
        #[cfg(feature = "graph")]
        ExportFormat::GraphNTriples => {
            export_graph_ntriples(&ext.entities, &ext.relations, input, base_uri)
        }
        // GraphCsv handled above; guard against future variants.
        _ => {
            return Err(format!(
                "Unsupported single-file export format: {:?}",
                format
            ))
        }
    };

    fs::write(&output_path, output_content)
        .map_err(|e| format!("Failed to write output: {}", e))?;

    // brat also copies the source text alongside the .ann file
    if matches!(format, ExportFormat::Brat) {
        let txt_path = output_dir.join(format!("{}.txt", stem));
        if !txt_path.exists() || overwrite {
            fs::write(&txt_path, content)
                .map_err(|e| format!("Failed to write text file: {}", e))?;
        }
    }

    Ok(())
}

#[cfg(feature = "graph")]
fn export_graph_ntriples(
    entities: &[anno_core::Entity],
    relations: &[anno_core::Relation],
    source: &Path,
    base_uri: &str,
) -> String {
    let base = base_uri.trim_end_matches('/');
    let stem = anno_graph::uri_safe(&source.file_stem().unwrap_or_default().to_string_lossy());
    let doc_iri = format!("{}/doc/{}", base, stem);

    let kg = anno_graph::entities_to_knowledge_graph(entities, relations, &doc_iri, base_uri);
    kg.triples()
        .map(|t| t.to_ntriples())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conll_splits_trailing_period_from_entity() {
        let text = "Apple CEO Tim Cook.";
        let entities = vec![
            anno_core::Entity::new("Apple", anno_core::EntityType::Organization, 0, 5, 0.9),
            anno_core::Entity::new("Tim Cook", anno_core::EntityType::Person, 10, 18, 0.9),
        ];
        let output = anno::export::to_conll(text, &entities);
        let lines: Vec<&str> = output.lines().collect();

        // "Apple" -> B-ORG
        assert_eq!(lines[0], "Apple\tB-ORG");
        // "CEO" -> O
        assert_eq!(lines[1], "CEO\tO");
        // "Tim" -> B-PER
        assert_eq!(lines[2], "Tim\tB-PER");
        // "Cook" (without period) -> I-PER
        assert_eq!(lines[3], "Cook\tI-PER");
        // "." -> O (separate token)
        assert_eq!(lines[4], ".\tO");
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn conll_no_trailing_punct_unchanged() {
        let text = "Tim Cook spoke";
        let entities = vec![anno_core::Entity::new(
            "Tim Cook",
            anno_core::EntityType::Person,
            0,
            8,
            0.9,
        )];
        let output = anno::export::to_conll(text, &entities);
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines[0], "Tim\tB-PER");
        assert_eq!(lines[1], "Cook\tI-PER");
        assert_eq!(lines[2], "spoke\tO");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn conll_multiple_punct_chars() {
        let text = "Really?!";
        let entities: Vec<anno_core::Entity> = vec![];
        let output = anno::export::to_conll(text, &entities);
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines[0], "Really\tO");
        assert_eq!(lines[1], "?!\tO");
    }
}
