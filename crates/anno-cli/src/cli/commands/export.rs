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
//! When the selected backend is `RelationCapable` (e.g. `tplinker`), graph-oriented formats
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
    /// With a RelationCapable model (e.g. `--model tplinker`), emits real semantic triples.
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
    /// Populated when the backend is RelationCapable; empty otherwise.
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
                .signals
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
            Ok((doc.text, Extracted::entities_only(entities)))
        } else if let Some(ref rm) = relation_model {
            let content =
                fs::read_to_string(file).map_err(|e| format!("Failed to read file: {}", e))?;
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

// =============================================================================
// Non-graph formats
// =============================================================================

fn export_brat(entities: &[anno_core::Entity], include_confidence: bool) -> String {
    let mut lines = Vec::new();
    for (idx, entity) in entities.iter().enumerate() {
        let tid = format!("T{}", idx + 1);
        let line = format!(
            "{}\t{} {} {}\t{}",
            tid,
            entity.entity_type.as_label(),
            entity.start(),
            entity.end(),
            entity.text
        );
        if include_confidence {
            let aid = format!("A{}", idx + 1);
            lines.push(line);
            lines.push(format!(
                "{}\tConfidence {} {:.2}",
                aid, tid, entity.confidence
            ));
        } else {
            lines.push(line);
        }
    }
    lines.join("\n")
}

fn export_conll(text: &str, entities: &[anno_core::Entity]) -> String {
    let mut lines = Vec::new();
    let mut char_idx = 0;
    for word in text.split_whitespace() {
        let word_start = text[char_idx..]
            .find(word)
            .map(|i| char_idx + i)
            .unwrap_or(char_idx);
        let word_end = word_start + word.len();
        char_idx = word_end;

        // Check if the whole word (including trailing punctuation) falls inside an entity span.
        let entity_full = entities
            .iter()
            .find(|e| word_start < e.end() && word_end > e.start());

        // Only split trailing punctuation when it's NOT inside an entity span.
        let trimmed = word.trim_end_matches(['.', ',', ';', ':', '!', '?', ')', ']']);
        let punct = &word[trimmed.len()..];
        let inside_entity = entity_full.map(|e| word_end <= e.end()).unwrap_or(false);

        if inside_entity {
            // Entire word (with punctuation) is inside entity -- keep together.
            let tag = match entity_full {
                Some(e) => {
                    if word_start <= e.start() {
                        format!("B-{}", e.entity_type.as_label())
                    } else {
                        format!("I-{}", e.entity_type.as_label())
                    }
                }
                None => "O".to_string(),
            };
            lines.push(format!("{}\t{}", word, tag));
        } else {
            let trimmed_end = word_start + trimmed.len();
            if !trimmed.is_empty() {
                let entity = entities
                    .iter()
                    .find(|e| word_start < e.end() && trimmed_end > e.start());
                let tag = match entity {
                    Some(e) => {
                        if word_start <= e.start() {
                            format!("B-{}", e.entity_type.as_label())
                        } else {
                            format!("I-{}", e.entity_type.as_label())
                        }
                    }
                    None => "O".to_string(),
                };
                lines.push(format!("{}\t{}", trimmed, tag));
            }

            if !punct.is_empty() {
                lines.push(format!("{}\tO", punct));
            }
        }
    }
    lines.join("\n")
}

fn export_jsonl(entities: &[anno_core::Entity], source: &Path, include_confidence: bool) -> String {
    entities
        .iter()
        .map(|e| {
            let mut obj = serde_json::json!({
                "text": e.text,
                "type": e.entity_type.as_label(),
                "start": e.start(),
                "end": e.end(),
                "source": source.to_string_lossy(),
            });
            if include_confidence {
                obj["confidence"] = serde_json::json!(e.confidence);
            }
            obj.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// =============================================================================
// Shared RDF helpers
// =============================================================================

fn escape_ntriples(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn uri_safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn doc_uri(base_uri: &str, source: &Path) -> String {
    let base = base_uri.trim_end_matches('/');
    let stem = uri_safe(&source.file_stem().unwrap_or_default().to_string_lossy());
    format!("<{}/doc/{}>", base, stem)
}

fn entity_uri(base_uri: &str, entity_type: &str, idx: usize, text: &str, start: usize) -> String {
    let base = base_uri.trim_end_matches('/');
    format!(
        "<{}/entity/{}/{}_{}_{}>",
        base,
        entity_type.to_lowercase(),
        idx,
        uri_safe(text),
        start,
    )
}

fn rel_predicate_uri(base_uri: &str, rel_type: &str) -> String {
    let base = base_uri.trim_end_matches('/');
    format!("<{}/rel/{}>", base, uri_safe(rel_type))
}

// =============================================================================
// N-Triples (plain, no lattix)
// =============================================================================

fn export_ntriples(
    entities: &[anno_core::Entity],
    relations: &[anno_core::Relation],
    source: &Path,
    base_uri: &str,
) -> String {
    let mut lines = Vec::new();
    let doc = doc_uri(base_uri, source);
    let base = base_uri.trim_end_matches('/');
    let anno_ns = format!("{}/vocab#", base);

    let rdf_type = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>";
    let rdfs_label = "<http://www.w3.org/2000/01/rdf-schema#label>";

    // Index entities by (start, type) so we can resolve relation head/tail to URIs.
    let ent_uri: Vec<String> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| entity_uri(base_uri, e.entity_type.as_label(), i, &e.text, e.start()))
        .collect();

    for (idx, entity) in entities.iter().enumerate() {
        let ent = &ent_uri[idx];
        let type_uri = format!("<{}{}Type>", anno_ns, entity.entity_type.as_label());
        lines.push(format!("{} {} {} .", ent, rdf_type, type_uri));
        lines.push(format!(
            "{} {} \"{}\" .",
            ent,
            rdfs_label,
            escape_ntriples(&entity.text)
        ));
        lines.push(format!(
            "{} <{}startOffset> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
            ent,
            anno_ns,
            entity.start()
        ));
        lines.push(format!(
            "{} <{}endOffset> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
            ent,
            anno_ns,
            entity.end()
        ));
        lines.push(format!(
            "{} <{}confidence> \"{}\"^^<http://www.w3.org/2001/XMLSchema#float> .",
            ent, anno_ns, entity.confidence
        ));
        lines.push(format!(
            "{} <http://www.w3.org/ns/prov#hadPrimarySource> {} .",
            ent, doc
        ));
    }

    // Semantic relation triples (available when backend is RelationCapable).
    // Head and tail entities are identified by their position in the entity list.
    for rel in relations {
        // Find matching entity URIs by matching head/tail text + offsets.
        let head_uri = ent_uri.iter().zip(entities.iter()).find_map(|(u, e)| {
            (e.text == rel.head.text && e.start() == rel.head.start()).then_some(u.as_str())
        });
        let tail_uri = ent_uri.iter().zip(entities.iter()).find_map(|(u, e)| {
            (e.text == rel.tail.text && e.start() == rel.tail.start()).then_some(u.as_str())
        });
        if let (Some(h), Some(t)) = (head_uri, tail_uri) {
            let pred = rel_predicate_uri(base_uri, &rel.relation_type);
            lines.push(format!("{} {} {} .", h, pred, t));
        }
    }

    lines.join("\n")
}

// =============================================================================
// JSON-LD
// =============================================================================

fn export_jsonld(
    entities: &[anno_core::Entity],
    relations: &[anno_core::Relation],
    source: &Path,
    include_confidence: bool,
    base_uri: &str,
) -> String {
    let base = base_uri.trim_end_matches('/');
    let entity_ns = format!("{}/entity", base);
    let anno_ns = format!("{}/vocab#", base);
    let doc_stem = uri_safe(&source.file_stem().unwrap_or_default().to_string_lossy());

    // Build entity ID map for relation resolution.
    let entity_ids: Vec<String> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            format!(
                "{}/{}/{}_{}_{}",
                entity_ns,
                e.entity_type.as_label().to_lowercase(),
                i,
                uri_safe(&e.text),
                e.start(),
            )
        })
        .collect();

    // Group relation triples by head entity ID.
    let mut rel_by_head: std::collections::HashMap<&str, Vec<serde_json::Value>> =
        std::collections::HashMap::new();
    for rel in relations {
        let head_id = entity_ids.iter().zip(entities.iter()).find_map(|(id, e)| {
            (e.text == rel.head.text && e.start() == rel.head.start()).then_some(id.as_str())
        });
        let tail_id = entity_ids.iter().zip(entities.iter()).find_map(|(id, e)| {
            (e.text == rel.tail.text && e.start() == rel.tail.start()).then_some(id.as_str())
        });
        if let (Some(h), Some(t)) = (head_id, tail_id) {
            rel_by_head.entry(h).or_default().push(serde_json::json!({
                "@type": format!("{}/rel/{}", base, uri_safe(&rel.relation_type)),
                "target": { "@id": t }
            }));
        }
    }

    let graph: Vec<serde_json::Value> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let id = &entity_ids[i];
            let mut node = serde_json::json!({
                "@id": id,
                "@type": format!("{}{}Type", anno_ns, e.entity_type.as_label()),
                "rdfs:label": e.text,
                "anno:startOffset": e.start(),
                "anno:endOffset": e.end(),
                "prov:hadPrimarySource": {
                    "@id": format!("{}/doc/{}", base, doc_stem)
                }
            });
            if include_confidence {
                node["anno:confidence"] = serde_json::json!(e.confidence);
            }
            if let Some(rels) = rel_by_head.get(id.as_str()) {
                node["anno:relations"] = serde_json::json!(rels);
            }
            node
        })
        .collect();

    let doc = serde_json::json!({
        "@context": {
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "prov": "http://www.w3.org/ns/prov#",
            "anno": anno_ns,
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        },
        "@graph": graph
    });

    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string())
}

// =============================================================================
// graph-ntriples (via anno-graph substrate)
// =============================================================================

/// N-Triples rendered via `anno-graph`.
///
/// All triple construction (offset/confidence/provenance triples, relation arcs) lives in
/// `anno-graph::entities_to_knowledge_graph` — this function is just the CLI glue.
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

// =============================================================================
// Graph CSV
// =============================================================================

/// Export entities as graph-compatible CSV node and edge tables.
///
/// Returns `(nodes_csv, edges_csv)`.
///
/// When the backend is `RelationCapable` (e.g. `--model tplinker`), edges are extracted
/// semantic triples. Otherwise they fall back to co-occurrence pairs (within 200 chars).
fn export_graph_csv(
    entities: &[anno_core::Entity],
    relations: &[anno_core::Relation],
    source: &Path,
    include_confidence: bool,
) -> (String, String) {
    let source_str = source.to_string_lossy();

    // Entity IDs: stable per-document keys.
    let entity_ids: Vec<String> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            format!(
                "{}:{}_{}",
                e.entity_type.as_label().to_lowercase(),
                i,
                uri_safe(&e.text)
            )
        })
        .collect();

    // Nodes CSV
    let mut nodes = String::from("id,entity_type,text,start,end,source");
    if include_confidence {
        nodes.push_str(",confidence");
    }
    nodes.push('\n');
    for (i, e) in entities.iter().enumerate() {
        nodes.push_str(&format!(
            "{},{},{},{},{},{}",
            csv_escape(&entity_ids[i]),
            csv_escape(e.entity_type.as_label()),
            csv_escape(&e.text),
            e.start(),
            e.end(),
            csv_escape(&source_str),
        ));
        if include_confidence {
            nodes.push_str(&format!(",{:.4}", e.confidence));
        }
        nodes.push('\n');
    }

    // Edges CSV: semantic relations when available, co-occurrence fallback otherwise.
    let mut edges = String::from("from,to,rel_type,confidence\n");

    if !relations.is_empty() {
        // Real semantic edges from a RelationCapable backend.
        for rel in relations {
            let head_id = entity_ids.iter().zip(entities.iter()).find_map(|(id, e)| {
                (e.text == rel.head.text && e.start() == rel.head.start()).then_some(id.as_str())
            });
            let tail_id = entity_ids.iter().zip(entities.iter()).find_map(|(id, e)| {
                (e.text == rel.tail.text && e.start() == rel.tail.start()).then_some(id.as_str())
            });
            if let (Some(h), Some(t)) = (head_id, tail_id) {
                edges.push_str(&format!(
                    "{},{},{},{:.4}\n",
                    csv_escape(h),
                    csv_escape(t),
                    csv_escape(&rel.relation_type),
                    rel.confidence,
                ));
            }
        }
    } else {
        // Co-occurrence fallback: entities within 200 chars in the same document.
        const COOCCUR_WINDOW: usize = 200;
        for (i, a) in entities.iter().enumerate() {
            for (j, b) in entities.iter().enumerate().skip(i + 1) {
                let distance = if a.end() <= b.start() {
                    b.start().saturating_sub(a.end())
                } else if b.end() <= a.start() {
                    a.start().saturating_sub(b.end())
                } else {
                    0
                };
                if distance <= COOCCUR_WINDOW {
                    edges.push_str(&format!(
                        "{},{},CO_OCCURS,1.0000\n",
                        csv_escape(&entity_ids[i]),
                        csv_escape(&entity_ids[j]),
                    ));
                }
            }
        }
    }

    (nodes, edges)
}

/// Minimal CSV field escaping.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
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
        let output = export_conll(text, &entities);
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
        let output = export_conll(text, &entities);
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
        let output = export_conll(text, &entities);
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines[0], "Really\tO");
        assert_eq!(lines[1], "?!\tO");
    }
}
