//! Export command - Export annotations to various formats
//!
//! Supports exporting to:
//! - brat standoff format (.ann files)
//! - CoNLL format (IOB/BIO tagging)
//! - JSONL (one entity per line)
//! - N-Triples (RDF triples for knowledge graphs)
//! - JSON-LD (linked data for knowledge graphs)

use clap::{Parser, ValueEnum};
use std::fs;
use std::path::{Path, PathBuf};

use super::super::output::color;
use super::super::parser::ModelBackend;

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

    /// Model backend to use for extraction
    #[arg(short, long, default_value = "stacked")]
    pub model: ModelBackend,

    /// Overwrite existing files
    #[arg(long)]
    pub overwrite: bool,

    /// Include confidence scores in output
    #[arg(long)]
    pub include_confidence: bool,

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
    /// N-Triples exported via `lattix` (graph/KG adapter; stable triple rendering).
    #[cfg(feature = "graph")]
    #[value(name = "lattix-ntriples")]
    LattixNTriples,
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

    // Create model
    let model = args.model.create_model()?;

    // Collect files to process
    let files: Vec<PathBuf> = if args.input.is_file() {
        vec![args.input.clone()]
    } else {
        fs::read_dir(&args.input)
            .map_err(|e| format!("Failed to read directory: {}", e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "txt"))
            .collect()
    };

    if files.is_empty() {
        return Err("No .txt files found in input".into());
    }

    if !args.quiet {
        eprintln!(
            "{} Exporting {} files to {:?} format",
            color("32", "[export]"),
            files.len(),
            args.format
        );
    }

    let mut success_count = 0;
    let mut error_count = 0;

    for file in &files {
        match export_file(
            file,
            &args.output,
            &*model,
            args.format,
            args.include_confidence,
            args.overwrite,
        ) {
            Ok(entity_count) => {
                success_count += 1;
                if !args.quiet {
                    eprintln!(
                        "  {} {:?} ({} entities)",
                        color("32", "✓"),
                        file.file_name().unwrap_or_default(),
                        entity_count
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

fn export_file(
    input: &Path,
    output_dir: &Path,
    model: &dyn anno::Model,
    format: ExportFormat,
    include_confidence: bool,
    overwrite: bool,
) -> Result<usize, String> {
    // Read input file
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    // Extract entities
    let entities = model
        .extract_entities(&content, None)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    let entity_count = entities.len();

    // Determine output filename
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    let output_path = match format {
        ExportFormat::Brat => output_dir.join(format!("{}.ann", stem)),
        ExportFormat::Conll => output_dir.join(format!("{}.conll", stem)),
        ExportFormat::Jsonl => output_dir.join(format!("{}.jsonl", stem)),
        ExportFormat::NTriples => output_dir.join(format!("{}.nt", stem)),
        ExportFormat::JsonLd => output_dir.join(format!("{}.jsonld", stem)),
        #[cfg(feature = "graph")]
        ExportFormat::LattixNTriples => output_dir.join(format!("{}.nt", stem)),
    };

    // Check if output exists
    if output_path.exists() && !overwrite {
        return Err(format!(
            "Output file already exists: {:?} (use --overwrite)",
            output_path
        ));
    }

    // Generate output content
    let output_content = match format {
        ExportFormat::Brat => export_brat(&entities, include_confidence),
        ExportFormat::Conll => export_conll(&content, &entities),
        ExportFormat::Jsonl => export_jsonl(&entities, input, include_confidence),
        ExportFormat::NTriples => export_ntriples(&entities, &content, input),
        ExportFormat::JsonLd => export_jsonld(&entities, &content, input, include_confidence),
        #[cfg(feature = "graph")]
        ExportFormat::LattixNTriples => export_lattix_ntriples(&entities, input),
    };

    // Write output
    fs::write(&output_path, output_content)
        .map_err(|e| format!("Failed to write output: {}", e))?;

    // For brat format, also copy the source text file
    if matches!(format, ExportFormat::Brat) {
        let txt_path = output_dir.join(format!("{}.txt", stem));
        if !txt_path.exists() || overwrite {
            fs::write(&txt_path, &content)
                .map_err(|e| format!("Failed to write text file: {}", e))?;
        }
    }

    Ok(entity_count)
}

/// Export to brat standoff format
fn export_brat(entities: &[anno_core::Entity], include_confidence: bool) -> String {
    let mut lines = Vec::new();

    for (idx, entity) in entities.iter().enumerate() {
        let tid = format!("T{}", idx + 1);
        let entity_type = entity.entity_type.as_label();

        // brat format: T1	Type Start End	Text
        let line = format!(
            "{}\t{} {} {}\t{}",
            tid, entity_type, entity.start, entity.end, entity.text
        );

        // Add confidence as attribute
        if include_confidence {
            let aid = format!("A{}", idx + 1);
            lines.push(line);
            lines.push(format!(
                "{}\tConfidence {} {:.2}",
                aid, tid, entity.confidence
            ));
            continue;
        }

        lines.push(line);
    }

    lines.join("\n")
}

/// Export to CoNLL IOB format
fn export_conll(text: &str, entities: &[anno_core::Entity]) -> String {
    let mut lines = Vec::new();
    let mut char_idx = 0;

    // Simple word tokenization
    for word in text.split_whitespace() {
        let word_start = text[char_idx..]
            .find(word)
            .map(|i| char_idx + i)
            .unwrap_or(char_idx);
        let word_end = word_start + word.len();
        char_idx = word_end;

        // Find entity covering this word
        let entity = entities.iter().find(|e| {
            // Word overlaps with entity
            word_start < e.end && word_end > e.start
        });

        let tag = match entity {
            Some(e) => {
                let is_begin = word_start <= e.start;
                if is_begin {
                    format!("B-{}", e.entity_type.as_label())
                } else {
                    format!("I-{}", e.entity_type.as_label())
                }
            }
            None => "O".to_string(),
        };

        lines.push(format!("{}\t{}", word, tag));
    }

    lines.join("\n")
}

/// Export to JSONL format
fn export_jsonl(entities: &[anno_core::Entity], source: &Path, include_confidence: bool) -> String {
    let mut lines = Vec::new();

    for entity in entities {
        let mut obj = serde_json::json!({
            "text": entity.text,
            "type": entity.entity_type.as_label(),
            "start": entity.start,
            "end": entity.end,
            "source": source.to_string_lossy(),
        });

        if include_confidence {
            obj["confidence"] = serde_json::json!(entity.confidence);
        }

        lines.push(obj.to_string());
    }

    lines.join("\n")
}

// =============================================================================
// Knowledge Graph Export Formats
// =============================================================================

/// Escape a string for N-Triples format (RDF).
fn escape_ntriples(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Generate a URI-safe identifier from text.
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

/// Export to N-Triples format (RDF triples for knowledge graphs).
///
/// Generates triples:
/// - Entity type assertions: `<entity> <rdf:type> <EntityType>`
/// - Entity text: `<entity> <rdfs:label> "text"`
/// - Entity position: `<entity> <anno:startOffset> "N"`
/// - Entity provenance: `<entity> <prov:hadPrimarySource> <document>`
///
/// For relations (when available), also generates:
/// - <head_entity> <relation_type> <tail_entity>
fn export_ntriples(entities: &[anno_core::Entity], _text: &str, source: &Path) -> String {
    let mut lines = Vec::new();
    let doc_uri = format!(
        "<http://example.org/doc/{}>",
        uri_safe(&source.to_string_lossy())
    );

    // Standard prefixes (expanded inline for N-Triples)
    let rdf_type = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>";
    let rdfs_label = "<http://www.w3.org/2000/01/rdf-schema#label>";
    let anno_ns = "http://example.org/anno#";
    let entity_ns = "http://example.org/entity/";

    for (idx, entity) in entities.iter().enumerate() {
        let entity_id = format!("e{}_{}", idx, uri_safe(&entity.text));
        let entity_uri = format!("<{}{}>", entity_ns, entity_id);
        let type_uri = format!("<{}{}Type>", anno_ns, entity.entity_type.as_label());

        // Type assertion
        lines.push(format!("{} {} {} .", entity_uri, rdf_type, type_uri));

        // Label (the actual text)
        lines.push(format!(
            "{} {} \"{}\" .",
            entity_uri,
            rdfs_label,
            escape_ntriples(&entity.text)
        ));

        // Position in document
        lines.push(format!(
            "{} <{}startOffset> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
            entity_uri, anno_ns, entity.start
        ));
        lines.push(format!(
            "{} <{}endOffset> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
            entity_uri, anno_ns, entity.end
        ));

        // Confidence score
        lines.push(format!(
            "{} <{}confidence> \"{}\"^^<http://www.w3.org/2001/XMLSchema#float> .",
            entity_uri, anno_ns, entity.confidence
        ));

        // Provenance
        lines.push(format!(
            "{} <http://www.w3.org/ns/prov#hadPrimarySource> {} .",
            entity_uri, doc_uri
        ));
    }

    // TODO: When relation extraction is integrated into Model trait,
    // add relation triples here:
    // <head_entity> <relation_type> <tail_entity> .

    lines.join("\n")
}

#[cfg(feature = "graph")]
fn escape_literal(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Export N-Triples using `lattix` as the triple renderer.
///
/// This produces N-Triples compatible with `lattix::KnowledgeGraph::from_ntriples_file(...)`.
#[cfg(feature = "graph")]
fn export_lattix_ntriples(entities: &[anno_core::Entity], source: &Path) -> String {
    use lattix::{KnowledgeGraph, Triple};

    let mut kg = KnowledgeGraph::with_capacity(entities.len().max(1), entities.len() * 8);

    let doc_iri = format!(
        "http://example.org/doc/{}",
        uri_safe(&source.to_string_lossy())
    );
    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label";
    let prov_src = "http://www.w3.org/ns/prov#hadPrimarySource";
    let anno_ns = "http://example.org/anno#";
    let entity_ns = "http://example.org/entity/";
    let xsd_int = "http://www.w3.org/2001/XMLSchema#integer";
    let xsd_float = "http://www.w3.org/2001/XMLSchema#float";

    for (idx, entity) in entities.iter().enumerate() {
        let entity_id = format!("e{}_{}", idx, uri_safe(&entity.text));
        let entity_iri = format!("{}{}", entity_ns, entity_id);
        let type_iri = format!("{}{}Type", anno_ns, entity.entity_type.as_label());

        kg.add_triple(Triple::new(
            entity_iri.as_str(),
            rdf_type,
            type_iri.as_str(),
        ));

        // Label literal
        kg.add_triple(Triple::new(
            entity_iri.as_str(),
            rdfs_label,
            format!("\"{}\"", escape_literal(&entity.text)),
        ));

        // Offsets + confidence as typed literals
        kg.add_triple(Triple::new(
            entity_iri.as_str(),
            format!("{}startOffset", anno_ns),
            format!("\"{}\"^^<{}>", entity.start, xsd_int),
        ));
        kg.add_triple(Triple::new(
            entity_iri.as_str(),
            format!("{}endOffset", anno_ns),
            format!("\"{}\"^^<{}>", entity.end, xsd_int),
        ));
        kg.add_triple(Triple::new(
            entity_iri.as_str(),
            format!("{}confidence", anno_ns),
            format!("\"{}\"^^<{}>", entity.confidence, xsd_float),
        ));

        // Provenance
        kg.add_triple(Triple::new(entity_iri.as_str(), prov_src, doc_iri.as_str()));

        // Convenience: doc mentions entity
        kg.add_triple(Triple::new(
            doc_iri.as_str(),
            format!("{}mentions", anno_ns),
            entity_iri.as_str(),
        ));
    }

    kg.triples()
        .map(|t| t.to_ntriples())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Export to JSON-LD format (linked data for knowledge graphs).
///
/// Produces a JSON-LD document with:
/// - @context with standard prefixes
/// - @graph array of entity nodes
/// - Each entity has @id, @type, label, offsets, and provenance
fn export_jsonld(
    entities: &[anno_core::Entity],
    _text: &str,
    source: &Path,
    include_confidence: bool,
) -> String {
    let entity_ns = "http://example.org/entity/";
    let anno_ns = "http://example.org/anno#";

    let mut graph = Vec::new();

    for (idx, entity) in entities.iter().enumerate() {
        let entity_id = format!("e{}_{}", idx, uri_safe(&entity.text));

        let mut node = serde_json::json!({
            "@id": format!("{}{}", entity_ns, entity_id),
            "@type": format!("{}{}Type", anno_ns, entity.entity_type.as_label()),
            "rdfs:label": entity.text,
            "anno:startOffset": entity.start,
            "anno:endOffset": entity.end,
            "prov:hadPrimarySource": {
                "@id": format!("http://example.org/doc/{}", uri_safe(&source.to_string_lossy()))
            }
        });

        if include_confidence {
            node["anno:confidence"] = serde_json::json!(entity.confidence);
        }

        graph.push(node);
    }

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
