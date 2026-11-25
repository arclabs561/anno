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
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::HashMap;
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
    ///   `--base-uri https://www.gutenberg.org/ebooks/` (Project Gutenberg texts)
    ///   `--base-uri https://dbpedia.org/resource/` (DBpedia-aligned)
    ///   --base-uri urn:anno:                            (default, stable URN prefix)
    #[arg(long, default_value = "urn:anno:")]
    pub base_uri: String,

    /// Versioned source IRI for a single RDF input. Defaults to a digest of its path and text.
    #[arg(long, value_name = "IRI", value_parser = parse_document_iri)]
    pub document_iri: Option<String>,

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

impl ExportFormat {
    fn is_rdf(self) -> bool {
        match self {
            Self::NTriples | Self::JsonLd => true,
            #[cfg(feature = "graph")]
            Self::GraphNTriples => true,
            _ => false,
        }
    }
}

fn parse_document_iri(value: &str) -> Result<String, String> {
    oxiri::Iri::parse(value)
        .map(|_| value.to_owned())
        .map_err(|error| format!("invalid absolute document IRI: {error}"))
}

/// Different source files can contain identical text. Editing one file creates a new revision.
fn file_document_iri(input: &Path, text: &str, base_uri: &str) -> Result<String, String> {
    let path = input
        .canonicalize()
        .map_err(|e| format!("Cannot identify source file {}: {e}", input.display()))?;
    #[cfg(windows)]
    let path_bytes: Vec<u8> = {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str()
            .encode_wide()
            .flat_map(u16::to_le_bytes)
            .collect()
    };
    #[cfg(not(windows))]
    let path_bytes = path.as_os_str().as_encoded_bytes();

    let mut hash = Sha256::new();
    hash.update((path_bytes.len() as u64).to_be_bytes());
    hash.update(path_bytes);
    hash.update((text.len() as u64).to_be_bytes());
    hash.update(text.as_bytes());
    let digest: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!(
        "{}/doc/sha256-{digest}",
        base_uri.trim_end_matches('/')
    ))
}

// =============================================================================
// Extraction result
// =============================================================================

struct Extracted {
    entities: Vec<anno::Entity>,
    /// Populated when the backend supports relation extraction; empty otherwise.
    relations: Vec<anno::Relation>,
}

impl Extracted {
    fn entities_only(entities: Vec<anno::Entity>) -> Self {
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

    if args.document_iri.is_some() && (!args.input.is_file() || !args.format.is_rdf()) {
        return Err("--document-iri requires one input file and an RDF export format".into());
    }

    // Collect files to process
    let files: Vec<PathBuf> = if args.input.is_file() {
        vec![args.input.clone()]
    } else {
        let mut inputs = Vec::new();
        let entries = fs::read_dir(&args.input)
            .map_err(|e| format!("Failed to read directory {}: {e}", args.input.display()))?;
        for entry in entries {
            let path = entry
                .map_err(|e| format!("Failed to list {}: {e}", args.input.display()))?
                .path();
            if !path
                .extension()
                .is_some_and(|e| e == "txt" || e == "json" || e == "jsonl")
            {
                continue;
            }
            let metadata = fs::metadata(&path)
                .map_err(|e| format!("Failed to inspect {}: {e}", path.display()))?;
            if metadata.is_file() {
                inputs.push(path);
            }
        }
        inputs
    };

    if files.is_empty() {
        return Err("No .txt/.json/.jsonl files found in input".into());
    }

    // A format appends the same suffixes to every stem, including both CSV or brat files.
    // --overwrite may replace an earlier run, but two inputs must never replace one another.
    let mut by_output_stem = HashMap::new();
    for file in &files {
        let stem = output_stem(file);
        if let Some(previous) = by_output_stem.insert(stem, file) {
            return Err(format!(
                "Inputs {previous:?} and {file:?} map to the same output name. Rename one or export them to separate directories."
            ));
        }
    }

    // Create output directory
    if !args.output.exists() {
        fs::create_dir_all(&args.output)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    // Check every destination before touching any output. Existing hard links
    // and symlinks can name an input even when their path strings differ.
    let sources = files
        .iter()
        .map(|path| file_identity(path))
        .collect::<std::io::Result<std::collections::HashSet<_>>>()
        .map_err(|e| format!("Cannot identify export inputs: {e}"))?;
    for file in &files {
        for path in output_paths(file, &args.output, args.format) {
            match fs::symlink_metadata(&path) {
                Ok(_) => {
                    let target = file_identity(&path)
                        .map_err(|e| format!("Cannot identify output {path:?}: {e}"))?;
                    if matches!(args.format, ExportFormat::Brat)
                        && path.extension().is_some_and(|ext| ext == "txt")
                        && file.extension().is_some_and(|ext| ext == "txt")
                        && file_identity(file).map_err(|e| e.to_string())? == target
                    {
                        continue; // The source already is the brat text companion.
                    }
                    if sources.contains(&target) {
                        return Err(format!("Output {path:?} aliases an input file"));
                    }
                    if !args.overwrite || !path.is_file() {
                        return Err(format!("Output already exists: {path:?}"));
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("Cannot inspect output {path:?}: {e}")),
            }
        }
    }

    // Choose extraction path: relation-capable model vs plain model.
    let relation_model = args.model.try_create_relation_model().transpose()?;
    let plain_model = if relation_model.is_none() {
        Some(args.model.create_model()?)
    } else {
        None
    };

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
            let entities: Vec<anno::Entity> = doc
                .signals()
                .iter()
                .filter_map(|s| {
                    let (start, end) = s.text_offsets()?;
                    let mut entity = anno::Entity::new(
                        s.surface(),
                        s.label.to_entity_type(),
                        start,
                        end,
                        s.confidence.value(),
                    );
                    if let anno::Location::Discontinuous { segments } = &s.location {
                        entity.set_discontinuous_span(anno::DiscontinuousSpan::new(
                            segments.iter().map(|&(start, end)| start..end).collect(),
                        ));
                    }
                    Some(entity)
                })
                .collect();
            Ok((doc.text().to_owned(), Extracted::entities_only(entities)))
        } else if let Some(ref rm) = relation_model {
            let content =
                fs::read_to_string(file).map_err(|e| format!("Failed to read file: {}", e))?;
            let (entities, relations) = rm
                .extract_relations_default(&content)
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
                    document_iri: args.document_iri.as_deref(),
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

    if error_count > 0 {
        Err(format!(
            "{error_count} exports failed, {success_count} succeeded"
        ))
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
    document_iri: Option<&'a str>,
}

fn output_stem(input: &Path) -> Cow<'_, str> {
    input.file_stem().unwrap_or_default().to_string_lossy()
}

// Unix metadata identifies hard links without holding a descriptor per input.
#[cfg(unix)]
fn file_identity(path: &Path) -> std::io::Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::metadata(path)?;
    Ok((meta.dev(), meta.ino()))
}

#[cfg(not(unix))]
fn file_identity(path: &Path) -> std::io::Result<same_file::Handle> {
    same_file::Handle::from_path(path)
}

pub(super) fn write_output(
    path: &Path,
    content: impl AsRef<[u8]>,
    overwrite: bool,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut staged = tempfile::NamedTempFile::new_in(path.parent().unwrap_or(Path::new(".")))?;
    staged.write_all(content.as_ref())?;
    if overwrite {
        staged.persist(path)?;
    } else {
        staged.persist_noclobber(path)?;
    }
    Ok(())
}

fn output_paths(input: &Path, dir: &Path, format: ExportFormat) -> Vec<PathBuf> {
    let suffixes: &[&str] = match format {
        ExportFormat::Brat => &[".ann", ".txt"],
        ExportFormat::Conll => &[".conll"],
        ExportFormat::Jsonl => &[".jsonl"],
        ExportFormat::NTriples => &[".nt"],
        ExportFormat::JsonLd => &[".jsonld"],
        ExportFormat::GraphCsv => &["-nodes.csv", "-edges.csv"],
        #[cfg(feature = "graph")]
        ExportFormat::GraphNTriples => &[".nt"],
    };
    let stem = output_stem(input);
    suffixes
        .iter()
        .map(|suffix| dir.join(format!("{stem}{suffix}")))
        .collect()
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
        document_iri,
    } = opts;
    let paths = output_paths(input, output_dir, format);

    if matches!(format, ExportFormat::GraphCsv) {
        let nodes_path = &paths[0];
        let edges_path = &paths[1];
        let (nodes_csv, edges_csv) = anno::export::to_graph_csv(
            &ext.entities,
            &ext.relations,
            &input.to_string_lossy(),
            include_confidence,
        );
        write_output(nodes_path, nodes_csv, overwrite)
            .map_err(|e| format!("Failed to write nodes CSV: {}", e))?;
        write_output(edges_path, edges_csv, overwrite)
            .map_err(|e| format!("Failed to write edges CSV: {}", e))?;

        return Ok(());
    }

    let output_path = &paths[0];
    if output_path.exists() && !overwrite {
        return Err(format!(
            "Output file already exists: {:?} (use --overwrite)",
            output_path
        ));
    }

    let source_label = input.to_string_lossy();
    let source_iri = || match document_iri {
        Some(iri) => Ok(iri.to_owned()),
        None => file_document_iri(input, content, base_uri),
    };
    let output_content = match format {
        ExportFormat::Brat => anno::export::to_brat(content, &ext.entities, include_confidence),
        ExportFormat::Conll => {
            if ext.entities.iter().any(anno::Entity::is_discontinuous) {
                return Err("CoNLL BIO cannot represent a discontinuous mention. Use brat to preserve its segments.".into());
            }
            anno::export::to_conll(content, &ext.entities)
        }
        ExportFormat::Jsonl => {
            anno::export::to_jsonl(&ext.entities, &source_label, include_confidence)
        }
        ExportFormat::NTriples => anno::export::to_ntriples_for_document(
            &ext.entities,
            &ext.relations,
            &source_iri()?,
            base_uri,
        ),
        ExportFormat::JsonLd => anno::export::to_jsonld_for_document(
            &ext.entities,
            &ext.relations,
            &source_iri()?,
            include_confidence,
            base_uri,
        ),
        #[cfg(feature = "graph")]
        ExportFormat::GraphNTriples => {
            export_graph_ntriples(&ext.entities, &ext.relations, &source_iri()?, base_uri)
        }
        // GraphCsv handled above; guard against future variants.
        _ => {
            return Err(format!(
                "Unsupported single-file export format: {:?}",
                format
            ))
        }
    };

    write_output(output_path, output_content, overwrite)
        .map_err(|e| format!("Failed to write output: {}", e))?;

    if matches!(format, ExportFormat::Brat) {
        let companion = &paths[1];
        let is_source = input.extension().is_some_and(|ext| ext == "txt")
            && companion.exists()
            && same_file::is_same_file(input, companion).map_err(|e| e.to_string())?;
        if !is_source {
            write_output(companion, content, overwrite)
                .map_err(|e| format!("Failed to write text file: {e}"))?;
        }
    }

    Ok(())
}

#[cfg(feature = "graph")]
fn export_graph_ntriples(
    entities: &[anno::Entity],
    relations: &[anno::Relation],
    doc_iri: &str,
    base_uri: &str,
) -> String {
    let kg = anno::graph::entities_to_knowledge_graph(entities, relations, doc_iri, base_uri);
    kg.triples()
        .map(|t| t.to_ntriples())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn file_identity_preserves_non_utf8_paths() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join(OsStr::from_bytes(b"report-\x80.txt"));
        let second = dir.path().join(OsStr::from_bytes(b"report-\x81.txt"));
        fs::write(&first, "Ada").unwrap();
        fs::write(&second, "Ada").unwrap();
        assert_eq!(first.to_string_lossy(), second.to_string_lossy());
        assert_ne!(
            file_document_iri(&first, "Ada", "urn:anno:").unwrap(),
            file_document_iri(&second, "Ada", "urn:anno:").unwrap()
        );
    }

    #[test]
    fn conll_splits_trailing_period_from_entity() {
        let text = "Apple CEO Tim Cook.";
        let entities = vec![
            anno::Entity::new("Apple", anno::EntityType::Organization, 0, 5, 0.9),
            anno::Entity::new("Tim Cook", anno::EntityType::Person, 10, 18, 0.9),
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
        let entities = vec![anno::Entity::new(
            "Tim Cook",
            anno::EntityType::Person,
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
        let entities: Vec<anno::Entity> = vec![];
        let output = anno::export::to_conll(text, &entities);
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines[0], "Really\tO");
        assert_eq!(lines[1], "?!\tO");
    }
}
