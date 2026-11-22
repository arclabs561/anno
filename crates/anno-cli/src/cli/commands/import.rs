//! Import command - Import annotations from various formats for training/evaluation
//!
//! Supported formats:
//! - brat standoff (.ann files)
//! - CoNLL format (IOB/BIO tagging)
//! - JSONL (one entity per line)
//! - N-Triples (.nt) — round-trips with `anno export --format ntriples`
//! - JSON-LD (.jsonld) — round-trips with `anno export --format jsonld`

use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};

use super::super::output::color;
use super::export::{write_output, ExportFormat};

/// Import annotations from different formats
#[derive(Parser, Debug)]
pub struct ImportArgs {
    /// Input file or directory
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// JSONL destination, replaced atomically after successful import
    #[arg(short, long, value_name = "PATH")]
    pub output: PathBuf,

    /// Import format
    #[arg(short, long, default_value = "brat")]
    pub format: ExportFormat,

    /// Include text file content in output
    #[arg(long)]
    pub include_text: bool,

    /// Skip files that fail to import and write the successful files' annotations.
    /// Directory listing errors and an entirely failed import still abort.
    #[arg(long)]
    pub allow_partial: bool,

    /// Suppress progress messages
    #[arg(short, long)]
    pub quiet: bool,
}

/// Imported annotation from external format.
#[derive(Debug, Clone)]
pub struct ImportedAnnotation {
    /// Entity text
    pub text: String,
    /// Entity type label
    pub entity_type: String,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
    /// Source annotation system
    pub source: String,
    /// Optional confidence score
    pub confidence: Option<f64>,
}

/// Run the import command.
pub fn run(args: ImportArgs) -> Result<(), String> {
    // Validate input
    if !args.input.exists() {
        return Err(format!("Input not found: {:?}", args.input));
    }

    // Collect files to process
    let files: Vec<PathBuf> = if args.input.is_file() {
        vec![args.input.clone()]
    } else {
        let ext = match args.format {
            ExportFormat::Brat => "ann",
            ExportFormat::Conll => "conll",
            ExportFormat::Jsonl => "jsonl",
            ExportFormat::NTriples => "nt",
            ExportFormat::JsonLd => "jsonld",
            #[cfg(feature = "graph")]
            ExportFormat::GraphNTriples => "nt",
            ExportFormat::GraphCsv => "csv",
        };
        let output_exists = args
            .output
            .try_exists()
            .map_err(|e| format!("Failed to inspect output: {e}"))?;
        let mut inputs = Vec::new();
        let entries = fs::read_dir(&args.input)
            .map_err(|e| format!("Failed to read directory {}: {e}", args.input.display()))?;
        for entry in entries {
            let path = entry
                .map_err(|e| format!("Failed to list {}: {e}", args.input.display()))?
                .path();
            if path.extension().is_none_or(|e| e != ext) {
                continue;
            }
            let metadata = fs::metadata(&path)
                .map_err(|e| format!("Failed to inspect {}: {e}", path.display()))?;
            if !metadata.is_file() {
                continue;
            }
            // A previous output is not input on the next directory import.
            if output_exists
                && same_file::is_same_file(&path, &args.output)
                    .map_err(|e| format!("Failed to identify input {}: {e}", path.display()))?
            {
                let parent = args
                    .output
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                if path.file_name() != args.output.file_name()
                    || !same_file::is_same_file(&args.input, parent)
                        .map_err(|e| format!("Failed to identify output directory: {e}"))?
                {
                    return Err(format!("Output aliases an input file: {}", path.display()));
                }
                continue;
            }
            inputs.push(path);
        }
        inputs
    };

    if files.is_empty() {
        return Err("No annotation files found in input".into());
    }

    if !args.quiet {
        eprintln!(
            "{} Importing {} files from {:?} format",
            color("32", "[import]"),
            files.len(),
            args.format
        );
    }

    let mut all_annotations = Vec::new();
    let mut success_count = 0;

    for file in &files {
        match import_file(file, args.format, args.include_text) {
            Ok(annotations) => {
                let count = annotations.len();
                all_annotations.extend(annotations);
                success_count += 1;
                if !args.quiet {
                    eprintln!(
                        "  {} {:?} ({} annotations)",
                        color("32", "✓"),
                        file.file_name().unwrap_or_default(),
                        count
                    );
                }
            }
            Err(e) if args.allow_partial => eprintln!("Skipping {}: {e}", file.display()),
            Err(e) => return Err(format!("Failed to import {}: {e}", file.display())),
        }
    }

    if success_count == 0 {
        return Err("All imports failed".into());
    }

    // Write output
    let output_content: String = all_annotations
        .iter()
        .map(|a| {
            serde_json::json!({
                "text": a.text,
                "type": a.entity_type,
                "start": a.start,
                "end": a.end,
                "source": a.source,
                "confidence": a.confidence,
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");

    write_output(&args.output, output_content, true)
        .map_err(|e| format!("Failed to write output {}: {e}", args.output.display()))?;

    if !args.quiet {
        eprintln!();
        eprintln!(
            "{} Imported {} annotations from {} files to {:?}",
            color("32", "[done]"),
            all_annotations.len(),
            success_count,
            args.output
        );
    }

    Ok(())
}

fn import_file(
    input: &PathBuf,
    format: ExportFormat,
    include_text: bool,
) -> Result<Vec<ImportedAnnotation>, String> {
    match format {
        ExportFormat::Brat => import_brat(input, include_text),
        ExportFormat::Conll => import_conll(input),
        ExportFormat::Jsonl => import_jsonl(input),
        ExportFormat::NTriples => import_ntriples(input),
        ExportFormat::JsonLd => import_jsonld(input),
        #[cfg(feature = "graph")]
        ExportFormat::GraphNTriples => import_ntriples(input),
        ExportFormat::GraphCsv => {
            Err("Import from `graph-csv` format is not yet supported. Use jsonl or brat.".into())
        }
    }
}

/// Import from brat standoff format
fn import_brat(input: &PathBuf, include_text: bool) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    // Try to read corresponding .txt file for entity text
    let txt_path = input.with_extension("txt");
    let txt_content = if include_text && txt_path.exists() {
        Some(fs::read_to_string(&txt_path).ok())
    } else {
        None
    };
    let txt_content = txt_content.flatten();

    let mut annotations = Vec::new();
    let mut confidences: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    // First pass: collect confidences from attributes
    for line in content.lines() {
        if line.starts_with('A') {
            // Attribute line: A1	Confidence T1 0.85
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 && parts[1].starts_with("Confidence") {
                let attr_parts: Vec<&str> = parts[1].split_whitespace().collect();
                if attr_parts.len() >= 3 {
                    let tid = attr_parts[1];
                    if let Ok(conf) = attr_parts[2].parse::<f64>() {
                        confidences.insert(tid.to_string(), conf);
                    }
                }
            }
        }
    }

    // Second pass: parse entity annotations
    for line in content.lines() {
        if line.starts_with('T') {
            // Entity line: T1	Type Start End	Text
            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            if parts.len() >= 3 {
                let tid = parts[0];
                let type_span: Vec<&str> = parts[1].split_whitespace().collect();
                if type_span.len() >= 3 {
                    let entity_type = type_span[0].to_string();
                    let start: usize = type_span[1].parse().map_err(|_| "Invalid start offset")?;
                    let end: usize = type_span[2].parse().map_err(|_| "Invalid end offset")?;

                    // Get text from annotation or from txt file
                    let text = if parts.len() > 2 && !parts[2].is_empty() {
                        parts[2].to_string()
                    } else if let Some(ref txt) = txt_content {
                        txt.chars().skip(start).take(end - start).collect()
                    } else {
                        format!("[{}:{}]", start, end)
                    };

                    annotations.push(ImportedAnnotation {
                        text,
                        entity_type,
                        start,
                        end,
                        source: input.to_string_lossy().to_string(),
                        confidence: confidences.get(tid).copied(),
                    });
                }
            }
        }
    }

    Ok(annotations)
}

/// Import from CoNLL IOB format
fn import_conll(input: &PathBuf) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let mut annotations: Vec<ImportedAnnotation> = Vec::new();
    let mut inside_entity = false;
    let mut char_idx = 0;

    for line in content.lines() {
        let mut fields = line.split_whitespace();
        let Some(word) = fields.next() else {
            inside_entity = false;
            continue;
        };
        if word == "-DOCSTART-" {
            inside_entity = false;
            continue;
        }
        let Some(second) = fields.next() else {
            continue;
        };
        // Anno exports word/tag pairs. CoNLL-2003 puts the NER tag last,
        // after the part-of-speech and chunk tags.
        let tag = if second == "O" || second.starts_with("B-") || second.starts_with("I-") {
            second
        } else {
            fields.next_back().unwrap_or(second)
        };
        let start = char_idx;
        let end = start + word.chars().count();
        char_idx = end + 1;

        if let Some(label) = tag.strip_prefix("I-") {
            if let Some(entity) = annotations.last_mut() {
                if inside_entity && entity.entity_type == label {
                    entity.text.push(' ');
                    entity.text.push_str(word);
                    entity.end = end;
                    continue;
                }
            }
        }
        let label = tag.strip_prefix("B-").or_else(|| tag.strip_prefix("I-"));
        inside_entity = label.is_some();
        if let Some(label) = label {
            annotations.push(ImportedAnnotation {
                text: word.to_string(),
                entity_type: label.to_string(),
                start,
                end,
                source: input.to_string_lossy().into_owned(),
                confidence: None,
            });
        }
    }

    Ok(annotations)
}

/// Import from JSONL format
fn import_jsonl(input: &PathBuf) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let mut annotations = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("Invalid JSON: {}", e))?;

        annotations.push(ImportedAnnotation {
            text: obj["text"].as_str().unwrap_or("").to_string(),
            entity_type: obj["type"].as_str().unwrap_or("").to_string(),
            start: obj["start"].as_u64().unwrap_or(0) as usize,
            end: obj["end"].as_u64().unwrap_or(0) as usize,
            source: obj["source"]
                .as_str()
                .unwrap_or(&input.to_string_lossy())
                .to_string(),
            confidence: obj["confidence"].as_f64(),
        });
    }

    Ok(annotations)
}

// =============================================================================
// N-Triples import
// =============================================================================

/// Import entity annotations from N-Triples (`.nt`) format.
///
/// Parses triples produced by `anno export --format ntriples` or `--format graph-ntriples`.
/// Groups triples by subject IRI, then extracts entity type, surface text, character offsets,
/// confidence, and provenance from the standard predicates written by `anno export`.
fn import_ntriples(input: &PathBuf) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let mut triples: Vec<(String, String, String)> = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(t) = parse_nt_line(line) {
            triples.push(t);
        }
    }

    // Group by subject.
    let mut by_subject: std::collections::HashMap<&str, Vec<(&str, &str)>> =
        std::collections::HashMap::new();
    for (s, p, o) in &triples {
        by_subject
            .entry(s.as_str())
            .or_default()
            .push((p.as_str(), o.as_str()));
    }

    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
    const PROV_SRC: &str = "http://www.w3.org/ns/prov#hadPrimarySource";

    let mut annotations = Vec::new();

    for pairs in by_subject.values() {
        // Skip document nodes (no rdfs:label).
        let label = pairs
            .iter()
            .find(|(p, _)| *p == RDFS_LABEL)
            .map(|(_, o)| unescape_literal(strip_literal(o)));
        let Some(text) = label else { continue };

        // `…vocab#PERType` → `PER`
        let entity_type = pairs
            .iter()
            .find(|(p, _)| *p == RDF_TYPE)
            .map(|(_, o)| {
                let iri = o.trim_start_matches('<').trim_end_matches('>');
                let after_hash = iri.rsplit('#').next().unwrap_or(iri);
                after_hash
                    .strip_suffix("Type")
                    .unwrap_or(after_hash)
                    .to_string()
            })
            .unwrap_or_else(|| "ENTITY".to_string());

        let start = pairs
            .iter()
            .find(|(p, _)| p.ends_with("startOffset"))
            .and_then(|(_, o)| strip_literal(o).parse::<usize>().ok())
            .unwrap_or(0);

        let end = pairs
            .iter()
            .find(|(p, _)| p.ends_with("endOffset"))
            .and_then(|(_, o)| strip_literal(o).parse::<usize>().ok())
            .unwrap_or(0);

        let confidence = pairs
            .iter()
            .find(|(p, _)| p.ends_with("confidence"))
            .and_then(|(_, o)| strip_literal(o).parse::<f64>().ok());

        let source = pairs
            .iter()
            .find(|(p, _)| *p == PROV_SRC)
            .map(|(_, o)| o.trim_start_matches('<').trim_end_matches('>').to_string())
            .unwrap_or_default();

        if !text.is_empty() && end > start {
            annotations.push(ImportedAnnotation {
                text,
                entity_type,
                start,
                end,
                source,
                confidence,
            });
        }
    }

    annotations.sort_unstable_by_key(|a| a.start);
    Ok(annotations)
}

/// Parse one N-Triples line into a (subject, predicate, object) triple.
fn parse_nt_line(line: &str) -> Option<(String, String, String)> {
    let line = line.strip_suffix(" .").or_else(|| line.strip_suffix('.'))?;
    let line = line.trim();

    let (s, rest) = parse_iri_or_bnode(line)?;
    let rest = rest.trim_start();
    let (p, rest) = parse_iri_or_bnode(rest)?;
    let rest = rest.trim_start();

    let o = if rest.starts_with('<') {
        let end = rest.find('>').unwrap_or(rest.len() - 1);
        rest[1..end].to_string()
    } else {
        rest.to_string()
    };

    Some((s, p, o))
}

fn parse_iri_or_bnode(s: &str) -> Option<(String, &str)> {
    if s.starts_with('<') {
        let end = s.find('>')?;
        Some((s[1..end].to_string(), &s[end + 1..]))
    } else if s.starts_with("_:") {
        let end = s.find(|c: char| c.is_whitespace()).unwrap_or(s.len());
        Some((s[..end].to_string(), &s[end..]))
    } else {
        None
    }
}

/// Extract value from an N-Triples literal: `"foo"^^<type>` → `"foo"`.
fn strip_literal(o: &str) -> &str {
    let o = o.trim();
    if let Some(inner) = o.strip_prefix('"') {
        inner.rfind('"').map(|i| &inner[..i]).unwrap_or(inner)
    } else {
        o.trim_start_matches('<').trim_end_matches('>')
    }
}

fn unescape_literal(s: &str) -> String {
    // Consume each exporter escape once. A literal backslash followed by n
    // must survive without becoming a newline.
    let mut chars = s.chars();
    let mut decoded = String::with_capacity(s.len());
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            decoded.push(ch);
            continue;
        }
        match chars.next() {
            Some('"') => decoded.push('"'),
            Some('\\') => decoded.push('\\'),
            Some('n') => decoded.push('\n'),
            Some('r') => decoded.push('\r'),
            Some('t') => decoded.push('\t'),
            Some(other) => {
                decoded.push('\\');
                decoded.push(other);
            }
            None => decoded.push('\\'),
        }
    }
    decoded
}

// =============================================================================
// JSON-LD import
// =============================================================================

/// Import entity annotations from JSON-LD (`.jsonld`) format.
///
/// Parses documents produced by `anno export --format jsonld`.
/// Each object in the `@graph` array maps to one `ImportedAnnotation`.
fn import_jsonld(input: &PathBuf) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let doc: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Invalid JSON-LD: {}", e))?;

    let graph = doc
        .get("@graph")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "JSON-LD document missing '@graph' array".to_string())?;

    let source_default = input.to_string_lossy().to_string();
    let mut annotations = Vec::new();

    for node in graph {
        // `@type` → `"…#PERType"` → `"PER"`
        let entity_type = node
            .get("@type")
            .and_then(|v| v.as_str())
            .map(|t| {
                let after = t.rsplit('#').next().unwrap_or(t);
                after.strip_suffix("Type").unwrap_or(after).to_string()
            })
            .unwrap_or_else(|| "ENTITY".to_string());

        let text = node
            .get("rdfs:label")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let start = node
            .get("anno:startOffset")
            .or_else(|| node.get("startOffset"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let end = node
            .get("anno:endOffset")
            .or_else(|| node.get("endOffset"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let confidence = node
            .get("anno:confidence")
            .or_else(|| node.get("confidence"))
            .and_then(|v| v.as_f64());

        let source = node
            .get("prov:hadPrimarySource")
            .and_then(|v| v.get("@id"))
            .and_then(|v| v.as_str())
            .unwrap_or(&source_default)
            .to_string();

        if !text.is_empty() && end > start {
            annotations.push(ImportedAnnotation {
                text,
                entity_type,
                start,
                end,
                source,
                confidence,
            });
        }
    }

    annotations.sort_unstable_by_key(|a| a.start);
    Ok(annotations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conll_import_uses_character_offsets_for_unicode_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unicode.conll");
        fs::write(&path, "東京\tB-LOC\n駅\tI-LOC\nopen\tO\n").unwrap();

        let annotations = import_conll(&path).unwrap();
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].text, "東京 駅");
        assert_eq!((annotations[0].start, annotations[0].end), (0, 4));
    }
    #[test]
    fn failed_import_preserves_existing_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("bad.jsonl");
        let output = dir.path().join("existing.jsonl");
        fs::write(&input, "{").unwrap();
        fs::write(&output, "previous annotations").unwrap();
        assert!(run(ImportArgs {
            input,
            output: output.clone(),
            format: ExportFormat::Jsonl,
            include_text: false,
            allow_partial: false,
            quiet: true,
        })
        .is_err());
        assert_eq!(fs::read_to_string(output).unwrap(), "previous annotations");
    }

    #[test]
    fn directory_import_excludes_existing_output_and_its_alias() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("result.jsonl");
        let original = r#"{ "text": "keep this exactly" }"#;
        fs::write(&output, original).unwrap();
        fs::write(dir.path().join("bad.jsonl"), "{").unwrap();
        let args = || ImportArgs {
            input: dir.path().into(),
            output: output.clone(),
            format: ExportFormat::Jsonl,
            include_text: false,
            allow_partial: false,
            quiet: true,
        };
        assert!(run(args()).is_err());
        assert_eq!(fs::read_to_string(&output).unwrap(), original);

        fs::write(dir.path().join("valid.jsonl"), r#"{"text":"new"}"#).unwrap();
        let alias = dir.path().join("alias.jsonl");
        fs::hard_link(&output, &alias).unwrap();
        assert!(run(args()).unwrap_err().contains("aliases an input file"));
        assert_eq!(fs::read_to_string(&alias).unwrap(), original);
        fs::remove_file(alias).unwrap();
        fs::remove_file(dir.path().join("bad.jsonl")).unwrap();

        run(args()).unwrap();
        let contents = fs::read_to_string(output).unwrap();
        let row: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(row["text"], "new");
    }

    #[test]
    fn conll2003_keeps_iob1_starts_and_sentence_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("sample.conll");
        fs::write(&input, "-DOCSTART- -X- O O\n\nZoë NNP B-NP I-PER\nLi NNP I-NP I-PER\nAcme NNP B-NP I-ORG\n\n東京 NNP B-NP I-ORG\n").unwrap();
        let rows = import_conll(&input).unwrap();
        let spans: Vec<_> = rows
            .iter()
            .map(|a| (a.text.as_str(), a.entity_type.as_str(), a.start, a.end))
            .collect();
        assert_eq!(
            spans,
            [
                ("Zoë Li", "PER", 0, 6),
                ("Acme", "ORG", 7, 11),
                ("東京", "ORG", 12, 14)
            ]
        );
    }
    #[test]
    fn ntriples_recovers_quotes_and_distinguishes_backslashes_from_controls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.nt");
        let text = "Zoë \"Z\" C:\\new\\tools\\report\r\n\tend\\";
        let triples = format!(
            "<urn:e> <http://www.w3.org/2000/01/rdf-schema#label> {} .\n\
             <urn:e> <urn:v#startOffset> \"0\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n\
             <urn:e> <urn:v#endOffset> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
            serde_json::to_string(text).unwrap(),
            text.chars().count()
        );
        fs::write(&path, triples).unwrap();
        let rows = import_ntriples(&path).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].text, text);
        assert_eq!(rows[0].end, text.chars().count());
    }
}
