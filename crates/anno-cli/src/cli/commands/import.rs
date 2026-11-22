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
use std::ops::Range;
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

    /// Align one CoNLL input to this original UTF-8 document for exact character offsets
    #[arg(long, value_name = "PATH")]
    pub source_text: Option<PathBuf>,

    /// Include source text. Multi-span brat records always need the matching .txt file.
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

/// Exact BRAT character offsets. Unlike an entity's normalized DiscontinuousSpan,
/// source segments can have empty endpoints marking leading/trailing line breaks.
#[derive(Debug, Clone)]
struct SourceSegments(Vec<Range<usize>>);

impl SourceSegments {
    fn new(ranges: Vec<Range<usize>>) -> Result<Self, String> {
        if ranges.len() < 2 {
            return Err("expected at least two source segments".into());
        }
        for (i, range) in ranges.iter().enumerate() {
            if range.start > range.end || (i > 0 && ranges[i - 1].end > range.start) {
                return Err(format!("invalid or out-of-order source segment {range:?}"));
            }
        }
        Ok(Self(ranges))
    }

    fn bounds(&self) -> Range<usize> {
        self.0[0].start..self.0[self.0.len() - 1].end
    }
}

#[derive(Debug, Clone)]
enum ImportedSpan {
    Contiguous(Range<usize>),
    Segmented(SourceSegments),
}

impl ImportedSpan {
    fn bounds(&self) -> Range<usize> {
        match self {
            Self::Contiguous(range) => range.clone(),
            Self::Segmented(segments) => segments.bounds(),
        }
    }
}

/// Imported annotation from external format.
#[derive(Debug, Clone)]
pub struct ImportedAnnotation {
    /// Entity text
    pub text: String,
    /// Entity type label
    pub entity_type: String,
    span: ImportedSpan,
    /// Source annotation system
    pub source: String,
    /// Optional confidence score
    pub confidence: Option<f64>,
}

fn same_existing_file(left: &Path, right: &Path) -> Result<bool, String> {
    same_file::is_same_file(left, right).map_err(|e| {
        format!(
            "Failed to compare {} and {}: {e}",
            left.display(),
            right.display()
        )
    })
}

// Compare directory entries even when the target (e.g. an optional companion)
// does not exist yet. Symlinked parent directories may spell the same location.
fn same_entry(left: &Path, right: &Path) -> Result<bool, String> {
    fn parent(path: &Path) -> &Path {
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
    }
    if left.file_name() != right.file_name() {
        return Ok(false);
    }
    same_existing_file(parent(left), parent(right))
}

/// Run the import command.
pub fn run(args: ImportArgs) -> Result<(), String> {
    // Validate input
    if !args.input.exists() {
        return Err(format!("Input not found: {:?}", args.input));
    }

    let output_exists = args
        .output
        .try_exists()
        .map_err(|e| format!("Failed to inspect output: {e}"))?;

    let source_text = if let Some(path) = &args.source_text {
        if !matches!(args.format, ExportFormat::Conll) || !args.input.is_file() {
            return Err("--source-text requires a single CoNLL input file".into());
        }
        if output_exists && same_existing_file(path, &args.output)? {
            return Err(format!(
                "Output aliases the original source text: {}",
                path.display()
            ));
        }
        Some(
            fs::read_to_string(path)
                .map_err(|e| format!("Failed to read source text {}: {e}", path.display()))?,
        )
    } else {
        None
    };

    // Collect files to process
    let files: Vec<PathBuf> = if args.input.is_file() {
        if output_exists && same_existing_file(&args.input, &args.output)? {
            return Err(format!(
                "Output aliases an input file: {}",
                args.input.display()
            ));
        }
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
            if output_exists && same_existing_file(&path, &args.output)? {
                // A previous JSONL result in a JSONL source directory is
                // excluded on reruns. Other formats contain source records.
                if matches!(args.format, ExportFormat::Jsonl) && same_entry(&path, &args.output)? {
                    continue;
                }
                return Err(format!("Output aliases an input file: {}", path.display()));
            }
            inputs.push(path);
        }
        inputs
    };

    if matches!(args.format, ExportFormat::Brat) {
        for file in &files {
            let companion = file.with_extension("txt");
            if same_entry(&companion, &args.output)?
                || (output_exists
                    && companion
                        .try_exists()
                        .map_err(|e| format!("Failed to inspect {}: {e}", companion.display()))?
                    && same_existing_file(&companion, &args.output)?)
            {
                return Err(format!(
                    "Output aliases a brat source companion: {}",
                    companion.display()
                ));
            }
        }
    }

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
        match import_file(file, args.format, args.include_text, source_text.as_deref()) {
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
            let bounds = a.span.bounds();
            let mut row = serde_json::json!({
                "text": a.text,
                "type": a.entity_type,
                "start": bounds.start,
                "end": bounds.end,
                "source": a.source,
                "confidence": a.confidence,
            });
            if let ImportedSpan::Segmented(segments) = &a.span {
                row["segments"] = serde_json::json!(segments.0);
            }
            row.to_string()
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
    source_text: Option<&str>,
) -> Result<Vec<ImportedAnnotation>, String> {
    match format {
        ExportFormat::Brat => import_brat(input, include_text),
        ExportFormat::Conll => import_conll(input, source_text),
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

    // A multi-span annotation needs the source to distinguish newlines from omitted words.
    let txt_path = input.with_extension("txt");
    let source = std::cell::OnceCell::new();
    let read_source = || {
        source
            .get_or_init(|| {
                fs::read_to_string(&txt_path).map(|text| {
                    let bytes = anno::offset::build_char_to_byte_map(&text);
                    (text, bytes)
                })
            })
            .as_ref()
    };

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
                if let Some((entity_type, spans)) = parts[1].trim().split_once(char::is_whitespace)
                {
                    let (span, text) = if spans.contains(';') {
                        let (source, bytes) = read_source().map_err(|e| {
                            format!(
                                "{tid}: need {} to check the gaps in a multi-span mention: {e}",
                                txt_path.display()
                            )
                        })?;
                        import_brat_segments(spans, parts[2], source, bytes)
                            .map_err(|e| format!("{tid}: {e}"))?
                    } else {
                        let range = brat_span(spans)?;
                        let (start, end) = (range.start, range.end);
                        let text = if !parts[2].is_empty() {
                            parts[2].to_string()
                        } else if let Some((txt, _)) =
                            include_text.then(read_source).and_then(Result::ok)
                        {
                            txt.chars().skip(start).take(end - start).collect()
                        } else {
                            format!("[{start}:{end}]")
                        };
                        (ImportedSpan::Contiguous(range), text)
                    };

                    annotations.push(ImportedAnnotation {
                        text,
                        entity_type: entity_type.to_string(),
                        span,
                        source: input.to_string_lossy().to_string(),
                        confidence: confidences.get(tid).copied(),
                    });
                }
            }
        }
    }

    Ok(annotations)
}

fn brat_span(span: &str) -> Result<Range<usize>, String> {
    let mut fields = span.split_whitespace();
    let start = fields
        .next()
        .and_then(|n| n.parse().ok())
        .ok_or("Invalid start offset")?;
    let end = fields
        .next()
        .and_then(|n| n.parse().ok())
        .ok_or("Invalid end offset")?;
    if start > end || fields.next().is_some() {
        return Err(format!("Invalid brat span '{span}'"));
    }
    Ok(start..end)
}

fn import_brat_segments(
    spans: &str,
    reference: &str,
    source: &str,
    bytes: &[usize],
) -> Result<(ImportedSpan, String), String> {
    let segments = SourceSegments::new(spans.split(';').map(brat_span).collect::<Result<_, _>>()?)?;
    let bounds = segments.bounds();
    if bounds.end >= bytes.len() {
        return Err("brat span exceeds source length".into());
    }
    let slice = |range: &Range<usize>| &source[bytes[range.start]..bytes[range.end]];
    let pieces: Vec<_> = segments.0.iter().map(slice).collect();
    let joined = pieces.join(" ");
    if !reference.is_empty() && reference != joined {
        return Err("brat reference text does not match its source spans".into());
    }
    let only_line_breaks = segments.0.windows(2).all(|pair| {
        slice(&(pair[0].end..pair[1].start))
            .chars()
            .all(|c| matches!(c, '\r' | '\n'))
    });
    let text = if only_line_breaks {
        slice(&bounds).to_owned()
    } else {
        joined
    };
    Ok((ImportedSpan::Segmented(segments), text))
}

/// Import from CoNLL IOB format
fn import_conll(
    input: &Path,
    mut remaining_source: Option<&str>,
) -> Result<Vec<ImportedAnnotation>, String> {
    let content = fs::read_to_string(input).map_err(|e| format!("Failed to read file: {}", e))?;

    let mut annotations: Vec<ImportedAnnotation> = Vec::new();
    let mut inside_entity = false;
    let mut char_idx = 0;

    for (index, line) in content.lines().enumerate() {
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
            if remaining_source.is_some() {
                return Err(format!("Missing CoNLL tag on line {}", index + 1));
            }
            continue;
        };
        // Anno exports word/tag pairs. CoNLL-2003 puts the NER tag last,
        // after the part-of-speech and chunk tags.
        let tag = if second == "O" || second.starts_with("B-") || second.starts_with("I-") {
            second
        } else {
            fields.next_back().unwrap_or(second)
        };
        let label = tag.strip_prefix("B-").or_else(|| tag.strip_prefix("I-"));
        if remaining_source.is_some() && tag != "O" && label.is_none_or(str::is_empty) {
            return Err(format!("Invalid CoNLL tag {tag:?} on line {}", index + 1));
        }
        let gap = if let Some(remaining) = remaining_source {
            let next = remaining.trim_start_matches(char::is_whitespace);
            let whitespace = &remaining[..remaining.len() - next.len()];
            let after_token = next.strip_prefix(word).ok_or_else(|| {
                format!(
                "CoNLL token {word:?} on line {} does not match the original text at character {}",
                index + 1, char_idx + whitespace.chars().count()
            )
            })?;
            remaining_source = Some(after_token);
            whitespace
        } else if char_idx == 0 {
            ""
        } else {
            " "
        };
        let start = char_idx + gap.chars().count();
        let end = start + word.chars().count();
        char_idx = end;

        if let Some(label) = tag.strip_prefix("I-") {
            if let Some(entity) = annotations.last_mut() {
                if inside_entity && entity.entity_type == label {
                    if let ImportedSpan::Contiguous(range) = &mut entity.span {
                        entity.text.push_str(gap);
                        entity.text.push_str(word);
                        range.end = end;
                        continue;
                    }
                }
            }
        }
        inside_entity = label.is_some();
        if let Some(label) = label {
            annotations.push(ImportedAnnotation {
                text: word.to_string(),
                entity_type: label.to_string(),
                span: ImportedSpan::Contiguous(start..end),
                source: input.to_string_lossy().into_owned(),
                confidence: None,
            });
        }
    }

    if remaining_source.is_some_and(|rest| !rest.trim().is_empty()) {
        return Err("CoNLL input ends before the original text".into());
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

        let span = if let Some(raw) = obj.get("segments") {
            let ranges = serde_json::from_value(raw.clone())
                .map_err(|e| format!("Invalid JSONL segments: {e}"))?;
            let segments = SourceSegments::new(ranges)?;
            let bounds = segments.bounds();
            if obj["start"].as_u64() != Some(bounds.start as u64)
                || obj["end"].as_u64() != Some(bounds.end as u64)
            {
                return Err("JSONL start/end must match the bounds of segments".into());
            }
            ImportedSpan::Segmented(segments)
        } else {
            let start = obj["start"].as_u64().unwrap_or(0) as usize;
            let end = obj["end"].as_u64().unwrap_or(0) as usize;
            ImportedSpan::Contiguous(start..end)
        };

        annotations.push(ImportedAnnotation {
            text: obj["text"].as_str().unwrap_or("").to_string(),
            entity_type: obj["type"].as_str().unwrap_or("").to_string(),
            span,
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
                span: ImportedSpan::Contiguous(start..end),
                source,
                confidence,
            });
        }
    }

    annotations.sort_unstable_by_key(|a| a.span.bounds().start);
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
                span: ImportedSpan::Contiguous(start..end),
                source,
                confidence,
            });
        }
    }

    annotations.sort_unstable_by_key(|a| a.span.bounds().start);
    Ok(annotations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brat_line_breaks_round_trip_with_their_source_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mentions.ann");
        for (mention, spans, reference) in [
            ("New\nYork", "2 5;6 10", "New York"),
            ("New\r\nYork", "2 5;7 11", "New York"),
            ("Zoë\r\n\nZoe\u{301}", "2 5;8 12", "Zoë Zoe\u{301}"),
            ("New \n York", "2 6;7 12", "New   York"),
            ("\nYork\r\n", "2 2;3 7;9 9", " York "),
            ("\r\n", "2 2;4 4", " "),
        ] {
            let text = format!("😀 {mention} {mention}");
            let end = 2 + mention.chars().count();
            let entity = anno::Entity::new(mention, anno::EntityType::Location, 2, end, 0.85);
            let ann = anno::export::to_brat(&text, &[entity], true);
            assert_eq!(
                ann,
                format!("T1\tLOC {spans}\t{reference}\nA1\tConfidence T1 0.85")
            );
            fs::write(path.with_extension("txt"), &text).unwrap();
            fs::write(&path, ann).unwrap();
            for include_text in [false, true] {
                let rows = import_brat(&path, include_text).unwrap();
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].text, mention);
                assert_eq!(rows[0].span.bounds(), 2..end);
                assert_eq!(rows[0].confidence, Some(0.85));
            }
        }
    }

    #[test]
    fn brat_does_not_guess_what_a_multi_span_gap_contains() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gap.ann");
        fs::write(&path, "T1\tLOC 0 3;8 12\tNew York").unwrap();
        let err = import_brat(&path, false).unwrap_err();
        assert!(err.contains("gap.txt"), "{err}");
        fs::write(path.with_extension("txt"), "New and York").unwrap();
        let rows = import_brat(&path, false).unwrap();
        assert_eq!(rows[0].text, "New York");
        assert_eq!(rows[0].span.bounds(), 0..12);

        fs::write(path.with_extension("txt"), "New\nYork").unwrap();
        for (spans, reference) in [
            ("0 3;4 8", "New Paris"),
            ("0 3;4 9", "New York"),
            ("4 8;0 3", "York New"),
            ("0 4;3 8", "New York"),
            ("0 3;8 4", "New York"),
        ] {
            fs::write(&path, format!("T1\tLOC {spans}\t{reference}")).unwrap();
            assert!(import_brat(&path, true).is_err(), "{spans}: {reference}");
        }
    }

    #[test]
    fn every_pair_of_character_ranges_keeps_its_source_segments() {
        // One-, two-, three- and four-byte scalars, a decomposed accent and CRLF.
        let source = "é🦊e\u{301}\r\n界";
        let chars: Vec<char> = source.chars().collect();
        let bytes = anno::offset::build_char_to_byte_map(source);
        let offsets = 0..=chars.len();
        let pairs: Vec<_> = offsets
            .clone()
            .flat_map(|start| offsets.clone().map(move |end| start..end))
            .collect();
        let mut jsonl = String::new();
        let mut expected = Vec::new();
        for first in &pairs {
            for second in &pairs {
                let bounds = first.start..second.end;
                let spans = format!(
                    "{} {};{} {}",
                    first.start, first.end, second.start, second.end
                );
                let ranges = [first.clone(), second.clone()];
                let endpoints = [first.start, first.end, second.start, second.end];
                if !endpoints.is_sorted() {
                    assert!(
                        import_brat_segments(&spans, "", source, &bytes).is_err(),
                        "{spans}"
                    );
                    continue;
                }

                // This oracle indexes Unicode scalars directly, independently of
                // the byte-offset map used by the BRAT reader.
                let left: String = chars[first.clone()].iter().collect();
                let right: String = chars[second.clone()].iter().collect();
                let reference = format!("{left} {right}");
                let text = if chars[first.end..second.start]
                    .iter()
                    .all(|c| matches!(c, '\r' | '\n'))
                {
                    chars[bounds.clone()].iter().collect()
                } else {
                    reference.clone()
                };
                let (span, surface) =
                    import_brat_segments(&spans, &reference, source, &bytes).unwrap();
                let ImportedSpan::Segmented(segment) = span else {
                    panic!("flattened {spans}")
                };
                assert_eq!(segment.0, ranges, "{spans}");
                assert_eq!(segment.bounds(), bounds, "{spans}");
                assert_eq!(surface, text, "{spans}");
                jsonl.push_str(&serde_json::json!({"text":text, "segments":ranges, "start":bounds.start, "end":bounds.end}).to_string());
                jsonl.push('\n');
                expected.push((ranges, text));
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ranges.jsonl");
        fs::write(&path, jsonl).unwrap();
        let rows = import_jsonl(&path).unwrap();
        assert_eq!(rows.len(), expected.len());
        for (row, (ranges, text)) in rows.into_iter().zip(expected) {
            let ImportedSpan::Segmented(segments) = row.span else {
                panic!("flattened JSONL range")
            };
            assert_eq!(segments.0, ranges);
            assert_eq!(row.text, text);
        }
    }

    #[test]
    fn jsonl_segment_validation_preserves_exact_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entity.jsonl");
        let row = serde_json::json!({
            "text": "first last",
            "start": 1,
            "end": 12,
            "segments": [{"start":1,"end":6}, {"start":8,"end":12}]
        });
        fs::write(&path, row.to_string()).unwrap();
        let rows = import_jsonl(&path).unwrap();
        let ImportedSpan::Segmented(segments) = &rows[0].span else {
            panic!("missing segments")
        };
        assert_eq!(segments.0, [1..6, 8..12]);
        for invalid in [
            serde_json::json!(null),
            serde_json::json!([]),
            serde_json::json!([{"start": 1, "end": 12}]),
            serde_json::json!([{"start": 8, "end": 12}, {"start": 1, "end": 6}]),
            serde_json::json!([{"start": 1, "end": 9}, {"start": 8, "end": 12}]),
            serde_json::json!([{"start": 1, "end": 6}, {"start": 13, "end": 12}]),
            serde_json::json!([{"start": 1, "end": 6}, {"start": 8, "end": 11}]),
            serde_json::json!([{"start": 1, "end": 6}, {"start": 8}]),
        ] {
            let mut bad = row.clone();
            bad["segments"] = invalid;
            fs::write(&path, bad.to_string()).unwrap();
            assert!(import_jsonl(&path).is_err(), "{bad}");
        }
    }

    #[test]
    fn conll_import_uses_character_offsets_for_unicode_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unicode.conll");
        fs::write(&path, "東京\tB-LOC\n駅\tI-LOC\nopen\tO\n").unwrap();

        let annotations = import_conll(&path, None).unwrap();
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].text, "東京 駅");
        assert_eq!(annotations[0].span.bounds(), 0..4);
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
            source_text: None,
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
            source_text: None,
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
        let rows = import_conll(&input, None).unwrap();
        let spans: Vec<_> = rows
            .iter()
            .map(|a| {
                (
                    a.text.as_str(),
                    a.entity_type.as_str(),
                    a.span.bounds().start,
                    a.span.bounds().end,
                )
            })
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
        assert_eq!(rows[0].span.bounds().end, text.chars().count());
    }
}
