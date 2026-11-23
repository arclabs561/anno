//! Utility functions for CLI commands

use std::collections::HashMap;
use std::fs;
use std::io::{self, IsTerminal, Read};

use anno::backends::coref::CorefBackend;
use anno::{
    CorefAttemptFailure, CorefRunReport, GroundedDocument, Identity, IdentityId, SignalId, TrackId,
};

#[cfg(not(any(feature = "extractor-readability", feature = "extractor-html2text")))]
compile_error!(
    "anno-cli requires at least one HTML extractor feature: \
     `extractor-readability` (default, best quality) or \
     `extractor-html2text` (no MPL-2.0 transitives, lower quality)"
);

/// Extracted text and the provenance recorded by the URL resolver.
#[cfg_attr(not(feature = "eval"), allow(dead_code))]
pub(crate) struct HtmlExtraction {
    pub text: String,
    pub extractor: deformat::Extractor,
    pub title: Option<String>,
    pub fallback: bool,
}

// Readability takes precedence when both backends are compiled.
#[cfg(feature = "extractor-readability")]
pub(crate) fn extract_html(html: &str, url: Option<&str>) -> HtmlExtraction {
    let article = deformat::extract_readable(html, url);
    HtmlExtraction {
        text: article.text,
        extractor: article.extractor,
        title: article.title,
        fallback: article.fallback,
    }
}

#[cfg(all(
    not(feature = "extractor-readability"),
    feature = "extractor-html2text"
))]
pub(crate) fn extract_html(html: &str, _url: Option<&str>) -> HtmlExtraction {
    // Render each cell in document order. Display columns interleave other cells
    // into wrapped identifiers before the entity detector sees them.
    let rendered = html2text::config::plain()
        .raw_mode(true)
        .string_from_read(html.as_bytes(), 80);
    let (text, extractor, fallback) = match rendered {
        Ok(text) => (text, deformat::Extractor::Html2text, false),
        Err(_) => (
            deformat::html::strip_to_text(html),
            deformat::Extractor::Strip,
            true,
        ),
    };
    HtmlExtraction {
        text,
        extractor,
        title: None,
        fallback,
    }
}

/// Convenience: extract HTML and return only the text. Same backend
/// routing as [`extract_html`].
pub(crate) fn extract_html_to_text(html: &str, url: Option<&str>) -> String {
    extract_html(html, url).text
}

/// Get input text from various sources (text arg, file, or stdin).
///
/// Text is returned unchanged, apart from explicit HTML/PDF extraction for file
/// input and detected HTML on stdin. Keeping plain source text intact preserves
/// the character offsets reported by downstream commands.
pub fn get_input_text(
    text: &Option<String>,
    file: Option<&str>,
    positional: &[String],
) -> Result<String, String> {
    // Check explicit text arg
    if let Some(t) = text {
        return Ok(t.clone());
    }

    // Check file arg
    if let Some(f) = file {
        return read_input_file(f);
    }

    // Check positional args
    if !positional.is_empty() {
        return Ok(positional.join(" "));
    }

    // Try stdin
    if !io::stdin().is_terminal() {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format_error("read stdin", &e.to_string()))?;
        if !buf.is_empty() {
            // Auto-detect HTML in stdin and strip it, same as --file does
            if deformat::detect::is_html(&buf) {
                eprintln!("note: detected HTML content on stdin, converting to text");
                return Ok(extract_html_to_text(&buf, None));
            }
            return Ok(buf);
        }
    }

    Err("No input text provided. Use -t 'text' or -f file or pipe via stdin".to_string())
}

/// Validate a user-supplied name used as one filename component.
pub(crate) fn validate_path_component<'a>(value: &'a str, kind: &str) -> Result<&'a str, String> {
    use std::path::{Component, Path};

    let is_single_normal_component = matches!(
        Path::new(value).components().next(),
        Some(Component::Normal(_))
    ) && Path::new(value).components().count() == 1;
    if value.is_empty() || value.contains(['/', '\\']) || !is_single_normal_component {
        return Err(format!("{kind} must be a non-empty filename component"));
    }
    Ok(value)
}

/// Read a file with consistent error handling.
///
/// If the file looks like a PDF (by extension or `%PDF` magic bytes), extracts text
/// via `deformat` (requires the `pdf` feature).
///
/// If the file looks like HTML (based on content sniffing or `.html`/`.htm` extension),
/// automatically strips tags and extracts text -- same logic as `--url` uses.
pub fn read_input_file(path: &str) -> Result<String, String> {
    // --- PDF detection (before reading as UTF-8, since PDFs are binary) ---
    let is_pdf_ext = path.ends_with(".pdf");
    let is_pdf_magic = if !is_pdf_ext {
        fs::read(path)
            .ok()
            .map(|bytes| bytes.starts_with(b"%PDF"))
            .unwrap_or(false)
    } else {
        false
    };

    if is_pdf_ext || is_pdf_magic {
        #[cfg(feature = "pdf")]
        {
            eprintln!("note: extracting text from PDF '{}'", path);
            let result = deformat::pdf::extract_file(std::path::Path::new(path))
                .map_err(|e| format_error("extract PDF text", &format!("{}: {}", path, e)))?;
            return Ok(result.text);
        }
        #[cfg(not(feature = "pdf"))]
        {
            return Err(format!(
                "File '{}' appears to be a PDF, but the `pdf` feature is not enabled. \
                 Rebuild with `--features pdf` to enable PDF extraction.",
                path
            ));
        }
    }

    let content = fs::read_to_string(path)
        .map_err(|e| format_error("read file", &format!("{}: {}", path, e)))?;

    // Detect HTML by extension or content
    let is_html_ext = path.ends_with(".html") || path.ends_with(".htm") || path.ends_with(".xhtml");
    if is_html_ext || deformat::detect::is_html(&content) {
        eprintln!(
            "note: detected HTML content in '{}', converting to text",
            path
        );
        Ok(extract_html_to_text(&content, None))
    } else {
        Ok(content)
    }
}

/// Parse a GroundedDocument from JSON with consistent error handling
pub fn parse_grounded_document(json: &str) -> Result<GroundedDocument, String> {
    serde_json::from_str(json)
        .map_err(|e| format_error("parse GroundedDocument JSON", &e.to_string()))
}

/// Format error message consistently
pub fn format_error(operation: &str, details: &str) -> String {
    format!("Failed to {}: {}", operation, details)
}

/// Log success message with color (respects quiet flag)
pub fn log_success(msg: &str, quiet: bool) {
    if !quiet {
        use super::output::color;
        eprintln!("{} {}", color("32", "✓"), msg);
    }
}

/// A gold-standard entity annotation for evaluation.
///
/// Loaded from JSONL files with `entities` arrays containing
/// `{text, label, start, end}` objects.
#[derive(Debug, Clone)]
pub struct GoldSpec {
    /// Surface text of the entity.
    pub text: String,
    /// Entity type label (e.g., "PER", "ORG").
    pub label: String,
    /// Character offset where entity begins.
    pub start: usize,
    /// Character offset where entity ends.
    pub end: usize,
    /// Validated covered character ranges for segmented file annotations.
    pub segments: Option<anno::DiscontinuousSpan>,
}

/// Parse gold spec with format: "text:label:start:end"
/// Uses rsplit to handle text containing colons (like URLs)
pub fn parse_gold_spec(s: &str) -> Option<GoldSpec> {
    // Split from right to handle colons in text
    let parts: Vec<&str> = s.rsplitn(4, ':').collect();
    if parts.len() < 4 {
        return None;
    }

    let end: usize = parts[0].parse().ok()?;
    let start: usize = parts[1].parse().ok()?;
    let label = parts[2].to_string();
    let text = parts[3].to_string();

    Some(GoldSpec {
        text,
        label,
        start,
        end,
        segments: None,
    })
}

pub(crate) fn source_sha256(text: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Load gold-standard annotations from a JSONL file.
///
/// Each line should be a JSON object with an `entities` array.
/// Entities are objects with `text`, `label`, `start`, and `end` fields.
pub fn load_gold_from_file(
    path: &str,
    source: &str,
) -> Result<(Vec<GoldSpec>, Option<String>), String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let offsets = std::cell::LazyCell::new(|| anno::offset::build_char_to_byte_map(source));
    let mut gold = Vec::new();
    let mut document_id = None;
    let digest = std::cell::LazyCell::new(|| source_sha256(source));

    for (line_num, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let entry: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("Invalid JSON in gold file at line {}: {}", line_num + 1, e))?;

        let location = || format!("{path}:{}", line_num + 1);
        if entry
            .get("text")
            .is_some_and(|text| text.as_str() != Some(source))
        {
            return Err(format!(
                "{}: gold source text differs from evaluation input",
                location()
            ));
        }
        if let Some(hash) = entry.get("source_sha256") {
            if !hash
                .as_str()
                .is_some_and(|s| s.eq_ignore_ascii_case(&digest))
            {
                return Err(format!(
                    "{}: gold source_sha256 differs from evaluation input",
                    location()
                ));
            }
        }
        for field in ["id", "doc_id", "document_id"] {
            if let Some(id) = entry.get(field) {
                let id = match id {
                    serde_json::Value::String(s) if !s.is_empty() => s.clone(),
                    serde_json::Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
                    _ => {
                        return Err(format!(
                            "{}: expected a nonempty string or integer {field}",
                            location()
                        ))
                    }
                };
                if document_id
                    .as_deref()
                    .is_some_and(|previous| previous != id)
                {
                    return Err(format!(
                        "{}: conflicting gold document IDs in a single-text evaluation",
                        location()
                    ));
                }
                document_id.get_or_insert(id);
            }
        }

        let entities = entry["entities"]
            .as_array()
            .ok_or_else(|| format!("{path}:{}: expected an entities array", line_num + 1))?;
        for (i, ent) in entities.iter().enumerate() {
            let invalid = |field: &str| {
                format!(
                    "{path}:{}: entity[{i}] has missing or invalid {field}",
                    line_num + 1
                )
            };
            let offset = |field: &str| {
                ent[field]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| invalid(field))
            };
            let start = offset("start")?;
            let end = offset("end")?;
            let text = ent["text"].as_str().ok_or_else(|| invalid("text"))?;
            let segments = if let Some(raw) = ent.get("segments").filter(|v| !v.is_null()) {
                let ranges: Vec<std::ops::Range<usize>> =
                    serde_json::from_value(raw.clone()).map_err(|_| invalid("segments"))?;
                if ranges.is_empty()
                    || start >= end
                    || end >= offsets.len()
                    || ranges.first().is_none_or(|r| r.start != start)
                    || ranges.last().is_none_or(|r| r.end != end)
                    || ranges.iter().any(|r| r.start > r.end)
                    || ranges.windows(2).any(|p| p[0].end > p[1].start)
                {
                    return Err(invalid("segments or bounding offsets"));
                }
                let joined = ranges
                    .iter()
                    .map(|r| &source[offsets[r.start]..offsets[r.end]])
                    .collect::<Vec<_>>()
                    .join(" ");
                if text != joined && text != &source[offsets[start]..offsets[end]] {
                    return Err(invalid("segment surface text"));
                }
                let coverage = anno::DiscontinuousSpan::new(ranges);
                if coverage.segments().is_empty() {
                    return Err(invalid("empty segment coverage"));
                }
                Some(coverage)
            } else {
                None
            };
            gold.push(GoldSpec {
                text: text.to_owned(),
                label: ent["type"]
                    .as_str()
                    .or_else(|| ent["label"].as_str())
                    .ok_or_else(|| invalid("type or label"))?
                    .to_owned(),
                start,
                end,
                segments,
            });
        }
    }

    Ok((gold, document_id))
}

/// Resolve coreference by grouping signals into tracks.
///
/// Tries neural f-coref (ONNX) first, falls back to heuristic mention-ranking.
pub fn resolve_coreference(doc: &mut GroundedDocument, text: &str, _signal_ids: &[SignalId]) {
    #[cfg(feature = "onnx")]
    let mut failures = Vec::new();
    #[cfg(not(feature = "onnx"))]
    let failures = Vec::new();
    // Try f-coref (neural) if ONNX model is available
    #[cfg(feature = "onnx")]
    {
        use anno::backends::coref::fcoref::FCoref;

        // Check standard locations for f-coref model
        let fcoref_paths = [
            std::env::var("FCOREF_MODEL_PATH").ok(),
            // XDG cache (~/.cache on Linux, also common on macOS)
            std::env::var("HOME")
                .ok()
                .map(|h| format!("{h}/.cache/anno/models/fcoref")),
            // macOS native cache (~/Library/Caches)
            dirs::cache_dir().map(|d| d.join("anno/models/fcoref").to_string_lossy().into_owned()),
        ];

        for path in fcoref_paths.iter().flatten() {
            if std::path::Path::new(path).join("encoder.onnx").exists()
                && try_neural_coref(doc, text, FCoref::from_path(path), &mut failures)
            {
                return;
            }
        }

        if try_neural_coref(
            doc,
            text,
            FCoref::from_pretrained("biu-nlp/f-coref"),
            &mut failures,
        ) {
            return;
        }
    }

    // Fallback: heuristic mention-ranking
    let coref = anno::backends::coref::mention_ranking::MentionRankingCoref::new();
    match coref.resolve_into_document(text, doc) {
        Ok(tracks) => doc.record_coref_run(CorefRunReport::Completed {
            backend: coref.name().into(),
            tracks_added: tracks.len(),
            prior_failures: failures,
        }),
        Err(e) => {
            use super::output::color;
            doc.record_coref_run(CorefRunReport::Failed {
                last_failure: CorefAttemptFailure::Inference {
                    backend: coref.name().into(),
                },
                prior_failures: failures,
            });
            eprintln!("{} coref failed: {}", color("33", "warning:"), e);
        }
    }
}

#[cfg(feature = "onnx")]
fn try_neural_coref(
    doc: &mut GroundedDocument,
    text: &str,
    model: anno::Result<anno::backends::coref::fcoref::FCoref>,
    failures: &mut Vec<CorefAttemptFailure>,
) -> bool {
    let model = match model {
        Ok(model) => model,
        Err(error) => {
            log::debug!("f-coref load failed: {error}");
            failures.push(CorefAttemptFailure::Load {
                backend: "fcoref".into(),
            });
            return false;
        }
    };
    let clusters = match model.resolve(text) {
        Ok(clusters) => clusters,
        Err(error) => {
            log::debug!("f-coref inference failed: {error}");
            failures.push(CorefAttemptFailure::Inference {
                backend: model.name().into(),
            });
            return false;
        }
    };
    match append_coref_clusters(doc, &clusters) {
        Ok(tracks_added) => {
            doc.record_coref_run(CorefRunReport::Completed {
                backend: model.name().into(),
                tracks_added,
                prior_failures: std::mem::take(failures),
            });
            true
        }
        Err(error) => {
            log::debug!("invalid f-coref output: {error}");
            failures.push(CorefAttemptFailure::InvalidOutput {
                backend: model.name().into(),
            });
            false
        }
    }
}

/// Validate all cluster shapes before mutation. Stage nonempty writes so a rejected
/// insertion leaves earlier tracks and signals untouched for the fallback resolver.
#[cfg(any(test, feature = "onnx", feature = "discourse"))]
pub(super) fn append_coref_clusters(
    doc: &mut GroundedDocument,
    clusters: &[anno::backends::coref::resolve::CorefCluster],
) -> Result<usize, String> {
    if clusters.is_empty() {
        return Ok(0);
    }
    let text = doc.text();
    let offsets = anno::offset::SpanConverter::new(text);
    let text_len = offsets.byte_to_char(text.len());
    for cluster in clusters {
        if cluster.spans.len() != cluster.mentions.len() {
            return Err(format!(
                "coreference cluster {} has {} mentions and {} spans",
                cluster.id,
                cluster.mentions.len(),
                cluster.spans.len()
            ));
        }
        if cluster
            .spans
            .iter()
            .zip(&cluster.mentions)
            .any(|(&(start, end), mention)| {
                start >= end
                    || end > text_len
                    || text[offsets.char_to_byte(start)..offsets.char_to_byte(end)] != *mention
            })
        {
            return Err(format!(
                "coreference cluster {} has an invalid source span",
                cluster.id
            ));
        }
    }
    let mut groups = clusters
        .iter()
        .filter(|cluster| cluster.mentions.len() >= 2)
        .peekable();
    if groups.peek().is_none() {
        return Ok(0);
    }
    let mut staged = doc.clone();
    let mut tracks_added = 0;
    for cluster in groups {
        let signals =
            cluster
                .mentions
                .iter()
                .zip(&cluster.spans)
                .map(|(mention, &(start, end))| {
                    anno::Signal::new(
                        0u64,
                        anno::Location::text(start, end),
                        mention.clone(),
                        anno::TypeLabel::from("COREF_mention"),
                        1.0,
                    )
                });
        let ids = staged.add_signals(signals);
        staged
            .create_track_from_signals(&cluster.canonical, &ids)
            .ok_or_else(|| format!("could not attach coreference cluster {}", cluster.id))?;
        tracks_added += 1;
    }
    *doc = staged;
    Ok(tracks_added)
}

/// Attach demo KB-style identities to tracks.
///
/// This is a debug-only helper used by `anno debug --link-kb`.
/// It is **offline** and does **not** query a real knowledge base.
pub fn link_tracks_to_kb(doc: &mut GroundedDocument) {
    // Small embedded demo table (no claim of completeness).
    let known_entities: HashMap<&str, (&str, &str)> = [
        (
            "barack obama",
            ("demo:barack_obama", "44th President of the United States"),
        ),
        (
            "angela merkel",
            ("demo:angela_merkel", "Chancellor of Germany 2005-2021"),
        ),
        ("berlin", ("demo:berlin", "Capital of Germany")),
        ("nato", ("demo:nato", "North Atlantic Treaty Organization")),
        (
            "donald trump",
            ("demo:donald_trump", "45th President of the United States"),
        ),
        (
            "joe biden",
            ("demo:joe_biden", "46th President of the United States"),
        ),
        (
            "vladimir putin",
            ("demo:vladimir_putin", "President of Russia"),
        ),
        (
            "emmanuel macron",
            ("demo:emmanuel_macron", "President of France"),
        ),
        ("lynn conway", ("demo:lynn_conway", "VLSI pioneer")),
        ("sophie wilson", ("demo:sophie_wilson", "Co-designed ARM")),
        (
            "albert einstein",
            ("demo:albert_einstein", "Theoretical physicist"),
        ),
        ("new york", ("demo:new_york", "City in New York State")),
        ("london", ("demo:london", "Capital of the United Kingdom")),
        ("paris", ("demo:paris", "Capital of France")),
        ("google", ("demo:google", "American technology company")),
        ("apple", ("demo:apple", "American technology company")),
        (
            "microsoft",
            ("demo:microsoft", "American technology company"),
        ),
        (
            "united nations",
            ("demo:united_nations", "International organization"),
        ),
        (
            "european union",
            ("demo:european_union", "Political and economic union"),
        ),
    ]
    .into_iter()
    .collect();

    // Collect track IDs first to avoid borrow issues
    let track_ids: Vec<TrackId> = doc.tracks().map(|t| t.id).collect();

    for track_id in track_ids {
        let (canonical, entity_type) = {
            let track = match doc.get_track(track_id) {
                Some(t) => t,
                None => continue,
            };
            (track.canonical_surface.clone(), track.entity_type.clone())
        };

        let canonical_lower = canonical.to_lowercase();

        // Look up in demo table
        if let Some(&(kb_id, description)) = known_entities.get(canonical_lower.as_str()) {
            let mut identity = Identity::from_kb(IdentityId::ZERO, &canonical, "demo", kb_id);
            identity.aliases.push(description.to_string());
            if let Some(etype) = &entity_type {
                identity.entity_type = Some(etype.clone());
            }

            let identity_id = doc.add_identity(identity);
            doc.link_track_to_identity(track_id, identity_id);
        } else {
            // Create a deterministic demo KB id (no external meaning).
            let kb_id = format!("demo:{}", canonical_lower.replace(' ', "_"));
            let identity = Identity::from_kb(IdentityId::ZERO, &canonical, "demo", &kb_id);
            let identity_id = doc.add_identity(identity);
            doc.link_track_to_identity(track_id, identity_id);
        }
    }
}

/// Find similar model names using simple string similarity
pub fn find_similar_models(query: &str, candidates: &[&str]) -> Vec<String> {
    let query_lower = query.to_lowercase();
    let mut matches: Vec<(f64, &str)> = candidates
        .iter()
        .filter_map(|&candidate| {
            let candidate_lower = candidate.to_lowercase();
            // Check if query is a prefix of candidate or vice versa
            if candidate_lower.starts_with(&query_lower)
                || query_lower.starts_with(&candidate_lower)
            {
                Some((0.9, candidate))
            } else if candidate_lower.contains(&query_lower)
                || query_lower.contains(&candidate_lower)
            {
                Some((0.7, candidate))
            } else {
                // Simple Levenshtein-like check (first char match)
                if candidate_lower.chars().next() == query_lower.chars().next() {
                    Some((0.5, candidate))
                } else {
                    None
                }
            }
        })
        .collect();

    matches.sort_by(|a, b| b.0.total_cmp(&a.0));
    matches
        .into_iter()
        .take(3)
        .map(|(_, name)| name.to_string())
        .collect()
}

/// Get cache directory
pub fn get_cache_dir() -> Result<std::path::PathBuf, String> {
    #[cfg(feature = "eval")]
    {
        use dirs::cache_dir;
        if let Some(mut cache) = cache_dir() {
            cache.push("anno");
            fs::create_dir_all(&cache)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
            Ok(cache)
        } else {
            // Fallback to current directory
            Ok(std::path::PathBuf::from(".anno-cache"))
        }
    }
    #[cfg(not(feature = "eval"))]
    {
        Ok(std::path::PathBuf::from(".anno-cache"))
    }
}

/// Get config directory
pub fn get_config_dir() -> Result<std::path::PathBuf, String> {
    // Allow tests and power-users to override config location.
    //
    // This keeps CLI tests hermetic (no writes to the user's real config dir)
    // and makes config storage predictable in constrained environments.
    if let Ok(dir) = std::env::var("ANNO_CONFIG_DIR") {
        let path = std::path::PathBuf::from(dir);
        fs::create_dir_all(&path)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
        return Ok(path);
    }

    #[cfg(feature = "eval")]
    {
        use dirs::config_dir;
        if let Some(mut config) = config_dir() {
            config.push("anno");
            fs::create_dir_all(&config)
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
            Ok(config)
        } else {
            // Fallback to current directory
            Ok(std::path::PathBuf::from(".anno-config"))
        }
    }
    #[cfg(not(feature = "eval"))]
    {
        Ok(std::path::PathBuf::from(".anno-config"))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn malformed_coref_clusters_leave_prior_document_untouched() {
        use anno::backends::coref::resolve::CorefCluster;
        let mut doc = anno::GroundedDocument::new("doc", "A B");
        let good = CorefCluster {
            id: 1,
            canonical: "A".into(),
            mentions: vec!["A".into(), "B".into()],
            spans: vec![(0, 1), (2, 3)],
        };
        let mut bad = good.clone();
        bad.id = 2;
        bad.mentions.pop();
        assert!(super::append_coref_clusters(&mut doc, &[good, bad]).is_err());
        assert!(doc.signals().is_empty());
        assert_eq!(doc.tracks().count(), 0);
        assert!(doc.coref_backend().is_none());
    }

    #[test]
    fn in_bounds_coref_surface_must_match_its_source_before_attachment() {
        use anno::backends::coref::resolve::CorefCluster;
        let mut doc = anno::GroundedDocument::new("doc", "🦊 Ada Ada");
        let valid = CorefCluster {
            id: 1,
            canonical: "Ada".into(),
            mentions: vec!["Ada".into(), "Ada".into()],
            spans: vec![(2, 5), (6, 9)],
        };
        let mut stale = valid.clone();
        stale.id = 2;
        stale.mentions[1] = "Eve".into();
        assert!(super::append_coref_clusters(&mut doc, &[valid.clone(), stale]).is_err());
        assert!(doc.signals().is_empty());
        assert_eq!(doc.tracks().count(), 0);
        assert_eq!(super::append_coref_clusters(&mut doc, &[valid]).unwrap(), 1);
        assert!(doc.validate().is_empty());
    }

    use super::*;
    #[cfg(feature = "eval")]
    use anno::{Entity, EntityType};

    // -------------------------------------------------------------------------
    // parse_gold_spec tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parse_gold_spec_basic() {
        let spec = parse_gold_spec("John:PER:0:4").unwrap();
        assert_eq!(spec.text, "John");
        assert_eq!(spec.label, "PER");
        assert_eq!(spec.start, 0);
        assert_eq!(spec.end, 4);
    }

    #[test]
    fn test_parse_gold_spec_with_colon_in_text() {
        // URLs have colons, should still parse correctly (rsplit from right)
        let spec = parse_gold_spec("http://example.com:URL:5:22").unwrap();
        assert_eq!(spec.text, "http://example.com");
        assert_eq!(spec.label, "URL");
        assert_eq!(spec.start, 5);
        assert_eq!(spec.end, 22);
    }

    #[test]
    fn test_parse_gold_spec_invalid() {
        assert!(parse_gold_spec("invalid").is_none());
        assert!(parse_gold_spec("only:two").is_none());
        assert!(parse_gold_spec("text:label:notanumber:4").is_none());
    }

    // -------------------------------------------------------------------------
    // find_similar_models tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_find_similar_models() {
        let candidates = &["gliner", "gliner-candle", "heuristic", "pattern", "stacked"];

        let matches = find_similar_models("gli", candidates);
        assert!(matches.contains(&"gliner".to_string()));
        assert!(matches.contains(&"gliner-candle".to_string()));

        let matches = find_similar_models("pattern", candidates);
        assert!(matches.contains(&"pattern".to_string()));

        let matches = find_similar_models("xyz", candidates);
        assert!(matches.is_empty() || matches.len() <= 3);
    }

    // -------------------------------------------------------------------------
    // create_entity_pair_relations tests
    // -------------------------------------------------------------------------

    #[cfg(feature = "eval")]
    use anno_eval::eval::relation::create_entity_pair_relations;

    #[cfg(feature = "eval")]
    #[test]
    fn test_create_entity_pair_relations_founded() {
        let entities = vec![
            Entity::new("Steve Jobs", EntityType::Person, 0, 10, 0.9),
            Entity::new("Apple", EntityType::Organization, 30, 35, 0.9),
        ];
        let text = "Steve Jobs, who founded Apple in 1976, changed the world.";
        let relation_types = &["FOUNDED", "WORKS_FOR"];

        let relations = create_entity_pair_relations(&entities, text, relation_types);

        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation_type, "FOUNDED");
    }

    #[cfg(feature = "eval")]
    #[test]
    fn test_create_entity_pair_relations_unknown_fallback() {
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 15, 18, 0.9),
        ];
        // No relation keywords between entities
        let text = "Alice met Bob yesterday.";
        // Even if gold types include specific relations, fallback should be UNKNOWN
        let relation_types = &["FOUNDED", "WORKS_FOR"];

        let relations = create_entity_pair_relations(&entities, text, relation_types);

        assert_eq!(relations.len(), 1);
        // Should use UNKNOWN, not first gold type (FOUNDED) - this prevents metric inflation
        assert_eq!(
            relations[0].relation_type, "UNKNOWN",
            "Unknown relations should use UNKNOWN, not first gold type"
        );
    }

    #[cfg(feature = "eval")]
    #[test]
    fn test_create_entity_pair_relations_max_distance() {
        // Entities very far apart should not form relations
        let entities = vec![
            Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
            Entity::new("Bob", EntityType::Person, 300, 303, 0.9),
        ];
        let text = &"x".repeat(400); // Text longer than max_distance
        let relation_types = &["RELATED"];

        let relations = create_entity_pair_relations(&entities, text, relation_types);

        // Entities are > 200 chars apart, should not create relation
        assert!(relations.is_empty());
    }

    #[cfg(feature = "eval")]
    #[test]
    fn test_create_entity_pair_relations_overlapping_entities() {
        // Overlapping entities should be skipped
        let entities = vec![
            Entity::new("New York", EntityType::Location, 0, 8, 0.9),
            Entity::new("New York City", EntityType::Location, 0, 13, 0.9), // Overlapping
        ];
        let text = "New York City is great.";
        let relation_types = &["LOCATED_IN"];

        let relations = create_entity_pair_relations(&entities, text, relation_types);

        // Overlapping entities should be skipped
        assert!(relations.is_empty());
    }

    // -------------------------------------------------------------------------
    // read_input_file tests (readability/strip + PDF detection)
    // -------------------------------------------------------------------------

    #[test]
    fn read_input_file_html_extracts_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.html");
        fs::write(
            &path,
            r#"<!DOCTYPE html>
            <html><head><title>Test</title></head>
            <body>
                <nav>Menu</nav>
                <p>Angela Merkel met Emmanuel Macron in Berlin.</p>
                <footer>Copyright</footer>
            </body></html>"#,
        )
        .unwrap();
        let text = read_input_file(path.to_str().unwrap()).unwrap();
        assert!(
            text.contains("Angela Merkel"),
            "should extract person names, got: {}",
            text
        );
        assert!(
            text.contains("Berlin"),
            "should extract location, got: {}",
            text
        );
        // `html2text` is deliberately a tag-strip fallback and retains
        // boilerplate. Readability extraction promises article isolation.
        #[cfg(feature = "extractor-readability")]
        {
            assert!(
                !text.contains("Menu"),
                "nav content should be stripped, got: {}",
                text
            );
            assert!(
                !text.contains("Copyright"),
                "footer content should be stripped, got: {}",
                text
            );
        }
    }

    #[test]
    fn read_input_file_plain_text_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "Tim Cook met Sundar Pichai in Seattle.").unwrap();
        let text = read_input_file(path.to_str().unwrap()).unwrap();
        assert_eq!(text, "Tim Cook met Sundar Pichai in Seattle.");
    }

    #[test]
    fn get_input_text_preserves_plain_source_text() {
        let text = " cargo run  --verbose  keeps  spaces ";
        assert_eq!(
            get_input_text(&Some(text.to_string()), None, &[]).unwrap(),
            text
        );

        let positional = vec!["cargo run".to_string(), "-v".to_string()];
        assert_eq!(
            get_input_text(&None, None, &positional).unwrap(),
            "cargo run -v"
        );
    }

    #[test]
    fn path_component_validation_rejects_directory_escapes() {
        assert_eq!(
            validate_path_component("report-01", "document id").unwrap(),
            "report-01"
        );
        for invalid in ["", ".", "..", "../outside", "nested/name", "nested\\name"] {
            assert!(
                validate_path_component(invalid, "document id").is_err(),
                "{invalid:?}"
            );
        }
    }

    #[test]
    fn read_input_file_html_detected_by_content() {
        let dir = tempfile::tempdir().unwrap();
        // No .html extension, but content looks like HTML
        let path = dir.path().join("page.dat");
        fs::write(&path, "<html><body><p>Hello World</p></body></html>").unwrap();
        let text = read_input_file(path.to_str().unwrap()).unwrap();
        assert!(text.contains("Hello World"));
        // Should not contain raw HTML tags
        assert!(!text.contains("<p>"));
    }

    #[test]
    fn read_input_file_nonexistent_errors() {
        let result = read_input_file("/tmp/anno-does-not-exist-12345.txt");
        assert!(result.is_err());
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn read_input_file_pdf_extension_detected() {
        let dir = tempfile::tempdir().unwrap();
        // Create a file with .pdf extension but not actually a PDF -- should error
        let path = dir.path().join("fake.pdf");
        fs::write(&path, "not a real pdf").unwrap();
        let result = read_input_file(path.to_str().unwrap());
        // Should attempt PDF extraction and fail (not valid PDF)
        assert!(result.is_err());
    }

    /// Stdin HTML content should be detected and stripped automatically.
    /// This tests the detection logic -- actual stdin piping is tested via CLI integration.
    /// Gated on `extractor-readability`: the assertions about boilerplate
    /// stripping (`<nav>` and "Skip this" being removed) only hold for
    /// readability extraction; html2text retains nav text by design.
    #[cfg(feature = "extractor-readability")]
    #[test]
    fn html_content_detected_for_stripping() {
        let html = r#"<!DOCTYPE html>
        <html><head><title>Test</title></head>
        <body>
        <nav>Skip this</nav>
        <article><p>Marie Curie discovered radium at the University of Paris.</p></article>
        <footer>Copyright</footer>
        </body></html>"#;
        assert!(
            deformat::detect::is_html(html),
            "should detect HTML content"
        );
        let text = super::extract_html(html, None).text;
        assert!(text.contains("Marie Curie"), "should extract article text");
        assert!(
            text.contains("University of Paris"),
            "should extract org name"
        );
        assert!(!text.contains("<nav>"), "should not contain HTML tags");
        assert!(
            !text.contains("Skip this"),
            "nav content should be stripped"
        );
    }

    /// Plain text should NOT be detected as HTML
    #[test]
    fn plain_text_not_detected_as_html() {
        let text = "Angela Merkel met Emmanuel Macron in Berlin on Tuesday.";
        assert!(
            !deformat::detect::is_html(text),
            "plain text should not trigger HTML detection"
        );
    }

    #[cfg(not(feature = "pdf"))]
    #[test]
    fn read_input_file_pdf_without_feature_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pdf");
        fs::write(&path, "%PDF-1.4 fake").unwrap();
        let result = read_input_file(path.to_str().unwrap());
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("pdf"),
            "should mention pdf feature"
        );
    }
}
