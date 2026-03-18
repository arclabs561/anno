//! Utility functions for CLI commands

use std::collections::HashMap;
use std::fs;
use std::io::{self, IsTerminal, Read};

use anno::{GroundedDocument, Identity, IdentityId, SignalId, TrackId};

/// Get input text from various sources (text arg, file, or stdin)
///
/// Sanitizes input to remove shell command fragments that might leak in
/// when using verbose flags with positional arguments.
pub fn get_input_text(
    text: &Option<String>,
    file: Option<&str>,
    positional: &[String],
) -> Result<String, String> {
    // Check explicit text arg
    if let Some(t) = text {
        return Ok(sanitize_input(t));
    }

    // Check file arg
    if let Some(f) = file {
        return read_input_file(f);
    }

    // Check positional args
    if !positional.is_empty() {
        let joined = positional.join(" ");
        return Ok(sanitize_input(&joined));
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
                let text = deformat::extract_readable(&buf, None).text;
                return Ok(text);
            }
            return Ok(sanitize_input(&buf));
        }
    }

    Err("No input text provided. Use -t 'text' or -f file or pipe via stdin".to_string())
}

/// Sanitize input text to remove shell command fragments.
///
/// Filters out common patterns that indicate shell command pollution:
/// - Path-like strings that look like command invocations
/// - Repeated flag patterns (e.g., "-vv" appearing in text)
/// - Shell variable references
fn sanitize_input(input: &str) -> String {
    let mut result = input.to_string();

    // Remove common shell command fragments that might leak in
    // Pattern: paths that look like command invocations (e.g., "Users/arc/Documents/dev/anno")
    // We only remove if they appear at word boundaries and look like absolute paths
    let path_patterns = [
        r"/Users/[^/]+/Documents/",
        r"/home/[^/]+/",
        r"/tmp/[^/]+/",
        r"cargo run",
        r"cargo test",
        r"target/debug/",
        r"target/release/",
    ];

    for pattern in &path_patterns {
        // Only remove if it's a standalone word/phrase, not part of actual text
        // This is conservative - we only remove obvious command fragments
        if let Some(pos) = result.find(pattern) {
            // Check if it's at the start or preceded by whitespace
            if pos == 0 || result[..pos].trim_end().ends_with(' ') {
                // Check if it's followed by whitespace or end of string
                let end_pos = pos + pattern.len();
                if end_pos >= result.len() || result[end_pos..].starts_with(' ') {
                    // This looks like a command fragment, remove it
                    result.replace_range(pos..end_pos, "");
                }
            }
        }
    }

    // Remove standalone flag patterns that shouldn't be in text (e.g., "-vv", "--verbose")
    // Only if they appear as separate words
    let flag_patterns = [
        (r" -vv ", " "),
        (r" -vvv ", " "),
        (r" --verbose ", " "),
        (r" -v ", " "), // Be careful with single -v as it might be part of words
    ];

    for (pattern, replacement) in &flag_patterns {
        result = result.replace(pattern, replacement);
    }

    // Clean up multiple spaces
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    result.trim().to_string()
}

/// Read a file with consistent error handling.
///
/// If the file looks like a PDF (by extension or `%PDF` magic bytes), extracts text
/// via `pdf-extract` (requires the `pdf` feature).
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
        // Use readability for article extraction, fall back to strip_to_text.
        // Same strategy as --url path (url_resolver.rs).
        let text = deformat::extract_readable(&content, None).text;
        Ok(text)
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
    })
}

/// Load gold-standard annotations from a JSONL file.
///
/// Each line should be a JSON object with an `entities` array.
/// Entities are objects with `text`, `label`, `start`, and `end` fields.
pub fn load_gold_from_file(path: &str) -> Result<Vec<GoldSpec>, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let mut gold = Vec::new();
    let mut warnings = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let entry: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("Invalid JSON in gold file at line {}: {}", line_num + 1, e))?;

        if let Some(entities) = entry["entities"].as_array() {
            for (i, ent) in entities.iter().enumerate() {
                // Validate required fields are present
                let start = match ent["start"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        warnings.push(format!(
                            "{}:{}: entity[{}] missing 'start' field, defaulting to 0",
                            path,
                            line_num + 1,
                            i
                        ));
                        0
                    }
                };
                let end = match ent["end"].as_u64() {
                    Some(v) => v as usize,
                    None => {
                        warnings.push(format!(
                            "{}:{}: entity[{}] missing 'end' field, defaulting to 0",
                            path,
                            line_num + 1,
                            i
                        ));
                        0
                    }
                };

                gold.push(GoldSpec {
                    text: ent["text"].as_str().unwrap_or("").to_string(),
                    label: ent["type"]
                        .as_str()
                        .or(ent["label"].as_str())
                        .unwrap_or("UNK")
                        .to_string(),
                    start,
                    end,
                });
            }
        }
    }

    // Report warnings to stderr
    for warning in &warnings {
        use super::output::color;
        eprintln!("{} {}", color("33", "warning:"), warning);
    }

    Ok(gold)
}

/// Resolve coreference by grouping signals into tracks.
///
/// The CLI delegates to the library implementation to keep semantics centralized.
pub fn resolve_coreference(doc: &mut GroundedDocument, text: &str, _signal_ids: &[SignalId]) {
    let coref = anno::backends::coref::mention_ranking::MentionRankingCoref::new();
    if let Err(e) = coref.resolve_into_document(text, doc) {
        use super::output::color;
        eprintln!("{} coref failed: {}", color("33", "warning:"), e);
    }
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

    matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
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
    fn read_input_file_html_strips_nav_footer() {
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

    #[test]
    fn read_input_file_plain_text_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "Tim Cook met Sundar Pichai in Seattle.").unwrap();
        let text = read_input_file(path.to_str().unwrap()).unwrap();
        assert_eq!(text, "Tim Cook met Sundar Pichai in Seattle.");
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
    #[test]
    fn html_content_detected_for_stripping() {
        let html = r#"<!DOCTYPE html>
        <html><head><title>Test</title></head>
        <body>
        <nav>Skip this</nav>
        <article><p>Elon Musk founded SpaceX in Hawthorne, California.</p></article>
        <footer>Copyright</footer>
        </body></html>"#;
        assert!(
            deformat::detect::is_html(html),
            "should detect HTML content"
        );
        let text = deformat::extract_readable(html, None).text;
        assert!(text.contains("Elon Musk"), "should extract article text");
        assert!(text.contains("SpaceX"), "should extract org name");
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
