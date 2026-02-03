//! Utility functions for CLI commands

use std::collections::HashMap;
use std::fs;
use std::io::{self, IsTerminal, Read};

#[cfg(feature = "eval-advanced")]
use anno_core::Entity;
use anno_core::{
    GroundedDocument, Identity, IdentityId, Location, Quantifier, Signal, SignalId, TrackId,
};

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

/// Read a file with consistent error handling
pub fn read_input_file(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format_error("read file", &format!("{}: {}", path, e)))
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

/// Log verbose message (only if verbose enabled)
pub fn log_verbose(msg: &str, verbose: bool) {
    if verbose {
        eprintln!("{}", msg);
    }
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

/// Detect if entity at position is negated
pub fn is_negated(text: &str, entity_start: usize) -> bool {
    // Performance: avoid allocating the entire prefix up to entity_start.
    // Negation cues are typically local (a few words before the entity), so scan a bounded window.
    const WINDOW_CHARS: usize = 200;
    let window_start = entity_start.saturating_sub(WINDOW_CHARS);
    let prefix: String = text
        .chars()
        .skip(window_start)
        .take(entity_start.saturating_sub(window_start))
        .collect();
    let words: Vec<&str> = prefix.split_whitespace().collect();
    let last_words: Vec<&str> = words.iter().rev().take(3).copied().collect();

    const NEGATION_WORDS: &[&str] = &[
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "without",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "couldn't",
        "shouldn't",
    ];

    for word in &last_words {
        if NEGATION_WORDS.contains(&word.to_lowercase().as_str()) {
            return true;
        }
    }

    false
}

/// Detect quantifier before entity
pub fn detect_quantifier(text: &str, entity_start: usize) -> Option<Quantifier> {
    // Performance: quantifiers are almost always immediately adjacent.
    const WINDOW_CHARS: usize = 80;
    let window_start = entity_start.saturating_sub(WINDOW_CHARS);
    let prefix: String = text
        .chars()
        .skip(window_start)
        .take(entity_start.saturating_sub(window_start))
        .collect();
    let words: Vec<&str> = prefix.split_whitespace().collect();

    words
        .last()
        .and_then(|word| match word.to_lowercase().as_str() {
            "every" | "all" | "each" | "any" => Some(Quantifier::Universal),
            "some" | "certain" | "a" | "an" => Some(Quantifier::Existential),
            "no" | "none" => Some(Quantifier::None),
            "the" | "this" | "that" | "these" | "those" => Some(Quantifier::Definite),
            _ => None,
        })
}

/// Flexible type matching for evaluation (handles PER/PERSON, LOC/LOCATION, etc.)
///
/// Handles common entity type variations across datasets:
/// - CoNLL-style: PER, LOC, ORG, MISC
/// - OntoNotes-style: PERSON, GPE, ORG, NORP, FAC, LOC
/// - spaCy-style: PERSON, GPE, ORG, LOC, DATE, TIME
///
/// # Examples
/// ```rust,ignore
/// assert!(types_match_flexible("PERSON", "PER"));
/// assert!(types_match_flexible("GPE", "LOC"));
/// assert!(types_match_flexible("MISC", "OTHER"));
/// ```
pub fn types_match_flexible(pred: &str, gold: &str) -> bool {
    let pred = pred.to_uppercase();
    let gold = gold.to_uppercase();

    if pred == gold {
        return true;
    }

    // Normalize to canonical groups for comparison
    let pred_canonical = normalize_entity_type(&pred);
    let gold_canonical = normalize_entity_type(&gold);

    pred_canonical == gold_canonical && pred_canonical != "UNKNOWN"
}

/// Normalize entity type to canonical form for flexible matching.
/// This is used internally by types_match_flexible.
fn normalize_entity_type(entity_type: &str) -> &'static str {
    match entity_type {
        // Person variants
        "PERSON" | "PER" | "PEOPLE" => "PERSON",
        // Location variants (including GPE for geopolitical entities)
        "LOCATION" | "LOC" | "GPE" | "GEO" | "PLACE" | "FAC" | "FACILITY" => "LOCATION",
        // Organization variants
        "ORGANIZATION" | "ORG" | "COMPANY" | "CORP" | "AGENCY" => "ORGANIZATION",
        // Miscellaneous/Other variants
        "MISC" | "MISCELLANEOUS" | "OTHER" | "O" => "MISC",
        // Date/Time variants
        "DATE" | "TIME" | "YEAR" | "MONTH" | "DAY" | "HOURS" | "DURATION" => "TEMPORAL",
        // Money variants
        "MONEY" | "CURRENCY" | "PRICE" | "AMOUNT" => "MONEY",
        // Quantity variants
        "QUANTITY" | "NUMBER" | "CARDINAL" | "ORDINAL" | "PERCENT" | "PERCENTAGE" => "QUANTITY",
        // Event variants
        "EVENT" | "INCIDENT" | "OCCURRENCE" => "EVENT",
        // Product variants
        "PRODUCT" | "WORK_OF_ART" | "ARTWORK" => "PRODUCT",
        // Language variants
        "LANGUAGE" | "LANG" => "LANGUAGE",
        // Nationalities, Religions, Political groups (NORP) - distinct from PERSON
        "NORP" | "NATIONALITY" | "RELIGION" | "POLITICAL_GROUP" => "NORP",
        // Law variants
        "LAW" | "LEGAL" | "REGULATION" => "LAW",
        // Anything else is unknown
        _ => "UNKNOWN",
    }
}

/// Normalize entity type to canonical form (owned String version).
/// Exported for use in per-type stats aggregation.
///
/// Handles case-insensitive input and returns owned String.
///
/// # Examples
/// ```rust,ignore
/// assert_eq!(normalize_entity_type_canonical("per"), "PERSON".to_string());
/// assert_eq!(normalize_entity_type_canonical("GPE"), "LOCATION".to_string());
/// ```
pub fn normalize_entity_type_canonical(entity_type: &str) -> String {
    normalize_entity_type(&entity_type.to_uppercase()).to_string()
}

/// Normalize an entity name for grouping (lowercase, trim)
pub fn normalize_entity_name(name: &str) -> String {
    let mut s = name.trim().to_string();
    if s.is_empty() {
        return s;
    }

    // Lowercase for case-insensitive grouping.
    s = s.to_lowercase();

    fn is_edge_punct(c: char) -> bool {
        matches!(
            c,
            '.' | ','
                | ';'
                | ':'
                | '!'
                | '?'
                | '"'
                | '\''
                | '’'
                | '“'
                | '”'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
        )
    }

    // Strip leading/trailing punctuation/quotes that often cling to spans.
    while s.chars().next().is_some_and(is_edge_punct) {
        s.remove(0);
    }
    while s.chars().last().is_some_and(is_edge_punct) {
        s.pop();
    }

    // Strip English possessives for grouping ("openai's" -> "openai").
    // Keep conservative: only strip at the very end.
    if s.ends_with("'s") || s.ends_with("’s") {
        s.pop(); // 's'
        s.pop(); // apostrophe
    } else if s.ends_with("s'") || s.ends_with("s’") {
        s.pop(); // apostrophe
    }

    // Strip trailing punctuation again (e.g., "openai's," -> "openai")
    while s.chars().last().is_some_and(is_edge_punct) {
        s.pop();
    }

    // Collapse internal whitespace.
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Simple heuristic for determining if a name is likely male.
pub fn is_likely_male(name: &str) -> bool {
    // Get first name (first word)
    let first_name = name.split_whitespace().next().unwrap_or("").to_lowercase();

    // Common male first names
    let male_names = [
        "james", "john", "robert", "michael", "william", "david", "richard", "joseph", "thomas",
        "charles", "barack", "donald", "joe", "george", "bill", "vladimir", "emmanuel", "boris",
        "xi", "narendra", "justin", "elon", "jeff", "mark", "steve", "tim", "satya", "sundar",
        "albert", "isaac", "stephen", "neil", "peter", "paul", "matthew", "andrew", "philip",
        "simon",
    ];

    male_names.contains(&first_name.as_str())
}

/// Simple heuristic for determining if a name is likely female.
pub fn is_likely_female(name: &str) -> bool {
    // Get first name (first word)
    let first_name = name.split_whitespace().next().unwrap_or("").to_lowercase();

    // Common female first names
    let female_names = [
        "mary",
        "patricia",
        "jennifer",
        "linda",
        "elizabeth",
        "angela",
        "marie",
        "susan",
        "margaret",
        "dorothy",
        "hillary",
        "nancy",
        "kamala",
        "michelle",
        "melania",
        "jill",
        "theresa",
        "ursula",
        "christine",
        "sanna",
        "jacinda",
        "oprah",
        "beyonce",
        "taylor",
        "sheryl",
        "marissa",
        "susan",
        "ginni",
        "diana",
        "catherine",
        "anne",
        "victoria",
        "queen",
        "jane",
        "sarah",
    ];

    female_names.contains(&first_name.as_str())
}

/// Resolve coreference by grouping signals into tracks.
///
/// This uses a simple rule-based approach:
/// 1. Named entities of the same type that overlap in their canonical form
/// 2. Pronouns detected in text and linked to nearest compatible antecedent
pub fn resolve_coreference(doc: &mut GroundedDocument, text: &str, signal_ids: &[SignalId]) {
    // Pronouns by gender
    let male_pronouns = ["he", "him", "his"];
    let female_pronouns = ["she", "her", "hers"];
    let neutral_pronouns = ["they", "them", "their", "theirs"]; // Can refer to any
    let org_pronouns = ["it", "its"];

    // First, detect pronouns in text that weren't found by NER
    // (pronoun_id, pronoun_type: "male", "female", "org", "any")
    let mut pronoun_signals: Vec<(SignalId, &str)> = Vec::new();

    // Build byte-to-char offset mapping for proper conversion
    let byte_to_char: Vec<usize> = text.char_indices().map(|(byte_idx, _)| byte_idx).collect();
    let char_count = text.chars().count();

    // Helper to convert byte offset to char offset
    let byte_to_char_offset = |byte_offset: usize| -> usize {
        byte_to_char
            .iter()
            .position(|&b| b == byte_offset)
            .unwrap_or_else(|| {
                // If exact match not found, it's at the end or between chars
                if byte_offset >= text.len() {
                    char_count
                } else {
                    // Find the char that contains this byte
                    byte_to_char
                        .iter()
                        .take_while(|&&b| b < byte_offset)
                        .count()
                }
            })
    };

    // Find pronouns in text and add them as signals
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();

    for (pronouns, ptype) in [
        (&male_pronouns[..], "male"),
        (&female_pronouns[..], "female"),
        (&org_pronouns[..], "org"),
        (&neutral_pronouns[..], "any"),
    ] {
        for &pronoun in pronouns {
            // Find all occurrences (byte offsets from find())
            let mut byte_start = 0;
            while let Some(pos) = text_lower[byte_start..].find(pronoun) {
                let abs_byte_start = byte_start + pos;
                let abs_byte_end = abs_byte_start + pronoun.len();

                // Convert to character offsets for Location::Text
                let char_start = byte_to_char_offset(abs_byte_start);
                let char_end = byte_to_char_offset(abs_byte_end);

                // Check word boundaries using character indices
                let is_word_start = char_start == 0
                    || !chars
                        .get(char_start.saturating_sub(1))
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);
                let is_word_end = char_end >= char_count
                    || !chars
                        .get(char_end)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);

                if is_word_start && is_word_end {
                    // Check if this position already has a signal (using char offsets)
                    let already_exists = doc.signals().iter().any(|s| {
                        if let Location::Text {
                            start: s_start,
                            end: s_end,
                        } = &s.location
                        {
                            *s_start == char_start && *s_end == char_end
                        } else {
                            false
                        }
                    });

                    if !already_exists {
                        // Add pronoun as a signal (use byte slice for surface text)
                        let surface = &text[abs_byte_start..abs_byte_end];
                        let signal = Signal::new(
                            SignalId::ZERO,                       // ID will be assigned by add_signal
                            Location::text(char_start, char_end), // Use char offsets
                            surface,
                            "PRON", // Special label for pronouns
                            0.9,
                        );
                        let sig_id = doc.add_signal(signal);
                        pronoun_signals.push((sig_id, ptype));
                    }
                }

                // Advance by one byte to find next occurrence
                byte_start = abs_byte_start + 1;
            }
        }
    }

    // Group NER signals by type
    let mut per_signals: Vec<SignalId> = Vec::new();
    let mut org_signals: Vec<SignalId> = Vec::new();
    let mut loc_signals: Vec<SignalId> = Vec::new();

    for &sig_id in signal_ids {
        if let Some(sig) = doc.get_signal(sig_id) {
            let label_lower = sig.label.to_lowercase();
            match label_lower.as_str() {
                "per" | "person" => per_signals.push(sig_id),
                "org" | "organization" => org_signals.push(sig_id),
                "loc" | "location" | "gpe" => loc_signals.push(sig_id),
                _ => {}
            }
        }
    }

    // Create tracks from named entities (group same-type entities by canonical form)
    let mut track_assignments: HashMap<SignalId, TrackId> = HashMap::new(); // signal_id -> track_id

    // For each entity type, create tracks by grouping similar surface forms
    for signals in [&per_signals, &org_signals, &loc_signals] {
        if signals.is_empty() {
            continue;
        }

        // Simple grouping: each unique entity gets its own track
        let mut canonical_groups: HashMap<String, Vec<SignalId>> = HashMap::new();

        for &sig_id in signals {
            if let Some(sig) = doc.get_signal(sig_id) {
                // Use lowercase canonical form for grouping
                let canonical = normalize_entity_name(&sig.surface);
                canonical_groups.entry(canonical).or_default().push(sig_id);
            }
        }

        // Create a track for each group
        for (canonical, group_signals) in canonical_groups {
            let track_id = doc.create_track_from_signals(&canonical, &group_signals);
            if let Some(tid) = track_id {
                for &sig_id in &group_signals {
                    track_assignments.insert(sig_id, tid);
                }
            }
        }
    }

    // Link pronouns to nearest compatible antecedent's track
    for (pronoun_id, pronoun_type) in &pronoun_signals {
        let pronoun_sig = match doc.get_signal(*pronoun_id) {
            Some(s) => s.clone(),
            None => continue,
        };

        let pronoun_start = match &pronoun_sig.location {
            Location::Text { start, .. } => *start,
            _ => continue,
        };

        // For person pronouns, we need to filter by gender compatibility
        let compatible_signals: Vec<SignalId> = match *pronoun_type {
            "male" => {
                // Filter to likely male names
                per_signals
                    .iter()
                    .filter(|&&id| {
                        doc.get_signal(id)
                            .map(|s| is_likely_male(&s.surface))
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            }
            "female" => {
                // Filter to likely female names
                per_signals
                    .iter()
                    .filter(|&&id| {
                        doc.get_signal(id)
                            .map(|s| is_likely_female(&s.surface))
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            }
            "org" => org_signals.clone(),
            "any" => per_signals
                .iter()
                .chain(org_signals.iter())
                .cloned()
                .collect(),
            _ => continue,
        };

        let mut nearest: Option<(SignalId, usize)> = None; // (signal_id, distance)

        for sig_id in &compatible_signals {
            if let Some(sig) = doc.get_signal(*sig_id) {
                if let Location::Text { end, .. } = &sig.location {
                    if *end < pronoun_start {
                        let distance = pronoun_start - end;
                        let should_update = match nearest {
                            None => true,
                            Some((_, prev_dist)) => distance < prev_dist,
                        };
                        if should_update {
                            nearest = Some((*sig_id, distance));
                        }
                    }
                }
            }
        }

        // Add pronoun to the track of its antecedent
        if let Some((antecedent_id, _)) = nearest {
            if let Some(&track_id) = track_assignments.get(&antecedent_id) {
                // Get position (number of signals already in track)
                let position = doc
                    .get_track(track_id)
                    .map(|t| t.signals.len() as u32)
                    .unwrap_or(0);
                // Add pronoun signal to this track (updates index)
                if doc.add_signal_to_track(*pronoun_id, track_id, position) {
                    track_assignments.insert(*pronoun_id, track_id);
                }
            }
        }
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

/// Create relation predictions from entity pairs using heuristics.
#[cfg(feature = "eval-advanced")]
pub fn create_entity_pair_relations(
    entities: &[Entity],
    text: &str,
    _relation_types: &[&str], // Kept for API compatibility, but no longer used
) -> Vec<anno_eval::eval::relation::RelationPrediction> {
    use anno_eval::eval::relation::RelationPrediction;

    let text_char_len = text.chars().count();
    let max_distance = 200; // Increased from 100 to catch cross-sentence relations

    let mut pred_relations = Vec::new();

    // Validate entities first to avoid panics
    let valid_entities: Vec<&Entity> = entities
        .iter()
        .filter(|e| e.start < e.end && e.end <= text_char_len && e.start < text_char_len)
        .collect();

    // Limit to avoid O(n²) explosion with many entities
    // Only consider pairs from first 50 entities to keep it tractable
    let max_entities = 50.min(valid_entities.len());

    for i in 0..max_entities {
        for j in (i + 1)..max_entities {
            let head = valid_entities[i];
            let tail = valid_entities[j];

            // Calculate distance using character offsets
            let distance = if tail.start >= head.end {
                tail.start - head.end
            } else if head.start >= tail.end {
                head.start - tail.end
            } else {
                // Overlapping entities - skip (they can't have a relation)
                continue;
            };

            if distance > max_distance {
                continue;
            }

            // Extract text between entities using character offsets (not byte offsets)
            let between_text = if head.end <= tail.start {
                text.chars()
                    .skip(head.end)
                    .take(tail.start - head.end)
                    .collect::<String>()
            } else {
                text.chars()
                    .skip(tail.end)
                    .take(head.start - tail.end)
                    .collect::<String>()
            };

            // Simple regex matching for common relations
            // Note: Using explicit "UNKNOWN" fallback to avoid inflating metrics
            // for any particular gold relation type
            let between_lower = between_text.to_lowercase();
            let rel_type = if between_lower.contains("founded") || between_lower.contains("founder")
            {
                "FOUNDED"
            } else if between_lower.contains("works for")
                || between_lower.contains("employee")
                || between_lower.contains("employed")
            {
                "WORKS_FOR"
            } else if between_lower.contains("located in")
                || between_lower.contains("based in")
                || between_lower.contains("headquartered")
            {
                "LOCATED_IN"
            } else if between_lower.contains("born in") || between_lower.contains("native of") {
                "BORN_IN"
            } else if between_lower.contains("ceo of")
                || between_lower.contains("president of")
                || between_lower.contains("leads")
            {
                "HEAD_OF"
            } else if between_lower.contains("acquired")
                || between_lower.contains("bought")
                || between_lower.contains("merged")
            {
                "ACQUIRED"
            } else {
                // Use explicit "UNKNOWN" to avoid inflating metrics for gold types
                // This ensures unknown relations don't count as correct matches
                "UNKNOWN"
            };

            pred_relations.push(RelationPrediction {
                head_span: (head.start, head.end),
                head_type: head.entity_type.as_label().to_string(),
                tail_span: (tail.start, tail.end),
                tail_type: tail.entity_type.as_label().to_string(),
                relation_type: rel_type.to_string(),
                confidence: 0.5,
            });
        }
    }

    pred_relations
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "eval-advanced")]
    use anno_core::{Entity, EntityType};

    // -------------------------------------------------------------------------
    // types_match_flexible tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_types_match_flexible_exact() {
        assert!(types_match_flexible("PERSON", "PERSON"));
        assert!(types_match_flexible("LOC", "LOC"));
        assert!(types_match_flexible("ORG", "ORG"));
    }

    #[test]
    fn test_types_match_flexible_person_variants() {
        assert!(types_match_flexible("PERSON", "PER"));
        assert!(types_match_flexible("PER", "PERSON"));
        assert!(types_match_flexible("PEOPLE", "PER"));
        assert!(types_match_flexible("PER", "PEOPLE"));
    }

    #[test]
    fn test_types_match_flexible_location_variants() {
        // LOC and LOCATION
        assert!(types_match_flexible("LOCATION", "LOC"));
        assert!(types_match_flexible("LOC", "LOCATION"));
        // GPE (geopolitical) should match LOC
        assert!(types_match_flexible("GPE", "LOC"));
        assert!(types_match_flexible("LOC", "GPE"));
        assert!(types_match_flexible("GPE", "LOCATION"));
        // FAC (facility) is also location-ish
        assert!(types_match_flexible("FAC", "LOC"));
        assert!(types_match_flexible("FACILITY", "LOCATION"));
    }

    #[test]
    fn test_types_match_flexible_org_variants() {
        assert!(types_match_flexible("ORGANIZATION", "ORG"));
        assert!(types_match_flexible("ORG", "ORGANIZATION"));
        assert!(types_match_flexible("COMPANY", "ORG"));
        assert!(types_match_flexible("CORP", "ORGANIZATION"));
    }

    #[test]
    fn test_types_match_flexible_misc_other() {
        assert!(types_match_flexible("MISC", "OTHER"));
        assert!(types_match_flexible("OTHER", "MISC"));
        assert!(types_match_flexible("MISCELLANEOUS", "MISC"));
    }

    #[test]
    fn test_types_match_flexible_temporal() {
        assert!(types_match_flexible("DATE", "TIME"));
        assert!(types_match_flexible("YEAR", "DATE"));
        assert!(types_match_flexible("DURATION", "DATE"));
    }

    #[test]
    fn test_types_match_flexible_no_match() {
        // Different categories should not match
        assert!(!types_match_flexible("PERSON", "ORG"));
        assert!(!types_match_flexible("LOC", "PER"));
        assert!(!types_match_flexible("DATE", "PERSON"));
        // Unknown types should not match anything
        assert!(!types_match_flexible("FOOBAR", "PERSON"));
        assert!(!types_match_flexible("UNKNOWN_TYPE", "LOC"));
    }

    #[test]
    fn test_types_match_flexible_case_insensitive() {
        assert!(types_match_flexible("person", "PER"));
        assert!(types_match_flexible("Person", "per"));
        assert!(types_match_flexible("PERSON", "per"));
    }

    // -------------------------------------------------------------------------
    // normalize_entity_type tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalize_entity_type() {
        assert_eq!(normalize_entity_type("PERSON"), "PERSON");
        assert_eq!(normalize_entity_type("PER"), "PERSON");
        assert_eq!(normalize_entity_type("LOC"), "LOCATION");
        assert_eq!(normalize_entity_type("GPE"), "LOCATION");
        assert_eq!(normalize_entity_type("ORG"), "ORGANIZATION");
        assert_eq!(normalize_entity_type("MISC"), "MISC");
        assert_eq!(normalize_entity_type("OTHER"), "MISC");
        assert_eq!(normalize_entity_type("DATE"), "TEMPORAL");
        assert_eq!(normalize_entity_type("UNKNOWN_XYZ"), "UNKNOWN");
    }

    #[test]
    fn test_normalize_entity_type_canonical() {
        // Case insensitive
        assert_eq!(normalize_entity_type_canonical("per"), "PERSON");
        assert_eq!(normalize_entity_type_canonical("Person"), "PERSON");
        assert_eq!(normalize_entity_type_canonical("PERSON"), "PERSON");

        // Location variants
        assert_eq!(normalize_entity_type_canonical("gpe"), "LOCATION");
        assert_eq!(normalize_entity_type_canonical("loc"), "LOCATION");

        // NORP is now separate from LANGUAGE
        assert_eq!(normalize_entity_type_canonical("NORP"), "NORP");
        assert_eq!(normalize_entity_type_canonical("LANGUAGE"), "LANGUAGE");

        // Unknown falls through
        assert_eq!(normalize_entity_type_canonical("FOOBAR"), "UNKNOWN");
    }

    // -------------------------------------------------------------------------
    // is_negated tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_negated() {
        assert!(is_negated("He is not a doctor", 12)); // "doctor" at 12
        assert!(is_negated("I never saw John", 11)); // "John" at 11
        assert!(!is_negated("He is a doctor", 8)); // "doctor" at 8
        assert!(!is_negated("The quick brown fox", 4)); // "quick" at 4
    }

    // -------------------------------------------------------------------------
    // detect_quantifier tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_detect_quantifier() {
        assert_eq!(
            detect_quantifier("Every student passed", 6),
            Some(Quantifier::Universal)
        );
        assert_eq!(
            detect_quantifier("Some students failed", 5),
            Some(Quantifier::Existential)
        );
        assert_eq!(
            detect_quantifier("No student failed", 3),
            Some(Quantifier::None)
        );
        assert_eq!(
            detect_quantifier("The student passed", 4),
            Some(Quantifier::Definite)
        );
        assert_eq!(detect_quantifier("Students passed", 0), None);
    }

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
    // normalize_entity_name tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalize_entity_name() {
        assert_eq!(normalize_entity_name("John Smith"), "john smith");
        assert_eq!(normalize_entity_name("  APPLE INC  "), "apple inc");
        assert_eq!(normalize_entity_name("Cupertino,"), "cupertino");
        assert_eq!(normalize_entity_name("OpenAI's"), "openai");
        assert_eq!(normalize_entity_name("OpenAI’s"), "openai");
        assert_eq!(normalize_entity_name(""), "");
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

    #[cfg(feature = "eval-advanced")]
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

    #[cfg(feature = "eval-advanced")]
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

    #[cfg(feature = "eval-advanced")]
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

    #[cfg(feature = "eval-advanced")]
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
    // Unicode tests (character offsets, not byte offsets)
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_negated_unicode() {
        // "café" has 4 chars but 5 bytes (é is 2 bytes in UTF-8)
        assert!(!is_negated("café John", 5)); // "John" starts at char 5
        assert!(is_negated("not café John", 9)); // "not" is in the prefix
    }

    #[test]
    fn test_detect_quantifier_unicode() {
        // "every café employee" - "employee" starts at char index 11
        assert_eq!(
            detect_quantifier("every café employee", 11),
            None // "café" is not a quantifier
        );
        // "every employee" still works
        assert_eq!(
            detect_quantifier("every employee", 6),
            Some(Quantifier::Universal)
        );
    }

    // -------------------------------------------------------------------------
    // Gender heuristics tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_likely_male() {
        assert!(is_likely_male("John Smith"));
        assert!(is_likely_male("Barack Obama"));
        assert!(!is_likely_male("Lynn Conway"));
        assert!(!is_likely_male("Unknown Person"));
    }

    #[test]
    fn test_is_likely_female() {
        assert!(is_likely_female("Lynn Conway"));
        assert!(is_likely_female("Hillary Clinton"));
        assert!(!is_likely_female("John Smith"));
        assert!(!is_likely_female("Unknown Person"));
    }
}
