//! Utility functions for CLI commands

use is_terminal::IsTerminal;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};

use anno_core::{GroundedDocument, Identity, Location, Quantifier, Signal};

/// Get input text from various sources (text arg, file, or stdin)
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
            return Ok(buf);
        }
    }

    Err("No input text provided. Use -t 'text' or -f file or pipe via stdin".to_string())
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

#[derive(Debug, Clone)]
pub struct GoldSpec {
    pub text: String,
    pub label: String,
    pub start: usize,
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
    let prefix: String = text.chars().take(entity_start).collect();
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
    let prefix: String = text.chars().take(entity_start).collect();
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
pub fn types_match_flexible(pred: &str, gold: &str) -> bool {
    let pred = pred.to_uppercase();
    let gold = gold.to_uppercase();

    if pred == gold {
        return true;
    }

    // Allow common mappings
    match (pred.as_str(), gold.as_str()) {
        // Person
        ("PERSON", "PER") | ("PER", "PERSON") => true,
        // Location
        ("LOCATION", "LOC") | ("LOC", "LOCATION") | ("LOCATION", "GPE") | ("GPE", "LOCATION") => {
            true
        }
        // Organization
        ("ORGANIZATION", "ORG") | ("ORG", "ORGANIZATION") => true,
        // Date/Time
        ("DATE", "YEAR") | ("YEAR", "DATE") | ("DATE", "HOURS") => true,
        _ => false,
    }
}

/// Normalize an entity name for grouping (lowercase, trim)
pub fn normalize_entity_name(name: &str) -> String {
    name.to_lowercase().trim().to_string()
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
pub fn resolve_coreference(doc: &mut GroundedDocument, text: &str, signal_ids: &[u64]) {
    // Pronouns by gender
    let male_pronouns = ["he", "him", "his"];
    let female_pronouns = ["she", "her", "hers"];
    let neutral_pronouns = ["they", "them", "their", "theirs"]; // Can refer to any
    let org_pronouns = ["it", "its"];

    // First, detect pronouns in text that weren't found by NER
    // (pronoun_id, pronoun_type: "male", "female", "org", "any")
    let mut pronoun_signals: Vec<(u64, &str)> = Vec::new();

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
                            0,
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
    let mut per_signals: Vec<u64> = Vec::new();
    let mut org_signals: Vec<u64> = Vec::new();
    let mut loc_signals: Vec<u64> = Vec::new();

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
    let mut track_assignments: HashMap<u64, u64> = HashMap::new(); // signal_id -> track_id

    // For each entity type, create tracks by grouping similar surface forms
    for signals in [&per_signals, &org_signals, &loc_signals] {
        if signals.is_empty() {
            continue;
        }

        // Simple grouping: each unique entity gets its own track
        let mut canonical_groups: HashMap<String, Vec<u64>> = HashMap::new();

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
        let compatible_signals: Vec<u64> = match *pronoun_type {
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

        let mut nearest: Option<(u64, usize)> = None; // (signal_id, distance)

        for sig_id in &compatible_signals {
            if let Some(sig) = doc.get_signal(*sig_id) {
                if let Location::Text { end, .. } = &sig.location {
                    if *end < pronoun_start {
                        let distance = pronoun_start - end;
                        if nearest.map_or(true, |(_, prev_dist)| distance < prev_dist) {
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

/// Link tracks to KB identities.
///
/// Creates placeholder Wikidata-style identities for each track.
/// In a production system, this would query a real KB like Wikidata.
pub fn link_tracks_to_kb(doc: &mut GroundedDocument) {
    // Well-known entities with Wikidata IDs
    let known_entities: HashMap<&str, (&str, &str)> = [
        (
            "barack obama",
            ("Q76", "44th President of the United States"),
        ),
        ("angela merkel", ("Q567", "Chancellor of Germany 2005-2021")),
        ("berlin", ("Q64", "Capital of Germany")),
        ("nato", ("Q7184", "North Atlantic Treaty Organization")),
        (
            "donald trump",
            ("Q22686", "45th President of the United States"),
        ),
        (
            "joe biden",
            ("Q6279", "46th President of the United States"),
        ),
        ("vladimir putin", ("Q7747", "President of Russia")),
        ("emmanuel macron", ("Q3052772", "President of France")),
        ("elon musk", ("Q317521", "CEO of Tesla and SpaceX")),
        ("marie curie", ("Q7186", "Physicist and chemist")),
        ("albert einstein", ("Q937", "Theoretical physicist")),
        ("new york", ("Q60", "City in New York State")),
        ("london", ("Q84", "Capital of the United Kingdom")),
        ("paris", ("Q90", "Capital of France")),
        ("google", ("Q95", "American technology company")),
        ("apple", ("Q312", "American technology company")),
        ("microsoft", ("Q2283", "American technology company")),
        ("united nations", ("Q1065", "International organization")),
        ("european union", ("Q458", "Political and economic union")),
    ]
    .into_iter()
    .collect();

    // Collect track IDs first to avoid borrow issues
    let track_ids: Vec<u64> = doc.tracks().map(|t| t.id).collect();

    for track_id in track_ids {
        let (canonical, entity_type) = {
            let track = match doc.get_track(track_id) {
                Some(t) => t,
                None => continue,
            };
            (track.canonical_surface.clone(), track.entity_type.clone())
        };

        let canonical_lower = canonical.to_lowercase();

        // Look up in known entities
        if let Some(&(qid, description)) = known_entities.get(canonical_lower.as_str()) {
            // Create identity from KB
            let mut identity = Identity::from_kb(
                0, // Will be assigned by add_identity
                &canonical, "wikidata", qid,
            );
            identity.aliases.push(description.to_string());
            if let Some(etype) = &entity_type {
                identity.entity_type = Some(etype.clone());
            }

            let identity_id = doc.add_identity(identity);
            doc.link_track_to_identity(track_id, identity_id);
        } else {
            // Create placeholder identity without KB link
            let identity = Identity::new(0, &canonical);
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
    relation_types: &[&str],
) -> Vec<anno::eval::relation::RelationPrediction> {
    use anno::eval::relation::RelationPrediction;

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
                || between_lower.contains("in ")
            {
                "LOCATED_IN"
            } else if between_lower.contains("born in") {
                "BORN_IN"
            } else {
                // Use first relation type from gold data as fallback, or "RELATED"
                relation_types.first().copied().unwrap_or("RELATED")
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
