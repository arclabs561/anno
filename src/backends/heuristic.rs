//! Heuristic NER - optimized for low Kolmogorov complexity
//!
//! A heuristic-based NER model that achieves reasonable performance with minimal
//! complexity. The goal is to minimize description length while maximizing
//! downstream quality.
//!
//! Core principles:
//! 1. Exploit structural signals (capitalization, punctuation) - "free" features
//! 2. Use high-precision patterns (Inc., Dr., in/from) - small fixed cost
//! 3. Avoid large lexicons - high description cost per marginal gain

use crate::{Entity, EntityType, ExtractionMethod, Model, Provenance, Result};

/// Heuristic NER model.
///
/// A heuristic-based NER model optimized for low Kolmogorov complexity.
/// Uses high-precision patterns with minimal lexical resources.
#[derive(Debug, Clone)]
pub struct HeuristicNER {
    /// Minimum confidence threshold for entity extraction.
    threshold: f64,
}

impl Default for HeuristicNER {
    fn default() -> Self {
        Self { threshold: 0.35 }
    }
}

impl HeuristicNER {
    /// Create a new HeuristicNER instance with default threshold.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new HeuristicNER with a custom confidence threshold.
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }
}

// High-precision patterns (small, fixed cost)
const ORG_SUFFIX: &[&str] = &[
    "inc.",
    "inc",
    "corp.",
    "corp",
    "ltd.",
    "ltd",
    "llc",
    "co.",
    "plc",
    "foundation",
    "institute",
    "university",
    "college",
    "bank",
    "group",
    "agency",
    // International suffixes
    "gmbh",
    "ag",
    "kg",
    "sa",
    "s.a.",
    "s.l.",
    "s.r.l.",
    "spa",
    "nv",
    "bv",
    "pty",
    "ab",
    "limited",
    "corporation",
    "incorporated",
    "company",
    "holding",
    "holdings",
];
const PERSON_PREFIX: &[&str] = &[
    "mr.", "mr", "ms.", "ms", "mrs.", "mrs", "dr.", "dr", "prof.", "prof",
];
const LOC_PREPOSITION: &[&str] = &[
    "in", "from", "at", "to", "near", // German
    "aus", "nach", "bei", "von", // French/Spanish/Italian
    "en", "de", "à", "dans", "por", "sur",
];
// Common words that look like entities but aren't (all caps acronyms, titles)
#[allow(dead_code)] // Used in classify_minimal
const SKIP_WORDS: &[&str] = &[
    "ceo",
    "cto",
    "cfo",
    "vp",
    "president",
    "chairman",
    "director",
];

// Words that commonly start sentences but are not entities
const COMMON_SENTENCE_STARTERS: &[&str] = &[
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "he",
    "she",
    "we",
    "they",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "with",
    "and",
    "but",
    "or",
    "so",
    "yet",
    "if",
    "because",
    "contact",
    "call",
    "email",
    "visit",
    "please",
    "see",
    "note",
    "today",
    "yesterday",
    "tomorrow",
    "now",
    "then",
    "what",
    "where",
    "when",
    "who",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
];

// Minimal lexical knowledge (50 items each - high ROI)
// These are the most common entities that are hard to distinguish structurally
#[allow(dead_code)] // Used in classify_minimal but compiler sometimes misses it
const KNOWN_ORGS: &[&str] = &[
    "google",
    "apple",
    "microsoft",
    "amazon",
    "facebook",
    "meta",
    "tesla",
    "twitter",
    "ibm",
    "intel",
    "nvidia",
    "oracle",
    "cisco",
    "samsung",
    "sony",
    "toyota",
    "honda",
    "bmw",
    "mercedes",
    "volkswagen",
    "nasa",
    "fbi",
    "cia",
    "nsa",
    "nato",
    "un",
    "eu",
    "bbc",
    "cnn",
    "nbc",
    "cbs",
    "abc",
    "fox",
    "nyt",
    "wsj",
    "reuters",
    "bloomberg",
    "spotify",
    "netflix",
    "uber",
    "airbnb",
    "paypal",
    "visa",
    "mastercard",
    "amex",
    // CJK Orgs
    "ソニー",
    "トヨタ",
    "ホンダ",
    "任天堂",
    "サムスン",
    "ファーウェイ",
    "アリババ",
    "テンセント",
    "华为",
    "阿里巴巴",
    "腾讯",
    "百度",
    "小米",
];

#[allow(dead_code)] // Used in classify_minimal but compiler sometimes misses it
const KNOWN_LOCS: &[&str] = &[
    "paris",
    "london",
    "tokyo",
    "berlin",
    "rome",
    "madrid",
    "moscow",
    "beijing",
    "shanghai",
    "dubai",
    "singapore",
    "sydney",
    "toronto",
    "chicago",
    "boston",
    "california",
    "texas",
    "florida",
    "new york",
    "washington",
    "europe",
    "asia",
    "africa",
    "america",
    "australia",
    "china",
    "india",
    "japan",
    "germany",
    "france",
    "italy",
    "spain",
    "brazil",
    "mexico",
    "russia",
    "korea",
    "canada",
    "uk",
    "usa",
    // CJK Locs
    "東京",
    "大阪",
    "京都",
    "北京",
    "上海",
    "香港",
    "ソウル",
    "台北",
    "中国",
    "日本",
    "韓国",
    "アメリカ",
    "イギリス",
    "フランス",
    "ドイツ",
];

#[allow(dead_code)] // Used in classify_minimal
const KNOWN_PERSONS: &[&str] = &[
    "john", "jane", "mary", "james", "robert", "michael", "william", "david", "richard", "joseph",
    "thomas", "charles", "barack", "donald", "joe", "george", "bill", "vladimir", "emmanuel",
    "boris", "narendra", "justin", "elon", "jeff", "mark", "steve", "tim", "satya", "sundar",
    "albert", "isaac", "stephen", "neil", "peter", "paul", "matthew", "andrew", "philip", "simon",
    "marie", "angela", "hillary", "nancy", "kamala", "michelle", "melania", "jill", "theresa",
    "ursula",
];

impl Model for HeuristicNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let mut entities: Vec<Entity> = Vec::new();

        // CJK Detection & Extraction
        // Since CJK doesn't use spaces, we scan for known entities directly
        let has_cjk = text.chars().any(
            |c| {
                ('\u{4e00}'..='\u{9fff}').contains(&c) || // CJK Unified Ideographs
            ('\u{3040}'..='\u{309f}').contains(&c) || // Hiragana
            ('\u{30a0}'..='\u{30ff}').contains(&c)
            }, // Katakana
        );

        if has_cjk {
            for &org in KNOWN_ORGS {
                // Simple substring search for CJK terms
                if org.chars().any(|c| c >= '\u{3040}') {
                    // Include Hiragana/Katakana
                    for (start_byte, _) in text.match_indices(org) {
                        let char_start = text[..start_byte].chars().count();
                        let char_end = char_start + org.chars().count();
                        // Avoid duplicates if already found (simple overlap check)
                        if !entities
                            .iter()
                            .any(|e| e.start == char_start && e.end == char_end)
                        {
                            entities.push(Entity::new(
                                org.to_string(),
                                EntityType::Organization,
                                char_start,
                                char_end,
                                0.9,
                            ));
                        }
                    }
                }
            }
            for &loc in KNOWN_LOCS {
                if loc.chars().any(|c| c >= '\u{3040}') {
                    for (start_byte, _) in text.match_indices(loc) {
                        let char_start = text[..start_byte].chars().count();
                        let char_end = char_start + loc.chars().count();
                        if !entities
                            .iter()
                            .any(|e| e.start == char_start && e.end == char_end)
                        {
                            entities.push(Entity::new(
                                loc.to_string(),
                                EntityType::Location,
                                char_start,
                                char_end,
                                0.9,
                            ));
                        }
                    }
                }
            }
        }

        // Build word list with character positions
        // Robust strategy: Scan text linearly, identifying word boundaries.
        // This avoids synchronization issues between split_whitespace and find.
        let mut words_with_pos: Vec<(&str, usize, usize)> = Vec::new();

        let mut in_word = false;
        let mut word_start_byte = 0;
        let mut word_start_char = 0;
        let mut char_pos = 0;

        for (i, c) in text.char_indices() {
            if c.is_whitespace() {
                if in_word {
                    // Word ended
                    let word = &text[word_start_byte..i];
                    words_with_pos.push((word, word_start_char, char_pos));
                    in_word = false;
                }
            } else if !in_word {
                in_word = true;
                word_start_byte = i;
                word_start_char = char_pos;
            }
            char_pos += 1;
        }
        // Last word
        if in_word {
            let word = &text[word_start_byte..];
            words_with_pos.push((word, word_start_char, char_pos));
        }

        let words: Vec<&str> = words_with_pos.iter().map(|(w, _, _)| *w).collect();

        let mut i = 0;
        while i < words.len() {
            let word = words[i];

            // Pre-check: Clean leading punctuation before checking capitalization
            let clean_leading = word.trim_start_matches(|c: char| !c.is_alphanumeric());
            if clean_leading.is_empty() {
                i += 1;
                continue;
            }

            // Only consider capitalized words as candidates
            if !clean_leading
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                i += 1;
                continue;
            }

            // Find span of consecutive capitalized words
            // Only allow "of" and "the" as connectors (not "and" which separates entities)
            let start_idx = i;

            // Filter: Skip common sentence starters if this is the first word of a span
            let first_word_lower = word.to_lowercase();
            let first_word_clean = first_word_lower.trim_matches(|c: char| !c.is_alphanumeric());
            if COMMON_SENTENCE_STARTERS.contains(&first_word_clean) {
                i += 1;
                continue;
            }

            while i < words.len() {
                let w = words[i];
                let w_clean = w.trim_start_matches(|c: char| !c.is_alphanumeric());

                // Check if word ends with closing parenthesis - implies end of group
                // Also check for sentence boundaries (. ! ?) unless it's a known suffix like Inc. or Mr.
                // NOTE: We assume '.' inside a word (e.g. U.S.A.) is fine, but '.' at end is boundary.
                // Unless next word is lower case (abbreviation).
                let ends_with_closing = w.ends_with([')', ']', '}']);
                let ends_with_punct = w.ends_with(['.', '!', '?']);

                let first_char_upper = w_clean
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);

                // Only "of" and "the" connect entity names (e.g., "Bank of America", "The New York Times")
                // "and" separates entities (e.g., "Paris and London" are two entities)
                let is_connector = matches!(w.to_lowercase().as_str(), "of" | "the");

                // Check next word
                let next_word_ok = if i + 1 < words.len() {
                    let next = words[i + 1];
                    let next_clean = next.trim_start_matches(|c: char| !c.is_alphanumeric());
                    let next_upper = next_clean
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false);

                    // Special case: "Inc", "Corp" etc can follow a closing parenthesis
                    // e.g. "Google) Inc" -> merged
                    let is_suffix = ORG_SUFFIX.contains(&&*next_clean.to_lowercase());

                    if (ends_with_closing || ends_with_punct) && !is_suffix {
                        false // Break span at closing parenthesis/punctuation unless followed by suffix
                    } else {
                        next_upper
                    }
                } else {
                    false
                };

                if first_char_upper || (is_connector && next_word_ok) {
                    i += 1;
                    // If this word ended with closing parenthesis, and we didn't break above (because next is suffix),
                    // continue. If we broke above, loop terminates.
                    if ends_with_closing || ends_with_punct {
                        let is_suffix_next = if let Some(next_w) = words.get(i) {
                            let clean = next_w.to_lowercase();
                            let clean_ref = clean.trim_matches(|c: char| !c.is_alphanumeric());
                            ORG_SUFFIX.contains(&clean_ref)
                        } else {
                            false
                        };

                        if !is_suffix_next {
                            break;
                        }
                    }
                } else {
                    break;
                }
            }
            let end_idx = i;

            if start_idx == end_idx {
                continue;
            }

            // Extract the span
            let span_words = &words[start_idx..end_idx];
            let mut entity_text = span_words.join(" ");

            // Check if previous word is a person prefix (e.g., "Dr.", "Mr.")
            let prev_word = if start_idx > 0 {
                Some(
                    words[start_idx - 1]
                        .to_lowercase()
                        .trim_end_matches('.')
                        .to_string(),
                )
            } else {
                None
            };
            let should_include_prefix = prev_word
                .as_ref()
                .map(|p| PERSON_PREFIX.contains(&p.as_str()))
                .unwrap_or(false);

            // If previous word is a person prefix, include it in the entity text
            if should_include_prefix {
                let prefix_word = &words[start_idx - 1];
                entity_text = format!("{} {}", prefix_word, entity_text);
                // Adjust start position to include prefix
                let prefix_char_start = words_with_pos[start_idx - 1].1;
                let char_start = prefix_char_start;
                let char_end = char_start + entity_text.chars().count();

                // Classify based on minimal rules
                let clean_span_words: Vec<&str> = entity_text.split_whitespace().collect();
                let (entity_type, confidence, reason) =
                    classify_minimal(&clean_span_words, &words, start_idx - 1);

                // Skip low-confidence and filtered entities
                if confidence >= self.threshold && !matches!(entity_type, EntityType::Other(_)) {
                    entities.push(Entity::with_provenance(
                        entity_text,
                        entity_type,
                        char_start,
                        char_end,
                        confidence,
                        Provenance {
                            source: "heuristic".into(),
                            method: ExtractionMethod::Heuristic,
                            pattern: Some(reason.into()),
                            raw_confidence: Some(confidence),
                            model_version: None,
                            timestamp: None,
                        },
                    ));
                }
                continue; // Skip the normal processing below
            }

            // Clean leading punctuation from first word (but not person prefixes)
            let leading_punct_len = entity_text.len()
                - entity_text
                    .trim_start_matches(|c: char| !c.is_alphanumeric())
                    .len();
            if leading_punct_len > 0 {
                entity_text = entity_text[leading_punct_len..].to_string();
            }

            // Clean trailing punctuation from the last word
            while entity_text.ends_with(|c: char| !c.is_alphanumeric()) {
                entity_text.pop();
            }

            // Skip if entity became empty after cleaning
            if entity_text.is_empty() {
                continue;
            }

            // Get character offsets from our position tracking
            // Correct start offset by adding leading punctuation length
            let char_start = words_with_pos[start_idx].1 + leading_punct_len;
            let char_end = char_start + entity_text.chars().count();

            // Classify based on minimal rules
            // Use cleaned span for classification to avoid punctuation noise
            let clean_span_words: Vec<&str> = entity_text.split_whitespace().collect();
            let (entity_type, confidence, reason) =
                classify_minimal(&clean_span_words, &words, start_idx);

            // Skip low-confidence and filtered entities
            if confidence >= self.threshold && !matches!(entity_type, EntityType::Other(_)) {
                entities.push(Entity::with_provenance(
                    entity_text,
                    entity_type,
                    char_start,
                    char_end,
                    confidence,
                    Provenance {
                        source: "heuristic".into(),
                        method: ExtractionMethod::Heuristic,
                        pattern: Some(reason.into()),
                        raw_confidence: Some(confidence),
                        model_version: None,
                        timestamp: None,
                    },
                ));
            }
        }

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "heuristic"
    }

    fn description(&self) -> &'static str {
        "Heuristic NER optimized for low complexity"
    }
}

fn classify_minimal(
    span: &[&str],
    all_words: &[&str],
    start_idx: usize,
) -> (EntityType, f64, &'static str) {
    let last_word = span.last().map(|s| s.to_lowercase()).unwrap_or_default();
    let first_word = span.first().map(|s| s.to_lowercase()).unwrap_or_default();
    let span_lower = span
        .iter()
        .map(|s| s.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    // Get context
    let prev_word = if start_idx > 0 {
        Some(all_words[start_idx - 1].to_lowercase())
    } else {
        None
    };

    // Filter: Skip common pronouns/articles/titles that get capitalized
    let skip_pronouns = [
        "the", "a", "an", "he", "she", "it", "they", "we", "i", "you",
    ];
    if span.len() == 1 && skip_pronouns.contains(&first_word.as_str()) {
        return (EntityType::Other("skip".into()), 0.0, "skip_pronoun");
    }
    // Filter: Skip job titles and common non-entity nouns
    let first_clean_lc = first_word
        .trim_end_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if span.len() == 1 && SKIP_WORDS.contains(&first_clean_lc.as_str()) {
        return (EntityType::Other("skip".into()), 0.0, "skip_word");
    }

    // Rule 1: ORG suffix (highest precision)
    let last_clean: &str = last_word.trim_end_matches(|c: char| !c.is_alphanumeric());
    if ORG_SUFFIX.contains(&last_clean) {
        return (EntityType::Organization, 0.85, "org_suffix");
    }

    // Rule 2: Known organization name
    let first_clean_text = first_word.trim_end_matches(|c: char| !c.is_alphanumeric());
    if KNOWN_ORGS.contains(&first_clean_text) || KNOWN_ORGS.contains(&span_lower.as_str()) {
        return (EntityType::Organization, 0.80, "known_org");
    }

    // Rule 3: Known location name
    if KNOWN_LOCS.contains(&first_clean_text) || KNOWN_LOCS.contains(&span_lower.as_str()) {
        return (EntityType::Location, 0.80, "known_location");
    }

    // Rule 3.5: Known person name
    if KNOWN_PERSONS.contains(&first_clean_text) {
        return (EntityType::Person, 0.75, "common_name");
    }

    // Rule 4: Person prefix in previous word
    if let Some(prev) = &prev_word {
        let prev_clean: &str = prev.trim_end_matches('.');
        if PERSON_PREFIX.contains(&prev_clean) {
            return (EntityType::Person, 0.80, "person_prefix_context");
        }
    }

    // Rule 5: First word is a title -> Person
    let first_clean: &str = first_word.trim_end_matches('.');
    if PERSON_PREFIX.contains(&first_clean) && span.len() >= 2 {
        return (EntityType::Person, 0.75, "person_prefix_span");
    }

    // Rule 6: Location preposition context
    if let Some(prev) = &prev_word {
        if LOC_PREPOSITION.contains(&prev.as_str()) {
            return (EntityType::Location, 0.70, "loc_context");
        }
    }

    // Rule 7: Two capitalized words (likely person name)
    // Unless it looks like a country/place (contains "United", "New", etc.)
    if span.len() == 2 {
        let place_indicators = ["united", "new", "south", "north", "west", "east", "great"];
        if place_indicators.contains(&first_word.as_str()) {
            return (EntityType::Location, 0.65, "loc_indicator");
        }
        return (EntityType::Person, 0.60, "two_word_name");
    }

    // Rule 8: Three+ words -> likely ORG or LOC, not PER
    if span.len() >= 3 {
        // "Bank of X" pattern -> ORG
        if span.len() >= 2 && span[1].to_lowercase() == "of" {
            return (EntityType::Organization, 0.65, "org_of_pattern");
        }
        return (EntityType::Organization, 0.50, "long_span_org");
    }

    // Rule 9: Single word at sentence start with no context - very low confidence
    if start_idx == 0 && prev_word.is_none() {
        return (EntityType::Person, 0.30, "single_start_word");
    }

    // Default: single capitalized word mid-sentence - assume Person
    (EntityType::Person, 0.45, "capitalized")
}

impl crate::NamedEntityCapable for HeuristicNER {}

// =============================================================================
// BatchCapable Trait Implementation
// =============================================================================

impl crate::BatchCapable for HeuristicNER {
    fn optimal_batch_size(&self) -> Option<usize> {
        Some(16) // HeuristicNER is fast, can handle larger batches
    }
}

// =============================================================================
// StreamingCapable Trait Implementation
// =============================================================================

impl crate::StreamingCapable for HeuristicNER {
    fn recommended_chunk_size(&self) -> usize {
        8192 // Characters - heuristics are lightweight
    }
}
