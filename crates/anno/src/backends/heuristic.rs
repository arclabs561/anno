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
            // Performance: Build byte-to-char mapping once for this text.
            // ROI: High - reused across all CJK substring matches.
            let converter = crate::offset::SpanConverter::new(text);

            for &org in KNOWN_ORGS {
                // Simple substring search for CJK terms
                if org.chars().any(|c| c >= '\u{3040}') {
                    let org_char_count = if org.is_ascii() {
                        org.len()
                    } else {
                        org.chars().count()
                    };

                    // Include Hiragana/Katakana
                    // Standard library substring search - efficient for typical NER workloads
                    for (start_byte, _) in text.match_indices(org) {
                        let char_start = converter.byte_to_char(start_byte);
                        let char_end = char_start + org_char_count;
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
                    let loc_char_count = if loc.is_ascii() {
                        loc.len()
                    } else {
                        loc.chars().count()
                    };

                    // Standard library substring search - efficient for typical NER workloads
                    for (start_byte, _) in text.match_indices(loc) {
                        let char_start = converter.byte_to_char(start_byte);
                        let char_end = char_start + loc_char_count;
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
            // Performance: Use entity_text.len() for ASCII, fallback to chars().count() for Unicode
            let char_end = char_start
                + if entity_text.is_ascii() {
                    entity_text.len()
                } else {
                    entity_text.chars().count()
                };

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

/// Domain-agnostic, language-agnostic acronym check.  True when every
/// alphabetic character is uppercase and there are at least 2 alphabetic
/// characters.  Unicode-aware: works for Latin (NASA), Cyrillic (НАТО),
/// and gracefully returns false for caseless scripts (CJK, Arabic).
fn is_acronym_word(w: &str) -> bool {
    let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
    let alpha_count = clean.chars().filter(|c| c.is_alphabetic()).count();
    alpha_count >= 2
        && clean
            .chars()
            .filter(|c| c.is_alphabetic())
            .all(|c| c.is_uppercase())
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

    // Rule 5.5: Acronym signal (domain-agnostic, language-agnostic).
    // Excludes SKIP_WORDS (CEO/CTO/VP) which are role titles, not entities.
    if span.len() >= 2 {
        let has_real_acronym = span.iter().any(|w| {
            is_acronym_word(w) && {
                let lc = w.to_lowercase();
                !SKIP_WORDS.contains(&lc.trim_matches(|c: char| !c.is_alphanumeric()))
            }
        });
        if has_real_acronym {
            return (EntityType::Organization, 0.70, "acronym_in_span");
        }
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

    // Rule 9: Single-word structural signals
    if span.len() == 1 {
        let word = span[0].trim_matches(|c: char| !c.is_alphanumeric());
        if word.len() == 1 {
            return (EntityType::Other("skip".into()), 0.0, "single_letter");
        }
        if is_acronym_word(word) {
            let lc = word.to_lowercase();
            if !SKIP_WORDS.contains(&lc.as_str()) {
                return (EntityType::Organization, 0.55, "single_acronym");
            }
        }
    }

    // Rule 10: Single word at sentence start with no context - very low confidence
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_person_detection() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("Dr. John Smith met with Mary.", None)
            .unwrap();

        let names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            names
                .iter()
                .any(|n| n.contains("John") || n.contains("Smith")),
            "Should detect John Smith: {:?}",
            names
        );
    }

    #[test]
    fn test_organization_suffix_detection() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("Apple Inc. announced new products.", None)
            .unwrap();

        let orgs: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Organization))
            .collect();
        assert!(!orgs.is_empty(), "Should detect Apple Inc. as organization");
    }

    #[test]
    fn test_location_preposition_context() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("She lived in Paris for years.", None)
            .unwrap();

        let locs: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Location))
            .collect();
        assert!(!locs.is_empty(), "Should detect Paris as location");
    }

    #[test]
    fn test_known_organizations() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("Google and Microsoft competed.", None)
            .unwrap();

        let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            texts.iter().any(|t| t.contains("Google")),
            "Should detect Google"
        );
        assert!(
            texts.iter().any(|t| t.contains("Microsoft")),
            "Should detect Microsoft"
        );
    }

    #[test]
    fn test_cjk_organization_detection() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("ソニーが新製品を発表しました。", None)
            .unwrap();

        let orgs: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Organization))
            .collect();
        assert!(
            !orgs.is_empty(),
            "Should detect Sony (ソニー) as organization"
        );
    }

    #[test]
    fn test_cjk_location_detection() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("東京オリンピックが開催された。", None)
            .unwrap();

        let locs: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Location))
            .collect();
        assert!(!locs.is_empty(), "Should detect Tokyo (東京) as location");
    }

    #[test]
    fn test_empty_text() {
        let ner = HeuristicNER::new();
        let entities = ner.extract_entities("", None).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_no_entities() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("the quick brown fox jumps over the lazy dog", None)
            .unwrap();
        // All lowercase, no entities expected
        assert!(
            entities.is_empty(),
            "Lowercase text should have no entities"
        );
    }

    #[test]
    fn test_threshold_filtering() {
        let low_threshold = HeuristicNER::with_threshold(0.1);
        let high_threshold = HeuristicNER::with_threshold(0.9);

        let text = "John works at Google.";
        let low_entities = low_threshold.extract_entities(text, None).unwrap();
        let high_entities = high_threshold.extract_entities(text, None).unwrap();

        // Lower threshold should capture more or equal entities
        assert!(low_entities.len() >= high_entities.len());
    }

    #[test]
    fn test_sentence_starter_filtering() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("The dog ran. It was fast.", None)
            .unwrap();

        // "The" and "It" should be filtered as common sentence starters
        let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            !texts.contains(&"The"),
            "Should filter 'The' as sentence starter"
        );
        assert!(!texts.contains(&"It"), "Should filter 'It' as pronoun");
    }

    #[test]
    fn test_person_prefix_detection() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("Prof. Einstein presented the theory.", None)
            .unwrap();

        let persons: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Person))
            .collect();
        assert!(
            !persons.is_empty(),
            "Should detect Prof. Einstein as person"
        );
    }

    #[test]
    fn test_multi_word_organization() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("Bank of America provides services.", None)
            .unwrap();

        let orgs: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Organization))
            .collect();
        assert!(!orgs.is_empty(), "Should detect 'Bank of America' pattern");
    }

    #[test]
    fn test_location_indicators() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("New Zealand is beautiful.", None)
            .unwrap();

        let locs: Vec<_> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Location))
            .collect();
        assert!(!locs.is_empty(), "Should detect 'New Zealand' as location");
    }

    #[test]
    fn test_model_trait_implementation() {
        let ner = HeuristicNER::new();

        assert_eq!(ner.name(), "heuristic");
        assert!(ner.is_available());
        assert!(!ner.supported_types().is_empty());
        assert!(ner.description().contains("Heuristic"));
    }

    #[test]
    fn test_entity_offsets_are_valid() {
        let ner = HeuristicNER::new();
        let text = "Barack Obama visited Berlin yesterday.";
        let entities = ner.extract_entities(text, None).unwrap();

        let char_count = text.chars().count();
        for entity in &entities {
            assert!(entity.start <= entity.end, "start should be <= end");
            assert!(entity.end <= char_count, "end should be within text");

            // Verify text matches span
            let extracted: String = text
                .chars()
                .skip(entity.start)
                .take(entity.end - entity.start)
                .collect();
            assert_eq!(
                extracted, entity.text,
                "Extracted text should match entity text"
            );
        }
    }

    #[test]
    fn test_unicode_text_handling() {
        let ner = HeuristicNER::new();
        let text = "François Müller from München met José García.";
        let entities = ner.extract_entities(text, None).unwrap();

        // Should handle diacritics correctly
        for entity in &entities {
            let extracted: String = text
                .chars()
                .skip(entity.start)
                .take(entity.end - entity.start)
                .collect();
            assert_eq!(extracted, entity.text, "Unicode offsets should be correct");
        }
    }

    #[test]
    fn test_provenance_is_set() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("Google announced today.", None)
            .unwrap();

        for entity in &entities {
            if let Some(ref prov) = entity.provenance {
                assert_eq!(prov.source, "heuristic");
                assert!(matches!(prov.method, ExtractionMethod::Heuristic));
            }
        }
    }

    // =========================================================================
    // Acronym signal tests (domain-agnostic, language-agnostic)
    // =========================================================================

    #[test]
    fn test_is_acronym_word_latin() {
        assert!(is_acronym_word("PARC"));
        assert!(is_acronym_word("IBM"));
        assert!(is_acronym_word("NASA"));
        assert!(is_acronym_word("N2K"));
        assert!(is_acronym_word("DARPA."));
        assert!(is_acronym_word("(NATO)"));
        assert!(!is_acronym_word("Xerox"));
        assert!(!is_acronym_word("Lynn"));
        assert!(!is_acronym_word("A"));
        assert!(!is_acronym_word("42"));
        assert!(!is_acronym_word(""));
    }

    #[test]
    fn test_is_acronym_word_cyrillic() {
        assert!(is_acronym_word("\u{041D}\u{0410}\u{0422}\u{041E}")); // НАТО
        assert!(is_acronym_word("\u{041C}\u{0418}\u{0414}"));         // МИД
        assert!(!is_acronym_word("\u{041C}\u{043E}\u{0441}\u{043A}\u{0432}\u{0430}")); // Москва
    }

    #[test]
    fn test_is_acronym_word_caseless_scripts() {
        assert!(!is_acronym_word("\u{6771}\u{4EAC}"));   // 東京 (CJK)
        assert!(!is_acronym_word("\u{30BD}\u{30CB}\u{30FC}")); // ソニー (Katakana)
        assert!(!is_acronym_word("\u{062D}\u{0645}\u{0627}\u{0633}")); // حماس (Arabic)
    }

    #[test]
    fn test_acronym_in_multi_word_span_signals_org() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities(
                "Lynn Conway worked at IBM and Xerox PARC in California.",
                None,
            )
            .unwrap();
        let xerox_parc = entities.iter().find(|e| e.text == "Xerox PARC");
        assert!(xerox_parc.is_some(), "Should detect 'Xerox PARC': {entities:?}");
        assert!(
            matches!(xerox_parc.unwrap().entity_type, EntityType::Organization),
            "Xerox PARC should be ORG, got {:?}",
            xerox_parc.unwrap().entity_type,
        );
    }

    #[test]
    fn test_acronym_no_regression_on_normal_names() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("Lynn Conway designed the processor.", None)
            .unwrap();
        let lynn = entities.iter().find(|e| e.text == "Lynn Conway");
        assert!(lynn.is_some(), "Should detect 'Lynn Conway': {entities:?}");
        assert!(
            matches!(lynn.unwrap().entity_type, EntityType::Person),
            "Lynn Conway should remain PER, got {:?}",
            lynn.unwrap().entity_type,
        );
    }

    #[test]
    fn test_single_acronym_signals_org() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("She joined DARPA last year.", None)
            .unwrap();
        let darpa = entities.iter().find(|e| e.text == "DARPA");
        assert!(darpa.is_some(), "Should detect 'DARPA': {entities:?}");
        assert!(
            matches!(darpa.unwrap().entity_type, EntityType::Organization),
            "DARPA should be ORG, got {:?}",
            darpa.unwrap().entity_type,
        );
    }

    #[test]
    fn test_known_loc_acronym_still_loc() {
        let ner = HeuristicNER::new();
        let entities = ner
            .extract_entities("She moved to USA last year.", None)
            .unwrap();
        let usa = entities.iter().find(|e| e.text == "USA");
        assert!(usa.is_some(), "Should detect 'USA': {entities:?}");
        assert!(
            matches!(usa.unwrap().entity_type, EntityType::Location),
            "USA should be LOC (gazetteer wins), got {:?}",
            usa.unwrap().entity_type,
        );
    }
}
