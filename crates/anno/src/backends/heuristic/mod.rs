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

use crate::{Entity, EntityType, ExtractionMethod, Language, Model, Provenance, Result};

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
    // Multi-word entity suffixes (services, technologies, etc.)
    "services",
    "technologies",
    "systems",
    "partners",
    "solutions",
    "industries",
    "enterprises",
    "laboratories",
    "labs",
    "association",
    "council",
    "commission",
    "committee",
    "authority",
    "bureau",
    "department",
    "ministry",
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
const SKIP_WORDS: &[&str] = &[
    // Job titles
    "ceo",
    "cto",
    "cfo",
    "coo",
    "vp",
    "president",
    "chairman",
    "director",
    "manager",
    "secretary",
    "treasurer",
    // Form field labels (capitalized in forms/UIs, not entities)
    "phone",
    "fax",
    "mobile",
    "telephone",
    "address",
    "website",
    "name",
    "occupation",
    "company",
    // Medical/financial/form field labels (capitalized at sentence start, not entities)
    "patient",
    "card",
    "account",
    "member",
    "recipient",
    "sender",
    "holder",
    "applicant",
    "invoice",
    "payment",
    "balance",
    "total",
    "subject",
    "contact",
    "reference",
    // German political titles (role words, not entity names)
    "bundeskanzler",
    "kanzler",
    "praesident",
    "minister",
    "buergermeister",
];

// Common acronyms/abbreviations that are not entities.
// All entries are lowercase (compared against lowercased input).
// Only checked against all-caps tokens (via is_acronym_word), so mixed-case names
// like "Ai" or "Id" are NOT affected. Entries are intentionally limited to
// international standards (ISO, tech protocols, currency codes) to avoid
// language/domain bias.
const COMMON_ACRONYMS: &[&str] = &[
    // Tech
    "lcd", "led", "html", "css", "http", "https", "api", "url", "dna", "rna", "cpu", "gpu", "ram",
    "rom", "usb", "pdf", "sql", "xml", "json", "csv", "atm", "gps", "wifi", "lan", "wan", "vpn",
    "ssl", "tls", "ssh", "ftp", "iso", "jpg", "png", "gif", "svg", "mp3", "mp4", "avi", "hd",
    "uhd", "ac", "dc", "tv", "pc", "os", "ui", "ux", "ai", "ml", "id", "ip", "io",
    // Units / time
    "mph", "rpm", "gmt", "utc", "am", "pm", // Economics / finance (not entities)
    "gdp", "gnp", "cpi", "roi", "ebitda", "ipo", "etf", "apy", "apr",
    // International abbreviations (used across languages)
    "etc", "aka", "eta",
    // Currency codes (not entities -- handled by regex/pattern backend)
    "usd", "eur", "gbp", "jpy", "chf", "cad", "aud", "nzd", "cny", "krw", "inr", "brl", "mxn",
    "sgd", "hkd", "sek", "nok", "dkk", "pln", "czk",
];

// Words that commonly start sentences but are not entities.
// English-only: this filter exploits capitalization, which is a Latin-script signal.
// CJK and other caseless scripts use a separate extraction path above.
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
    // Day names (capitalized but not entities)
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    // Month names (capitalized but not entities)
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
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
    // Common title-cased words in headlines/UI (not entities).
    // NOTE: avoid words that commonly START multi-word entity names
    // (e.g., "new" -> "New York", "commission" -> "European Commission").
    "here",
    "found",
    "listen",
    "switch",
    "download",
    "submit",
    "apply",
    "search",
    "about",
    "subscribe",
    "read",
    "more",
    "show",
    "hide",
    "next",
    "previous",
    "back",
    "home",
    "old",
    "update",
    "updated",
    "published",
    // Literary/formal sentence starters (often capitalized but not entities)
    "often",
    "nevertheless",
    "however",
    "meanwhile",
    "furthermore",
    "moreover",
    "therefore",
    "otherwise",
    "although",
    "indeed",
    "perhaps",
    "certainly",
    "apparently",
    "obviously",
    "no",
    "yes",
    "oh",
    "out",
    "up",
    "off",
    "upon",
    "still",
    "once",
    "even",
    "just",
    "never",
    "always",
    "only",
    "silent",
    "belated",
    "soon",
    "later",
    "such",
    "most",
    "some",
    "many",
    "each",
    "every",
    "both",
    "all",
    "few",
    "much",
    "several",
    "other",
    "another",
    "any",
    "after",
    "almost",
];

// Minimal lexical knowledge (50 items each - high ROI)
// These are the most common entities that are hard to distinguish structurally
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

const KNOWN_PERSONS: &[&str] = &[
    "john", "jane", "mary", "james", "robert", "michael", "william", "david", "richard", "joseph",
    "thomas", "charles", "barack", "donald", "joe", "george", "bill", "vladimir", "emmanuel",
    "boris", "narendra", "justin", "jeff", "mark", "steve", "tim", "satya", "sundar", "albert",
    "isaac", "stephen", "neil", "peter", "paul", "matthew", "andrew", "philip", "simon", "marie",
    "angela", "hillary", "nancy", "kamala", "michelle", "melania", "jill", "theresa", "ursula",
];

impl Model for HeuristicNER {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
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
                            .any(|e| e.start() == char_start && e.end() == char_end)
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
                            .any(|e| e.start() == char_start && e.end() == char_end)
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

            // Skip title/role words at span start (e.g., "Bundeskanzler", "CEO").
            // They're not entities themselves, and the following name will form its own span.
            if SKIP_WORDS.contains(&first_word_clean) {
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

                // Break span at title words (CEO, President, etc.) when mid-span.
                // Prevents "Apple CEO Tim Cook" from being bundled as one ORG span;
                // instead produces "Apple" and "CEO Tim Cook" which classify independently.
                if i > start_idx && first_char_upper {
                    let w_lower = w_clean.to_lowercase();
                    let w_trimmed = w_lower.trim_end_matches(|c: char| !c.is_alphanumeric());
                    if SKIP_WORDS.contains(&w_trimmed) {
                        break;
                    }
                }

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
                // Use original text positions instead of joined text length
                let char_start = words_with_pos[start_idx - 1].1;
                let char_end = words_with_pos[end_idx - 1].2;

                // Classify based on minimal rules
                let clean_span_words: Vec<&str> = entity_text.split_whitespace().collect();
                let (entity_type, confidence, reason) =
                    classify_minimal(&clean_span_words, &words, start_idx - 1);

                // Skip low-confidence and filtered entities
                if confidence >= self.threshold && confidence > 0.0 {
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
                            raw_confidence: Some(confidence.into()),
                            model_version: None,
                            timestamp: None,
                        },
                    ));
                }
                continue; // Skip the normal processing below
            }

            // Compute offsets from original text positions
            let raw_char_start = words_with_pos[start_idx].1;
            let raw_char_end = words_with_pos[end_idx - 1].2;

            // Clean leading punctuation (count chars, not bytes)
            let leading_punct_chars = entity_text
                .chars()
                .take_while(|c| !c.is_alphanumeric())
                .count();
            if leading_punct_chars > 0 {
                entity_text = entity_text.chars().skip(leading_punct_chars).collect();
            }

            // Clean trailing punctuation
            let trailing_punct_chars = entity_text
                .chars()
                .rev()
                .take_while(|c| !c.is_alphanumeric())
                .count();
            for _ in 0..trailing_punct_chars {
                entity_text.pop();
            }

            // Skip if entity became empty after cleaning
            if entity_text.is_empty() {
                continue;
            }

            let char_start = raw_char_start + leading_punct_chars;
            let char_end = raw_char_end - trailing_punct_chars;

            // Classify based on minimal rules
            // Use cleaned span for classification to avoid punctuation noise
            let clean_span_words: Vec<&str> = entity_text.split_whitespace().collect();
            let (entity_type, confidence, reason) =
                classify_minimal(&clean_span_words, &words, start_idx);

            // Skip low-confidence and filtered entities
            if confidence >= self.threshold && confidence > 0.0 {
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
                        raw_confidence: Some(confidence.into()),
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

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities::default()
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
        return (
            EntityType::custom("skip", anno_core::EntityCategory::Misc),
            0.0,
            "skip_pronoun",
        );
    }
    // Filter: Skip fiscal quarter abbreviations (Q1-Q4) and multi-word fiscal
    // patterns like "Q3 FY2025", "Q1 2024" -- not entities
    {
        let trimmed = first_word.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed.len() == 2 {
            let bytes = trimmed.as_bytes();
            if (bytes[0] == b'Q' || bytes[0] == b'q') && (bytes[1] >= b'1' && bytes[1] <= b'4') {
                // Single "Q3" or multi-word "Q3 FY2025", "Q3 2024"
                if span.len() == 1 {
                    return (
                        EntityType::custom("skip", anno_core::EntityCategory::Misc),
                        0.0,
                        "skip_fiscal_quarter",
                    );
                }
                // Check if remaining words are fiscal year indicators
                let rest_is_fiscal = span[1..].iter().all(|w| {
                    let wl = w.to_lowercase();
                    wl.starts_with("fy")
                        || wl.chars().all(|c| c.is_ascii_digit())
                        || wl == "h1"
                        || wl == "h2"
                });
                if rest_is_fiscal {
                    return (
                        EntityType::custom("skip", anno_core::EntityCategory::Misc),
                        0.0,
                        "skip_fiscal_quarter",
                    );
                }
            }
        }
    }

    // Filter: Skip job titles and common non-entity nouns
    let first_clean_lc = first_word
        .trim_end_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if span.len() == 1 && SKIP_WORDS.contains(&first_clean_lc.as_str()) {
        return (
            EntityType::custom("skip", anno_core::EntityCategory::Misc),
            0.0,
            "skip_word",
        );
    }
    // Filter: Skip standalone person prefixes (Dr, Mr, Prof) -- they'll be
    // absorbed into the next span via the prefix-inclusion logic.
    if span.len() == 1 && PERSON_PREFIX.contains(&first_clean_lc.as_str()) {
        return (
            EntityType::custom("skip", anno_core::EntityCategory::Misc),
            0.0,
            "skip_prefix",
        );
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
                let clean = lc.trim_matches(|c: char| !c.is_alphanumeric());
                !SKIP_WORDS.contains(&clean) && !COMMON_ACRONYMS.contains(&clean)
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
        if span[1].to_lowercase() == "of" {
            return (EntityType::Organization, 0.65, "org_of_pattern");
        }
        // Title-prefixed name: "CEO Shuntaro Furukawa", "President Barack Obama"
        // First word is a job title -> rest is likely a person name
        let first_clean_lower = first_word
            .trim_end_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if SKIP_WORDS.contains(&first_clean_lower.as_str()) {
            return (EntityType::Person, 0.65, "title_prefixed_name");
        }
        return (EntityType::Organization, 0.50, "long_span_org");
    }

    // Rule 9: Single-word structural signals
    if span.len() == 1 {
        let word = span[0].trim_matches(|c: char| !c.is_alphanumeric());
        if word.len() == 1 {
            return (
                EntityType::custom("skip", anno_core::EntityCategory::Misc),
                0.0,
                "single_letter",
            );
        }
        if is_acronym_word(word) {
            let lc = word.to_lowercase();
            if SKIP_WORDS.contains(&lc.as_str()) || COMMON_ACRONYMS.contains(&lc.as_str()) {
                return (
                    EntityType::custom("skip", anno_core::EntityCategory::Misc),
                    0.0,
                    "skip_acronym",
                );
            }
            return (EntityType::Organization, 0.55, "single_acronym");
        }
    }

    // Rule 9.5: Hyphenated compounds with a non-entity prefix (e.g., "DNA-based", "LCD-equipped")
    if span.len() == 1 {
        let word = span[0].trim_matches(|c: char| !c.is_alphanumeric());
        if let Some(prefix) = word.split('-').next() {
            let prefix_lc = prefix.to_lowercase();
            if COMMON_ACRONYMS.contains(&prefix_lc.as_str()) {
                return (
                    EntityType::custom("skip", anno_core::EntityCategory::Misc),
                    0.0,
                    "skip_hyphenated_acronym",
                );
            }
        }
    }

    // Rule 10: Single word at sentence start -- low confidence.
    // Sentence-initial capitalization is unreliable: headlines ("Death toll rises"),
    // German grammar (all nouns capitalized), and section headers all produce false
    // positives. Detect sentence start by: no prev_word, or prev_word ends with
    // sentence-terminal punctuation.
    let is_sentence_start = prev_word.is_none()
        || prev_word
            .as_ref()
            .map(|w| w.ends_with('.') || w.ends_with('!') || w.ends_with('?'))
            .unwrap_or(false);
    if is_sentence_start {
        return (EntityType::Person, 0.30, "single_start_word");
    }

    // Default: single capitalized word mid-sentence - assume Person
    (EntityType::Person, 0.45, "capitalized")
}

#[cfg(test)]
mod tests;
