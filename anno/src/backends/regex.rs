//! Regex-based NER - Extracts entities via regex patterns only.
//!
//! No hardcoded gazetteers. Only extracts entities that can be reliably
//! identified by their format:
//! - Dates: ISO 8601, MM/DD/YYYY, "January 15, 2024", "Jan 15"
//!   - Multilingual: Japanese 年月日, German/French/Spanish/Italian/Portuguese/Dutch months
//! - Times: "3:30 PM", "14:00", "10am"
//! - Money: $100, $1.5M, "50 dollars", €500
//! - Percentages: 15%, 3.5%
//! - Emails: user@example.com
//! - URLs: `https://example.com`
//! - Phone numbers: (555) 123-4567, +1-555-123-4567
//!
//! For Person/Organization/Location, use ML models (BERT ONNX, GLiNER).

use crate::{Entity, EntityType, Model, Result};
use once_cell::sync::Lazy;
use regex::Regex;

/// Regex-based NER - extracts entities with recognizable formats using regex patterns.
///
/// Reliable extraction without ML models. Does NOT attempt to identify
/// Person/Organization/Location - those require contextual understanding.
///
/// # Supported Entity Types
///
/// | Type | Examples |
/// |------|----------|
/// | Date | "2024-01-15", "January 15, 2024", "2024年1月15日", "15 Januar" |
/// | Time | "3:30 PM", "14:00", "10am" |
/// | Money | "$100", "€50", "5 million dollars" |
/// | Percent | "15%", "3.5%" |
/// | Email | "user@example.com" |
/// | URL | `https://example.com` |
/// | Phone | "(555) 123-4567", "+1-555-1234" |
///
/// # Example
///
/// ```rust
/// use anno::{RegexNER, Model};
///
/// let ner = RegexNER::new();
/// let entities = ner.extract_entities(
///     "Meeting at 3:30 PM on Jan 15. Contact: bob@acme.com",
///     None
/// ).unwrap();
///
/// assert!(entities.len() >= 3); // time, date, email
/// ```
pub struct RegexNER;

impl RegexNER {
    /// Create a new regex-based NER.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for RegexNER {
    fn default() -> Self {
        Self::new()
    }
}

// Static regex patterns - compiled once, reused forever
static DATE_ISO: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").expect("valid regex"));

static DATE_US: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b").expect("valid regex"));

static DATE_EU: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b").expect("valid regex"));

static DATE_WRITTEN_FULL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b").expect("valid regex")
});

static DATE_WRITTEN_SHORT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b").expect("valid regex")
});

static DATE_WRITTEN_EU: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?(?:\s+\d{4})?\b").expect("valid regex")
});

// =============================================================================
// Japanese Date Format: YYYY年MM月DD日
// =============================================================================

static DATE_JAPANESE: Lazy<Regex> = Lazy::new(|| {
    // Matches: 2024年1月15日, 2024年01月15日, etc.
    Regex::new(r"\d{4}年\d{1,2}月\d{1,2}日").expect("valid regex")
});

// =============================================================================
// Multilingual Month Names
// =============================================================================

// German months: Januar, Februar, März, April, Mai, Juni, Juli, August, September, Oktober, November, Dezember
static DATE_GERMAN_FULL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{1,2}(?:\.)?(?:,?\s*\d{4})?\b").expect("valid regex")
});

static DATE_GERMAN_EU: Lazy<Regex> = Lazy::new(|| {
    // "15. Januar 2024" or "15 Januar"
    Regex::new(r"(?i)\b\d{1,2}\.?\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)(?:\s+\d{4})?\b").expect("valid regex")
});

// French months: janvier, février, mars, avril, mai, juin, juillet, août, septembre, octobre, novembre, décembre
static DATE_FRENCH_FULL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+\d{1,2}(?:,?\s*\d{4})?\b").expect("valid regex")
});

static DATE_FRENCH_EU: Lazy<Regex> = Lazy::new(|| {
    // "15 janvier 2024" or "1er janvier"
    Regex::new(r"(?i)\b\d{1,2}(?:er)?\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)(?:\s+\d{4})?\b").expect("valid regex")
});

// Spanish months: enero, febrero, marzo, abril, mayo, junio, julio, agosto, septiembre, octubre, noviembre, diciembre
static DATE_SPANISH_EU: Lazy<Regex> = Lazy::new(|| {
    // "15 de enero de 2024" or "15 enero 2024"
    Regex::new(r"(?i)\b\d{1,2}\s+(?:de\s+)?(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+(?:de\s+)?\d{4})?\b").expect("valid regex")
});

// Italian months: gennaio, febbraio, marzo, aprile, maggio, giugno, luglio, agosto, settembre, ottobre, novembre, dicembre
static DATE_ITALIAN_EU: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)(?:\s+\d{4})?\b").expect("valid regex")
});

// Portuguese months: janeiro, fevereiro, março, abril, maio, junho, julho, agosto, setembro, outubro, novembro, dezembro
static DATE_PORTUGUESE_EU: Lazy<Regex> = Lazy::new(|| {
    // "15 de janeiro de 2024"
    Regex::new(r"(?i)\b\d{1,2}\s+(?:de\s+)?(?:janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)(?:\s+(?:de\s+)?\d{4})?\b").expect("valid regex")
});

// Dutch months: januari, februari, maart, april, mei, juni, juli, augustus, september, oktober, november, december
static DATE_DUTCH_EU: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d{1,2}\s+(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)(?:\s+\d{4})?\b").expect("valid regex")
});

// Russian months (Cyrillic): январь, февраль, март, апрель, май, июнь, июль, август, сентябрь, октябрь, ноябрь, декабрь
static DATE_RUSSIAN_EU: Lazy<Regex> = Lazy::new(|| {
    // "15 января 2024" - uses genitive case forms
    Regex::new(r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?\b").expect("valid regex")
});

// Chinese date format: YYYY年MM月DD日 (same as Japanese but also common)
// Already covered by DATE_JAPANESE

// Korean date format: YYYY년 MM월 DD일
static DATE_KOREAN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일").expect("valid regex"));

static TIME_12H: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b").expect("valid regex")
});

static TIME_24H: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b").expect("valid regex"));

static TIME_SIMPLE: Lazy<Regex> = Lazy::new(|| {
    // Note: No trailing \b because a.m./p.m. end with .
    Regex::new(r"(?i)\b\d{1,2}\s*(?:am\b|pm\b|a\.m\.|p\.m\.)").expect("valid regex")
});

static MONEY_SYMBOL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[$€£¥][\d,]+(?:\.\d{1,2})?(?:\s*(?:billion|million|thousand|B|M|K|bn|mn))?")
        .expect("valid regex")
});

static MONEY_WRITTEN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)\b\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP|yen|JPY)\b",
    )
    .expect("valid regex")
});

static MONEY_MAGNITUDE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)\b\d+(?:\.\d+)?\s*(?:billion|million|trillion)\s*(?:dollars?|euros?|pounds?)?\b",
    )
    .expect("valid regex")
});

static PERCENT: Lazy<Regex> = Lazy::new(|| {
    // Note: No trailing \b because % is not a word character
    Regex::new(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|pct\b)").expect("valid regex")
});

static EMAIL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b").expect("valid regex")
});

static URL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\bhttps?://[^\s<>\[\]{}|\\^`\x00-\x1f]+").expect("valid regex"));

static PHONE_US: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b").expect("valid regex")
});

static PHONE_INTL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b").expect("valid regex")
});

static MENTION: Lazy<Regex> = Lazy::new(|| {
    // @username - supports letters, numbers, underscore, dot (but not starting/ending with dot)
    Regex::new(r"\B@[\w](?:[\w.]*[\w])?").expect("valid regex")
});

static HASHTAG: Lazy<Regex> = Lazy::new(|| {
    // #hashtag - supports letters, numbers, underscore
    Regex::new(r"\B#\w+").expect("valid regex")
});

impl Model for RegexNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        use crate::offset::SpanConverter;
        use anno_core::Provenance;
        let mut entities = Vec::new();

        // Performance optimization: Build SpanConverter once for all byte-to-char conversions
        // ROI: High - called once per extract_entities, saves O(n) per regex match
        let converter = SpanConverter::new(text);

        // Helper to add entity if no overlap
        // Note: regex returns byte offsets, but we convert to char offsets
        // for consistency with evaluation (GoldEntity uses char offsets).
        let mut add_entity =
            |m: regex::Match, entity_type: EntityType, confidence: f64, pattern: &'static str| {
                // Convert byte offsets to character offsets for Unicode correctness
                // Use optimized SpanConverter instead of bytes_to_chars
                let char_start = converter.byte_to_char(m.start());
                let char_end = converter.byte_to_char(m.end());
                if !overlaps(&entities, char_start, char_end) {
                    entities.push(Entity::with_provenance(
                        m.as_str(),
                        entity_type,
                        char_start,
                        char_end,
                        confidence,
                        Provenance::pattern(pattern),
                    ));
                }
            };

        // Dates (high confidence - very specific patterns)
        // English dates
        let date_patterns_en: &[(&Lazy<Regex>, &'static str)] = &[
            (&DATE_ISO, "DATE_ISO"),
            (&DATE_US, "DATE_US"),
            (&DATE_EU, "DATE_EU"),
            (&DATE_WRITTEN_FULL, "DATE_WRITTEN_FULL"),
            (&DATE_WRITTEN_SHORT, "DATE_WRITTEN_SHORT"),
            (&DATE_WRITTEN_EU, "DATE_WRITTEN_EU"),
        ];
        for (pattern, name) in date_patterns_en {
            for m in pattern.find_iter(text) {
                add_entity(m, EntityType::Date, 0.95, name);
            }
        }

        // Multilingual dates (Japanese, Korean, German, French, Spanish, etc.)
        let date_patterns_i18n: &[(&Lazy<Regex>, &'static str)] = &[
            (&DATE_JAPANESE, "DATE_JAPANESE"),
            (&DATE_KOREAN, "DATE_KOREAN"),
            (&DATE_GERMAN_FULL, "DATE_GERMAN_FULL"),
            (&DATE_GERMAN_EU, "DATE_GERMAN_EU"),
            (&DATE_FRENCH_FULL, "DATE_FRENCH_FULL"),
            (&DATE_FRENCH_EU, "DATE_FRENCH_EU"),
            (&DATE_SPANISH_EU, "DATE_SPANISH_EU"),
            (&DATE_ITALIAN_EU, "DATE_ITALIAN_EU"),
            (&DATE_PORTUGUESE_EU, "DATE_PORTUGUESE_EU"),
            (&DATE_DUTCH_EU, "DATE_DUTCH_EU"),
            (&DATE_RUSSIAN_EU, "DATE_RUSSIAN_EU"),
        ];
        for (pattern, name) in date_patterns_i18n {
            for m in pattern.find_iter(text) {
                add_entity(m, EntityType::Date, 0.93, name); // Slightly lower confidence for i18n
            }
        }

        // Times
        let time_patterns: &[(&Lazy<Regex>, &'static str)] = &[
            (&TIME_12H, "TIME_12H"),
            (&TIME_24H, "TIME_24H"),
            (&TIME_SIMPLE, "TIME_SIMPLE"),
        ];
        for (pattern, name) in time_patterns {
            for m in pattern.find_iter(text) {
                add_entity(m, EntityType::Time, 0.90, name);
            }
        }

        // Money (high confidence)
        let money_patterns: &[(&Lazy<Regex>, &'static str)] = &[
            (&MONEY_SYMBOL, "MONEY_SYMBOL"),
            (&MONEY_WRITTEN, "MONEY_WRITTEN"),
            (&MONEY_MAGNITUDE, "MONEY_MAGNITUDE"),
        ];
        for (pattern, name) in money_patterns {
            for m in pattern.find_iter(text) {
                add_entity(m, EntityType::Money, 0.95, name);
            }
        }

        // Percentages
        for m in PERCENT.find_iter(text) {
            add_entity(m, EntityType::Percent, 0.95, "PERCENT");
        }

        // Emails (very high confidence - very specific pattern)
        for m in EMAIL.find_iter(text) {
            add_entity(m, EntityType::Email, 0.98, "EMAIL");
        }

        // URLs (very high confidence)
        for m in URL.find_iter(text) {
            add_entity(m, EntityType::Url, 0.98, "URL");
        }

        // Phone numbers (medium confidence - can have false positives)
        let phone_patterns: &[(&Lazy<Regex>, &'static str)] =
            &[(&PHONE_US, "PHONE_US"), (&PHONE_INTL, "PHONE_INTL")];
        for (pattern, name) in phone_patterns {
            for m in pattern.find_iter(text) {
                add_entity(m, EntityType::Phone, 0.85, name);
            }
        }

        // Social Media (@mentions and #hashtags) - note: mapping to Other for now as specific types don't exist yet
        for m in MENTION.find_iter(text) {
            // Using a custom "Mention" type via Other
            // In future refactor: Add EntityType::Mention
            let char_start = converter.byte_to_char(m.start());
            let char_end = converter.byte_to_char(m.end());
            if !overlaps(&entities, char_start, char_end) {
                // We use EntityType::Other for now, but specific string "Mention"
                entities.push(Entity::with_provenance(
                    m.as_str(),
                    EntityType::Other("Mention".to_string()),
                    char_start,
                    char_end,
                    0.95,
                    Provenance::pattern("MENTION"),
                ));
            }
        }

        for m in HASHTAG.find_iter(text) {
            let char_start = converter.byte_to_char(m.start());
            let char_end = converter.byte_to_char(m.end());
            if !overlaps(&entities, char_start, char_end) {
                entities.push(Entity::with_provenance(
                    m.as_str(),
                    EntityType::Other("Hashtag".to_string()),
                    char_start,
                    char_end,
                    0.95,
                    Provenance::pattern("HASHTAG"),
                ));
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort by position for consistent output
        entities.sort_unstable_by_key(|e| e.start);

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Date,
            EntityType::Time,
            EntityType::Money,
            EntityType::Percent,
            EntityType::Email,
            EntityType::Url,
            EntityType::Phone,
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "regex"
    }

    fn description(&self) -> &'static str {
        "Regex-based NER (dates, times, money, percentages, emails, URLs, phones)"
    }
}

/// Check if a span overlaps with existing entities.
fn overlaps(entities: &[Entity], start: usize, end: usize) -> bool {
    entities.iter().any(|e| !(end <= e.start || start >= e.end))
}

// Capability marker: RegexNER extracts structured entities via regex
impl crate::StructuredEntityCapable for RegexNER {}

#[cfg(test)]
mod tests {
    use super::*;

    fn ner() -> RegexNER {
        RegexNER::new()
    }

    fn extract(text: &str) -> Vec<Entity> {
        ner().extract_entities(text, None).unwrap()
    }

    fn has_type(entities: &[Entity], ty: &EntityType) -> bool {
        entities.iter().any(|e| &e.entity_type == ty)
    }

    fn count_type(entities: &[Entity], ty: &EntityType) -> usize {
        entities.iter().filter(|e| &e.entity_type == ty).count()
    }

    fn find_text<'a>(entities: &'a [Entity], text: &str) -> Option<&'a Entity> {
        entities.iter().find(|e| e.text == text)
    }

    // ========================================================================
    // Date Tests
    // ========================================================================

    #[test]
    fn date_iso_format() {
        let e = extract("Meeting on 2024-01-15.");
        assert!(find_text(&e, "2024-01-15").is_some());
    }

    #[test]
    fn date_us_format() {
        let e = extract("Due by 12/31/2024 and 1/5/24.");
        assert_eq!(count_type(&e, &EntityType::Date), 2);
    }

    #[test]
    fn date_eu_format() {
        let e = extract("Released on 31.12.2024.");
        assert!(find_text(&e, "31.12.2024").is_some());
    }

    #[test]
    fn date_written_full() {
        let cases = [
            "January 15, 2024",
            "February 28",
            "March 1st, 2024",
            "December 25th",
        ];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
        }
    }

    #[test]
    fn date_written_short() {
        let cases = ["Jan 15, 2024", "Feb 28", "Mar. 1st", "Dec 25th, 2024"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
        }
    }

    #[test]
    fn date_eu_written() {
        let cases = ["15 January 2024", "28th February", "1st March 2024"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
        }
    }

    // ========================================================================
    // Time Tests
    // ========================================================================

    #[test]
    fn time_12h_format() {
        let cases = ["3:30 PM", "10:00 am", "12:30:45 p.m.", "9:00 AM"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Time), "Failed: {}", case);
        }
    }

    #[test]
    fn time_24h_format() {
        let cases = ["14:30", "09:00", "23:59:59", "0:00"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Time), "Failed: {}", case);
        }
    }

    #[test]
    fn time_simple() {
        let cases = ["3pm", "10 AM", "9 a.m."];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Time), "Failed: {}", case);
        }
    }

    // ========================================================================
    // Money Tests
    // ========================================================================

    #[test]
    fn money_dollar_basic() {
        let cases = ["$100", "$1,000", "$99.99", "$1,234,567.89"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Money), "Failed: {}", case);
        }
    }

    #[test]
    fn money_with_magnitude() {
        let cases = ["$5 million", "$1.5B", "$100K", "$2 billion"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Money), "Failed: {}", case);
        }
    }

    #[test]
    fn money_other_currencies() {
        let cases = ["€500", "£100", "¥1000"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Money), "Failed: {}", case);
        }
    }

    #[test]
    fn money_unicode_offsets_correct() {
        // Regression test: Entity offsets must be CHARACTER offsets, not byte offsets.
        // Euro sign (€) is 3 bytes but 1 character.
        // This test catches the bug where regex byte offsets were stored directly.
        let text = "Price: €50 then €100";
        let ner = RegexNER::new();
        let entities = ner.extract_entities(text, None).unwrap();

        // "Price: " = 7 chars, so first € is at char 7
        // "€50 then " = 9 chars, so second € is at char 16
        let money: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .collect();

        assert_eq!(money.len(), 2, "Expected 2 money entities, got {:?}", money);

        // First entity: "€50" at char 7
        assert_eq!(money[0].start, 7, "First € should be at char 7, not byte 7");
        assert_eq!(money[0].end, 10, "First entity end should be char 10");

        // Second entity: "€100" at char 16
        assert_eq!(
            money[1].start, 16,
            "Second € should be at char 16, not byte 18"
        );
        assert_eq!(money[1].end, 20, "Second entity end should be char 20");
    }

    #[test]
    fn money_written() {
        let cases = [
            "50 dollars",
            "100 USD",
            "500 euros",
            "1000 EUR",
            "200 pounds",
        ];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Money), "Failed: {}", case);
        }
    }

    #[test]
    fn money_magnitude_written() {
        let cases = ["5 billion dollars", "1.5 million euros", "100 million"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Money), "Failed: {}", case);
        }
    }

    // ========================================================================
    // Percent Tests
    // ========================================================================

    #[test]
    fn percent_basic() {
        let cases = ["15%", "3.5%", "100%", "0.01%"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Percent), "Failed: {}", case);
        }
    }

    #[test]
    fn percent_written() {
        let cases = ["15 percent", "50 pct"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Percent), "Failed: {}", case);
        }
    }

    // ========================================================================
    // Email Tests
    // ========================================================================

    #[test]
    fn email_basic() {
        let cases = [
            "user@example.com",
            "john.doe@company.org",
            "support+ticket@help.co.uk",
            "test_123@sub.domain.io",
        ];
        for case in cases {
            let e = extract(case);
            assert!(
                e.iter().any(|e| e.entity_type == EntityType::Email),
                "Failed: {}",
                case
            );
        }
    }

    // ========================================================================
    // URL Tests
    // ========================================================================

    #[test]
    fn url_basic() {
        let cases = [
            "https://example.com",
            "http://www.google.com",
            "https://sub.domain.co.uk/path?query=1",
            "http://localhost:8080/api",
        ];
        for case in cases {
            let e = extract(case);
            assert!(
                e.iter().any(|e| e.entity_type == EntityType::Url),
                "Failed: {}",
                case
            );
        }
    }

    // ========================================================================
    // Phone Tests
    // ========================================================================

    #[test]
    fn phone_us_format() {
        let cases = [
            "(555) 123-4567",
            "555-123-4567",
            "555.123.4567",
            "1-555-123-4567",
            "+1 555 123 4567",
        ];
        for case in cases {
            let e = extract(case);
            assert!(
                e.iter().any(|e| e.entity_type == EntityType::Phone),
                "Failed: {}",
                case
            );
        }
    }

    #[test]
    fn phone_international() {
        let cases = ["+44 20 7946 0958", "+81 3 1234 5678"];
        for case in cases {
            let e = extract(case);
            assert!(
                e.iter().any(|e| e.entity_type == EntityType::Phone),
                "Failed: {}",
                case
            );
        }
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn mixed_entities() {
        let text = "Meeting on Jan 15 at 3:30 PM. Cost: $500. Contact: bob@acme.com or (555) 123-4567. Completion: 75%.";
        let e = extract(text);

        assert!(has_type(&e, &EntityType::Date), "Should have Date: {:?}", e);
        assert!(has_type(&e, &EntityType::Time), "Should have Time: {:?}", e);
        assert!(
            has_type(&e, &EntityType::Money),
            "Should have Money: {:?}",
            e
        );
        assert!(
            has_type(&e, &EntityType::Percent),
            "Should have Percent: {:?}",
            e
        );
        assert!(
            e.iter().any(|e| e.entity_type == EntityType::Email),
            "Should have Email: {:?}",
            e
        );
        assert!(
            e.iter().any(|e| e.entity_type == EntityType::Phone),
            "Should have Phone: {:?}",
            e
        );
    }

    #[test]
    fn no_person_org_loc() {
        let e = extract("John Smith works at Google in New York.");
        // Should NOT extract Person/Org/Location
        assert!(!has_type(&e, &EntityType::Person));
        assert!(!has_type(&e, &EntityType::Organization));
        assert!(!has_type(&e, &EntityType::Location));
    }

    #[test]
    fn entities_sorted_by_position() {
        let e = extract("$100 on 2024-01-01 at 50%");
        let positions: Vec<usize> = e.iter().map(|e| e.start).collect();
        let mut sorted = positions.clone();
        sorted.sort();
        assert_eq!(positions, sorted);
    }

    #[test]
    fn no_overlapping_entities() {
        let e = extract("The price is $1,000,000 (1 million dollars).");
        for i in 0..e.len() {
            for j in (i + 1)..e.len() {
                let overlap = e[i].start < e[j].end && e[j].start < e[i].end;
                assert!(!overlap, "Overlap: {:?} and {:?}", e[i], e[j]);
            }
        }
    }

    #[test]
    fn empty_text() {
        let e = extract("");
        assert!(e.is_empty());
    }

    #[test]
    fn no_entities_text() {
        let e = extract("The quick brown fox jumps over the lazy dog.");
        assert!(e.is_empty());
    }

    #[test]
    fn entity_spans_correct() {
        let text = "Cost: $100";
        let e = extract(text);
        let money = find_text(&e, "$100").unwrap();
        assert_eq!(&text[money.start..money.end], "$100");
    }

    #[test]
    fn provenance_attached() {
        use anno_core::ExtractionMethod;

        let text = "Contact: test@email.com on 2024-01-15";
        let e = extract(text);

        // All entities should have provenance
        for entity in &e {
            assert!(
                entity.provenance.is_some(),
                "Missing provenance for {:?}",
                entity
            );
            let prov = entity.provenance.as_ref().unwrap();

            // Source should be "pattern"
            assert_eq!(prov.source.as_ref(), "pattern");
            assert_eq!(prov.method, ExtractionMethod::Pattern);

            // Pattern name should be set
            assert!(
                prov.pattern.is_some(),
                "Missing pattern name for {:?}",
                entity
            );
        }

        // Check specific pattern names
        let email = find_text(&e, "test@email.com").unwrap();
        assert_eq!(
            email
                .provenance
                .as_ref()
                .unwrap()
                .pattern
                .as_ref()
                .unwrap()
                .as_ref(),
            "EMAIL"
        );

        let date = find_text(&e, "2024-01-15").unwrap();
        assert_eq!(
            date.provenance
                .as_ref()
                .unwrap()
                .pattern
                .as_ref()
                .unwrap()
                .as_ref(),
            "DATE_ISO"
        );
    }

    // ========================================================================
    // Multilingual Date Tests
    // ========================================================================

    #[test]
    fn japanese_date_format() {
        let cases = ["2024年1月15日", "2024年12月31日", "2000年01月01日"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
            assert_eq!(e[0].text, case);
        }
    }

    #[test]
    fn korean_date_format() {
        let cases = ["2024년 1월 15일", "2024년 12월 31일"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
        }
    }

    #[test]
    fn german_month_names() {
        let cases = [
            ("15. Januar 2024", "15. Januar 2024"),
            ("3 März 2023", "3 März 2023"),
            ("25 Dezember", "25 Dezember"),
        ];
        for (text, expected) in cases {
            let e = extract(text);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", text);
            assert!(
                find_text(&e, expected).is_some(),
                "Expected '{}' in: {}",
                expected,
                text
            );
        }
    }

    #[test]
    fn french_month_names() {
        let cases = ["15 janvier 2024", "1er février 2023", "25 décembre"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
        }
    }

    #[test]
    fn spanish_month_names() {
        let cases = ["15 de enero de 2024", "5 marzo 2023", "25 diciembre"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
        }
    }

    #[test]
    fn italian_month_names() {
        let e = extract("15 gennaio 2024");
        assert!(has_type(&e, &EntityType::Date));
    }

    #[test]
    fn portuguese_month_names() {
        let e = extract("15 de janeiro de 2024");
        assert!(has_type(&e, &EntityType::Date));
    }

    #[test]
    fn dutch_month_names() {
        let e = extract("15 januari 2024");
        assert!(has_type(&e, &EntityType::Date));
    }

    #[test]
    fn russian_month_names() {
        let e = extract("15 января 2024");
        assert!(has_type(&e, &EntityType::Date));
    }

    #[test]
    fn multilingual_dates_with_context() {
        // Test that multilingual dates work in context with other text
        let text = "Meeting on 2024年1月15日 at the office. Follow-up on 15 janvier.";
        let e = extract(text);
        let dates: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .collect();
        assert_eq!(dates.len(), 2, "Expected 2 dates, got {:?}", dates);
    }
}

// =============================================================================
// BatchCapable and StreamingCapable Trait Implementations
// =============================================================================

impl crate::BatchCapable for RegexNER {
    fn extract_entities_batch(
        &self,
        texts: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<Vec<Entity>>> {
        texts
            .iter()
            .map(|text| self.extract_entities(text, language))
            .collect()
    }

    fn optimal_batch_size(&self) -> Option<usize> {
        Some(64) // Regex matching is fast, can handle larger batches
    }
}

impl crate::StreamingCapable for RegexNER {
    fn recommended_chunk_size(&self) -> usize {
        10_000 // Regex matching handles larger chunks efficiently
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn extraction_never_panics(text in ".*") {
            let ner = RegexNER::new();
            let _ = ner.extract_entities(&text, None);
        }

        #[test]
        fn entities_within_text_bounds(text in ".{1,200}") {
            let ner = RegexNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                for e in entities {
                    prop_assert!(e.start <= text.len());
                    prop_assert!(e.end <= text.len());
                    prop_assert!(e.start <= e.end);
                }
            }
        }

        #[test]
        fn dollar_amounts_detected(amount in 1u32..10000) {
            let text = format!("Cost: ${}", amount);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();
            prop_assert!(entities.iter().any(|e| e.entity_type == EntityType::Money));
        }

        #[test]
        fn percentages_detected(pct in 1u32..100) {
            let text = format!("{}% complete", pct);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();
            prop_assert!(entities.iter().any(|e| e.entity_type == EntityType::Percent));
        }

        #[test]
        fn emails_detected(user in "[a-z]{3,10}", domain in "[a-z]{3,8}") {
            let text = format!("Contact: {}@{}.com", user, domain);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();
            prop_assert!(entities.iter().any(|e|
                e.entity_type == EntityType::Email
            ));
        }

        #[test]
        fn urls_detected(path in "[a-z]{1,10}") {
            let text = format!("Visit https://example.com/{}", path);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();
            prop_assert!(entities.iter().any(|e|
                e.entity_type == EntityType::Url
            ));
        }

        #[test]
        fn iso_dates_detected(y in 2000u32..2030, m in 1u32..13, d in 1u32..29) {
            let text = format!("Date: {:04}-{:02}-{:02}", y, m, d);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();
            prop_assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
        }

        #[test]
        fn no_overlapping_entities(text in ".{0,100}") {
            let ner = RegexNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                for i in 0..entities.len() {
                    for j in (i + 1)..entities.len() {
                        let e1 = &entities[i];
                        let e2 = &entities[j];
                        let overlap = e1.start < e2.end && e2.start < e1.end;
                        prop_assert!(!overlap, "Overlap: {:?} and {:?}", e1, e2);
                    }
                }
            }
        }
    }
}
