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

mod table;

use crate::{Entity, EntityType, Language, Model, Result};
use regex::Regex;
use std::sync::LazyLock;

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
/// Output spans are flat. A numeric date component in an HTTP(S) path can
/// be included in the complete URL span.
/// Dates in the surrounding text have their own spans.
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
static DATE_ISO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").expect("valid regex"));

// A serial field can contain a perfectly valid ISO substring. Limit its scope to
// the field's value token, so a sibling date field or a date-bearing filename
// elsewhere still reaches the ISO matcher.
static IDENTIFIER_VALUES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?ix)\b(?:ids?|identifiers?|numbers?|skus?|serials?|",
        r"(?:serial|lot|batch|accession|catalog(?:ue)?|inventory|part|booking|badge)",
        r"[\t\p{Zs}]+(?:no\.?|codes?|keys?))",
        r#"(?:["'”’]?[\t\p{Zs}]*[:=][\t\p{Zs}]*|[\t\p{Zs}]+(?:is|was)[\t\p{Zs}]+)"#,
        r#"["'“‘«]?[\p{L}\p{N}_.‑-]+"#,
    ))
    .expect("valid regex")
});

// Date-shaped inventory numbers use the same typography. Require a field/value
// boundary or a calendar phrase. Across a line break, require a standalone
// heading so the last column label cannot bind to the first value of a TSV row.
static DATE_ISO_PRINTED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?imx)(?:^\p{Zs}*(?:\p{L}+\p{Zs})?(?:date|deadline|expiry|expiration)",
        r"\p{Zs}*[:=]?\p{Zs}*\r?\n\p{Zs}*",
        r"|\b(?:date|deadline|expiry|expiration)(?:",
        r"[\t\p{Zs}]*[:=,][\t\p{Zs}]*|\t+|[\t\p{Zs}]+\(",
        r"|[\t\p{Zs}]+(?:is|was)[\t\p{Zs}]+)",
        r"|\b(?:as[\t\p{Zs}]+of|dated|due[\t\p{Zs}]+on|effective[\t\p{Zs}]+(?:on|from)|",
        r"scheduled[\t\p{Zs}]+(?:on|for)|",
        r"(?:premier(?:e|ed)|held|issued|published|released|expires?|expired|",
        r"began|ended|started|finished|occurred|opened|closed|arrived|departed|",
        r"launched|concluded|took[\t\p{Zs}]+place|curtain[\t\p{Zs}]+rose)",
        r"[\t\p{Zs}]+on)[\t\p{Zs}]+",
        r#")[“‘«"']?([0-9]{4}[-‑][0-9]{2}[-‑][0-9]{2})\b"#,
    ))
    .expect("valid regex")
});

// These predicates identify the period during which something holds. A pair
// of date-shaped endpoints joined by "from … to" alone can be a range of IDs.
static PRINTED_VALIDITY_PERIOD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?ix)\b(?:valid|effective|open|closed|available|unavailable|",
        r"(?:in|out[\t\p{Zs}]+of)[\t\p{Zs}]+service|in[\t\p{Zs}]+(?:effect|force)|",
        r"lasted|took[\t\p{Zs}]+place|",
        r"(?:exhibition|festival|session|campaign|season|conference|tournament|workshop|",
        r"program(?:me)?|trial|restoration|construction|rehearsals?)[\t\p{Zs}]+(?:runs|ran))",
        r"[\t\p{Zs}]+from[\t\p{Zs}]+([0-9]{4}[-‑][0-9]{2}[-‑][0-9]{2})",
        r"[\t\p{Zs}]+(?:to|through|until)[\t\p{Zs}]+([0-9]{4}[-‑][0-9]{2}[-‑][0-9]{2})\b",
    ))
    .expect("valid regex")
});

// Dotted filenames, hostnames and hyphenated inventory suffixes may begin with a date.
fn printed_date_ends_here(suffix: &str) -> bool {
    let mut chars = suffix.chars();
    match chars.next() {
        None => true,
        Some('.' | ':' | ',') => chars.next().is_none_or(|c| !c.is_alphanumeric()),
        Some(c) => c.is_whitespace() || "!?;)]}’”»\"'".contains(c),
    }
}

/// Check Gregorian month lengths after DATE_ISO has matched the numeric shape.
fn is_calendar_date(iso: &str) -> bool {
    // DATE_ISO also recognizes Unicode decimal digits. Keep those lexical matches.
    if !iso.is_ascii() {
        return true;
    }
    let mut fields = iso.split('-').map(str::parse::<u16>);
    let (Some(Ok(year)), Some(Ok(month)), Some(Ok(day))) =
        (fields.next(), fields.next(), fields.next())
    else {
        return false;
    };
    // ISO year 0000 is the proleptic Gregorian year corresponding to 1 BCE.
    let leap_year =
        year.is_multiple_of(400) || (year.is_multiple_of(4) && !year.is_multiple_of(100));
    let last_day = match month {
        2 if leap_year => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => return false,
    };
    (1..=last_day).contains(&day)
}

static DATE_US: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b").expect("valid regex"));

static DATE_EU: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b").expect("valid regex"));

// Unpunctuated calendar dates used in astronomical catalogs (2001 Jan 09).
// A single inline space keeps a preceding column or line from supplying the year.
static DATE_YEAR_MONTH_DAY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b[0-9]{4} (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) (?:0?[1-9]|[12][0-9]|3[01])\b").expect("valid regex")
});

static DATE_WRITTEN_FULL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b").expect("valid regex")
});

static DATE_WRITTEN_SHORT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b").expect("valid regex")
});

static DATE_WRITTEN_EU: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?(?:\s+\d{4})?\b").expect("valid regex")
});

// Month-year only (English): "April 2018", "December 2024"
// Must come AFTER full date patterns so "April 15, 2018" matches the longer pattern first.
static DATE_MONTH_YEAR_EN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b").expect("valid regex")
});

// Month-year only (German): "Oktober 2024", "Januar 2023"
static DATE_MONTH_YEAR_DE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}\b").expect("valid regex")
});

// =============================================================================
// Japanese Date Format: YYYY年MM月DD日
// =============================================================================

static DATE_JAPANESE: LazyLock<Regex> = LazyLock::new(|| {
    // Matches: 2024年1月15日, 2024年01月15日, etc.
    Regex::new(r"\d{4}年\d{1,2}月\d{1,2}日").expect("valid regex")
});

// =============================================================================
// Multilingual Month Names
// =============================================================================

// German months: Januar, Februar, März, April, Mai, Juni, Juli, August, September, Oktober, November, Dezember
static DATE_GERMAN_FULL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{1,2}(?:\.)?(?:,?\s*\d{4})?\b").expect("valid regex")
});

static DATE_GERMAN_EU: LazyLock<Regex> = LazyLock::new(|| {
    // "15. Januar 2024" or "15 Januar"
    Regex::new(r"(?i)\b\d{1,2}\.?\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)(?:\s+\d{4})?\b").expect("valid regex")
});

// French months: janvier, février, mars, avril, mai, juin, juillet, août, septembre, octobre, novembre, décembre
static DATE_FRENCH_FULL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+\d{1,2}(?:,?\s*\d{4})?\b").expect("valid regex")
});

static DATE_FRENCH_EU: LazyLock<Regex> = LazyLock::new(|| {
    // "15 janvier 2024" or "1er janvier"
    Regex::new(r"(?i)\b\d{1,2}(?:er)?\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)(?:\s+\d{4})?\b").expect("valid regex")
});

// Spanish months: enero, febrero, marzo, abril, mayo, junio, julio, agosto, septiembre, octubre, noviembre, diciembre
static DATE_SPANISH_EU: LazyLock<Regex> = LazyLock::new(|| {
    // "15 de enero de 2024" or "15 enero 2024"
    Regex::new(r"(?i)\b\d{1,2}\s+(?:de\s+)?(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+(?:de\s+)?\d{4})?\b").expect("valid regex")
});

// Italian months: gennaio, febbraio, marzo, aprile, maggio, giugno, luglio, agosto, settembre, ottobre, novembre, dicembre
static DATE_ITALIAN_EU: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)(?:\s+\d{4})?\b").expect("valid regex")
});

// Portuguese months: janeiro, fevereiro, março, abril, maio, junho, julho, agosto, setembro, outubro, novembro, dezembro
static DATE_PORTUGUESE_EU: LazyLock<Regex> = LazyLock::new(|| {
    // "15 de janeiro de 2024"
    Regex::new(r"(?i)\b\d{1,2}\s+(?:de\s+)?(?:janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)(?:\s+(?:de\s+)?\d{4})?\b").expect("valid regex")
});

// Dutch months: januari, februari, maart, april, mei, juni, juli, augustus, september, oktober, november, december
static DATE_DUTCH_EU: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b\d{1,2}\s+(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)(?:\s+\d{4})?\b").expect("valid regex")
});

// Russian months (Cyrillic): январь, февраль, март, апрель, май, июнь, июль, август, сентябрь, октябрь, ноябрь, декабрь
static DATE_RUSSIAN_EU: LazyLock<Regex> = LazyLock::new(|| {
    // "15 января 2024" - uses genitive case forms
    Regex::new(r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?\b").expect("valid regex")
});

// Chinese date format: YYYY年MM月DD日 (same as Japanese but also common)
// Already covered by DATE_JAPANESE

// Korean date format: YYYY년 MM월 DD일
static DATE_KOREAN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일").expect("valid regex"));

static TIME_12H: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b").expect("valid regex")
});

static TIME_24H: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?|24:\d+(?::\d+)*)\b")
        .expect("valid regex")
});

static TIME_SIMPLE: LazyLock<Regex> = LazyLock::new(|| {
    // Note: No trailing \b because a.m./p.m. end with .
    Regex::new(r"(?i)\b\d{1,2}\s*(?:am\b|pm\b|a\.m\.|p\.m\.)").expect("valid regex")
});

static MONEY_SYMBOL: LazyLock<Regex> = LazyLock::new(|| {
    // Print prices may use one nonbreaking space between the symbol and digits.
    // Commas belong to the amount only between digits.
    // Spelled-out magnitudes match in any case; the single-letter forms stay
    // uppercase-only. The trailing `\b` keeps "$20 Kits" from becoming "$20 K".
    Regex::new(
        r"[$€£¥][\x{00A0}\x{202F}]?\d+(?:,\d+)*(?:\.\d{1,2})?(?:\s*(?:(?i:billion|million|thousand|bn|mn)|B|M|K)\b)?",
    )
    .expect("valid regex")
});

static MONEY_WRITTEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b\d+(?:,\d{3})*(?:[.,]\d{1,2})?\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP|yen|JPY)\b",
    )
    .expect("valid regex")
});

static MONEY_CODE_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:USD|EUR|GBP|JPY|CHF|CAD|AUD)\s*\d+(?:[,\.]\d+)*(?:\s*(?:billion|million|trillion|thousand|Mrd|Mio|Bn|Mn|B|M|K|bn|mn))?\b",
    )
    .expect("valid regex")
});

static MONEY_MAGNITUDE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b\d+(?:[.,]\d+)?\s*(?:billion|million|trillion|Mrd|Mio|Bn|Mn)(?:\s+(?:dollars?|euros?|pounds?|USD|EUR|GBP|JPY|CHF|CAD|AUD))?\b",
    )
    .expect("valid regex")
});

static PERCENT: LazyLock<Regex> = LazyLock::new(|| {
    // Note: No trailing \b because % is not a word character.
    // Supports both period and comma as decimal separator (European format).
    Regex::new(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent\b|pct\b)").expect("valid regex")
});

static EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b").expect("valid regex")
});

static URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\bhttps?://[^\s<>"\[\]{}|\\^`\x00-\x1f]+"#).expect("valid regex")
});

// An apostrophe within a word is part of the path, even in quoted prose.
fn word_apostrophe(text: &str, byte: usize) -> bool {
    text[..byte]
        .chars()
        .next_back()
        .is_some_and(char::is_alphabetic)
        && text[byte + 1..]
            .chars()
            .next()
            .is_some_and(char::is_alphabetic)
}

// Mask matching outer parentheses or a closing single quote before scanning.
// A bare URL can contain any of these characters. Equal-width spaces keep offsets.
fn url_text(text: &str) -> std::borrow::Cow<'_, str> {
    let mut result = std::borrow::Cow::Borrowed(text);
    if !text.contains(['(', '（', '\'']) {
        return result;
    }
    let mut wrapper = None;
    let mut depth = 0;
    for (byte, ch) in text.char_indices() {
        if let Some((open, close)) = wrapper {
            if ch == close {
                if ch == '\'' && word_apostrophe(text, byte) {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    let space = if close == '）' { "　" } else { " " };
                    result
                        .to_mut()
                        .replace_range(byte..byte + ch.len_utf8(), space);
                    wrapper = None;
                }
            } else if ch == open {
                depth += 1;
            } else if ch.is_whitespace() {
                wrapper = None;
            }
        } else if matches!(ch, '(' | '（' | '\'') {
            let rest = &text.as_bytes()[byte + ch.len_utf8()..];
            if [b"http://".as_slice(), b"https://"].iter().any(|scheme| {
                rest.get(..scheme.len())
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(scheme))
            }) {
                wrapper = Some((
                    ch,
                    match ch {
                        '(' => ')',
                        '（' => '）',
                        _ => ch,
                    },
                ));
                depth = 1;
            }
        }
    }
    result
}

// Determine each URL boundary once, including for links with many date components.
fn date_link_range(text: &str, url: regex::Match<'_>) -> Option<std::ops::Range<usize>> {
    let wrapped = matches!(
        (
            text[..url.start()].chars().next_back(),
            text[url.end()..].chars().next()
        ),
        (Some('('), Some(')'))
            | (Some('（'), Some('）'))
            | (Some('<'), Some('>'))
            | (Some('"'), Some('"'))
            | (Some('\''), Some('\''))
    );
    let value = if wrapped {
        url.as_str()
    } else {
        let value = url.as_str().trim_end_matches(['.', ',']);
        // Unwrapped commas, quotes and other punctuation may introduce another
        // field or sentence. Keep the original predictions in ambiguous text.
        if !value.char_indices().all(|(byte, ch)| {
            ch.is_alphanumeric()
                || "-._~:/?#@%&=+".contains(ch)
                || (ch == '\'' && word_apostrophe(value, byte))
        }) {
            return None;
        }
        value
    };
    let end = url.start() + value.len();
    Some(url.start()..end)
}

fn link_containing_date(
    text: &str,
    date: regex::Match<'_>,
    links: &[std::ops::Range<usize>],
) -> Option<std::ops::Range<usize>> {
    let next = links.partition_point(|link| link.end <= date.start());
    let link = links.get(next)?;
    let start = date.start();
    (link.start < start
        && date.end() <= link.end
        && text[link.start..start].ends_with(['/', '-', '=', '#']))
    .then(|| link.clone())
}

// A percent escape is exactly '%' followed by two ASCII hex digits in the URL.
fn is_url_escape(m: regex::Match<'_>, urls: &[regex::Match<'_>]) -> bool {
    if !m.as_str().ends_with('%') {
        return false;
    }
    let next = urls.partition_point(|url| url.end() <= m.start());
    let Some(url) = urls.get(next).filter(|url| url.start() <= m.start()) else {
        return false;
    };
    let after_percent = m.end() - url.start();
    url.as_str()
        .as_bytes()
        .get(after_percent..after_percent + 2)
        .is_some_and(|pair| pair.iter().all(u8::is_ascii_hexdigit))
}

static PHONE_US: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:\+1[-.\s]?\(?|\b1[-.\s]?\(?|\(|\b)\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        .expect("valid regex")
});

static PHONE_INTL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\+[1-9]\d{0,2}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b").expect("valid regex")
});

static SHORT_PLUS_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\+\d{4}$").expect("valid regex"));

static PHONE_LOCAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{3}[-.\s]?\d{4}\b").expect("valid regex"));

/// Words that signal the following number is a phone number (for PHONE_LOCAL context filter).
static PHONE_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:call|tel|telephone|phone|fax|dial|ring|mobile|cell|contact)\b[:\s]*$")
        .expect("valid regex")
});

static MENTION: LazyLock<Regex> = LazyLock::new(|| {
    // @username - supports letters, numbers, underscore, dot (but not starting/ending with dot)
    Regex::new(r"\B@[\w](?:[\w.]*[\w])?").expect("valid regex")
});

static HASHTAG: LazyLock<Regex> = LazyLock::new(|| {
    // #hashtag - supports letters, numbers, underscore
    Regex::new(r"\B#\w+").expect("valid regex")
});

impl Model for RegexNER {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        use crate::offset::SpanConverter;
        use crate::Provenance;
        let mut entities = Vec::new();
        let url_input = url_text(text);
        let urls: Vec<_> = URL.find_iter(&url_input).collect();
        let date_links: Vec<_> = urls
            .iter()
            .filter_map(|&url| date_link_range(text, url))
            .collect();

        // Performance optimization: Build SpanConverter once for all byte-to-char conversions
        // ROI: High - called once per extract_entities, saves O(n) per regex match
        let converter = SpanConverter::new(text);

        // Helper to add entity if no overlap
        // Note: regex returns byte offsets, but we convert to char offsets
        // for consistency with evaluation (GoldEntity uses char offsets).
        let mut add_entity =
            |m: regex::Match, entity_type: EntityType, confidence: f64, pattern: &'static str| {
                let link = match pattern {
                    "DATE_ISO" | "DATE_US" => link_containing_date(text, m, &date_links),
                    _ => None,
                };
                let (range, entity_type, confidence, pattern) = match link {
                    Some(range) => (range, EntityType::Url, 0.98, "URL"),
                    None => (m.range(), entity_type, confidence, pattern),
                };
                let char_start = converter.byte_to_char(range.start);
                let char_end = converter.byte_to_char(range.end);
                if !overlaps(&entities, char_start, char_end) {
                    entities.push(Entity::with_provenance(
                        &text[range],
                        entity_type,
                        char_start,
                        char_end,
                        confidence,
                        Provenance::pattern(pattern),
                    ));
                }
            };

        // Dates (high confidence - very specific patterns)
        let date_text = table::with_cell_boundaries(text);
        let mut identifiers = IDENTIFIER_VALUES.find_iter(&date_text).peekable();
        for m in DATE_ISO.find_iter(&date_text) {
            while identifiers.next_if(|id| id.end() <= m.start()).is_some() {}
            if !identifiers.peek().is_some_and(|id| id.start() <= m.start())
                && is_calendar_date(m.as_str())
            {
                add_entity(m, EntityType::Date, 0.95, "DATE_ISO");
            }
        }
        // English dates
        let date_patterns_en: &[(&LazyLock<Regex>, &'static str)] = &[
            (&DATE_US, "DATE_US"),
            (&DATE_EU, "DATE_EU"),
            (&DATE_YEAR_MONTH_DAY, "DATE_YEAR_MONTH_DAY"),
            (&DATE_WRITTEN_FULL, "DATE_WRITTEN_FULL"),
            (&DATE_WRITTEN_SHORT, "DATE_WRITTEN_SHORT"),
            (&DATE_WRITTEN_EU, "DATE_WRITTEN_EU"),
        ];
        for (pattern, name) in date_patterns_en {
            for m in pattern.find_iter(&date_text) {
                add_entity(m, EntityType::Date, 0.95, name);
            }
        }

        // Multilingual dates (Japanese, Korean, German, French, Spanish, etc.)
        let date_patterns_i18n: &[(&LazyLock<Regex>, &'static str)] = &[
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
            for m in pattern.find_iter(&date_text) {
                add_entity(m, EntityType::Date, 0.93, name); // Slightly lower confidence for i18n
            }
        }

        // Month-year only patterns (English and German).
        // These come AFTER both English full dates AND i18n full dates so longer
        // patterns like "15. Januar 2024" or "April 15, 2018" are preferred.
        let date_month_year: &[(&LazyLock<Regex>, &'static str)] = &[
            (&DATE_MONTH_YEAR_EN, "DATE_MONTH_YEAR_EN"),
            (&DATE_MONTH_YEAR_DE, "DATE_MONTH_YEAR_DE"),
        ];
        for (pattern, name) in date_month_year {
            for m in pattern.find_iter(&date_text) {
                add_entity(m, EntityType::Date, 0.90, name);
            }
        }

        // Times
        let time_patterns: &[(&LazyLock<Regex>, &'static str)] = &[
            (&TIME_12H, "TIME_12H"),
            (&TIME_24H, "TIME_24H"),
            (&TIME_SIMPLE, "TIME_SIMPLE"),
        ];
        for (pattern, name) in time_patterns {
            for m in pattern.find_iter(text) {
                // Consume the whole 24:... token so 24:00:01 cannot yield
                // either a partial midnight or a spurious 00:01.
                if *name == "TIME_24H"
                    && m.as_str().starts_with("24:")
                    && !matches!(m.as_str(), "24:00" | "24:00:00")
                {
                    continue;
                }
                add_entity(m, EntityType::Time, 0.90, name);
            }
        }

        // Money (high confidence)
        let money_patterns: &[(&LazyLock<Regex>, &'static str)] = &[
            (&MONEY_SYMBOL, "MONEY_SYMBOL"),
            (&MONEY_CODE_PREFIX, "MONEY_CODE_PREFIX"),
            (&MONEY_WRITTEN, "MONEY_WRITTEN"),
            (&MONEY_MAGNITUDE, "MONEY_MAGNITUDE"),
        ];
        for (pattern, name) in money_patterns {
            for m in pattern.find_iter(text) {
                add_entity(m, EntityType::Money, 0.95, name);
            }
        }

        // Preserve numeric substrings before a percent-encoded octet in the address.
        for m in PERCENT.find_iter(text) {
            if is_url_escape(m, &urls) {
                continue;
            }
            add_entity(m, EntityType::Percent, 0.95, "PERCENT");
        }

        // Emails (very high confidence - very specific pattern)
        for m in EMAIL.find_iter(text) {
            add_entity(m, EntityType::Email, 0.98, "EMAIL");
        }

        for m in urls {
            add_entity(m, EntityType::Url, 0.98, "URL");
        }

        // Phone numbers (medium confidence - can have false positives)
        for m in PHONE_US.find_iter(text) {
            add_entity(m, EntityType::Phone, 0.85, "PHONE_US");
        }
        for m in PHONE_INTL.find_iter(text) {
            // Short compact numbers can also be UTC offsets (+1100). Unicode
            // code points use the explicitly marked U+hhhh form.
            let prefix = &text[..m.start()];
            let number = m.as_str();
            let is_codepoint = prefix.ends_with(['U', 'u'])
                && (5..=7).contains(&number.len())
                && number[1..].bytes().all(|byte| byte.is_ascii_digit());
            if is_codepoint
                || (SHORT_PLUS_NUMBER.is_match(number) && !PHONE_CONTEXT.is_match(prefix))
            {
                continue;
            }
            add_entity(m, EntityType::Phone, 0.85, "PHONE_INTL");
        }
        // Local 7-digit phone numbers -- only when preceded by phone-context words
        // to avoid matching page numbers, codes, zip codes, etc.
        for m in PHONE_LOCAL.find_iter(text) {
            let prefix = &text[..m.start()];
            if PHONE_CONTEXT.is_match(prefix) {
                add_entity(m, EntityType::Phone, 0.65, "PHONE_LOCAL");
            }
        }

        // The new printed-date rule runs after URLs so a query such as
        // ?date=2040‑02‑29 cannot displace the contact entity.
        if date_text.contains('‑') {
            for caps in PRINTED_VALIDITY_PERIOD.captures_iter(&date_text) {
                let start = caps.get(1).expect("start date capture");
                let end = caps.get(2).expect("end date capture");
                let first = start.as_str().replace('‑', "-");
                let last = end.as_str().replace('‑', "-");
                if printed_date_ends_here(&date_text[end.end()..])
                    && is_calendar_date(&first)
                    && is_calendar_date(&last)
                    && first <= last
                {
                    for endpoint in [start, end] {
                        if endpoint.as_str().contains('‑') {
                            add_entity(endpoint, EntityType::Date, 0.95, "DATE_ISO");
                        }
                    }
                }
            }
            for caps in DATE_ISO_PRINTED.captures_iter(&date_text) {
                let m = caps.get(1).expect("date capture");
                if m.as_str().contains('‑')
                    && printed_date_ends_here(&date_text[m.end()..])
                    && is_calendar_date(&m.as_str().replace('‑', "-"))
                {
                    add_entity(m, EntityType::Date, 0.95, "DATE_ISO");
                }
            }
        }

        // Social Media (@mentions and #hashtags) - note: mapping to Other for now as specific types don't exist yet
        for m in MENTION.find_iter(text) {
            // Using a custom "Mention" type via Other
            // In future refactor: Add EntityType::Mention
            let char_start = converter.byte_to_char(m.start());
            let char_end = converter.byte_to_char(m.end());
            if !overlaps(&entities, char_start, char_end) {
                // We use EntityType::custom for now, but specific string "Mention"
                entities.push(Entity::with_provenance(
                    m.as_str(),
                    EntityType::custom("Mention", crate::EntityCategory::Misc),
                    char_start,
                    char_end,
                    0.95,
                    Provenance::pattern("MENTION"),
                ));
            }
        }

        for m in HASHTAG.find_iter(text) {
            // Skip matches followed by a hyphen: these are typically invoice/reference
            // numbers (e.g. "#2024-0042"), not hashtags.
            if text.as_bytes().get(m.end()) == Some(&b'-') {
                continue;
            }
            let char_start = converter.byte_to_char(m.start());
            let char_end = converter.byte_to_char(m.end());
            if !overlaps(&entities, char_start, char_end) {
                entities.push(Entity::with_provenance(
                    m.as_str(),
                    EntityType::custom("Hashtag", crate::EntityCategory::Misc),
                    char_start,
                    char_end,
                    0.95,
                    Provenance::pattern("HASHTAG"),
                ));
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort by position for consistent output
        entities.sort_unstable_by_key(|e| e.start());

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

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities::default()
    }
}

/// Check if a span overlaps with existing entities.
fn overlaps(entities: &[Entity], start: usize, end: usize) -> bool {
    entities
        .iter()
        .any(|e| !(end <= e.start() || start >= e.end()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labeled_entities(text: &str) -> Vec<(String, String)> {
        RegexNER::new()
            .extract_entities(text, None)
            .unwrap()
            .into_iter()
            .map(|e| (e.entity_type.as_label().to_string(), e.text))
            .collect()
    }

    #[test]
    fn symbol_prices_keep_print_spacing_and_exclude_list_punctuation() {
        let text = "🧾 €\u{a0}27, £\u{202f}4.50, $3,240.05, and ¥12. Signs: $, £, €\n47.";
        let money: Vec<_> = extract(text)
            .into_iter()
            .filter(|entity| entity.entity_type == EntityType::Money)
            .collect();
        let spans: Vec<_> = money.iter().map(|entity| entity.text.as_str()).collect();
        assert_eq!(spans, ["€\u{a0}27", "£\u{202f}4.50", "$3,240.05", "¥12"]);
        for entity in money {
            let slice: String = text
                .chars()
                .skip(entity.start())
                .take(entity.end() - entity.start())
                .collect();
            assert_eq!(slice, entity.text);
        }
    }

    /// A magnitude word matches in any case and only as a whole word, so
    /// "$5 Billion" is not cut to "$5 B" and "$20 Kits" is not "$20 K".
    #[test]
    fn money_magnitude_suffix_is_not_truncated() {
        let ents = labeled_entities("The deal is worth $5 Billion today.");
        assert!(
            ents.iter().any(|(t, s)| t == "MONEY" && s == "$5 Billion"),
            "{ents:?}"
        );
        let ents = labeled_entities("We sold $20 Kits yesterday.");
        assert!(!ents.iter().any(|(_, s)| s == "$20 K"), "{ents:?}");
        assert!(
            ents.iter().any(|(t, s)| t == "MONEY" && s == "$20"),
            "{ents:?}"
        );
    }

    /// PHONE_US needs a word boundary before a leading digit, so it does not
    /// match the last ten digits of a longer digit run.
    #[test]
    fn phone_not_matched_inside_long_digit_run() {
        let ents = labeled_entities("Order 12345678901234 shipped, ts=1700000000000.");
        assert!(!ents.iter().any(|(t, _)| t == "PHONE"), "{ents:?}");
    }

    fn ner() -> RegexNER {
        RegexNER::new()
    }

    fn extract(text: &str) -> Vec<Entity> {
        ner()
            .extract_entities(text, None)
            .expect("NER extraction should succeed")
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
    fn iso_identifier_fields_end_before_neighboring_dates() {
        let source = concat!(
            "📦 Serial: KT-2028-01-19-Q; inspection date: 2028-01-21. ",
            "The lot number is 2068-03-15. Booking: 2026-12-11. ",
            "The meeting date is embedded in ref MTG-2025-04-08-A. ",
            "The manifest reads {\"id\":\"2023-07-06\"}.",
        );
        let dates: Vec<_> = extract(source)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .map(|e| e.text)
            .collect();
        assert_eq!(dates, ["2028-01-21", "2026-12-11", "2025-04-08"]);
    }

    #[test]
    fn calendar_event_predicates_do_not_extend_to_identifier_ranges() {
        let source = concat!(
            "The conference ran from 2050‑01‑06 until 2050‑01‑09. ",
            "Conference badge numbers ran from 2040‑01‑08 to 2040‑02‑05. ",
            "The tour began on 2042‑03‑01. Tracking number: 2042-03-02. ",
            "The date code: 2030-08-13. The firmware relied on 2037‑08‑15.",
        );
        let dates: Vec<_> = extract(source)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .map(|e| e.text)
            .collect();
        assert_eq!(
            dates,
            ["2050‑01‑06", "2050‑01‑09", "2042‑03‑01", "2030-08-13"]
        );
    }

    #[test]
    fn printed_validity_periods_require_a_calendar_predicate_and_valid_endpoints() {
        let source = concat!(
            "Café 🚌 out of service from 2032‑02‑29 through 2032-03‑02. ",
            "The serials run from 2033‑03‑01 to 2034‑04‑01. ",
            "Valid from 2100‑02‑29 until 2100‑03‑01. ",
            "Open from 2042‑03‑07 to 2041‑02‑06. ",
            "Closed from 2042‑03‑07 to 2042‑03‑09.txt.",
        );
        let dates: Vec<_> = extract(source)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .map(|e| {
                assert_eq!(
                    source
                        .chars()
                        .skip(e.start())
                        .take(e.end() - e.start())
                        .collect::<String>(),
                    e.text
                );
                e.text
            })
            .collect();
        assert_eq!(dates, ["2032‑02‑29", "2032-03‑02"]);
    }

    #[test]
    fn printed_dates_keep_unicode_offsets_and_respect_field_context() {
        let text = concat!(
            "Café 🗓 Premiere on 2040‑02‑29. Lot number: 2040‑02‑29. ",
            "Expiry date: 2100‑02‑29. Invoice date: 2024-02‑29. ",
            "Inspection date: 2034‑10‑07-A. Publication date: 2040‑01‑31.pdf. ",
            "Hearing date: pending. Booking reference: 2028‑04‑21.",
        );
        let entities = extract(text);
        let dates: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .collect();
        let expected = ["2040‑02‑29", "2024-02‑29"];
        assert_eq!(dates.len(), expected.len());
        for (entity, value) in dates.into_iter().zip(expected) {
            let start = text[..text.find(value).unwrap()].chars().count();
            assert_eq!(entity.text, value);
            assert_eq!(
                (entity.start(), entity.end()),
                (start, start + value.chars().count())
            );
        }
    }

    #[test]
    fn printed_date_fields_keep_the_value_in_its_source_context() {
        let text = concat!(
            "🧊 Révision date is “2044‑02‑29”. Delivery scheduled for 2043‑12‑17.\n",
            "Expiry date:\r\n  2042‑09‑30\nRelease date, «2041‑10‑17».\n",
            "Visit date\n2039‑02‑28\n",
            "Date code is 2038‑05‑18. Lot number: 2031‑05‑04.\n",
            "Date: pending. Barcode is 2032‑08‑03. Date:\n\n2036‑02‑29\n",
            "Inventory code\tDate\n2036‑03‑18\tPending\n",
            "The date was 2037‑06‑30.csv. The date was 2045‑02‑29.",
        );
        let dates: Vec<_> = extract(text)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .map(|e| (e.start(), e.end(), e.text))
            .collect();
        let expected: Vec<_> = [
            "2044‑02‑29",
            "2043‑12‑17",
            "2042‑09‑30",
            "2041‑10‑17",
            "2039‑02‑28",
        ]
        .into_iter()
        .map(|value| {
            let start = text[..text.find(value).unwrap()].chars().count();
            (start, start + value.chars().count(), value.to_owned())
        })
        .collect();
        assert_eq!(dates, expected);
    }

    #[test]
    fn printed_date_does_not_displace_a_url_with_a_date_parameter() {
        let url = "https://example.org/reset?date=2052‑07‑03";
        let entities = extract(url);
        assert!(entities
            .iter()
            .any(|e| e.entity_type == EntityType::Url && e.text == url));
        assert!(!has_type(&entities, &EntityType::Date));
    }

    #[test]
    fn iso_dates_follow_the_gregorian_calendar() {
        let text = concat!(
            "0000-00-00 2024-01-00 2024-13-01 2024-04-31 1900-02-29 2023-02-29 ",
            "0000-02-29 2000-02-29 2024-02-29 2023-04-30 9999-12-31 ٢٠٢٤-٠٢-٢٩",
        );
        let dates: Vec<_> = extract(text)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .map(|e| e.text)
            .collect();
        assert_eq!(
            dates,
            [
                "0000-02-29",
                "2000-02-29",
                "2024-02-29",
                "2023-04-30",
                "9999-12-31",
                "٢٠٢٤-٠٢-٢٩"
            ]
        );
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
    fn dates_stay_inside_underlined_table_cells() {
        // RFC 3339 §5.7 has separate month-number, name and day-limit cells.
        // Add genuine dates in prose and a table cell, plus a non-ASCII prefix.
        let text = concat!(
            "Préface: May\n  2020.\n\n",
            "  Code  Designation       Data\n",
            "  ----  ----------------  ----------\n",
            "  01    January           31\n",
            "  02    February, leap    29\n",
            "  03    March 2020        31\n",
            "  éà    14 juin 2020      30\r\n",
            "\nAnnounced on April          12, 2021.\n",
        );
        let dates: Vec<_> = extract(text)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .collect();
        let gold = [
            "May\n  2020",
            "March 2020",
            "14 juin 2020",
            "April          12, 2021",
        ];
        assert_eq!(dates.len(), gold.len(), "{dates:?}");
        for expected in gold {
            let start = text[..text.find(expected).unwrap()].chars().count();
            let end = start + expected.chars().count();
            assert!(dates
                .iter()
                .any(|e| e.text == expected && e.start() == start && e.end() == end));
        }
    }

    #[test]
    fn decorative_rules_do_not_split_prose() {
        for text in [
            "May          12\n---  -----\nNo date here.",
            "Some  Text\n----  ----\nMay          12",
        ] {
            assert!(find_text(&extract(text), "May          12").is_some());
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
    fn year_first_date_uses_inline_year_only() {
        for date in ["2001 Jan 09", "2020 September 30", "2020 Jul 5"] {
            assert!(find_text(&extract(date), date).is_some(), "{date}");
        }
        for text in ["2001\nJan 09", "2001    Jan 09"] {
            let entities = extract(text);
            assert!(find_text(&entities, text).is_none());
            assert!(find_text(&entities, "Jan 09").is_some());
        }
    }

    #[test]
    fn end_of_day_requires_zero_minutes_and_seconds() {
        for valid in ["24:00", "24:00:00"] {
            assert!(find_text(&extract(valid), valid).is_some(), "{valid}");
        }
        for invalid in ["24:01", "24:00:01", "24:00:99"] {
            assert!(!has_type(&extract(invalid), &EntityType::Time), "{invalid}");
        }
    }

    #[test]
    fn utc_offsets_are_not_international_phones() {
        for offset in ["+0330", "+1100", "+1245", "U+2069", "U+10000", "U+100000"] {
            assert!(!has_type(&extract(offset), &EntityType::Phone), "{offset}");
        }
        assert!(has_type(&extract("tel: +1245"), &EntityType::Phone));
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
        let entities = ner
            .extract_entities(text, None)
            .expect("NER extraction should succeed");

        // "Price: " = 7 chars, so first € is at char 7
        // "€50 then " = 9 chars, so second € is at char 16
        let money: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .collect();

        assert_eq!(money.len(), 2, "Expected 2 money entities, got {:?}", money);

        // First entity: "€50" at char 7
        assert_eq!(
            money[0].start(),
            7,
            "First € should be at char 7, not byte 7"
        );
        assert_eq!(money[0].end(), 10, "First entity end should be char 10");

        // Second entity: "€100" at char 16
        assert_eq!(
            money[1].start(),
            16,
            "Second € should be at char 16, not byte 18"
        );
        assert_eq!(money[1].end(), 20, "Second entity end should be char 20");
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
    fn url_percent_escapes_do_not_hide_the_address_or_an_outside_percentage() {
        let url = "https://example.org/search?q=75%25%20cocoa";
        let text = format!("🦊 {url} (12.5% of recipes)");
        let entities = extract(&text);
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Url
            && e.text == url
            && e.start() == 2
            && e.end() == 2 + url.chars().count()));
        let percentages: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Percent)
            .map(|e| e.text.as_str())
            .collect();
        assert_eq!(percentages, ["12.5%"]);
    }

    #[test]
    fn incomplete_or_nonhex_escape_keeps_percentage_detection() {
        for suffix in ["", "F", "0G"] {
            let text = format!("https://example.org/path/17%{suffix}");
            assert!(extract(&text)
                .iter()
                .any(|e| e.entity_type == EntityType::Percent && e.text == "17%"));
        }
    }

    #[test]
    fn markdown_url_wrappers_preserve_literal_parentheses_and_source_ranges() {
        let text = "🦊 [doc](HTTPS://example.org/guide)[ref](http://example.org/A_(B)) bare https://example.org/A_(B)";
        let urls: Vec<_> = extract(text)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::Url)
            .map(|e| (e.start(), e.end(), e.text))
            .collect();
        let expected: Vec<_> = [
            "HTTPS://example.org/guide",
            "http://example.org/A_(B)",
            "https://example.org/A_(B)",
        ]
        .iter()
        .map(|&url| {
            let byte = text.find(url).unwrap();
            let start = text[..byte].chars().count();
            (start, start + url.chars().count(), url.to_string())
        })
        .collect();
        assert_eq!(urls, expected);
    }

    #[test]
    fn fullwidth_url_wrappers_preserve_internal_and_unwrapped_punctuation() {
        let text = "（HTTPS://example.org/作品（上））続き（https://example.jp/資料）終。 https://example.org/作品（上）";
        let urls: Vec<_> = extract(text)
            .into_iter()
            .filter(|e| e.entity_type == EntityType::Url)
            .map(|e| e.text)
            .collect();
        assert_eq!(
            urls,
            [
                "HTTPS://example.org/作品（上）",
                "https://example.jp/資料",
                "https://example.org/作品（上）"
            ]
        );
    }

    #[test]
    fn dated_links_respect_field_delimiters_and_neighboring_dates() {
        for quote in ['"', '\''] {
            let url = "https://example.org/2024-06-15/O'Brien/at/12:30";
            let text = format!("📚 {quote}{url}{quote},{quote}2027-04-21{quote}");
            let entities = extract(&text);
            assert!(entities
                .iter()
                .any(|e| e.entity_type == EntityType::Url && e.text == url));
            assert!(entities
                .iter()
                .any(|e| e.entity_type == EntityType::Date && e.text == "2027-04-21"));
            assert!(!has_type(&entities, &EntityType::Time));
        }
        for (text, date) in [
            ("https://example.org/2024-06-15,2024-07-02", "2024-07-02"),
            (
                "https://example.org/report。2024年6月15日。",
                "2024年6月15日",
            ),
            ("https://example.org/key.May 3, 2026", "May 3, 2026"),
        ] {
            assert!(extract(text)
                .iter()
                .any(|e| e.entity_type == EntityType::Date && e.text == date));
        }
    }

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
        let positions: Vec<usize> = e.iter().map(|e| e.start()).collect();
        let mut sorted = positions.clone();
        sorted.sort();
        assert_eq!(positions, sorted);
    }

    #[test]
    fn no_overlapping_entities() {
        let e = extract("The price is $1,000,000 (1 million dollars).");
        for i in 0..e.len() {
            for j in (i + 1)..e.len() {
                let overlap = e[i].start() < e[j].end() && e[j].start() < e[i].end();
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
        use crate::offset::TextSpan;

        let text = "Cost: $100";
        let e = extract(text);
        let money = find_text(&e, "$100").expect("money entity should be found");
        assert_eq!(
            TextSpan::from_chars(text, money.start(), money.end()).extract(text),
            "$100"
        );
    }

    #[test]
    fn date_shaped_controller_model_after_on_is_not_a_calendar_date() {
        let text =
            "The firmware runs on 2040‑02‑29, the controller model. Installed on 2040-03-01.";
        let entities = extract(text);
        let dates: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .map(|e| e.text.as_str())
            .collect();
        assert_eq!(dates, ["2040-03-01"]);
    }

    #[test]
    fn provenance_attached() {
        use crate::ExtractionMethod;

        let text = "Contact: test@email.com on 2024-01-15";
        let e = extract(text);

        // All entities should have provenance
        for entity in &e {
            assert!(
                entity.provenance.is_some(),
                "Missing provenance for {:?}",
                entity
            );
            let prov = entity
                .provenance
                .as_ref()
                .expect("provenance should be set");

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
        let email = find_text(&e, "test@email.com").expect("email entity should be found");
        assert_eq!(
            email
                .provenance
                .as_ref()
                .expect("provenance should be set")
                .pattern
                .as_ref()
                .expect("pattern should be set")
                .as_ref(),
            "EMAIL"
        );

        let date = find_text(&e, "2024-01-15").expect("date entity should be found");
        assert_eq!(
            date.provenance
                .as_ref()
                .expect("provenance should be set")
                .pattern
                .as_ref()
                .expect("pattern should be set")
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

    // ========================================================================
    // Month-year only patterns
    // ========================================================================

    #[test]
    fn month_year_only_english() {
        let cases = ["April 2018", "December 2024", "May 2022"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
        }
    }

    #[test]
    fn month_year_only_german() {
        let cases = ["Oktober 2024", "Januar 2023", "März 2025"];
        for case in cases {
            let e = extract(case);
            assert!(has_type(&e, &EntityType::Date), "Failed: {}", case);
        }
    }

    #[test]
    fn full_date_preferred_over_month_year() {
        // "April 15, 2018" should match as a full date, not just "April 2018"
        let text = "The event on April 15, 2018 was great.";
        let e = extract(text);
        let dates: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Date)
            .collect();
        assert_eq!(dates.len(), 1, "Expected 1 date, got {:?}", dates);
        assert!(
            dates[0].text.contains("15"),
            "Should match full date with day, got: {}",
            dates[0].text
        );
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

    // ========================================================================
    // Fix 5: Money magnitude -- no trailing whitespace
    // ========================================================================

    #[test]
    fn money_magnitude_no_trailing_whitespace() {
        let cases = [
            "5 billion in revenue",
            "1.5 trillion was allocated",
            "100 million for research",
        ];
        for case in cases {
            let e = extract(case);
            for entity in &e {
                assert_eq!(
                    entity.text,
                    entity.text.trim(),
                    "Money entity '{}' should have no trailing whitespace in: '{}'",
                    entity.text,
                    case
                );
            }
        }
    }

    #[test]
    fn money_magnitude_with_currency_still_works() {
        let cases = [
            ("5 billion dollars", "5 billion dollars"),
            ("1.5 million euros", "1.5 million euros"),
            ("100 trillion pounds", "100 trillion pounds"),
        ];
        for (text, expected) in cases {
            let e = extract(text);
            assert!(
                find_text(&e, expected).is_some(),
                "Should match '{}' in '{}', got: {:?}",
                expected,
                text,
                e
            );
        }
    }

    // ========================================================================
    // Fix 6: Currency code prefix patterns
    // ========================================================================

    #[test]
    fn money_code_prefix_basic() {
        let cases = [
            ("EUR 500", "EUR 500"),
            ("GBP 100", "GBP 100"),
            ("USD 1,000", "USD 1,000"),
            ("JPY 50000", "JPY 50000"),
            ("CHF 200", "CHF 200"),
            ("CAD 750", "CAD 750"),
            ("AUD 300", "AUD 300"),
        ];
        for (text, expected) in cases {
            let e = extract(text);
            assert!(
                find_text(&e, expected).is_some(),
                "Should detect '{}' as money, got: {:?}",
                expected,
                e
            );
            let money = find_text(&e, expected).unwrap();
            assert_eq!(
                money.entity_type,
                EntityType::Money,
                "'{}' should be MONEY type",
                expected
            );
        }
    }

    #[test]
    fn money_code_prefix_with_magnitude() {
        let cases = ["EUR 1.2 million", "GBP 500 billion", "USD 3.5M", "JPY 100K"];
        for case in cases {
            let e = extract(case);
            assert!(
                has_type(&e, &EntityType::Money),
                "Should detect money in '{}', got: {:?}",
                case,
                e
            );
        }
    }

    #[test]
    fn money_code_prefix_case_insensitive() {
        let cases = ["eur 500", "Eur 1000", "gbp 250"];
        for case in cases {
            let e = extract(case);
            assert!(
                has_type(&e, &EntityType::Money),
                "Case-insensitive currency code '{}' should match, got: {:?}",
                case,
                e
            );
        }
    }

    #[test]
    fn money_code_prefix_in_context() {
        let text = "The budget allocated EUR 1.2 million for research and GBP 500 for travel.";
        let e = extract(text);
        let money: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::Money)
            .collect();
        assert!(
            money.len() >= 2,
            "Should detect at least 2 money entities, got: {:?}",
            money
        );
    }

    #[test]
    fn money_code_prefix_offsets_correct() {
        let text = "Price: EUR 500 then USD 1000";
        let e = extract(text);
        for entity in &e {
            if entity.entity_type == EntityType::Money {
                let extracted: String = text
                    .chars()
                    .skip(entity.start())
                    .take(entity.end() - entity.start())
                    .collect();
                assert_eq!(
                    extracted, entity.text,
                    "Char offsets must match entity text"
                );
            }
        }
    }

    // ========================================================================
    // Existing pattern regression: suffix-style currency codes still work
    // ========================================================================

    #[test]
    fn money_code_suffix_still_works() {
        let cases = ["100 USD", "500 EUR", "200 GBP", "1000 JPY"];
        for case in cases {
            let e = extract(case);
            assert!(
                has_type(&e, &EntityType::Money),
                "Suffix-style '{}' should still match, got: {:?}",
                case,
                e
            );
        }
    }

    #[test]
    fn money_european_decimal_comma() {
        // European decimal comma: €3,50 means 3 euros 50 cents
        let cases = ["€3,50", "€12,99", "£7,50"];
        for case in cases {
            let e = extract(case);
            assert!(
                has_type(&e, &EntityType::Money),
                "European decimal '{}' should be MONEY, got: {:?}",
                case,
                e
            );
        }
    }

    #[test]
    fn money_european_magnitude_abbreviations() {
        // German/European magnitude abbreviations via MONEY_CODE_PREFIX
        let cases = ["EUR 3,2 Mrd", "EUR 3.2 billion", "EUR 5,7 Mio"];
        for case in cases {
            let e = extract(case);
            assert!(
                has_type(&e, &EntityType::Money),
                "European magnitude '{}' should be MONEY, got: {:?}",
                case,
                e
            );
        }
    }

    #[test]
    fn money_us_format_unchanged() {
        // Existing US formats must still work
        let cases = ["$14,999.00", "$1,000,000", "$3.50"];
        for case in cases {
            let e = extract(case);
            assert!(
                has_type(&e, &EntityType::Money),
                "US format '{}' should still be MONEY, got: {:?}",
                case,
                e
            );
        }
    }

    // ========================================================================
    // Hashtag false-positive regression: invoice/reference numbers
    // ========================================================================

    #[test]
    fn hashtag_not_triggered_by_invoice_numbers() {
        let text = "Invoice #2024-0042 is due";
        let e = extract(text);
        let hashtags: Vec<_> = e
            .iter()
            .filter(|e| e.entity_type == EntityType::custom("Hashtag", crate::EntityCategory::Misc))
            .collect();
        assert!(
            hashtags.is_empty(),
            "Invoice number '#2024-0042' should not be tagged as Hashtag, got: {:?}",
            hashtags
        );
    }

    #[test]
    fn hashtag_still_matches_normal_tags() {
        let text = "Trending #rust today";
        let e = extract(text);
        assert!(
            has_type(
                &e,
                &EntityType::custom("Hashtag", crate::EntityCategory::Misc)
            ),
            "Normal hashtag '#rust' should still match, got: {:?}",
            e
        );
    }

    /// N6: MONEY_MAGNITUDE should capture trailing currency codes like USD/EUR.
    #[test]
    fn money_magnitude_with_trailing_currency_code() {
        for case in &["22.1 billion USD", "3.5 million EUR", "1 trillion GBP"] {
            let e = extract(case);
            let money: Vec<_> = e
                .iter()
                .filter(|e| e.entity_type == EntityType::Money)
                .collect();
            assert!(
                !money.is_empty(),
                "'{}' should be tagged as MONEY, got: {:?}",
                case,
                e
            );
            // The full string including currency code should be captured
            let full_match = money.iter().any(|m| {
                m.text.contains("USD") || m.text.contains("EUR") || m.text.contains("GBP")
            });
            assert!(
                full_match,
                "'{}' should include the currency code in the span, got: {:?}",
                case,
                money.iter().map(|m| &m.text).collect::<Vec<_>>()
            );
        }
    }
}

#[cfg(test)]
mod proptests;
