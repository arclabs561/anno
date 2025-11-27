//! Declarative pattern configuration for PatternNER.
//!
//! This provides a cleaner, more maintainable way to define patterns
//! without boilerplate. Patterns are defined once and compiled lazily.

use crate::EntityType;
use once_cell::sync::Lazy;
use regex::Regex;

/// A pattern definition: regex + entity type + confidence + name.
pub struct PatternDef {
    /// The compiled regex pattern.
    pub regex: &'static Lazy<Regex>,
    /// The entity type to assign to matches.
    pub entity_type: EntityType,
    /// Confidence score (0.0-1.0).
    pub confidence: f64,
    /// Pattern name for provenance tracking.
    pub name: &'static str,
}

/// All pattern definitions, organized by priority.
///
/// Higher priority patterns are checked first. Within same priority,
/// order in the slice determines precedence.
pub static PATTERNS: Lazy<Vec<PatternDef>> = Lazy::new(|| {
    vec![
        // =====================================================================
        // High confidence: Very specific formats
        // =====================================================================
        
        // Emails (extremely specific)
        PatternDef { regex: &EMAIL, entity_type: EntityType::Email, confidence: 0.98, name: "EMAIL" },
        
        // URLs (extremely specific)
        PatternDef { regex: &URL, entity_type: EntityType::Url, confidence: 0.98, name: "URL" },
        
        // ISO dates (unambiguous)
        PatternDef { regex: &DATE_ISO, entity_type: EntityType::Date, confidence: 0.98, name: "DATE_ISO" },
        
        // =====================================================================
        // High confidence: Format-based detection
        // =====================================================================
        
        // Money with symbols
        PatternDef { regex: &MONEY_SYMBOL, entity_type: EntityType::Money, confidence: 0.95, name: "MONEY_SYMBOL" },
        PatternDef { regex: &MONEY_WRITTEN, entity_type: EntityType::Money, confidence: 0.95, name: "MONEY_WRITTEN" },
        PatternDef { regex: &MONEY_MAGNITUDE, entity_type: EntityType::Money, confidence: 0.92, name: "MONEY_MAGNITUDE" },
        
        // Dates (specific formats)
        PatternDef { regex: &DATE_US, entity_type: EntityType::Date, confidence: 0.95, name: "DATE_US" },
        PatternDef { regex: &DATE_EU, entity_type: EntityType::Date, confidence: 0.95, name: "DATE_EU" },
        PatternDef { regex: &DATE_WRITTEN_FULL, entity_type: EntityType::Date, confidence: 0.95, name: "DATE_WRITTEN_FULL" },
        PatternDef { regex: &DATE_WRITTEN_SHORT, entity_type: EntityType::Date, confidence: 0.95, name: "DATE_WRITTEN_SHORT" },
        PatternDef { regex: &DATE_WRITTEN_EU, entity_type: EntityType::Date, confidence: 0.95, name: "DATE_WRITTEN_EU" },
        
        // Percentages
        PatternDef { regex: &PERCENT, entity_type: EntityType::Percent, confidence: 0.95, name: "PERCENT" },
        
        // =====================================================================
        // Medium confidence: Times and phones (can have false positives)
        // =====================================================================
        
        // Times
        PatternDef { regex: &TIME_12H, entity_type: EntityType::Time, confidence: 0.90, name: "TIME_12H" },
        PatternDef { regex: &TIME_24H, entity_type: EntityType::Time, confidence: 0.88, name: "TIME_24H" },
        PatternDef { regex: &TIME_SIMPLE, entity_type: EntityType::Time, confidence: 0.85, name: "TIME_SIMPLE" },
        
        // Phone numbers (many false positives possible)
        PatternDef { regex: &PHONE_US, entity_type: EntityType::Phone, confidence: 0.85, name: "PHONE_US" },
        PatternDef { regex: &PHONE_INTL, entity_type: EntityType::Phone, confidence: 0.85, name: "PHONE_INTL" },
    ]
});

// =============================================================================
// Regex Definitions (compiled once, lazily)
// =============================================================================
// Note: These patterns are compile-time constants. If any regex is invalid,
// it's a programmer error that should panic immediately with a clear message.

// Date patterns
static DATE_ISO: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").expect("DATE_ISO regex is invalid")
});
static DATE_US: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b").expect("DATE_US regex is invalid")
});
static DATE_EU: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b").expect("DATE_EU regex is invalid")
});
static DATE_WRITTEN_FULL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b")
        .expect("DATE_WRITTEN_FULL regex is invalid")
});
static DATE_WRITTEN_SHORT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b")
        .expect("DATE_WRITTEN_SHORT regex is invalid")
});
static DATE_WRITTEN_EU: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?(?:\s+\d{4})?\b")
        .expect("DATE_WRITTEN_EU regex is invalid")
});

// Time patterns
static TIME_12H: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b")
        .expect("TIME_12H regex is invalid")
});
static TIME_24H: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b")
        .expect("TIME_24H regex is invalid")
});
static TIME_SIMPLE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d{1,2}\s*(?:am\b|pm\b|a\.m\.|p\.m\.)")
        .expect("TIME_SIMPLE regex is invalid")
});

// Money patterns
static MONEY_SYMBOL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[$€£¥][\d,]+(?:\.\d{1,2})?(?:\s*(?:billion|million|thousand|B|M|K|bn|mn))?")
        .expect("MONEY_SYMBOL regex is invalid")
});
static MONEY_WRITTEN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:dollars?|USD|euros?|EUR|pounds?|GBP|yen|JPY)\b")
        .expect("MONEY_WRITTEN regex is invalid")
});
static MONEY_MAGNITUDE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b\d+(?:\.\d+)?\s*(?:billion|million|trillion)\s*(?:dollars?|euros?|pounds?)?\b")
        .expect("MONEY_MAGNITUDE regex is invalid")
});

// Percent pattern
static PERCENT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|pct\b)")
        .expect("PERCENT regex is invalid")
});

// Contact patterns
static EMAIL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
        .expect("EMAIL regex is invalid")
});
static URL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bhttps?://[^\s<>\[\]{}|\\^`\x00-\x1f]+")
        .expect("URL regex is invalid")
});
static PHONE_US: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        .expect("PHONE_US regex is invalid")
});
static PHONE_INTL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b")
        .expect("PHONE_INTL regex is invalid")
});

/// Get all supported entity types (for Model trait).
pub fn supported_types() -> Vec<EntityType> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patterns_compile() {
        // Force lazy evaluation
        assert!(!PATTERNS.is_empty());
    }

    #[test]
    fn test_email_pattern() {
        assert!(EMAIL.is_match("test@example.com"));
        assert!(!EMAIL.is_match("not an email"));
    }

    #[test]
    fn test_url_pattern() {
        assert!(URL.is_match("https://example.com"));
        assert!(URL.is_match("http://sub.domain.org/path?query=1"));
        assert!(!URL.is_match("not a url"));
    }

    #[test]
    fn test_money_patterns() {
        assert!(MONEY_SYMBOL.is_match("$100"));
        assert!(MONEY_SYMBOL.is_match("€50.00"));
        assert!(MONEY_WRITTEN.is_match("100 dollars"));
        assert!(MONEY_MAGNITUDE.is_match("5 million dollars"));
    }
}

