//! Parenthetical text analysis and entity extraction.
//!
//! # Overview
//!
//! Parentheticals are text enclosed in parentheses, brackets, or similar delimiters
//! that often contain valuable entity-related information:
//!
//! - **Aliases**: "Barack Obama (Barry)" - alternate names
//! - **Abbreviations**: "World Health Organization (WHO)"
//! - **Clarifications**: "The Big Apple (New York City)"
//! - **Stock tickers**: "Apple Inc. (AAPL)"
//! - **Temporal bounds**: "Napoleon Bonaparte (1769-1821)"
//! - **Translations**: "台北 (Taipei)"
//! - **Descriptions**: "John Smith (CEO of Acme Corp)"
//!
//! # Integration with Coalesce
//!
//! Parenthetical information provides crucial aliases for cross-document
//! entity coalescing. When "WHO" appears in one document and "World Health
//! Organization" in another, the parenthetical establishes the link.
//!
//! # Example
//!
//! ```rust
//! use anno::preprocess::parenthetical::{ParentheticalExtractor, ParentheticalType};
//!
//! let extractor = ParentheticalExtractor::new();
//! let text = "Apple Inc. (AAPL) reported earnings.";
//! let results = extractor.extract(text);
//!
//! assert_eq!(results.len(), 1);
//! assert_eq!(results[0].antecedent, "Apple Inc.");
//! assert_eq!(results[0].content, "AAPL");
//! assert_eq!(results[0].parenthetical_type, ParentheticalType::Ticker);
//! ```

use crate::offset::TextSpan;
use anno_core::Confidence;
use serde::{Deserialize, Serialize};

/// Type of parenthetical content.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ParentheticalType {
    /// Abbreviation/acronym: "World Health Organization (WHO)"
    Abbreviation,
    /// Stock ticker: "Apple Inc. (AAPL)"
    Ticker,
    /// Alternate name/alias: "William Shakespeare (The Bard)"
    Alias,
    /// Temporal bounds: "Napoleon (1769-1821)"
    TemporalBounds,
    /// Translation/transliteration: "北京 (Beijing)"
    Translation,
    /// Clarification/description: "the company (based in Seattle)"
    Clarification,
    /// Cross-reference: "see Section 3 (above)"
    CrossReference,
    /// Citation: "[Smith et al., 2020]"
    Citation,
    /// Role/title: "John Smith (CEO)"
    Role,
    /// Location qualifier: "Cambridge (Massachusetts)"
    LocationQualifier,
    /// Quantity/measurement: "500ml (about 2 cups)"
    Measurement,
    /// Unknown type
    #[default]
    Unknown,
}

/// A parenthetical extraction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parenthetical {
    /// The text preceding the parenthetical (the "antecedent")
    pub antecedent: String,
    /// The content inside the parentheses
    pub content: String,
    /// Start offset of the entire span (antecedent + parenthetical)
    pub start: usize,
    /// End offset of the entire span
    pub end: usize,
    /// Start offset of just the parenthetical content
    pub content_start: usize,
    /// End offset of just the parenthetical content
    pub content_end: usize,
    /// Type of parenthetical
    pub parenthetical_type: ParentheticalType,
    /// Confidence in the classification
    pub confidence: Confidence,
    /// Whether this creates an alias relationship
    pub is_alias: bool,
}

impl Parenthetical {
    /// Create a new parenthetical.
    pub fn new(
        antecedent: &str,
        content: &str,
        start: usize,
        end: usize,
        content_start: usize,
        content_end: usize,
    ) -> Self {
        Self {
            antecedent: antecedent.to_string(),
            content: content.to_string(),
            start,
            end,
            content_start,
            content_end,
            parenthetical_type: ParentheticalType::Unknown,
            confidence: Confidence::new(0.5),
            is_alias: false,
        }
    }

    /// Set the type.
    pub fn with_type(mut self, ptype: ParentheticalType) -> Self {
        self.parenthetical_type = ptype;
        self
    }

    /// Check if this represents an abbreviation.
    pub fn is_abbreviation(&self) -> bool {
        matches!(self.parenthetical_type, ParentheticalType::Abbreviation)
    }

    /// Check if this represents a stock ticker.
    pub fn is_ticker(&self) -> bool {
        matches!(self.parenthetical_type, ParentheticalType::Ticker)
    }

    /// Check if this provides temporal bounds for an entity.
    pub fn is_temporal(&self) -> bool {
        matches!(self.parenthetical_type, ParentheticalType::TemporalBounds)
    }

    /// Get the alias if this parenthetical creates one.
    ///
    /// For abbreviations and aliases, returns the content.
    /// For some types, the antecedent might be the alias.
    pub fn get_alias(&self) -> Option<&str> {
        if self.is_alias {
            Some(&self.content)
        } else {
            None
        }
    }
}

/// Extractor for parenthetical information.
#[derive(Debug, Clone, Default)]
pub struct ParentheticalExtractor {
    /// Minimum antecedent length to consider
    min_antecedent_len: usize,
    /// Maximum parenthetical content length
    max_content_len: usize,
}

impl ParentheticalExtractor {
    /// Create a new extractor with default settings.
    pub fn new() -> Self {
        Self {
            min_antecedent_len: 2,
            max_content_len: 100,
        }
    }

    /// Set minimum antecedent length.
    pub fn with_min_antecedent(mut self, len: usize) -> Self {
        self.min_antecedent_len = len;
        self
    }

    /// Extract parentheticals from text.
    pub fn extract(&self, text: &str) -> Vec<Parenthetical> {
        let mut results = Vec::new();
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i].1 == '(' {
                let open_idx = chars[i].0;

                // Find matching close paren
                let mut depth = 1;
                let mut j = i + 1;
                while j < chars.len() && depth > 0 {
                    match chars[j].1 {
                        '(' => depth += 1,
                        ')' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }

                if depth == 0 && j > i + 1 {
                    let close_idx = chars[j - 1].0;
                    let content_start = open_idx + 1;
                    let content_end = close_idx;
                    let content = &text[content_start..content_end];

                    // Skip if content too long
                    if content.chars().count() <= self.max_content_len {
                        // Find antecedent (text before the parenthetical)
                        let (antecedent, antecedent_start_byte, _antecedent_end_byte) =
                            self.find_antecedent(text, open_idx);

                        if antecedent.chars().count() >= self.min_antecedent_len {
                            let start_byte = antecedent_start_byte;
                            let end_byte = close_idx + 1; // ')' is ASCII
                            let span = TextSpan::from_bytes(text, start_byte, end_byte);
                            let content_span =
                                TextSpan::from_bytes(text, content_start, content_end);

                            let mut paren = Parenthetical::new(
                                &antecedent,
                                content,
                                span.char_start,
                                span.char_end,
                                content_span.char_start,
                                content_span.char_end,
                            );

                            // Classify the parenthetical
                            paren = self.classify(paren);

                            results.push(paren);
                        }
                    }
                }
                i = j;
            } else {
                i += 1;
            }
        }

        results
    }

    /// Find the antecedent (text before the parenthetical).
    ///
    /// Returns:
    /// - antecedent text (trimmed)
    /// - antecedent start byte offset (inclusive)
    /// - antecedent end byte offset (exclusive)
    fn find_antecedent(&self, text: &str, paren_start: usize) -> (String, usize, usize) {
        if paren_start == 0 {
            return (String::new(), 0, 0);
        }

        // Work backwards from the parenthesis
        let before = &text[..paren_start];
        let trimmed = before.trim_end();
        let trimmed_end = trimmed.len(); // byte offset in `text` (trimmed is a prefix slice)

        // Find the start of the phrase, but ignore periods in common abbreviations
        // like "Inc.", "Corp.", "Ltd.", "Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr."
        let abbrev_suffixes = [
            "Inc.", "Corp.", "Ltd.", "LLC.", "Co.", "Ltd", "Dr.", "Mr.", "Mrs.", "Ms.", "Jr.",
            "Sr.", "Ph.D.", "M.D.", "Prof.", "Rev.", "Gen.", "Col.", "Capt.", "Sgt.", "St.", "Mt.",
            "Ave.", "Blvd.", "Rd.",
        ];

        // Find sentence boundaries, but skip if it's part of an abbreviation
        let mut phrase_start = 0;
        let bytes = trimmed.as_bytes();

        for i in (0..bytes.len()).rev() {
            let c = bytes[i] as char;
            if c == '.' || c == ',' || c == ';' || c == ':' || c == '\n' {
                // Check if this is an abbreviation
                let suffix = &trimmed[..=i];
                let is_abbrev = abbrev_suffixes.iter().any(|abbr| suffix.ends_with(abbr));

                if !is_abbrev || c != '.' {
                    phrase_start = i + 1;
                    break;
                }
            }
        }

        // Skip any leading whitespace between phrase_start and trimmed_end.
        let mut antecedent_start = phrase_start;
        for (rel, c) in trimmed[phrase_start..].char_indices() {
            if !c.is_whitespace() {
                antecedent_start = phrase_start + rel;
                break;
            }
        }

        let antecedent = trimmed[antecedent_start..trimmed_end].to_string();
        (antecedent, antecedent_start, trimmed_end)
    }

    /// Classify the type of parenthetical.
    fn classify(&self, mut paren: Parenthetical) -> Parenthetical {
        let content = paren.content.trim();
        let antecedent = paren.antecedent.trim();

        // Check for stock ticker: all caps, 1-5 letters
        if content.len() <= 5
            && content.chars().all(|c| c.is_ascii_uppercase())
            && !content.is_empty()
        {
            // Check if antecedent looks like a company name
            if antecedent.ends_with("Inc.")
                || antecedent.ends_with("Corp.")
                || antecedent.ends_with("Ltd.")
                || antecedent.ends_with("LLC")
                || antecedent.ends_with("Company")
            {
                paren.parenthetical_type = ParentheticalType::Ticker;
                paren.is_alias = true;
                paren.confidence = Confidence::new(0.9);
                return paren;
            }
        }

        // Check for abbreviation/acronym
        if self.is_likely_abbreviation(antecedent, content) {
            paren.parenthetical_type = ParentheticalType::Abbreviation;
            paren.is_alias = true;
            paren.confidence = Confidence::new(0.85);
            return paren;
        }

        // Check for temporal bounds (years, date ranges)
        if self.is_temporal_bounds(content) {
            paren.parenthetical_type = ParentheticalType::TemporalBounds;
            paren.confidence = Confidence::new(0.9);
            return paren;
        }

        // Check for translation (contains non-ASCII)
        if !content.is_ascii() || !antecedent.is_ascii() {
            paren.parenthetical_type = ParentheticalType::Translation;
            paren.is_alias = true;
            paren.confidence = Confidence::new(0.7);
            return paren;
        }

        // Check for role/title
        if self.is_role(content) {
            paren.parenthetical_type = ParentheticalType::Role;
            paren.confidence = Confidence::new(0.8);
            return paren;
        }

        // Check for location qualifier
        if self.is_location_qualifier(content) {
            paren.parenthetical_type = ParentheticalType::LocationQualifier;
            paren.confidence = Confidence::new(0.75);
            return paren;
        }

        // Check for citation
        if content.starts_with('[')
            || content.contains("et al")
            || content.contains("19")
            || content.contains("20")
        {
            paren.parenthetical_type = ParentheticalType::Citation;
            paren.confidence = Confidence::new(0.7);
            return paren;
        }

        // Default to alias if short content that looks like a name
        if content.split_whitespace().count() <= 3
            && content
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
        {
            paren.parenthetical_type = ParentheticalType::Alias;
            paren.is_alias = true;
            paren.confidence = Confidence::new(0.6);
            return paren;
        }

        // Default to clarification
        paren.parenthetical_type = ParentheticalType::Clarification;
        paren.confidence = Confidence::new(0.5);
        paren
    }

    /// Check if content is likely an abbreviation of antecedent.
    fn is_likely_abbreviation(&self, antecedent: &str, content: &str) -> bool {
        // All caps content
        if !content
            .chars()
            .all(|c| c.is_uppercase() || c.is_whitespace() || c == '.')
        {
            return false;
        }

        // Check if initials match
        let antecedent_initials: String = antecedent
            .split_whitespace()
            .filter_map(|w| w.chars().next())
            .filter(|c| c.is_uppercase())
            .collect();

        let content_letters: String = content.chars().filter(|c| c.is_alphabetic()).collect();

        if antecedent_initials == content_letters {
            return true;
        }

        // Check if content could be abbreviation (3+ uppercase letters)
        content.len() >= 2 && content.len() <= 10
    }

    /// Check if content represents temporal bounds (birth-death years, etc.)
    fn is_temporal_bounds(&self, content: &str) -> bool {
        // Match patterns like "1769-1821", "b. 1950", "1920s", "born 1985"
        let patterns = [
            r"^\d{4}\s*[-–—]\s*\d{4}$",            // 1769-1821
            r"^\d{4}\s*[-–—]\s*(present|\d{4})?$", // 1990-present or 1990-
            r"^b\.\s*\d{4}$",                      // b. 1950
            r"^d\.\s*\d{4}$",                      // d. 2020
            r"^born\s+\d{4}$",                     // born 1985
            r"^\d{4}s$",                           // 1920s
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if re.is_match(content) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if content looks like a role/title.
    fn is_role(&self, content: &str) -> bool {
        let role_indicators = [
            "CEO",
            "CFO",
            "CTO",
            "COO",
            "CMO",
            "President",
            "Director",
            "Manager",
            "Chairman",
            "Senator",
            "Governor",
            "Mayor",
            "Minister",
            "Dr.",
            "Prof.",
            "Rev.",
            "founder",
            "co-founder",
            "editor",
        ];

        let lower = content.to_lowercase();
        role_indicators
            .iter()
            .any(|r| lower.contains(&r.to_lowercase()))
    }

    /// Check if content is a location qualifier.
    fn is_location_qualifier(&self, content: &str) -> bool {
        let qualifiers = [
            "UK",
            "US",
            "USA",
            "England",
            "Scotland",
            "Wales",
            "Massachusetts",
            "California",
            "Texas",
            "New York",
            "Ontario",
            "Quebec",
            "Bavaria",
            "Saxony",
        ];

        qualifiers.iter().any(|q| content.contains(q))
    }
}

/// Alias pair extracted from parentheticals.
///
/// Used for feeding into coalesce module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasPair {
    /// Primary name/surface form
    pub primary: String,
    /// Alias name/surface form
    pub alias: String,
    /// Source document ID
    pub doc_id: Option<String>,
    /// Confidence in this alias relationship
    pub confidence: Confidence,
    /// Type of alias relationship
    pub alias_type: ParentheticalType,
}

impl AliasPair {
    /// Create from a parenthetical.
    pub fn from_parenthetical(paren: &Parenthetical, doc_id: Option<&str>) -> Option<Self> {
        if !paren.is_alias {
            return None;
        }

        Some(Self {
            primary: paren.antecedent.clone(),
            alias: paren.content.clone(),
            doc_id: doc_id.map(|s| s.to_string()),
            confidence: paren.confidence,
            alias_type: paren.parenthetical_type.clone(),
        })
    }
}

/// Extract alias pairs from text for coalescing.
pub fn extract_aliases(text: &str, doc_id: Option<&str>) -> Vec<AliasPair> {
    let extractor = ParentheticalExtractor::new();
    let parentheticals = extractor.extract(text);

    parentheticals
        .iter()
        .filter_map(|p| AliasPair::from_parenthetical(p, doc_id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset::TextSpan;

    #[test]
    fn test_abbreviation_extraction() {
        let extractor = ParentheticalExtractor::new();
        let text = "The World Health Organization (WHO) announced new guidelines.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].antecedent, "The World Health Organization");
        assert_eq!(results[0].content, "WHO");
        assert_eq!(
            results[0].parenthetical_type,
            ParentheticalType::Abbreviation
        );
        assert!(results[0].is_alias);
    }

    #[test]
    fn test_ticker_extraction() {
        let extractor = ParentheticalExtractor::new();
        let text = "Apple Inc. (AAPL) reported strong earnings.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "AAPL");
        assert_eq!(results[0].parenthetical_type, ParentheticalType::Ticker);
    }

    #[test]
    fn test_temporal_bounds() {
        let extractor = ParentheticalExtractor::new();
        let text = "Napoleon Bonaparte (1769-1821) was Emperor of France.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "1769-1821");
        assert_eq!(
            results[0].parenthetical_type,
            ParentheticalType::TemporalBounds
        );
    }

    #[test]
    fn test_translation() {
        let extractor = ParentheticalExtractor::new();
        let text = "北京 (Beijing) is the capital.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].parenthetical_type,
            ParentheticalType::Translation
        );
    }

    #[test]
    fn test_role_extraction() {
        let extractor = ParentheticalExtractor::new();
        let text = "Tim Cook (CEO of Apple) spoke at the conference.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].parenthetical_type, ParentheticalType::Role);
    }

    #[test]
    fn test_parenthetical_offsets_are_character_offsets_with_unicode_prefix() {
        // ü is multi-byte, so byte offsets != char offsets; ensure we store char offsets.
        let extractor = ParentheticalExtractor::new();
        let text = "Müller (CEO) spoke.";
        let results = extractor.extract(text);
        assert_eq!(results.len(), 1);

        let p = &results[0];
        let span_text = TextSpan::from_chars(text, p.start, p.end).extract(text);
        assert_eq!(span_text, "Müller (CEO)");

        let content_text = TextSpan::from_chars(text, p.content_start, p.content_end).extract(text);
        assert_eq!(content_text, "CEO");
    }

    #[test]
    fn test_alias_pair_extraction() {
        let text = "The United Nations (UN) held a meeting.";
        let aliases = extract_aliases(text, Some("doc1"));

        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0].primary, "The United Nations");
        assert_eq!(aliases[0].alias, "UN");
        assert_eq!(aliases[0].doc_id, Some("doc1".to_string()));
    }

    #[test]
    fn test_multiple_parentheticals() {
        let extractor = ParentheticalExtractor::new();
        let text = "Microsoft Corp. (MSFT) and Apple Inc. (AAPL) are tech giants.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_nested_parentheses_skipped() {
        let extractor = ParentheticalExtractor::new();
        let text = "Complex formula (f(x) = x^2) is quadratic.";
        let results = extractor.extract(text);

        // Should still extract the outer parenthetical
        assert_eq!(results.len(), 1);
    }
}
