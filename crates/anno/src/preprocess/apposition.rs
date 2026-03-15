//! Apposition and Alias Pattern Extraction.
//!
//! Extracts alias relationships from various linguistic patterns beyond parentheticals.
//!
//! # Supported Patterns
//!
//! | Pattern | Example | Type |
//! |---------|---------|------|
//! | **Also known as** | "Peter Parker, also known as Spider-Man" | [`AppositionType::AlsoKnownAs`] |
//! | **AKA** | "Ringo Starr, aka Richard Starkey" | [`AppositionType::Aka`] |
//! | **Born** | "Lady Gaga, born Stefani Germanotta" | [`AppositionType::BirthName`] |
//! | **Formerly** | "Mumbai, formerly Bombay" | [`AppositionType::FormerlyKnownAs`] |
//! | **Now** | "Facebook, now Meta" | [`AppositionType::NowKnownAs`] |
//! | **Nickname** | "Dwayne 'The Rock' Johnson" | [`AppositionType::Nickname`] |
//! | **Colon** | "AWS: Amazon Web Services" | [`AppositionType::ColonExpansion`] |
//! | **Née** | "Hillary Clinton, née Rodham" | [`AppositionType::Nee`] |
//! | **Better known as** | "Marshall Mathers, better known as Eminem" | [`AppositionType::BetterKnownAs`] |
//!
//! # Canonical vs. Alternate Forms
//!
//! Different patterns have different directionality:
//!
//! - **"born X"**: The birth name (X) is canonical (legal name)
//! - **"better known as X"**: X is canonical (more recognized)
//! - **"formerly X"**: Current name is canonical
//! - **"aka X"**: Primary mention is canonical
//!
//! Use [`Apposition::canonical()`] and [`Apposition::alternate()`] to get the correct form.
//!
//! # Integration with Coalesce
//!
//! Extracted aliases integrate with `anno::coalesce` for cross-document entity linking:
//!
//! ```text
//! Appositions ──► AliasPairs ──► Coalesce ──► Unified Identities
//! ```
//!
//! # Example
//!
//! ```rust
//! use anno::preprocess::apposition::{AppositionExtractor, AppositionType};
//!
//! let extractor = AppositionExtractor::new();
//! let text = "Lady Gaga, born Stefani Germanotta, is a singer.";
//! let results = extractor.extract(text);
//!
//! assert_eq!(results.len(), 1);
//! assert_eq!(results[0].primary, "Lady Gaga");
//! assert_eq!(results[0].alias, "Stefani Germanotta");
//! assert_eq!(results[0].apposition_type, AppositionType::BirthName);
//!
//! // Birth name is the canonical (legal) form
//! assert_eq!(results[0].canonical(), "Stefani Germanotta");
//! assert_eq!(results[0].alternate(), "Lady Gaga");
//! ```
//!
//! # Linguistic Background
//!
//! These patterns are related to but distinct from:
//!
//! - **Appositions**: Noun phrases that rename another noun ("Obama, the president")
//! - **Parentheticals**: Insertions that can be removed without loss of grammaticality
//! - **Copular constructions**: "X is Y" relationships
//!
//! This module focuses specifically on **alias-introducing patterns** that establish
//! identity relationships between different surface forms of the same entity.

use anno_core::Confidence;
use serde::{Deserialize, Serialize};

/// Type of apposition/alias pattern.
///
/// Each type has different semantics for which form is "canonical":
///
/// - `BirthName`, `RealName`: The alias is canonical (legal name)
/// - `BetterKnownAs`, `NowKnownAs`: The alias is canonical (current/famous name)
/// - `FormerlyKnownAs`, `Aka`, `AlsoKnownAs`: The primary is canonical
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AppositionType {
    /// Standard appositive: "Obama, the president, ..."
    Appositive,
    /// Also known as: "Spider-Man, also known as Peter Parker"
    AlsoKnownAs,
    /// AKA abbreviation: "Ringo Starr, aka Richard Starkey"
    Aka,
    /// Nickname in quotes: "Dwayne 'The Rock' Johnson"
    Nickname,
    /// Birth name: "Lady Gaga, born Stefani Germanotta"
    BirthName,
    /// Former name: "Mumbai, formerly Bombay"
    FormerlyKnownAs,
    /// Renamed: "Meta, formerly Facebook"
    Renamed,
    /// Now known as: "Facebook, now Meta"
    NowKnownAs,
    /// Colon expansion: "AWS: Amazon Web Services"
    ColonExpansion,
    /// Or alternative: "Myanmar (or Burma)"
    OrAlternative,
    /// Real name: "Eminem, real name Marshall Mathers"
    RealName,
    /// Better known as: "Marshall Mathers, better known as Eminem"
    BetterKnownAs,
    /// Née (maiden name): "Hillary Clinton, née Rodham"
    Nee,
    /// Styled as: "Prince, styled as ꛦ for a period"
    StyledAs,
    /// Generic alias
    #[default]
    Generic,
}

/// An extracted apposition/alias relationship.
///
/// Represents an alias relationship between two surface forms of an entity.
/// The `primary` field contains the first-mentioned form, and `alias` contains
/// the form introduced by the pattern.
///
/// Use [`canonical()`](Apposition::canonical) and [`alternate()`](Apposition::alternate)
/// to get the semantically appropriate form regardless of mention order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Apposition {
    /// Primary entity text (first-mentioned form)
    pub primary: String,
    /// Alias/alternate text (introduced by pattern)
    pub alias: String,
    /// Start offset of entire span in source text
    pub start: usize,
    /// End offset of entire span in source text
    pub end: usize,
    /// Type of apposition pattern
    pub apposition_type: AppositionType,
    /// Confidence in extraction (0.0-1.0)
    pub confidence: Confidence,
    /// Direction: true if primary→alias is the name→alias direction
    pub primary_is_canonical: bool,
}

impl Apposition {
    /// Create a new apposition with default settings.
    ///
    /// By default, assumes the primary (first-mentioned) form is canonical.
    pub fn new(primary: &str, alias: &str, start: usize, end: usize) -> Self {
        Self {
            primary: primary.to_string(),
            alias: alias.to_string(),
            start,
            end,
            apposition_type: AppositionType::Generic,
            confidence: Confidence::new(0.7),
            primary_is_canonical: true,
        }
    }

    /// Set the apposition type.
    #[must_use]
    pub fn with_type(mut self, atype: AppositionType) -> Self {
        self.apposition_type = atype;
        self
    }

    /// Mark that the alias is the canonical form.
    ///
    /// Use for patterns like "born X" or "better known as X" where the
    /// introduced form is the canonical one.
    #[must_use]
    pub fn alias_is_canonical(mut self) -> Self {
        self.primary_is_canonical = false;
        self
    }

    /// Get the canonical (preferred) form.
    ///
    /// # Examples
    ///
    /// - "Lady Gaga, born Stefani Germanotta" → "Stefani Germanotta" (legal name)
    /// - "Mumbai, formerly Bombay" → "Mumbai" (current name)
    pub fn canonical(&self) -> &str {
        if self.primary_is_canonical {
            &self.primary
        } else {
            &self.alias
        }
    }

    /// Get the alternate (non-canonical) form.
    pub fn alternate(&self) -> &str {
        if self.primary_is_canonical {
            &self.alias
        } else {
            &self.primary
        }
    }
}

/// Extractor for appositions and alias patterns.
///
/// # Configuration
///
/// Each pattern category can be individually enabled/disabled:
///
/// ```rust
/// use anno::preprocess::apposition::AppositionExtractor;
///
/// let extractor = AppositionExtractor::new();
/// // All patterns enabled by default
/// ```
#[derive(Debug, Clone, Default)]
pub struct AppositionExtractor {
    /// Extract AKA patterns
    extract_aka: bool,
    /// Extract nickname patterns
    extract_nicknames: bool,
    /// Extract formerly/now patterns
    extract_rename: bool,
    /// Extract colon expansions
    extract_colon: bool,
}

impl AppositionExtractor {
    /// Create a new extractor with all patterns enabled.
    pub fn new() -> Self {
        Self {
            extract_aka: true,
            extract_nicknames: true,
            extract_rename: true,
            extract_colon: true,
        }
    }

    /// Extract all alias patterns from text.
    ///
    /// Returns a deduplicated list sorted by position. Overlapping
    /// extractions are resolved by keeping the highest-confidence one.
    pub fn extract(&self, text: &str) -> Vec<Apposition> {
        let mut results = Vec::new();

        // AKA patterns
        if self.extract_aka {
            results.extend(self.extract_aka_patterns(text));
        }

        // Born patterns
        results.extend(self.extract_born_patterns(text));

        // Formerly patterns
        if self.extract_rename {
            results.extend(self.extract_rename_patterns(text));
        }

        // Nickname patterns
        if self.extract_nicknames {
            results.extend(self.extract_nickname_patterns(text));
        }

        // Colon expansions
        if self.extract_colon {
            results.extend(self.extract_colon_patterns(text));
        }

        // Née patterns
        results.extend(self.extract_nee_patterns(text));

        // Sort by position, deduplicate overlapping
        results.sort_by_key(|a| a.start);
        self.remove_overlaps(results)
    }

    /// Extract "also known as" / "aka" patterns.
    fn extract_aka_patterns(&self, text: &str) -> Vec<Apposition> {
        let mut results = Vec::new();
        let _lower = text.to_lowercase();

        // Pattern: "X, also known as Y"
        let patterns = [
            (
                r"([A-Z][^,]+),\s*also known as\s+([A-Z][^,.]+)",
                AppositionType::AlsoKnownAs,
                true,
            ),
            (
                r"([A-Z][^,]+),\s*a\.k\.a\.?\s+([A-Z][^,.]+)",
                AppositionType::Aka,
                true,
            ),
            (
                r"([A-Z][^,]+),\s*aka\s+([A-Z][^,.]+)",
                AppositionType::Aka,
                true,
            ),
            (
                r"([A-Z][^,]+),\s*better known as\s+([A-Z][^,.]+)",
                AppositionType::BetterKnownAs,
                false,
            ),
            (
                r"([A-Z][^,]+),\s*real name\s+([A-Z][^,.]+)",
                AppositionType::RealName,
                false,
            ),
        ];

        for (pattern, atype, primary_canonical) in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(text) {
                    if let (Some(m1), Some(m2)) = (cap.get(1), cap.get(2)) {
                        let mut appo = Apposition::new(
                            m1.as_str().trim(),
                            m2.as_str().trim(),
                            cap.get(0).expect("regex match should have group 0").start(),
                            cap.get(0).expect("regex match should have group 0").end(),
                        )
                        .with_type(atype.clone());

                        if !*primary_canonical {
                            appo = appo.alias_is_canonical();
                        }
                        appo.confidence = Confidence::new(0.9);

                        results.push(appo);
                    }
                }
            }
        }

        results
    }

    /// Extract "born X" patterns.
    fn extract_born_patterns(&self, text: &str) -> Vec<Apposition> {
        let mut results = Vec::new();

        // Pattern: "X, born Y" or "X (born Y)"
        let patterns = [
            r"([A-Z][A-Za-z\s]+),\s*born\s+([A-Z][A-Za-z\s]+?)(?:[,.]|$)",
            r"([A-Z][A-Za-z\s]+)\s*\(born\s+([A-Z][A-Za-z\s]+)\)",
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(text) {
                    if let (Some(m1), Some(m2)) = (cap.get(1), cap.get(2)) {
                        let appo = Apposition::new(
                            m1.as_str().trim(),
                            m2.as_str().trim(),
                            cap.get(0).expect("regex match should have group 0").start(),
                            cap.get(0).expect("regex match should have group 0").end(),
                        )
                        .with_type(AppositionType::BirthName)
                        .alias_is_canonical(); // Birth name is the canonical legal name

                        results.push(appo);
                    }
                }
            }
        }

        results
    }

    /// Extract "formerly" / "now" / "previously" patterns.
    fn extract_rename_patterns(&self, text: &str) -> Vec<Apposition> {
        let mut results = Vec::new();

        let patterns = [
            (
                r"([A-Z][A-Za-z\s]+),\s*formerly\s+(?:known as\s+)?([A-Z][A-Za-z\s]+?)(?:[,.]|$)",
                AppositionType::FormerlyKnownAs,
                true,
            ),
            (
                r"([A-Z][A-Za-z\s]+),\s*previously\s+(?:known as\s+)?([A-Z][A-Za-z\s]+?)(?:[,.]|$)",
                AppositionType::FormerlyKnownAs,
                true,
            ),
            (
                r"([A-Z][A-Za-z\s]+),\s*now\s+(?:known as\s+)?([A-Z][A-Za-z\s]+?)(?:[,.]|$)",
                AppositionType::NowKnownAs,
                false,
            ),
            (
                r"([A-Z][A-Za-z\s]+),\s*currently\s+(?:known as\s+)?([A-Z][A-Za-z\s]+?)(?:[,.]|$)",
                AppositionType::NowKnownAs,
                false,
            ),
        ];

        for (pattern, atype, primary_canonical) in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(text) {
                    if let (Some(m1), Some(m2)) = (cap.get(1), cap.get(2)) {
                        let mut appo = Apposition::new(
                            m1.as_str().trim(),
                            m2.as_str().trim(),
                            cap.get(0).expect("regex match should have group 0").start(),
                            cap.get(0).expect("regex match should have group 0").end(),
                        )
                        .with_type(atype.clone());

                        if !*primary_canonical {
                            appo = appo.alias_is_canonical();
                        }
                        appo.confidence = Confidence::new(0.85);

                        results.push(appo);
                    }
                }
            }
        }

        results
    }

    /// Extract nickname patterns (quotes).
    fn extract_nickname_patterns(&self, text: &str) -> Vec<Apposition> {
        let mut results = Vec::new();

        // Pattern: "FirstName 'Nickname' LastName" or "FirstName "Nickname" LastName"
        let patterns = [
            r#"([A-Z][a-z]+)\s+'([A-Z][^']+)'\s+([A-Z][a-z]+)"#,
            r#"([A-Z][a-z]+)\s+"([A-Z][^"]+)"\s+([A-Z][a-z]+)"#,
            r#"([A-Z][a-z]+)\s+'([A-Z][^']+)'\s+([A-Z][a-z]+)"#, // curly quotes
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(text) {
                    if let (Some(first), Some(nick), Some(last)) =
                        (cap.get(1), cap.get(2), cap.get(3))
                    {
                        let full_name = format!("{} {}", first.as_str(), last.as_str());
                        let appo = Apposition::new(
                            &full_name,
                            nick.as_str(),
                            cap.get(0).expect("regex match should have group 0").start(),
                            cap.get(0).expect("regex match should have group 0").end(),
                        )
                        .with_type(AppositionType::Nickname);

                        results.push(appo);
                    }
                }
            }
        }

        results
    }

    /// Extract colon expansion patterns.
    fn extract_colon_patterns(&self, text: &str) -> Vec<Apposition> {
        let mut results = Vec::new();

        // Pattern: "ABBREV: Full Name" - match capitalized name after colon
        // The full form typically ends at lowercase word start or punctuation
        if let Ok(re) = regex::Regex::new(r"([A-Z]{2,8}):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)") {
            for cap in re.captures_iter(text) {
                if let (Some(abbrev), Some(full)) = (cap.get(1), cap.get(2)) {
                    let full_text = full.as_str().trim();
                    let group_0 = cap.get(0).expect("regex match should have group 0");
                    let appo =
                        Apposition::new(full_text, abbrev.as_str(), group_0.start(), group_0.end())
                            .with_type(AppositionType::ColonExpansion);

                    results.push(appo);
                }
            }
        }

        results
    }

    /// Extract "née" patterns for maiden names.
    fn extract_nee_patterns(&self, text: &str) -> Vec<Apposition> {
        let mut results = Vec::new();

        let patterns = [
            r"([A-Z][A-Za-z\s]+),\s*née\s+([A-Z][a-z]+)",
            r"([A-Z][A-Za-z\s]+),\s*nee\s+([A-Z][a-z]+)",
            r"([A-Z][A-Za-z\s]+)\s*\(née\s+([A-Z][a-z]+)\)",
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for cap in re.captures_iter(text) {
                    if let (Some(m1), Some(m2)) = (cap.get(1), cap.get(2)) {
                        let appo = Apposition::new(
                            m1.as_str().trim(),
                            m2.as_str().trim(),
                            cap.get(0).expect("regex match should have group 0").start(),
                            cap.get(0).expect("regex match should have group 0").end(),
                        )
                        .with_type(AppositionType::Nee);

                        results.push(appo);
                    }
                }
            }
        }

        results
    }

    /// Remove overlapping extractions, keeping highest confidence.
    fn remove_overlaps(&self, mut appos: Vec<Apposition>) -> Vec<Apposition> {
        appos.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut result = Vec::new();
        for appo in appos {
            let overlaps = result
                .iter()
                .any(|a: &Apposition| appo.start < a.end && appo.end > a.start);
            if !overlaps {
                result.push(appo);
            }
        }

        result.sort_by_key(|a| a.start);
        result
    }
}

/// Combined alias extraction from all sources.
///
/// Combines parentheticals and appositions into a unified alias list.
/// Returns tuples of (canonical, alternate, confidence).
///
/// # Example
///
/// ```rust
/// use anno::preprocess::apposition::extract_all_aliases;
///
/// let text = "Apple Inc. (AAPL), formerly Apple Computer, is based in Cupertino.";
/// let aliases = extract_all_aliases(text);
///
/// // Will find both the ticker parenthetical and the formerly pattern
/// for (canonical, alternate, confidence) in aliases {
///     println!("{} = {} ({})", canonical, alternate, confidence);
/// }
/// ```
pub fn extract_all_aliases(text: &str) -> Vec<(String, String, Confidence)> {
    use super::parenthetical::ParentheticalExtractor;

    let mut aliases = Vec::new();

    // Parentheticals
    let paren_ext = ParentheticalExtractor::new();
    for paren in paren_ext.extract(text) {
        if paren.is_alias {
            aliases.push((paren.antecedent, paren.content, paren.confidence));
        }
    }

    // Appositions
    let appo_ext = AppositionExtractor::new();
    for appo in appo_ext.extract(text) {
        aliases.push((
            appo.canonical().to_string(),
            appo.alternate().to_string(),
            appo.confidence,
        ));
    }

    aliases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aka_pattern() {
        let extractor = AppositionExtractor::new();
        let text = "Peter Parker, also known as Spider-Man, saved the city.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].primary, "Peter Parker");
        assert_eq!(results[0].alias, "Spider-Man");
        assert_eq!(results[0].apposition_type, AppositionType::AlsoKnownAs);
    }

    #[test]
    fn test_born_pattern() {
        let extractor = AppositionExtractor::new();
        let text = "Lady Gaga, born Stefani Germanotta, is a famous singer.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].primary, "Lady Gaga");
        assert_eq!(results[0].alias, "Stefani Germanotta");
        assert_eq!(results[0].apposition_type, AppositionType::BirthName);
        // Birth name is canonical
        assert_eq!(results[0].canonical(), "Stefani Germanotta");
    }

    #[test]
    fn test_formerly_pattern() {
        let extractor = AppositionExtractor::new();
        let text = "Mumbai, formerly Bombay, is India's largest city.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].primary, "Mumbai");
        assert_eq!(results[0].alias, "Bombay");
        assert_eq!(results[0].apposition_type, AppositionType::FormerlyKnownAs);
    }

    #[test]
    fn test_nickname_pattern() {
        let extractor = AppositionExtractor::new();
        let text = "Dwayne 'The Rock' Johnson is an actor.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].primary, "Dwayne Johnson");
        assert_eq!(results[0].alias, "The Rock");
        assert_eq!(results[0].apposition_type, AppositionType::Nickname);
    }

    #[test]
    fn test_colon_pattern() {
        let extractor = AppositionExtractor::new();
        let text = "AWS: Amazon Web Services provides cloud computing.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].alias, "AWS");
        assert_eq!(results[0].primary, "Amazon Web Services");
    }

    #[test]
    fn test_nee_pattern() {
        let extractor = AppositionExtractor::new();
        let text = "Hillary Clinton, née Rodham, was Secretary of State.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].alias, "Rodham");
        assert_eq!(results[0].apposition_type, AppositionType::Nee);
    }

    #[test]
    fn test_combined_extraction() {
        let text = "Apple Inc. (AAPL), formerly Apple Computer, launched the iPhone.";
        let aliases = extract_all_aliases(text);

        // Should find both the ticker parenthetical and the formerly pattern
        assert!(!aliases.is_empty());
    }

    #[test]
    fn test_better_known_as() {
        let extractor = AppositionExtractor::new();
        let text = "Marshall Mathers, better known as Eminem, is a rapper.";
        let results = extractor.extract(text);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].apposition_type, AppositionType::BetterKnownAs);
        // "Eminem" is the canonical (better known) form
        assert_eq!(results[0].canonical(), "Eminem");
    }
}
