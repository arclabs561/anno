//! Lexicon-based NER backend.
//!
//! Provides exact-match entity lookup using gazetteers/lexicons.
//! Useful for closed-domain entities (stock tickers, medical codes, known catalogs).
//!
//! # Research Context
//!
//! Gazetteers are most valuable when:
//! 1. **Domain is closed**: Fixed, known entity lists
//! 2. **Text is short**: where context is insufficient (see the “gazetteer + neural” literature)
//! 3. **Used as features**: Input to neural model, not final output
//!
//! # Usage
//!
//! ```rust
//! use anno::{Model, LexiconNER};
//! use anno::{HashMapLexicon, EntityType};
//!
//! // Create a domain-specific lexicon
//! let mut lexicon = HashMapLexicon::new("stock_tickers");
//! lexicon.insert("AAPL", EntityType::Organization, 0.99);
//! lexicon.insert("GOOGL", EntityType::Organization, 0.99);
//!
//! // Use as a backend
//! let ner = LexiconNER::new(lexicon);
//! let entities = ner
//!     .extract_entities("AAPL stock rose today.", None)
//!     .unwrap();
//! ```
//!
//! # Integration with StackedNER
//!
//! LexiconNER can be used as a layer in StackedNER for hybrid extraction:
//!
//! ```rust
//! use anno::{Model, StackedNER, RegexNER, LexiconNER};
//! use anno::{HashMapLexicon, EntityCategory, EntityType};
//!
//! let mut lexicon = HashMapLexicon::new("medical_codes");
//! lexicon.insert("ICD-10", EntityType::custom("CODE", EntityCategory::Misc), 0.95);
//!
//! let ner = StackedNER::builder()
//!     .layer(RegexNER::new())           // Structured entities
//!     .layer(LexiconNER::new(lexicon))  // Domain-specific lookup
//!     .build();
//! ```

use crate::{Entity, EntityType, Language, Model, Result};
use anno_core::Lexicon;
use std::sync::Arc;

/// NER backend that uses exact-match lexicon lookup.
///
/// Scans text for known entities from a lexicon/gazetteer.
/// Best for closed-domain entities where the full list is known.
///
/// This is a **library-only** backend with no CLI entry point (`--model lexicon`
/// is not available). Use it programmatically through the [`Model`] trait.
pub struct LexiconNER {
    lexicon: Arc<dyn Lexicon + Send + Sync>,
    case_sensitive: bool,
    /// Minimum word boundary requirement (true = only match whole words)
    word_boundary: bool,
}

impl LexiconNER {
    /// Create a new LexiconNER with the given lexicon.
    pub fn new(lexicon: impl Lexicon + 'static) -> Self {
        Self {
            lexicon: Arc::new(lexicon),
            case_sensitive: false,
            word_boundary: true,
        }
    }

    /// Create with case-sensitive matching.
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Create with word boundary requirement.
    ///
    /// If `true`, only matches whole words (default).
    /// If `false`, matches substrings (e.g., "Apple" matches in "AppleInc").
    pub fn with_word_boundary(mut self, word_boundary: bool) -> Self {
        self.word_boundary = word_boundary;
        self
    }

    /// Get a reference to the underlying lexicon.
    pub fn lexicon(&self) -> &dyn Lexicon {
        self.lexicon.as_ref()
    }
}

impl Model for LexiconNER {
    fn extract_entities(&self, text: &str, language: Option<Language>) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // For efficiency with large lexicons, we scan the text and check potential spans
        // against the lexicon. This is O(n*m) where n=text length, m=avg entity length.
        // For production with large lexicons, consider Aho-Corasick algorithm.

        let text_chars: Vec<char> = text.chars().collect();
        let text_len = text_chars.len();

        // Detect if this is a CJK language (no word boundaries)
        let is_cjk = language.is_some_and(|l| l.is_cjk());

        // Helper to check if character is a word boundary marker
        // For CJK: punctuation and whitespace are boundaries
        // For other languages: alphanumeric vs non-alphanumeric
        let is_word_boundary_char = |c: char| -> bool {
            if is_cjk {
                // CJK: punctuation, whitespace, and some CJK punctuation marks
                c.is_whitespace()
                    || matches!(
                        c,
                        '。' | '，' | '、' | '；' | '：' | '？' | '！' | '・' | // CJK punctuation (Chinese/Japanese)
                    '.' | ',' | ';' | ':' | '?' | '!' | '(' | ')' | '[' | ']' | '{' | '}'
                    )
            } else {
                // Non-CJK: non-alphanumeric characters
                !c.is_alphanumeric()
            }
        };

        // Try all possible spans (word boundaries if word_boundary=true, or all substrings)
        for start in 0..text_len {
            // Try spans of increasing length
            for end in (start + 1)..=text_len.min(start + 50) {
                // Limit max span length
                let span_text: String = text_chars[start..end].iter().collect();

                // Check word boundary if required
                if self.word_boundary {
                    let is_word_start =
                        start == 0 || is_word_boundary_char(text_chars[start.saturating_sub(1)]);
                    let is_word_end = end >= text_len || is_word_boundary_char(text_chars[end]);
                    if !is_word_start || !is_word_end {
                        continue;
                    }
                }

                // Try exact match
                // For case-insensitive: we need to check if lexicon has the entry in any case
                // Since Lexicon trait only supports exact lookup, we try both original and lowercase
                // In a production system, consider using a case-normalized lexicon or Aho-Corasick
                let matched = if self.case_sensitive {
                    self.lexicon.lookup(&span_text)
                } else {
                    // Try original case first, then lowercase
                    // Note: This assumes lexicon entries are stored in a specific case
                    // For better case-insensitive matching, lexicon should normalize internally
                    self.lexicon
                        .lookup(&span_text)
                        .or_else(|| {
                            let lower = span_text.to_lowercase();
                            if lower != span_text {
                                self.lexicon.lookup(&lower)
                            } else {
                                None
                            }
                        })
                        // Also try with first letter capitalized (common pattern)
                        .or_else(|| {
                            let mut capitalized = span_text.to_lowercase();
                            if let Some(first) = capitalized.chars().next() {
                                capitalized.replace_range(
                                    0..first.len_utf8(),
                                    &first.to_uppercase().to_string(),
                                );
                                if capitalized != span_text {
                                    self.lexicon.lookup(&capitalized)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                };

                if let Some((entity_type, confidence)) = matched {
                    // start/end are already character indices (from text_chars: Vec<char>)
                    let char_start = start;
                    let char_end = end;

                    // Extract actual text span (preserving original case)
                    let actual_span: String = text.chars().skip(start).take(end - start).collect();

                    let provenance = anno_core::Provenance {
                        source: std::borrow::Cow::Borrowed("lexicon"),
                        method: anno_core::ExtractionMethod::Heuristic,
                        pattern: Some(std::borrow::Cow::Owned(format!(
                            "lexicon:{}",
                            self.lexicon.source()
                        ))),
                        raw_confidence: Some(confidence),
                        model_version: None,
                        timestamp: None,
                    };

                    entities.push(Entity::with_provenance(
                        actual_span,
                        entity_type,
                        char_start,
                        char_end,
                        confidence,
                        provenance,
                    ));

                    // Skip ahead to avoid overlapping matches (greedy matching)
                    break;
                }
            }
        }

        // Sort by position and remove overlaps (keep longest)
        entities.sort_by_key(|e| (e.start(), e.end()));
        let mut deduped: Vec<Entity> = Vec::new();
        for entity in entities {
            if deduped.is_empty() || !deduped.last().unwrap().overlaps(&entity) {
                deduped.push(entity);
            } else {
                // Keep the longer span
                let last = deduped.last_mut().unwrap();
                if entity.end() - entity.start() > last.end() - last.start() {
                    *last = entity;
                }
            }
        }

        Ok(deduped)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        // We can't enumerate all types from the lexicon trait alone
        // Return empty vec - types will be discovered during extraction
        // For better type reporting, consider adding an entries() method to Lexicon trait
        vec![]
    }

    fn is_available(&self) -> bool {
        !self.lexicon.is_empty()
    }

    fn name(&self) -> &'static str {
        "lexicon"
    }

    fn description(&self) -> &'static str {
        "Exact-match lexicon/gazetteer lookup"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno_core::HashMapLexicon;

    #[test]
    fn test_lexicon_ner_basic() {
        let mut lexicon = HashMapLexicon::new("test");
        lexicon.insert("Apple", EntityType::Organization, 0.99);
        lexicon.insert("Microsoft", EntityType::Organization, 0.99);

        let ner = LexiconNER::new(lexicon);
        let entities = ner
            .extract_entities("Apple and Microsoft are tech companies.", None)
            .unwrap();

        assert_eq!(entities.len(), 2);
        assert!(entities
            .iter()
            .any(|e| e.text == "Apple" && e.entity_type == EntityType::Organization));
        assert!(entities
            .iter()
            .any(|e| e.text == "Microsoft" && e.entity_type == EntityType::Organization));
    }

    #[test]
    fn test_lexicon_ner_case_insensitive() {
        let mut lexicon = HashMapLexicon::new("test");
        lexicon.insert("Apple", EntityType::Organization, 0.99);

        let ner = LexiconNER::new(lexicon);
        let entities = ner.extract_entities("apple stock rose.", None).unwrap();

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "apple");
    }

    #[test]
    fn test_lexicon_ner_word_boundary() {
        let mut lexicon = HashMapLexicon::new("test");
        lexicon.insert("Apple", EntityType::Organization, 0.99);

        let ner = LexiconNER::new(lexicon);
        let entities = ner
            .extract_entities("AppleInc is a company.", None)
            .unwrap();

        // With word boundary, "Apple" should not match in "AppleInc"
        // Note: This test may need adjustment based on word boundary detection logic
        assert_eq!(entities.len(), 0);
    }

    #[test]
    fn test_lexicon_ner_no_word_boundary() {
        let mut lexicon = HashMapLexicon::new("test");
        lexicon.insert("Apple", EntityType::Organization, 0.99);

        let ner = LexiconNER::new(lexicon).with_word_boundary(false);
        let entities = ner.extract_entities("AppleInc", None).unwrap();

        // Without word boundary, "Apple" should match in "AppleInc"
        assert!(entities.iter().any(|e| e.text == "Apple"));
    }

    #[test]
    fn test_lexicon_ner_unicode_offsets() {
        let mut lexicon = HashMapLexicon::new("test");
        lexicon.insert("東京", EntityType::Location, 0.99);

        let ner = LexiconNER::new(lexicon);
        let text = "Visit 東京 for tourism.";
        let entities = ner.extract_entities(text, None).unwrap();

        assert_eq!(entities.len(), 1);
        let entity = &entities[0];
        assert_eq!(entity.text, "東京");
        assert!(entity.start() < entity.end());
        assert!(entity.end() <= text.chars().count());
    }
}
