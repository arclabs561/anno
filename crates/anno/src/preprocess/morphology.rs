//! Morphological preprocessing for polysynthetic and agglutinative languages.
//!
//! # Overview
//!
//! This module provides preprocessing support for morphologically complex languages
//! where standard tokenization fails. Polysynthetic languages (Cherokee, Navajo, Mohawk)
//! encode entire sentences in single words; agglutinative languages (Quechua, Turkish)
//! have productive morpheme concatenation.
//!
//! # Problem Statement
//!
//! Standard NER assumes word-level spans work well for entity boundaries. For polysynthetic
//! languages, a single word may contain:
//! - Subject, object, and verb
//! - Tense, aspect, mood markers
//! - Evidentiality markers
//! - Named entity references
//!
//! Example (Mohawk): "wahshakotahráhkwen" = "he told someone something about him"
//!
//! # Approach
//!
//! 1. **Morpheme segmentation**: Split words into morphemes before NER
//! 2. **Entity span mapping**: Map morpheme spans back to character offsets
//! 3. **Pro-drop handling**: Insert placeholder nodes for null arguments
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::preprocess::morphology::{MorphologicalPreprocessor, SegmentationStrategy};
//!
//! let preprocessor = MorphologicalPreprocessor::new()
//!     .with_strategy(SegmentationStrategy::BPE { vocab_size: 5000 })
//!     .with_prodrop_expansion(true);
//!
//! let segmented = preprocessor.segment("wahshakotahráhkwen")?;
//! // Returns morpheme sequence with offset mapping
//! ```
//!
//! # References
//!
//! - qxoRef (Quechua): 3,137 morphemes across 1,413 words
//! - Cherokee syllabary: 85 characters representing CV syllables
//! - Navajo: Complex verbal morphology with prefix templates

use crate::offset::TextSpan;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy for morphological segmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SegmentationStrategy {
    /// Byte-Pair Encoding (BPE) based segmentation
    BPE {
        /// Target vocabulary size
        vocab_size: usize,
    },
    /// Character-level segmentation (fallback)
    Character,
    /// Syllable-based segmentation (for syllabic scripts like Cherokee)
    Syllable,
    /// Rule-based segmentation using morpheme boundaries
    RuleBased {
        /// Boundary markers (e.g., "-" for hyphenated morphemes)
        boundary_chars: Vec<char>,
    },
    /// External morphological analyzer (FST-based)
    External {
        /// Path to analyzer model
        model_path: String,
    },
}

impl Default for SegmentationStrategy {
    fn default() -> Self {
        SegmentationStrategy::RuleBased {
            boundary_chars: vec!['-'],
        }
    }
}

/// A morpheme with its position in the original text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Morpheme {
    /// The morpheme text
    pub text: String,
    /// Start offset in original text (character)
    pub start: usize,
    /// End offset in original text (character)
    pub end: usize,
    /// Morpheme type (if known)
    pub morph_type: Option<MorphemeType>,
    /// Gloss (if available)
    pub gloss: Option<String>,
}

/// Types of morphemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MorphemeType {
    /// Root/stem morpheme
    Root,
    /// Prefix
    Prefix,
    /// Suffix
    Suffix,
    /// Infix
    Infix,
    /// Circumfix
    Circumfix,
    /// Clitic
    Clitic,
    /// Unknown type
    Unknown,
}

/// Result of morphological segmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationResult {
    /// Original text
    pub original: String,
    /// Sequence of morphemes
    pub morphemes: Vec<Morpheme>,
    /// Whether pro-drop placeholders were inserted
    pub has_prodrop_placeholders: bool,
    /// Mapping from morpheme indices to character spans
    pub span_map: Vec<(usize, usize)>,
}

impl SegmentationResult {
    /// Get morpheme text joined with separator.
    pub fn joined(&self, separator: &str) -> String {
        self.morphemes
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>()
            .join(separator)
    }

    /// Map a morpheme span back to character offsets.
    pub fn morpheme_to_char_span(
        &self,
        morph_start: usize,
        morph_end: usize,
    ) -> Option<(usize, usize)> {
        if morph_start >= self.morphemes.len() || morph_end > self.morphemes.len() {
            return None;
        }
        let char_start = self.morphemes[morph_start].start;
        let char_end = self.morphemes[morph_end - 1].end;
        Some((char_start, char_end))
    }
}

/// Configuration for pro-drop handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProdropConfig {
    /// Insert placeholder for null subjects
    pub expand_null_subjects: bool,
    /// Insert placeholder for null objects
    pub expand_null_objects: bool,
    /// Placeholder token for null arguments
    pub placeholder_token: String,
}

impl Default for ProdropConfig {
    fn default() -> Self {
        Self {
            expand_null_subjects: true,
            expand_null_objects: false,
            placeholder_token: "[NULL]".to_string(),
        }
    }
}

/// Preprocessor for morphologically complex languages.
pub struct MorphologicalPreprocessor {
    strategy: SegmentationStrategy,
    prodrop_config: Option<ProdropConfig>,
    /// BPE vocabulary (if using BPE strategy)
    bpe_vocab: Option<HashMap<String, usize>>,
    /// Syllable inventory (if using syllable strategy)
    syllable_inventory: Option<Vec<String>>,
}

impl MorphologicalPreprocessor {
    /// Create a new preprocessor with default settings.
    pub fn new() -> Self {
        Self {
            strategy: SegmentationStrategy::default(),
            prodrop_config: None,
            bpe_vocab: None,
            syllable_inventory: None,
        }
    }

    /// Set the segmentation strategy.
    pub fn with_strategy(mut self, strategy: SegmentationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable pro-drop expansion.
    pub fn with_prodrop_expansion(mut self, config: ProdropConfig) -> Self {
        self.prodrop_config = Some(config);
        self
    }

    /// Load BPE vocabulary from file.
    pub fn load_bpe_vocab(&mut self, vocab: HashMap<String, usize>) {
        self.bpe_vocab = Some(vocab);
    }

    /// Load syllable inventory for syllabic scripts.
    pub fn load_syllable_inventory(&mut self, inventory: Vec<String>) {
        self.syllable_inventory = Some(inventory);
    }

    /// Segment text into morphemes.
    pub fn segment(&self, text: &str) -> Result<SegmentationResult> {
        let morphemes = match &self.strategy {
            SegmentationStrategy::BPE { vocab_size: _ } => self.segment_bpe(text)?,
            SegmentationStrategy::Character => self.segment_character(text),
            SegmentationStrategy::Syllable => self.segment_syllable(text)?,
            SegmentationStrategy::RuleBased { boundary_chars } => {
                self.segment_rule_based(text, boundary_chars)
            }
            SegmentationStrategy::External { model_path: _ } => {
                // External analyzers would be called here
                return Err(Error::FeatureNotAvailable(
                    "External morphological analyzer not yet implemented".to_string(),
                ));
            }
        };

        let span_map: Vec<(usize, usize)> = morphemes.iter().map(|m| (m.start, m.end)).collect();

        Ok(SegmentationResult {
            original: text.to_string(),
            morphemes,
            has_prodrop_placeholders: false,
            span_map,
        })
    }

    /// Character-level segmentation (baseline).
    fn segment_character(&self, text: &str) -> Vec<Morpheme> {
        text.char_indices()
            .map(|(i, c)| Morpheme {
                text: c.to_string(),
                start: i,
                end: i + c.len_utf8(),
                morph_type: Some(MorphemeType::Unknown),
                gloss: None,
            })
            .collect()
    }

    /// Rule-based segmentation using boundary characters.
    fn segment_rule_based(&self, text: &str, boundary_chars: &[char]) -> Vec<Morpheme> {
        let mut morphemes = Vec::new();
        let mut current_start = 0;
        let mut current_text = String::new();

        for (i, c) in text.char_indices() {
            if boundary_chars.contains(&c) {
                // Save current morpheme if non-empty
                if !current_text.is_empty() {
                    let span = TextSpan::from_bytes(text, current_start, i);
                    morphemes.push(Morpheme {
                        text: current_text.clone(),
                        start: span.char_start,
                        end: span.char_end,
                        morph_type: Some(MorphemeType::Unknown),
                        gloss: None,
                    });
                    current_text.clear();
                }
                current_start = i + c.len_utf8();
            } else {
                if current_text.is_empty() {
                    current_start = i;
                }
                current_text.push(c);
            }
        }

        // Don't forget the last morpheme
        if !current_text.is_empty() {
            let span = TextSpan::from_bytes(text, current_start, text.len());
            morphemes.push(Morpheme {
                text: current_text,
                start: span.char_start,
                end: span.char_end,
                morph_type: Some(MorphemeType::Unknown),
                gloss: None,
            });
        }

        morphemes
    }

    /// Syllable-based segmentation (for Cherokee, etc.).
    fn segment_syllable(&self, text: &str) -> Result<Vec<Morpheme>> {
        let inventory = self
            .syllable_inventory
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Syllable inventory not loaded".to_string()))?;

        let mut morphemes = Vec::new();
        let mut pos = 0; // byte offset

        // Greedy matching from syllable inventory
        while pos < text.len() {
            let mut matched = false;
            let remaining = &text[pos..];

            // Try to match longest syllable first
            for syllable in inventory.iter().rev() {
                // Assumes sorted by length
                if remaining.starts_with(syllable) {
                    let span = TextSpan::from_bytes(text, pos, pos + syllable.len());
                    morphemes.push(Morpheme {
                        text: syllable.clone(),
                        start: span.char_start,
                        end: span.char_end,
                        morph_type: Some(MorphemeType::Unknown),
                        gloss: None,
                    });
                    pos += syllable.len();
                    matched = true;
                    break;
                }
            }

            // Fallback to single character if no syllable matches
            if !matched {
                let c = text[pos..]
                    .chars()
                    .next()
                    .expect("pos should be within text bounds");
                let span = TextSpan::from_bytes(text, pos, pos + c.len_utf8());
                morphemes.push(Morpheme {
                    text: c.to_string(),
                    start: span.char_start,
                    end: span.char_end,
                    morph_type: Some(MorphemeType::Unknown),
                    gloss: None,
                });
                pos += c.len_utf8();
            }
        }

        Ok(morphemes)
    }

    /// BPE-based segmentation.
    fn segment_bpe(&self, text: &str) -> Result<Vec<Morpheme>> {
        let _vocab = self
            .bpe_vocab
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("BPE vocabulary not loaded".to_string()))?;

        // Simplified BPE: character-level with merge rules
        // Real implementation would use proper BPE algorithm
        Ok(self.segment_character(text))
    }
}

impl Default for MorphologicalPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Cherokee syllabary inventory (85 syllables).
///
/// Returns the Cherokee syllables sorted by length (longest first).
pub fn cherokee_syllable_inventory() -> Vec<String> {
    // Cherokee syllabary characters (U+13A0 to U+13F4)
    let syllables: Vec<String> = (0x13A0..=0x13F4)
        .filter_map(char::from_u32)
        .map(|c| c.to_string())
        .collect();
    syllables
}

/// Common Quechua morpheme boundaries.
pub fn quechua_boundary_chars() -> Vec<char> {
    vec!['-', '='] // Hyphen for morpheme, equals for clitic
}

/// Common Navajo prefix templates.
///
/// Navajo verbs have a complex template of prefix positions.
/// This returns common prefix morphemes.
pub fn navajo_prefix_inventory() -> Vec<String> {
    vec![
        // Object markers
        "shi-".to_string(), // 1sg object
        "ni-".to_string(),  // 2sg object
        "bi-".to_string(),  // 3rd person object
        // Subject markers
        "-ish".to_string(), // 1sg subject
        "-í".to_string(),   // 2sg subject
        // Aspect markers
        "yi-".to_string(), // perfective
        "na-".to_string(), // iterative
    ]
}

/// Trait for morphological analysis.
///
/// Implement this trait to integrate external morphological analyzers
/// (e.g., FST-based analyzers like HFST, Foma, or language-specific tools).
pub trait MorphologicalAnalyzer: Send + Sync {
    /// Analyze a word and return its morphemes.
    fn analyze(&self, word: &str) -> Result<Vec<Morpheme>>;

    /// Get the language code this analyzer supports.
    fn language_code(&self) -> &str;

    /// Whether this analyzer supports glossing.
    fn supports_glossing(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_segmentation_offsets_are_character_offsets_on_unicode() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::RuleBased {
                boundary_chars: vec!['-'],
            });

        // ü is multi-byte; offsets must still be character offsets.
        let text = "über-alles";
        let result = preprocessor.segment(text).expect("segment");
        assert_eq!(result.morphemes.len(), 2);

        assert_eq!(result.morphemes[0].text, "über");
        assert_eq!(result.morphemes[0].start, 0);
        assert_eq!(result.morphemes[0].end, 4);

        assert_eq!(result.morphemes[1].text, "alles");
        assert_eq!(result.morphemes[1].start, 5);
        assert_eq!(result.morphemes[1].end, 10);
    }

    #[test]
    fn test_rule_based_segmentation() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::RuleBased {
                boundary_chars: vec!['-'],
            });

        let result = preprocessor
            .segment("wasi-kuna-y-ki")
            .expect("valid Quechua word should segment");
        assert_eq!(result.morphemes.len(), 4);
        assert_eq!(result.morphemes[0].text, "wasi");
        assert_eq!(result.morphemes[1].text, "kuna");
        assert_eq!(result.morphemes[2].text, "y");
        assert_eq!(result.morphemes[3].text, "ki");
    }

    #[test]
    fn test_character_segmentation() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::Character);

        let result = preprocessor.segment("hello").unwrap();
        assert_eq!(result.morphemes.len(), 5);
    }

    #[test]
    fn test_span_mapping() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::RuleBased {
                boundary_chars: vec!['-'],
            });

        let result = preprocessor
            .segment("wasi-kuna")
            .expect("Quechua compound should segment");

        // Map morphemes 0-2 (both morphemes) back to character span
        let span = result
            .morpheme_to_char_span(0, 2)
            .expect("valid morpheme indices should map to span");
        assert_eq!(span, (0, 9)); // "wasi-kuna".len() == 9
    }

    #[test]
    fn test_cherokee_inventory() {
        let inventory = cherokee_syllable_inventory();
        assert!(!inventory.is_empty());
        // Cherokee syllabary has 85+ characters
        assert!(inventory.len() >= 85);
    }

    #[test]
    fn test_empty_string_handling() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::Character);
        let result = preprocessor.segment("").unwrap();
        assert!(result.morphemes.is_empty());
        assert_eq!(result.original, "");
    }

    #[test]
    fn test_unicode_handling() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::Character);

        // Cherokee syllabary
        let result = preprocessor
            .segment("ᏣᎳᎩ")
            .expect("Cherokee word should segment");
        assert_eq!(result.morphemes.len(), 3);

        // Nahuatl with diacritics
        let result = preprocessor.segment("Nāhuatl").unwrap();
        assert_eq!(result.morphemes.len(), 7);
    }

    #[test]
    fn test_rule_based_boundary_only() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::RuleBased {
                boundary_chars: vec!['-'],
            });

        // Input with only boundary chars
        let result = preprocessor
            .segment("---")
            .expect("punctuation should segment");
        assert!(result.morphemes.is_empty());
    }

    #[test]
    fn test_rule_based_no_boundaries() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::RuleBased {
                boundary_chars: vec!['-'],
            });

        // Input with no boundary chars
        let result = preprocessor.segment("word").unwrap();
        assert_eq!(result.morphemes.len(), 1);
        assert_eq!(result.morphemes[0].text, "word");
    }

    #[test]
    fn test_quechua_segmentation() {
        let preprocessor =
            MorphologicalPreprocessor::new().with_strategy(SegmentationStrategy::RuleBased {
                boundary_chars: quechua_boundary_chars(),
            });

        // Quechua word with hyphens
        let result = preprocessor
            .segment("wasi-kuna-y-ki")
            .expect("valid Quechua word should segment");
        assert_eq!(result.morphemes.len(), 4);

        // Verify span mapping works
        assert_eq!(
            result
                .morpheme_to_char_span(0, 1)
                .expect("valid morpheme indices should map to span"),
            (0, 4)
        ); // "wasi"
    }

    #[test]
    fn test_navajo_inventory() {
        let inventory = navajo_prefix_inventory();
        assert!(!inventory.is_empty());
        // Should have basic Navajo prefixes (note: stored with hyphens)
        assert!(inventory
            .iter()
            .any(|p| p.contains("na") || p.contains("ni") || p.contains("bi")));
    }
}
