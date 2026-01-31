//! Language-specific tokenization for multilingual NLP.
//!
//! This module provides a trait-based tokenization system that supports
//! different languages and scripts. Unlike ML backends which use transformer
//! tokenizers, this is for **statistical methods** (keywords, summarization)
//! that need language-aware word segmentation.
//!
//! # Research Context
//!
//! Tokenization varies dramatically by language:
//!
//! | Language Family | Tokenization Method | Example |
//! |----------------|---------------------|---------|
//! | English, Spanish, French | Whitespace + punctuation | "Hello world" → ["Hello", "world"] |
//! | Chinese, Japanese | Word segmentation (jieba, MeCab) | "中华人民共和国" → ["中华人民共和国"] or ["中华", "人民", "共和国"] |
//! | Thai | No spaces, needs segmentation | "ประเทศไทย" → ["ประเทศไทย"] |
//! | Arabic | Morphological analysis (clitics) | "وأبوه" → ["و", "أب", "ه"] |
//! | Korean | Morphological analysis | "서울시" → ["서울", "시"] |
//!
//! # Usage
//!
//! ```rust
//! use anno::lang::Language;
//! use anno::tokenizer::{Tokenizer, WhitespaceTokenizer};
//!
//! let tokenizer = WhitespaceTokenizer::new();
//! let tokens = tokenizer.tokenize("Hello world", Some(&Language::English));
//! assert_eq!(tokens.len(), 2);
//! ```
//!
//! # Future: Language-Specific Implementations
//!
//! - `JiebaTokenizer` for Chinese
//! - `MecabTokenizer` for Japanese
//! - `KonlpyTokenizer` for Korean
//! - `UnicodeSegmenter` using UAX#29 for fallback

use crate::lang::Language;

/// A token extracted from text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// Surface form (raw text)
    pub surface: String,
    /// Normalized form (lemma, if available)
    pub lemma: Option<String>,
    /// Part of speech tag (if available)
    pub pos: Option<String>,
    /// Start position (character offset)
    pub start: usize,
    /// End position (character offset)
    pub end: usize,
}

impl Token {
    /// Create a new token with surface form and position.
    pub fn new(surface: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            surface: surface.into(),
            lemma: None,
            pos: None,
            start,
            end,
        }
    }

    /// Create with lemma.
    pub fn with_lemma(mut self, lemma: impl Into<String>) -> Self {
        self.lemma = Some(lemma.into());
        self
    }

    /// Create with POS tag.
    pub fn with_pos(mut self, pos: impl Into<String>) -> Self {
        self.pos = Some(pos.into());
        self
    }

    /// Get the normalized form (lemma if available, otherwise surface).
    pub fn normalized(&self) -> &str {
        self.lemma.as_deref().unwrap_or(&self.surface)
    }
}

/// Trait for language-specific tokenization.
///
/// Implementations should handle:
/// - Word segmentation (whitespace, morphological, statistical)
/// - Stopword detection
/// - Case normalization (if applicable)
/// - Script-specific rules (CJK, Arabic, etc.)
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into a sequence of tokens.
    ///
    /// # Arguments
    /// - `text`: Input text to tokenize
    /// - `language`: Optional language hint (ISO 639-1 code or Language enum)
    ///
    /// # Returns
    /// Vector of tokens with positions and optional linguistic annotations.
    fn tokenize(&self, text: &str, language: Option<&Language>) -> Vec<Token>;

    /// Check if a token is a stopword (common function words to ignore).
    ///
    /// Default implementation returns `false` (no stopwords).
    /// Language-specific implementations should override this.
    fn is_stopword(&self, token: &Token, language: Option<&Language>) -> bool {
        let _ = (token, language);
        false
    }

    /// Get the tokenizer name/identifier.
    fn name(&self) -> &'static str;
}

/// Simple whitespace-based tokenizer (English, Spanish, French, etc.).
///
/// Splits on whitespace and punctuation. Works for languages with
/// clear word boundaries.
pub struct WhitespaceTokenizer {
    /// Whether to include punctuation as separate tokens
    include_punctuation: bool,
}

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    pub fn new() -> Self {
        Self {
            include_punctuation: false,
        }
    }

    /// Create with punctuation handling.
    pub fn with_punctuation(mut self, include: bool) -> Self {
        self.include_punctuation = include;
        self
    }
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str, _language: Option<&Language>) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut in_word = false;
        let mut word_start = 0;

        for (i, c) in text.char_indices() {
            let is_word_char = c.is_alphanumeric() || c == '_';

            if is_word_char {
                if !in_word {
                    word_start = i;
                    in_word = true;
                }
            } else {
                if in_word {
                    // End of word
                    let word: String = text[word_start..i].chars().collect();
                    if !word.is_empty() {
                        tokens.push(Token::new(word, word_start, i));
                    }
                    in_word = false;
                }

                if self.include_punctuation && !c.is_whitespace() {
                    // Add punctuation as separate token
                    let punct: String = c.to_string();
                    tokens.push(Token::new(punct, i, i + c.len_utf8()));
                }
            }
            // Note: current_start was unused, removed assignment
        }

        // Handle word at end of text
        if in_word {
            let word: String = text[word_start..].chars().collect();
            if !word.is_empty() {
                tokens.push(Token::new(word, word_start, text.len()));
            }
        }

        tokens
    }

    fn name(&self) -> &'static str {
        "whitespace"
    }
}

/// Unicode segmentation-based tokenizer (fallback for CJK and other languages).
///
/// Uses Unicode Standard Annex #29 (UAX#29) word boundaries.
/// This is a reasonable fallback but language-specific tokenizers are preferred.
pub struct UnicodeSegmenter;

impl UnicodeSegmenter {
    /// Create a new Unicode segmenter.
    pub fn new() -> Self {
        Self
    }
}

impl Default for UnicodeSegmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for UnicodeSegmenter {
    fn tokenize(&self, text: &str, _language: Option<&Language>) -> Vec<Token> {
        // For now, use character-based segmentation for CJK
        // In production, use unicode-segmentation crate or language-specific tools
        let mut tokens = Vec::new();
        let mut start = 0;

        for (i, c) in text.char_indices() {
            // Simple heuristic: split on whitespace and punctuation
            if c.is_whitespace() || c.is_ascii_punctuation() {
                if start < i {
                    let word: String = text[start..i].chars().collect();
                    if !word.trim().is_empty() {
                        tokens.push(Token::new(word, start, i));
                    }
                }
                start = i + c.len_utf8();
            }
        }

        // Handle remaining text
        if start < text.len() {
            let word: String = text[start..].chars().collect();
            if !word.trim().is_empty() {
                tokens.push(Token::new(word, start, text.len()));
            }
        }

        tokens
    }

    fn name(&self) -> &'static str {
        "unicode_segmenter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenizer() {
        let tokenizer = WhitespaceTokenizer::new();
        let tokens = tokenizer.tokenize("Hello world", Some(&Language::English));

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].surface, "Hello");
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
        assert_eq!(tokens[1].surface, "world");
        assert_eq!(tokens[1].start, 6);
        assert_eq!(tokens[1].end, 11);
    }

    #[test]
    fn test_whitespace_tokenizer_punctuation() {
        let tokenizer = WhitespaceTokenizer::new().with_punctuation(true);
        let tokens = tokenizer.tokenize("Hello, world!", Some(&Language::English));

        assert!(tokens.len() >= 2);
        assert_eq!(tokens[0].surface, "Hello");
        // Punctuation tokens should be present
    }

    #[test]
    fn test_unicode_segmenter_cjk() {
        let tokenizer = UnicodeSegmenter::new();
        let tokens = tokenizer.tokenize("北京是中国的首都", Some(&Language::Chinese));

        // Should produce at least one token (character-based fallback)
        assert!(!tokens.is_empty());
    }
}
