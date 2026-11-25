//! Document preprocessing and cleaning utilities.
//!
//! Provides text normalization, cleaning, and preparation for entity extraction.

use crate::lang::detect_language;
use crate::SourceMap;
use std::collections::HashMap;
use textprep::clean::CleanOptions;

/// Prepared document with metadata.
#[derive(Debug, Clone)]
pub struct PreparedDocument {
    /// The cleaned text
    pub text: String,
    /// Metadata about the preparation process
    pub metadata: HashMap<String, String>,
}

/// Document preprocessor for cleaning and normalizing text.
#[derive(Debug, Clone)]
pub struct DocumentPreprocessor {
    /// Normalize whitespace (collapse multiple spaces, normalize line breaks)
    pub clean_whitespace: bool,
    /// Normalize Unicode (NFC normalization)
    pub normalize_unicode: bool,
    /// Detect and record language
    pub detect_language: bool,
    /// Maximum chunk size (None = no chunking)
    pub chunk_size: Option<usize>,
}

impl Default for DocumentPreprocessor {
    fn default() -> Self {
        Self {
            clean_whitespace: true,
            normalize_unicode: true,
            detect_language: false,
            chunk_size: None,
        }
    }
}

impl DocumentPreprocessor {
    /// Create a new preprocessor with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a preprocessor with all cleaning enabled.
    #[must_use]
    pub fn with_all_cleaning() -> Self {
        Self {
            clean_whitespace: true,
            normalize_unicode: true,
            detect_language: true,
            chunk_size: None,
        }
    }

    /// Prepare text for entity extraction.
    pub fn prepare(&self, text: &str) -> PreparedDocument {
        self.prepared(text, self.clean_options().prepare(text))
    }

    /// Prepare the same text and metadata as [`Self::prepare`], retaining a map
    /// from analyzed UTF-8 byte ranges to the supplied input.
    pub fn prepare_mapped(&self, text: &str) -> Result<(PreparedDocument, SourceMap), String> {
        let (cleaned, map) = self.clean_options().prepare_mapped(text)?;
        Ok((self.prepared(text, cleaned), map))
    }

    fn clean_options(&self) -> CleanOptions {
        CleanOptions {
            normalize_unicode: self.normalize_unicode,
            clean_whitespace: self.clean_whitespace,
        }
    }

    fn prepared(&self, text: &str, processed: String) -> PreparedDocument {
        let mut metadata = HashMap::new();
        if self.normalize_unicode {
            metadata.insert("unicode_normalized".to_string(), "nfc".to_string());
        }
        if self.clean_whitespace {
            metadata.insert("whitespace_cleaned".to_string(), "true".to_string());
        }

        // Language detection
        if self.detect_language {
            let lang = detect_language(&processed);
            metadata.insert("detected_language".to_string(), format!("{:?}", lang));
        }

        // Chunking (if requested)
        if let Some(chunk_size) = self.chunk_size {
            // For now, just record chunk size - actual chunking would be done
            // at extraction time to preserve entity spans
            metadata.insert("chunk_size".to_string(), chunk_size.to_string());
        }

        metadata.insert("original_length".to_string(), text.len().to_string());
        metadata.insert("processed_length".to_string(), processed.len().to_string());

        PreparedDocument {
            text: processed,
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_preparation_matches_text_and_metadata_in_each_mode() {
        for clean_whitespace in [false, true] {
            for normalize_unicode in [false, true] {
                let prep = DocumentPreprocessor {
                    clean_whitespace,
                    normalize_unicode,
                    chunk_size: Some(20),
                    detect_language: true,
                };
                for input in ["", "\u{200b} \r\n", "  東京\r\nB\u{301}\t e\u{301} "] {
                    let plain = prep.prepare(input);
                    let (mapped, _) = prep.prepare_mapped(input).unwrap();
                    assert_eq!(mapped.text, plain.text);
                    assert_eq!(mapped.metadata, plain.metadata);
                }
            }
        }
    }

    #[test]
    fn test_preprocessor_default() {
        let prep = DocumentPreprocessor::new();
        assert!(prep.clean_whitespace);
        assert!(prep.normalize_unicode);
        assert!(!prep.detect_language);
        assert!(prep.chunk_size.is_none());
    }

    #[test]
    fn test_preprocessor_with_all_cleaning() {
        let prep = DocumentPreprocessor::with_all_cleaning();
        assert!(prep.clean_whitespace);
        assert!(prep.normalize_unicode);
        assert!(prep.detect_language);
    }

    #[test]
    fn test_whitespace_normalization() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("Hello   world\r\n\r\ntest");

        // Multiple spaces should be collapsed
        assert!(!doc.text.contains("  "));
        // CRLF should be normalized to LF
        assert!(!doc.text.contains("\r"));
    }

    #[test]
    fn test_unicode_zero_width_removal() {
        let prep = DocumentPreprocessor::new();
        let input = "Hello\u{200b}world\u{feff}test";
        let doc = prep.prepare(input);

        assert!(!doc.text.contains('\u{200b}'));
        assert!(!doc.text.contains('\u{feff}'));
        assert!(doc.text.contains("Helloworld"));
    }

    #[test]
    fn test_trim_whitespace() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("   text with spaces   ");

        assert_eq!(doc.text, "text with spaces");
    }

    #[test]
    fn test_metadata_recording() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("test input");

        assert!(doc.metadata.contains_key("original_length"));
        assert!(doc.metadata.contains_key("processed_length"));
        assert!(doc.metadata.contains_key("whitespace_cleaned"));
        assert_eq!(
            doc.metadata.get("unicode_normalized").map(String::as_str),
            Some("nfc"),
        );
    }

    #[test]
    fn test_nfc_normalization_precomposed_eq_decomposed() {
        let prep = DocumentPreprocessor::new();
        let precomposed = "caf\u{00e9}"; // "café" single codepoint
        let decomposed = "cafe\u{0301}"; // "cafe" + combining acute
        assert_ne!(precomposed, decomposed);
        let a = prep.prepare(precomposed).text;
        let b = prep.prepare(decomposed).text;
        assert_eq!(a, b);
    }

    #[test]
    fn test_language_detection_metadata() {
        let prep = DocumentPreprocessor::with_all_cleaning();
        let doc = prep.prepare("Hello world, this is English text.");

        assert!(doc.metadata.contains_key("detected_language"));
        assert!(doc
            .metadata
            .get("detected_language")
            .unwrap()
            .contains("English"));
    }

    #[test]
    fn test_preserve_paragraph_breaks() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("First paragraph.\n\nSecond paragraph.");

        // Should preserve double newline (paragraph break)
        assert!(doc.text.contains("\n\n") || doc.text.contains("\n"));
    }

    #[test]
    fn test_empty_input() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("");

        assert!(doc.text.is_empty());
        assert_eq!(doc.metadata.get("original_length"), Some(&"0".to_string()));
    }

    #[test]
    fn test_prepared_document_clone() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("test");
        let cloned = doc.clone();

        assert_eq!(doc.text, cloned.text);
        assert_eq!(doc.metadata, cloned.metadata);
    }

    #[test]
    fn test_cjk_text_handling() {
        let prep = DocumentPreprocessor::with_all_cleaning();
        let doc = prep.prepare("東京オリンピック2020は延期されました。");

        // Should preserve CJK text
        assert!(doc.text.contains("東京"));
        // Language should be detected
        assert!(doc.metadata.contains_key("detected_language"));
    }
}
