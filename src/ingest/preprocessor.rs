//! Document preprocessing and cleaning utilities.
//!
//! Provides text normalization, cleaning, and preparation for entity extraction.

use crate::lang::detect_language;
use std::collections::HashMap;

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
        let mut processed = text.to_string();
        let mut metadata = HashMap::new();

        // Unicode normalization (NFC)
        // Note: For now, we do basic normalization without external crate
        // Full NFC normalization would require unicode-normalization crate
        if self.normalize_unicode {
            // Basic normalization: remove zero-width characters, normalize line breaks
            processed = processed
                .chars()
                .filter(|c| !matches!(c, '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}'))
                .collect();
            metadata.insert("unicode_normalized".to_string(), "basic".to_string());
        }

        // Whitespace cleaning
        if self.clean_whitespace {
            // Normalize line breaks to \n
            processed = processed.replace("\r\n", "\n").replace('\r', "\n");

            // Collapse multiple spaces (but preserve single spaces)
            let mut cleaned = String::with_capacity(processed.len());
            let mut last_was_space = false;
            for ch in processed.chars() {
                if ch.is_whitespace() {
                    if !last_was_space {
                        // Preserve newlines but collapse other whitespace
                        if ch == '\n' {
                            cleaned.push('\n');
                        } else {
                            cleaned.push(' ');
                        }
                        last_was_space = true;
                    } else if ch == '\n' && !cleaned.ends_with('\n') {
                        // Preserve consecutive newlines (paragraph breaks)
                        cleaned.push('\n');
                    }
                } else {
                    cleaned.push(ch);
                    last_was_space = false;
                }
            }

            // Trim leading/trailing whitespace
            processed = cleaned.trim().to_string();
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
