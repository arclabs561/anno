//! Unicode script detection for routing similarity algorithms.
//!
//! Extracted from similarity.rs to isolate potential compiler issues.

use serde::{Deserialize, Serialize};

/// Unicode script categories for routing similarity algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Script {
    /// Latin script (English, French, German, etc.)
    Latin,
    /// CJK (Chinese, Japanese Kanji, Korean Hanja)
    Cjk,
    /// Japanese Hiragana/Katakana
    Kana,
    /// Korean Hangul
    Hangul,
    /// Arabic script
    Arabic,
    /// Cyrillic script (Russian, etc.)
    Cyrillic,
    /// Devanagari (Hindi, Sanskrit, etc.)
    Devanagari,
    /// Greek script
    Greek,
    /// Hebrew script
    Hebrew,
    /// Thai script
    Thai,
    /// Mixed or unknown
    Mixed,
}

impl Script {
    /// Detect the dominant script in a string.
    ///
    /// Returns the script that appears most frequently.
    /// For mixed scripts (e.g., "東京 (Tokyo)"), returns Mixed if multiple scripts
    /// have significant presence (>= 20% of characters).
    pub fn detect(s: &str) -> Self {
        // Helper function to check if codepoint is in range
        #[inline(always)]
        fn in_range(cp: u32, start: u32, end: u32) -> bool {
            start <= cp && cp <= end
        }

        let mut counts = [0u32; 11]; // One per Script variant
        let mut total_chars = 0u32;

        for c in s.chars() {
            // Skip whitespace and punctuation for script detection
            if c.is_whitespace() || c.is_ascii_punctuation() {
                continue;
            }
            total_chars += 1;

            let cp = c as u32;
            // Use explicit range checks with helper to avoid compiler issues
            if cp <= 0x007F || in_range(cp, 0x0080, 0x024F) {
                counts[0] += 1; // Latin
            } else if in_range(cp, 0x4E00, 0x9FFF) || in_range(cp, 0x3400, 0x4DBF) {
                counts[1] += 1; // CJK
            } else if in_range(cp, 0x3040, 0x309F) || in_range(cp, 0x30A0, 0x30FF) {
                counts[2] += 1; // Kana
            } else if in_range(cp, 0xAC00, 0xD7AF) || in_range(cp, 0x1100, 0x11FF) {
                counts[3] += 1; // Hangul
            } else if in_range(cp, 0x0600, 0x06FF) || in_range(cp, 0x0750, 0x077F) {
                counts[4] += 1; // Arabic
            } else if in_range(cp, 0x0400, 0x04FF) || in_range(cp, 0x0500, 0x052F) {
                counts[5] += 1; // Cyrillic
            } else if in_range(cp, 0x0900, 0x097F) {
                counts[6] += 1; // Devanagari
            } else if in_range(cp, 0x0370, 0x03FF) || in_range(cp, 0x1F00, 0x1FFF) {
                counts[7] += 1; // Greek
            } else if in_range(cp, 0x0590, 0x05FF) {
                counts[8] += 1; // Hebrew
            } else if in_range(cp, 0x0E00, 0x0E7F) {
                counts[9] += 1; // Thai
            } else {
                counts[10] += 1; // Other
            }
        }

        if total_chars == 0 {
            return Script::Mixed;
        }

        // Check if multiple scripts have significant presence (>= 20%)
        // Use at least 1 as threshold to avoid counting zero-count scripts
        let threshold = ((total_chars as f32 * 0.2) as u32).max(1);
        let significant_scripts = counts.iter().filter(|&&c| c >= threshold).count();

        // If 2+ scripts are significant, return Mixed
        if significant_scripts >= 2 {
            return Script::Mixed;
        }

        // Find dominant script
        let scripts = [
            Script::Latin,
            Script::Cjk,
            Script::Kana,
            Script::Hangul,
            Script::Arabic,
            Script::Cyrillic,
            Script::Devanagari,
            Script::Greek,
            Script::Hebrew,
            Script::Thai,
            Script::Mixed,
        ];

        let max_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(i, _)| i)
            .unwrap_or(10);

        scripts[max_idx]
    }

    /// Whether this script uses word boundaries (spaces).
    pub fn has_word_boundaries(&self) -> bool {
        matches!(
            self,
            Script::Latin
                | Script::Cyrillic
                | Script::Greek
                | Script::Arabic
                | Script::Hebrew
                | Script::Devanagari
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_detection_latin() {
        assert_eq!(Script::detect("Hello World"), Script::Latin);
        assert_eq!(Script::detect("Marie Curie"), Script::Latin);
    }

    #[test]
    fn test_script_detection_cjk() {
        assert_eq!(Script::detect("北京"), Script::Cjk);
        // Note: "中华人民共和国" might be detected as Mixed if it contains punctuation
        // Test with pure CJK characters
        assert_eq!(Script::detect("中华人民共和国"), Script::Cjk);
        // Test with longer pure CJK text
        assert_eq!(Script::detect("中华人民共和国是伟大的国家"), Script::Cjk);
    }

    #[test]
    fn test_script_detection_mixed() {
        // Mixed script strings should be detected as Mixed
        assert_eq!(Script::detect("東京 (Tokyo)"), Script::Mixed);
    }
}
