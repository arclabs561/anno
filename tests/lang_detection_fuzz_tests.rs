//! Property-based tests for language detection.
//!
//! These tests verify that language detection never panics and
//! handles arbitrary Unicode input correctly.

use anno::lang::{detect_language, Language};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Language detection should never panic on arbitrary input.
    #[test]
    fn language_detection_never_panics(
        text in ".{0,500}",
    ) {
        let lang = detect_language(&text);

        // Should always return a valid Language
        match lang {
            Language::English | Language::German | Language::French |
            Language::Spanish | Language::Italian | Language::Portuguese |
            Language::Russian | Language::Chinese | Language::Japanese |
            Language::Korean | Language::Arabic | Language::Hebrew |
            Language::Other => {}
        }
    }

    /// Language detection should handle empty/whitespace text.
    #[test]
    fn language_detection_empty_text(
        text in "\\s*",  // Whitespace only
    ) {
        let lang = detect_language(&text);
        // Empty/whitespace should default to English
        prop_assert_eq!(lang, Language::English,
            "Empty/whitespace text should default to English, got {:?}", lang);
    }

    /// Language detection should handle very long text.
    #[test]
    fn language_detection_long_text(
        text in ".{1000,10000}",
    ) {
        let lang = detect_language(&text);

        // Should not panic and should return valid language
        match lang {
            Language::English | Language::German | Language::French |
            Language::Spanish | Language::Italian | Language::Portuguese |
            Language::Russian | Language::Chinese | Language::Japanese |
            Language::Korean | Language::Arabic | Language::Hebrew |
            Language::Other => {}
        }
    }

    /// Language detection should be deterministic.
    #[test]
    fn language_detection_deterministic(
        text in ".{0,500}",
    ) {
        let lang1 = detect_language(&text);
        let lang2 = detect_language(&text);

        // Should produce same result
        prop_assert_eq!(lang1, lang2,
            "Language detection not deterministic for '{}': {:?} != {:?}",
            text, lang1, lang2);
    }

    /// Language detection should handle pure ASCII.
    #[test]
    fn language_detection_ascii(
        text in "[ -~]{1,500}",  // ASCII only
    ) {
        let lang = detect_language(&text);

        // ASCII text should default to English (or detect based on special chars)
        match lang {
            Language::English | Language::German | Language::French |
            Language::Spanish | Language::Italian | Language::Portuguese |
            Language::Russian | Language::Chinese | Language::Japanese |
            Language::Korean | Language::Arabic | Language::Hebrew |
            Language::Other => {} // All are valid for ASCII
        }
    }

    /// Language detection should handle CJK characters.
    #[test]
    fn language_detection_cjk(
        text in "[\\u{4e00}-\\u{9fff}\\u{3040}-\\u{30ff}\\u{ac00}-\\u{d7af}]{5,100}",
    ) {
        // Filter to ensure we have enough alphabetic CJK characters
        let alphabetic_count = text.chars().filter(|c| c.is_alphabetic() && (
            ('\u{4e00}'..='\u{9fff}').contains(c) ||
            ('\u{3040}'..='\u{30ff}').contains(c) ||
            ('\u{ac00}'..='\u{d7af}').contains(c)
        )).count();

        if alphabetic_count >= 3 {
            let lang = detect_language(&text);
            // Should detect CJK language when we have enough alphabetic chars
            prop_assert!(matches!(lang, Language::Chinese | Language::Japanese | Language::Korean),
                "CJK text should detect CJK language, got {:?}", lang);
        }
    }

    /// Language detection should handle Arabic/Hebrew (RTL).
    #[test]
    fn language_detection_rtl(
        text in "[\\u{0600}-\\u{06ff}\\u{0590}-\\u{05ff}]{5,100}",
    ) {
        // Filter to ensure we have enough alphabetic RTL characters
        let alphabetic_count = text.chars().filter(|c| c.is_alphabetic() && (
            ('\u{0600}'..='\u{06ff}').contains(c) ||
            ('\u{0590}'..='\u{05ff}').contains(c)
        )).count();

        if alphabetic_count >= 3 {
            let lang = detect_language(&text);
            // Should detect RTL language when we have enough alphabetic chars
            prop_assert!(matches!(lang, Language::Arabic | Language::Hebrew),
                "RTL text should detect RTL language, got {:?}", lang);
        }
    }

    /// Language detection should handle Cyrillic.
    #[test]
    fn language_detection_cyrillic(
        text in "[\\u{0400}-\\u{04ff}]{5,100}",
    ) {
        let lang = detect_language(&text);

        // Should detect Russian (heuristic may need multiple chars)
        prop_assert_eq!(lang, Language::Russian,
            "Cyrillic text should detect Russian, got {:?}", lang);
    }

    /// Language detection should handle mixed scripts.
    #[test]
    fn language_detection_mixed_scripts(
        ascii_part in "[a-zA-Z ]{0,50}",
        cjk_part in "[\\u{4e00}-\\u{9fff}]{0,50}",
    ) {
        let text = format!("{}{}", ascii_part, cjk_part);
        if !text.is_empty() {
            let lang = detect_language(&text);

            // Should return a valid language (may prefer one script over another)
            match lang {
                Language::English | Language::German | Language::French |
                Language::Spanish | Language::Italian | Language::Portuguese |
                Language::Russian | Language::Chinese | Language::Japanese |
                Language::Korean | Language::Arabic | Language::Hebrew |
                Language::Other => {}
            }
        }
    }
}
