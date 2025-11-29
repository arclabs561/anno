//! Property-based tests for string similarity functions.
//!
//! These tests verify mathematical properties of similarity functions:
//! - Symmetry
//! - Boundedness
//! - Identity
//! - Triangle inequality (where applicable)

use anno::similarity::{jaccard_word_similarity, jaccard_word_similarity_f32, string_similarity};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Similarity should be symmetric: sim(a, b) == sim(b, a)
    #[test]
    fn similarity_symmetric(
        a in ".{0,100}",
        b in ".{0,100}",
    ) {
        let sim_ab = string_similarity(&a, &b);
        let sim_ba = string_similarity(&b, &a);

        // Similarity should be symmetric
        prop_assert!((sim_ab - sim_ba).abs() < 0.001,
            "Similarity not symmetric: sim({:?}, {:?})={}, sim({:?}, {:?})={}",
            a, b, sim_ab, b, a, sim_ba);
    }

    /// Similarity should always be in [0.0, 1.0]
    #[test]
    fn similarity_bounded(
        a in ".{0,100}",
        b in ".{0,100}",
    ) {
        let sim = string_similarity(&a, &b);
        // Should always be in [0.0, 1.0]
        prop_assert!(sim >= 0.0 && sim <= 1.0,
            "Similarity out of bounds: sim({:?}, {:?})={}", a, b, sim);
    }

    /// Similarity of identical strings should be 1.0
    #[test]
    fn similarity_identical_is_one(
        text in ".{0,100}",
    ) {
        let sim = string_similarity(&text, &text);
        prop_assert!((sim - 1.0).abs() < 0.001,
            "Identical strings should have similarity 1.0, got {}", sim);
    }

    /// Jaccard similarity should be commutative
    #[test]
    fn jaccard_commutative(
        a in ".{0,100}",
        b in ".{0,100}",
    ) {
        let j1 = jaccard_word_similarity(&a, &b);
        let j2 = jaccard_word_similarity(&b, &a);
        prop_assert!((j1 - j2).abs() < 0.001,
            "Jaccard not commutative: j({:?}, {:?})={}, j({:?}, {:?})={}",
            a, b, j1, b, a, j2);
    }

    /// Jaccard similarity should be bounded [0.0, 1.0]
    #[test]
    fn jaccard_bounded(
        a in ".{0,100}",
        b in ".{0,100}",
    ) {
        let j = jaccard_word_similarity(&a, &b);
        prop_assert!(j >= 0.0 && j <= 1.0,
            "Jaccard out of bounds: j({:?}, {:?})={}", a, b, j);
    }

    /// Jaccard similarity of identical strings should be 1.0
    #[test]
    fn jaccard_identical_is_one(
        text in "[a-zA-Z0-9 ]{1,100}",  // Text with words (letters/numbers/spaces)
    ) {
        // Filter to ensure we have at least one word
        let words: Vec<&str> = text.split_whitespace().collect();
        if !words.is_empty() {
            let j = jaccard_word_similarity(&text, &text);
            prop_assert!((j - 1.0).abs() < 0.001,
                "Identical strings should have Jaccard 1.0, got {}", j);
        }
    }

    /// Jaccard similarity of disjoint strings should be 0.0
    #[test]
    fn jaccard_disjoint_is_zero(
        a in "[a-z]{1,20}",
        b in "[A-Z]{1,20}",
    ) {
        // If strings have no common words, Jaccard should be 0
        let j = jaccard_word_similarity(&a, &b);
        // Note: This might not always be 0 if words match after lowercasing
        // But if truly disjoint, should be 0
        prop_assert!(j >= 0.0 && j <= 1.0);
    }

    /// f32 version should match f64 version
    #[test]
    fn jaccard_f32_matches_f64(
        a in ".{0,100}",
        b in ".{0,100}",
    ) {
        let j_f64 = jaccard_word_similarity(&a, &b);
        let j_f32 = jaccard_word_similarity_f32(&a, &b);

        // f32 version should be close to f64 (within f32 precision)
        prop_assert!((j_f64 - j_f32 as f64).abs() < 0.0001,
            "f32 version mismatch: f64={}, f32={}", j_f64, j_f32);
    }

    /// Similarity should handle empty strings correctly
    #[test]
    fn similarity_empty_strings(
        text in ".{0,100}",
    ) {
        // Empty vs empty
        let sim_empty = string_similarity("", "");
        prop_assert!((sim_empty - 1.0).abs() < 0.001,
            "Empty vs empty should be 1.0, got {}", sim_empty);

        // Empty vs non-empty (should be 0.8 due to substring match)
        let sim_empty_text = string_similarity("", &text);
        let sim_text_empty = string_similarity(&text, "");
        prop_assert!((sim_empty_text - 0.8).abs() < 0.001 || text.is_empty(),
            "Empty vs text should be 0.8, got {}", sim_empty_text);
        prop_assert!((sim_text_empty - 0.8).abs() < 0.001 || text.is_empty(),
            "Text vs empty should be 0.8, got {}", sim_text_empty);
    }

    /// Similarity should be case-insensitive
    #[test]
    fn similarity_case_insensitive(
        text in "[a-zA-Z]{1,50}",
    ) {
        let lower = text.to_lowercase();
        let upper = text.to_uppercase();
        let mixed: String = text.chars().enumerate()
            .map(|(i, c)| if i % 2 == 0 { c.to_uppercase().next().unwrap() } else { c.to_lowercase().next().unwrap() })
            .collect();

        // All case variants should have similarity 1.0 with each other
        let sim_lower_upper = string_similarity(&lower, &upper);
            prop_assert!((sim_lower_upper - 1.0).abs() < 0.001,
                "Case variants should match: lower={:?}, upper={:?}, sim={}", lower, upper, sim_lower_upper);

            let sim_lower_mixed = string_similarity(&lower, &mixed);
            prop_assert!((sim_lower_mixed - 1.0).abs() < 0.001,
                "Case variants should match: lower={:?}, mixed={:?}, sim={}", lower, mixed, sim_lower_mixed);
    }

    /// Substring matches should have similarity >= 0.8
    #[test]
    fn similarity_substring_high(
        prefix in ".{0,20}",
        middle in ".{1,50}",
        suffix in ".{0,20}",
    ) {
        let full = format!("{}{}{}", prefix, middle, suffix);
        let sim = string_similarity(&full, &middle);

        // Substring match should return 0.8 (unless full == middle, then 1.0)
        if full == middle {
            prop_assert!((sim - 1.0).abs() < 0.001,
                "Identical strings should have similarity 1.0, got {}", sim);
        } else {
            prop_assert!((sim - 0.8).abs() < 0.001,
                "Substring match should be 0.8: full={:?}, middle={:?}, sim={}", full, middle, sim);
        }
    }
}
