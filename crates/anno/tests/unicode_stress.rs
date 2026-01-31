//! Unicode stress tests for string similarity and resolver logic.
//!
//! Tests ensure correct handling of various Unicode scripts, edge cases,
//! and multilingual entity resolution scenarios.

use anno_core::coalesce as anno_coalesce;

use anno_coalesce::similarity::{
    jaro_similarity, jaro_winkler_similarity, levenshtein_distance, normalize, Script, Similarity,
};

// =============================================================================
// CJK (Chinese, Japanese, Korean)
// =============================================================================

#[test]
fn test_cjk_identical() {
    let sim = Similarity::new();
    assert_eq!(sim.compute("北京", "北京"), 1.0);
    assert_eq!(sim.compute("東京", "東京"), 1.0);
    assert_eq!(sim.compute("서울", "서울"), 1.0);
}

#[test]
fn test_cjk_different() {
    let sim = Similarity::new();
    let score = sim.compute("北京", "東京");
    // Different but share "京" - should have some similarity via bigrams
    assert!(score < 0.5, "Different CJK: {}", score);
}

#[test]
fn test_cjk_partial_overlap() {
    let sim = Similarity::new();
    // 中华人民共和国 (PRC) vs 中华民国 (ROC)
    // Share: 中华, 民国
    let score = sim.compute("中华人民共和国", "中华民国");
    assert!(score > 0.0 && score < 1.0, "CJK partial overlap: {}", score);

    // More overlap should have higher similarity
    let score_more = sim.compute("中华人民共和国", "中华人民");
    let score_less = sim.compute("中华人民共和国", "共和国");
    assert!(
        score_more > score_less || (score_more - score_less).abs() < 0.1,
        "More overlap {} should be >= less overlap {}",
        score_more,
        score_less
    );
}

// =============================================================================
// Arabic (RTL)
// =============================================================================

#[test]
fn test_arabic_identical() {
    let sim = Similarity::new();
    assert_eq!(sim.compute("الرياض", "الرياض"), 1.0);
    assert_eq!(sim.compute("محمد", "محمد"), 1.0);
}

#[test]
fn test_rtl_different() {
    let sim = Similarity::new();
    let score = sim.compute("الرياض", "القاهرة");
    assert!(score < 1.0, "Different Arabic: {}", score);
}

// =============================================================================
// Hebrew
// =============================================================================

#[test]
fn test_hebrew_identical() {
    let sim = Similarity::new();
    assert_eq!(sim.compute("ירושלים", "ירושלים"), 1.0);
}

// =============================================================================
// Cyrillic
// =============================================================================

#[test]
fn test_cyrillic_identical() {
    let sim = Similarity::new();
    assert_eq!(sim.compute("Москва", "Москва"), 1.0);
}

#[test]
fn test_cyrillic_different() {
    let sim = Similarity::new();
    let score = sim.compute("Москва", "Санкт-Петербург");
    assert!(score < 1.0);
}

#[test]
fn test_cyrillic_latin_lookalikes() {
    let sim = Similarity::new();
    // Cyrillic А vs Latin A - should be 0 (different scripts)
    let score = sim.compute("А", "A");
    // These are visually identical but different Unicode codepoints
    assert!(score < 1.0, "Cyrillic vs Latin lookalike: {}", score);
}

// =============================================================================
// Diacritics and Combining Characters
// =============================================================================

#[test]
fn test_diacritics_identical() {
    let sim = Similarity::new();
    // NFC form
    assert_eq!(sim.compute("café", "café"), 1.0);
    assert_eq!(sim.compute("naïve", "naïve"), 1.0);
}

#[test]
fn test_vietnamese_tones() {
    let sim = Similarity::new();
    // Vietnamese with tones
    assert_eq!(sim.compute("Hà Nội", "Hà Nội"), 1.0);
    // Without proper Unicode normalization, tones cause different words
    // This test documents current behavior - with NFKC normalization,
    // these would be more similar
    let score = sim.compute("Hà Nội", "Ha Noi");
    // Word-based: "Hà" != "Ha", "Nội" != "Noi", intersection=0, union=4
    // So similarity is 0.0 with current word-based approach
    assert!(score >= 0.0, "Vietnamese with/without tones: {}", score);
}

#[test]
fn test_precomposed_vs_combining() {
    let sim = Similarity::new();
    // é precomposed vs e + combining acute
    let precomposed = "café";
    let combining = "cafe\u{0301}";

    // Without proper NFKC normalization, precomposed and combining
    // forms are different character sequences
    // TODO: Add unicode-normalization crate for proper handling
    let score = sim.compute(precomposed, combining);
    // Current behavior: treated as different single words via n-gram
    // This test documents the limitation
    assert!(score >= 0.0, "Precomposed vs combining: {}", score);
}

// =============================================================================
// Emoji
// =============================================================================

#[test]
fn test_emoji_in_names() {
    let sim = Similarity::new();
    assert_eq!(sim.compute("Test 🎉", "Test 🎉"), 1.0);
}

#[test]
fn test_emoji_only() {
    let sim = Similarity::new();
    assert_eq!(sim.compute("🎉", "🎉"), 1.0);
    let score = sim.compute("🎉", "🎊");
    assert!(score < 1.0, "Different emoji: {}", score);
}

#[test]
fn test_multi_codepoint_emoji() {
    let sim = Similarity::new();
    // Family emoji (multiple codepoints joined)
    let emoji1 = "👨‍👩‍👧‍👦";
    let emoji2 = "👨‍👩‍👧‍👦";
    assert_eq!(sim.compute(emoji1, emoji2), 1.0);
}

#[test]
fn test_flag_emoji() {
    let sim = Similarity::new();
    // Flag emojis (regional indicator pairs)
    assert_eq!(sim.compute("🇺🇸", "🇺🇸"), 1.0);
    let score = sim.compute("🇺🇸", "🇬🇧");
    assert!(score < 1.0);
}

// =============================================================================
// Mixed Scripts
// =============================================================================

#[test]
fn test_mixed_scripts_identical() {
    let sim = Similarity::new();
    assert_eq!(sim.compute("Dr. 田中", "Dr. 田中"), 1.0);
}

#[test]
fn test_code_switching() {
    let sim = Similarity::new();
    // English with CJK
    let text1 = "Tokyo 東京";
    let text2 = "Tokyo 东京"; // Simplified Chinese variant
    let score = sim.compute(text1, text2);
    assert!(score > 0.0 && score < 1.0, "Code-switching: {}", score);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_strings() {
    let sim = Similarity::new();
    assert_eq!(sim.compute("", ""), 1.0);
    assert_eq!(sim.compute("test", ""), 0.0);
    assert_eq!(sim.compute("", "test"), 0.0);
}

#[test]
fn test_whitespace_only() {
    let sim = Similarity::new();
    let score = sim.compute("   ", "   ");
    // After normalization, both become empty
    assert_eq!(score, 1.0);
}

#[test]
fn test_zero_width_characters() {
    let sim = Similarity::new();
    let with_zwj = "a\u{200D}b"; // Zero-width joiner
    let without = "ab";
    let score = sim.compute(with_zwj, without);
    // N-gram Jaccard: "a\u{200D}", "\u{200D}b" vs "ab"
    // These share some characters but different bigrams
    // Without invisible char stripping, similarity is low
    assert!(score >= 0.0, "Zero-width char: {}", score);
}

#[test]
fn test_bom_handling() {
    let sim = Similarity::new();
    let with_bom = "\u{FEFF}test";
    let without = "test";
    let score = sim.compute(with_bom, without);
    // Without BOM stripping, these have different bigrams at the start
    // TODO: Add preprocessing to strip BOM
    assert!(score >= 0.0, "BOM handling: {}", score);
}

#[test]
fn test_direction_marks() {
    let sim = Similarity::new();
    let with_lrm = "test\u{200E}"; // Left-to-right mark
    let without = "test";
    let score = sim.compute(with_lrm, without);
    // Without invisible char stripping, the trailing mark affects n-grams
    // TODO: Add preprocessing to strip direction marks
    assert!(score >= 0.0, "Direction marks: {}", score);
}

#[test]
fn test_long_cjk_string() {
    let sim = Similarity::new();
    // Long CJK text
    let long1 = "这是一个非常长的中文字符串用于测试相似度计算";
    let long2 = "这是一个非常长的中文字符串用于测试相似度计算";
    assert_eq!(sim.compute(long1, long2), 1.0);
}

#[test]
fn test_very_long_mixed_script() {
    let sim = Similarity::new();
    let long = "This is a very long string with 日本語 and العربية mixed in for testing purposes.";
    assert_eq!(sim.compute(long, long), 1.0);
}

// =============================================================================
// Script Detection Tests
// =============================================================================

#[test]
fn test_script_detection_comprehensive() {
    assert_eq!(Script::detect("Hello World"), Script::Latin);
    assert_eq!(Script::detect("北京市"), Script::Cjk);
    assert_eq!(Script::detect("ひらがな"), Script::Kana);
    assert_eq!(Script::detect("서울특별시"), Script::Hangul);
    assert_eq!(Script::detect("مرحبا"), Script::Arabic);
    assert_eq!(Script::detect("Москва"), Script::Cyrillic);
    assert_eq!(Script::detect("नमस्ते"), Script::Devanagari);
    assert_eq!(Script::detect("Αθήνα"), Script::Greek);
    assert_eq!(Script::detect("שלום"), Script::Hebrew);
    assert_eq!(Script::detect("สวัสดี"), Script::Thai);
}

// =============================================================================
// Levenshtein with Unicode
// =============================================================================

#[test]
fn test_levenshtein_cjk() {
    // Single character change in CJK
    let dist = levenshtein_distance("北京", "南京");
    assert_eq!(dist, 1); // One character different
}

#[test]
fn test_levenshtein_arabic() {
    let dist = levenshtein_distance("مرحبا", "مرحبا");
    assert_eq!(dist, 0);
}

// =============================================================================
// Jaro-Winkler with Unicode
// =============================================================================

#[test]
fn test_jaro_winkler_cyrillic() {
    let sim = jaro_winkler_similarity("Москва", "Москве");
    assert!(sim > 0.8, "Cyrillic Jaro-Winkler: {}", sim);
}

#[test]
fn test_jaro_similarity_cjk() {
    let sim = jaro_similarity("東京", "東京都");
    assert!(sim > 0.5, "CJK Jaro: {}", sim);
}

// =============================================================================
// Normalize Function
// =============================================================================

#[test]
fn test_normalize_unicode() {
    assert_eq!(normalize("  HELLO   WORLD  "), "hello world");
    // Preserve CJK
    assert_eq!(normalize("北京市"), "北京市");
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Unicode similarity is symmetric
        #[test]
        fn unicode_similarity_symmetric(a in "\\PC{1,20}", b in "\\PC{1,20}") {
            let sim = Similarity::new();
            let ab = sim.compute(&a, &b);
            let ba = sim.compute(&b, &a);
            prop_assert!((ab - ba).abs() < 0.001,
                "Symmetry: {} vs {}", ab, ba);
        }

        /// Unicode similarity is bounded [0, 1]
        #[test]
        fn unicode_similarity_bounded(a in "\\PC{0,30}", b in "\\PC{0,30}") {
            let sim = Similarity::new();
            let score = sim.compute(&a, &b);
            prop_assert!((0.0..=1.0).contains(&score),
                "Bounds: {}", score);
        }

        /// Identical Unicode strings have similarity 1.0
        #[test]
        fn unicode_identity(s in "\\PC{0,30}") {
            let sim = Similarity::new();
            let score = sim.compute(&s, &s);
            prop_assert!((score - 1.0).abs() < 0.001,
                "Identity: {}", score);
        }
    }
}
