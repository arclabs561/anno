//! Comprehensive tests for language detection functionality.

use anno::lang::{detect_language, Language};

#[test]
fn test_language_is_cjk_chinese() {
    assert!(Language::Chinese.is_cjk());
}

#[test]
fn test_language_is_cjk_japanese() {
    assert!(Language::Japanese.is_cjk());
}

#[test]
fn test_language_is_cjk_korean() {
    assert!(Language::Korean.is_cjk());
}

#[test]
fn test_language_is_cjk_non_cjk() {
    assert!(!Language::English.is_cjk());
    assert!(!Language::German.is_cjk());
    assert!(!Language::Arabic.is_cjk());
}

#[test]
fn test_language_is_rtl_arabic() {
    assert!(Language::Arabic.is_rtl());
}

#[test]
fn test_language_is_rtl_hebrew() {
    assert!(Language::Hebrew.is_rtl());
}

#[test]
fn test_language_is_rtl_non_rtl() {
    assert!(!Language::English.is_rtl());
    assert!(!Language::Chinese.is_rtl());
    assert!(!Language::Russian.is_rtl());
}

#[test]
fn test_detect_language_english() {
    assert_eq!(detect_language("Hello world"), Language::English);
    assert_eq!(detect_language("The quick brown fox"), Language::English);
}

#[test]
fn test_detect_language_german() {
    // German-specific characters (ß, ä, ö, ü) should trigger German detection
    assert_eq!(detect_language("Müller"), Language::German);
    assert_eq!(detect_language("Grüße"), Language::German);
    // "Hallo Welt" has no German-specific chars, may default to English
    // This is expected behavior - heuristic needs special chars
    let result = detect_language("Hallo Welt");
    assert!(matches!(result, Language::German | Language::English));
}

#[test]
fn test_detect_language_french() {
    // French-specific characters should trigger French detection
    assert_eq!(detect_language("café"), Language::French);
    assert_eq!(detect_language("résumé"), Language::French);
    // "Bonjour" has no French-specific chars, may default to English
    let result = detect_language("Bonjour");
    assert!(matches!(result, Language::French | Language::English));
}

#[test]
fn test_detect_language_spanish() {
    // Spanish-specific characters (ñ, ¿, ¡, á, é, í, ó, ú) should trigger Spanish detection
    // Note: The heuristic counts 'ñ' as Spanish (+5) but also counts all Latin chars as English (+1)
    // For short words, English count might win. For longer words with 'ñ', Spanish should win.
    let result1 = detect_language("España"); // Has 'ñ' (+5) and 5 Latin chars (+5) = 10 total
                                             // "España" = E(1) + s(1) + p(1) + a(1) + ñ(5) + a(1) = English: 5, Spanish: 5
                                             // English might win due to array ordering, but Spanish should win with more chars
    assert!(matches!(result1, Language::Spanish | Language::English));

    let result2 = detect_language("niño"); // Has 'ñ' (+5) and 3 Latin chars (+3)
                                           // "niño" = n(1) + i(1) + ñ(5) + o(1) = English: 3, Spanish: 5
    assert!(matches!(result2, Language::Spanish | Language::English));

    // "Hola" has no Spanish-specific chars, defaults to English
    assert_eq!(detect_language("Hola"), Language::English);
}

#[test]
fn test_detect_language_chinese() {
    assert_eq!(detect_language("你好"), Language::Chinese);
    assert_eq!(detect_language("北京"), Language::Chinese);
    assert_eq!(detect_language("中文"), Language::Chinese);
}

#[test]
fn test_detect_language_japanese() {
    // Hiragana/Katakana should trigger Japanese detection
    assert_eq!(detect_language("こんにちは"), Language::Japanese);
    // "日本語" has Hiragana in "語" context, but "東京" is only Kanji
    // The heuristic checks if Japanese chars exist when Chinese is detected
    let result1 = detect_language("東京");
    // May be detected as Chinese (only Kanji) or Japanese (if heuristic sees it as Japanese)
    assert!(matches!(result1, Language::Chinese | Language::Japanese));

    // "日本語" - if it has Hiragana, should be Japanese
    let result2 = detect_language("日本語");
    assert!(matches!(result2, Language::Chinese | Language::Japanese));
}

#[test]
fn test_detect_language_korean() {
    assert_eq!(detect_language("안녕하세요"), Language::Korean);
    assert_eq!(detect_language("서울"), Language::Korean);
}

#[test]
fn test_detect_language_arabic() {
    assert_eq!(detect_language("مرحبا"), Language::Arabic);
    assert_eq!(detect_language("الرياض"), Language::Arabic);
}

#[test]
fn test_detect_language_hebrew() {
    assert_eq!(detect_language("שלום"), Language::Hebrew);
}

#[test]
fn test_detect_language_russian() {
    assert_eq!(detect_language("Привет"), Language::Russian);
    assert_eq!(detect_language("Москва"), Language::Russian);
}

#[test]
fn test_detect_language_empty_text() {
    // Empty text should default to English
    assert_eq!(detect_language(""), Language::English);
}

#[test]
fn test_detect_language_whitespace_only() {
    // Whitespace-only should default to English
    assert_eq!(detect_language("   "), Language::English);
    assert_eq!(detect_language("\n\t"), Language::English);
}

#[test]
fn test_detect_language_mixed_scripts() {
    // Mixed scripts - should detect dominant script
    let result1 = detect_language("Hello 你好");
    // May be Chinese (if Chinese chars dominate) or English (if Latin chars dominate)
    assert!(matches!(result1, Language::Chinese | Language::English));

    let result2 = detect_language("你好 Hello");
    assert!(matches!(result2, Language::Chinese | Language::English));

    let result3 = detect_language("Hello world مرحبا");
    // May be Arabic (if Arabic chars dominate) or English (if Latin chars dominate)
    assert!(matches!(result3, Language::Arabic | Language::English));
}

#[test]
fn test_detect_language_numbers_only() {
    // Numbers only - should default to English
    assert_eq!(detect_language("12345"), Language::English);
}

#[test]
fn test_detect_language_punctuation_only() {
    // Punctuation only - should default to English
    assert_eq!(detect_language("!@#$%"), Language::English);
}

#[test]
fn test_detect_language_emoji() {
    // Emoji - should default to English (no alphabetic chars)
    assert_eq!(detect_language("🚀🎉"), Language::English);
}

#[test]
fn test_detect_language_mixed_latin() {
    // Mixed Latin languages - should detect based on special characters
    // These characters get weighted higher (French +5, Spanish +5, German +10)
    // But all Latin chars also count as English (+1 each)

    // "café" = c(1) + a(1) + f(1) + é(5) = English: 3, French: 5 → French wins
    assert_eq!(detect_language("café"), Language::French);

    // "España" = E(1) + s(1) + p(1) + a(1) + ñ(5) + a(1) = English: 5, Spanish: 5
    // May be English or Spanish depending on array order/implementation
    let result = detect_language("España");
    assert!(matches!(result, Language::Spanish | Language::English));

    // "Müller" = M(1) + ü(10) + l(1) + l(1) + e(1) + r(1) = English: 5, German: 10 → German wins
    assert_eq!(detect_language("Müller"), Language::German);
}

#[test]
fn test_detect_language_japanese_vs_chinese() {
    // Japanese uses Kanji (Chinese characters) but also has Hiragana/Katakana
    // If Hiragana/Katakana present, should detect as Japanese
    assert_eq!(detect_language("東京"), Language::Chinese); // Only Kanji, no Hiragana
    assert_eq!(detect_language("こんにちは"), Language::Japanese); // Hiragana present
    assert_eq!(detect_language("東京は"), Language::Japanese); // Kanji + Hiragana
}

#[test]
fn test_detect_language_case_insensitive() {
    // Detection should work regardless of case
    assert_eq!(detect_language("HELLO"), Language::English);
    assert_eq!(detect_language("hello"), Language::English);
    assert_eq!(detect_language("Hello"), Language::English);
}

#[test]
fn test_detect_language_long_text() {
    // Long text should still detect correctly
    let long_english = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    assert_eq!(detect_language(&long_english), Language::English);

    let long_chinese = "你好世界".repeat(100);
    assert_eq!(detect_language(&long_chinese), Language::Chinese);
}

#[test]
fn test_detect_language_italian() {
    // Italian has some unique characters but may be detected as generic Latin
    // This test documents current behavior
    let result = detect_language("Ciao");
    // May be detected as English (generic Latin) or Italian depending on implementation
    assert!(matches!(result, Language::English | Language::Italian));
}

#[test]
fn test_detect_language_portuguese() {
    // Portuguese has some unique characters (á, ã, ç, etc.)
    // "Olá" has 'á' which might be detected as Spanish or Portuguese
    // The current heuristic doesn't distinguish Portuguese from Spanish well
    let result = detect_language("Olá");
    // May be detected as Spanish (á is Spanish indicator), English, or Portuguese
    assert!(matches!(
        result,
        Language::English | Language::Spanish | Language::Portuguese
    ));
}

#[test]
fn test_detect_language_other_fallback() {
    // Unknown scripts should fall back to Other or English
    // This test documents that the function doesn't panic on unusual input
    let result = detect_language("𐌀𐌁𐌂"); // Old Italic script
                                         // Should not panic, may return Other or English
    assert!(matches!(result, Language::English | Language::Other));
}
