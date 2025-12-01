//! Multilingual NER tests - exposing limitations of pattern and statistical approaches.
//!
//! These tests demonstrate where RegexNER and HeuristicNER fail on non-English text,
//! serving as a baseline for future ML-based improvements.
//!
//! ## Key Findings from Research
//!
//! 1. **RegexNER (regex)**: Works well across languages for structured entities
//!    (dates, emails, URLs, money with $ € £ ¥) but fails for:
//!    - Language-specific date formats (Japanese 年月日, Arabic numerals)
//!    - Currency names in other languages
//!
//! 2. **HeuristicNER (heuristics)**: Heavily English-biased because:
//!    - Relies on capitalization (fails for Chinese, Japanese, Arabic, Hebrew)
//!    - Uses English context words ("Mr.", "Inc.", "in New York")
//!    - English first names gazetteer
//!
//! ## Language Categories by Difficulty
//!
//! | Category | Languages | Capitalization | Challenge |
//! |----------|-----------|----------------|-----------|
//! | Easy | German, Spanish, French | Yes (like English) | Different word lists |
//! | Medium | Russian, Greek | Yes (different alphabet) | Script differences |
//! | Hard | Chinese, Japanese | No | No capitalization signal |
//! | Very Hard | Arabic, Hebrew | No + RTL | Script direction + no caps |
//!
//! ## References
//!
//! - WikiANN (PAN-X): 282 languages, PER/LOC/ORG
//! - MultiCoNER: 12 languages, 33 fine-grained types
//! - CoNLL 2002/2003: Spanish, Dutch, German, English

use anno::{Entity, EntityType, HeuristicNER, Model, RegexNER, StackedNER};

// =============================================================================
// Test Helpers
// =============================================================================

fn pattern() -> RegexNER {
    RegexNER::new()
}

fn stats() -> HeuristicNER {
    HeuristicNER::new()
}

fn stacked() -> StackedNER {
    StackedNER::default()
}

fn has_type(entities: &[Entity], ty: EntityType) -> bool {
    entities.iter().any(|e| e.entity_type == ty)
}

fn find_text<'a>(entities: &'a [Entity], text: &str) -> Option<&'a Entity> {
    entities.iter().find(|e| e.text == text)
}

fn entity_texts(entities: &[Entity]) -> Vec<&str> {
    entities.iter().map(|e| e.text.as_str()).collect()
}

// =============================================================================
// PATTERN NER: Language-Agnostic Structured Entities
// =============================================================================

mod pattern_multilingual {
    use super::*;

    // -------------------------------------------------------------------------
    // These SHOULD work across all languages (format-based)
    // -------------------------------------------------------------------------

    #[test]
    fn iso_dates_universal() {
        // ISO 8601 is language-agnostic
        let cases = [
            "会议日期 2024-01-15", // Chinese
            "Datum: 2024-01-15",   // German
            "Fecha: 2024-01-15",   // Spanish
            "التاريخ 2024-01-15",  // Arabic
            "日付: 2024-01-15",    // Japanese
            "Дата: 2024-01-15",    // Russian
        ];

        for text in cases {
            let e = pattern().extract_entities(text, None).unwrap();
            assert!(
                has_type(&e, EntityType::Date),
                "Should find ISO date in: {}",
                text
            );
        }
    }

    #[test]
    fn emails_universal() {
        // Email format is truly universal
        let cases = [
            "联系: test@example.com",    // Chinese
            "Kontakt: test@example.com", // German
            "Contato: test@example.com", // Portuguese
            "連絡先: test@example.com",  // Japanese
            "Контакт: test@example.com", // Russian
            "الاتصال: test@example.com",  // Arabic
        ];

        for text in cases {
            let e = pattern().extract_entities(text, None).unwrap();
            assert!(
                find_text(&e, "test@example.com").is_some(),
                "Should find email in: {}",
                text
            );
        }
    }

    #[test]
    fn urls_universal() {
        let cases = [
            "访问 https://example.com",
            "Besuchen Sie https://example.com",
            "Посетите https://example.com",
            "訪問 https://example.com",
        ];

        for text in cases {
            let e = pattern().extract_entities(text, None).unwrap();
            assert!(
                has_type(&e, EntityType::Url),
                "Should find URL in: {}",
                text
            );
        }
    }

    #[test]
    fn money_with_symbols_universal() {
        // Currency symbols work across contexts
        let cases = [
            ("价格 $100", "$100"),
            ("Preis: €500", "€500"),
            ("Prix: £200", "£200"),
            ("価格: ¥10000", "¥10000"),
        ];

        for (text, expected) in cases {
            let e = pattern().extract_entities(text, None).unwrap();
            assert!(
                find_text(&e, expected).is_some(),
                "Should find {} in: {}",
                expected,
                text
            );
        }
    }

    // -------------------------------------------------------------------------
    // These FAIL - language-specific date/money formats
    // -------------------------------------------------------------------------

    #[test]
    fn german_date_format() {
        // German uses DD.MM.YYYY (supported)
        let text = "Termin am 15.01.2024";
        let e = pattern().extract_entities(text, None).unwrap();
        // This SHOULD work - we have DATE_EU pattern
        assert!(
            has_type(&e, EntityType::Date),
            "German date format: {:?}",
            e
        );
    }

    #[test]
    fn japanese_date_format_supported() {
        // Japanese 年月日 format - NOW SUPPORTED
        let text = "会議は2024年1月15日です";
        let e = pattern().extract_entities(text, None).unwrap();
        assert!(
            has_type(&e, EntityType::Date),
            "Japanese date format (年月日) should be supported: {:?}",
            e
        );
        let date = e
            .iter()
            .find(|e| e.entity_type == EntityType::Date)
            .unwrap();
        assert_eq!(date.text, "2024年1月15日");
    }

    #[test]
    fn french_date_written() {
        // "15 janvier 2024" - SHOULD work with DATE_WRITTEN_EU pattern
        let text = "Réunion le 15 January 2024"; // Using English month name
        let e = pattern().extract_entities(text, None).unwrap();
        assert!(
            has_type(&e, EntityType::Date),
            "Date with English month: {:?}",
            e
        );
    }

    #[test]
    fn french_month_names_supported() {
        // French month names - NOW SUPPORTED
        let cases = [
            ("Réunion le 15 janvier 2024", "15 janvier 2024"),
            ("Le 1er février", "1er février"),
            ("Date: 25 décembre 2023", "25 décembre 2023"),
        ];
        for (text, expected) in cases {
            let e = pattern().extract_entities(text, None).unwrap();
            assert!(
                has_type(&e, EntityType::Date),
                "French month should be supported in: {}",
                text
            );
            let date = e
                .iter()
                .find(|e| e.entity_type == EntityType::Date)
                .unwrap();
            assert_eq!(date.text, expected, "Wrong text for: {}", text);
        }
    }

    #[test]
    fn german_month_names_supported() {
        // German month names - NOW SUPPORTED
        let cases = [
            ("Termin am 15. Januar 2024", "15. Januar 2024"),
            ("Am 3 März beginnt", "3 März"),
            ("Der 25 Dezember ist", "25 Dezember"),
        ];
        for (text, expected) in cases {
            let e = pattern().extract_entities(text, None).unwrap();
            assert!(
                has_type(&e, EntityType::Date),
                "German month should be supported in: {}",
                text
            );
            let date = e
                .iter()
                .find(|e| e.entity_type == EntityType::Date)
                .unwrap();
            assert_eq!(date.text, expected, "Wrong text for: {}", text);
        }
    }

    #[test]
    fn spanish_month_names_supported() {
        let cases = [
            ("Fecha: 15 de enero de 2024", "15 de enero de 2024"),
            ("El 5 marzo", "5 marzo"),
        ];
        for (text, expected) in cases {
            let e = pattern().extract_entities(text, None).unwrap();
            assert!(
                has_type(&e, EntityType::Date),
                "Spanish month should be supported in: {}",
                text
            );
            let date = e
                .iter()
                .find(|e| e.entity_type == EntityType::Date)
                .unwrap();
            assert_eq!(date.text, expected, "Wrong text for: {}", text);
        }
    }

    #[test]
    fn italian_month_names_supported() {
        let text = "Data: 15 gennaio 2024";
        let e = pattern().extract_entities(text, None).unwrap();
        assert!(has_type(&e, EntityType::Date), "Italian month: {:?}", e);
    }

    #[test]
    fn portuguese_month_names_supported() {
        let text = "Data: 15 de janeiro de 2024";
        let e = pattern().extract_entities(text, None).unwrap();
        assert!(has_type(&e, EntityType::Date), "Portuguese month: {:?}", e);
    }

    #[test]
    fn dutch_month_names_supported() {
        let text = "Datum: 15 januari 2024";
        let e = pattern().extract_entities(text, None).unwrap();
        assert!(has_type(&e, EntityType::Date), "Dutch month: {:?}", e);
    }

    #[test]
    fn russian_month_names_supported() {
        let text = "Дата: 15 января 2024";
        let e = pattern().extract_entities(text, None).unwrap();
        assert!(has_type(&e, EntityType::Date), "Russian month: {:?}", e);
    }

    #[test]
    fn korean_date_format_supported() {
        let text = "날짜: 2024년 1월 15일";
        let e = pattern().extract_entities(text, None).unwrap();
        assert!(has_type(&e, EntityType::Date), "Korean date: {:?}", e);
    }

    #[test]
    fn arabic_indic_numerals_supported() {
        // Eastern Arabic numerals (٠١٢٣٤٥٦٧٨٩)
        // Rust regex has Unicode support by default - \d matches all Unicode digits!
        let text = "السعر: $١٢٣"; // $123 in Arabic-Indic numerals
        let e = pattern().extract_entities(text, None).unwrap();
        // Positive finding: Rust regex \d matches Unicode digits
        assert!(
            has_type(&e, EntityType::Money),
            "Arabic-Indic numerals ARE supported (Rust regex \\d is Unicode-aware): {:?}",
            e
        );
        // Verify the text was captured correctly
        let money = e
            .iter()
            .find(|e| e.entity_type == EntityType::Money)
            .unwrap();
        assert_eq!(money.text, "$١٢٣");
    }

    #[test]
    fn chinese_currency_words_not_supported() {
        // Chinese currency expression without symbol
        let text = "价格是一百美元"; // "The price is 100 dollars"
        let e = pattern().extract_entities(text, None).unwrap();
        // Documents limitation: no pattern for Chinese currency words
        assert!(
            !has_type(&e, EntityType::Money),
            "Chinese currency words NOT supported (would need NLP): {:?}",
            e
        );
    }
}

// =============================================================================
// STATISTICAL NER: English-Centric Heuristics
// =============================================================================

mod statistical_multilingual {
    use super::*;

    // -------------------------------------------------------------------------
    // Languages WITH capitalization (should partially work)
    // -------------------------------------------------------------------------

    #[test]
    fn german_capitalized_entities() {
        // German capitalizes ALL nouns, not just proper nouns
        // This creates massive false positive problem
        let text = "Der Mann arbeitet bei der Firma in der Stadt.";
        // "The man works at the company in the city."
        // Mann, Firma, Stadt are all capitalized but NOT entities

        let e = stats().extract_entities(text, None).unwrap();

        // Statistical NER will likely find false positives
        // This test documents the problem
        println!("German common nouns (capitalized): {:?}", entity_texts(&e));
        // Don't assert - just document the behavior
    }

    #[test]
    fn german_real_entities() {
        let text = "Angela Merkel arbeitet in Berlin.";
        let e = stats().extract_entities(text, None).unwrap();

        // Should find Angela Merkel (capitalized, looks like name)
        // Should find Berlin (capitalized, after "in")
        println!("German entities: {:?}", entity_texts(&e));

        // May or may not work - context words are English
    }

    #[test]
    fn spanish_entities() {
        // Spanish capitalizes proper nouns like English
        let text = "Pablo García trabaja en Madrid para Telefónica.";
        let e = stats().extract_entities(text, None).unwrap();

        println!("Spanish entities: {:?}", entity_texts(&e));
        // May find "Pablo García" (capitalized sequence)
        // May find "Madrid" (capitalized after something)
        // May find "Telefónica" (capitalized)
    }

    #[test]
    fn french_entities() {
        let text = "Emmanuel Macron habite à Paris.";
        let e = stats().extract_entities(text, None).unwrap();

        println!("French entities: {:?}", entity_texts(&e));
        // Similar to Spanish - capitalization helps
    }

    // -------------------------------------------------------------------------
    // Languages WITHOUT capitalization (will fail badly)
    // -------------------------------------------------------------------------

    #[test]
    fn chinese_no_capitalization() {
        // Chinese has no capitalization, but heuristic NER uses known entity lists
        let text = "李明在北京的阿里巴巴公司工作";
        // "Li Ming works at Alibaba Company in Beijing"
        // 李明 = Person, 北京 = Location, 阿里巴巴 = Organization

        let e = stats().extract_entities(text, None).unwrap();

        // HeuristicNER now matches known entities (阿里巴巴, 北京) from KNOWN_ORGS/KNOWN_LOCS
        // So it will find some entities even without capitalization
        assert!(!e.is_empty(), "Chinese: {:?}", e);
        // Should find at least 北京 (Beijing) and 阿里巴巴 (Alibaba) from known lists
        assert!(
            e.iter().any(|ent| ent.text.contains("北京")),
            "Should find 北京"
        );
        assert!(
            e.iter().any(|ent| ent.text.contains("阿里巴巴")),
            "Should find 阿里巴巴"
        );
    }

    #[test]
    fn japanese_no_capitalization() {
        // Japanese also lacks capitalization, but heuristic NER uses known entity lists
        let text = "田中太郎は東京のソニーで働いています";
        // "Taro Tanaka works at Sony in Tokyo"

        let e = stats().extract_entities(text, None).unwrap();

        // HeuristicNER now matches known entities (東京, ソニー) from KNOWN_LOCS/KNOWN_ORGS
        // So it will find some entities even without capitalization
        assert!(!e.is_empty(), "Japanese: {:?}", e);
        // Should find at least 東京 (Tokyo) and ソニー (Sony) from known lists
        assert!(
            e.iter().any(|ent| ent.text.contains("東京")),
            "Should find 東京"
        );
        assert!(
            e.iter().any(|ent| ent.text.contains("ソニー")),
            "Should find ソニー"
        );
    }

    #[test]
    fn korean_no_capitalization() {
        let text = "김철수는 서울에서 삼성전자에 다닙니다";
        // "Kim Cheolsu works at Samsung Electronics in Seoul"

        let e = stats().extract_entities(text, None).unwrap();

        // WILL FAIL
        assert!(e.is_empty(), "Korean: {:?}", e);
    }

    #[test]
    fn arabic_rtl_no_caps() {
        // Arabic: RTL + no capitalization
        let text = "يعمل أحمد في القاهرة لشركة مايكروسوفت";
        // "Ahmed works in Cairo for Microsoft"

        let e = stats().extract_entities(text, None).unwrap();

        // WILL FAIL - no capitalization + RTL complexity
        assert!(e.is_empty(), "Arabic: {:?}", e);
    }

    #[test]
    fn hebrew_rtl_no_caps() {
        let text = "דוד עובד בירושלים בחברת גוגל";
        // "David works in Jerusalem at Google company"

        let e = stats().extract_entities(text, None).unwrap();

        assert!(e.is_empty(), "Hebrew: {:?}", e);
    }

    // -------------------------------------------------------------------------
    // Mixed scripts (partial success)
    // -------------------------------------------------------------------------

    #[test]
    fn chinese_with_english_names() {
        // When English names appear in Chinese text, capitalization helps
        let text = "Steve Jobs创立了Apple公司";
        // "Steve Jobs founded Apple company"

        let e = stats().extract_entities(text, None).unwrap();

        // Might find "Steve Jobs" and "Apple" due to capitalization
        println!("Chinese+English: {:?}", entity_texts(&e));

        // The English parts might be detected
        let found_steve = e.iter().any(|e| e.text.contains("Steve"));
        let found_apple = e.iter().any(|e| e.text.contains("Apple"));

        if found_steve || found_apple {
            println!("Partial success: English names detected in Chinese text");
        }
    }

    #[test]
    fn japanese_with_katakana() {
        // Katakana often used for foreign names - provides some signal
        let text = "マイクロソフトのビル・ゲイツ氏";
        // "Microsoft's Bill Gates"

        let e = stats().extract_entities(text, None).unwrap();

        // Katakana doesn't help our English-based heuristics
        println!("Japanese katakana: {:?}", entity_texts(&e));
    }
}

// =============================================================================
// STACKED NER: Combined Behavior
// =============================================================================

mod stacked_multilingual {
    use super::*;

    #[test]
    fn stacked_chinese_partial() {
        // Pattern layer finds structured entities
        // Statistical layer finds nothing (no caps)
        let text = "会议日期 2024-01-15，联系 test@example.com，费用 $100";

        let e = stacked().extract_entities(text, None).unwrap();

        // Pattern should find: date, email, money
        assert!(has_type(&e, EntityType::Date), "Should find date");
        assert!(has_type(&e, EntityType::Email), "Should find email");
        assert!(has_type(&e, EntityType::Money), "Should find money");

        // But NO named entities (Person, Org, Location)
        assert!(!has_type(&e, EntityType::Person), "Should NOT find person");
        assert!(
            !has_type(&e, EntityType::Organization),
            "Should NOT find org"
        );
        assert!(
            !has_type(&e, EntityType::Location),
            "Should NOT find location"
        );
    }

    #[test]
    fn stacked_german_mixed() {
        let text = "Angela Merkel besucht Berlin am 2024-01-15. Kontakt: merkel@gov.de";

        let e = stacked().extract_entities(text, None).unwrap();

        // Pattern: date, email
        assert!(has_type(&e, EntityType::Date), "Should find date");
        assert!(has_type(&e, EntityType::Email), "Should find email");

        // Statistical: might find Angela Merkel, Berlin
        // (capitalized, though context words won't match perfectly)
        println!("German stacked: {:?}", entity_texts(&e));
    }
}

// =============================================================================
// POTENTIAL IMPROVEMENTS (documented as tests)
// =============================================================================

mod improvement_opportunities {
    #![allow(unused_imports)]
    use super::*;

    /// Japanese date pattern: YYYY年MM月DD日
    /// Adding this would be straightforward:
    /// ```regex
    /// (\d{4})年(\d{1,2})月(\d{1,2})日
    /// ```
    #[test]
    fn document_japanese_date_pattern() {
        let text = "2024年1月15日";
        let pattern = r"(\d{4})年(\d{1,2})月(\d{1,2})日";
        let re = regex::Regex::new(pattern).unwrap();
        assert!(re.is_match(text), "Japanese date pattern works");
    }

    /// Multilingual month names
    /// Could extend DATE_WRITTEN patterns with:
    /// - German: Januar, Februar, März, April, Mai, Juni, Juli, August, September, Oktober, November, Dezember
    /// - French: janvier, février, mars, avril, mai, juin, juillet, août, septembre, octobre, novembre, décembre
    /// - Spanish: enero, febrero, marzo, abril, mayo, junio, julio, agosto, septiembre, octubre, noviembre, diciembre
    #[test]
    fn document_multilingual_months() {
        let months_de = [
            "Januar",
            "Februar",
            "März",
            "April",
            "Mai",
            "Juni",
            "Juli",
            "August",
            "September",
            "Oktober",
            "November",
            "Dezember",
        ];
        let months_fr = [
            "janvier",
            "février",
            "mars",
            "avril",
            "mai",
            "juin",
            "juillet",
            "août",
            "septembre",
            "octobre",
            "novembre",
            "décembre",
        ];
        let months_es = [
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ];

        // All defined - would need to be added to pattern.rs
        assert_eq!(months_de.len(), 12);
        assert_eq!(months_fr.len(), 12);
        assert_eq!(months_es.len(), 12);
    }

    /// Unicode-aware digit matching in Rust regex
    ///
    /// POSITIVE FINDING: Rust regex crate has Unicode support enabled by default!
    /// The `\d` character class matches all Unicode decimal digits, including:
    /// - ASCII: 0-9
    /// - Arabic-Indic: ٠-٩
    /// - Extended Arabic-Indic (Persian): ۰-۹
    /// - And many more Unicode digit characters
    #[test]
    fn document_unicode_digits_supported() {
        let text = "١٢٣"; // 123 in Arabic-Indic

        // Rust regex \d matches Unicode digits by default
        let re = regex::Regex::new(r"\d+").unwrap();
        let matches = re.is_match(text);

        // This is a POSITIVE finding - Unicode digits work out of the box
        assert!(
            matches,
            "Rust regex \\d DOES match Arabic-Indic numerals (Unicode support is default)"
        );

        // Verify we can extract the match
        let m = re.find(text).unwrap();
        assert_eq!(m.as_str(), "١٢٣");

        // This means RegexNER works with Arabic-Indic numerals automatically!
    }

    /// For non-capitalizing languages, could use:
    /// 1. Script detection (Chinese characters = CJK block)
    /// 2. Character n-gram features
    /// 3. Dictionary-based lookup (gazetteers)
    /// 4. ML backends (GLiNER, NuNER)
    #[test]
    fn document_cjk_approaches() {
        // Unicode script detection
        let chinese = '中';
        let japanese_hiragana = 'あ';
        let japanese_katakana = 'ア';
        let _korean = '한';

        // Could use Unicode blocks to detect script
        // CJK Unified Ideographs: U+4E00..U+9FFF
        let is_cjk = |c: char| matches!(c as u32, 0x4E00..=0x9FFF);

        assert!(is_cjk(chinese));
        // Hiragana/Katakana are different blocks
        assert!(!is_cjk(japanese_hiragana));
        assert!(!is_cjk(japanese_katakana));
    }
}

// =============================================================================
// BENCHMARK: Expected Performance by Language
// =============================================================================

/// Summary of expected performance by language and backend.
///
/// | Language | RegexNER | HeuristicNER | Notes |
/// |----------|------------|----------------|-------|
/// | English | High | Medium | Reference implementation |
/// | German | High | Low-Medium | All nouns capitalized (FP problem) |
/// | French | High | Medium | Caps work, need FR months |
/// | Spanish | High | Medium | Similar to French |
/// | Russian | Medium | Low | Cyrillic script, caps work |
/// | Chinese | Medium | None | No caps, need ML/gazetteers |
/// | Japanese | Low | None | No caps, multiple scripts |
/// | Arabic | Low | None | RTL, no caps, different numerals |
/// | Korean | Medium | None | No caps, need ML |
///
/// For production multilingual NER, use:
/// - GLiNER/NuNER for named entities (zero-shot works across languages)
/// - RegexNER for structured entities (extend date patterns)
#[test]
fn performance_expectations_documented() {
    // This test exists to document expected behavior
    // See table above in doc comment

    // RegexNER: ~95%+ precision on supported patterns, any language
    // HeuristicNER: ~60-70% F1 English, near-zero for CJK/Arabic

    assert!(true, "Documentation test");
}

// =============================================================================
// REGRESSION: Unicode Offset Handling
// =============================================================================

mod unicode_offsets {
    use super::*;

    #[test]
    fn chinese_text_offsets_valid() {
        let text = "价格是 $100 美元";
        let e = pattern().extract_entities(text, None).unwrap();

        for entity in &e {
            // Offsets should be character offsets, not byte offsets
            let char_count = text.chars().count();
            assert!(
                entity.start <= char_count,
                "Start {} > char count {}",
                entity.start,
                char_count
            );
            assert!(
                entity.end <= char_count,
                "End {} > char count {}",
                entity.end,
                char_count
            );

            // Extract by char offset should match text
            let extracted: String = text
                .chars()
                .skip(entity.start)
                .take(entity.end - entity.start)
                .collect();
            assert_eq!(extracted, entity.text, "Offset mismatch");
        }
    }

    #[test]
    fn emoji_text_offsets_valid() {
        let text = "📧 Email: test@example.com 🎉";
        let e = pattern().extract_entities(text, None).unwrap();

        let email = find_text(&e, "test@example.com").expect("Should find email");

        // Verify extraction by offset works
        let extracted: String = text
            .chars()
            .skip(email.start)
            .take(email.end - email.start)
            .collect();
        assert_eq!(extracted, "test@example.com");
    }

    #[test]
    fn mixed_script_offsets_valid() {
        // Mix of ASCII, CJK, emoji
        let text = "日期: 2024-01-15 📅 费用: $100";
        let e = pattern().extract_entities(text, None).unwrap();

        for entity in &e {
            let extracted: String = text
                .chars()
                .skip(entity.start)
                .take(entity.end - entity.start)
                .collect();
            assert_eq!(extracted, entity.text, "Offset mismatch for {:?}", entity);
        }
    }
}
