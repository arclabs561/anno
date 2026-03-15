use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn extraction_never_panics(text in ".*") {
        let ner = RegexNER::new();
        let _ = ner.extract_entities(&text, None);
    }

    #[test]
    fn entities_within_text_bounds(text in ".{1,200}") {
        let ner = RegexNER::new();
        if let Ok(entities) = ner.extract_entities(&text, None) {
            let text_char_len = text.chars().count();
            for e in entities {
                prop_assert!(e.start() <= text_char_len);
                prop_assert!(e.end() <= text_char_len);
                prop_assert!(e.start() <= e.end());
            }
        }
    }

    #[test]
    fn dollar_amounts_detected(amount in 1u32..10000) {
        let text = format!("Cost: ${}", amount);
        let ner = RegexNER::new();
        let entities = ner.extract_entities(&text, None).unwrap();
        prop_assert!(entities.iter().any(|e| e.entity_type == EntityType::Money));
    }

    #[test]
    fn percentages_detected(pct in 1u32..100) {
        let text = format!("{}% complete", pct);
        let ner = RegexNER::new();
        let entities = ner.extract_entities(&text, None).unwrap();
        prop_assert!(entities.iter().any(|e| e.entity_type == EntityType::Percent));
    }

    #[test]
    fn emails_detected(user in "[a-z]{3,10}", domain in "[a-z]{3,8}") {
        let text = format!("Contact: {}@{}.com", user, domain);
        let ner = RegexNER::new();
        let entities = ner.extract_entities(&text, None).unwrap();
        prop_assert!(entities.iter().any(|e|
            e.entity_type == EntityType::Email
        ));
    }

    #[test]
    fn urls_detected(path in "[a-z]{1,10}") {
        let text = format!("Visit https://example.com/{}", path);
        let ner = RegexNER::new();
        let entities = ner.extract_entities(&text, None).unwrap();
        prop_assert!(entities.iter().any(|e|
            e.entity_type == EntityType::Url
        ));
    }

    #[test]
    fn iso_dates_detected(y in 2000u32..2030, m in 1u32..13, d in 1u32..29) {
        let text = format!("Date: {:04}-{:02}-{:02}", y, m, d);
        let ner = RegexNER::new();
        let entities = ner.extract_entities(&text, None).unwrap();
        prop_assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
    }

    #[test]
    fn no_overlapping_entities(text in ".{0,100}") {
        let ner = RegexNER::new();
        if let Ok(entities) = ner.extract_entities(&text, None) {
            for i in 0..entities.len() {
                for j in (i + 1)..entities.len() {
                    let e1 = &entities[i];
                    let e2 = &entities[j];
                    let overlap = e1.start() < e2.end() && e2.start() < e1.end();
                    prop_assert!(!overlap, "Overlap: {:?} and {:?}", e1, e2);
                }
            }
        }
    }
}
