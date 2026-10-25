//! Span integrity tests for the BERT BIO decoder.
//!
//! These tests verify that subword tokenization does not truncate entity spans.
//! Trailing-punctuation trimming for word-boundary backends is covered by unit
//! tests next to the production code (`backends/nuner/tests.rs`).

// =============================================================================
// BERT BIO decoder: subword continuation
// =============================================================================
// These tests require the `onnx` feature and actual model weights, so they
// are guarded behind `#[cfg(feature = "onnx")]` and `#[ignore]` (run with
// `cargo test --features onnx -- --ignored`).

#[cfg(feature = "onnx")]
mod bert_subword {
    /// Regression: "Christine Lagarde" was truncated to "Christine Lagard"
    /// because the final subword "##e" received an O label.
    #[test]
    #[ignore] // requires model download
    fn christine_lagarde_full_span() {
        let model = anno::BertNEROnnx::new("protectai/bert-base-NER-onnx").unwrap();
        let entities = model
            .extract_entities("ECB President Christine Lagarde spoke today.", None)
            .unwrap();

        let per_texts: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, anno::EntityType::Person))
            .map(|e| e.text.as_str())
            .collect();

        assert!(
            per_texts.contains(&"Christine Lagarde"),
            "Expected full 'Christine Lagarde', got: {:?}",
            per_texts,
        );
    }

    /// Regression: multi-subword names should not be truncated.
    #[test]
    #[ignore] // requires model download
    fn shuntaro_furukawa_full_span() {
        let model = anno::BertNEROnnx::new("protectai/bert-base-NER-onnx").unwrap();
        let entities = model
            .extract_entities("Nintendo CEO Shuntaro Furukawa announced new games.", None)
            .unwrap();

        let per_texts: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, anno::EntityType::Person))
            .map(|e| e.text.as_str())
            .collect();

        // The model should at minimum capture "Furukawa" without truncation.
        // Full "Shuntaro Furukawa" is ideal.
        let has_furukawa = per_texts.iter().any(|t| t.ends_with("Furukawa"));
        assert!(
            has_furukawa,
            "Expected name ending in 'Furukawa', got: {:?}",
            per_texts,
        );
    }

    /// Entity at end of sentence (boundary condition).
    #[test]
    #[ignore] // requires model download
    fn entity_at_sentence_end() {
        let model = anno::BertNEROnnx::new("protectai/bert-base-NER-onnx").unwrap();
        let entities = model
            .extract_entities("The meeting was held in Strasbourg.", None)
            .unwrap();

        let loc_texts: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, anno::EntityType::Location))
            .map(|e| e.text.as_str())
            .collect();

        assert!(
            loc_texts.contains(&"Strasbourg"),
            "Expected 'Strasbourg', got: {:?}",
            loc_texts,
        );
    }
}

// =============================================================================
// Pattern backend: European decimal comma
// =============================================================================

mod pattern_percent {
    use anno::Model;

    #[test]
    fn european_decimal_comma() {
        let model = anno::RegexNER::new();
        let entities = model
            .extract_entities("Die EZB senkte den Leitzins auf 3,75% im Juni.", None)
            .unwrap();

        let pct: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, anno::EntityType::Percent))
            .map(|e| e.text.as_str())
            .collect();

        assert!(
            pct.contains(&"3,75%"),
            "Expected '3,75%' as single PERCENT entity, got: {:?}",
            pct,
        );
    }

    #[test]
    fn period_decimal_still_works() {
        let model = anno::RegexNER::new();
        let entities = model
            .extract_entities("Inflation rose 0.25% in Q4.", None)
            .unwrap();

        let pct: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, anno::EntityType::Percent))
            .map(|e| e.text.as_str())
            .collect();

        assert!(
            pct.contains(&"0.25%"),
            "Expected '0.25%' as single PERCENT entity, got: {:?}",
            pct,
        );
    }

    #[test]
    fn whole_number_percent() {
        let model = anno::RegexNER::new();
        let entities = model
            .extract_entities("The rate is 5% per annum.", None)
            .unwrap();

        let pct: Vec<&str> = entities
            .iter()
            .filter(|e| matches!(e.entity_type, anno::EntityType::Percent))
            .map(|e| e.text.as_str())
            .collect();

        assert!(pct.contains(&"5%"), "Expected '5%', got: {:?}", pct,);
    }
}
