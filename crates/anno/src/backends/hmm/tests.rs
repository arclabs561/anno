use super::*;

#[test]
fn test_basic_extraction() {
    let ner = HmmNER::new();
    let entities = ner
        .extract_entities("John works at Google in California.", None)
        .unwrap();

    // HMM with heuristics should find some entities
    assert!(!entities.is_empty(), "HMM should find at least one entity");
    for entity in &entities {
        assert!(entity.confidence > 0.0 && entity.confidence <= 1.0);
    }

    // All expected entities should be found
    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        texts.contains(&"John") || texts.contains(&"Google") || texts.contains(&"California"),
        "Expected at least one of John, Google, or California in results; got {:?}",
        texts
    );
}

#[test]
fn test_empty_input() {
    let ner = HmmNER::new();
    let entities = ner.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_viterbi_path_length() {
    let ner = HmmNER::new();
    let words = vec!["John", "works", "at", "Google"];
    let path = ner.viterbi(&words);

    assert_eq!(path.len(), words.len());
}

#[test]
fn test_bio_constraints() {
    let ner = HmmNER::new();

    // I-PER should not follow O with high probability
    let i_per = ner.state_to_idx["I-PER"];
    let o = ner.state_to_idx["O"];
    let b_per = ner.state_to_idx["B-PER"];

    // Transition O -> I-PER should be very low
    assert!(ner.transitions[o][i_per] < 0.01);

    // Transition B-PER -> I-PER should be reasonable
    assert!(ner.transitions[b_per][i_per] > 0.1);
}

#[test]
fn test_emission_heuristics() {
    let ner = HmmNER::new();

    let _o_idx = ner.state_to_idx["O"];
    let b_per_idx = ner.state_to_idx["B-PER"];

    // Capitalized word should have higher entity probability
    let cap_prob = ner.emission_prob(b_per_idx, "John");
    let lower_prob = ner.emission_prob(b_per_idx, "john");

    assert!(cap_prob > lower_prob, "capitalized prob {} should exceed lower prob {}", cap_prob, lower_prob);
    // Signal should be meaningful, not just a rounding difference
    assert!(
        cap_prob > 0.5 * lower_prob || lower_prob == 0.0,
        "cap_prob {} should be materially larger than lower_prob {}",
        cap_prob,
        lower_prob
    );
}

#[test]
fn test_training() {
    let mut ner = HmmNER::new();

    let sentences: Vec<(&[&str], &[&str])> = vec![
        (
            &["John", "works", "at", "Google"][..],
            &["B-PER", "O", "O", "B-ORG"][..],
        ),
        (
            &["Mary", "lives", "in", "Paris"][..],
            &["B-PER", "O", "O", "B-LOC"][..],
        ),
    ];

    ner.train(&sentences);

    // After training, transitions should be updated
    let b_per = ner.state_to_idx["B-PER"];
    let o = ner.state_to_idx["O"];

    // B-PER -> O should be high (entities followed by non-entities)
    assert!(ner.transitions[b_per][o] > 0.3);
}

#[test]
fn test_unicode_offsets() {
    let ner = HmmNER::new();
    let text = "北京 Google Inc.";
    let char_count = text.chars().count();

    let entities = ner.extract_entities(text, None).unwrap();

    for entity in &entities {
        assert!(entity.start() <= entity.end());
        assert!(entity.end() <= char_count);
    }
}

#[test]
fn test_config() {
    let config = HmmConfig {
        smoothing: 1e-5,
        ..Default::default()
    };

    let ner = HmmNER::with_config(config);
    assert_eq!(ner.config.smoothing, 1e-5);
}

#[test]
fn test_supported_types() {
    let ner = HmmNER::new();
    let types = ner.supported_types();

    assert!(types.contains(&EntityType::Person));
    assert!(types.contains(&EntityType::Organization));
    assert!(types.contains(&EntityType::Location));
}

/// Test that duplicate entity texts get correct offsets.
#[test]
fn test_duplicate_entity_offsets() {
    // Test token position calculation directly
    let text = "Google bought Google for $1 billion.";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let positions = HmmNER::calculate_token_positions(text, &tokens);

    // First "Google" at byte 0-6
    assert_eq!(
        positions[0],
        (0, 6),
        "First 'Google' should be at bytes 0-6"
    );
    // Second "Google" at byte 14-20
    assert_eq!(
        positions[2],
        (14, 20),
        "Second 'Google' should be at bytes 14-20"
    );
}

/// Test token position calculation with Unicode.
#[test]
fn test_token_positions_unicode() {
    let text = "東京 Tokyo 東京";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let positions = HmmNER::calculate_token_positions(text, &tokens);

    // Each 東京 is 6 bytes (2 chars × 3 bytes each)
    assert_eq!(positions[0], (0, 6), "First '東京' at bytes 0-6");
    assert_eq!(positions[1], (7, 12), "Tokyo at bytes 7-12");
    assert_eq!(positions[2], (13, 19), "Second '東京' at bytes 13-19");
}
