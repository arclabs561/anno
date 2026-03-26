use super::*;

#[test]
fn test_basic_extraction() {
    let ner = HeuristicCrfNER::new();
    let entities = ner
        .extract_entities("John Smith works at Google Inc.", None)
        .unwrap();

    // Should find some entities with the heuristic fallback
    // (Exact results depend on heuristic tuning)
    assert!(entities
        .iter()
        .all(|e| e.confidence > 0.0 && e.confidence <= 1.0));
}

#[test]
fn test_empty_input() {
    let ner = HeuristicCrfNER::new();
    let entities = ner.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_whitespace_only() {
    let ner = HeuristicCrfNER::new();
    let entities = ner.extract_entities("   \n\t  ", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_viterbi_respects_bio_constraints() {
    let ner = HeuristicCrfNER::new();

    // Create emissions that would prefer I-PER after O
    // But CRF transitions should prevent this
    let emissions = vec![
        vec![0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], // O preferred
        vec![0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], // I-PER has high score
    ];

    let path = ner.viterbi_decode(&emissions);

    // Should NOT have I-PER (idx 2) after O (idx 0) due to transition constraints
    // Instead should have B-PER (idx 1) or O
    if path[0] == 0 {
        // If first is O, second should not be I-*
        assert!(
            path[1] == 0 || ner.labels[path[1]].starts_with("B-"),
            "Invalid BIO sequence: O followed by {}",
            ner.labels[path[1]]
        );
    }
}

#[test]
fn test_unicode_offsets() {
    let ner = HeuristicCrfNER::new();
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
    let config = HeuristicCrfConfig {
        hidden_size: 512,
        num_layers: 3,
        dropout: 0.3,
        use_char_embeddings: false,
        max_seq_len: 256,
    };

    let ner = HeuristicCrfNER::with_config(config.clone());
    assert_eq!(ner.config().hidden_size, 512);
    assert_eq!(ner.config().num_layers, 3);
}

#[test]
fn test_transition_matrix_shape() {
    let ner = HeuristicCrfNER::new();
    let n = ner.labels.len();

    assert_eq!(ner.transitions.len(), n);
    for row in &ner.transitions {
        assert_eq!(row.len(), n);
    }
}

#[test]
fn test_supported_types() {
    let ner = HeuristicCrfNER::new();
    let types = ner.supported_types();

    assert!(types.contains(&EntityType::Person));
    assert!(types.contains(&EntityType::Organization));
    assert!(types.contains(&EntityType::Location));
}

/// Test that duplicate entity texts get correct offsets.
///
/// This test verifies the fix for a bug where `text.find()` was used to locate
/// entity positions, which always returned the first occurrence. When the same
/// entity text appeared multiple times, subsequent occurrences would have
/// incorrect offsets pointing to the first occurrence.
#[test]
fn test_duplicate_entity_offsets() {
    let ner = HeuristicCrfNER::new();

    // Text with "Google" appearing twice
    let text = "Google bought Google for $1 billion.";

    // Test token position calculation directly
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let positions = HeuristicCrfNER::calculate_token_positions(text, &tokens);

    // "Google" appears at indices 0 and 2 in tokens
    // First "Google" at byte 0-6
    assert_eq!(
        positions[0],
        (0, 6),
        "First 'Google' should be at bytes 0-6"
    );
    // Second "Google" at byte 14-20 (after "Google bought ")
    assert_eq!(
        positions[2],
        (14, 20),
        "Second 'Google' should be at bytes 14-20"
    );

    // Also test with the full extraction
    let entities = ner.extract_entities(text, None).unwrap();

    // If any Google entities are found, verify they have distinct offsets
    let google_entities: Vec<_> = entities
        .iter()
        .filter(|e| e.text.contains("Google"))
        .collect();

    if google_entities.len() >= 2 {
        assert_ne!(
            google_entities[0].start(),
            google_entities[1].start(),
            "Duplicate entities should have different start positions"
        );
    }
}

/// Test token position calculation with Unicode.
#[test]
fn test_token_positions_unicode() {
    let text = "東京 Tokyo 東京 Osaka";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let positions = HeuristicCrfNER::calculate_token_positions(text, &tokens);

    // Each 東京 is 6 bytes (2 chars × 3 bytes each)
    assert_eq!(positions[0], (0, 6), "First '東京' at bytes 0-6");
    assert_eq!(positions[1], (7, 12), "Tokyo at bytes 7-12");
    assert_eq!(positions[2], (13, 19), "Second '東京' at bytes 13-19");
    assert_eq!(positions[3], (20, 25), "Osaka at bytes 20-25");
}

#[test]
fn test_config_defaults() {
    let config = HeuristicCrfConfig::default();
    assert_eq!(config.hidden_size, 256);
    assert_eq!(config.num_layers, 2);
    assert!((config.dropout - 0.5).abs() < f32::EPSILON);
    assert!(config.use_char_embeddings);
    assert_eq!(config.max_seq_len, 512);
}

#[test]
fn test_vocab_lookup_empty() {
    let ner = HeuristicCrfNER::new();
    // Default model has empty vocab (no pre-trained embeddings loaded)
    assert!(ner.vocab().is_empty());
    assert_eq!(ner.vocab_lookup("hello"), None);
}

#[test]
fn test_labels_accessor() {
    let ner = HeuristicCrfNER::new();
    let labels = ner.labels();
    assert_eq!(labels.len(), 9);
    assert_eq!(labels[0], "O");
    assert!(labels.iter().any(|l| l == "B-PER"));
    assert!(labels.iter().any(|l| l == "I-PER"));
    assert!(labels.iter().any(|l| l == "B-ORG"));
    assert!(labels.iter().any(|l| l == "B-LOC"));
}
