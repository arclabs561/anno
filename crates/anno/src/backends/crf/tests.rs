use super::*;

#[test]
fn test_crf_basic() {
    let ner = CrfNER::new();
    let entities = ner
        .extract_entities("John Smith works at Google in California", None)
        .unwrap();

    // With our default heuristic weights + gazetteers, we should usually get some entities.
    // (Trained weights will do better, but defaults should not be totally dead.)
    assert!(!entities.is_empty(), "Expected some entities, got none");
}

#[test]
fn test_word_shape() {
    assert_eq!(CrfNER::word_shape("John"), "Xx");
    assert_eq!(CrfNER::word_shape("USA"), "X");
    assert_eq!(CrfNER::word_shape("hello"), "x");
    assert_eq!(CrfNER::word_shape("123"), "0");
    assert_eq!(CrfNER::word_shape("Hello123"), "Xx0");
}

#[test]
fn test_tokenize() {
    let tokens = CrfNER::tokenize("Hello world");
    assert_eq!(tokens, vec!["Hello", "world"]);
}

#[test]
fn test_empty_input() {
    let ner = CrfNER::new();
    let entities = ner.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_gazetteer_lookup() {
    let ner = CrfNER::new();

    // Gazetteer should contain common entities
    assert!(ner.gazetteers[&EntityType::Person].contains(&"John".to_string()));
    assert!(ner.gazetteers[&EntityType::Location].contains(&"California".to_string()));
    assert!(ner.gazetteers[&EntityType::Organization].contains(&"Google".to_string()));
}

#[test]
fn test_viterbi_returns_valid_labels() {
    let ner = CrfNER::new();
    let tokens = vec!["John", "works", "at", "Google"];
    let labels = ner.viterbi_decode(&tokens);

    assert_eq!(labels.len(), tokens.len());
    for label in &labels {
        assert!(ner.labels.contains(label));
    }
}

#[test]
fn test_common_verbs_not_in_entities() {
    let ner = CrfNER::new();

    // Test that common verbs don't get tagged as part of entities
    let entities = ner
        .extract_entities("John Smith works at Apple", None)
        .unwrap();

    // Should find John Smith and Apple, but NOT "works"
    let entity_texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    for entity_text in &entity_texts {
        assert!(
            !entity_text.contains("works"),
            "Entity '{}' should not contain 'works'",
            entity_text
        );
    }
}

#[test]
fn test_weights_for_common_words() {
    // This test asserts properties of the heuristic weight table, which includes
    // many `word.lower=...` entries. The bundled trained weights are intentionally
    // compact and omit token-identity features to keep the shipped file size small.
    #[cfg(feature = "bundled-crf-weights")]
    {
        return;
    }

    #[allow(unreachable_code)]
    let ner = CrfNER::new();

    // Check that weights exist for common stop words
    assert!(
        ner.weights.contains_key("word.lower=works:O"),
        "Missing weight for word.lower=works:O"
    );
    assert!(
        ner.weights.contains_key("word.lower=works:I-PER"),
        "Missing weight for word.lower=works:I-PER"
    );

    // Check that O weight is positive and I-* weight is negative
    let o_weight = *ner.weights.get("word.lower=works:O").unwrap();
    let i_per_weight = *ner.weights.get("word.lower=works:I-PER").unwrap();
    assert!(
        o_weight > 0.0,
        "O weight should be positive, got {}",
        o_weight
    );
    assert!(
        i_per_weight < 0.0,
        "I-PER weight should be negative, got {}",
        i_per_weight
    );
}

#[test]
fn test_unicode_char_offsets() {
    // Test that entity offsets are character-based, not byte-based
    let ner = CrfNER::new();

    // "北京" is 2 chars, 6 bytes. "Beijing" is 7 chars, 7 bytes.
    // Text "北京 Beijing" is 10 chars, 14 bytes.
    let text = "北京 Beijing";
    assert_eq!(text.len(), 14, "Expected 14 bytes");
    assert_eq!(text.chars().count(), 10, "Expected 10 characters");

    let entities = ner.extract_entities(text, None).unwrap();

    // Regardless of what entities are found, check all offsets are valid char offsets
    let char_count = text.chars().count();
    for entity in &entities {
        assert!(
            entity.start() <= entity.end(),
            "Invalid span: start {} > end {}",
            entity.start(),
            entity.end()
        );
        assert!(
            entity.end() <= char_count,
            "Entity end {} exceeds char count {} for text {:?}",
            entity.end(),
            char_count,
            text
        );

        // Also verify we can extract the text at those offsets
        let extracted: String = text
            .chars()
            .skip(entity.start())
            .take(entity.end() - entity.start())
            .collect();
        assert!(
            !extracted.is_empty() || entity.start() == entity.end(),
            "Empty extraction for entity at {}..{} in {:?}",
            entity.start(),
            entity.end(),
            text
        );
    }
}

#[test]
fn test_multilingual_inputs_no_panic_and_valid_spans() {
    let ner = CrfNER::new();
    let texts = [
        // Latin
        "Marie Curie discovered radium in Paris.",
        // CJK
        "習近平在北京會見了普京。",
        // Arabic (RTL)
        "التقى محمد بن سلمان بالرئيس في الرياض",
        // Cyrillic
        "Путин встретился с Си Цзиньпином в Москве.",
        // Devanagari
        "प्रधान मंत्री शर्मा दिल्ली में मिले।",
    ];

    for text in texts {
        let entities = ner.extract_entities(text, None).unwrap();
        let char_count = text.chars().count();
        for e in entities {
            assert!(e.start() <= e.end());
            assert!(e.end() <= char_count);
            let _span: String = text
                .chars()
                .skip(e.start())
                .take(e.end() - e.start())
                .collect();
        }
    }
}

/// Test that duplicate entity texts get correct offsets.
#[test]
fn test_duplicate_entity_offsets() {
    // Test token position calculation directly
    let text = "Google bought Google for $1 billion.";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let positions = CrfNER::calculate_token_positions(text, &tokens);

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
    let positions = CrfNER::calculate_token_positions(text, &tokens);

    // Each 東京 is 6 bytes (2 chars × 3 bytes each)
    assert_eq!(positions[0], (0, 6), "First '東京' at bytes 0-6");
    assert_eq!(positions[1], (7, 12), "Tokyo at bytes 7-12");
    assert_eq!(positions[2], (13, 19), "Second '東京' at bytes 13-19");
}

// ---------------------------------------------------------------------------
// Algorithm-focused unit tests
// ---------------------------------------------------------------------------

/// Build a minimal CrfNER with only the supplied weights and the standard BIO
/// label set. No gazetteers, no feature templates beyond the basics.
fn minimal_crf(weights: HashMap<String, f64>) -> CrfNER {
    CrfNER {
        weights,
        gazetteers: HashMap::new(),
        labels: vec![
            "O".to_string(),
            "B-PER".to_string(),
            "I-PER".to_string(),
            "B-ORG".to_string(),
            "I-ORG".to_string(),
            "B-LOC".to_string(),
            "I-LOC".to_string(),
            "B-MISC".to_string(),
            "I-MISC".to_string(),
        ],
        templates: vec![],
    }
}

/// Viterbi on empty input returns empty labels.
#[test]
fn test_viterbi_empty_input() {
    let ner = minimal_crf(HashMap::new());
    let labels = ner.viterbi_decode(&[]);
    assert!(labels.is_empty(), "Empty tokens should yield empty labels");
}

/// Viterbi on a single token should return exactly one label.
#[test]
fn test_viterbi_single_token() {
    let ner = minimal_crf(HashMap::new());
    let labels = ner.viterbi_decode(&["hello"]);
    assert_eq!(labels.len(), 1);
    // With no weights except the default +0.5 O bias inside score_label,
    // the single token should be labeled O.
    assert_eq!(labels[0], "O");
}

/// When emission weights strongly prefer B-PER for a capitalized word,
/// Viterbi should pick B-PER.
#[test]
fn test_viterbi_strong_emission_overrides_o_bias() {
    let mut w = HashMap::new();
    // Make B-PER overwhelmingly attractive for the "bias" feature
    w.insert("bias:B-PER".to_string(), 20.0);
    let ner = minimal_crf(w);

    let labels = ner.viterbi_decode(&["Alice"]);
    assert_eq!(labels, vec!["B-PER"]);
}

/// Transition weights enforce BIO validity: O -> I-PER is penalized,
/// so the decoder should prefer O -> B-PER -> I-PER over O -> I-PER.
#[test]
fn test_viterbi_bio_transition_constraint() {
    let mut w = HashMap::new();

    // First token: strongly O
    w.insert("BOS:O".to_string(), 10.0);
    // Second and third tokens: want some PER tag
    w.insert("word.lower=john:B-PER".to_string(), 8.0);
    w.insert("word.lower=john:I-PER".to_string(), 6.0);
    w.insert("word.lower=smith:I-PER".to_string(), 8.0);
    w.insert("word.lower=smith:B-PER".to_string(), 4.0);

    // Valid transition: B-PER -> I-PER is rewarded
    w.insert("trans:B-PER->I-PER".to_string(), 5.0);
    // Invalid transition: O -> I-PER is heavily penalized
    w.insert("trans:O->I-PER".to_string(), -50.0);

    let ner = minimal_crf(w);
    let labels = ner.viterbi_decode(&["The", "John", "Smith"]);

    assert_eq!(labels.len(), 3);
    // "John" should be B-PER (not I-PER, since previous is O)
    assert_eq!(
        labels[1], "B-PER",
        "Expected B-PER after O, not {}",
        labels[1]
    );
    // "Smith" should be I-PER (continuing the entity)
    assert_eq!(
        labels[2], "I-PER",
        "Expected I-PER continuation, not {}",
        labels[2]
    );
}

/// Cross-type I- transitions are blocked: B-PER -> I-ORG should not happen.
#[test]
fn test_viterbi_cross_type_transition_blocked() {
    let mut w = HashMap::new();

    // First token strongly B-PER
    w.insert("bias:B-PER".to_string(), 15.0);
    // Second token: I-ORG has a high emission but cross-type transition is penalized
    w.insert("word.lower=inc:I-ORG".to_string(), 5.0);
    w.insert("word.lower=inc:O".to_string(), 1.0);
    w.insert("trans:B-PER->I-ORG".to_string(), -50.0);

    let ner = minimal_crf(w);
    let labels = ner.viterbi_decode(&["Alice", "Inc"]);

    assert_eq!(labels.len(), 2);
    // Second token must NOT be I-ORG because the cross-type penalty is too high
    assert_ne!(
        labels[1], "I-ORG",
        "Cross-type transition B-PER -> I-ORG should be blocked"
    );
}

/// score_label returns the default O bias (0.5) when no weights match.
#[test]
fn test_score_label_default_o_bias() {
    let ner = minimal_crf(HashMap::new());
    let features = vec!["some_unknown_feature".to_string()];

    let o_score = ner.score_label(&features, "O");
    let b_per_score = ner.score_label(&features, "B-PER");

    assert!(
        o_score > b_per_score,
        "O should score higher than B-PER with no matching weights: O={}, B-PER={}",
        o_score,
        b_per_score
    );
    // O gets +0.5 bias, B-PER gets 0.0
    assert!(
        (o_score - 0.5).abs() < 1e-9,
        "O score should be 0.5, got {}",
        o_score
    );
    assert!(
        (b_per_score - 0.0).abs() < 1e-9,
        "B-PER score should be 0.0, got {}",
        b_per_score
    );
}

/// score_label accumulates weights from both "feature:label" and "feature" keys.
#[test]
fn test_score_label_weight_accumulation() {
    let mut w = HashMap::new();
    w.insert("feat_a:B-PER".to_string(), 2.0);
    w.insert("feat_b:B-PER".to_string(), 3.0);
    // A type-independent weight (applied at 0.5x)
    w.insert("feat_c".to_string(), 4.0);

    let ner = minimal_crf(w);
    let features = vec![
        "feat_a".to_string(),
        "feat_b".to_string(),
        "feat_c".to_string(),
    ];

    let score = ner.score_label(&features, "B-PER");
    // feat_a:B-PER = 2.0, feat_b:B-PER = 3.0, feat_c (independent) = 4.0 * 0.5 = 2.0
    // B-PER has no O bias, so total = 2.0 + 3.0 + 2.0 = 7.0
    assert!(
        (score - 7.0).abs() < 1e-9,
        "Expected score 7.0, got {}",
        score
    );
}

/// extract_features produces BOS for the first token and EOS for the last.
#[test]
fn test_extract_features_bos_eos() {
    let ner = minimal_crf(HashMap::new());
    let tokens = vec!["Hello", "world"];

    let feats_first = ner.extract_features(&tokens, 0, "O");
    assert!(
        feats_first.contains(&"BOS".to_string()),
        "First token should have BOS feature"
    );
    assert!(
        !feats_first.contains(&"EOS".to_string()),
        "First token should not have EOS"
    );

    let feats_last = ner.extract_features(&tokens, 1, "O");
    assert!(
        feats_last.contains(&"EOS".to_string()),
        "Last token should have EOS feature"
    );
    assert!(
        !feats_last.contains(&"BOS".to_string()),
        "Last token should not have BOS"
    );
}

/// extract_features for a single token should have both BOS and EOS.
#[test]
fn test_extract_features_single_token_bos_and_eos() {
    let ner = minimal_crf(HashMap::new());
    let tokens = vec!["Only"];
    let feats = ner.extract_features(&tokens, 0, "O");
    assert!(feats.contains(&"BOS".to_string()), "Single token needs BOS");
    assert!(feats.contains(&"EOS".to_string()), "Single token needs EOS");
}

/// extract_features includes the expected word identity and shape features.
#[test]
fn test_extract_features_word_identity() {
    let ner = minimal_crf(HashMap::new());
    let tokens = vec!["John"];
    let feats = ner.extract_features(&tokens, 0, "O");

    assert!(feats.contains(&"bias".to_string()), "Missing bias feature");
    assert!(
        feats.contains(&"word.lower=john".to_string()),
        "Missing word.lower feature"
    );
    assert!(
        feats.contains(&"word.shape=Xx".to_string()),
        "Missing word.shape feature"
    );
    assert!(
        feats.contains(&"word.istitle=True".to_string()),
        "Missing word.istitle feature"
    );
    assert!(
        feats.contains(&"word.isupper=False".to_string()),
        "Missing word.isupper feature"
    );
}

/// labels_to_entities correctly converts BIO labels into Entity spans.
#[test]
fn test_labels_to_entities_simple() {
    let ner = minimal_crf(HashMap::new());
    let text = "John Smith works at Google";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let labels = vec![
        "B-PER".to_string(),
        "I-PER".to_string(),
        "O".to_string(),
        "O".to_string(),
        "B-ORG".to_string(),
    ];

    let entities = ner.labels_to_entities(text, &tokens, &labels);

    assert_eq!(
        entities.len(),
        2,
        "Expected 2 entities, got {}",
        entities.len()
    );

    // First entity: "John Smith" (Person)
    assert_eq!(entities[0].text, "John Smith");
    assert_eq!(entities[0].entity_type, EntityType::Person);
    assert_eq!(entities[0].start(), 0);
    assert_eq!(entities[0].end(), 10);

    // Second entity: "Google" (Organization)
    assert_eq!(entities[1].text, "Google");
    assert_eq!(entities[1].entity_type, EntityType::Organization);
    assert_eq!(entities[1].start(), 20);
    assert_eq!(entities[1].end(), 26);
}

/// labels_to_entities handles entity at end of sequence (no trailing O).
#[test]
fn test_labels_to_entities_trailing_entity() {
    let ner = minimal_crf(HashMap::new());
    let text = "lives in Paris";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let labels = vec!["O".to_string(), "O".to_string(), "B-LOC".to_string()];

    let entities = ner.labels_to_entities(text, &tokens, &labels);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "Paris");
    assert_eq!(entities[0].entity_type, EntityType::Location);
}

/// labels_to_entities with all-O labels produces no entities.
#[test]
fn test_labels_to_entities_all_outside() {
    let ner = minimal_crf(HashMap::new());
    let text = "nothing special here";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let labels = vec!["O".to_string(), "O".to_string(), "O".to_string()];

    let entities = ner.labels_to_entities(text, &tokens, &labels);
    assert!(
        entities.is_empty(),
        "All-O labels should produce no entities"
    );
}

/// word_shape compresses repeated characters: "AAAA" -> "X", "aaaa" -> "x".
#[test]
fn test_word_shape_compression() {
    assert_eq!(CrfNER::word_shape("AAAA"), "X");
    assert_eq!(CrfNER::word_shape("aaaa"), "x");
    assert_eq!(CrfNER::word_shape("AaBb"), "XxXx");
    assert_eq!(CrfNER::word_shape("123-456"), "0-0");
    assert_eq!(CrfNER::word_shape("O'Brien"), "X'Xx");
}

/// Viterbi with a two-label system and explicit transition matrix to verify
/// the dynamic programming produces the globally optimal path.
#[test]
fn test_viterbi_global_optimality() {
    // Construct a scenario where the locally greedy choice at position 1
    // differs from the globally optimal path.
    //
    // Labels: O (idx 0), B-PER (idx 1), and the rest are available but
    // we only put weights on O and B-PER.
    //
    // Token 0: O scores 5, B-PER scores 3
    // Token 1: O scores 1, B-PER scores 4
    // Transition: O->B-PER = 0, B-PER->B-PER = 10 (huge bonus)
    //
    // Greedy: picks O at t=0 (5 > 3), then B-PER at t=1 (1+0+4 < 3+10+4).
    // Viterbi: picks B-PER at t=0 (3), then B-PER at t=1 (3 + 10 + 4 = 17),
    //          which beats O at t=0 (5), then B-PER at t=1 (5 + 0 + 4 = 9).
    let mut w = HashMap::new();
    w.insert("bias:O".to_string(), 5.0);
    w.insert("bias:B-PER".to_string(), 3.0);
    // Override at position 1 to make B-PER more attractive
    w.insert("word.lower=b:B-PER".to_string(), 1.0);
    w.insert("trans:B-PER->B-PER".to_string(), 10.0);

    let ner = minimal_crf(w);
    let labels = ner.viterbi_decode(&["a", "b"]);

    assert_eq!(labels.len(), 2);
    // The globally optimal path should be B-PER, B-PER
    assert_eq!(
        labels,
        vec!["B-PER", "B-PER"],
        "Viterbi should find the globally optimal path, not the greedy one"
    );
}

/// Default weights include BIO constraint transitions: O -> I-* is penalized at -10.
#[test]
fn test_default_weights_bio_constraints() {
    let w = CrfNER::default_weights();

    // O -> I-* transitions should be strongly negative
    for tag in ["I-PER", "I-ORG", "I-LOC", "I-MISC"] {
        let key = format!("trans:O->{}", tag);
        let val = w.get(&key).copied().unwrap_or(0.0);
        assert!(
            val < -5.0,
            "O -> {} should be heavily penalized, got {}",
            tag,
            val
        );
    }

    // Same-type B -> I should be positive
    for (b, i) in [("B-PER", "I-PER"), ("B-ORG", "I-ORG"), ("B-LOC", "I-LOC")] {
        let key = format!("trans:{}->{}", b, i);
        let val = w.get(&key).copied().unwrap_or(0.0);
        assert!(val > 0.0, "{} -> {} should be positive, got {}", b, i, val);
    }

    // Cross-type B -> I should be strongly negative
    for (b, i) in [("B-PER", "I-ORG"), ("B-ORG", "I-LOC"), ("B-LOC", "I-PER")] {
        let key = format!("trans:{}->{}", b, i);
        let val = w.get(&key).copied().unwrap_or(0.0);
        assert!(
            val < -5.0,
            "{} -> {} should be heavily penalized, got {}",
            b,
            i,
            val
        );
    }
}

// ---------------------------------------------------------------------------
// Additional algorithm tests
// ---------------------------------------------------------------------------

/// extract_features includes prefix and suffix features for words >= 2 chars.
#[test]
fn test_extract_features_prefix_suffix() {
    let ner = minimal_crf(HashMap::new());
    let tokens = vec!["Johnson"];
    let feats = ner.extract_features(&tokens, 0, "O");

    assert!(
        feats.contains(&"prefix2=jo".to_string()),
        "Missing prefix2 feature, got: {:?}",
        feats
    );
    assert!(
        feats.contains(&"prefix3=joh".to_string()),
        "Missing prefix3 feature"
    );
    assert!(
        feats.contains(&"suffix2=on".to_string()),
        "Missing suffix2 feature"
    );
    assert!(
        feats.contains(&"suffix3=son".to_string()),
        "Missing suffix3 feature"
    );
}

/// Short words (1 char) should not produce prefix/suffix features.
#[test]
fn test_extract_features_no_prefix_suffix_for_short_word() {
    let ner = minimal_crf(HashMap::new());
    let tokens = vec!["I"];
    let feats = ner.extract_features(&tokens, 0, "O");

    let has_prefix = feats.iter().any(|f| f.starts_with("prefix"));
    let has_suffix = feats.iter().any(|f| f.starts_with("suffix"));
    assert!(
        !has_prefix,
        "Single-char word should have no prefix feature"
    );
    assert!(
        !has_suffix,
        "Single-char word should have no suffix feature"
    );
}

/// extract_features includes context features for previous and next words.
#[test]
fn test_extract_features_context_words() {
    let ner = minimal_crf(HashMap::new());
    let tokens = vec!["Dr", "John", "Smith"];
    let feats = ner.extract_features(&tokens, 1, "O");

    // Previous word features
    assert!(
        feats.contains(&"-1:word.lower=dr".to_string()),
        "Missing -1:word.lower feature"
    );
    // "Dr" is titlecase, not all-uppercase
    assert!(
        feats.contains(&"-1:word.istitle=True".to_string()),
        "Missing -1:word.istitle for 'Dr'"
    );
    assert!(
        feats.contains(&"-1:word.isupper=False".to_string()),
        "Dr is not all-uppercase"
    );

    // Next word features
    assert!(
        feats.contains(&"+1:word.lower=smith".to_string()),
        "Missing +1:word.lower feature"
    );
    assert!(
        feats.contains(&"+1:word.istitle=True".to_string()),
        "Missing +1:word.istitle feature"
    );
}

/// extract_features: digit-only words get word.isdigit=True.
#[test]
fn test_extract_features_digit_word() {
    let ner = minimal_crf(HashMap::new());
    let tokens = vec!["2024"];
    let feats = ner.extract_features(&tokens, 0, "O");

    assert!(
        feats.contains(&"word.isdigit=True".to_string()),
        "All-digit word should have isdigit=True"
    );
    assert!(
        feats.contains(&"word.isupper=False".to_string()),
        "Digits are not uppercase"
    );
}

/// extract_features: mixed word gets word.isdigit=False.
#[test]
fn test_extract_features_mixed_word_not_digit() {
    let ner = minimal_crf(HashMap::new());
    let tokens = vec!["Room42"];
    let feats = ner.extract_features(&tokens, 0, "O");

    assert!(
        feats.contains(&"word.isdigit=False".to_string()),
        "Mixed word should have isdigit=False"
    );
}

/// extract_features with gazetteers: a Person gazetteer match emits a gaz:PER feature.
#[test]
fn test_extract_features_gazetteer_match() {
    let ner = CrfNER::new(); // Full model with gazetteers
    let tokens = vec!["John", "works"];
    let feats = ner.extract_features(&tokens, 0, "O");

    assert!(
        feats.contains(&"gaz:PER".to_string()),
        "John should match Person gazetteer, features: {:?}",
        feats
    );
}

/// extract_features with gazetteers: non-matching word has no gaz feature.
#[test]
fn test_extract_features_no_gazetteer_match() {
    let ner = CrfNER::new();
    let tokens = vec!["works"];
    let feats = ner.extract_features(&tokens, 0, "O");

    let has_gaz = feats.iter().any(|f| f.starts_with("gaz:"));
    assert!(
        !has_gaz,
        "'works' should not match any gazetteer, features: {:?}",
        feats
    );
}

/// tokenize: multiple spaces and leading/trailing whitespace.
#[test]
fn test_tokenize_whitespace_variants() {
    assert_eq!(
        CrfNER::tokenize("  Hello   world  "),
        vec!["Hello", "world"]
    );
    assert!(CrfNER::tokenize("").is_empty());
    assert!(CrfNER::tokenize("   ").is_empty());
    assert_eq!(CrfNER::tokenize("single"), vec!["single"]);
}

/// tokenize: tabs and newlines count as whitespace.
#[test]
fn test_tokenize_tabs_newlines() {
    assert_eq!(
        CrfNER::tokenize("Hello\tworld\nfoo"),
        vec!["Hello", "world", "foo"]
    );
}

/// labels_to_entities: consecutive B- tags (back-to-back entities, no I- continuation).
#[test]
fn test_labels_to_entities_consecutive_b_tags() {
    let ner = minimal_crf(HashMap::new());
    let text = "John Mary works";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let labels = vec!["B-PER".to_string(), "B-PER".to_string(), "O".to_string()];

    let entities = ner.labels_to_entities(text, &tokens, &labels);
    assert_eq!(
        entities.len(),
        2,
        "Two consecutive B-PER should yield 2 entities"
    );
    assert_eq!(entities[0].text, "John");
    assert_eq!(entities[1].text, "Mary");
}

/// labels_to_entities: MISC entity type.
#[test]
fn test_labels_to_entities_misc_type() {
    let ner = minimal_crf(HashMap::new());
    let text = "World Cup";
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let labels = vec!["B-MISC".to_string(), "I-MISC".to_string()];

    let entities = ner.labels_to_entities(text, &tokens, &labels);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "World Cup");
    assert_eq!(
        entities[0].entity_type,
        EntityType::custom("MISC", EntityCategory::Misc)
    );
}

/// labels_to_entities: empty tokens and labels produces no entities.
#[test]
fn test_labels_to_entities_empty() {
    let ner = minimal_crf(HashMap::new());
    let entities = ner.labels_to_entities("", &[], &[]);
    assert!(entities.is_empty());
}

/// calculate_token_positions on empty input.
#[test]
fn test_calculate_token_positions_empty() {
    let positions = CrfNER::calculate_token_positions("", &[]);
    assert!(positions.is_empty());
}

/// calculate_token_positions with single token.
#[test]
fn test_calculate_token_positions_single() {
    let positions = CrfNER::calculate_token_positions("Hello", &["Hello"]);
    assert_eq!(positions, vec![(0, 5)]);
}

/// calculate_token_positions preserves ordering with punctuation-adjacent tokens.
#[test]
fn test_calculate_token_positions_with_punctuation() {
    let text = "Hello, world!";
    // split_whitespace would give ["Hello,", "world!"]
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let positions = CrfNER::calculate_token_positions(text, &tokens);
    assert_eq!(positions[0], (0, 6)); // "Hello,"
    assert_eq!(positions[1], (7, 13)); // "world!"
}

/// word_shape handles empty string.
#[test]
fn test_word_shape_empty() {
    assert_eq!(CrfNER::word_shape(""), "");
}

/// word_shape handles mixed punctuation and Unicode.
#[test]
fn test_word_shape_unicode_letters() {
    // Uppercase Cyrillic followed by lowercase
    let shape = CrfNER::word_shape("Москва");
    assert_eq!(shape, "Xx", "Titlecase Cyrillic should be Xx");
}

/// Viterbi on a longer sequence: labels length matches tokens length.
#[test]
fn test_viterbi_longer_sequence_length_invariant() {
    let ner = CrfNER::new();
    let tokens: Vec<&str> = "The quick brown fox jumps over the lazy dog near London"
        .split_whitespace()
        .collect();
    let labels = ner.viterbi_decode(&tokens);
    assert_eq!(
        labels.len(),
        tokens.len(),
        "Viterbi must return exactly one label per token"
    );
    for label in &labels {
        assert!(
            ner.labels.contains(label),
            "Label '{}' not in label set",
            label
        );
    }
}

/// Viterbi: all-lowercase common words should mostly be labeled O.
#[test]
fn test_viterbi_common_words_are_outside() {
    let ner = CrfNER::new_heuristic();
    let tokens = vec!["the", "quick", "brown", "fox"];
    let labels = ner.viterbi_decode(&tokens);
    for (tok, label) in tokens.iter().zip(labels.iter()) {
        assert_eq!(
            label, "O",
            "Common lowercase word '{}' should be O, got '{}'",
            tok, label
        );
    }
}

/// score_label: matching both "feature:label" and "feature" keys accumulates correctly.
#[test]
fn test_score_label_both_keys_present() {
    let mut w = HashMap::new();
    w.insert("myfeat:O".to_string(), 3.0);
    w.insert("myfeat".to_string(), 2.0); // type-independent, applied at 0.5x
    let ner = minimal_crf(w);

    let features = vec!["myfeat".to_string()];
    let score = ner.score_label(&features, "O");
    // 3.0 (label-specific) + 2.0 * 0.5 (type-independent) + 0.5 (O bias) = 4.5
    assert!((score - 4.5).abs() < 1e-9, "Expected 4.5, got {}", score);
}

/// new_heuristic produces a model that does not use shipped/trained weights.
#[test]
fn test_new_heuristic_uses_default_weights() {
    let heuristic = CrfNER::new_heuristic();
    let default_w = CrfNER::default_weights();
    // The heuristic model's weights should match the default weights exactly.
    assert_eq!(
        heuristic.weights.len(),
        default_w.len(),
        "Heuristic model should have same number of weights as default_weights()"
    );
    for (key, val) in &default_w {
        let got = heuristic.weights.get(key).copied().unwrap_or(f64::NAN);
        assert!(
            (got - val).abs() < 1e-12,
            "Weight mismatch for '{}': expected {}, got {}",
            key,
            val,
            got
        );
    }
}

// =========================================================================
// Sentence boundary clipping
// =========================================================================

#[test]
fn sentence_boundary_detection() {
    let text = "Max Planck Institute respectively. Doudna said something.";
    let boundaries = super::sentence_boundary_offsets(text);
    assert!(
        !boundaries.is_empty(),
        "Should detect boundary at period before 'Doudna'"
    );
}

#[test]
fn crf_no_cross_sentence_span() {
    // If CRF produces an entity spanning "respectively. Doudna", the post-processor
    // should clip it at the sentence boundary.
    let mut entities = vec![Entity::new(
        "Institute respectively. Doudna",
        EntityType::Person,
        10,
        40,
        0.7,
    )];
    let text = "Max Planck Institute respectively. Doudna said something else here.";
    super::clip_entities_at_sentence_boundaries(text, &mut entities);
    // Entity should be clipped to end before "Doudna"
    for e in &entities {
        assert!(
            e.end() <= 34,
            "Entity should not cross sentence boundary: {:?}",
            e
        );
        assert!(
            !e.text.contains("Doudna"),
            "Entity text should not contain 'Doudna': {:?}",
            e
        );
    }
}
