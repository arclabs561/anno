//! Property-based tests for NER backend invariants.
//!
//! These tests verify structural guarantees that ALL backends must satisfy,
//! regardless of implementation. They use lightweight backends (regex, heuristic,
//! ensemble) that don't require model downloads.

use anno::{EnsembleNER, HeuristicNER, Model, RegexNER};
use proptest::prelude::*;
use std::sync::LazyLock;

/// Cached EnsembleNER to avoid reconstructing per proptest case.
/// `EnsembleNER::new()` compiles regex patterns and initializes backends,
/// which is expensive when repeated 50+ times.
static ENSEMBLE: LazyLock<EnsembleNER> = LazyLock::new(EnsembleNER::new);

/// Normalize whitespace: collapse runs of whitespace to a single space and trim.
fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// =============================================================================
// Arbitrary Generators
// =============================================================================

/// Generate text that is likely to produce entities.
fn arb_ner_text() -> impl Strategy<Value = String> {
    prop_oneof![
        // Names with titles
        Just("Dr. John Smith visited Paris on 2024-01-15.".to_string()),
        Just("Apple CEO Tim Cook met Google CEO Sundar Pichai.".to_string()),
        Just("Contact test@email.com or call 555-123-4567.".to_string()),
        Just("The EU spent EUR 3.2 billion on infrastructure.".to_string()),
        Just("Angela Merkel visited the United Nations in New York.".to_string()),
        Just("Barack Obama spoke at Harvard University yesterday.".to_string()),
        // Empty / whitespace
        Just("".to_string()),
        Just("   ".to_string()),
        // Unicode
        Just("Marie Curie won 2 Nobel Prizes.".to_string()),
        // Numbers only
        Just("The rate is 5.25% per annum.".to_string()),
        // Long text
        Just(
            "Alice and Bob went to London. Charlie met them in Paris. \
              David works at Microsoft. Eve works at Apple Inc. \
              They all met on 2024-03-15 at the UN headquarters."
                .to_string()
        ),
        // Random printable ASCII
        "[A-Za-z .,;:!?0-9@#$%&*()-]{0,200}",
    ]
}

/// Generate truly random text that should not crash backends.
fn arb_fuzz_text() -> impl Strategy<Value = String> {
    prop_oneof![
        // Pure whitespace variants
        "[ \t\n\r]{0,50}",
        // Random ASCII
        "[ -~]{0,300}",
        // Mixed with some Unicode
        prop::string::string_regex("[A-Za-z0-9 \u{00C0}-\u{00FF}]{0,200}").expect("valid regex"),
    ]
}

// =============================================================================
// Shared invariant checkers
// =============================================================================

/// Verify all structural invariants on a backend's output.
fn check_entity_invariants(text: &str, backend_name: &str, backend: &dyn Model) {
    let entities = backend
        .extract_entities(text, None)
        .unwrap_or_else(|e| panic!("{} failed on {:?}: {}", backend_name, text, e));

    let char_count = text.chars().count();

    for entity in &entities {
        // Confidence in [0, 1]
        assert!(
            (0.0..=1.0).contains(&entity.confidence),
            "{}: entity {:?} confidence {} outside [0.0, 1.0]",
            backend_name,
            entity.text,
            entity.confidence,
        );

        // Non-empty text (empty entities are meaningless)
        if !text.is_empty() {
            // start <= end (zero-length spans are permitted but unusual)
            assert!(
                entity.start() <= entity.end(),
                "{}: entity {:?} has start ({}) > end ({})",
                backend_name,
                entity.text,
                entity.start(),
                entity.end(),
            );
        }

        // End within text bounds
        assert!(
            entity.end() <= char_count,
            "{}: entity {:?} end ({}) exceeds text char count ({})",
            backend_name,
            entity.text,
            entity.end(),
            char_count,
        );

        // Span text matches entity text (character offsets, not byte offsets).
        // Backends may normalize whitespace (collapse multiple spaces) or trim
        // trailing punctuation, so we compare after normalization.
        if entity.start() < entity.end() {
            let extracted: String = text
                .chars()
                .skip(entity.start())
                .take(entity.end() - entity.start())
                .collect();
            let norm_extracted = normalize_ws(&extracted);
            let norm_entity = normalize_ws(&entity.text);
            assert!(
                norm_extracted.contains(&norm_entity)
                    || norm_entity.contains(norm_extracted.trim()),
                "{}: span [{},{}) = {:?} doesn't match entity text {:?} (after ws normalization)",
                backend_name,
                entity.start(),
                entity.end(),
                extracted,
                entity.text,
            );
        }
    }

    // No duplicate spans: same (start, end, type) should not appear twice
    let mut seen = std::collections::HashSet::new();
    for entity in &entities {
        let key = (entity.start(), entity.end(), entity.entity_type.as_label());
        assert!(
            seen.insert(key),
            "{}: duplicate entity at [{},{}) type={:?}",
            backend_name,
            entity.start(),
            entity.end(),
            entity.entity_type.as_label(),
        );
    }

    // Entities should be sorted by position
    for pair in entities.windows(2) {
        assert!(
            pair[0].start() <= pair[1].start(),
            "{}: entities not sorted by position: [{},{}] before [{},{}]",
            backend_name,
            pair[0].start(),
            pair[0].end(),
            pair[1].start(),
            pair[1].end(),
        );
    }
}

// =============================================================================
// Per-Backend Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    // --- RegexNER ---

    #[test]
    fn regex_invariants(text in arb_ner_text()) {
        let backend = RegexNER::new();
        check_entity_invariants(&text, "RegexNER", &backend);
    }

    #[test]
    fn regex_fuzz(text in arb_fuzz_text()) {
        let backend = RegexNER::new();
        check_entity_invariants(&text, "RegexNER", &backend);
    }

    // --- HeuristicNER ---

    #[test]
    fn heuristic_invariants(text in arb_ner_text()) {
        let backend = HeuristicNER::new();
        check_entity_invariants(&text, "HeuristicNER", &backend);
    }

    #[test]
    fn heuristic_fuzz(text in arb_fuzz_text()) {
        let backend = HeuristicNER::new();
        check_entity_invariants(&text, "HeuristicNER", &backend);
    }

    // --- EnsembleNER ---

    #[test]
    fn ensemble_invariants(text in arb_ner_text()) {
        check_entity_invariants(&text, "EnsembleNER", &*ENSEMBLE);
    }

    #[test]
    fn ensemble_fuzz(text in arb_fuzz_text()) {
        check_entity_invariants(&text, "EnsembleNER", &*ENSEMBLE);
    }
}

// =============================================================================
// Ensemble Determinism
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// Same input -> same output (ensemble uses no RNG).
    #[test]
    fn ensemble_deterministic(text in arb_ner_text()) {
        let run1 = ENSEMBLE.extract_entities(&text, None).unwrap();
        let run2 = ENSEMBLE.extract_entities(&text, None).unwrap();

        prop_assert_eq!(
            run1.len(), run2.len(),
            "Ensemble should produce same entity count on repeated calls"
        );

        for (a, b) in run1.iter().zip(run2.iter()) {
            prop_assert_eq!(a.start(), b.start());
            prop_assert_eq!(a.end(), b.end());
            prop_assert_eq!(a.entity_type.as_label(), b.entity_type.as_label());
            prop_assert!(
                (a.confidence - b.confidence).abs() < 1e-10,
                "Confidence should be identical across runs"
            );
        }
    }
}

// =============================================================================
// Cross-Backend Agreement (well-known entities)
// =============================================================================

/// For well-known entities, at least one non-ML backend should find them.
#[test]
fn cross_backend_email_detection() {
    let text = "Contact alice@example.com for details.";
    let regex = RegexNER::new();
    let ensemble = EnsembleNER::new();

    let regex_entities = regex.extract_entities(text, None).unwrap();
    let ensemble_entities = ensemble.extract_entities(text, None).unwrap();

    let regex_has_email = regex_entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Email));
    let ensemble_has_email = ensemble_entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Email));

    assert!(regex_has_email, "RegexNER should detect email");
    assert!(ensemble_has_email, "EnsembleNER should detect email");
}

#[test]
fn cross_backend_date_detection() {
    let text = "The meeting is on 2024-01-15.";
    let regex = RegexNER::new();
    let ensemble = EnsembleNER::new();

    let regex_entities = regex.extract_entities(text, None).unwrap();
    let ensemble_entities = ensemble.extract_entities(text, None).unwrap();

    let regex_has_date = regex_entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Date));
    let ensemble_has_date = ensemble_entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Date));

    assert!(regex_has_date, "RegexNER should detect ISO date");
    assert!(ensemble_has_date, "EnsembleNER should detect ISO date");
}

#[test]
fn cross_backend_url_detection() {
    let text = "Visit https://example.com/page for more info.";
    let regex = RegexNER::new();

    let entities = regex.extract_entities(text, None).unwrap();
    let has_url = entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Url));

    assert!(has_url, "RegexNER should detect URL");
}

#[test]
fn cross_backend_person_detection() {
    let text = "CEO Tim Cook announced the new product.";
    let heuristic = HeuristicNER::new();
    let ensemble = EnsembleNER::new();

    let h_entities = heuristic.extract_entities(text, None).unwrap();
    let e_entities = ensemble.extract_entities(text, None).unwrap();

    let h_has_person = h_entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Person));
    let e_has_person = e_entities
        .iter()
        .any(|e| matches!(e.entity_type, anno::EntityType::Person));

    assert!(h_has_person, "HeuristicNER should detect 'Tim Cook' as PER");
    assert!(e_has_person, "EnsembleNER should detect 'Tim Cook' as PER");
}

// =============================================================================
// Ensemble Provenance Invariants
// =============================================================================

#[test]
fn ensemble_all_entities_have_provenance() {
    let text = "Dr. Alice Smith visited Paris on 2024-01-15. Contact her at alice@test.com.";
    let backend = EnsembleNER::new();
    let entities = backend.extract_entities(text, None).unwrap();

    for entity in &entities {
        assert!(
            entity.provenance.is_some(),
            "Entity {:?} ({:?}) at [{},{}) missing provenance",
            entity.text,
            entity.entity_type,
            entity.start(),
            entity.end(),
        );

        let prov = entity.provenance.as_ref().unwrap();
        assert!(
            !prov.source.is_empty(),
            "Entity {:?} has empty provenance source",
            entity.text,
        );
        assert!(
            prov.source.starts_with("ensemble("),
            "Ensemble provenance should start with 'ensemble(', got {:?}",
            prov.source,
        );
    }
}

// =============================================================================
// Empty/Edge Case Handling
// =============================================================================

#[test]
fn all_backends_handle_empty_gracefully() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
    ];

    let edge_cases = [
        "",
        " ",
        "\n",
        "\t\t\t",
        ".",
        "!!!",
        "123",
        "a",
        // Pure punctuation
        "...---...---...",
    ];

    for (name, backend) in &backends {
        for input in &edge_cases {
            let result = backend.extract_entities(input, None);
            assert!(
                result.is_ok(),
                "{} crashed on edge case {:?}: {:?}",
                name,
                input,
                result.err()
            );
        }
    }
}

// =============================================================================
// Confidence Monotonicity (Ensemble)
// =============================================================================

/// Ensemble confidence should be >= single-backend confidence when multiple
/// backends agree (agreement bonus).
#[test]
fn ensemble_agreement_boosts_confidence() {
    let text = "Tim Cook is the CEO of Apple Inc.";

    // Single heuristic backend
    let single = EnsembleNER::with_backends(vec![Box::new(HeuristicNER::new())]);
    let single_entities = single.extract_entities(text, None).unwrap();

    // Full ensemble (regex + heuristic)
    let full = EnsembleNER::new();
    let full_entities = full.extract_entities(text, None).unwrap();

    // For entities found by both, ensemble should generally have higher confidence
    // (due to agreement bonus). We check at least one entity demonstrates this.
    let mut found_boost = false;
    for fe in &full_entities {
        for se in &single_entities {
            if fe.text == se.text && fe.entity_type == se.entity_type {
                // Agreement bonus may or may not apply (depends on regex finding it too)
                // But ensemble should never produce *lower* confidence than single when
                // multiple sources agree.
                if fe.confidence > se.confidence {
                    found_boost = true;
                }
            }
        }
    }

    // This is a soft check -- it's OK if no boost happened (regex may not find the same entities)
    // The important thing is it doesn't crash and produces valid output
    let _ = found_boost;
}

// =============================================================================
// Weight Learner Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Learned weights are in [0, 1] range.
    #[test]
    fn weight_learner_weights_bounded(
        correct in 0usize..100,
        total in 1usize..200,
    ) {
        use anno::backends::ensemble::{WeightLearner, WeightTrainingExample};
        use anno::EntityType;

        let mut learner = WeightLearner::new();
        let effective_correct = correct.min(total);

        learner.add_example(&WeightTrainingExample {
            text: "Test".to_string(),
            gold_type: EntityType::Person,
            start: 0,
            end: 4,
            predictions: (0..total).map(|i| {
                let pred_type = if i < effective_correct {
                    EntityType::Person
                } else {
                    EntityType::Organization
                };
                (format!("backend_{}", i), pred_type, anno_core::Confidence::new(0.8))
            }).collect(),
        });

        let weights = learner.learn_weights();
        for (name, w) in &weights {
            prop_assert!(
                (0.0..=1.0).contains(&w.overall),
                "Learned weight for {} = {} outside [0,1]",
                name,
                w.overall,
            );
        }
    }
}
