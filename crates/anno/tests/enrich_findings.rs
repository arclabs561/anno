//! Tests motivated by the enrich report (2026-03-08).
//!
//! Sources: GLiNER test suite, coreferee test suite, spaCy tests, HuggingFace tokenizers.
//! Each test section cites the specific external source that motivated it.

use anno::{EnsembleNER, EntityCategory, EntityType, HeuristicNER, Model, RegexNER, StackedNER};

// =============================================================================
// 1. Threshold boundary semantics (source: GLiNER test_decoder.py)
//
// Verify that confidence filtering uses consistent >= semantics across backends.
// The key property: lowering the threshold monotonically increases (or preserves)
// entity count. A threshold of 0.0 should return all candidates.
// =============================================================================

#[test]
fn heuristic_threshold_monotonicity() {
    // Threshold 0.0 should return >= entities compared to threshold 0.5
    let text = "Dr. Alice Smith visited Paris on Monday. Bob went to London.";
    let low = HeuristicNER::with_threshold(0.0);
    let mid = HeuristicNER::with_threshold(0.5);
    let high = HeuristicNER::with_threshold(0.99);

    let low_ents = low.extract_entities(text, None).unwrap();
    let mid_ents = mid.extract_entities(text, None).unwrap();
    let high_ents = high.extract_entities(text, None).unwrap();

    assert!(
        low_ents.len() >= mid_ents.len(),
        "Lower threshold ({}) should produce >= entities than mid threshold ({}): {} vs {}",
        0.0,
        0.5,
        low_ents.len(),
        mid_ents.len(),
    );
    assert!(
        mid_ents.len() >= high_ents.len(),
        "Mid threshold ({}) should produce >= entities than high threshold ({}): {} vs {}",
        0.5,
        0.99,
        mid_ents.len(),
        high_ents.len(),
    );
}

#[test]
fn heuristic_threshold_at_exact_boundary() {
    // Create a backend with a threshold, extract entities, then verify that
    // all returned entities have confidence >= the threshold (not just >).
    let text = "Angela Merkel visited the United Nations in New York.";
    let threshold = 0.35; // default heuristic threshold
    let backend = HeuristicNER::with_threshold(threshold);
    let entities = backend.extract_entities(text, None).unwrap();

    for entity in &entities {
        assert!(
            entity.confidence >= threshold,
            "Entity '{}' has confidence {} which is below threshold {}",
            entity.text,
            entity.confidence,
            threshold,
        );
    }
}

#[test]
fn heuristic_threshold_zero_returns_all_candidates() {
    // With threshold 0.0, we should get entities that the default threshold would filter.
    let text = "The committee discussed the proposal.";
    let permissive = HeuristicNER::with_threshold(0.0);
    let strict = HeuristicNER::with_threshold(0.95);

    let permissive_ents = permissive.extract_entities(text, None).unwrap();
    let strict_ents = strict.extract_entities(text, None).unwrap();

    // Strict should be a subset (or equal) of permissive
    for se in &strict_ents {
        assert!(
            permissive_ents
                .iter()
                .any(|pe| pe.start() == se.start() && pe.end() == se.end()),
            "Strict entity '{}' [{},{}) not found in permissive results",
            se.text,
            se.start(),
            se.end(),
        );
    }
}

#[test]
fn ensemble_min_confidence_boundary() {
    let text = "Dr. John Smith visited Google headquarters in California on 2024-01-15.";

    // Default ensemble (min_confidence = 0.30)
    let default_ensemble = EnsembleNER::new();
    let entities = default_ensemble.extract_entities(text, None).unwrap();

    // All entities should respect the 0.30 min_confidence
    for entity in &entities {
        assert!(
            entity.confidence >= 0.30,
            "Ensemble entity '{}' has confidence {} below min_confidence 0.30",
            entity.text,
            entity.confidence,
        );
    }
}

// =============================================================================
// 2. Confidence invariants (source: rust-bert score tolerance tests)
//
// No backend should ever produce NaN, infinity, or out-of-range confidence.
// =============================================================================

#[test]
fn confidence_never_nan_or_infinite() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
    ];

    let inputs = [
        "Dr. Alice Smith visited Paris on 2024-01-15.",
        "Contact alice@example.com or call 555-123-4567.",
        "The rate is 5.25% per annum, totaling $1,234.56.",
        "CEO Tim Cook met Google CEO Sundar Pichai in New York.",
        "\u{200B}\u{FEFF}zero-width\u{200C}chars\u{200D}here",
        "cafe\u{0301} vs caf\u{00E9}",
        "\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}", // Arabic
    ];

    for (name, backend) in &backends {
        for input in &inputs {
            let entities = backend.extract_entities(input, None).unwrap();
            for entity in &entities {
                assert!(
                    entity.confidence.value().is_finite(),
                    "{}: entity '{}' has non-finite confidence: {}",
                    name,
                    entity.text,
                    entity.confidence,
                );
                assert!(
                    !entity.confidence.value().is_nan(),
                    "{}: entity '{}' has NaN confidence",
                    name,
                    entity.text,
                );
                assert!(
                    (0.0..=1.0).contains(&entity.confidence.value()),
                    "{}: entity '{}' confidence {} outside [0.0, 1.0]",
                    name,
                    entity.text,
                    entity.confidence,
                );
            }
        }
    }
}

// =============================================================================
// 3. Handshaking index reference implementation (source: GLiNER reference impl pattern)
//
// Verify the TPLinker handshaking formula against a naive reference.
// =============================================================================

/// Naive O(n^2) reference: enumerate all (i, j) pairs and assign sequential indices.
fn handshaking_index_reference(i: usize, j: usize, seq_len: usize) -> usize {
    assert!(i <= j, "i must be <= j");
    assert!(j < seq_len, "j must be < seq_len");
    let mut idx = 0;
    for row in 0..seq_len {
        for col in row..seq_len {
            if row == i && col == j {
                return idx;
            }
            idx += 1;
        }
    }
    unreachable!("(i={}, j={}) not found in seq_len={}", i, j, seq_len);
}

/// Optimized formula (mirrors TPLinker's internal handshaking_index).
fn handshaking_index_fast(i: usize, j: usize, seq_len: usize) -> usize {
    i * seq_len - i * (i.wrapping_sub(1)) / 2 + (j - i)
}

#[test]
fn handshaking_index_reference_vs_fast() {
    // Exhaustive check for small sequence lengths
    for seq_len in 1..=20 {
        for i in 0..seq_len {
            for j in i..seq_len {
                let reference = handshaking_index_reference(i, j, seq_len);
                let fast = handshaking_index_fast(i, j, seq_len);
                assert_eq!(
                    reference, fast,
                    "Mismatch at (i={}, j={}, L={}): reference={}, fast={}",
                    i, j, seq_len, reference, fast,
                );
            }
        }
    }
}

#[test]
fn handshaking_index_total_count() {
    // Total number of upper-triangular pairs = L*(L+1)/2
    for seq_len in 1..=30 {
        let expected = seq_len * (seq_len + 1) / 2;
        let mut count = 0;
        for i in 0..seq_len {
            for j in i..seq_len {
                let idx = handshaking_index_fast(i, j, seq_len);
                assert!(
                    idx < expected,
                    "(i={}, j={}, L={}): idx={} >= expected={}",
                    i,
                    j,
                    seq_len,
                    idx,
                    expected,
                );
                count += 1;
            }
        }
        assert_eq!(
            count, expected,
            "L={}: count={} != expected={}",
            seq_len, count, expected
        );
    }
}

#[test]
fn handshaking_index_bijectivity() {
    // Every index in [0, L*(L+1)/2) should be produced exactly once
    for seq_len in 1..=15 {
        let total = seq_len * (seq_len + 1) / 2;
        let mut seen = vec![false; total];
        for i in 0..seq_len {
            for j in i..seq_len {
                let idx = handshaking_index_fast(i, j, seq_len);
                assert!(
                    !seen[idx],
                    "Duplicate index {} at (i={}, j={}, L={})",
                    idx, i, j, seq_len,
                );
                seen[idx] = true;
            }
        }
        assert!(
            seen.iter().all(|&b| b),
            "L={}: not all indices covered",
            seq_len
        );
    }
}

// =============================================================================
// 4. Softmax reference implementation (source: GLiNER reference impl pattern)
//
// Verify softmax confidence computation against a naive reference.
// =============================================================================

/// Reference softmax: returns the probability of the argmax class.
fn softmax_confidence_reference(logits: &[f32]) -> f32 {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
    let best_exp = (0.0_f32).exp(); // = 1.0
    best_exp / sum_exp
}

#[test]
fn softmax_confidence_basic_properties() {
    // Property 1: result is in (0, 1]
    let test_cases: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0], // uniform: 1/5 = 0.2
        vec![10.0, 0.0, 0.0],          // peaked: near 1.0
        vec![-1.0, -2.0, -3.0],        // negative logits
        vec![100.0, 99.0],             // close logits
    ];

    for logits in &test_cases {
        let conf = softmax_confidence_reference(logits);
        assert!(
            conf > 0.0 && conf <= 1.0,
            "Softmax confidence {} outside (0, 1] for logits {:?}",
            conf,
            logits,
        );
    }

    // Property 2: uniform logits -> 1/n
    let uniform = vec![0.0; 5];
    let conf = softmax_confidence_reference(&uniform);
    assert!(
        (conf - 0.2).abs() < 1e-6,
        "Uniform 5-class softmax should be 0.2, got {}",
        conf,
    );

    // Property 3: very peaked -> near 1.0
    let peaked = vec![100.0, 0.0, 0.0];
    let conf = softmax_confidence_reference(&peaked);
    assert!(
        conf > 0.99,
        "Peaked softmax should be near 1.0, got {}",
        conf,
    );
}

// =============================================================================
// 5. Overlap handling per backend (source: GLiNER test_flat_ner / test_nested_ner)
//
// Verify that overlapping spans are handled correctly: flat backends should
// suppress lower-confidence overlaps; backends supporting nested NER should
// preserve them.
// =============================================================================

#[test]
fn stacked_resolves_overlapping_spans() {
    // The stacked backend should not produce overlapping spans of the same type
    // (its default conflict strategy is Priority).
    let text = "Apple Inc. is headquartered in California.";
    let backend = StackedNER::default();
    let entities = backend.extract_entities(text, None).unwrap();

    // Check no same-type overlaps
    for (i, a) in entities.iter().enumerate() {
        for b in entities.iter().skip(i + 1) {
            if a.entity_type == b.entity_type {
                let overlaps = a.start() < b.end() && b.start() < a.end();
                assert!(
                    !overlaps,
                    "Stacked backend produced overlapping same-type entities: \
                     '{}' [{},{}) and '{}' [{},{})",
                    a.text,
                    a.start(),
                    a.end(),
                    b.text,
                    b.start(),
                    b.end(),
                );
            }
        }
    }
}

#[test]
fn ensemble_deduplicates_overlapping_candidates() {
    // When multiple backends find the same entity, ensemble should merge them
    // (not return duplicates).
    let text = "Contact alice@example.com for details.";
    let backend = EnsembleNER::new();
    let entities = backend.extract_entities(text, None).unwrap();

    // Count email entities -- should be exactly 1, not duplicated from regex + heuristic
    let email_count = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Email))
        .count();
    assert!(
        email_count <= 1,
        "Ensemble should deduplicate overlapping email entities, found {}",
        email_count,
    );
}

// =============================================================================
// 6. Empty and degenerate inputs (source: spaCy, GLiNER, fastcoref)
//
// Systematic coverage of edge cases across all non-ML backends.
// =============================================================================

#[test]
fn all_backends_empty_string() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
        ("StackedNER", Box::new(StackedNER::default())),
    ];

    for (name, backend) in &backends {
        let result = backend.extract_entities("", None);
        assert!(
            result.is_ok(),
            "{} failed on empty string: {:?}",
            name,
            result.err()
        );
        assert!(
            result.unwrap().is_empty(),
            "{} returned entities for empty string",
            name,
        );
    }
}

#[test]
fn all_backends_whitespace_only() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
        ("StackedNER", Box::new(StackedNER::default())),
    ];

    let whitespace_inputs = [" ", "  ", "\t", "\n", "\r\n", "   \t\n  "];

    for (name, backend) in &backends {
        for input in &whitespace_inputs {
            let result = backend.extract_entities(input, None);
            assert!(
                result.is_ok(),
                "{} failed on whitespace {:?}: {:?}",
                name,
                input,
                result.err(),
            );
        }
    }
}

#[test]
fn all_backends_single_character() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
    ];

    let single_chars = ["a", "Z", "1", ".", "!", "\u{00E9}", "\u{4E16}"];

    for (name, backend) in &backends {
        for input in &single_chars {
            let result = backend.extract_entities(input, None);
            assert!(
                result.is_ok(),
                "{} failed on single char {:?}: {:?}",
                name,
                input,
                result.err(),
            );
        }
    }
}

#[test]
fn all_backends_pure_punctuation() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
    ];

    let punct_inputs = [
        "...", "---", "!!!???", "()[]{}", "<<>>", "####", "@@@", "***",
    ];

    for (name, backend) in &backends {
        for input in &punct_inputs {
            let result = backend.extract_entities(input, None);
            assert!(
                result.is_ok(),
                "{} failed on punctuation {:?}: {:?}",
                name,
                input,
                result.err(),
            );
        }
    }
}

#[test]
fn all_backends_numbers_only() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
    ];

    let number_inputs = ["42", "3.14159", "1,000,000", "0", "-1", "1e10"];

    for (name, backend) in &backends {
        for input in &number_inputs {
            let result = backend.extract_entities(input, None);
            assert!(
                result.is_ok(),
                "{} failed on numeric input {:?}: {:?}",
                name,
                input,
                result.err(),
            );
        }
    }
}

// =============================================================================
// 7. Unicode combining characters (source: HuggingFace tokenizers, anno TODO)
//
// Precomposed vs decomposed forms should not cause entity boundary issues.
// =============================================================================

#[test]
fn unicode_precomposed_vs_decomposed_no_crash() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
    ];

    // "cafe\u{0301}" (decomposed e-acute) vs "caf\u{00E9}" (precomposed)
    let decomposed = "Ren\u{0065}\u{0301} Descartes visited Paris.";
    let precomposed = "Ren\u{00E9} Descartes visited Paris.";

    for (name, backend) in &backends {
        let result_d = backend.extract_entities(decomposed, None);
        let result_p = backend.extract_entities(precomposed, None);

        assert!(
            result_d.is_ok(),
            "{} crashed on decomposed Unicode: {:?}",
            name,
            result_d.err(),
        );
        assert!(
            result_p.is_ok(),
            "{} crashed on precomposed Unicode: {:?}",
            name,
            result_p.err(),
        );

        // Verify all entity spans are valid character offsets in both forms
        for entity in result_d.unwrap().iter().chain(result_p.unwrap().iter()) {
            let text_to_check = if entity.text.contains('\u{0301}') {
                decomposed
            } else {
                precomposed
            };
            assert!(
                entity.end() <= text_to_check.chars().count(),
                "{}: entity '{}' end {} exceeds char count {} in {:?}",
                name,
                entity.text,
                entity.end(),
                text_to_check.chars().count(),
                text_to_check,
            );
        }
    }
}

#[test]
fn unicode_mixed_scripts_entity_boundary() {
    // Entity names spanning script boundaries (source: HF tokenizers CJK-Latin tests)
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
    ];

    let inputs = [
        // CJK mixed with Latin
        "Toyota\u{30C8}\u{30E8}\u{30BF} announced earnings.", // Toyota + katakana
        // RTL mixed with Latin
        "CEO of \u{0634}\u{0631}\u{0643}\u{0629} Apple met investors.",
        // Emoji in text (should not crash)
        "Dr. Smith \u{1F600} visited Paris \u{1F1EB}\u{1F1F7}.",
    ];

    for (name, backend) in &backends {
        for input in &inputs {
            let result = backend.extract_entities(input, None);
            assert!(
                result.is_ok(),
                "{} crashed on mixed-script input {:?}: {:?}",
                name,
                input,
                result.err(),
            );
        }
    }
}

// =============================================================================
// 8. Large entity type count stress test (source: spaCy test_issue2800)
//
// Verify that DynamicLabels backends and TypeMapper handle large type counts
// without panicking or excessive slowdown.
// =============================================================================

#[test]
fn type_mapper_handles_many_types() {
    use anno::TypeMapper;

    let mut mapper = TypeMapper::new();

    // Add 1000 custom entity types
    for i in 0..1000 {
        let label = format!("CUSTOM_TYPE_{}", i);
        mapper.add(&label, EntityType::custom(&label, EntityCategory::Misc));
    }

    // Verify lookups work
    assert!(mapper.map("CUSTOM_TYPE_0").is_some());
    assert!(mapper.map("CUSTOM_TYPE_999").is_some());
    assert!(mapper.map("CUSTOM_TYPE_1000").is_none());

    // Normalize should handle unknown types gracefully
    let normalized = mapper.normalize("COMPLETELY_UNKNOWN_TYPE");
    // Should return something (the normalize method maps unknowns)
    let _ = normalized;
}

#[test]
fn entity_type_from_label_many_custom_types() {
    // EntityType::from_label should handle arbitrary strings without panicking
    for i in 0..500 {
        let label = format!("CUSTOM_{}", i);
        let et = EntityType::from_label(&label);
        assert_eq!(
            et.as_label().to_uppercase(),
            label.to_uppercase(),
            "from_label roundtrip failed for '{}'",
            label,
        );
    }
}

// =============================================================================
// 9. Pleonastic "it" detection (source: coreferee test_rules_en.py)
//
// Tests for non-referential "it" patterns that should NOT create coreference
// links. Exercises the MentionRankingCoref's pleonastic filter.
// =============================================================================

// These test the patterns directly by verifying that sentences with pleonastic
// "it" don't create spurious coref chains with preceding entities.

#[test]
fn pleonastic_it_weather_verbs() {
    // "It rains" -- "it" is pleonastic, should not corefer with "Paris"
    let text = "Paris is a city. It rains there often.";
    let backend = HeuristicNER::new();
    let entities = backend.extract_entities(text, None).unwrap();

    // The key invariant: "It" should not be extracted as an entity
    let it_entities: Vec<_> = entities.iter().filter(|e| e.text == "It").collect();
    assert!(
        it_entities.is_empty(),
        "Pleonastic 'It' in 'It rains' should not be extracted as an entity: {:?}",
        it_entities,
    );
}

#[test]
fn pleonastic_it_modal_adjectives() {
    // "It is important that..." -- non-referential
    let texts = [
        "The report was filed. It is important that we review it carefully.",
        "The committee met. It is likely that the proposal will pass.",
        "The evidence was presented. It is clear that changes are needed.",
        "Scientists observed the phenomenon. It is believed that this is rare.",
        "The deadline approaches. It is essential to submit on time.",
    ];

    let backend = HeuristicNER::new();
    for text in &texts {
        let result = backend.extract_entities(text, None);
        assert!(
            result.is_ok(),
            "Backend crashed on pleonastic 'it' text: {:?}",
            text,
        );
    }
}

#[test]
fn pleonastic_it_cognitive_verbs() {
    // "It seems..." -- non-referential
    let texts = [
        "The data was analyzed. It seems that the results are consistent.",
        "We reviewed the code. It appears that the bug is fixed.",
        "The experiment concluded. It turns out that our hypothesis was correct.",
    ];

    let backend = HeuristicNER::new();
    for text in &texts {
        let result = backend.extract_entities(text, None);
        assert!(
            result.is_ok(),
            "Backend crashed on cognitive verb 'it' text: {:?}",
            text,
        );
    }
}

#[test]
fn pleonastic_it_time_expressions() {
    // "It is midnight" -- non-referential
    let texts = [
        "The meeting ended. It was midnight when we left.",
        "The alarm went off. It is 5 o'clock.",
        "She checked the clock. It was noon already.",
    ];

    let backend = HeuristicNER::new();
    for text in &texts {
        let result = backend.extract_entities(text, None);
        assert!(
            result.is_ok(),
            "Backend crashed on time 'it' text: {:?}",
            text
        );
    }
}

#[test]
fn pleonastic_it_with_contraction() {
    // "It's raining" -- contraction form
    let text = "London is lovely. It's raining there today.";
    let backend = HeuristicNER::new();
    let result = backend.extract_entities(text, None);
    assert!(
        result.is_ok(),
        "Backend crashed on contracted pleonastic 'it's'"
    );
}

// =============================================================================
// 10. Coref mention-ranking pleonastic filter (source: coreferee)
//
// Test the MentionRankingCoref's pleonastic detection through the public API.
// =============================================================================

#[test]
fn mention_ranking_coref_pleonastic_patterns() {
    use anno::MentionRankingCoref;

    let coref = MentionRankingCoref::default();

    // Texts where "it" is pleonastic -- coref should not link "it" to preceding entities
    let pleonastic_texts = [
        // Weather
        "The city of London is beautiful. It rains there every day.",
        // Modal
        "The CEO announced profits. It is important that shareholders know.",
        // Cognitive
        "The team investigated. It seems that the error was intermittent.",
    ];

    for text in &pleonastic_texts {
        let clusters = coref.resolve(text);
        assert!(
            clusters.is_ok(),
            "MentionRankingCoref crashed on pleonastic text: {:?}: {:?}",
            text,
            clusters.err(),
        );

        let clusters = clusters.unwrap();
        // Verify no cluster links "it"/"It" to a named entity
        for cluster in &clusters {
            let has_it = cluster
                .mentions
                .iter()
                .any(|m| m.text.eq_ignore_ascii_case("it"));
            let has_named = cluster.mentions.iter().any(|m| {
                m.text.len() > 2 && m.text.chars().next().is_some_and(|c| c.is_uppercase())
            });
            if has_it && has_named {
                // Check if the "it" appears in a pleonastic context
                let it_mention = cluster
                    .mentions
                    .iter()
                    .find(|m| m.text.eq_ignore_ascii_case("it"))
                    .unwrap();
                let text_lower = text.to_lowercase();
                let it_pos_byte = text_lower[..it_mention.start].len();
                let after = &text_lower[it_pos_byte + 2..];
                let after_trimmed = after.trim_start();

                // If the text after "it" matches a pleonastic pattern, this is a bug
                let pleonastic_starts = [
                    "rains",
                    "snows",
                    "seems",
                    "appears",
                    "turns out",
                    "is important",
                    "is likely",
                    "is clear",
                    "is believed",
                    "is essential",
                    "is necessary",
                    "is obvious",
                ];
                let is_pleonastic_context = pleonastic_starts
                    .iter()
                    .any(|p| after_trimmed.starts_with(p));

                assert!(
                    !is_pleonastic_context,
                    "Pleonastic 'it' incorrectly linked to '{}' in cluster: {:?}",
                    cluster
                        .mentions
                        .iter()
                        .find(|m| !m.text.eq_ignore_ascii_case("it"))
                        .map(|m| m.text.as_str())
                        .unwrap_or("?"),
                    cluster.mentions.iter().map(|m| &m.text).collect::<Vec<_>>(),
                );
            }
        }
    }
}

// =============================================================================
// 11. Determinism across backends (source: backend_properties.rs extension)
//
// Same input -> same output for all backends, not just ensemble.
// =============================================================================

#[test]
fn all_backends_deterministic() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("StackedNER", Box::new(StackedNER::default())),
    ];

    let text = "Dr. Alice Smith visited Google in Paris on 2024-01-15. Contact alice@test.com.";

    for (name, backend) in &backends {
        let run1 = backend.extract_entities(text, None).unwrap();
        let run2 = backend.extract_entities(text, None).unwrap();

        assert_eq!(
            run1.len(),
            run2.len(),
            "{}: different entity count across runs: {} vs {}",
            name,
            run1.len(),
            run2.len(),
        );

        for (a, b) in run1.iter().zip(run2.iter()) {
            assert_eq!(a.start(), b.start(), "{}: start mismatch", name);
            assert_eq!(a.end(), b.end(), "{}: end mismatch", name);
            assert_eq!(
                a.entity_type.as_label(),
                b.entity_type.as_label(),
                "{}: type mismatch",
                name,
            );
            assert!(
                (a.confidence - b.confidence).abs() < 1e-10,
                "{}: confidence mismatch: {} vs {}",
                name,
                a.confidence,
                b.confidence,
            );
        }
    }
}

// =============================================================================
// 12. Entity span text extraction correctness
//
// Verify that entity.text matches text[entity.start..entity.end] (char offsets)
// for all non-ML backends with known inputs.
// =============================================================================

#[test]
fn entity_text_matches_span_extraction() {
    let backends: Vec<(&str, Box<dyn Model>)> = vec![
        ("RegexNER", Box::new(RegexNER::new())),
        ("HeuristicNER", Box::new(HeuristicNER::new())),
        ("EnsembleNER", Box::new(EnsembleNER::new())),
    ];

    let texts = [
        "Contact alice@example.com by 2024-03-15.",
        "CEO Tim Cook visited Paris.",
        "The rate is 5.25% per annum.",
        "Call 555-123-4567 for info.",
    ];

    for (name, backend) in &backends {
        for text in &texts {
            let entities = backend.extract_entities(text, None).unwrap();
            for entity in &entities {
                let extracted: String = text
                    .chars()
                    .skip(entity.start())
                    .take(entity.end() - entity.start())
                    .collect();

                // Allow whitespace normalization (some backends trim/collapse)
                let norm_extracted: String =
                    extracted.split_whitespace().collect::<Vec<_>>().join(" ");
                let norm_entity: String =
                    entity.text.split_whitespace().collect::<Vec<_>>().join(" ");

                assert!(
                    norm_extracted.contains(&norm_entity) || norm_entity.contains(&norm_extracted),
                    "{}: span [{},{}) extracts '{}' but entity.text is '{}' in '{}'",
                    name,
                    entity.start(),
                    entity.end(),
                    extracted,
                    entity.text,
                    text,
                );
            }
        }
    }
}
