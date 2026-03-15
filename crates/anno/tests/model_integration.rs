//! Integration tests for NER + coref model backends.
//!
//! These tests exercise the full pipeline: NER extraction -> coref resolution.
//! They require model downloads and the `onnx` feature flag.
//!
//! Run with:
//!   cargo test -p anno-lib --features "onnx analysis" --test model_integration -- --ignored

#![cfg(all(feature = "onnx", feature = "analysis"))]

use anno::eval::coref_resolver::SimpleCorefResolver;
use anno::{EntityType, Model, StackedNER};

// =============================================================================
// NER -> Coref Pipeline
// =============================================================================

/// Full pipeline: stacked NER -> coref resolution.
/// Verifies that NER entities feed correctly into coref chains.
#[test]
#[ignore]
fn stacked_ner_to_coref_pipeline() {
    let text = "Marie Curie discovered radium. She won two Nobel Prizes. \
                Curie was born in Poland.";

    let ner = StackedNER::default();
    let entities = ner
        .extract_entities(text, None)
        .expect("NER should succeed");

    // Should find at least Marie Curie
    let person_entities: Vec<_> = entities
        .iter()
        .filter(|e| matches!(e.entity_type, EntityType::Person))
        .collect();
    assert!(
        !person_entities.is_empty(),
        "Should find person entities, got: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    // Feed into coref resolver
    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Should produce at least one chain
    assert!(
        !chains.is_empty(),
        "Coref should produce at least one chain from {:?}",
        entities.iter().map(|e| &e.text).collect::<Vec<_>>()
    );

    // Marie Curie chain should have > 1 mention (at least "Marie Curie" + "Curie")
    let curie_chain = chains.iter().find(|c| {
        c.mentions
            .iter()
            .any(|m| m.text.contains("Curie") || m.text.contains("Marie"))
    });
    assert!(
        curie_chain.is_some(),
        "Should have a chain containing 'Curie'"
    );
    let curie_chain = curie_chain.unwrap();
    assert!(
        curie_chain.len() >= 2,
        "Curie chain should have >= 2 mentions, got {}: {:?}",
        curie_chain.len(),
        curie_chain
            .mentions
            .iter()
            .map(|m| &m.text)
            .collect::<Vec<_>>()
    );
}

/// Multi-entity pipeline: verifies separate coref chains for different entities.
#[test]
#[ignore]
fn multi_entity_coref_separation() {
    let text = "Tim Cook leads Apple. Sundar Pichai runs Google. \
                Cook announced new products. Pichai presented AI features.";

    let ner = StackedNER::default();
    let entities = ner
        .extract_entities(text, None)
        .expect("NER should succeed");

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Should have at least 2 chains (Cook and Pichai should not merge)
    let cook_chain = chains
        .iter()
        .find(|c| c.mentions.iter().any(|m| m.text.contains("Cook")));
    let pichai_chain = chains
        .iter()
        .find(|c| c.mentions.iter().any(|m| m.text.contains("Pichai")));

    if let (Some(cc), Some(pc)) = (cook_chain, pichai_chain) {
        // They should be in different chains
        let cook_mentions: Vec<&str> = cc.mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            !cook_mentions.iter().any(|m| m.contains("Pichai")),
            "Cook and Pichai should be in separate chains. Cook chain: {:?}",
            cook_mentions
        );
        let pichai_mentions: Vec<&str> = pc.mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            !pichai_mentions.iter().any(|m| m.contains("Cook")),
            "Pichai chain should not contain Cook. Pichai chain: {:?}",
            pichai_mentions
        );
    }
}

// =============================================================================
// NER Entity Structural Invariants (with real models)
// =============================================================================

/// All entities from stacked NER have valid char offsets and extractable text.
#[test]
#[ignore]
fn stacked_ner_entity_offsets_are_char_based() {
    let texts = [
        "Dr. Angela Merkel visited the United Nations in New York on 2024-01-15.",
        "Contact alice@example.com or call +1-555-123-4567.",
        "The EU invested EUR 500 million in renewable energy.",
        // Unicode: ensure char offsets, not byte offsets
        "Li Wei (\u{674E}\u{4F1F}) met with \u{5B89}\u{500D}\u{664B}\u{4E09} in Tokyo.",
    ];

    let ner = StackedNER::default();

    for text in &texts {
        let char_count = text.chars().count();
        let entities = ner
            .extract_entities(text, None)
            .expect("NER should succeed");

        for entity in &entities {
            // start < end
            assert!(
                entity.start() < entity.end(),
                "Entity {:?}: start ({}) must be < end ({})",
                entity.text,
                entity.start(),
                entity.end()
            );

            // Within bounds (char count, not byte count)
            assert!(
                entity.end() <= char_count,
                "Entity {:?}: end ({}) exceeds char count ({})",
                entity.text,
                entity.end(),
                char_count
            );

            // Confidence in [0, 1]
            assert!(
                (0.0..=1.0).contains(&entity.confidence),
                "Entity {:?}: confidence {} outside [0, 1]",
                entity.text,
                entity.confidence
            );

            // Extractable text matches
            let extracted: String = text
                .chars()
                .skip(entity.start())
                .take(entity.end() - entity.start())
                .collect();
            // Allow whitespace normalization
            let norm_extracted = extracted.split_whitespace().collect::<Vec<_>>().join(" ");
            let norm_entity = entity.text.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                norm_extracted.contains(&norm_entity) || norm_entity.contains(&norm_extracted),
                "Span [{},{}) = {:?} doesn't match entity text {:?}",
                entity.start(),
                entity.end(),
                extracted,
                entity.text
            );
        }
    }
}

/// Stacked NER handles empty/whitespace input without crashing.
#[test]
#[ignore]
fn stacked_ner_edge_cases() {
    let ner = StackedNER::default();

    let edge_cases = ["", " ", "\n\n", "\t", ".", "!!!", "123 456"];
    for input in &edge_cases {
        let result = ner.extract_entities(input, None);
        assert!(
            result.is_ok(),
            "StackedNER crashed on {:?}: {:?}",
            input,
            result.err()
        );
    }
}

// =============================================================================
// Ensemble NER (with ONNX GLiNER)
// =============================================================================

/// Ensemble with GLiNER backend produces richer entities than heuristic alone.
#[test]
#[ignore]
fn ensemble_with_gliner_produces_entities() {
    let ner = anno::EnsembleNER::new();
    let text = "Barack Obama spoke at Harvard University in Cambridge, Massachusetts.";
    let entities = ner
        .extract_entities(text, None)
        .expect("Ensemble should succeed");

    assert!(
        !entities.is_empty(),
        "Ensemble (with GLiNER) should find entities in well-formed English text"
    );

    // Should find at least a person or organization
    let has_named_entity = entities.iter().any(|e| {
        matches!(
            e.entity_type,
            EntityType::Person | EntityType::Organization | EntityType::Location
        )
    });
    assert!(
        has_named_entity,
        "Should find at least one PER/ORG/LOC, got: {:?}",
        entities
            .iter()
            .map(|e| (&e.text, &e.entity_type))
            .collect::<Vec<_>>()
    );

    // All should have provenance
    for entity in &entities {
        assert!(
            entity.provenance.is_some(),
            "Entity {:?} missing provenance",
            entity.text
        );
    }
}

// =============================================================================
// Coref Chain Structural Invariants
// =============================================================================

/// Coref chains from real NER output satisfy structural invariants.
#[test]
#[ignore]
fn coref_chain_structural_invariants() {
    let text = "John Smith works at Microsoft. He is a senior engineer. \
                Smith joined the company in 2015. He leads the AI team.";

    let ner = StackedNER::default();
    let entities = ner
        .extract_entities(text, None)
        .expect("NER should succeed");

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    for chain in &chains {
        // Non-empty
        assert!(!chain.is_empty(), "Chain should not be empty");

        // All mentions have valid text
        for mention in &chain.mentions {
            assert!(!mention.text.is_empty(), "Mention text should not be empty");
        }

        // Canonical mention exists
        let canonical = chain.canonical_mention();
        assert!(
            canonical.is_some(),
            "Chain should have a canonical mention: {:?}",
            chain.mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
        );
    }
}

/// Coref resolver doesn't merge incompatible entity types.
#[test]
#[ignore]
fn coref_type_separation() {
    let text = "Apple reported strong earnings. Tim Cook presented the results. \
                The company expects continued growth.";

    let ner = StackedNER::default();
    let entities = ner
        .extract_entities(text, None)
        .expect("NER should succeed");

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Person entities (Tim Cook) should not be in same chain as Organization (Apple)
    for chain in &chains {
        let has_person = chain.mentions.iter().any(|m| {
            entities
                .iter()
                .any(|e| e.text == m.text && matches!(e.entity_type, EntityType::Person))
        });
        let has_org = chain.mentions.iter().any(|m| {
            entities
                .iter()
                .any(|e| e.text == m.text && matches!(e.entity_type, EntityType::Organization))
        });

        // A chain should not mix persons and organizations
        assert!(
            !(has_person && has_org),
            "Chain mixes Person and Organization: {:?}",
            chain.mentions.iter().map(|m| &m.text).collect::<Vec<_>>()
        );
    }
}

// =============================================================================
// GLiNER ONNX backend
// =============================================================================

#[test]
#[ignore]
fn gliner_onnx_basic() {
    use anno::backends::GLiNEROnnx;
    use anno::DEFAULT_GLINER_MODEL;

    let model = GLiNEROnnx::new(DEFAULT_GLINER_MODEL).expect("GLiNER model load should succeed");
    assert!(model.is_available());

    let entities = model
        .extract_entities("Tim Cook is the CEO of Apple Inc.", None)
        .expect("GLiNER extraction should succeed");

    assert!(!entities.is_empty(), "GLiNER should find entities");

    for entity in &entities {
        assert!((0.0..=1.0).contains(&entity.confidence));
        assert!(entity.start() < entity.end());
        assert!(!entity.text.is_empty());
    }
}

// =============================================================================
// NuNER ONNX backend
// =============================================================================

#[test]
#[ignore]
fn nuner_onnx_basic() {
    use anno::backends::NuNER;

    let model =
        NuNER::from_pretrained("numind/NuNER_Zero").expect("NuNER model load should succeed");
    assert!(model.is_available());

    let entities = model
        .extract_entities(
            "Angela Merkel visited the European Parliament in Strasbourg.",
            None,
        )
        .expect("NuNER extraction should succeed");

    // NuNER may or may not find entities depending on default labels
    for entity in &entities {
        assert!((0.0..=1.0).contains(&entity.confidence));
        assert!(entity.start() < entity.end());
        assert!(!entity.text.is_empty());
    }
}

// =============================================================================
// BERT NER ONNX backend
// =============================================================================

#[test]
#[ignore]
fn bert_ner_onnx_basic() {
    use anno::BertNEROnnx;

    let model = BertNEROnnx::new("protectai/bert-base-NER-onnx")
        .expect("BERT NER model load should succeed");
    assert!(model.is_available());

    let entities = model
        .extract_entities("Alice works at Google in London.", None)
        .expect("BERT NER extraction should succeed");

    assert!(!entities.is_empty(), "BERT NER should find entities");

    for entity in &entities {
        assert!((0.0..=1.0).contains(&entity.confidence));
        assert!(entity.start() < entity.end());
        assert!(!entity.text.is_empty());
    }
}

// =============================================================================
// Determinism: repeated runs produce identical results
// =============================================================================

#[test]
#[ignore]
fn stacked_ner_deterministic() {
    let text = "Dr. Angela Merkel visited the European Parliament in Strasbourg.";
    let ner = StackedNER::default();

    let run1 = ner.extract_entities(text, None).unwrap();
    let run2 = ner.extract_entities(text, None).unwrap();

    assert_eq!(
        run1.len(),
        run2.len(),
        "Stacked NER should be deterministic"
    );

    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.text, b.text);
        assert_eq!(a.start(), b.start());
        assert_eq!(a.end(), b.end());
        assert_eq!(a.entity_type, b.entity_type);
        assert!(
            (a.confidence - b.confidence).abs() < 1e-10,
            "Confidence should be identical"
        );
    }
}
