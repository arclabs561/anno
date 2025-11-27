//! Integration tests for coreference resolution.
//!
//! Tests the full pipeline: NER → Coreference Resolution → Metrics

use anno::eval::coref::{CorefChain, Mention};
use anno::eval::coref_metrics::{b_cubed_score, conll_f1, muc_score};
use anno::eval::coref_resolver::{CorefConfig, SimpleCorefResolver};
use anno::{Entity, EntityType, Model, PatternNER, StackedNER};

// =============================================================================
// Resolver → Metrics Integration
// =============================================================================

#[test]
fn test_resolver_produces_evaluable_chains() {
    // Create entities that should corefer
    let entities = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("he", EntityType::Person, 30, 32, 0.8),
        Entity::new("Smith", EntityType::Person, 60, 65, 0.85),
        Entity::new("Apple Inc", EntityType::Organization, 100, 109, 0.9),
        Entity::new("the company", EntityType::Organization, 130, 141, 0.7),
    ];

    // Resolve coreference
    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Should have 2 chains: John cluster and Apple cluster
    assert!(chains.len() >= 2, "Expected at least 2 chains, got {}", chains.len());

    // The John chain should have 3 mentions
    let john_chain = chains
        .iter()
        .find(|c| c.mentions.iter().any(|m| m.text == "John Smith"))
        .expect("Should have a John Smith chain");

    assert!(john_chain.len() >= 2, "John chain should have at least 2 mentions");
}

#[test]
fn test_resolver_metrics_integration() {
    // Gold standard chains
    let gold_chains = vec![
        CorefChain::new(vec![
            Mention::new("John", 0, 4),
            Mention::new("he", 20, 22),
            Mention::new("him", 40, 43),
        ]),
        CorefChain::new(vec![
            Mention::new("Mary", 50, 54),
            Mention::new("she", 70, 73),
        ]),
    ];

    // Create entities matching the gold standard
    let entities = vec![
        Entity::new("John", EntityType::Person, 0, 4, 0.9),
        Entity::new("he", EntityType::Person, 20, 22, 0.8),
        Entity::new("him", EntityType::Person, 40, 43, 0.8),
        Entity::new("Mary", EntityType::Person, 50, 54, 0.9),
        Entity::new("she", EntityType::Person, 70, 73, 0.8),
    ];

    // Resolve
    let resolver = SimpleCorefResolver::default();
    let pred_chains = resolver.resolve_to_chains(&entities);

    // The resolver should produce chains that can be evaluated
    // Even if not perfect, metrics should be computable
    let (muc_p, muc_r, muc_f1) = muc_score(&pred_chains, &gold_chains);
    let (b3_p, b3_r, b3_f1) = b_cubed_score(&pred_chains, &gold_chains);
    let conll_f1_score = conll_f1(&pred_chains, &gold_chains);

    // Sanity checks
    assert!(muc_p >= 0.0 && muc_p <= 1.0, "MUC precision out of range");
    assert!(muc_r >= 0.0 && muc_r <= 1.0, "MUC recall out of range");
    assert!(b3_p >= 0.0 && b3_p <= 1.0, "B3 precision out of range");
    assert!(conll_f1_score >= 0.0 && conll_f1_score <= 1.0, "CoNLL F1 out of range");

    // With our simple resolver, we should get decent scores on this easy case
    // Note: The resolver groups by entity type + name matching
    println!("MUC: P={:.2} R={:.2} F1={:.2}", muc_p, muc_r, muc_f1);
    println!("B³:  P={:.2} R={:.2} F1={:.2}", b3_p, b3_r, b3_f1);
    println!("CoNLL F1: {:.2}", conll_f1_score);
}

#[test]
fn test_perfect_resolution_gives_perfect_score() {
    // Create a simple case where our resolver should get perfect score
    let gold_chains = vec![CorefChain::new(vec![
        Mention::new("Alice", 0, 5),
        Mention::new("Alice", 20, 25),
    ])];

    let entities = vec![
        Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
        Entity::new("Alice", EntityType::Person, 20, 25, 0.9),
    ];

    let resolver = SimpleCorefResolver::default();
    let pred_chains = resolver.resolve_to_chains(&entities);

    let (_muc_p, _muc_r, muc_f1) = muc_score(&pred_chains, &gold_chains);

    // Exact name match should give perfect score
    assert!(
        muc_f1 > 0.99,
        "Exact match should give near-perfect MUC F1, got {}",
        muc_f1
    );
}

// =============================================================================
// NER → Coreference Pipeline
// =============================================================================

#[test]
fn test_ner_to_coref_pipeline() {
    let text = "John Smith went to the store. He bought milk. Smith paid $5.99.";

    // Step 1: Extract entities with PatternNER (will get the money)
    let pattern_ner = PatternNER::new();
    let entities = pattern_ner.extract_entities(text, None).unwrap();

    // PatternNER finds money, dates, etc. - it won't find John Smith
    // But we can test that whatever it finds can go through the resolver
    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Should not crash, should produce valid chains
    for chain in &chains {
        assert!(!chain.mentions.is_empty());
        for mention in &chain.mentions {
            assert!(mention.start <= mention.end);
        }
    }
}

#[test]
fn test_stacked_ner_to_coref_pipeline() {
    let text = "The CEO of Apple visited Google. He met their executives.";

    // Step 1: Extract entities with StackedNER
    let stacked_ner = StackedNER::default();
    let entities = stacked_ner.extract_entities(text, None).unwrap();

    // Step 2: Resolve coreference
    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Validate output
    for chain in &chains {
        assert!(!chain.mentions.is_empty());
    }

    println!("Found {} entities, {} chains", entities.len(), chains.len());
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_input() {
    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&[]);
    assert!(chains.is_empty());
}

#[test]
fn test_single_entity() {
    let entities = vec![Entity::new("John", EntityType::Person, 0, 4, 0.9)];

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Single entity = singleton chain
    assert_eq!(chains.len(), 1);
    assert!(chains[0].is_singleton());
}

#[test]
fn test_no_coreference() {
    // All different entities, no coreference
    let entities = vec![
        Entity::new("John", EntityType::Person, 0, 4, 0.9),
        Entity::new("Mary", EntityType::Person, 10, 14, 0.9),
        Entity::new("Apple", EntityType::Organization, 20, 25, 0.9),
    ];

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Each entity should be its own chain (3 singletons)
    assert_eq!(chains.len(), 3);
    for chain in &chains {
        assert!(chain.is_singleton());
    }
}

#[test]
fn test_config_affects_resolution() {
    // Strict config: high similarity threshold
    let strict_config = CorefConfig {
        similarity_threshold: 0.99,
        max_pronoun_distance: 1,
        fuzzy_matching: false,
        include_singletons: true,
    };

    let entities = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("Smith", EntityType::Person, 50, 55, 0.9),
    ];

    let strict_resolver = SimpleCorefResolver::new(strict_config);
    let strict_chains = strict_resolver.resolve_to_chains(&entities);

    let lenient_resolver = SimpleCorefResolver::default();
    let lenient_chains = lenient_resolver.resolve_to_chains(&entities);

    // Lenient should merge (fuzzy matching on), strict might not
    let lenient_non_singletons: Vec<_> =
        lenient_chains.iter().filter(|c| !c.is_singleton()).collect();
    let strict_non_singletons: Vec<_> =
        strict_chains.iter().filter(|c| !c.is_singleton()).collect();

    // With fuzzy matching, lenient should find the "Smith" match
    assert!(
        lenient_non_singletons.len() >= strict_non_singletons.len(),
        "Lenient resolver should find at least as many coreferent pairs"
    );
}

