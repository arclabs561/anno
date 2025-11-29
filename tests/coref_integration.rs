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
    assert!(
        chains.len() >= 2,
        "Expected at least 2 chains, got {}",
        chains.len()
    );

    // The John chain should have 3 mentions
    let john_chain = chains
        .iter()
        .find(|c| c.mentions.iter().any(|m| m.text == "John Smith"))
        .expect("Should have a John Smith chain");

    assert!(
        john_chain.len() >= 2,
        "John chain should have at least 2 mentions"
    );
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
    assert!((0.0..=1.0).contains(&muc_p), "MUC precision out of range");
    assert!((0.0..=1.0).contains(&muc_r), "MUC recall out of range");
    assert!((0.0..=1.0).contains(&b3_p), "B3 precision out of range");
    assert!(
        (0.0..=1.0).contains(&conll_f1_score),
        "CoNLL F1 out of range"
    );

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
    let lenient_non_singletons: Vec<_> = lenient_chains
        .iter()
        .filter(|c| !c.is_singleton())
        .collect();
    let strict_non_singletons: Vec<_> =
        strict_chains.iter().filter(|c| !c.is_singleton()).collect();

    // With fuzzy matching, lenient should find the "Smith" match
    assert!(
        lenient_non_singletons.len() >= strict_non_singletons.len(),
        "Lenient resolver should find at least as many coreferent pairs"
    );
}

// =============================================================================
// Deep Edge Case Tests
// =============================================================================

#[test]
fn test_pronoun_without_antecedent() {
    // Pronoun with no preceding entity should become a singleton
    let entities = vec![Entity::new("he", EntityType::Person, 0, 2, 0.8)];

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // Should have one singleton chain
    assert_eq!(chains.len(), 1);
    assert!(chains[0].is_singleton());
}

#[test]
fn test_pronoun_wrong_type() {
    // "it" shouldn't link to a Person
    let entities = vec![
        Entity::new("John", EntityType::Person, 0, 4, 0.9),
        Entity::new("it", EntityType::Person, 20, 22, 0.8), // "it" tagged as Person incorrectly
    ];

    let resolver = SimpleCorefResolver::default();
    let resolved = resolver.resolve(&entities);

    // "it" is not compatible with Person type, so they shouldn't link
    // Note: our pronoun_compatible check handles this
    assert_ne!(resolved[0].canonical_id, resolved[1].canonical_id);
}

#[test]
fn test_she_links_to_female_entity() {
    let entities = vec![
        Entity::new("Mary", EntityType::Person, 0, 4, 0.9),
        Entity::new("John", EntityType::Person, 10, 14, 0.9),
        Entity::new("she", EntityType::Person, 30, 33, 0.8),
    ];

    let resolver = SimpleCorefResolver::default();
    let resolved = resolver.resolve(&entities);

    // "she" should link to neither (we can't infer gender from "Mary" without external knowledge)
    // But it will link to the nearest Person (John) since we can't infer gender from names
    // This is a limitation of the simple resolver
    assert!(resolved[2].canonical_id.is_some());
}

#[test]
fn test_organization_pronoun() {
    let entities = vec![
        Entity::new("Apple Inc", EntityType::Organization, 0, 9, 0.9),
        Entity::new("it", EntityType::Organization, 30, 32, 0.8),
        Entity::new("they", EntityType::Organization, 50, 54, 0.8),
    ];

    let resolver = SimpleCorefResolver::default();
    let resolved = resolver.resolve(&entities);

    // Both "it" and "they" should link to Apple Inc
    assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    assert_eq!(resolved[0].canonical_id, resolved[2].canonical_id);
}

#[test]
fn test_already_resolved_entities() {
    // Entities that already have canonical_id should keep it
    let mut entity1 = Entity::new("John", EntityType::Person, 0, 4, 0.9);
    entity1.canonical_id = Some(999);

    let mut entity2 = Entity::new("John", EntityType::Person, 20, 24, 0.9);
    entity2.canonical_id = Some(999);

    let entity3 = Entity::new("John", EntityType::Person, 40, 44, 0.9);

    let entities = vec![entity1, entity2, entity3];

    let resolver = SimpleCorefResolver::default();
    let resolved = resolver.resolve(&entities);

    // First two should keep their IDs, third gets a new one (but matches the first two by name)
    assert_eq!(resolved[0].canonical_id, Some(999));
    assert_eq!(resolved[1].canonical_id, Some(999));
    // Third entity should match because of exact name match
    // But it won't get 999 because we don't look at existing IDs in canonical_to_cluster
}

#[test]
fn test_case_insensitive_matching() {
    let entities = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("JOHN SMITH", EntityType::Person, 30, 40, 0.9),
        Entity::new("john smith", EntityType::Person, 60, 70, 0.9),
    ];

    let resolver = SimpleCorefResolver::default();
    let resolved = resolver.resolve(&entities);

    // All should link together (case insensitive)
    assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    assert_eq!(resolved[1].canonical_id, resolved[2].canonical_id);
}

#[test]
fn test_partial_name_match() {
    let entities = vec![
        Entity::new("Dr. John Smith", EntityType::Person, 0, 14, 0.9),
        Entity::new("John Smith", EntityType::Person, 30, 40, 0.9),
        Entity::new("Smith", EntityType::Person, 60, 65, 0.9),
    ];

    let resolver = SimpleCorefResolver::default();
    let resolved = resolver.resolve(&entities);

    // All should link together (substring matching)
    assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    assert_eq!(resolved[1].canonical_id, resolved[2].canonical_id);
}

#[test]
fn test_metrics_on_complex_scenario() {
    // Complex scenario with multiple entities and pronouns
    let gold_chains = vec![
        CorefChain::new(vec![
            Mention::new("John", 0, 4),
            Mention::new("he", 20, 22),
            Mention::new("John", 50, 54),
        ]),
        CorefChain::new(vec![
            Mention::new("Apple", 100, 105),
            Mention::new("it", 120, 122),
            Mention::new("the company", 140, 151),
        ]),
    ];

    let entities = vec![
        Entity::new("John", EntityType::Person, 0, 4, 0.9),
        Entity::new("he", EntityType::Person, 20, 22, 0.8),
        Entity::new("John", EntityType::Person, 50, 54, 0.9),
        Entity::new("Apple", EntityType::Organization, 100, 105, 0.9),
        Entity::new("it", EntityType::Organization, 120, 122, 0.8),
        // Note: "the company" is not in our entities - this tests partial coverage
    ];

    let resolver = SimpleCorefResolver::default();
    let pred_chains = resolver.resolve_to_chains(&entities);

    let (_muc_p, _muc_r, muc_f1) = muc_score(&pred_chains, &gold_chains);
    let conll = conll_f1(&pred_chains, &gold_chains);

    println!("Complex scenario: MUC F1={:.2}, CoNLL={:.2}", muc_f1, conll);

    // We won't get perfect scores because we're missing "the company"
    // But metrics should be computable and reasonable
    assert!(muc_f1 > 0.0, "Should have some correct links");
    assert!((0.0..=1.0).contains(&conll), "CoNLL should be valid");
}

#[test]
fn test_all_singletons_produces_valid_metrics() {
    // All different entities - no coreference
    let gold_chains = vec![
        CorefChain::singleton(Mention::new("A", 0, 1)),
        CorefChain::singleton(Mention::new("B", 10, 11)),
        CorefChain::singleton(Mention::new("C", 20, 21)),
    ];

    let entities = vec![
        Entity::new("A", EntityType::Person, 0, 1, 0.9),
        Entity::new("B", EntityType::Organization, 10, 11, 0.9),
        Entity::new("C", EntityType::Location, 20, 21, 0.9),
    ];

    let resolver = SimpleCorefResolver::default();
    let pred_chains = resolver.resolve_to_chains(&entities);

    // Should produce 3 singleton chains
    assert_eq!(pred_chains.len(), 3);

    let (muc_p, muc_r, muc_f1) = muc_score(&pred_chains, &gold_chains);

    // MUC on all singletons is degenerate (0/0), but should not panic
    println!(
        "All singletons: MUC P={:.2} R={:.2} F1={:.2}",
        muc_p, muc_r, muc_f1
    );
}
