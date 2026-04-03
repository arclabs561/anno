//! End-to-end integration tests for the RAG coreference pipeline.
//!
//! Exercises resolve_for_rag() with realistic multi-sentence inputs,
//! verifying pronoun rewriting, offset validity, cataphora support,
//! and the coref chain evaluation path.

use anno::eval::coref_resolver::{CorefConfig, SimpleCorefResolver};
use anno::rag::{resolve_for_rag, RagCorefConfig};
use anno::{CorefChain, Entity, EntityType, Mention, MentionType};

// =============================================================================
// RAG pronoun rewriting
// =============================================================================

#[test]
fn rag_basic_pronoun_rewrite() {
    let text = "Alice went to the store. She bought milk. She came home.";
    let entities = vec![
        Entity::new("Alice", EntityType::Person, 0, 5, 0.95),
        Entity::new("She", EntityType::Person, 25, 28, 0.85),
        Entity::new("She", EntityType::Person, 42, 45, 0.85),
    ];

    let result = resolve_for_rag(text, &entities, None);

    // Pronouns should be replaced with "Alice"
    assert!(
        result.text.contains("Alice bought milk"),
        "First pronoun not rewritten: {}",
        result.text
    );
    assert!(
        result.text.contains("Alice came home"),
        "Second pronoun not rewritten: {}",
        result.text
    );
    assert_eq!(result.rewrites.len(), 2, "Expected 2 rewrites");
}

#[test]
fn rag_rewrite_offsets_are_valid() {
    let text = "Dr. Smith published a paper. He presented it at the conference.";
    let entities = vec![
        Entity::new("Dr. Smith", EntityType::Person, 0, 9, 0.95),
        Entity::new("He", EntityType::Person, 29, 31, 0.85),
    ];

    let result = resolve_for_rag(text, &entities, None);

    for rewrite in &result.rewrites {
        // Original offsets should be within the original text
        assert!(
            rewrite.start < text.len(),
            "rewrite.start {} out of bounds (text len {})",
            rewrite.start,
            text.len()
        );
        assert!(
            rewrite.end <= text.len(),
            "rewrite.end {} out of bounds (text len {})",
            rewrite.end,
            text.len()
        );
        assert!(
            rewrite.start < rewrite.end,
            "rewrite start {} >= end {}",
            rewrite.start,
            rewrite.end
        );
        // The original text at those offsets should match the original pronoun
        let original_slice: String = text
            .chars()
            .skip(rewrite.start)
            .take(rewrite.end - rewrite.start)
            .collect();
        assert_eq!(
            original_slice.to_lowercase(),
            rewrite.original.to_lowercase(),
            "Offset mismatch: slice='{}' vs original='{}'",
            original_slice,
            rewrite.original
        );
    }
}

#[test]
fn rag_self_contained_sentences() {
    // Each sentence in the rewritten text should be understandable without context.
    let text = "Marie Curie discovered radium. She won the Nobel Prize. She was born in Poland.";
    let entities = vec![
        Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95),
        Entity::new("She", EntityType::Person, 31, 34, 0.85),
        Entity::new("She", EntityType::Person, 56, 59, 0.85),
    ];

    let result = resolve_for_rag(text, &entities, None);

    // Split into sentences and verify each one is self-contained
    let sentences: Vec<&str> = result.text.split(". ").collect();
    for sentence in &sentences {
        // No sentence should start with a pronoun -- they should all reference the antecedent
        let trimmed = sentence.trim();
        assert!(
            !trimmed.starts_with("She ") && !trimmed.starts_with("He "),
            "Sentence not self-contained (starts with pronoun): '{}'",
            trimmed
        );
    }
}

#[test]
fn rag_cataphoric_reference() {
    // Cataphoric: pronoun appears BEFORE its antecedent.
    // "Before she left, Dr. Kim locked the door."
    let text = "Before she left, Dr. Kim locked the door.";
    let entities = vec![
        Entity::new("she", EntityType::Person, 7, 10, 0.85),
        Entity::new("Dr. Kim", EntityType::Person, 17, 24, 0.95),
    ];

    let config = RagCorefConfig {
        resolve_cataphora: true,
        ..Default::default()
    };

    let result = resolve_for_rag(text, &entities, Some(config));

    // With cataphora enabled, "she" should be resolved to "Dr. Kim"
    assert!(
        result.text.contains("Dr. Kim left") || result.text.contains("Dr. Kim locked"),
        "Cataphoric pronoun not resolved: '{}'",
        result.text
    );
}

#[test]
fn rag_no_entities_returns_original() {
    let text = "No entities here.";
    let result = resolve_for_rag(text, &[], None);
    assert_eq!(result.text, text);
    assert_eq!(result.rewrites.len(), 0);
    assert_eq!(result.unresolved_count, 0);
}

#[test]
fn rag_unicode_text_offsets() {
    // Offsets are character offsets, not byte offsets.
    // "Li Wei" uses only ASCII but test with surrounding Unicode.
    let text = "\u{201C}Li Wei\u{201D} went to Beijing. He visited the Great Wall.";
    // \u{201C} = 1 char, "Li Wei" = 6 chars, \u{201D} = 1 char => 8 chars before space
    // " went to Beijing. " = 19 chars => "He" starts at char 27
    let li_wei_start = 1; // after opening quote
    let li_wei_end = 7; // "Li Wei" is 6 chars
    let chars: Vec<char> = text.chars().collect();
    let he_pos = chars
        .windows(2)
        .position(|w| w[0] == 'H' && w[1] == 'e')
        .expect("Should find 'He' in text");

    let entities = vec![
        Entity::new("Li Wei", EntityType::Person, li_wei_start, li_wei_end, 0.95),
        Entity::new("He", EntityType::Person, he_pos, he_pos + 2, 0.85),
    ];

    let result = resolve_for_rag(text, &entities, None);

    // "He" should be replaced with "Li Wei"
    assert!(
        result.text.contains("Li Wei visited"),
        "Unicode pronoun rewrite failed: '{}'",
        result.text
    );
}

// =============================================================================
// Coref chain evaluation path
// =============================================================================

#[test]
fn coref_resolver_groups_coreferent_entities() {
    let entities = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.95),
        Entity::new("he", EntityType::Person, 30, 32, 0.85),
        Entity::new("Microsoft", EntityType::Organization, 45, 54, 0.90),
        Entity::new("it", EntityType::Organization, 70, 72, 0.80),
        Entity::new("Smith", EntityType::Person, 90, 95, 0.90),
    ];

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // There should be at least 2 chains (one for Person cluster, one for Org cluster)
    assert!(
        chains.len() >= 2,
        "Expected at least 2 chains, got {}",
        chains.len()
    );

    // Find the Person chain (should contain "John Smith" and likely "Smith" and "he")
    let person_chain = chains
        .iter()
        .find(|c| c.mentions.iter().any(|m| m.text == "John Smith"));
    assert!(
        person_chain.is_some(),
        "Should have a chain containing 'John Smith'"
    );

    let person_chain = person_chain.unwrap();
    assert!(
        person_chain.len() >= 2,
        "Person chain should have at least 2 mentions, got {}",
        person_chain.len()
    );

    // "Smith" should be in the same chain as "John Smith" (substring match)
    let has_smith = person_chain.mentions.iter().any(|m| m.text == "Smith");
    assert!(
        has_smith,
        "Chain should group 'Smith' with 'John Smith': {:?}",
        person_chain
            .mentions
            .iter()
            .map(|m| &m.text)
            .collect::<Vec<_>>()
    );
}

#[test]
fn coref_canonical_mention_prefers_proper_nouns() {
    // Build a chain manually with typed mentions
    let mentions = vec![
        Mention::with_type("he", 50, 52, MentionType::Pronominal),
        Mention::with_type("the CEO", 30, 37, MentionType::Nominal),
        Mention::with_type("John Smith", 0, 10, MentionType::Proper),
    ];

    let chain = CorefChain::new(mentions);
    let canonical = chain.canonical_mention();

    assert!(canonical.is_some(), "Chain should have a canonical mention");
    assert_eq!(
        canonical.unwrap().text,
        "John Smith",
        "Canonical mention should be the proper noun, not pronoun/nominal"
    );
}

#[test]
fn coref_canonical_mention_falls_back_to_longest() {
    // Without mention types, canonical selection falls back to longest mention
    let mentions = vec![
        Mention::new("he", 50, 52),
        Mention::new("the president of the company", 0, 28),
        Mention::new("him", 60, 63),
    ];

    let chain = CorefChain::new(mentions);
    let canonical = chain.canonical_mention();

    assert!(canonical.is_some());
    assert_eq!(
        canonical.unwrap().text,
        "the president of the company",
        "Without mention types, should pick longest mention"
    );
}

#[test]
fn coref_resolver_with_custom_config() {
    let config = CorefConfig {
        max_pronoun_lookback: 1,
        fuzzy_matching: false,
        include_singletons: false,
        use_name_gazetteer: true,
        acronym_matching: true,
        relaxed_head_match: true,
        proper_containment: true,
        precise_constructs: true,
        strict_head_match: true,
        proper_head_word_match: true,
    };

    let resolver = SimpleCorefResolver::new(config);
    let entities = vec![
        Entity::new("Alice", EntityType::Person, 0, 5, 0.95),
        Entity::new("Bob", EntityType::Person, 20, 23, 0.90),
    ];

    let chains = resolver.resolve_to_chains(&entities);

    // With singletons disabled, separate entities should still form chains
    // (each unique entity forms at least one chain unless excluded)
    // Without fuzzy matching, "Alice" and "Bob" should be in separate chains
    let alice_chain = chains
        .iter()
        .find(|c| c.mentions.iter().any(|m| m.text == "Alice"));
    let bob_chain = chains
        .iter()
        .find(|c| c.mentions.iter().any(|m| m.text == "Bob"));

    // They should not be merged into the same chain
    if let (Some(ac), Some(_bc)) = (alice_chain, bob_chain) {
        let alice_mentions: Vec<&str> = ac.mentions.iter().map(|m| m.text.as_str()).collect();
        assert!(
            !alice_mentions.contains(&"Bob"),
            "Alice and Bob should be in separate chains"
        );
    }
}
