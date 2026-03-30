//! Cross-crate E2E test for the RAG coreference pipeline.
//!
//! Exercises: Entity creation -> SimpleCorefResolver -> resolve_for_rag() -> pronoun replacement,
//! cataphoric resolution, and offset validation.
//!
//! Feature-gated: `#[cfg(feature = "analysis")]`

#![cfg(feature = "analysis")]

use anno::core::coref::entities_to_chains;
use anno::eval::coref_resolver::SimpleCorefResolver;
use anno::rag::{resolve_for_rag, RagCorefConfig};
use anno::{Entity, EntityType, MentionType};

// =============================================================================
// Helpers
// =============================================================================

/// Verify that all rewrite offsets are valid Unicode scalar offsets within the
/// original text. This catches byte-vs-char offset bugs.
fn assert_offsets_valid(text: &str, result: &anno::rag::RagCorefResult) {
    let char_count = text.chars().count();
    for rw in &result.rewrites {
        assert!(
            rw.start <= rw.end,
            "rewrite start ({}) > end ({})",
            rw.start,
            rw.end
        );
        assert!(
            rw.end <= char_count,
            "rewrite end ({}) exceeds text char count ({})",
            rw.end,
            char_count
        );
        // Verify the original text at those char offsets matches the rewrite's `original` field.
        let extracted: String = text
            .chars()
            .skip(rw.start)
            .take(rw.end - rw.start)
            .collect();
        assert_eq!(
            extracted, rw.original,
            "char slice [{},{}) = {:?}, expected {:?}",
            rw.start, rw.end, extracted, rw.original
        );
    }
}

// =============================================================================
// 1. Basic pronoun replacement (anaphora)
// =============================================================================

#[test]
fn anaphoric_pronoun_replacement() {
    let text = "Alice went to the store. She bought milk. Her friend Bob was there.";
    let entities = vec![
        Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
        Entity::new("She", EntityType::Person, 25, 28, 0.8),
        Entity::new("Her", EntityType::Person, 42, 45, 0.8),
        Entity::new("Bob", EntityType::Person, 53, 56, 0.9),
    ];

    let result = resolve_for_rag(text, &entities, None);

    // "She" and "Her" should be replaced with "Alice"
    assert!(
        result.text.contains("Alice bought milk"),
        "expected 'Alice bought milk', got: {}",
        result.text
    );
    assert!(
        result.text.contains("Alice friend Bob") || result.text.contains("Alice's friend Bob"),
        "expected 'Her' replaced with 'Alice', got: {}",
        result.text
    );
    assert!(result.rewrites.len() >= 2, "expected at least 2 rewrites");
    assert_offsets_valid(text, &result);
}

// =============================================================================
// 2. Unicode offset correctness
// =============================================================================

#[test]
fn unicode_offsets_are_char_not_byte() {
    // Multi-byte characters before the pronoun. "Zurich" with u-umlaut = 6 chars but 7 bytes.
    let text = "Z\u{00fc}rich is a city. It is in Switzerland.";
    //          Z u-uml r i c h   i s   a   c  i  t  y  .     I  t  ...
    // chars:   0  1   2 3 4 5  6 7 8 9 10 11 12 13 14 15 16 17 18 ...

    let entities = vec![
        Entity::new("Z\u{00fc}rich", EntityType::Location, 0, 6, 0.9),
        Entity::new("It", EntityType::Location, 18, 20, 0.8),
    ];

    let result = resolve_for_rag(text, &entities, None);

    // "It" should be replaced with "Zurich" (with umlaut)
    assert!(
        result.text.contains("Z\u{00fc}rich is in Switzerland"),
        "expected Zurich replacement, got: {}",
        result.text
    );
    assert_offsets_valid(text, &result);
}

// =============================================================================
// 3. Cataphoric (forward-pointing) reference
// =============================================================================

#[test]
fn cataphoric_pronoun_resolution() {
    // "she" appears before "Maria" -- cataphoric reference.
    let text = "When she arrived at the airport, Maria checked in immediately.";
    let entities = vec![
        Entity::new("she", EntityType::Person, 5, 8, 0.8),
        Entity::new("Maria", EntityType::Person, 32, 37, 0.9),
    ];

    let result = resolve_for_rag(text, &entities, None);

    assert_eq!(
        result.text, "When Maria arrived at the airport, Maria checked in immediately.",
        "cataphoric 'she' should resolve to 'Maria'"
    );
    assert_eq!(result.rewrites.len(), 1);
    assert_eq!(result.rewrites[0].original, "she");
    assert_eq!(result.rewrites[0].replacement, "Maria");
    assert_eq!(result.unresolved_count, 0);
    assert_offsets_valid(text, &result);
}

#[test]
fn cataphoric_disabled_leaves_pronoun() {
    let text = "When she arrived at the airport, Maria checked in immediately.";
    let entities = vec![
        Entity::new("she", EntityType::Person, 5, 8, 0.8),
        Entity::new("Maria", EntityType::Person, 32, 37, 0.9),
    ];

    let config = RagCorefConfig {
        resolve_cataphora: false,
        ..Default::default()
    };
    let result = resolve_for_rag(text, &entities, Some(config));

    // Without cataphora, "she" stays unresolved.
    assert_eq!(result.text, text);
    assert_eq!(result.unresolved_count, 1);
}

// =============================================================================
// 4. SimpleCorefResolver -> CorefChain path
// =============================================================================

#[test]
fn resolver_produces_chains_with_canonical_ids() {
    let entities = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.95),
        Entity::new("the company", EntityType::Organization, 15, 26, 0.9),
        Entity::new("He", EntityType::Person, 28, 30, 0.8),
        Entity::new("John", EntityType::Person, 45, 49, 0.9),
    ];

    let resolver = SimpleCorefResolver::default();
    let resolved = resolver.resolve(&entities);

    // Every entity must have a canonical_id assigned.
    for entity in &resolved {
        assert!(
            entity.canonical_id.is_some(),
            "entity {:?} missing canonical_id",
            entity.text
        );
    }

    // "John Smith", "He", and "John" should share the same canonical_id (Person cluster).
    let john_id = resolved[0].canonical_id.unwrap();
    let company_id = resolved[1].canonical_id.unwrap();
    let he_id = resolved[2].canonical_id.unwrap();
    let john2_id = resolved[3].canonical_id.unwrap();

    assert_eq!(john_id, john2_id, "'John Smith' and 'John' should corefer");

    // "He" has a canonical_id (confirmed by unwrap above). It should be in
    // the Person cluster (john_id), not the Org cluster (company_id).
    assert_ne!(
        he_id, company_id,
        "'He' (Person) should not be in the company cluster"
    );

    // "the company" should be in a different cluster from the Person entities.
    assert_ne!(
        company_id, john_id,
        "'the company' (Org) should not corefer with 'John Smith' (Person)"
    );
}

#[test]
fn resolve_to_chains_groups_coreferent_mentions() {
    let entities = vec![
        Entity::new("Alice", EntityType::Person, 0, 5, 0.95),
        Entity::new("She", EntityType::Person, 10, 13, 0.8),
        Entity::new("Bob", EntityType::Person, 20, 23, 0.9),
    ];

    let resolver = SimpleCorefResolver::default();
    let chains = resolver.resolve_to_chains(&entities);

    // There should be at least 2 chains: one for Alice+She, one for Bob.
    assert!(
        chains.len() >= 2,
        "expected >= 2 chains, got {}",
        chains.len()
    );

    // Find the multi-mention chain (Alice + She).
    let multi_chains: Vec<_> = chains.iter().filter(|c| c.len() > 1).collect();
    assert!(
        !multi_chains.is_empty(),
        "expected at least one multi-mention chain"
    );

    let alice_chain = multi_chains
        .iter()
        .find(|c| c.mentions.iter().any(|m| m.text == "Alice"))
        .expect("should have a chain containing Alice");
    assert!(
        alice_chain.mentions.iter().any(|m| m.text == "She"),
        "'She' should be in Alice's chain"
    );
}

// =============================================================================
// 5. entities_to_chains preserves structure
// =============================================================================

#[test]
fn entities_to_chains_roundtrip() {
    let resolver = SimpleCorefResolver::default();
    let entities = vec![
        Entity::new("Dr. Smith", EntityType::Person, 0, 9, 0.95),
        Entity::new("he", EntityType::Person, 15, 17, 0.8),
        Entity::new("the hospital", EntityType::Organization, 25, 37, 0.9),
    ];

    let resolved = resolver.resolve(&entities);
    let chains = entities_to_chains(&resolved);

    // Total mentions across all chains should equal total entities.
    let total_mentions: usize = chains.iter().map(|c| c.len()).sum();
    assert_eq!(total_mentions, entities.len());
}

// =============================================================================
// 6. Canonical mention prefers proper nouns
// =============================================================================

#[test]
fn canonical_mention_prefers_proper_noun() {
    use anno::Mention;

    let proper = Mention::with_type("Dr. Elizabeth Chen", 0, 18, MentionType::Proper);
    let pronoun = Mention::with_type("she", 25, 28, MentionType::Pronominal);
    let nominal = Mention::with_type("the doctor", 35, 45, MentionType::Nominal);

    let chain = anno::CorefChain::new(vec![proper, pronoun, nominal]);
    let canonical = chain.canonical_mention().expect("chain is non-empty");

    assert_eq!(
        canonical.text, "Dr. Elizabeth Chen",
        "canonical mention should be the proper noun"
    );
    assert_eq!(canonical.mention_type, Some(MentionType::Proper));
}

// =============================================================================
// 7. Multi-sentence self-contained paragraph
// =============================================================================

#[test]
fn multi_sentence_paragraph_becomes_self_contained() {
    let text = concat!(
        "The European Central Bank raised interest rates on Thursday. ",
        "It cited persistent inflation in the eurozone. ",
        "Its president Christine Lagarde held a press conference."
    );
    let entities = vec![
        Entity::new(
            "The European Central Bank",
            EntityType::Organization,
            0,
            25,
            0.95,
        ),
        Entity::new("It", EntityType::Organization, 61, 63, 0.8),
        Entity::new("Its", EntityType::Organization, 108, 111, 0.8),
        Entity::new("Christine Lagarde", EntityType::Person, 122, 139, 0.9),
    ];

    let result = resolve_for_rag(text, &entities, None);

    // "It" -> "The European Central Bank"
    assert!(
        result.text.contains("The European Central Bank cited"),
        "expected 'It' replaced, got: {}",
        result.text
    );
    // "Its" -> "The European Central Bank" (case-preserving: "Its" starts uppercase -> "The")
    // The replacement splices at char boundaries, so we just check the antecedent appears.
    assert!(
        result.text.contains("The European Central Bank president")
            || result
                .text
                .contains("The European Central Bank's president"),
        "expected 'Its' replaced, got: {}",
        result.text
    );
    assert_offsets_valid(text, &result);
}

// =============================================================================
// 8. Empty and edge cases
// =============================================================================

#[test]
fn empty_entities_returns_original_text() {
    let text = "Nothing to resolve here.";
    let result = resolve_for_rag(text, &[], None);
    assert_eq!(result.text, text);
    assert!(result.rewrites.is_empty());
    assert_eq!(result.unresolved_count, 0);
}

#[test]
fn no_pronouns_returns_original_text() {
    let text = "Alice and Bob went to Paris.";
    let entities = vec![
        Entity::new("Alice", EntityType::Person, 0, 5, 0.9),
        Entity::new("Bob", EntityType::Person, 10, 13, 0.9),
        Entity::new("Paris", EntityType::Location, 22, 27, 0.9),
    ];
    let result = resolve_for_rag(text, &entities, None);
    assert_eq!(result.text, text);
    assert!(result.rewrites.is_empty());
}
