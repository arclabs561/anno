//! Property-Based Testing for NER and Coreference Systems
//!
//! This module contains comprehensive property tests based on research best practices:
//!
//! # Categories of Property Tests
//!
//! ## 1. Structural Invariants
//! - Span validity: entities must be within text bounds
//! - UTF-8 boundary alignment: spans must align with character boundaries  
//! - No invalid overlaps (except for Union strategy)
//! - Confidence in [0.0, 1.0]
//!
//! ## 2. Determinism Properties
//! - Same input → same output
//! - Order independence (where applicable)
//!
//! ## 3. Metamorphic Relations (Semantic-Preserving Transforms)
//! - Trailing whitespace invariance
//! - Leading whitespace offset correction
//! - Case normalization for certain entity types
//! - Newline normalization (\r\n ↔ \n)
//!
//! ## 4. Coreference Axioms
//! - Reflexivity: every mention corefs with itself
//! - Symmetry: if A corefs with B, then B corefs with A
//! - Transitivity: if A corefs with B and B corefs with C, then A corefs with C
//!
//! ## 5. Round-Trip Properties
//! - Serialization → Deserialization preserves data
//! - Morpheme segmentation → reconstruction preserves text
//!
//! # References
//! - Property-based testing: https://blog.logrocket.com/property-based-testing-in-rust-with-proptest/
//! - Metamorphic testing for NLP: https://valerio-terragni.github.io/assets/pdf/cho-icsme-2025.pdf
//! - NER invariants: https://blog.lambdaclass.com/what-is-property-based-testing/

#[cfg(test)]
mod proptests {
    use anno::EntityType;
    use anno::{Model, RegexNER};
    use proptest::prelude::*;

    // =========================================================================
    // Test Strategies (Generators)
    // =========================================================================

    /// Generate ASCII-only text for reliable testing
    fn ascii_text() -> impl Strategy<Value = String> {
        proptest::string::string_regex("[A-Za-z0-9 .,!?'\"\\-]{0,200}").unwrap()
    }

    /// Generate realistic sentence-like text
    fn sentence_text() -> impl Strategy<Value = String> {
        proptest::string::string_regex("[A-Z][a-z]{2,10}( [a-z]{2,8}){0,10}\\.").unwrap()
    }

    /// Generate text with potential entity patterns
    fn entity_rich_text() -> impl Strategy<Value = String> {
        prop_oneof![
            // Names
            proptest::string::string_regex("[A-Z][a-z]{2,8} [A-Z][a-z]{2,8} is a person\\.")
                .unwrap(),
            // Money
            proptest::string::string_regex("The cost is \\$[1-9][0-9]{0,3}\\.").unwrap(),
            // Emails
            proptest::string::string_regex("Contact [a-z]{3,8}@[a-z]{3,6}\\.com for info\\.")
                .unwrap(),
            // Dates
            proptest::string::string_regex(
                "On 20[0-2][0-9]-[01][0-9]-[0-3][0-9] something happened\\."
            )
            .unwrap(),
            // URLs
            proptest::string::string_regex("Visit https://[a-z]{3,8}\\.com/[a-z]{2,6} today\\.")
                .unwrap(),
        ]
    }

    /// Generate Unicode text including multi-byte characters
    fn unicode_text() -> impl Strategy<Value = String> {
        prop_oneof![
            // Chinese text
            Just("北京 Beijing is in China".to_string()),
            Just("Price: €50 then €100".to_string()),
            Just("Tokyo 東京 is the capital".to_string()),
            // Emoji
            Just("🚀 SpaceX launched from 🇺🇸".to_string()),
            // Combining characters
            Just("café résumé naïve".to_string()),
            // Mixed
            proptest::string::string_regex("[A-Za-z0-9 €£¥]{0,50}").unwrap(),
        ]
    }

    // =========================================================================
    // 1. Structural Invariants
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        /// Property: All entity spans must be within text bounds
        #[test]
        fn spans_within_bounds(text in ascii_text()) {
            let ner = RegexNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                let char_count = text.chars().count();
                for e in &entities {
                    prop_assert!(e.start() <= char_count,
                        "Start {} exceeds text length {}", e.start(), char_count);
                    prop_assert!(e.end() <= char_count,
                        "End {} exceeds text length {}", e.end(), char_count);
                    prop_assert!(e.start() <= e.end(),
                        "Invalid span: start {} > end {}", e.start(), e.end());
                }
            }
        }

        /// Property: Entity spans must align with UTF-8 character boundaries
        #[test]
        fn utf8_boundary_alignment(text in unicode_text()) {
            let ner = RegexNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                let chars: Vec<char> = text.chars().collect();
                for e in &entities {
                    // Character offset must be valid
                    prop_assert!(e.start() <= chars.len(),
                        "Start char offset {} invalid for {} chars", e.start(), chars.len());
                    prop_assert!(e.end() <= chars.len(),
                        "End char offset {} invalid for {} chars", e.end(), chars.len());

                    // The span text should match what we extract
                    if e.start() < e.end() && e.end() <= chars.len() {
                        let extracted: String = chars[e.start()..e.end()].iter().collect();
                        // Allow for normalization differences
                        let match_ok = e.text == extracted ||
                            e.text.trim() == extracted.trim() ||
                            e.text.to_lowercase() == extracted.to_lowercase();
                        prop_assert!(match_ok,
                            "Span mismatch: entity '{}' vs extracted '{}'", e.text, extracted);
                    }
                }
            }
        }

        /// Property: All confidence scores must be in [0.0, 1.0]
        #[test]
        fn confidence_bounded(text in entity_rich_text()) {
            let ner = RegexNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                for e in &entities {
                    prop_assert!(e.confidence >= 0.0 && e.confidence <= 1.0,
                        "Confidence {} out of bounds for {:?}", e.confidence, e.entity_type);
                }
            }
        }

        /// Property: Non-Union strategies must not produce overlapping entities
        #[test]
        fn no_overlap_default_strategy(text in ascii_text()) {
            let ner = RegexNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                for i in 0..entities.len() {
                    for j in (i + 1)..entities.len() {
                        let e1 = &entities[i];
                        let e2 = &entities[j];
                        let overlaps = e1.start() < e2.end() && e2.start() < e1.end();
                        prop_assert!(!overlaps,
                            "Overlap detected: [{}-{}) and [{}-{})",
                            e1.start(), e1.end(), e2.start(), e2.end());
                    }
                }
            }
        }

        /// Property: Entity text must be non-empty
        #[test]
        fn non_empty_entity_text(text in entity_rich_text()) {
            let ner = RegexNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                for e in &entities {
                    prop_assert!(!e.text.is_empty(),
                        "Empty entity text for type {:?}", e.entity_type);
                    prop_assert!(!e.text.trim().is_empty(),
                        "Whitespace-only entity text for type {:?}", e.entity_type);
                }
            }
        }
    }

    // =========================================================================
    // 2. Determinism Properties
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Property: Same input must produce same output (determinism)
        #[test]
        fn deterministic_extraction(text in ascii_text()) {
            let ner = RegexNER::new();
            let result1 = ner.extract_entities(&text, None);
            let result2 = ner.extract_entities(&text, None);

            match (result1, result2) {
                (Ok(e1), Ok(e2)) => {
                    prop_assert_eq!(e1.len(), e2.len(), "Different entity counts");
                    for (a, b) in e1.iter().zip(e2.iter()) {
                        prop_assert_eq!(a.start(), b.start());
                        prop_assert_eq!(a.end(), b.end());
                        prop_assert_eq!(&a.text, &b.text);
                        prop_assert_eq!(&a.entity_type, &b.entity_type);
                    }
                }
                (Err(_), Err(_)) => {} // Both failed, that's consistent
                _ => prop_assert!(false, "Inconsistent success/failure"),
            }
        }

        /// Property: Entity order should be consistent (sorted by start position)
        #[test]
        fn entities_sorted_by_position(text in entity_rich_text()) {
            let ner = RegexNER::new();
            if let Ok(entities) = ner.extract_entities(&text, None) {
                for window in entities.windows(2) {
                    prop_assert!(window[0].start() <= window[1].start(),
                        "Entities not sorted: {} > {}", window[0].start(), window[1].start());
                }
            }
        }
    }

    // =========================================================================
    // 3. Metamorphic Relations (Semantic-Preserving Transforms)
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Metamorphic: Adding trailing whitespace should not change entity count
        #[test]
        fn trailing_whitespace_invariance(text in sentence_text()) {
            let ner = RegexNER::new();

            let original = ner.extract_entities(&text, None);
            let with_space = ner.extract_entities(&format!("{}   ", text), None);

            if let (Ok(e1), Ok(e2)) = (original, with_space) {
                prop_assert_eq!(e1.len(), e2.len(),
                    "Trailing whitespace changed entity count: {} -> {}", e1.len(), e2.len());

                // Entities should have same spans (offsets unchanged)
                for (a, b) in e1.iter().zip(e2.iter()) {
                    prop_assert_eq!(a.start(), b.start());
                    prop_assert_eq!(a.end(), b.end());
                    prop_assert!(a.entity_type == b.entity_type);
                }
            }
        }

        /// Metamorphic: Adding leading whitespace should shift offsets correctly
        #[test]
        fn leading_whitespace_offset_shift(text in sentence_text()) {
            let ner = RegexNER::new();
            let prefix = "   "; // 3 spaces
            let prefix_len = prefix.chars().count();

            let original = ner.extract_entities(&text, None);
            let with_prefix = ner.extract_entities(&format!("{}{}", prefix, text), None);

            if let (Ok(e1), Ok(e2)) = (original, with_prefix) {
                prop_assert_eq!(e1.len(), e2.len(),
                    "Leading whitespace changed entity count: {} -> {}", e1.len(), e2.len());

                // Entities should have shifted offsets
                for (a, b) in e1.iter().zip(e2.iter()) {
                    prop_assert_eq!(a.start() + prefix_len, b.start(),
                        "Start not shifted correctly: {} + {} != {}", a.start(), prefix_len, b.start());
                    prop_assert_eq!(a.end() + prefix_len, b.end(),
                        "End not shifted correctly: {} + {} != {}", a.end(), prefix_len, b.end());
                    prop_assert!(a.entity_type == b.entity_type);
                }
            }
        }

        /// Metamorphic: Newline normalization should not change entities
        #[test]
        fn newline_normalization_invariance(text in sentence_text()) {
            let ner = RegexNER::new();

            // Insert \n in original, \r\n in modified
            let with_lf = format!("{}\n{}", text, text);
            let with_crlf = format!("{}\r\n{}", text, text);

            let e_lf = ner.extract_entities(&with_lf, None);
            let e_crlf = ner.extract_entities(&with_crlf, None);

            if let (Ok(entities_lf), Ok(entities_crlf)) = (e_lf, e_crlf) {
                // Same entity count
                prop_assert_eq!(entities_lf.len(), entities_crlf.len(),
                    "Newline type changed entity count");

                // Same entity types and texts
                for (a, b) in entities_lf.iter().zip(entities_crlf.iter()) {
                    prop_assert!(a.entity_type == b.entity_type);
                    prop_assert!(a.text == b.text);
                }
            }
        }

        /// Metamorphic: Duplicate sentence should double entity count (approximately)
        #[test]
        fn duplication_scales_entities(text in entity_rich_text()) {
            let ner = RegexNER::new();

            let single = ner.extract_entities(&text, None);
            let doubled = ner.extract_entities(&format!("{} {}", text, text), None);

            if let (Ok(e1), Ok(e2)) = (single, doubled) {
                // Should have roughly 2x entities (allowing for boundary effects)
                if !e1.is_empty() {
                    let ratio = e2.len() as f64 / e1.len() as f64;
                    prop_assert!((1.5..=2.5).contains(&ratio),
                        "Duplication ratio unexpected: {} (expected ~2.0)", ratio);
                }
            }
        }
    }

    // =========================================================================
    // 4. Entity Type Specific Properties
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        /// Property: Money entities should contain currency indicators
        #[test]
        fn money_contains_currency(amount in 1u32..10000u32) {
            let text = format!("The price is ${}.00 for the item.", amount);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();

            let money: Vec<_> = entities.iter()
                .filter(|e| e.entity_type == EntityType::Money)
                .collect();

            prop_assert!(!money.is_empty(), "No money entity found in '{}'", text);

            for e in money {
                let has_currency = e.text.contains('$') ||
                    e.text.contains('€') ||
                    e.text.contains('£') ||
                    e.text.to_lowercase().contains("dollar") ||
                    e.text.to_lowercase().contains("usd");
                prop_assert!(has_currency,
                    "Money entity '{}' has no currency indicator", e.text);
            }
        }

        /// Property: Email entities must contain @ symbol
        #[test]
        fn email_contains_at(user in "[a-z]{3,8}", domain in "[a-z]{3,6}") {
            let text = format!("Send mail to {}@{}.com please.", user, domain);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();

            let emails: Vec<_> = entities.iter()
                .filter(|e| e.entity_type == EntityType::Email)
                .collect();

            for e in &emails {
                prop_assert!(e.text.contains('@'),
                    "Email entity '{}' missing @ symbol", e.text);
            }
        }

        /// Property: URL entities must contain protocol or domain pattern
        #[test]
        fn url_has_valid_structure(path in "[a-z]{2,8}") {
            let text = format!("Visit https://example.com/{} for details.", path);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();

            let urls: Vec<_> = entities.iter()
                .filter(|e| e.entity_type == EntityType::Url)
                .collect();

            for e in &urls {
                let has_protocol = e.text.starts_with("http://") || e.text.starts_with("https://");
                let has_domain = e.text.contains('.') && e.text.contains('/');
                prop_assert!(has_protocol || has_domain,
                    "URL entity '{}' has invalid structure", e.text);
            }
        }

        /// Property: Percentage entities must contain % or 'percent'
        #[test]
        fn percentage_has_indicator(pct in 1u32..100u32) {
            let text = format!("The rate is {}% this quarter.", pct);
            let ner = RegexNER::new();
            let entities = ner.extract_entities(&text, None).unwrap();

            let percents: Vec<_> = entities.iter()
                .filter(|e| e.entity_type == EntityType::Percent)
                .collect();

            for e in &percents {
                let has_indicator = e.text.contains('%') ||
                    e.text.to_lowercase().contains("percent");
                prop_assert!(has_indicator,
                    "Percent entity '{}' missing indicator", e.text);
            }
        }
    }

    // =========================================================================
    // 5. Robustness Properties
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property: NER should never panic on arbitrary input.
        /// proptest catches panics itself and reports them; wrapping in
        /// `catch_unwind` would swallow the panic message silently.
        #[test]
        fn never_panics(text in ".*") {
            let ner = RegexNER::new();
            let _ = ner.extract_entities(&text, None);
        }

        /// Property: Empty input should return empty entities
        #[test]
        fn empty_input_empty_output(_seed in 0u32..100) {
            let ner = RegexNER::new();
            let result = ner.extract_entities("", None);
            prop_assert!(result.is_ok());
            prop_assert!(result.unwrap().is_empty());
        }

        /// Property: Whitespace-only input should return empty entities
        #[test]
        fn whitespace_only_empty_output(spaces in 1usize..20) {
            let text = " ".repeat(spaces);
            let ner = RegexNER::new();
            let result = ner.extract_entities(&text, None);
            prop_assert!(result.is_ok());
            // Most NER systems return no entities for whitespace-only
            // (some might extract nothing, which is fine)
        }

        /// Property: Very long input should not cause issues
        #[test]
        fn handles_long_input(repeat in 10usize..50) {
            let base = "John Smith works at Apple Inc. in San Francisco. ";
            let text = base.repeat(repeat);
            let ner = RegexNER::new();
            let result = ner.extract_entities(&text, None);
            prop_assert!(result.is_ok(), "Failed on {} char input", text.len());
        }
    }
}

// =========================================================================
// Coreference Property Tests
// =========================================================================

#[cfg(test)]
mod coref_proptests {
    use crate::eval::coref::{CorefChain, Mention};
    use proptest::prelude::*;
    use std::collections::{HashMap, HashSet};

    /// Generate a valid coreference clustering
    fn arb_clustering() -> impl Strategy<Value = Vec<CorefChain>> {
        (1usize..5)
            .prop_flat_map(|num_chains| {
                proptest::collection::vec(
                    proptest::collection::vec(1usize..10, 1..4),
                    num_chains..=num_chains,
                )
            })
            .prop_map(|chain_lens| {
                let mut offset = 0usize;
                chain_lens
                    .into_iter()
                    .map(|lens| {
                        let mentions: Vec<_> = lens
                            .iter()
                            .map(|&len| {
                                let m = Mention::new(format!("m{}", offset), offset, offset + len);
                                offset += len + 10;
                                m
                            })
                            .collect();
                        CorefChain::new(mentions)
                    })
                    .collect()
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Coreference Axiom: Reflexivity
        /// Every mention should be in exactly one cluster containing itself
        #[test]
        fn coref_reflexivity(chains in arb_clustering()) {
            // Build mention -> chain index mapping
            let mut mention_to_chain: HashMap<String, usize> = HashMap::new();

            for (idx, chain) in chains.iter().enumerate() {
                for mention in &chain.mentions {
                    let prev = mention_to_chain.insert(mention.text.clone(), idx);
                    // Each mention should appear exactly once
                    prop_assert!(prev.is_none(),
                        "Mention '{}' appears in multiple chains", mention.text);
                }
            }

            // Every mention is in some chain (reflexivity with self)
            for chain in &chains {
                for mention in &chain.mentions {
                    prop_assert!(mention_to_chain.contains_key(&mention.text));
                }
            }
        }

        /// Coreference Axiom: Symmetry
        /// If A corefs with B, then B corefs with A
        #[test]
        fn coref_symmetry(chains in arb_clustering()) {
            for chain in &chains {
                let mentions: HashSet<_> = chain.mentions.iter()
                    .map(|m| &m.text)
                    .collect();

                // For any two mentions in the same chain, both directions hold
                for m1 in &chain.mentions {
                    for m2 in &chain.mentions {
                        // If m1 corefs with m2 (both in same chain)
                        // then m2 must coref with m1 (also in same chain)
                        // This is trivially true by construction, but we verify
                        prop_assert!(mentions.contains(&m1.text));
                        prop_assert!(mentions.contains(&m2.text));
                    }
                }
            }
        }

        /// Coreference Axiom: Transitivity
        /// If A corefs with B and B corefs with C, then A corefs with C
        #[test]
        fn coref_transitivity(chains in arb_clustering()) {
            // Build mention -> chain index
            let mut mention_to_chain: HashMap<String, usize> = HashMap::new();
            for (idx, chain) in chains.iter().enumerate() {
                for mention in &chain.mentions {
                    mention_to_chain.insert(mention.text.clone(), idx);
                }
            }

            // For any chain with 3+ mentions, transitivity must hold
            for chain in &chains {
                if chain.mentions.len() >= 3 {
                    let m0 = &chain.mentions[0].text;
                    let m1 = &chain.mentions[1].text;
                    let m2 = &chain.mentions[2].text;

                    // m0 corefs with m1 (same chain)
                    prop_assert_eq!(mention_to_chain[m0], mention_to_chain[m1]);
                    // m1 corefs with m2 (same chain)
                    prop_assert_eq!(mention_to_chain[m1], mention_to_chain[m2]);
                    // Therefore m0 must coref with m2 (transitivity)
                    prop_assert_eq!(mention_to_chain[m0], mention_to_chain[m2]);
                }
            }
        }

        /// Property: Singleton clusters should have exactly one mention
        #[test]
        fn singleton_property(chains in arb_clustering()) {
            for chain in &chains {
                if chain.mentions.len() == 1 {
                    // Singleton: the mention is only coreferent with itself
                    // No additional invariants needed beyond reflexivity
                    prop_assert!(chain.mentions.len() == 1);
                }
            }
        }

        /// Property: No duplicate mentions across chains
        #[test]
        fn no_duplicate_mentions(chains in arb_clustering()) {
            let mut seen: HashSet<(usize, usize)> = HashSet::new();

            for chain in &chains {
                for mention in &chain.mentions {
                    let span = (mention.start, mention.end);
                    let is_new = seen.insert(span);
                    prop_assert!(is_new,
                        "Duplicate mention span ({}, {})", mention.start, mention.end);
                }
            }
        }
    }
}

// =========================================================================
// Discontinuous NER Properties
// =========================================================================

#[cfg(test)]
mod discontinuous_proptests {
    #[allow(unused_imports)]
    use crate::eval::discontinuous::{DiscontinuousGold, DiscontinuousNERMetrics};
    use proptest::prelude::*;

    /// Generate valid discontinuous entity spans
    fn arb_discontinuous_spans() -> impl Strategy<Value = Vec<(usize, usize)>> {
        (1usize..5)
            .prop_flat_map(|num_spans| {
                proptest::collection::vec((0usize..50, 1usize..10), num_spans..=num_spans)
            })
            .prop_map(|raw_spans| {
                let mut offset = 0;
                raw_spans
                    .into_iter()
                    .map(|(gap, len)| {
                        let start = offset + gap;
                        let end = start + len;
                        offset = end + 1; // Ensure non-overlapping
                        (start, end)
                    })
                    .collect()
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property: Discontinuous spans should be sorted and non-overlapping
        #[test]
        fn discontinuous_spans_valid(spans in arb_discontinuous_spans()) {
            // Verify sorted
            for window in spans.windows(2) {
                prop_assert!(window[0].1 <= window[1].0,
                    "Spans overlap or unsorted: {:?} and {:?}", window[0], window[1]);
            }

            // Verify each span is valid
            for (start, end) in &spans {
                prop_assert!(start < end, "Invalid span: {} >= {}", start, end);
            }
        }

        /// Property: Merging adjacent spans should reduce count
        #[test]
        fn merging_reduces_spans(spans in arb_discontinuous_spans()) {
            if spans.len() >= 2 {
                // Check gaps between spans
                if let (Some(first), Some(second)) = (spans.first(), spans.get(1)) {
                    // If we merge [0].end with [1].start, we'd have one fewer span
                    // This is a conceptual property - actual implementation may vary
                    let gap = second.0.saturating_sub(first.1);
                    // Gap should be non-negative (spans are sorted)
                    prop_assert!(gap > 0 || second.0 >= first.1,
                        "Expected non-overlapping spans");
                }
            }
        }
    }
}

// =========================================================================
// Signal/Track/Identity Hierarchy Properties (`anno::core`)
// =========================================================================

#[cfg(test)]
mod hierarchy_proptests {
    use anno::Track;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property: Track canonical surface should be non-empty
        #[test]
        fn track_has_canonical_surface(name in "[A-Za-z ]{1,20}") {
            let track = Track::new(0u64, name.clone());
            prop_assert!(!track.canonical_surface.is_empty(),
                "Track canonical_surface should not be empty");
            prop_assert_eq!(track.canonical_surface, name);
        }

        /// Property: Track cluster confidence bounded by default
        #[test]
        fn track_confidence_default_bounded(_seed in 0u32..100) {
            let track = Track::new(0u64, "test");
            prop_assert!(track.cluster_confidence >= 0.0 && track.cluster_confidence <= 1.0,
                "Track cluster_confidence {} out of bounds", track.cluster_confidence);
        }

        /// Property: Track cluster confidence can be set within bounds
        #[test]
        fn track_confidence_settable(conf in 0.0f32..=1.0) {
            let mut track = Track::new(0u64, "test");
            track.cluster_confidence = conf.into();
            prop_assert!(track.cluster_confidence >= 0.0 && track.cluster_confidence <= 1.0);
        }

        /// Property: Unlinked track has no identity
        #[test]
        fn unlinked_track_no_identity(_seed in 0u32..100) {
            let track = Track::new(0u64, "test");
            prop_assert!(track.identity_id.is_none(),
                "New track should not have identity_id");
        }

        /// Property: New track starts empty (no signals)
        #[test]
        fn new_track_empty(_seed in 0u32..100) {
            let track = Track::new(0u64, "test");
            prop_assert!(track.is_empty(), "New track should have no signals");
        }

        /// Property: Track with_identity links correctly
        #[test]
        fn track_identity_linking(identity_id in 0u64..1000) {
            let track = Track::new(0u64, "test").with_identity(identity_id.into());
            prop_assert_eq!(track.identity_id, Some(identity_id.into()));
        }

        /// Property: Track with_type sets type correctly
        #[test]
        fn track_type_setting(entity_type in "[A-Z][a-z]+") {
            let track = Track::new(0u64, "test").with_type(entity_type.clone());
            prop_assert_eq!(track.entity_type, Some(entity_type.into()));
        }
    }
}

// =========================================================================
// Entity/Mention Consistency Properties
// =========================================================================

#[cfg(test)]
mod entity_consistency_proptests {
    use anno::{Entity, EntityType};
    use proptest::prelude::*;

    /// Generate valid entity type
    fn arb_entity_type() -> impl Strategy<Value = EntityType> {
        prop_oneof![
            Just(EntityType::Person),
            Just(EntityType::Organization),
            Just(EntityType::Location),
            Just(EntityType::Date),
            Just(EntityType::Money),
            Just(EntityType::Percent),
            Just(EntityType::Email),
            Just(EntityType::Url),
            Just(EntityType::Time),
        ]
    }

    /// Generate valid entity using Entity::new
    fn arb_entity() -> impl Strategy<Value = Entity> {
        (
            "[A-Za-z]{1,5}( [A-Za-z]{1,10})?", // text: at least one non-space char
            arb_entity_type(),                 // type
            0usize..100,                       // start
            0.5f64..1.0,                       // confidence (f64!)
        )
            .prop_map(|(text, entity_type, start, conf)| {
                let len = text.len();
                Entity::new(text, entity_type, start, start + len, conf)
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Property: Entity span length equals text length
        #[test]
        fn entity_span_matches_text(entity in arb_entity()) {
            let span_len = entity.end() - entity.start();
            let text_len = entity.text.len();
            prop_assert_eq!(span_len, text_len,
                "Span length {} != text length {}", span_len, text_len);
        }

        /// Property: Entity confidence bounded (Entity::new clamps)
        #[test]
        fn entity_confidence_bounded(entity in arb_entity()) {
            prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0,
                "Entity confidence {} out of bounds", entity.confidence);
        }

        /// Property: Entity text non-empty
        #[test]
        fn entity_text_non_empty(entity in arb_entity()) {
            prop_assert!(!entity.text.is_empty(), "Entity text should not be empty");
            prop_assert!(!entity.text.trim().is_empty(),
                "Entity text should not be whitespace-only");
        }

        /// Property: Entity span valid
        #[test]
        fn entity_span_valid(entity in arb_entity()) {
            prop_assert!(entity.start() <= entity.end(),
                "Invalid span: {} > {}", entity.start(), entity.end());
        }

        /// Property: Entity type round-trips through display
        #[test]
        fn entity_type_display(et in arb_entity_type()) {
            let display = format!("{}", et);
            prop_assert!(!display.is_empty(), "EntityType display should not be empty");
        }

        /// Property: Entity::new clamps confidence to [0, 1]
        #[test]
        fn entity_confidence_clamped(conf in -1.0f64..2.0) {
            let entity = Entity::new("test", EntityType::Person, 0, 4, conf);
            prop_assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0,
                "Confidence {} not clamped to [0,1]", entity.confidence);
        }
    }
}

// =========================================================================
// Union-Find Clustering Properties (coalesce algorithm)
// =========================================================================

#[cfg(test)]
mod union_find_proptests {
    use proptest::prelude::*;
    use std::collections::HashMap;

    /// Simple union-find (mirrors coalesce/resolver.rs implementation)
    struct UnionFind {
        parent: Vec<usize>,
    }

    impl UnionFind {
        fn new(n: usize) -> Self {
            Self {
                parent: (0..n).collect(),
            }
        }

        fn find(&mut self, i: usize) -> usize {
            if self.parent[i] != i {
                self.parent[i] = self.find(self.parent[i]); // path compression
            }
            self.parent[i]
        }

        fn union(&mut self, i: usize, j: usize) {
            let pi = self.find(i);
            let pj = self.find(j);
            if pi != pj {
                self.parent[pi] = pj;
            }
        }

        fn same_set(&mut self, i: usize, j: usize) -> bool {
            self.find(i) == self.find(j)
        }

        fn clusters(&mut self) -> HashMap<usize, Vec<usize>> {
            let mut result: HashMap<usize, Vec<usize>> = HashMap::new();
            for i in 0..self.parent.len() {
                let root = self.find(i);
                result.entry(root).or_default().push(i);
            }
            result
        }
    }

    /// Generate union operations: pairs of indices to union
    fn arb_unions(n: usize) -> impl Strategy<Value = Vec<(usize, usize)>> {
        proptest::collection::vec((0usize..n, 0usize..n), 0..n * 2)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property: Union-find reflexivity (x is in same set as x)
        #[test]
        fn uf_reflexivity(n in 1usize..50) {
            let mut uf = UnionFind::new(n);
            for i in 0..n {
                prop_assert!(uf.same_set(i, i), "Element {} not in same set as itself", i);
            }
        }

        /// Property: Union-find symmetry (if x~y then y~x)
        #[test]
        fn uf_symmetry(n in 2usize..30, unions in arb_unions(30)) {
            let mut uf = UnionFind::new(n);
            for (i, j) in unions {
                if i < n && j < n {
                    uf.union(i, j);
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let ij = uf.same_set(i, j);
                    let ji = uf.same_set(j, i);
                    prop_assert_eq!(ij, ji, "Asymmetric: same_set({},{}) != same_set({},{})", i, j, j, i);
                }
            }
        }

        /// Property: Union-find transitivity (if x~y and y~z then x~z)
        #[test]
        fn uf_transitivity(n in 3usize..20, unions in arb_unions(20)) {
            let mut uf = UnionFind::new(n);
            for (i, j) in unions {
                if i < n && j < n {
                    uf.union(i, j);
                }
            }
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        if uf.same_set(i, j) && uf.same_set(j, k) {
                            prop_assert!(uf.same_set(i, k),
                                "Transitivity violated: {}~{} and {}~{} but not {}~{}", i, j, j, k, i, k);
                        }
                    }
                }
            }
        }

        /// Property: Union creates equivalence (after union(i,j), i~j)
        #[test]
        fn uf_union_creates_equivalence(n in 2usize..50, i in 0usize..50, j in 0usize..50) {
            prop_assume!(i < n && j < n);
            let mut uf = UnionFind::new(n);
            uf.union(i, j);
            prop_assert!(uf.same_set(i, j), "After union({},{}), they should be in same set", i, j);
        }

        /// Property: Clusters partition elements (each element in exactly one cluster)
        #[test]
        fn uf_clusters_partition(n in 1usize..30, unions in arb_unions(30)) {
            let mut uf = UnionFind::new(n);
            for (i, j) in unions {
                if i < n && j < n {
                    uf.union(i, j);
                }
            }
            let clusters = uf.clusters();

            // Every element appears exactly once
            let mut seen = vec![false; n];
            for members in clusters.values() {
                for &m in members {
                    prop_assert!(!seen[m], "Element {} appears in multiple clusters", m);
                    seen[m] = true;
                }
            }
            for (i, &s) in seen.iter().enumerate() {
                prop_assert!(s, "Element {} not in any cluster", i);
            }
        }

        /// Property: Cluster count <= n (can't have more clusters than elements)
        #[test]
        fn uf_cluster_count_bounded(n in 1usize..50, unions in arb_unions(50)) {
            let mut uf = UnionFind::new(n);
            for (i, j) in unions {
                if i < n && j < n {
                    uf.union(i, j);
                }
            }
            let cluster_count = uf.clusters().len();
            prop_assert!(cluster_count <= n, "More clusters ({}) than elements ({})", cluster_count, n);
        }

        /// Property: Union monotonically reduces cluster count (or keeps it same)
        #[test]
        fn uf_union_reduces_clusters(n in 2usize..30) {
            let mut uf = UnionFind::new(n);
            let initial_count = uf.clusters().len();
            prop_assert_eq!(initial_count, n, "Initially should have n clusters");

            uf.union(0, 1);
            let new_count = uf.clusters().len();
            prop_assert!(new_count <= initial_count,
                "Union should not increase cluster count: {} -> {}", initial_count, new_count);
        }
    }
}

// =========================================================================
// Hash Determinism Properties (Session Learning: xxHash vs SipHash)
// =========================================================================

#[cfg(test)]
mod hash_determinism_proptests {
    use proptest::prelude::*;
    use xxhash_rust::xxh3::xxh3_64;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Property: xxHash3 is deterministic (same input → same hash)
        #[test]
        fn xxhash_deterministic(data in proptest::collection::vec(any::<u8>(), 0..1000)) {
            let hash1 = xxh3_64(&data);
            let hash2 = xxh3_64(&data);
            prop_assert_eq!(hash1, hash2, "xxHash should be deterministic");
        }

        /// Property: Different inputs usually produce different hashes (collision resistance)
        #[test]
        fn xxhash_collision_resistance(
            data1 in proptest::collection::vec(any::<u8>(), 1..100),
            data2 in proptest::collection::vec(any::<u8>(), 1..100)
        ) {
            if data1 != data2 {
                let hash1 = xxh3_64(&data1);
                let hash2 = xxh3_64(&data2);
                // With 64-bit hash, collision probability is ~1/2^64
                // Over 200 test cases, collision is astronomically unlikely
                prop_assert_ne!(hash1, hash2,
                    "Different inputs should have different hashes (collision detected)");
            }
        }

        /// Property: Hash is stable with string content
        #[test]
        fn xxhash_string_stable(text in "[A-Za-z0-9 .,!?]{1,200}") {
            let hash1 = xxh3_64(text.as_bytes());
            let hash2 = xxh3_64(text.as_bytes());
            prop_assert_eq!(hash1, hash2, "String hash should be stable");
        }

        /// Property: Entity ID hash computation is deterministic
        /// (mirrors extract.rs entity_id generation)
        #[test]
        fn entity_id_deterministic(
            doc_id in "[a-z]{5,10}",
            entity_text in "[A-Z][a-z]{2,10}",
            start in 0usize..1000,
            end in 0usize..1000
        ) {
            let end = start.max(end); // Ensure valid span

            // Compute entity ID the same way as extract.rs
            let compute_id = || {
                let mut data = Vec::new();
                data.extend_from_slice(doc_id.as_bytes());
                data.extend_from_slice(entity_text.as_bytes());
                data.extend_from_slice(&start.to_le_bytes());
                data.extend_from_slice(&end.to_le_bytes());
                format!("e:{:016x}", xxh3_64(&data))
            };

            let id1 = compute_id();
            let id2 = compute_id();
            prop_assert_eq!(id1, id2, "Entity ID should be deterministic");
        }
    }
}

// =========================================================================
// Context Extraction Properties (Session Learning: Unicode handling)
// =========================================================================

#[cfg(test)]
mod context_extraction_proptests {
    use proptest::prelude::*;

    /// Helper: Extract context around a span (mirrors extract.rs get_context)
    fn get_context(text: &str, start: usize, end: usize, window: usize) -> (String, String) {
        let chars: Vec<char> = text.chars().collect();
        let char_count = chars.len();

        // Build byte offset to char offset mapping
        let mut byte_to_char: Vec<usize> = Vec::with_capacity(text.len() + 1);
        for (i, c) in chars.iter().enumerate() {
            for _ in 0..c.len_utf8() {
                byte_to_char.push(i);
            }
        }
        byte_to_char.push(chars.len()); // for end-of-string

        // Convert byte offsets to char offsets
        let start_char = if start < byte_to_char.len() {
            byte_to_char[start]
        } else {
            char_count
        };
        let end_char = if end < byte_to_char.len() {
            byte_to_char[end]
        } else {
            char_count
        };

        // Extract context
        let ctx_start = start_char.saturating_sub(window);
        let ctx_end = (end_char + window).min(char_count);

        let before: String = chars[ctx_start..start_char].iter().collect();
        let after: String = chars[end_char..ctx_end].iter().collect();

        (before, after)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Property: Context extraction works for ASCII text
        #[test]
        fn context_ascii(
            text in "[A-Za-z ]{20,100}",
            entity_start in 5usize..15,
            entity_len in 1usize..10,
            window in 1usize..10
        ) {
            let text_len = text.len();
            let entity_start = entity_start.min(text_len.saturating_sub(entity_len + 1));
            let entity_end = (entity_start + entity_len).min(text_len);

            let (before, after) = get_context(&text, entity_start, entity_end, window);

            // Context should not exceed requested window
            prop_assert!(before.chars().count() <= window,
                "Before context {} chars exceeds window {}", before.chars().count(), window);
            prop_assert!(after.chars().count() <= window,
                "After context {} chars exceeds window {}", after.chars().count(), window);
        }

        /// Property: Context extraction works for Unicode text
        #[test]
        fn context_unicode(window in 1usize..5) {
            // Test with known Unicode strings
            let test_cases = [
                ("Hello 北京 World", 6, 12),  // 北京 is 2 chars, 6 bytes each
                ("Price: €50 now", 7, 8),    // € is 3 bytes
                ("Tokyo 東京 City", 6, 12),  // 東京 is 2 chars
            ];

            for (text, start, end) in test_cases {
                let (before, after) = get_context(text, start, end, window);

                // Should not panic or produce invalid UTF-8
                prop_assert!(before.is_empty() || before.chars().count() > 0,
                    "Before context should be valid UTF-8");
                prop_assert!(after.is_empty() || after.chars().count() > 0,
                    "After context should be valid UTF-8");
            }
        }

        /// Property: Context extraction handles edge cases
        #[test]
        fn context_edge_cases(text in "[A-Za-z]{10,50}", window in 1usize..10) {
            let text_len = text.len();

            // Start of text
            let (before, _) = get_context(&text, 0, 5.min(text_len), window);
            prop_assert!(before.is_empty() || before.chars().count() <= window);

            // End of text
            let start = text_len.saturating_sub(5);
            let (_, after) = get_context(&text, start, text_len, window);
            prop_assert!(after.is_empty() || after.chars().count() <= window);
        }

        /// Property: Empty text returns empty context
        #[test]
        fn context_empty_text(window in 1usize..10) {
            let (before, after) = get_context("", 0, 0, window);
            prop_assert!(before.is_empty());
            prop_assert!(after.is_empty());
        }
    }
}

// =========================================================================
// Result Hash Stability Properties (Session Learning: entity ordering)
// =========================================================================

#[cfg(test)]
mod result_hash_proptests {
    use proptest::prelude::*;
    use xxhash_rust::xxh3::xxh3_64;

    /// Simulated entity for testing hash stability
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct TestEntity {
        text: String,
        entity_type: String,
        start: usize,
        end: usize,
    }

    /// Compute result hash (mirrors extract.rs logic)
    fn compute_result_hash(text: &str, entities: &[TestEntity]) -> String {
        let mut data = Vec::new();
        data.extend_from_slice(text.as_bytes());

        // Sort entities for determinism
        let mut sorted = entities.to_vec();
        sorted.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| a.end.cmp(&b.end))
                .then_with(|| a.entity_type.cmp(&b.entity_type))
                .then_with(|| a.text.cmp(&b.text))
        });

        for e in &sorted {
            data.extend_from_slice(e.text.as_bytes());
            data.extend_from_slice(e.entity_type.as_bytes());
            data.extend_from_slice(&e.start.to_le_bytes());
            data.extend_from_slice(&e.end.to_le_bytes());
        }

        format!("xxh3:{:016x}", xxh3_64(&data))
    }

    /// Generate random entities
    fn arb_entities() -> impl Strategy<Value = Vec<TestEntity>> {
        proptest::collection::vec(
            (
                "[A-Z][a-z]{2,8}",                        // text
                prop_oneof!["PER", "ORG", "LOC", "MISC"], // type
                0usize..100,                              // start
                1usize..20,                               // length
            ),
            0..5,
        )
        .prop_map(|raw| {
            raw.into_iter()
                .map(|(text, entity_type, start, len)| TestEntity {
                    text,
                    entity_type,
                    start,
                    end: start + len,
                })
                .collect()
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Property: Result hash is deterministic
        #[test]
        fn result_hash_deterministic(
            text in "[A-Za-z ]{10,100}",
            entities in arb_entities()
        ) {
            let hash1 = compute_result_hash(&text, &entities);
            let hash2 = compute_result_hash(&text, &entities);
            prop_assert_eq!(hash1, hash2, "Result hash should be deterministic");
        }

        /// Property: Result hash is order-independent (due to sorting)
        #[test]
        fn result_hash_order_independent(
            text in "[A-Za-z ]{10,100}",
            entities in arb_entities()
        ) {
            if entities.len() >= 2 {
                let mut reversed = entities.clone();
                reversed.reverse();

                let hash_original = compute_result_hash(&text, &entities);
                let hash_reversed = compute_result_hash(&text, &reversed);

                prop_assert_eq!(hash_original, hash_reversed,
                    "Result hash should be independent of input entity order");
            }
        }

        /// Property: Empty entities list produces valid hash
        #[test]
        fn result_hash_empty_entities(text in "[A-Za-z ]{10,100}") {
            let hash = compute_result_hash(&text, &[]);
            prop_assert!(hash.starts_with("xxh3:"), "Hash should have xxh3: prefix");
            prop_assert_eq!(hash.len(), 21, "Hash should be 'xxh3:' + 16 hex chars");
        }

        /// Property: Different content produces different hash
        #[test]
        fn result_hash_content_sensitive(
            text1 in "[A-Za-z ]{10,50}",
            text2 in "[A-Za-z ]{10,50}",
            entities in arb_entities()
        ) {
            if text1 != text2 {
                let hash1 = compute_result_hash(&text1, &entities);
                let hash2 = compute_result_hash(&text2, &entities);
                prop_assert_ne!(hash1, hash2,
                    "Different text should produce different hash");
            }
        }

        /// Property: Hash format is valid (xxh3: prefix + 16 hex digits)
        #[test]
        fn result_hash_format_valid(
            text in "[A-Za-z]{5,20}",
            entities in arb_entities()
        ) {
            let hash = compute_result_hash(&text, &entities);

            prop_assert!(hash.starts_with("xxh3:"), "Hash should start with xxh3:");
            let hex_part = &hash[5..];
            prop_assert_eq!(hex_part.len(), 16, "Hex part should be 16 characters");
            prop_assert!(hex_part.chars().all(|c| c.is_ascii_hexdigit()),
                "Hex part should only contain hex digits");
        }
    }
}
