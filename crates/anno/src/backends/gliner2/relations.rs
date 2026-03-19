//! GLiNER2 relation extraction heuristics.
//!
//! Delegates to the shared backend-agnostic heuristic in
//! [`crate::backends::inference::relation_extraction::extract_relation_triples_simple`].

use crate::Entity;

/// Extract relations using the shared heuristic pipeline (triggers + type fallback + dedup).
#[cfg(any(feature = "onnx", feature = "candle"))]
pub(crate) fn extract_relations_heuristic(
    entities: &[Entity],
    text: &str,
    relation_types: &[&str],
    threshold: f32,
) -> Vec<crate::backends::inference::RelationTriple> {
    use crate::backends::inference::relation_extraction::{
        extract_relation_triples_simple, RelationExtractionConfig,
    };

    let config = RelationExtractionConfig {
        threshold: crate::Confidence::new(threshold as f64),
        max_span_distance: 120,
        extract_triggers: false,
    };
    extract_relation_triples_simple(entities, text, relation_types, &config)
}

#[cfg(test)]
#[cfg(any(feature = "onnx", feature = "candle"))]
mod tests {
    use super::*;

    /// Helper: build a minimal Entity at given char offsets.
    fn entity(text: &str, ty: crate::EntityType, start: usize, end: usize) -> Entity {
        Entity::new(text, ty, start, end, 0.9)
    }

    // Tests for get_likely_relations are in relation_extraction.rs (shared implementation).
    // These tests verify the delegation wrapper works correctly.

    #[test]
    fn trigger_pattern_extracts_works_for() {
        let text = "Alice works for Acme Corp in the city";
        let entities = vec![
            entity("Alice", crate::EntityType::Person, 0, 5),
            entity("Acme Corp", crate::EntityType::Organization, 16, 25),
        ];
        let rel_types: Vec<&str> = vec!["WORKS_FOR", "LOCATED_IN"];
        let rels = extract_relations_heuristic(&entities, text, &rel_types, 0.0);
        let labels: Vec<&str> = rels.iter().map(|r| r.relation_type.as_str()).collect();
        assert!(
            labels.contains(&"WORKS_FOR"),
            "expected WORKS_FOR from trigger 'works for', got {labels:?}"
        );
    }

    #[test]
    fn empty_entities_returns_empty() {
        let rels = extract_relations_heuristic(&[], "some text", &["WORKS_FOR"], 0.0);
        assert!(rels.is_empty());
    }

    #[test]
    fn single_entity_returns_empty() {
        let entities = vec![entity("Alice", crate::EntityType::Person, 0, 5)];
        let rels = extract_relations_heuristic(&entities, "Alice is here", &[], 0.0);
        assert!(rels.is_empty(), "need at least two entities for a relation");
    }

    #[test]
    fn high_threshold_filters_low_confidence() {
        let text = "Alice works for Bob";
        let entities = vec![
            entity("Alice", crate::EntityType::Person, 0, 5),
            entity("Bob", crate::EntityType::Person, 16, 19),
        ];
        let rels = extract_relations_heuristic(&entities, text, &[], 0.99);
        // With threshold 0.99, most heuristic relations should be filtered out.
        // (proximity * base_score * avg_confidence will rarely exceed 0.99)
        assert!(
            rels.is_empty() || rels.iter().all(|r| r.confidence >= 0.99),
            "all surviving relations must meet threshold"
        );
    }

    #[test]
    fn deduplicates_to_top_per_pair() {
        // "born in" can trigger both a trigger-pattern and a type-based relation.
        // The output should have at most one relation per directed (head, tail) pair.
        let text = "Alice born in Paris";
        let entities = vec![
            entity("Alice", crate::EntityType::Person, 0, 5),
            entity("Paris", crate::EntityType::Location, 14, 19),
        ];
        let rels = extract_relations_heuristic(&entities, text, &[], 0.0);
        let pair_count = rels
            .iter()
            .filter(|r| r.head_idx == 0 && r.tail_idx == 1)
            .count();
        assert!(
            pair_count <= 1,
            "expected at most 1 relation per directed pair, got {pair_count}"
        );
    }

    /// N7: Reverse-duplicate relations (A→B and B→A) should be deduplicated.
    #[test]
    fn no_reverse_duplicate_relations() {
        // Two entities that could trigger relations in both directions
        let text = "Tim Cook works at Apple Inc. in Cupertino";
        let entities = vec![
            entity("Tim Cook", crate::EntityType::Person, 0, 8),
            entity("Apple Inc.", crate::EntityType::Organization, 18, 28),
            entity("Cupertino", crate::EntityType::Location, 32, 41),
        ];
        let rels = extract_relations_heuristic(&entities, text, &[], 0.0);

        // For each undirected pair, there should be at most one relation
        let mut seen_undirected = std::collections::HashSet::new();
        for r in &rels {
            let canonical = if r.head_idx <= r.tail_idx {
                (r.head_idx, r.tail_idx)
            } else {
                (r.tail_idx, r.head_idx)
            };
            assert!(
                seen_undirected.insert(canonical),
                "Found duplicate relation for pair ({}, {}): {:?}",
                r.head_idx,
                r.tail_idx,
                r.relation_type
            );
        }
    }
}
