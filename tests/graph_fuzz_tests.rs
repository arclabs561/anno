use anno::graph::GraphDocument;
use anno::Relation;
use proptest::prelude::*;

#[path = "fuzz_strategies.rs"]
mod fuzz_strategies;
use fuzz_strategies::entity_strategy;

proptest! {
    #[test]
    fn test_minimal(entities in proptest::collection::vec(entity_strategy(), 0..5)) {
        let graph = GraphDocument::from_extraction(&entities, &[], None);
        let _ = graph.node_count();
    }
    #[test]
    fn graph_export_networkx_valid_json(entities in proptest::collection::vec(entity_strategy(), 0..20)) {
        let graph = GraphDocument::from_extraction(&entities, &[], None);
        let nx_json = graph.to_networkx_json();
        let parsed: serde_json::Value = serde_json::from_str(&nx_json).expect("NetworkX JSON should be valid");
        prop_assert!(parsed.get("nodes").is_some());
        prop_assert!(parsed.get("links").is_some());
    }
}
