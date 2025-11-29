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
}
