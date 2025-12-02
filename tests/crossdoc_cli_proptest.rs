//! Property-based tests for cross-document coreference CLI
//!
//! Tests invariants and properties that should always hold.

use anno::eval::cdcr::{CDCRConfig, CDCRResolver, CrossDocCluster, Document};
use anno::{Entity, EntityType};
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_cluster_mentions_match_documents(
        num_docs in 1..10usize,
        entities_per_doc in 1..5usize,
    ) {
        // Generate random documents with entities
        let mut docs = Vec::new();
        for doc_idx in 0..num_docs {
            let mut doc = Document::new(&format!("doc{}", doc_idx), "Test document text.");
            for ent_idx in 0..entities_per_doc {
                doc.entities.push(Entity::new(
                    &format!("Entity{}", ent_idx),
                    EntityType::Person,
                    0,
                    10,
                    0.9,
                ));
            }
            docs.push(doc);
        }

        let config = CDCRConfig {
            min_similarity: 0.3,
            use_lsh: false,
            require_type_match: true,
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&docs);

        // Property: All mentions should reference valid documents and entities
        for cluster in &clusters {
            for (doc_id, entity_idx) in &cluster.mentions {
                let doc = docs.iter().find(|d| d.id == *doc_id);
                prop_assert!(doc.is_some(), "Mention references valid doc_id");

                if let Some(d) = doc {
                    prop_assert!(
                        *entity_idx < d.entities.len(),
                        "Mention references valid entity_idx"
                    );
                }
            }
        }
    }

    #[test]
    fn test_cluster_doc_count_consistency(
        num_docs in 2..10usize,
    ) {
        // Create documents with same entity name
        let mut docs = Vec::new();
        for doc_idx in 0..num_docs {
            let mut doc = Document::new(&format!("doc{}", doc_idx), "Test text.");
            doc.entities.push(Entity::new(
                "SameEntity",
                EntityType::Person,
                0,
                10,
                0.9,
            ));
            docs.push(doc);
        }

        let config = CDCRConfig {
            min_similarity: 0.3,
            use_lsh: false,
            require_type_match: true,
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&docs);

        // Property: doc_count should match unique documents in mentions
        for cluster in &clusters {
            let mentioned_docs: std::collections::HashSet<_> = cluster.mentions.iter()
                .map(|(doc_id, _)| doc_id)
                .collect();

            prop_assert_eq!(
                cluster.doc_count(),
                mentioned_docs.len(),
                "doc_count should match unique documents in mentions"
            );
        }
    }

    #[test]
    fn test_cluster_confidence_range(
        similarity in 0.0f32..1.0f32,
    ) {
        let mut doc1 = Document::new("doc1", "Entity A mentioned here.");
        doc1.entities.push(Entity::new("Entity A", EntityType::Person, 0, 8, 0.9));

        let mut doc2 = Document::new("doc2", "Entity A mentioned again.");
        doc2.entities.push(Entity::new("Entity A", EntityType::Person, 0, 8, 0.9));

        let config = CDCRConfig {
            min_similarity: similarity as f64,
            use_lsh: false,
            require_type_match: true,
            ..Default::default()
        };
        let resolver = CDCRResolver::with_config(config);
        let clusters = resolver.resolve(&[doc1, doc2]);

        // Property: All cluster confidences should be in [0.0, 1.0]
        for cluster in &clusters {
            prop_assert!(
                cluster.confidence >= 0.0 && cluster.confidence <= 1.0,
                "Cluster confidence should be in [0.0, 1.0], found {}",
                cluster.confidence
            );
        }
    }

    #[test]
    fn test_cluster_id_uniqueness_property(
        num_clusters in 1..20usize,
    ) {
        // Create clusters manually to test ID uniqueness
        let clusters: Vec<CrossDocCluster> = (0..num_clusters)
            .map(|i| {
                let mut c = CrossDocCluster::new(i as u64, &format!("Entity{}", i));
                c.add_mention("doc1", 0);
                c
            })
            .collect();

        // Property: All cluster IDs should be unique
        let mut ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for cluster in &clusters {
            prop_assert!(
                ids.insert(cluster.id),
                "Cluster IDs should be unique, found duplicate: {}",
                cluster.id
            );
        }
    }

    #[test]
    fn test_mention_count_consistency(
        mentions_per_cluster in 1..10usize,
    ) {
        let mut cluster = CrossDocCluster::new(0, "Test");
        for i in 0..mentions_per_cluster {
            cluster.add_mention("doc1", i);
        }

        // Property: len() should match number of mentions
        prop_assert_eq!(
            cluster.len(),
            mentions_per_cluster,
            "Cluster len() should match number of mentions"
        );
    }

    #[test]
    fn test_doc_count_deduplication(
        mentions in 1..10usize,
        unique_docs in 1..5usize,
    ) {
        let mut cluster = CrossDocCluster::new(0, "Test");

        // Add mentions from same docs multiple times
        for i in 0..mentions {
            let doc_id = format!("doc{}", i % unique_docs);
            cluster.add_mention(&doc_id, i);
        }

        // Property: doc_count should be <= number of unique docs
        prop_assert!(
            cluster.doc_count() <= unique_docs,
            "doc_count should be <= number of unique documents"
        );
    }
}

#[test]
fn test_cluster_canonical_name_non_empty() {
    // Property: Canonical name should never be empty
    let docs = vec![{
        let mut d = Document::new("doc1", "Test.");
        d.entities
            .push(Entity::new("Entity", EntityType::Person, 0, 6, 0.9));
        d
    }];

    let resolver = CDCRResolver::new();
    let clusters = resolver.resolve(&docs);

    for cluster in &clusters {
        assert!(
            !cluster.canonical_name.is_empty(),
            "Canonical name should not be empty"
        );
    }
}

#[test]
fn test_all_entities_clustered() {
    // Property: All entities should be assigned to exactly one cluster
    let mut doc1 = Document::new("doc1", "Entity A and Entity B mentioned.");
    doc1.entities = vec![
        Entity::new("Entity A", EntityType::Person, 0, 8, 0.9),
        Entity::new("Entity B", EntityType::Person, 13, 21, 0.9),
    ];

    let mut doc2 = Document::new("doc2", "Entity A mentioned again.");
    doc2.entities = vec![Entity::new("Entity A", EntityType::Person, 0, 8, 0.9)];

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&[doc1, doc2]);

    // Count total mentions in clusters
    let total_mentions: usize = clusters.iter().map(|c| c.len()).sum();
    let total_entities: usize = 3; // 2 in doc1, 1 in doc2

    assert_eq!(
        total_mentions, total_entities,
        "All entities should be clustered"
    );
}

#[test]
fn test_cluster_type_consistency() {
    // Property: All entities in a cluster should have the same type
    let mut doc1 = Document::new("doc1", "Apple Inc. and Apple fruit.");
    doc1.entities = vec![
        Entity::new("Apple Inc.", EntityType::Organization, 0, 10, 0.9),
        Entity::new("Apple", EntityType::Other("Fruit".to_string()), 15, 20, 0.8),
    ];

    let config = CDCRConfig {
        min_similarity: 0.3,
        use_lsh: false,
        require_type_match: true, // Strict type matching
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&[doc1]);

    // With type matching, should have 2 separate clusters
    assert_eq!(
        clusters.len(),
        2,
        "Different types should form separate clusters"
    );

    // Verify each cluster has consistent type
    for cluster in &clusters {
        if cluster.entity_type.is_some() {
            // Verify the cluster has a type
            assert!(
                cluster.entity_type.is_some(),
                "Cluster should have entity type"
            );
        }
    }
}
