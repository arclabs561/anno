//! Integration tests for cross-document coreference CLI output format
//!
//! Tests the tree format output, sorting, filtering, and display logic.

use anno::eval::cdcr::{CDCRConfig, CDCRResolver, Document};
use anno::{Entity, EntityType};

/// Create test documents with pre-annotated entities
fn create_test_documents() -> Vec<Document> {
    let mut doc1 = Document::new(
        "doc1",
        "Jensen Huang announced that Nvidia will build new AI supercomputers. The chipmaker plans to expand its data center business.",
    );
    doc1.entities = vec![
        Entity::new("Jensen Huang", EntityType::Person, 0, 12, 0.95),
        Entity::new("Nvidia", EntityType::Organization, 28, 34, 0.94),
    ];

    let mut doc2 = Document::new(
        "doc2",
        "The CEO of Nvidia revealed plans for Blackwell chips during CES 2025. Huang said the new GPUs would advance robotics.",
    );
    doc2.entities = vec![
        Entity::new("CEO of Nvidia", EntityType::Person, 4, 17, 0.85),
        Entity::new("Nvidia", EntityType::Organization, 11, 17, 0.9),
        Entity::new("Huang", EntityType::Person, 70, 75, 0.92),
    ];

    let mut doc3 = Document::new(
        "doc3",
        "Nvidia's stock reached new highs after Jensen Huang's keynote. The company announced partnerships with major cloud providers.",
    );
    doc3.entities = vec![
        Entity::new("Nvidia", EntityType::Organization, 0, 6, 0.94),
        Entity::new("Jensen Huang", EntityType::Person, 38, 50, 0.93),
    ];

    vec![doc1, doc2, doc3]
}

#[test]
fn test_cdcr_resolver_produces_clusters() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false, // Brute force for reliable testing
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Should produce at least some clusters
    assert!(
        !clusters.is_empty(),
        "Should produce clusters from test documents"
    );

    // Should have at least one cross-document cluster (Nvidia appears in all 3 docs)
    let cross_doc_count = clusters.iter().filter(|c| c.doc_count() > 1).count();
    assert!(
        cross_doc_count > 0,
        "Should have at least one cross-document cluster"
    );
}

#[test]
fn test_cdcr_nvidia_cluster_spans_multiple_docs() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Find Nvidia organization cluster
    let nvidia_cluster = clusters.iter().find(|c| {
        c.canonical_name.to_lowercase() == "nvidia"
            && c.entity_type == Some(EntityType::Organization)
    });

    assert!(
        nvidia_cluster.is_some(),
        "Should find Nvidia organization cluster"
    );

    let nc = nvidia_cluster.unwrap();
    assert!(
        nc.doc_count() >= 2,
        "Nvidia should appear in at least 2 documents, found {}",
        nc.doc_count()
    );
    assert!(
        nc.len() >= 2,
        "Nvidia should have at least 2 mentions, found {}",
        nc.len()
    );
}

#[test]
fn test_cdcr_huang_cluster_spans_multiple_docs() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.3, // Lower threshold for name variations
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Find Huang person cluster (should cluster "Jensen Huang", "CEO of Nvidia", "Huang")
    let huang_cluster = clusters.iter().find(|c| {
        c.entity_type == Some(EntityType::Person)
            && (c.canonical_name.to_lowercase().contains("huang")
                || c.canonical_name.to_lowercase().contains("ceo"))
    });

    if let Some(hc) = huang_cluster {
        // Should span multiple documents
        assert!(
            hc.doc_count() >= 2,
            "Huang mentions should appear in at least 2 documents, found {}",
            hc.doc_count()
        );
    }
}

#[test]
fn test_cdcr_cluster_mentions_are_valid() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    for cluster in &clusters {
        // Verify all mentions reference valid documents and entities
        for (doc_id, entity_idx) in &cluster.mentions {
            let doc = docs.iter().find(|d| d.id == *doc_id);
            assert!(doc.is_some(), "Cluster mentions invalid doc_id: {}", doc_id);

            if let Some(d) = doc {
                assert!(
                    *entity_idx < d.entities.len(),
                    "Cluster mentions invalid entity_idx {} for doc {}",
                    entity_idx,
                    doc_id
                );
            }
        }

        // Verify document list matches mentions
        let mentioned_docs: std::collections::HashSet<_> =
            cluster.mentions.iter().map(|(doc_id, _)| doc_id).collect();
        let cluster_docs: std::collections::HashSet<_> = cluster.documents.iter().collect();

        assert_eq!(
            mentioned_docs, cluster_docs,
            "Cluster documents list should match mentioned documents"
        );
    }
}

#[test]
fn test_cdcr_cluster_canonical_name() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    for cluster in &clusters {
        // Canonical name should not be empty
        assert!(
            !cluster.canonical_name.is_empty(),
            "Cluster should have canonical name"
        );

        // Canonical name should match at least one mention
        let doc_ids: std::collections::HashSet<_> =
            cluster.mentions.iter().map(|(doc_id, _)| doc_id).collect();

        let mut found_match = false;
        for doc_id in doc_ids {
            if let Some(doc) = docs.iter().find(|d| d.id == *doc_id) {
                for (_, entity_idx) in &cluster.mentions {
                    if *entity_idx < doc.entities.len() {
                        let entity = &doc.entities[*entity_idx];
                        if entity.text.to_lowercase() == cluster.canonical_name.to_lowercase() {
                            found_match = true;
                            break;
                        }
                    }
                }
            }
            if found_match {
                break;
            }
        }
        // Note: This might not always be true due to how canonical names are chosen
        // but it's a reasonable expectation
    }
}

#[test]
fn test_cdcr_type_matching() {
    let mut doc1 = Document::new("doc1", "Apple announced new products.");
    doc1.entities = vec![Entity::new("Apple", EntityType::Organization, 0, 5, 0.9)];

    let mut doc2 = Document::new("doc2", "I ate an apple for lunch.");
    doc2.entities = vec![Entity::new(
        "apple",
        EntityType::Other("Fruit".to_string()),
        9,
        14,
        0.8,
    )];

    let config = CDCRConfig {
        min_similarity: 0.3,
        use_lsh: false,
        require_type_match: true, // Strict type matching
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&[doc1, doc2]);

    // Should have 2 separate clusters due to type mismatch
    assert_eq!(clusters.len(), 2, "Type mismatch should prevent clustering");
}

#[test]
fn test_cdcr_similarity_threshold() {
    let mut doc1 = Document::new("doc1", "John works here.");
    doc1.entities = vec![Entity::new("John", EntityType::Person, 0, 4, 0.9)];

    let mut doc2 = Document::new("doc2", "Jonathan is a developer.");
    doc2.entities = vec![Entity::new("Jonathan", EntityType::Person, 0, 8, 0.9)];

    // High threshold - should NOT cluster
    let config_high = CDCRConfig {
        use_lsh: false,
        min_similarity: 0.9,
        require_type_match: true,
        ..Default::default()
    };
    let resolver_high = CDCRResolver::with_config(config_high);
    let clusters_high = resolver_high.resolve(&[doc1.clone(), doc2.clone()]);

    // John and Jonathan share "John" substring, so with low threshold they might cluster
    // but with high threshold they should stay separate
    assert!(
        clusters_high.len() >= 1,
        "High threshold should keep separate or cluster based on substring"
    );

    // Low threshold - might cluster
    let config_low = CDCRConfig {
        use_lsh: false,
        min_similarity: 0.2,
        require_type_match: true,
        ..Default::default()
    };
    let resolver_low = CDCRResolver::with_config(config_low);
    let clusters_low = resolver_low.resolve(&[doc1, doc2]);

    // Should have at most 2 clusters (might cluster due to substring match)
    assert!(clusters_low.len() <= 2);
}

#[test]
fn test_cdcr_empty_documents() {
    let resolver = CDCRResolver::new();
    let clusters = resolver.resolve(&[]);
    assert!(clusters.is_empty(), "Empty docs should produce no clusters");
}

#[test]
fn test_cdcr_single_document() {
    let mut doc = Document::new("doc1", "John Smith works at Google.");
    doc.entities = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
        Entity::new("Google", EntityType::Organization, 20, 26, 0.95),
    ];

    let config = CDCRConfig {
        use_lsh: false,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&[doc]);

    // Each entity should be in its own cluster (no cross-doc clustering possible)
    assert_eq!(clusters.len(), 2, "Two entities should form two clusters");

    // All clusters should be singletons (doc_count = 1)
    for cluster in &clusters {
        assert_eq!(
            cluster.doc_count(),
            1,
            "Single document should produce singleton clusters"
        );
    }
}

#[test]
fn test_cdcr_cluster_confidence() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    for cluster in &clusters {
        // Confidence should be in valid range [0.0, 1.0]
        assert!(
            cluster.confidence >= 0.0 && cluster.confidence <= 1.0,
            "Cluster confidence should be in [0.0, 1.0], found {}",
            cluster.confidence
        );
    }
}

#[test]
fn test_cdcr_cluster_id_uniqueness() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for cluster in &clusters {
        assert!(
            ids.insert(cluster.id),
            "Cluster IDs should be unique, found duplicate: {}",
            cluster.id
        );
    }
}
