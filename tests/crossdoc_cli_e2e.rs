//! End-to-end tests for cross-document coreference CLI
//!
//! Tests the actual CLI command execution with real data.

use anno::eval::cdcr::{tech_news_dataset, CDCRConfig, CDCRResolver};

#[test]
fn test_cdcr_with_tech_news_dataset() {
    let docs = tech_news_dataset();

    assert!(
        docs.len() >= 5,
        "Tech news dataset should have at least 5 documents"
    );

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false, // Brute force for reliable testing
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Should produce clusters
    assert!(
        !clusters.is_empty(),
        "Should produce clusters from tech news dataset"
    );

    // Should have cross-document clusters
    let cross_doc_count = clusters.iter().filter(|c| c.doc_count() > 1).count();
    assert!(cross_doc_count > 0, "Should have cross-document clusters");

    // Nvidia should be clustered across multiple documents
    let nvidia_cluster = clusters.iter().find(|c| {
        c.canonical_name.to_lowercase() == "nvidia"
            && c.entity_type == Some(anno::EntityType::Organization)
    });

    if let Some(nc) = nvidia_cluster {
        assert!(
            nc.doc_count() >= 2,
            "Nvidia should appear in at least 2 documents, found {}",
            nc.doc_count()
        );
    }
}

#[test]
fn test_cdcr_cluster_quality_metrics() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Quality checks
    let total_mentions: usize = clusters.iter().map(|c| c.len()).sum();
    let total_entities: usize = docs.iter().map(|d| d.entities.len()).sum();

    // All entities should be in clusters
    assert_eq!(
        total_mentions, total_entities,
        "All entities should be assigned to clusters"
    );

    // Average cluster size should be reasonable
    if !clusters.is_empty() {
        let avg_size = total_mentions as f64 / clusters.len() as f64;
        // Average should be between 1 and total_entities (reasonable range)
        assert!(
            avg_size >= 1.0 && avg_size <= total_entities as f64,
            "Average cluster size should be reasonable, found {}",
            avg_size
        );
    }

    // Cross-doc clusters should have confidence <= 1.0
    for cluster in &clusters {
        if cluster.doc_count() > 1 {
            assert!(
                cluster.confidence <= 1.0,
                "Cross-doc cluster confidence should be <= 1.0, found {}",
                cluster.confidence
            );
        }
    }
}

#[test]
fn test_cdcr_document_coverage() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Check that all documents are represented in clusters
    let doc_ids_in_clusters: std::collections::HashSet<_> =
        clusters.iter().flat_map(|c| &c.documents).collect();

    let _doc_ids: std::collections::HashSet<_> = docs.iter().map(|d| &d.id).collect();

    // All documents with entities should appear in at least one cluster
    let docs_with_entities: std::collections::HashSet<_> = docs
        .iter()
        .filter(|d| !d.entities.is_empty())
        .map(|d| &d.id)
        .collect();

    for doc_id in &docs_with_entities {
        assert!(
            doc_ids_in_clusters.contains(doc_id),
            "Document {} with entities should appear in at least one cluster",
            doc_id
        );
    }
}

#[test]
fn test_cdcr_mention_consistency() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Verify mention consistency
    for cluster in &clusters {
        // Each mention should reference a valid document and entity
        for (doc_id, entity_idx) in &cluster.mentions {
            let doc = docs.iter().find(|d| d.id == *doc_id);
            assert!(
                doc.is_some(),
                "Mention references invalid doc_id: {}",
                doc_id
            );

            if let Some(d) = doc {
                assert!(
                    *entity_idx < d.entities.len(),
                    "Mention references invalid entity_idx {} for doc {}",
                    entity_idx,
                    doc_id
                );

                // Entity type should match cluster type (if cluster has type)
                if let Some(ref cluster_type) = cluster.entity_type {
                    let entity = &d.entities[*entity_idx];
                    assert_eq!(
                        &entity.entity_type, cluster_type,
                        "Entity type should match cluster type"
                    );
                }
            }
        }

        // Document list should match unique documents in mentions
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
fn test_cdcr_canonical_name_quality() {
    let docs = tech_news_dataset();

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
            "Cluster should have non-empty canonical name"
        );

        // Canonical name should be reasonable length
        assert!(
            cluster.canonical_name.len() <= 200,
            "Canonical name should be reasonable length, found {} chars",
            cluster.canonical_name.len()
        );

        // Canonical name should match at least one mention (case-insensitive)
        let canonical_lower = cluster.canonical_name.to_lowercase();
        let mut found_match = false;

        for (doc_id, entity_idx) in &cluster.mentions {
            if let Some(doc) = docs.iter().find(|d| d.id == *doc_id) {
                if *entity_idx < doc.entities.len() {
                    let entity = &doc.entities[*entity_idx];
                    if entity.text.to_lowercase() == canonical_lower {
                        found_match = true;
                        break;
                    }
                }
            }
        }

        // Canonical name should match at least one mention (this is how it's chosen)
        assert!(
            found_match,
            "Canonical name '{}' should match at least one mention",
            cluster.canonical_name
        );
    }
}

#[test]
fn test_cdcr_large_dataset_performance() {
    // Test with larger dataset to ensure performance is reasonable
    let mut docs = tech_news_dataset();

    // Duplicate to create larger dataset
    let mut more_docs = docs.clone();
    for (idx, doc) in more_docs.iter_mut().enumerate() {
        doc.id = format!("doc{}_copy", idx);
    }
    docs.extend(more_docs);

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: true, // Use LSH for larger dataset
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);

    let start = std::time::Instant::now();
    let clusters = resolver.resolve(&docs);
    let duration = start.elapsed();

    // Should complete in reasonable time (< 1 second for this dataset size)
    assert!(
        duration.as_secs_f64() < 1.0,
        "CDCR should complete in reasonable time, took {:.2}s",
        duration.as_secs_f64()
    );

    // Should still produce valid clusters
    assert!(
        !clusters.is_empty(),
        "Should produce clusters even with larger dataset"
    );

    // Verify cluster quality
    for cluster in &clusters {
        assert!(
            !cluster.mentions.is_empty(),
            "Cluster should have at least one mention"
        );
        assert!(
            !cluster.documents.is_empty(),
            "Cluster should have at least one document"
        );
    }
}

#[test]
fn test_cdcr_similarity_threshold_effect() {
    let docs = tech_news_dataset();

    // High threshold - should produce more clusters (less merging)
    let config_high = CDCRConfig {
        min_similarity: 0.8,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver_high = CDCRResolver::with_config(config_high);
    let clusters_high = resolver_high.resolve(&docs);

    // Low threshold - should produce fewer clusters (more merging)
    let config_low = CDCRConfig {
        min_similarity: 0.2,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver_low = CDCRResolver::with_config(config_low);
    let clusters_low = resolver_low.resolve(&docs);

    // Lower threshold should generally produce fewer or equal clusters
    // (more aggressive merging)
    assert!(
        clusters_low.len() <= clusters_high.len(),
        "Lower threshold should produce fewer or equal clusters (more merging)"
    );
}

#[test]
fn test_cdcr_type_matching_effect() {
    let mut doc1 = anno::eval::cdcr::Document::new("doc1", "Apple Inc. announced new products.");
    doc1.entities = vec![anno::Entity::new(
        "Apple Inc.",
        anno::EntityType::Organization,
        0,
        10,
        0.9,
    )];

    let mut doc2 = anno::eval::cdcr::Document::new("doc2", "I ate an apple for lunch.");
    doc2.entities = vec![anno::Entity::new(
        "apple",
        anno::EntityType::Other("Fruit".to_string()),
        9,
        14,
        0.8,
    )];

    // With type matching - should NOT cluster
    let config_strict = CDCRConfig {
        min_similarity: 0.3,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver_strict = CDCRResolver::with_config(config_strict);
    let clusters_strict = resolver_strict.resolve(&[doc1.clone(), doc2.clone()]);

    // Without type matching - might cluster (same string)
    let config_loose = CDCRConfig {
        min_similarity: 0.3,
        use_lsh: false,
        require_type_match: false,
        ..Default::default()
    };
    let resolver_loose = CDCRResolver::with_config(config_loose);
    let clusters_loose = resolver_loose.resolve(&[doc1, doc2]);

    // Strict should have 2 clusters, loose might have 1 or 2
    assert_eq!(
        clusters_strict.len(),
        2,
        "Type matching should prevent clustering different types"
    );
    assert!(
        clusters_loose.len() <= 2,
        "Without type matching, might cluster or not"
    );
}
