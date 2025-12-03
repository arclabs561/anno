//! Integration tests for cross-document coreference CLI
//!
//! Tests the full workflow: document loading, entity extraction, clustering, and output formatting.

use anno::eval::cdcr::{political_news_dataset, tech_news_dataset, CDCRConfig, CDCRResolver};
use std::collections::HashMap;

#[test]
fn test_full_workflow_tech_news() {
    let docs = tech_news_dataset();

    // Simulate CLI workflow
    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Verify results
    assert!(!clusters.is_empty());

    // Check Nvidia cluster (should be cross-doc)
    let nvidia = clusters.iter().find(|c| {
        c.canonical_name.to_lowercase() == "nvidia"
            && c.entity_type == Some(anno::EntityType::Organization)
    });

    assert!(nvidia.is_some(), "Should find Nvidia cluster");
    if let Some(n) = nvidia {
        assert!(n.doc_count() >= 2, "Nvidia should span multiple docs");
    }
}

#[test]
fn test_filtering_cross_doc_only() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let all_clusters = resolver.resolve(&docs);

    // Filter to only cross-doc clusters
    let cross_doc_only: Vec<_> = all_clusters.iter().filter(|c| c.doc_count() > 1).collect();

    assert!(cross_doc_only.len() <= all_clusters.len());
    assert!(cross_doc_only.iter().all(|c| c.doc_count() > 1));
}

#[test]
fn test_filtering_by_entity_type() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let all_clusters = resolver.resolve(&docs);

    // Filter to only Organization clusters
    let org_clusters: Vec<_> = all_clusters
        .iter()
        .filter(|c| c.entity_type == Some(anno::EntityType::Organization))
        .collect();

    assert!(org_clusters.len() <= all_clusters.len());
    assert!(org_clusters
        .iter()
        .all(|c| c.entity_type == Some(anno::EntityType::Organization)));
}

#[test]
fn test_min_cluster_size_filter() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let all_clusters = resolver.resolve(&docs);

    // Filter to clusters with at least 2 mentions
    let min_size = 2;
    let filtered: Vec<_> = all_clusters
        .iter()
        .filter(|c| c.len() >= min_size)
        .collect();

    assert!(filtered.len() <= all_clusters.len());
    assert!(filtered.iter().all(|c| c.len() >= min_size));
}

#[test]
fn test_sorting_by_importance() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let mut clusters = resolver.resolve(&docs);

    // Sort by importance: doc_count desc, then len desc, then name asc
    clusters.sort_by(|a, b| {
        b.doc_count()
            .cmp(&a.doc_count())
            .then_with(|| b.len().cmp(&a.len()))
            .then_with(|| a.canonical_name.cmp(&b.canonical_name))
    });

    // Verify sorting
    for i in 1..clusters.len() {
        let prev = &clusters[i - 1];
        let curr = &clusters[i];

        // Either prev has more docs, or same docs but more mentions, or same everything
        assert!(
            prev.doc_count() > curr.doc_count()
                || (prev.doc_count() == curr.doc_count() && prev.len() >= curr.len())
                || (prev.doc_count() == curr.doc_count()
                    && prev.len() == curr.len()
                    && prev.canonical_name <= curr.canonical_name),
            "Clusters should be sorted by importance"
        );
    }
}

#[test]
fn test_political_news_dataset() {
    let docs = political_news_dataset();

    assert!(
        docs.len() >= 4,
        "Political dataset should have at least 4 documents"
    );

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Should produce clusters
    assert!(!clusters.is_empty());

    // Should have cross-doc clusters (Biden, Scholz appear multiple times)
    let cross_doc_count = clusters.iter().filter(|c| c.doc_count() > 1).count();
    assert!(cross_doc_count > 0, "Should have cross-document clusters");
}

#[test]
fn test_mixed_datasets() {
    let mut docs = tech_news_dataset();
    docs.extend(political_news_dataset());

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Should produce clusters
    assert!(!clusters.is_empty());

    // Should have clusters from both domains
    let tech_entities: std::collections::HashSet<_> =
        ["Nvidia", "Jensen Huang", "Huang", "AMD", "Intel"]
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

    let pol_entities: std::collections::HashSet<_> =
        ["Biden", "Scholz", "NATO", "Washington", "Berlin"]
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

    let found_tech = clusters
        .iter()
        .any(|c| tech_entities.contains(&c.canonical_name.to_lowercase()));
    let found_pol = clusters
        .iter()
        .any(|c| pol_entities.contains(&c.canonical_name.to_lowercase()));

    assert!(found_tech, "Should find tech entities");
    assert!(found_pol, "Should find political entities");
}

#[test]
fn test_cluster_statistics() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Calculate statistics
    let total_entities: usize = docs.iter().map(|d| d.entities.len()).sum();
    let total_mentions: usize = clusters.iter().map(|c| c.len()).sum();
    let cross_doc_count = clusters.iter().filter(|c| c.doc_count() > 1).count();
    let singleton_count = clusters.len() - cross_doc_count;

    // All entities should be in clusters
    assert_eq!(
        total_mentions, total_entities,
        "All entities should be clustered"
    );

    // Statistics should be consistent
    assert!(cross_doc_count + singleton_count == clusters.len());
    assert!(cross_doc_count >= 0);
    // singleton_count is usize, so >= 0 is always true
    // Assertion removed as it's redundant for usize type

    // Average cluster size
    if !clusters.is_empty() {
        let avg_size = total_mentions as f64 / clusters.len() as f64;
        assert!(
            avg_size >= 1.0,
            "Average cluster size should be at least 1.0"
        );
    }
}

#[test]
fn test_document_path_mapping() {
    // Simulate document path mapping as done in CLI
    let mut doc_paths: HashMap<String, String> = HashMap::new();
    doc_paths.insert("tech_01".to_string(), "/data/tech_01.txt".to_string());
    doc_paths.insert("tech_02".to_string(), "/data/tech_02.txt".to_string());
    doc_paths.insert("tech_03".to_string(), "/data/tech_03.txt".to_string());

    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Verify path mapping works
    for cluster in &clusters {
        for doc_id in &cluster.documents {
            if let Some(path) = doc_paths.get(doc_id) {
                // Path should be accessible
                assert!(!path.is_empty(), "Path should not be empty");
            }
        }
    }
}

#[test]
fn test_output_format_completeness() {
    let docs = tech_news_dataset();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    // Verify all clusters have required fields for output
    for cluster in &clusters {
        assert!(
            !cluster.canonical_name.is_empty(),
            "Should have canonical name"
        );
        assert!(
            !cluster.mentions.is_empty() || cluster.is_empty(),
            "Should have mentions or be empty"
        );
        assert!(
            !cluster.documents.is_empty() || cluster.is_empty(),
            "Should have documents or be empty"
        );
        assert!(
            cluster.confidence >= 0.0 && cluster.confidence <= 1.0,
            "Confidence should be valid"
        );
    }
}

#[test]
fn test_edge_case_empty_documents() {
    let docs = vec![];

    let resolver = CDCRResolver::new();
    let clusters = resolver.resolve(&docs);

    assert!(
        clusters.is_empty(),
        "Empty documents should produce no clusters"
    );
}

#[test]
fn test_edge_case_single_entity() {
    let mut doc = anno::eval::cdcr::Document::new("doc1", "Apple announced new products.");
    doc.entities = vec![anno::Entity::new(
        "Apple",
        anno::EntityType::Organization,
        0,
        5,
        0.9,
    )];

    let resolver = CDCRResolver::new();
    let clusters = resolver.resolve(&[doc]);

    assert_eq!(clusters.len(), 1, "Single entity should form one cluster");
    assert_eq!(clusters[0].len(), 1, "Cluster should have one mention");
    assert_eq!(clusters[0].doc_count(), 1, "Should be singleton cluster");
}

#[test]
fn test_edge_case_duplicate_entities_same_doc() {
    let mut doc = anno::eval::cdcr::Document::new(
        "doc1",
        "Apple announced new products. Apple also revealed partnerships.",
    );
    doc.entities = vec![
        anno::Entity::new("Apple", anno::EntityType::Organization, 0, 5, 0.9),
        anno::Entity::new("Apple", anno::EntityType::Organization, 35, 40, 0.9),
    ];

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&[doc]);

    // Should cluster same entities in same document
    assert!(
        clusters.len() <= 1,
        "Same entities in same doc should cluster"
    );
    if !clusters.is_empty() {
        assert_eq!(clusters[0].len(), 2, "Should have 2 mentions");
        assert_eq!(clusters[0].doc_count(), 1, "Should be in one document");
    }
}

#[test]
fn test_performance_with_many_documents() {
    // Create many documents with overlapping entities
    let mut docs = Vec::new();
    for i in 0..20 {
        let mut doc = anno::eval::cdcr::Document::new(
            &format!("doc{}", i),
            &format!("Nvidia announced new products. Company {} is expanding.", i),
        );
        doc.entities = vec![anno::Entity::new(
            "Nvidia",
            anno::EntityType::Organization,
            0,
            6,
            0.9,
        )];
        docs.push(doc);
    }

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: true, // Use LSH for performance
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);

    let start = std::time::Instant::now();
    let clusters = resolver.resolve(&docs);
    let duration = start.elapsed();

    // Should complete quickly
    assert!(
        duration.as_secs_f64() < 1.0,
        "Should complete in reasonable time"
    );

    // Nvidia should be clustered across all documents
    let nvidia_cluster = clusters
        .iter()
        .find(|c| c.canonical_name.to_lowercase() == "nvidia");

    assert!(nvidia_cluster.is_some(), "Should find Nvidia cluster");
    if let Some(nc) = nvidia_cluster {
        assert!(nc.doc_count() >= 2, "Nvidia should span multiple docs");
    }
}
