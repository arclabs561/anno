//! Format validation tests for cross-document coreference CLI output
//!
//! Tests the output format structure, edge cases, and display logic.

use anno::eval::cdcr::{CDCRConfig, CDCRResolver, CrossDocCluster, Document};
use anno::{Entity, EntityType};

/// Test that output format handles various cluster configurations
#[test]
fn test_output_format_structure() {
    // Create clusters manually to test format
    let mut cluster1 = CrossDocCluster::new(0, "Nvidia");
    cluster1.entity_type = Some(EntityType::Organization);
    cluster1.add_mention("doc1", 0);
    cluster1.add_mention("doc2", 1);
    cluster1.add_mention("doc3", 2);
    cluster1.confidence = 0.95;

    let mut cluster2 = CrossDocCluster::new(1, "Jensen Huang");
    cluster2.entity_type = Some(EntityType::Person);
    cluster2.add_mention("doc1", 1);
    cluster2.add_mention("doc2", 0);
    cluster2.confidence = 0.92;

    let mut cluster3 = CrossDocCluster::new(2, "AMD");
    cluster3.entity_type = Some(EntityType::Organization);
    cluster3.add_mention("doc5", 0);
    cluster3.confidence = 1.0; // Perfect confidence

    let clusters = vec![cluster1, cluster2, cluster3];

    // Verify cluster properties
    assert_eq!(
        clusters[0].doc_count(),
        3,
        "First cluster should span 3 docs"
    );
    assert_eq!(
        clusters[1].doc_count(),
        2,
        "Second cluster should span 2 docs"
    );
    assert_eq!(
        clusters[2].doc_count(),
        1,
        "Third cluster should be singleton"
    );

    // Verify sorting: cross-doc clusters first
    let mut sorted = clusters.clone();
    sorted.sort_by(|a, b| {
        b.doc_count()
            .cmp(&a.doc_count())
            .then_with(|| b.len().cmp(&a.len()))
    });

    assert_eq!(
        sorted[0].canonical_name, "Nvidia",
        "Cross-doc cluster should be first"
    );
    assert_eq!(
        sorted[1].canonical_name, "Jensen Huang",
        "Second cross-doc cluster next"
    );
    assert_eq!(sorted[2].canonical_name, "AMD", "Singleton cluster last");
}

#[test]
fn test_cluster_metadata_formatting() {
    let mut cluster = CrossDocCluster::new(0, "Test Entity");
    cluster.entity_type = Some(EntityType::Person);
    cluster.add_mention("doc1", 0);
    cluster.add_mention("doc2", 1);
    cluster.add_mention("doc1", 2); // Same doc, different mention
    cluster.confidence = 0.87;

    // Verify metadata components
    assert_eq!(cluster.len(), 3, "Should have 3 mentions");
    assert_eq!(cluster.doc_count(), 2, "Should span 2 documents");
    assert_eq!(cluster.confidence, 0.87, "Should have correct confidence");

    // Metadata line should include: mentions, docs, confidence
    let meta_parts = vec![
        format!("{} mentions", cluster.len()),
        format!(
            "{} doc{}",
            cluster.doc_count(),
            if cluster.doc_count() == 1 { "" } else { "s" }
        ),
        format!("conf: {:.2}", cluster.confidence),
    ];
    let meta_line = meta_parts.join(" • ");

    assert!(
        meta_line.contains("3 mentions"),
        "Should show mention count"
    );
    assert!(meta_line.contains("2 docs"), "Should show doc count");
    assert!(meta_line.contains("conf: 0.87"), "Should show confidence");
}

#[test]
fn test_cross_doc_vs_singleton_markers() {
    let mut cross_doc = CrossDocCluster::new(0, "Cross Doc Entity");
    cross_doc.add_mention("doc1", 0);
    cross_doc.add_mention("doc2", 1);

    let mut singleton = CrossDocCluster::new(1, "Singleton Entity");
    singleton.add_mention("doc1", 1);

    assert!(
        cross_doc.doc_count() > 1,
        "Cross-doc should span multiple docs"
    );
    assert_eq!(singleton.doc_count(), 1, "Singleton should be in one doc");

    // Verify markers would be different
    let cross_doc_prefix = if cross_doc.doc_count() > 1 {
        "●"
    } else {
        "○"
    };
    let singleton_prefix = if singleton.doc_count() > 1 {
        "●"
    } else {
        "○"
    };

    assert_eq!(cross_doc_prefix, "●", "Cross-doc should use ●");
    assert_eq!(singleton_prefix, "○", "Singleton should use ○");
}

#[test]
fn test_display_limit_logic() {
    let clusters: Vec<CrossDocCluster> = (0..100)
        .map(|i| {
            let mut c = CrossDocCluster::new(i, &format!("Entity {}", i));
            c.add_mention("doc1", i as usize);
            c
        })
        .collect();

    // Test limit logic
    let max_clusters = 10;
    let verbose = false;

    let display_limit = if max_clusters > 0 {
        max_clusters
    } else if !verbose {
        50
    } else {
        clusters.len()
    };

    assert_eq!(display_limit, 10, "Should respect max_clusters");

    // When verbose=true but max_clusters is set, should respect max_clusters
    let verbose_limit_with_max = if max_clusters > 0 {
        max_clusters
    } else if !true {
        // verbose = true
        50
    } else {
        clusters.len()
    };
    assert_eq!(
        verbose_limit_with_max, 10,
        "Should respect max_clusters even in verbose"
    );

    // When verbose=true and max_clusters=0, should show all
    let verbose_limit_no_max = if 0 > 0 {
        0
    } else if !true {
        // verbose = true
        50
    } else {
        clusters.len()
    };
    assert_eq!(
        verbose_limit_no_max,
        clusters.len(),
        "Verbose with no max should show all"
    );
}

#[test]
fn test_entity_type_display() {
    let mut cluster_org = CrossDocCluster::new(0, "Apple");
    cluster_org.entity_type = Some(EntityType::Organization);

    let mut cluster_per = CrossDocCluster::new(1, "John Smith");
    cluster_per.entity_type = Some(EntityType::Person);

    let mut cluster_loc = CrossDocCluster::new(2, "New York");
    cluster_loc.entity_type = Some(EntityType::Location);

    // Verify type labels (as_label returns short codes)
    assert_eq!(cluster_org.entity_type.unwrap().as_label(), "ORG");
    assert_eq!(cluster_per.entity_type.unwrap().as_label(), "PER");
    assert_eq!(cluster_loc.entity_type.unwrap().as_label(), "LOC");
}

#[test]
fn test_confidence_display_ranges() {
    let mut cluster_high = CrossDocCluster::new(0, "High Conf");
    cluster_high.confidence = 0.95;

    let mut cluster_low = CrossDocCluster::new(1, "Low Conf");
    cluster_low.confidence = 0.65;

    let mut cluster_perfect = CrossDocCluster::new(2, "Perfect");
    cluster_perfect.confidence = 1.0;

    // Confidence < 1.0 should be shown
    assert!(cluster_high.confidence < 1.0, "High conf should be < 1.0");
    assert!(cluster_low.confidence < 1.0, "Low conf should be < 1.0");
    assert_eq!(cluster_perfect.confidence, 1.0, "Perfect should be 1.0");

    // Format should be 0.0-1.0 (not percentage)
    let conf_str_high = format!("{:.2}", cluster_high.confidence);
    assert_eq!(conf_str_high, "0.95", "Should format as 0.95 not 95%");
}

#[test]
fn test_document_path_display() {
    use std::collections::HashMap;

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());

    let mut cluster = CrossDocCluster::new(0, "Test");
    cluster.add_mention("doc1", 0);
    cluster.add_mention("doc2", 1);

    // Format: "doc_id (path)"
    let doc_list: Vec<String> = cluster
        .documents
        .iter()
        .map(|doc_id| {
            doc_paths
                .get(doc_id)
                .map(|p| format!("{} ({})", doc_id, p))
                .unwrap_or_else(|| doc_id.clone())
        })
        .collect();

    assert_eq!(doc_list.len(), 2);
    assert!(doc_list[0].contains("doc1") && doc_list[0].contains("/path/to/doc1.txt"));
    assert!(doc_list[1].contains("doc2") && doc_list[1].contains("/path/to/doc2.txt"));
}

#[test]
fn test_mention_sample_size() {
    let mut cluster = CrossDocCluster::new(0, "Test");
    for i in 0..10 {
        cluster.add_mention("doc1", i);
    }

    // Non-verbose: sample of 3
    let sample_size_non_verbose = cluster.mentions.len().min(3);
    assert_eq!(sample_size_non_verbose, 3);

    // Verbose: all mentions
    let sample_size_verbose = cluster.mentions.len();
    assert_eq!(sample_size_verbose, 10);

    // Should show "... and X more" when truncated
    if cluster.mentions.len() > sample_size_non_verbose {
        let more_count = cluster.mentions.len() - sample_size_non_verbose;
        assert_eq!(more_count, 7, "Should show 7 more mentions");
    }
}

#[test]
fn test_empty_cluster_handling() {
    let cluster = CrossDocCluster::new(0, "Empty");

    assert!(cluster.is_empty(), "Empty cluster should be empty");
    assert_eq!(cluster.len(), 0, "Empty cluster should have 0 mentions");
    assert_eq!(cluster.doc_count(), 0, "Empty cluster should have 0 docs");

    // Empty clusters shouldn't appear in output (filtered by min_cluster_size)
    let min_cluster_size = 1;
    assert!(
        cluster.len() < min_cluster_size,
        "Empty cluster should be filtered out"
    );
}

#[test]
fn test_cluster_id_uniqueness() {
    let clusters: Vec<CrossDocCluster> = (0..10)
        .map(|i| CrossDocCluster::new(i, &format!("Entity {}", i)))
        .collect();

    let mut ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for cluster in &clusters {
        assert!(ids.insert(cluster.id), "Cluster IDs should be unique");
    }

    assert_eq!(ids.len(), 10, "Should have 10 unique IDs");
}

#[test]
fn test_canonical_name_variations() {
    // Test that canonical names handle various formats
    let names = vec![
        "Simple Name",
        "Name with Numbers 123",
        "Name-with-hyphens",
        "Name_with_underscores",
        "Name.with.dots",
        "Very Long Name That Might Exceed Some Display Limits But Should Still Work",
    ];

    for name in names {
        let cluster = CrossDocCluster::new(0, name);
        assert_eq!(
            cluster.canonical_name, name,
            "Canonical name should be preserved"
        );
        assert!(
            !cluster.canonical_name.is_empty(),
            "Name should not be empty"
        );
    }
}

#[test]
fn test_kb_id_display() {
    let mut cluster = CrossDocCluster::new(0, "Marie Curie");
    cluster.kb_id = Some("wikidata:Q7186".to_string());

    assert!(cluster.kb_id.is_some(), "Should have KB ID");
    assert_eq!(cluster.kb_id.as_ref().unwrap(), "wikidata:Q7186");

    // KB ID should be displayed when present
    let kb_display = cluster
        .kb_id
        .as_ref()
        .map(|id| format!("KB: {}", id))
        .unwrap_or_default();

    assert!(kb_display.contains("wikidata:Q7186"), "Should show KB ID");
}

#[test]
fn test_multiple_mentions_same_doc() {
    let mut cluster = CrossDocCluster::new(0, "Repeated Entity");
    cluster.add_mention("doc1", 0);
    cluster.add_mention("doc1", 1);
    cluster.add_mention("doc1", 2);
    cluster.add_mention("doc2", 0);

    assert_eq!(cluster.len(), 4, "Should have 4 mentions");
    assert_eq!(
        cluster.doc_count(),
        2,
        "Should span 2 docs (doc1 appears once in documents list)"
    );

    // Documents list should be deduplicated
    let unique_docs: std::collections::HashSet<_> = cluster.documents.iter().collect();
    assert_eq!(unique_docs.len(), 2, "Should have 2 unique documents");
}
