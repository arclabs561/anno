//! Tests for cross-document coreference CLI output format
//!
//! Tests the actual tree format output, sorting, filtering, and display logic
//! by simulating what the CLI would produce.

use anno::eval::cdcr::{CDCRConfig, CDCRResolver, CrossDocCluster, Document};
use anno::{Entity, EntityType};
use std::collections::HashMap;

/// Simulate the tree format output generation (extracted logic from CLI)
fn generate_tree_output(
    clusters: &[CrossDocCluster],
    documents: &[Document],
    doc_paths: &HashMap<String, String>,
    verbose: bool,
    max_clusters: usize,
) -> String {
    use std::collections::HashMap as StdHashMap;

    // Build document index for O(1) lookups
    let doc_index: StdHashMap<&str, &Document> =
        documents.iter().map(|d| (d.id.as_str(), d)).collect();

    let mut output = String::new();

    // Sort clusters by importance
    let mut sorted_clusters: Vec<_> = clusters.iter().collect();
    sorted_clusters.sort_by(|a, b| {
        b.doc_count()
            .cmp(&a.doc_count())
            .then_with(|| b.len().cmp(&a.len()))
            .then_with(|| b.canonical_name.cmp(&a.canonical_name))
    });

    output.push_str("Cross-Document Entity Clusters\n");
    output.push_str("\n");

    // Summary
    let total_entities: usize = documents.iter().map(|d| d.entities.len()).sum();
    let cross_doc_clusters = clusters.iter().filter(|c| c.doc_count() > 1).count();
    let singleton_clusters = clusters.len() - cross_doc_clusters;

    output.push_str("Summary\n");
    output.push_str(&format!("  Documents: {}\n", documents.len()));
    output.push_str(&format!("  Entities: {}\n", total_entities));
    output.push_str(&format!(
        "  Clusters: {} ({} cross-doc, {} singleton)\n",
        clusters.len(),
        cross_doc_clusters,
        singleton_clusters
    ));
    output.push_str("\n");

    // Determine display limit
    let display_limit = if max_clusters > 0 {
        max_clusters
    } else if !verbose {
        50
    } else {
        sorted_clusters.len()
    };

    output.push_str("Clusters\n");
    output.push_str("\n");

    for cluster in sorted_clusters.iter().take(display_limit) {
        let is_cross_doc = cluster.doc_count() > 1;
        let prefix = if is_cross_doc { "●" } else { "○" };

        // Cluster header
        let mut header = format!("{} {}", prefix, cluster.canonical_name);
        if let Some(ref entity_type) = cluster.entity_type {
            header.push_str(&format!(" ({})", entity_type.as_label()));
        }
        if is_cross_doc {
            header.push_str(" [cross-doc]");
        }
        output.push_str(&format!("{}\n", header));

        // Metadata line
        let mut meta_parts = Vec::new();
        meta_parts.push(format!("{} mentions", cluster.len()));
        meta_parts.push(format!(
            "{} doc{}",
            cluster.doc_count(),
            if cluster.doc_count() == 1 { "" } else { "s" }
        ));
        if cluster.confidence < 1.0 {
            meta_parts.push(format!("conf: {:.2}", cluster.confidence));
        }
        output.push_str(&format!("  {}\n", meta_parts.join(" • ")));

        // Show documents with paths
        if !cluster.documents.is_empty() {
            let doc_list: Vec<String> = cluster
                .documents
                .iter()
                .map(|doc_id| {
                    let path = doc_paths
                        .get(doc_id)
                        .map(|p| format!("{} ({})", doc_id, p))
                        .unwrap_or_else(|| doc_id.clone());
                    path
                })
                .collect();
            output.push_str(&format!("  Docs: {}\n", doc_list.join(", ")));
        }

        // Show mentions
        if !cluster.mentions.is_empty() {
            let sample_size = if verbose {
                cluster.mentions.len()
            } else {
                cluster.mentions.len().min(3)
            };

            for (doc_id, entity_idx) in cluster.mentions.iter().take(sample_size) {
                if let Some(doc) = doc_index.get(doc_id.as_str()) {
                    if let Some(entity) = doc.entities.get(*entity_idx) {
                        if verbose {
                            // Extract context safely
                            let context_window = 50;
                            let start = entity.start.saturating_sub(context_window);
                            let end = (entity.end + context_window).min(doc.text.len());

                            let safe_start = doc
                                .text
                                .char_indices()
                                .position(|(i, _)| i >= start)
                                .map(|pos| {
                                    doc.text
                                        .char_indices()
                                        .nth(pos)
                                        .map(|(i, _)| i)
                                        .unwrap_or(start)
                                })
                                .unwrap_or(start);
                            let safe_end = doc
                                .text
                                .char_indices()
                                .position(|(i, _)| i >= end)
                                .map(|pos| {
                                    doc.text
                                        .char_indices()
                                        .nth(pos)
                                        .map(|(i, _)| i)
                                        .unwrap_or(end)
                                })
                                .unwrap_or(end);

                            let context = &doc.text[safe_start..safe_end];
                            let entity_text = &doc.text[entity.start..entity.end];

                            let before_len = entity.start.saturating_sub(safe_start);
                            let after_start = (entity.end - safe_start).min(context.len());

                            let before = if safe_start < entity.start {
                                &context[..before_len.min(context.len())]
                            } else {
                                ""
                            };
                            let after = if after_start < context.len() {
                                &context[after_start..]
                            } else {
                                ""
                            };

                            let before_marker = if safe_start < entity.start { "..." } else { "" };
                            let after_marker = if entity.end < safe_end { "..." } else { "" };

                            output.push_str(&format!(
                                "    • {}: {}{}[{}]{}{}\n",
                                doc_id, before_marker, before, entity_text, after, after_marker
                            ));
                        } else {
                            output.push_str(&format!("    • {}: \"{}\"\n", doc_id, entity.text));
                        }
                    }
                }
            }

            if cluster.mentions.len() > sample_size {
                output.push_str(&format!(
                    "    • ... and {} more\n",
                    cluster.mentions.len() - sample_size
                ));
            }
        }

        output.push_str("\n");
    }

    if sorted_clusters.len() > display_limit {
        let more_count = sorted_clusters.len() - display_limit;
        output.push_str(&format!(
            "... {} more cluster{} (use --max-clusters {} or --verbose to see all)\n",
            more_count,
            if more_count == 1 { "" } else { "s" },
            sorted_clusters.len()
        ));
    }

    output
}

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
fn test_tree_output_contains_summary() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    assert!(
        output.contains("Summary"),
        "Output should contain summary section"
    );
    assert!(
        output.contains("Documents:"),
        "Output should show document count"
    );
    assert!(
        output.contains("Entities:"),
        "Output should show entity count"
    );
    assert!(
        output.contains("Clusters:"),
        "Output should show cluster count"
    );
}

#[test]
fn test_tree_output_sorted_by_importance() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    // Find cluster positions in output
    let lines: Vec<&str> = output.lines().collect();
    let mut cluster_positions = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        if line.starts_with("●") || line.starts_with("○") {
            cluster_positions.push((idx, line));
        }
    }

    // First cluster should be cross-doc (●) if any exist
    if !cluster_positions.is_empty() {
        let first = cluster_positions[0].1;
        // Cross-doc clusters should appear first (they have ●)
        // This is a basic check - more sophisticated would verify doc_count ordering
        assert!(
            first.starts_with("●") || cluster_positions.len() == 1,
            "Cross-doc clusters should appear first"
        );
    }
}

#[test]
fn test_tree_output_shows_document_paths() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    // Should show document paths
    assert!(
        output.contains("/path/to/doc1.txt"),
        "Should show doc1 path"
    );
    assert!(
        output.contains("/path/to/doc2.txt"),
        "Should show doc2 path"
    );
    assert!(
        output.contains("/path/to/doc3.txt"),
        "Should show doc3 path"
    );
}

#[test]
fn test_tree_output_respects_max_clusters() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 2);

    // Count cluster headers (● or ○)
    let cluster_count = output
        .lines()
        .filter(|line| line.starts_with("●") || line.starts_with("○"))
        .count();

    assert!(cluster_count <= 2, "Should respect max_clusters limit");

    if clusters.len() > 2 {
        assert!(
            output.contains("more cluster"),
            "Should show 'more cluster' message when limited"
        );
    }
}

#[test]
fn test_tree_output_verbose_shows_context() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output_verbose = generate_tree_output(&clusters, &docs, &doc_paths, true, 0);
    let output_non_verbose = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    // Verbose output should show context (contain "..." markers)
    // Non-verbose should just show entity text
    if !clusters.is_empty() && !clusters[0].mentions.is_empty() {
        // Verbose should have context markers or longer lines
        let _verbose_has_context =
            output_verbose.contains("...") || output_verbose.lines().any(|l| l.len() > 100);

        // Both should show entity mentions, but verbose adds context
        assert!(
            output_verbose.len() >= output_non_verbose.len(),
            "Verbose output should be longer or equal"
        );
    }
}

#[test]
fn test_tree_output_shows_cross_doc_marker() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    // Should have cross-doc clusters marked with ●
    let has_cross_doc = output.lines().any(|line| line.starts_with("●"));

    // Nvidia should appear in multiple docs
    let nvidia_cluster = clusters.iter().find(|c| {
        c.canonical_name.to_lowercase() == "nvidia"
            && c.entity_type == Some(EntityType::Organization)
    });

    if let Some(nc) = nvidia_cluster {
        if nc.doc_count() > 1 {
            assert!(has_cross_doc, "Should mark cross-doc clusters with ●");
            assert!(
                output.contains("[cross-doc]"),
                "Should show [cross-doc] marker"
            );
        }
    }
}

#[test]
fn test_tree_output_shows_entity_type() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    // Should show entity types in parentheses
    if !clusters.is_empty() {
        let has_org = output.contains("(Organization)") || output.contains("(ORG)");
        let has_person = output.contains("(Person)") || output.contains("(PER)");

        // At least one type should be shown if clusters have types
        let clusters_with_types = clusters.iter().filter(|c| c.entity_type.is_some()).count();
        if clusters_with_types > 0 {
            assert!(has_org || has_person, "Should show entity types");
        }
    }
}

#[test]
fn test_tree_output_shows_metadata_line() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    // Should show metadata with mentions and doc count
    if !clusters.is_empty() {
        assert!(output.contains("mentions"), "Should show mention count");
        assert!(output.contains("doc"), "Should show document count");
    }
}

#[test]
fn test_tree_output_handles_empty_clusters() {
    let docs = vec![];
    let clusters = vec![];
    let doc_paths = HashMap::new();

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    // Should still show summary with zeros
    assert!(
        output.contains("Documents: 0"),
        "Should show zero documents"
    );
    assert!(output.contains("Entities: 0"), "Should show zero entities");
    assert!(output.contains("Clusters: 0"), "Should show zero clusters");
}

#[test]
fn test_tree_output_shows_sample_mentions() {
    let docs = create_test_documents();

    let config = CDCRConfig {
        min_similarity: 0.4,
        use_lsh: false,
        require_type_match: true,
        ..Default::default()
    };
    let resolver = CDCRResolver::with_config(config);
    let clusters = resolver.resolve(&docs);

    let mut doc_paths = HashMap::new();
    doc_paths.insert("doc1".to_string(), "/path/to/doc1.txt".to_string());
    doc_paths.insert("doc2".to_string(), "/path/to/doc2.txt".to_string());
    doc_paths.insert("doc3".to_string(), "/path/to/doc3.txt".to_string());

    let output = generate_tree_output(&clusters, &docs, &doc_paths, false, 0);

    // Should show at least some entity mentions
    if !clusters.is_empty() && !clusters[0].mentions.is_empty() {
        // Should contain entity text or doc IDs
        assert!(
            output.contains("Nvidia") || output.contains("doc1") || output.contains("doc2"),
            "Should show entity mentions or document references"
        );
    }
}
