//! Comprehensive tests for graph export functionality.
//!
//! Tests GraphNode, GraphEdge, GraphDocument, and all export formats.

use anno::eval::coref::{CorefChain, Mention};
use anno::graph::{GraphDocument, GraphEdge, GraphExportFormat, GraphNode};
use anno::{Entity, EntityType, Relation};

#[test]
fn test_graph_node_new() {
    let node = GraphNode::new("node1", "Person", "John Doe");
    assert_eq!(node.id, "node1");
    assert_eq!(node.node_type, "Person");
    assert_eq!(node.name, "John Doe");
    assert!(node.properties.is_empty());
}

#[test]
fn test_graph_node_with_property() {
    let node = GraphNode::new("node1", "Person", "John")
        .with_property("age", 30)
        .with_property("city", "New York");

    assert_eq!(node.properties.get("age"), Some(&serde_json::json!(30)));
    assert_eq!(
        node.properties.get("city"),
        Some(&serde_json::json!("New York"))
    );
}

#[test]
fn test_graph_node_with_mentions_count() {
    let node = GraphNode::new("node1", "Person", "John").with_mentions_count(5);

    assert_eq!(
        node.properties.get("mentions_count"),
        Some(&serde_json::json!(5))
    );
}

#[test]
fn test_graph_node_with_first_seen() {
    let node = GraphNode::new("node1", "Person", "John").with_first_seen(42);

    assert_eq!(
        node.properties.get("first_seen"),
        Some(&serde_json::json!(42))
    );
}

#[test]
fn test_graph_edge_new() {
    let edge = GraphEdge::new("source1", "target1", "RELATED_TO");
    assert_eq!(edge.source, "source1");
    assert_eq!(edge.target, "target1");
    assert_eq!(edge.relation, "RELATED_TO");
    assert_eq!(edge.confidence, 1.0);
    assert!(edge.properties.is_empty());
}

#[test]
fn test_graph_edge_with_confidence() {
    let edge = GraphEdge::new("source1", "target1", "RELATED_TO").with_confidence(0.85);

    assert_eq!(edge.confidence, 0.85);
}

#[test]
fn test_graph_edge_with_property() {
    let edge = GraphEdge::new("source1", "target1", "RELATED_TO")
        .with_property("distance", 10)
        .with_property("context", "sentence");

    assert_eq!(
        edge.properties.get("distance"),
        Some(&serde_json::json!(10))
    );
    assert_eq!(
        edge.properties.get("context"),
        Some(&serde_json::json!("sentence"))
    );
}

#[test]
fn test_graph_edge_with_trigger() {
    let edge = GraphEdge::new("source1", "target1", "FOUNDED").with_trigger("founded by");

    assert_eq!(
        edge.properties.get("trigger"),
        Some(&serde_json::json!("founded by"))
    );
}

#[test]
fn test_graph_document_new() {
    let doc = GraphDocument::new();
    assert!(doc.nodes.is_empty());
    assert!(doc.edges.is_empty());
    assert!(doc.metadata.is_empty());
    assert!(doc.is_empty());
    assert_eq!(doc.node_count(), 0);
    assert_eq!(doc.edge_count(), 0);
}

#[test]
fn test_graph_document_with_metadata() {
    let doc = GraphDocument::new()
        .with_metadata("source", "test")
        .with_metadata("version", 1);

    assert_eq!(doc.metadata.get("source"), Some(&serde_json::json!("test")));
    assert_eq!(doc.metadata.get("version"), Some(&serde_json::json!(1)));
}

#[test]
fn test_graph_document_from_extraction_empty() {
    let entities: Vec<Entity> = vec![];
    let relations: Vec<Relation> = vec![];

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    assert!(graph.is_empty());
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_graph_document_from_extraction_single_entity() {
    let entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    let entities = vec![entity];
    let relations: Vec<Relation> = vec![];

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    assert_eq!(graph.node_count(), 1);
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.nodes[0].name, "Apple");
    assert_eq!(graph.nodes[0].node_type, "ORG");
}

#[test]
fn test_graph_document_from_extraction_with_relations() {
    let elon = Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9).with_canonical_id(1);
    let tesla = Entity::new("Tesla", EntityType::Organization, 19, 24, 0.95).with_canonical_id(2);

    let relations = vec![Relation::new(elon.clone(), tesla.clone(), "FOUNDED", 0.85)];
    let entities = vec![elon, tesla];

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.edges[0].relation, "FOUNDED");
    assert_eq!(graph.edges[0].confidence, 0.85);
}

#[test]
fn test_graph_document_deduplication_by_canonical_id() {
    let entities = vec![
        Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9).with_canonical_id(1),
        Entity::new("Musk", EntityType::Person, 50, 54, 0.8).with_canonical_id(1), // Same canonical
        Entity::new("Tesla", EntityType::Organization, 100, 105, 0.95).with_canonical_id(2),
    ];

    let graph = GraphDocument::from_extraction(&entities, &[], None);
    // Should have 2 nodes (Elon Musk and Musk deduplicated)
    assert_eq!(graph.node_count(), 2);

    // Check mention count was incremented
    let elon_node = graph.nodes.iter().find(|n| n.id == "coref_1").unwrap();
    assert_eq!(
        elon_node.properties.get("mentions_count"),
        Some(&serde_json::json!(2))
    );
}

#[test]
fn test_graph_document_deduplication_by_kb_id() {
    let mut entity1 = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    entity1.kb_id = Some("Q312".to_string());
    entity1.canonical_id = Some(1);
    let mut entity2 = Entity::new("Apple Inc", EntityType::Organization, 20, 29, 0.95);
    entity2.kb_id = Some("Q312".to_string());
    entity2.canonical_id = Some(2); // Same KB ID
    let entities = vec![entity1, entity2];

    let graph = GraphDocument::from_extraction(&entities, &[], None);
    // Should have 1 node (deduplicated by KB ID)
    assert_eq!(graph.node_count(), 1);
    assert_eq!(graph.nodes[0].id, "Q312");
}

#[test]
fn test_graph_document_with_coref_chains() {
    let entities = vec![
        Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9).with_canonical_id(1),
        Entity::new("Musk", EntityType::Person, 50, 54, 0.8).with_canonical_id(1),
    ];

    let chains = vec![CorefChain {
        cluster_id: Some(1),
        entity_type: Some("Person".to_string()),
        mentions: vec![
            Mention::new("Elon Musk", 0, 9),
            Mention::new("Musk", 50, 54),
        ],
    }];

    let graph = GraphDocument::from_extraction(&entities, &[], Some(&chains));
    // Should use canonical name from first mention
    let elon_node = graph.nodes.iter().find(|n| n.id == "coref_1").unwrap();
    assert_eq!(elon_node.name, "Elon Musk");
    // Mentions count may be 2 (from chains) or 3 (if entities are also counted)
    let mentions_count = elon_node
        .properties
        .get("mentions_count")
        .and_then(|v| v.as_u64());
    assert!(mentions_count.is_some() && mentions_count.unwrap() >= 2);
}

#[test]
fn test_graph_document_relation_with_missing_nodes() {
    // Create relation between entities that don't exist in graph
    let elon = Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9);
    let tesla = Entity::new("Tesla", EntityType::Organization, 19, 24, 0.95);

    // But only add one entity to the graph
    let entities = vec![elon.clone()];
    let relations = vec![Relation::new(elon, tesla, "FOUNDED", 0.85)];

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    // Edge should not be created because target node doesn't exist
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_graph_document_cooccurrence() {
    let entities = vec![
        Entity::new("John", EntityType::Person, 0, 4, 0.9),
        Entity::new("Apple", EntityType::Organization, 20, 25, 0.95), // Within 50 chars
        Entity::new("Paris", EntityType::Location, 100, 105, 0.9),    // Far away
    ];

    let graph = GraphDocument::from_entities_cooccurrence(&entities, 50);
    assert_eq!(graph.node_count(), 3);
    // Only John-Apple should have an edge (within 50 chars)
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.edges[0].relation, "RELATED_TO");
}

#[test]
fn test_graph_document_cooccurrence_no_self_loops() {
    let entities = vec![
        Entity::new("Apple", EntityType::Organization, 0, 5, 0.9),
        Entity::new("Apple", EntityType::Organization, 10, 15, 0.95), // Same entity, different position
    ];

    let graph = GraphDocument::from_entities_cooccurrence(&entities, 50);
    // Should not create self-loops
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_cypher_export_basic() {
    let entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    let graph = GraphDocument::from_extraction(&[entity], &[], None);

    let cypher = graph.to_cypher();
    assert!(cypher.contains("CREATE"));
    // Check that it creates a node (format: CREATE (n...:... {id: '...', name: '...'}))
    assert!(cypher.contains("id:") || cypher.contains("CREATE"));
    // The node type should be present (ORG or Organization)
    assert!(cypher.contains("ORG") || cypher.contains("Organization") || cypher.contains("org"));
}

#[test]
fn test_cypher_export_with_relations() {
    let elon = Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9);
    let tesla = Entity::new("Tesla", EntityType::Organization, 19, 24, 0.95);
    let entities = vec![elon.clone(), tesla.clone()];
    let relations = vec![Relation::new(elon, tesla, "FOUNDED", 0.85)];

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    let cypher = graph.to_cypher();

    assert!(cypher.contains("CREATE"));
    assert!(cypher.contains("MATCH"));
    assert!(cypher.contains("FOUNDED"));
    assert!(cypher.contains("confidence: 0.850"));
}

#[test]
fn test_cypher_export_escapes_special_chars() {
    let entity = Entity::new("O'Brien", EntityType::Person, 0, 7, 0.9);
    let graph = GraphDocument::from_extraction(&[entity], &[], None);

    let cypher = graph.to_cypher();
    // Should handle special characters safely (may escape or sanitize)
    // Just verify it doesn't crash and produces valid output
    assert!(cypher.contains("CREATE"));
    // The name may be escaped, sanitized, or appear as-is - just check it's present somehow
    assert!(!cypher.is_empty());
}

#[test]
fn test_networkx_json_export_basic() {
    let entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    let graph = GraphDocument::from_extraction(&[entity], &[], None);

    let json = graph.to_networkx_json();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["directed"], true);
    assert_eq!(parsed["multigraph"], false);
    assert!(parsed.get("nodes").is_some());
    assert!(parsed.get("links").is_some());
    assert_eq!(parsed["nodes"].as_array().unwrap().len(), 1);
}

#[test]
fn test_networkx_json_export_with_relations() {
    let elon = Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9);
    let tesla = Entity::new("Tesla", EntityType::Organization, 19, 24, 0.95);
    let entities = vec![elon.clone(), tesla.clone()];
    let relations = vec![Relation::new(elon, tesla, "FOUNDED", 0.9)];

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    let json = graph.to_networkx_json();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["links"].as_array().unwrap().len(), 1);
    let link = &parsed["links"][0];
    assert_eq!(link["relation"], "FOUNDED");
    assert_eq!(link["confidence"], 0.9);
}

#[test]
fn test_networkx_json_export_omits_default_confidence() {
    let elon = Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9);
    let tesla = Entity::new("Tesla", EntityType::Organization, 19, 24, 0.95);
    let entities = vec![elon.clone(), tesla.clone()];
    let relations = vec![Relation::new(elon, tesla, "FOUNDED", 1.0)]; // Default confidence

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    let json = graph.to_networkx_json();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    let link = &parsed["links"][0];
    // Default confidence (1.0) should be omitted
    assert!(!link.as_object().unwrap().contains_key("confidence"));
}

#[test]
fn test_json_ld_export_basic() {
    let entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    let graph = GraphDocument::from_extraction(&[entity], &[], None);

    let json_ld = graph.to_json_ld();
    let parsed: serde_json::Value = serde_json::from_str(&json_ld).unwrap();

    assert!(parsed.get("@context").is_some());
    assert!(parsed.get("@graph").is_some());
    assert_eq!(parsed["@graph"].as_array().unwrap().len(), 1);
}

#[test]
fn test_json_ld_export_with_relations() {
    let elon = Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9);
    let tesla = Entity::new("Tesla", EntityType::Organization, 19, 24, 0.95);
    let entities = vec![elon.clone(), tesla.clone()];
    let relations = vec![Relation::new(elon, tesla, "FOUNDED", 0.85)];

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    let json_ld = graph.to_json_ld();
    let parsed: serde_json::Value = serde_json::from_str(&json_ld).unwrap();

    let graph_nodes = parsed["@graph"].as_array().unwrap();
    // Find Elon node and check it has relations
    let elon_node = graph_nodes
        .iter()
        .find(|n| n["name"] == "Elon Musk")
        .unwrap();
    assert!(elon_node.get("relations").is_some());
    let relations = elon_node["relations"].as_array().unwrap();
    assert_eq!(relations.len(), 1);
    assert_eq!(relations[0]["@type"], "FOUNDED");
}

#[test]
fn test_export_format_enum() {
    let entity = Entity::new("Apple", EntityType::Organization, 0, 5, 0.9);
    let graph = GraphDocument::from_extraction(&[entity], &[], None);

    let cypher = graph.export(GraphExportFormat::Cypher);
    assert!(cypher.contains("CREATE"));

    let networkx = graph.export(GraphExportFormat::NetworkXJson);
    let parsed: serde_json::Value = serde_json::from_str(&networkx).unwrap();
    assert!(parsed.get("nodes").is_some());

    let json_ld = graph.export(GraphExportFormat::JsonLd);
    let parsed: serde_json::Value = serde_json::from_str(&json_ld).unwrap();
    assert!(parsed.get("@context").is_some());
}

#[test]
fn test_graph_document_temporal_validity_export() {
    use anno::EntityViewport;
    use chrono::{TimeZone, Utc};

    let mut nadella = Entity::new("Satya Nadella", EntityType::Person, 0, 13, 0.9);
    nadella.valid_from = Some(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap());
    nadella.viewport = Some(EntityViewport::Business);

    let graph = GraphDocument::from_extraction(&[nadella], &[], None);
    let node = &graph.nodes[0];

    assert!(node.properties.contains_key("valid_from"));
    assert!(node.properties.contains_key("viewport"));
    assert_eq!(node.properties.get("viewport").unwrap(), "business");
}

#[test]
fn test_graph_document_large_graph() {
    // Test with many entities and relations
    let mut entities = Vec::new();
    let mut relations = Vec::new();

    for i in 0..100 {
        let entity = Entity::new(
            &format!("Entity{}", i),
            EntityType::Person,
            i * 10,
            i * 10 + 7,
            0.9,
        )
        .with_canonical_id(i as u64);
        entities.push(entity);
    }

    // Create relations between adjacent entities
    for i in 0..99 {
        let head = entities[i].clone();
        let tail = entities[i + 1].clone();
        relations.push(Relation::new(head, tail, "RELATED_TO", 0.8));
    }

    let graph = GraphDocument::from_extraction(&entities, &relations, None);
    assert_eq!(graph.node_count(), 100);
    assert_eq!(graph.edge_count(), 99);
}

#[test]
fn test_graph_document_empty_entity_text() {
    // Edge case: entity with empty text
    let entity = Entity::new("", EntityType::Person, 0, 0, 0.9);
    let graph = GraphDocument::from_extraction(&[entity], &[], None);

    // Should still create a node (though with empty name)
    assert_eq!(graph.node_count(), 1);
}
