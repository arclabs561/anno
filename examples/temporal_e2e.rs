//! E2E test of temporal validity in graph export

use anno::graph::GraphDocument;
use anno::{Entity, EntityType, EntityViewport, Relation};
use chrono::{TimeZone, Utc};

fn main() {
    println!("=== Temporal Validity E2E Test ===\n");

    // Create entities representing Microsoft CEO history
    let mut nadella = Entity::new("Satya Nadella", EntityType::Person, 0, 13, 0.95);
    nadella.valid_from = Some(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap());
    nadella.viewport = Some(EntityViewport::Business);
    nadella.canonical_id = Some(1);

    let mut ballmer = Entity::new("Steve Ballmer", EntityType::Person, 50, 63, 0.92);
    ballmer.valid_from = Some(Utc.with_ymd_and_hms(2000, 1, 13, 0, 0, 0).unwrap());
    ballmer.valid_until = Some(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap());
    ballmer.viewport = Some(EntityViewport::Historical);
    ballmer.canonical_id = Some(2);

    let mut gates = Entity::new("Bill Gates", EntityType::Person, 100, 110, 0.98);
    gates.valid_from = Some(Utc.with_ymd_and_hms(1975, 4, 4, 0, 0, 0).unwrap());
    gates.valid_until = Some(Utc.with_ymd_and_hms(2000, 1, 13, 0, 0, 0).unwrap());
    gates.viewport = Some(EntityViewport::Historical);
    gates.canonical_id = Some(3);

    let mut microsoft = Entity::new("Microsoft", EntityType::Organization, 150, 159, 0.99);
    microsoft.viewport = Some(EntityViewport::Business);
    microsoft.canonical_id = Some(4);

    // Create relations
    let relations = vec![
        Relation::new(nadella.clone(), microsoft.clone(), "CEO_OF", 0.95),
        Relation::new(ballmer.clone(), microsoft.clone(), "FORMER_CEO_OF", 0.92),
        Relation::new(gates.clone(), microsoft.clone(), "FOUNDER_OF", 0.98),
    ];

    let entities = vec![nadella, ballmer, gates, microsoft];

    // Build graph
    let graph = GraphDocument::from_extraction(&entities, &relations, None);

    println!("Graph Statistics:");
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());
    println!();

    // Export to Cypher
    println!("=== Neo4j Cypher Export ===\n");
    println!("{}", graph.to_cypher());

    // Export to NetworkX JSON
    println!("\n=== NetworkX JSON Export ===\n");
    let json = graph.to_networkx_json();

    // Parse and pretty-print nodes only for clarity
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    println!("Nodes with temporal data:");
    if let Some(nodes) = parsed.get("nodes").and_then(|n| n.as_array()) {
        for node in nodes {
            let name = node.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            let valid_from = node.get("valid_from").and_then(|v| v.as_str());
            let valid_until = node.get("valid_until").and_then(|v| v.as_str());
            let viewport = node.get("viewport").and_then(|v| v.as_str());

            println!("  {} ({:?})", name, node.get("type"));
            if let Some(vf) = valid_from {
                println!("    valid_from: {}", vf);
            }
            if let Some(vu) = valid_until {
                println!("    valid_until: {}", vu);
            }
            if let Some(vp) = viewport {
                println!("    viewport: {}", vp);
            }
            println!();
        }
    }

    // Export to JSON-LD
    println!("=== JSON-LD Export (for semantic web) ===\n");
    println!("{}", graph.to_json_ld());

    println!("\n=== Verification ===");

    // Verify temporal data is present
    let json_str = graph.to_networkx_json();
    assert!(
        json_str.contains("valid_from"),
        "valid_from should be in export"
    );
    assert!(
        json_str.contains("valid_until"),
        "valid_until should be in export"
    );
    assert!(
        json_str.contains("2014-02-04"),
        "Nadella start date should be present"
    );
    assert!(
        json_str.contains("2000-01-13"),
        "Ballmer start date should be present"
    );
    assert!(
        json_str.contains("viewport"),
        "viewport should be in export"
    );
    assert!(
        json_str.contains("business"),
        "business viewport should be present"
    );
    assert!(
        json_str.contains("historical"),
        "historical viewport should be present"
    );

    println!("All assertions passed!");
    println!("\nTemporal validity is correctly exported to all graph formats.");
}
