//! Graph RAG integration: convert NER/Coref output to graph format.
//!
//! This module bridges the gap between NER/IE extraction and graph databases
//! like Neo4j or NetworkX for RAG (Retrieval-Augmented Generation) applications.
//!
//! # The "Fractured Graph" Problem
//!
//! Without coreference resolution, the same real-world entity creates multiple
//! disconnected nodes:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                    FRACTURED GRAPH (BAD)                          │
//! ├────────────────────────────────────────────────────────────────────┤
//! │                                                                    │
//! │    [Elon Musk]          [Musk]          [he]          [The CEO]   │
//! │         │                  │               │               │      │
//! │         v                  v               v               v      │
//! │    "founded Tesla"   "bought Twitter"  "said..."    "announced"  │
//! │                                                                    │
//! │    → 4 disconnected nodes, no cross-reference!                    │
//! └────────────────────────────────────────────────────────────────────┘
//!
//! ┌────────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED GRAPH (GOOD)                           │
//! ├────────────────────────────────────────────────────────────────────┤
//! │                                                                    │
//! │                       [Elon Musk]                                  │
//! │                     (canonical node)                               │
//! │                    /      |      \                                │
//! │                   v       v       v                               │
//! │           "founded"  "bought"  "announced"                        │
//! │               |          |          |                             │
//! │               v          v          v                             │
//! │           [Tesla]   [Twitter]   [layoffs]                        │
//! │                                                                    │
//! │    → 1 node with all relationships connected                      │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno_core::graph::{GraphDocument, GraphExportFormat};
//! use anno_core::{Entity, EntityType, Relation};
//! use anno_core::eval::coref::CorefChain;
//!
//! // From NER extraction
//! let elon = Entity::new("Elon Musk", EntityType::Person, 0, 9, 0.9)
//!     .with_canonical_id(1);
//! let tesla = Entity::new("Tesla", EntityType::Organization, 19, 24, 0.95)
//!     .with_canonical_id(2);
//!
//! // From relation extraction (head and tail are Entity clones)
//! let relations = vec![
//!     Relation::new(elon.clone(), tesla.clone(), "FOUNDED", 0.85),
//! ];
//!
//! let entities = vec![elon, tesla];
//!
//! // Build graph document
//! let graph = GraphDocument::from_extraction(&entities, &relations, None);
//!
//! // Export to Neo4j Cypher
//! println!("{}", graph.to_cypher());
//!
//! // Export to NetworkX JSON
//! println!("{}", graph.to_networkx_json());
//! ```
//!
//! # Research Background
//!
//! - **GraphRAG** (Microsoft, 2024-2025): Combines knowledge graphs with vector retrieval
//! - **Entity Linking**: Maps extracted mentions to canonical KB entities
//! - **Coreference Resolution**: Clusters mentions referring to the same entity

use crate::entity::{Entity, Relation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Core Data Structures
// =============================================================================

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier (from canonical_id, kb_id, or generated)
    pub id: String,
    /// Node type/label (from EntityType)
    pub node_type: String,
    /// Display name (canonical mention text)
    pub name: String,
    /// Additional properties
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

impl GraphNode {
    /// Create a new graph node.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        node_type: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            node_type: node_type.into(),
            name: name.into(),
            properties: HashMap::new(),
        }
    }

    /// Add a property to the node.
    #[must_use]
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Add mention count property.
    #[must_use]
    pub fn with_mentions_count(self, count: usize) -> Self {
        self.with_property("mentions_count", count)
    }

    /// Add first occurrence offset.
    #[must_use]
    pub fn with_first_seen(self, offset: usize) -> Self {
        self.with_property("first_seen", offset)
    }
}

/// An edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Relation type
    pub relation: String,
    /// Confidence score (0.0 - 1.0)
    #[serde(default)]
    pub confidence: f64,
    /// Additional properties
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

impl GraphEdge {
    /// Create a new graph edge.
    #[must_use]
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        relation: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            relation: relation.into(),
            confidence: 1.0,
            properties: HashMap::new(),
        }
    }

    /// Set confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Add a property to the edge.
    #[must_use]
    pub fn with_property(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Add trigger text property.
    #[must_use]
    pub fn with_trigger(self, trigger: impl Into<String>) -> Self {
        self.with_property("trigger", trigger.into())
    }
}

/// A complete graph document ready for export.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphDocument {
    /// Nodes (entities)
    pub nodes: Vec<GraphNode>,
    /// Edges (relations)
    pub edges: Vec<GraphEdge>,
    /// Document metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GraphDocument {
    /// Create an empty graph document.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build graph document from NER/IE extraction results.
    ///
    /// # Arguments
    /// * `entities` - Extracted entities (should have `canonical_id` set if coref was run)
    /// * `relations` - Extracted relations between entities
    /// * `coref_chains` - Optional coreference chains for canonical mention resolution
    ///
    /// # Returns
    /// A `GraphDocument` with deduplicated nodes (by canonical_id) and edges.
    #[must_use]
    pub fn from_extraction(
        entities: &[Entity],
        relations: &[Relation],
        // Note: coref_chains removed from anno-core - will be added back in anno crate
        // For now, canonical_id on entities is sufficient
        _coref_chains: Option<()>,
    ) -> Self {
        let mut doc = Self::new();

        // Build canonical mention map from entities with canonical_id
        // (CorefChain support will be added in anno crate)
        let canonical_mentions: HashMap<u64, (&str, usize)> = HashMap::new();

        // Track seen canonical IDs to avoid duplicate nodes
        let mut seen_nodes: HashMap<String, usize> = HashMap::new();
        let mut entity_to_node: HashMap<usize, String> = HashMap::new();

        // 1. Create nodes from entities (deduplicated by canonical_id)
        for (idx, entity) in entities.iter().enumerate() {
            let node_id = get_node_id(entity);

            // Check if we already have this canonical entity
            if let Some(&existing_idx) = seen_nodes.get(&node_id) {
                // Update mention count
                if let Some(count) = doc.nodes[existing_idx].properties.get_mut("mentions_count") {
                    if let Some(n) = count.as_u64() {
                        *count = serde_json::Value::from(n + 1);
                    }
                }
                entity_to_node.insert(idx, node_id);
                continue;
            }

            // Determine canonical name
            let (name, mentions_count) = if let Some(canonical_id) = entity.canonical_id {
                canonical_mentions
                    .get(&canonical_id)
                    .map(|(text, count)| (text.to_string(), *count))
                    .unwrap_or_else(|| (entity.text.clone(), 1))
            } else {
                (entity.text.clone(), 1)
            };

            let mut node = GraphNode::new(&node_id, entity.entity_type.as_label(), name)
                .with_mentions_count(mentions_count)
                .with_first_seen(entity.start);

            // Add temporal validity if present
            if let Some(valid_from) = &entity.valid_from {
                node = node.with_property("valid_from", valid_from.to_rfc3339());
            }
            if let Some(valid_until) = &entity.valid_until {
                node = node.with_property("valid_until", valid_until.to_rfc3339());
            }

            // Add viewport if present
            if let Some(viewport) = &entity.viewport {
                node = node.with_property("viewport", viewport.as_str());
            }

            seen_nodes.insert(node_id.clone(), doc.nodes.len());
            entity_to_node.insert(idx, node_id);
            doc.nodes.push(node);
        }

        // 2. Create edges from relations
        for relation in relations {
            // Get node IDs directly from relation entities
            let source_node_id = get_node_id(&relation.head);
            let target_node_id = get_node_id(&relation.tail);

            // Only create edge if both nodes exist in the graph
            let source_exists = seen_nodes.contains_key(&source_node_id);
            let target_exists = seen_nodes.contains_key(&target_node_id);

            if source_exists && target_exists {
                let edge =
                    GraphEdge::new(&source_node_id, &target_node_id, &relation.relation_type)
                        .with_confidence(relation.confidence);

                doc.edges.push(edge);
            }
        }

        doc
    }

    /// Build graph from entities only, inferring co-occurrence relations.
    ///
    /// Uses a simple heuristic: entities within `max_distance` characters
    /// are considered related. This is useful when no explicit relation
    /// extraction was performed.
    #[must_use]
    pub fn from_entities_cooccurrence(entities: &[Entity], max_distance: usize) -> Self {
        let mut doc = Self::new();
        let mut entity_to_node: HashMap<usize, String> = HashMap::new();
        let mut seen_nodes: HashMap<String, usize> = HashMap::new();

        // Create nodes
        for (idx, entity) in entities.iter().enumerate() {
            let node_id = get_node_id(entity);

            if seen_nodes.contains_key(&node_id) {
                entity_to_node.insert(idx, node_id);
                continue;
            }

            let mut node = GraphNode::new(&node_id, entity.entity_type.as_label(), &entity.text)
                .with_first_seen(entity.start);

            // Add temporal validity if present
            if let Some(valid_from) = &entity.valid_from {
                node = node.with_property("valid_from", valid_from.to_rfc3339());
            }
            if let Some(valid_until) = &entity.valid_until {
                node = node.with_property("valid_until", valid_until.to_rfc3339());
            }

            // Add viewport if present
            if let Some(viewport) = &entity.viewport {
                node = node.with_property("viewport", viewport.as_str());
            }

            seen_nodes.insert(node_id.clone(), doc.nodes.len());
            entity_to_node.insert(idx, node_id);
            doc.nodes.push(node);
        }

        // Create co-occurrence edges
        for (i, entity_a) in entities.iter().enumerate() {
            for (j, entity_b) in entities.iter().enumerate().skip(i + 1) {
                let distance = if entity_a.end <= entity_b.start {
                    entity_b.start.saturating_sub(entity_a.end)
                } else if entity_b.end <= entity_a.start {
                    entity_a.start.saturating_sub(entity_b.end)
                } else {
                    0 // overlapping
                };

                if distance <= max_distance {
                    if let (Some(source), Some(target)) =
                        (entity_to_node.get(&i), entity_to_node.get(&j))
                    {
                        // Don't create self-loops
                        if source != target {
                            let edge = GraphEdge::new(source, target, "RELATED_TO")
                                .with_property("distance", distance);
                            doc.edges.push(edge);
                        }
                    }
                }
            }
        }

        doc
    }

    /// Export to Neo4j Cypher CREATE statements.
    #[must_use]
    pub fn to_cypher(&self) -> String {
        let mut cypher = String::new();

        // Create nodes
        for node in &self.nodes {
            let props = format_cypher_props(&node.properties, &node.name);
            cypher.push_str(&format!(
                "CREATE (n{}:{} {{id: '{}'{}}});\n",
                sanitize_cypher_name(&node.id),
                sanitize_cypher_name(&node.node_type),
                escape_cypher_string(&node.id),
                props
            ));
        }

        cypher.push('\n');

        // Create edges
        for edge in &self.edges {
            let props = if edge.confidence < 1.0 {
                format!(" {{confidence: {:.3}}}", edge.confidence)
            } else {
                String::new()
            };

            cypher.push_str(&format!(
                "MATCH (a {{id: '{}'}}), (b {{id: '{}'}}) CREATE (a)-[:{}{}]->(b);\n",
                escape_cypher_string(&edge.source),
                escape_cypher_string(&edge.target),
                sanitize_cypher_name(&edge.relation),
                props
            ));
        }

        cypher
    }

    /// Export to NetworkX-compatible JSON format.
    ///
    /// This format can be loaded directly with:
    /// ```python
    /// import networkx as nx
    /// import json
    /// with open('graph.json') as f:
    ///     data = json.load(f)
    /// G = nx.node_link_graph(data)
    /// ```
    #[must_use]
    pub fn to_networkx_json(&self) -> String {
        #[derive(Serialize)]
        struct NetworkXGraph<'a> {
            directed: bool,
            multigraph: bool,
            graph: HashMap<String, serde_json::Value>,
            nodes: Vec<NetworkXNode<'a>>,
            links: Vec<NetworkXLink<'a>>,
        }

        #[derive(Serialize)]
        struct NetworkXNode<'a> {
            id: &'a str,
            #[serde(rename = "type")]
            node_type: &'a str,
            name: &'a str,
            #[serde(flatten)]
            properties: &'a HashMap<String, serde_json::Value>,
        }

        #[derive(Serialize)]
        struct NetworkXLink<'a> {
            source: &'a str,
            target: &'a str,
            relation: &'a str,
            #[serde(skip_serializing_if = "is_default_confidence")]
            confidence: f64,
            #[serde(flatten)]
            properties: &'a HashMap<String, serde_json::Value>,
        }

        fn is_default_confidence(c: &f64) -> bool {
            (*c - 1.0).abs() < f64::EPSILON
        }

        let graph = NetworkXGraph {
            directed: true,
            multigraph: false,
            graph: self.metadata.clone(),
            nodes: self
                .nodes
                .iter()
                .map(|n| NetworkXNode {
                    id: &n.id,
                    node_type: &n.node_type,
                    name: &n.name,
                    properties: &n.properties,
                })
                .collect(),
            links: self
                .edges
                .iter()
                .map(|e| NetworkXLink {
                    source: &e.source,
                    target: &e.target,
                    relation: &e.relation,
                    confidence: e.confidence,
                    properties: &e.properties,
                })
                .collect(),
        };

        serde_json::to_string_pretty(&graph).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export to JSON-LD format (for semantic web applications).
    #[must_use]
    pub fn to_json_ld(&self) -> String {
        #[derive(Serialize)]
        struct JsonLd<'a> {
            #[serde(rename = "@context")]
            context: JsonLdContext,
            #[serde(rename = "@graph")]
            graph: Vec<JsonLdNode<'a>>,
        }

        #[derive(Serialize)]
        struct JsonLdContext {
            #[serde(rename = "@vocab")]
            vocab: &'static str,
            name: &'static str,
            #[serde(rename = "type")]
            type_: &'static str,
        }

        #[derive(Serialize)]
        struct JsonLdNode<'a> {
            #[serde(rename = "@id")]
            id: &'a str,
            #[serde(rename = "@type")]
            node_type: &'a str,
            name: &'a str,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            relations: Vec<JsonLdRelation<'a>>,
        }

        #[derive(Serialize)]
        struct JsonLdRelation<'a> {
            #[serde(rename = "@type")]
            relation_type: &'a str,
            target: &'a str,
        }

        // Group edges by source
        let mut node_edges: HashMap<&str, Vec<&GraphEdge>> = HashMap::new();
        for edge in &self.edges {
            node_edges.entry(&edge.source).or_default().push(edge);
        }

        let doc = JsonLd {
            context: JsonLdContext {
                vocab: "http://schema.org/",
                name: "http://schema.org/name",
                type_: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            },
            graph: self
                .nodes
                .iter()
                .map(|n| JsonLdNode {
                    id: &n.id,
                    node_type: &n.node_type,
                    name: &n.name,
                    relations: node_edges
                        .get(n.id.as_str())
                        .map(|edges| {
                            edges
                                .iter()
                                .map(|e| JsonLdRelation {
                                    relation_type: &e.relation,
                                    target: &e.target,
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                })
                .collect(),
        };

        serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string())
    }

    /// Add metadata to the graph document.
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Build graph document from a GroundedDocument.
    ///
    /// Converts the Signal → Track → Identity hierarchy to a graph format
    /// suitable for RAG applications (Neo4j, NetworkX, etc.).
    ///
    /// # Arguments
    /// * `doc` - The GroundedDocument to convert
    ///
    /// # Returns
    /// A GraphDocument with nodes from entities and edges inferred from
    /// co-occurrence or track relationships.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno_core::grounded::GroundedDocument;
    /// use anno_core::graph::GraphDocument;
    ///
    /// let doc = GroundedDocument::new("doc1", "Marie Curie won the Nobel Prize.");
    /// // ... add signals, tracks, identities ...
    ///
    /// let graph = GraphDocument::from_grounded_document(&doc);
    /// println!("{}", graph.to_cypher());
    /// ```
    #[must_use]
    pub fn from_grounded_document(doc: &crate::grounded::GroundedDocument) -> Self {
        // EntityType conversion handled inline below

        // Convert signals to entities
        let entities: Vec<crate::Entity> = doc.to_entities();

        // Note: coref chains support moved to anno crate
        // For now, use canonical_id from entities directly

        // No relations available in GroundedDocument yet
        // Could be extended in the future to store relations
        let relations: Vec<crate::entity::Relation> = Vec::new();

        Self::from_extraction(&entities, &relations, None)
    }
}

// =============================================================================
// Export Format Enum
// =============================================================================

/// Supported graph export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphExportFormat {
    /// Neo4j Cypher CREATE statements
    Cypher,
    /// NetworkX-compatible JSON (node_link_graph format)
    NetworkXJson,
    /// JSON-LD for semantic web
    JsonLd,
}

impl GraphDocument {
    /// Export to the specified format.
    #[must_use]
    pub fn export(&self, format: GraphExportFormat) -> String {
        match format {
            GraphExportFormat::Cypher => self.to_cypher(),
            GraphExportFormat::NetworkXJson => self.to_networkx_json(),
            GraphExportFormat::JsonLd => self.to_json_ld(),
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get a stable node ID for an entity.
fn get_node_id(entity: &Entity) -> String {
    // Priority: kb_id > canonical_id > content hash
    if let Some(ref kb_id) = entity.kb_id {
        return kb_id.clone();
    }
    if let Some(canonical_id) = entity.canonical_id {
        return format!("coref_{}", canonical_id);
    }
    // Fall back to a content-based ID
    format!(
        "{}:{}",
        entity.entity_type.as_label().to_lowercase(),
        entity.text.to_lowercase().replace(' ', "_")
    )
}

/// Format properties for Cypher (excluding name which is handled separately).
fn format_cypher_props(props: &HashMap<String, serde_json::Value>, name: &str) -> String {
    let mut parts = vec![format!("name: '{}'", escape_cypher_string(name))];

    for (key, value) in props {
        let formatted = match value {
            serde_json::Value::String(s) => format!("{}: '{}'", key, escape_cypher_string(s)),
            serde_json::Value::Number(n) => format!("{}: {}", key, n),
            serde_json::Value::Bool(b) => format!("{}: {}", key, b),
            _ => continue,
        };
        parts.push(formatted);
    }

    if parts.len() > 1 {
        format!(", {}", parts[1..].join(", "))
    } else {
        String::new()
    }
}

/// Escape special characters in Cypher strings.
fn escape_cypher_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Sanitize names for Cypher identifiers.
fn sanitize_cypher_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    fn make_entity(text: &str, entity_type: EntityType, start: usize) -> Entity {
        Entity::new(text, entity_type, start, start + text.len(), 0.9)
    }

    #[test]
    fn test_graph_from_entities() {
        let elon = make_entity("Elon Musk", EntityType::Person, 0).with_canonical_id(1);
        let tesla = make_entity("Tesla", EntityType::Organization, 19).with_canonical_id(2);

        let relations = vec![Relation::with_trigger(
            elon.clone(),
            tesla.clone(),
            "FOUNDED",
            10,
            17,
            0.85,
        )];
        let entities = vec![elon, tesla];

        let graph = GraphDocument::from_extraction(&entities, &relations, None);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.edges[0].relation, "FOUNDED");
    }

    #[test]
    fn test_graph_deduplication() {
        let entities = vec![
            make_entity("Elon Musk", EntityType::Person, 0).with_canonical_id(1),
            make_entity("Musk", EntityType::Person, 50).with_canonical_id(1), // Same canonical
            make_entity("Tesla", EntityType::Organization, 100).with_canonical_id(2),
        ];

        let graph = GraphDocument::from_extraction(&entities, &[], None);

        // Should have 2 nodes (Elon Musk and Musk deduplicated)
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_cypher_export() {
        let entities = vec![make_entity("Apple", EntityType::Organization, 0)];
        let graph = GraphDocument::from_extraction(&entities, &[], None);

        let cypher = graph.to_cypher();
        assert!(cypher.contains("CREATE"));
        assert!(cypher.contains(":ORG"));
    }

    #[test]
    fn test_networkx_json() {
        let entity_a = make_entity("A", EntityType::Person, 0);
        let entity_b = make_entity("B", EntityType::Organization, 10);
        let entities = vec![entity_a.clone(), entity_b.clone()];
        let relations = vec![Relation::new(entity_a, entity_b, "WORKS_AT", 0.9)];

        let graph = GraphDocument::from_extraction(&entities, &relations, None);
        let json = graph.to_networkx_json();

        // Verify JSON structure
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("nodes").is_some());
        assert!(parsed.get("links").is_some());
        assert_eq!(parsed["directed"], true);
    }

    #[test]
    fn test_cooccurrence_graph() {
        let entities = vec![
            make_entity("A", EntityType::Person, 0),
            make_entity("B", EntityType::Organization, 20),
            make_entity("C", EntityType::Location, 100), // Far away
        ];

        let graph = GraphDocument::from_entities_cooccurrence(&entities, 50);

        // A and B are within 50 chars, C is not
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 1); // Only A-B edge
    }

    #[test]
    fn test_json_ld_export() {
        let entities = vec![make_entity("Test", EntityType::Person, 0)];
        let graph = GraphDocument::from_extraction(&entities, &[], None);

        let json_ld = graph.to_json_ld();
        let parsed: serde_json::Value = serde_json::from_str(&json_ld).unwrap();

        assert!(parsed.get("@context").is_some());
        assert!(parsed.get("@graph").is_some());
    }

    #[test]
    fn test_temporal_validity_export() {
        use crate::EntityViewport;
        use chrono::{TimeZone, Utc};

        // Create entity with temporal validity (CEO tenure)
        let mut nadella = make_entity("Satya Nadella", EntityType::Person, 0);
        nadella.valid_from = Some(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap());
        nadella.viewport = Some(EntityViewport::Business);

        // Create entity with historical validity (past CEO)
        let mut ballmer = make_entity("Steve Ballmer", EntityType::Person, 50);
        ballmer.valid_from = Some(Utc.with_ymd_and_hms(2000, 1, 13, 0, 0, 0).unwrap());
        ballmer.valid_until = Some(Utc.with_ymd_and_hms(2014, 2, 4, 0, 0, 0).unwrap());
        ballmer.viewport = Some(EntityViewport::Historical);

        let entities = vec![nadella, ballmer];
        let graph = GraphDocument::from_extraction(&entities, &[], None);

        // Verify temporal properties are exported
        assert_eq!(graph.node_count(), 2);

        // Check Nadella node has valid_from but no valid_until
        let nadella_node = graph
            .nodes
            .iter()
            .find(|n| n.name == "Satya Nadella")
            .unwrap();
        assert!(nadella_node.properties.contains_key("valid_from"));
        assert!(!nadella_node.properties.contains_key("valid_until"));
        assert_eq!(nadella_node.properties.get("viewport").unwrap(), "business");

        // Check Ballmer node has both valid_from and valid_until
        let ballmer_node = graph
            .nodes
            .iter()
            .find(|n| n.name == "Steve Ballmer")
            .unwrap();
        assert!(ballmer_node.properties.contains_key("valid_from"));
        assert!(ballmer_node.properties.contains_key("valid_until"));
        assert_eq!(
            ballmer_node.properties.get("viewport").unwrap(),
            "historical"
        );

        // Verify JSON export includes temporal data
        let json = graph.to_networkx_json();
        assert!(json.contains("valid_from"));
        assert!(json.contains("valid_until"));
        assert!(json.contains("2014-02-04")); // Nadella start date
        assert!(json.contains("2000-01-13")); // Ballmer start date
    }
}
