//! Adapters between `anno-core` and `lattix`.
//!
//! This crate exists to avoid introducing a dependency edge from `anno-core` → `lattix` while
//! still letting downstream tooling treat `lattix` as the graph/KG substrate.

use lattix::{KnowledgeGraph, Triple};

fn escape_literal(s: &str) -> String {
    // Minimal N-Triples literal escaping; keep it predictable.
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Convert an `anno_core::GraphDocument` into a `lattix::KnowledgeGraph`.
///
/// Mapping:
/// - Each `GraphEdge` becomes one triple: `(source, relation, target)`
/// - Each `GraphNode` adds best-effort `rdf:type` and `rdfs:label` triples when present
#[must_use]
pub fn to_lattix_knowledge_graph(g: &anno_core::GraphDocument) -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::new();

    // Nodes: best-effort metadata.
    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
    const ANNO_NODE_TYPE: &str = "urn:anno:node_type:";

    for n in &g.nodes {
        if !n.node_type.is_empty() {
            kg.add_triple(Triple::new(
                n.id.as_str(),
                RDF_TYPE,
                format!("{ANNO_NODE_TYPE}{}", n.node_type),
            ));
        }
        // `GraphNode.name` is always present (may be empty).
        if !n.name.trim().is_empty() {
            let lit = format!("\"{}\"", escape_literal(n.name.as_str()));
            kg.add_triple(Triple::new(n.id.as_str(), RDFS_LABEL, lit));
        }
    }

    // Edges: the main content.
    for e in &g.edges {
        let mut t = Triple::new(e.source.as_str(), e.relation.as_str(), e.target.as_str());
        if e.confidence.is_finite() {
            t = t.with_confidence((e.confidence as f32).clamp(0.0, 1.0));
        }
        kg.add_triple(t);
    }

    kg
}

/// Convert a `GroundedDocument` into a `lattix::KnowledgeGraph`.
#[must_use]
pub fn grounded_to_lattix_knowledge_graph(doc: &anno_core::GroundedDocument) -> KnowledgeGraph {
    let g = anno_core::GraphDocument::from_grounded_document(doc);
    to_lattix_knowledge_graph(&g)
}
