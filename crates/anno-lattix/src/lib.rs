//! Adapters between `anno-core` and `lattix`.
//!
//! This crate exists to avoid introducing a dependency edge from `anno-core` → `lattix` while
//! still letting downstream tooling (CLI, eval harnesses, other workspace crates) treat `lattix`
//! as the canonical graph substrate.

/// Convert an `anno_core::GraphDocument` (interop/export shape) into a `lattix::GraphDocument`.
#[must_use]
pub fn to_lattix_graph_document(g: &anno_core::GraphDocument) -> lattix::GraphDocument {
    lattix::GraphDocument {
        nodes: g
            .nodes
            .iter()
            .map(|n| lattix::GraphNode {
                id: n.id.clone(),
                node_type: n.node_type.clone(),
                name: n.name.clone(),
                properties: n.properties.clone(),
            })
            .collect(),
        edges: g
            .edges
            .iter()
            .map(|e| lattix::GraphEdge {
                source: e.source.clone(),
                target: e.target.clone(),
                relation: e.relation.clone(),
                confidence: e.confidence,
                properties: e.properties.clone(),
            })
            .collect(),
        metadata: g.metadata.clone(),
    }
}

/// Convert a `GroundedDocument` into a `lattix::GraphDocument`.
///
/// Today this uses `anno_core::GraphDocument::from_grounded_document` as the extraction-to-graph
/// adapter, then maps the interchange types into `lattix`.
#[must_use]
pub fn grounded_to_lattix_graph(doc: &anno_core::GroundedDocument) -> lattix::GraphDocument {
    let g = anno_core::GraphDocument::from_grounded_document(doc);
    to_lattix_graph_document(&g)
}
