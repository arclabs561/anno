//! Adapters between `anno` and `lattix` (graph/KG export).
//!
//! Available when the `graph` feature is enabled.
//!
//! ## Functions
//!
//! - [`crate::graph::entities_to_knowledge_graph`] -- the preferred export path: takes raw
//!   extraction output (`Entity` + `Relation` slices) and emits a fully-annotated
//!   `KnowledgeGraph` including character-offset, confidence, and provenance triples.
//!   Used by `anno export --format graph-ntriples`.

use lattix::{GraphEdge, GraphNode, Triple};

// Re-export the lattix types that `anno::graph`'s public API surface returns,
// so callers don't need a separate `lattix` dep just to type-name what
// `anno::graph` produces. A breaking change in `lattix` then becomes a
// breaking change in `anno`'s `graph` feature, which is correct (and caught
// by `cargo-semver-checks`).
pub use lattix::{GraphDocument, GraphExportFormat, KnowledgeGraph};

/// Convert a `GroundedDocument` into a `lattix::exchange::GraphDocument`.
///
/// Mentions of the same catalog record share a node. A KB ID is used verbatim when
/// it belongs to one catalog in this document. If multiple catalogs use the same ID,
/// their nodes receive distinct IDs derived from the catalog and record ID.
/// Relations are not currently stored in `GroundedDocument`. To include
/// extraction-time relations, use [`entities_to_graph_document`] directly with
/// the `Relation` slice from the extraction backend.
#[must_use]
pub fn grounded_to_graph_document(doc: &crate::GroundedDocument) -> GraphDocument {
    use std::collections::{HashMap, HashSet};

    let mut entities = doc.to_entities();
    let mut catalogs = HashMap::new();
    let mut ambiguous = HashSet::new();
    for track in doc.tracks().filter(|track| !track.signals().is_empty()) {
        let Some(identity) = doc.identity_for_track(track.id) else {
            continue;
        };
        if let Some(id) = identity.kb_id() {
            let catalog = identity.kb_name();
            if catalogs
                .insert(id, catalog)
                .is_some_and(|prior| prior != catalog)
            {
                ambiguous.insert(id);
            }
        }
    }

    if !ambiguous.is_empty() {
        let mut aliases = HashMap::new();
        for (signal, entity) in doc.signals().iter().zip(&mut entities) {
            let Some(identity) = doc.identity_for_signal(signal.id) else {
                continue;
            };
            let Some(id) = identity.kb_id().filter(|id| ambiguous.contains(id)) else {
                continue;
            };
            // The JSON pair distinguishes missing and empty catalog names and arbitrary
            // punctuation. Its closing bracket cannot occur at the end of a fallback node ID.
            let alias = aliases.entry((identity.kb_name(), id)).or_insert_with(|| {
                let mut candidate = serde_json::json!([identity.kb_name(), id]).to_string();
                while catalogs.contains_key(candidate.as_str()) {
                    candidate.insert(0, '~');
                }
                candidate
            });
            entity.kb_id = Some(alias.clone());
        }
    }
    entities_to_graph_document(&entities, &[])
}

/// Convert entities and relations into a `lattix::exchange::GraphDocument`.
#[must_use]
pub fn entities_to_graph_document(
    entities: &[crate::Entity],
    relations: &[crate::Relation],
) -> GraphDocument {
    let mut doc = GraphDocument::new();
    let mut seen_nodes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut entity_to_node: std::collections::HashMap<usize, String> =
        std::collections::HashMap::new();

    let get_node_id = |e: &crate::Entity| -> String {
        if let Some(ref kb_id) = e.kb_id {
            return kb_id.clone();
        }
        if let Some(canonical_id) = e.canonical_id {
            return format!("coref_{}", canonical_id);
        }
        format!(
            "{}:{}",
            e.entity_type.as_label().to_lowercase(),
            uri_safe(&e.text)
        )
    };

    for (idx, entity) in entities.iter().enumerate() {
        let node_id = get_node_id(entity);

        if let Some(&existing_idx) = seen_nodes.get(&node_id) {
            if let Some(count) = doc.nodes[existing_idx].properties.get_mut("mentions_count") {
                if let Some(n) = count.as_u64() {
                    *count = serde_json::Value::from(n + 1);
                }
            }
            entity_to_node.insert(idx, node_id);
            continue;
        }

        let node = GraphNode::new(&node_id, entity.entity_type.as_label(), &entity.text)
            .with_mentions_count(1)
            .with_first_seen(entity.start());

        seen_nodes.insert(node_id.clone(), doc.nodes.len());
        entity_to_node.insert(idx, node_id);
        doc.nodes.push(node);
    }

    let mut seen_edges: std::collections::HashMap<(String, String, String), usize> =
        std::collections::HashMap::new();
    for relation in relations {
        let source_node_id = get_node_id(&relation.head);
        let target_node_id = get_node_id(&relation.tail);

        if seen_nodes.contains_key(&source_node_id) && seen_nodes.contains_key(&target_node_id) {
            let key = (
                source_node_id.clone(),
                target_node_id.clone(),
                relation.relation_type.clone(),
            );
            if let Some(&idx) = seen_edges.get(&key) {
                if let Some(existing) = doc.edges.get_mut(idx) {
                    existing.confidence = existing.confidence.max(relation.confidence.value());
                }
            } else {
                let edge =
                    GraphEdge::new(&source_node_id, &target_node_id, &relation.relation_type)
                        .with_confidence(relation.confidence.value());
                doc.edges.push(edge);
                seen_edges.insert(key, doc.edges.len().saturating_sub(1));
            }
        }
    }
    doc
}

// ---------------------------------------------------------------------------
// URI helpers (shared across all export paths)
// ---------------------------------------------------------------------------

/// Make a string safe for use inside a URI path segment.
pub fn uri_safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn escape_literal(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// Raw entity/relation extraction output → KnowledgeGraph
// ---------------------------------------------------------------------------

/// Build a fully-annotated `KnowledgeGraph` from raw NER + relation extraction output.
///
/// Each entity becomes a subject with:
/// - `rdf:type` assertion
/// - `rdfs:label` (surface text)
/// - character offset and confidence typed literals
/// - `prov:hadPrimarySource` provenance link to the document IRI
///
/// When `relations` is non-empty (i.e. the model supports relation extraction), each triple becomes
/// a predicate arc: `<head_entity> <{base}/rel/{type}> <tail_entity>`.
///
/// # Arguments
///
/// - `entities` — extracted entity spans
/// - `relations` — semantic triples; empty for entity-only backends
/// - `doc_iri` — IRI identifying this document (e.g. `https://www.gutenberg.org/ebooks/doc/pg1342`)
/// - `base_uri` — namespace prefix (e.g. `https://www.gutenberg.org/ebooks/`)
///
/// # Returns
///
/// A `KnowledgeGraph` whose triples can be serialised to N-Triples via
/// `kg.triples().map(|t| t.to_ntriples()).collect::<Vec<_>>().join("\n")`.
#[must_use]
pub fn entities_to_knowledge_graph(
    entities: &[crate::Entity],
    relations: &[crate::Relation],
    doc_iri: &str,
    base_uri: &str,
) -> KnowledgeGraph {
    let mut kg =
        KnowledgeGraph::with_capacity(entities.len().max(1), entities.len() * 7 + relations.len());

    let base = base_uri.trim_end_matches('/');
    let anno_ns = format!("{}/vocab#", base);

    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
    const PROV_SOURCE: &str = "http://www.w3.org/ns/prov#hadPrimarySource";
    const XSD_INT: &str = "http://www.w3.org/2001/XMLSchema#integer";
    const XSD_FLOAT: &str = "http://www.w3.org/2001/XMLSchema#float";

    let entity_iris = crate::export::occurrence_iris(entities, doc_iri, base_uri);

    for (idx, entity) in entities.iter().enumerate() {
        let iri = &entity_iris[idx];
        let type_iri = format!("{}{}Type", anno_ns, uri_safe(entity.entity_type.as_label()));

        kg.add_triple(Triple::new(iri.as_str(), RDF_TYPE, type_iri.as_str()));
        kg.add_triple(Triple::new(
            iri.as_str(),
            RDFS_LABEL,
            format!("\"{}\"", escape_literal(&entity.text)),
        ));
        kg.add_triple(Triple::new(
            iri.as_str(),
            format!("{}startOffset", anno_ns),
            format!("\"{}\"^^<{}>", entity.start(), XSD_INT),
        ));
        kg.add_triple(Triple::new(
            iri.as_str(),
            format!("{}endOffset", anno_ns),
            format!("\"{}\"^^<{}>", entity.end(), XSD_INT),
        ));
        kg.add_triple(Triple::new(
            iri.as_str(),
            format!("{}confidence", anno_ns),
            format!("\"{}\"^^<{}>", entity.confidence, XSD_FLOAT),
        ));
        kg.add_triple(Triple::new(iri.as_str(), PROV_SOURCE, doc_iri));
        kg.add_triple(Triple::new(
            doc_iri,
            format!("{}mentions", anno_ns),
            iri.as_str(),
        ));
    }

    for (rel, h, t) in crate::export::resolved_relations(entities, relations) {
        let pred = format!("{}/rel/{}", base, uri_safe(&rel.relation_type));
        let mut triple = Triple::new(
            entity_iris[h].as_str(),
            pred.as_str(),
            entity_iris[t].as_str(),
        );
        if rel.confidence.value().is_finite() {
            triple = triple.with_confidence(f32::from(rel.confidence));
        }
        kg.add_triple(triple);
    }

    kg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Entity, EntityType, Relation};

    fn ent(text: &str, start: usize, end: usize, ty: EntityType) -> Entity {
        Entity::new(text, ty, start, end, 0.9)
    }

    #[test]
    fn grounded_graph_separates_catalogs_that_reuse_an_id() {
        use crate::{GroundedDocument, Identity, Location, Signal};

        let mut doc = GroundedDocument::new("d", "X X X X X");
        let catalog_links = [
            ("alpha", "42"),
            ("alpha", "42"),
            ("beta", "42"),
            ("alpha", "unique"),
            // A raw ID may look exactly like a generated scoped ID.
            ("alpha", r#"["alpha","42"]"#),
        ];
        for (i, (catalog, id)) in catalog_links.into_iter().enumerate() {
            let signal = doc.add_signal(Signal::new(
                0,
                Location::text(i * 2, i * 2 + 1),
                "X",
                "PER",
                1.0,
            ));
            let track = doc.create_track_from_signals("X", &[signal]).unwrap();
            let identity = doc.add_identity(Identity::from_kb(0, "X", catalog, id));
            doc.link_track_to_identity(track, identity);
        }
        let graph = grounded_to_graph_document(&doc);
        assert_eq!(graph.nodes.len(), 4);
        let nodes = &graph.nodes;
        assert_eq!(nodes[0].properties["mentions_count"], 2);
        assert_eq!(nodes[1].properties["mentions_count"], 1);
        assert_ne!(nodes[0].id, nodes[1].id);
        assert_eq!(nodes[2].id, "unique");
        assert_eq!(nodes[3].id, r#"["alpha","42"]"#);
        assert_ne!(nodes[0].id, nodes[3].id);
    }

    #[test]
    fn kg_produces_type_label_provenance_triples() {
        let entities = vec![ent("Lynn Conway", 0, 11, EntityType::Person)];
        let kg = entities_to_knowledge_graph(&entities, &[], "urn:test:doc/d1", "urn:test:");
        let triples: Vec<String> = kg.triples().map(|t| t.to_ntriples()).collect();

        assert!(
            triples.len() >= 6,
            "expected ≥6 triples, got {}",
            triples.len()
        );
        assert!(triples
            .iter()
            .any(|t| t.contains("rdf-syntax-ns#type") && t.contains("PERType")));
        assert!(triples.iter().any(|t| t.contains("rdf-schema#label")));
        assert!(triples
            .iter()
            .any(|t| t.contains("prov#hadPrimarySource") || t.contains("prov/ns#")));
    }

    #[test]
    fn kg_includes_relation_arc() {
        let head = ent("Steve Jobs", 0, 10, EntityType::Person);
        let tail = ent("Apple", 19, 24, EntityType::Organization);
        let rel = Relation::new(head.clone(), tail.clone(), "founded", 0.85);

        let kg = entities_to_knowledge_graph(&[head, tail], &[rel], "urn:test:doc/d2", "urn:test:");
        let triples: Vec<String> = kg.triples().map(|t| t.to_ntriples()).collect();
        assert!(
            triples.iter().any(|t| t.contains("rel/founded")),
            "missing relation triple; triples:\n{}",
            triples.join("\n")
        );
    }

    #[test]
    fn empty_entities_empty_kg() {
        let kg = entities_to_knowledge_graph(&[], &[], "urn:test:doc/empty", "urn:test:");
        assert_eq!(kg.triples().count(), 0);
    }

    #[test]
    fn uri_safe_replaces_specials() {
        assert_eq!(uri_safe("Lynn Conway"), "Lynn_Conway");
        assert_eq!(uri_safe("IBM"), "IBM");
        assert_eq!(uri_safe("New York"), "New_York");
    }
}
