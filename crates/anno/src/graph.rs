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
use std::collections::HashMap;

// Re-export the lattix types that `anno::graph`'s public API surface returns,
// so callers don't need a separate `lattix` dep just to type-name what
// `anno::graph` produces. A breaking change in `lattix` then becomes a
// breaking change in `anno`'s `graph` feature, which is correct (and caught
// by `cargo-semver-checks`).
pub use lattix::{GraphDocument, GraphExportFormat, KnowledgeGraph};

/// Group signals by their qualified KB entry, document identity, track, or occurrence.
/// Within one export, a KB accession used by only one named catalog keeps its
/// spelling. The `kb_name` and `kb_id` properties retain the full external key
/// for consumers combining exports. Catalogs sharing an accession receive distinct IDs. Document-local
/// references include the document ID. Raw KB accessions take precedence if
/// they resemble generated keys. Grounded documents contain no extraction-time relations.
#[must_use]
pub fn grounded_to_graph_document(doc: &crate::GroundedDocument) -> GraphDocument {
    let mut catalogs = HashMap::new();
    let mut ambiguous = std::collections::HashSet::new();
    for track in doc.tracks().filter(|track| !track.signals().is_empty()) {
        if let Some((name, accession)) = doc
            .identity_for_track(track.id)
            .and_then(crate::Identity::qualified_kb_ref)
        {
            if catalogs
                .insert(accession, name)
                .is_some_and(|prior| prior != name)
            {
                ambiguous.insert(accession);
            }
        }
    }

    // Each generated key starts with its kind or a JSON array. A leading '~'
    // therefore distinguishes escaped keys without creating another collision.
    let generated_id = |mut candidate: String| {
        while catalogs.contains_key(candidate.as_str()) {
            candidate.insert(0, '~');
        }
        candidate
    };

    let mut graph = GraphDocument::new();
    let mut node_indices = HashMap::new();
    let document = crate::export::iri_component(doc.id());
    for signal in doc.signals() {
        let identity = doc.identity_for_signal(signal.id);
        let kb_ref = identity.and_then(crate::Identity::qualified_kb_ref);
        let node_id = if let Some(identity) = identity {
            match kb_ref {
                Some((name, accession)) if ambiguous.contains(accession) => {
                    generated_id(serde_json::json!([name, accession]).to_string())
                }
                Some((_, accession)) => accession.to_owned(),
                None => generated_id(format!("identity/{document}/{}", identity.id)),
            }
        } else if let Some(track) = doc.track_for_signal(signal.id) {
            generated_id(format!("track/{document}/{}", track.id))
        } else {
            generated_id(format!("signal/{document}/{}", signal.id))
        };
        add_mention(
            &mut graph,
            &mut node_indices,
            &node_id,
            signal.label(),
            &signal.surface,
            signal.text_offsets().map_or(0, |(start, _)| start),
            kb_ref,
        );
    }
    graph
}

fn add_mention(
    graph: &mut GraphDocument,
    node_indices: &mut HashMap<String, usize>,
    node_id: &str,
    label: &str,
    surface: &str,
    start: usize,
    kb_ref: Option<(&str, &str)>,
) {
    if let Some(&index) = node_indices.get(node_id) {
        let node = &mut graph.nodes[index];
        let count = node.properties["mentions_count"]
            .as_u64()
            .expect("node initialized with count");
        node.properties
            .insert("mentions_count".into(), (count + 1).into());
    } else {
        node_indices.insert(node_id.into(), graph.nodes.len());
        let mut node = GraphNode::new(node_id, label, surface)
            .with_mentions_count(1)
            .with_first_seen(start);
        if let Some((catalog, accession)) = kb_ref {
            node = node
                .with_property("kb_name", catalog)
                .with_property("kb_id", accession);
        }
        graph.nodes.push(node);
    }
}

/// Convert a single document's flat entities and relations into a graph.
/// Explicit KB or canonical IDs group mentions. Each unassigned mention has
/// its own node. The flat format lacks a document ID and KB namespace, so
/// these node IDs are local to this export. Ambiguous endpoints are omitted.
#[must_use]
pub fn entities_to_graph_document(
    entities: &[crate::Entity],
    relations: &[crate::Relation],
) -> GraphDocument {
    let mut graph = GraphDocument::new();
    let mut node_indices = HashMap::new();
    let mut entity_nodes = Vec::with_capacity(entities.len());
    for (index, entity) in entities.iter().enumerate() {
        let node_id = if let Some(accession) = &entity.kb_id {
            format!("flat/kb/{}", crate::export::iri_component(accession))
        } else if let Some(canonical_id) = entity.canonical_id {
            format!("flat/track/{canonical_id}")
        } else {
            format!("flat/mention/{index}")
        };
        add_mention(
            &mut graph,
            &mut node_indices,
            &node_id,
            entity.entity_type.as_label(),
            &entity.text,
            entity.start(),
            None,
        );
        entity_nodes.push(node_id);
    }

    let mut edge_indices: HashMap<_, usize> = HashMap::new();
    for (relation, head, tail) in crate::export::resolved_relations(entities, relations) {
        let source = &entity_nodes[head];
        let target = &entity_nodes[tail];
        let key = (source, target, &relation.relation_type);
        if let Some(&index) = edge_indices.get(&key) {
            let existing = &mut graph.edges[index];
            existing.confidence = existing.confidence.max(relation.confidence.value());
        } else {
            edge_indices.insert(key, graph.edges.len());
            graph.edges.push(
                GraphEdge::new(source.as_str(), target.as_str(), &relation.relation_type)
                    .with_confidence(relation.confidence.value()),
            );
        }
    }
    graph
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
        assert_eq!(nodes[2].properties["kb_name"], "alpha");
        assert_eq!(nodes[2].properties["kb_id"], "unique");
        assert_eq!(nodes[1].properties["kb_name"], "beta");
        assert_eq!(nodes[1].properties["kb_id"], "42");
        assert_eq!(nodes[3].id, r#"["alpha","42"]"#);
        assert_ne!(nodes[0].id, nodes[3].id);
    }

    #[test]
    fn grounded_graph_scopes_kb_accessions_and_keeps_distinct_unassigned_signals() {
        let mut doc = crate::GroundedDocument::new("d/1", "Ada Ada Ada Ada Ada Ada");
        let mut signals = Vec::new();
        for occurrence in 0..6 {
            signals.push(
                doc.add_signal_from_text_nth("Ada", "PER", 0.9, occurrence)
                    .unwrap(),
            );
        }
        for (signal, (kb, accession)) in
            signals
                .iter()
                .zip([("A/B", "42"), ("A", "B/42"), ("A/B", "42"), ("B", "42")])
        {
            let id = doc.add_identity(crate::Identity::from_kb(0, "Ada", kb, accession));
            let track = doc.create_track_from_signals("Ada", &[*signal]).unwrap();
            doc.link_track_to_identity(track, id);
        }

        let graph = grounded_to_graph_document(&doc);
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.nodes[0].id, r#"["A/B","42"]"#);
        assert_eq!(graph.nodes[0].properties["mentions_count"], 2);
        assert_eq!(graph.nodes[1].id, "B/42");
        assert_eq!(graph.nodes[2].id, r#"["B","42"]"#);
        assert_eq!(graph.nodes[3].id, "signal/d%2F1/4");
        assert_eq!(graph.nodes[4].id, "signal/d%2F1/5");
    }

    #[test]
    fn grounded_graph_reserves_raw_accessions_before_allocating_occurrence_ids() {
        let mut doc = crate::GroundedDocument::new("d", "X A B");
        doc.add_signal_from_text("X", "PER", 1.0).unwrap();
        for (surface, accession) in [("A", "signal/d/0"), ("B", "~signal/d/0")] {
            let s = doc.add_signal_from_text(surface, "PER", 1.0).unwrap();
            let t = doc.create_track_from_signals(surface, &[s]).unwrap();
            let id = doc.add_identity(crate::Identity::from_kb(0, surface, "catalog", accession));
            doc.link_track_to_identity(t, id);
        }
        let graph = grounded_to_graph_document(&doc);
        assert_eq!(graph.nodes[0].id, "~~signal/d/0");
        assert_eq!(graph.nodes[1].id, "signal/d/0");
        assert_eq!(graph.nodes[2].id, "~signal/d/0");
    }

    #[test]
    fn grounded_graph_groups_distinct_tracks_sharing_an_unlinked_identity() {
        let mut doc = crate::GroundedDocument::new("d", "Ada she");
        let identity = doc.add_identity(crate::Identity::new(0, "Ada"));
        for surface in ["Ada", "she"] {
            let signal = doc.add_signal_from_text(surface, "PER", 0.9).unwrap();
            let track = doc.create_track_from_signals(surface, &[signal]).unwrap();
            doc.link_track_to_identity(track, identity);
        }
        let graph = grounded_to_graph_document(&doc);
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].id, "identity/d/0");
        assert_eq!(graph.nodes[0].properties["mentions_count"], 2);
    }

    #[test]
    fn grounded_graph_keeps_unspecified_kb_namespaces_document_local() {
        let mut entity = ent("Ada", 0, 3, EntityType::Person);
        entity.link_to_kb("42");
        for (doc_id, namespace) in [("one", Some("unknown")), ("two", None), ("three", Some(""))] {
            let mut doc = crate::GroundedDocument::from_entities(
                doc_id,
                "Ada",
                std::slice::from_ref(&entity),
            );
            let identity = doc.identities_map_mut().values_mut().next().unwrap();
            identity.kb_name = namespace.map(str::to_owned);
            let graph = grounded_to_graph_document(&doc);
            assert_eq!(graph.nodes[0].id, format!("identity/{doc_id}/0"));
        }
    }

    #[test]
    fn flat_graph_relations_bind_by_occurrence_before_grouping() {
        let first = ent("Ada", 0, 3, EntityType::Person);
        let later = ent("Ada", 10, 13, EntityType::Person);
        let mut kb = ent("Acme", 20, 24, EntityType::Organization);
        kb.link_to_kb("coref_0");
        let track = ent("Bob", 30, 33, EntityType::Person).with_canonical_id(0);
        let outside = ent("Ada", 40, 43, EntityType::Person);
        let relations = [
            Relation::new(later.clone(), kb.clone(), "founded", 0.7),
            Relation::new(later.clone(), kb.clone(), "founded", 0.9),
            Relation::new(outside, kb.clone(), "founded", 0.99),
        ];
        let graph = entities_to_graph_document(&[first, later, kb, track], &relations);
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].source, graph.nodes[1].id);
        assert_eq!(graph.edges[0].target, graph.nodes[2].id);
        assert_eq!(graph.edges[0].confidence, 0.9);
    }

    #[test]
    fn flat_graph_uses_endpoint_kb_evidence_and_skips_ambiguous_clones() {
        let mention = ent("Ada", 0, 3, EntityType::Person);
        let mut a = mention.clone();
        a.link_to_kb("A");
        let mut b = mention.clone();
        b.link_to_kb("B");
        let tail = ent("Bob", 10, 13, EntityType::Person);
        let relations = [
            Relation::new(b.clone(), tail.clone(), "meets", 0.9),
            Relation::new(mention, tail.clone(), "ambiguous", 0.9),
        ];
        let graph = entities_to_graph_document(&[a, b, tail], &relations);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].source, graph.nodes[1].id);
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
