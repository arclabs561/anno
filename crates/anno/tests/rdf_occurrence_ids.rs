use anno::{export, Entity, EntityType};
use serde_json::Value;

const BASE: &str = "https://example.org/archive";

fn json_nodes(entities: &[Entity], label: &str) -> Vec<Value> {
    let output = export::to_jsonld(entities, &[], label, true, BASE);
    serde_json::from_str::<Value>(&output).unwrap()["@graph"]
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn occurrences_are_scoped_to_the_document_and_independent_of_prediction_order() {
    let first = Entity::new("Ada", EntityType::Person, 0, 3, 0.9);
    let next = Entity::new("Ada", EntityType::Person, 12, 15, 0.8);
    let old_order = json_nodes(&[first.clone(), next.clone()], "a/b");
    let new_order = json_nodes(&[next, first.clone()], "a/b");
    let other = json_nodes(&[first], "a_b");
    assert_ne!(old_order[0]["@id"], old_order[1]["@id"]);
    assert_eq!(old_order[0]["@id"], new_order[1]["@id"]);
    assert_eq!(old_order[1]["@id"], new_order[0]["@id"]);
    assert_ne!(old_order[0]["@id"], other[0]["@id"]);
    assert_ne!(
        old_order[0]["prov:hadPrimarySource"],
        other[0]["prov:hadPrimarySource"]
    );
}

#[cfg(feature = "graph")]
#[test]
fn merging_graphs_preserves_both_occurrences_and_their_sources() {
    let entities = [Entity::new("Ada", EntityType::Person, 0, 3, 0.9)];
    let source_a = "https://example.org/a?revision=1#page";
    let source_b = "https://example.org/a?revision=2#page";
    let mut a = anno::graph::entities_to_knowledge_graph(&entities, &[], source_a, BASE);
    let b = anno::graph::entities_to_knowledge_graph(&entities, &[], source_b, BASE);
    a.merge(&b);
    let mentions: std::collections::HashSet<String> = a
        .triples()
        .map(|t| t.to_ntriples())
        .filter(|t| t.contains("prov#hadPrimarySource"))
        .map(|t| t.split_whitespace().next().unwrap().to_owned())
        .collect();
    assert_eq!(mentions.len(), 2, "each source owns a different mention");
}

#[test]
fn separate_visual_and_discontinuous_anchors_remain_distinct() {
    use anno::{DiscontinuousSpan, Span};

    let box_on = |page| {
        Entity::from_visual(
            "Ada",
            EntityType::Person,
            Span::bbox_on_page(0.0, 0.0, 0.5, 0.5, page),
            0.9,
        )
    };
    let part = |ranges| {
        let mut entity = Entity::new("New York", EntityType::Location, 0, 12, 0.9);
        entity.set_discontinuous_span(DiscontinuousSpan::new(ranges));
        entity
    };
    let nodes = json_nodes(
        &[
            box_on(1),
            box_on(2),
            part(vec![0..3, 8..12]),
            part(vec![0..3, 6..12]),
        ],
        "doc",
    );
    let ids: std::collections::HashSet<_> =
        nodes.iter().map(|n| n["@id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 4);
}

#[test]
fn relations_attach_once_and_accept_equivalent_contiguous_spans() {
    use anno::{DiscontinuousSpan, Relation};

    let first = Entity::new("Ada", EntityType::Person, 0, 3, 0.9).with_canonical_id(1u64);
    let other = first.clone().with_canonical_id(2u64);
    let mut equivalent = first.clone();
    equivalent.set_discontinuous_span(DiscontinuousSpan::contiguous(0, 3));
    let city = Entity::new("Paris", EntityType::Location, 10, 15, 0.99);
    for (listed, endpoint) in [(first.clone(), equivalent.clone()), (equivalent, first)] {
        let entities = [listed, other.clone(), city.clone()];
        let relation = [Relation::new(endpoint, city.clone(), "visits", 0.9)];
        let json: Value = serde_json::from_str(&export::to_jsonld_for_document(
            &entities,
            &relation,
            "urn:revision:1",
            true,
            BASE,
        ))
        .unwrap();
        assert_eq!(json["@graph"][0]["@id"], json["@graph"][1]["@id"]);
        assert_eq!(json["@context"]["target"], "anno:target");
        let target = &json["@graph"][2]["@id"];
        assert_eq!(
            &json["@graph"][0]["anno:relations"][0]["target"]["@id"],
            target
        );
        assert!(json["@graph"][1].get("anno:relations").is_none());

        let nt = export::to_ntriples_for_document(&entities, &relation, "urn:revision:1", BASE);
        assert_eq!(
            nt.lines()
                .filter(|line| line.contains("/rel/visits>"))
                .count(),
            1
        );
        #[cfg(feature = "graph")]
        {
            let kg = anno::graph::entities_to_knowledge_graph(
                &entities,
                &relation,
                "urn:revision:1",
                BASE,
            );
            assert_eq!(
                kg.triples()
                    .filter(|t| t.to_ntriples().contains("/rel/visits>"))
                    .count(),
                1
            );
        }
    }
}
