use anno::{export, Entity, EntityType, Relation};
use std::collections::BTreeSet;

#[test]
fn rdf_writers_preserve_distinct_relation_labels() {
    let base = "https://example.org/kb/";
    let entities = [
        Entity::new("Ada", EntityType::Person, 0, 3, 0.9),
        Entity::new("May", EntityType::Person, 8, 11, 0.9),
    ];
    let labels = [
        ("WORKS_AT", "WORKS_AT"),
        ("part-of", "part-of"),
        ("affilié_à", "affilié_à"),
        ("part_of", "part_of"),
        ("part:of", "part%3Aof"),
        ("part%3Aof", "part%253Aof"),
        ("part/of", "part%2Fof"),
        ("part of", "part%20of"),
        ("🔗", "%F0%9F%94%97"),
    ];
    let relations: Vec<_> = labels
        .iter()
        .map(|(label, _)| Relation::new(entities[0].clone(), entities[1].clone(), *label, 0.8))
        .collect();
    let prefix = format!("{base}rel/");
    let expected: BTreeSet<_> = labels
        .iter()
        .map(|(_, iri)| format!("{prefix}{iri}"))
        .collect();
    let nt = export::to_ntriples(&entities, &relations, "doc", base);
    let predicates: BTreeSet<_> = nt
        .lines()
        .filter_map(|line| line.split_whitespace().nth(1))
        .map(|iri| iri.trim_start_matches('<').trim_end_matches('>'))
        .filter(|iri| iri.starts_with(&prefix))
        .map(str::to_owned)
        .collect();
    assert_eq!(predicates, expected, "plain N-Triples predicates");

    let ld: serde_json::Value = serde_json::from_str(&export::to_jsonld(
        &entities, &relations, "doc", false, base,
    ))
    .unwrap();
    let kinds: BTreeSet<_> = ld["@graph"][0]["anno:relations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|rel| rel["@type"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(kinds, expected, "JSON-LD relation types");

    #[cfg(feature = "graph")]
    {
        let graph =
            anno::graph::entities_to_knowledge_graph(&entities, &relations, "urn:doc", base);
        let predicates: BTreeSet<_> = graph
            .triples()
            .map(|t| t.predicate().as_str())
            .filter(|iri| iri.starts_with(&prefix))
            .map(str::to_owned)
            .collect();
        assert_eq!(predicates, expected, "lattix predicates");
    }
}
