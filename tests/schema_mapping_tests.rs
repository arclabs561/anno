//! Comprehensive tests for schema mapping functionality.

use anno::schema::{map_to_canonical, CanonicalType, CoarseType, DatasetSchema, SchemaMapper};
use anno::{EntityCategory, EntityType};

#[test]
fn test_canonical_type_name() {
    assert_eq!(CanonicalType::Person.name(), "PERSON");
    assert_eq!(CanonicalType::Organization.name(), "ORG");
    assert_eq!(CanonicalType::GeopoliticalEntity.name(), "GPE");
    assert_eq!(CanonicalType::Date.name(), "DATE");
    assert_eq!(CanonicalType::Money.name(), "MONEY");
    assert_eq!(CanonicalType::Misc.name(), "MISC");
}

#[test]
fn test_canonical_type_category() {
    assert_eq!(CanonicalType::Person.category(), EntityCategory::Agent);
    assert_eq!(CanonicalType::Group.category(), EntityCategory::Agent);
    assert_eq!(
        CanonicalType::Organization.category(),
        EntityCategory::Organization
    );
    assert_eq!(
        CanonicalType::GeopoliticalEntity.category(),
        EntityCategory::Place
    );
    assert_eq!(CanonicalType::Date.category(), EntityCategory::Temporal);
    assert_eq!(CanonicalType::Money.category(), EntityCategory::Numeric);
}

#[test]
fn test_canonical_type_to_entity_type() {
    assert_eq!(CanonicalType::Person.to_entity_type(), EntityType::Person);
    assert_eq!(
        CanonicalType::Organization.to_entity_type(),
        EntityType::Organization
    );
    assert_eq!(CanonicalType::Date.to_entity_type(), EntityType::Date);
    assert_eq!(CanonicalType::Money.to_entity_type(), EntityType::Money);
}

#[test]
fn test_dataset_schema_labels_conll2003() {
    let labels = DatasetSchema::CoNLL2003.labels();
    assert!(labels.contains(&"PER"));
    assert!(labels.contains(&"LOC"));
    assert!(labels.contains(&"ORG"));
    assert!(labels.contains(&"MISC"));
    assert_eq!(labels.len(), 4);
}

#[test]
fn test_dataset_schema_labels_ontonotes() {
    let labels = DatasetSchema::OntoNotes.labels();
    assert!(labels.contains(&"PERSON"));
    assert!(labels.contains(&"NORP"));
    assert!(labels.contains(&"GPE"));
    assert!(labels.contains(&"FAC"));
    assert!(labels.contains(&"ORG"));
    assert!(labels.contains(&"DATE"));
    assert!(labels.contains(&"MONEY"));
    assert_eq!(labels.len(), 18);
}

#[test]
fn test_dataset_schema_labels_multinerd() {
    let labels = DatasetSchema::MultiNERD.labels();
    assert!(labels.contains(&"PER"));
    assert!(labels.contains(&"LOC"));
    assert!(labels.contains(&"ORG"));
    assert!(labels.contains(&"ANIM"));
    assert!(labels.contains(&"DIS"));
    assert_eq!(labels.len(), 15);
}

#[test]
fn test_schema_mapper_for_dataset_conll2003() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);
    assert_eq!(mapper.source_schema, DatasetSchema::CoNLL2003);

    assert_eq!(mapper.to_canonical("PER"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("LOC"), CanonicalType::Location);
    assert_eq!(mapper.to_canonical("ORG"), CanonicalType::Organization);
    assert_eq!(mapper.to_canonical("MISC"), CanonicalType::Misc);
}

#[test]
fn test_schema_mapper_for_dataset_ontonotes() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);

    assert_eq!(mapper.to_canonical("PERSON"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("NORP"), CanonicalType::Group); // Important: NORP is Group, not Org!
    assert_eq!(
        mapper.to_canonical("GPE"),
        CanonicalType::GeopoliticalEntity
    );
    assert_eq!(mapper.to_canonical("FAC"), CanonicalType::Facility);
    assert_eq!(mapper.to_canonical("LOC"), CanonicalType::NaturalLocation);
    assert_eq!(mapper.to_canonical("ORG"), CanonicalType::Organization);
}

#[test]
fn test_schema_mapper_case_insensitive() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);

    assert_eq!(mapper.to_canonical("PER"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("per"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("Per"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("PeR"), CanonicalType::Person);
}

#[test]
fn test_schema_mapper_unknown_label() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);

    // Unknown label should map to Misc
    assert_eq!(mapper.to_canonical("UNKNOWN"), CanonicalType::Misc);
    assert_eq!(mapper.to_canonical("XYZ"), CanonicalType::Misc);
}

#[test]
fn test_schema_mapper_information_loss() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);

    // NORP should have information loss
    let loss = mapper.information_loss("NORP");
    assert!(loss.is_some());
    let loss = loss.unwrap();
    assert_eq!(loss.original, "NORP");
    assert_eq!(loss.canonical, CanonicalType::Group);
    assert!(!loss.lost_semantics.is_empty());

    // FAC should have information loss
    let loss = mapper.information_loss("FAC");
    assert!(loss.is_some());

    // PERSON should NOT have information loss (direct mapping)
    let loss = mapper.information_loss("PERSON");
    assert!(loss.is_none());
}

#[test]
fn test_schema_mapper_all_losses() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);

    let losses: Vec<_> = mapper.all_losses().collect();
    assert!(!losses.is_empty());

    // Should include NORP, FAC, LOC
    let has_norp = losses.iter().any(|l| l.original == "NORP");
    assert!(has_norp);
}

#[test]
fn test_schema_mapper_to_entity_type() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);

    assert_eq!(mapper.to_entity_type("PER"), EntityType::Person);
    assert_eq!(mapper.to_entity_type("ORG"), EntityType::Organization);
    assert_eq!(mapper.to_entity_type("LOC"), EntityType::Location);
}

#[test]
fn test_schema_mapper_label_overlap_same_schema() {
    let mapper1 = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);
    let mapper2 = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);

    // Same schema should have 100% overlap
    let overlap = mapper1.label_overlap(&mapper2);
    assert!((overlap - 1.0).abs() < 0.001);
}

#[test]
fn test_schema_mapper_label_overlap_different_schemas() {
    let mapper1 = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);
    let mapper2 = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);

    // CoNLL and OntoNotes share Person, Org, Location
    let overlap = mapper1.label_overlap(&mapper2);
    assert!(overlap > 0.0);
    assert!(overlap < 1.0);
}

#[test]
fn test_schema_mapper_label_overlap_disjoint() {
    let mapper1 = SchemaMapper::for_dataset(DatasetSchema::BC5CDR);
    let mapper2 = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);

    // BC5CDR (Chemical, Disease) vs CoNLL (PER, LOC, ORG, MISC) - no overlap
    let overlap = mapper1.label_overlap(&mapper2);
    assert_eq!(overlap, 0.0);
}

#[test]
fn test_map_to_canonical_with_schema() {
    // With schema, should use precise mapping
    let entity_type = map_to_canonical("PER", Some(DatasetSchema::CoNLL2003));
    assert_eq!(entity_type, EntityType::Person);

    let entity_type = map_to_canonical("NORP", Some(DatasetSchema::OntoNotes));
    // Should map to custom type for Group
    assert_eq!(entity_type.category(), EntityCategory::Agent);
}

#[test]
fn test_map_to_canonical_without_schema() {
    // Without schema, should use heuristic
    let entity_type = map_to_canonical("PER", None);
    assert_eq!(entity_type, EntityType::Person);

    let entity_type = map_to_canonical("PERSON", None);
    assert_eq!(entity_type, EntityType::Person);
}

#[test]
fn test_map_to_canonical_strips_bio_prefixes() {
    // Should strip B-, I-, E-, S-, L-, U- prefixes
    assert_eq!(
        map_to_canonical("B-PER", Some(DatasetSchema::CoNLL2003)),
        EntityType::Person
    );
    assert_eq!(
        map_to_canonical("I-PER", Some(DatasetSchema::CoNLL2003)),
        EntityType::Person
    );
    assert_eq!(
        map_to_canonical("E-PER", Some(DatasetSchema::CoNLL2003)),
        EntityType::Person
    );
    assert_eq!(
        map_to_canonical("S-PER", Some(DatasetSchema::CoNLL2003)),
        EntityType::Person
    );
    assert_eq!(
        map_to_canonical("L-PER", Some(DatasetSchema::CoNLL2003)),
        EntityType::Person
    );
    assert_eq!(
        map_to_canonical("U-PER", Some(DatasetSchema::CoNLL2003)),
        EntityType::Person
    );
}

#[test]
fn test_coarse_type_from_canonical() {
    assert_eq!(
        CoarseType::from_canonical(CanonicalType::Person),
        CoarseType::Person
    );
    assert_eq!(
        CoarseType::from_canonical(CanonicalType::Organization),
        CoarseType::Organization
    );
    assert_eq!(
        CoarseType::from_canonical(CanonicalType::Location),
        CoarseType::Location
    );
    assert_eq!(
        CoarseType::from_canonical(CanonicalType::Date),
        CoarseType::DateTime
    );
    assert_eq!(
        CoarseType::from_canonical(CanonicalType::Money),
        CoarseType::Numeric
    );
    assert_eq!(
        CoarseType::from_canonical(CanonicalType::Time),
        CoarseType::DateTime
    );
}

#[test]
fn test_coarse_type_from_label() {
    // from_label uses OntoNotes schema mapper
    // OntoNotes uses "PERSON" not "PER", "ORG" not "ORGANIZATION", etc.
    assert_eq!(CoarseType::from_label("PERSON"), CoarseType::Person);
    assert_eq!(CoarseType::from_label("ORG"), CoarseType::Organization);
    assert_eq!(CoarseType::from_label("LOC"), CoarseType::Location);
    assert_eq!(CoarseType::from_label("GPE"), CoarseType::Location); // GeopoliticalEntity -> Location
    assert_eq!(CoarseType::from_label("DATE"), CoarseType::DateTime);
    assert_eq!(CoarseType::from_label("TIME"), CoarseType::DateTime);
    assert_eq!(CoarseType::from_label("MONEY"), CoarseType::Numeric);
    // Unknown labels should map to Other (via Misc canonical type)
    assert_eq!(CoarseType::from_label("UNKNOWN"), CoarseType::Other);
}

#[test]
fn test_schema_mapper_multinerd_mappings() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::MultiNERD);

    assert_eq!(mapper.to_canonical("PER"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("ANIM"), CanonicalType::Animal);
    assert_eq!(mapper.to_canonical("DIS"), CanonicalType::Disease);
    assert_eq!(mapper.to_canonical("FOOD"), CanonicalType::Food);
    assert_eq!(mapper.to_canonical("PLANT"), CanonicalType::Plant);
}

#[test]
fn test_schema_mapper_wnut17_mappings() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::WNUT17);

    assert_eq!(mapper.to_canonical("person"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("location"), CanonicalType::Location);
    assert_eq!(
        mapper.to_canonical("corporation"),
        CanonicalType::Organization
    );
    assert_eq!(mapper.to_canonical("product"), CanonicalType::Product);
    assert_eq!(
        mapper.to_canonical("creative-work"),
        CanonicalType::CreativeWork
    );
    assert_eq!(mapper.to_canonical("group"), CanonicalType::Group);
}

#[test]
fn test_schema_mapper_bc5cdr_mappings() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::BC5CDR);

    assert_eq!(mapper.to_canonical("Chemical"), CanonicalType::Chemical);
    assert_eq!(mapper.to_canonical("Disease"), CanonicalType::Disease);
}

#[test]
fn test_schema_mapper_mit_movie_mappings() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::MITMovie);

    // Actor, Director, Character all map to Person (with loss)
    assert_eq!(mapper.to_canonical("Actor"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("Director"), CanonicalType::Person);
    assert_eq!(mapper.to_canonical("Character"), CanonicalType::Person);

    // Should have information loss
    assert!(mapper.information_loss("Actor").is_some());
    assert!(mapper.information_loss("Director").is_some());

    assert_eq!(mapper.to_canonical("Title"), CanonicalType::CreativeWork);
    assert_eq!(mapper.to_canonical("Year"), CanonicalType::Date);
}

#[test]
fn test_schema_mapper_round_trip() {
    // Test that mapping is consistent
    let mapper = SchemaMapper::for_dataset(DatasetSchema::CoNLL2003);

    let canonical = mapper.to_canonical("PER");
    let entity_type = canonical.to_entity_type();
    assert_eq!(entity_type, EntityType::Person);

    // Should be able to map back (though not perfect due to information loss)
    let canonical2 = mapper.to_canonical("PER");
    assert_eq!(canonical, canonical2);
}

#[test]
fn test_information_loss_structure() {
    let mapper = SchemaMapper::for_dataset(DatasetSchema::OntoNotes);
    let loss = mapper.information_loss("NORP").unwrap();

    assert_eq!(loss.original, "NORP");
    assert_eq!(loss.canonical, CanonicalType::Group);
    assert!(!loss.lost_semantics.is_empty());
    assert!(
        loss.lost_semantics.contains("Nationalities")
            || loss.lost_semantics.contains("religions")
            || loss.lost_semantics.contains("politics")
    );
}

#[test]
fn test_schema_mapper_all_datasets() {
    // Test that all dataset schemas can be created
    let schemas = vec![
        DatasetSchema::CoNLL2003,
        DatasetSchema::OntoNotes,
        DatasetSchema::MultiNERD,
        DatasetSchema::FewNERD,
        DatasetSchema::CrossNER,
        DatasetSchema::BC5CDR,
        DatasetSchema::NCBIDisease,
        DatasetSchema::MITMovie,
        DatasetSchema::MITRestaurant,
        DatasetSchema::WNUT17,
    ];

    for schema in schemas {
        let mapper = SchemaMapper::for_dataset(schema);
        assert_eq!(mapper.source_schema, schema);
        // Should have at least one mapping
        assert!(!schema.labels().is_empty());
    }
}
