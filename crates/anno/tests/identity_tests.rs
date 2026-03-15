//! Tests for Identity tracking and management.

use anno::{EntityType, Identity, IdentityId, IdentitySource, TypeLabel};

#[test]
fn test_identity_creation() {
    let id = IdentityId::new(1);
    let identity = Identity::new(id, "John Smith");

    assert_eq!(identity.id, id);
    assert_eq!(identity.canonical_name, "John Smith");
    assert!(identity.entity_type.is_none());
    assert!(identity.kb_id.is_none());
    assert_eq!(identity.confidence, 1.0);
}

#[test]
fn test_identity_from_knowledge_base() {
    let id = IdentityId::new(7186);
    let identity = Identity::from_kb(id, "Marie Curie", "wikidata", "Q7186");

    assert_eq!(identity.canonical_name, "Marie Curie");
    assert_eq!(identity.kb_name, Some("wikidata".to_string()));
    assert_eq!(identity.kb_id, Some("Q7186".to_string()));

    match &identity.source {
        Some(IdentitySource::KnowledgeBase { kb_name, kb_id }) => {
            assert_eq!(kb_name, "wikidata");
            assert_eq!(kb_id, "Q7186");
        }
        _ => panic!("Expected KnowledgeBase source"),
    }
}

#[test]
fn test_identity_aliases() {
    let id = IdentityId::new(100);
    let mut identity = Identity::new(id, "Robert Smith");

    identity.add_alias("Bob Smith");
    identity.add_alias("R. Smith");
    identity.add_alias("Bobby");

    assert_eq!(identity.aliases.len(), 3);
    assert!(identity.aliases.contains(&"Bob Smith".to_string()));
    assert!(identity.aliases.contains(&"Bobby".to_string()));
}

#[test]
fn test_identity_with_embedding() {
    let id = IdentityId::new(200);
    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    let identity = Identity::new(id, "Test Entity").with_embedding(embedding.clone());

    assert_eq!(identity.embedding, Some(embedding));
}

#[test]
fn test_identity_with_entity_type() {
    let id = IdentityId::new(300);
    let mut identity = Identity::new(id, "Apple Inc.");
    identity.entity_type = Some(TypeLabel::from("Organization"));

    assert_eq!(identity.entity_type, Some(TypeLabel::from("Organization")));
}

#[test]
fn test_identity_confidence() {
    let id = IdentityId::new(400);
    let mut identity = Identity::new(id, "Uncertain Entity");
    identity.set_confidence(0.75);

    assert!((identity.confidence.value() - 0.75).abs() < f64::EPSILON);
}

#[test]
fn test_identity_equality() {
    let id1 = IdentityId::new(500);
    let id2 = IdentityId::new(500);

    let identity1 = Identity::new(id1, "Test");
    let identity2 = Identity::new(id2, "Test");

    assert_eq!(identity1, identity2);
}

#[test]
fn test_identity_source_variants() {
    // KnowledgeBase source
    let kb_source = IdentitySource::KnowledgeBase {
        kb_name: "wikidata".to_string(),
        kb_id: "Q42".to_string(),
    };
    assert!(matches!(kb_source, IdentitySource::KnowledgeBase { .. }));

    // CrossDocCoref source (from clustering tracks)
    let coref_source = IdentitySource::CrossDocCoref {
        track_refs: Vec::new(),
    };
    assert!(matches!(coref_source, IdentitySource::CrossDocCoref { .. }));

    // Hybrid source (both clustered and KB-linked)
    let hybrid_source = IdentitySource::Hybrid {
        track_refs: Vec::new(),
        kb_name: "wikidata".to_string(),
        kb_id: "Q937".to_string(),
    };
    assert!(matches!(hybrid_source, IdentitySource::Hybrid { .. }));
}

#[test]
fn test_identity_serialization() {
    let id = IdentityId::new(600);
    let mut identity = Identity::from_kb(id, "Test Entity", "test_kb", "T123");
    identity.add_alias("Alias 1");
    identity.description = Some("A test entity".to_string());

    // Serialize to JSON
    let json = serde_json::to_string(&identity).expect("Serialization failed");

    // Deserialize back
    let deserialized: Identity = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(deserialized.canonical_name, "Test Entity");
    assert_eq!(deserialized.kb_id, Some("T123".to_string()));
    assert_eq!(deserialized.aliases, vec!["Alias 1"]);
}

#[test]
fn test_identity_deserialize_ignores_legacy_box_embedding_field() {
    // Older serialized identities may contain a `box_embedding` field.
    // The stable `Identity` type intentionally does not carry box embeddings;
    // those live in `provisional::ProvisionalIdentity`.
    let json = serde_json::json!({
        "id": 7186,
        "canonical_name": "Marie Curie",
        "entity_type": "Person",
        "kb_id": "Q7186",
        "kb_name": "wikidata",
        "description": null,
        "embedding": null,
        "box_embedding": { "min": [0.0, 0.0], "max": [1.0, 1.0] },
        "aliases": [],
        "confidence": 1.0,
        "source": null
    });

    let identity: Identity = serde_json::from_value(json).expect("legacy JSON should deserialize");
    assert_eq!(identity.id, IdentityId::new(7186));
    assert_eq!(identity.canonical_name, "Marie Curie");
    assert_eq!(identity.kb_id.as_deref(), Some("Q7186"));
    assert!(
        matches!(
            identity.entity_type.as_ref(),
            Some(TypeLabel::Core(EntityType::Person))
        ),
        "entity_type should parse as the canonical core PERSON type"
    );
    // `TypeLabel` renders core entity types in canonical CoNLL/OntoNotes labels.
    assert_eq!(
        identity.entity_type.as_ref().map(|t| t.as_str()),
        Some("PER")
    );
}

#[test]
fn test_identity_with_description() {
    let id = IdentityId::new(700);
    let mut identity = Identity::new(id, "Albert Einstein");
    identity.description = Some("German-born theoretical physicist".to_string());
    identity.entity_type = Some(TypeLabel::from("Person"));

    assert!(identity.description.is_some());
    assert!(identity
        .description
        .as_ref()
        .expect("description set above")
        .contains("physicist"));
}

#[test]
fn test_multiple_identities_distinct() {
    let id1 = IdentityId::new(937); // Einstein's Wikidata Q number
    let id2 = IdentityId::new(7186); // Curie's Wikidata Q number

    let einstein = Identity::from_kb(id1, "Albert Einstein", "wikidata", "Q937");
    let curie = Identity::from_kb(id2, "Marie Curie", "wikidata", "Q7186");

    assert_ne!(einstein.id, curie.id);
    assert_ne!(einstein.canonical_name, curie.canonical_name);
    assert_ne!(einstein.kb_id, curie.kb_id);
}

#[test]
fn test_identity_default_confidence() {
    let identity = Identity::new(1000, "Default Confidence Entity");
    assert_eq!(identity.confidence, 1.0);
}

#[test]
fn test_identity_empty_aliases() {
    let identity = Identity::new(1001, "No Aliases");
    assert!(identity.aliases.is_empty());
}

#[test]
fn test_identity_no_embedding_by_default() {
    let identity = Identity::new(1002, "No Embedding");
    assert!(identity.embedding.is_none());
}
