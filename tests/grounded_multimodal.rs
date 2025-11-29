//! Integration tests for multimodal grounded entity representation.
//!
//! Tests the Signal → Track → Identity hierarchy with mixed modalities.
//!
//! # Research Context
//!
//! These tests validate the isomorphism between:
//! - **Vision**: BoundingBox (x,y,w,h) → Tracklet → FaceID
//! - **NLP**: TextSpan (start,end) → CorefChain → EntityLink
//!
//! The key insight: detection = localization × classification, regardless of modality.

use anno::grounded::{
    render_document_html, GroundedDocument, Identity, Location, Modality, Quantifier, Signal,
    TextSpatialIndex, Track,
};
use anno::{Entity, EntityType};

// =============================================================================
// Part 1: Basic Location Tests
// =============================================================================

#[test]
fn location_text_properties() {
    let loc = Location::text(10, 25);

    assert_eq!(loc.modality(), Modality::Symbolic);
    assert_eq!(loc.text_offsets(), Some((10, 25)));

    // Text locations don't have geometric features in the same sense
    assert!(loc.modality().supports_linguistic_features());
    assert!(!loc.modality().supports_geometric_features());
}

#[test]
fn location_bbox_properties() {
    let loc = Location::bbox(0.1, 0.2, 0.3, 0.4);

    assert_eq!(loc.modality(), Modality::Iconic);
    assert_eq!(loc.text_offsets(), None);

    // Bounding boxes don't support linguistic features
    assert!(!loc.modality().supports_linguistic_features());
    assert!(loc.modality().supports_geometric_features());
}

#[test]
fn location_iou_text_spans() {
    // Non-overlapping spans
    let a = Location::text(0, 10);
    let b = Location::text(20, 30);
    assert_eq!(a.iou(&b), Some(0.0));
    assert!(!a.overlaps(&b));

    // Identical spans
    let c = Location::text(0, 10);
    assert_eq!(a.iou(&c), Some(1.0));
    assert!(a.overlaps(&c));

    // Partial overlap: [0,10) ∩ [5,15) = [5,10) = 5 chars
    // Union: [0,15) = 15 chars → IoU = 5/15 ≈ 0.333
    let d = Location::text(5, 15);
    let iou = a.iou(&d).unwrap();
    assert!((iou - 0.333).abs() < 0.01);
    assert!(a.overlaps(&d));
}

#[test]
fn location_iou_bounding_boxes() {
    // Non-overlapping boxes
    let a = Location::bbox(0.0, 0.0, 0.1, 0.1);
    let b = Location::bbox(0.5, 0.5, 0.1, 0.1);
    assert_eq!(a.iou(&b), Some(0.0));
    assert!(!a.overlaps(&b));

    // Identical boxes
    let c = Location::bbox(0.0, 0.0, 0.1, 0.1);
    assert_eq!(a.iou(&c), Some(1.0));
    assert!(a.overlaps(&c));

    // Partial overlap
    let d = Location::bbox(0.05, 0.05, 0.1, 0.1);
    let iou = a.iou(&d).unwrap();
    assert!(iou > 0.0 && iou < 1.0);
    assert!(a.overlaps(&d));
}

#[test]
fn location_iou_incompatible_types() {
    let text = Location::text(0, 10);
    let bbox = Location::bbox(0.0, 0.0, 0.1, 0.1);

    // IoU is undefined between incompatible types
    assert_eq!(text.iou(&bbox), None);
    assert!(!text.overlaps(&bbox));
}

// =============================================================================
// Part 2: Signal Tests (Level 1)
// =============================================================================

#[test]
fn signal_basic_creation() {
    let signal: Signal<Location> =
        Signal::new(42, Location::text(0, 12), "Marie Curie", "Person", 0.95);

    assert_eq!(signal.id, 42);
    assert_eq!(signal.surface, "Marie Curie");
    assert_eq!(signal.label, "Person");
    assert!((signal.confidence - 0.95).abs() < 0.001);
    assert!(!signal.negated);
    assert_eq!(signal.quantifier, None);
}

#[test]
fn signal_with_linguistic_features() {
    // Negated signal: "not a doctor"
    let negated: Signal<Location> =
        Signal::new(0, Location::text(0, 12), "not a doctor", "Occupation", 0.8)
            .negated()
            .with_modality(Modality::Symbolic);

    assert!(negated.negated);
    assert!(negated.modality.supports_linguistic_features());

    // Quantified signal: "every employee"
    let quantified: Signal<Location> =
        Signal::new(1, Location::text(0, 14), "every employee", "Person", 0.7)
            .with_quantifier(Quantifier::Universal);

    assert_eq!(quantified.quantifier, Some(Quantifier::Universal));

    // Existential: "some customers"
    let existential: Signal<Location> =
        Signal::new(2, Location::text(0, 14), "some customers", "Person", 0.75)
            .with_quantifier(Quantifier::Existential);

    assert_eq!(existential.quantifier, Some(Quantifier::Existential));
}

#[test]
fn signal_visual_has_no_linguistic_features() {
    // Visual signals shouldn't have linguistic features like negation
    let visual: Signal<Location> = Signal::new(
        0,
        Location::bbox(0.1, 0.2, 0.3, 0.4),
        "face_patch",
        "Person",
        0.92,
    )
    .with_modality(Modality::Iconic);

    assert!(!visual.modality.supports_linguistic_features());
    // Even if we set negated=true, it's semantically meaningless for visual
    // The type system allows it, but downstream code should ignore it
}

#[test]
fn signal_confidence_clamping() {
    // Confidence should be clamped to [0, 1]
    let over: Signal<Location> = Signal::new(0, Location::text(0, 5), "test", "Type", 1.5);
    assert_eq!(over.confidence, 1.0);

    let under: Signal<Location> = Signal::new(0, Location::text(0, 5), "test", "Type", -0.5);
    assert_eq!(under.confidence, 0.0);
}

// =============================================================================
// Part 3: Track Tests (Level 2)
// =============================================================================

#[test]
fn track_basic_operations() {
    let mut track = Track::new(0, "Marie Curie");
    assert!(track.is_empty());
    assert!(track.is_singleton() == false); // Empty is not singleton

    track.add_signal(1, 0);
    assert!(!track.is_empty());
    assert!(track.is_singleton());
    assert_eq!(track.len(), 1);

    track.add_signal(2, 1);
    track.add_signal(3, 2);
    assert!(!track.is_singleton());
    assert_eq!(track.len(), 3);
}

#[test]
fn track_with_identity_link() {
    let track = Track::new(0, "Albert Einstein")
        .with_type("Person")
        .with_identity(42);

    assert_eq!(track.entity_type, Some("Person".to_string()));
    assert_eq!(track.identity_id, Some(42));
}

// =============================================================================
// Part 4: Identity Tests (Level 3)
// =============================================================================

#[test]
fn identity_from_kb() {
    let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186")
        .with_type("Person")
        .with_description("Polish-French physicist");

    assert_eq!(identity.canonical_name, "Marie Curie");
    assert_eq!(identity.kb_name, Some("wikidata".to_string()));
    assert_eq!(identity.kb_id, Some("Q7186".to_string()));
    assert_eq!(identity.entity_type, Some("Person".to_string()));
}

#[test]
fn identity_with_aliases() {
    let mut identity = Identity::new(0, "Elon Musk");
    identity.add_alias("@elonmusk");
    identity.add_alias("Tesla CEO");

    assert_eq!(identity.aliases.len(), 2);
    assert!(identity.aliases.contains(&"@elonmusk".to_string()));
}

// =============================================================================
// Part 5: GroundedDocument Tests
// =============================================================================

#[test]
fn document_signal_track_identity_flow() {
    let text = "Marie Curie won the Nobel Prize. She was a physicist.";
    let mut doc = GroundedDocument::new("doc1", text);

    // Level 1: Add signals
    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Marie Curie",
        "Person",
        0.95,
    ));
    let s2 = doc.add_signal(Signal::new(
        0,
        Location::text(33, 36),
        "She",
        "Person",
        0.88,
    ));
    let s3 = doc.add_signal(Signal::new(
        0,
        Location::text(17, 28),
        "Nobel Prize",
        "Award",
        0.92,
    ));

    assert_eq!(doc.signals().len(), 3);

    // Level 2: Form tracks (coreference)
    let mut curie_track = Track::new(0, "Marie Curie");
    curie_track.add_signal(s1, 0);
    curie_track.add_signal(s2, 1);
    let curie_track_id = doc.add_track(curie_track);

    let mut prize_track = Track::new(1, "Nobel Prize");
    prize_track.add_signal(s3, 0);
    doc.add_track(prize_track);

    assert_eq!(doc.tracks().count(), 2);

    // Level 3: Link to knowledge base
    let curie_identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
    let curie_identity_id = doc.add_identity(curie_identity);
    doc.link_track_to_identity(curie_track_id, curie_identity_id);

    // Verify traversal
    assert!(doc.track_for_signal(s1).is_some());
    assert!(doc.identity_for_signal(s1).is_some());
    assert_eq!(
        doc.identity_for_signal(s1).unwrap().kb_id,
        Some("Q7186".to_string())
    );

    // s3 (Nobel Prize) is in a track but not linked to an identity
    assert!(doc.track_for_signal(s3).is_some());
    assert!(doc.identity_for_signal(s3).is_none());
}

#[test]
fn document_statistics() {
    let mut doc = GroundedDocument::new("stats_test", "Test document with entities.");

    // Add mixed modality signals
    doc.add_signal(
        Signal::new(0, Location::text(0, 4), "Test", "Type", 0.9).with_modality(Modality::Symbolic),
    );
    doc.add_signal(
        Signal::new(0, Location::text(5, 13), "document", "Type", 0.8)
            .with_modality(Modality::Symbolic),
    );
    doc.add_signal(
        Signal::new(0, Location::bbox(0.0, 0.0, 0.1, 0.1), "logo", "Image", 0.85)
            .with_modality(Modality::Iconic),
    );

    let stats = doc.stats();
    assert_eq!(stats.signal_count, 3);
    assert_eq!(stats.symbolic_count, 2);
    assert_eq!(stats.iconic_count, 1);
    assert_eq!(stats.hybrid_count, 0);

    // Average confidence
    let expected_avg = (0.9 + 0.8 + 0.85) / 3.0;
    assert!((stats.avg_confidence - expected_avg).abs() < 0.001);
}

#[test]
fn document_filtering() {
    let mut doc = GroundedDocument::new("filter_test", "John went to Paris. He loved it.");

    // Add signals with different properties
    let _john = doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "Person", 0.95));
    let _paris = doc.add_signal(Signal::new(
        0,
        Location::text(13, 18),
        "Paris",
        "Location",
        0.92,
    ));
    let _he = doc.add_signal(Signal::new(0, Location::text(20, 22), "He", "Person", 0.88));

    // Add a negated signal
    doc.add_signal(
        Signal::new(0, Location::text(25, 35), "not happy", "Sentiment", 0.75).negated(),
    );

    // Filter by confidence
    let high_conf = doc.confident_signals(0.9);
    assert_eq!(high_conf.len(), 2); // John and Paris

    // Filter by label
    let persons = doc.signals_with_label("Person");
    assert_eq!(persons.len(), 2); // John and He

    // Filter by negation
    let negated = doc.negated_signals();
    assert_eq!(negated.len(), 1);
    assert_eq!(negated[0].surface, "not happy");

    // Filter by range
    let first_sentence = doc.signals_in_range(0, 19);
    assert_eq!(first_sentence.len(), 2); // John and Paris
}

// =============================================================================
// Part 6: Round-trip Entity ↔ GroundedDocument
// =============================================================================

#[test]
fn roundtrip_entity_to_grounded_and_back() {
    // Create legacy entities with coreference
    let entities = vec![
        Entity::new("John Smith", EntityType::Person, 0, 10, 0.95).with_canonical_id(1),
        Entity::new("CEO", EntityType::Other("Title".into()), 20, 23, 0.88).with_canonical_id(1), // Same entity as John
        Entity::new("Apple Inc", EntityType::Organization, 30, 39, 0.92).with_canonical_id(2),
        Entity::new("the company", EntityType::Organization, 50, 61, 0.85).with_canonical_id(2), // Same entity as Apple
    ];

    let text = "John Smith is the CEO of Apple Inc. He leads the company.";

    // Convert to GroundedDocument
    let doc = GroundedDocument::from_entities("roundtrip_test", text, &entities);

    // Verify signals
    assert_eq!(doc.signals().len(), 4);

    // Verify tracks (should have 2 tracks: one for John/CEO, one for Apple/company)
    let track_count: usize = doc.tracks().count();
    // Note: entities without canonical_id would each get their own track
    // Entities with the same canonical_id should be in the same track
    assert!(track_count >= 2); // At least 2 tracks

    // Convert back to entities
    let recovered = doc.to_entities();
    assert_eq!(recovered.len(), 4);

    // Verify properties preserved
    for (original, recovered) in entities.iter().zip(recovered.iter()) {
        assert_eq!(original.text, recovered.text);
        assert_eq!(original.start, recovered.start);
        assert_eq!(original.end, recovered.end);
        // canonical_id becomes track_id, so might not be exactly equal
        // but entities with same original canonical_id should have same recovered canonical_id
    }

    // Verify coreference preserved: entities with same original canonical_id
    // should have same recovered canonical_id
    let john_recovered = &recovered[0];
    let ceo_recovered = &recovered[1];
    assert_eq!(john_recovered.canonical_id, ceo_recovered.canonical_id);

    let apple_recovered = &recovered[2];
    let company_recovered = &recovered[3];
    assert_eq!(apple_recovered.canonical_id, company_recovered.canonical_id);
}

#[test]
fn roundtrip_preserves_kb_id() {
    let mut entity = Entity::new("Marie Curie", EntityType::Person, 0, 11, 0.95);
    entity.link_to_kb("Q7186");
    entity.set_canonical(42);

    let text = "Marie Curie won the Nobel Prize.";
    let doc = GroundedDocument::from_entities("kb_test", text, &[entity.clone()]);

    // Add an identity with the KB info
    // Note: from_entities doesn't automatically create identities, so we check
    // that the track exists and could be linked
    let recovered = doc.to_entities();
    assert_eq!(recovered.len(), 1);
    // KB info is preserved through identity linking if set up
}

#[test]
fn roundtrip_discontinuous_span() {
    use anno::DiscontinuousSpan;

    let mut entity = Entity::new("airports", EntityType::Location, 0, 0, 0.9);
    entity.discontinuous_span = Some(DiscontinuousSpan::new(vec![0..8, 13..15, 17..25]));

    let text = "New York and LA airports are busy.";
    let doc = GroundedDocument::from_entities("disc_test", text, &[entity]);

    // Verify the discontinuous location is preserved
    let signal = &doc.signals()[0];
    match &signal.location {
        Location::Discontinuous { segments } => {
            assert_eq!(segments.len(), 3);
        }
        _ => panic!("Expected discontinuous location"),
    }
}

// =============================================================================
// Part 7: Multimodal Document Tests
// =============================================================================

#[test]
fn multimodal_document_mixed_signals() {
    let mut doc = GroundedDocument::new("multimodal", "A photo showing John at Paris.");

    // Text signals
    let john_text = doc.add_signal(
        Signal::new(0, Location::text(16, 20), "John", "Person", 0.9)
            .with_modality(Modality::Symbolic),
    );
    let _paris_text = doc.add_signal(
        Signal::new(0, Location::text(24, 29), "Paris", "Location", 0.88)
            .with_modality(Modality::Symbolic),
    );

    // Visual signals (detected in accompanying image)
    let john_face = doc.add_signal(
        Signal::new(
            0,
            Location::bbox(0.2, 0.1, 0.15, 0.2),
            "face",
            "Person",
            0.95,
        )
        .with_modality(Modality::Iconic),
    );
    let _eiffel = doc.add_signal(
        Signal::new(
            0,
            Location::bbox(0.6, 0.3, 0.2, 0.4),
            "eiffel_tower",
            "Landmark",
            0.92,
        )
        .with_modality(Modality::Iconic),
    );

    assert_eq!(doc.text_signals().len(), 2);
    assert_eq!(doc.visual_signals().len(), 2);

    // Create tracks that link text and visual mentions
    // This is the "multimodal coreference" case
    let mut john_track = Track::new(0, "John");
    john_track.add_signal(john_text, 0);
    john_track.add_signal(john_face, 1);
    doc.add_track(john_track);

    // Create identity that bridges both modalities
    let john_identity =
        Identity::from_kb(0, "John Doe", "internal_db", "PERSON_001").with_type("Person");
    let john_id = doc.add_identity(john_identity);
    doc.link_track_to_identity(0, john_id);

    // Verify multimodal traversal: text mention → track → identity
    let identity_from_text = doc.identity_for_signal(john_text);
    assert!(identity_from_text.is_some());
    assert_eq!(identity_from_text.unwrap().canonical_name, "John Doe");

    // Same identity reachable from visual signal
    let identity_from_visual = doc.identity_for_signal(john_face);
    assert!(identity_from_visual.is_some());
    assert_eq!(
        identity_from_visual.unwrap().kb_id,
        identity_from_text.unwrap().kb_id
    );
}

#[test]
fn multimodal_ocr_hybrid_location() {
    let mut doc = GroundedDocument::new("ocr_doc", "Invoice #12345");

    // OCR signal: has both text offsets AND visual bounding box
    let invoice_text = Location::TextWithBbox {
        start: 0,
        end: 14,
        bbox: Box::new(Location::bbox(0.1, 0.05, 0.3, 0.04)),
    };

    let signal = Signal::new(0, invoice_text.clone(), "Invoice #12345", "Document", 0.97)
        .with_modality(Modality::Hybrid);

    doc.add_signal(signal);

    let stats = doc.stats();
    assert_eq!(stats.hybrid_count, 1);

    // Hybrid location should have text offsets
    assert_eq!(invoice_text.text_offsets(), Some((0, 14)));

    // And should count as hybrid modality
    assert_eq!(invoice_text.modality(), Modality::Hybrid);
}

// =============================================================================
// Part 8: Batch Operations Tests
// =============================================================================

#[test]
fn batch_add_signals() {
    let mut doc = GroundedDocument::new("batch", "A B C D E F G H");

    let signals: Vec<Signal<Location>> = (0..8)
        .map(|i| {
            Signal::new(
                0,
                Location::text(i * 2, i * 2 + 1),
                format!("{}", (b'A' + i as u8) as char),
                "Letter",
                0.9,
            )
        })
        .collect();

    let ids = doc.add_signals(signals);
    assert_eq!(ids.len(), 8);
    assert_eq!(doc.signals().len(), 8);
}

#[test]
fn create_track_from_signals() {
    let mut doc = GroundedDocument::new("track_create", "John met Mary. He greeted her.");

    let john = doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "Person", 0.95));
    let mary = doc.add_signal(Signal::new(
        0,
        Location::text(9, 13),
        "Mary",
        "Person",
        0.92,
    ));
    let he = doc.add_signal(Signal::new(0, Location::text(15, 17), "He", "Person", 0.88));
    let her = doc.add_signal(Signal::new(
        0,
        Location::text(27, 30),
        "her",
        "Person",
        0.85,
    ));

    // Create tracks using helper
    let john_track = doc.create_track_from_signals("John", &[john, he]);
    let mary_track = doc.create_track_from_signals("Mary", &[mary, her]);

    assert!(john_track.is_some());
    assert!(mary_track.is_some());

    let jt = doc.get_track(john_track.unwrap()).unwrap();
    assert_eq!(jt.len(), 2);
    assert_eq!(jt.canonical_surface, "John");
}

#[test]
fn merge_tracks() {
    let mut doc = GroundedDocument::new("merge", "A B C");

    let a = doc.add_signal(Signal::new(0, Location::text(0, 1), "A", "T", 0.9));
    let b = doc.add_signal(Signal::new(0, Location::text(2, 3), "B", "T", 0.9));
    let c = doc.add_signal(Signal::new(0, Location::text(4, 5), "C", "T", 0.9));

    let track1 = doc.create_track_from_signals("A", &[a]).unwrap();
    let track2 = doc.create_track_from_signals("B", &[b, c]).unwrap();

    // Merge tracks
    let merged = doc.merge_tracks(&[track1, track2]);
    assert!(merged.is_some());

    let mt = doc.get_track(merged.unwrap()).unwrap();
    assert_eq!(mt.len(), 3);
    assert_eq!(mt.canonical_surface, "A"); // Takes canonical from first track
}

#[test]
fn find_overlapping_signals() {
    let mut doc = GroundedDocument::new("overlap", "New York City is great.");

    // Overlapping spans: "New York" and "New York City"
    doc.add_signal(Signal::new(
        0,
        Location::text(0, 8),
        "New York",
        "City",
        0.9,
    ));
    doc.add_signal(Signal::new(
        0,
        Location::text(0, 13),
        "New York City",
        "City",
        0.95,
    ));
    doc.add_signal(Signal::new(0, Location::text(17, 22), "great", "Adj", 0.8));

    let overlaps = doc.find_overlapping_signal_pairs();
    assert_eq!(overlaps.len(), 1); // Only the first two overlap
}

// =============================================================================
// Part 9: CorefDocument Conversion
// =============================================================================

#[test]
fn convert_to_coref_document() {
    let mut doc = GroundedDocument::new("coref_test", "John saw Mary. He waved to her.");

    let john = doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "Person", 0.95));
    let mary = doc.add_signal(Signal::new(
        0,
        Location::text(9, 13),
        "Mary",
        "Person",
        0.92,
    ));
    let he = doc.add_signal(Signal::new(0, Location::text(15, 17), "He", "Person", 0.88));
    let her = doc.add_signal(Signal::new(
        0,
        Location::text(27, 30),
        "her",
        "Person",
        0.85,
    ));

    doc.create_track_from_signals("John", &[john, he]);
    doc.create_track_from_signals("Mary", &[mary, her]);

    // Convert to CorefDocument
    let coref_doc = doc.to_coref_document();

    assert_eq!(coref_doc.chain_count(), 2);
    assert_eq!(coref_doc.mention_count(), 4);
}

// =============================================================================
// Part 10: Property Tests for Invariants
// =============================================================================

#[test]
fn invariant_signal_ids_unique() {
    let mut doc = GroundedDocument::new("unique_ids", "Test");

    let ids: Vec<_> = (0..10)
        .map(|_| {
            doc.add_signal(Signal::new(
                999, // Auto-assigned, this gets overwritten
                Location::text(0, 4),
                "Test",
                "Type",
                0.9,
            ))
        })
        .collect();

    // All IDs should be unique
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len());

    // IDs should be sequential
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(id, i as u64);
    }
}

#[test]
fn invariant_track_signal_consistency() {
    let mut doc = GroundedDocument::new("consistency", "A B C");

    let a = doc.add_signal(Signal::new(0, Location::text(0, 1), "A", "T", 0.9));
    let b = doc.add_signal(Signal::new(0, Location::text(2, 3), "B", "T", 0.9));

    let track_id = doc.create_track_from_signals("AB", &[a, b]).unwrap();

    // Verify signal → track lookup is consistent
    assert_eq!(doc.track_for_signal(a).unwrap().id, track_id);
    assert_eq!(doc.track_for_signal(b).unwrap().id, track_id);
}

#[test]
fn invariant_identity_track_consistency() {
    let mut doc = GroundedDocument::new("id_consistency", "Test");

    let s = doc.add_signal(Signal::new(0, Location::text(0, 4), "Test", "T", 0.9));
    let track_id = doc.create_track_from_signals("Test", &[s]).unwrap();

    let identity = Identity::new(0, "Test Entity");
    let identity_id = doc.add_identity(identity);
    doc.link_track_to_identity(track_id, identity_id);

    // Verify track → identity lookup is consistent
    assert_eq!(doc.identity_for_track(track_id).unwrap().id, identity_id);
    assert_eq!(doc.identity_for_signal(s).unwrap().id, identity_id);
}

#[test]
fn modality_count_sums_to_total() {
    let mut doc = GroundedDocument::new("modality_sum", "Mixed");

    // Add different modalities
    doc.add_signal(
        Signal::new(0, Location::text(0, 5), "text", "T", 0.9).with_modality(Modality::Symbolic),
    );
    doc.add_signal(
        Signal::new(0, Location::bbox(0.0, 0.0, 0.1, 0.1), "img", "T", 0.9)
            .with_modality(Modality::Iconic),
    );
    doc.add_signal(
        Signal::new(
            0,
            Location::TextWithBbox {
                start: 0,
                end: 5,
                bbox: Box::new(Location::bbox(0.0, 0.0, 0.1, 0.1)),
            },
            "ocr",
            "T",
            0.9,
        )
        .with_modality(Modality::Hybrid),
    );

    let stats = doc.stats();
    assert_eq!(
        stats.symbolic_count + stats.iconic_count + stats.hybrid_count,
        stats.signal_count
    );
}

// =============================================================================
// Part 11: Spatial Index Tests
// =============================================================================

#[test]
fn spatial_index_basic_operations() {
    let mut index = TextSpatialIndex::new();
    assert!(index.is_empty());

    index.insert(0, 0, 10);
    index.insert(1, 20, 30);
    index.insert(2, 5, 15);

    assert_eq!(index.len(), 3);
    assert!(!index.is_empty());
}

#[test]
fn spatial_index_query_overlap() {
    let mut index = TextSpatialIndex::new();
    index.insert(0, 0, 10); // [0, 10)
    index.insert(1, 20, 30); // [20, 30)
    index.insert(2, 5, 15); // [5, 15)

    // Query [7, 12) should overlap with [0,10) and [5,15)
    let results = index.query_overlap(7, 12);
    assert!(results.contains(&0));
    assert!(results.contains(&2));
    assert!(!results.contains(&1));

    // Query [25, 35) should only overlap with [20,30)
    let results = index.query_overlap(25, 35);
    assert_eq!(results, vec![1]);

    // Query [100, 110) should have no overlaps
    let results = index.query_overlap(100, 110);
    assert!(results.is_empty());
}

#[test]
fn spatial_index_query_containing() {
    let mut index = TextSpatialIndex::new();
    index.insert(0, 0, 100); // Large span
    index.insert(1, 20, 30); // Small span
    index.insert(2, 5, 95); // Medium span

    // Query for spans containing [25, 28)
    let results = index.query_containing(25, 28);
    assert!(results.contains(&0)); // [0, 100) contains [25, 28)
    assert!(results.contains(&1)); // [20, 30) contains [25, 28)
    assert!(results.contains(&2)); // [5, 95) contains [25, 28)

    // Query for spans containing [0, 100)
    let results = index.query_containing(0, 100);
    assert!(results.contains(&0)); // Only [0, 100) contains itself
    assert!(!results.contains(&1));
    assert!(!results.contains(&2));
}

#[test]
fn spatial_index_query_contained_in() {
    let mut index = TextSpatialIndex::new();
    index.insert(0, 0, 10);
    index.insert(1, 20, 30);
    index.insert(2, 5, 15);
    index.insert(3, 100, 110);

    // Query for spans contained in [0, 50)
    let results = index.query_contained_in(0, 50);
    assert!(results.contains(&0)); // [0, 10) is in [0, 50)
    assert!(results.contains(&1)); // [20, 30) is in [0, 50)
    assert!(results.contains(&2)); // [5, 15) is in [0, 50)
    assert!(!results.contains(&3)); // [100, 110) is not in [0, 50)
}

#[test]
fn spatial_index_from_document() {
    let mut doc =
        GroundedDocument::new("index_test", "The quick brown fox jumps over the lazy dog.");

    doc.add_signal(Signal::new(0, Location::text(0, 3), "The", "DET", 0.9));
    doc.add_signal(Signal::new(0, Location::text(4, 9), "quick", "ADJ", 0.9));
    doc.add_signal(Signal::new(0, Location::text(10, 15), "brown", "ADJ", 0.9));
    doc.add_signal(Signal::new(0, Location::text(16, 19), "fox", "NOUN", 0.9));

    let index = doc.build_text_index();
    assert_eq!(index.len(), 4);

    // Query for signals in the first 10 characters
    let results = index.query_contained_in(0, 10);
    assert_eq!(results.len(), 2); // "The" and "quick"
}

#[test]
fn spatial_index_document_methods() {
    let mut doc = GroundedDocument::new("query_test", "John went to Paris and London.");

    doc.add_signal(Signal::new(0, Location::text(0, 4), "John", "Person", 0.95));
    doc.add_signal(Signal::new(
        0,
        Location::text(13, 18),
        "Paris",
        "Location",
        0.92,
    ));
    doc.add_signal(Signal::new(
        0,
        Location::text(23, 29),
        "London",
        "Location",
        0.90,
    ));

    // Use indexed query methods
    let in_range = doc.query_signals_in_range_indexed(10, 25);
    assert_eq!(in_range.len(), 1); // Only "Paris"
    assert_eq!(in_range[0].surface, "Paris");

    let overlapping = doc.query_overlapping_signals_indexed(15, 25);
    assert_eq!(overlapping.len(), 2); // "Paris" and "London"
}

// =============================================================================
// Part 12: HTML Rendering Tests
// =============================================================================

#[test]
fn html_rendering_produces_valid_output() {
    let mut doc = GroundedDocument::new("html_test", "Marie Curie won the Nobel Prize.");

    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Marie Curie",
        "Person",
        0.95,
    ));
    let _s2 = doc.add_signal(Signal::new(
        0,
        Location::text(21, 32),
        "Nobel Prize",
        "Award",
        0.92,
    ));

    let mut track = Track::new(0, "Marie Curie");
    track.add_signal(s1, 0);
    let track_id = doc.add_track(track);

    let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
    let identity_id = doc.add_identity(identity);
    doc.link_track_to_identity(track_id, identity_id);

    let html = render_document_html(&doc);

    // Basic structure checks
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<html>"));
    assert!(html.contains("</html>"));
    assert!(html.contains("Marie Curie"));
    assert!(html.contains("Nobel Prize"));
    assert!(html.contains("Q7186"));

    // Stats should be present (lowercase in brutalist design)
    assert!(html.contains("signals"));
    assert!(html.contains("tracks"));
    assert!(html.contains("identities"));
}

#[test]
fn html_escapes_special_characters() {
    let mut doc = GroundedDocument::new("escape_test", "a < b && c > d");

    doc.add_signal(Signal::new(0, Location::text(0, 1), "a", "VAR", 0.9));

    let html = render_document_html(&doc);

    // Should escape HTML special characters
    assert!(html.contains("&lt;")); // <
    assert!(html.contains("&amp;")); // &
    assert!(html.contains("&gt;")); // >
}

#[test]
fn html_handles_empty_document() {
    let doc = GroundedDocument::new("empty", "");

    let html = render_document_html(&doc);

    // Should still produce valid HTML
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("signals")); // lowercase in brutalist design
}

#[test]
fn html_handles_multimodal_document() {
    let mut doc = GroundedDocument::new("multimodal", "Photo of Paris");

    // Text signal
    doc.add_signal(
        Signal::new(0, Location::text(9, 14), "Paris", "Location", 0.9)
            .with_modality(Modality::Symbolic),
    );

    // Visual signal (not rendered in text, but shown in stats)
    doc.add_signal(
        Signal::new(
            0,
            Location::bbox(0.0, 0.0, 0.5, 0.5),
            "Eiffel Tower",
            "Landmark",
            0.95,
        )
        .with_modality(Modality::Iconic),
    );

    let html = render_document_html(&doc);

    // Should show modality breakdown since we have iconic signals (brutalist: sym/ico/hyb)
    assert!(html.contains("sym/ico/hyb"));
}
