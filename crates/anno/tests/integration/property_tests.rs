//! Property-based tests for `anno::core` types.
//!
//! These tests verify fundamental invariants of the Signal → Track → Identity hierarchy
//! and the Location type's geometric operations.

use anno::core::grounded::{
    Identity, IdentitySource, Location, Modality, Signal, SignalRef, Track, TrackRef,
};
use anno::TypeLabel;
use anno::{Confidence, IdentityId, SignalId};
use proptest::prelude::*;

// =============================================================================
// Arbitrary Generators
// =============================================================================

/// Generate a text location with valid offsets
fn arb_text_location() -> impl Strategy<Value = Location> {
    (0usize..10000, 1usize..1000).prop_map(|(start, len)| Location::text(start, start + len))
}

/// Generate any location type
fn arb_location() -> impl Strategy<Value = Location> {
    arb_text_location()
}

/// Generate a signal with text location
fn arb_signal() -> impl Strategy<Value = Signal<Location>> {
    (
        any::<u64>(),
        arb_text_location(),
        "[A-Za-z ]{1,50}",    // surface
        "(PER|ORG|LOC|MISC)", // label
        0.0f32..1.0,          // confidence
    )
        .prop_map(|(id, location, surface, label, confidence)| Signal {
            id: id.into(),
            location,
            surface,
            label: TypeLabel::from(label.as_str()),
            confidence: Confidence::new(confidence as f64),
            hierarchical: None,
            provenance: None,
            modality: Modality::Symbolic,
            normalized: None,
            negated: false,
            quantifier: None,
        })
}

/// Generate a track with valid structure
fn arb_track() -> impl Strategy<Value = Track> {
    (
        any::<u64>(),
        "[A-Za-z ]{1,30}",                      // canonical surface
        prop::option::of("(PER|ORG|LOC|MISC)"), // entity type
        0.0f32..1.0,                            // cluster confidence
        prop::collection::vec((any::<u64>(), 0u32..1000), 1..10), // signals
    )
        .prop_map(
            |(id, canonical_surface, entity_type, confidence, signal_specs)| {
                let mut track = Track::new(id, canonical_surface);
                track.entity_type = entity_type.map(|s| TypeLabel::from(s.as_str()));
                track.cluster_confidence = Confidence::new(confidence as f64);
                for (sig_id, pos) in signal_specs {
                    track.add_signal(sig_id, pos);
                }
                track
            },
        )
}

/// Generate an identity
fn arb_identity() -> impl Strategy<Value = Identity> {
    (
        any::<u64>(),
        "[A-Za-z ]{1,50}",                              // canonical name
        prop::option::of("(PER|ORG|LOC|MISC)"),         // entity type
        prop::option::of("Q[0-9]{1,10}"),               // kb_id
        0.0f32..1.0,                                    // confidence
        prop::collection::vec("[A-Za-z ]{1,20}", 0..5), // aliases
    )
        .prop_map(
            |(id, canonical_name, entity_type, kb_id, confidence, aliases)| {
                let mut identity = Identity::new(id, canonical_name);
                identity.entity_type = entity_type.map(|s| TypeLabel::from(s.as_str()));
                identity.kb_id = kb_id;
                identity.confidence = Confidence::new(confidence as f64);
                for alias in aliases {
                    identity.add_alias(alias);
                }
                identity
            },
        )
}

// =============================================================================
// Location Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // --- Modality ---

    /// Text locations have Symbolic modality
    #[test]
    fn text_location_is_symbolic(loc in arb_text_location()) {
        prop_assert_eq!(loc.modality(), Modality::Symbolic);
    }



    // --- Text Offsets ---

    /// text_offsets() returns Some for text locations
    #[test]
    fn text_location_has_offsets(start in 0usize..10000, len in 1usize..1000) {
        let loc = Location::text(start, start + len);
        let offsets = loc.text_offsets();
        prop_assert!(offsets.is_some());
        let (s, e) = offsets.expect("text_offsets must be Some for text locations");
        prop_assert_eq!(s, start);
        prop_assert_eq!(e, start + len);
    }



    // --- Overlap ---

    /// A location always overlaps with itself
    #[test]
    fn location_overlaps_self(loc in arb_location()) {
        prop_assert!(loc.overlaps(&loc));
    }

    /// Overlap is symmetric
    #[test]
    fn overlap_symmetric(
        start1 in 0usize..5000,
        len1 in 1usize..500,
        start2 in 0usize..5000,
        len2 in 1usize..500
    ) {
        let loc1 = Location::text(start1, start1 + len1);
        let loc2 = Location::text(start2, start2 + len2);
        prop_assert_eq!(loc1.overlaps(&loc2), loc2.overlaps(&loc1));
    }

    /// Non-overlapping text spans don't overlap
    #[test]
    fn disjoint_text_no_overlap(
        start1 in 0usize..1000,
        len1 in 1usize..100
    ) {
        let loc1 = Location::text(start1, start1 + len1);
        let loc2 = Location::text(start1 + len1 + 100, start1 + len1 + 200);
        prop_assert!(!loc1.overlaps(&loc2));
    }



    // --- IoU (Intersection over Union) ---

    /// IoU of identical locations is 1.0
    #[test]
    fn iou_identical_is_one(loc in arb_location()) {
        if let Some(iou) = loc.iou(&loc) {
            prop_assert!((iou - 1.0).abs() < 0.0001, "IoU of identical should be 1.0, got {}", iou);
        }
    }

    /// IoU is symmetric
    #[test]
    fn iou_symmetric(loc1 in arb_text_location(), loc2 in arb_text_location()) {
        let iou1 = loc1.iou(&loc2);
        let iou2 = loc2.iou(&loc1);
        prop_assert_eq!(iou1, iou2, "IoU should be symmetric");
    }

    /// IoU is bounded [0, 1]
    #[test]
    fn iou_bounded(loc1 in arb_text_location(), loc2 in arb_text_location()) {
        if let Some(iou) = loc1.iou(&loc2) {
            prop_assert!(
                (0.0..=1.0).contains(&iou),
                "IoU should be in [0,1], got {}",
                iou
            );
        }
    }

    /// Disjoint locations have IoU = 0
    #[test]
    fn disjoint_iou_zero(start in 0usize..1000, len in 1usize..100) {
        let loc1 = Location::text(start, start + len);
        let loc2 = Location::text(start + len + 100, start + len + 200);
        if let Some(iou) = loc1.iou(&loc2) {
            prop_assert!((iou - 0.0).abs() < 0.0001, "Disjoint IoU should be 0, got {}", iou);
        }
    }

    /// Contained location has IoU = inner_area / outer_area
    #[test]
    fn contained_iou(
        start in 100usize..1000,
        outer_len in 100usize..500,
        inner_offset in 10usize..50,
        inner_len in 10usize..50
    ) {
        // Ensure inner is actually inside outer
        prop_assume!(inner_offset + inner_len < outer_len);

        let outer = Location::text(start, start + outer_len);
        let inner = Location::text(start + inner_offset, start + inner_offset + inner_len);

        if let Some(iou) = outer.iou(&inner) {
            // IoU = intersection / union = inner_len / outer_len
            let expected = inner_len as f64 / outer_len as f64;
            prop_assert!(
                (iou - expected).abs() < 0.01,
                "Contained IoU should be {}, got {}",
                expected,
                iou
            );
        }
    }
}

// =============================================================================
// Signal Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Signals have valid confidence bounds
    #[test]
    fn signal_confidence_bounded(signal in arb_signal()) {
        prop_assert!(signal.confidence.value() >= 0.0 && signal.confidence.value() <= 1.0);
    }

    /// Signals have non-empty surface
    #[test]
    fn signal_surface_non_empty(signal in arb_signal()) {
        prop_assert!(!signal.surface.is_empty());
    }

    /// Signals have non-empty label
    #[test]
    fn signal_label_non_empty(signal in arb_signal()) {
        prop_assert!(!signal.label.as_str().is_empty());
    }

    /// Signal location text offsets are valid (start < end)
    #[test]
    fn signal_text_offsets_valid(signal in arb_signal()) {
        if let Some((start, end)) = signal.location.text_offsets() {
            prop_assert!(start < end, "start ({}) should be < end ({})", start, end);
        }
    }
}

// =============================================================================
// Track Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Tracks have valid cluster confidence bounds
    #[test]
    fn track_confidence_bounded(track in arb_track()) {
        prop_assert!(
            track.cluster_confidence.value() >= 0.0 && track.cluster_confidence.value() <= 1.0,
            "Cluster confidence should be in [0,1], got {}",
            track.cluster_confidence
        );
    }

    /// Tracks have non-empty canonical surface
    #[test]
    fn track_canonical_non_empty(track in arb_track()) {
        prop_assert!(!track.canonical_surface.is_empty());
    }

    /// Track len() matches signals count
    #[test]
    fn track_len_matches_signals(track in arb_track()) {
        prop_assert_eq!(track.len(), track.signals.len());
    }

    /// Track is_empty() is consistent with len()
    #[test]
    fn track_is_empty_consistent(track in arb_track()) {
        prop_assert_eq!(track.is_empty(), track.signals.is_empty());
    }

    /// Track is_singleton() is consistent with len()
    #[test]
    fn track_is_singleton_consistent(track in arb_track()) {
        prop_assert_eq!(track.is_singleton(), track.len() == 1);
    }

    /// New track has no identity by default
    #[test]
    fn new_track_no_identity(id in any::<u64>(), surface in "[A-Za-z]{1,20}") {
        let track = Track::new(id, surface);
        prop_assert!(track.identity_id.is_none());
    }

    /// with_identity sets identity_id
    #[test]
    fn with_identity_sets_id(id in any::<u64>(), surface in "[A-Za-z]{1,20}", identity_id in any::<u64>()) {
        let identity_id = IdentityId::new(identity_id);
        let track = Track::new(id, surface).with_identity(identity_id);
        prop_assert_eq!(track.identity_id, Some(identity_id));
    }

    /// with_type sets entity_type
    #[test]
    fn with_type_sets_type(id in any::<u64>(), surface in "[A-Za-z]{1,20}", etype in "(PER|ORG|LOC)") {
        let track = Track::new(id, surface).with_type(&etype);
        prop_assert_eq!(track.entity_type, Some(TypeLabel::from(etype.as_str())));
    }
}

// =============================================================================
// Identity Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Identities have valid confidence bounds
    #[test]
    fn identity_confidence_bounded(identity in arb_identity()) {
        prop_assert!(
            identity.confidence.value() >= 0.0 && identity.confidence.value() <= 1.0,
            "Identity confidence should be in [0,1], got {}",
            identity.confidence
        );
    }

    /// Identities have non-empty canonical name
    #[test]
    fn identity_canonical_non_empty(identity in arb_identity()) {
        prop_assert!(!identity.canonical_name.is_empty());
    }

    /// New identity has empty aliases
    #[test]
    fn new_identity_empty_aliases(id in any::<u64>(), name in "[A-Za-z]{1,20}") {
        let identity = Identity::new(id, name);
        prop_assert!(identity.aliases.is_empty());
    }

    /// add_alias increases alias count
    #[test]
    fn add_alias_increases_count(id in any::<u64>(), name in "[A-Za-z]{1,20}", alias in "[A-Za-z]{1,10}") {
        let mut identity = Identity::new(id, name);
        let before = identity.aliases.len();
        identity.add_alias(alias);
        prop_assert_eq!(identity.aliases.len(), before + 1);
    }

    /// Identity serialization roundtrip
    #[test]
    fn identity_serde_roundtrip(identity in arb_identity()) {
        let json = serde_json::to_string(&identity).expect("Serialization failed");
        let deserialized: Identity = serde_json::from_str(&json).expect("Deserialization failed");

        prop_assert_eq!(identity.id, deserialized.id);
        prop_assert_eq!(identity.canonical_name, deserialized.canonical_name);
        prop_assert_eq!(identity.entity_type, deserialized.entity_type);
        prop_assert_eq!(identity.kb_id, deserialized.kb_id);
        prop_assert_eq!(identity.aliases, deserialized.aliases);
    }
}

// =============================================================================
// Hierarchy Invariants
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Signal → Track: signals can be added to track
    #[test]
    fn signals_can_join_track(
        track_id in any::<u64>(),
        surface in "[A-Za-z]{1,20}",
        signal_ids in prop::collection::vec(any::<u64>(), 1..10)
    ) {
        let mut track = Track::new(track_id, surface);
        let signal_ids: Vec<SignalId> = signal_ids.into_iter().map(SignalId::new).collect();
        for (pos, sig_id) in signal_ids.iter().enumerate() {
            track.add_signal(*sig_id, pos as u32);
        }

        prop_assert_eq!(track.len(), signal_ids.len());
        for (pos, sig_id) in signal_ids.iter().enumerate() {
            prop_assert_eq!(track.signals[pos].signal_id, *sig_id);
        }
    }

    /// Track → Identity: track can link to identity
    #[test]
    fn track_can_link_identity(
        track_id in any::<u64>(),
        surface in "[A-Za-z]{1,20}",
        identity_id in any::<u64>()
    ) {
        let identity_id = IdentityId::new(identity_id);
        let track = Track::new(track_id, surface).with_identity(identity_id);
        prop_assert_eq!(track.identity_id, Some(identity_id));
    }

    /// Identity can track multiple cross-doc tracks
    #[test]
    fn identity_cross_doc_tracks(
        identity_id in any::<u64>(),
        name in "[A-Za-z]{1,20}",
        track_refs in prop::collection::vec(
            ("[a-z]{3,8}", any::<u64>()),
            1..5
        )
    ) {
        let refs: Vec<TrackRef> = track_refs
            .into_iter()
            .map(|(doc_id, track_id)| TrackRef { doc_id, track_id: track_id.into() })
            .collect();

        let mut identity = Identity::new(identity_id, name);
        identity.source = Some(IdentitySource::CrossDocCoref {
            track_refs: refs.clone(),
        });

        match &identity.source {
            Some(IdentitySource::CrossDocCoref { track_refs }) => {
                prop_assert_eq!(track_refs.len(), refs.len());
            }
            _ => prop_assert!(false, "Expected CrossDocCoref source"),
        }
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_zero_length_text_span() {
    let loc = Location::text(100, 100);
    // Zero-length spans are technically valid but unusual
    assert_eq!(loc.text_offsets(), Some((100, 100)));
}

#[test]
fn test_location_default() {
    let loc = Location::default();
    assert_eq!(loc, Location::Text { start: 0, end: 0 });
}

#[test]
fn test_track_empty_after_creation() {
    let track = Track::new(1, "Test");
    assert!(track.is_empty());
    assert!(!track.is_singleton());
}

#[test]
fn test_signal_ref_equality() {
    let ref1 = SignalRef {
        signal_id: 42.into(),
        position: 5,
    };
    let ref2 = SignalRef {
        signal_id: 42.into(),
        position: 5,
    };
    assert_eq!(ref1, ref2);
}

#[test]
fn test_track_ref_equality() {
    let ref1 = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 42.into(),
    };
    let ref2 = TrackRef {
        doc_id: "doc1".to_string(),
        track_id: 42.into(),
    };
    assert_eq!(ref1, ref2);
}

#[test]
fn test_identity_source_variants_all() {
    let kb = IdentitySource::KnowledgeBase {
        kb_name: "wikidata".to_string(),
        kb_id: "Q42".to_string(),
    };
    assert!(matches!(kb, IdentitySource::KnowledgeBase { .. }));

    let coref = IdentitySource::CrossDocCoref {
        track_refs: vec![TrackRef {
            doc_id: "doc1".to_string(),
            track_id: 1.into(),
        }],
    };
    assert!(matches!(coref, IdentitySource::CrossDocCoref { .. }));

    let hybrid = IdentitySource::Hybrid {
        track_refs: vec![],
        kb_name: "wikidata".to_string(),
        kb_id: "Q42".to_string(),
    };
    assert!(matches!(hybrid, IdentitySource::Hybrid { .. }));
}

// =============================================================================
// Phi-Feature Linguistic Invariants
// =============================================================================
//
// These property tests encode fundamental constraints from linguistic theory.
// They verify that our phi-feature system respects cross-linguistic universals.

use anno::core::types::{Gender, MentionType, Number, Person, PhiFeatures};

/// Generate an arbitrary Person
fn arb_person() -> impl Strategy<Value = Person> {
    prop_oneof![
        Just(Person::First),
        Just(Person::Second),
        Just(Person::Third),
        Just(Person::Unknown),
    ]
}

/// Generate an arbitrary Number
fn arb_number() -> impl Strategy<Value = Number> {
    prop_oneof![
        Just(Number::Singular),
        Just(Number::Dual),
        Just(Number::Plural),
        Just(Number::Unknown),
    ]
}

/// Generate an arbitrary Gender
fn arb_gender() -> impl Strategy<Value = Gender> {
    prop_oneof![
        Just(Gender::Masculine),
        Just(Gender::Feminine),
        Just(Gender::Neutral),
        Just(Gender::Unknown),
    ]
}

/// Generate arbitrary PhiFeatures
fn arb_phi_features() -> impl Strategy<Value = PhiFeatures> {
    (arb_person(), arb_number(), arb_gender())
        .prop_map(|(person, number, gender)| PhiFeatures::new(person, number, gender))
}

/// Generate an arbitrary MentionType
fn arb_mention_type() -> impl Strategy<Value = MentionType> {
    prop_oneof![
        Just(MentionType::Proper),
        Just(MentionType::Nominal),
        Just(MentionType::Pronominal),
        Just(MentionType::Zero),
        Just(MentionType::Unknown),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // =========================================================================
    // Reflexivity: Every phi-feature is compatible with itself
    // =========================================================================

    /// Compatibility is reflexive (a feature matches itself)
    ///
    /// This is a fundamental logical property. If an entity has gender X,
    /// it should be compatible with gender X.
    #[test]
    fn phi_compatibility_reflexive(phi in arb_phi_features()) {
        prop_assert!(phi.is_compatible(&phi), "Phi-features should be self-compatible");
    }

    #[test]
    fn number_compatibility_reflexive(n in arb_number()) {
        prop_assert!(n.is_compatible(&n), "Number should be self-compatible");
    }

    #[test]
    fn gender_compatibility_reflexive(g in arb_gender()) {
        prop_assert!(g.is_compatible(&g), "Gender should be self-compatible");
    }

    // =========================================================================
    // Symmetry: Compatibility is symmetric
    // =========================================================================

    /// Number compatibility is symmetric
    ///
    /// If A is compatible with B, then B is compatible with A.
    /// This ensures coreference linking is order-independent.
    #[test]
    fn number_compatibility_symmetric(n1 in arb_number(), n2 in arb_number()) {
        prop_assert_eq!(
            n1.is_compatible(&n2),
            n2.is_compatible(&n1),
            "Number compatibility must be symmetric"
        );
    }

    /// Gender compatibility is symmetric
    #[test]
    fn gender_compatibility_symmetric(g1 in arb_gender(), g2 in arb_gender()) {
        prop_assert_eq!(
            g1.is_compatible(&g2),
            g2.is_compatible(&g1),
            "Gender compatibility must be symmetric"
        );
    }

    /// Person compatibility is symmetric
    #[test]
    fn person_compatibility_symmetric(p1 in arb_person(), p2 in arb_person()) {
        prop_assert_eq!(
            p1.is_compatible(&p2),
            p2.is_compatible(&p1),
            "Person compatibility must be symmetric"
        );
    }

    // =========================================================================
    // Unknown/Neutral as Wildcards
    // =========================================================================

    /// Unknown number is compatible with all numbers
    ///
    /// This models singular "they", ambiguous "you", and languages
    /// where number is not grammatically marked.
    #[test]
    fn unknown_number_is_wildcard(n in arb_number()) {
        prop_assert!(
            Number::Unknown.is_compatible(&n),
            "Unknown should be compatible with {:?}",
            n
        );
    }

    /// Unknown gender is compatible with all genders
    ///
    /// This models neopronouns, epicene nouns, and contexts
    /// where gender is not determined.
    #[test]
    fn unknown_gender_is_wildcard(g in arb_gender()) {
        prop_assert!(
            Gender::Unknown.is_compatible(&g),
            "Unknown should be compatible with {:?}",
            g
        );
    }

    /// Neutral gender is compatible with all genders
    ///
    /// "They" can refer to anyone regardless of gender.
    /// This is the singular "they" principle.
    #[test]
    fn neutral_gender_is_wildcard(g in arb_gender()) {
        prop_assert!(
            Gender::Neutral.is_compatible(&g),
            "Neutral (they) should be compatible with {:?}",
            g
        );
    }

    // =========================================================================
    // Binary Gender Exclusion
    // =========================================================================

    /// Masculine and Feminine are mutually exclusive
    ///
    /// This is the core binary gender constraint. "John... she" fails.
    #[test]
    fn binary_gender_exclusion(_ignored: ()) {
        prop_assert!(!Gender::Masculine.is_compatible(&Gender::Feminine));
        prop_assert!(!Gender::Feminine.is_compatible(&Gender::Masculine));
    }

    // =========================================================================
    // Person is a Hard Constraint (Cross-Linguistic Universal)
    // =========================================================================

    /// Different persons are never compatible
    ///
    /// "I went home. She ate dinner." - "I" and "She" cannot corefer.
    /// This is true in ALL known human languages.
    #[test]
    fn person_exclusion(p1 in arb_person(), p2 in arb_person()) {
        // Person::Unknown is a wildcard (compatible with everything),
        // so only assert incompatibility when neither variant is Unknown.
        if p1 != p2 && p1 != Person::Unknown && p2 != Person::Unknown {
            prop_assert!(
                !p1.is_compatible(&p2),
                "Different concrete persons {:?} and {:?} should not be compatible",
                p1,
                p2
            );
        }
    }

    // =========================================================================
    // Dual-Plural Compatibility (Semitic/Sanskrit/Ancient Greek)
    // =========================================================================

    /// Dual is compatible with plural
    ///
    /// In Arabic, Hebrew, Sanskrit: a pair (dual) can be referred to with
    /// plural pronouns. This is a cross-linguistic pattern.
    #[test]
    fn dual_plural_compatibility(_ignored: ()) {
        prop_assert!(Number::Dual.is_compatible(&Number::Plural));
        prop_assert!(Number::Plural.is_compatible(&Number::Dual));
    }

    /// Singular is NOT compatible with plural/dual
    ///
    /// "The dog... they" is ungrammatical (unless singular they).
    /// This models strict number agreement.
    #[test]
    fn singular_plural_exclusion(_ignored: ()) {
        prop_assert!(!Number::Singular.is_compatible(&Number::Plural));
        prop_assert!(!Number::Singular.is_compatible(&Number::Dual));
    }

    // =========================================================================
    // MentionType Ordering (Accessibility Hierarchy)
    // =========================================================================

    /// MentionType ordering is total (all pairs are comparable)
    #[test]
    fn mention_type_total_order(t1 in arb_mention_type(), t2 in arb_mention_type()) {
        // Total order means exactly one of: t1 < t2, t1 == t2, t1 > t2
        let lt = t1 < t2;
        let eq = t1 == t2;
        let gt = t1 > t2;
        // Exactly one of the three conditions must be true
        let count = (lt as u8) + (eq as u8) + (gt as u8);
        prop_assert_eq!(count, 1, "Exactly one ordering relation should hold");
    }

    /// MentionType ordering is transitive
    #[test]
    fn mention_type_transitive(
        t1 in arb_mention_type(),
        t2 in arb_mention_type(),
        t3 in arb_mention_type()
    ) {
        if t1 <= t2 && t2 <= t3 {
            prop_assert!(t1 <= t3, "Ordering should be transitive");
        }
    }

    /// Proper nouns are always >= other types (canonical mention selection)
    #[test]
    fn proper_is_maximal(t in arb_mention_type()) {
        prop_assert!(
            MentionType::Proper >= t,
            "Proper should be >= {:?}",
            t
        );
    }

    /// Zero pronouns are always <= other types (least informative)
    #[test]
    fn zero_is_minimal(t in arb_mention_type()) {
        prop_assert!(
            MentionType::Zero <= t,
            "Zero should be <= {:?}",
            t
        );
    }

    // =========================================================================
    // Salience Weight Bounds
    // =========================================================================

    /// Salience weights are in valid range (0, 1]
    #[test]
    fn salience_weight_bounded(t in arb_mention_type()) {
        let w = t.salience_weight();
        prop_assert!(w > 0.0, "Salience should be positive");
        prop_assert!(w <= 1.0, "Salience should be <= 1.0");
    }

    /// Salience ordering matches accessibility hierarchy
    #[test]
    fn salience_matches_accessibility(t1 in arb_mention_type(), t2 in arb_mention_type()) {
        if t1 > t2 {
            prop_assert!(
                t1.salience_weight() >= t2.salience_weight(),
                "Higher accessibility should mean higher salience"
            );
        }
    }

    // =========================================================================
    // Anaphoricity Constraints
    // =========================================================================

    /// Anaphoric types require antecedents
    ///
    /// Pronouns and zeros cannot introduce entities.
    #[test]
    fn anaphoric_types_require_antecedent(t in arb_mention_type()) {
        if t == MentionType::Pronominal || t == MentionType::Zero {
            prop_assert!(t.requires_antecedent(), "{:?} should require antecedent", t);
            prop_assert!(!t.can_introduce_entity(), "{:?} should not introduce entities", t);
        }
    }

    /// Only Zero has no surface form
    #[test]
    fn only_zero_is_surfaceless(t in arb_mention_type()) {
        if t == MentionType::Zero {
            prop_assert!(!t.has_surface_form(), "Zero should have no surface form");
        } else {
            prop_assert!(t.has_surface_form(), "{:?} should have surface form", t);
        }
    }
}
