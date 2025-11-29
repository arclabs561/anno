//! Grounded entity representation with Signal → Track → Identity hierarchy.
//!
//! This example demonstrates the unified abstraction for entity detection that
//! works across modalities (text, visual, audio, etc.).
//!
//! # Research Motivation
//!
//! Traditional NER conflates three distinct levels:
//! 1. **Signal** (Level 1): Raw detections - "there's something here"
//! 2. **Track** (Level 2): Within-document coreference - "these are the same entity"
//! 3. **Identity** (Level 3): Cross-document linking - "this is Q7186 in Wikidata"
//!
//! This separation enables:
//! - Better embedding alignment (mentions vs KB entries have different representations)
//! - Efficient streaming (signals can be processed incrementally)
//! - Clear evaluation (each level can be evaluated separately)
//!
//! # The Detection Isomorphism
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    VISION                    TEXT (NER)              │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ Localization Unit  │ BoundingBox (x,y,w,h)  │ TextSpan (start,end)  │
//! │ Signal             │ Detection              │ Mention               │
//! │ Track (Level 2)    │ Tracklet (MOT)         │ CorefChain            │
//! │ Identity (Level 3) │ Face Recognition       │ Entity Linking        │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Run with: `cargo run --example grounded`

use anno::grounded::{GroundedDocument, Identity, Location, Modality, Quantifier, Signal, Track};
use anno::{Entity, EntityType, HierarchicalConfidence};

fn main() {
    println!("=== Grounded Entity Representation ===\n");

    // -------------------------------------------------------------------------
    // Part 1: Understanding Locations (The Universal Localization Unit)
    // -------------------------------------------------------------------------
    println!("--- Part 1: Locations ---\n");

    // Text location: 1D interval
    let text_loc = Location::text(0, 12);
    println!("Text location: {:?}", text_loc);
    println!("  Modality: {:?}", text_loc.modality());
    println!("  Text offsets: {:?}", text_loc.text_offsets());

    // Bounding box: 2D rectangle
    let bbox_loc = Location::bbox(0.1, 0.2, 0.3, 0.4);
    println!("\nBounding box: {:?}", bbox_loc);
    println!("  Modality: {:?}", bbox_loc.modality());

    // IoU (Intersection over Union) works for compatible types
    let loc1 = Location::text(0, 10);
    let loc2 = Location::text(5, 15);
    let iou = loc1.iou(&loc2).unwrap();
    println!("\nIoU([0,10), [5,15)) = {:.3}", iou);
    // Intersection: [5,10) = 5 chars
    // Union: [0,15) = 15 chars
    // IoU = 5/15 = 0.333

    // -------------------------------------------------------------------------
    // Part 2: Signals (Level 1 - Raw Detections)
    // -------------------------------------------------------------------------
    println!("\n--- Part 2: Signals (Level 1) ---\n");

    // A signal is the atomic unit of detection
    let signal1: Signal<Location> =
        Signal::new(0, Location::text(0, 12), "Marie Curie", "Person", 0.95);
    println!("Signal 1: {:?}", signal1.surface);
    println!("  Label: {}", signal1.label);
    println!("  Confidence: {:.2}", signal1.confidence);

    // Signals can have linguistic features (only relevant for symbolic modality)
    let negated_signal: Signal<Location> = Signal::new(
        1,
        Location::text(0, 14),
        "not a scientist",
        "Occupation",
        0.8,
    )
    .negated()
    .with_modality(Modality::Symbolic);

    println!("\nNegated signal: {:?}", negated_signal.surface);
    println!("  Negated: {}", negated_signal.negated);
    println!(
        "  Modality supports linguistic features: {}",
        negated_signal.modality.supports_linguistic_features()
    );

    // Quantified signal
    let quantified: Signal<Location> =
        Signal::new(2, Location::text(0, 14), "every employee", "Person", 0.7)
            .with_quantifier(Quantifier::Universal);

    println!("\nQuantified signal: {:?}", quantified.surface);
    println!("  Quantifier: {:?}", quantified.quantifier);

    // -------------------------------------------------------------------------
    // Part 3: Tracks (Level 2 - Within-Document Coreference)
    // -------------------------------------------------------------------------
    println!("\n--- Part 3: Tracks (Level 2) ---\n");

    let mut track = Track::new(0, "Marie Curie");
    track.add_signal(0, 0); // "Marie Curie" at position 0
    track.add_signal(1, 1); // "She" at position 1
    track.add_signal(2, 2); // "the physicist" at position 2

    println!("Track: {:?}", track.canonical_surface);
    println!("  Signals: {} mentions", track.len());
    println!("  Is singleton: {}", track.is_singleton());

    // -------------------------------------------------------------------------
    // Part 4: Identities (Level 3 - Knowledge Base Linking)
    // -------------------------------------------------------------------------
    println!("\n--- Part 4: Identities (Level 3) ---\n");

    let mut identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186").with_type("Person");
    identity.add_alias("Maria Sklodowska");
    identity.add_alias("Madame Curie");

    println!("Identity: {:?}", identity.canonical_name);
    println!("  KB: {:?} / {:?}", identity.kb_name, identity.kb_id);
    println!("  Aliases: {:?}", identity.aliases);

    // -------------------------------------------------------------------------
    // Part 5: GroundedDocument - The Complete Picture
    // -------------------------------------------------------------------------
    println!("\n--- Part 5: GroundedDocument ---\n");

    let text = "Marie Curie won the Nobel Prize in Physics. She was a pioneering physicist.";
    println!("Document: \"{}\"\n", text);

    let mut doc = GroundedDocument::new("doc1", text);

    // Add signals (Level 1)
    let s1 = doc.add_signal(Signal::new(
        0,
        Location::text(0, 12),
        "Marie Curie",
        "Person",
        0.95,
    ));
    let s2 = doc.add_signal(Signal::new(
        1,
        Location::text(44, 47),
        "She",
        "Person",
        0.88,
    ));
    let s3 = doc.add_signal(Signal::new(
        2,
        Location::text(17, 29),
        "Nobel Prize",
        "Award",
        0.92,
    ));

    println!("Added {} signals", doc.signals().len());

    // Form tracks (Level 2)
    let mut track1 = Track::new(0, "Marie Curie");
    track1.add_signal(s1, 0);
    track1.add_signal(s2, 1);
    let track1_id = doc.add_track(track1);

    let mut track2 = Track::new(1, "Nobel Prize");
    track2.add_signal(s3, 0);
    doc.add_track(track2);

    println!("Formed {} tracks", doc.tracks().count());

    // Add identity and link (Level 3)
    let identity = Identity::from_kb(0, "Marie Curie", "wikidata", "Q7186");
    let identity_id = doc.add_identity(identity);
    doc.link_track_to_identity(track1_id, identity_id);

    println!("Linked to {} identities", doc.identities().count());

    // Traverse the hierarchy
    println!("\nHierarchy traversal:");
    for signal in doc.signals() {
        let track = doc.track_for_signal(signal.id);
        let identity = doc.identity_for_signal(signal.id);

        println!(
            "  Signal '{}' (confidence: {:.2})",
            signal.surface, signal.confidence
        );
        if let Some(t) = track {
            println!(
                "    → Track '{}' ({} mentions)",
                t.canonical_surface,
                t.len()
            );
        }
        if let Some(i) = identity {
            println!("    → Identity '{}' (KB: {:?})", i.canonical_name, i.kb_id);
        }
    }

    // -------------------------------------------------------------------------
    // Part 6: Conversion to/from Legacy Entity Format
    // -------------------------------------------------------------------------
    println!("\n--- Part 6: Legacy Compatibility ---\n");

    // Convert to legacy Entity format
    let entities = doc.to_entities();
    println!("Converted to {} legacy entities", entities.len());

    for entity in &entities {
        println!(
            "  Entity: '{}' ({}) at [{}, {})",
            entity.text, entity.entity_type, entity.start, entity.end
        );
        if let Some(kb_id) = &entity.kb_id {
            println!("    KB ID: {}", kb_id);
        }
    }

    // Convert back from legacy entities
    let legacy_entities = vec![
        Entity::new("Albert Einstein", EntityType::Person, 0, 15, 0.9),
        Entity::new(
            "physicist",
            EntityType::Other("Occupation".into()),
            30,
            39,
            0.85,
        ),
    ];

    let doc2 = GroundedDocument::from_entities(
        "doc2",
        "Albert Einstein was a brilliant physicist.",
        &legacy_entities,
    );
    println!(
        "\nConverted from {} legacy entities:",
        legacy_entities.len()
    );
    println!("  → {} signals", doc2.signals().len());
    println!("  → {} tracks", doc2.tracks().count());

    // -------------------------------------------------------------------------
    // Part 7: The Semiotic Gap - Icon vs Symbol
    // -------------------------------------------------------------------------
    println!("\n--- Part 7: Modality Distinction ---\n");

    println!("The semiotic gap between modalities:\n");

    let iconic = Modality::Iconic;
    let symbolic = Modality::Symbolic;
    let hybrid = Modality::Hybrid;

    println!("Iconic (vision):");
    println!(
        "  Supports linguistic features: {}",
        iconic.supports_linguistic_features()
    );
    println!(
        "  Supports geometric features: {}",
        iconic.supports_geometric_features()
    );

    println!("\nSymbolic (text):");
    println!(
        "  Supports linguistic features: {}",
        symbolic.supports_linguistic_features()
    );
    println!(
        "  Supports geometric features: {}",
        symbolic.supports_geometric_features()
    );

    println!("\nHybrid (OCR):");
    println!(
        "  Supports linguistic features: {}",
        hybrid.supports_linguistic_features()
    );
    println!(
        "  Supports geometric features: {}",
        hybrid.supports_geometric_features()
    );

    println!("\nKey insight: Linguistic features like negation ('not a doctor'),");
    println!("quantification ('every employee'), and recursion ('the claim that...')");
    println!("only apply to symbolic modalities. Visual detection is about geometry.");

    // -------------------------------------------------------------------------
    // Part 8: Document Statistics
    // -------------------------------------------------------------------------
    println!("\n--- Part 8: Document Statistics ---\n");

    let stats = doc.stats();
    println!("{}", stats);

    // Show filtering capabilities
    println!("Text signals: {}", doc.text_signals().len());
    println!(
        "High-confidence signals (>0.9): {}",
        doc.confident_signals(0.9).len()
    );
    println!("Linked tracks: {}", doc.linked_tracks().count());
    println!("Unlinked tracks: {}", doc.unlinked_tracks().count());

    // -------------------------------------------------------------------------
    // Part 9: Batch Operations
    // -------------------------------------------------------------------------
    println!("\n--- Part 9: Batch Operations ---\n");

    let mut batch_doc = GroundedDocument::new(
        "batch_demo",
        "John went to Paris. He loved it. Paris is beautiful.",
    );

    // Batch add signals
    let signals = vec![
        Signal::new(0, Location::text(0, 4), "John", "Person", 0.95),
        Signal::new(0, Location::text(13, 18), "Paris", "Location", 0.92),
        Signal::new(0, Location::text(20, 22), "He", "Person", 0.88),
        Signal::new(0, Location::text(33, 38), "Paris", "Location", 0.90),
    ];
    let ids = batch_doc.add_signals(signals);
    println!("Batch added {} signals", ids.len());

    // Create tracks using the helper
    let john_track = batch_doc.create_track_from_signals("John", &[ids[0], ids[2]]);
    let paris_track = batch_doc.create_track_from_signals("Paris", &[ids[1], ids[3]]);
    println!(
        "Created tracks: John={:?}, Paris={:?}",
        john_track, paris_track
    );

    // Find overlapping signals (useful for nested entity detection)
    let overlaps = batch_doc.find_overlapping_signal_pairs();
    println!("Overlapping signal pairs: {}", overlaps.len());

    // Filter signals by range
    let in_first_sentence = batch_doc.signals_in_range(0, 19);
    println!(
        "Signals in first sentence: {:?}",
        in_first_sentence
            .iter()
            .map(|s| &s.surface)
            .collect::<Vec<_>>()
    );

    // -------------------------------------------------------------------------
    // Part 10: Hierarchical Confidence
    // -------------------------------------------------------------------------
    println!("\n--- Part 10: Hierarchical Confidence ---\n");

    let hier_conf = HierarchicalConfidence::new(
        0.95, // linkage: "is there an entity here?"
        0.88, // type: "is it a Person?"
        0.92, // boundary: "are the boundaries correct?"
    );

    println!("Hierarchical confidence:");
    println!("  Linkage (any entity?): {:.2}", hier_conf.linkage);
    println!("  Type (correct type?): {:.2}", hier_conf.type_score);
    println!("  Boundary (correct span?): {:.2}", hier_conf.boundary);
    println!("  Combined (geometric mean): {:.2}", hier_conf.combined());

    let mut signal_with_hier = Signal::new(0, Location::text(0, 12), "Marie Curie", "Person", 0.9);
    signal_with_hier.hierarchical = Some(hier_conf);

    println!("\nSignal with hierarchical confidence:");
    if let Some(h) = &signal_with_hier.hierarchical {
        println!(
            "  Passes threshold (0.8, 0.8, 0.8): {}",
            h.passes_threshold(0.8, 0.8, 0.8)
        );
    }

    // -------------------------------------------------------------------------
    // Part 11: Spatial Index for Efficient Range Queries
    // -------------------------------------------------------------------------
    println!("\n--- Part 11: Spatial Index ---\n");

    // For documents with many signals, the spatial index provides O(log n + k) queries
    let index = batch_doc.build_text_index();
    println!("Built spatial index with {} entries", index.len());

    // Query signals in a range
    let in_range = index.query_contained_in(0, 20);
    println!("Signals fully in [0, 20): {:?}", in_range);

    // Query overlapping signals
    let overlapping = index.query_overlap(10, 25);
    println!("Signals overlapping [10, 25): {:?}", overlapping);

    // -------------------------------------------------------------------------
    // Part 12: HTML Visualization
    // -------------------------------------------------------------------------
    println!("\n--- Part 12: HTML Visualization ---\n");

    // Generate HTML for analysis
    let html = anno::grounded::render_document_html(&doc);
    println!("Generated {} bytes of HTML", html.len());

    // In production, you'd write this to a file:
    // std::fs::write("grounded_analysis.html", &html).unwrap();
    println!("HTML preview (first 500 chars):");
    println!("{}", &html[..500.min(html.len())]);

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("\n=== Summary ===\n");
    println!("The Signal → Track → Identity hierarchy provides:");
    println!("1. Clear separation of detection, coreference, and linking");
    println!("2. Unified treatment of text and visual signals");
    println!("3. Efficient streaming/incremental processing");
    println!("4. Better embedding alignment for RAG applications");
    println!("5. Linguistic features (negation, quantification) for text");
    println!("6. Backwards compatibility with legacy Entity format");
    println!("7. Document statistics for analysis and debugging");
    println!("8. Batch operations for efficient bulk processing");
    println!("9. Overlap detection for nested entity handling");
    println!("10. Spatial indexing for O(log n) range queries");
    println!("11. HTML visualization for debugging and analysis");
}
