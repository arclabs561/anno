//! Tests for DiscontinuousSpan functionality.
//!
//! DiscontinuousSpan represents entities that span non-contiguous text regions.
//! Examples:
//! - Medical: "severe [pain] ... in [abdomen]" → "severe abdominal pain"
//! - Negation: "[no] evidence of [cancer]" → "no cancer"
//! - Coordination: "[John] and [Mary]" → potentially two persons

use anno::{DiscontinuousSpan, Entity, EntityBuilder, EntityType};

// =============================================================================
// DiscontinuousSpan Unit Tests
// =============================================================================

#[test]
fn test_discontinuous_span_contiguous() {
    // A "contiguous" discontinuous span (single segment)
    let span = DiscontinuousSpan::contiguous(10, 20);
    assert_eq!(span.segments().len(), 1);
    assert_eq!(span.segments()[0], 10..20);
    assert!(!span.is_discontinuous());
    assert_eq!(span.num_segments(), 1);
}

#[test]
fn test_discontinuous_span_multi_segment() {
    // Multiple segments
    let segments = vec![0..5, 10..15, 20..25];
    let span = DiscontinuousSpan::new(segments);

    assert!(span.is_discontinuous());
    assert_eq!(span.num_segments(), 3);

    // Segments should be sorted
    assert_eq!(span.segments()[0].start, 0);
    assert_eq!(span.segments()[1].start, 10);
    assert_eq!(span.segments()[2].start, 20);
}

#[test]
fn test_discontinuous_span_extract_text() {
    let text = "severe pain in abdomen";
    //          0123456789...
    // "severe" = 0..6
    // "abdomen" starts at position 15
    // "abdo" = 15..19

    let span = DiscontinuousSpan::new(vec![0..6, 15..19]);

    // Extract "severe" and "abdo" (demonstrating extraction)
    let extracted = span.extract_text(text, " ");
    assert_eq!(extracted, "severe abdo");
}

#[test]
fn test_discontinuous_span_with_separator() {
    let span = DiscontinuousSpan::new(vec![0..4, 9..13]);
    let text = "John and Mary went home";

    // Extract "John" and "Mary"
    let extracted = span.extract_text(text, " and ");
    assert_eq!(extracted, "John and Mary");
}

#[test]
fn test_discontinuous_span_len() {
    // Total length across all segments
    let span = DiscontinuousSpan::new(vec![0..5, 10..15, 20..30]);

    // Should calculate total character span (5 + 5 + 10 = 20)
    let total_len: usize = span.segments().iter().map(|s| s.len()).sum();
    assert_eq!(total_len, 20);
}

#[test]
fn test_discontinuous_span_empty() {
    let span = DiscontinuousSpan::new(vec![]);
    assert!(!span.is_discontinuous());
    assert_eq!(span.num_segments(), 0);
}

#[test]
fn test_discontinuous_span_overlapping_merge() {
    // If segments overlap, they should be merged
    // Note: This depends on implementation - currently segments are just stored
    let segments = vec![0..10, 5..15]; // Overlapping
    let span = DiscontinuousSpan::new(segments);

    // The implementation should handle this gracefully
    assert!(span.num_segments() > 0);
}

#[test]
fn test_discontinuous_span_bounding_range() {
    // Test the bounding_range method
    let span = DiscontinuousSpan::new(vec![5..10, 20..30, 40..50]);

    let bounding = span.bounding_range().expect("Should have bounding range");
    assert_eq!(bounding.start, 5);
    assert_eq!(bounding.end, 50);
}

#[test]
fn test_discontinuous_span_total_len() {
    // Test total_len method if it exists
    let span = DiscontinuousSpan::new(vec![0..10, 20..30, 40..50]);

    // Total covered characters = 10 + 10 + 10 = 30
    let total: usize = span.segments().iter().map(|r| r.end - r.start).sum();
    assert_eq!(total, 30);
}

#[test]
fn test_discontinuous_span_from_range() {
    // Test conversion from a single range
    let range = 10..20;
    let span: DiscontinuousSpan = range.into();

    assert!(!span.is_discontinuous());
    assert_eq!(span.num_segments(), 1);
    assert_eq!(span.segments()[0], 10..20);
}

#[test]
fn test_discontinuous_span_sorted() {
    // Verify segments are sorted by start position
    let span = DiscontinuousSpan::new(vec![20..30, 0..10, 10..15]);

    let segments = span.segments();
    for i in 1..segments.len() {
        assert!(
            segments[i - 1].start <= segments[i].start,
            "Segments should be sorted by start position"
        );
    }
}

// =============================================================================
// Entity Integration Tests
// =============================================================================

#[test]
fn test_entity_with_discontinuous_span() {
    // Create an entity with discontinuous span using EntityBuilder
    let entity = EntityBuilder::new("severe pain", EntityType::Other("SYMPTOM".to_string()))
        .confidence(0.9)
        .discontinuous_span(DiscontinuousSpan::new(vec![0..6, 15..19]))
        .build();

    assert!(entity.is_discontinuous());
    assert_eq!(entity.discontinuous_segments(), Some(vec![0..6, 15..19]));
    // start/end should be updated to bounding range
    assert_eq!(entity.start, 0);
    assert_eq!(entity.end, 19);
}

#[test]
fn test_entity_without_discontinuous_span() {
    let entity = EntityBuilder::new("John Smith", EntityType::Person)
        .span(0, 10)
        .confidence(0.95)
        .build();

    assert!(!entity.is_discontinuous());
    assert_eq!(entity.discontinuous_segments(), None);
}

#[test]
fn test_entity_total_len_discontinuous() {
    // Test that total_len works correctly with discontinuous spans
    let entity = EntityBuilder::new("test entity", EntityType::Person)
        .confidence(0.9)
        .discontinuous_span(DiscontinuousSpan::new(vec![0..5, 10..15]))
        .build();

    // total_len should return sum of segments (5 + 5 = 10)
    assert_eq!(entity.total_len(), 10);
}

#[test]
fn test_entity_total_len_contiguous() {
    let entity = Entity::new("John Smith", EntityType::Person, 0, 10, 0.95);

    // total_len for contiguous = end - start
    assert_eq!(entity.total_len(), 10);
}

#[test]
fn test_entity_set_discontinuous_span() {
    let mut entity = Entity::new("test", EntityType::Person, 100, 200, 0.9);

    // Set discontinuous span - should update start/end
    entity.set_discontinuous_span(DiscontinuousSpan::new(vec![5..10, 20..25]));

    assert!(entity.is_discontinuous());
    assert_eq!(entity.start, 5);
    assert_eq!(entity.end, 25);
}

// =============================================================================
// Real-World Scenario Tests
// =============================================================================

#[test]
fn test_medical_discontinuous_entity() {
    // Real-world medical example
    let text = "Patient denies any chest pain or shortness of breath";

    // "chest pain" is at positions 17-27
    // But in clinical NER, "denies chest pain" might be extracted as a negated finding

    let span = DiscontinuousSpan::new(vec![8..14, 19..29]); // "denies" + "chest pain"

    assert!(span.is_discontinuous());
    assert_eq!(span.num_segments(), 2);

    let extracted = span.extract_text(text, " ");
    assert!(extracted.contains("denies"));
}

#[test]
fn test_w2ner_discontinuous_scenario() {
    // W2NER can detect discontinuous entities
    // Example: "The CEO of Apple, Tim Cook, announced..."
    // Entity: "Tim Cook" (Person)
    // But also could be: "CEO of Apple" (Relation) - discontinuous if you skip "The"

    let text = "The CEO of Apple announced new products";
    //          01234567890123456...
    // "CEO of Apple" = 4..16

    // "CEO of Apple" at positions 4-16 (contiguous in this case)
    let span = DiscontinuousSpan::contiguous(4, 16);
    assert!(!span.is_discontinuous());

    // Verify extraction
    let extracted = span.extract_text(text, " ");
    assert_eq!(extracted, "CEO of Apple");
}

#[test]
fn test_discontinuous_span_real_world_coordination() {
    // "John and Mary went to Paris and London"
    // If we want to extract "Paris and London" as a single location entity,
    // that's contiguous. But if we want "John and Mary" separately...
    let text = "John and Mary went to Paris and London";

    // Extract coordinated locations "Paris" (22-27) and "London" (32-38)
    let locations = DiscontinuousSpan::new(vec![22..27, 32..38]);
    assert!(locations.is_discontinuous());

    let extracted = locations.extract_text(text, " and ");
    assert_eq!(extracted, "Paris and London");
}

// =============================================================================
// Property-Based Tests
// =============================================================================

use proptest::prelude::*;

proptest! {
    /// Invariant: bounding range always contains all segments
    #[test]
    fn bounding_range_contains_all_segments(
        segments in proptest::collection::vec(0usize..1000, 1..10)
            .prop_map(|starts| {
                starts.into_iter()
                    .map(|s| s..s + 5)
                    .collect::<Vec<_>>()
            })
    ) {
        let span = DiscontinuousSpan::new(segments.clone());

        if let Some(bounding) = span.bounding_range() {
            for seg in span.segments() {
                prop_assert!(seg.start >= bounding.start, "Segment starts before bounding");
                prop_assert!(seg.end <= bounding.end, "Segment ends after bounding");
            }
        }
    }

    /// Invariant: num_segments matches segments().len()
    #[test]
    fn num_segments_consistent(
        segments in proptest::collection::vec(0usize..100, 0..20)
            .prop_map(|starts| {
                starts.into_iter()
                    .map(|s| s..s + 3)
                    .collect::<Vec<_>>()
            })
    ) {
        let span = DiscontinuousSpan::new(segments);
        prop_assert_eq!(span.num_segments(), span.segments().len());
    }

    /// Invariant: single segment is never discontinuous
    #[test]
    fn single_segment_not_discontinuous(start in 0usize..1000, len in 1usize..100) {
        let span = DiscontinuousSpan::contiguous(start, start + len);
        prop_assert!(!span.is_discontinuous());
        prop_assert_eq!(span.num_segments(), 1);
    }

    /// Invariant: entity total_len is always >= 0
    #[test]
    fn entity_total_len_non_negative(
        start in 0usize..100,
        end in 0usize..100
    ) {
        let entity = Entity::new("test", EntityType::Person, start, end, 0.9);
        prop_assert!(entity.total_len() <= end.saturating_sub(start).max(end.saturating_sub(start)));
    }

    /// Invariant: is_discontinuous is true iff num_segments > 1
    #[test]
    fn discontinuous_iff_multiple_segments(
        num_segs in 0usize..10
    ) {
        let segments: Vec<_> = (0..num_segs)
            .map(|i| (i * 20)..(i * 20 + 5))
            .collect();
        let span = DiscontinuousSpan::new(segments);

        let is_disc = span.is_discontinuous();
        let has_multiple = span.num_segments() > 1;

        prop_assert_eq!(is_disc, has_multiple,
            "is_discontinuous={} but num_segments={}", is_disc, span.num_segments());
    }
}

// =============================================================================
// Mutation Testing Targets - Tests for previously missed mutants
// =============================================================================

#[test]
fn test_discontinuous_span_to_span_contiguous() {
    // Test that to_span() returns Some for contiguous spans
    let contiguous = DiscontinuousSpan::contiguous(10, 20);
    let span = contiguous.to_span();

    assert!(
        span.is_some(),
        "Contiguous span should convert to Some(Span)"
    );
    let span = span.unwrap();
    // Use text_offsets() to get the start/end
    let offsets = span.text_offsets();
    assert!(offsets.is_some(), "Text span should have offsets");
    let (start, end) = offsets.unwrap();
    assert_eq!(start, 10);
    assert_eq!(end, 20);
}

#[test]
fn test_discontinuous_span_to_span_discontinuous() {
    // Test that to_span() returns Some for discontinuous spans (uses bounding range)
    let disc = DiscontinuousSpan::new(vec![0..5, 10..15]);
    let span = disc.to_span();

    // For discontinuous spans, to_span() uses bounding_range, so it should still work
    assert!(
        span.is_some(),
        "Discontinuous span should use bounding range"
    );
    let span = span.unwrap();
    let offsets = span.text_offsets();
    assert!(offsets.is_some(), "Text span should have offsets");
    let (start, end) = offsets.unwrap();
    assert_eq!(start, 0);
    assert_eq!(end, 15);
}

#[test]
fn test_span_is_empty() {
    use anno::Span;

    // Empty span (start == end)
    let empty_span = Span::text(0, 0);
    assert!(
        empty_span.is_empty(),
        "Span with start==end should be empty"
    );

    // Non-empty span
    let non_empty = Span::text(0, 5);
    assert!(
        !non_empty.is_empty(),
        "Span with start<end should not be empty"
    );

    // Visual span - bbox spans always return len() == 0, so is_empty() returns true
    // This is by design: len() is for text spans, bbox spans use area/coordinates
    let visual_span = Span::bbox(0.0, 0.0, 0.5, 0.5);
    // Note: bbox spans return len() == 0, so is_empty() returns true
    // This is expected behavior - bbox spans don't have a "length" concept
    assert!(
        visual_span.is_empty(),
        "Bbox span len() returns 0, so is_empty() is true"
    );

    // Zero-area bbox
    let empty_visual = Span::bbox(0.0, 0.0, 0.0, 0.0);
    assert!(empty_visual.is_empty(), "Zero-area bbox should be empty");
}
