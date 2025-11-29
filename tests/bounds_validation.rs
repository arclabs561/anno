//! Test for bounds validation bugs in entity creation and usage.

use anno::{Entity, EntityType};

#[test]
fn test_entity_with_invalid_bounds() {
    let text = "Hello World";
    let char_len = text.chars().count();

    // Create entity with out-of-bounds end
    let bad_entity = Entity::new("World", EntityType::Location, 6, char_len + 10, 0.9);

    // This should be caught by validation
    let issues = bad_entity.validate(text);
    assert!(!issues.is_empty(), "Should detect out-of-bounds entity");

    // But the entity was created successfully - this is the bug
    // The entity can be used before validation, causing panics
}

#[test]
fn test_entity_text_extraction_panic() {
    let text = "Hello World";

    // Create entity with invalid bounds
    let bad_entity = Entity::new("World", EntityType::Location, 6, 100, 0.9);

    // This will panic if we try to extract text using char offsets
    // because we're using character offsets but text slicing needs byte offsets
    // Actually, wait - Entity uses char offsets, so we need to convert

    // The real issue: if someone tries to use the entity without validation
    // and assumes the offsets are valid, they might panic
    let issues = bad_entity.validate(text);
    assert!(!issues.is_empty());
}

#[test]
fn test_entity_start_greater_than_end() {
    let text = "Hello World";

    // Create entity with start > end
    let bad_entity = Entity::new("World", EntityType::Location, 10, 6, 0.9);

    let issues = bad_entity.validate(text);
    assert!(!issues.is_empty(), "Should detect invalid span");
}

#[test]
fn test_entity_validate_boundary_conditions() {
    let text = "Hello World";
    let char_count = text.chars().count();

    // Test boundary: seg.end == char_count (should be valid)
    let entity_at_end = Entity::new("World", EntityType::Location, 6, char_count, 0.9);
    let issues_at_end = entity_at_end.validate(text);
    // Should be valid - end == char_count is allowed
    let has_out_of_bounds = issues_at_end
        .iter()
        .any(|i| matches!(i, anno::ValidationIssue::SpanOutOfBounds { .. }));
    assert!(
        !has_out_of_bounds,
        "Entity ending at text boundary should be valid"
    );

    // Test boundary: seg.end > char_count (should fail)
    let entity_over_end = Entity::new("World", EntityType::Location, 6, char_count + 1, 0.9);
    let issues_over_end = entity_over_end.validate(text);
    let has_out_of_bounds = issues_over_end
        .iter()
        .any(|i| matches!(i, anno::ValidationIssue::SpanOutOfBounds { .. }));
    assert!(
        has_out_of_bounds,
        "Entity ending after text boundary should fail validation"
    );

    // Test with discontinuous span boundary condition
    use anno::DiscontinuousSpan;
    let mut entity = Entity::new("test", EntityType::Person, 0, 4, 0.9);
    // Create discontinuous span where one segment ends exactly at boundary
    let disc_span = DiscontinuousSpan::new(vec![0..4, char_count..char_count + 1]);
    entity.set_discontinuous_span(disc_span);
    let issues = entity.validate(text);
    // The second segment is out of bounds
    let has_out_of_bounds = issues
        .iter()
        .any(|i| matches!(i, anno::ValidationIssue::SpanOutOfBounds { .. }));
    assert!(
        has_out_of_bounds,
        "Discontinuous span with out-of-bounds segment should fail"
    );
}

#[test]
fn test_entity_text_mismatch() {
    let text = "Hello World";

    // Create entity where stored text doesn't match span
    let bad_entity = Entity::new("Wrong", EntityType::Location, 0, 5, 0.9);

    let issues = bad_entity.validate(text);
    // Should detect text mismatch
    assert!(issues
        .iter()
        .any(|i| matches!(i, anno::ValidationIssue::TextMismatch { .. })));
}
