//! Performance invariants: ensure optimizations don't regress performance.
//!
//! These tests verify that optimized code paths are at least as fast as
//! unoptimized versions, and that performance characteristics are maintained.

use anno::{Entity, EntityType, Model, StackedNER};
use std::time::Instant;

#[test]
fn extract_text_with_len_performance() {
    // Verify that extract_text_with_len is at least as fast as extract_text
    // when called multiple times (should be faster due to cached length)
    let text = "This is a test string with some content. ".repeat(100);
    let text_char_count = text.chars().count();
    let entity = Entity::new("test", EntityType::Person, 0, 50, 0.5);

    // Warm up
    let _ = entity.extract_text(&text);
    let _ = entity.extract_text_with_len(&text, text_char_count);

    // Benchmark extract_text (recalculates length each time)
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = entity.extract_text(&text);
    }
    let extract_text_time = start.elapsed();

    // Benchmark extract_text_with_len (uses cached length)
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = entity.extract_text_with_len(&text, text_char_count);
    }
    let extract_text_with_len_time = start.elapsed();

    // extract_text_with_len should be at least as fast (usually faster)
    // Allow 10% margin for measurement noise
    let extract_text_nanos = extract_text_time.as_nanos() as f64;
    let extract_text_with_len_nanos = extract_text_with_len_time.as_nanos() as f64;
    assert!(
        extract_text_with_len_nanos <= extract_text_nanos * 1.1,
        "extract_text_with_len should be at least as fast: cached={:?}, original={:?}",
        extract_text_with_len_time,
        extract_text_time
    );
}

#[test]
fn stacked_ner_deterministic_performance() {
    // Verify that repeated calls to StackedNER have consistent performance
    // (no performance degradation from caching or state)
    let text = "Contact me at test@example.com on January 15, 2024. Visit https://example.com for more info.";
    let ner = StackedNER::default();

    // Warm up
    let _ = ner.extract_entities(&text, None).unwrap();

    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _ = ner.extract_entities(&text, None).unwrap();
        times.push(start.elapsed());
    }

    // Performance should be consistent (within 2x variance for measurement noise)
    let min_time = times.iter().min().unwrap();
    let max_time = times.iter().max().unwrap();
    let min_nanos = min_time.as_nanos() as f64;
    let max_nanos = max_time.as_nanos() as f64;

    assert!(
        max_nanos <= min_nanos * 2.0,
        "Performance should be consistent: min={:?}, max={:?}",
        min_time,
        max_time
    );
}

#[test]
fn validate_with_len_performance() {
    // Verify that validate_with_len is at least as fast as validate
    let text = "This is a test string. ".repeat(50);
    let text_char_count = text.chars().count();
    let entity = Entity::new("test", EntityType::Person, 0, 20, 0.5);

    // Warm up
    let _ = entity.validate(&text);
    let _ = entity.validate_with_len(&text, text_char_count);

    // Benchmark validate
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = entity.validate(&text);
    }
    let validate_time = start.elapsed();

    // Benchmark validate_with_len
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = entity.validate_with_len(&text, text_char_count);
    }
    let validate_with_len_time = start.elapsed();

    // validate_with_len should be at least as fast
    let validate_nanos = validate_time.as_nanos() as f64;
    let validate_with_len_nanos = validate_with_len_time.as_nanos() as f64;
    assert!(
        validate_with_len_nanos <= validate_nanos * 1.1,
        "validate_with_len should be at least as fast: cached={:?}, original={:?}",
        validate_with_len_time,
        validate_time
    );
}
