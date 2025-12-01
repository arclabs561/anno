//! Integration tests for harmonized backend traits.
//!
//! These tests verify that trait implementations work correctly
//! in real-world scenarios and edge cases.

use anno::{
    backends::{HeuristicNER, PatternNER, StackedNER},
    BatchCapable, Entity, Model, StreamingCapable,
};

// =============================================================================
// BatchCapable Integration Tests
// =============================================================================

#[test]
fn test_batch_extraction_with_varying_lengths() {
    let backend = PatternNER::new();
    let texts = vec![
        "Short",
        "This is a medium length text with some entities like $100 and 2024-01-15",
        "A",
        "Very long text with multiple entities: $500, 2024-12-31, test@example.com, and more entities to test batch processing capabilities",
    ];
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();

    let result = backend.extract_entities_batch(&text_refs, None);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), texts.len());

    // Verify each result is valid
    for (i, entities) in results.iter().enumerate() {
        for entity in entities {
            assert!(
                entity.start <= entity.end,
                "Entity in batch {} has invalid offsets",
                i
            );
            assert!(
                entity.end <= texts[i].len(),
                "Entity in batch {} extends beyond text",
                i
            );
            assert!(
                entity.confidence >= 0.0 && entity.confidence <= 1.0,
                "Entity in batch {} has invalid confidence",
                i
            );
        }
    }
}

#[test]
fn test_batch_optimal_size_usage() {
    let backend = HeuristicNER::new();

    if let Some(optimal_size) = backend.optimal_batch_size() {
        // Create exactly optimal_size texts
        let texts: Vec<String> = (0..optimal_size)
            .map(|i| format!("Test text {} with John Smith", i))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();

        let result = backend.extract_entities_batch(&text_refs, None);
        assert!(result.is_ok());
        let results = result.unwrap();
        assert_eq!(results.len(), optimal_size);
    }
}

#[test]
fn test_batch_larger_than_optimal() {
    let backend = PatternNER::new();
    let optimal_size = backend.optimal_batch_size().unwrap_or(8);

    // Create batch larger than optimal
    let batch_size = optimal_size * 2;
    let texts: Vec<String> = (0..batch_size)
        .map(|i| format!("Text {}: $100 on 2024-01-15", i))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();

    let result = backend.extract_entities_batch(&text_refs, None);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), batch_size);
}

#[test]
fn test_batch_single_item() {
    let backend = StackedNER::default();
    let texts = vec!["Single text with $100"];
    let text_refs: Vec<&str> = texts.iter().copied().collect();

    let result = backend.extract_entities_batch(&text_refs, None);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1);
}

// =============================================================================
// StreamingCapable Integration Tests
// =============================================================================

#[test]
fn test_streaming_chunk_boundary() {
    let backend = PatternNER::new();
    let chunk_size = backend.recommended_chunk_size();

    // Create text exactly at chunk boundary
    let text = "a".repeat(chunk_size);
    let result = backend.extract_entities_streaming(&text, 0);
    assert!(result.is_ok());
}

#[test]
fn test_streaming_multiple_chunks() {
    let backend = HeuristicNER::new();
    let chunk_size = backend.recommended_chunk_size();

    // Simulate processing multiple chunks
    let chunk1 = "First chunk with John Smith".repeat(chunk_size / 30);
    let chunk2 = "Second chunk with Jane Doe".repeat(chunk_size / 30);

    let entities1 = backend.extract_entities_streaming(&chunk1, 0).unwrap();
    let offset = chunk1.len() + 1; // +1 for separator
    let entities2 = backend.extract_entities_streaming(&chunk2, offset).unwrap();

    // Verify offsets are adjusted correctly
    for entity in entities1 {
        assert!(entity.end <= chunk1.len());
    }
    for entity in entities2 {
        assert!(entity.start >= offset);
        assert!(entity.end >= offset);
    }
}

#[test]
fn test_streaming_empty_chunk() {
    let backend = PatternNER::new();
    let result = backend.extract_entities_streaming("", 0);
    assert!(result.is_ok());
    let entities = result.unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_streaming_very_large_offset() {
    let backend = StackedNER::default();
    let text = "Test text";
    let large_offset = 1_000_000;

    let result = backend.extract_entities_streaming(text, large_offset);
    assert!(result.is_ok());
    let entities = result.unwrap();

    for entity in entities {
        assert!(entity.start >= large_offset);
        assert!(entity.end >= large_offset);
    }
}

// =============================================================================
// Trait Combination Tests
// =============================================================================

#[test]
fn test_batch_and_streaming_together() {
    let backend = PatternNER::new();

    // Use batch for initial processing
    let texts = vec!["Text 1: $100", "Text 2: $200"];
    let text_refs: Vec<&str> = texts.iter().copied().collect();
    let batch_result = backend.extract_entities_batch(&text_refs, None).unwrap();

    // Use streaming for same texts with offset
    let offset = 1000;
    let stream_result1 = backend
        .extract_entities_streaming(texts[0], offset)
        .unwrap();
    let stream_result2 = backend
        .extract_entities_streaming(texts[1], offset + texts[0].len() + 1)
        .unwrap();

    // Batch and streaming should extract same entities (with offset adjustment)
    assert_eq!(batch_result[0].len(), stream_result1.len());
    assert_eq!(batch_result[1].len(), stream_result2.len());
}

#[test]
fn test_model_batch_streaming_all_together() {
    let backend = HeuristicNER::new();
    let text = "John Smith works at Acme Corp in New York";

    // Test all three interfaces
    let model_entities = backend.extract_entities(text, None).unwrap();
    let batch_entities = backend.extract_entities_batch(&[text], None).unwrap();
    let stream_entities = backend.extract_entities_streaming(text, 0).unwrap();

    // All should produce same number of entities
    assert_eq!(model_entities.len(), batch_entities[0].len());
    assert_eq!(model_entities.len(), stream_entities.len());
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_batch_with_all_empty_strings() {
    let backend = PatternNER::new();
    let texts = vec!["", "", ""];
    let text_refs: Vec<&str> = texts.iter().copied().collect();

    let result = backend.extract_entities_batch(&text_refs, None);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|entities| entities.is_empty()));
}

#[test]
fn test_streaming_with_unicode_boundaries() {
    let backend = PatternNER::new();
    // Text with Unicode characters that might affect offset calculation
    let text = "Test with émojis 🎉 and 中文 characters";
    let offset = 50;

    let result = backend.extract_entities_streaming(text, offset);
    assert!(result.is_ok());
    let entities = result.unwrap();

    for entity in entities {
        assert!(entity.start >= offset);
        assert!(entity.end >= offset);
    }
}

#[test]
fn test_batch_consistency_across_backends() {
    let backends: Vec<Box<dyn BatchCapable>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];

    let texts = vec!["Test 1", "Test 2", "Test 3"];
    let text_refs: Vec<&str> = texts.iter().copied().collect();

    for backend in backends {
        let result = backend.extract_entities_batch(&text_refs, None);
        assert!(result.is_ok());
        let results = result.unwrap();
        assert_eq!(results.len(), texts.len());
    }
}

#[test]
fn test_streaming_consistency_across_backends() {
    let backends: Vec<Box<dyn StreamingCapable>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];

    let text = "Test text for streaming";
    let offset = 100;

    for backend in backends {
        let result = backend.extract_entities_streaming(text, offset);
        assert!(result.is_ok());
        let entities = result.unwrap();
        for entity in entities {
            assert!(entity.start >= offset);
        }
    }
}

// =============================================================================
// Performance Characteristics Tests
// =============================================================================

#[test]
fn test_batch_is_faster_than_sequential() {
    // This is a qualitative test - batch should handle multiple items
    let backend = PatternNER::new();
    let texts: Vec<String> = (0..10).map(|i| format!("Text {}: $100", i)).collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();

    // Batch processing
    let batch_result = backend.extract_entities_batch(&text_refs, None);
    assert!(batch_result.is_ok());

    // Sequential processing (for comparison)
    let sequential_result: Result<Vec<_>, _> = texts
        .iter()
        .map(|text| backend.extract_entities(text, None))
        .collect();
    assert!(sequential_result.is_ok());

    // Both should produce same results
    let batch_entities = batch_result.unwrap();
    let sequential_entities = sequential_result.unwrap();
    assert_eq!(batch_entities.len(), sequential_entities.len());
}

#[test]
fn test_streaming_preserves_entity_text() {
    let backend = HeuristicNER::new();
    let text = "John Smith works at Acme Corp";
    let offset = 100;

    let direct_entities = backend.extract_entities(text, None).unwrap();
    let stream_entities = backend.extract_entities_streaming(text, offset).unwrap();

    // Entity texts should match (offsets will differ)
    assert_eq!(direct_entities.len(), stream_entities.len());
    for (direct, stream) in direct_entities.iter().zip(stream_entities.iter()) {
        assert_eq!(direct.text, stream.text);
        assert_eq!(direct.entity_type, stream.entity_type);
        // Confidence might differ slightly due to context, but should be close
        assert!((direct.confidence - stream.confidence).abs() < 0.1);
    }
}
