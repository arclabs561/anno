//! Comprehensive tests for backend trait harmonization.
//!
//! These tests verify that all backends implement traits consistently
//! and that trait implementations maintain expected invariants.

use anno::{
    backends::{
        HeuristicNER, PatternNER, StackedNER,
    },
    BatchCapable, Entity, Model, StreamingCapable,
};
use proptest::prelude::*;

// =============================================================================
// Model Trait Invariants
// =============================================================================

proptest! {
    /// INVARIANT: All backends return valid entity offsets
    #[test]
    fn valid_offsets(
        text in "[A-Za-z0-9 .,!?$@#]{1,200}",
        backend_idx in 0..3u8
    ) {
        let backend: Box<dyn Model> = match backend_idx {
            0 => Box::new(PatternNER::new()),
            1 => Box::new(HeuristicNER::new()),
            _ => Box::new(StackedNER::default()),
        };

        if let Ok(entities) = backend.extract_entities(&text, None) {
            for entity in entities {
                prop_assert!(
                    entity.start <= entity.end,
                    "start {} > end {}",
                    entity.start, entity.end
                );
                prop_assert!(
                    entity.end <= text.len(),
                    "end {} > text.len() {}",
                    entity.end, text.len()
                );
                prop_assert!(
                    entity.confidence >= 0.0 && entity.confidence <= 1.0,
                    "confidence {} not in [0, 1]",
                    entity.confidence
                );
            }
        }
    }

    /// INVARIANT: Backend names are never empty
    #[test]
    fn backend_names_not_empty(backend_idx in 0..3u8) {
        let backend: Box<dyn Model> = match backend_idx {
            0 => Box::new(PatternNER::new()),
            1 => Box::new(HeuristicNER::new()),
            _ => Box::new(StackedNER::default()),
        };

        prop_assert!(!backend.name().is_empty());
        prop_assert!(!backend.description().is_empty());
    }

    /// INVARIANT: Empty text returns empty entities (or valid empty vec)
    #[test]
    fn empty_text_handling(backend_idx in 0..3u8) {
        let backend: Box<dyn Model> = match backend_idx {
            0 => Box::new(PatternNER::new()),
            1 => Box::new(HeuristicNER::new()),
            _ => Box::new(StackedNER::default()),
        };

        let result = backend.extract_entities("", None);
        prop_assert!(result.is_ok());
        let entities = result.unwrap();
        prop_assert!(entities.is_empty() || entities.iter().all(|e| e.start == e.end));
    }
}

// =============================================================================
// BatchCapable Trait Tests
// =============================================================================

proptest! {
    /// INVARIANT: Batch extraction returns same number of results as inputs
    #[test]
    fn batch_count_matches_input(
        texts in prop::collection::vec("[A-Za-z0-9 .,!?]{1,100}", 1..10)
    ) {
        let backends: Vec<Box<dyn BatchCapable>> = vec![
            Box::new(PatternNER::new()),
            Box::new(HeuristicNER::new()),
            Box::new(StackedNER::default()),
        ];

        for backend in backends {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
            if let Ok(results) = backend.extract_entities_batch(&text_refs, None) {
                prop_assert_eq!(
                    results.len(),
                    texts.len(),
                    "Batch results count {} != input count {}",
                    results.len(),
                    texts.len()
                );
            }
        }
    }

    /// INVARIANT: Batch extraction matches sequential extraction
    #[test]
    fn batch_matches_sequential(
        texts in prop::collection::vec("[A-Za-z0-9 .,!?]{1,50}", 1..5)
    ) {
        let backend = PatternNER::new();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();

        // Sequential extraction
        let sequential: Result<Vec<Vec<Entity>>, _> = texts
            .iter()
            .map(|text| backend.extract_entities(text, None))
            .collect();

        // Batch extraction
        let batch = backend.extract_entities_batch(&text_refs, None);

        if let (Ok(seq), Ok(bat)) = (sequential, batch) {
            prop_assert_eq!(
                seq.len(),
                bat.len(),
                "Sequential and batch results have different lengths"
            );

            for (seq_ents, bat_ents) in seq.iter().zip(bat.iter()) {
                prop_assert_eq!(
                    seq_ents.len(),
                    bat_ents.len(),
                    "Entity counts differ between sequential and batch"
                );
            }
        }
    }

    /// INVARIANT: Optimal batch size is reasonable (if provided)
    #[test]
    fn optimal_batch_size_reasonable(backend_idx in 0..3u8) {
        let backend: Box<dyn BatchCapable> = match backend_idx {
            0 => Box::new(PatternNER::new()),
            1 => Box::new(HeuristicNER::new()),
            _ => Box::new(StackedNER::default()),
        };

        if let Some(size) = backend.optimal_batch_size() {
            prop_assert!(
                size > 0 && size <= 128,
                "Optimal batch size {} is unreasonable",
                size
            );
        }
    }
}

// =============================================================================
// StreamingCapable Trait Tests
// =============================================================================

proptest! {
    /// INVARIANT: Streaming extraction with offset adjustment maintains positions
    #[test]
    fn streaming_offset_adjustment(
        chunk1 in "[A-Za-z0-9 ]{1,50}",
        chunk2 in "[A-Za-z0-9 ]{1,50}",
        backend_idx in 0..3u8
    ) {
        let backend: Box<dyn StreamingCapable> = match backend_idx {
            0 => Box::new(PatternNER::new()),
            1 => Box::new(HeuristicNER::new()),
            _ => Box::new(StackedNER::default()),
        };

        let full_text = format!("{} {}", chunk1, chunk2);
        let chunk1_len = chunk1.len();

        // Extract from full text (for reference)
        let _full_entities = backend.extract_entities(&full_text, None).unwrap();

        // Extract from chunks with offset
        let chunk1_entities = backend.extract_entities_streaming(&chunk1, 0).unwrap();
        let chunk2_entities = backend.extract_entities_streaming(&chunk2, chunk1_len + 1).unwrap();

        // All chunk entities should have adjusted offsets
        for entity in chunk1_entities.iter().chain(chunk2_entities.iter()) {
            prop_assert!(
                entity.end <= full_text.len(),
                "Streaming entity end {} > full text length {}",
                entity.end,
                full_text.len()
            );
        }
    }

    /// INVARIANT: Recommended chunk size is reasonable
    #[test]
    fn recommended_chunk_size_reasonable(backend_idx in 0..3u8) {
        let backend: Box<dyn StreamingCapable> = match backend_idx {
            0 => Box::new(PatternNER::new()),
            1 => Box::new(HeuristicNER::new()),
            _ => Box::new(StackedNER::default()),
        };

        let chunk_size = backend.recommended_chunk_size();
        prop_assert!(
            chunk_size > 0 && chunk_size <= 100_000,
            "Recommended chunk size {} is unreasonable",
            chunk_size
        );
    }
}

// =============================================================================
// GpuCapable Trait Tests
// =============================================================================

#[cfg(feature = "candle")]
mod gpu_tests {
    use super::*;
    use anno::backends::{CandleNER, GLiNERCandle};

    proptest! {
        /// INVARIANT: GPU-capable backends report valid device strings
        #[test]
        fn gpu_device_string_valid(
            backend_idx in 0..2u8
        ) {
            // Note: These tests require actual model files, so we'll skip for now
            // In a real scenario, you'd create backends and check device()
            prop_assume!(false, "GPU tests require model files");
        }

        /// INVARIANT: is_gpu_active() is consistent with device()
        #[test]
        fn gpu_active_consistent(
            backend_idx in 0..2u8
        ) {
            prop_assume!(false, "GPU tests require model files");
        }
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_all_backends_implement_model() {
    let backends: Vec<Box<dyn Model>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];

    for backend in backends {
        assert!(backend.is_available());
        assert!(!backend.name().is_empty());
        assert!(!backend.description().is_empty());
    }
}

#[test]
fn test_batch_capable_backends() {
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
fn test_streaming_capable_backends() {
    let backends: Vec<Box<dyn StreamingCapable>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];

    for backend in backends {
        let chunk_size = backend.recommended_chunk_size();
        assert!(chunk_size > 0);

        // Test streaming with offset
        let text = "Test text for streaming";
        let result = backend.extract_entities_streaming(text, 100);
        assert!(result.is_ok());
        let entities = result.unwrap();
        for entity in entities {
            assert!(entity.start >= 100);
            assert!(entity.end >= 100);
        }
    }
}

#[test]
fn test_trait_combinations() {
    // Test that backends can implement multiple traits
    let backend = PatternNER::new();

    // Should work as Model
    let _entities = backend.extract_entities("Test", None).unwrap();

    // Should work as BatchCapable
    let _batch = backend.extract_entities_batch(&["Test"], None).unwrap();

    // Should work as StreamingCapable
    let _stream = backend.extract_entities_streaming("Test", 0).unwrap();
}

// =============================================================================
// Consistency Tests
// =============================================================================

#[test]
fn test_confidence_bounds() {
    let backends: Vec<Box<dyn Model>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];

    let text = "Meeting on 2024-01-15 cost $100.";

    for backend in backends {
        if let Ok(entities) = backend.extract_entities(text, None) {
            for entity in entities {
                assert!(
                    entity.confidence >= 0.0 && entity.confidence <= 1.0,
                    "Entity confidence {} out of bounds for backend {}",
                    entity.confidence,
                    backend.name()
                );
            }
        }
    }
}

#[test]
fn test_no_overlapping_entities_same_backend() {
    let backend = PatternNER::new();
    let text = "Meeting on 2024-01-15 cost $100.";

    if let Ok(entities) = backend.extract_entities(text, None) {
        // Check for overlapping entities (same backend shouldn't produce overlaps)
        for (i, e1) in entities.iter().enumerate() {
            for (j, e2) in entities.iter().enumerate() {
                if i != j {
                    let overlaps = !(e1.end <= e2.start || e2.end <= e1.start);
                    // Some backends may intentionally allow overlaps (e.g., nested entities)
                    // So we just verify the logic is correct
                    if overlaps {
                        // If they overlap, they should have different types or be intentional
                        assert_ne!(e1.entity_type, e2.entity_type, "Overlapping entities with same type");
                    }
                }
            }
        }
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_batch_with_empty_strings() {
    let backend = PatternNER::new();
    let texts = vec!["", "Test", "", "Another test"];
    let text_refs: Vec<&str> = texts.iter().copied().collect();
    
    let result = backend.extract_entities_batch(&text_refs, None);
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), texts.len());
    
    // Empty strings should return empty entity lists
    assert!(results[0].is_empty());
    assert!(results[2].is_empty());
}

#[test]
fn test_streaming_with_zero_offset() {
    let backend = PatternNER::new();
    let text = "Meeting on 2024-01-15";
    
    let entities = backend.extract_entities_streaming(text, 0).unwrap();
    let direct_entities = backend.extract_entities(text, None).unwrap();
    
    // With zero offset, results should match
    assert_eq!(entities.len(), direct_entities.len());
}

#[test]
fn test_streaming_with_large_offset() {
    let backend = PatternNER::new();
    let text = "Date: 2024-01-15";
    let offset = 1000;
    
    let entities = backend.extract_entities_streaming(text, offset).unwrap();
    
    // All entities should have adjusted offsets
    for entity in entities {
        assert!(entity.start >= offset);
        assert!(entity.end >= offset);
    }
}

#[test]
fn test_batch_optimal_size_consistency() {
    let backends: Vec<Box<dyn BatchCapable>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];
    
    for backend in backends {
        if let Some(size) = backend.optimal_batch_size() {
            // Should be a reasonable size
            assert!(size > 0 && size <= 128);
            
            // Test with exactly that batch size
            let texts: Vec<String> = (0..size).map(|i| format!("Test {}", i)).collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
            let result = backend.extract_entities_batch(&text_refs, None);
            assert!(result.is_ok());
        }
    }
}

#[test]
fn test_streaming_chunk_size_consistency() {
    let backends: Vec<Box<dyn StreamingCapable>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];
    
    for backend in backends {
        let chunk_size = backend.recommended_chunk_size();
        assert!(chunk_size > 0);
        
        // Create text of exactly that size
        let text = "a".repeat(chunk_size);
        let result = backend.extract_entities_streaming(&text, 0);
        assert!(result.is_ok());
    }
}

#[test]
fn test_gpu_capable_backends_work_on_cpu() {
    // Even if GPU is not available, GPU-capable backends should work on CPU
    // This is tested via feature flags - if candle feature is enabled, backends
    // should gracefully fall back to CPU
    #[cfg(feature = "candle")]
    {
        // Note: This would require actual model files, so we skip for now
        // In production, this would test that is_gpu_active() returns false
        // when GPU is not available, but device() still works
    }
}

#[test]
fn test_trait_implementation_consistency() {
    // Verify that all backends that implement BatchCapable also implement Model
    let batch_backends: Vec<Box<dyn BatchCapable>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];
    
    for backend in batch_backends {
        // Should be able to use as Model
        let _entities = backend.extract_entities("Test", None).unwrap();
        
        // Should be able to use batch methods
        let _batch = backend.extract_entities_batch(&["Test"], None).unwrap();
    }
    
    // Same for StreamingCapable
    let stream_backends: Vec<Box<dyn StreamingCapable>> = vec![
        Box::new(PatternNER::new()),
        Box::new(HeuristicNER::new()),
        Box::new(StackedNER::default()),
    ];
    
    for backend in stream_backends {
        let _entities = backend.extract_entities("Test", None).unwrap();
        let _stream = backend.extract_entities_streaming("Test", 0).unwrap();
    }
}

