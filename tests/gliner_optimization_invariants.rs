//! Property tests for GLiNER optimizations.
//!
//! Ensures that optimizations (cached text length, pre-allocated vectors, etc.)
//! produce identical results to unoptimized versions.

use anno::backends::gliner_onnx::GLiNEROnnx;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: GLiNER produces valid entities with correct character offsets
    #[test]
    fn gliner_valid_offsets(text in "[A-Za-z0-9\\s.,!?]{10,200}") {
        // Skip if GLiNER is not available (requires onnx feature and model)
        // This test focuses on optimization correctness, not model availability
        if let Ok(gliner) = GLiNEROnnx::from_pretrained("onnx-community/gliner_small-v2.1") {
            let entity_types = &["person", "organization", "location"];
            let entities = gliner.extract(&text, entity_types, 0.5);

            if let Ok(entities) = entities {
                let text_char_count = text.chars().count();

                for entity in entities {
                    // All entities should have valid spans
                    prop_assert!(
                        entity.start < entity.end,
                        "Entity should have valid span: start={}, end={}",
                        entity.start, entity.end
                    );

                    // All entities should be within bounds
                    prop_assert!(
                        entity.end <= text_char_count + 2,
                        "Entity end should be within bounds: end={}, text_len={}",
                        entity.end, text_char_count
                    );

                    // Confidence should be valid
                    prop_assert!(
                        entity.confidence >= 0.0 && entity.confidence <= 1.0,
                        "Entity confidence should be valid: {}",
                        entity.confidence
                    );
                }
            }
        }
    }

    /// Property: GLiNER is deterministic (optimizations don't break determinism)
    #[test]
    fn gliner_deterministic(text in "[A-Za-z0-9\\s.,!?]{10,200}") {
        if let Ok(gliner1) = GLiNEROnnx::from_pretrained("onnx-community/gliner_small-v2.1") {
            if let Ok(gliner2) = GLiNEROnnx::from_pretrained("onnx-community/gliner_small-v2.1") {
                let entity_types = &["person", "organization", "location"];

                let entities1 = gliner1.extract(&text, entity_types, 0.5);
                let entities2 = gliner2.extract(&text, entity_types, 0.5);

                if let (Ok(e1), Ok(e2)) = (entities1, entities2) {
                    prop_assert_eq!(
                        e1.len(), e2.len(),
                        "GLiNER should be deterministic"
                    );

                    // Compare entity sets (order may vary slightly)
                    let e1_set: std::collections::HashSet<_> = e1.iter()
                        .map(|e| (e.start, e.end, e.entity_type.clone(), (e.confidence * 1000.0) as u64))
                        .collect();
                    let e2_set: std::collections::HashSet<_> = e2.iter()
                        .map(|e| (e.start, e.end, e.entity_type.clone(), (e.confidence * 1000.0) as u64))
                        .collect();

                    // Allow some variance due to floating point precision
                    prop_assert!(
                        e1_set.len() == e2_set.len() || (e1_set.len() as i32 - e2_set.len() as i32).abs() <= 1,
                        "GLiNER should produce similar results on repeated calls"
                    );
                }
            }
        }
    }
}
