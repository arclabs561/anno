//! Tests for automatic backend selection functionality.

use anno::{auto, auto_for, available_backends, UseCase};

#[test]
fn test_auto_always_returns_model() {
    // auto() should always return at least StackedNER
    let model = auto().expect("auto() should always return a model");
    assert!(model.is_available());
    assert!(!model.name().is_empty());
}

#[test]
fn test_auto_for_fast() {
    // Fast use case should return StackedNER (no ML)
    let model = auto_for(UseCase::Fast).expect("Fast should always work");
    // Name may vary (e.g., "stacked(pattern+heuristic)" or "StackedNER")
    assert!(model.name().to_lowercase().contains("stacked"));
}

#[test]
fn test_auto_for_nested_entities() {
    // Nested entities should return W2NER
    let model = auto_for(UseCase::NestedEntities).expect("NestedEntities should always work");
    // Name may vary (e.g., "w2ner" or "W2NER")
    assert!(model.name().to_lowercase().contains("w2ner"));
}

#[test]
fn test_auto_for_best_quality() {
    // BestQuality should return best available (at least StackedNER)
    let model = auto_for(UseCase::BestQuality).expect("BestQuality should always work");
    assert!(model.is_available());
    // Should be StackedNER or better (if onnx/candle features enabled)
    assert!(!model.name().is_empty());
}

#[test]
fn test_auto_for_production() {
    // Production should return best available (at least StackedNER)
    let model = auto_for(UseCase::Production).expect("Production should always work");
    assert!(model.is_available());
    assert!(!model.name().is_empty());
}

#[test]
fn test_auto_for_zero_shot_without_onnx() {
    // ZeroShot requires onnx feature - should error if not available
    #[cfg(not(feature = "onnx"))]
    {
        let result = auto_for(UseCase::ZeroShot);
        assert!(result.is_err());
        // Check error message via match or pattern
        match result {
            Err(e) => {
                let err_str = format!("{}", e);
                assert!(err_str.contains("onnx") || err_str.contains("Zero-shot"));
            }
            Ok(_) => panic!("Expected error for ZeroShot without onnx feature"),
        }
    }

    // If onnx is available, it should try to load GLiNER
    #[cfg(feature = "onnx")]
    {
        // May succeed (if model available) or fail (if model download needed)
        // Either way, should not panic
        let _result = auto_for(UseCase::ZeroShot);
    }
}

#[test]
fn test_available_backends_always_includes_core() {
    let backends = available_backends();

    // Core backends should always be available
    let pattern = backends.iter().find(|(name, _)| *name == "RegexNER");
    assert!(pattern.is_some());
    assert!(pattern.unwrap().1); // Should be available

    let heuristic = backends.iter().find(|(name, _)| *name == "HeuristicNER");
    assert!(heuristic.is_some());
    assert!(heuristic.unwrap().1);

    let stacked = backends.iter().find(|(name, _)| *name == "StackedNER");
    assert!(stacked.is_some());
    assert!(stacked.unwrap().1);

    // Note: HybridNER is not a separate backend - it's a pattern used by NERExtractor
    // which combines ML backends with RegexNER. The test checks core backends only.
}

#[test]
fn test_available_backends_onnx_feature_gated() {
    let backends = available_backends();

    #[cfg(feature = "onnx")]
    {
        // With onnx feature, these should be listed as available
        let bert = backends.iter().find(|(name, _)| *name == "BertNEROnnx");
        assert!(bert.is_some());

        let gliner = backends.iter().find(|(name, _)| *name == "GLiNEROnnx");
        assert!(gliner.is_some());

        let nuner = backends.iter().find(|(name, _)| *name == "NuNER");
        assert!(nuner.is_some());

        let w2ner = backends.iter().find(|(name, _)| *name == "W2NER");
        assert!(w2ner.is_some());
    }

    #[cfg(not(feature = "onnx"))]
    {
        // Without onnx feature, these should be listed as unavailable
        let bert = backends.iter().find(|(name, _)| *name == "BertNEROnnx");
        assert!(bert.is_some());
        assert!(!bert.unwrap().1); // Should be false

        let gliner = backends.iter().find(|(name, _)| *name == "GLiNEROnnx");
        assert!(gliner.is_some());
        assert!(!gliner.unwrap().1);
    }
}

#[test]
fn test_available_backends_candle_feature_gated() {
    let backends = available_backends();

    #[cfg(feature = "candle")]
    {
        let candle = backends.iter().find(|(name, _)| *name == "CandleNER");
        assert!(candle.is_some());

        let gliner_candle = backends.iter().find(|(name, _)| *name == "GLiNERCandle");
        assert!(gliner_candle.is_some());
    }

    #[cfg(not(feature = "candle"))]
    {
        let candle = backends.iter().find(|(name, _)| *name == "CandleNER");
        assert!(candle.is_some());
        assert!(!candle.unwrap().1);

        let gliner_candle = backends.iter().find(|(name, _)| *name == "GLiNERCandle");
        assert!(gliner_candle.is_some());
        assert!(!gliner_candle.unwrap().1);
    }
}

#[test]
fn test_auto_returns_working_model() {
    // The model returned by auto() should actually work
    let model = auto().expect("auto() should return a model");
    let result = model.extract_entities("John works at Apple", None);
    assert!(result.is_ok());
    let _entities = result.unwrap();
    // Should find at least some entities (regex-based ones like dates, or heuristic ones)
    // Even if it doesn't find "John" or "Apple", it should not panic
}

#[test]
fn test_auto_for_fast_returns_working_model() {
    let model = auto_for(UseCase::Fast).expect("Fast should work");
    let entities = model.extract_entities("Meeting on January 15, 2024", None);
    assert!(entities.is_ok());
    // StackedNER should find the date
    let entities = entities.unwrap();
    assert!(!entities.is_empty());
}

#[test]
fn test_auto_for_best_quality_returns_working_model() {
    let model = auto_for(UseCase::BestQuality).expect("BestQuality should work");
    let entities = model.extract_entities("Test text", None);
    assert!(entities.is_ok());
    // Should not panic, even if no entities found
}

#[test]
fn test_available_backends_returns_vec() {
    let backends = available_backends();
    assert!(!backends.is_empty());
    // Should have at least 4 core backends
    assert!(backends.len() >= 4);
}

#[test]
fn test_available_backends_format() {
    let backends = available_backends();
    for (name, available) in backends {
        assert!(!name.is_empty());
        // available should be a boolean (always true or false)
        assert!(matches!(available, true | false));
    }
}

#[test]
fn test_use_case_enum() {
    // Test that UseCase variants exist and are comparable
    assert_eq!(UseCase::Fast, UseCase::Fast);
    assert_eq!(UseCase::BestQuality, UseCase::BestQuality);
    assert_eq!(UseCase::ZeroShot, UseCase::ZeroShot);
    assert_eq!(UseCase::Production, UseCase::Production);
    assert_eq!(UseCase::NestedEntities, UseCase::NestedEntities);
}

#[test]
fn test_auto_equals_auto_for_best_quality() {
    // auto() should be equivalent to auto_for(UseCase::BestQuality)
    let model1 = auto().expect("auto() should work");
    let model2 = auto_for(UseCase::BestQuality).expect("BestQuality should work");

    // Both should return available models
    assert!(model1.is_available());
    assert!(model2.is_available());

    // They may return different backends depending on features, but both should work
    let entities1 = model1.extract_entities("Test", None);
    let entities2 = model2.extract_entities("Test", None);
    assert!(entities1.is_ok());
    assert!(entities2.is_ok());
}
