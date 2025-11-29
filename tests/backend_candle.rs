//! Tests for Candle-based NER backends.
//!
//! These tests require the `candle` feature and model downloads.
//! Run with: `cargo test --features candle -- --ignored candle`

// =============================================================================
// CandleNER Tests
// =============================================================================

#[cfg(feature = "candle")]
mod candle_ner {
    use anno::CandleNER;
    use anno::Model;

    /// Test CandleNER with real model - requires network and candle feature.
    #[test]
    #[ignore] // Run with: cargo test --features candle -- --ignored candle_ner_real
    fn test_candle_ner_real_model() {
        println!("\n=== CandleNER Real Model Test ===\n");

        // Load real model (dslim/bert-base-NER)
        let ner = match CandleNER::from_pretrained("dslim/bert-base-NER") {
            Ok(n) => n,
            Err(e) => {
                println!("Skipping CandleNER test (model not available): {}", e);
                return;
            }
        };

        assert!(
            ner.is_available(),
            "CandleNER should be available after loading"
        );

        let test_cases = [
            "Steve Jobs founded Apple in California.",
            "Microsoft CEO Satya Nadella announced new products.",
            "The European Union headquarters is in Brussels, Belgium.",
            "Dr. Jane Smith works at Harvard University.",
        ];

        for text in test_cases {
            println!("Input: {}", text);

            match ner.extract_entities(text, None) {
                Ok(entities) => {
                    for e in &entities {
                        println!(
                            "  - {} ({}, {:.2})",
                            e.text,
                            e.entity_type.as_label(),
                            e.confidence
                        );
                    }
                    if entities.is_empty() {
                        println!("  (no entities found)");
                    }
                }
                Err(e) => println!("  Error: {}", e),
            }
            println!();
        }
    }

    /// Test CandleNER device selection.
    #[test]
    #[ignore]
    fn test_candle_ner_device() {
        // This test verifies device selection logic
        let ner = match CandleNER::from_pretrained("dslim/bert-base-NER") {
            Ok(n) => n,
            Err(e) => {
                println!("Skipping device test: {}", e);
                return;
            }
        };

        let device = ner.device();
        println!("CandleNER using device: {}", device);

        // Device should be one of: "cpu", "metal", "cuda"
        assert!(
            device == "cpu" || device == "metal" || device == "cuda",
            "Unknown device: {}",
            device
        );
    }
}

// =============================================================================
// GLiNERCandle Tests
// =============================================================================

#[cfg(feature = "candle")]
mod gliner_candle {
    use anno::backends::gliner_candle::GLiNERCandle;
    use anno::Model;

    /// Test GLiNERCandle with real model - requires network and candle feature.
    #[test]
    #[ignore] // Run with: cargo test --features candle -- --ignored gliner_candle_real
    fn test_gliner_candle_real_model() {
        println!("\n=== GLiNERCandle Real Model Test ===\n");

        // Load real model
        let ner = match GLiNERCandle::from_pretrained("urchade/gliner_small-v2.1") {
            Ok(n) => n,
            Err(e) => {
                println!("Skipping GLiNERCandle test (model not available): {}", e);
                return;
            }
        };

        assert!(
            ner.is_available(),
            "GLiNERCandle should be available after loading"
        );

        // Test zero-shot extraction with custom labels
        let test_cases: [(&str, &[&str]); 3] = [
            (
                "Steve Jobs founded Apple in California.",
                &["person", "company", "location"],
            ),
            (
                "CRISPR-Cas9 was developed by Jennifer Doudna at UC Berkeley.",
                &["technology", "scientist", "university"],
            ),
            (
                "The MacBook Pro costs $1,999 and has M3 processor.",
                &["product_model", "price", "specification"],
            ),
        ];

        for (text, labels) in test_cases {
            println!("Input: {}", text);
            println!("Labels: {:?}", labels);

            match ner.extract(text, labels, 0.5) {
                Ok(entities) => {
                    for e in &entities {
                        println!(
                            "  - {} ({}, {:.2})",
                            e.text,
                            e.entity_type.as_label(),
                            e.confidence
                        );
                    }
                    if entities.is_empty() {
                        println!("  (no entities found)");
                    }
                }
                Err(e) => println!("  Error: {}", e),
            }
            println!();
        }
    }

    /// Test GLiNERCandle device selection.
    #[test]
    #[ignore]
    fn test_gliner_candle_device() {
        let ner = match GLiNERCandle::from_pretrained("urchade/gliner_small-v2.1") {
            Ok(n) => n,
            Err(e) => {
                println!("Skipping device test: {}", e);
                return;
            }
        };

        let device = ner.device();
        println!("GLiNERCandle using device: {}", device);

        assert!(
            device == "cpu" || device == "metal" || device == "cuda",
            "Unknown device: {}",
            device
        );
    }
}

// =============================================================================
// Comparison Tests
// =============================================================================

#[cfg(feature = "candle")]
mod comparison {
    #[allow(unused_imports)] // Used conditionally in onnx block
    use anno::Model;

    /// Compare Candle vs ONNX backends on the same inputs.
    #[test]
    #[ignore] // Run with: cargo test --features "candle onnx" -- --ignored compare_backends
    fn test_candle_vs_onnx_consistency() {
        println!("\n=== Candle vs ONNX Backend Comparison ===\n");

        #[cfg(feature = "onnx")]
        {
            use anno::{BertNEROnnx, CandleNER};

            let candle = match CandleNER::from_pretrained("dslim/bert-base-NER") {
                Ok(n) => Some(n),
                Err(e) => {
                    println!("CandleNER not available: {}", e);
                    None
                }
            };

            let onnx = match BertNEROnnx::new("dslim/bert-base-NER") {
                Ok(n) => Some(n),
                Err(e) => {
                    println!("BertNEROnnx not available: {}", e);
                    None
                }
            };

            if candle.is_none() || onnx.is_none() {
                println!("Skipping comparison (one or both backends unavailable)");
                return;
            }

            let candle = candle.unwrap();
            let onnx = onnx.unwrap();

            let test_cases = [
                "Steve Jobs founded Apple.",
                "Microsoft is based in Seattle.",
                "Dr. Jane Smith works at MIT.",
            ];

            for text in test_cases {
                println!("Input: {}", text);

                let candle_entities = candle.extract_entities(text, None).unwrap_or_default();
                let onnx_entities = onnx.extract_entities(text, None).unwrap_or_default();

                println!(
                    "  Candle: {:?}",
                    candle_entities.iter().map(|e| &e.text).collect::<Vec<_>>()
                );
                println!(
                    "  ONNX:   {:?}",
                    onnx_entities.iter().map(|e| &e.text).collect::<Vec<_>>()
                );

                // They should find similar entities (not necessarily identical due to implementation differences)
                println!(
                    "  Match: {} entities vs {} entities",
                    candle_entities.len(),
                    onnx_entities.len()
                );
                println!();
            }
        }

        #[cfg(not(feature = "onnx"))]
        println!("Skipping comparison (onnx feature not enabled)");
    }
}
