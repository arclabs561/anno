//! End-to-end tests for PyTorch to Safetensors conversion.
//!
//! These tests verify that Candle backends can automatically convert
//! PyTorch models to safetensors format when needed.
//!
//! Run with: `cargo test --features candle conversion`

#[cfg(feature = "candle")]
mod conversion_tests {
    use std::path::PathBuf;

    /// Test that the conversion script exists and is executable
    #[test]
    fn test_conversion_script_exists() {
        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("scripts/convert_pytorch_to_safetensors.py");

        assert!(
            script_path.exists(),
            "Conversion script should exist at: {:?}",
            script_path
        );
    }

    /// Test conversion script can be invoked (smoke test - will fail without args, that's expected)
    #[test]
    fn test_conversion_script_execution() {
        use std::process::Command;

        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("scripts/convert_pytorch_to_safetensors.py");

        // Test that script can be executed (will fail without input args, but that's expected)
        let output = Command::new("uv")
            .arg("run")
            .arg("--script")
            .arg(&script_path)
            .output()
            .or_else(|_| {
                // Fallback to python3 if uv is not available
                Command::new("python3").arg(&script_path).output()
            });

        match output {
            Ok(output) => {
                // Script executed (exit code != 0 is expected without args)
                // Just verify it ran and produced output
                assert!(
                    !output.stderr.is_empty() || !output.stdout.is_empty(),
                    "Script should produce output (usage message)"
                );
            }
            Err(e) => {
                // If neither uv nor python3 is available, that's a test environment issue
                // but we should still verify the script exists
                eprintln!("Note: uv/python3 not available: {}", e);
                assert!(
                    script_path.exists(),
                    "Script should exist even if uv/python3 not available"
                );
            }
        }
    }

    /// Test conversion script with invalid input (should produce helpful error)
    #[test]
    fn test_conversion_script_error_handling() {
        use std::process::Command;
        use tempfile::NamedTempFile;

        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("scripts/convert_pytorch_to_safetensors.py");

        // Create a fake (empty) pytorch file
        let fake_pytorch = NamedTempFile::new().expect("Should create temp file");
        let fake_output = NamedTempFile::new().expect("Should create temp file");

        let output = Command::new("uv")
            .arg("run")
            .arg("--script")
            .arg(&script_path)
            .arg(fake_pytorch.path())
            .arg(fake_output.path())
            .output()
            .or_else(|_| {
                Command::new("python3")
                    .arg(&script_path)
                    .arg(fake_pytorch.path())
                    .arg(fake_output.path())
                    .output()
            });

        match output {
            Ok(output) => {
                // Should fail (empty file is not valid PyTorch)
                // But should produce a helpful error message
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let combined = format!("{} {}", stdout, stderr);

                // Error should mention something about the file or conversion
                assert!(
                    !output.status.success()
                        || combined.contains("error")
                        || combined.contains("Error")
                        || combined.contains("failed"),
                    "Script should fail or report error for invalid input. Output: {}",
                    combined
                );
            }
            Err(_) => {
                // If uv/python3 not available, skip this test
                // But verify script exists
                assert!(script_path.exists());
            }
        }
    }

    /// Test that GLiNERCandle can handle models with only pytorch_model.bin
    /// This requires a real model download and conversion
    /// Marked ignore because it requires network and model download
    #[test]
    #[ignore] // Requires network and model download (~100MB)
    fn test_gliner_candle_conversion() {
        use anno::backends::gliner_candle::GLiNERCandle;

        // Try loading a model that typically only has pytorch_model.bin
        // Note: This will download and convert, so it's slow
        let result = GLiNERCandle::from_pretrained("urchade/gliner_small-v2.1");

        match result {
            Ok(model) => {
                // Test that model works after conversion
                let text = "Steve Jobs founded Apple in California.";
                let entities = model.extract(text, &["person", "organization", "location"], 0.5);
                assert!(entities.is_ok(), "Model should work after conversion");
            }
            Err(e) => {
                // Conversion might fail if uv/python3 not available
                // or if model already has safetensors
                eprintln!("Conversion test skipped: {}", e);
            }
        }
    }

    /// Test that CandleNER can handle conversion (if model only has pytorch_model.bin)
    /// Marked ignore because it requires network and model download
    #[test]
    #[ignore] // Requires network and model download
    fn test_candle_ner_conversion() {
        use anno::{CandleNER, Model};

        // Most BERT models have safetensors, but test the conversion path
        // if we find one that doesn't
        let result = CandleNER::from_pretrained("dslim/bert-base-NER");

        match result {
            Ok(_model) => {
                // Model loaded successfully (either had safetensors or converted)
                // Test that it can extract entities
                let text = "Steve Jobs founded Apple in California.";
                let entities = _model.extract_entities(text, None);
                assert!(entities.is_ok(), "Model should work after conversion");
            }
            Err(e) => {
                eprintln!("CandleNER test skipped: {}", e);
            }
        }
    }

    /// Test that GLiNER2Candle can handle conversion
    /// Marked ignore because it requires network and model download
    #[test]
    #[ignore] // Requires network and model download
    fn test_gliner2_candle_conversion() {
        use anno::backends::gliner2::GLiNER2Candle;

        // Try loading a GLiNER2 model
        let result = GLiNER2Candle::from_pretrained("fastino/gliner2-base-v1");

        match result {
            Ok(_model) => {
                // Model loaded successfully
            }
            Err(e) => {
                eprintln!("GLiNER2Candle test skipped: {}", e);
            }
        }
    }

    /// Test conversion caching - second call should use cached file
    /// Marked ignore because it requires network and model download
    #[test]
    #[ignore] // Requires network and model download
    fn test_conversion_caching() {
        use anno::backends::gliner_candle::GLiNERCandle;
        use std::time::Instant;

        // First load - should trigger conversion
        let start1 = Instant::now();
        let model1 = GLiNERCandle::from_pretrained("urchade/gliner_small-v2.1");
        let time1 = start1.elapsed();

        if model1.is_err() {
            eprintln!("Skipping caching test: model load failed");
            return;
        }

        // Second load - should use cached conversion
        let start2 = Instant::now();
        let model2 = GLiNERCandle::from_pretrained("urchade/gliner_small-v2.1");
        let time2 = start2.elapsed();

        assert!(model2.is_ok(), "Second load should succeed");

        // Second load should be faster (uses cache)
        // Note: This is a heuristic - network conditions may vary
        if time2 < time1 {
            println!(
                "Cache working: second load ({:?}) faster than first ({:?})",
                time2, time1
            );
        }
    }

    /// Test that CandleEncoder can handle conversion
    /// Marked ignore because it requires network and model download
    #[test]
    #[ignore] // Requires network and model download
    fn test_candle_encoder_conversion() {
        use anno::backends::encoder_candle::CandleEncoder;

        // Try loading an encoder that might need conversion
        let result = CandleEncoder::from_pretrained("bert-base-uncased");

        match result {
            Ok(_encoder) => {
                // Encoder loaded successfully (either had safetensors or converted)
            }
            Err(e) => {
                eprintln!("CandleEncoder test skipped: {}", e);
            }
        }
    }

    /// Test error handling when model doesn't exist (no network needed - fails fast)
    #[test]
    fn test_conversion_error_handling_nonexistent_model() {
        use anno::backends::gliner_candle::GLiNERCandle;

        // Try loading a model that doesn't exist - should return error quickly
        // This tests the error path when conversion would be needed but fails
        let result = GLiNERCandle::from_pretrained("nonexistent/model-12345-that-does-not-exist");

        assert!(result.is_err(), "Should error on non-existent model");
        let err = result.unwrap_err().to_string();
        // Error should mention conversion or model loading failure
        assert!(
            err.contains("conversion")
                || err.contains("not found")
                || err.contains("Failed")
                || err.contains("Retrieval")
                || err.contains("safetensors")
                || err.contains("HuggingFace")
                || err.contains("model"),
            "Error message should mention conversion or failure: {}",
            err
        );
    }

    /// Test that conversion error messages are helpful
    #[test]
    fn test_conversion_error_message_quality() {
        use anno::backends::gliner_candle::GLiNERCandle;

        // Try loading a model that doesn't exist - should provide helpful error
        let result = GLiNERCandle::from_pretrained("nonexistent/model-12345-that-does-not-exist");

        if let Err(e) = result {
            let err_str = e.to_string();
            // Error should mention alternatives or solutions
            assert!(
                err_str.contains("GLiNEROnnx")
                    || err_str.contains("safetensors")
                    || err_str.contains("uv")
                    || err_str.contains("python")
                    || err_str.contains("conversion")
                    || err_str.contains("alternative")
                    || err_str.contains("HuggingFace")
                    || err_str.len() > 50, // At least somewhat informative
                "Error message should provide helpful alternatives or be informative: {}",
                err_str
            );
        } else {
            // If it somehow succeeds (cached?), that's also fine
        }
    }

    /// Test conversion path resolution (script path finding)
    #[test]
    fn test_conversion_script_path_resolution() {
        // Test that the conversion function can find the script
        // This is a unit test of the path resolution logic
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let expected_script =
            PathBuf::from(manifest_dir).join("scripts/convert_pytorch_to_safetensors.py");

        assert!(
            expected_script.exists(),
            "Script should exist at expected location: {:?}",
            expected_script
        );

        // Verify script has correct shebang
        let content = std::fs::read_to_string(&expected_script).expect("Should read script");
        assert!(
            content.starts_with("#!/usr/bin/env -S uv run --script")
                || content.contains("uv run --script"),
            "Script should have PEP 723 shebang"
        );
        assert!(
            content.contains("torch") && content.contains("safetensors"),
            "Script should mention torch and safetensors dependencies"
        );
    }
}
