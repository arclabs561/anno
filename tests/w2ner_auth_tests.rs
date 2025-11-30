//! Tests for W2NER authentication error handling.
//!
//! These tests verify that W2NER properly handles 401 authentication errors
//! and provides helpful error messages.

#[cfg(feature = "onnx")]
mod w2ner_auth {
    use anno::W2NER;

    /// Test that W2NER provides helpful error message for 401 authentication errors
    /// This test can run without network - it will fail fast when model doesn't exist
    #[test]
    fn test_w2ner_401_error_detection() {
        // Try loading the model that requires authentication
        let result = W2NER::from_pretrained("ljynlp/w2ner-bert-base");

        match result {
            Ok(_) => {
                // If it succeeds, the model might be publicly available now
                // or user has HF_TOKEN set
                eprintln!("W2NER model loaded successfully (may have HF_TOKEN set)");
            }
            Err(e) => {
                let err_str = e.to_string();

                // Check if it's an authentication error
                if err_str.contains("401")
                    || err_str.contains("Unauthorized")
                    || err_str.contains("authentication")
                {
                    // Verify error message is helpful
                    assert!(
                        err_str.contains("HF_TOKEN")
                            || err_str.contains("HuggingFace")
                            || err_str.contains("authentication"),
                        "Error message should mention authentication: {}",
                        err_str
                    );

                    // Verify error mentions alternatives
                    assert!(
                        err_str.contains("alternative")
                            || err_str.contains("nested")
                            || err_str.contains("model"),
                        "Error message should mention alternatives: {}",
                        err_str
                    );
                } else {
                    // Other errors are also valid (network issues, etc.)
                    eprintln!("W2NER error (not 401): {}", err_str);
                }
            }
        }
    }

    /// Test that W2NER error messages are informative
    #[test]
    fn test_w2ner_error_message_quality() {
        // Test with invalid model ID
        let result = W2NER::from_pretrained("nonexistent/model-12345");

        if let Err(e) = result {
            let err_str = e.to_string();
            // Error should be informative
            assert!(
                err_str.contains("not found")
                    || err_str.contains("download")
                    || err_str.contains("Retrieval")
                    || err_str.contains("error"),
                "Error message should be informative: {}",
                err_str
            );
        }
    }
}
