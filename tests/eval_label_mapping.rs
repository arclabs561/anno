//! Tests for label mapping in evaluation framework.

#[cfg(feature = "eval")]
mod tests {
    use anno::eval::task_evaluator::TaskEvaluator;

    #[test]
    fn test_label_mapping_standard_labels() {
        // Test standard CoNLL-style labels
        let test_cases = vec![
            ("PER", "person"),
            ("ORG", "organization"),
            ("LOC", "location"),
            ("MISC", "misc"),
            ("PERSON", "person"),
            ("ORGANIZATION", "organization"),
            ("LOCATION", "location"),
        ];

        for (input, expected) in test_cases {
            let input_refs = vec![input];
            let mapped = TaskEvaluator::map_dataset_labels_to_model(&input_refs, "nuner");
            assert_eq!(
                mapped[0], expected,
                "Label mapping failed: {} should map to {}",
                input, expected
            );
        }
    }

    #[test]
    fn test_label_mapping_case_insensitive() {
        // Label mapping should handle case variations
        let test_cases = vec![
            ("per", "person"),
            ("Per", "person"),
            ("PER", "person"),
            ("org", "organization"),
            ("Org", "organization"),
            ("ORG", "organization"),
        ];

        for (input, expected) in test_cases {
            let input_refs = vec![input];
            let mapped = TaskEvaluator::map_dataset_labels_to_model(&input_refs, "nuner");
            assert_eq!(
                mapped[0], expected,
                "Label mapping should be case-insensitive: {} should map to {}",
                input, expected
            );
        }
    }

    #[test]
    fn test_label_mapping_preserves_unknown() {
        // Unknown labels should be preserved (lowercased)
        let test_cases = vec!["CUSTOM_TYPE", "UNKNOWN", "RARE_ENTITY"];

        for input in test_cases {
            let input_refs = vec![input];
            let mapped = TaskEvaluator::map_dataset_labels_to_model(&input_refs, "nuner");
            // Should be lowercased but otherwise preserved
            assert_eq!(
                mapped[0],
                input.to_lowercase(),
                "Unknown label should be preserved (lowercased): {} -> {}",
                input,
                mapped[0]
            );
        }
    }

    #[test]
    fn test_label_mapping_multiple_labels() {
        // Test mapping multiple labels at once
        let input = vec!["PER", "ORG", "LOC"];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&input, "nuner");

        assert_eq!(mapped.len(), 3);
        assert_eq!(mapped[0], "person");
        assert_eq!(mapped[1], "organization");
        assert_eq!(mapped[2], "location");
    }

    #[test]
    fn test_label_mapping_different_backends() {
        // Different backends should produce same mappings (for standard labels)
        let input = vec!["PER", "ORG"];

        let nuner_mapped = TaskEvaluator::map_dataset_labels_to_model(&input, "nuner");
        let gliner_mapped = TaskEvaluator::map_dataset_labels_to_model(&input, "gliner_onnx");
        let gliner_candle_mapped =
            TaskEvaluator::map_dataset_labels_to_model(&input, "gliner_candle");

        // All should produce same mappings for standard labels
        assert_eq!(nuner_mapped, gliner_mapped);
        assert_eq!(nuner_mapped, gliner_candle_mapped);
    }

    #[test]
    fn test_label_mapping_empty_input() {
        // Empty input should produce empty output
        let input: Vec<&str> = vec![];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&input, "nuner");
        assert_eq!(mapped.len(), 0);
    }

    #[test]
    fn test_label_mapping_whitespace_handling() {
        // Labels with whitespace are lowercased but not trimmed (current behavior)
        // Note: The function doesn't trim, so whitespace is preserved in lowercase
        let input = vec!["PER ", " ORG"];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&input, "nuner");

        // Current behavior: whitespace is preserved (lowercased)
        // Note: "PER " -> "per " (lowercased, whitespace preserved)
        // Note: " ORG" -> " org" (lowercased, whitespace preserved)
        assert_eq!(mapped[0], "per "); // Lowercased but not trimmed
        assert_eq!(mapped[1], " org"); // Lowercased but not trimmed (note: leading space)

        // Without whitespace, mapping works correctly
        let input_clean = vec!["PER", "ORG"];
        let mapped_clean = TaskEvaluator::map_dataset_labels_to_model(&input_clean, "nuner");
        assert_eq!(mapped_clean[0], "person");
        assert_eq!(mapped_clean[1], "organization");
    }

    #[test]
    fn test_label_mapping_special_characters() {
        // Labels with special characters should be preserved (lowercased)
        let input = vec!["PER-SON", "ORG_ENTITY"];
        let mapped = TaskEvaluator::map_dataset_labels_to_model(&input, "nuner");

        // Should be lowercased but otherwise preserved
        assert_eq!(mapped[0], "per-son");
        assert_eq!(mapped[1], "org_entity");
    }
}
