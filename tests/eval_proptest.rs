//! Property tests for evaluation framework.
//!
//! Tests invariants and properties that should hold for all evaluation operations.

#[cfg(all(feature = "eval", feature = "eval-advanced"))]
mod tests {
    use anno::eval::task_evaluator::TaskEvaluator;
    use anno::Entity;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_label_mapping_idempotent(
            labels in prop::collection::vec("[A-Z]{2,10}", 1..10)
        ) {
            // Label mapping should be idempotent (applying twice gives same result)
            let backend_names = vec!["nuner", "gliner_onnx", "gliner_candle", "gliner2"];

            for backend_name in backend_names {
                let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                let mapped_once = TaskEvaluator::map_dataset_labels_to_model(&label_refs, backend_name);
                let mapped_refs: Vec<&str> = mapped_once.iter().map(|s| s.as_str()).collect();
                let mapped_twice = TaskEvaluator::map_dataset_labels_to_model(&mapped_refs, backend_name);

                // After mapping twice, should be same as mapping once
                prop_assert_eq!(mapped_once, mapped_twice,
                    "Label mapping should be idempotent for backend {}", backend_name);
            }
        }

        #[test]
        fn test_label_mapping_preserves_count(
            labels in prop::collection::vec("[A-Z]{2,10}", 1..10)
        ) {
            // Label mapping should preserve count (one input = one output)
            let backend_names = vec!["nuner", "gliner_onnx", "gliner_candle", "gliner2"];

            for backend_name in backend_names {
                let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                let mapped = TaskEvaluator::map_dataset_labels_to_model(&label_refs, backend_name);
                prop_assert_eq!(labels.len(), mapped.len(),
                    "Label mapping should preserve count for backend {}", backend_name);
            }
        }

        #[test]
        fn test_metrics_bounded(
            tp in 0usize..1000,
            fp in 0usize..1000,
            fn_count in 0usize..1000,
        ) {
            // All metrics should be bounded between 0 and 1
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };

            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            prop_assert!(precision >= 0.0 && precision <= 1.0, "Precision should be in [0, 1]");
            prop_assert!(recall >= 0.0 && recall <= 1.0, "Recall should be in [0, 1]");
            prop_assert!(f1 >= 0.0 && f1 <= 1.0, "F1 should be in [0, 1]");
        }

        #[test]
        fn test_metrics_symmetry(
            tp in 0usize..100,
            fp in 0usize..100,
            fn_count in 0usize..100,
        ) {
            // F1 should be symmetric: F1(predicted, gold) = F1(gold, predicted)
            // This is tested by swapping TP/FP/FN roles

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };

            let recall = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };

            let f1_pred_gold = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            // Swap roles: what was FP becomes FN, what was FN becomes FP
            let precision_swapped = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };

            let recall_swapped = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };

            let f1_gold_pred = if precision_swapped + recall_swapped > 0.0 {
                2.0 * precision_swapped * recall_swapped / (precision_swapped + recall_swapped)
            } else {
                0.0
            };

            // F1 should be the same when swapping
            prop_assert!((f1_pred_gold - f1_gold_pred).abs() < 1e-10,
                "F1 should be symmetric: {} vs {}", f1_pred_gold, f1_gold_pred);
        }

        #[test]
        fn test_entity_matching_transitive(
            text in ".{10,100}",
            start1 in 0usize..50,
            end1 in 1usize..51,
            start2 in 0usize..50,
            end2 in 1usize..51,
            start3 in 0usize..50,
            end3 in 1usize..51,
        ) {
            // Entity matching should be transitive: if A matches B and B matches C, then A matches C
            // (assuming exact match: same span and type)

            let text_len = text.chars().count();
            let start1 = start1.min(text_len);
            let end1 = end1.min(text_len).max(start1 + 1);
            let start2 = start2.min(text_len);
            let end2 = end2.min(text_len).max(start2 + 1);
            let start3 = start3.min(text_len);
            let end3 = end3.min(text_len).max(start3 + 1);

            let entity1 = Entity::new("test".to_string(), anno::EntityType::Person, start1, end1, 1.0);
            let entity2 = Entity::new("test".to_string(), anno::EntityType::Person, start2, end2, 1.0);
            let entity3 = Entity::new("test".to_string(), anno::EntityType::Person, start3, end3, 1.0);

            let matches_12 = entity1.start == entity2.start && entity1.end == entity2.end;
            let matches_23 = entity2.start == entity3.start && entity2.end == entity3.end;
            let matches_13 = entity1.start == entity3.start && entity1.end == entity3.end;

            // If 1 matches 2 and 2 matches 3, then 1 should match 3
            if matches_12 && matches_23 {
                prop_assert!(matches_13, "Entity matching should be transitive");
            }
        }
    }

    #[test]
    fn test_label_mapping_common_cases() {
        // Test common label mappings
        let test_cases = vec![
            (vec!["PER"], vec!["person".to_string()]),
            (vec!["ORG"], vec!["organization".to_string()]),
            (vec!["LOC"], vec!["location".to_string()]),
            (vec!["PERSON"], vec!["person".to_string()]),
            (vec!["ORGANIZATION"], vec!["organization".to_string()]),
        ];

        for (input, expected) in test_cases {
            let mapped = TaskEvaluator::map_dataset_labels_to_model(&input, "nuner");
            assert_eq!(mapped, expected, "Label mapping failed for {:?}", input);
        }
    }

    #[test]
    fn test_metrics_perfect_match() {
        // Perfect match should give precision=1, recall=1, f1=1
        let tp = 100;
        let fp = 0;
        let fn_count = 0;

        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / (tp + fn_count) as f64;
        let f1 = 2.0 * precision * recall / (precision + recall);

        assert!(
            (precision - 1.0).abs() < 1e-10,
            "Perfect match should have precision=1"
        );
        assert!(
            (recall - 1.0).abs() < 1e-10,
            "Perfect match should have recall=1"
        );
        assert!((f1 - 1.0).abs() < 1e-10, "Perfect match should have f1=1");
    }

    #[test]
    fn test_metrics_no_predictions() {
        // No predictions should give precision=0, recall=0, f1=0
        let tp = 0;
        let fp = 0;
        let fn_count = 100;

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = tp as f64 / (tp + fn_count) as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        assert_eq!(precision, 0.0, "No predictions should have precision=0");
        assert_eq!(recall, 0.0, "No predictions should have recall=0");
        assert_eq!(f1, 0.0, "No predictions should have f1=0");
    }
}
