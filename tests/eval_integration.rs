//! Integration tests for the eval module components.
//!
//! Tests that the various evaluation modules work together correctly:
//! - Drift detection
//! - Threshold analysis
//! - Dataset comparison
//! - Harness integration
//!
//! Requires `eval-advanced` feature.
#![cfg(feature = "eval-advanced")]

use anno::eval::{
    compare_datasets, compute_stats, estimate_difficulty,
    interpret_curve, DriftConfig, DriftDetector,
    PredictionWithConfidence, ThresholdAnalyzer,
};
use anno::eval::dataset_comparison::EstimatedDifficulty;
use anno::eval::synthetic::{all_datasets, Domain};
use anno::eval::harness::{EvalConfig, EvalHarness};
use anno::{Model, PatternNER};

// =============================================================================
// Drift Detection Tests
// =============================================================================

#[test]
fn test_drift_detection_no_drift_scenario() {
    let mut detector = DriftDetector::new(DriftConfig {
        min_samples: 10,
        window_size: 10,
        num_windows: 2,
        confidence_drift_threshold: 0.1,
        ..Default::default()
    });

    // Log consistent predictions
    for i in 0..20 {
        detector.log_prediction(i as u64, 0.90, "PER", "John Smith");
    }

    let report = detector.analyze();
    
    // With consistent data, confidence drift should not be significant
    assert!(!report.confidence_drift.is_significant, 
        "Expected no significant confidence drift with consistent predictions");
    
    // Check that we got expected window count
    assert!(!report.windows.is_empty(), "Should have computed windows");
}

#[test]
fn test_drift_detection_confidence_drift() {
    let mut detector = DriftDetector::new(DriftConfig {
        min_samples: 10,
        window_size: 10,
        num_windows: 2,
        confidence_drift_threshold: 0.1,
        ..Default::default()
    });

    // Window 1: High confidence
    for i in 0..10 {
        detector.log_prediction(i as u64, 0.95, "PER", "John");
    }

    // Window 2: Low confidence (significant drop)
    for i in 10..20 {
        detector.log_prediction(i as u64, 0.55, "PER", "John");
    }

    let report = detector.analyze();
    
    assert!(report.confidence_drift.is_significant,
        "Expected significant confidence drift when confidence drops from 0.95 to 0.55");
    assert!(report.confidence_drift.drift_amount < 0.0,
        "Drift amount should be negative (confidence decreased)");
}

#[test]
fn test_drift_detection_vocabulary_shift() {
    let mut detector = DriftDetector::new(DriftConfig {
        min_samples: 10,
        window_size: 10,
        num_windows: 2,
        vocab_drift_threshold: 0.3,
        ..Default::default()
    });

    // Window 1: Common names
    for i in 0..10 {
        detector.log_prediction(i as u64, 0.90, "PER", "John Smith");
    }

    // Window 2: Completely different vocabulary
    for i in 10..20 {
        detector.log_prediction(i as u64, 0.90, "PER", "Xiangjun Wei Zhang");
    }

    let report = detector.analyze();
    
    // Should detect vocabulary shift
    assert!(report.vocabulary_drift.new_token_rate > 0.0,
        "Expected non-zero new token rate with vocabulary shift");
}

#[test]
fn test_drift_detector_reset() {
    let mut detector = DriftDetector::default();
    
    // Log some data
    detector.log_prediction(0, 0.9, "PER", "Test");
    detector.log_prediction(1, 0.9, "PER", "Test");
    
    // Reset
    detector.reset();
    
    // Should report insufficient data
    let report = detector.analyze();
    assert!(!report.drift_detected);
    assert!(report.summary.contains("Insufficient") || report.summary.contains("insufficient"));
}

// =============================================================================
// Threshold Analysis Tests
// =============================================================================

#[test]
fn test_threshold_analysis_basic() {
    let predictions = vec![
        PredictionWithConfidence::new("John", "PER", 0.95, true),
        PredictionWithConfidence::new("Google", "ORG", 0.88, true),
        PredictionWithConfidence::new("wrong", "PER", 0.45, false),
        PredictionWithConfidence::new("another", "LOC", 0.30, false),
    ];

    let analyzer = ThresholdAnalyzer::new(10);
    let curve = analyzer.analyze(&predictions);

    assert_eq!(curve.total_predictions, 4);
    assert_eq!(curve.total_correct, 2);
    assert!(curve.optimal_threshold >= 0.0 && curve.optimal_threshold <= 1.0);
    assert!(curve.optimal_f1 >= 0.0 && curve.optimal_f1 <= 1.0);
    assert!(curve.auc_pr >= 0.0 && curve.auc_pr <= 1.0);
}

#[test]
fn test_threshold_analysis_perfect_predictions() {
    let predictions = vec![
        PredictionWithConfidence::new("A", "T", 0.9, true),
        PredictionWithConfidence::new("B", "T", 0.8, true),
        PredictionWithConfidence::new("C", "T", 0.7, true),
    ];

    let analyzer = ThresholdAnalyzer::new(10);
    let curve = analyzer.analyze(&predictions);

    // All correct = perfect precision at all thresholds with predictions
    for point in &curve.points {
        if point.num_predictions > 0 {
            assert!((point.precision - 1.0).abs() < 0.01,
                "Expected 100% precision for all-correct predictions, got {}", 
                point.precision);
        }
    }
}

#[test]
fn test_threshold_analysis_high_confidence_false_positives() {
    // Scenario: Model is overconfident on wrong predictions
    let predictions = vec![
        PredictionWithConfidence::new("correct1", "T", 0.99, true),
        PredictionWithConfidence::new("WRONG1", "T", 0.95, false),  // High conf FP
        PredictionWithConfidence::new("WRONG2", "T", 0.90, false),  // High conf FP
        PredictionWithConfidence::new("correct2", "T", 0.50, true),
    ];

    let analyzer = ThresholdAnalyzer::default();
    let curve = analyzer.analyze(&predictions);

    // Should have relatively low precision even at high thresholds
    let high_threshold_point = curve.points.iter()
        .find(|p| p.threshold >= 0.9)
        .expect("Should have a high threshold point");
    
    // At threshold 0.9, we keep 3 predictions (0.99, 0.95, 0.90) with 1 correct
    // Precision should be around 1/3 = 33%
    assert!(high_threshold_point.precision < 0.5,
        "Expected low precision at high threshold due to overconfident FPs");

    // Insights should note this
    let insights = interpret_curve(&curve);
    assert!(!insights.is_empty(), "Should generate insights");
}

#[test]
fn test_threshold_analysis_empty_predictions() {
    let predictions: Vec<PredictionWithConfidence> = vec![];
    
    let analyzer = ThresholdAnalyzer::default();
    let curve = analyzer.analyze(&predictions);

    assert_eq!(curve.total_predictions, 0);
    assert!(curve.points.is_empty());
}

// =============================================================================
// Dataset Comparison Tests
// =============================================================================

#[test]
fn test_dataset_comparison_same_domain() {
    let dataset = all_datasets();
    
    // Compare dataset with itself
    let comparison = compare_datasets(&dataset, &dataset);

    // Should be identical
    assert!(comparison.type_divergence < 0.01,
        "Type divergence should be ~0 for same dataset, got {}", 
        comparison.type_divergence);
    assert!((comparison.vocab_overlap - 1.0).abs() < 0.01,
        "Vocab overlap should be ~1.0 for same dataset, got {}",
        comparison.vocab_overlap);
    assert!((comparison.entity_text_overlap - 1.0).abs() < 0.01,
        "Entity overlap should be ~1.0 for same dataset");
}

#[test]
fn test_dataset_comparison_different_domains() {
    let all_data = all_datasets();
    
    let news_data: Vec<_> = all_data.iter()
        .filter(|e| matches!(e.domain, Domain::News))
        .cloned()
        .collect();
    let tech_data: Vec<_> = all_data.iter()
        .filter(|e| matches!(e.domain, Domain::Technical))
        .cloned()
        .collect();
    
    if news_data.is_empty() || tech_data.is_empty() {
        // Skip if synthetic dataset doesn't have these domains
        return;
    }

    let comparison = compare_datasets(&news_data, &tech_data);

    // Different domains should show some divergence
    assert!(comparison.type_divergence >= 0.0);
    assert!(comparison.vocab_overlap >= 0.0 && comparison.vocab_overlap <= 1.0);
    assert!(comparison.estimated_domain_gap >= 0.0);
    
    // Should generate recommendations
    assert!(!comparison.recommendations.is_empty(),
        "Should provide recommendations for cross-domain comparison");
}

#[test]
fn test_dataset_stats_computation() {
    let dataset = all_datasets();
    let stats = compute_stats(&dataset);

    assert!(stats.num_examples > 0);
    assert!(stats.num_entities > 0);
    assert!(stats.vocab_size > 0);
    assert!(!stats.type_distribution.is_empty());
    assert!(stats.avg_entities_per_example > 0.0);
}

#[test]
fn test_difficulty_estimation() {
    let dataset = all_datasets();
    let stats = compute_stats(&dataset);
    let difficulty = estimate_difficulty(&stats);

    // Should return a valid difficulty level
    assert!(matches!(
        difficulty.difficulty,
        EstimatedDifficulty::Easy | EstimatedDifficulty::Medium | EstimatedDifficulty::Hard | EstimatedDifficulty::VeryHard
    ));
    
    assert!(difficulty.score >= 0.0 && difficulty.score <= 1.0);
}

// =============================================================================
// Harness Integration Tests
// =============================================================================

#[test]
fn test_harness_with_default_config() {
    // EvalHarness::with_defaults() creates harness with default backends registered
    let harness = EvalHarness::with_defaults().expect("Should create harness");
    assert!(harness.backend_count() > 0, "Should have registered backends");
}

#[test]
fn test_harness_synthetic_evaluation() {
    let config = EvalConfig {
        breakdown_by_difficulty: true,
        breakdown_by_domain: true,
        warmup: false, // Skip warmup for faster tests
        ..EvalConfig::default()
    };

    let harness = EvalHarness::with_config(config).expect("Should create harness");
    let results = harness.run_synthetic().expect("Should run synthetic evaluation");

    // Should have results for at least one backend
    assert!(!results.backends.is_empty(), "Should have backend results");
    
    // Check that metrics are bounded correctly
    for backend in &results.backends {
        assert!(backend.f1.mean >= 0.0 && backend.f1.mean <= 1.0,
            "F1 should be in [0, 1], got {}", backend.f1.mean);
        assert!(backend.precision.mean >= 0.0 && backend.precision.mean <= 1.0,
            "Precision should be in [0, 1]");
        assert!(backend.recall.mean >= 0.0 && backend.recall.mean <= 1.0,
            "Recall should be in [0, 1]");
    }
    
    // Check breakdown by difficulty if requested
    if let Some(by_diff) = &results.by_difficulty {
        assert!(!by_diff.is_empty(), "Should have difficulty breakdown");
    }
    
    // Check breakdown by domain if requested
    if let Some(by_domain) = &results.by_domain {
        assert!(!by_domain.is_empty(), "Should have domain breakdown");
    }
}

// =============================================================================
// Cross-Module Integration Tests
// =============================================================================

#[test]
fn test_threshold_analysis_with_real_predictions() {
    // Use PatternNER to make predictions, then analyze thresholds
    let ner = PatternNER::new();
    
    let test_cases = vec![
        ("Meeting on 2024-01-15 for $100", vec![("2024-01-15", "DATE"), ("$100", "MONEY")]),
        ("The temperature is 25%", vec![("25%", "PERCENT")]),
        ("Email: test@example.com", vec![("test@example.com", "EMAIL")]),
    ];

    let mut predictions = Vec::new();
    
    for (text, expected) in test_cases {
        let entities = ner.extract_entities(text, None).expect("Should extract");
        
        for entity in entities {
            // Check if this entity matches any expected
            let is_correct = expected.iter().any(|(exp_text, _)| {
                entity.text == *exp_text
            });
            
            predictions.push(PredictionWithConfidence::new(
                &entity.text,
                entity.entity_type.as_label(),
                entity.confidence,
                is_correct,
            ));
        }
    }

    if !predictions.is_empty() {
        let analyzer = ThresholdAnalyzer::default();
        let curve = analyzer.analyze(&predictions);
        
        // Should produce valid analysis
        assert!(curve.auc_pr >= 0.0);
        assert!(!curve.points.is_empty());
    }
}

#[test]
fn test_drift_detector_with_harness_data() {
    // Use synthetic data from harness to feed drift detector
    let all_data = all_datasets();
    
    let mut detector = DriftDetector::new(DriftConfig {
        min_samples: 5,
        window_size: 5,
        num_windows: 2,
        ..Default::default()
    });

    // Log entities from synthetic dataset as if they were predictions
    for (i, example) in all_data.iter().take(20).enumerate() {
        for entity in &example.entities {
            detector.log_prediction(
                i as u64,
                0.85, // Simulated confidence
                entity.entity_type.as_label(),
                &entity.text,
            );
        }
    }

    let report = detector.analyze();
    
    // Should produce a valid report (may or may not detect drift)
    assert!(!report.summary.is_empty());
    assert!(!report.recommendations.is_empty());
}

