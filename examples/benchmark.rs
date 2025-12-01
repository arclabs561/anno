//! Quality benchmark comparing NER backends on synthetic and real datasets.
//!
//! Run with:
//!   cargo run --example benchmark                              # Zero-dep backends, synthetic only
//!   cargo run --example benchmark --features onnx              # + BERT ONNX backend
//!   cargo run --example benchmark --features eval-advanced     # + real dataset evaluation
//!   cargo run --example benchmark --features onnx,eval-advanced # Full evaluation
//!
//! Shows:
//! - Per-backend quality metrics (F1, Precision, Recall)
//! - Per-difficulty breakdown (Easy/Medium/Hard/Adversarial)
//! - Per-domain breakdown (News/Financial/Technical/etc.)
//! - Variance across domains using MetricWithVariance
//! - Real dataset evaluation (WikiGold, WNUT-17, CoNLL-2003) when eval-advanced feature enabled
//! - Gender bias evaluation (WinoBias-style)
//! - Demographic bias evaluation (ethnicity, region, script)

use anno::eval::calibration::{calibration_grade, CalibrationEvaluator};
use anno::eval::dataset_comparison::{compare_datasets, estimate_difficulty};
use anno::eval::dataset_quality::check_leakage;
use anno::eval::demographic_bias::{
    create_diverse_location_dataset, create_diverse_name_dataset, DemographicBiasEvaluator,
};
use anno::eval::drift::{DriftConfig, DriftDetector};
use anno::eval::gender_bias::{create_winobias_templates, GenderBiasEvaluator};
use anno::eval::harness::{EvalConfig, EvalHarness};
use anno::eval::learning_curve::{DataPoint, LearningCurveAnalyzer};
use anno::eval::length_bias::{create_length_varied_dataset, EntityLengthEvaluator};
use anno::eval::ood_detection::{ood_rate_grade, OODDetector};
use anno::eval::robustness::{robustness_grade, RobustnessEvaluator};
use anno::eval::synthetic::Domain;
use anno::eval::synthetic::{all_datasets, dataset_stats};
use anno::eval::temporal_bias::{create_temporal_name_dataset, TemporalBiasEvaluator};
use anno::eval::threshold_analysis::{
    interpret_curve, PredictionWithConfidence, ThresholdAnalyzer,
};
use anno::eval::MetricWithVariance;
use anno::eval::SimpleCorefResolver;
use anno::RegexNER;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NER Quality Benchmark ===\n");

    // Dataset overview
    let stats = dataset_stats();
    println!(
        "Dataset: {} examples, {} entities",
        stats.total_examples, stats.total_entities
    );
    println!(
        "Domains: {:?}",
        stats.examples_per_domain.keys().collect::<Vec<_>>()
    );
    println!(
        "Difficulties: {:?}\n",
        stats.examples_per_difficulty.keys().collect::<Vec<_>>()
    );

    // Configure evaluation
    let config = EvalConfig {
        breakdown_by_difficulty: true,
        breakdown_by_domain: true,
        warmup: true,
        warmup_iterations: 3,
        ..EvalConfig::default()
    };

    // Create harness with custom config and default backends
    let harness = EvalHarness::with_config(config)?;

    // Print registered backends
    println!("Backends: {}", harness.backend_count());
    for (name, desc, _) in harness.registry().iter() {
        println!("  - {}: {}", name, desc);
    }
    println!();

    // Run evaluation
    println!("Evaluating on synthetic data...\n");
    let results = harness.run_synthetic()?;

    // === Overall Results ===
    println!("=== Overall Results ===\n");
    println!(
        "{:<16} {:>8} {:>10} {:>8} {:>10} {:>12}",
        "Backend", "F1", "Precision", "Recall", "Found/Exp", "Time"
    );
    println!("{}", "-".repeat(70));

    for backend in &results.backends {
        println!(
            "{:<16} {:>7.1}% {:>9.1}% {:>7.1}% {:>5}/{:<5} {:>10.1}ms",
            backend.backend_name,
            backend.f1.mean * 100.0,
            backend.precision.mean * 100.0,
            backend.recall.mean * 100.0,
            backend.total_found,
            backend.total_expected,
            backend.total_duration_ms
        );
    }

    // === Per-type metrics for best backend ===
    // Find StackedNER and show per-type breakdown from its per_dataset results
    if let Some(stacked) = results
        .backends
        .iter()
        .find(|b| b.backend_name == "StackedNER")
    {
        if let Some(dataset_result) = stacked.per_dataset.first() {
            println!("\n=== Per-Type Metrics (StackedNER) ===\n");
            print_per_type_metrics(&dataset_result.per_type);
        }
    }

    // === Breakdown by difficulty with variance ===
    if let Some(by_difficulty) = &results.by_difficulty {
        println!("\n=== Results by Difficulty ===\n");

        // Pick StackedNER for detailed analysis
        let mut difficulty_f1s: Vec<f64> = Vec::new();

        println!(
            "{:<14} {:>8} {:>10} {:>8} {:>8}",
            "Difficulty", "F1", "Precision", "Recall", "Count"
        );
        println!("{}", "-".repeat(52));

        for difficulty in &["Easy", "Medium", "Hard", "Adversarial"] {
            if let Some(results_list) = by_difficulty.get(*difficulty) {
                // Find StackedNER result
                if let Some(result) = results_list.iter().find(|r| r.backend_name == "StackedNER") {
                    println!(
                        "{:<14} {:>7.1}% {:>9.1}% {:>7.1}% {:>8}",
                        difficulty,
                        result.f1 * 100.0,
                        result.precision * 100.0,
                        result.recall * 100.0,
                        result.num_examples
                    );
                    difficulty_f1s.push(result.f1);
                }
            }
        }

        // Show variance across difficulties
        if !difficulty_f1s.is_empty() {
            let diff_variance = MetricWithVariance::from_samples(&difficulty_f1s);
            println!("\nVariance across difficulties: {}", diff_variance);
            println!(
                "  Range: {:.1}% - {:.1}%",
                diff_variance.min * 100.0,
                diff_variance.max * 100.0
            );
        }
    }

    // === Breakdown by domain with variance ===
    if let Some(by_domain) = &results.by_domain {
        println!("\n=== Results by Domain (StackedNER) ===\n");

        let mut domain_f1s: Vec<f64> = Vec::new();

        println!(
            "{:<16} {:>8} {:>10} {:>8} {:>8}",
            "Domain", "F1", "Precision", "Recall", "Count"
        );
        println!("{}", "-".repeat(54));

        // Sort domains by F1 score for readability
        let mut domain_results: Vec<_> = by_domain
            .iter()
            .filter_map(|(domain, results_list)| {
                results_list
                    .iter()
                    .find(|r| r.backend_name == "StackedNER")
                    .map(|r| (domain, r))
            })
            .collect();
        domain_results.sort_by(|a, b| {
            b.1.f1
                .partial_cmp(&a.1.f1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (domain, result) in &domain_results {
            println!(
                "{:<16} {:>7.1}% {:>9.1}% {:>7.1}% {:>8}",
                domain,
                result.f1 * 100.0,
                result.precision * 100.0,
                result.recall * 100.0,
                result.num_examples
            );
            domain_f1s.push(result.f1);
        }

        // Show variance across domains
        if !domain_f1s.is_empty() {
            let domain_variance = MetricWithVariance::from_samples(&domain_f1s);
            println!("\nVariance across domains: {}", domain_variance);
            println!(
                "  Range: {:.1}% - {:.1}%",
                domain_variance.min * 100.0,
                domain_variance.max * 100.0
            );
            println!(
                "  CV (coefficient of variation): {:.1}%",
                domain_variance.coefficient_of_variation() * 100.0
            );
        }
    }

    // === Entity type distribution ===
    println!("\n=== Entity Type Distribution in Dataset ===\n");
    println!("{:<20} {:>8} {:>10}", "Type", "Count", "Percent");
    println!("{}", "-".repeat(40));

    let mut sorted_types: Vec<_> = results
        .dataset_stats
        .entity_type_distribution
        .iter()
        .collect();
    sorted_types.sort_by(|a, b| b.1.cmp(a.1));

    let total: usize = sorted_types.iter().map(|(_, c)| **c).sum();
    for (type_name, count) in sorted_types {
        println!(
            "{:<20} {:>8} {:>9.1}%",
            type_name,
            count,
            (*count as f64 / total as f64) * 100.0
        );
    }

    // === Summary ===
    println!("\n=== Summary ===\n");

    // Find best backend
    if let Some(best) = results.backends.iter().max_by(|a, b| {
        a.f1.mean
            .partial_cmp(&b.f1.mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "Best backend: {} (F1: {:.1}%)",
            best.backend_name,
            best.f1.mean * 100.0
        );
    }

    println!("\nKey observations:");
    println!("  - RegexNER: High precision on DATE/MONEY/PERCENT/EMAIL/URL/PHONE");
    println!("  - HeuristicNER: Baseline for PER/ORG/LOC (heuristic-based)");
    println!("  - StackedNER: Best zero-dependency option");

    #[cfg(not(feature = "onnx"))]
    {
        println!("\nTo test ML backends with higher accuracy:");
        println!("  cargo run --example quality_bench --features onnx");
    }

    #[cfg(feature = "onnx")]
    {
        println!("\nML backend (BertNEROnnx) provides significant improvement on named entities.");
    }

    // Optionally save HTML report
    if std::env::var("SAVE_HTML").is_ok() {
        let html = results.to_html();
        std::fs::write("eval_results.html", &html)?;
        println!("\nHTML report saved to eval_results.html");
    }

    // === Real Dataset Evaluation ===
    #[cfg(feature = "eval-advanced")]
    {
        println!("\n=== Real Dataset Evaluation ===\n");
        run_real_dataset_evaluation()?;
    }
    #[cfg(not(feature = "eval-advanced"))]
    {
        println!("\n=== Real Dataset Evaluation ===\n");
        println!("Skipped (requires --features eval-advanced)");
        println!("Real datasets: WikiGold, WNUT-17, CoNLL-2003");
    }

    // === Bias Evaluation ===
    println!("\n=== Bias Evaluation ===\n");
    run_bias_evaluation()?;

    Ok(())
}

/// Run comprehensive bias evaluations
fn run_bias_evaluation() -> Result<(), Box<dyn std::error::Error>> {
    // --- Gender Bias (WinoBias-style) ---
    println!("--- Gender Bias (Coreference) ---\n");

    let resolver = SimpleCorefResolver::default();
    let templates = create_winobias_templates();
    let gender_evaluator = GenderBiasEvaluator::new(false);
    let gender_results = gender_evaluator.evaluate_resolver(&resolver, &templates);

    println!(
        "{:<20} {:>10} {:>12}",
        "Stereotype Type", "Accuracy", "Count"
    );
    println!("{}", "-".repeat(44));
    println!(
        "{:<20} {:>9.1}% {:>12}",
        "Pro-stereotypical",
        gender_results.pro_stereotype_accuracy * 100.0,
        gender_results.num_pro
    );
    println!(
        "{:<20} {:>9.1}% {:>12}",
        "Anti-stereotypical",
        gender_results.anti_stereotype_accuracy * 100.0,
        gender_results.num_anti
    );
    println!(
        "\nBias Gap: {:.1}% (lower is better)",
        gender_results.bias_gap * 100.0
    );

    if !gender_results.per_pronoun.is_empty() {
        println!("\nPer-Pronoun Accuracy:");
        let mut pronouns: Vec<_> = gender_results.per_pronoun.iter().collect();
        pronouns.sort_by(|a, b| a.0.cmp(b.0));
        for (pronoun, accuracy) in pronouns {
            println!("  {:<8}: {:.1}%", pronoun, accuracy * 100.0);
        }
    }

    // --- Demographic Bias (NER) ---
    println!("\n--- Demographic Bias (NER) ---\n");
    println!("⚠️  NOTE: RegexNER cannot detect PERSON or LOCATION entities.");
    println!("    These bias evaluations require ML backends (--features onnx).");
    println!("    Results below demonstrate the API structure only.\n");

    let ner = RegexNER::new();
    let names = create_diverse_name_dataset();
    let locations = create_diverse_location_dataset();
    let demo_evaluator = DemographicBiasEvaluator::new(false);

    let name_results = demo_evaluator.evaluate_ner(&ner, &names);
    let location_results = demo_evaluator.evaluate_locations(&ner, &locations);

    // Only show results if there's actual data (i.e., ML backend detected something)
    let has_name_data = name_results.by_ethnicity.values().any(|&v| v > 0.0);
    let has_location_data = location_results.by_region.values().any(|&v| v > 0.0);

    if has_name_data {
        println!("Name Recognition by Ethnicity:");
        println!("{:<20} {:>12}", "Ethnicity", "Recognition");
        println!("{}", "-".repeat(34));

        let mut ethnicity_sorted: Vec<_> = name_results.by_ethnicity.iter().collect();
        ethnicity_sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (ethnicity, rate) in ethnicity_sorted {
            println!("{:<20} {:>11.1}%", ethnicity, rate * 100.0);
        }

        println!(
            "\nEthnicity Parity Gap: {:.1}% (lower is better)",
            name_results.ethnicity_parity_gap * 100.0
        );
        println!(
            "Script Bias Gap: {:.1}% (Latin vs non-Latin)",
            name_results.script_bias_gap * 100.0
        );
    } else {
        println!("Name Recognition: [SKIPPED - RegexNER cannot detect PERSON]");
    }

    if has_location_data {
        println!("\nLocation Recognition by Region:");
        println!("{:<20} {:>12}", "Region", "Recognition");
        println!("{}", "-".repeat(34));

        let mut region_sorted: Vec<_> = location_results.by_region.iter().collect();
        region_sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (region, rate) in region_sorted {
            println!("{:<20} {:>11.1}%", region, rate * 100.0);
        }

        println!(
            "\nRegional Parity Gap: {:.1}% (lower is better)",
            location_results.regional_parity_gap * 100.0
        );
    } else {
        println!("\nLocation Recognition: [SKIPPED - RegexNER cannot detect LOCATION]");
    }

    // --- Temporal Bias (Names by Decade) ---
    println!("\n--- Temporal Bias (Names by Decade) ---\n");

    let temporal_names = create_temporal_name_dataset();
    let temporal_evaluator = TemporalBiasEvaluator::default();
    let temporal_results = temporal_evaluator.evaluate(&ner, &temporal_names);

    let has_temporal_data =
        temporal_results.historical_rate > 0.0 || temporal_results.modern_rate > 0.0;

    if has_temporal_data {
        println!("{:<20} {:>12}", "Time Period", "Recognition");
        println!("{}", "-".repeat(34));
        println!(
            "{:<20} {:>11.1}%",
            "Historical (pre-1950)",
            temporal_results.historical_rate * 100.0
        );
        println!(
            "{:<20} {:>11.1}%",
            "Modern (post-2000)",
            temporal_results.modern_rate * 100.0
        );
        println!(
            "{:<20} {:>11.1}%",
            "Classic names",
            temporal_results.classic_rate * 100.0
        );
        println!(
            "{:<20} {:>11.1}%",
            "Trendy names",
            temporal_results.trendy_rate * 100.0
        );

        println!(
            "\nHistorical-Modern Gap: {:.1}% (lower is better)",
            temporal_results.historical_modern_gap * 100.0
        );
        println!(
            "Temporal Parity Gap: {:.1}% (max gap across decades)",
            temporal_results.temporal_parity_gap * 100.0
        );
    } else {
        println!("[SKIPPED - RegexNER cannot detect PERSON entities]");
    }

    // --- Entity Length Bias ---
    println!("\n--- Entity Length Bias ---\n");

    let length_examples = create_length_varied_dataset();
    let length_evaluator = EntityLengthEvaluator::default();
    let length_results = length_evaluator.evaluate(&ner, &length_examples);

    println!("{:<16} {:>12}", "Length Bucket", "Recognition");
    println!("{}", "-".repeat(30));

    let mut char_sorted: Vec<_> = length_results.by_char_bucket.iter().collect();
    char_sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (bucket, rate) in char_sorted {
        println!("{:<16} {:>11.1}%", bucket, rate * 100.0);
    }

    println!(
        "\nChar Length Parity Gap: {:.1}%",
        length_results.char_length_parity_gap * 100.0
    );
    println!(
        "Short vs Long Gap: {:.1}%",
        length_results.short_vs_long_gap * 100.0
    );

    if length_results.avg_recognized_char_length > 0.0
        || length_results.avg_missed_char_length > 0.0
    {
        println!(
            "Avg recognized entity length: {:.1} chars",
            length_results.avg_recognized_char_length
        );
        println!(
            "Avg missed entity length: {:.1} chars",
            length_results.avg_missed_char_length
        );
    }

    // --- Name Frequency Bias ---
    println!("\n--- Name Frequency Bias ---\n");

    println!("{:<16} {:>12}", "Frequency", "Recognition");
    println!("{}", "-".repeat(30));

    let mut freq_sorted: Vec<_> = name_results.by_frequency.iter().collect();
    freq_sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (freq, rate) in freq_sorted {
        println!("{:<16} {:>11.1}%", freq, rate * 100.0);
    }

    // --- Robustness Testing ---
    println!("\n--- Robustness Testing ---\n");

    // Create test cases with some DATE entities RegexNER can handle
    let robustness_cases: Vec<(String, Vec<anno::Entity>)> = vec![
        (
            "Meeting on January 15, 2024.".to_string(),
            vec![anno::Entity::new(
                "January 15, 2024",
                anno::EntityType::Date,
                11,
                27,
                0.95,
            )],
        ),
        (
            "Cost: $500.00 total.".to_string(),
            vec![anno::Entity::new(
                "$500.00",
                anno::EntityType::Money,
                6,
                13,
                0.95,
            )],
        ),
    ];

    let robustness_eval = RobustnessEvaluator::default();
    let robustness_results = robustness_eval.evaluate(&ner, &robustness_cases);

    println!(
        "Baseline F1: {:.1}%",
        robustness_results.baseline_f1 * 100.0
    );
    println!(
        "Avg Perturbed F1: {:.1}%",
        robustness_results.avg_perturbed_f1 * 100.0
    );
    println!(
        "Robustness Score: {:.1}% ({})",
        robustness_results.robustness_score * 100.0,
        robustness_grade(robustness_results.robustness_score)
    );
    println!(
        "Worst perturbation: {}",
        robustness_results.worst_perturbation
    );

    // --- OOD Detection Demo ---
    println!("\n--- OOD Detection ---\n");
    println!("⚠️  NOTE: This OOD detector uses vocabulary overlap only.");
    println!("    It detects novel tokens, NOT semantic distribution shift.");
    println!("    For embedding-based OOD detection, use ML backends.\n");

    let training_entities = vec![
        "John Smith",
        "Jane Doe",
        "Google",
        "Microsoft",
        "New York",
        "London",
    ];
    let ood_detector = OODDetector::default().fit(&training_entities);

    let test_entities: Vec<(&str, Option<f64>)> = vec![
        ("John Smith", Some(0.95)),    // In-distribution (exact match)
        ("Jane Doe", Some(0.90)),      // In-distribution (exact match)
        ("Xiangjun Chen", Some(0.45)), // OOD (unfamiliar tokens)
        ("山田太郎", Some(0.30)),      // OOD (different script)
    ];

    let ood_results = ood_detector.analyze(&test_entities);
    println!(
        "OOD Rate: {:.1}% ({})",
        ood_results.ood_rate * 100.0,
        ood_rate_grade(ood_results.ood_rate)
    );
    println!(
        "Vocab Coverage: {:.1}%",
        ood_results.vocab_stats.coverage_ratio * 100.0
    );
    if !ood_results.sample_ood_entities.is_empty() {
        println!("Sample OOD entities: {:?}", ood_results.sample_ood_entities);
    }
    println!("\nLimitations:");
    println!("  - Cannot detect domain shift with overlapping vocabulary");
    println!("  - \"John Smith\" in medical text would appear in-distribution");
    println!("  - For production use, combine with embedding-space analysis");

    // --- Dataset Quality Demo ---
    println!("\n--- Dataset Quality Metrics ---\n");

    // Quick leakage check using synthetic data
    let all_examples = all_datasets();
    let (train_texts, test_texts): (Vec<_>, Vec<_>) = all_examples
        .iter()
        .enumerate()
        .partition(|(i, _)| *i % 5 != 0); // 80/20 split

    let train_strs: Vec<_> = train_texts.iter().map(|(_, e)| e.text.as_str()).collect();
    let test_strs: Vec<_> = test_texts.iter().map(|(_, e)| e.text.as_str()).collect();

    let (leaked, leak_ratio) = check_leakage(&train_strs, &test_strs);
    println!("Train/Test Leakage Check:");
    println!("  Leaked samples: {} ({:.1}%)", leaked, leak_ratio * 100.0);
    println!(
        "  Status: {}",
        if leaked == 0 { "Clean" } else { "Warning!" }
    );

    // --- Calibration Demo ---
    println!("\n--- Calibration Demo ---\n");
    println!("⚠️  IMPORTANT: Calibration metrics are only meaningful for probabilistic");
    println!("    confidence scores (e.g., softmax outputs from neural models).");
    println!("    RegexNER outputs hardcoded values (0.95) - NOT calibrated.");
    println!("    HeuristicNER outputs heuristic scores - NOT calibrated.");
    println!("    Use ExtractionMethod::is_calibrated() to check.\n");

    // Simulated predictions - demonstrating what neural model output looks like
    println!("Demo using simulated neural model outputs:");
    let predictions = vec![
        (0.95, true),  // High confidence, correct
        (0.88, true),  // High confidence, correct
        (0.75, true),  // Medium confidence, correct
        (0.60, false), // Low confidence, incorrect (good calibration)
        (0.55, false), // Low confidence, incorrect (good calibration)
        (0.92, false), // High confidence, incorrect (overconfident!)
    ];

    let cal_results = CalibrationEvaluator::compute(&predictions);
    println!("Expected Calibration Error (ECE): {:.3}", cal_results.ece);
    println!("Calibration Grade: {}", calibration_grade(cal_results.ece));
    println!(
        "Confidence Gap: {:.1}% (correct: {:.0}%, incorrect: {:.0}%)",
        cal_results.confidence_gap * 100.0,
        cal_results.avg_confidence_correct * 100.0,
        cal_results.avg_confidence_incorrect * 100.0
    );

    println!("\nTo run calibration on real predictions:");
    println!("  cargo run --example quality_bench --features onnx");

    // --- Learning Curve Demo ---
    println!("\n--- Learning Curve Analysis ---\n");
    println!("Demo using hypothetical training data points.");
    println!("In production, collect these by training at different data sizes.\n");

    // Simulated learning curve data (typical NER model behavior)
    let learning_data = vec![
        DataPoint {
            train_size: 100,
            f1: 0.55,
            precision: 0.58,
            recall: 0.52,
        },
        DataPoint {
            train_size: 500,
            f1: 0.72,
            precision: 0.75,
            recall: 0.69,
        },
        DataPoint {
            train_size: 1000,
            f1: 0.80,
            precision: 0.82,
            recall: 0.78,
        },
        DataPoint {
            train_size: 2000,
            f1: 0.84,
            precision: 0.85,
            recall: 0.83,
        },
        DataPoint {
            train_size: 5000,
            f1: 0.87,
            precision: 0.88,
            recall: 0.86,
        },
    ];

    let curve_analyzer = LearningCurveAnalyzer::new(learning_data);
    let curve_analysis = curve_analyzer.analyze();

    println!(
        "Data Efficiency: {:.2}% F1 per 100 samples",
        curve_analysis.efficiency.f1_per_100_samples
    );
    println!(
        "Saturation Level: {:.0}%",
        curve_analysis.efficiency.saturation_level * 100.0
    );
    println!(
        "More data would help: {}",
        if curve_analysis.more_data_would_help() {
            "Yes"
        } else {
            "No (saturated)"
        }
    );

    if let Some(samples) = curve_analysis.samples_for_target(0.90) {
        println!("Estimated samples for 90% F1: ~{}", samples);
    }

    println!("\nNote: Extrapolation assumes log-linear learning curve.");
    println!("      Real gains depend on data quality and diversity.");

    // === Threshold Analysis (Precision-Recall Curves) ===
    println!("\n=== Threshold Analysis ===\n");

    // Simulate predictions with varying confidence levels
    let simulated_predictions = vec![
        PredictionWithConfidence::new("John Smith", "PER", 0.95, true),
        PredictionWithConfidence::new("Google", "ORG", 0.92, true),
        PredictionWithConfidence::new("New York", "LOC", 0.88, true),
        PredictionWithConfidence::new("maybe-person", "PER", 0.45, false), // FP
        PredictionWithConfidence::new("Apple", "ORG", 0.78, true),
        PredictionWithConfidence::new("random", "PER", 0.35, false), // FP
        PredictionWithConfidence::new("London", "LOC", 0.85, true),
        PredictionWithConfidence::new("unclear", "ORG", 0.55, false), // FP
        PredictionWithConfidence::new("Microsoft", "ORG", 0.91, true),
        PredictionWithConfidence::new("Paris", "LOC", 0.82, true),
    ];

    let threshold_analyzer = ThresholdAnalyzer::new(10);
    let curve = threshold_analyzer.analyze(&simulated_predictions);

    println!(
        "Total predictions: {}, Correct: {}",
        curve.total_predictions, curve.total_correct
    );
    println!("Optimal threshold: {:.2}", curve.optimal_threshold);
    println!("  F1 at optimal: {:.1}%", curve.optimal_f1 * 100.0);
    println!(
        "  Precision: {:.1}%, Recall: {:.1}%",
        curve.optimal_precision * 100.0,
        curve.optimal_recall * 100.0
    );
    println!("AUC-PR: {:.3}", curve.auc_pr);

    if let Some(t) = curve.high_precision_threshold {
        println!("High-precision (>=95%) threshold: {:.2}", t);
    }

    println!("\nInsights:");
    for insight in interpret_curve(&curve) {
        println!("  - {}", insight);
    }

    // === Dataset Comparison (Cross-Domain Analysis) ===
    println!("\n=== Dataset Comparison ===\n");

    // Compare different synthetic dataset domains
    let all_data = all_datasets();
    let news_data: Vec<_> = all_data
        .iter()
        .filter(|e| matches!(e.domain, Domain::News))
        .cloned()
        .collect();
    let tech_data: Vec<_> = all_data
        .iter()
        .filter(|e| matches!(e.domain, Domain::Technical))
        .cloned()
        .collect();

    if !news_data.is_empty() && !tech_data.is_empty() {
        let comparison = compare_datasets(&news_data, &tech_data);

        println!("Comparing News vs Technical domains:");
        println!(
            "  News: {} examples, {} entities",
            comparison.stats_a.num_examples, comparison.stats_a.num_entities
        );
        println!(
            "  Tech: {} examples, {} entities",
            comparison.stats_b.num_examples, comparison.stats_b.num_entities
        );
        println!(
            "  Type divergence: {:.3} (0=identical, 1=disjoint)",
            comparison.type_divergence
        );
        println!(
            "  Vocabulary overlap: {:.1}%",
            comparison.vocab_overlap * 100.0
        );
        println!(
            "  Entity overlap: {:.1}%",
            comparison.entity_text_overlap * 100.0
        );
        println!(
            "  Estimated domain gap: {:.2}",
            comparison.estimated_domain_gap
        );

        if !comparison.recommendations.is_empty() {
            println!("\nRecommendations:");
            for rec in &comparison.recommendations {
                println!("  - {}", rec);
            }
        }

        // Show difficulty estimate for news domain
        let news_difficulty = estimate_difficulty(&comparison.stats_a);
        println!(
            "\nNews domain difficulty: {:?} (score: {:.2})",
            news_difficulty.difficulty, news_difficulty.score
        );
        if !news_difficulty.factors.is_empty() {
            for factor in &news_difficulty.factors {
                println!("  - {}", factor);
            }
        }
    }

    // === Drift Detection Demo ===
    println!("\n=== Drift Detection Demo ===\n");

    let mut drift_detector = DriftDetector::new(DriftConfig {
        min_samples: 10,
        window_size: 10,
        num_windows: 2,
        confidence_drift_threshold: 0.1,
        ..Default::default()
    });

    // Simulate first window: high confidence, consistent types
    for i in 0..10 {
        drift_detector.log_prediction(i as u64, 0.92, "PER", "John Smith");
    }

    // Simulate second window: lower confidence, new vocabulary
    for i in 10..20 {
        drift_detector.log_prediction(i as u64, 0.65, "PER", "Xiangjun Wei");
    }

    let drift_report = drift_detector.analyze();
    println!("Drift detected: {}", drift_report.drift_detected);
    println!("Summary: {}", drift_report.summary);

    if drift_report.confidence_drift.is_significant {
        println!(
            "  Confidence drift: {:.2} -> {:.2} (change: {:.2})",
            drift_report.confidence_drift.baseline_mean,
            drift_report.confidence_drift.current_mean,
            drift_report.confidence_drift.drift_amount
        );
    }

    if drift_report.vocabulary_drift.is_significant {
        println!(
            "  Vocabulary drift: {:.1}% new tokens",
            drift_report.vocabulary_drift.new_token_rate * 100.0
        );
    }

    if !drift_report.recommendations.is_empty() {
        println!("\nRecommendations:");
        for rec in &drift_report.recommendations {
            println!("  - {}", rec);
        }
    }

    // --- Summary ---
    println!("\n--- Summary ---\n");
    println!("Note: RegexNER only detects structured entities (DATE/MONEY/etc.),");
    println!("not PERSON/LOCATION, so demographic/temporal bias results will be 0%.");
    println!("For meaningful bias evaluation, use ML backends:");
    println!("  cargo run --example quality_bench --features onnx");

    println!("\nKey research findings (Mishra et al. 2020, Jeong & Kang 2021):");
    println!("  - Character-based models (ELMo-style) show least demographic bias");
    println!("  - Debiased embeddings do NOT help resolve NER bias");
    println!("  - Entity length bias correlates with training data distribution");

    println!("\nEvaluation modules available:");
    println!("  - Gender bias (WinoBias-style coreference)");
    println!("  - Demographic bias (ethnicity, region, script)");
    println!("  - Temporal bias (names by decade)");
    println!("  - Entity length bias");
    println!("  - Robustness testing (perturbations)");
    println!("  - Calibration metrics (ECE, confidence gap)");
    println!("  - OOD detection (vocabulary coverage)");
    println!("  - Dataset quality (leakage, redundancy)");
    println!("  - Learning curve analysis");
    println!("  - Threshold analysis (precision-recall curves)");
    println!("  - Dataset comparison (cross-domain analysis)");
    println!("  - Drift detection (production monitoring)");

    Ok(())
}

fn print_per_type_metrics(per_type: &HashMap<String, anno::eval::TypeMetrics>) {
    let mut sorted_types: Vec<_> = per_type.iter().collect();
    sorted_types.sort_by(|a, b| {
        b.1.f1
            .partial_cmp(&a.1.f1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "{:<3} {:<14} {:>8} {:>10} {:>8} {:>12}",
        "", "Type", "F1", "Precision", "Recall", "Correct/Exp"
    );
    println!("{}", "-".repeat(60));

    for (entity_type, metrics) in sorted_types {
        let status = if metrics.f1 > 0.9 {
            "[+]" // Excellent
        } else if metrics.f1 > 0.7 {
            "[~]" // Good
        } else if metrics.f1 > 0.3 {
            "[?]" // Moderate
        } else if metrics.expected > 0 {
            "[-]" // Poor
        } else {
            "   " // N/A
        };

        println!(
            "{:<3} {:<14} {:>7.1}% {:>9.1}% {:>7.1}% {:>6}/{:<5}",
            status,
            entity_type,
            metrics.f1 * 100.0,
            metrics.precision * 100.0,
            metrics.recall * 100.0,
            metrics.correct,
            metrics.expected
        );
    }
}

/// Run evaluation on real NER datasets
#[cfg(feature = "eval-advanced")]
fn run_real_dataset_evaluation() -> Result<(), Box<dyn std::error::Error>> {
    use anno::eval::datasets::GoldEntity;
    use anno::eval::evaluate_ner_model;
    use anno::eval::loader::{DatasetId, DatasetLoader};
    use anno::StackedNER;

    let loader = DatasetLoader::new()?;
    let model = StackedNER::default();

    let datasets = [
        DatasetId::WikiGold,
        DatasetId::Wnut17,
        DatasetId::CoNLL2003Sample,
        DatasetId::BC5CDR,        // Biomedical (chemicals, diseases)
        DatasetId::MitRestaurant, // Domain-specific (food/location)
    ];

    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>12}",
        "Dataset", "F1", "Precision", "Recall", "Entities"
    );
    println!("{}", "-".repeat(64));

    for dataset_id in &datasets {
        match loader.load_or_download(*dataset_id) {
            Ok(dataset) => {
                // Convert to evaluation format
                let test_cases: Vec<(String, Vec<GoldEntity>)> = dataset
                    .sentences
                    .iter()
                    .filter(|s| !s.tokens.is_empty())
                    .map(|s| (s.text(), s.entities()))
                    .collect();

                let total_entities: usize = test_cases.iter().map(|(_, e)| e.len()).sum();

                // Sample for efficiency (max 500 sentences)
                let sample: Vec<_> = test_cases.into_iter().take(500).collect();

                match evaluate_ner_model(&model, &sample) {
                    Ok(results) => {
                        println!(
                            "{:<20} {:>9.1}% {:>9.1}% {:>9.1}% {:>12}",
                            dataset_id.name(),
                            results.f1 * 100.0,
                            results.precision * 100.0,
                            results.recall * 100.0,
                            total_entities
                        );
                    }
                    Err(e) => {
                        println!("{:<20} Eval error: {}", dataset_id.name(), e);
                    }
                }
            }
            Err(e) => {
                println!("{:<20} Load error: {}", dataset_id.name(), e);
            }
        }
    }

    println!("\nNote: StackedNER (zero-dep) has limited accuracy on real datasets.");
    println!(
        "For better results, use: cargo run --example quality_bench --features onnx,eval-advanced"
    );

    Ok(())
}
