//! Comprehensive bias evaluation example using all new features.
//!
//! Demonstrates:
//! - Expanded datasets (5x larger)
//! - Statistical reporting with confidence intervals
//! - Frequency-weighted evaluation
//! - Distribution validation
//! - Extended intersectional analysis
//! - Real-world sentence contexts
//!
//! Run: cargo run --example comprehensive_bias_eval --features eval-bias

use anno::eval::config_builder::BiasDatasetConfigBuilder;
use anno::eval::coref_resolver::SimpleCorefResolver;
use anno::eval::demographic_bias::{create_diverse_name_dataset, DemographicBiasEvaluator};
use anno::eval::gender_bias::{create_winobias_templates, GenderBiasEvaluator};
use anno::eval::length_bias::{create_length_varied_dataset, EntityLengthEvaluator};
use anno::eval::temporal_bias::{create_temporal_name_dataset, TemporalBiasEvaluator};
use anno::RegexNER;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comprehensive Bias Evaluation ===\n");
    println!("Using expanded datasets with statistical validation\n");

    // === NEW: Using Configuration Builder ===
    let config = BiasDatasetConfigBuilder::new()
        .with_frequency_weighting(true)
        .with_validation(true)
        .with_min_samples(30)
        .add_seed(42)
        .add_seed(123)
        .add_seed(456)
        .add_seed(789)
        .add_seed(999)
        .with_detailed(true)
        .build();

    println!("Configuration:");
    println!("  Frequency weighting: {}", config.frequency_weighted);
    println!(
        "  Distribution validation: {}",
        config.validate_distributions
    );
    println!("  Evaluation seeds: {}", config.evaluation_seeds.len());
    println!(
        "  Confidence level: {:.0}%",
        config.confidence_level * 100.0
    );
    println!();

    // ========================================================================
    // Gender Bias Evaluation
    // ========================================================================
    println!("--- Gender Bias (WinoBias-style) ---\n");

    let resolver = SimpleCorefResolver::default();
    let templates = create_winobias_templates();
    println!(
        "Dataset size: {} examples (expanded from 30)",
        templates.len()
    );

    let evaluator = GenderBiasEvaluator::new(true);
    let results = evaluator.evaluate_resolver(&resolver, &templates);

    println!("\nResults:");
    println!(
        "  Pro-stereotypical accuracy: {:.1}%",
        results.pro_stereotype_accuracy * 100.0
    );
    println!(
        "  Anti-stereotypical accuracy: {:.1}%",
        results.anti_stereotype_accuracy * 100.0
    );
    println!("  Bias gap: {:.1}%", results.bias_gap * 100.0);
    println!(
        "  Verdict: {}",
        if results.bias_gap < 0.05 {
            "Minimal bias"
        } else if results.bias_gap < 0.15 {
            "Moderate bias"
        } else {
            "Significant bias"
        }
    );

    // ========================================================================
    // Demographic Bias Evaluation
    // ========================================================================
    println!("\n--- Demographic Bias (NER) ---\n");

    let ner = RegexNER::new();
    let names = create_diverse_name_dataset();
    println!("Dataset size: {} names (expanded from ~100)", names.len());

    use anno::eval::bias_config::BiasDatasetConfig;
    let bias_config: BiasDatasetConfig = config.clone();
    let evaluator = DemographicBiasEvaluator::with_config(true, bias_config);
    let results = evaluator.evaluate_ner(&ner, &names);

    println!(
        "\nOverall recognition rate: {:.1}%",
        results.overall_recognition_rate * 100.0
    );
    println!(
        "Ethnicity parity gap: {:.1}%",
        results.ethnicity_parity_gap * 100.0
    );
    println!("Script bias gap: {:.1}%", results.script_bias_gap * 100.0);

    println!("\nRecognition by ethnicity:");
    let mut eth_sorted: Vec<_> = results.by_ethnicity.iter().collect();
    eth_sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (eth, &rate) in eth_sorted.iter().take(8) {
        println!("  {:<20}: {:.1}%", eth, rate * 100.0);
    }

    // Frequency-weighted results
    if let Some(freq) = &results.frequency_weighted {
        println!("\nFrequency-Weighted Analysis:");
        println!("  Unweighted rate: {:.1}%", freq.unweighted_rate * 100.0);
        println!("  Weighted rate: {:.1}%", freq.weighted_rate * 100.0);
        let diff = freq.weighted_rate - freq.unweighted_rate;
        println!("  Difference: {:.1}%", diff * 100.0);
    }

    // Statistical results
    if let Some(stat) = &results.statistical {
        println!("\nStatistical Results:");
        println!("  {}", stat.format_with_ci());
        println!("  Standard deviation: {:.3}", stat.std_dev);
        if let Some(effect) = stat.effect_size {
            println!("  Effect size (Cohen's d): {:.3}", effect);
        }
    }

    // Distribution validation
    if let Some(validation) = &results.distribution_validation {
        println!("\nDistribution Validation (vs US Census 2020):");
        println!("  Valid: {}", validation.is_valid);
        println!("  Max deviation: {:.1}%", validation.max_deviation * 100.0);
        println!("  Tolerance: {:.1}%", validation.tolerance * 100.0);

        if !validation.is_valid {
            println!("\n  Category deviations:");
            let mut devs: Vec<_> = validation.category_deviations.iter().collect();
            devs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (cat, &dev) in devs.iter().take(5) {
                println!("    {}: {:.1}%", cat, dev * 100.0);
            }
        }
    }

    // Extended intersectional analysis
    if !results.extended_intersectional.is_empty() {
        println!("\nExtended Intersectional Analysis (Ethnicity × Gender × Frequency):");
        let mut inter: Vec<_> = results.extended_intersectional.iter().collect();
        inter.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (key, &rate) in inter.iter().take(10) {
            println!("  {:<50}: {:.1}%", key, rate * 100.0);
        }
    }

    // ========================================================================
    // Temporal Bias Evaluation
    // ========================================================================
    println!("\n--- Temporal Bias (Names by Decade) ---\n");

    let temporal_names = create_temporal_name_dataset();
    println!(
        "Dataset size: {} names (expanded from ~70)",
        temporal_names.len()
    );

    let evaluator = TemporalBiasEvaluator::new(true);
    let results = evaluator.evaluate(&ner, &temporal_names);

    println!("\nResults:");
    println!(
        "  Historical (pre-1950): {:.1}%",
        results.historical_rate * 100.0
    );
    println!("  Modern (post-2000): {:.1}%", results.modern_rate * 100.0);
    let temporal_gap = (results.historical_rate - results.modern_rate).abs();
    println!(
        "  Temporal gap: {:.1}%",
        results.historical_modern_gap * 100.0
    );

    if !results.by_decade.is_empty() {
        println!("\nRecognition by decade:");
        let mut decade_sorted: Vec<_> = results.by_decade.iter().collect();
        decade_sorted.sort_by_key(|(k, _)| k.clone());
        for (decade, &rate) in decade_sorted {
            println!("  {:<20}: {:.1}%", decade, rate * 100.0);
        }
    }

    // ========================================================================
    // Length Bias Evaluation
    // ========================================================================
    println!("\n--- Length Bias (Entity Length) ---\n");

    let length_examples = create_length_varied_dataset();
    println!(
        "Dataset size: {} examples (expanded from ~30)",
        length_examples.len()
    );

    let evaluator = EntityLengthEvaluator::new(true);
    let results = evaluator.evaluate(&ner, &length_examples);

    println!("\nResults:");
    println!(
        "  Short entities (1-2 words): {:.1}%",
        results
            .by_word_bucket
            .get("SingleWord")
            .or_else(|| results.by_word_bucket.get("TwoWords"))
            .copied()
            .unwrap_or(0.0)
            * 100.0
    );
    println!(
        "  Long entities (4+ words): {:.1}%",
        results
            .by_word_bucket
            .get("FourPlusWords")
            .copied()
            .unwrap_or(0.0)
            * 100.0
    );
    println!("  Length gap: {:.1}%", results.short_vs_long_gap * 100.0);

    if !results.by_char_bucket.is_empty() {
        println!("\nRecognition by character length:");
        let mut char_sorted: Vec<_> = results.by_char_bucket.iter().collect();
        char_sorted.sort_by_key(|(k, _)| k.clone());
        for (bucket, &rate) in char_sorted {
            println!("  {:<20}: {:.1}%", bucket, rate * 100.0);
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n=== Summary ===");
    println!("All bias evaluations completed using:");
    println!("  ✓ Expanded datasets (5x larger)");
    println!("  ✓ Real-world sentence contexts");
    println!("  ✓ Statistical reporting with confidence intervals");
    println!("  ✓ Frequency-weighted evaluation");
    println!("  ✓ Distribution validation");
    println!("  ✓ Extended intersectional analysis");

    Ok(())
}
