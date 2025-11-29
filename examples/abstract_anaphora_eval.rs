//! Generate abstract anaphora evaluation report.
//!
//! Run: `cargo run --example abstract_anaphora_eval`
//!
//! This demonstrates the gap between nominal coreference resolution
//! (which rule-based systems handle decently) and abstract anaphora
//! (which requires event detection and discourse modeling).
//!
//! Compares:
//! 1. Simple resolver (baseline) - only handles nominal coreference
//! 2. Discourse-aware resolver - uses event extraction for abstract anaphora

use anno::eval::{AbstractAnaphoraDataset, AbstractAnaphoraEvaluator};
use std::fs;

fn main() {
    println!("=== Abstract Anaphora Evaluation ===\n");
    println!("Based on research by:");
    println!("  - Kolhatkar & Hirst (2012): 'this-issue' anaphors");
    println!("  - Marasović et al. (2017): LSTM-Siamese mention-ranking");
    println!("  - Schmid (2000): Shell noun taxonomy (~670 nouns)\n");

    // Create extended dataset (includes shell noun examples)
    let dataset = AbstractAnaphoraDataset::extended();
    let stats = dataset.stats();

    println!("Extended Dataset:");
    println!("  Total cases:  {}", stats.total);
    println!("  Nominal:      {} (baseline)", stats.nominal);
    println!("  Event:        {}", stats.event);
    println!("  Fact:         {}", stats.fact);
    println!("  Proposition:  {}", stats.proposition);
    println!("  Situation:    {}", stats.situation);
    println!();

    // =========================================================================
    // BASELINE: Simple Resolver (nominal only)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("=== BASELINE: Simple Coreference Resolver ===");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let simple_evaluator = AbstractAnaphoraEvaluator::default();
    let simple_results = simple_evaluator.evaluate(&dataset);
    println!("{}", simple_results.summary());

    // =========================================================================
    // IMPROVED: Discourse-Aware Resolver (with event extraction)
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("=== IMPROVED: Discourse-Aware Resolver (with Event Extraction) ===");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let discourse_evaluator = AbstractAnaphoraEvaluator::discourse_aware();
    let discourse_results = discourse_evaluator.evaluate(&dataset);
    println!("{}", discourse_results.summary());

    // =========================================================================
    // COMPARISON
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("=== IMPROVEMENT SUMMARY ===");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let simple_gap = simple_results.accuracy_gap();
    let discourse_gap = discourse_results.accuracy_gap();
    let abstract_improvement =
        discourse_results.abstract_accuracy - simple_results.abstract_accuracy;

    println!("                      Simple     Discourse-Aware   Change");
    println!("                      ------     ---------------   ------");
    println!(
        "Nominal Accuracy:     {:5.1}%          {:5.1}%       {:+.1}pp",
        simple_results.nominal_accuracy * 100.0,
        discourse_results.nominal_accuracy * 100.0,
        (discourse_results.nominal_accuracy - simple_results.nominal_accuracy) * 100.0
    );
    println!(
        "Abstract Accuracy:    {:5.1}%          {:5.1}%       {:+.1}pp  ← KEY METRIC",
        simple_results.abstract_accuracy * 100.0,
        discourse_results.abstract_accuracy * 100.0,
        abstract_improvement * 100.0
    );
    println!(
        "Accuracy Gap:         {:5.1}pp         {:5.1}pp       {:+.1}pp",
        simple_gap * 100.0,
        discourse_gap * 100.0,
        (discourse_gap - simple_gap) * 100.0
    );

    if abstract_improvement > 0.0 {
        println!(
            "\n✓ Discourse-aware resolver improved abstract anaphora by {:.1}pp",
            abstract_improvement * 100.0
        );
    }

    // Shell noun analysis
    println!("\n=== SHELL NOUN ANALYSIS ===");
    let shell_analysis = simple_evaluator.analyze_shell_nouns(&dataset);
    println!("{}", shell_analysis.summary());

    // Generate HTML report
    let html = discourse_results.to_html(&dataset);
    let report_path = "abstract_anaphora_analysis.html";
    fs::write(report_path, &html).expect("Failed to write HTML report");
    println!("HTML report written to: {}", report_path);

    // LEA metric analysis
    println!("\n=== LEA METRIC (Moosavi & Strube 2016) ===");
    let lea = discourse_results.compute_lea_scores(&dataset);
    println!("{}", lea.summary());

    // Key finding
    println!("\n=== KEY FINDING ===");
    if discourse_results.abstract_accuracy > simple_results.abstract_accuracy {
        println!(
            "Event extraction REDUCED the gap from {:.0}pp to {:.0}pp.",
            simple_gap * 100.0,
            discourse_gap * 100.0
        );
        println!(
            "Abstract anaphora accuracy improved from {:.0}% to {:.0}%.",
            simple_results.abstract_accuracy * 100.0,
            discourse_results.abstract_accuracy * 100.0
        );
    } else {
        println!(
            "The {:.0} percentage point gap between nominal ({:.0}%) and abstract ({:.0}%)",
            discourse_gap * 100.0,
            discourse_results.nominal_accuracy * 100.0,
            discourse_results.abstract_accuracy * 100.0
        );
        println!("shows that more work is needed on event/proposition detection.");
    }

    println!("\n=== SHELL NOUN INSIGHT ===");
    println!(
        "Detected {} shell nouns (e.g., 'this problem', 'the fact').",
        shell_analysis.total_shell_nouns
    );
    println!(
        "{:.0}% are demonstrative ('this X') - strong abstract anaphora signal.",
        shell_analysis.demonstrative_ratio() * 100.0
    );
    println!(
        "{:.0}% have antecedent type matching their semantic class.",
        shell_analysis.type_match_ratio() * 100.0
    );

    println!("\n=== NEXT STEPS ===");
    println!("To further improve abstract anaphora resolution:");
    println!("  1. Enable GLiNER for neural event extraction:");
    println!("     EventExtractor::with_gliner(\"urchade/gliner_small-v2.1\")");
    println!("  2. See docs/ABSTRACT_ANAPHORA_RESEARCH.md for research directions.");
}
