//! Basic evaluation example - evaluating Pattern and Statistical NER backends.
//!
//! Run with: cargo run --example eval_basic --features eval
//!
//! This example shows:
//! - How to use the BackendEvaluator to test NER backends
//! - How to generate markdown/HTML reports
//! - How to access per-domain and per-entity-type metrics

use anno::eval::backend_eval::{BackendEvaluator, EvalConfig};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                     NER Backend Evaluation");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // 1. Quick evaluation on a subset of data
    // =========================================================================
    println!("1. Quick evaluation (50 examples)...\n");

    let config = EvalConfig {
        include_pattern: true,
        include_heuristic: true,
        include_stacked: true,
        include_gliner: true, // Requires --features onnx
        max_examples: 50,
        ..Default::default()
    };

    let evaluator = BackendEvaluator::with_config(config);
    let report = evaluator.run_comprehensive();

    println!("Evaluated {} examples\n", report.total_examples);

    // Print overall results
    println!("┌─────────────────┬───────────┬────────┬────────┐");
    println!("│ Backend         │ Precision │ Recall │ F1     │");
    println!("├─────────────────┼───────────┼────────┼────────┤");
    for backend in &report.backends {
        println!(
            "│ {:15} │ {:7.1}%  │ {:6.1}% │ {:6.1}% │",
            backend.name,
            backend.overall.precision * 100.0,
            backend.overall.recall * 100.0,
            backend.overall.f1 * 100.0,
        );
    }
    println!("└─────────────────┴───────────┴────────┴────────┘\n");

    // =========================================================================
    // 2. Domain-specific evaluation
    // =========================================================================
    println!("2. Technology domain evaluation...\n");

    let tech_evaluator = BackendEvaluator::with_config(EvalConfig {
        include_pattern: true,
        include_heuristic: false,
        include_stacked: false,
        ..Default::default()
    });

    let tech_report = tech_evaluator.run_technology();

    if let Some(backend) = tech_report.backends.first() {
        println!(
            "Technology dataset ({} examples):",
            tech_report.total_examples
        );
        println!("  Pattern NER F1: {:.1}%\n", backend.overall.f1 * 100.0);
    }

    // =========================================================================
    // 3. Healthcare domain evaluation
    // =========================================================================
    println!("3. Healthcare domain evaluation...\n");

    let health_report = tech_evaluator.run_healthcare();

    if let Some(backend) = health_report.backends.first() {
        println!(
            "Healthcare dataset ({} examples):",
            health_report.total_examples
        );
        println!("  Pattern NER F1: {:.1}%\n", backend.overall.f1 * 100.0);
    }

    // =========================================================================
    // 4. Per-entity-type breakdown
    // =========================================================================
    println!("4. Per-entity-type breakdown...\n");

    if let Some(backend) = report.backends.first() {
        println!("Entity type performance for {}:", backend.name);

        let mut types: Vec<_> = backend.by_entity_type.iter().collect();
        types.sort_by(|a, b| b.1.f1.partial_cmp(&a.1.f1).unwrap());

        for (entity_type, metrics) in types.iter().take(8) {
            let bar = "█".repeat((metrics.f1 * 20.0) as usize);
            println!("  {:12} {:5.1}% {}", entity_type, metrics.f1 * 100.0, bar);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                        Evaluation Complete");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // 5. Generate reports
    // =========================================================================
    println!("Generating reports...\n");

    // Markdown report
    let md = report.to_markdown();
    println!("Markdown report preview (first 500 chars):");
    println!("─────────────────────────────────────────────");
    println!("{}", &md[..md.len().min(500)]);
    println!("─────────────────────────────────────────────\n");

    // You can save reports to files:
    // std::fs::write("eval_report.md", &md).unwrap();
    // std::fs::write("eval_report.html", report.to_html()).unwrap();

    println!("Done! To save full reports, uncomment the file writes in the example.");
}
