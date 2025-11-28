//! Quick evaluation example using the unified EvalReport.
//!
//! This is the recommended way to evaluate an NER model.
//!
//! Run: cargo run --example quick_eval

use anno::eval::{EvalReport, ReportBuilder, TestCase};
use anno::eval::report::GoldEntity;
use anno::PatternNER;

fn main() {
    println!("=== Quick NER Evaluation ===\n");

    // Create model
    let model = PatternNER::new();
    
    // Option 1: Use built-in synthetic data (quick sanity check)
    println!("--- Evaluation on synthetic data ---\n");
    let report = ReportBuilder::new("PatternNER")
        .with_error_analysis(true)
        .build(&model);
    
    println!("{}", report.summary());
    
    // Option 2: Provide custom test data
    println!("\n--- Evaluation on custom data ---\n");
    let custom_tests = vec![
        TestCase {
            text: "Send invoice to alice@company.com by March 15".into(),
            gold_entities: vec![
                GoldEntity {
                    text: "alice@company.com".into(),
                    entity_type: "EMAIL".into(),
                    start: 16,
                    end: 33,
                },
                GoldEntity {
                    text: "March 15".into(),
                    entity_type: "DATE".into(),
                    start: 37,
                    end: 45,
                },
            ],
        },
        TestCase {
            text: "Meeting at 2:30 PM, budget $50,000".into(),
            gold_entities: vec![
                GoldEntity {
                    text: "2:30 PM".into(),
                    entity_type: "TIME".into(),
                    start: 11,
                    end: 18,
                },
                GoldEntity {
                    text: "$50,000".into(),
                    entity_type: "MONEY".into(),
                    start: 27,
                    end: 34,
                },
            ],
        },
    ];
    
    let custom_report = ReportBuilder::new("PatternNER")
        .with_test_data(custom_tests)
        .with_error_analysis(true)
        .build(&model);
    
    println!("{}", custom_report.summary());
    
    // Option 3: Export as JSON for tooling
    if let Ok(json) = custom_report.to_json() {
        println!("\n--- JSON export (first 500 chars) ---\n");
        println!("{}...", &json[..json.len().min(500)]);
    }
}

