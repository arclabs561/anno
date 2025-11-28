//! Bias analysis example for NER models.
//!
//! Demonstrates how to check for length bias using pattern-based entities.
//! For demographic bias on names, use an ML-based model that detects PERSON.
//!
//! Run: cargo run --example bias_check

use anno::eval::length_bias::create_length_varied_dataset;
use anno::Model;
use anno::PatternNER;

fn main() {
    println!("=== NER Bias Analysis ===\n");

    let model = PatternNER::new();
    
    // Check supported entity types
    let supported = model.supported_types();
    println!("Supported entity types: {:?}\n", supported);

    // 1. Length Bias Analysis
    // Tests whether the model performs differently on short vs long entities
    println!("--- Entity Length Bias ---\n");
    
    let length_data = create_length_varied_dataset();
    
    let mut short_correct = 0;
    let mut short_total = 0;
    let mut long_correct = 0;
    let mut long_total = 0;
    
    for example in &length_data {
        // Create a sentence with the entity for testing
        let sentence = &example.sentence;
        let predictions = model.extract_entities(sentence, None).unwrap_or_default();
        
        let is_short = example.char_length <= 10;
        
        // Check if entity was found
        let found = predictions.iter().any(|p| {
            // Check if prediction contains the expected entity text
            let pred_text = &sentence[p.start..p.end];
            pred_text == example.entity_text
        });
        
        if is_short {
            short_total += 1;
            if found { short_correct += 1; }
        } else {
            long_total += 1;
            if found { long_correct += 1; }
        }
    }
    
    let short_rate = if short_total > 0 { short_correct as f64 / short_total as f64 } else { 0.0 };
    let long_rate = if long_total > 0 { long_correct as f64 / long_total as f64 } else { 0.0 };
    
    println!("Short entities (<=10 chars):");
    println!("  Recall: {:.1}% ({}/{})", short_rate * 100.0, short_correct, short_total);
    println!("\nLong entities (>10 chars):");
    println!("  Recall: {:.1}% ({}/{})", long_rate * 100.0, long_correct, long_total);
    println!("\nGap: {:.1}%", (short_rate - long_rate).abs() * 100.0);
    
    // Interpretation
    let gap = (short_rate - long_rate).abs();
    println!("\nInterpretation:");
    if gap < 0.05 {
        println!("  Minimal length bias detected.");
    } else if gap < 0.15 {
        println!("  Moderate length bias. Consider testing on more varied entity lengths.");
    } else {
        println!("  Significant length bias. Model performs differently on short vs long entities.");
    }

    // 2. Pattern Coverage Analysis
    println!("\n--- Pattern Coverage by Type ---\n");
    
    // Test each supported type with sample entities
    let test_cases = vec![
        ("DATE", vec!["January 15", "2024-01-01", "March 3rd, 2024"]),
        ("TIME", vec!["3:00 PM", "14:30", "noon"]),
        ("EMAIL", vec!["user@example.com", "test@test.org"]),
        ("MONEY", vec!["$1,234.56", "€500", "$99"]),
        ("URL", vec!["https://example.com", "http://test.org"]),
    ];
    
    for (entity_type, examples) in test_cases {
        let mut found = 0;
        for example in &examples {
            let text = format!("Test: {}", example);
            let predictions = model.extract_entities(&text, None).unwrap_or_default();
            if predictions.iter().any(|p| &text[p.start..p.end] == *example) {
                found += 1;
            }
        }
        let rate = found as f64 / examples.len() as f64;
        println!("  {:6}: {:.0}% coverage ({}/{})", 
            entity_type, rate * 100.0, found, examples.len());
    }

    println!("\n=== Summary ===");
    println!("\nPatternNER is best suited for:");
    println!("  - Structured patterns (dates, emails, URLs, money)");
    println!("\nFor name/organization analysis, use an ML model with:");
    println!("  cargo run --example bias_check --features onnx");
}
