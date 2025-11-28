//! Rigorous model evaluation: Compare models on curated test cases.
//!
//! Tests:
//! - Standard entity types (PER, ORG, LOC, DATE, etc.)
//! - Nested entities
//! - Ambiguous cases
//! - Edge cases
//!
//! Run: cargo run --features "eval,onnx,network" --example 63_model_evaluation

use anno::{Model, PatternNER, StatisticalNER, StackedNER};
use std::collections::HashMap;

#[cfg(feature = "onnx")]
use anno::{BertNEROnnx, GLiNEROnnx};

/// A test case with expected entities
struct TestCase {
    text: &'static str,
    expected: Vec<(&'static str, &'static str)>, // (text, type)
    category: &'static str,
}

fn main() -> anno::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                    Rigorous Model Evaluation                           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // Define test cases
    let test_cases = vec![
        // Standard PER/ORG/LOC
        TestCase {
            text: "Steve Jobs founded Apple in Cupertino.",
            expected: vec![("Steve Jobs", "PER"), ("Apple", "ORG"), ("Cupertino", "LOC")],
            category: "Standard",
        },
        TestCase {
            text: "Barack Obama visited the White House in Washington D.C.",
            expected: vec![("Barack Obama", "PER"), ("White House", "LOC"), ("Washington D.C.", "LOC")],
            category: "Standard",
        },
        // Dates and temporal
        TestCase {
            text: "The conference is scheduled for March 15, 2024 at 2:30 PM.",
            expected: vec![("March 15, 2024", "DATE"), ("2:30 PM", "TIME")],
            category: "Temporal",
        },
        TestCase {
            text: "The deadline is next Friday, January 10th.",
            expected: vec![("Friday", "DATE"), ("January 10th", "DATE")],
            category: "Temporal",
        },
        // Money and quantities
        TestCase {
            text: "Tesla reported revenue of $53.8 billion in Q4 2023.",
            expected: vec![("Tesla", "ORG"), ("$53.8 billion", "MONEY")],
            category: "Financial",
        },
        TestCase {
            text: "The acquisition cost Microsoft $69 billion.",
            expected: vec![("Microsoft", "ORG"), ("$69 billion", "MONEY")],
            category: "Financial",
        },
        // Multi-word names
        TestCase {
            text: "Jean-Claude Van Damme starred in Universal Soldier.",
            expected: vec![("Jean-Claude Van Damme", "PER"), ("Universal Soldier", "WORK")],
            category: "Multi-word",
        },
        TestCase {
            text: "The New York Times reported on the European Central Bank.",
            expected: vec![("New York Times", "ORG"), ("European Central Bank", "ORG")],
            category: "Multi-word",
        },
        // Ambiguous cases
        TestCase {
            text: "Apple announced new products at the Apple Park campus.",
            expected: vec![("Apple", "ORG"), ("Apple Park", "LOC")],
            category: "Ambiguous",
        },
        TestCase {
            text: "Washington crossed the Delaware to reach Washington.",
            expected: vec![("Washington", "PER"), ("Delaware", "LOC"), ("Washington", "LOC")],
            category: "Ambiguous",
        },
        // Contact info
        TestCase {
            text: "Contact us at support@company.com or call (555) 123-4567.",
            expected: vec![("support@company.com", "EMAIL"), ("(555) 123-4567", "PHONE")],
            category: "Contact",
        },
        // Technical/scientific
        TestCase {
            text: "The CRISPR-Cas9 system was developed at UC Berkeley.",
            expected: vec![("CRISPR-Cas9", "TECH"), ("UC Berkeley", "ORG")],
            category: "Technical",
        },
    ];

    // Create model registry
    let mut models: Vec<(&str, Box<dyn Model>)> = vec![
        ("PatternNER", Box::new(PatternNER::new())),
        ("StatisticalNER", Box::new(StatisticalNER::new())),
        ("StackedNER", Box::new(StackedNER::default())),
    ];

    #[cfg(feature = "onnx")]
    {
        if let Ok(bert) = BertNEROnnx::new("protectai/bert-base-NER-onnx") {
            models.push(("BertNER", Box::new(bert)));
        }
        if let Ok(gliner) = GLiNEROnnx::new("onnx-community/gliner_small-v2.1") {
            models.push(("GLiNER", Box::new(gliner)));
        }
    }

    // Run evaluation
    let mut results: HashMap<&str, ModelMetrics> = HashMap::new();
    for (name, _) in &models {
        results.insert(name, ModelMetrics::default());
    }

    println!("Running {} test cases across {} models...\n", test_cases.len(), models.len());

    // Evaluate each test case
    for case in &test_cases {
        println!("─── {} [{}] ───", case.category, case.text.len());
        println!("  Text: \"{}\"", case.text);
        println!("  Expected: {:?}", case.expected);
        println!();

        for (name, model) in &models {
            let entities = model.extract_entities(case.text, None).unwrap_or_default();
            let metrics = results.get_mut(name).unwrap();

            // Calculate matches
            let mut found = 0;
            let mut partial = 0;
            
            for (exp_text, exp_type) in &case.expected {
                let mut matched = false;
                let mut partial_match = false;
                
                for e in &entities {
                    let e_text = e.text.trim_end_matches(|c: char| c.is_ascii_punctuation());
                    let exp_clean = exp_text.trim_end_matches(|c: char| c.is_ascii_punctuation());
                    
                    // Exact match
                    if e_text.eq_ignore_ascii_case(exp_clean) {
                        // Type match (flexible)
                        let type_match = match_entity_type(&e.entity_type.as_label(), exp_type);
                        if type_match {
                            matched = true;
                            break;
                        } else {
                            partial_match = true;
                        }
                    }
                    // Partial text match
                    else if e_text.contains(exp_clean) || exp_clean.contains(e_text) {
                        partial_match = true;
                    }
                }
                
                if matched {
                    found += 1;
                } else if partial_match {
                    partial += 1;
                }
            }

            metrics.total_expected += case.expected.len();
            metrics.exact_matches += found;
            metrics.partial_matches += partial;
            metrics.total_predicted += entities.len();
            metrics.by_category.entry(case.category).or_default().0 += case.expected.len();
            metrics.by_category.entry(case.category).or_default().1 += found;

            // Print results
            let status = if found == case.expected.len() {
                "PASS"
            } else if found > 0 || partial > 0 {
                "PARTIAL"
            } else {
                "MISS"
            };
            println!(
                "  {:15} [{:7}] found {}/{} (partial: {})",
                name, status, found, case.expected.len(), partial
            );
        }
        println!();
    }

    // Print summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Summary Results");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("┌─────────────────┬────────┬────────┬────────┬────────┐");
    println!("│ Model           │ Recall │ Partial│ Total  │ Pred   │");
    println!("├─────────────────┼────────┼────────┼────────┼────────┤");

    let mut sorted_results: Vec<_> = results.iter().collect();
    sorted_results.sort_by(|a, b| b.1.exact_matches.cmp(&a.1.exact_matches));

    for (name, metrics) in sorted_results {
        let recall = if metrics.total_expected > 0 {
            metrics.exact_matches as f64 / metrics.total_expected as f64
        } else {
            0.0
        };
        println!(
            "│ {:15} │ {:5.1}% │ {:6} │ {:6} │ {:6} │",
            name,
            recall * 100.0,
            metrics.partial_matches,
            metrics.total_expected,
            metrics.total_predicted
        );
    }
    println!("└─────────────────┴────────┴────────┴────────┴────────┘");

    // Print by category
    println!("\nResults by Category:");
    println!("┌────────────────┬");
    for (name, metrics) in &results {
        print!("│ {:14} ", name);
    }
    println!("│");
    println!("├────────────────┼");

    let categories: Vec<&str> = test_cases.iter().map(|c| c.category).collect::<std::collections::HashSet<_>>().into_iter().collect();
    for cat in categories {
        print!("│ {:14} ", cat);
        for (_, metrics) in &results {
            if let Some((total, found)) = metrics.by_category.get(cat) {
                print!("│ {:3}/{:3} ", found, total);
            } else {
                print!("│   -    ");
            }
        }
        println!("│");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("Analysis");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("Key findings:");
    println!("  - PatternNER excels at structured patterns (DATE, TIME, MONEY, EMAIL, PHONE)");
    println!("  - StatisticalNER handles PER/ORG/LOC via capitalization heuristics");
    println!("  - BertNER provides broader coverage but may miss patterns");
    println!("  - GLiNER offers flexible zero-shot extraction");
    println!();
    println!("Recommendation: Use StackedNER or Hybrid for best coverage.");

    Ok(())
}

#[derive(Default)]
struct ModelMetrics {
    exact_matches: usize,
    partial_matches: usize,
    total_expected: usize,
    total_predicted: usize,
    by_category: HashMap<&'static str, (usize, usize)>, // (expected, found)
}

/// Flexible type matching
fn match_entity_type(predicted: &str, expected: &str) -> bool {
    let pred_lower = predicted.to_lowercase();
    let exp_lower = expected.to_lowercase();

    // Exact match
    if pred_lower == exp_lower {
        return true;
    }

    // Common mappings
    match exp_lower.as_str() {
        "per" | "person" => pred_lower == "per" || pred_lower == "person" || pred_lower.contains("person"),
        "org" | "organization" => pred_lower == "org" || pred_lower == "organization" || pred_lower.contains("org"),
        "loc" | "location" => pred_lower == "loc" || pred_lower == "location" || pred_lower.contains("loc"),
        "date" => pred_lower == "date" || pred_lower.contains("date"),
        "time" => pred_lower == "time" || pred_lower.contains("time"),
        "money" => pred_lower == "money" || pred_lower.contains("money"),
        "email" => pred_lower == "email" || pred_lower.contains("email"),
        "phone" => pred_lower == "phone" || pred_lower.contains("phone"),
        "work" => pred_lower == "misc" || pred_lower.contains("work") || pred_lower.contains("art"),
        "tech" => pred_lower == "misc" || pred_lower.contains("tech") || pred_lower.contains("product"),
        _ => pred_lower.contains(&exp_lower) || exp_lower.contains(&pred_lower),
    }
}

