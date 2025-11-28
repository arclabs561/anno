//! Hybrid model evaluation: Shows how combining models improves coverage.
//!
//! Key insight: ML models miss structured patterns, Pattern models miss named entities.
//! Solution: Combine them intelligently.
//!
//! Run: cargo run --features "onnx" --example hybrid

use anno::{Entity, Model, HybridNER, PatternNER};
use std::collections::HashSet;

#[cfg(feature = "onnx")]
use anno::{BertNEROnnx, GLiNEROnnx};

fn main() -> anno::Result<()> {
    println!("Hybrid Model Evaluation");
    println!("=======================\n");

    let test_cases = [
        // ML models excel at these
        ("Steve Jobs founded Apple in Cupertino.", "Named Entities"),
        ("Barack Obama met with Angela Merkel in Berlin.", "Multi-word Names"),
        
        // Pattern models excel at these
        ("Contact us at john@company.com or call (555) 123-4567.", "Contact Info"),
        ("The deadline is March 15, 2024 at 3:00 PM.", "Temporal"),
        ("Total investment: $50 million USD.", "Financial"),
        
        // Both need to work together
        ("Apple CEO Tim Cook announced $3.2 billion in revenue on January 15.", "Mixed"),
        ("Email support@tesla.com about the March 2024 earnings.", "Mixed"),
    ];

    // Initialize models
    let pattern = PatternNER::new();
    let hybrid_basic = HybridNER::pattern_only();

    #[cfg(feature = "onnx")]
    let bert = BertNEROnnx::new("protectai/bert-base-NER-onnx").ok();
    #[cfg(feature = "onnx")]
    let gliner = GLiNEROnnx::new("onnx-community/gliner_small-v2.1").ok();

    println!("Entity Extraction Comparison");
    println!("----------------------------\n");

    for (text, category) in &test_cases {
        println!("[{}]", category);
        println!("  Text: \"{}\"\n", text);

        // Pattern only
        let pattern_entities = pattern.extract_entities(text, None).unwrap_or_default();
        println!("  PatternNER:     {}", format_entities(&pattern_entities));

        // BERT only
        #[cfg(feature = "onnx")]
        if let Some(ref bert_model) = bert {
            let bert_entities = bert_model.extract_entities(text, None).unwrap_or_default();
            println!("  BertNER:        {}", format_entities(&bert_entities));
        }

        // GLiNER only
        #[cfg(feature = "onnx")]
        if let Some(ref gliner_model) = gliner {
            let gliner_entities = gliner_model.extract_entities(text, None).unwrap_or_default();
            println!("  GLiNER:         {}", format_entities(&gliner_entities));
        }

        // Hybrid (Pattern + ML)
        #[cfg(feature = "onnx")]
        if let (Some(ref gliner_model), Some(ref _bert_model)) = (&gliner, &bert) {
            // Manual hybrid: combine GLiNER + Pattern
            let ml_entities = gliner_model.extract_entities(text, None).unwrap_or_default();
            let combined = merge_entities(&pattern_entities, &ml_entities);
            println!("  GLiNER+Pattern: {} [COMBINED]", format_entities(&combined));
        }

        // Basic hybrid from library
        let basic_hybrid = hybrid_basic.extract_entities(text, None).unwrap_or_default();
        println!("  HybridNER:      {}", format_entities(&basic_hybrid));

        println!();
    }

    // Summary comparison
    println!("Coverage Analysis");
    println!("-----------------\n");

    let mut total_pattern = 0;
    let mut total_ml = 0;
    let mut total_combined = 0;

    for (text, _) in &test_cases {
        let pattern_entities = pattern.extract_entities(text, None).unwrap_or_default();
        total_pattern += pattern_entities.len();

        #[cfg(feature = "onnx")]
        if let Some(ref gliner_model) = gliner {
            let ml_entities = gliner_model.extract_entities(text, None).unwrap_or_default();
            total_ml += ml_entities.len();
            
            let combined = merge_entities(&pattern_entities, &ml_entities);
            total_combined += combined.len();
        }
    }

    println!("Entity counts across {} test cases:", test_cases.len());
    println!("  PatternNER:     {:2} entities", total_pattern);
    #[cfg(feature = "onnx")]
    println!("  GLiNER:         {:2} entities", total_ml);
    #[cfg(feature = "onnx")]
    println!("  Combined:       {:2} entities", total_combined);

    println!("\nComplementary Strengths:");
    println!();
    println!("  PatternNER excels at:");
    println!("    - Dates (March 15, 2024)");
    println!("    - Times (3:00 PM)");
    println!("    - Money ($50 million)");
    println!("    - Emails (john@company.com)");
    println!("    - Phone numbers ((555) 123-4567)");
    println!();
    println!("  ML Models (BERT/GLiNER) excel at:");
    println!("    - Person names (Tim Cook, Angela Merkel)");
    println!("    - Organization names (Apple, Tesla)");
    println!("    - Location names (Berlin, Cupertino)");
    println!("    - Multi-word entities (New York Times)");
    println!();
    println!("  Combined approach provides:");
    println!("    - Best of both worlds");
    println!("    - Higher recall without sacrificing precision");
    println!("    - Robust handling of diverse entity types");

    println!("\nRecommended Usage");
    println!("-----------------\n");

    println!("For production use, consider:");
    println!();
    println!("  // Option 1: Use HybridNER (automatic ML + Pattern)");
    println!("  let model = HybridNER::default();");
    println!();
    println!("  // Option 2: Manual combination with deduplication");
    println!("  let pattern = PatternNER::new();");
    println!("  let ml = GLiNEROnnx::new(\"...\")?;");
    println!("  let entities = merge_entities(pattern.extract(text)?, ml.extract(text)?);");
    println!();
    println!("  // Option 3: Use StackedNER for Pattern + Statistical");
    println!("  let model = StackedNER::default();");

    Ok(())
}

fn format_entities(entities: &[Entity]) -> String {
    if entities.is_empty() {
        return "(none)".to_string();
    }
    entities
        .iter()
        .map(|e| format!("{}[{}]", e.text.trim_end_matches(|c: char| c.is_ascii_punctuation()), short_type(&e.entity_type.as_label())))
        .collect::<Vec<_>>()
        .join(", ")
}

fn short_type(t: &str) -> &str {
    match t {
        "DATE" | "date" => "D",
        "TIME" | "time" => "T",
        "MONEY" | "money" => "$",
        "EMAIL" | "email" => "@",
        "PHONE" | "phone" => "#",
        "PER" | "person" => "P",
        "ORG" | "organization" => "O",
        "LOC" | "location" => "L",
        "MISC" | "misc" => "M",
        _ => "?",
    }
}

/// Merge entities from two sources, deduplicating overlapping spans.
fn merge_entities(a: &[Entity], b: &[Entity]) -> Vec<Entity> {
    let mut result: Vec<Entity> = a.to_vec();
    let used_spans: HashSet<(usize, usize)> = a.iter().map(|e| (e.start, e.end)).collect();

    for entity in b {
        // Check if this span overlaps with existing entities
        let overlaps = used_spans.iter().any(|(s, e)| {
            // Check for any overlap
            !(entity.end <= *s || entity.start >= *e)
        });

        if !overlaps {
            result.push(entity.clone());
        }
    }

    // Sort by start position
    result.sort_by_key(|e| e.start);
    result
}

