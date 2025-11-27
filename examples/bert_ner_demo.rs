//! BERT NER ONNX Demo - Standard Named Entity Recognition
//!
//! This example demonstrates high-quality NER using BERT with ONNX Runtime.
//!
//! Run with:
//!   cargo run --example bert_ner_demo --features onnx
//!
//! # Entity Types
//!
//! BERT NER models are pre-trained on standard entity types:
//! - PER (Person): Names of people
//! - ORG (Organization): Companies, agencies, institutions
//! - LOC (Location): Places, cities, countries
//! - MISC (Miscellaneous): Events, nationalities, etc.
//!
//! For custom entity types (e.g., movie genres, legal terms), use GLiNER
//! which supports zero-shot entity detection with any label.

#[cfg(feature = "onnx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use anno::BertNEROnnx;
    use std::time::Instant;

    println!("=== BERT NER ONNX Demo ===\n");

    // Load model
    println!("Loading BERT NER model...");
    let start = Instant::now();
    let model = BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL)?;
    println!("Model loaded in {:.1}s: {}\n", start.elapsed().as_secs_f64(), model.model_name());

    // Test texts covering different entity types
    let test_texts = [
        ("Tech News", "Elon Musk announced that Tesla will open a new factory in Berlin, Germany."),
        ("Politics", "President Biden met with Chancellor Scholz at the White House on Monday."),
        ("Sports", "Lionel Messi led Argentina to victory in the World Cup final in Qatar."),
        ("Business", "Microsoft acquired Activision Blizzard for $69 billion, pending FTC approval."),
        ("Science", "NASA's James Webb Space Telescope captured images of the Andromeda galaxy."),
        ("History", "Leonardo da Vinci painted the Mona Lisa in Florence during the Renaissance."),
        ("Legal", "The Supreme Court ruled in Brown v. Board of Education that segregation is unconstitutional."),
        ("Entertainment", "Christopher Nolan directed Oppenheimer starring Cillian Murphy and Robert Downey Jr."),
    ];

    for (domain, text) in &test_texts {
        println!("--- {} ---", domain);
        println!("Text: {}\n", text);

        let start = Instant::now();
        match model.extract_entities(text, None) {
            Ok(entities) => {
                let elapsed = start.elapsed();
                if entities.is_empty() {
                    println!("  No entities found\n");
                } else {
                    for e in &entities {
                        println!(
                            "  {:?}: \"{}\" (chars {}-{}, {:.1}% conf)",
                            e.entity_type, e.text, e.start, e.end, e.confidence * 100.0
                        );
                    }
                    println!("  [{} entities in {:.1}ms]\n", entities.len(), elapsed.as_secs_f64() * 1000.0);
                }
            }
            Err(e) => {
                println!("  Error: {}\n", e);
            }
        }
    }

    // Performance summary
    println!("=== Performance Notes ===\n");
    println!("BERT NER achieves ~50-60% F1 on standard benchmarks (CoNLL-2003, WikiGold).");
    println!("For higher accuracy, consider:");
    println!("  - Using larger models (bert-large-NER)");
    println!("  - Domain-specific fine-tuning");
    println!("  - GLiNER for zero-shot custom entity types");
    println!("\nFor fastest inference:");
    println!("  - Enable GPU via CUDA/CoreML execution providers");
    println!("  - Use quantized models (INT8)");
    println!("  - Batch multiple texts together");

    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    println!("This example requires the 'onnx' feature.");
    println!("Run with: cargo run --example bert_ner_demo --features onnx");
}

