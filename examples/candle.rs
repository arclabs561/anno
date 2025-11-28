//! Demonstration of GLiNER using the pure Rust Candle backend.
//!
//! This example shows:
//! - Pluggable encoder architecture (BERT/ModernBERT)
//! - Late interaction span-label matching
//! - Zero-shot entity extraction
//!
//! Run with:
//! ```bash
//! cargo run --example candle --features candle
//! ```

use anno::{Model, PatternNER, StatisticalNER};

#[cfg(feature = "candle")]
use anno::backends::gliner_candle::GLiNERCandle;

fn main() -> anno::Result<()> {
    println!("GLiNER Candle Architecture Demo");
    println!("===============================\n");

    // Test texts
    let texts = [
        "Elon Musk founded SpaceX in 2002 and Tesla in 2003.",
        "The Mona Lisa by Leonardo da Vinci is displayed at the Louvre in Paris.",
        "Tim Cook announced the iPhone 15 at Apple Park in September 2023.",
        "Amazon's headquarters in Seattle employs over 75,000 people.",
    ];

    // Pattern-based baseline
    println!("PatternNER (zero-dependency baseline)");
    println!("-------------------------------------\n");
    
    let pattern_ner = PatternNER::new();
    for text in &texts {
        let entities = pattern_ner.extract_entities(text, None)?;
        println!("Text: {}", text);
        println!("  Entities: {:?}", 
            entities.iter().map(|e| (&e.text, &e.entity_type)).collect::<Vec<_>>()
        );
        println!();
    }

    // Statistical baseline
    println!("StatisticalNER (heuristic-based baseline)");
    println!("-----------------------------------------\n");
    
    let statistical_ner = StatisticalNER::new();
    for text in &texts {
        let entities = statistical_ner.extract_entities(text, None)?;
        println!("Text: {}", text);
        println!("  Entities: {:?}",
            entities.iter().map(|e| (&e.text, &e.entity_type)).collect::<Vec<_>>()
        );
        println!();
    }

    // GLiNER Architecture Explanation
    println!("GLiNER Architecture (Bi-Encoder for Zero-Shot NER)");
    println!("--------------------------------------------------\n");
    
    println!("GLiNER treats NER as a bi-encoder matching problem:\n");
    println!(r#"
    ┌──────────────────────────────────────────────────────────────────┐
    │                         GLiNER Pipeline                          │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Input Text: "Elon Musk founded SpaceX"                          │
    │       │                                                          │
    │       ▼                                                          │
    │  ┌─────────────┐                                                 │
    │  │  Tokenizer  │ → [CLS] Elon Musk founded SpaceX [SEP]          │
    │  └──────┬──────┘                                                 │
    │         │                                                        │
    │         ▼                                                        │
    │  ┌─────────────────────────────────────────┐                     │
    │  │      Transformer Encoder                │                     │
    │  │  (BERT / DeBERTa / ModernBERT)          │                     │
    │  │                                         │                     │
    │  │  Token Embeddings [seq_len × hidden]    │                     │
    │  └──────┬──────────────────────────────────┘                     │
    │         │                                                        │
    │         ├────────────────┬─────────────────┐                     │
    │         │                │                 │                     │
    │         ▼                ▼                 ▼                     │
    │  ┌────────────┐   ┌────────────┐   ┌────────────┐                │
    │  │ Span Rep   │   │ Label Enc  │   │   Width    │                │
    │  │ Layer      │   │  Layer     │   │   Emb      │                │
    │  └──────┬─────┘   └──────┬─────┘   └──────┬─────┘                │
    │         │                │                │                      │
    │         ▼                ▼                ▼                      │
    │  span_emb = concat(start_emb, end_emb, width_emb)                │
    │  label_emb = project(encoder(label_text))                        │
    │                                                                  │
    │         └────────────────┴─────────────────┘                     │
    │                          │                                       │
    │                          ▼                                       │
    │  ┌──────────────────────────────────────────────┐                │
    │  │         Span-Label Matcher                   │                │
    │  │  score = sigmoid(span_emb · label_emb / τ)   │                │
    │  │                                              │                │
    │  │  Late Interaction: Compare all spans with    │                │
    │  │  all labels in embedding space               │                │
    │  └──────────────────────────────────────────────┘                │
    │                          │                                       │
    │                          ▼                                       │
    │  Output: [(span="Elon Musk", label="person", score=0.95), ...]   │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    "#);

    // Encoder Variants
    println!("\nPluggable Encoder Architectures");
    println!("-------------------------------\n");

    println!("Encoder           Context  Position    Accuracy  Speed");
    println!("----------------  -------  ----------  --------  -----------------");
    println!("BERT-base            512   Absolute    Good      Fast");
    println!("DeBERTaV3-base       512   Relative    Better    Medium");
    println!("ModernBERT-base     8192   RoPE        Best      Fast (unpadded)");
    println!("ModernBERT-large    8192   RoPE        SOTA      Medium (unpadded)\n");

    println!("Key Innovations in ModernBERT:");
    println!("  • RoPE (Rotary Position Embeddings): Better extrapolation");
    println!("  • GeGLU Activation: Improved FFN performance");
    println!("  • Unpadding: Process as 1D stream, no padding overhead");
    println!("  • Flash Attention: Memory-efficient attention\n");

    // GLiNER Model Variants
    println!("GLiNER Model Zoo");
    println!("----------------\n");

    let models = [
        ("onnx-community/gliner_small-v2.1", "DeBERTaV3", "Small", "109M", "Fast, CPU-friendly"),
        ("onnx-community/gliner_medium-v2.1", "DeBERTaV3", "Medium", "183M", "Balanced"),
        ("onnx-community/gliner_large-v2.1", "DeBERTaV3", "Large", "304M", "Accurate"),
        ("knowledgator/modern-gliner-bi-base-v1.0", "ModernBERT", "Base", "149M", "8K context, fast"),
        ("knowledgator/modern-gliner-bi-large-v1.0", "ModernBERT", "Large", "395M", "SOTA, 8K ctx"),
    ];

    println!("Model ID                                     Encoder     Size    Params  Notes");
    println!("-------------------------------------------  ----------  ------  ------  ------------------");
    for (model_id, encoder, size, params, notes) in models {
        println!("{:43}  {:10}  {:6}  {:6}  {}",
            model_id, encoder, size, params, notes);
    }
    println!();

    // Candle Backend Demo
    #[cfg(feature = "candle")]
    {
        println!("Candle Backend Status");
        println!("---------------------\n");

        // Check device
        match anno::backends::gliner_candle::best_device() {
            Ok(device) => {
                println!("✓ Device: {:?}", device);
                println!("  - Metal (Apple Silicon): {}", 
                    if cfg!(all(target_os = "macos", feature = "metal")) { "Enabled" } else { "Disabled" });
                println!("  - CUDA (NVIDIA GPU): {}",
                    if cfg!(feature = "cuda") { "Enabled" } else { "Disabled" });
            }
            Err(e) => println!("✗ Device selection failed: {}", e),
        }

        println!("\nNote: Full GLiNER inference requires model weights.");
        println!("The architecture is implemented; weight loading is WIP.\n");
        
        // Try to create model (will fail gracefully without network)
        println!("Attempting to initialize GLiNER-Candle (requires network)...");
        match GLiNERCandle::new("answerdotai/ModernBERT-base") {
            Ok(model) => {
                println!("✓ Model initialized successfully");
                println!("  - Model: {}", model.model_name());
                println!("  - Device: {:?}", model.device());
            }
            Err(e) => {
                println!("✗ Model initialization failed: {}", e);
                println!("  (This is expected if network access is disabled or model not cached)");
            }
        }
    }

    #[cfg(not(feature = "candle"))]
    {
        println!("Candle Backend Not Available");
        println!("----------------------------\n");
        println!("The 'candle' feature is not enabled. To use GLiNER-Candle:");
        println!("  cargo run --example candle_gliner_demo --features candle");
    }

    // Late Interaction Explanation
    println!("\nLate Interaction: The Key to Zero-Shot NER");
    println!("-------------------------------------------\n");

    println!("Traditional NER: Train separate classifiers per entity type");
    println!("  → Adding new types requires retraining\n");

    println!("GLiNER: Encode types and spans into same space, then match");
    println!("  → New types = new labels in embedding space (no retraining!)\n");

    println!("Late Interaction Pattern:");
    println!("  1. Encode text spans → [num_spans × hidden_dim]");
    println!("  2. Encode label text → [num_labels × hidden_dim]");
    println!("  3. Compute similarity matrix → [num_spans × num_labels]");
    println!("  4. Apply threshold → Extract high-confidence matches\n");

    println!("This is the same insight behind ColBERT retrieval!");
    println!("  ColBERT: query_tokens × doc_tokens → MaxSim");
    println!("  GLiNER:  text_spans × label_types → Sigmoid(dot/τ)\n");

    // Summary
    println!("Summary: Anno Backend Hierarchy");
    println!("-------------------------------\n");

    println!("Zero Dependencies:");
    println!("  - PatternNER     Regex patterns (dates, emails, etc.)");
    println!("  - StatisticalNER Capitalization heuristics");
    println!();
    println!("ONNX Runtime (cross-platform):");
    println!("  - BertNEROnnx    Standard BERT token classification");
    println!("  - GLiNEROnnx     GLiNER zero-shot");
    println!();
    println!("Pure Rust Candle (native GPU):");
    println!("  - GLiNERCandle   Metal/CUDA accelerated GLiNER");
    println!();
    println!("Composites:");
    println!("  - StackedNER     Combine backends with conflict resolution");
    println!("  - HybridNER      ML + Pattern fallback");

    Ok(())
}

