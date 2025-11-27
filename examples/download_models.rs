//! Download all NER models for offline use.
//!
//! This example pre-downloads model weights from HuggingFace so they're cached
//! for offline use. Run once with network access, then use offline.
//!
//! # Usage
//!
//! ```bash
//! # Download all models (ONNX + Candle)
//! cargo run --example download_models --features "onnx,candle"
//!
//! # Download only ONNX models
//! cargo run --example download_models --features onnx
//!
//! # Download only Candle models
//! cargo run --example download_models --features candle
//! ```
//!
//! # Models Downloaded
//!
//! | Model | Size | Backend | Use Case |
//! |-------|------|---------|----------|
//! | `protectai/bert-base-NER-onnx` | ~400MB | ONNX | Traditional NER |
//! | `onnx-community/gliner_small-v2.1` | ~150MB | ONNX | Zero-shot NER |
//! | `dslim/bert-base-NER` | ~400MB | Candle | Rust-native NER |
//!
//! # Cache Location
//!
//! Models are cached in the HuggingFace cache directory:
//! - Linux: `~/.cache/huggingface/hub/`
//! - macOS: `~/.cache/huggingface/hub/`
//! - Windows: `C:\Users\<user>\.cache\huggingface\hub\`
//!
//! # Lazy Loading
//!
//! If you don't run this example, models are downloaded automatically on first use.
//! This example is useful for:
//! - Pre-warming cache for offline deployment
//! - Checking which models are available
//! - Ensuring download works before production

use std::time::Instant;

/// Models to download for ONNX backend
#[cfg(feature = "onnx")]
const ONNX_MODELS: &[(&str, &str)] = &[
    ("protectai/bert-base-NER-onnx", "Traditional BERT NER"),
    ("onnx-community/gliner_small-v2.1", "GLiNER zero-shot NER"),
    // Uncomment for more models:
    // ("onnx-community/gliner-multitask-large-v0.5", "GLiNER + relations"),
];

/// Models to download for Candle backend
#[cfg(feature = "candle")]
const CANDLE_MODELS: &[(&str, &str)] = &[
    ("dslim/bert-base-NER", "BERT NER (Candle)"),
    ("answerdotai/ModernBERT-base", "ModernBERT encoder"),
    // Uncomment for more:
    // ("microsoft/deberta-v3-base", "DeBERTa v3 encoder"),
];

fn main() {
    println!("=== Anno Model Downloader ===\n");
    
    let start = Instant::now();
    #[allow(unused_mut)] // Used conditionally by feature flags
    let mut downloaded = 0;
    #[allow(unused_mut)] // Used conditionally by feature flags
    let mut failed = 0;

    // ONNX models
    #[cfg(feature = "onnx")]
    {
        println!("--- ONNX Models ---\n");
        for (model_id, description) in ONNX_MODELS {
            match download_onnx_model(model_id, description) {
                Ok(_) => downloaded += 1,
                Err(e) => {
                    eprintln!("  FAILED: {}", e);
                    failed += 1;
                }
            }
        }
        println!();
    }

    #[cfg(not(feature = "onnx"))]
    {
        println!("--- ONNX Models (skipped - enable 'onnx' feature) ---\n");
    }

    // Candle models
    #[cfg(feature = "candle")]
    {
        println!("--- Candle Models ---\n");
        for (model_id, description) in CANDLE_MODELS {
            match download_candle_model(model_id, description) {
                Ok(_) => downloaded += 1,
                Err(e) => {
                    eprintln!("  FAILED: {}", e);
                    failed += 1;
                }
            }
        }
        println!();
    }

    #[cfg(not(feature = "candle"))]
    {
        println!("--- Candle Models (skipped - enable 'candle' feature) ---\n");
    }

    let elapsed = start.elapsed();
    println!("=== Summary ===");
    println!("Downloaded: {}", downloaded);
    println!("Failed: {}", failed);
    println!("Time: {:.1}s", elapsed.as_secs_f64());
    
    if downloaded > 0 {
        println!("\nModels cached in: ~/.cache/huggingface/hub/");
        println!("You can now use anno offline!");
    }
    
    #[cfg(not(any(feature = "onnx", feature = "candle")))]
    {
        println!("\nNo ML features enabled!");
        println!("Run with: cargo run --example download_models --features \"onnx,candle\"");
    }
}

#[cfg(feature = "onnx")]
fn download_onnx_model(model_id: &str, description: &str) -> Result<(), String> {
    use hf_hub::api::sync::Api;
    
    print!("  {} ({})... ", model_id, description);
    let start = Instant::now();
    
    let api = Api::new().map_err(|e| format!("API init: {}", e))?;
    let repo = api.model(model_id.to_string());
    
    // Download model files
    let files = ["model.onnx", "onnx/model.onnx", "tokenizer.json", "config.json"];
    let mut found_model = false;
    
    for file in files {
        match repo.get(file) {
            Ok(path) => {
                if file.contains("model.onnx") {
                    found_model = true;
                    let size = std::fs::metadata(&path)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    print!("{:.1}MB ", size as f64 / 1_000_000.0);
                }
            }
            Err(_) => continue, // Some files optional
        }
    }
    
    if !found_model {
        return Err("model.onnx not found".to_string());
    }
    
    println!("OK ({:.1}s)", start.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(feature = "candle")]
fn download_candle_model(model_id: &str, description: &str) -> Result<(), String> {
    use hf_hub::api::sync::Api;
    
    print!("  {} ({})... ", model_id, description);
    let start = Instant::now();
    
    let api = Api::new().map_err(|e| format!("API init: {}", e))?;
    let repo = api.model(model_id.to_string());
    
    // Download model files (safetensors preferred)
    let weight_files = ["model.safetensors", "pytorch_model.bin"];
    let mut found_weights = false;
    
    for file in weight_files {
        if let Ok(path) = repo.get(file) {
            found_weights = true;
            let size = std::fs::metadata(&path)
                .map(|m| m.len())
                .unwrap_or(0);
            print!("{:.1}MB ", size as f64 / 1_000_000.0);
            break;
        }
    }
    
    // Also get tokenizer and config
    let _ = repo.get("tokenizer.json");
    let _ = repo.get("config.json");
    
    if !found_weights {
        return Err("weights not found".to_string());
    }
    
    println!("OK ({:.1}s)", start.elapsed().as_secs_f64());
    Ok(())
}

