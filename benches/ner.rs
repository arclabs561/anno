//! Performance benchmarks for all NER backends.
//!
//! This benchmark tests all available backends and automatically downloads
//! models from HuggingFace if needed. Models are cached for subsequent runs.
//!
//! # Usage
//!
//! ```bash
//! # Benchmark all backends (downloads models automatically)
//! cargo bench --features "eval,onnx,candle" --bench ner
//!
//! # Pre-download models (optional, for offline use)
//! cargo run --example download_models --features "onnx,candle"
//!
//! # Pre-download datasets (optional, for offline use)
//! cargo run --example download_datasets --features eval-advanced
//!
//! # Backup all caches (models + datasets)
//! ./examples/backup_cache.sh [backup-dir]
//! ```

use anno::{
    backends::{HeuristicNER, StackedNER},
    Model, PatternNER, DEFAULT_BERT_ONNX_MODEL, DEFAULT_CANDLE_MODEL, DEFAULT_GLINER_CANDLE_MODEL,
    DEFAULT_GLINER_MODEL, DEFAULT_NUNER_MODEL,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "candle")]
use anno::backends::gliner_candle::GLiNERCandle;

const BENCH_TEXT: &str =
    "Meeting scheduled for January 15, 2025 at $500 per hour, estimated 15% completion. Apple Inc. announced new products in Cupertino, California.";

// W2NER model - try alternative if main one fails
#[cfg(feature = "onnx")]
const W2NER_MODEL: &str = "ljynlp/w2ner-bert-base";
#[cfg(feature = "onnx")]
const W2NER_ALTERNATIVE: &str = "harry-kpv-hf/ai-rec-ner-onnx-model"; // Alternative NER model if W2NER unavailable

fn bench_all_backends(c: &mut Criterion) {
    // Always available backends
    bench_pattern_ner(c);
    bench_heuristic_ner(c);
    bench_stacked_ner(c);

    // ONNX backends (if feature enabled)
    #[cfg(feature = "onnx")]
    {
        bench_bert_onnx(c);
        bench_gliner_onnx(c);
        bench_nuner(c);
        bench_w2ner(c);
    }

    // Candle backends (if feature enabled)
    #[cfg(feature = "candle")]
    {
        bench_candle_ner(c);
        bench_gliner_candle(c);
    }
}

fn bench_pattern_ner(c: &mut Criterion) {
    let ner = PatternNER::new();
    c.bench_function("PatternNER", |b| {
        b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
    });
}

fn bench_heuristic_ner(c: &mut Criterion) {
    let ner = HeuristicNER::new();
    c.bench_function("HeuristicNER", |b| {
        b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
    });
}

fn bench_stacked_ner(c: &mut Criterion) {
    let ner = StackedNER::default();
    c.bench_function("StackedNER", |b| {
        b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
    });
}

#[cfg(feature = "onnx")]
fn bench_bert_onnx(c: &mut Criterion) {
    eprintln!(
        "[Bench] Loading BertNEROnnx from {}...",
        DEFAULT_BERT_ONNX_MODEL
    );
    match anno::backends::BertNEROnnx::new(DEFAULT_BERT_ONNX_MODEL) {
        Ok(ner) => {
            eprintln!("[Bench] BertNEROnnx loaded successfully");
            c.bench_function("BertNEROnnx", |b| {
                b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
            });
        }
        Err(e) => {
            eprintln!("[Bench] BertNEROnnx failed to load: {} (skipping)", e);
        }
    }
}

#[cfg(feature = "onnx")]
fn bench_gliner_onnx(c: &mut Criterion) {
    eprintln!(
        "[Bench] Loading GLiNEROnnx from {}...",
        DEFAULT_GLINER_MODEL
    );
    match anno::backends::GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
        Ok(ner) => {
            eprintln!("[Bench] GLiNEROnnx loaded successfully");
            c.bench_function("GLiNEROnnx", |b| {
                b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
            });
        }
        Err(e) => {
            eprintln!("[Bench] GLiNEROnnx failed to load: {} (skipping)", e);
        }
    }
}

#[cfg(feature = "onnx")]
fn bench_nuner(c: &mut Criterion) {
    eprintln!("[Bench] Loading NuNER from {}...", DEFAULT_NUNER_MODEL);
    match anno::backends::NuNER::from_pretrained(DEFAULT_NUNER_MODEL) {
        Ok(ner) => {
            eprintln!("[Bench] NuNER loaded successfully");
            c.bench_function("NuNER", |b| {
                b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
            });
        }
        Err(e) => {
            eprintln!("[Bench] NuNER failed to load: {} (skipping)", e);
        }
    }
}

#[cfg(feature = "onnx")]
fn bench_w2ner(c: &mut Criterion) {
    eprintln!("[Bench] Loading W2NER from {}...", W2NER_MODEL);
    match anno::backends::W2NER::from_pretrained(W2NER_MODEL) {
        Ok(ner) => {
            eprintln!("[Bench] W2NER loaded successfully");
            c.bench_function("W2NER", |b| {
                b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
            });
        }
        Err(e) => {
            eprintln!("[Bench] W2NER failed to load: {} (skipping)", e);
        }
    }
}

#[cfg(feature = "candle")]
fn bench_candle_ner(c: &mut Criterion) {
    // Try default model first
    eprintln!("[Bench] Loading CandleNER from {}...", DEFAULT_CANDLE_MODEL);
    let result = anno::backends::CandleNER::from_pretrained(DEFAULT_CANDLE_MODEL);

    // If default fails due to missing tokenizer.json, try alternative
    let ner = match result {
        Ok(ner) => {
            eprintln!(
                "[Bench] CandleNER loaded successfully from {}",
                DEFAULT_CANDLE_MODEL
            );
            ner
        }
        Err(e) => {
            eprintln!(
                "[Bench] CandleNER failed with {}: {} (trying alternative...)",
                DEFAULT_CANDLE_MODEL, e
            );
            // The error should now be more informative, but if it still fails,
            // it means the model format is incompatible
            eprintln!("[Bench] CandleNER failed to load: {} (skipping)", e);
            return;
        }
    };

    c.bench_function("CandleNER", |b| {
        b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
    });
}

#[cfg(feature = "candle")]
fn bench_gliner_candle(c: &mut Criterion) {
    use anno::backends::gliner_candle::GLiNERCandle;
    use anno::DEFAULT_GLINER_CANDLE_MODEL;

    // Try default model first (may have safetensors)
    eprintln!(
        "[Bench] Loading GLiNERCandle from {}...",
        DEFAULT_GLINER_CANDLE_MODEL
    );
    let ner = match GLiNERCandle::from_pretrained(DEFAULT_GLINER_CANDLE_MODEL) {
        Ok(ner) => {
            eprintln!(
                "[Bench] GLiNERCandle loaded successfully from {}",
                DEFAULT_GLINER_CANDLE_MODEL
            );
            ner
        }
        Err(e) => {
            eprintln!(
                "[Bench] GLiNERCandle failed with {}: {} (trying alternative...)",
                DEFAULT_GLINER_CANDLE_MODEL, e
            );
            // Try alternative model
            const ALTERNATIVE: &str = "knowledgator/gliner-x-small";
            match GLiNERCandle::from_pretrained(ALTERNATIVE) {
                Ok(ner) => {
                    eprintln!(
                        "[Bench] GLiNERCandle loaded successfully from {}",
                        ALTERNATIVE
                    );
                    ner
                }
                Err(e2) => {
                    eprintln!(
                        "[Bench] GLiNERCandle failed to load from both models: {} / {} (skipping)",
                        e, e2
                    );
                    return;
                }
            }
        }
    };

    c.bench_function("GLiNERCandle", |b| {
        b.iter(|| ner.extract_entities(black_box(BENCH_TEXT), None))
    });
}

criterion_group!(benches, bench_all_backends);
criterion_main!(benches);
