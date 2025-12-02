//! Performance benchmarks for optimization impact.
//!
//! This benchmark measures the performance improvements from:
//! - Caching `text.chars().count()` in hot paths
//! - Pre-allocating vectors to reduce reallocations
//! - Using unstable sorting when stability isn't needed
//! - Optimized span conversion with `SpanConverter`
//!
//! # Usage
//!
//! ```bash
//! # Benchmark optimization impact
//! cargo bench --bench optimization_performance
//!
//! # With ONNX backends
//! cargo bench --bench optimization_performance --features onnx
//! ```

use anno::{Entity, EntityType, HeuristicNER, Model, RegexNER, StackedNER};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Test texts of varying lengths to measure optimization impact
const SHORT_TEXT: &str = "Apple Inc. was founded by Steve Jobs.";
const MEDIUM_TEXT: &str = "Apple Inc. was founded by Steve Jobs in California in 1976. The company is headquartered in Cupertino and is known for products like the iPhone, iPad, and Mac computers.";
const LONG_TEXT: &str = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in California. The company is headquartered in Cupertino and is known for products like the iPhone, iPad, and Mac computers. Tim Cook became CEO in 2011 after Jobs stepped down. Apple is one of the world's largest technology companies and has a market capitalization of over $3 trillion. The company operates in over 100 countries and has retail stores worldwide.";

fn bench_text_length_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_length_impact");

    let texts = vec![
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ];

    // RegexNER
    let regex_ner = RegexNER::new();
    for (text_name, text) in &texts {
        group.bench_with_input(BenchmarkId::new("RegexNER", text_name), text, |b, t| {
            b.iter(|| {
                let _ = regex_ner.extract_entities(black_box(t), None);
            });
        });
    }

    // HeuristicNER
    let heuristic_ner = HeuristicNER::new();
    for (text_name, text) in &texts {
        group.bench_with_input(BenchmarkId::new("HeuristicNER", text_name), text, |b, t| {
            b.iter(|| {
                let _ = heuristic_ner.extract_entities(black_box(t), None);
            });
        });
    }

    // StackedNER
    let stacked_ner = StackedNER::default();
    for (text_name, text) in &texts {
        group.bench_with_input(BenchmarkId::new("StackedNER", text_name), text, |b, t| {
            b.iter(|| {
                let _ = stacked_ner.extract_entities(black_box(t), None);
            });
        });
    }

    group.finish();
}

fn bench_repeated_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("repeated_extraction");

    // Test repeated extraction on same text (should benefit from optimizations)
    let text = MEDIUM_TEXT;
    let iterations = 100;

    // RegexNER
    let regex_ner = RegexNER::new();
    group.bench_function(format!("RegexNER_repeated_{}", iterations), |b| {
        b.iter(|| {
            for _ in 0..iterations {
                let _ = regex_ner.extract_entities(black_box(text), None);
            }
        });
    });

    // HeuristicNER
    let heuristic_ner = HeuristicNER::new();
    group.bench_function(format!("HeuristicNER_repeated_{}", iterations), |b| {
        b.iter(|| {
            for _ in 0..iterations {
                let _ = heuristic_ner.extract_entities(black_box(text), None);
            }
        });
    });

    // StackedNER
    let stacked_ner = StackedNER::default();
    group.bench_function(format!("StackedNER_repeated_{}", iterations), |b| {
        b.iter(|| {
            for _ in 0..iterations {
                let _ = stacked_ner.extract_entities(black_box(text), None);
            }
        });
    });

    group.finish();
}

fn bench_entity_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_count_scaling");

    // Create texts with varying numbers of potential entities
    let texts = vec![
        ("few_entities", "Apple Inc. was founded by Steve Jobs."),
        ("many_entities", "Apple Inc. was founded by Steve Jobs in California in 1976. Microsoft was founded by Bill Gates in Washington in 1975. Google was founded by Larry Page and Sergey Brin in California in 1998. Amazon was founded by Jeff Bezos in Seattle in 1994."),
    ];

    // RegexNER
    let regex_ner = RegexNER::new();
    for (text_name, text) in &texts {
        group.bench_with_input(BenchmarkId::new("RegexNER", text_name), text, |b, t| {
            b.iter(|| {
                let _ = regex_ner.extract_entities(black_box(t), None);
            });
        });
    }

    // HeuristicNER
    let heuristic_ner = HeuristicNER::new();
    for (text_name, text) in &texts {
        group.bench_with_input(BenchmarkId::new("HeuristicNER", text_name), text, |b, t| {
            b.iter(|| {
                let _ = heuristic_ner.extract_entities(black_box(t), None);
            });
        });
    }

    // StackedNER
    let stacked_ner = StackedNER::default();
    for (text_name, text) in &texts {
        group.bench_with_input(BenchmarkId::new("StackedNER", text_name), text, |b, t| {
            b.iter(|| {
                let _ = stacked_ner.extract_entities(black_box(t), None);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "onnx")]
fn bench_onnx_optimizations(c: &mut Criterion) {
    use anno::backends::{BertNEROnnx, GLiNEROnnx};
    use anno::{DEFAULT_BERT_ONNX_MODEL, DEFAULT_GLINER_MODEL};

    let mut group = c.benchmark_group("onnx_optimizations");

    // Test ONNX backends with different text lengths
    let texts = vec![
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ];

    // BertNEROnnx
    if let Ok(bert) = BertNEROnnx::new(DEFAULT_BERT_ONNX_MODEL) {
        for (text_name, text) in &texts {
            group.bench_with_input(BenchmarkId::new("BertNEROnnx", text_name), text, |b, t| {
                b.iter(|| {
                    let _ = bert.extract_entities(black_box(t), None);
                });
            });
        }
    }

    // GLiNEROnnx
    if let Ok(gliner) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
        let entity_types = &["person", "organization", "location"];
        for (text_name, text) in &texts {
            group.bench_with_input(BenchmarkId::new("GLiNEROnnx", text_name), text, |b, t| {
                b.iter(|| {
                    let _ = gliner.extract(black_box(t), black_box(entity_types), 0.5);
                });
            });
        }
    }

    group.finish();
}

#[cfg(not(feature = "onnx"))]
fn bench_onnx_optimizations(_c: &mut Criterion) {
    // Stub when onnx feature is not enabled
}

criterion_group!(
    benches,
    bench_text_length_impact,
    bench_repeated_extraction,
    bench_entity_count_scaling,
    bench_onnx_optimizations
);
criterion_main!(benches);
