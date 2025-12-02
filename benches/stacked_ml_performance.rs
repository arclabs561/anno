//! Performance benchmarks for StackedNER with ML backends.
//!
//! These benchmarks measure:
//! - ML-first vs ML-fallback performance
//! - Multiple ML backend performance
//! - Comparison with default StackedNER
//! - Batch processing performance

use anno::backends::stacked::ConflictStrategy;
use anno::{BatchCapable, HeuristicNER, Model, RegexNER, StackedNER};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "onnx")]
fn create_gliner() -> Option<anno::GLiNEROnnx> {
    anno::GLiNEROnnx::new("onnx-community/gliner_small-v2.1").ok()
}

#[cfg(feature = "onnx")]
fn create_bert() -> Option<anno::BertNEROnnx> {
    anno::BertNEROnnx::new(anno::DEFAULT_BERT_ONNX_MODEL).ok()
}

fn bench_ml_first_vs_fallback(c: &mut Criterion) {
    #[cfg(feature = "onnx")]
    {
        if let (Some(gliner1), Some(gliner2)) = (create_gliner(), create_gliner()) {
            let ml_first = StackedNER::with_ml_first(Box::new(gliner1));
            let ml_fallback = StackedNER::with_ml_fallback(Box::new(gliner2));
            let default = StackedNER::default();

            let text = "Apple Inc. was founded by Steve Jobs in 1976. Microsoft was founded by Bill Gates in 1975.";

            let mut group = c.benchmark_group("ml_first_vs_fallback");

            group.bench_function("default", |b| {
                b.iter(|| {
                    black_box(default.extract_entities(black_box(text), None).unwrap());
                });
            });

            group.bench_function("ml_first", |b| {
                b.iter(|| {
                    black_box(ml_first.extract_entities(black_box(text), None).unwrap());
                });
            });

            group.bench_function("ml_fallback", |b| {
                b.iter(|| {
                    black_box(ml_fallback.extract_entities(black_box(text), None).unwrap());
                });
            });

            group.finish();
        }
    }
}

fn bench_multiple_ml_backends(c: &mut Criterion) {
    #[cfg(feature = "onnx")]
    {
        if let (Some(gliner1), Some(gliner2), Some(bert)) =
            (create_gliner(), create_gliner(), create_bert())
        {
            let single_ml = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner1))
                .layer(HeuristicNER::new())
                .build();

            let multiple_ml = StackedNER::builder()
                .layer(RegexNER::new())
                .layer_boxed(Box::new(gliner2))
                .layer_boxed(Box::new(bert))
                .layer(HeuristicNER::new())
                .build();

            let text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.";

            let mut group = c.benchmark_group("multiple_ml_backends");

            group.bench_function("single_ml", |b| {
                b.iter(|| {
                    black_box(single_ml.extract_entities(black_box(text), None).unwrap());
                });
            });

            group.bench_function("multiple_ml", |b| {
                b.iter(|| {
                    black_box(multiple_ml.extract_entities(black_box(text), None).unwrap());
                });
            });

            group.finish();
        }
    }
}

fn bench_ml_conflict_strategies(c: &mut Criterion) {
    #[cfg(feature = "onnx")]
    {
        if let Some(gliner) = create_gliner() {
            let strategies = [
                ("priority", ConflictStrategy::Priority),
                ("longest_span", ConflictStrategy::LongestSpan),
                ("highest_conf", ConflictStrategy::HighestConf),
                ("union", ConflictStrategy::Union),
            ];

            let text = "New York City is located in New York state.";

            let mut group = c.benchmark_group("ml_conflict_strategies");

            for (name, strategy) in strategies.iter() {
                if let Some(gliner) = create_gliner() {
                    let stacked = StackedNER::builder()
                        .layer(RegexNER::new())
                        .layer_boxed(Box::new(gliner))
                        .layer(HeuristicNER::new())
                        .strategy(*strategy)
                        .build();

                    group.bench_with_input(
                        BenchmarkId::from_parameter(name),
                        &stacked,
                        |b, ner| {
                            b.iter(|| {
                                black_box(ner.extract_entities(black_box(text), None).unwrap());
                            });
                        },
                    );
                }
            }

            group.finish();
        }
    }
}

fn bench_ml_batch_processing(c: &mut Criterion) {
    #[cfg(feature = "onnx")]
    {
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let texts: Vec<String> = (0..10)
                .map(|i| format!("Company {} was founded in {}.", i, 1970 + i))
                .collect();

            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            let mut group = c.benchmark_group("ml_batch_processing");

            group.bench_function("batch", |b| {
                b.iter(|| {
                    black_box(
                        ner.extract_entities_batch(black_box(&text_refs), None)
                            .unwrap(),
                    );
                });
            });

            group.bench_function("sequential", |b| {
                b.iter(|| {
                    for text in &text_refs {
                        black_box(ner.extract_entities(black_box(text), None).unwrap());
                    }
                });
            });

            group.finish();
        }
    }
}

fn bench_ml_text_length_scaling(c: &mut Criterion) {
    #[cfg(feature = "onnx")]
    {
        if let Some(gliner) = create_gliner() {
            let ner = StackedNER::with_ml_first(Box::new(gliner));

            let base_text = "Apple Inc. was founded by Steve Jobs. ";
            let lengths = [1, 10, 50, 100];

            let mut group = c.benchmark_group("ml_text_length_scaling");

            for &len in lengths.iter() {
                let text = base_text.repeat(len);
                group.bench_with_input(BenchmarkId::from_parameter(len), &text, |b, text| {
                    b.iter(|| {
                        black_box(ner.extract_entities(black_box(text), None).unwrap());
                    });
                });
            }

            group.finish();
        }
    }
}

criterion_group!(
    benches,
    bench_ml_first_vs_fallback,
    bench_multiple_ml_backends,
    bench_ml_conflict_strategies,
    bench_ml_batch_processing,
    bench_ml_text_length_scaling
);
criterion_main!(benches);
