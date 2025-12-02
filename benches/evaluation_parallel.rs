//! Benchmarks for parallel vs sequential evaluation processing.
//!
//! Tests the performance improvement from parallel evaluation processing
//! using the `eval-parallel` feature.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(all(feature = "eval", feature = "eval-parallel"))]
fn create_mock_dataset(sentence_count: usize) -> Vec<String> {
    (0..sentence_count)
        .map(|i| {
            format!(
                "Apple CEO Tim Cook announced new products in Cupertino, California. Sentence {}.",
                i
            )
        })
        .collect()
}

#[cfg(all(feature = "eval", feature = "eval-parallel"))]
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    use anno::eval::backend_factory::BackendFactory;

    let mut group = c.benchmark_group("evaluation_parallel");
    group.sample_size(10);

    for &sentence_count in &[10, 50, 100] {
        let texts = create_mock_dataset(sentence_count);

        // Test with RegexNER (fast, no model loading)
        if let Ok(backend) = BackendFactory::create("pattern") {
            // Sequential benchmark
            group.bench_with_input(
                BenchmarkId::new("sequential", sentence_count),
                &texts,
                |b, texts| {
                    b.iter(|| {
                        let mut total = 0;
                        for text in texts {
                            let _ = backend.extract_entities(black_box(text), None);
                            total += text.len();
                        }
                        black_box(total)
                    })
                },
            );

            // Parallel benchmark
            #[cfg(feature = "eval-parallel")]
            {
                use rayon::prelude::*;
                group.bench_with_input(
                    BenchmarkId::new("parallel", sentence_count),
                    &texts,
                    |b, texts| {
                        b.iter(|| {
                            let total: usize = texts
                                .par_iter()
                                .map(|text| {
                                    let _ = backend.extract_entities(black_box(text), None);
                                    text.len()
                                })
                                .sum();
                            black_box(total)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(all(feature = "eval", feature = "eval-parallel"))]
fn bench_zero_shot_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_shot_caching");
    group.sample_size(10); // Increase sample size to meet criterion's minimum

    let texts = create_mock_dataset(50);

    // Test NuNER caching (if available)
    #[cfg(feature = "onnx")]
    {
        // Without caching (recreate backend each time)
        group.bench_function("nuner_no_cache", |b| {
            b.iter(|| {
                let mut total = 0;
                for text in &texts {
                    // Simulate recreating backend (slow)
                    if let Ok(nuner) =
                        anno::backends::NuNER::from_pretrained(anno::DEFAULT_NUNER_MODEL)
                    {
                        let _ = nuner.extract(black_box(text), &["person", "organization"], 0.5);
                        total += text.len();
                    }
                }
                black_box(total)
            })
        });

        // With caching (backend created once)
        group.bench_function("nuner_with_cache", |b| {
            if let Ok(nuner) = anno::backends::NuNER::from_pretrained(anno::DEFAULT_NUNER_MODEL) {
                b.iter(|| {
                    let mut total = 0;
                    for text in &texts {
                        let _ = nuner.extract(black_box(text), &["person", "organization"], 0.5);
                        total += text.len();
                    }
                    black_box(total)
                })
            }
        });
    }

    group.finish();
}

#[cfg(all(feature = "eval", feature = "eval-parallel"))]
criterion_group!(
    benches,
    bench_parallel_vs_sequential,
    bench_zero_shot_caching
);
#[cfg(all(feature = "eval", feature = "eval-parallel"))]
criterion_main!(benches);

#[cfg(not(all(feature = "eval", feature = "eval-parallel")))]
fn main() {
    eprintln!("This benchmark requires 'eval' and 'eval-parallel' features");
}
