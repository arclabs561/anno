//! Benchmarks for parallel vs sequential evaluation processing.
//!
//! Tests the performance improvement from parallel evaluation processing
//! using the `eval-parallel` feature.

use anno::eval::task_evaluator::TaskEvaluator;
use anno::eval::task_mapping::{DatasetId, Task};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

#[cfg(all(feature = "eval", feature = "eval-parallel"))]
fn create_mock_dataset(sentence_count: usize) -> anno::eval::loader::LoadedDataset {
    use anno::eval::loader::{LoadedDataset, Sentence};
    use anno::eval::types::GoldEntity;

    let sentences: Vec<Sentence> = (0..sentence_count)
        .map(|i| {
            let text = format!(
                "Apple CEO Tim Cook announced new products in Cupertino, California. Sentence {}.",
                i
            );
            Sentence::new(
                text.clone(),
                vec![
                    GoldEntity {
                        text: "Apple".to_string(),
                        entity_type: "ORG".to_string(),
                        start: text.find("Apple").unwrap(),
                        end: text.find("Apple").unwrap() + "Apple".len(),
                    },
                    GoldEntity {
                        text: "Tim Cook".to_string(),
                        entity_type: "PER".to_string(),
                        start: text.find("Tim Cook").unwrap(),
                        end: text.find("Tim Cook").unwrap() + "Tim Cook".len(),
                    },
                ],
            )
        })
        .collect();

    LoadedDataset {
        sentences,
        metadata: std::collections::HashMap::new(),
    }
}

#[cfg(all(feature = "eval", feature = "eval-parallel"))]
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    use anno::eval::backend_factory::BackendFactory;

    let mut group = c.benchmark_group("evaluation_parallel");
    group.sample_size(10);

    for &sentence_count in &[10, 50, 100] {
        let dataset = create_mock_dataset(sentence_count);

        // Test with RegexNER (fast, no model loading)
        if let Ok(backend) = BackendFactory::create("pattern") {
            let evaluator = TaskEvaluator::new().unwrap();

            // Sequential benchmark (simulated - actual sequential is feature-gated)
            group.bench_with_input(
                BenchmarkId::new("sequential", sentence_count),
                &(evaluator.clone(), dataset.clone(), backend.as_ref()),
                |b, (evaluator, dataset, backend)| {
                    b.iter(|| {
                        // Simulate sequential processing
                        let mut total = 0;
                        for sentence in &dataset.sentences {
                            let _ = backend.extract_entities(&sentence.text(), None);
                            total += sentence.text().len();
                        }
                        black_box(total)
                    })
                },
            );

            // Parallel benchmark
            #[cfg(feature = "eval-parallel")]
            {
                use rayon::prelude::*;
                let evaluator_par = evaluator.clone();
                group.bench_with_input(
                    BenchmarkId::new("parallel", sentence_count),
                    &(evaluator_par, dataset.clone(), backend.as_ref()),
                    |b, (evaluator, dataset, backend)| {
                        b.iter(|| {
                            let total: usize = dataset
                                .sentences
                                .par_iter()
                                .map(|sentence| {
                                    let _ = backend.extract_entities(&sentence.text(), None);
                                    sentence.text().len()
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
    use anno::eval::backend_factory::BackendFactory;

    let mut group = c.benchmark_group("zero_shot_caching");
    group.sample_size(5);

    let dataset = create_mock_dataset(50);

    // Test NuNER caching (if available)
    #[cfg(feature = "onnx")]
    {
        if let Ok(backend) = BackendFactory::create("nuner") {
            let _evaluator = TaskEvaluator::new().unwrap();

            // Without caching (recreate backend each time)
            group.bench_function("nuner_no_cache", |b| {
                b.iter(|| {
                    let mut total = 0;
                    for sentence in &dataset.sentences {
                        // Simulate recreating backend (slow)
                        if let Ok(nuner) =
                            anno::backends::NuNER::from_pretrained(anno::DEFAULT_NUNER_MODEL)
                        {
                            let _ =
                                nuner.extract(&sentence.text(), &["person", "organization"], 0.5);
                            total += sentence.text().len();
                        }
                    }
                    black_box(total)
                })
            });

            // With caching (backend created once)
            group.bench_function("nuner_with_cache", |b| {
                if let Ok(nuner) = anno::backends::NuNER::from_pretrained(anno::DEFAULT_NUNER_MODEL)
                {
                    b.iter(|| {
                        let mut total = 0;
                        for sentence in &dataset.sentences {
                            let _ =
                                nuner.extract(&sentence.text(), &["person", "organization"], 0.5);
                            total += sentence.text().len();
                        }
                        black_box(total)
                    })
                }
            });
        }
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
