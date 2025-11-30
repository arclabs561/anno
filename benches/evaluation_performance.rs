use anno::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

fn create_test_sentences(count: usize) -> Vec<String> {
    let texts = vec![
        "Apple CEO Tim Cook announced new products in Cupertino.",
        "Microsoft and Google are competing in the cloud market.",
        "Barack Obama was the 44th President of the United States.",
        "The company is based in San Francisco, California.",
        "Amazon Web Services provides cloud computing services.",
        "Elon Musk founded Tesla and SpaceX in California.",
        "The United Nations headquarters is in New York City.",
        "Facebook changed its name to Meta in 2021.",
        "Jeff Bezos founded Amazon in Seattle, Washington.",
        "The European Union has 27 member countries.",
    ];

    (0..count)
        .map(|i| texts[i % texts.len()].to_string())
        .collect()
}

fn bench_sequential_processing(c: &mut Criterion) {
    let sentences = create_test_sentences(100);

    #[cfg(feature = "onnx")]
    {
        let mut group = c.benchmark_group("evaluation_processing");
        group.sample_size(10);

        // Sequential processing
        group.bench_function("sequential_100_sentences", |b| {
            b.iter(|| {
                let mut total = 0;
                for sentence in &sentences {
                    total += sentence.len();
                }
                black_box(total)
            })
        });

        // Parallel processing (if rayon available)
        #[cfg(feature = "eval-parallel")]
        {
            use rayon::prelude::*;
            group.bench_function("parallel_100_sentences", |b| {
                b.iter(|| {
                    let total: usize = sentences.par_iter().map(|s| s.len()).sum();
                    black_box(total)
                })
            });
        }

        group.finish();
    }
}

fn bench_backend_extraction(c: &mut Criterion) {
    let text = "Apple CEO Tim Cook announced new products in Cupertino, California.";

    #[cfg(feature = "onnx")]
    {
        let mut group = c.benchmark_group("backend_extraction");
        group.sample_size(20);

        // Pattern NER (baseline)
        let pattern_ner = PatternNER::new();
        group.bench_function("pattern_ner", |b| {
            b.iter(|| {
                let _ = pattern_ner.extract_entities(black_box(text), None);
            })
        });

        // Stacked NER
        let stacked = StackedNER::default();
        group.bench_function("stacked_ner", |b| {
            b.iter(|| {
                let _ = stacked.extract_entities(black_box(text), None);
            })
        });

        // GLiNER ONNX (if available)
        if let Ok(gliner) = GLiNEROnnx::new(DEFAULT_GLINER_MODEL) {
            group.bench_function("gliner_onnx", |b| {
                b.iter(|| {
                    let _ = gliner.extract_entities(black_box(text), None);
                })
            });
        }

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_sequential_processing,
    bench_backend_extraction
);
criterion_main!(benches);
