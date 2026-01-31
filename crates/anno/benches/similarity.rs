use anno::linking::candidate;
use anno_core::coalesce::similarity::Similarity;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_similarity(c: &mut Criterion) {
    let sim = Similarity::new();

    let a = "Marie Curie";
    let b = "Curie";
    let cjk_a = "北京市";
    let cjk_b = "北京";

    c.bench_function("coalesce::Similarity::compute (latin)", |bch| {
        bch.iter(|| black_box(sim.compute(black_box(a), black_box(b))))
    });

    c.bench_function("coalesce::Similarity::compute (cjk)", |bch| {
        bch.iter(|| black_box(sim.compute(black_box(cjk_a), black_box(cjk_b))))
    });

    c.bench_function("linking::candidate::string_similarity (latin)", |bch| {
        bch.iter(|| black_box(candidate::string_similarity(black_box(a), black_box(b))))
    });

    c.bench_function("linking::candidate::string_similarity (cjk)", |bch| {
        bch.iter(|| {
            black_box(candidate::string_similarity(
                black_box(cjk_a),
                black_box(cjk_b),
            ))
        })
    });

    // When `--features gramdex` is enabled, compare the two implementations directly.
    #[cfg(feature = "gramdex")]
    {
        c.bench_function(
            "linking::candidate::string_similarity_textprep (latin)",
            |bch| {
                bch.iter(|| {
                    black_box(candidate::string_similarity_textprep(
                        black_box(a),
                        black_box(b),
                    ))
                })
            },
        );

        c.bench_function(
            "linking::candidate::string_similarity_gramdex (latin)",
            |bch| {
                bch.iter(|| {
                    black_box(candidate::string_similarity_gramdex(
                        black_box(a),
                        black_box(b),
                    ))
                })
            },
        );
    }
}

criterion_group!(benches, bench_similarity);
criterion_main!(benches);
