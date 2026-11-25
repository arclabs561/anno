use anno::backends::coref::mention_ranking::MentionRankingCoref;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ---------------------------------------------------------------------------
// Coref test inputs
// ---------------------------------------------------------------------------

const SIMPLE_COREF: &str =
    "Marie Curie discovered radium. She won the Nobel Prize. Curie later moved to Paris.";

const MULTI_ENTITY: &str = "Ada Lovelace worked with Charles Babbage on the Analytical Engine. \
    She wrote the first algorithm. Babbage designed the machine in London. \
    Lovelace published her notes in 1843.";

const DENSE_PRONOUNS: &str = "Grace Hopper joined the Navy in 1943. She developed COBOL. \
    Her work at Harvard was groundbreaking. They named a ship after her. \
    Hopper received the Presidential Medal of Freedom. She retired as a rear admiral.";

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_coref_resolve(c: &mut Criterion) {
    let coref = MentionRankingCoref::new();
    let mut group = c.benchmark_group("coref/resolve");

    for (name, text) in [
        ("simple", SIMPLE_COREF),
        ("multi_entity", MULTI_ENTITY),
        ("dense_pronouns", DENSE_PRONOUNS),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &text| {
            b.iter(|| coref.resolve(black_box(text)));
        });
    }
    group.finish();
}

fn bench_coref_scaling(c: &mut Criterion) {
    let coref = MentionRankingCoref::new();
    let base = "Sophie Wilson designed ARM. She worked at Acorn. Wilson later joined Broadcom. ";
    let mut group = c.benchmark_group("coref/scaling");

    for n in [1, 3, 5, 10] {
        let text = base.repeat(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n}x")),
            &text,
            |b, text| {
                b.iter(|| coref.resolve(black_box(text)));
            },
        );
    }
    group.finish();
}

fn bench_coref_to_grounded(c: &mut Criterion) {
    let coref = MentionRankingCoref::new();
    let text = MULTI_ENTITY;
    let mut group = c.benchmark_group("coref/resolve_to_grounded");

    group.bench_function("multi_entity", |b| {
        b.iter(|| coref.resolve_to_grounded(black_box(text)));
    });
    group.finish();
}

fn bench_grounded_validation(c: &mut Criterion) {
    use anno::{GroundedDocument, Location, Signal};

    let mut group = c.benchmark_group("grounded/validate");
    for count in [256, 512, 1024, 2048] {
        let mut doc = GroundedDocument::new("dense", "甲 🦊 ".repeat(count));
        for index in 0..count {
            let start = 4 * index + 2;
            doc.add_signal(Signal::new(
                0,
                Location::text(start, start + 1),
                "🦊",
                "ANIMAL",
                1.0,
            ));
        }
        assert!(doc.is_valid());
        group.bench_with_input(BenchmarkId::from_parameter(count), &doc, |b, doc| {
            b.iter(|| black_box(doc).is_valid())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_coref_resolve,
    bench_coref_scaling,
    bench_coref_to_grounded,
    bench_grounded_validation
);
criterion_main!(benches);
