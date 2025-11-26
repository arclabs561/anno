use anno::{Model, PatternNER};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_pattern_ner(c: &mut Criterion) {
    let ner = PatternNER::new();
    let text = "Meeting scheduled for January 15, 2025 at $500 per hour, estimated 15% completion.";

    c.bench_function("pattern_ner", |b| {
        b.iter(|| ner.extract_entities(black_box(text), None))
    });
}

criterion_group!(benches, bench_pattern_ner);
criterion_main!(benches);

