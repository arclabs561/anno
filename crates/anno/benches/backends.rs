use anno::backends::crf::CrfNER;
use anno::backends::heuristic::HeuristicNER;
use anno::backends::hmm::HmmNER;
use anno::backends::regex::RegexNER;
use anno::backends::stacked::StackedNER;
use anno::Model;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Test inputs (increasing complexity)
// ---------------------------------------------------------------------------

const SHORT: &str = "Marie Curie discovered radium.";

const MEDIUM: &str = "Apple CEO Tim Cook met Google CEO Sundar Pichai in Seattle. \
    The deal, worth EUR 3.2 billion, closes on March 15, 2026. \
    Contact press@apple.com or call +1-555-867-5309.";

const LONG: &str = "The European Central Bank raised interest rates by 25 basis points \
    on January 15, 2026. ECB President Christine Lagarde announced the decision at a \
    press conference in Frankfurt. The move affects all 20 eurozone member states. \
    Analysts at Goldman Sachs and JPMorgan Chase had predicted the increase. \
    Germany's Bundeskanzler Olaf Scholz called the decision 'necessary for stability.' \
    The EUR/USD exchange rate moved to 1.0842 following the announcement. \
    Meanwhile, Federal Reserve Chair Jerome Powell signaled that the Fed would hold \
    rates steady at its next meeting in Washington, DC on February 1, 2026. \
    Contact: ecb-press@ecb.europa.eu or +49-69-1344-0.";

fn inputs() -> Vec<(&'static str, &'static str)> {
    vec![("short", SHORT), ("medium", MEDIUM), ("long", LONG)]
}

// ---------------------------------------------------------------------------
// Individual backend benchmarks
// ---------------------------------------------------------------------------

fn bench_regex(c: &mut Criterion) {
    let model = RegexNER::new();
    let mut group = c.benchmark_group("regex");
    for (name, text) in inputs() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &text| {
            b.iter(|| model.extract_entities(black_box(text), None));
        });
    }
    group.finish();
}

fn bench_heuristic(c: &mut Criterion) {
    let model = HeuristicNER::new();
    let mut group = c.benchmark_group("heuristic");
    for (name, text) in inputs() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &text| {
            b.iter(|| model.extract_entities(black_box(text), None));
        });
    }
    group.finish();
}

fn bench_crf(c: &mut Criterion) {
    let model = CrfNER::new();
    let mut group = c.benchmark_group("crf");
    for (name, text) in inputs() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &text| {
            b.iter(|| model.extract_entities(black_box(text), None));
        });
    }
    group.finish();
}

fn bench_hmm(c: &mut Criterion) {
    let model = HmmNER::new();
    let mut group = c.benchmark_group("hmm");
    for (name, text) in inputs() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &text| {
            b.iter(|| model.extract_entities(black_box(text), None));
        });
    }
    group.finish();
}

fn bench_stacked_default(c: &mut Criterion) {
    let model = StackedNER::new();
    let mut group = c.benchmark_group("stacked_default");
    for (name, text) in inputs() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, &text| {
            b.iter(|| model.extract_entities(black_box(text), None));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Cross-backend comparison (same input, all non-ML backends)
// ---------------------------------------------------------------------------

fn bench_cross_backend(c: &mut Criterion) {
    let regex = RegexNER::new();
    let heuristic = HeuristicNER::new();
    let crf = CrfNER::new();
    let hmm = HmmNER::new();
    let stacked = StackedNER::new();

    let text = MEDIUM;
    let mut group = c.benchmark_group("cross_backend/medium");

    group.bench_function("regex", |b| {
        b.iter(|| regex.extract_entities(black_box(text), None));
    });
    group.bench_function("heuristic", |b| {
        b.iter(|| heuristic.extract_entities(black_box(text), None));
    });
    group.bench_function("crf", |b| {
        b.iter(|| crf.extract_entities(black_box(text), None));
    });
    group.bench_function("hmm", |b| {
        b.iter(|| hmm.extract_entities(black_box(text), None));
    });
    group.bench_function("stacked", |b| {
        b.iter(|| stacked.extract_entities(black_box(text), None));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Entity count scaling (stacked on increasingly large input)
// ---------------------------------------------------------------------------

fn bench_scaling(c: &mut Criterion) {
    let model = StackedNER::new();
    let base = "Dr. Sophie Wilson designed the ARM processor at Acorn Computers in Cambridge. ";
    let mut group = c.benchmark_group("scaling/stacked");

    for n in [1, 5, 10, 50] {
        let text = base.repeat(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n}x")),
            &text,
            |b, text| {
                b.iter(|| model.extract_entities(black_box(text), None));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_regex,
    bench_heuristic,
    bench_crf,
    bench_hmm,
    bench_stacked_default,
    bench_cross_backend,
    bench_scaling,
);
criterion_main!(benches);
