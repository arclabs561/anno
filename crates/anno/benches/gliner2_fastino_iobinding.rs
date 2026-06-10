//! Phase 3.5 M14: criterion benchmark comparing
//! [`anno::backends::gliner2_fastino::ExecutionMode::Standard`] and
//! [`anno::backends::gliner2_fastino::ExecutionMode::IoBinding`] on the
//! same workload.
//!
//! Run with the cached `SemplificaAI/gliner2-multi-v1-onnx` snapshot:
//!
//! ```bash
//! cargo bench --bench gliner2_fastino_iobinding --features gliner2-fastino
//! ```
//!
//! On CPU we expect IoBinding to be 1.5-3× faster than Standard on the
//! "long" input (more tensor passes between sessions amortise the
//! per-session-boundary copy savings). On a GPU host (with
//! `--features gliner2-fastino-cuda`), the gap should widen
//! substantially because device→host copies are the bottleneck.
//!
//! **Skipped automatically when the snapshot isn't cached** — the
//! `from_pretrained_with_config` call would attempt to download
//! ~6 GB. Each `bench_*` group eprintln-skips if model load fails.

#![cfg(feature = "gliner2-fastino")]

use anno::backends::gliner2_fastino::{ExecutionMode, GLiNER2Fastino, GLiNER2FastinoConfig};
use anno::backends::inference::ZeroShotNER;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const MODEL_ID: &str = "SemplificaAI/gliner2-multi-v1-onnx";

const SHORT: &str = "Marie Curie discovered radium.";

const MEDIUM: &str = "Apple CEO Tim Cook met Google CEO Sundar Pichai in Seattle. \
    The deal, worth EUR 3.2 billion, closes on March 15, 2026.";

const LONG: &str = "The European Central Bank raised interest rates by 25 basis points \
    on January 15, 2026. ECB President Christine Lagarde announced the decision at a \
    press conference in Frankfurt. The move affects all 20 eurozone member states. \
    Analysts at Goldman Sachs and JPMorgan Chase had predicted the increase. \
    Germany's Bundeskanzler Olaf Scholz called the decision 'necessary for stability.' \
    Meanwhile, Federal Reserve Chair Jerome Powell signaled the Fed would hold rates steady \
    at its next meeting in Washington, DC on February 1, 2026.";

const TYPES: &[&str] = &["person", "organization", "location", "date"];

fn try_load(mode: ExecutionMode) -> Option<GLiNER2Fastino> {
    GLiNER2Fastino::from_pretrained_with_config(
        MODEL_ID,
        GLiNER2FastinoConfig::default().with_execution_mode(mode),
    )
    .ok()
}

fn inputs() -> Vec<(&'static str, &'static str)> {
    vec![("short", SHORT), ("medium", MEDIUM), ("long", LONG)]
}

fn bench_extract_with_types(c: &mut Criterion) {
    let standard = match try_load(ExecutionMode::Standard) {
        Some(m) => m,
        None => {
            eprintln!(
                "[gliner2_fastino_iobinding bench] {MODEL_ID} not cached; skipping benches. \
                 Run an integration test against the model first to populate the HF cache."
            );
            return;
        }
    };
    let iobinding = match try_load(ExecutionMode::IoBinding) {
        Some(m) => m,
        None => {
            eprintln!(
                "[gliner2_fastino_iobinding bench] IoBinding load failed; skipping IoBinding side."
            );
            return;
        }
    };

    let mut group = c.benchmark_group("gliner2_fastino_extract_with_types");
    // Lower per-iter sample size — each iteration runs an 8-session ONNX
    // chain (~5-50 ms depending on input length).
    group.sample_size(20);

    for (name, text) in inputs() {
        group.bench_with_input(BenchmarkId::new("standard", name), text, |b, t| {
            b.iter(|| {
                let _ = ZeroShotNER::extract_with_types(
                    black_box(&standard),
                    black_box(t),
                    black_box(TYPES),
                    black_box(0.5),
                )
                .expect("standard extract");
            });
        });
        group.bench_with_input(BenchmarkId::new("iobinding", name), text, |b, t| {
            b.iter(|| {
                let _ = ZeroShotNER::extract_with_types(
                    black_box(&iobinding),
                    black_box(t),
                    black_box(TYPES),
                    black_box(0.5),
                )
                .expect("iobinding extract");
            });
        });
    }
    group.finish();
}

fn bench_classify(c: &mut Criterion) {
    let standard = match try_load(ExecutionMode::Standard) {
        Some(m) => m,
        None => return,
    };
    let iobinding = match try_load(ExecutionMode::IoBinding) {
        Some(m) => m,
        None => return,
    };

    let labels = ["positive", "negative", "neutral"];
    let text = "I absolutely loved every minute of the show — wonderful experience!";

    let mut group = c.benchmark_group("gliner2_fastino_classify");
    group.sample_size(30);

    group.bench_function("standard", |b| {
        b.iter(|| {
            let _ = standard
                .classify(black_box(text), black_box(&labels), black_box(0.5))
                .expect("standard classify");
        });
    });
    group.bench_function("iobinding", |b| {
        b.iter(|| {
            let _ = iobinding
                .classify(black_box(text), black_box(&labels), black_box(0.5))
                .expect("iobinding classify");
        });
    });
    group.finish();
}

criterion_group!(benches, bench_extract_with_types, bench_classify);
criterion_main!(benches);
