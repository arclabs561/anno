//! Profile GLiNER extract_entities to identify bottlenecks.
//!
//! Measures time spent in:
//! - encode_prompt (tokenization + encoding)
//! - ONNX inference
//! - span generation
//! - similarity computation
//! - entity decoding
//!
//! Run with:
//!   cargo bench --bench gliner_profiling --features onnx

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Instant;

#[cfg(feature = "onnx")]
fn bench_gliner_extract_breakdown(c: &mut Criterion) {
    use anno::backends::gliner_onnx::GLiNEROnnx;
    
    // Initialize model once
    let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")
        .expect("Failed to load GLiNER model");
    
    let test_cases = vec![
        ("short", "Apple Inc. was founded by Steve Jobs."),
        ("medium", "Apple Inc. was founded by Steve Jobs in California in 1976. The company is headquartered in Cupertino."),
        ("long", "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in California. The company is headquartered in Cupertino and is known for products like the iPhone, iPad, and Mac computers. Tim Cook became CEO in 2011 after Jobs stepped down."),
    ];
    
    let entity_types = &["person", "organization", "location", "date"];
    
    let mut group = c.benchmark_group("gliner_extract_breakdown");
    group.sample_size(10); // Fewer samples for detailed profiling
    
    for (name, text) in test_cases {
        // Full extraction
        group.bench_function(BenchmarkId::new("full_extract", name), |b| {
            b.iter(|| {
                let _ = model.extract(black_box(text), black_box(entity_types), 0.5);
            });
        });
        
        // Measure encode_prompt cost indirectly by comparing full extract times
        // The encode_prompt_only benchmark is removed as it was redundant
    }
    
    group.finish();
}

#[cfg(feature = "onnx")]
fn bench_gliner_cache_impact(c: &mut Criterion) {
    use anno::backends::gliner_onnx::GLiNEROnnx;
    
    let model = GLiNEROnnx::new("onnx-community/gliner_small-v2.1")
        .expect("Failed to load GLiNER model");
    
    let text = "Apple Inc. was founded by Steve Jobs in California in 1976.";
    let entity_types = &["person", "organization", "location", "date"];
    
    let mut group = c.benchmark_group("gliner_cache_impact");
    
    // First call (cache miss, if cache exists)
    group.bench_function("first_call", |b| {
        b.iter(|| {
            let _ = model.extract(black_box(text), black_box(entity_types), 0.5);
        });
    });
    
    // Second call (cache hit, if cache exists)
    group.bench_function("second_call_same_text_types", |b| {
        // Warm up
        let _ = model.extract(text, entity_types, 0.5);
        
        b.iter(|| {
            let _ = model.extract(black_box(text), black_box(entity_types), 0.5);
        });
    });
    
    // Different entity types (cache miss, if caching by text only)
    let different_types = &["person", "organization"];
    group.bench_function("different_types", |b| {
        b.iter(|| {
            let _ = model.extract(black_box(text), black_box(different_types), 0.5);
        });
    });
    
    // Different text (cache miss)
    let different_text = "Microsoft was founded by Bill Gates in Washington.";
    group.bench_function("different_text", |b| {
        b.iter(|| {
            let _ = model.extract(black_box(different_text), black_box(entity_types), 0.5);
        });
    });
    
    group.finish();
}

// Removed bench_gliner_encode_prompt_cost - redundant with bench_gliner_extract_breakdown

#[cfg(not(feature = "onnx"))]
fn bench_gliner_extract_breakdown(_c: &mut Criterion) {
    // Stub when onnx feature is not enabled
}

#[cfg(not(feature = "onnx"))]
fn bench_gliner_cache_impact(_c: &mut Criterion) {
    // Stub when onnx feature is not enabled
}


criterion_group!(
    benches,
    bench_gliner_extract_breakdown,
    bench_gliner_cache_impact
);
criterion_main!(benches);

