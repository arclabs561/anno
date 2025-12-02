//! Performance benchmarks for StackedNER overlap detection and conflict resolution.
//!
//! This benchmark tests StackedNER's performance with:
//! - Varying numbers of entities
//! - Different conflict resolution strategies
//! - Multiple overlapping entities
//! - Large entity sets
//!
//! # Optimizations
//!
//! StackedNER has been optimized with:
//! - Cached `text.chars().count()` to avoid repeated O(n) operations
//! - Pre-allocated vectors to reduce reallocations
//! - Unstable sorting for better performance when stability isn't needed
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench stacked_ner_performance
//! ```

use anno::backends::stacked::ConflictStrategy;
use anno::{Entity, EntityType, MockModel, Model, StackedNER};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Generate test entities with varying overlap patterns
fn generate_test_entities(count: usize, overlap_ratio: f64) -> Vec<Entity> {
    let mut entities = Vec::new();
    let base_text = "This is a test sentence with multiple entities that may overlap. ";
    let base_char_count = base_text.chars().count();

    for i in 0..count {
        let start = (i * 10) % base_char_count.max(20).saturating_sub(10);
        let start = start.max(0);
        let max_end = (start + 20).min(base_char_count);

        // Create overlapping entities based on overlap_ratio
        let actual_start = if i > 0 && (i as f64 * overlap_ratio) as usize > 0 {
            // Make some entities overlap with previous ones
            let prev_end = entities.last().map(|e: &Entity| e.end).unwrap_or(0);
            if prev_end > start && prev_end < base_char_count {
                (prev_end - 5).max(0) // Overlap by 5 chars, but ensure valid
            } else {
                start
            }
        } else {
            start
        };

        // Ensure end is valid and > start
        let actual_end = if actual_start < max_end {
            (actual_start + 10 + (i % 10))
                .min(max_end)
                .max(actual_start + 1)
        } else {
            actual_start + 1
        };

        // Final validation
        let actual_start = actual_start.min(base_char_count.saturating_sub(1));
        let actual_end = actual_end.min(base_char_count).max(actual_start + 1);

        let entity_type = match i % 4 {
            0 => EntityType::Person,
            1 => EntityType::Organization,
            2 => EntityType::Location,
            _ => EntityType::Date,
        };

        // Extract text from base_text using character offsets
        let text_chars: Vec<char> = base_text.chars().collect();
        let text = if actual_start < text_chars.len() && actual_end <= text_chars.len() {
            text_chars[actual_start..actual_end].iter().collect()
        } else {
            format!("Entity{}", i) // Fallback
        };

        // Ensure we have valid entity
        if actual_start < actual_end && actual_end <= base_char_count {
            entities.push(Entity::new(
                text,
                entity_type,
                actual_start,
                actual_end,
                0.5 + (i as f64 % 5.0) / 10.0, // Confidence 0.5-0.9
            ));
        }
    }

    entities
}

fn bench_overlap_detection_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlap_detection_scaling");

    for entity_count in [10, 50, 100, 200, 500].iter() {
        for overlap_pct in [0, 25, 50, 75, 90].iter() {
            let overlap_ratio = *overlap_pct as f64 / 100.0;
            let entities = generate_test_entities(*entity_count, overlap_ratio);

            // Create a mock model that returns these entities
            // Use static string slices for names (acceptable for benchmarks)
            let model_name = match entity_count {
                10 => "mock_10",
                50 => "mock_50",
                100 => "mock_100",
                200 => "mock_200",
                500 => "mock_500",
                _ => "mock_other",
            };
            let model = MockModel::new(model_name).with_entities(entities.clone());

            let ner = StackedNER::builder().layer(model).build();

            let test_text =
                "This is a test sentence with multiple entities that may overlap. ".repeat(10);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}_entities_{}%_overlap",
                    entity_count, overlap_pct
                )),
                &test_text,
                |b, text| {
                    b.iter(|| black_box(ner.extract_entities(black_box(text), None).unwrap()));
                },
            );
        }
    }

    group.finish();
}

fn bench_conflict_strategy_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("conflict_strategy_performance");

    let strategies = [
        ConflictStrategy::Priority,
        ConflictStrategy::LongestSpan,
        ConflictStrategy::HighestConf,
        ConflictStrategy::Union,
    ];

    // Generate entities with high overlap
    let entities = generate_test_entities(100, 0.75);
    let model = MockModel::new("mock_conflict").with_entities(entities);

    let test_text = "This is a test sentence with multiple entities that may overlap. ".repeat(10);

    for strategy in strategies.iter() {
        let ner = StackedNER::builder()
            .layer(model.clone())
            .strategy(*strategy)
            .build();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", strategy)),
            &test_text,
            |b, text| {
                b.iter(|| black_box(ner.extract_entities(black_box(text), None).unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_many_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("many_layers");

    for layer_count in [1, 2, 3, 5, 10].iter() {
        let mut builder = StackedNER::builder();

        // Use static string slices for layer names (acceptable for benchmarks)
        let layer_names = [
            "layer0", "layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7",
            "layer8", "layer9",
        ];
        for i in 0..*layer_count {
            let entities = generate_test_entities(20, 0.3);
            let layer_name = layer_names[i.min(layer_names.len() - 1)];
            let model = MockModel::new(layer_name).with_entities(entities);
            builder = builder.layer(model);
        }

        let ner = builder.build();
        let test_text =
            "This is a test sentence with multiple entities that may overlap. ".repeat(5);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_layers", layer_count)),
            &test_text,
            |b, text| {
                b.iter(|| black_box(ner.extract_entities(black_box(text), None).unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_real_world_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_patterns");

    let test_cases = vec![
        ("short", "Email ceo@apple.com about Apple stock for $100"),
        ("medium", "Apple CEO Tim Cook announced new products in Cupertino, California on January 15, 2025. Contact: tim@apple.com for $100/hr."),
    ];

    // Add long test case
    let long_text = "Apple CEO Tim Cook announced new products in Cupertino, California on January 15, 2025. Contact: tim@apple.com for $100/hr. ".repeat(10);
    let mut test_cases_with_long = test_cases;
    test_cases_with_long.push(("long", &long_text));

    let ner = StackedNER::default();

    for (name, text) in test_cases_with_long {
        group.bench_with_input(BenchmarkId::from_parameter(name), text, |b, t| {
            b.iter(|| black_box(ner.extract_entities(black_box(t), None).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_overlap_detection_scaling,
    bench_conflict_strategy_performance,
    bench_many_layers,
    bench_real_world_patterns
);
criterion_main!(benches);
