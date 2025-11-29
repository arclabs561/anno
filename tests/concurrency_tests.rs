//! Concurrency and thread-safety tests for NER backends.
//!
//! These tests verify that:
//! - Backends can be used from multiple threads safely
//! - Arc<dyn Model> works correctly
//! - No data races occur under concurrent load

use anno::{Entity, EntityType, HeuristicNER, Model, PatternNER, StackedNER};
use std::sync::Arc;
use std::thread;

// =============================================================================
// Thread Safety Tests
// =============================================================================

#[test]
fn pattern_ner_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PatternNER>();
}

#[test]
fn statistical_ner_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HeuristicNER>();
}

#[test]
fn stacked_ner_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StackedNER>();
}

// =============================================================================
// Concurrent Extraction Tests
// =============================================================================

#[test]
fn pattern_ner_concurrent_extraction() {
    let ner = Arc::new(PatternNER::new());
    let texts = vec![
        "Cost: $100",
        "Date: 2024-01-15",
        "Email: test@example.com",
        "Phone: (555) 123-4567",
        "Time: 3:30pm",
        "Percent: 25%",
    ];

    let handles: Vec<_> = texts
        .into_iter()
        .map(|text| {
            let ner = Arc::clone(&ner);
            let text = text.to_string();
            thread::spawn(move || {
                let entities = ner.extract_entities(&text, None).unwrap();
                (text, entities)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All should have found at least one entity
    for (text, entities) in &results {
        assert!(
            !entities.is_empty(),
            "Should find entities in '{}': {:?}",
            text,
            entities
        );
    }
}

#[test]
fn statistical_ner_concurrent_extraction() {
    let ner = Arc::new(HeuristicNER::new());
    let texts = vec![
        "Dr. John Smith is here",
        "Apple Inc. announced profits",
        "Meeting in New York City",
        "Mr. Robert Johnson called",
        "Microsoft Corporation released",
        "Visit San Francisco today",
    ];

    let handles: Vec<_> = texts
        .into_iter()
        .map(|text| {
            let ner = Arc::clone(&ner);
            let text = text.to_string();
            thread::spawn(move || {
                let entities = ner.extract_entities(&text, None).unwrap();
                (text, entities)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Most should have found at least one entity (heuristic, not guaranteed)
    let found_count = results.iter().filter(|(_, e)| !e.is_empty()).count();
    assert!(
        found_count >= 3,
        "Should find entities in at least half the texts: {:?}",
        results
    );
}

#[test]
fn stacked_ner_concurrent_extraction() {
    let ner = Arc::new(StackedNER::new());
    let texts = vec![
        "Dr. Smith charges $100/hr",
        "Meeting on 2024-01-15 in NYC",
        "Contact: john@test.com",
        "Call Apple at (555) 123-4567",
        "Revenue up 25% for Google Inc.",
        "Event at 3pm with Mr. Johnson",
    ];

    let handles: Vec<_> = texts
        .into_iter()
        .map(|text| {
            let ner = Arc::clone(&ner);
            let text = text.to_string();
            thread::spawn(move || {
                let entities = ner.extract_entities(&text, None).unwrap();
                (text, entities)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All should have found at least one entity (mixed patterns + statistical)
    for (text, entities) in &results {
        assert!(
            !entities.is_empty(),
            "Should find entities in '{}': {:?}",
            text,
            entities
        );
    }
}

#[test]
fn high_concurrency_pattern_ner() {
    let ner = Arc::new(PatternNER::new());
    let num_threads = 50;
    let text = "Price: $100, date: 2024-01-15, email: test@test.com";

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let ner = Arc::clone(&ner);
            let text = text.to_string();
            thread::spawn(move || ner.extract_entities(&text, None).unwrap())
        })
        .collect();

    let results: Vec<Vec<Entity>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All results should be identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            result.len(),
            first.len(),
            "Thread {} got different count: {} vs {}",
            i,
            result.len(),
            first.len()
        );
    }
}

#[test]
fn concurrent_different_texts_same_model() {
    let ner = Arc::new(StackedNER::new());

    // Generate unique texts for each thread
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let ner = Arc::clone(&ner);
            thread::spawn(move || {
                let text = format!(
                    "Dr. Person{} charges ${}/hr on 2024-01-{:02}",
                    i,
                    i * 100,
                    i + 1
                );
                let entities = ner.extract_entities(&text, None).unwrap();
                (i, text, entities)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Verify each result makes sense for its input
    for (i, text, entities) in &results {
        // Should find at least money (pattern)
        let has_money = entities.iter().any(|e| e.entity_type == EntityType::Money);
        assert!(
            has_money,
            "Thread {} text '{}' should have money: {:?}",
            i, text, entities
        );
    }
}

// =============================================================================
// Arc<dyn Model> Tests
// =============================================================================

#[test]
fn arc_dyn_model_concurrent() {
    let models: Vec<Arc<dyn Model + Send + Sync>> = vec![
        Arc::new(PatternNER::new()),
        Arc::new(HeuristicNER::new()),
        Arc::new(StackedNER::new()),
    ];

    let text = "Dr. Smith charges $100 on 2024-01-15";

    let handles: Vec<_> = models
        .into_iter()
        .map(|model| {
            let text = text.to_string();
            thread::spawn(move || {
                let name = model.name();
                let entities = model.extract_entities(&text, None).unwrap();
                (name, entities)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // PatternNER should find money and date (guaranteed)
    // HeuristicNER is heuristic - may or may not find
    // StackedNER combines both

    // At minimum, pattern should find something
    let pattern_found = results
        .iter()
        .any(|(name, e)| *name == "pattern" && !e.is_empty());
    assert!(pattern_found, "PatternNER should find entities");

    // Stacked should find something (includes pattern)
    let stacked_found = results
        .iter()
        .any(|(name, e)| name.starts_with("stacked") && !e.is_empty());
    assert!(stacked_found, "StackedNER should find entities");
}

// =============================================================================
// Determinism Tests (same input = same output across threads)
// =============================================================================

#[test]
fn pattern_ner_deterministic_across_threads() {
    let ner = Arc::new(PatternNER::new());
    let text = "Meeting on 2024-01-15 at 3:30pm, cost $500";

    // Run 100 times across threads
    let handles: Vec<_> = (0..100)
        .map(|_| {
            let ner = Arc::clone(&ner);
            let text = text.to_string();
            thread::spawn(move || ner.extract_entities(&text, None).unwrap())
        })
        .collect();

    let results: Vec<Vec<Entity>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All should produce identical results
    let first = &results[0];
    for result in &results[1..] {
        assert_eq!(
            result.len(),
            first.len(),
            "Entity count should be deterministic"
        );
        for (a, b) in first.iter().zip(result.iter()) {
            assert_eq!(a.text, b.text, "Entity text should be deterministic");
            assert_eq!(a.start, b.start, "Entity start should be deterministic");
            assert_eq!(a.end, b.end, "Entity end should be deterministic");
            assert_eq!(
                a.entity_type, b.entity_type,
                "Entity type should be deterministic"
            );
        }
    }
}

// =============================================================================
// Stress Test (marked ignore - run with --ignored)
// =============================================================================

#[test]
#[ignore]
fn stress_test_high_thread_count() {
    let ner = Arc::new(StackedNER::new());
    let num_threads = 200;
    let iterations_per_thread = 50;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let ner = Arc::clone(&ner);
            thread::spawn(move || {
                let mut success_count = 0;
                for i in 0..iterations_per_thread {
                    let text = format!(
                        "Thread {} iteration {}: Dr. Smith charges ${} on 2024-{:02}-{:02}",
                        thread_id,
                        i,
                        thread_id * 100 + i,
                        (thread_id % 12) + 1,
                        (i % 28) + 1
                    );
                    if ner.extract_entities(&text, None).is_ok() {
                        success_count += 1;
                    }
                }
                success_count
            })
        })
        .collect();

    let total_success: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let expected = num_threads * iterations_per_thread;

    assert_eq!(
        total_success, expected,
        "All {} extractions should succeed",
        expected
    );
}
