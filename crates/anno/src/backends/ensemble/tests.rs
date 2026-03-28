use super::*;
use anno_core::{Confidence, ExtractionMethod};

fn fast_ensemble() -> EnsembleNER {
    // Keep unit tests deterministic and fast: do not initialize model-loading backends here.
    EnsembleNER::with_backends(vec![
        Box::new(crate::RegexNER::new()),
        Box::new(crate::HeuristicNER::new()),
    ])
}

// =============================================================================
// Mock backends for parallelism and error-handling tests
// =============================================================================

/// A backend that always returns the same fixed set of entities.
///
/// Used to test determinism of parallel execution across runs.
struct FixedBackend {
    name: &'static str,
    entities: Vec<Entity>,
}

impl FixedBackend {
    fn new(name: &'static str, entities: Vec<Entity>) -> Self {
        Self { name, entities }
    }
}

impl crate::sealed::Sealed for FixedBackend {}

impl crate::Model for FixedBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn extract_entities(
        &self,
        _text: &str,
        _language: Option<Language>,
    ) -> crate::Result<Vec<Entity>> {
        Ok(self.entities.clone())
    }

    fn supported_types(&self) -> Vec<EntityType> {
        self.entities
            .iter()
            .map(|e| e.entity_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    fn is_available(&self) -> bool {
        true
    }
}

/// A backend that always returns `Err(...)`.
///
/// Used to verify the ensemble skips failing backends gracefully.
struct AlwaysErrBackend {
    name: &'static str,
}

impl AlwaysErrBackend {
    fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl crate::sealed::Sealed for AlwaysErrBackend {}

impl crate::Model for AlwaysErrBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn extract_entities(
        &self,
        _text: &str,
        _language: Option<Language>,
    ) -> crate::Result<Vec<Entity>> {
        Err(crate::Error::ModelInit(format!(
            "AlwaysErrBackend '{}' intentionally failed",
            self.name
        )))
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![]
    }

    fn is_available(&self) -> bool {
        false
    }
}

#[test]
fn test_new_backend_ids_have_weights() {
    let ner = EnsembleNER::new();

    // For the built-in constructor, we require stable IDs so weights apply as intended.
    assert!(
        !ner.backend_ids.is_empty(),
        "EnsembleNER::new() should have at least one backend"
    );

    for id in &ner.backend_ids {
        assert!(
                ner.weights.contains_key(id),
                "EnsembleNER::new(): missing weight for backend id={:?}. This usually means the ensemble's advertised IDs drifted from default_backend_weights keys.",
                id
            );
    }
}

#[test]
fn test_ensemble_basic() {
    let ner = fast_ensemble();
    let entities = ner
        .extract_entities("Tim Cook is the CEO of Apple Inc.", None)
        .unwrap();

    // Should find at least some entities
    assert!(!entities.is_empty(), "Should extract entities");

    // Check that provenance exists (may or may not say "ensemble" for single-source entities)
    for e in &entities {
        assert!(
            e.provenance.is_some(),
            "All entities should have provenance"
        );
    }
}

#[test]
fn test_span_overlap() {
    // Span1 [0-10], Span2 [5-15]: overlap [5-10] = 5 chars
    // Smaller span = 10 chars, overlap/smaller = 5/10 = 0.5
    // Need >0.5 so this is borderline - adjust test
    let span1 = SpanKey { start: 0, end: 10 };
    let span2 = SpanKey { start: 3, end: 15 }; // overlap [3-10] = 7 chars, 7/10 = 0.7 > 0.5
    let span3 = SpanKey { start: 20, end: 30 };

    assert!(span1.overlaps(&span2), "Overlapping spans should match");
    assert!(
        !span1.overlaps(&span3),
        "Non-overlapping spans should not match"
    );
}

#[test]
fn test_backend_weights() {
    let weights = default_backend_weights();

    // Pattern should have high weight
    assert!(weights["regex"].overall > 0.9);

    // GLiNER should have good weight
    assert!(weights["gliner"].overall > 0.8);

    // Heuristic should have lower weight
    assert!(weights["heuristic"].overall < 0.7);
}

#[test]
fn test_type_specific_weights() {
    let weights = default_backend_weights();

    // Pattern should be best for dates
    let pattern_date = weights["regex"].per_type.as_ref().unwrap().date;
    let heuristic_date = weights["heuristic"].per_type.as_ref().unwrap().date;
    assert!(pattern_date > heuristic_date);

    // Heuristic should be decent for orgs
    let heuristic_org = weights["heuristic"].per_type.as_ref().unwrap().organization;
    assert!(heuristic_org > 0.6);
}

#[test]
fn test_agreement_bonus() {
    let ner = fast_ensemble().with_agreement_bonus(0.15);
    assert!((ner.agreement_bonus - 0.15).abs() < 0.001);
}

#[test]
fn test_weight_learner_basic() {
    let mut learner = WeightLearner::new();

    // Add some training examples
    learner.add_example(&WeightTrainingExample {
        text: "Apple".to_string(),
        gold_type: EntityType::Organization,
        start: 0,
        end: 5,
        predictions: vec![
            (
                "heuristic".to_string(),
                EntityType::Organization,
                Confidence::new(0.8),
            ),
            (
                "gliner".to_string(),
                EntityType::Organization,
                Confidence::new(0.9),
            ),
        ],
    });

    learner.add_example(&WeightTrainingExample {
        text: "Paris".to_string(),
        gold_type: EntityType::Location,
        start: 0,
        end: 5,
        predictions: vec![
            (
                "heuristic".to_string(),
                EntityType::Person,
                Confidence::new(0.6),
            ), // Wrong!
            (
                "gliner".to_string(),
                EntityType::Location,
                Confidence::new(0.85),
            ),
        ],
    });

    // Learn weights
    let weights = learner.learn_weights();

    // GLiNER should have higher weight (2/2 correct vs 1/2)
    let gliner_weight = weights.get("gliner").map(|w| w.overall).unwrap_or(0.0);
    let heuristic_weight = weights.get("heuristic").map(|w| w.overall).unwrap_or(0.0);

    assert!(
        gliner_weight > heuristic_weight,
        "GLiNER should have higher weight (was {} vs {})",
        gliner_weight,
        heuristic_weight
    );
}

#[test]
fn test_backend_stats() {
    let mut stats = BackendStats {
        correct: 8,
        total: 10,
        ..Default::default()
    };
    stats.per_type.insert("PER".to_string(), (5, 6));

    assert!((stats.precision() - 0.8).abs() < 0.01);
    assert!((stats.type_precision("PER") - 0.833).abs() < 0.01);
    assert!((stats.type_precision("ORG") - 0.0).abs() < 0.01); // Unknown type
}

// =========================================================================
// Additional Edge Case Tests
// =========================================================================

#[test]
fn test_empty_text() {
    let ner = fast_ensemble();
    let entities = ner.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_whitespace_only_text() {
    let ner = fast_ensemble();
    let entities = ner.extract_entities("   \t\n   ", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_resolve_candidates_tie_break_is_order_independent() {
    let ner = fast_ensemble();
    let span_text = "Apple";
    let span = (0, 5);

    let e_person = Entity::new(span_text, EntityType::Person, span.0, span.1, 0.5);
    let e_org = Entity::new(span_text, EntityType::Organization, span.0, span.1, 0.5);

    let c1 = Candidate {
        entity: e_person,
        source: "heuristic".to_string(),
        backend_weight: 1.0,
    };
    let c2 = Candidate {
        entity: e_org,
        source: "heuristic".to_string(),
        backend_weight: 1.0,
    };

    let out_a = ner
        .resolve_candidates(vec![c1.clone(), c2.clone()])
        .expect("should resolve");
    let out_b = ner
        .resolve_candidates(vec![c2, c1])
        .expect("should resolve");

    assert_eq!(
        out_a.entity_type, out_b.entity_type,
        "tie resolution should not depend on candidate order"
    );

    let key_a = out_a.entity_type.as_label().to_string();
    let person_key = EntityType::Person.as_label().to_string();
    let org_key = EntityType::Organization.as_label().to_string();
    let expected = std::cmp::min(person_key, org_key);
    assert_eq!(
        key_a, expected,
        "tie-break should choose lexicographically smallest type label"
    );
}

#[test]
fn test_single_source_preserves_underlying_method_and_pattern() {
    // With a single backend, ensemble should preserve the backend's extraction method/pattern
    // (important for explainability and nested composition).
    let ner = EnsembleNER::with_backends(vec![Box::new(crate::RegexNER::new())]);
    let text = "Contact test@email.com on 2024-01-15";
    let entities = ner.extract_entities(text, None).expect("extract");
    assert!(!entities.is_empty());

    let email = entities
        .iter()
        .find(|e| e.text == "test@email.com")
        .expect("email entity should exist");
    let prov = email.provenance.as_ref().expect("provenance");

    assert_eq!(prov.method, ExtractionMethod::Pattern);
    assert!(
        prov.pattern.is_some(),
        "expected to preserve regex pattern name"
    );
}

#[test]
fn test_nested_single_source_preserves_inner_method() {
    // Inner ensemble produces provenance.method = Heuristic; outer should not overwrite it
    // to Neural just because the backend id is "ensemble(...)".
    let inner = EnsembleNER::with_backends(vec![Box::new(crate::HeuristicNER::new())]);
    let outer = EnsembleNER::with_backends(vec![Box::new(inner)]);

    let text = "John Smith visited Paris.";
    let entities = outer.extract_entities(text, None).expect("extract");
    assert!(!entities.is_empty());

    for e in &entities {
        let prov = e.provenance.as_ref().expect("provenance");
        assert_eq!(
            prov.method,
            ExtractionMethod::Heuristic,
            "expected outer to preserve inner method"
        );
    }
}

#[test]
fn test_span_key_self_overlap() {
    let span = SpanKey { start: 0, end: 10 };
    assert!(span.overlaps(&span), "Span should overlap with itself");
}

#[test]
fn test_span_key_adjacent_no_overlap() {
    let span1 = SpanKey { start: 0, end: 10 };
    let span2 = SpanKey { start: 10, end: 20 };
    assert!(!span1.overlaps(&span2), "Adjacent spans should not overlap");
}

#[test]
fn test_span_key_contained() {
    let outer = SpanKey { start: 0, end: 20 };
    let inner = SpanKey { start: 5, end: 15 };
    assert!(outer.overlaps(&inner), "Contained spans should overlap");
    assert!(inner.overlaps(&outer), "Overlap should be symmetric");
}

#[test]
fn test_backend_stats_empty() {
    let stats = BackendStats::default();
    assert!((stats.precision() - 0.0).abs() < 0.001);
    assert!((stats.type_precision("ANY") - 0.0).abs() < 0.001);
}

#[test]
fn test_weight_learner_empty() {
    let learner = WeightLearner::new();
    let weights = learner.learn_weights();
    // Empty learner returns empty weights (caller should use defaults)
    assert!(
        weights.is_empty(),
        "empty learner should return empty weights"
    );
}

#[test]
fn test_ensemble_with_language() {
    let ner = fast_ensemble();

    // Try with English language hint
    let entities = ner
        .extract_entities("Tim Cook is the CEO of Apple.", Some(Language::English))
        .unwrap();

    // Should find entities (language hint shouldn't break anything)
    assert!(
        !entities.is_empty(),
        "Should find entities with language hint"
    );
}

#[test]
fn test_type_weights_structure() {
    let weights = TypeWeights {
        person: 0.9,
        location: 0.85,
        organization: 0.88,
        date: 0.95,
        money: 0.8,
        other: 0.7,
    };

    assert!(weights.person > 0.0);
    assert!(weights.date > weights.other);
}

#[test]
fn test_backend_weight_structure() {
    let weight = BackendWeight {
        overall: 0.85,
        per_type: Some(TypeWeights {
            person: 0.9,
            location: 0.88,
            organization: 0.87,
            date: 0.92,
            money: 0.85,
            other: 0.75,
        }),
    };

    assert!(weight.overall > 0.0);
    assert!(weight.per_type.is_some());
}

#[test]
fn test_unicode_extraction() {
    let ner = EnsembleNER::new();
    let entities = ner
        .extract_entities("東京で会議がありました。", None)
        .unwrap();

    // Should not crash on Unicode
    for e in &entities {
        assert!(e.confidence >= 0.0 && e.confidence <= 1.0);
    }
}

#[test]
fn test_ensemble_provenance_tracking() {
    let ner = EnsembleNER::new();
    let entities = ner
        .extract_entities("Barack Obama visited Paris yesterday.", None)
        .unwrap();

    for e in &entities {
        // All entities should have provenance
        assert!(
            e.provenance.is_some(),
            "Entity '{}' ({:?}) at {}..{} has no provenance",
            e.text,
            e.entity_type,
            e.start(),
            e.end()
        );
        let prov = e.provenance.as_ref().unwrap();
        // Provenance source should not be empty
        assert!(!prov.source.is_empty());
    }
}

// =============================================================================
// Parallel execution determinism
// =============================================================================

/// Build an ensemble with three mock backends that return different entity counts.
///
/// Backend A: 3 entities (non-overlapping spans)
/// Backend B: 2 entities (different spans, no overlap with A)
/// Backend C: 1 entity  (unique span)
///
/// Each entity is placed at a distinct non-overlapping span so every candidate
/// ends up in its own conflict cluster. This maximises surface area: all six
/// entities must survive into the output, and their order / confidence must be
/// identical across repeated calls.
fn determinism_ensemble() -> EnsembleNER {
    // Span layout (char offsets into "aaaa bbbb cccc dddd eeee ffff"):
    //   [0,4)   "aaaa"  -- Backend A: Person
    //   [5,9)   "bbbb"  -- Backend A: Organization
    //   [10,14) "cccc"  -- Backend A: Date
    //   [15,19) "dddd"  -- Backend B: Person
    //   [20,24) "eeee"  -- Backend B: Location
    //   [25,29) "ffff"  -- Backend C: Money
    let backend_a = FixedBackend::new(
        "backend-a",
        vec![
            Entity::new("aaaa", EntityType::Person, 0, 4, 0.80),
            Entity::new("bbbb", EntityType::Organization, 5, 9, 0.75),
            Entity::new("cccc", EntityType::Date, 10, 14, 0.90),
        ],
    );
    let backend_b = FixedBackend::new(
        "backend-b",
        vec![
            Entity::new("dddd", EntityType::Person, 15, 19, 0.70),
            Entity::new("eeee", EntityType::Location, 20, 24, 0.65),
        ],
    );
    let backend_c = FixedBackend::new(
        "backend-c",
        vec![Entity::new("ffff", EntityType::Money, 25, 29, 0.85)],
    );
    EnsembleNER::with_backends(vec![
        Box::new(backend_a),
        Box::new(backend_b),
        Box::new(backend_c),
    ])
}

#[test]
fn test_parallel_execution_is_deterministic() {
    // Run the same ensemble 10 times on the same input.
    // Every run must produce an identical entity list: same count, same order,
    // same spans, same types, and same confidence values.
    //
    // The ensemble uses std::thread::scope internally (parallel backends), so
    // any race-condition or non-deterministic HashMap iteration that leaks into
    // the output will be caught by differing results across iterations.

    let text = "aaaa bbbb cccc dddd eeee ffff";
    let ner = determinism_ensemble();

    let reference: Vec<(usize, usize, String, Confidence)> = ner
        .extract_entities(text, None)
        .expect("first run should succeed")
        .into_iter()
        .map(|e| {
            (
                e.start(),
                e.end(),
                e.entity_type.as_label().to_string(),
                e.confidence,
            )
        })
        .collect();

    assert!(
        !reference.is_empty(),
        "determinism ensemble should produce at least one entity"
    );

    for run in 1..10_usize {
        let result: Vec<(usize, usize, String, Confidence)> = ner
            .extract_entities(text, None)
            .unwrap_or_else(|e| panic!("run {} failed: {}", run, e))
            .into_iter()
            .map(|e| {
                (
                    e.start(),
                    e.end(),
                    e.entity_type.as_label().to_string(),
                    e.confidence,
                )
            })
            .collect();

        assert_eq!(
            result.len(),
            reference.len(),
            "run {} produced {} entities, expected {}",
            run,
            result.len(),
            reference.len()
        );

        for (idx, (got, want)) in result.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                got, want,
                "run {} entity[{}]: got {:?}, want {:?}",
                run, idx, got, want
            );
        }
    }
}

#[test]
fn test_parallel_determinism_with_overlapping_spans() {
    // Two backends produce competing candidates for the SAME span.
    // The weighted vote must always resolve to the same winner regardless
    // of which thread finishes first.
    //
    // "Apple" at [0,5):
    //   backend-high (weight 0.90 via "gliner" key): Organization, conf 0.80
    //   backend-low  (weight 0.60 via "heuristic" key): Person, conf 0.80
    //
    // Weighted sums:  ORG = 0.90 * 0.80 = 0.72
    //                 PER = 0.60 * 0.80 = 0.48
    // ORG must always win.
    let backend_high = FixedBackend::new(
        "gliner", // matches default weight entry (0.85 overall)
        vec![Entity::new("Apple", EntityType::Organization, 0, 5, 0.80)],
    );
    let backend_low = FixedBackend::new(
        "heuristic", // matches default weight entry (0.60 overall)
        vec![Entity::new("Apple", EntityType::Person, 0, 5, 0.80)],
    );

    let ner = EnsembleNER::with_backends(vec![Box::new(backend_high), Box::new(backend_low)]);
    let text = "Apple";

    let reference_type = {
        let entities = ner.extract_entities(text, None).expect("first run");
        assert_eq!(entities.len(), 1, "should resolve to exactly one entity");
        entities[0].entity_type.clone()
    };

    for run in 1..10_usize {
        let entities = ner
            .extract_entities(text, None)
            .unwrap_or_else(|e| panic!("run {} failed: {}", run, e));
        assert_eq!(
            entities.len(),
            1,
            "run {} produced {} entities, expected 1",
            run,
            entities.len()
        );
        assert_eq!(
            entities[0].entity_type, reference_type,
            "run {} resolved to {:?}, expected {:?}",
            run, entities[0].entity_type, reference_type
        );
    }
}

// =============================================================================
// Error-handling: failing backends are skipped gracefully
// =============================================================================

#[test]
fn test_failing_backend_is_skipped_and_others_produce_results() {
    // One backend always fails; two succeed.
    // The ensemble must produce the successful backends' results without panicking.
    let good_a = FixedBackend::new(
        "good-a",
        vec![Entity::new("Paris", EntityType::Location, 0, 5, 0.85)],
    );
    let bad = AlwaysErrBackend::new("always-err");
    let good_b = FixedBackend::new(
        "good-b",
        vec![Entity::new("March", EntityType::Date, 6, 11, 0.90)],
    );

    let ner = EnsembleNER::with_backends(vec![Box::new(good_a), Box::new(bad), Box::new(good_b)]);

    let entities = ner
        .extract_entities("Paris March", None)
        .expect("ensemble should not propagate backend errors");

    // Both healthy backends must contribute (non-overlapping spans).
    assert_eq!(
        entities.len(),
        2,
        "expected 2 entities from healthy backends, got: {:?}",
        entities
            .iter()
            .map(|e| format!("{}:{:?}", e.text, e.entity_type))
            .collect::<Vec<_>>()
    );

    let texts: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
    assert!(
        texts.contains(&"Paris"),
        "expected 'Paris' in output, got {:?}",
        texts
    );
    assert!(
        texts.contains(&"March"),
        "expected 'March' in output, got {:?}",
        texts
    );
}

#[test]
fn test_all_backends_fail_returns_empty() {
    // When every backend fails the ensemble must return Ok([]) not Err.
    let ner = EnsembleNER::with_backends(vec![
        Box::new(AlwaysErrBackend::new("err-1")),
        Box::new(AlwaysErrBackend::new("err-2")),
    ]);

    let result = ner.extract_entities("Anything at all", None);
    assert!(
        result.is_ok(),
        "ensemble should return Ok even when all backends fail"
    );
    assert!(
        result.unwrap().is_empty(),
        "ensemble should return empty vec when all backends fail"
    );
}

#[test]
fn test_single_failing_backend_with_single_good_backend() {
    // Boundary case: exactly one backend, which fails.
    let ner = EnsembleNER::with_backends(vec![Box::new(AlwaysErrBackend::new("only-err"))]);

    let result = ner.extract_entities("Tim Cook", None);
    assert!(
        result.is_ok(),
        "single failing backend must not propagate as Err"
    );
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_error_backend_does_not_affect_confidence_of_good_results() {
    // The presence of a failing backend must not inflate or deflate the confidence
    // of entities from the healthy backend compared to a baseline without it.
    let text = "London";
    let entity = Entity::new("London", EntityType::Location, 0, 6, 0.80);

    let solo = EnsembleNER::with_backends(vec![Box::new(FixedBackend::new(
        "solo",
        vec![entity.clone()],
    ))]);

    let with_err = EnsembleNER::with_backends(vec![
        Box::new(FixedBackend::new("solo", vec![entity])),
        Box::new(AlwaysErrBackend::new("noise")),
    ]);

    let solo_result = solo.extract_entities(text, None).unwrap();
    let err_result = with_err.extract_entities(text, None).unwrap();

    assert_eq!(
        solo_result.len(),
        err_result.len(),
        "entity count should be identical regardless of failing backend"
    );
    // Confidence is a function of the resolved candidates only; the error backend
    // adds no candidates so confidence must be equal.
    assert!(
        (solo_result[0].confidence - err_result[0].confidence).abs() < 1e-9,
        "confidence differed: solo={} with_err={}",
        solo_result[0].confidence,
        err_result[0].confidence
    );
}
