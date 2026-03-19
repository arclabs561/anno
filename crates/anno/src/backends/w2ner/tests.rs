use super::*;
use crate::backends::inference::HandshakingCell;
use crate::EntityCategory;

#[test]
fn test_w2ner_relation_conversion() {
    assert_eq!(W2NERRelation::from_index(0), W2NERRelation::None);
    assert_eq!(W2NERRelation::from_index(1), W2NERRelation::NNW);
    assert_eq!(W2NERRelation::from_index(2), W2NERRelation::THW);

    assert_eq!(W2NERRelation::None.to_index(), 0);
    assert_eq!(W2NERRelation::NNW.to_index(), 1);
    assert_eq!(W2NERRelation::THW.to_index(), 2);
}

#[test]
fn test_w2ner_config_defaults() {
    let config = W2NERConfig::default();
    assert!((config.threshold.value() - 0.5).abs() < f64::EPSILON);
    assert!(config.allow_nested);
    assert!(config.allow_discontinuous);
    assert_eq!(config.entity_labels.len(), 3);
}

#[test]
fn test_decode_simple_entity() {
    let w2ner = W2NER::new();
    let tokens = ["New", "York", "City"];

    // THW marker: tail=2, head=0 (entity spans all 3 tokens)
    let matrix = HandshakingMatrix {
        cells: vec![HandshakingCell {
            i: 2, // tail
            j: 0, // head
            label_idx: W2NERRelation::THW.to_index() as u16,
            score: 0.9,
        }],
        seq_len: 3,
        num_labels: 3,
    };

    let entities = w2ner.decode_from_matrix(&matrix, &tokens, 0);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].0, 0); // start
    assert_eq!(entities[0].1, 3); // end
}

#[test]
fn test_decode_nested_entities() {
    let w2ner = W2NER::with_config(W2NERConfig {
        allow_nested: true,
        ..Default::default()
    });

    let tokens = ["University", "of", "California", "Berkeley"];

    let matrix = HandshakingMatrix {
        cells: vec![
            // Full entity: tail=3, head=0
            HandshakingCell {
                i: 3,
                j: 0,
                label_idx: W2NERRelation::THW.to_index() as u16,
                score: 0.95,
            },
            // Nested: tail=2, head=2 (just "California")
            HandshakingCell {
                i: 2,
                j: 2,
                label_idx: W2NERRelation::THW.to_index() as u16,
                score: 0.85,
            },
        ],
        seq_len: 4,
        num_labels: 3,
    };

    let entities = w2ner.decode_from_matrix(&matrix, &tokens, 0);
    assert_eq!(entities.len(), 2);
}

#[test]
fn test_remove_nested() {
    let entities = vec![
        (0, 4, 0.9), // outer
        (2, 3, 0.8), // nested
    ];

    let filtered = decode::remove_nested(&entities);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0], (0, 4, 0.9));
}

#[test]
fn test_grid_to_matrix() {
    // 3x3 grid with 3 relations (None, NNW, THW)
    let seq_len = 3;
    let num_rels = 3;
    let mut grid = vec![0.0f32; seq_len * seq_len * num_rels];

    // Set THW at (2, 0) with score 0.9
    // Index formula: i * seq_len * num_rels + j * num_rels + rel_idx
    let i = 2;
    let j = 0;
    let rel_thw = 2;
    let idx = i * seq_len * num_rels + j * num_rels + rel_thw;
    grid[idx] = 0.9;

    let matrix = W2NER::grid_to_matrix(&grid, seq_len, num_rels, 0.5);
    assert_eq!(matrix.cells.len(), 1);
    assert_eq!(matrix.cells[0].i, 2);
    assert_eq!(matrix.cells[0].j, 0);
}

#[test]
fn test_label_mapping() {
    assert_eq!(decode::map_label_to_entity_type("PER"), EntityType::Person);
    assert_eq!(
        decode::map_label_to_entity_type("org"),
        EntityType::Organization
    );
    assert_eq!(
        decode::map_label_to_entity_type("GPE"),
        EntityType::Location
    );
    assert_eq!(
        decode::map_label_to_entity_type("CUSTOM"),
        EntityType::custom("CUSTOM", EntityCategory::Misc)
    );
}

#[test]
fn test_empty_input() {
    let w2ner = W2NER::new();
    let entities = w2ner.extract_entities("", None).unwrap();
    assert!(entities.is_empty());
}

#[test]
fn test_not_available_without_model() {
    let w2ner = W2NER::new();
    // Without model loaded, should not be available
    assert!(!w2ner.is_available());
}

// ─────────────────────────────────────────────────────────────────────────
// decode_discontinuous_from_matrix tests
// ─────────────────────────────────────────────────────────────────────────

fn make_cell(i: u32, j: u32, label: u16, score: f32) -> HandshakingCell {
    HandshakingCell {
        i,
        j,
        label_idx: label,
        score,
    }
}

/// Build a minimal HandshakingMatrix from explicit cells.
fn mat(cells: Vec<HandshakingCell>, seq_len: usize) -> HandshakingMatrix {
    HandshakingMatrix {
        cells,
        seq_len,
        num_labels: 3,
    }
}

const NNW: u16 = 1;
const THW: u16 = 2;

#[test]
fn discontinuous_contiguous_entity_three_words() {
    // "New York City" — NNW(0,1), NNW(1,2), THW(tail=2, head=0)
    let w2ner = W2NER::new();
    let tokens = ["New", "York", "City"];
    let matrix = mat(
        vec![
            make_cell(0, 1, NNW, 0.9), // New-York NNW
            make_cell(1, 2, NNW, 0.9), // York-City NNW
            make_cell(2, 0, THW, 0.9), // tail=City, head=New
        ],
        3,
    );
    let result = w2ner.decode_discontinuous_from_matrix(&matrix, &tokens, 0.5);
    assert_eq!(result.len(), 1, "should find exactly one entity");
    let (_, spans, _) = &result[0];
    assert_eq!(
        spans.len(),
        1,
        "all three words are adjacent → one contiguous span"
    );
    assert_eq!(spans[0], (0, 3)); // words 0..3
}

#[test]
fn discontinuous_two_part_entity() {
    // "severe ... pain" — tokens = ["severe", "and", "moderate", "pain"]
    // Entity = (severe, pain) but "and moderate" is NOT part of it.
    // NNW between "severe" and "pain" missing → two segments.
    // THW(tail=3, head=0): entity boundary severe..pain
    // NNW(0,1) absent, NNW(1,2) absent, NNW(2,3) present (moderate-pain NNW)
    // Gap at (0,1): severe | and → segment 1 = (0,1)
    // Gap at (1,2): and | moderate → not needed because we only look at head..tail
    // Actually we look from head=0 to tail=3:
    //   i=0: NNW(0,1)? no → gap → segment (0,1)
    //   i=1: NNW(1,2)? no → gap → segment (1,2)
    //   i=2: NNW(2,3)? yes → no gap
    // → segments: (0,1), (1,2), (2,4)
    // That's 3 segments for "severe", "and", "moderate pain".
    //
    // For a cleaner two-segment test:
    // tokens = ["severe", "pain"]; THW(1,0), no NNW
    let w2ner = W2NER::new();
    let tokens = ["severe", "pain"];
    let matrix = mat(
        vec![
            make_cell(1, 0, THW, 0.9), // tail=pain, head=severe
                                       // No NNW(0,1) → gap between severe and pain
        ],
        2,
    );
    let result = w2ner.decode_discontinuous_from_matrix(&matrix, &tokens, 0.5);
    assert_eq!(result.len(), 1);
    let (_, spans, _) = &result[0];
    assert_eq!(
        spans.len(),
        2,
        "missing NNW should produce 2 disjoint segments"
    );
    assert_eq!(spans[0], (0, 1)); // "severe" alone
    assert_eq!(spans[1], (1, 2)); // "pain" alone
}

#[test]
fn discontinuous_empty_matrix() {
    let w2ner = W2NER::new();
    let tokens = ["a", "b", "c"];
    let matrix = mat(vec![], 3);
    let result = w2ner.decode_discontinuous_from_matrix(&matrix, &tokens, 0.5);
    assert!(result.is_empty(), "no cells → no entities");
}

#[test]
fn discontinuous_multiple_entities() {
    // "Google Apple" — two single-word entities
    // THW(0,0): tail=Google, head=Google → entity (0,0)
    // THW(1,1): tail=Apple, head=Apple → entity (1,1)
    let w2ner = W2NER::new();
    let tokens = ["Google", "Apple"];
    let matrix = mat(
        vec![
            make_cell(0, 0, THW, 0.9), // single-word entity Google
            make_cell(1, 1, THW, 0.9), // single-word entity Apple
        ],
        2,
    );
    let result = w2ner.decode_discontinuous_from_matrix(&matrix, &tokens, 0.5);
    assert_eq!(result.len(), 2, "two entities");
    // Each should have one segment of length 1
    for (_, spans, _) in &result {
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].1 - spans[0].0, 1);
    }
}

#[test]
fn discontinuous_threshold_filters_low_score() {
    let w2ner = W2NER::new();
    let tokens = ["New", "York"];
    let matrix = mat(
        vec![
            make_cell(1, 0, THW, 0.3), // score 0.3 < threshold 0.5
        ],
        2,
    );
    let result = w2ner.decode_discontinuous_from_matrix(&matrix, &tokens, 0.5);
    assert!(
        result.is_empty(),
        "low-score THW should be filtered by threshold"
    );
}

#[test]
fn test_errors_without_model() {
    let w2ner = W2NER::new();
    // Without model, should return an explicit error (no silent empty fallback).
    let err = w2ner
        .extract_entities("Steve Jobs founded Apple", None)
        .unwrap_err();
    assert!(
        matches!(
            err,
            crate::Error::ModelInit(_) | crate::Error::FeatureNotAvailable(_)
        ),
        "unexpected error: {:?}",
        err
    );
}
