#![cfg(feature = "onnx")]

use anno::{FCoref, FCorefConfig};

#[test]
fn resolve_retains_the_final_segment_mention() {
    let model = FCoref::from_path_with_config(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fcoref-marker"),
        FCorefConfig {
            max_doc_tokens: Some(1026),
            max_span_length: 1,
            top_lambda: 0.0025, // Keep three candidates out of 1026 tokens.
            ..Default::default()
        },
    )
    .unwrap();
    let mut words = vec!["x"; 1024];
    for index in [0, 509, 1023] {
        words[index] = "Zoë";
    }
    let clusters = model.resolve(&words.join(" ")).unwrap();
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].mentions, ["Zoë", "Zoë", "Zoë"]);
    assert_eq!(clusters[0].spans, [(0, 3), (1020, 1023), (2050, 2053)]);
}

#[test]
fn word_analysis_obeys_the_document_budget_across_segments() {
    let text = "Zoë x Zoë x Zoë";
    let words = [(0, 4), (5, 6), (7, 11), (12, 13), (14, 18)];
    let load = |budget| {
        FCoref::from_path_with_config(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fcoref-marker"),
            FCorefConfig {
                max_doc_tokens: Some(budget),
                max_segment_len: 3,
                max_span_length: 1,
                top_lambda: 0.5,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let error = load(6).analyze_words(text, &words).unwrap_err().to_string();
    assert!(
        error.contains("7 tokens, exceeding max_doc_tokens 6"),
        "{error}"
    );
    let analysis = load(7).analyze_words(text, &words).unwrap();
    assert_eq!(analysis.mentions.len(), 3);
    assert_eq!(analysis.clusters.len(), 1);
    assert_eq!(analysis.clusters[0].spans, [(0, 3), (6, 9), (12, 15)]);
}
