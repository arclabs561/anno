#![cfg(feature = "onnx")]

use anno::{FCoref, FCorefConfig};

#[test]
fn resolve_retains_the_final_segment_mention() {
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fcoref-marker");
    // Downstream crates can still construct all five fields explicitly.
    let config = FCorefConfig {
        max_span_length: 1,
        top_lambda: 0.0025,
        max_segment_len: 512,
        optimization_level: 3,
        num_threads: 4,
    };
    let mut words = vec!["x"; 1024];
    for index in [0, 509, 1023] {
        words[index] = "Zoë";
    }
    let text = words.join(" ");
    let default_budget = FCoref::from_path_with_config(model_path, config.clone()).unwrap();
    let error = default_budget.resolve(&text).unwrap_err().to_string();
    assert!(
        error.contains("1026 tokens, exceeding max_doc_tokens 512"),
        "{error}"
    );

    let model = FCoref::from_path_with_document_budget(model_path, config, Some(1026)).unwrap();
    let clusters = model.resolve(&text).unwrap();
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].mentions, ["Zoë", "Zoë", "Zoë"]);
    assert_eq!(clusters[0].spans, [(0, 3), (1020, 1023), (2050, 2053)]);
}
