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
