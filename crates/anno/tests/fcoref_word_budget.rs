#![cfg(feature = "onnx")]

use anno::{FCoref, FCorefConfig};
use regex::Regex;

#[test]
#[ignore = "requires the pinned FP32 biu-nlp/f-coref export in FCOREF_MODEL_DIR"]
fn reference_word_mentions_cross_the_encoder_boundary() {
    let model_dir = std::env::var("FCOREF_MODEL_DIR")
        .unwrap_or_else(|_| format!("{}/fcoref_onnx", env!("CARGO_MANIFEST_DIR")));
    let text = "x ".repeat(502)
        + "Alice reviewed the proposal. She approved it. Bob objected. She overruled him.";
    let words: Vec<_> = Regex::new(r"\w+|[^\s\w]")
        .unwrap()
        .find_iter(&text)
        .map(|m| (m.start(), m.end()))
        .collect();
    // FastCoref at ae224139, FP32 biu-nlp/f-coref checkpoint e91dfff1.
    // The recorded 521-token reference and export procedure are linked in PR #150.
    let reference: Vec<(usize, usize, f32)> =
        serde_json::from_str(include_str!("fixtures/fcoref_word_budget.json")).unwrap();

    for budget in [Some(520), Some(521), None] {
        let model = FCoref::from_path_with_document_budget(
            &model_dir,
            FCorefConfig {
                num_threads: 2,
                ..Default::default()
            },
            budget,
        )
        .unwrap();
        let result = model.analyze_words(&text, &words);
        if budget == Some(520) {
            let error = result.unwrap_err().to_string();
            assert!(
                error.contains("521 tokens") && error.contains("520"),
                "{error}"
            );
            continue;
        }
        let result = result.unwrap();
        assert_eq!(result.mentions.len(), reference.len());
        for (mention, &(start, end, logit)) in result.mentions.iter().zip(&reference) {
            assert_eq!(
                (mention.character_start, mention.character_end),
                (start, end)
            );
            // This fixture is ASCII, so scalar positions also index the source bytes.
            assert_eq!(mention.text, text[start..end]);
            assert!(
                (mention.raw_mention_score - logit).abs() < 0.01,
                "{mention:?}"
            );
        }
        let clusters: Vec<_> = result.clusters.into_iter().map(|c| c.spans).collect();
        assert_eq!(
            clusters,
            [
                vec![(1004, 1009), (1033, 1036), (1064, 1067)],
                vec![(1019, 1031), (1046, 1048)],
                vec![(1050, 1053), (1078, 1081)],
            ]
        );
    }
}
