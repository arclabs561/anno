#![cfg(feature = "language")]
use anyhow::{Result, ensure};
use aurelius_lanes::{
    contracts::{Document, DocumentKey},
    language::{LanguageConfig, LanguageRuntime},
};

fn document(text: &str) -> Document {
    Document {
        key: DocumentKey {
            scope: "fixture-org".into(),
            source: "fixture-source".into(),
            revision: "fixture-revision".into(),
        },
        text: text.into(),
        context: None,
        metadata: Default::default(),
    }
}
#[test]
fn configured_document_budget_reaches_word_scoring() -> Result<()> {
    let model_dir = std::env::var("AURELIUS_TEST_LANGUAGE_MODEL_DIR")?;
    let text = "x ".repeat(502)
        + "Alice reviewed the proposal. She approved it. Bob objected. She overruled him.";
    let source = document(&text);
    let runtime = LanguageRuntime::load(LanguageConfig {
        model_dir: model_dir.clone(),
        max_document_tokens: 521,
        threads: 2,
    })?;
    let result = runtime.analyze(&source)?;
    result.validate(&source)?;
    let mut clusters = Vec::new();
    for cluster in &result.coreference_hypotheses {
        let mut names = Vec::new();
        for id in cluster {
            let mention = result.mentions.iter().find(|m| &m.id == id).unwrap();
            names.push(mention.text.as_str());
        }
        names.sort_unstable();
        clusters.push(names);
    }
    clusters.sort_unstable();
    assert_eq!(
        clusters,
        [
            vec!["Alice", "She", "She"],
            vec!["Bob", "him"],
            vec!["it", "the proposal"],
        ]
    );
    drop(runtime);
    let runtime = LanguageRuntime::load(LanguageConfig {
        model_dir,
        max_document_tokens: 520,
        threads: 2,
    })?;
    ensure!(runtime.analyze(&source).is_err());
    Ok(())
}
