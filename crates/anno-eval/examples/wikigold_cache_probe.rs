//! Measure one normal WikiGold load, its parsed annotations and deterministic NER score.
//! Run with an isolated ANNO_CACHE_DIR. The workflow compares three independent invocations.

use anno::core::grounded::html_source::source_sha256;
use anno_eval::eval::evaluate_ner_model;
use anno_eval::eval::loader::{DatasetId, DatasetLoader, LoadableDatasetId};
use anno_eval::HeuristicNER;
use serde_json::json;
use std::{error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let id = DatasetId::WikiGold;
    let loadable = LoadableDatasetId::try_from(id)?;
    let loader = DatasetLoader::new()?;

    let start = Instant::now();
    let dataset = loader.load_or_download(loadable)?;
    let load_us = start.elapsed().as_micros();
    let text = fs::read_to_string(loader.cache_dir().join(id.cache_filename()))?;
    let cases = dataset.to_test_cases();
    let score = evaluate_ner_model(&HeuristicNER::new(), &cases)?;
    // Ordered tokens and BIO labels cover annotation boundaries. Test cases additionally
    // cover the reconstructed sentence, gold span offsets, and normalized entity types.
    let token_labels = serde_json::to_string(&dataset.sentences)?;
    let gold_spans = serde_json::to_string(&cases)?;
    let receipt = json!({
        "source": dataset.data_source,
        "load_us": load_us,
        "dataset": {
            "sha256": source_sha256(&text),
            "bytes": text.len(),
            "stats": dataset.stats(),
            "token_labels_sha256": source_sha256(&token_labels),
            "gold_spans_sha256": source_sha256(&gold_spans),
        },
        "heuristic_score": {
            "sentences": cases.len(), "predicted": score.found, "expected": score.expected,
            "precision": score.precision, "recall": score.recall, "f1": score.f1,
        }
    });
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
