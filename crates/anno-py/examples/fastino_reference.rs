//! Emit a Rust Fastino oracle for the Python binding's cached-model test.
//!
//! Run with `ANNO_NO_DOWNLOADS=1 cargo run -p anno-py --example
//! fastino_reference --features fastino` after the pinned model is cached.

use anno::backends::gliner2_fastino::{GLiNER2Fastino, SUPPORTED_GLINER2_FASTINO_MODEL};
use anno::backends::inference::ZeroShotNER;
use serde_json::json;

const NER_TEXT: &str = "🌍 Acme Corp signed a deal with Globex in Paris.";
const NER_LABELS: &[&str] = &["organization", "location"];
const CLASSIFICATION_TEXT: &str = "This product is wonderful, I love it.";
const CLASSIFICATION_LABELS: &[&str] = &["positive", "negative", "neutral"];
const THRESHOLD: f32 = 0.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = GLiNER2Fastino::from_pretrained(SUPPORTED_GLINER2_FASTINO_MODEL)?;
    let entities = model.extract_with_types(NER_TEXT, NER_LABELS, THRESHOLD)?;
    let classifications = model.classify(CLASSIFICATION_TEXT, CLASSIFICATION_LABELS, 0.0)?;

    let oracle = json!({
        "model": SUPPORTED_GLINER2_FASTINO_MODEL,
        "ner": {
            "text": NER_TEXT,
            "labels": NER_LABELS,
            "threshold": THRESHOLD,
            "entities": entities.into_iter().map(|entity| json!({
                "text": entity.text,
                "label": entity.entity_type.to_string(),
                "start": entity.start(),
                "end": entity.end(),
                "confidence": entity.confidence.value(),
            })).collect::<Vec<_>>(),
        },
        "classification": {
            "text": CLASSIFICATION_TEXT,
            "labels": CLASSIFICATION_LABELS,
            "results": classifications.into_iter().map(|(label, probability)| json!({
                "label": label,
                "probability": probability,
            })).collect::<Vec<_>>(),
        },
    });

    println!("{}", serde_json::to_string(&oracle)?);
    Ok(())
}
