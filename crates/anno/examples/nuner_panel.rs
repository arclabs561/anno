//! Run a labeled-prompt NER panel through one public NuNER session. Inputs stay on stdin.
use anno::{core::grounded::html_source::source_sha256, Model, NuNER};
use serde::Deserialize;
use serde_json::json;
use std::{error::Error, io, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    texts: Vec<String>,
    labels: Vec<String>,
    threshold: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let checkpoint = std::env::args()
        .nth(1)
        .ok_or("expected a checkpoint directory")?;
    let panel: Request = serde_json::from_reader(io::stdin().lock())?;
    if panel.labels.is_empty() || !(0.0..=1.0).contains(&panel.threshold) {
        return Err("expected a nonempty label list and threshold in [0, 1]".into());
    }
    let labels: Vec<_> = panel.labels.iter().map(String::as_str).collect();
    let model = NuNER::from_pretrained(&checkpoint)?;
    let ner = model
        .as_zero_shot()
        .ok_or("model lacks zero-shot extraction")?;
    let mut rows = Vec::with_capacity(panel.texts.len());
    for text in &panel.texts {
        let before = Instant::now();
        let entities = ner.extract_with_types(text, &labels, panel.threshold)?;
        let elapsed = before.elapsed().as_secs_f64();
        let entities: Vec<_> = entities.into_iter()
            .filter(|e| e.confidence.value() >= f64::from(panel.threshold))
            .map(|e| json!({"type": e.entity_type.as_label(), "text": e.text, "start": e.start(), "end": e.end()}))
            .collect();
        rows.push(json!({"source_sha256": source_sha256(text),
                         "entities": entities, "seconds": elapsed}));
    }
    serde_json::to_writer(
        io::stdout().lock(),
        &json!({
            "model": model.name(), "checkpoint": checkpoint, "labels": panel.labels,
            "threshold": panel.threshold, "rows": rows,
        }),
    )?;
    Ok(())
}
