//! Replay the default NuNER fallback, optionally preceded by BERT, on JSON strings.
//! `--probe` records only scheduling. A local NuNER checkpoint runs actual inference.
use anno::backends::onnx::BertNEROnnx;
use anno::backends::stacked::{StackedExtractionPolicy, StackedLayerOutcome};
use anno::backends::{HeuristicNER, RegexNER, StackedNER};
use anno::core::grounded::html_source::source_sha256;
use anno::NuNER;
use serde_json::json;
use std::error::Error;
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let arg = std::env::args()
        .nth(1)
        .ok_or("expected --probe or a local checkpoint directory")?;
    let probe = arg == "--probe";
    let nuner = if probe {
        NuNER::new()
    } else {
        NuNER::from_pretrained(&arg)?
    };
    let bert = std::env::args().nth(2);
    let mut builder = StackedNER::builder();
    if let Some(ref path) = bert {
        builder = builder.layer(BertNEROnnx::new(path)?);
    }
    let model = builder
        .layer(nuner.with_threshold(0.9))
        .layer(RegexNER::new())
        .layer(HeuristicNER::new())
        .build();
    let policy = if probe {
        StackedExtractionPolicy::BestEffort
    } else {
        StackedExtractionPolicy::Strict
    };
    if !probe {
        // Below the routing length cutoff in both builds. Model startup is outside the timings.
        model.extract_entities_with_report("Rosa in Lisbon.", None, policy)?;
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let text: String = serde_json::from_str(&line?)?;
        let start = Instant::now();
        let report = model.extract_entities_with_report(&text, None, policy)?;
        let seconds = start.elapsed().as_secs_f64();
        if bert.is_some()
            && !matches!(report.layer_outcomes().first(), Some(StackedLayerOutcome::Succeeded { layer, .. }) if layer == "bert-onnx")
        {
            return Err("BERT did not successfully process this input".into());
        }
        let ran = match report.layer_outcomes().get(usize::from(bert.is_some())) {
            Some(StackedLayerOutcome::Succeeded { layer, .. }) if layer == "nuner" => true,
            Some(StackedLayerOutcome::Skipped { layer, .. }) if layer == "nuner" => false,
            Some(StackedLayerOutcome::Failed { layer, .. }) if probe && layer == "nuner" => true,
            _ => return Err("unexpected NuNER layer outcome".into()),
        };
        let mut spans = Vec::new();
        if !probe {
            let chars: Vec<_> = text.chars().collect();
            for entity in report
                .entities()
                .iter()
                .filter(|e| e.confidence.value() >= 0.5)
            {
                let span = chars
                    .get(entity.start()..entity.end())
                    .ok_or("invalid character span")?;
                if span.is_empty() || span.iter().collect::<String>() != entity.text {
                    return Err("span differs from the original input".into());
                }
                spans.push((entity.start(), entity.end(), entity.entity_type.as_label()));
            }
        }
        serde_json::to_writer(
            &mut out,
            &json!({"input_sha256": source_sha256(&text), "ran": ran, "spans": spans, "seconds": seconds}),
        )?;
        writeln!(&mut out)?;
    }
    out.flush()?;
    Ok(())
}
