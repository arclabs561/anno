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
    let arg = std::env::args().nth(1).ok_or(
        "expected --probe, --default-no-bert, --default-with-bert, or a local checkpoint directory",
    )?;
    let probe = arg == "--probe";
    let default_stack = matches!(arg.as_str(), "--default-no-bert" | "--default-with-bert");
    let second = std::env::args().nth(2);
    let opt_in = default_stack && second.as_deref() == Some("--nuner-fallback");
    if default_stack && second.is_some() && !opt_in {
        return Err("default stack accepts only --nuner-fallback".into());
    }
    let bert = second.filter(|_| !default_stack);
    let expect_bert = arg == "--default-with-bert" || bert.is_some();
    let model = if default_stack {
        // The CLI Auto and Stacked routes use this constructor. The replay controls
        // model availability with an isolated, offline Hugging Face cache.
        let model = StackedNER::default();
        if opt_in {
            model.with_nuner_fallback()
        } else {
            model
        }
    } else {
        let nuner = if probe {
            NuNER::new()
        } else {
            NuNER::from_pretrained(&arg)?
        };
        let mut builder = StackedNER::builder();
        if let Some(ref path) = bert {
            builder = builder.layer(BertNEROnnx::new(path)?);
        }
        builder
            .layer(nuner.with_threshold(0.9))
            .layer(RegexNER::new())
            .layer(HeuristicNER::new())
            .build()
    };
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
        if expect_bert
            && !matches!(report.layer_outcomes().first(), Some(StackedLayerOutcome::Succeeded { layer, .. }) if layer == "bert-onnx")
        {
            return Err("BERT did not successfully process this input".into());
        }
        let ran = match report.layer_outcomes().get(usize::from(expect_bert)) {
            Some(StackedLayerOutcome::Succeeded { layer, .. }) if layer == "nuner" => true,
            Some(StackedLayerOutcome::Skipped { layer, .. }) if layer == "nuner" => false,
            Some(StackedLayerOutcome::Failed { layer, .. }) if probe && layer == "nuner" => true,
            _ => return Err("unexpected NuNER layer outcome".into()),
        };
        let layers: Vec<_> = report
            .layer_outcomes()
            .iter()
            .map(|outcome| match outcome {
                StackedLayerOutcome::Succeeded { layer, .. } => (layer.as_str(), "succeeded"),
                StackedLayerOutcome::Skipped { layer, .. } => (layer.as_str(), "skipped"),
                StackedLayerOutcome::Failed { layer, .. } => (layer.as_str(), "failed"),
                _ => ("unknown", "unknown"),
            })
            .collect();
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
            &json!({"input_sha256": source_sha256(&text), "ran": ran, "layers": layers, "spans": spans, "seconds": seconds}),
        )?;
        writeln!(&mut out)?;
    }
    out.flush()?;
    Ok(())
}
