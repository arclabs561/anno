//! Zero-shot NER span-F1 on a real dataset, through anno-eval's own harness.
//!
//! Registers GLiNER as a backend in `EvalHarness` and runs it on the
//! auto-downloaded CoNLL-2003 sample via `run_real_datasets`. The harness owns
//! dataset loading, label passing, and SemEval span-F1 scoring, so this example
//! is just wiring: pick a backend, pick a dataset, print the metrics.
//!
//! ```sh
//! cargo run --release --features "onnx eval" --example wnut_ner_f1
//! ```
//! First run downloads the GLiNER model (cached after) and the dataset sample.
//!
//! The chunking offset-inversion panic this used to hit is fixed (see
//! `Entity::shift_by`). Separately, the `CoNLL2003Sample` loader currently
//! returns one giant document with 0 gold entities (it appears to run the CoNLL
//! parser over HF-JSON rows), so the F1 here reads 0 until that loader defect is
//! fixed. The wiring (register backend, run dataset, score) is correct and will
//! report a real F1 once a dataset loads gold spans.

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("needs the `onnx` feature: cargo run --features onnx --example wnut_ner_f1");
}

#[cfg(feature = "onnx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use anno::GLiNEROnnx;
    use anno_eval::eval::harness::{EvalConfig, EvalHarness};
    use anno_eval::eval::loader::DatasetId;

    let mut harness = EvalHarness::new(EvalConfig::default())?;
    harness.register(
        "GLiNER",
        "zero-shot GLiNER small v2.1",
        Box::new(GLiNEROnnx::new("onnx-community/gliner_small-v2.1")?),
    );

    let results = harness.run_real_datasets(&[DatasetId::CoNLL2003Sample])?;

    println!("\nNER span-F1 (SemEval micro, via EvalHarness):");
    for backend in &results.backends {
        for d in &backend.per_dataset {
            println!(
                "  {} on {}: P {:.3}  R {:.3}  F1 {:.3}  ({} examples, {} gold entities)",
                d.backend_name, d.dataset_name, d.precision, d.recall, d.f1, d.num_examples, d.num_gold_entities,
            );
        }
    }
    Ok(())
}
