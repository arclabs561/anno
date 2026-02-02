//! Real E2E classical NER evaluation (entity-level F1).
//!
//! This is a small, bounded harness intended for validating:
//! - CRF heuristic vs CRF bundled weights
//! - HMM heuristic vs HMM bundled params
//!
//! It uses the `anno-eval` dataset loader and the legacy strict/exact/partial/type metrics.

use anno_eval::eval::loader::{DatasetLoader, LoadableDatasetId};
use anno_eval::eval::ner_metrics::evaluate_entities;
use anno_eval::backends::{CrfNER, HmmNER};
use anno_eval::{Model, Result};

fn print_row(name: &str, r: &anno_eval::eval::ner_metrics::NerEvalResults) {
    let p = r.strict.precision_exact();
    let rec = r.strict.recall_exact();
    let f1 = r.strict.f1_exact();
    println!(
        "{name:24} strict P={:.4} R={:.4} F1={:.4} (COR={} INC={} MIS={} SPU={})",
        p,
        rec,
        f1,
        r.strict.correct,
        r.strict.incorrect,
        r.strict.missed,
        r.strict.spurious
    );
}

fn eval_model<M: Model>(model: &M, ds: &anno_eval::eval::loader::LoadedDataset, max_sentences: usize) -> Result<anno_eval::eval::ner_metrics::NerEvalResults> {
    let mut out = anno_eval::eval::ner_metrics::NerEvalResults::new();
    let n = ds.sentences.len().min(max_sentences);
    for s in ds.sentences.iter().take(n) {
        let text = s.text();
        let gold = s
            .entities()
            .into_iter()
            .map(|g| anno_eval::Entity::new(g.text, g.entity_type, g.start, g.end, 1.0))
            .collect::<Vec<_>>();
        let pred = model.extract_entities(&text, None)?;
        let r = evaluate_entities(&gold, &pred);
        out.merge(&r);
    }
    Ok(out)
}

fn main() -> Result<()> {
    let max_sentences = std::env::var("ANNO_CLASSICAL_E2E_MAX_SENTENCES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let max_sentences = if max_sentences == 0 { usize::MAX } else { max_sentences };

    let dataset = std::env::var("ANNO_CLASSICAL_E2E_DATASET")
        .ok()
        .unwrap_or_else(|| "WikiGold".to_string());
    let id: LoadableDatasetId = dataset.parse()?;

    let loader = DatasetLoader::new()?;
    let ds = loader.load_or_download(id)?;
    let stats = ds.stats();
    println!("dataset={} sentences={} entities={}", stats.name, stats.sentences, stats.entities);

    // Models
    let crf_heur = CrfNER::new_heuristic();
    let crf = CrfNER::new();
    let hmm_heur = HmmNER::new_heuristic();
    let hmm = HmmNER::new();

    let r_crf_heur = eval_model(&crf_heur, &ds, max_sentences)?;
    let r_crf = eval_model(&crf, &ds, max_sentences)?;
    let r_hmm_heur = eval_model(&hmm_heur, &ds, max_sentences)?;
    let r_hmm = eval_model(&hmm, &ds, max_sentences)?;

    println!();
    print_row("crf (heuristic)", &r_crf_heur);
    print_row("crf (default)", &r_crf);
    print_row("hmm (heuristic)", &r_hmm_heur);
    print_row("hmm (default)", &r_hmm);

    // Feature visibility (audit).
    println!();
    println!(
        "features: bundled-crf-weights={} bundled-hmm-params={}",
        cfg!(feature = "bundled-crf-weights"),
        cfg!(feature = "bundled-hmm-params")
    );

    Ok(())
}

