//! Deterministic muxer decision-loop smoke example (offline).
//!
//! This example demonstrates a realistic local workflow without dataset/model downloads:
//! 1) choose an arm with muxer MAB on dataset-scoped history,
//! 2) write synthetic outcome rows compatible with `muxer_agg_lib`,
//! 3) aggregate and print a compact report.
//!
//! Run:
//! `cargo run -p anno-eval --example muxer_decision_loop --features eval`

use anno_eval::eval::loader::DatasetId;
use anno_eval::muxer_history::BackendHistory;
use muxer::{MabConfig, Outcome};

fn simulated_outcome(backend: &str, round: usize) -> (Outcome, Option<&'static str>) {
    match backend {
        // Usually strong, occasional low-signal run.
        "stacked" => {
            let soft_junk = round.is_multiple_of(7);
            let quality = if soft_junk { 0.12 } else { 0.86 };
            (
                Outcome::with_quality(!soft_junk, soft_junk, false, 20, 650, quality),
                if soft_junk { Some("low_signal") } else { None },
            )
        }
        // Mid-quality baseline with frequent soft junk.
        "heuristic" => {
            let soft_junk = round.is_multiple_of(2);
            let quality = if soft_junk { 0.18 } else { 0.55 };
            (
                Outcome::with_quality(!soft_junk, soft_junk, false, 8, 180, quality),
                if soft_junk { Some("low_signal") } else { None },
            )
        }
        // Low-latency structured backend with occasional hard failure.
        _ => {
            let hard_fail = round.is_multiple_of(3);
            (
                Outcome::with_quality(
                    !hard_fail,
                    hard_fail,
                    hard_fail,
                    4,
                    90,
                    if hard_fail { 0.0 } else { 0.35 },
                ),
                if hard_fail { Some("timeout") } else { None },
            )
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = DatasetId::Wnut17;
    let arms = vec![
        "stacked".to_string(),
        "heuristic".to_string(),
        "pattern".to_string(),
    ];

    let mut history = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: Default::default(),
        fail_kinds: Default::default(),
        exp3ix_state: None,
        linucb_state: None,
    };

    let mut cfg = MabConfig {
        exploration_c: 0.7,
        ..MabConfig::default()
    };
    cfg.set_weight(muxer::Extract::SoftJunkRate, 0.8);
    cfg.set_weight(muxer::Extract::HardJunkRate, 1.6);

    let out_dir = std::env::temp_dir().join(format!("anno-muxer-example-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir)?;
    let jsonl_path = out_dir.join("muxer_outcomes.jsonl");

    let mut jsonl = String::new();
    for round in 0..12usize {
        let summaries = history.summaries_for(None, &arms, Some(&[dataset]), true, 0);
        let decision = muxer::select_mab_explain(&arms, &summaries, cfg.clone());
        let chosen = decision.selection.chosen.clone();

        let (outcome, fail_kind_raw) = simulated_outcome(&chosen, round);
        let fail_kind = fail_kind_raw.map(str::to_string);

        history.push_with_fail_kind(&chosen, outcome, fail_kind.clone());
        history.push_with_fail_kind(
            &BackendHistory::dataset_key(&chosen, dataset),
            outcome,
            fail_kind.clone(),
        );

        let row = serde_json::json!({
            "schema_version": 1,
            "record_type": "outcome",
            "run_id": "example",
            "strategy": "ml-only",
            "slice": "ner.lang=en.dom=social_media",
            "dataset": format!("{dataset:?}"),
            "backend": chosen,
            "ok": outcome.ok,
            "junk": outcome.junk,
            "hard_junk": outcome.hard_junk,
            "fail_kind": fail_kind,
            "primary_f1": outcome.quality_score.unwrap_or(0.0),
            "elapsed_ms": outcome.elapsed_ms,
            "cost_units": outcome.cost_units
        });
        jsonl.push_str(&serde_json::to_string(&row)?);
        jsonl.push('\n');
    }
    std::fs::write(&jsonl_path, jsonl)?;

    let agg = anno_eval::muxer_agg_lib::aggregate_jsonl_paths(&[jsonl_path.as_path()])
        .map_err(|e| format!("aggregate failed: {e}"))?;

    let groups = agg
        .get("groups")
        .and_then(|v| v.as_array())
        .ok_or("aggregate output missing groups")?;
    if groups.is_empty() {
        return Err("expected at least one aggregated group".into());
    }

    let lines_outcome = agg
        .get("lines_outcome")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    println!(
        "wrote {lines_outcome} synthetic outcomes to {}",
        jsonl_path.display()
    );
    println!("aggregated groups: {}", groups.len());

    for arm in &arms {
        if let Some(top) = history.chosen_fail_kinds_top_for(arm, Some(&[dataset]), true, 2) {
            let kinds = top
                .iter()
                .map(|r| format!("{}={}", r.kind, r.count))
                .collect::<Vec<_>>()
                .join(", ");
            println!("arm={arm:<10} fail_kinds={kinds}");
        } else {
            println!("arm={arm:<10} fail_kinds=-");
        }
    }

    Ok(())
}
