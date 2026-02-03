//! Shared aggregation logic for muxer JSONL logs.
//!
//! This is used by:
//! - `bin/muxer_agg.rs` (CLI-ish tool)
//! - `bin/muxer_repeat.rs` (repeat runner)

use crate::eval::loader::DatasetId;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, serde::Serialize)]
struct GroupRow {
    task: String,
    lang: String,
    domain: String,
    backend: String,
    n: u64,
    ok: u64,
    junk: u64,
    hard_junk: u64,
    mean_primary_f1: Option<f64>,
    std_primary_f1: Option<f64>,
    stderr_primary_f1: Option<f64>,
    mean_elapsed_ms: Option<f64>,
    std_elapsed_ms: Option<f64>,
    stderr_elapsed_ms: Option<f64>,
    mean_cost_units: Option<f64>,
    std_cost_units: Option<f64>,
    stderr_cost_units: Option<f64>,
}

#[derive(Debug, Default, Clone)]
struct Agg {
    n: u64,
    ok: u64,
    junk: u64,
    hard_junk: u64,
    sum_f1: f64,
    sum_f1_sq: f64,
    n_f1: u64,
    sum_ms: f64,
    sum_ms_sq: f64,
    n_ms: u64,
    sum_cost: f64,
    sum_cost_sq: f64,
    n_cost: u64,
}

fn mean_std(sum: f64, sum_sq: f64, n: u64) -> Option<(f64, f64)> {
    if n == 0 {
        return None;
    }
    let n_f = n as f64;
    let mean = sum / n_f;
    // Population variance; this is a lightweight diagnostic, not a statistical CI.
    let var = (sum_sq / n_f) - mean * mean;
    let std = var.max(0.0).sqrt();
    Some((mean, std))
}

fn stderr(std: f64, n: u64) -> Option<f64> {
    if n == 0 {
        return None;
    }
    Some(std / (n as f64).sqrt())
}

fn parse_task_from_slice(slice: &str) -> String {
    slice.split('.').next().unwrap_or(slice).to_string()
}

/// Aggregate one or more muxer JSONL logs.
pub fn aggregate_jsonl_paths(paths: &[impl AsRef<Path>]) -> Result<serde_json::Value, String> {
    let mut groups: BTreeMap<(String, String, String, String), Agg> = BTreeMap::new();
    let mut lines_total: u64 = 0;
    let mut lines_parsed: u64 = 0;
    let mut lines_outcome: u64 = 0;

    let inputs: Vec<String> = paths
        .iter()
        .map(|p| p.as_ref().to_string_lossy().to_string())
        .collect();

    for p in paths {
        let content = std::fs::read_to_string(p.as_ref()).map_err(|e| e.to_string())?;
        for line in content.lines() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            lines_total += 1;
            let v: serde_json::Value = match serde_json::from_str(t) {
                Ok(v) => v,
                Err(_) => continue,
            };
            lines_parsed += 1;
            let record_type = v.get("record_type").and_then(|x| x.as_str());
            if record_type != Some("outcome") {
                continue;
            }
            lines_outcome += 1;

            let slice = v.get("slice").and_then(|x| x.as_str()).unwrap_or("");
            let task = parse_task_from_slice(slice);
            let backend = v
                .get("backend")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let dataset_s = v.get("dataset").and_then(|x| x.as_str()).unwrap_or("");
            let (lang, domain) = match dataset_s.parse::<DatasetId>() {
                Ok(ds) => (ds.language().to_string(), ds.domain().to_string()),
                Err(_) => ("unknown".to_string(), "unknown".to_string()),
            };

            let ok = v.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
            let junk = v.get("junk").and_then(|x| x.as_bool()).unwrap_or(false);
            let hard_junk = v
                .get("hard_junk")
                .and_then(|x| x.as_bool())
                .unwrap_or(false);

            let f1 = v.get("primary_f1").and_then(|x| x.as_f64());
            let elapsed_ms = v
                .get("elapsed_ms")
                .and_then(|x| x.as_u64())
                .map(|x| x as f64);
            let cost_units = v
                .get("cost_units")
                .and_then(|x| x.as_u64())
                .map(|x| x as f64);

            let k = (task, lang, domain, backend);
            let a = groups.entry(k).or_default();
            a.n += 1;
            if ok {
                a.ok += 1;
            }
            if junk {
                a.junk += 1;
            }
            if hard_junk {
                a.hard_junk += 1;
            }
            if let Some(f1) = f1 {
                a.sum_f1 += f1;
                a.sum_f1_sq += f1 * f1;
                a.n_f1 += 1;
            }
            if let Some(ms) = elapsed_ms {
                a.sum_ms += ms;
                a.sum_ms_sq += ms * ms;
                a.n_ms += 1;
            }
            if let Some(c) = cost_units {
                a.sum_cost += c;
                a.sum_cost_sq += c * c;
                a.n_cost += 1;
            }
        }
    }

    let rows: Vec<GroupRow> = groups
        .into_iter()
        .map(|((task, lang, domain, backend), a)| {
            let ms = mean_std(a.sum_ms, a.sum_ms_sq, a.n_ms);
            let f1 = mean_std(a.sum_f1, a.sum_f1_sq, a.n_f1);
            let cost = mean_std(a.sum_cost, a.sum_cost_sq, a.n_cost);
            GroupRow {
                task,
                lang,
                domain,
                backend,
                n: a.n,
                ok: a.ok,
                junk: a.junk,
                hard_junk: a.hard_junk,
                mean_primary_f1: f1.map(|(m, _)| m),
                std_primary_f1: f1.map(|(_, s)| s),
                stderr_primary_f1: f1.and_then(|(_, s)| stderr(s, a.n_f1)),
                mean_elapsed_ms: ms.map(|(m, _)| m),
                std_elapsed_ms: ms.map(|(_, s)| s),
                stderr_elapsed_ms: ms.and_then(|(_, s)| stderr(s, a.n_ms)),
                mean_cost_units: cost.map(|(m, _)| m),
                std_cost_units: cost.map(|(_, s)| s),
                stderr_cost_units: cost.and_then(|(_, s)| stderr(s, a.n_cost)),
            }
        })
        .collect();

    Ok(serde_json::json!({
        "inputs": inputs,
        "lines_total": lines_total,
        "lines_parsed_json": lines_parsed,
        "lines_outcome": lines_outcome,
        "groups": rows,
    }))
}
