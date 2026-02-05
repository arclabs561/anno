//! Audit muxer decisions JSONL for stability/convergence signals.
//!
//! This tool is intentionally lightweight and dependency-free (besides serde_json).
//! It is meant for reading the JSONL emitted by `matrix_muxer_ci` via `ANNO_MUXER_DECISIONS_FILE`.

use serde_json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Default, Clone)]
struct BackendAgg {
    chosen: u64,
    outcomes: u64,
    mean_f1_sum: f64,
    reward_sum: f64,
    reward_n: u64,
    reward_adj_sum: f64,
    reward_adj_n: u64,
    pen_out_sum: f64,
    pen_out_n: u64,
    pen_dec_sum: f64,
    pen_dec_n: u64,
    display: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct RunAgg {
    n_decisions: u64,
    n_outcomes: u64,
    chosen_seq: Vec<String>,
    chosen_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Default, Clone)]
struct SliceAgg {
    chosen_seq: Vec<String>,
    chosen_counts: BTreeMap<String, u64>,
    n_outcomes: u64,
    // Reward metrics (derived from outcome rows, attributed to decisions via run_id+round).
    chosen_reward_sum: f64,
    chosen_reward_adj_sum: f64,
    chosen_reward_n: u64,
    reward_seq: Vec<Option<f64>>,
    reward_adj_seq: Vec<Option<f64>>,
    per_backend_reward_sum: BTreeMap<String, f64>,
    per_backend_reward_n: BTreeMap<String, u64>,
    per_backend_reward_adj_sum: BTreeMap<String, f64>,
    per_backend_reward_adj_n: BTreeMap<String, u64>,
}

fn normalize_strategy(s: &str) -> String {
    let t = s.trim().to_ascii_lowercase();
    let t = t.replace('_', "-");
    let t = t.replace("--", "-");
    match t.as_str() {
        "mlonly" | "ml-only" | "ml" | "bestfirst" | "best-first" => "ml-only".to_string(),
        "worstfirst" | "worst-first" => "worst-first".to_string(),
        other => other.to_string(),
    }
}

fn shannon_entropy(counts: &BTreeMap<String, u64>) -> f64 {
    let total: f64 = counts.values().copied().sum::<u64>() as f64;
    if total <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in counts.values() {
        let p = (c as f64) / total;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

fn switching_rate(seq: &[String]) -> f64 {
    if seq.len() <= 1 {
        return 0.0;
    }
    let mut switches = 0u64;
    for i in 1..seq.len() {
        if seq[i] != seq[i - 1] {
            switches += 1;
        }
    }
    (switches as f64) / ((seq.len() - 1) as f64)
}

fn top_arm_and_share(counts: &BTreeMap<String, u64>) -> (Option<String>, f64) {
    let total: u64 = counts.values().copied().sum();
    if total == 0 {
        return (None, 0.0);
    }
    let mut best: Option<(&String, u64)> = None;
    for (k, &v) in counts {
        match best {
            None => best = Some((k, v)),
            Some((bk, bv)) => {
                if v > bv || (v == bv && k < bk) {
                    best = Some((k, v));
                }
            }
        }
    }
    let Some((k, v)) = best else {
        return (None, 0.0);
    };
    (Some(k.clone()), (v as f64) / (total as f64))
}

fn clamp01(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn best_arm_by_mean_reward(
    sum: &BTreeMap<String, f64>,
    n: &BTreeMap<String, u64>,
) -> (Option<String>, f64) {
    let mut best: Option<(String, f64)> = None;
    for (k, &s) in sum {
        let nn = n.get(k).copied().unwrap_or(0);
        if nn == 0 {
            continue;
        }
        let m = s / (nn as f64);
        match &best {
            None => best = Some((k.clone(), m)),
            Some((bk, bm)) => {
                if m > *bm || ((m - *bm).abs() <= 1e-12 && k < bk) {
                    best = Some((k.clone(), m));
                }
            }
        }
    }
    best.map(|(k, m)| (Some(k), m)).unwrap_or((None, 0.0))
}

fn pretty_backend_display(backend_id: &str, display: Option<&str>) -> Option<String> {
    let d = display.map(|s| s.trim()).filter(|s| !s.is_empty())?;
    // Prefer an explicit "backend_id(...)" name when present.
    let d = d.to_string();
    if let Some(inner) = d.strip_prefix("stacked(").and_then(|x| x.strip_suffix(')')) {
        let parts: Vec<&str> = inner
            .split('+')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if !parts.is_empty() {
            return Some(format!("{}(priority: {})", backend_id, parts.join(" -> ")));
        }
    }
    if let Some(inner) = d
        .strip_prefix("ensemble(")
        .and_then(|x| x.strip_suffix(')'))
    {
        let parts: Vec<&str> = inner
            .split('|')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if !parts.is_empty() {
            return Some(format!(
                "{}(weighted_vote: {})",
                backend_id,
                parts.join(" | ")
            ));
        }
    }
    Some(d)
}

fn main() -> Result<(), String> {
    let path = env::args()
        .nth(1)
        .or_else(|| env::var("ANNO_MUXER_DECISIONS_FILE").ok())
        .ok_or_else(|| {
            "usage: muxer_audit <decisions.jsonl> (or set ANNO_MUXER_DECISIONS_FILE)".to_string()
        })?;

    let f = File::open(&path).map_err(|e| format!("open {path}: {e}"))?;
    let r = BufReader::new(f);

    let mut by_run: BTreeMap<String, RunAgg> = BTreeMap::new();
    let mut by_slice: BTreeMap<String, SliceAgg> = BTreeMap::new();
    let mut by_backend: BTreeMap<String, BackendAgg> = BTreeMap::new();
    #[derive(Debug, Clone)]
    struct PendingDecision {
        slice_key: String,
        idx: usize,
        chosen: String,
    }
    let mut pending: BTreeMap<String, PendingDecision> = BTreeMap::new();
    let mut parse_errors = 0u64;
    let mut lines = 0u64;

    for line in r.lines() {
        lines += 1;
        let line = line.map_err(|e| format!("read line: {e}"))?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(t) {
            Ok(v) => v,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };

        let run_id = v
            .get("run_id")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown")
            .to_string();
        let round = v.get("round").and_then(|x| x.as_u64()).unwrap_or(1);
        let pending_key = format!("{run_id}#round={round}");

        let is_outcome = v
            .get("record_type")
            .and_then(|x| x.as_str())
            .map(|s| s == "outcome")
            .unwrap_or(false);

        if is_outcome {
            let backend = v
                .get("backend")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            let backend_for_counts = backend.clone();
            let backend_display = v.get("backend_display").and_then(|x| x.as_str());
            let backend_display_pretty = pretty_backend_display(&backend, backend_display);
            let f1 = v.get("primary_f1").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let pen = v.get("monitoring_penalty").and_then(|x| x.as_f64());
            let ok = v.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
            let slice = v
                .get("slice")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            let strategy = v
                .get("strategy")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            let strategy = normalize_strategy(&strategy);
            let ml_only_policy = v
                .get("ml_only_policy")
                .and_then(|x| x.as_str())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty());
            let slice_key = if strategy == "ml-only" {
                if let Some(p) = ml_only_policy {
                    format!("{strategy}/{p}::{slice}")
                } else {
                    format!("{strategy}::{slice}")
                }
            } else {
                format!("{strategy}::{slice}")
            };

            let reward = if ok { clamp01(f1) } else { 0.0 };
            let reward_adj = if let Some(p) = pen {
                if p.is_finite() && p > 0.0 {
                    reward * (-p).exp()
                } else {
                    reward
                }
            } else {
                reward
            };

            let ba = by_backend.entry(backend).or_default();
            ba.outcomes += 1;
            ba.mean_f1_sum += f1;
            ba.reward_sum += reward;
            ba.reward_n += 1;
            ba.reward_adj_sum += reward_adj;
            ba.reward_adj_n += 1;
            if ba.display.is_none() {
                ba.display = backend_display_pretty;
            }
            if let Some(p) = pen {
                if p.is_finite() {
                    ba.pen_out_sum += p;
                    ba.pen_out_n += 1;
                }
            }
            by_run.entry(run_id.clone()).or_default().n_outcomes += 1;
            by_slice.entry(slice_key.clone()).or_default().n_outcomes += 1;

            // Attribute reward to the decision (best-effort: run_id+round).
            if let Some(pd) = pending.remove(&pending_key) {
                if let Some(sa) = by_slice.get_mut(&pd.slice_key) {
                    if pd.idx < sa.reward_seq.len() {
                        sa.reward_seq[pd.idx] = Some(reward);
                        sa.reward_adj_seq[pd.idx] = Some(reward_adj);
                    }
                    sa.chosen_reward_sum += reward;
                    sa.chosen_reward_adj_sum += reward_adj;
                    sa.chosen_reward_n += 1;

                    *sa.per_backend_reward_sum
                        .entry(pd.chosen.clone())
                        .or_insert(0.0) += reward;
                    *sa.per_backend_reward_n
                        .entry(pd.chosen.clone())
                        .or_insert(0) += 1;
                    *sa.per_backend_reward_adj_sum
                        .entry(pd.chosen.clone())
                        .or_insert(0.0) += reward_adj;
                    *sa.per_backend_reward_adj_n.entry(pd.chosen).or_insert(0) += 1;
                }
            } else {
                // Some log streams emit outcome rows without explicit decision rows.
                // In that case, treat the outcome as an implicit decision so convergence
                // diagnostics remain meaningful.
                let ra = by_run.entry(run_id.clone()).or_default();
                ra.n_decisions += 1;
                ra.chosen_seq.push(backend_for_counts.clone());
                *ra.chosen_counts
                    .entry(backend_for_counts.clone())
                    .or_insert(0) += 1;

                let sa = by_slice.entry(slice_key).or_default();
                sa.chosen_seq.push(backend_for_counts.clone());
                sa.reward_seq.push(Some(reward));
                sa.reward_adj_seq.push(Some(reward_adj));
                *sa.chosen_counts
                    .entry(backend_for_counts.clone())
                    .or_insert(0) += 1;

                sa.chosen_reward_sum += reward;
                sa.chosen_reward_adj_sum += reward_adj;
                sa.chosen_reward_n += 1;
                *sa.per_backend_reward_sum
                    .entry(backend_for_counts.clone())
                    .or_insert(0.0) += reward;
                *sa.per_backend_reward_n
                    .entry(backend_for_counts.clone())
                    .or_insert(0) += 1;
                *sa.per_backend_reward_adj_sum
                    .entry(backend_for_counts.clone())
                    .or_insert(0.0) += reward_adj;
                *sa.per_backend_reward_adj_n
                    .entry(backend_for_counts)
                    .or_insert(0) += 1;

                // Also treat it as chosen for backend aggregate counts.
                ba.chosen += 1;
            }
        } else {
            let chosen = v
                .get("chosen")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            let slice = v
                .get("slice")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            let strategy = v
                .get("strategy")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string();
            let strategy = normalize_strategy(&strategy);
            let ml_only_policy = v
                .get("ml_only_policy")
                .and_then(|x| x.as_str())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty());
            let slice_key = if strategy == "ml-only" {
                if let Some(p) = ml_only_policy {
                    format!("{strategy}/{p}::{slice}")
                } else {
                    format!("{strategy}::{slice}")
                }
            } else {
                format!("{strategy}::{slice}")
            };

            // Penalty can be present on decision lines too (e.g., older logs missing outcome penalty).
            let pen = v.get("chosen_monitoring_penalty").and_then(|x| x.as_f64());

            by_run.entry(run_id.clone()).or_default().n_decisions += 1;
            let ra = by_run.get_mut(&run_id).unwrap();
            ra.chosen_seq.push(chosen.clone());
            *ra.chosen_counts.entry(chosen.clone()).or_insert(0) += 1;

            let ba = by_backend.entry(chosen.clone()).or_default();
            ba.chosen += 1;
            if let Some(p) = pen {
                if p.is_finite() {
                    ba.pen_dec_sum += p;
                    ba.pen_dec_n += 1;
                }
            }

            let slice_key_for_pending = slice_key.clone();
            let sa = by_slice.entry(slice_key).or_default();
            sa.chosen_seq.push(chosen.clone());
            sa.reward_seq.push(None);
            sa.reward_adj_seq.push(None);
            *sa.chosen_counts.entry(chosen.clone()).or_insert(0) += 1;

            let idx = sa.chosen_seq.len().saturating_sub(1);
            pending.insert(
                pending_key,
                PendingDecision {
                    slice_key: slice_key_for_pending,
                    idx,
                    chosen,
                },
            );
        }
    }

    println!("muxer_audit: path={path}");
    println!("muxer_audit: lines={} parse_errors={}", lines, parse_errors);
    println!("muxer_audit: runs={}", by_run.len());
    println!("muxer_audit: backends={}", by_backend.len());
    println!();

    println!("## Runs (convergence proxies)");
    for (run_id, ra) in &by_run {
        let h = shannon_entropy(&ra.chosen_counts);
        let sw = switching_rate(&ra.chosen_seq);
        let top_share = ra
            .chosen_counts
            .values()
            .copied()
            .max()
            .map(|m| (m as f64) / (ra.n_decisions.max(1) as f64))
            .unwrap_or(0.0);
        println!(
            "- run_id={} decisions={} outcomes={} entropy={:.3} top_share={:.3} switch_rate={:.3}",
            run_id, ra.n_decisions, ra.n_outcomes, h, top_share, sw
        );
    }
    println!();

    println!("## Slice groups (convergence proxies across file order)");
    for (k, sa) in &by_slice {
        let h = shannon_entropy(&sa.chosen_counts);
        let sw = switching_rate(&sa.chosen_seq);
        let total = sa.chosen_seq.len() as u64;
        let (top_arm, top_share) = top_arm_and_share(&sa.chosen_counts);
        let mean_reward = if sa.chosen_reward_n > 0 {
            sa.chosen_reward_sum / (sa.chosen_reward_n as f64)
        } else {
            0.0
        };
        let mean_reward_adj = if sa.chosen_reward_n > 0 {
            sa.chosen_reward_adj_sum / (sa.chosen_reward_n as f64)
        } else {
            0.0
        };
        let (best_arm, best_mean_reward) =
            best_arm_by_mean_reward(&sa.per_backend_reward_sum, &sa.per_backend_reward_n);
        let regret_proxy = (best_mean_reward - mean_reward).max(0.0);
        print!(
            "- slice={} decisions={} outcomes={} entropy={:.3} top_share={:.3} switch_rate={:.3} top_arm={} mean_reward={:.3} mean_reward_adj={:.3} best_arm_by_mean_reward={} best_mean_reward={:.3} regret_proxy={:.3}",
            k,
            total,
            sa.n_outcomes,
            h,
            top_share,
            sw,
            top_arm.as_deref().unwrap_or("-"),
            mean_reward,
            mean_reward_adj,
            best_arm.as_deref().unwrap_or("-"),
            best_mean_reward,
            regret_proxy
        );

        // Lightweight "convergence" sketch over prefixes:
        // - top-share + top-arm (by frequency)
        // - mean reward and regret proxy vs best arm by mean reward (among tried so far)
        let checkpoints = [5usize, 10, 20, 50, 100];
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        let mut r_sum: BTreeMap<String, f64> = BTreeMap::new();
        let mut r_n: BTreeMap<String, u64> = BTreeMap::new();
        let mut chosen_r_sum = 0.0;
        let mut chosen_r_n = 0u64;
        let mut next_i = 0usize;
        for (i, b) in sa.chosen_seq.iter().enumerate() {
            *counts.entry(b.clone()).or_insert(0) += 1;
            if let Some(r) = sa.reward_seq.get(i).and_then(|x| *x) {
                chosen_r_sum += r;
                chosen_r_n += 1;
                *r_sum.entry(b.clone()).or_insert(0.0) += r;
                *r_n.entry(b.clone()).or_insert(0) += 1;
            }
            if next_i < checkpoints.len() && i + 1 == checkpoints[next_i] {
                let (ta, ts) = top_arm_and_share(&counts);
                let prefix_mean_reward = if chosen_r_n > 0 {
                    chosen_r_sum / (chosen_r_n as f64)
                } else {
                    0.0
                };
                let (_best_arm_p, best_mean_p) = best_arm_by_mean_reward(&r_sum, &r_n);
                let regret_p = (best_mean_p - prefix_mean_reward).max(0.0);
                print!(
                    " top_share@{}={:.3} top_arm@{}={} mean_reward@{}={:.3} regret@{}={:.3}",
                    checkpoints[next_i],
                    ts,
                    checkpoints[next_i],
                    ta.as_deref().unwrap_or("-"),
                    checkpoints[next_i],
                    prefix_mean_reward,
                    checkpoints[next_i],
                    regret_p
                );
                next_i += 1;
            }
        }
        println!();
    }
    println!();

    println!("## Backends");
    for (b, ba) in &by_backend {
        let mean_f1 = if ba.outcomes > 0 {
            ba.mean_f1_sum / (ba.outcomes as f64)
        } else {
            0.0
        };
        let mean_reward = if ba.reward_n > 0 {
            ba.reward_sum / (ba.reward_n as f64)
        } else {
            0.0
        };
        let mean_reward_adj = if ba.reward_adj_n > 0 {
            ba.reward_adj_sum / (ba.reward_adj_n as f64)
        } else {
            0.0
        };
        let mean_pen_out = if ba.pen_out_n > 0 {
            ba.pen_out_sum / (ba.pen_out_n as f64)
        } else {
            0.0
        };
        let mean_pen_dec = if ba.pen_dec_n > 0 {
            ba.pen_dec_sum / (ba.pen_dec_n as f64)
        } else {
            0.0
        };
        println!(
            "- backend={} display={} chosen={} outcomes={} mean_f1={:.3} mean_reward={:.3} mean_reward_adj={:.3} mean_penalty_decision={:.3} (n={}) mean_penalty_outcome={:.3} (n={})",
            b,
            ba.display.as_deref().unwrap_or("-"),
            ba.chosen,
            ba.outcomes,
            mean_f1,
            mean_reward,
            mean_reward_adj,
            mean_pen_dec,
            ba.pen_dec_n,
            mean_pen_out,
            ba.pen_out_n
        );
    }

    Ok(())
}
