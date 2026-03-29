//! CI-friendly randomized matrix test (muxer-backed).
//!
//! This replaces the older archived test harness with something that:
//! - compiles under `--features eval` (no `cli` feature required)
//! - uses `muxer` for deterministic, windowed MAB-style backend selection
//! - respects `ANNO_CI_SEED` and `ANNO_SAMPLE_STRATEGY`
//! - prefers cache in CI, with an opt-in download fallback to avoid no-op runs
//!
//! Environment variables:
//! - `ANNO_CI_SEED`: u64 seed (default: 0)
//! - `ANNO_SAMPLE_STRATEGY`: `random` | `ml-only` | `worst-first` (default: `ml-only`)
//! - `ANNO_MUXER_MODE`: optional mode default (`triage` | `measure`). Used only when
//!   `ANNO_SAMPLE_STRATEGY` is unset.
//! - `ANNO_MATRIX_TASK`: optional task override (e.g. `discontinuous-ner`, `re`, `intra-coref`)
//! - `ANNO_MATRIX_REQUIRE_CACHED`: if true, run in cache-only mode (no fetch); if selection yields
//!   nothing and `ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY=1`, fall back to “try to fetch once”
//! - `ANNO_MATRIX_INCLUDE_NON_AUTOMATABLE`: if `1`/`true`, include non-automatable datasets in candidates
//! - `ANNO_MATRIX_INCLUDE_SLOW_DATASETS`: if `1`/`true`, allow known-slow datasets even under `ANNO_MUXER_PROFILE=fast*`
//! - `ANNO_MATRIX_COVERAGE_REPORT`: if set, write a JSON coverage report to this path
//! - `ANNO_HISTORY_FILE`: optional JSON path for muxer history
//! - `ANNO_MAX_EXAMPLES`: max examples per dataset (default: 5 in CI, 25 locally)
//! - `ANNO_CACHE_DIR`: cache root (datasets live under `$ANNO_CACHE_DIR/datasets`)
//! - `ANNO_ML_IN_MATRIX`: include ML-ish backends in candidates (`1`/`true`)
//! - `ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY`: if `1`/`true`, fall back to downloads when cache-only selection yields nothing
//!
//! Muxer tuning (optional; defaults keep the harness stable/fast):
//! - `ANNO_MUXER_WINDOW_CAP`: window size per arm (default: 50)
//! - `ANNO_MUXER_PER_DATASET`: enable dataset-scoped history + selection (default: true)
//! - `ANNO_MUXER_BACKENDS_PER_RUN`: max backends evaluated per run (default: 2)
//! - `ANNO_MUXER_DATASETS_PER_RUN`: max datasets evaluated per run (default: 2)
//! - `ANNO_MUXER_EXPLORATION_C`: UCB exploration coefficient (default: 0.8)
//! - `ANNO_MUXER_JUNK_WEIGHT`: penalty weight for soft junk (default: 0.8)
//! - `ANNO_MUXER_HARD_JUNK_WEIGHT`: penalty weight for hard junk (default: 1.6)
//! - `ANNO_MUXER_COST_WEIGHT`: penalty weight for mean cost (default: 0.0)
//! - `ANNO_MUXER_LATENCY_WEIGHT`: penalty weight for mean latency (default: 0.0)
//! - `ANNO_MUXER_MAX_MEAN_ELAPSED_MS`: optional constraint (>=0); filters slow arms in ml-only
//! - `ANNO_MUXER_LATENCY_GUARDRAIL_ALLOW_FEWER`: if true, ml-only may return <K arms instead of falling back
//! - `ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT`: if true, untried arms are treated as ineligible under the latency guardrail
//! - `ANNO_MUXER_PROFILE`: presets for the latency guardrail (`off` | `fast` | `fast-strict` | `regress`)
//! - `ANNO_MUXER_MAX_JUNK_RATE`: optional constraint (0..1)
//! - `ANNO_MUXER_MAX_HARD_JUNK_RATE`: optional constraint (0..1)
//! - `ANNO_MUXER_MAX_HTTP_429_RATE`: optional constraint (0..1) (harness sets 429=false today)
//! - `ANNO_MUXER_MAX_MEAN_COST_UNITS`: optional constraint (>=0)
//! - Monitoring window split (monitored selection only):
//!   - `ANNO_MUXER_MONITOR_RECENT_CAP`: “recent” window size for change monitors (default: 20)
//! - Drift monitoring (monitored selection only; all optional):
//!   - `ANNO_MUXER_MAX_DRIFT`: optional drift guard threshold
//!   - `ANNO_MUXER_DRIFT_WEIGHT`: penalty weight for drift (0 disables)
//!   - `ANNO_MUXER_DRIFT_METRIC`: `hellinger` | `rao` | `js` (default: hellinger)
//!   - `ANNO_MUXER_DRIFT_MIN_BASELINE`: minimum baseline samples (default: 20)
//!   - `ANNO_MUXER_DRIFT_MIN_RECENT`: minimum recent samples (default: 10)
//!   - `ANNO_MUXER_DRIFT_TOL`: numerical tolerance (default: 1e-12)
//! - Change monitoring (monitored selection only; all optional):
//!   - `ANNO_MUXER_MAX_CATKL`: optional threshold on `S = n_recent * KL(q_recent || p0_baseline)`
//!   - `ANNO_MUXER_CATKL_WEIGHT`: penalty weight for catKL (0 disables)
//!   - `ANNO_MUXER_CATKL_ALPHA`: Dirichlet smoothing pseudo-count (default: 1e-3)
//!   - `ANNO_MUXER_CATKL_MIN_BASELINE`: minimum baseline samples (default: 40)
//!   - `ANNO_MUXER_CATKL_MIN_RECENT`: minimum recent samples (default: 20)
//!   - `ANNO_MUXER_MAX_CUSUM`: optional threshold on categorical CUSUM score over the recent window
//!   - `ANNO_MUXER_CUSUM_WEIGHT`: penalty weight for CUSUM (0 disables)
//!   - `ANNO_MUXER_CUSUM_ALPHA`: smoothing pseudo-count (default: 1e-3)
//!   - `ANNO_MUXER_CUSUM_MIN_BASELINE`: minimum baseline samples (default: 40)
//!   - `ANNO_MUXER_CUSUM_MIN_RECENT`: minimum recent samples (default: 20)
//!   - `ANNO_MUXER_CUSUM_ALT_P`: optional CSV 4-vector (normalized) over
//!     `[ok_clean, ok_soft_junk, ok_hard_junk, fail]`
//! - `ANNO_MUXER_JUNK_F1_NER`: junk threshold for NER F1 (default: 0.05)
//! - `ANNO_MUXER_JUNK_F1_COREF`: junk threshold for CoNLL F1 (default: 0.20)
//! - `ANNO_MUXER_JUNK_F1_RELATION`: junk threshold for strict F1 (default: 0.10)
//! - `ANNO_MUXER_PRIOR_CALLS`: pseudo-call prior budget for smoothing (default: 6)
//! - `ANNO_MUXER_PRIOR_BY_FACETS`: if true (default), prefer facet-matched priors (lang+domain)
//! - `ANNO_MUXER_NOVELTY`: if true (default), explore unseen arms within a slice before MAB/worst-first
//! - `ANNO_MUXER_CONTROL_K`: reserve K deterministic-random “control” picks (default: 0/off)
//! - `ANNO_MUXER_VERBOSE`: print chosen slice + per-result outcomes (`1`/`true`)
//! - `ANNO_MUXER_DECISIONS_FILE`: optional path to write selection decisions as JSONL
//! - `ANNO_MUXER_DECISIONS_DEFAULT`: if true (default locally, false in CI), write decisions JSONL to a default cache file
//! - `ANNO_MUXER_DECISIONS_TOP`: max candidate rows to include per decision (default: 8)
//! - `ANNO_MUXER_FIXED_BACKEND`: force evaluation of a specific backend (or comma-separated list)
//! - `ANNO_MUXER_FIXED_DATASETS`: force evaluation on specific datasets (comma-separated `DatasetId` debug names)
//! - `ANNO_MUXER_PIN_LANG`: restrict dataset sampling to one or more languages (comma-separated)
//! - `ANNO_MUXER_PIN_DOMAIN`: restrict dataset sampling to one or more domains (comma-separated)
//! - `ANNO_MUXER_PIN_BACKEND`: restrict backend candidates to one or more backends (comma-separated)
//! - Aliases (same semantics):
//!   - `ANNO_MUXER_FORCE_BACKEND` == `ANNO_MUXER_FIXED_BACKEND`
//!   - `ANNO_MUXER_FORCE_DATASETS` == `ANNO_MUXER_FIXED_DATASETS`
//!   - `ANNO_MUXER_FILTER_LANG` == `ANNO_MUXER_PIN_LANG`
//!   - `ANNO_MUXER_FILTER_DOMAIN` == `ANNO_MUXER_PIN_DOMAIN`
//!   - `ANNO_MUXER_FILTER_BACKEND` == `ANNO_MUXER_PIN_BACKEND`
//! - `ANNO_MATRIX_SWEEP`: if true, enables `test_matrix_sweep_all_backends_once` (opt-in)
//! - `ANNO_MATRIX_SWEEP_STRICT`: if true (default when sweep enabled), fail the test on any skip/failure
//! - `ANNO_MATRIX_SWEEP_MAX_EXAMPLES`: max examples per dataset in sweep (default: 5)
//!
//! Worst-first tuning (regression hunting):
//! - `ANNO_WORST_EXPLORATION_C`: exploration coefficient for worst-first (default: 0.8)
//! - `ANNO_WORST_HARD_WEIGHT`: weight for hard failures in worst-first (default: 1.0)
//! - `ANNO_WORST_SOFT_WEIGHT`: weight for soft junk in worst-first (default: 0.2 locally, 0.0 in CI)
//!
//! Notes:
//! - “worst-first” here means “prioritize historically bad outcomes” where “bad” is
//!   an eval failure or low F1, recorded into a muxer window.

// This module is compiled under `anno-eval`'s `eval` feature.
// Tests within this file are still `#[cfg(test)]` as usual.

use crate::eval::backend_factory::BackendFactory;
use crate::eval::loader::{DatasetId, DatasetLoader, LoadableDatasetId};
use crate::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use crate::eval::task_mapping::{
    backend_tasks, dataset_tasks, get_task_backends, get_task_datasets, Task,
};
use crate::muxer_harness as mh;
#[cfg(test)]
use crate::muxer_history::HistoryWindow;
use crate::muxer_history::{BackendHistory, FailKindCount};
use muxer::{
    CandidateDebug, Decision, DecisionNote, Exp3Ix, Exp3IxConfig, Exp3IxState, MabConfig,
    MabSelectionDecision, Outcome, Summary,
};
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Local compat types for logging (removed from muxer >= 0.3.12)
// ---------------------------------------------------------------------------

/// Local version constant for JSONL decision logs.
const MUXER_VERSION: &str = "0.3.12-local";

/// Score kind tag for MAB scalar scores in decision logs.
const LOG_SCORE_KIND_MAB_SCALAR: &str = "mab_scalar";

/// Serializable top-candidate row for decision logs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LogTopCandidate {
    arm: String,
    score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    calls: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ok_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    junk_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hard_junk_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mean_quality_score: Option<f64>,
}

/// Serializable top-candidate list for decision logs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LogTopCandidates {
    kind: String,
    rows: Vec<LogTopCandidate>,
}

/// Per-round guardrail state for multi-pick MAB logging.
#[derive(Debug, Clone)]
struct MultiPickGuardrailRound {
    eligible: Vec<String>,
    stop_early: bool,
    fallback_used: bool,
}

/// One round of a multi-pick MAB selection.
#[derive(Debug, Clone)]
struct MultiPickMabRound {
    mab: MabSelectionDecision,
    guardrail: MultiPickGuardrailRound,
}

/// Stop reason for multi-pick MAB.
#[derive(Debug, Clone)]
struct MultiPickMabStop {
    guardrail: MultiPickGuardrailRound,
}

/// Result of a multi-pick MAB selection (local replacement for removed MabKExplain).
#[derive(Debug, Clone)]
struct MultiPickMabResult {
    chosen: Vec<String>,
    rounds: Vec<MultiPickMabRound>,
    stop: Option<MultiPickMabStop>,
}

/// Serializable round log for multi-pick MAB decision logging.
#[derive(Debug, Clone, serde::Serialize)]
struct MabKRoundLog {
    round: usize,
    remaining: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chosen: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    explore_first: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    constraints_fallback_used: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    constraints_eligible_arms: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_candidates: Option<LogTopCandidates>,
}

/// Per-round guardrail state for multi-pick Exp3Ix logging.
#[derive(Debug, Clone)]
struct Exp3IxGuardrailRound {
    eligible: Vec<String>,
    #[allow(dead_code)]
    stop_early: bool,
}

/// One round of a multi-pick Exp3Ix selection.
#[derive(Debug, Clone)]
struct Exp3IxRoundDetail {
    decision: Decision,
    prob_used: f64,
    guardrail: Exp3IxGuardrailRound,
}

/// Result of a multi-pick Exp3Ix selection (local replacement for removed Exp3IxKExplain).
#[derive(Debug, Clone)]
struct Exp3IxKExplain {
    chosen: Vec<String>,
    state: Exp3IxState,
    rounds: Vec<Exp3IxRoundDetail>,
}

/// Serializable round log for Exp3Ix decision logging.
#[derive(Debug, Clone, serde::Serialize)]
struct Exp3IxKRoundLog {
    round: usize,
    remaining: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chosen: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    explore_first: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_candidates: Option<LogTopCandidates>,
}

/// Multi-pick MAB selection: call `select_mab_explain` (or monitored variant) in a loop,
/// removing chosen arms between rounds.
fn select_mab_k_explain(
    arms: &[String],
    summaries_for: impl Fn(&[String]) -> std::collections::BTreeMap<String, Summary>,
    cfg: MabConfig,
    _guardrail_cfg: mh::LatencyGuardrailConfig,
    k: usize,
) -> MultiPickMabResult {
    let mut remaining = arms.to_vec();
    let mut chosen = Vec::new();
    let mut rounds = Vec::new();

    for _ in 0..k {
        if remaining.is_empty() {
            break;
        }
        let sums = summaries_for(&remaining);
        let d = muxer::select_mab_explain(&remaining, &sums, cfg);
        let pick = d.selection.chosen.clone();
        rounds.push(MultiPickMabRound {
            mab: d,
            guardrail: MultiPickGuardrailRound {
                eligible: remaining.clone(),
                stop_early: false,
                fallback_used: false,
            },
        });
        chosen.push(pick.clone());
        remaining.retain(|b| b != &pick);
    }
    MultiPickMabResult {
        chosen,
        rounds,
        stop: None,
    }
}

/// Multi-pick monitored MAB selection.
fn select_mab_k_monitored_explain(
    arms: &[String],
    summaries_for: impl Fn(&[String]) -> std::collections::BTreeMap<String, Summary>,
    monitored_for: impl Fn(&[String]) -> std::collections::BTreeMap<String, muxer::MonitoredWindow>,
    drift_cfg: muxer::DriftConfig,
    cfg: muxer::MonitoredMabConfig,
    _guardrail_cfg: mh::LatencyGuardrailConfig,
    k: usize,
) -> MultiPickMabResult {
    let mut remaining = arms.to_vec();
    let mut chosen = Vec::new();
    let mut rounds = Vec::new();

    for _ in 0..k {
        if remaining.is_empty() {
            break;
        }
        let sums = summaries_for(&remaining);
        let mon = monitored_for(&remaining);
        let d = muxer::select_mab_monitored_explain_with_summaries(
            &remaining, &sums, &mon, drift_cfg, cfg,
        );
        let pick = d.selection.chosen.clone();
        rounds.push(MultiPickMabRound {
            mab: d,
            guardrail: MultiPickGuardrailRound {
                eligible: remaining.clone(),
                stop_early: false,
                fallback_used: false,
            },
        });
        chosen.push(pick.clone());
        remaining.retain(|b| b != &pick);
    }
    MultiPickMabResult {
        chosen,
        rounds,
        stop: None,
    }
}

/// Multi-pick Exp3Ix selection with guardrail filtering.
fn exp3ix_decide_k_guardrailed(
    cfg: Exp3IxConfig,
    state: Option<Exp3IxState>,
    eligible: &[String],
    summaries: &std::collections::BTreeMap<String, Summary>,
    guardrail_cfg: mh::LatencyGuardrailConfig,
    k: usize,
    decision_seed: u64,
) -> Exp3IxKExplain {
    let mut ex = Exp3Ix::new(cfg);
    if let Some(st) = state {
        ex.restore(st);
    }

    // Apply latency guardrail to get eligible set.
    let filtered: Vec<String> = if let Some(max_ms) = guardrail_cfg.max_mean_ms {
        eligible
            .iter()
            .filter(|b| {
                if let Some(s) = summaries.get(b.as_str()) {
                    if guardrail_cfg.require_measured && s.calls == 0 {
                        return false;
                    }
                    s.calls == 0 || s.mean_elapsed_ms() <= max_ms
                } else {
                    !guardrail_cfg.require_measured
                }
            })
            .cloned()
            .collect()
    } else {
        eligible.to_vec()
    };
    let final_eligible = if filtered.is_empty() {
        eligible.to_vec()
    } else {
        filtered
    };

    let all_arms: Vec<String> = {
        let mut s: BTreeSet<String> = eligible.iter().cloned().collect();
        for k in summaries.keys() {
            s.insert(k.clone());
        }
        s.into_iter().collect()
    };

    let mut remaining = final_eligible.clone();
    let mut chosen = Vec::new();
    let mut rounds = Vec::new();

    for round_idx in 0..k {
        if remaining.is_empty() {
            break;
        }
        let round_seed = decision_seed ^ (round_idx as u64 + 1);
        let d = ex
            .decide_deterministic_filtered(&all_arms, &remaining, round_seed)
            .unwrap_or_else(|| Decision {
                policy: muxer::DecisionPolicy::Exp3Ix,
                chosen: remaining[0].clone(),
                probs: None,
                notes: vec![],
            });
        let pick = d.chosen.clone();
        let prob = d
            .probs
            .as_ref()
            .and_then(|m| m.get(&pick).copied())
            .unwrap_or(0.0);
        rounds.push(Exp3IxRoundDetail {
            decision: d,
            prob_used: prob,
            guardrail: Exp3IxGuardrailRound {
                eligible: remaining.clone(),
                stop_early: false,
            },
        });
        chosen.push(pick.clone());
        remaining.retain(|b| b != &pick);
    }

    Exp3IxKExplain {
        chosen,
        state: ex.snapshot(),
        rounds,
    }
}

/// Build typed round logs for multi-pick MAB (local replacement for removed log_mab_k_rounds_typed).
fn log_mab_k_rounds_typed(mk: &MultiPickMabResult, top_n: usize) -> Vec<MabKRoundLog> {
    let mut logs = Vec::new();
    for (i, r) in mk.rounds.iter().enumerate() {
        let d = &r.mab;
        let mut rows: Vec<LogTopCandidate> = d
            .selection
            .candidates
            .iter()
            .map(|c| {
                let score = c.objective_success;
                LogTopCandidate {
                    arm: c.name.clone(),
                    score,
                    calls: Some(c.calls),
                    ok_rate: Some(c.ok_rate),
                    junk_rate: Some(c.junk_rate),
                    hard_junk_rate: Some(c.hard_junk_rate),
                    mean_quality_score: c.mean_quality_score,
                }
            })
            .collect();
        rows.sort_by(|a, b| b.score.total_cmp(&a.score));
        rows.truncate(top_n.max(1));
        logs.push(MabKRoundLog {
            round: i + 1,
            remaining: r.guardrail.eligible.clone(),
            chosen: Some(d.selection.chosen.clone()),
            explore_first: Some(d.explore_first),
            constraints_fallback_used: Some(d.constraints_fallback_used),
            constraints_eligible_arms: Some(d.eligible_arms.clone()),
            top_candidates: Some(LogTopCandidates {
                kind: LOG_SCORE_KIND_MAB_SCALAR.to_string(),
                rows,
            }),
        });
    }
    // Add stop row if present.
    if let Some(ref _s) = mk.stop {
        logs.push(MabKRoundLog {
            round: logs.len() + 1,
            remaining: Vec::new(),
            chosen: None,
            explore_first: None,
            constraints_fallback_used: None,
            constraints_eligible_arms: None,
            top_candidates: None,
        });
    }
    logs
}

/// Build typed round logs for multi-pick Exp3Ix.
fn log_exp3ix_k_rounds_typed(ex: &Exp3IxKExplain, top_n: usize) -> Vec<Exp3IxKRoundLog> {
    let mut logs = Vec::new();
    for (i, r) in ex.rounds.iter().enumerate() {
        let probs = r.decision.probs.as_ref();
        let mut rows: Vec<LogTopCandidate> = r
            .guardrail
            .eligible
            .iter()
            .map(|arm| {
                let score = probs.and_then(|m| m.get(arm).copied()).unwrap_or(0.0);
                LogTopCandidate {
                    arm: arm.clone(),
                    score,
                    calls: None,
                    ok_rate: None,
                    junk_rate: None,
                    hard_junk_rate: None,
                    mean_quality_score: None,
                }
            })
            .collect();
        rows.sort_by(|a, b| b.score.total_cmp(&a.score));
        rows.truncate(top_n.max(1));
        let explore_first = r
            .decision
            .notes
            .iter()
            .any(|n| matches!(n, DecisionNote::ExploreFirst));
        logs.push(Exp3IxKRoundLog {
            round: i + 1,
            remaining: r.guardrail.eligible.clone(),
            chosen: Some(r.decision.chosen.clone()),
            explore_first: Some(explore_first),
            top_candidates: Some(LogTopCandidates {
                kind: "exp3ix_prob".to_string(),
                rows,
            }),
        });
    }
    logs
}

/// Compat shim for exp3ix_decide_persisted (removed from muxer >= 0.3.12).
/// Returns (Decision, Exp3IxState) like the old function.
#[cfg(test)]
fn exp3ix_decide_persisted(
    cfg: Exp3IxConfig,
    state: Option<Exp3IxState>,
    arms: &[String],
    eligible: &[String],
    decision_seed: u64,
) -> Option<(Decision, Exp3IxState)> {
    let mut ex = Exp3Ix::new(cfg);
    if let Some(st) = state {
        ex.restore(st);
    }
    let d = ex.decide_deterministic_filtered(arms, eligible, decision_seed)?;
    Some((d, ex.snapshot()))
}

/// Compat shim for exp3ix_update_persisted (removed from muxer >= 0.3.12).
#[cfg(test)]
fn exp3ix_update_persisted(
    cfg: Exp3IxConfig,
    state: Exp3IxState,
    arm: &str,
    reward: f64,
    prob_used: f64,
) -> Exp3IxState {
    let mut ex = Exp3Ix::new(cfg);
    ex.restore(state);
    ex.update_reward_with_prob(arm, reward, prob_used);
    ex.snapshot()
}

/// Compat shim for exp3ix_update_persisted used in non-test code.
fn exp3ix_update_persisted_prod(
    cfg: Exp3IxConfig,
    state: Exp3IxState,
    arm: &str,
    reward: f64,
    prob_used: f64,
) -> Exp3IxState {
    let mut ex = Exp3Ix::new(cfg);
    ex.restore(state);
    ex.update_reward_with_prob(arm, reward, prob_used);
    ex.snapshot()
}

#[derive(Debug, Clone, Copy)]
enum SampleStrategy {
    Random,
    MlOnly,
    WorstFirst,
    /// **Estimation-first**: select (backend, dataset) cells to maximize information
    /// about the full quality matrix.  Prioritizes cells with fewest observations or
    /// highest uncertainty, rather than routing to the "best" arm.
    ///
    /// This is the right objective when the goal is measurement (what is the true F1
    /// of each backend on each dataset?) rather than exploitation (route to the best).
    /// Detection comes naturally: cells with stale observations or high variance are
    /// prioritized, so changes are caught as a byproduct of estimation.
    Estimate,
}

#[derive(Debug, Clone, Copy)]
enum MuxerMode {
    /// Regression hunting: prioritize historically broken backends, run regression
    /// detection on the full quality matrix.  Maps to `WorstFirst` strategy.
    Triage,
    /// Stable measurement: route to the best backends for reliable F1 estimates.
    /// Maps to `MlOnly` strategy.
    Measure,
    /// Matrix coverage: fill the quality matrix by selecting least-observed cells.
    /// Maps to `Estimate` strategy.  Useful for systematic benchmarking.
    Coverage,
}

impl MuxerMode {
    fn from_env() -> Option<Self> {
        let v = std::env::var("ANNO_MUXER_MODE").ok()?;
        let t = v.trim().to_ascii_lowercase();
        if t.is_empty() {
            return None;
        }
        match t.as_str() {
            // Preferred names:
            "triage" => Some(Self::Triage),
            "measure" => Some(Self::Measure),
            "coverage" => Some(Self::Coverage),

            // Back-compat aliases:
            "bug" | "bug-hunt" | "bughunt" | "regress" | "regression" => Some(Self::Triage),
            "perf" | "perf-estimate" | "perfestimate" | "measurement" => Some(Self::Measure),
            "estimate" | "benchmark" | "fill" | "sweep" => Some(Self::Coverage),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MlOnlyPolicy {
    Exp3Ix,
    Mab,
    /// Contextual bandit (LinUCB): uses dataset-derived feature vectors to learn
    /// generalizable routing patterns across slices.
    ///
    /// Breaks the non-contextual collapse: in the flat regime, estimation and detection
    /// are proportional (D_eff = K-1).  With context features, the design measure gains
    /// spatial dimensions and objectives genuinely diverge.
    LinUcb,
}

impl MlOnlyPolicy {
    fn from_env() -> Self {
        match std::env::var("ANNO_MUXER_MLONLY_POLICY")
            .ok()
            .unwrap_or_else(|| "exp3ix".to_string())
            .to_lowercase()
            .as_str()
        {
            "mab" => Self::Mab,
            "exp3ix" | "exp3-ix" | "exp3" => Self::Exp3Ix,
            "linucb" | "lin-ucb" | "contextual" => Self::LinUcb,
            _ => Self::Exp3Ix,
        }
    }

    fn id_str(self) -> &'static str {
        match self {
            Self::Exp3Ix => "exp3ix",
            Self::Mab => "mab",
            Self::LinUcb => "linucb",
        }
    }
}

impl SampleStrategy {
    fn from_env() -> Self {
        // Explicit strategy wins.
        if let Ok(s) = std::env::var("ANNO_SAMPLE_STRATEGY") {
            match s.trim().to_ascii_lowercase().as_str() {
                "random" => return Self::Random,
                "worst-first" | "worstfirst" => return Self::WorstFirst,
                // Historical name: "ml-only" (selection is "best-ish", but not necessarily ML-only).
                "ml-only" | "mlonly" | "ml" | "best-first" | "bestfirst" => return Self::MlOnly,
                "estimate" | "estimation" | "measure-all" | "coverage" => return Self::Estimate,
                _ => {}
            }
        }

        // Otherwise, mode chooses the default.
        match MuxerMode::from_env() {
            Some(MuxerMode::Triage) => Self::WorstFirst,
            Some(MuxerMode::Measure) => Self::MlOnly,
            Some(MuxerMode::Coverage) => Self::Estimate,
            None => Self::MlOnly,
        }
    }

    fn id_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::MlOnly => "ml-only",
            Self::WorstFirst => "worst-first",
            Self::Estimate => "estimate",
        }
    }
}

fn in_ci() -> bool {
    // Treat only GitHub Actions as CI here.
    //
    // Many local dev shells set `CI=1`, but for the matrix harness we want local default
    // behavior to be “try to fetch once, then use cache”.
    std::env::var("GITHUB_ACTIONS").is_ok()
}

fn matrix_require_cached() -> bool {
    // Default:
    // - CI prefers cache; opt-in download fallback via `ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY`
    // - local runs default to “try to fetch”
    //
    // Override:
    // - if user explicitly sets `ANNO_MATRIX_REQUIRE_CACHED`, honor it even in CI
    if std::env::var("ANNO_MATRIX_REQUIRE_CACHED").is_ok() {
        return mh::env_bool("ANNO_MATRIX_REQUIRE_CACHED", false);
    }
    in_ci()
}

#[cfg(test)]
fn ci_seed() -> u64 {
    std::env::var("ANNO_CI_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn max_examples_per_dataset() -> usize {
    // Keep this intentionally small: this test runs inside `cargo test` and must not time out.
    let default = if in_ci() { 5 } else { 25 };
    std::env::var("ANNO_MAX_EXAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    mh::env_usize(name, default)
}

fn trunc(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

#[test]
fn test_exp3ix_can_outperform_mab_when_summaries_equal_but_reward_differs() {
    // Constructed "capability gap" test:
    // - summaries are identical, so MAB cannot distinguish arms
    // - scalar reward differs, so EXP3-IX can learn and win
    let slice = "capgap";
    let seed = 0u64;

    let arms = vec!["a".to_string(), "b".to_string()];
    let cfg = Exp3IxConfig {
        horizon: 200,
        confidence_delta: None,
        seed: 0,
        decay: 1.0,
    };
    let mut st: Option<Exp3IxState> = None;

    // Keep summaries equal forever.
    let s = Summary {
        calls: 10,
        ok: 10,
        junk: 0,
        hard_junk: 0,
        cost_units: 0,
        elapsed_ms_sum: 0,
        mean_quality_score: None,
    };

    // MAB will deterministically pick one arm under identical summaries.
    let mut summaries = BTreeMap::new();
    summaries.insert("a".to_string(), s);
    summaries.insert("b".to_string(), s);
    let mab_choice =
        muxer::select_mab_explain(&arms, &summaries, mh::monitored_mab_config_from_env().base)
            .selection
            .chosen;

    // Make the MAB-chosen arm worse so EXP3-IX has an opportunity to beat it.
    let r_a = if mab_choice == "a" { 0.6 } else { 0.9 };
    let r_b = if mab_choice == "b" { 0.6 } else { 0.9 };

    let mut total_mab = 0.0;
    let mut total_exp3 = 0.0;

    for t in 0..200u64 {
        let (d, st2) = exp3ix_decide_persisted(
            cfg,
            st.take(),
            &arms,
            &arms,
            seed ^ (t + 1) ^ mh::stable_hash64(0, slice),
        )
        .unwrap();
        let chosen = d.chosen.clone();
        let r = if chosen == "a" { r_a } else { r_b };
        total_exp3 += r;
        let p = d
            .probs
            .as_ref()
            .and_then(|m| m.get(&chosen).copied())
            .unwrap_or(0.0);
        st = Some(exp3ix_update_persisted(cfg, st2, &chosen, r, p));

        let r_mab = if mab_choice == "a" { r_a } else { r_b };
        total_mab += r_mab;
    }

    assert!(
        total_exp3 > total_mab + 5.0,
        "exp3ix should beat mab in graded-reward/identical-summary scenario (exp3={} mab={} mab_choice={})",
        total_exp3,
        total_mab,
        mab_choice
    );
}

#[derive(Debug, Clone, Copy)]
struct WorstFirstConfig {
    exploration_c: f64,
    hard_weight: f64,
    soft_weight: f64,
}

fn worst_first_config_from_env() -> WorstFirstConfig {
    // Local default: include some soft junk so worst-first also surfaces "broken outputs" (low_signal)
    // regressions, not just hard failures. Keep CI default strict to avoid churn.
    let default_soft = if in_ci() { 0.0 } else { 0.2 };
    let soft_weight = std::env::var("ANNO_WORST_SOFT_WEIGHT")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(default_soft)
        .max(0.0);
    WorstFirstConfig {
        exploration_c: mh::env_f64("ANNO_WORST_EXPLORATION_C", 0.8).max(0.0),
        hard_weight: mh::env_f64("ANNO_WORST_HARD_WEIGHT", 1.0).max(0.0),
        soft_weight,
    }
}

fn default_control_k_for_mode() -> usize {
    // Perf-estimate mode wants some unbiased "coverage" anchor to reduce selection bias when
    // interpreting estimated dataset performance. Bug-hunt mode wants to spend budget on failures.
    //
    // Keep CI default off unless explicitly set.
    if in_ci() {
        return 0;
    }
    if std::env::var("ANNO_MUXER_CONTROL_K").is_ok() {
        return 0;
    }
    match MuxerMode::from_env() {
        Some(MuxerMode::Measure) => 1,
        _ => 0,
    }
}

fn exp3ix_config_from_env(seed: u64) -> Exp3IxConfig {
    Exp3IxConfig {
        horizon: mh::env_usize("ANNO_MUXER_EXP3_HORIZON", 1000).max(1),
        confidence_delta: None,
        seed,
        decay: mh::env_f64("ANNO_MUXER_EXP3_DECAY", 1.0).clamp(0.01, 1.0),
    }
}

fn monitoring_enabled(cfg: &muxer::MonitoredMabConfig) -> bool {
    cfg.max_drift.is_some()
        || cfg.drift_weight > 0.0
        || cfg.max_catkl.is_some()
        || cfg.catkl_weight > 0.0
        || cfg.max_cusum.is_some()
        || cfg.cusum_weight > 0.0
}

fn monitoring_scores_for_backend(
    history: &BackendHistory,
    backend: &str,
    monitor_recent_cap: usize,
    cfg: &muxer::MonitoredMabConfig,
    drift_cfg: muxer::DriftConfig,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    // Monitoring is defined on backend-global windows only.
    let m = history.monitored_for_backends(&[backend.to_string()], monitor_recent_cap);
    let Some(w) = m.get(backend) else {
        return (None, None, None);
    };

    let drift =
        muxer::monitor::drift_between_windows(w.baseline(), w.recent(), drift_cfg).map(|d| d.score);
    let catkl = muxer::monitor::catkl_score_between_windows(
        w.baseline(),
        w.recent(),
        cfg.catkl_alpha,
        drift_cfg.tol,
        cfg.catkl_min_baseline,
        cfg.catkl_min_recent,
    );
    let cusum = muxer::monitor::cusum_score_between_windows(
        w.baseline(),
        w.recent(),
        cfg.cusum_alpha,
        drift_cfg.tol,
        cfg.cusum_min_baseline,
        cfg.cusum_min_recent,
        cfg.cusum_alt_p,
    );
    (drift, catkl, cusum)
}

fn monitoring_penalty(
    drift: Option<f64>,
    catkl: Option<f64>,
    cusum: Option<f64>,
    cfg: &muxer::MonitoredMabConfig,
    drift_metric: muxer::DriftMetric,
) -> f64 {
    // Normalize scores into [0,1] (best-effort), then take a weighted sum.
    let drift_max = match drift_metric {
        muxer::DriftMetric::Hellinger => 1.0,
        muxer::DriftMetric::Rao => core::f64::consts::PI,
        muxer::DriftMetric::JensenShannon => core::f64::consts::LN_2,
        _ => 1.0,
    };
    let drift_norm = drift.unwrap_or(0.0).max(0.0).min(drift_max) / drift_max;
    let catkl_norm = {
        let x = catkl.unwrap_or(0.0).max(0.0);
        x / (1.0 + x)
    };
    let cusum_norm = {
        let x = cusum.unwrap_or(0.0).max(0.0);
        x / (1.0 + x)
    };
    cfg.drift_weight.max(0.0) * drift_norm
        + cfg.catkl_weight.max(0.0) * catkl_norm
        + cfg.cusum_weight.max(0.0) * cusum_norm
}

#[derive(Debug, Clone, serde::Serialize)]
struct WorstFirstRoundLog {
    remaining: Vec<String>,
    chosen: String,
    explore_first: bool,
    exploration_c: f64,
    hard_weight: f64,
    soft_weight: f64,
    top_candidates: LogTopCandidates,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DecisionLog {
    schema_version: u32,
    muxer_version: String,
    run_id: String,
    strategy: String,
    /// Optional: disambiguates the ML-only selection policy (`exp3ix` or `mab`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ml_only_policy: Option<String>,
    slice: String,
    muxer_profile: Option<String>,
    latency_guardrail_max_mean_ms: Option<u64>,
    latency_guardrail_allow_fewer: Option<bool>,
    latency_guardrail_require_measured: Option<bool>,
    round: usize,
    datasets: Vec<String>,
    remaining: Vec<String>,
    chosen: Option<String>,
    explore_first: Option<bool>,
    constraints_fallback_used: Option<bool>,
    eligible_arms: Option<Vec<String>>,
    top_candidates: Option<LogTopCandidates>,
    /// If present, these arms were selected as deterministic-random "control" picks (bias anchor).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    control_arms: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chosen_fail_kinds_top: Option<Vec<FailKindCount>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mab_k_round: Option<MabKRoundLog>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exp3ix_rounds: Option<Vec<Exp3IxKRoundLog>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    worst_first_round: Option<WorstFirstRoundLog>,

    /// Optional monitoring metadata (drift/catKL/CUSUM) when enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    monitoring_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    monitoring_fallback_used: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    monitoring_eligible_arms: Option<Vec<String>>,

    /// Optional per-chosen monitoring scores (best-effort, backend-global).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chosen_drift_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chosen_catkl_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chosen_cusum_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chosen_monitoring_penalty: Option<f64>,
}

fn append_jsonl<T: serde::Serialize>(path: &str, v: &T) {
    if path.trim().is_empty() {
        return;
    }
    // Robust for developer-local paths like `.generated/foo.jsonl`.
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "matrix-muxer: failed to create decision log dir {:?}: {e}",
                    parent
                );
                return;
            }
        }
    }
    let line = match serde_json::to_string(v) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("matrix-muxer: failed to serialize decision log: {e}");
            return;
        }
    };
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                eprintln!("matrix-muxer: failed to write decision log to {path}: {e}");
            }
        }
        Err(e) => {
            eprintln!("matrix-muxer: failed to open decision log file {path}: {e}");
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct DecisionOutcomeLog {
    // Independent schema for post-run outcome events.
    schema_version: u32,
    record_type: String,
    muxer_version: String,
    run_id: String,
    strategy: String,
    /// Optional: disambiguates the ML-only selection policy (`exp3ix` or `mab`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ml_only_policy: Option<String>,
    slice: String,
    dataset: String,
    backend: String,
    /// Optional backend display name (may include composition details).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    backend_display: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    primary_f1: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    junk_f1_threshold: Option<f64>,
    ok: bool,
    junk: bool,
    hard_junk: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fail_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    elapsed_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cost_units: Option<u64>,

    /// Optional monitoring scores at outcome time (best-effort, backend-global).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    drift_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    catkl_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cusum_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    monitoring_penalty: Option<f64>,
}

#[test]
fn test_decision_log_schema_smoke() {
    let log = DecisionLog {
        schema_version: 6,
        muxer_version: MUXER_VERSION.to_string(),
        run_id: "seed=0 slice=ner strategy=MlOnly".to_string(),
        strategy: "ml-only".to_string(),
        ml_only_policy: None,
        slice: "ner".to_string(),
        muxer_profile: None,
        latency_guardrail_max_mean_ms: None,
        latency_guardrail_allow_fewer: None,
        latency_guardrail_require_measured: None,
        round: 1,
        datasets: vec!["WNUT-17".to_string()],
        remaining: vec!["crf".to_string(), "stacked".to_string()],
        chosen: Some("crf".to_string()),
        explore_first: Some(false),
        constraints_fallback_used: None,
        eligible_arms: None,
        top_candidates: Some(LogTopCandidates {
            kind: LOG_SCORE_KIND_MAB_SCALAR.to_string(),
            rows: Vec::new(),
        }),
        control_arms: None,
        chosen_fail_kinds_top: Some(vec![FailKindCount {
            kind: "timeout".to_string(),
            count: 2,
        }]),
        mab_k_round: None,
        exp3ix_rounds: None,
        worst_first_round: None,
        monitoring_enabled: None,
        monitoring_fallback_used: None,
        monitoring_eligible_arms: None,
        chosen_drift_score: None,
        chosen_catkl_score: None,
        chosen_cusum_score: None,
        chosen_monitoring_penalty: None,
    };

    let s = serde_json::to_string(&log).expect("serialize DecisionLog");
    let v: serde_json::Value = serde_json::from_str(&s).expect("parse DecisionLog JSON");

    assert_eq!(v["schema_version"].as_u64(), Some(6));
    assert_eq!(v["muxer_version"].as_str(), Some(MUXER_VERSION));
    assert_eq!(
        v["top_candidates"]["kind"].as_str(),
        Some(LOG_SCORE_KIND_MAB_SCALAR)
    );
    assert!(v["top_candidates"]["rows"].is_array());
    assert!(v["chosen_fail_kinds_top"].is_array());
}

#[test]
fn test_monitoring_penalty_is_monotone_and_reward_adjustment_is_bounded() {
    let cfg = muxer::MonitoredMabConfig {
        drift_weight: 1.0,
        catkl_weight: 2.0,
        cusum_weight: 3.0,
        drift_metric: muxer::DriftMetric::Hellinger,
        ..muxer::MonitoredMabConfig::default()
    };

    let p0 = monitoring_penalty(Some(0.0), Some(0.0), Some(0.0), &cfg, cfg.drift_metric);
    let p1 = monitoring_penalty(Some(0.2), Some(0.5), Some(1.0), &cfg, cfg.drift_metric);
    let p2 = monitoring_penalty(Some(0.9), Some(2.0), Some(4.0), &cfg, cfg.drift_metric);
    assert!(p0 <= p1 + 1e-12);
    assert!(p1 <= p2 + 1e-12);

    let r = 0.7;
    let r0 = (r * (-p0).exp()).clamp(0.0, 1.0);
    let r1 = (r * (-p1).exp()).clamp(0.0, 1.0);
    let r2 = (r * (-p2).exp()).clamp(0.0, 1.0);
    assert!((0.0..=1.0).contains(&r0));
    assert!((0.0..=1.0).contains(&r1));
    assert!((0.0..=1.0).contains(&r2));
    assert!(r0 + 1e-12 >= r1);
    assert!(r1 + 1e-12 >= r2);
}

fn matrix_task_override() -> Option<Task> {
    let raw = std::env::var("ANNO_MATRIX_TASK").ok()?;
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    let task = Task::from_code(t)?;
    if !TaskEvaluator::is_task_supported(task) {
        panic!(
            "matrix-muxer: task override {} is not supported by TaskEvaluator (catalog-only)",
            task.code()
        );
    }
    Some(task)
}

fn eligible_tasks_for_run(loader: &DatasetLoader, require_cached: bool) -> Vec<Task> {
    let mut out: Vec<Task> = Vec::new();
    for &task in Task::all() {
        if !TaskEvaluator::is_task_supported(task) {
            continue;
        }
        // Must have at least one matrix-eligible backend under current env gating.
        if backend_candidates(SampleStrategy::Random, &[task]).is_empty() {
            continue;
        }

        // Must have at least one dataset candidate (loadable, and cached if required).
        if !candidate_datasets_for_tasks(loader, &[task], require_cached).is_empty() {
            out.push(task);
        }
    }

    out.sort_by_key(|t| t.code());
    out.dedup();
    out
}

fn selected_task_for_run(seed: u64, loader: &DatasetLoader, require_cached: bool) -> Option<Task> {
    if let Some(t) = matrix_task_override() {
        return Some(t);
    }
    let mut tasks = eligible_tasks_for_run(loader, require_cached);
    if tasks.is_empty() {
        return None;
    }

    // Optional coarse “perspective” hint for task selection.
    //
    // This is intentionally soft guidance: if the requested perspective yields no eligible tasks,
    // we fall back to the full eligible set.
    //
    // Supported values:
    // - ner
    // - re / relation
    // - coref
    // - temporal
    if let Ok(p) = std::env::var("ANNO_MATRIX_PERSPECTIVE") {
        let p = p.trim().to_ascii_lowercase();
        if !p.is_empty() && p != "mixed" {
            let filtered: Vec<Task> = tasks
                .iter()
                .copied()
                .filter(|t| match p.as_str() {
                    "ner" => matches!(t, Task::NER | Task::DiscontinuousNER),
                    "re" | "relation" | "relation_extraction" => {
                        matches!(t, Task::RelationExtraction)
                    }
                    "coref" => t.is_coref_family(),
                    "temporal" => *t == Task::Temporal,
                    _ => true,
                })
                .collect();
            if !filtered.is_empty() {
                tasks = filtered;
            }
        }
    }
    let items: Vec<String> = tasks.iter().map(|t| t.code().to_string()).collect();
    let chosen = mh::pick_random_subset(seed ^ 0xC0DE_CAFE, &items, 1)
        .into_iter()
        .next()?;
    tasks.into_iter().find(|t| t.code() == chosen)
}

fn history_path(slice_tag: &str) -> PathBuf {
    if let Ok(p) = std::env::var("ANNO_HISTORY_FILE") {
        return PathBuf::from(p);
    }

    fn salt_slug() -> Option<String> {
        let s = std::env::var("ANNO_MUXER_HISTORY_SALT").ok()?;
        let t = s.trim();
        if t.is_empty() {
            return None;
        }
        // Keep filenames portable: ASCII-ish slug only.
        let mut out = String::new();
        for ch in t.chars().take(64) {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                out.push(ch);
            } else {
                out.push('_');
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
    let salt = salt_slug();
    let suffix = match salt.as_deref() {
        None => format!("muxer_history.{slice_tag}.json"),
        Some(s) => format!("muxer_history.{slice_tag}.salt={s}.json"),
    };

    // Prefer ANNO_CACHE_DIR so it matches CI caching.
    if let Ok(dir) = std::env::var("ANNO_CACHE_DIR") {
        return PathBuf::from(dir).join(suffix);
    }

    // Fallback: place next to default dataset cache root.
    // DatasetLoader::new() uses platform cache: ~/.cache/anno/datasets (linux).
    // We'll store history alongside `.../anno/`.
    if let Some(base) = dirs::cache_dir() {
        return base.join("anno").join(suffix);
    }
    PathBuf::from(".").join(suffix)
}

/// Path for the **global** LinUCB state file (shared across all task/facet slices).
///
/// This is the key to breaking the non-contextual collapse: by accumulating
/// observations from biomedical, social, news, etc. contexts into a single
/// ridge-regression model, the per-arm theta vectors learn feature-specific
/// weights instead of collapsing to a single direction.
fn linucb_global_state_path() -> PathBuf {
    if let Ok(p) = std::env::var("ANNO_LINUCB_STATE_FILE") {
        return PathBuf::from(p);
    }
    let suffix = "linucb_global_state.json";
    if let Ok(dir) = std::env::var("ANNO_CACHE_DIR") {
        return PathBuf::from(dir).join(suffix);
    }
    if let Some(base) = dirs::cache_dir() {
        return base.join("anno").join(suffix);
    }
    PathBuf::from(".").join(suffix)
}

/// Path for the eval-results JSONL file (and its SQLite sibling).
///
/// Resolution order mirrors `TaskEvaluator::new()`:
/// 1. `ANNO_EVAL_HISTORY` — explicit override
/// 2. `ANNO_CACHE_DIR` — CI-consistent cache directory
/// 3. `dirs::cache_dir()/anno/eval-results.jsonl` — platform default
fn eval_history_jsonl_path() -> PathBuf {
    if let Ok(p) = std::env::var("ANNO_EVAL_HISTORY") {
        return PathBuf::from(p);
    }
    if let Ok(dir) = std::env::var("ANNO_CACHE_DIR") {
        return PathBuf::from(dir).join("eval-results.jsonl");
    }
    if let Some(base) = dirs::cache_dir() {
        return base.join("anno").join("eval-results.jsonl");
    }
    PathBuf::from("eval-results.jsonl")
}

fn load_linucb_global_state(path: &PathBuf) -> Option<muxer::LinUcbState> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn save_linucb_global_state(path: &PathBuf, state: &muxer::LinUcbState) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let json = match serde_json::to_string_pretty(state) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("matrix-muxer: failed to serialize LinUCB state: {}", e);
            return;
        }
    };
    // Atomic write: write to temp file, then rename.  Prevents corruption if
    // two harness processes write simultaneously.
    let tmp_path = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp_path, &json) {
        eprintln!(
            "matrix-muxer: failed to write LinUCB temp file {}: {}",
            tmp_path.display(),
            e
        );
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        eprintln!(
            "matrix-muxer: failed to rename LinUCB state {} -> {}: {}",
            tmp_path.display(),
            path.display(),
            e
        );
    }
}

fn decisions_path() -> Option<String> {
    if let Ok(p) = std::env::var("ANNO_MUXER_DECISIONS_FILE") {
        let t = p.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    if in_ci() {
        return None;
    }
    if !mh::env_bool("ANNO_MUXER_DECISIONS_DEFAULT", true) {
        return None;
    }

    fn salt_slug() -> Option<String> {
        // Keep this separate from history salt so you can rotate logs independently.
        let s = std::env::var("ANNO_MUXER_DECISIONS_SALT").ok()?;
        let t = s.trim();
        if t.is_empty() {
            return None;
        }
        let mut out = String::new();
        for ch in t.chars().take(64) {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                out.push(ch);
            } else {
                out.push('_');
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
    let salt = salt_slug();
    let suffix = match salt.as_deref() {
        None => "muxer_decisions.jsonl".to_string(),
        Some(s) => format!("muxer_decisions.salt={s}.jsonl"),
    };

    if let Ok(dir) = std::env::var("ANNO_CACHE_DIR") {
        return Some(
            PathBuf::from(dir)
                .join(suffix)
                .to_string_lossy()
                .to_string(),
        );
    }
    if let Some(base) = dirs::cache_dir() {
        return Some(base.join("anno").join(suffix).to_string_lossy().to_string());
    }
    Some(
        PathBuf::from(".")
            .join(suffix)
            .to_string_lossy()
            .to_string(),
    )
}

fn slice_for_run(
    seed: u64,
    loader: &DatasetLoader,
    require_cached: bool,
) -> Option<(String, Vec<Task>)> {
    let task = selected_task_for_run(seed, loader, require_cached)?;
    Some((task.code().to_string(), vec![task]))
}

// Note: muxer history persistence lives in `crate::muxer_history`.

fn pick_random_subset(seed: u64, items: &[String], k: usize) -> Vec<String> {
    mh::pick_random_subset(seed, items, k)
}

fn junk_f1_threshold(task: Task) -> f64 {
    // Heuristic thresholds: we only need a coarse “low-signal” cutoff for routing.
    //
    // - NER: strict_f1 is usually meaningful even on small samples.
    // - Coref/CDCR: conll_f1 tends to be lower/noisier; keep a slightly lower cutoff.
    // - Relation: strict_f1 can be sparse; keep a low cutoff to avoid marking everything junk.
    match task {
        // Keep this low: baselines like CRF can be ~0.05–0.15 on tough slices; we only want to
        // label “basically broken” outputs as junk.
        Task::NER => mh::env_f64("ANNO_MUXER_JUNK_F1_NER", 0.05),
        Task::IntraDocCoref | Task::InterDocCoref => mh::env_f64("ANNO_MUXER_JUNK_F1_COREF", 0.20),
        // Relation strict_f1 is often sparse on small slices; keep the default low enough
        // to distinguish “completely broken (0.0)” from “weak baseline (~0.03–0.06)”.
        Task::RelationExtraction => mh::env_f64("ANNO_MUXER_JUNK_F1_RELATION", 0.02),
        _ => 0.30,
    }
}

fn monitored_config_from_env() -> muxer::MonitoredMabConfig {
    mh::monitored_mab_config_from_env()
}

#[allow(clippy::too_many_arguments)]
fn select_backends(
    strategy: SampleStrategy,
    seed: u64,
    slice_tag: &str,
    history: &BackendHistory,
    prior: Option<&BackendHistory>,
    candidates_in_order: &[String],
    datasets: Option<&[DatasetId]>,
    k: usize,
    prior_calls: u64,
) -> Vec<String> {
    if candidates_in_order.is_empty() || k == 0 {
        return Vec::new();
    }

    match strategy {
        SampleStrategy::Random => pick_random_subset(seed, candidates_in_order, k),
        SampleStrategy::MlOnly => {
            // Optional selection-bias anchor: reserve K deterministic-random "control" picks.
            //
            // Perf-estimate mode defaults to 1 locally (unless explicitly overridden) to reduce
            // selection bias in performance interpretation.
            // Never let the *default* control pick consume the entire budget; if k==1, reserving
            // a control pick would fully bypass ML selection and produce confusing outcome-only
            // logs. Users can still force `control_k == k` explicitly via env.
            let control_k = default_control_k_for_mode()
                .min(k.saturating_sub(1))
                .max(mh::control_k_from_env());
            let (control, remaining_k) = mh::split_control_budget(
                seed,
                candidates_in_order,
                k,
                mh::ControlConfig::with_k(control_k),
            );
            let mut candidates_for_muxer: Vec<String> = candidates_in_order.to_vec();
            candidates_for_muxer.retain(|b| !control.contains(b));

            let per_dataset = mh::env_bool("ANNO_MUXER_PER_DATASET", true);

            // MAB selection (muxer): pick historically "best" arms (high ok_rate, low junk),
            // with exploration to avoid fixating too early.
            let mon_cfg = monitored_config_from_env();
            let drift_cfg = mh::drift_config_from_env(mon_cfg.drift_metric);
            let is_monitored = monitoring_enabled(&mon_cfg);
            let monitor_recent_cap = mh::env_usize("ANNO_MUXER_MONITOR_RECENT_CAP", 20);
            let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
            let decisions_path = decisions_path();
            let decisions_top = mh::env_usize("ANNO_MUXER_DECISIONS_TOP", 8).max(1);
            let run_id = format!("seed={} slice={} strategy={:?}", seed, slice_tag, strategy);

            let guard = mh::latency_guardrail_from_env();
            let mut mk_opt = None;
            let fill = mh::policy_fill_k_observed_with(
                seed,
                &candidates_for_muxer,
                remaining_k,
                // MAB selection already includes an explore-first phase keyed on observed calls.
                // Harness-level novelty pre-chooses unseen arms *outside* the muxer decision path,
                // which can suppress decision logs and obscure learning dynamics.
                false,
                guard,
                |b| {
                    let s = history.observed_summary_for(b, datasets, per_dataset);
                    (s.calls, s.elapsed_ms_sum)
                },
                |eligible, remaining_k| {
                    let summaries_for = |remaining: &[String]| {
                        history.summaries_for(prior, remaining, datasets, per_dataset, prior_calls)
                    };
                    let guardrail_cfg = mh::LatencyGuardrailConfig {
                        // Latency guardrail is applied on observed summaries above, so priors can't
                        // mask "require measured" or skew mean latency.
                        max_mean_ms: None,
                        require_measured: false,
                        allow_fewer: guard.allow_fewer,
                    };
                    let mk = if is_monitored {
                        select_mab_k_monitored_explain(
                            eligible,
                            summaries_for,
                            |remaining| {
                                history.monitored_for_backends(remaining, monitor_recent_cap)
                            },
                            drift_cfg,
                            mon_cfg.clone(),
                            guardrail_cfg,
                            remaining_k,
                        )
                    } else {
                        select_mab_k_explain(
                            eligible,
                            summaries_for,
                            mon_cfg.base.clone(),
                            guardrail_cfg,
                            remaining_k,
                        )
                    };
                    let chosen = mk.chosen.clone();
                    mk_opt = Some(mk);
                    chosen
                },
            );

            let mut out = control.clone();
            out.extend(fill.chosen.clone());
            out.truncate(k);

            if fill.stopped_early && verbose {
                eprintln!(
                    "matrix-muxer: ml-only latency guardrail filtered all remaining arms (max_mean_ms={:.0}); stopping early",
                    guard.max_mean_ms.unwrap_or(0.0)
                );
            }

            if let Some(mk) = mk_opt.as_ref() {
                for (round, r) in mk.rounds.iter().enumerate() {
                    let d = &r.mab;
                    if verbose {
                        eprintln!(
                            "matrix-muxer: mab round={} remaining={} chosen={} explore_first={} constraints_fallback={}",
                            round + 1,
                            r.guardrail.eligible.len(),
                            d.selection.chosen,
                            d.explore_first,
                            d.constraints_fallback_used
                        );
                        if guard.max_mean_ms.is_some() && r.guardrail.stop_early {
                            eprintln!(
                                "matrix-muxer: ml-only latency guardrail filtered all remaining arms (max_mean_ms={:.0}); stopping early (chosen={})",
                                guard.max_mean_ms.unwrap_or(0.0),
                                round
                            );
                        }
                        if guard.max_mean_ms.is_some() && r.guardrail.fallback_used {
                            eprintln!(
                                "matrix-muxer: ml-only latency guardrail filtered all arms (max_mean_ms={:.0}); falling back",
                                guard.max_mean_ms.unwrap_or(0.0)
                            );
                        }
                        if d.constraints_fallback_used {
                            eprintln!(
                                "matrix-muxer: mab constraints filtered all arms; fell back to full set (eligible={})",
                                d.eligible_arms.len()
                            );
                        }
                        if d.explore_first {
                            eprintln!("matrix-muxer: mab explore-first chose an untried arm");
                        }
                        // Keep selection debugging bounded: show only a few candidate rows by scalar score.
                        // Keep selection debugging bounded: show only a few candidate rows by scalar score.
                        let mut rows: Vec<(f64, &CandidateDebug)> = Vec::new();
                        for c in &d.selection.candidates {
                            let drift = c.drift_score.unwrap_or(0.0);
                            let catkl = c.catkl_score.unwrap_or(0.0);
                            let cusum = c.cusum_score.unwrap_or(0.0);
                            let score = c.objective_success
                                - mon_cfg.base.cost_weight * c.mean_cost_units
                                - mon_cfg.base.latency_weight * c.mean_elapsed_ms
                                - mon_cfg.base.hard_junk_weight * c.hard_junk_rate
                                - mon_cfg.base.junk_weight * c.soft_junk_rate
                                - mon_cfg.drift_weight * drift
                                - mon_cfg.catkl_weight * catkl
                                - mon_cfg.cusum_weight * cusum;
                            rows.push((score, c));
                        }
                        rows.sort_by(|a, b| {
                            b.0.total_cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name))
                        });
                        for (score, c) in rows.into_iter().take(5) {
                            eprintln!(
                                "matrix-muxer: mab cand arm={} calls={} ok={:.2} junk={:.2} hard={:.2} mean_ms={:.0} ucb={:.2} obj_ok={:.2} score={:.2}",
                                c.name,
                                c.calls,
                                c.ok_rate,
                                c.junk_rate,
                                c.hard_junk_rate,
                                c.mean_elapsed_ms,
                                c.ucb,
                                c.objective_success,
                                score
                            );
                        }
                    }
                }
                if verbose {
                    if let Some(ref s) = mk.stop {
                        if guard.max_mean_ms.is_some() && s.guardrail.stop_early {
                            eprintln!(
                                "matrix-muxer: ml-only latency guardrail filtered all remaining arms (max_mean_ms={:.0}); stopping early (chosen={})",
                                guard.max_mean_ms.unwrap_or(0.0),
                                mk.chosen.len()
                            );
                        }
                        if guard.max_mean_ms.is_some() && s.guardrail.fallback_used {
                            eprintln!(
                                "matrix-muxer: ml-only latency guardrail filtered all arms (max_mean_ms={:.0}); falling back",
                                guard.max_mean_ms.unwrap_or(0.0)
                            );
                        }
                    }
                }
            }

            if let (Some(p), Some(mk)) = (decisions_path.as_ref(), mk_opt.as_ref()) {
                let round_logs = log_mab_k_rounds_typed(mk, decisions_top);
                let ds: Vec<String> = datasets
                    .unwrap_or(&[])
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect();
                let profile = std::env::var("ANNO_MUXER_PROFILE").ok();
                for (i, rl) in round_logs.into_iter().enumerate() {
                    let chosen_fail_kinds_top = rl.chosen.as_deref().and_then(|b| {
                        history.chosen_fail_kinds_top_for(b, datasets, per_dataset, 3)
                    });
                    let (
                        chosen_drift_score,
                        chosen_catkl_score,
                        chosen_cusum_score,
                        chosen_monitoring_penalty,
                    ) = if i < mk.rounds.len() {
                        let d = &mk.rounds[i].mab;
                        let c = d
                            .selection
                            .candidates
                            .iter()
                            .find(|c| c.name == d.selection.chosen);
                        if let Some(c) = c {
                            let mon_cfg = mh::monitored_mab_config_from_env();
                            let p = monitoring_penalty(
                                c.drift_score,
                                c.catkl_score,
                                c.cusum_score,
                                &mon_cfg,
                                mon_cfg.drift_metric,
                            );
                            (c.drift_score, c.catkl_score, c.cusum_score, Some(p))
                        } else {
                            (None, None, None, None)
                        }
                    } else {
                        (None, None, None, None)
                    };
                    let (monitoring_enabled, monitoring_fallback_used, monitoring_eligible_arms) =
                        if i < mk.rounds.len() {
                            let d = &mk.rounds[i].mab;
                            let enabled = d.drift_guard.is_some()
                                || d.catkl_guard.is_some()
                                || d.cusum_guard.is_some();
                            if !enabled {
                                (None, None, None)
                            } else if let Some(ref g) = d.cusum_guard {
                                (
                                    Some(true),
                                    Some(g.fallback_used),
                                    Some(g.eligible_arms.clone()),
                                )
                            } else if let Some(ref g) = d.catkl_guard {
                                (
                                    Some(true),
                                    Some(g.fallback_used),
                                    Some(g.eligible_arms.clone()),
                                )
                            } else if let Some(ref g) = d.drift_guard {
                                (
                                    Some(true),
                                    Some(g.fallback_used),
                                    Some(g.eligible_arms.clone()),
                                )
                            } else {
                                (Some(true), None, None)
                            }
                        } else {
                            // Stop row: no per-round monitoring decision.
                            (None, None, None)
                        };
                    append_jsonl(
                        p,
                        &DecisionLog {
                            schema_version: 6,
                            muxer_version: MUXER_VERSION.to_string(),
                            run_id: run_id.clone(),
                            strategy: "ml-only".to_string(),
                            ml_only_policy: Some(MlOnlyPolicy::Mab.id_str().to_string()),
                            slice: slice_tag.to_string(),
                            muxer_profile: profile.clone(),
                            latency_guardrail_max_mean_ms: guard
                                .max_mean_ms
                                .map(|x| x.round() as u64),
                            latency_guardrail_allow_fewer: Some(guard.allow_fewer),
                            latency_guardrail_require_measured: Some(guard.require_measured),
                            round: rl.round,
                            datasets: ds.clone(),
                            remaining: rl.remaining.clone(),
                            chosen: rl.chosen.clone(),
                            explore_first: rl.explore_first,
                            constraints_fallback_used: rl.constraints_fallback_used,
                            eligible_arms: rl.constraints_eligible_arms.clone(),
                            top_candidates: rl.top_candidates.clone(),
                            control_arms: if control.is_empty() {
                                None
                            } else {
                                Some(control.clone())
                            },
                            chosen_fail_kinds_top,
                            mab_k_round: Some(rl),
                            exp3ix_rounds: None,
                            worst_first_round: None,
                            monitoring_enabled,
                            monitoring_fallback_used,
                            monitoring_eligible_arms,
                            chosen_drift_score,
                            chosen_catkl_score,
                            chosen_cusum_score,
                            chosen_monitoring_penalty,
                        },
                    );
                }
            } else if let Some(p) = decisions_path.as_ref() {
                // Control-only or guardrail-short-circuit: still write a minimal decision row so
                // downstream audits don't collapse to "decisions=0".
                let ds: Vec<String> = datasets
                    .unwrap_or(&[])
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect();
                let profile = std::env::var("ANNO_MUXER_PROFILE").ok();
                let chosen_first = out.first().cloned();
                let chosen_fail_kinds_top = chosen_first
                    .as_deref()
                    .and_then(|b| history.chosen_fail_kinds_top_for(b, datasets, per_dataset, 3));
                append_jsonl(
                    p,
                    &DecisionLog {
                        schema_version: 6,
                        muxer_version: MUXER_VERSION.to_string(),
                        run_id: run_id.clone(),
                        strategy: "ml-only".to_string(),
                        ml_only_policy: Some(MlOnlyPolicy::Mab.id_str().to_string()),
                        slice: slice_tag.to_string(),
                        muxer_profile: profile.clone(),
                        latency_guardrail_max_mean_ms: guard.max_mean_ms.map(|x| x.round() as u64),
                        latency_guardrail_allow_fewer: Some(guard.allow_fewer),
                        latency_guardrail_require_measured: Some(guard.require_measured),
                        round: 1,
                        datasets: ds,
                        remaining: candidates_in_order.to_vec(),
                        chosen: chosen_first,
                        explore_first: None,
                        constraints_fallback_used: None,
                        eligible_arms: None,
                        top_candidates: None,
                        control_arms: if control.is_empty() {
                            None
                        } else {
                            Some(control.clone())
                        },
                        chosen_fail_kinds_top,
                        mab_k_round: None,
                        exp3ix_rounds: None,
                        worst_first_round: None,
                        monitoring_enabled: None,
                        monitoring_fallback_used: None,
                        monitoring_eligible_arms: None,
                        chosen_drift_score: None,
                        chosen_catkl_score: None,
                        chosen_cusum_score: None,
                        chosen_monitoring_penalty: None,
                    },
                );
            }
            out
        }
        SampleStrategy::WorstFirst => {
            // Worst-first selection is intentionally *not* muxer::select_mab:
            // muxer is designed to pick "good" providers. Here we want to bias toward
            // historically bad/flaky arms (to find regressions), while still exploring.
            //
            // Score: higher means "worse", with an exploration term favoring under-sampled arms.
            let wcfg = worst_first_config_from_env();
            let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
            let decisions_path = decisions_path();
            let decisions_top = mh::env_usize("ANNO_MUXER_DECISIONS_TOP", 8).max(1);
            let run_id = format!("seed={} slice={} strategy={:?}", seed, slice_tag, strategy);
            let mut remaining: Vec<String> = candidates_in_order.to_vec();
            let mut chosen: Vec<String> = Vec::new();

            for round in 0..k.min(remaining.len()) {
                let summaries = history.summaries_for(
                    prior,
                    &remaining,
                    datasets,
                    mh::env_bool("ANNO_MUXER_PER_DATASET", true),
                    prior_calls,
                );

                let per_dataset = mh::env_bool("ANNO_MUXER_PER_DATASET", true);
                let (pick, explore_first) = mh::worst_first_pick_one(
                    seed ^ 0x574F_5253 ^ (round as u64),
                    &remaining,
                    mh::WorstFirstConfig {
                        exploration_c: wcfg.exploration_c,
                        hard_weight: wcfg.hard_weight,
                        soft_weight: wcfg.soft_weight,
                    },
                    |b| history.observed_summary_for(b, datasets, per_dataset).calls,
                    |b| {
                        let s = summaries.get(b).copied().unwrap_or_default();
                        (s.calls, s.hard_junk_rate(), s.soft_junk_rate())
                    },
                )
                .unwrap_or_else(|| (remaining[round].clone(), false));

                // Keep verbose candidate debug: compute the same rows as before for printing.
                let total_calls: f64 = summaries
                    .values()
                    .map(|s| (s.calls as f64).max(1.0))
                    .sum::<f64>()
                    .max(1.0);
                let mut rows: Vec<(f64, String, Summary)> = Vec::new();
                for b in &remaining {
                    let s = summaries.get(b).copied().unwrap_or_default();
                    let calls = (s.calls as f64).max(1.0);
                    let hard_junk = s.hard_junk_rate();
                    let soft_junk = s.soft_junk_rate();
                    let exploration = wcfg.exploration_c * ((total_calls.ln() / calls).sqrt());
                    let score =
                        wcfg.hard_weight * hard_junk + wcfg.soft_weight * soft_junk + exploration;
                    rows.push((score, b.clone(), s));
                }
                rows.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

                if verbose {
                    eprintln!(
                        "matrix-muxer: worst-first round={} remaining={} chosen={} explore_first={}",
                        round + 1,
                        remaining.len(),
                        pick,
                        explore_first
                    );
                    for (score, arm, s) in rows.iter().take(5) {
                        eprintln!(
                            "matrix-muxer: worst cand arm={} calls={} ok={:.2} junk={:.2} hard={:.2} soft={:.2} score={:.2}",
                            arm,
                            s.calls,
                            s.ok_rate(),
                            s.junk_rate(),
                            s.hard_junk_rate(),
                            s.soft_junk_rate(),
                            score
                        );
                    }
                }
                if let Some(ref p) = decisions_path {
                    let top_candidates: LogTopCandidates = LogTopCandidates {
                        kind: "worst_first".to_string(),
                        rows: rows
                            .iter()
                            .take(decisions_top.max(1))
                            .map(|(score, arm, s)| LogTopCandidate {
                                arm: arm.clone(),
                                score: *score,
                                calls: Some(s.calls),
                                ok_rate: Some(s.ok_rate()),
                                junk_rate: Some(s.junk_rate()),
                                hard_junk_rate: Some(s.hard_junk_rate()),
                                mean_quality_score: s.mean_quality_score,
                            })
                            .collect(),
                    };
                    let ds: Vec<String> = datasets
                        .unwrap_or(&[])
                        .iter()
                        .map(|d| format!("{d:?}"))
                        .collect();
                    let profile = std::env::var("ANNO_MUXER_PROFILE").ok();

                    let mon_cfg = mh::monitored_mab_config_from_env();
                    let mon_enabled = monitoring_enabled(&mon_cfg);
                    let monitor_recent_cap = mh::env_usize("ANNO_MUXER_MONITOR_RECENT_CAP", 20);
                    let drift_cfg = mh::drift_config_from_env(mon_cfg.drift_metric);
                    let (
                        chosen_drift_score,
                        chosen_catkl_score,
                        chosen_cusum_score,
                        chosen_monitoring_penalty,
                    ) = if mon_enabled {
                        let (d, k, u) = monitoring_scores_for_backend(
                            history,
                            &pick,
                            monitor_recent_cap,
                            &mon_cfg,
                            drift_cfg,
                        );
                        let p = monitoring_penalty(d, k, u, &mon_cfg, mon_cfg.drift_metric);
                        (d, k, u, Some(p))
                    } else {
                        (None, None, None, None)
                    };

                    append_jsonl(
                        p,
                        &DecisionLog {
                            schema_version: 6,
                            muxer_version: MUXER_VERSION.to_string(),
                            run_id: run_id.clone(),
                            strategy: "worst-first".to_string(),
                            ml_only_policy: None,
                            slice: slice_tag.to_string(),
                            muxer_profile: profile,
                            latency_guardrail_max_mean_ms: None,
                            latency_guardrail_allow_fewer: None,
                            latency_guardrail_require_measured: None,
                            round: round + 1,
                            datasets: ds,
                            remaining: remaining.clone(),
                            chosen: Some(pick.clone()),
                            explore_first: Some(explore_first),
                            constraints_fallback_used: None,
                            eligible_arms: None,
                            top_candidates: Some(top_candidates),
                            control_arms: None,
                            chosen_fail_kinds_top: history.chosen_fail_kinds_top_for(
                                &pick,
                                datasets,
                                mh::env_bool("ANNO_MUXER_PER_DATASET", true),
                                3,
                            ),
                            mab_k_round: None,
                            exp3ix_rounds: None,
                            worst_first_round: Some(WorstFirstRoundLog {
                                remaining: remaining.clone(),
                                chosen: pick.clone(),
                                explore_first,
                                exploration_c: wcfg.exploration_c,
                                hard_weight: wcfg.hard_weight,
                                soft_weight: wcfg.soft_weight,
                                top_candidates: LogTopCandidates {
                                    kind: "worst_first".to_string(),
                                    rows: rows
                                        .iter()
                                        .take(decisions_top.max(1))
                                        .map(|(score, arm, s)| LogTopCandidate {
                                            arm: arm.clone(),
                                            score: *score,
                                            calls: Some(s.calls),
                                            ok_rate: Some(s.ok_rate()),
                                            junk_rate: Some(s.junk_rate()),
                                            hard_junk_rate: Some(s.hard_junk_rate()),
                                            mean_quality_score: s.mean_quality_score,
                                        })
                                        .collect(),
                                },
                            }),
                            monitoring_enabled: Some(mon_enabled),
                            monitoring_fallback_used: None,
                            monitoring_eligible_arms: None,
                            chosen_drift_score,
                            chosen_catkl_score,
                            chosen_cusum_score,
                            chosen_monitoring_penalty,
                        },
                    );
                }
                remaining.retain(|b| b != &pick);
                chosen.push(pick);
            }

            chosen
        }
        SampleStrategy::Estimate => {
            // Estimation-first: pick the backends with fewest observations on the
            // chosen datasets.  This is A-optimal experimental design applied to the
            // quality matrix: spread observations to minimize worst-case estimation
            // error across all cells.
            //
            // The key difference from MlOnly/WorstFirst: we don't care about
            // routing quality or regression hunting.  We care about filling the
            // matrix uniformly so every (backend, dataset) cell has a reliable F1
            // estimate.  Detection comes as a byproduct: cells with stale
            // observations are naturally prioritized.
            let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);

            // Score each backend by total observation count across the target datasets.
            // Lower count = higher priority (we want to fill the least-observed cells).
            //
            // Query from SQLite eval-history DB for accurate counts across ALL historical
            // runs, not just the muxer's sliding window.
            let cell_counts: std::collections::HashMap<(String, String), u64> = {
                let hist_path = eval_history_jsonl_path();
                crate::eval::history::EvalHistory::new(&hist_path)
                    .ok()
                    .and_then(|h| h.cell_observation_counts().ok())
                    .unwrap_or_default()
            };

            let mut scored: Vec<(u64, String)> = candidates_in_order
                .iter()
                .map(|b| {
                    let calls: u64 = datasets
                        .unwrap_or(&[])
                        .iter()
                        .map(|d| {
                            let key = (b.clone(), d.name().to_string());
                            cell_counts.get(&key).copied().unwrap_or(0)
                        })
                        .sum();
                    (calls, b.clone())
                })
                .collect();

            // Sort by observation count ascending (least-observed first).
            // Tie-break: deterministic hash for stability.
            scored.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| mh::stable_hash64(seed, &a.1).cmp(&mh::stable_hash64(seed, &b.1)))
            });

            let chosen: Vec<String> = scored.into_iter().take(k).map(|(_, b)| b).collect();

            if verbose {
                eprintln!(
                    "matrix-muxer: estimate chosen={:?} (least-observed on {:?})",
                    chosen,
                    datasets
                        .unwrap_or(&[])
                        .iter()
                        .map(|d| format!("{d:?}"))
                        .collect::<Vec<_>>()
                );
            }

            chosen
        }
    }
}

fn backend_candidates(_strategy: SampleStrategy, tasks: &[Task]) -> Vec<String> {
    // Start from what is actually feature-enabled in this build.
    let mut available: BTreeSet<String> = BackendFactory::available_backends()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    if tasks.iter().any(|t| t.is_coref_family()) {
        available.extend(
            crate::eval::backend_factory::BackendFactory::available_coref_resolvers()
                .into_iter()
                .map(|s| s.to_string()),
        );
    }

    // Keep the matrix aligned with what the evaluator will actually run:
    // TaskEvaluator filters explicit backends through `get_task_backends(task)`.
    let allowed: BTreeSet<&'static str> =
        tasks.iter().flat_map(|t| get_task_backends(*t)).collect();

    // CI policy: include ML by default (it should be exercised regularly).
    // Local policy: keep ML opt-in unless explicitly enabled.
    let mut allow_ml = match std::env::var("ANNO_ML_IN_MATRIX")
        .ok()
        .map(|v| v.trim().to_string())
    {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        Some(_) => in_ci(),
        None => in_ci(),
    };

    // If the user explicitly forced a backend list, treat that as an explicit opt-in for whatever
    // feature-enabled backends they requested (including ML-ish ones). This prevents surprising
    // “fixed backend not found in candidates” behavior on local runs.
    if std::env::var("ANNO_MUXER_FIXED_BACKEND")
        .or_else(|_| std::env::var("ANNO_MUXER_FORCE_BACKEND"))
        .ok()
        .is_some_and(|v| !v.trim().is_empty())
    {
        allow_ml = true;
    }

    // Some supported tasks have only ML-ish implementations in this repo. Allow them without requiring
    // global `ANNO_ML_IN_MATRIX=1`, otherwise they never get tested.
    let allow_task_baseline_ml = tasks.iter().any(|t| {
        matches!(
            t,
            Task::DiscontinuousNER
                | Task::EventExtraction
                | Task::TextClassification
                | Task::SpeechActClassification
                | Task::Temporal
                | Task::DiscourseRelations
                | Task::DiscourseSegmentation
        )
    });

    let mut out: Vec<String> = Vec::new();
    // Some ML backends require gated HuggingFace models. If no token is present, they will always
    // be “skipped” and waste matrix budget.
    anno::env::load_dotenv();
    let has_hf_token = anno::env::has_hf_token();
    for b in allowed {
        // gliner_poly is now implemented (bi-encoder ONNX export via DeBERTa + BGE).
        if b == "w2ner" && !has_hf_token {
            continue;
        }

        // Baselines: always include if present in this build.
        //
        // Note: `ensemble` is intentionally treated as a baseline arm here because its default
        // constructor is composed of non-ML backends (regex + heuristic) and only adds ML-ish
        // components behind feature flags (`onnx`, `candle`, etc.).
        if matches!(b, "stacked" | "crf" | "hmm" | "heuristic" | "ensemble") {
            if available.contains(b) {
                out.push(b.to_string());
            }
            continue;
        }

        // Task-specific “baseline” arms (non-ML-ish from the matrix’s point of view).
        //
        // Without these, some perspectives (e.g. relation) would have an empty candidate set
        // unless ML is explicitly enabled, which makes the harness a no-op.
        if b == "tplinker" {
            // Relation extraction is the primary consumer; include if feature-enabled.
            if available.contains(b) {
                out.push(b.to_string());
            }
            continue;
        }

        // Coref-family resolvers do not implement `Model`, so they are not returned by
        // `BackendFactory::available_backends()`. They are still valid eval “arms”.
        if b == "coref_resolver" || b == "mention_ranking" || b == "box" {
            out.push(b.to_string());
            continue;
        }

        // Everything else is treated as optional/ML-ish.
        if (allow_ml || allow_task_baseline_ml) && available.contains(b) {
            out.push(b.to_string());
        }
    }

    // Regardless of strategy, only return backends that are allowed for the requested tasks.
    //
    // Sampling policy (random vs ml-only vs worst-first) controls *how we pick*, not which
    // backends are even eligible for the task slice.
    out.sort();
    out.dedup();
    out
}

fn candidate_datasets_for_tasks(
    loader: &DatasetLoader,
    tasks: &[Task],
    require_cached: bool,
) -> Vec<DatasetId> {
    let mut candidates: Vec<DatasetId> = Vec::new();
    for &t in tasks {
        for ds in get_task_datasets(t) {
            candidates.push(ds);
        }
    }
    candidates.sort_by_key(|d| format!("{d:?}"));
    candidates.dedup();

    // Fast-profile sweeps are meant to be quick, lightweight sanity checks.
    // Filter out a small set of known-slow datasets that can OOM/timeout in debug-mode runs.
    //
    // Override: set `ANNO_MATRIX_INCLUDE_SLOW_DATASETS=1`.
    let profile = std::env::var("ANNO_MUXER_PROFILE")
        .ok()
        .unwrap_or_else(|| "off".to_string())
        .trim()
        .to_ascii_lowercase();
    let fast_profile = matches!(profile.as_str(), "fast" | "fast-strict");
    let include_slow = mh::env_bool("ANNO_MATRIX_INCLUDE_SLOW_DATASETS", false);

    let mut out: Vec<DatasetId> = Vec::new();
    let temporal_requested = tasks.contains(&Task::Temporal);
    for ds in candidates {
        // CI policy: avoid non-automatable sources unless explicitly requested.
        //
        // The registry contains many useful-but-not-directly-downloadable datasets (gated corpora,
        // dead links, “contact authors”, etc.). Those are still valuable, but running them in CI
        // wastes matrix budget and mostly produces hard-junk outcomes.
        if !mh::env_bool("ANNO_MATRIX_INCLUDE_NON_AUTOMATABLE", false)
            && !ds.is_automatable_download()
        {
            // Dataset fails the automation check (broken URL, paywall, or known-bad
            // parse like TweetNER7).  Skip it.  The is_automatable_download() hard
            // exclusion list is authoritative -- these datasets produce hard-junk
            // outcomes even when cached, so no cache fallback.
            continue;
        }
        if fast_profile
            && !include_slow
            && matches!(ds, DatasetId::OntoNotesSample | DatasetId::BioMNER)
        {
            continue;
        }
        let ts = dataset_tasks(ds);
        // Current contract: `temporal` runs through the NER-style evaluator (BIO tagging).
        // Avoid scheduling “temporal-but-not-NER” datasets (e.g., temporal RE) until a dedicated
        // temporal evaluator exists.
        if temporal_requested && ts.contains(&Task::Temporal) && !ts.contains(&Task::NER) {
            continue;
        }
        let Ok(loadable) = LoadableDatasetId::try_from(ds) else {
            continue;
        };
        if require_cached && !loader.is_cached(loadable) && !loader.s3_enabled() {
            continue;
        }
        out.push(ds);
    }
    out.sort_by_key(|d| format!("{d:?}"));
    out.dedup();
    out
}

#[test]
fn test_fast_profile_filters_known_slow_datasets() {
    let _env = env_lock();
    anno::env::load_dotenv();
    let loader = DatasetLoader::new().expect("DatasetLoader::new");

    // Guard: save+restore env so this test doesn't pollute others.
    let old_profile = std::env::var("ANNO_MUXER_PROFILE").ok();
    let old_include_slow = std::env::var("ANNO_MATRIX_INCLUDE_SLOW_DATASETS").ok();
    struct Restore {
        k: &'static str,
        v: Option<String>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.v.as_deref() {
                None => std::env::remove_var(self.k),
                Some(v) => std::env::set_var(self.k, v),
            }
        }
    }
    let _r1 = Restore {
        k: "ANNO_MUXER_PROFILE",
        v: old_profile,
    };
    let _r2 = Restore {
        k: "ANNO_MATRIX_INCLUDE_SLOW_DATASETS",
        v: old_include_slow,
    };

    std::env::set_var("ANNO_MUXER_PROFILE", "fast");
    std::env::remove_var("ANNO_MATRIX_INCLUDE_SLOW_DATASETS");
    let ds = candidate_datasets_for_tasks(&loader, &[Task::NER], false);
    assert!(
        !ds.contains(&DatasetId::OntoNotesSample),
        "fast profile should exclude OntoNotesSample by default"
    );
    assert!(
        !ds.contains(&DatasetId::BioMNER),
        "fast profile should exclude BioMNER by default"
    );

    std::env::set_var("ANNO_MATRIX_INCLUDE_SLOW_DATASETS", "1");
    let ds2 = candidate_datasets_for_tasks(&loader, &[Task::NER], false);
    assert!(
        ds2.contains(&DatasetId::OntoNotesSample),
        "include override should allow OntoNotesSample"
    );
    assert!(
        ds2.contains(&DatasetId::BioMNER),
        "include override should allow BioMNER"
    );
}

#[cfg(test)]
#[derive(Debug, Clone, serde::Serialize)]
struct DistributionReport {
    require_cached: bool,
    iterations: usize,
    chosen_task_counts: BTreeMap<String, u64>,
    chosen_dataset_counts: BTreeMap<String, u64>,
    chosen_backend_counts: BTreeMap<String, u64>,
    per_task_dataset_counts: BTreeMap<String, BTreeMap<String, u64>>,
    per_task_backend_counts: BTreeMap<String, BTreeMap<String, u64>>,
}

#[test]
fn test_matrix_distribution_report() {
    let Ok(path) = std::env::var("ANNO_MATRIX_DISTRIBUTION_REPORT") else {
        return;
    };
    let path = path.trim();
    if path.is_empty() {
        return;
    }

    anno::env::load_dotenv();
    let loader = DatasetLoader::new().expect("DatasetLoader::new");
    let require_cached = matrix_require_cached();

    let iters = mh::env_usize("ANNO_MATRIX_DISTRIBUTION_ITERS", 200).max(1);
    let datasets_per_run = mh::env_usize("ANNO_MUXER_DATASETS_PER_RUN", 2).max(1);
    let backends_per_run = mh::env_usize("ANNO_MUXER_BACKENDS_PER_RUN", 2).max(1);

    // Precompute eligibility once; this test is about distribution, not download/IO.
    let eligible_tasks = eligible_tasks_for_run(&loader, require_cached);
    if eligible_tasks.is_empty() {
        panic!(
            "matrix-dist: no eligible tasks (require_cached={} cache_dir={:?})",
            require_cached,
            loader.cache_dir()
        );
    }

    let mut task_to_datasets: BTreeMap<String, Vec<DatasetId>> = BTreeMap::new();
    let mut task_to_backends: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for &t in &eligible_tasks {
        let tag = t.code().to_string();
        task_to_datasets.insert(
            tag.clone(),
            candidate_datasets_for_tasks(&loader, &[t], require_cached),
        );
        task_to_backends.insert(
            tag.clone(),
            backend_candidates(SampleStrategy::Random, &[t]),
        );
    }

    let mut task_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut ds_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut be_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut per_task_ds: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    let mut per_task_be: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();

    for i in 0..iters {
        let seed = (ci_seed() ^ 0xD157_0000).wrapping_add(i as u64);
        let task_idx =
            (mh::stable_hash64(seed ^ 0x51EE_AAAA, "task") as usize) % eligible_tasks.len();
        let task = eligible_tasks[task_idx];
        let slice_tag = task.code().to_string();
        let datasets = task_to_datasets
            .get(&slice_tag)
            .cloned()
            .unwrap_or_default();
        let backends_all = task_to_backends
            .get(&slice_tag)
            .cloned()
            .unwrap_or_default();

        if datasets.is_empty() || backends_all.is_empty() {
            continue;
        };
        *task_counts.entry(slice_tag.clone()).or_insert(0) += 1;

        // Pick datasets uniformly within-task.
        let mut chosen_datasets: Vec<DatasetId> = Vec::new();
        for j in 0..datasets_per_run {
            let idx = (mh::stable_hash64(seed ^ 0xDADA_BEEF ^ (j as u64), "dataset") as usize)
                % datasets.len();
            chosen_datasets.push(datasets[idx]);
        }
        chosen_datasets.sort_by_key(|d| format!("{d:?}"));
        chosen_datasets.dedup();
        for ds in &chosen_datasets {
            let ds_s = format!("{ds:?}");
            *ds_counts.entry(ds_s.clone()).or_insert(0) += 1;
            *per_task_ds
                .entry(slice_tag.clone())
                .or_default()
                .entry(ds_s)
                .or_insert(0) += 1;
        }

        // Backends: filter to those compatible with the chosen datasets, then sample.
        let mut candidates = backends_all;
        candidates.retain(|b| {
            chosen_datasets
                .iter()
                .all(|d| TaskEvaluator::is_backend_compatible(b, *d))
        });
        if candidates.is_empty() {
            continue;
        }
        let picked = mh::pick_random_subset(seed ^ 0xBADC_0FFE, &candidates, backends_per_run);
        for b in &picked {
            *be_counts.entry(b.clone()).or_insert(0) += 1;
            *per_task_be
                .entry(slice_tag.clone())
                .or_default()
                .entry(b.clone())
                .or_insert(0) += 1;
        }
    }

    let report = DistributionReport {
        require_cached,
        iterations: iters,
        chosen_task_counts: task_counts,
        chosen_dataset_counts: ds_counts,
        chosen_backend_counts: be_counts,
        per_task_dataset_counts: per_task_ds,
        per_task_backend_counts: per_task_be,
    };

    let content =
        serde_json::to_string_pretty(&report).expect("distribution report JSON should serialize");
    if let Err(e) = std::fs::write(path, content) {
        panic!("failed to write ANNO_MATRIX_DISTRIBUTION_REPORT to {path}: {e}");
    }
}

fn coref_dataset_is_usable(loader: &DatasetLoader, ds: DatasetId, require_cached: bool) -> bool {
    if !ds.is_coreference() {
        return true;
    }
    if require_cached {
        loader
            .load_coref(ds)
            .map(|docs| !docs.is_empty())
            .unwrap_or(false)
    } else {
        loader
            .load_or_download_coref(ds)
            .map(|docs| !docs.is_empty())
            .unwrap_or(false)
    }
}

fn relation_dataset_is_usable(loader: &DatasetLoader, ds: DatasetId, require_cached: bool) -> bool {
    if !ds.is_relation_extraction() {
        return true;
    }
    if require_cached {
        loader
            .load_relation(ds)
            .map(|docs| !docs.is_empty())
            .unwrap_or(false)
    } else {
        loader
            .load_or_download_relation(ds)
            .map(|docs| !docs.is_empty())
            .unwrap_or(false)
    }
}

fn general_dataset_is_usable(loader: &DatasetLoader, ds: DatasetId, require_cached: bool) -> bool {
    let Ok(loadable) = LoadableDatasetId::try_from(ds) else {
        return false;
    };
    let loaded = if require_cached {
        loader.load(loadable)
    } else {
        loader.load_or_download(loadable)
    };
    match loaded {
        Ok(d) => !d.sentences.is_empty(),
        Err(_) => false,
    }
}

fn dataset_is_usable(loader: &DatasetLoader, ds: DatasetId, require_cached: bool) -> bool {
    if ds.is_relation_extraction() {
        return relation_dataset_is_usable(loader, ds, require_cached);
    }
    if ds.is_coreference() {
        return coref_dataset_is_usable(loader, ds, require_cached);
    }
    general_dataset_is_usable(loader, ds, require_cached)
}

fn choose_datasets_for_run(
    seed: u64,
    loader: &DatasetLoader,
    tasks: &[Task],
    require_cached: bool,
    datasets_per_run: usize,
) -> Vec<DatasetId> {
    let mut candidates = candidate_datasets_for_tasks(loader, tasks, require_cached);
    if candidates.is_empty() || datasets_per_run == 0 {
        return Vec::new();
    }

    // Optional facet pins for dataset selection.
    //
    // This is intentionally simple and coarse: it constrains which datasets we *sample* so muxer
    // histories don't bleed across unrelated language/domain buckets.
    fn env_csv_set(key: &str) -> Option<std::collections::BTreeSet<String>> {
        let raw = std::env::var(key).ok()?;
        let t = raw.trim();
        if t.is_empty() {
            return None;
        }
        let mut out = std::collections::BTreeSet::new();
        for part in t.split(',') {
            let s = part.trim().to_ascii_lowercase();
            if !s.is_empty() {
                out.insert(s);
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
    let pin_lang =
        env_csv_set("ANNO_MUXER_PIN_LANG").or_else(|| env_csv_set("ANNO_MUXER_FILTER_LANG"));
    let pin_dom =
        env_csv_set("ANNO_MUXER_PIN_DOMAIN").or_else(|| env_csv_set("ANNO_MUXER_FILTER_DOMAIN"));
    if pin_lang.is_some() || pin_dom.is_some() {
        // If the candidate set contains any datasets whose declared language matches the pin,
        // keep the pin strict.
        //
        // Otherwise, allow `language=mul` datasets (e.g. WikiANN) as a fallback so pinned-language
        // runs can still proceed on multilingual datasets when no monolingual dataset is loadable
        // in the current environment.
        let has_exact_lang_match = pin_lang.as_ref().is_some_and(|pins| {
            candidates
                .iter()
                .any(|d| pins.contains(&d.language().to_ascii_lowercase()))
        });
        candidates.retain(|d| {
            let d_lang = d.language().to_ascii_lowercase();
            let lang_ok = pin_lang.as_ref().is_none_or(|pins| {
                if pins.contains(&d_lang) {
                    return true;
                }
                // Fallback: allow multilingual datasets if no exact match exists.
                !has_exact_lang_match && d_lang == "mul"
            });
            let dom_ok = pin_dom
                .as_ref()
                .is_none_or(|s| s.contains(&d.domain().to_ascii_lowercase()));
            lang_ok && dom_ok
        });
    }
    if candidates.is_empty() {
        return Vec::new();
    }

    // Deterministic pseudo-random order (stable hash), then take the first K that are
    // usable under the current cache/download setting.
    //
    // This keeps selection stable while avoiding “wasted” runs on datasets that are nominally
    // registered but cannot be loaded/downloaded (or parse to empty) in this environment.
    let mut scored: Vec<(u64, DatasetId)> = candidates
        .iter()
        .copied()
        .map(|d| (mh::stable_hash64(seed ^ 0xDADA_BEEF, &format!("{d:?}")), d))
        .collect();
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| format!("{:?}", a.1).cmp(&format!("{:?}", b.1)))
    });

    let mut chosen: Vec<DatasetId> = Vec::new();
    for (_, ds) in scored {
        if chosen.len() >= datasets_per_run {
            break;
        }
        if dataset_is_usable(loader, ds, require_cached) {
            chosen.push(ds);
        }
    }
    chosen
}

#[cfg(test)]
#[derive(Debug, Clone, serde::Serialize)]
struct CoverageRow {
    task: String,
    candidate_datasets: usize,
    cached_candidate_datasets: usize,
    available_backends: Vec<String>,
    runnable_pairs: usize,
    notes: Vec<String>,
}

#[test]
fn test_matrix_coverage_report() {
    let Ok(path) = std::env::var("ANNO_MATRIX_COVERAGE_REPORT") else {
        return;
    };
    let path = path.trim();
    if path.is_empty() {
        return;
    }

    anno::env::load_dotenv();
    let loader = DatasetLoader::new().expect("DatasetLoader::new");
    let require_cached = matrix_require_cached();

    let available_models: BTreeSet<String> = BackendFactory::available_backends()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    let available_coref: BTreeSet<String> =
        crate::eval::backend_factory::BackendFactory::available_coref_resolvers()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

    let mut rows: Vec<CoverageRow> = Vec::new();
    for &task in Task::all() {
        if !TaskEvaluator::is_task_supported(task) {
            // Keep catalogued tasks visible, but mark as non-runnable until an evaluator exists.
            rows.push(CoverageRow {
                task: task.code().to_string(),
                candidate_datasets: candidate_datasets_for_tasks(&loader, &[task], false).len(),
                cached_candidate_datasets: candidate_datasets_for_tasks(&loader, &[task], true)
                    .len(),
                available_backends: Vec::new(),
                runnable_pairs: 0,
                notes: vec!["task is catalogued but not supported by TaskEvaluator".to_string()],
            });
            continue;
        }
        let is_coref = task.is_coref_family();
        let available: &BTreeSet<String> = if is_coref {
            // For coref-family tasks, include coreference resolvers which are not `Model`s.
            // (They are evaluated as “arms” by TaskEvaluator.)
            //
            // Note: We intentionally do *not* include model backends here unless they
            // explicitly advertise coref-family support in `get_task_backends(task)`.
            &available_coref
        } else {
            &available_models
        };
        let candidates = candidate_datasets_for_tasks(&loader, &[task], false);
        let cached_candidates = candidate_datasets_for_tasks(&loader, &[task], true);

        // Backends that are feature-enabled for this task (ignoring matrix gating like `ANNO_ML_IN_MATRIX`).
        let mut feature_backends: Vec<String> = get_task_backends(task)
            .iter()
            .map(|b| b.to_string())
            .filter(|b| available.contains(b))
            .collect();
        feature_backends.sort();
        feature_backends.dedup();

        // Backends that the matrix harness will actually consider under current env gating.
        let mut backends: Vec<String> = backend_candidates(SampleStrategy::Random, &[task]);
        backends.sort();
        backends.dedup();

        // Conservative notion of “runnable”: we have at least one backend *and* at least one
        // dataset candidate, filtered by the evaluator’s dataset compatibility gate.
        let mut runnable_pairs: usize = 0;
        for b in &backends {
            for &ds in &candidates {
                if require_cached {
                    let Ok(loadable) = LoadableDatasetId::try_from(ds) else {
                        continue;
                    };
                    if !loader.is_cached(loadable) {
                        continue;
                    }
                }
                if TaskEvaluator::is_backend_compatible(b, ds) {
                    runnable_pairs += 1;
                }
            }
        }

        let mut notes: Vec<String> = Vec::new();
        if backends.is_empty() {
            if !feature_backends.is_empty() {
                // Most common reasons:
                // - ML gating (`ANNO_ML_IN_MATRIX`): feature-enabled, but excluded from matrix by default.
                // - Missing HF token: some arms are gated because they'd always skip.
                if feature_backends.iter().any(|b| b == "w2ner") && !anno::env::has_hf_token() {
                    notes.push(
                        "backends exist but are gated (set HF_API_TOKEN to enable w2ner)"
                            .to_string(),
                    );
                }
                if feature_backends.iter().any(|b| {
                    matches!(
                        b.as_str(),
                        "bert_onnx"
                            | "candle_ner"
                            | "nuner"
                            | "gliner_onnx"
                            | "gliner_candle"
                            | "gliner2"
                            | "w2ner"
                            | "deberta_v3"
                            | "albert"
                            | "universal_ner"
                    )
                }) && std::env::var("ANNO_ML_IN_MATRIX")
                    .ok()
                    .is_none_or(|v| !(v == "1" || v.eq_ignore_ascii_case("true")))
                {
                    notes
                        .push("backends exist but are gated (set ANNO_ML_IN_MATRIX=1)".to_string());
                }
            } else {
                notes.push("no backends available in this feature set".to_string());
            }
        }
        if candidates.is_empty() {
            notes.push("no datasets loadable for this task".to_string());
        }
        if require_cached && cached_candidates.is_empty() {
            notes.push("no cached datasets available for this task".to_string());
        }

        rows.push(CoverageRow {
            task: task.code().to_string(),
            candidate_datasets: candidates.len(),
            cached_candidate_datasets: cached_candidates.len(),
            available_backends: backends,
            runnable_pairs,
            notes,
        });
    }

    rows.sort_by(|a, b| a.task.cmp(&b.task));
    let out = serde_json::json!({
        "require_cached": require_cached,
        "cache_dir": format!("{:?}", loader.cache_dir()),
        "rows": rows,
    });

    let content =
        serde_json::to_string_pretty(&out).expect("coverage report JSON should serialize");
    if let Err(e) = std::fs::write(path, content) {
        panic!("failed to write ANNO_MATRIX_COVERAGE_REPORT to {path}: {e}");
    }
}

/// Run one randomized muxer-backed matrix sample for the given seed.
///
/// This is used by both:
/// - unit tests (CI harness)
/// - developer tooling (`muxer_repeat`)
pub fn run_randomized_matrix_sample_with_seed(seed: u64) {
    anno::env::load_dotenv();
    let strategy = SampleStrategy::from_env();
    // Default window cap: 50 observations per arm.
    // For throughput-aware sizing use mh::suggested_window_cap(calls_per_arm, change_rate).
    let window_cap = env_usize("ANNO_MUXER_WINDOW_CAP", 50);

    let loader = DatasetLoader::new().expect("DatasetLoader::new");
    let require_cached = matrix_require_cached();

    let try_download_on_empty = mh::env_bool("ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY", false);

    let (slice_tag, tasks, require_cached_for_run) =
        if let Some((slice_tag, tasks)) = slice_for_run(seed, &loader, require_cached) {
            (slice_tag, tasks, require_cached)
        } else if require_cached && try_download_on_empty {
            // Cache-only selection yielded nothing. Fall back to “try to fetch once” so CI
            // doesn't silently become a no-op when caches are cold.
            //
            // This will still use S3 first if configured.
            let Some((slice_tag, tasks)) = slice_for_run(seed, &loader, false) else {
                eprintln!(
                    "matrix-muxer: no eligible tasks even after download fallback (cache_dir={:?})",
                    loader.cache_dir()
                );
                return;
            };
            (slice_tag, tasks, false)
        } else {
            eprintln!(
                "matrix-muxer: no eligible tasks (require_cached={} cache_dir={:?})",
                require_cached,
                loader.cache_dir()
            );
            return;
        };

    // Datasets: choose a small number of compatible datasets.
    //
    // Local default is “try to fetch”: we pay the download cost once and reuse cache later.
    // CI prefers cache, but may fall back to downloads when caches are cold.
    let datasets_per_run_default = if require_cached_for_run { 2 } else { 3 };
    let datasets_per_run =
        env_usize("ANNO_MUXER_DATASETS_PER_RUN", datasets_per_run_default).max(1);

    // Optional override: force evaluation of a specific backend (or comma-separated list).
    // When set, we also bias dataset sampling toward datasets compatible with that backend so
    // measure-mode results stay meaningful.
    //
    // Note on semantics:
    // - FIXED/FORCE is a hard override (no fallback if it yields nothing).
    // - PIN/FILTER is a soft constraint (filters candidates, but still allows selection).
    let fixed_backends_requested: Option<Vec<String>> = std::env::var("ANNO_MUXER_FIXED_BACKEND")
        .or_else(|_| std::env::var("ANNO_MUXER_FORCE_BACKEND"))
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        });
    let pinned_backends_requested: Option<Vec<String>> = std::env::var("ANNO_MUXER_PIN_BACKEND")
        .or_else(|_| std::env::var("ANNO_MUXER_FILTER_BACKEND"))
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        });
    let fixed_datasets_requested: Option<String> = std::env::var("ANNO_MUXER_FIXED_DATASETS")
        .or_else(|_| std::env::var("ANNO_MUXER_FORCE_DATASETS"))
        .ok();
    if fixed_backends_requested.is_some()
        && pinned_backends_requested.is_some()
        && mh::env_bool("ANNO_MUXER_VERBOSE", false)
    {
        eprintln!("matrix-muxer: both FIXED_BACKEND and PIN_BACKEND are set; FIXED will win for selection");
    }
    let mut chosen_datasets: Vec<DatasetId>;

    // Optional override: fixed datasets (comma-separated DatasetId debug names).
    //
    // This is the most “pandas groupby”-like pin: you are explicitly choosing the group.
    if let Some(raw) = fixed_datasets_requested.as_deref() {
        let t = raw.trim();
        if t.is_empty() {
            eprintln!("matrix-muxer: ANNO_MUXER_FIXED_DATASETS is set but empty");
            return;
        }
        let mut map: std::collections::BTreeMap<String, DatasetId> =
            std::collections::BTreeMap::new();
        for &d in DatasetId::all().iter() {
            map.insert(format!("{d:?}").to_ascii_lowercase(), d);
        }
        let mut fixed: Vec<DatasetId> = Vec::new();
        for part in t.split(',') {
            let key = part.trim().to_ascii_lowercase();
            if key.is_empty() {
                continue;
            }
            if let Some(&d) = map.get(&key) {
                fixed.push(d);
            } else {
                eprintln!("matrix-muxer: unknown dataset in ANNO_MUXER_FIXED_DATASETS: {part}");
            }
        }
        fixed.dedup();
        // Keep this harness bounded: respect `datasets_per_run` unless the user also increased it.
        if fixed.len() > datasets_per_run {
            fixed.truncate(datasets_per_run);
        }
        let mut rejected: Vec<(DatasetId, &'static str)> = Vec::new();
        let mut retained = Vec::new();
        for ds in fixed {
            if !tasks.iter().all(|t| dataset_tasks(ds).contains(t)) {
                rejected.push((
                    ds,
                    "dataset metadata does not declare the requested task(s)",
                ));
                continue;
            }
            let ok = if require_cached_for_run {
                dataset_is_usable(&loader, ds, true)
                    || (try_download_on_empty && dataset_is_usable(&loader, ds, false))
            } else {
                dataset_is_usable(&loader, ds, false)
            };
            if !ok {
                rejected.push((ds, "loader could not load or download any sentences"));
                continue;
            }
            retained.push(ds);
        }
        if retained.is_empty() {
            eprintln!("matrix-muxer: no usable fixed datasets remained after filtering");
            for (ds, why) in rejected {
                eprintln!("matrix-muxer: fixed dataset rejected: {ds:?}: {why}");
            }
            return;
        }
        chosen_datasets = retained;
    } else {
        // Default: choose a small number of compatible datasets.
        chosen_datasets = choose_datasets_for_run(
            seed,
            &loader,
            &tasks,
            require_cached_for_run,
            datasets_per_run,
        );
        if chosen_datasets.is_empty() {
            if require_cached_for_run && try_download_on_empty {
                chosen_datasets =
                    choose_datasets_for_run(seed, &loader, &tasks, false, datasets_per_run);
            }
            if chosen_datasets.is_empty() {
                eprintln!(
                    "matrix-muxer: no eligible datasets for tasks={:?} (require_cached={} cache_dir={:?})",
                    tasks,
                    require_cached_for_run,
                    loader.cache_dir()
                );
                return;
            }
        }
    }

    // Note: estimation-first dataset override is applied later, after backend
    // candidates are resolved (see below).

    if let Some(ref fixed) = fixed_backends_requested {
        if fixed.is_empty() {
            eprintln!("matrix-muxer: ANNO_MUXER_FIXED_BACKEND is set but empty");
            return;
        }
        // Prefer datasets compatible with the fixed backend(s).
        //
        // Semantics note:
        // - If the user also provided FIXED_DATASETS, that is a hard override. We must not
        //   “helpfully” resample to a different dataset group, because that defeats the point of
        //   the fixed facet.
        // - If datasets are not fixed, we can bias sampling toward compatible datasets so
        //   measure-mode results remain meaningful.
        let want = datasets_per_run;
        let also_fixed_ds = std::env::var("ANNO_MUXER_FIXED_DATASETS")
            .or_else(|_| std::env::var("ANNO_MUXER_FORCE_DATASETS"))
            .ok()
            .is_some();
        let mut filtered = chosen_datasets
            .into_iter()
            .filter(|d| {
                fixed
                    .iter()
                    .all(|b| TaskEvaluator::is_backend_compatible(b, *d))
            })
            .collect::<Vec<_>>();
        if !also_fixed_ds && filtered.len() < want {
            let mut pool = choose_datasets_for_run(
                seed ^ 0xF1E3_DA7A,
                &loader,
                &tasks,
                require_cached_for_run,
                want.saturating_mul(10).max(want),
            );
            pool.retain(|d| {
                fixed
                    .iter()
                    .all(|b| TaskEvaluator::is_backend_compatible(b, *d))
            });
            pool.truncate(want.min(pool.len()));
            filtered = pool;
        } else {
            filtered.truncate(want);
        }
        if filtered.is_empty() {
            let msg = if also_fixed_ds {
                "matrix-muxer: fixed backend + fixed datasets conflict (no compatible datasets remain)"
            } else {
                "matrix-muxer: fixed backend set but no compatible datasets remain"
            };
            eprintln!("{msg} tasks={tasks:?} fixed_backend={fixed:?}");
            return;
        }
        chosen_datasets = filtered;
    }

    // Make muxer history facet-aware by scoping to a coarse dataset slice.
    //
    // This does NOT change task/dataset sampling; it only changes how muxer learns and selects
    // backends, so different domains/languages don't collapse into one average history.
    //
    // Default: enabled. Set `ANNO_MUXER_SLICE_BY_DATASET_FACETS=0` to revert to task-only.
    let mut slice_tag_for_muxer = mh::muxer_slice_tag(
        &slice_tag,
        &chosen_datasets,
        mh::env_bool("ANNO_MUXER_SLICE_BY_DATASET_FACETS", true),
    )
    .unwrap_or_else(|e| {
        eprintln!("matrix-muxer: invalid muxer slice tag: {e} (falling back to task-only)");
        // `slice_tag` here is a task code (e.g. `ner`) and should always parse.
        mh::SliceTag::parse(&slice_tag).unwrap()
    })
    .to_string();

    // If the user pinned exactly one language/domain, but the chosen dataset is multilingual
    // (`language=mul`) or otherwise mixed, it is still useful to scope muxer history to the
    // pinned facet (so runs don't bleed into the global "mul" bucket).
    //
    // This is only a history-key choice; it does not change what examples are evaluated.
    if mh::env_bool("ANNO_MUXER_SLICE_BY_DATASET_FACETS", true) {
        fn env_single_slug(keys: &[&str]) -> Option<String> {
            for &k in keys {
                if let Ok(raw) = std::env::var(k) {
                    let mut parts = raw
                        .split(',')
                        .map(|s| s.trim().to_ascii_lowercase())
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>();
                    parts.sort();
                    parts.dedup();
                    if parts.len() == 1 {
                        return Some(parts[0].clone());
                    }
                }
            }
            None
        }

        let pin_lang = env_single_slug(&["ANNO_MUXER_PIN_LANG", "ANNO_MUXER_FILTER_LANG"]);
        let pin_dom = env_single_slug(&["ANNO_MUXER_PIN_DOMAIN", "ANNO_MUXER_FILTER_DOMAIN"]);

        if pin_lang.is_some() || pin_dom.is_some() {
            let mut langs: Vec<&'static str> =
                chosen_datasets.iter().map(|d| d.language()).collect();
            langs.sort();
            langs.dedup();
            let mut doms: Vec<&'static str> = chosen_datasets.iter().map(|d| d.domain()).collect();
            doms.sort();
            doms.dedup();

            let lang_is_ambiguous = langs.len() != 1 || langs[0].eq_ignore_ascii_case("mul");
            let dom_is_ambiguous = doms.len() != 1;
            if (lang_is_ambiguous && pin_lang.is_some()) || (dom_is_ambiguous && pin_dom.is_some())
            {
                let lang = pin_lang
                    .unwrap_or_else(|| langs.first().copied().unwrap_or("unknown").to_string());
                let dom = pin_dom
                    .unwrap_or_else(|| doms.first().copied().unwrap_or("unknown").to_string());
                let tagged = format!("{}.lang={}.dom={}", slice_tag, lang, dom);
                if let Ok(st) = mh::SliceTag::parse(&tagged) {
                    slice_tag_for_muxer = st.to_string();
                }
            }
        }
    }

    let hist_path = history_path(&slice_tag_for_muxer);
    let mut history = BackendHistory::load(&hist_path, window_cap);
    let prior_hist_path = history_path(&slice_tag);
    let prior_history = if prior_hist_path != hist_path {
        Some(BackendHistory::load(&prior_hist_path, window_cap))
    } else {
        None
    };
    let prior_calls = mh::prior_calls_from_env();

    // Backends: choose a *tiny* subset per run (keep this test fast).
    //
    // Important: choose datasets first so MAB summaries can be dataset-scoped, reducing
    // objective bleed across unrelated datasets.
    let mut candidates = backend_candidates(strategy, &tasks);
    if candidates.is_empty() {
        eprintln!("matrix-muxer: no backend candidates available for this feature set");
        return;
    }

    // Filter backend candidates down to those that can actually serve the chosen tasks.
    candidates.retain(|b| {
        let ts = backend_tasks(b);
        tasks.iter().any(|t| ts.contains(t))
    });
    // Further filter by dataset-level compatibility (entity types, etc.) to avoid wasting
    // matrix budget on expected incompatibilities (e.g., heuristic on WNUT-17).
    candidates.retain(|b| {
        chosen_datasets
            .iter()
            .all(|d| TaskEvaluator::is_backend_compatible(b, *d))
    });
    if let Some(ref pinned) = pinned_backends_requested {
        if pinned.is_empty() {
            eprintln!("matrix-muxer: ANNO_MUXER_PIN_BACKEND is set but empty");
            return;
        }
        candidates.retain(|b| pinned.iter().any(|p| p == b));
    }
    if candidates.is_empty() {
        eprintln!(
            "matrix-muxer: no backend candidates support tasks={:?} for this feature set",
            tasks
        );
        return;
    }

    // Estimation-first: override dataset selection to prioritize least-observed datasets.
    //
    // For the Estimate strategy, query the SQLite eval-history DB for observation counts
    // (the ground truth for matrix coverage), then pick the datasets with fewest
    // total observations across all backends.  Falls back to muxer history if DB unavailable.
    if matches!(strategy, SampleStrategy::Estimate) && fixed_datasets_requested.is_none() {
        let all_ds = candidate_datasets_for_tasks(&loader, &tasks, require_cached_for_run);
        if !all_ds.is_empty() {
            // Try to get counts from SQLite eval-history DB (fast, accurate, uses all
            // historical data).  This is the ground truth for matrix coverage.
            // Path resolution uses ANNO_EVAL_HISTORY → ANNO_CACHE_DIR → platform cache dir
            // so CI writes and reads from the same cached location.
            let db_counts: std::collections::HashMap<String, u64> = {
                let hist_path = eval_history_jsonl_path();
                crate::eval::history::EvalHistory::new(&hist_path)
                    .ok()
                    .and_then(|h| h.dataset_observation_counts().ok())
                    .unwrap_or_default()
            };

            let mut scored: Vec<(u64, DatasetId)> = all_ds
                .iter()
                .map(|&d| {
                    let name = d.name().to_string();
                    let total = db_counts.get(&name).copied().unwrap_or(0);
                    (total, d)
                })
                .collect();
            scored.sort_by(|a, b| {
                a.0.cmp(&b.0).then_with(|| {
                    mh::stable_hash64(seed ^ 0xE571, &format!("{:?}", a.1))
                        .cmp(&mh::stable_hash64(seed ^ 0xE571, &format!("{:?}", b.1)))
                })
            });
            let override_ds: Vec<DatasetId> = scored
                .into_iter()
                .take(datasets_per_run)
                .map(|(_, d)| d)
                .collect();
            if !override_ds.is_empty() {
                if mh::env_bool("ANNO_MUXER_VERBOSE", false) {
                    eprintln!(
                        "matrix-muxer: estimate dataset override: {:?} -> {:?} (least-observed in eval DB)",
                        chosen_datasets.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
                        override_ds.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
                    );
                }
                chosen_datasets = override_ds;
                // Re-filter candidates for compatibility with the new datasets.
                candidates.retain(|b| {
                    chosen_datasets
                        .iter()
                        .all(|d| TaskEvaluator::is_backend_compatible(b, *d))
                });
            }
        }
    }

    let per_dataset = mh::env_bool("ANNO_MUXER_PER_DATASET", true);
    let backends_per_run_default = if matches!(MlOnlyPolicy::from_env(), MlOnlyPolicy::LinUcb) {
        // LinUCB benefits from more arms per run: each observation updates the
        // per-arm ridge regression, accelerating convergence.
        if require_cached {
            3
        } else {
            4
        }
    } else if require_cached {
        2
    } else {
        3
    };
    let backends_per_run =
        mh::env_usize("ANNO_MUXER_BACKENDS_PER_RUN", backends_per_run_default).max(1);

    let mut exp3ix_state_for_update: Option<Exp3IxState> = None;
    let mut exp3ix_tickets_for_update: Vec<(String, f64)> = Vec::new();
    let mut linucb_for_update: Option<(muxer::LinUcb, [f64; mh::CONTEXT_DIM])> = None;
    let mut outcome_run_id_override: Option<String> = None;
    let mut outcome_strategy_override: Option<String> = None;
    let chosen_backends = if let Some(ref fixed) = fixed_backends_requested {
        // Force evaluation of a specific backend (or ordered list), while still using the muxer
        // sampler for task/dataset selection and history reporting.
        let mut chosen: Vec<String> = candidates
            .iter()
            .filter(|c| fixed.iter().any(|f| f == *c))
            .cloned()
            .collect();
        chosen.truncate(backends_per_run.min(chosen.len()));
        if chosen.is_empty() {
            eprintln!(
                "matrix-muxer: ANNO_MUXER_FIXED_BACKEND is set but no matches in candidates (fixed={fixed:?})"
            );
            return;
        }

        if mh::env_bool("ANNO_MUXER_VERBOSE", false) {
            eprintln!("matrix-muxer: fixed_backend_override chosen={chosen:?}");
        }
        if let Ok(p) = std::env::var("ANNO_MUXER_DECISIONS_FILE") {
            let t = p.trim();
            if !t.is_empty() {
                let ds: Vec<String> = chosen_datasets.iter().map(|d| format!("{d:?}")).collect();
                let run_id = format!(
                    "seed={} slice={} strategy=fixed fixed={}",
                    seed,
                    slice_tag_for_muxer,
                    fixed.join(",")
                );
                outcome_run_id_override = Some(run_id.clone());
                outcome_strategy_override = Some("fixed".to_string());
                append_jsonl(
                    t,
                    &DecisionLog {
                        schema_version: 6,
                        muxer_version: MUXER_VERSION.to_string(),
                        run_id,
                        strategy: "fixed".to_string(),
                        ml_only_policy: None,
                        slice: slice_tag_for_muxer.to_string(),
                        muxer_profile: std::env::var("ANNO_MUXER_PROFILE").ok(),
                        latency_guardrail_max_mean_ms: None,
                        latency_guardrail_allow_fewer: None,
                        latency_guardrail_require_measured: None,
                        round: 1,
                        datasets: ds,
                        remaining: candidates.clone(),
                        chosen: chosen.first().cloned(),
                        explore_first: None,
                        constraints_fallback_used: None,
                        eligible_arms: None,
                        top_candidates: None,
                        control_arms: None,
                        chosen_fail_kinds_top: chosen.first().and_then(|b| {
                            history.chosen_fail_kinds_top_for(
                                b,
                                Some(&chosen_datasets),
                                mh::env_bool("ANNO_MUXER_PER_DATASET", true),
                                3,
                            )
                        }),
                        mab_k_round: None,
                        exp3ix_rounds: None,
                        worst_first_round: None,
                        monitoring_enabled: None,
                        monitoring_fallback_used: None,
                        monitoring_eligible_arms: None,
                        chosen_drift_score: None,
                        chosen_catkl_score: None,
                        chosen_cusum_score: None,
                        chosen_monitoring_penalty: None,
                    },
                );
            }
        }

        chosen
    } else if matches!(strategy, SampleStrategy::MlOnly)
        && matches!(MlOnlyPolicy::from_env(), MlOnlyPolicy::Exp3Ix)
    {
        let profile = std::env::var("ANNO_MUXER_PROFILE").ok();

        // Build eligible set + decide under latency guardrail in muxer (single shared semantics).
        let guard = mh::latency_guardrail_from_env();
        // Optional selection-bias anchor: reserve K deterministic-random control picks.
        let (control, remaining_k) = mh::split_control_budget(
            seed,
            &candidates,
            backends_per_run,
            mh::ControlConfig::with_k(mh::control_k_from_env()),
        );
        let mut candidates_for_policy = candidates.clone();
        candidates_for_policy.retain(|b| !control.contains(b));

        let decision_seed = mh::stable_hash64(seed, &format!("anno-exp3ix:{slice_tag_for_muxer}"));
        let mut exp3ix_explain: Option<Exp3IxKExplain> = None;
        let mut exp3ix_monitoring: Option<(bool, bool, Vec<String>)> = None;
        let fill = mh::policy_fill_k_observed_with(
            seed ^ 0xE8D3_1A00,
            &candidates_for_policy,
            remaining_k,
            // EXP3-IX already has an "explore-first" phase (based on persisted `uses`).
            //
            // Harness-level novelty pre-chooses unseen arms *outside* the EXP3-IX decision path,
            // which means we cannot log probabilities nor update EXP3-IX state for those picks.
            // That weakens convergence and produces confusing audit gaps.
            false,
            guard,
            |b| {
                let s = history.observed_summary_for(b, Some(&chosen_datasets), per_dataset);
                (s.calls, s.elapsed_ms_sum)
            },
            |eligible, remaining_k| {
                // Build summaries (may include priors) for the eligible set.
                let summaries = history.summaries_for(
                    prior_history.as_ref(),
                    eligible,
                    Some(&chosen_datasets),
                    per_dataset,
                    prior_calls,
                );

                // Optional monitoring guardrails for EXP3-IX:
                // EXP3-IX is stochastic and reward-driven; we apply monitoring as:
                // - hard filters when `max_*` thresholds are set
                // - soft penalties at update-time when `*_weight > 0` (see below)
                let monitor_cfg = mh::monitored_mab_config_from_env();
                let mon_enabled = monitoring_enabled(&monitor_cfg);
                let monitor_recent_cap = mh::env_usize("ANNO_MUXER_MONITOR_RECENT_CAP", 20);
                let drift_cfg = mh::drift_config_from_env(monitor_cfg.drift_metric);
                let (eligible_after_monitor, monitor_fallback_used) = if mon_enabled
                    && (monitor_cfg.max_drift.is_some()
                        || monitor_cfg.max_catkl.is_some()
                        || monitor_cfg.max_cusum.is_some())
                {
                    let monitored = history.monitored_for_backends(eligible, monitor_recent_cap);
                    let mut kept: Vec<String> = Vec::new();
                    for a in eligible {
                        let Some(w) = monitored.get(a) else {
                            kept.push(a.clone());
                            continue;
                        };

                        let drift = muxer::monitor::drift_between_windows(
                            w.baseline(),
                            w.recent(),
                            muxer::DriftConfig {
                                metric: monitor_cfg.drift_metric,
                                ..drift_cfg
                            },
                        )
                        .map(|d| d.score);
                        let catkl = muxer::monitor::catkl_score_between_windows(
                            w.baseline(),
                            w.recent(),
                            monitor_cfg.catkl_alpha,
                            drift_cfg.tol,
                            monitor_cfg.catkl_min_baseline,
                            monitor_cfg.catkl_min_recent,
                        );
                        let cusum = muxer::monitor::cusum_score_between_windows(
                            w.baseline(),
                            w.recent(),
                            monitor_cfg.cusum_alpha,
                            drift_cfg.tol,
                            monitor_cfg.cusum_min_baseline,
                            monitor_cfg.cusum_min_recent,
                            monitor_cfg.cusum_alt_p,
                        );

                        let violates = monitor_cfg
                            .max_drift
                            .map(|thr| drift.map(|x| x > thr).unwrap_or(false))
                            .unwrap_or(false)
                            || monitor_cfg
                                .max_catkl
                                .map(|thr| catkl.map(|x| x > thr).unwrap_or(false))
                                .unwrap_or(false)
                            || monitor_cfg
                                .max_cusum
                                .map(|thr| cusum.map(|x| x > thr).unwrap_or(false))
                                .unwrap_or(false);

                        if !violates {
                            kept.push(a.clone());
                        }
                    }
                    let fallback_used = kept.is_empty();
                    let eligible_arms = if fallback_used {
                        eligible.to_vec()
                    } else {
                        kept
                    };
                    (eligible_arms, fallback_used)
                } else {
                    (eligible.to_vec(), false)
                };

                exp3ix_monitoring = Some((
                    mon_enabled,
                    monitor_fallback_used,
                    eligible_after_monitor.clone(),
                ));
                if mh::env_bool("ANNO_MUXER_VERBOSE", false) && mon_enabled {
                    eprintln!(
                        "matrix-muxer: exp3ix monitoring enabled (eligible_before={} eligible_after={} fallback={})",
                        eligible.len(),
                        eligible_after_monitor.len(),
                        monitor_fallback_used
                    );
                }

                let ex = exp3ix_decide_k_guardrailed(
                    exp3ix_config_from_env(seed),
                    history.exp3ix_state.clone(),
                    &eligible_after_monitor,
                    &summaries,
                    mh::LatencyGuardrailConfig {
                        // Guardrail is applied above on observed stats.
                        max_mean_ms: None,
                        require_measured: false,
                        allow_fewer: guard.allow_fewer,
                    },
                    remaining_k,
                    decision_seed,
                );

                let picked = ex.chosen.clone();
                assert!(
                    !picked.is_empty() || guard.allow_fewer,
                    "exp3ix selection returned no arms; eligible={}",
                    eligible.len()
                );

                // Keep the pre-update state (includes probs used for sampling).
                exp3ix_state_for_update = Some(ex.state.clone());
                exp3ix_tickets_for_update = ex
                    .rounds
                    .iter()
                    .map(|r| (r.decision.chosen.clone(), r.prob_used))
                    .collect();

                exp3ix_explain = Some(ex);
                picked
            },
        );
        let control_for_log = control.clone();
        let mut chosen = control;
        chosen.extend(fill.chosen);
        chosen.truncate(backends_per_run);

        let explore_first = exp3ix_explain
            .as_ref()
            .and_then(|ex| ex.rounds.first())
            .map(|r| {
                r.decision
                    .notes
                    .iter()
                    .any(|n| matches!(n, DecisionNote::ExploreFirst))
            })
            .unwrap_or(false);

        // Optional decision logging (bounded).
        if mh::env_bool("ANNO_MUXER_VERBOSE", false) {
            let eligible = exp3ix_explain
                .as_ref()
                .and_then(|ex| ex.rounds.first())
                .map(|r| r.guardrail.eligible.len())
                .unwrap_or(0);
            eprintln!(
                "matrix-muxer: exp3ix chosen={:?} explore_first={} arms={} profile={}",
                chosen,
                explore_first,
                eligible,
                profile.clone().unwrap_or_else(|| "off".to_string())
            );
        }
        let decisions_path = decisions_path();
        if let Some(ref p) = decisions_path {
            let decisions_top = mh::env_usize("ANNO_MUXER_DECISIONS_TOP", 8).max(1);
            let run_id = format!(
                "seed={} slice={} strategy={:?}",
                seed, slice_tag_for_muxer, strategy
            );
            let ds: Vec<String> = chosen_datasets.iter().map(|d| format!("{d:?}")).collect();
            if let Some(ex) = exp3ix_explain.as_ref() {
                let exp3ix_rounds = log_exp3ix_k_rounds_typed(ex, decisions_top);
                let eligible_arms = ex
                    .rounds
                    .first()
                    .map(|r| r.guardrail.eligible.clone())
                    .unwrap_or_default();
                let (
                    chosen_drift_score,
                    chosen_catkl_score,
                    chosen_cusum_score,
                    chosen_monitoring_penalty,
                ) = if let Some(chosen_backend) = chosen.first() {
                    let mon_cfg = mh::monitored_mab_config_from_env();
                    let mon_enabled = monitoring_enabled(&mon_cfg);
                    if mon_enabled {
                        let monitor_recent_cap = mh::env_usize("ANNO_MUXER_MONITOR_RECENT_CAP", 20);
                        let drift_cfg = mh::drift_config_from_env(mon_cfg.drift_metric);
                        let (d, k, u) = monitoring_scores_for_backend(
                            &history,
                            chosen_backend,
                            monitor_recent_cap,
                            &mon_cfg,
                            drift_cfg,
                        );
                        let p = monitoring_penalty(d, k, u, &mon_cfg, mon_cfg.drift_metric);
                        (d, k, u, Some(p))
                    } else {
                        (None, None, None, None)
                    }
                } else {
                    (None, None, None, None)
                };
                append_jsonl(
                    p,
                    &DecisionLog {
                        schema_version: 6,
                        muxer_version: MUXER_VERSION.to_string(),
                        run_id,
                        strategy: "ml-only".to_string(),
                        ml_only_policy: Some(MlOnlyPolicy::Exp3Ix.id_str().to_string()),
                        slice: slice_tag_for_muxer.to_string(),
                        muxer_profile: profile.clone(),
                        // Guardrail is applied on observed stats above; keep logging consistent with env.
                        latency_guardrail_max_mean_ms: guard.max_mean_ms.map(|x| x.round() as u64),
                        latency_guardrail_allow_fewer: Some(guard.allow_fewer),
                        latency_guardrail_require_measured: Some(guard.require_measured),
                        round: 1,
                        datasets: ds,
                        remaining: eligible_arms.clone(),
                        chosen: chosen.first().cloned(),
                        explore_first: Some(explore_first),
                        constraints_fallback_used: None,
                        eligible_arms: Some(eligible_arms),
                        top_candidates: exp3ix_rounds
                            .first()
                            .and_then(|r| r.top_candidates.clone()),
                        control_arms: if control_for_log.is_empty() {
                            None
                        } else {
                            Some(control_for_log)
                        },
                        chosen_fail_kinds_top: chosen.first().and_then(|b| {
                            history.chosen_fail_kinds_top_for(
                                b,
                                Some(&chosen_datasets),
                                per_dataset,
                                3,
                            )
                        }),
                        mab_k_round: None,
                        exp3ix_rounds: Some(exp3ix_rounds),
                        worst_first_round: None,
                        monitoring_enabled: exp3ix_monitoring.as_ref().map(|x| x.0),
                        monitoring_fallback_used: exp3ix_monitoring.as_ref().map(|x| x.1),
                        monitoring_eligible_arms: exp3ix_monitoring.as_ref().map(|x| x.2.clone()),
                        chosen_drift_score,
                        chosen_catkl_score,
                        chosen_cusum_score,
                        chosen_monitoring_penalty,
                    },
                );
            } else {
                // Control-only or guardrail-short-circuit: still write a minimal decision row so
                // downstream audits don't collapse to "decisions=0".
                let chosen_first = chosen.first().cloned();
                let chosen_fail_kinds_top = chosen_first.as_deref().and_then(|b| {
                    history.chosen_fail_kinds_top_for(b, Some(&chosen_datasets), per_dataset, 3)
                });
                append_jsonl(
                    p,
                    &DecisionLog {
                        schema_version: 6,
                        muxer_version: MUXER_VERSION.to_string(),
                        run_id,
                        strategy: "ml-only".to_string(),
                        ml_only_policy: Some(MlOnlyPolicy::Exp3Ix.id_str().to_string()),
                        slice: slice_tag_for_muxer.to_string(),
                        muxer_profile: profile.clone(),
                        latency_guardrail_max_mean_ms: guard.max_mean_ms.map(|x| x.round() as u64),
                        latency_guardrail_allow_fewer: Some(guard.allow_fewer),
                        latency_guardrail_require_measured: Some(guard.require_measured),
                        round: 1,
                        datasets: ds,
                        remaining: candidates.clone(),
                        chosen: chosen_first,
                        explore_first: None,
                        constraints_fallback_used: None,
                        eligible_arms: None,
                        top_candidates: None,
                        control_arms: if control_for_log.is_empty() {
                            None
                        } else {
                            Some(control_for_log.clone())
                        },
                        chosen_fail_kinds_top,
                        mab_k_round: None,
                        exp3ix_rounds: None,
                        worst_first_round: None,
                        monitoring_enabled: exp3ix_monitoring.as_ref().map(|x| x.0),
                        monitoring_fallback_used: exp3ix_monitoring.as_ref().map(|x| x.1),
                        monitoring_eligible_arms: exp3ix_monitoring.as_ref().map(|x| x.2.clone()),
                        chosen_drift_score: None,
                        chosen_catkl_score: None,
                        chosen_cusum_score: None,
                        chosen_monitoring_penalty: None,
                    },
                );
            }
        }
        chosen
    } else if matches!(strategy, SampleStrategy::MlOnly)
        && matches!(MlOnlyPolicy::from_env(), MlOnlyPolicy::LinUcb)
    {
        // Contextual bandit (LinUCB): route based on dataset-derived feature vectors.
        //
        // Key design choice: LinUCB state is stored in a **global** history file
        // (not per-slice), so it accumulates observations across all task/language/domain
        // slices.  This is what breaks the non-contextual collapse: the theta vectors
        // learn from biomedical AND social media AND news contexts simultaneously,
        // producing genuinely different weights per arm per feature dimension.
        //
        // Without cross-slice learning, the theta vectors collapse to rank 1 (all
        // proportional) because all observations come from the same feature regime.
        let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
        let guard = mh::latency_guardrail_from_env();
        let context = mh::context_features(&chosen_datasets);

        let linucb_alpha = mh::env_f64("ANNO_MUXER_LINUCB_ALPHA", 1.0);
        let linucb_lambda = mh::env_f64("ANNO_MUXER_LINUCB_LAMBDA", 1.0);
        let linucb_decay = mh::env_f64("ANNO_MUXER_LINUCB_DECAY", 0.98);
        let cfg = muxer::LinUcbConfig {
            dim: mh::CONTEXT_DIM,
            alpha: linucb_alpha,
            lambda: linucb_lambda,
            seed,
            decay: linucb_decay,
        };
        let mut linucb = muxer::LinUcb::new(cfg);

        // Restore from GLOBAL LinUCB state (cross-slice learning).
        // This is separate from the per-slice BackendHistory: the per-slice file
        // tracks outcomes for MAB/EXP3-IX, while the global file accumulates the
        // LinUCB ridge-regression state across all slices.
        let linucb_global_path = linucb_global_state_path();
        if let Some(st) = load_linucb_global_state(&linucb_global_path) {
            linucb.restore(st.clone());
            if verbose {
                eprintln!(
                    "matrix-muxer: linucb restored GLOBAL state with {} arms from {}",
                    st.arms.len(),
                    linucb_global_path.display()
                );
            }
        }

        let result = muxer::policy_fill_k_contextual(
            seed ^ 0x4C55_4342, // "LUCB"
            &candidates,
            backends_per_run,
            // Disable harness-level novelty for LinUCB: the UCB bonus already
            // handles exploration (untried arms have high bonus).  Harness
            // novelty pre-picks arms by dataset-scoped call count, which
            // overrides LinUCB's learned scores and slows convergence.
            false,
            guard,
            &mut linucb,
            &context,
            |b| {
                let s = history.observed_summary_for(b, Some(&chosen_datasets), per_dataset);
                (s.calls, s.elapsed_ms_sum)
            },
        );

        if verbose {
            eprintln!(
                "matrix-muxer: linucb chosen={:?} context={:?} arms={} profile=contextual",
                result.fill.chosen,
                context
                    .iter()
                    .map(|x| format!("{:.2}", x))
                    .collect::<Vec<_>>(),
                candidates.len()
            );
            for (arm, (ucb, mean, bonus)) in &result.scores {
                eprintln!(
                    "matrix-muxer: linucb score arm={} ucb={:.3} mean={:.3} bonus={:.3}",
                    arm, ucb, mean, bonus
                );
            }
        }

        // Store the linucb instance + context for reward update after evaluation.
        linucb_for_update = Some((linucb, context));

        result.fill.chosen
    } else if matches!(strategy, SampleStrategy::MlOnly)
        && matches!(MlOnlyPolicy::from_env(), MlOnlyPolicy::Mab)
    {
        // Fall back to legacy deterministic MAB selection (may return K backends).
        if per_dataset {
            select_backends(
                strategy,
                seed,
                &slice_tag_for_muxer,
                &history,
                prior_history.as_ref(),
                &candidates,
                Some(&chosen_datasets),
                backends_per_run,
                prior_calls,
            )
        } else {
            select_backends(
                strategy,
                seed,
                &slice_tag_for_muxer,
                &history,
                prior_history.as_ref(),
                &candidates,
                None,
                backends_per_run,
                prior_calls,
            )
        }
    } else if per_dataset {
        select_backends(
            strategy,
            seed,
            &slice_tag_for_muxer,
            &history,
            prior_history.as_ref(),
            &candidates,
            Some(&chosen_datasets),
            backends_per_run,
            prior_calls,
        )
    } else {
        select_backends(
            strategy,
            seed,
            &slice_tag_for_muxer,
            &history,
            prior_history.as_ref(),
            &candidates,
            None,
            backends_per_run,
            prior_calls,
        )
    };

    // Degenerate-slice diagnostics: log when the bandit machinery is vacuous.
    if chosen_backends.len() <= 1 {
        eprintln!(
            "matrix-muxer: WARN degenerate slice: |arms|={} for slice={} -- bandit selection is vacuous (single arm, nothing to learn)",
            chosen_backends.len(),
            slice_tag_for_muxer
        );
    }
    if candidates.len() <= 1 && chosen_backends.len() <= 1 {
        eprintln!(
            "matrix-muxer: WARN only {} candidate backend(s) available for this task/feature set -- consider enabling more features (e.g. --features onnx)",
            candidates.len()
        );
    }

    let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
    if verbose {
        eprintln!(
            "matrix-muxer: slice={} strategy={:?} seed={} per_dataset={} datasets={:?} backends={:?}",
            slice_tag_for_muxer,
            strategy,
            seed,
            per_dataset,
            chosen_datasets,
            chosen_backends
        );
        let profile = std::env::var("ANNO_MUXER_PROFILE")
            .ok()
            .unwrap_or_else(|| "off".to_string());
        let guard = mh::latency_guardrail_from_env();
        if guard.max_mean_ms.is_some()
            || profile.trim().to_lowercase() != "off"
            || std::env::var("ANNO_MUXER_MAX_MEAN_ELAPSED_MS").is_ok()
        {
            eprintln!(
                "matrix-muxer: latency_guardrail profile={} max_mean_ms={:?} allow_fewer={} require_measured={}",
                profile,
                guard.max_mean_ms.map(|x| x.round() as u64),
                guard.allow_fewer,
                guard.require_measured
            );
        }
    }

    let eval = TaskEvaluator::new().expect("TaskEvaluator::new");

    let config = TaskEvalConfig {
        tasks,
        datasets: chosen_datasets,
        backends: chosen_backends.clone(),
        max_examples: Some(max_examples_per_dataset()),
        seed: Some(seed),
        require_cached,
        relation_threshold: 0.5,
        robustness: false,
        compute_familiarity: false,
        temporal_stratification: false,
        confidence_intervals: false,
        custom_coref_resolver: None,
        coref_use_gold_mentions: std::env::var("ANNO_COREF_GOLD")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
    };

    let results = match eval.evaluate_all(config) {
        Ok(r) => r,
        Err(e) => {
            // If the requested slice has no valid combinations, treat as non-fatal.
            eprintln!("matrix-muxer: evaluation returned error: {}", e);
            return;
        }
    };

    // Update per-backend muxer windows.
    //
    // Record outcomes into per-backend windows.
    //
    // Muxer semantics:
    // - Outcome.ok: "request succeeded" (here: evaluation succeeded)
    // - Outcome.junk: low-signal output (here: very low F1)
    // - Outcome.hard_junk: hard failure (here: evaluation failed)
    for r in &results.results {
        // Don’t poison muxer history with “skips” (feature-gated, incompatible, etc.).
        // These are not actionable regressions for the routing policy; they’re configuration.
        if r.is_skipped() {
            if mh::env_bool("ANNO_MUXER_VERBOSE", false) {
                let err = r
                    .error
                    .as_deref()
                    .map(|s| trunc(s, 200))
                    .unwrap_or_else(|| "-".to_string());
                eprintln!(
                    "matrix-muxer: skipped task={:?} dataset={:?} backend={} err={}",
                    r.task, r.dataset, r.backend, err
                );
            }
            continue;
        }

        // Prefer task-appropriate primary F1:
        // - NER: "f1"
        // - Coref/CDCR: "conll_f1"
        // - Relation: "strict_f1"
        let f1 = r.primary_f1().unwrap_or(0.0);
        let thr = junk_f1_threshold(r.task);
        let dur_ms = r.duration_ms.unwrap_or(0.0).max(0.0);

        // Outcome mapping:
        // - ok: “succeeded AND not junk” (so ok_rate reflects both stability and quality)
        // - junk: “low-signal output” (low F1) OR hard failure
        // - hard_junk: “hard failure” (evaluation failed)
        let hard_junk = !r.success;
        let junk = hard_junk || f1 < thr;
        let ok = r.success && !junk;

        let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
        if verbose {
            let err = r
                .error
                .as_deref()
                .map(|s| trunc(s, 200))
                .unwrap_or_else(|| "-".to_string());
            let rel_counts = if matches!(r.task, Task::RelationExtraction) {
                let gold = r.metrics.get("num_gold_relations").copied().unwrap_or(0.0) as u64;
                let pred = r
                    .metrics
                    .get("num_predicted_relations")
                    .copied()
                    .unwrap_or(0.0) as u64;
                let oracle = r.metrics.get("oracle_docs_used").copied().unwrap_or(0.0) as u64;
                let oracle_tpl = r
                    .metrics
                    .get("oracle_tplinker_docs_used")
                    .copied()
                    .unwrap_or(0.0) as u64;
                if oracle_tpl > 0 {
                    format!(
                        " gold={} pred={} oracle_docs={} oracle_tpl_docs={}",
                        gold, pred, oracle, oracle_tpl
                    )
                } else {
                    format!(" gold={} pred={} oracle_docs={}", gold, pred, oracle)
                }
            } else {
                "".to_string()
            };
            eprintln!(
                "matrix-muxer: result task={:?} dataset={:?} backend={} success={} f1={:.3} thr={:.3} ok={} junk={} hard={} ms={:.0}{} err={}",
                r.task,
                r.dataset,
                r.backend,
                r.success,
                f1,
                thr,
                ok,
                junk,
                hard_junk,
                dur_ms,
                rel_counts,
                err
            );
        }

        let cost = r.num_examples as u64;
        let elapsed = dur_ms as u64;
        let o = match r.primary_f1() {
            Some(f1) => Outcome::with_quality(ok, junk, hard_junk, cost, elapsed, f1),
            None => Outcome::new(ok, junk, hard_junk, cost, elapsed),
        };
        let fail_kind = if hard_junk {
            Some(
                r.error
                    .as_deref()
                    .map(|e| mh::classify_failure_kind(e).to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
            )
        } else if junk {
            // Soft failure: quality/junk threshold.
            Some("low_signal".to_string())
        } else {
            None
        };

        // Optional: append an "observed outcome" event to the decision JSONL. This complements the
        // earlier selection decision logs (which reflect *history at selection time*).
        if let Some(ref p) = decisions_path() {
            let run_id = outcome_run_id_override.clone().unwrap_or_else(|| {
                format!(
                    "seed={} slice={} strategy={:?}",
                    seed, slice_tag_for_muxer, strategy
                )
            });
            let strategy_for_log = outcome_strategy_override
                .clone()
                .unwrap_or_else(|| strategy.id_str().to_string());
            let ml_only_policy_for_log = if strategy_for_log == SampleStrategy::MlOnly.id_str() {
                Some(MlOnlyPolicy::from_env().id_str().to_string())
            } else {
                None
            };
            let f1 = r.primary_f1().unwrap_or(0.0);
            let thr = junk_f1_threshold(r.task);

            let mon_cfg = mh::monitored_mab_config_from_env();
            let mon_enabled = monitoring_enabled(&mon_cfg);
            let monitor_recent_cap = mh::env_usize("ANNO_MUXER_MONITOR_RECENT_CAP", 20);
            let drift_cfg = mh::drift_config_from_env(mon_cfg.drift_metric);
            let (drift_score, catkl_score, cusum_score) = if mon_enabled {
                monitoring_scores_for_backend(
                    &history,
                    &r.backend,
                    monitor_recent_cap,
                    &mon_cfg,
                    drift_cfg,
                )
            } else {
                (None, None, None)
            };
            let monitoring_penalty = if mon_enabled {
                Some(monitoring_penalty(
                    drift_score,
                    catkl_score,
                    cusum_score,
                    &mon_cfg,
                    mon_cfg.drift_metric,
                ))
            } else {
                None
            };
            append_jsonl(
                p,
                &DecisionOutcomeLog {
                    schema_version: 3,
                    record_type: "outcome".to_string(),
                    muxer_version: MUXER_VERSION.to_string(),
                    run_id,
                    strategy: strategy_for_log,
                    ml_only_policy: ml_only_policy_for_log,
                    slice: slice_tag_for_muxer.to_string(),
                    dataset: format!("{:?}", r.dataset),
                    backend: r.backend.clone(),
                    backend_display: r.backend_display.clone(),
                    primary_f1: Some(f1),
                    junk_f1_threshold: Some(thr),
                    ok,
                    junk,
                    hard_junk,
                    fail_kind: fail_kind.clone(),
                    elapsed_ms: Some(dur_ms as u64),
                    cost_units: Some(r.num_examples as u64),
                    drift_score,
                    catkl_score,
                    cusum_score,
                    monitoring_penalty,
                },
            );
        }
        // Update both:
        // - global per-backend window (back-compat + overall view)
        // - dataset-scoped per-backend window (preferred for selection within a slice)
        history.push_with_fail_kind(&r.backend, o, fail_kind.clone());
        if mh::env_bool("ANNO_MUXER_PER_DATASET", true) {
            let k = BackendHistory::dataset_key(&r.backend, r.dataset);
            history.push_with_fail_kind(&k, o, fail_kind);
        }
    }

    // Update EXP3-IX state (ml-only policy) from the scalar reward signal.
    //
    // Reward mapping: we use the task-appropriate primary F1 clamped to [0, 1], and 0 on failure.
    // This intentionally differs from the MAB Summary (which is more "binary quality" oriented).
    if matches!(strategy, SampleStrategy::MlOnly)
        && matches!(MlOnlyPolicy::from_env(), MlOnlyPolicy::Exp3Ix)
    {
        if let Some(mut st) = exp3ix_state_for_update {
            for (chosen, prob_used) in exp3ix_tickets_for_update {
                let chosen = chosen.as_str();
                let mut sum = 0.0;
                let mut n = 0u64;
                for r in &results.results {
                    if r.is_skipped() {
                        continue;
                    }
                    if r.backend.as_str() != chosen {
                        continue;
                    }
                    let f1 = r.primary_f1().unwrap_or(0.0).clamp(0.0, 1.0);
                    let reward = if r.success { f1 } else { 0.0 };
                    sum += reward;
                    n += 1;
                }
                if n > 0 {
                    let r01 = (sum / (n as f64)).clamp(0.0, 1.0);
                    // Optional monitoring penalty: reduce reward when backend appears to drift / change.
                    //
                    // This is intentionally multiplicative: `r_adj = r01 * exp(-penalty)` keeps reward in [0,1]
                    // and preserves ranking when penalty=0.
                    let mon_cfg = mh::monitored_mab_config_from_env();
                    let mon_enabled = monitoring_enabled(&mon_cfg);
                    let monitor_recent_cap = mh::env_usize("ANNO_MUXER_MONITOR_RECENT_CAP", 20);
                    let drift_cfg = mh::drift_config_from_env(mon_cfg.drift_metric);
                    let (drift, catkl, cusum) = if mon_enabled {
                        monitoring_scores_for_backend(
                            &history,
                            chosen,
                            monitor_recent_cap,
                            &mon_cfg,
                            drift_cfg,
                        )
                    } else {
                        (None, None, None)
                    };
                    let penalty = if mon_enabled {
                        monitoring_penalty(drift, catkl, cusum, &mon_cfg, mon_cfg.drift_metric)
                    } else {
                        0.0
                    };
                    let r01 = if penalty > 0.0 {
                        (r01 * (-penalty).exp()).clamp(0.0, 1.0)
                    } else {
                        r01
                    };
                    if mh::env_bool("ANNO_MUXER_VERBOSE", false) && mon_enabled && penalty > 0.0 {
                        eprintln!(
                            "matrix-muxer: exp3ix reward adjusted backend={} base={:.3} penalty={:.3} adj={:.3} drift={:?} catkl={:?} cusum={:?}",
                            chosen,
                            (sum / (n as f64)).clamp(0.0, 1.0),
                            penalty,
                            r01,
                            drift,
                            catkl,
                            cusum
                        );
                    }
                    st = exp3ix_update_persisted_prod(
                        exp3ix_config_from_env(seed),
                        st,
                        chosen,
                        r01,
                        prob_used,
                    );
                }
            }
            history.exp3ix_state = Some(st);
        }
    }

    // LinUCB reward update: feed F1 scores back into the contextual policy.
    // State is saved to the GLOBAL file (cross-slice learning).
    if let Some((mut linucb, context)) = linucb_for_update {
        let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
        // Multi-objective reward: quality (F1) minus latency penalty.
        //
        // The scalarized reward is: reward = f1 - latency_weight * (ms / max_ms)
        // clamped to [0, 1].  This lets LinUCB learn the quality/latency Pareto
        // front: when latency_weight > 0, fast backends (bert_onnx ~1s) are
        // preferred over slow ones (nuner ~6s) unless the F1 gain justifies
        // the latency cost.  Default latency_weight=0 preserves F1-only behavior.
        let latency_weight = mh::env_f64("ANNO_MUXER_LINUCB_LATENCY_WEIGHT", 0.0);
        let latency_max_ms = mh::env_f64("ANNO_MUXER_LINUCB_LATENCY_MAX_MS", 10000.0);
        for r in &results.results {
            if r.is_skipped() {
                continue;
            }
            let f1 = r.primary_f1().unwrap_or(0.0).clamp(0.0, 1.0);
            let base_reward = if r.success { f1 } else { 0.0 };
            let latency_penalty = if latency_weight > 0.0 {
                let ms = r.duration_ms.unwrap_or(0.0).max(0.0);
                latency_weight * (ms / latency_max_ms).min(1.0)
            } else {
                0.0
            };
            let reward = (base_reward - latency_penalty).clamp(0.0, 1.0);
            linucb.update_reward(&r.backend, &context, reward);
            if verbose {
                if latency_weight > 0.0 {
                    eprintln!(
                        "matrix-muxer: linucb update arm={} reward={:.3} (f1={:.3} lat_pen={:.3} ms={:.0}) ctx=[{}]",
                        r.backend, reward, f1, latency_penalty,
                        r.duration_ms.unwrap_or(0.0),
                        context.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>().join(",")
                    );
                } else {
                    eprintln!(
                        "matrix-muxer: linucb update arm={} reward={:.3} (f1={:.3} success={}) ctx=[{}]",
                        r.backend, reward, f1, r.success,
                        context.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>().join(",")
                    );
                }
            }
        }
        let snapshot = linucb.snapshot();
        save_linucb_global_state(&linucb_global_state_path(), &snapshot);
        // Also keep in per-slice history for backward compat / inspection.
        history.linucb_state = Some(snapshot);
    }

    // Persist history best-effort for future runs.
    history.save(&hist_path);

    // Post-evaluation diagnostics: detect reward collapse (all backends scored identically).
    //
    // When every backend produces the same F1 (or all are junk), the bandit has no signal to
    // learn from. Log this so operators know the slice is degenerate.
    if chosen_backends.len() >= 2 {
        let mut per_backend_f1: std::collections::BTreeMap<String, Vec<f64>> =
            std::collections::BTreeMap::new();
        for r in &results.results {
            if r.is_skipped() {
                continue;
            }
            per_backend_f1
                .entry(r.backend.clone())
                .or_default()
                .push(r.primary_f1().unwrap_or(0.0));
        }
        let means: Vec<f64> = per_backend_f1
            .values()
            .map(|vs| {
                if vs.is_empty() {
                    0.0
                } else {
                    vs.iter().sum::<f64>() / (vs.len() as f64)
                }
            })
            .collect();
        if means.len() >= 2 {
            let max_f1 = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_f1 = means.iter().copied().fold(f64::INFINITY, f64::min);
            let all_junk = means.iter().all(|&m| m < 0.05);
            if (max_f1 - min_f1).abs() < 1e-6 {
                eprintln!(
                    "matrix-muxer: WARN reward collapse: all {} backends scored identical mean F1={:.3} on slice={} -- bandit has no signal to learn from",
                    means.len(),
                    max_f1,
                    slice_tag_for_muxer
                );
            }
            if all_junk {
                eprintln!(
                    "matrix-muxer: WARN all backends are junk (mean F1 < 0.05) on slice={} -- consider enabling ML backends (--features onnx) or checking dataset compatibility",
                    slice_tag_for_muxer
                );
            }
        }
    }

    // Change-point detection: compare recent F1 to historical F1 for each cell.
    //
    // This is the bug-detection mechanism: when a code change breaks a backend's
    // quality on a dataset, the recent F1 will drop relative to the historical mean.
    // We use a simple median-split comparison: if the drop exceeds 0.1, flag it.
    //
    // This runs on the full SQLite eval-history (not just the current run), so it
    // accumulates evidence across many runs and detects gradual degradation.
    // Gated: runs when ANNO_CHECK_REGRESSIONS=1, ANNO_MUXER_VERBOSE=1, or
    // ANNO_MUXER_MODE=triage (triage mode implies "find what broke").
    if mh::env_bool("ANNO_CHECK_REGRESSIONS", false)
        || mh::env_bool("ANNO_MUXER_VERBOSE", false)
        || matches!(MuxerMode::from_env(), Some(MuxerMode::Triage))
    {
        let hist_path = eval_history_jsonl_path();
        if let Ok(h) = crate::eval::history::EvalHistory::new(&hist_path) {
            // Use the recent-window method (last 5 observations vs n-comparable historical).
            //
            // NOTE: this is a heuristic -- it can't fully distinguish code regressions
            // from sampling variance because different seeds/subsets produce different F1.
            // The precise mechanism is detect_regressions_by_commit() which compares
            // identical conditions across git commits.
            //
            // Threshold: Cohen's d >= 2.0 (very large).  At this level, false alarms
            // from sampling noise are rare (requires a ~2 standard deviation shift).
            if let Ok(alerts) = h.detect_regressions_recent(
                5,   // compare last 5 observations
                2.0, // Cohen's d >= 2.0 (very large -- only flags severe regressions)
                20,  // min 20 total observations per cell
            ) {
                for alert in &alerts {
                    eprintln!(
                        "matrix-muxer: REGRESSION {}/{}: F1 dropped {:.3} ({:.3} -> {:.3}, n={}/{})",
                        alert.backend,
                        alert.dataset,
                        alert.drop,
                        alert.old_mean,
                        alert.new_mean,
                        alert.n_old,
                        alert.n_new,
                    );
                }
                if alerts.is_empty() && mh::env_bool("ANNO_MUXER_VERBOSE", false) {
                    eprintln!("matrix-muxer: no regressions detected (good)");
                }
            }
        }
    }

    // If we got here, the harness executed. Failures are recorded, not fatal.
    // (This job is intended to find regressions over time, not block all merges on flaky data.)
}

#[cfg(test)]
#[test]
#[ignore] // Slow (often >60s); run with: cargo test -p anno-eval --lib test_randomized_matrix_sample -- --ignored --include-ignored
fn test_randomized_matrix_sample() {
    run_randomized_matrix_sample_with_seed(ci_seed());
}

#[test]
fn test_matrix_sweep_all_backends_once() {
    // Opt-in: this can be expensive if you enable ML backends and caches are cold.
    if !mh::env_bool("ANNO_MATRIX_SWEEP", false) {
        return;
    }

    anno::env::load_dotenv();

    let seed = ci_seed();
    let loader = DatasetLoader::new().expect("DatasetLoader::new");
    let require_cached = matrix_require_cached();

    // Sweep is an explicit “prove things run” tool: require an explicit task so the user knows
    // what they are validating.
    let Some(task) = matrix_task_override() else {
        let msg = "matrix-sweep: set ANNO_MATRIX_TASK to choose what to validate (e.g. ner, intra-coref, re)";
        if mh::env_bool("ANNO_MATRIX_SWEEP_STRICT", true) {
            panic!("{msg}");
        }
        eprintln!("{msg}");
        return;
    };
    let tasks = vec![task];

    let strict = mh::env_bool("ANNO_MATRIX_SWEEP_STRICT", true);
    let max_examples = mh::env_usize("ANNO_MATRIX_SWEEP_MAX_EXAMPLES", 5).max(1);
    let chosen = choose_datasets_for_run(seed ^ 0x51EE_AAAA, &loader, &tasks, require_cached, 1);
    let Some(&dataset) = chosen.first() else {
        let msg = format!(
            "matrix-sweep: no eligible datasets for tasks={:?} (require_cached={} cache_dir={:?})",
            tasks,
            require_cached,
            loader.cache_dir()
        );
        if strict {
            panic!("{msg}");
        }
        eprintln!("{msg}");
        return;
    };

    // Sweep candidate set:
    // - reuse the same baseline/ML gating logic as the harness’ default (MlOnly) strategy,
    //   but run *all* eligible arms in one go (after compatibility filters).
    let mut candidates = backend_candidates(SampleStrategy::MlOnly, &tasks);
    candidates.retain(|b| {
        let ts = backend_tasks(b);
        tasks.iter().any(|t| ts.contains(t))
    });
    candidates.retain(|b| TaskEvaluator::is_backend_compatible(b, dataset));

    if candidates.is_empty() {
        let msg = format!(
            "matrix-sweep: no backend candidates for tasks={:?} dataset={:?}",
            tasks, dataset
        );
        if strict {
            panic!("{msg}");
        }
        eprintln!("{msg}");
        return;
    }

    let eval = TaskEvaluator::new().expect("TaskEvaluator::new");
    let config = TaskEvalConfig {
        tasks,
        datasets: vec![dataset],
        backends: candidates.clone(),
        max_examples: Some(max_examples),
        seed: Some(seed),
        require_cached,
        relation_threshold: 0.5,
        robustness: false,
        compute_familiarity: false,
        temporal_stratification: false,
        confidence_intervals: false,
        custom_coref_resolver: None,
        coref_use_gold_mentions: std::env::var("ANNO_COREF_GOLD")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
    };

    let results = match eval.evaluate_all(config) {
        Ok(r) => r,
        Err(e) => {
            if strict {
                panic!("matrix-sweep: evaluation returned error: {e}");
            } else {
                eprintln!("matrix-sweep: evaluation returned error: {e}");
                return;
            }
        }
    };

    let mut bad: Vec<String> = Vec::new();
    for r in &results.results {
        if r.is_skipped() || !r.success {
            let err = r
                .error
                .as_deref()
                .map(|s| trunc(s, 200))
                .unwrap_or_else(|| "-".to_string());
            bad.push(format!(
                "task={:?} dataset={:?} backend={} skipped={} success={} err={}",
                r.task,
                r.dataset,
                r.backend,
                r.is_skipped(),
                r.success,
                err
            ));
        }
    }

    if !bad.is_empty() {
        eprintln!(
            "matrix-sweep: failures={} strict={} dataset={:?} candidates={:?}",
            bad.len(),
            strict,
            dataset,
            candidates
        );
        for line in &bad {
            eprintln!("matrix-sweep: {line}");
        }
        if strict {
            panic!("matrix-sweep: one or more backends failed or were skipped");
        }
    }
}

#[test]
fn test_matrix_muxer_outcome_uses_primary_f1_keys() {
    // This is a seam test: it ensures we don't accidentally treat non-NER tasks as "f1 missing → 0",
    // which would mark good runs as junk and skew muxer history.
    use crate::eval::task_evaluator::TaskEvalResult;
    use crate::eval::task_mapping::Task;
    use std::collections::HashMap;

    fn mk(
        task: Task,
        dataset: DatasetId,
        success: bool,
        metrics: &[(&str, f64)],
    ) -> TaskEvalResult {
        let mut m = HashMap::new();
        for (k, v) in metrics {
            m.insert((*k).to_string(), *v);
        }
        TaskEvalResult {
            task,
            dataset,
            backend: "stub".to_string(),
            backend_display: None,
            seed: 0,
            success,
            error: None,
            metrics: m,
            num_examples: 1,
            duration_ms: Some(1.0),
            label_shift: None,
            robustness: None,
            stratified: None,
            confidence_intervals: None,
            kb_version: None,
        }
    }

    // Coref uses conll_f1 as its primary F1. This should not be treated as missing.
    let coref_ok = mk(
        Task::IntraDocCoref,
        DatasetId::GAP,
        true,
        &[("conll_f1", 0.9)],
    );
    assert_eq!(coref_ok.primary_f1(), Some(0.9));

    // Relation uses strict_f1 as its primary F1.
    let rel_ok = mk(
        Task::RelationExtraction,
        DatasetId::DocRED,
        true,
        &[("strict_f1", 0.9)],
    );
    assert_eq!(rel_ok.primary_f1(), Some(0.9));
}

#[test]
fn test_muxer_prior_prefers_facet_matched_history() {
    // Ensure facet-scoped slices borrow from the most relevant prior when possible.
    //
    // Setup:
    // - prior history has dataset-scoped windows for backend `stacked` on:
    //   - Wnut17 (en + social_media) = always ok
    //   - GermEvalDiscontinuous (de + ... ) = always junk
    // - current history is empty
    // - dataset slice is [Wnut17], so facet prior should pick the Wnut17 window.
    let mut prior = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };

    let mut wnut = HistoryWindow::new(50);
    for _ in 0..10 {
        wnut.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1,
            quality_score: None,
        });
    }
    prior.windows.insert(
        BackendHistory::dataset_key("stacked", DatasetId::Wnut17),
        wnut,
    );

    let mut de = HistoryWindow::new(50);
    for _ in 0..10 {
        de.push(Outcome {
            ok: false,
            junk: true,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1,
            quality_score: None,
        });
    }
    prior.windows.insert(
        BackendHistory::dataset_key("stacked", DatasetId::GermEvalDiscontinuous),
        de,
    );

    let current = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };

    let arms = vec!["stacked".to_string()];
    let summaries = current.summaries_for(Some(&prior), &arms, Some(&[DatasetId::Wnut17]), true, 6);
    let s = summaries.get("stacked").copied().unwrap_or_default();
    assert!(s.calls >= 6);
    assert!(
        s.ok_rate() > 0.5,
        "facet prior should bias ok_rate upward; got ok_rate={}",
        s.ok_rate()
    );
}

#[cfg(test)]
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

#[test]
fn test_latency_guardrail_require_measured_uses_observed_calls() {
    // Regression test: prior smoothing should not make an arm count as “measured” for the
    // latency guardrail. `require_measured` must be based on observed calls.
    let _env = env_lock();
    let mut prior = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };
    let mut w = HistoryWindow::new(50);
    for _ in 0..10 {
        w.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1,
            quality_score: None,
        });
    }
    prior.windows.insert("stacked".to_string(), w);

    let current = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };

    // Guard: save+restore env so this test doesn't pollute others.
    let old_max = std::env::var("ANNO_MUXER_MAX_MEAN_ELAPSED_MS").ok();
    let old_req = std::env::var("ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT").ok();
    let old_nov = std::env::var("ANNO_MUXER_NOVELTY").ok();
    struct Restore {
        k: &'static str,
        v: Option<String>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.v.as_deref() {
                None => std::env::remove_var(self.k),
                Some(v) => std::env::set_var(self.k, v),
            }
        }
    }
    let _r1 = Restore {
        k: "ANNO_MUXER_MAX_MEAN_ELAPSED_MS",
        v: old_max,
    };
    let _r2 = Restore {
        k: "ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT",
        v: old_req,
    };
    let _r3 = Restore {
        k: "ANNO_MUXER_NOVELTY",
        v: old_nov,
    };

    std::env::set_var("ANNO_MUXER_MAX_MEAN_ELAPSED_MS", "2");
    std::env::set_var("ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT", "1");
    std::env::set_var("ANNO_MUXER_NOVELTY", "0"); // don't short-circuit into novelty selection

    let chosen = select_backends(
        SampleStrategy::MlOnly,
        0,
        "ner",
        &current,
        Some(&prior),
        &["stacked".to_string()],
        None,
        1,
        6,
    );

    // With require_measured=1 and no observed calls, we must not pick it.
    assert!(
        chosen.is_empty(),
        "arm should be filtered as unmeasured (observed calls == 0) despite prior"
    );
}

#[test]
fn test_latency_guardrail_require_measured_prefers_observed_measured_arm() {
    // Tougher regression test: with multiple arms, require_measured must exclude arms that are
    // only “measured” via priors, and still allow selection of a truly observed arm.
    let _env = env_lock();
    let mut prior = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };
    let mut w_prior = HistoryWindow::new(50);
    for _ in 0..10 {
        w_prior.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1, // fast
            quality_score: None,
        });
    }
    prior.windows.insert("prior_only".to_string(), w_prior);

    // Current history has only one arm observed (measured_fast).
    let mut current = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };
    let mut w_obs = HistoryWindow::new(50);
    for _ in 0..3 {
        w_obs.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1, // fast
            quality_score: None,
        });
    }
    current.windows.insert("measured_fast".to_string(), w_obs);

    // Guard: save+restore env so this test doesn't pollute others.
    let old_max = std::env::var("ANNO_MUXER_MAX_MEAN_ELAPSED_MS").ok();
    let old_req = std::env::var("ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT").ok();
    let old_nov = std::env::var("ANNO_MUXER_NOVELTY").ok();
    struct Restore {
        k: &'static str,
        v: Option<String>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.v.as_deref() {
                None => std::env::remove_var(self.k),
                Some(v) => std::env::set_var(self.k, v),
            }
        }
    }
    let _r1 = Restore {
        k: "ANNO_MUXER_MAX_MEAN_ELAPSED_MS",
        v: old_max,
    };
    let _r2 = Restore {
        k: "ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT",
        v: old_req,
    };
    let _r3 = Restore {
        k: "ANNO_MUXER_NOVELTY",
        v: old_nov,
    };

    std::env::set_var("ANNO_MUXER_MAX_MEAN_ELAPSED_MS", "2");
    std::env::set_var("ANNO_MUXER_LATENCY_GUARDRAIL_REQUIRE_MEASUREMENT", "1");
    std::env::set_var("ANNO_MUXER_NOVELTY", "0");

    let chosen = select_backends(
        SampleStrategy::MlOnly,
        0,
        "ner",
        &current,
        Some(&prior),
        &["prior_only".to_string(), "measured_fast".to_string()],
        None,
        1,
        6,
    );
    assert_eq!(chosen, vec!["measured_fast".to_string()]);
}

#[test]
fn test_control_k_prefix_is_deterministic_and_reserved() {
    // Regression test: control picks should be a deterministic prefix and must not be re-picked.
    let _env = env_lock();
    let old = std::env::var("ANNO_MUXER_CONTROL_K").ok();
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.0.as_deref() {
                None => std::env::remove_var("ANNO_MUXER_CONTROL_K"),
                Some(v) => std::env::set_var("ANNO_MUXER_CONTROL_K", v),
            }
        }
    }
    let _r = Restore(old);
    std::env::set_var("ANNO_MUXER_CONTROL_K", "1");

    let history = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };
    let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let expected = mh::pick_random_subset(0xC0E1_1A11, &arms, 1)
        .first()
        .cloned()
        .unwrap();
    let chosen = select_backends(
        SampleStrategy::MlOnly,
        0,
        "ner",
        &history,
        None,
        &arms,
        None,
        2,
        0,
    );
    assert!(!chosen.is_empty());
    assert_eq!(chosen[0], expected);
    assert_eq!(chosen.len(), 2);
    assert_ne!(chosen[0], chosen[1]);
}

#[test]
fn test_novelty_still_triggers_under_priors() {
    // Regression test: with priors enabled, "calls" may be non-zero in smoothed summaries
    // even when an arm has never been tried in this slice. Novelty should still pick the
    // slice-unseen arm.
    let _env = env_lock();
    let mut prior = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };
    let mut w = HistoryWindow::new(50);
    for _ in 0..10 {
        w.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1,
            quality_score: None,
        });
    }
    prior.windows.insert("stacked".to_string(), w);

    let current = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };

    // Guard: save+restore env so this test doesn't pollute others.
    let old_nov = std::env::var("ANNO_MUXER_NOVELTY").ok();
    struct Restore {
        k: &'static str,
        v: Option<String>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.v.as_deref() {
                None => std::env::remove_var(self.k),
                Some(v) => std::env::set_var(self.k, v),
            }
        }
    }
    let _r = Restore {
        k: "ANNO_MUXER_NOVELTY",
        v: old_nov,
    };

    std::env::set_var("ANNO_MUXER_NOVELTY", "1");

    let chosen = select_backends(
        SampleStrategy::MlOnly,
        0,
        "ner",
        &current,
        Some(&prior),
        &["stacked".to_string()],
        None,
        1,
        6,
    );
    assert_eq!(
        chosen.as_slice(),
        &["stacked".to_string()],
        "novelty should pick the slice-unseen arm even when priors exist"
    );
}

#[test]
fn test_measure_mode_default_control_does_not_bypass_ml_when_k_is_one() {
    // Regression test: in measure-mode, the default control pick must not consume the entire
    // budget when k==1. Otherwise ML selection never runs and we only get outcome-only logs.
    let _env = env_lock();

    let tmp = std::env::temp_dir().join("anno-matrix-muxer-test-measure-k1.jsonl");
    let _ = std::fs::remove_file(&tmp);

    // Guard: save+restore env so this test doesn't pollute others.
    let old_dec = std::env::var("ANNO_MUXER_DECISIONS_FILE").ok();
    let old_mode = std::env::var("ANNO_MUXER_MODE").ok();
    let old_ctl = std::env::var("ANNO_MUXER_CONTROL_K").ok();
    struct Restore {
        k: &'static str,
        v: Option<String>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.v.as_deref() {
                None => std::env::remove_var(self.k),
                Some(v) => std::env::set_var(self.k, v),
            }
        }
    }
    let _r1 = Restore {
        k: "ANNO_MUXER_DECISIONS_FILE",
        v: old_dec,
    };
    let _r2 = Restore {
        k: "ANNO_MUXER_MODE",
        v: old_mode,
    };
    let _r3 = Restore {
        k: "ANNO_MUXER_CONTROL_K",
        v: old_ctl,
    };

    std::env::set_var(
        "ANNO_MUXER_DECISIONS_FILE",
        tmp.to_string_lossy().to_string(),
    );
    std::env::set_var("ANNO_MUXER_MODE", "measure");
    std::env::remove_var("ANNO_MUXER_CONTROL_K");

    let history = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };
    let arms = vec!["a".to_string(), "b".to_string()];
    let _chosen = select_backends(
        SampleStrategy::MlOnly,
        0,
        "ner",
        &history,
        None,
        &arms,
        None,
        1,
        0,
    );

    let s = std::fs::read_to_string(&tmp).expect("read decisions log");
    assert!(
        s.contains("\"mab_k_round\""),
        "expected an ML decision row (mab_k_round present) for k==1; log={s}"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_control_only_still_writes_minimal_decision_log_row() {
    // Regression test: with ANNO_MUXER_CONTROL_K set and n >= 2, the decision log must include
    // control_arms (≥1 control pick) alongside a MAB round for the remaining slots.
    //
    // Note: with n=1 muxer always reserves the single slot for MAB (max_control = min(k, n-1) = 0).
    // Use n=2 so control_k=1 yields 1 control arm + 1 MAB pick.
    let _env = env_lock();

    let tmp = std::env::temp_dir().join("anno-matrix-muxer-test-control-only.jsonl");
    let _ = std::fs::remove_file(&tmp);

    // Guard: save+restore env so this test doesn't pollute others.
    let old_dec = std::env::var("ANNO_MUXER_DECISIONS_FILE").ok();
    let old_mode = std::env::var("ANNO_MUXER_MODE").ok();
    let old_ctl = std::env::var("ANNO_MUXER_CONTROL_K").ok();
    struct Restore {
        k: &'static str,
        v: Option<String>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.v.as_deref() {
                None => std::env::remove_var(self.k),
                Some(v) => std::env::set_var(self.k, v),
            }
        }
    }
    let _r1 = Restore {
        k: "ANNO_MUXER_DECISIONS_FILE",
        v: old_dec,
    };
    let _r2 = Restore {
        k: "ANNO_MUXER_MODE",
        v: old_mode,
    };
    let _r3 = Restore {
        k: "ANNO_MUXER_CONTROL_K",
        v: old_ctl,
    };

    std::env::set_var(
        "ANNO_MUXER_DECISIONS_FILE",
        tmp.to_string_lossy().to_string(),
    );
    std::env::set_var("ANNO_MUXER_MODE", "measure");
    std::env::set_var("ANNO_MUXER_CONTROL_K", "1");

    let history = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
        linucb_state: None,
    };
    let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let _chosen = select_backends(
        SampleStrategy::MlOnly,
        0,
        "ner",
        &history,
        None,
        &arms,
        None,
        2, // n=2: max_control = min(1, 2-1) = 1 → 1 control arm + 1 MAB pick
        0,
    );

    let s = std::fs::read_to_string(&tmp).expect("read decisions log");
    assert!(
        s.contains("\"control_arms\""),
        "expected decision row with control_arms when control_k=1, n=2; log={s}"
    );
    // MAB also runs for the remaining n-control_arms slots.
    assert!(
        s.contains("\"mab_k_round\""),
        "expected mab_k_round alongside control_arms; log={s}"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ─── eval_history_jsonl_path resolution ──────────────────────────────────────

#[test]
fn eval_history_path_anno_eval_history_wins() {
    let _env = env_lock();
    let key = "ANNO_EVAL_HISTORY";
    let old = std::env::var(key).ok();
    std::env::set_var(key, "/explicit/eval-results.jsonl");
    std::env::remove_var("ANNO_CACHE_DIR");
    let p = eval_history_jsonl_path();
    match old.as_deref() {
        None => std::env::remove_var(key),
        Some(v) => std::env::set_var(key, v),
    }
    assert_eq!(p, std::path::PathBuf::from("/explicit/eval-results.jsonl"));
}

#[test]
fn eval_history_path_anno_cache_dir_fallback() {
    let _env = env_lock();
    let key_hist = "ANNO_EVAL_HISTORY";
    let key_cache = "ANNO_CACHE_DIR";
    let old_hist = std::env::var(key_hist).ok();
    let old_cache = std::env::var(key_cache).ok();
    std::env::remove_var(key_hist);
    std::env::set_var(key_cache, "/my/cache");
    let p = eval_history_jsonl_path();
    match old_hist.as_deref() {
        None => std::env::remove_var(key_hist),
        Some(v) => std::env::set_var(key_hist, v),
    }
    match old_cache.as_deref() {
        None => std::env::remove_var(key_cache),
        Some(v) => std::env::set_var(key_cache, v),
    }
    assert_eq!(
        p,
        std::path::PathBuf::from("/my/cache/eval-results.jsonl"),
        "ANNO_CACHE_DIR should be used when ANNO_EVAL_HISTORY is unset"
    );
}

#[test]
fn eval_history_path_anno_eval_history_beats_cache_dir() {
    let _env = env_lock();
    let key_hist = "ANNO_EVAL_HISTORY";
    let key_cache = "ANNO_CACHE_DIR";
    let old_hist = std::env::var(key_hist).ok();
    let old_cache = std::env::var(key_cache).ok();
    std::env::set_var(key_hist, "/override/eval.jsonl");
    std::env::set_var(key_cache, "/should/be/ignored");
    let p = eval_history_jsonl_path();
    match old_hist.as_deref() {
        None => std::env::remove_var(key_hist),
        Some(v) => std::env::set_var(key_hist, v),
    }
    match old_cache.as_deref() {
        None => std::env::remove_var(key_cache),
        Some(v) => std::env::set_var(key_cache, v),
    }
    assert_eq!(
        p,
        std::path::PathBuf::from("/override/eval.jsonl"),
        "ANNO_EVAL_HISTORY must beat ANNO_CACHE_DIR"
    );
}
