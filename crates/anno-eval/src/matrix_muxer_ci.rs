//! CI-friendly randomized matrix test (muxer-backed).
//!
//! This replaces the older archived test harness with something that:
//! - compiles under `--features eval-advanced` (no `cli` feature required)
//! - uses `muxer` for deterministic, windowed MAB-style backend selection
//! - respects `ANNO_CI_SEED` and `ANNO_SAMPLE_STRATEGY`
//! - prefers cache in CI, but can still download to avoid no-op runs
//!
//! Environment variables:
//! - `ANNO_CI_SEED`: u64 seed (default: 0)
//! - `ANNO_SAMPLE_STRATEGY`: `random` | `ml-only` | `worst-first` (default: `ml-only`)
//! - `ANNO_MUXER_MODE`: optional mode default (`triage` | `measure`). Used only when
//!   `ANNO_SAMPLE_STRATEGY` is unset.
//! - `ANNO_MATRIX_TASK`: optional task override (e.g. `discontinuous-ner`, `re`, `intra-coref`)
//! - `ANNO_MATRIX_REQUIRE_CACHED`: if true, prefer cache, but allow fallback downloads when empty
//! - `ANNO_MATRIX_COVERAGE_REPORT`: if set, write a JSON coverage report to this path
//! - `ANNO_HISTORY_FILE`: optional JSON path for muxer history
//! - `ANNO_MAX_EXAMPLES`: max examples per dataset (default: 20 in CI, 50 locally)
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

// This module is compiled under `anno-eval`'s `eval-advanced` feature.
// Tests within this file are still `#[cfg(test)]` as usual.

use crate::eval::backend_factory::BackendFactory;
use crate::eval::loader::{DatasetId, DatasetLoader, LoadableDatasetId};
use crate::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use crate::eval::task_mapping::{
    backend_tasks, dataset_tasks, get_task_backends, get_task_datasets, Task,
};
use crate::muxer_harness as mh;
use muxer::{Exp3IxConfig, Exp3IxState, MabConfig, Outcome, Summary, Window};
use std::collections::VecDeque;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy)]
enum SampleStrategy {
    Random,
    MlOnly,
    WorstFirst,
}

#[derive(Debug, Clone, Copy)]
enum MuxerMode {
    Triage,
    Measure,
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

            // Back-compat aliases:
            "bug" | "bug-hunt" | "bughunt" | "regress" | "regression" => Some(Self::Triage),
            "perf" | "perf-estimate" | "perfestimate" | "estimate" | "measurement" => {
                Some(Self::Measure)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MlOnlyPolicy {
    Exp3Ix,
    Mab,
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
            _ => Self::Exp3Ix,
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
                "ml-only" | "mlonly" | "ml" => return Self::MlOnly,
                _ => {}
            }
        }

        // Otherwise, mode chooses the default.
        match MuxerMode::from_env() {
            Some(MuxerMode::Triage) => Self::WorstFirst,
            Some(MuxerMode::Measure) => Self::MlOnly,
            None => Self::MlOnly,
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
    // - CI prefers cache, but can fall back to downloads when needed
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
    let default = if in_ci() { 10 } else { 25 };
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
        http_429: 0,
        junk: 0,
        hard_junk: 0,
        cost_units: 0,
        elapsed_ms_sum: 0,
    };

    // MAB will deterministically pick one arm under identical summaries.
    let mut summaries = BTreeMap::new();
    summaries.insert("a".to_string(), s);
    summaries.insert("b".to_string(), s);
    let mab_choice = muxer::select_mab_explain(&arms, &summaries, mab_config_from_env())
        .selection
        .chosen;

    // Make the MAB-chosen arm worse so EXP3-IX has an opportunity to beat it.
    let r_a = if mab_choice == "a" { 0.6 } else { 0.9 };
    let r_b = if mab_choice == "b" { 0.6 } else { 0.9 };

    let mut total_mab = 0.0;
    let mut total_exp3 = 0.0;

    for t in 0..200u64 {
        let (d, st2) = muxer::exp3ix_decide_persisted(
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
        st = Some(muxer::exp3ix_update_persisted(cfg, st2, &chosen, r, p));

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

#[derive(Debug, Clone, serde::Serialize)]
struct WorstFirstRoundLog {
    remaining: Vec<String>,
    chosen: String,
    explore_first: bool,
    exploration_c: f64,
    hard_weight: f64,
    soft_weight: f64,
    top_candidates: muxer::LogTopCandidates,
}

#[derive(Debug, Clone, serde::Serialize)]
struct FailKindCount {
    kind: String,
    count: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DecisionLog {
    schema_version: u32,
    muxer_version: String,
    run_id: String,
    strategy: String,
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
    top_candidates: Option<muxer::LogTopCandidates>,
    /// If present, these arms were selected as deterministic-random "control" picks (bias anchor).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    control_arms: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chosen_fail_kinds_top: Option<Vec<FailKindCount>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mab_k_round: Option<muxer::MabKRoundLog>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exp3ix_rounds: Option<Vec<muxer::Exp3IxKRoundLog>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    worst_first_round: Option<WorstFirstRoundLog>,
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
    slice: String,
    dataset: String,
    backend: String,
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
}

#[test]
fn test_decision_log_schema_smoke() {
    let log = DecisionLog {
        schema_version: 6,
        muxer_version: muxer::MUXER_VERSION.to_string(),
        run_id: "seed=0 slice=ner strategy=MlOnly".to_string(),
        strategy: "ml-only".to_string(),
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
        top_candidates: Some(muxer::LogTopCandidates {
            kind: muxer::LOG_SCORE_KIND_MAB_SCALAR.to_string(),
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
    };

    let s = serde_json::to_string(&log).expect("serialize DecisionLog");
    let v: serde_json::Value = serde_json::from_str(&s).expect("parse DecisionLog JSON");

    assert_eq!(v["schema_version"].as_u64(), Some(6));
    assert_eq!(v["muxer_version"].as_str(), Some(muxer::MUXER_VERSION));
    assert_eq!(
        v["top_candidates"]["kind"].as_str(),
        Some(muxer::LOG_SCORE_KIND_MAB_SCALAR)
    );
    assert!(v["top_candidates"]["rows"].is_array());
    assert!(v["chosen_fail_kinds_top"].is_array());
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BackendHistory {
    #[serde(default)]
    version: u32,
    window_cap: usize,
    windows: BTreeMap<String, Window>,
    /// Optional per-outcome failure kind strings aligned with each window buffer.
    ///
    /// This is triage-oriented metadata (low-cardinality strings). Selection should not depend on
    /// this directly; it exists to help bug-finding converge faster.
    #[serde(default)]
    fail_kinds: BTreeMap<String, VecDeque<Option<String>>>,
    #[serde(default)]
    exp3ix_state: Option<Exp3IxState>,
}

impl BackendHistory {
    fn load(path: &PathBuf, window_cap: usize) -> Self {
        #[derive(Debug, Clone, serde::Deserialize)]
        struct WindowSerde {
            #[serde(default)]
            cap: usize,
            #[serde(default)]
            buf: VecDeque<Outcome>,
        }
        #[derive(Debug, Clone, serde::Deserialize)]
        struct BackendHistorySerde {
            #[serde(default)]
            version: u32,
            #[serde(default)]
            window_cap: usize,
            #[serde(default)]
            windows: BTreeMap<String, WindowSerde>,
            #[serde(default)]
            fail_kinds: BTreeMap<String, VecDeque<Option<String>>>,
            #[serde(default)]
            exp3ix_state: Option<Exp3IxState>,
        }

        let bytes = fs::read(path).ok();
        if let Some(bytes) = bytes {
            if let Ok(h) = serde_json::from_slice::<BackendHistorySerde>(&bytes) {
                let cap = window_cap.max(1);
                let _ = h.window_cap; // legacy metadata; env-controlled cap is the source of truth

                // Upgrade path:
                // - version 0/1: legacy; ok may mean “bad” or “ran”; recover “non-junk success”
                //   as best-effort from the other fields.
                // - version 2: ok was recorded as “evaluation succeeded”; upgrade to “non-junk success”.
                // - version >=3: current semantics; enforce invariants defensively.
                let mut windows: BTreeMap<String, Window> = BTreeMap::new();
                let mut fail_kinds: BTreeMap<String, VecDeque<Option<String>>> = BTreeMap::new();
                for (k, w) in h.windows {
                    let _ = w.cap;
                    let mut out = Window::new(cap);
                    for mut o in w.buf {
                        match h.version {
                            0 | 1 => {
                                // Best-effort: treat “success” as “not hard_junk”, and require not junk.
                                // (Some legacy files inverted `ok`; do not trust it.)
                                o.ok = !o.hard_junk && !o.junk;
                            }
                            2 => {
                                // v2 stored ok := evaluation succeeded. Make it quality-aware.
                                o.ok = o.ok && !o.hard_junk && !o.junk;
                            }
                            _ => {
                                // v3+; keep ok but enforce basic invariant.
                                o.ok = o.ok && !o.hard_junk && !o.junk;
                            }
                        }
                        out.push(o);
                    }
                    windows.insert(k, out);
                }
                // Carry failure kinds forward best-effort, truncating to cap.
                for (k, mut fk) in h.fail_kinds {
                    while fk.len() > cap {
                        fk.pop_front();
                    }
                    fail_kinds.insert(k, fk);
                }

                return Self {
                    version: 3,
                    window_cap: cap,
                    windows,
                    fail_kinds,
                    exp3ix_state: h.exp3ix_state,
                };
            }
        }

        Self {
            version: 3,
            window_cap: window_cap.max(1),
            windows: BTreeMap::new(),
            fail_kinds: BTreeMap::new(),
            exp3ix_state: None,
        }
    }

    fn save(&self, path: &PathBuf) {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec_pretty(self) {
            let _ = fs::write(path, bytes);
        }
    }

    fn window_mut(&mut self, backend: &str) -> &mut Window {
        self.windows
            .entry(backend.to_string())
            .or_insert_with(|| Window::new(self.window_cap))
    }

    fn fail_kinds_mut(&mut self, key: &str) -> &mut VecDeque<Option<String>> {
        self.fail_kinds
            .entry(key.to_string())
            .or_insert_with(VecDeque::new)
    }

    fn push_with_fail_kind(&mut self, key: &str, o: Outcome, fail_kind: Option<String>) {
        self.window_mut(key).push(o);
        let cap = self.window_cap;
        let fk = self.fail_kinds_mut(key);
        fk.push_back(fail_kind);
        while fk.len() > cap {
            fk.pop_front();
        }
    }

    fn dataset_key(backend: &str, dataset: DatasetId) -> String {
        // Stable, human-readable key. `{:?}` is the enum variant name.
        format!("{backend}@@{:?}", dataset)
    }

    fn observed_summary_for(
        &self,
        backend: &str,
        datasets: Option<&[DatasetId]>,
        per_dataset: bool,
    ) -> Summary {
        // Prefer dataset-scoped windows when we have a dataset slice:
        // this avoids "objective bleed" across unrelated datasets while still allowing
        // a stable per-backend decision.
        if per_dataset {
            if let Some(datasets) = datasets {
                let mut agg = Summary::default();
                for &ds in datasets {
                    let k = Self::dataset_key(backend, ds);
                    if let Some(w) = self.windows.get(&k) {
                        let s = w.summary();
                        if s.calls == 0 {
                            continue;
                        }
                        agg.calls = agg.calls.saturating_add(s.calls);
                        agg.ok = agg.ok.saturating_add(s.ok);
                        agg.http_429 = agg.http_429.saturating_add(s.http_429);
                        agg.junk = agg.junk.saturating_add(s.junk);
                        agg.hard_junk = agg.hard_junk.saturating_add(s.hard_junk);
                        agg.cost_units = agg.cost_units.saturating_add(s.cost_units);
                        agg.elapsed_ms_sum = agg.elapsed_ms_sum.saturating_add(s.elapsed_ms_sum);
                    }
                }
                if agg.calls > 0 {
                    return agg;
                }
            }
        }
        // Fallback: global per-backend window (older history files, or when dataset-scoped windows
        // are not yet populated).
        self.windows
            .get(backend)
            .map(|w| w.summary())
            .unwrap_or_default()
    }

    fn base_prior_summary_for(prior: &BackendHistory, backend: &str) -> Summary {
        prior
            .windows
            .get(backend)
            .map(|w| w.summary())
            .unwrap_or_default()
    }

    fn facet_prior_summary_for(
        prior: &BackendHistory,
        backend: &str,
        lang: &'static str,
        dom: &'static str,
    ) -> Summary {
        // Aggregate dataset-scoped windows from the *prior* history, filtered to datasets whose
        // facets match the current slice.
        let prefix = format!("{backend}@@");
        let mut agg = Summary::default();
        for (k, w) in &prior.windows {
            let Some(suffix) = k.strip_prefix(&prefix) else {
                continue;
            };
            let Ok(ds) = suffix.parse::<DatasetId>() else {
                continue;
            };
            if ds.language() != lang || ds.domain() != dom {
                continue;
            }
            let s = w.summary();
            if s.calls == 0 {
                continue;
            }
            agg.calls = agg.calls.saturating_add(s.calls);
            agg.ok = agg.ok.saturating_add(s.ok);
            agg.http_429 = agg.http_429.saturating_add(s.http_429);
            agg.junk = agg.junk.saturating_add(s.junk);
            agg.hard_junk = agg.hard_junk.saturating_add(s.hard_junk);
            agg.cost_units = agg.cost_units.saturating_add(s.cost_units);
            agg.elapsed_ms_sum = agg.elapsed_ms_sum.saturating_add(s.elapsed_ms_sum);
        }
        agg
    }

    fn summaries_for(
        &self,
        prior: Option<&BackendHistory>,
        arms: &[String],
        datasets: Option<&[DatasetId]>,
        per_dataset: bool,
        prior_calls: u64,
    ) -> BTreeMap<String, Summary> {
        let mut out = BTreeMap::new();
        for a in arms {
            let mut s = self.observed_summary_for(a, datasets, per_dataset);
            if prior_calls > 0 {
                if let Some(prior) = prior {
                    let mut prior_s = Self::base_prior_summary_for(prior, a);
                    if mh::prior_by_facets_from_env() && per_dataset {
                        if let Some(datasets) = datasets {
                            if let Some((lang, dom)) = mh::facet_prior_filter(datasets) {
                                let facet = Self::facet_prior_summary_for(prior, a, lang, dom);
                                if facet.calls > 0 {
                                    prior_s = facet;
                                }
                            }
                        }
                    }
                    mh::apply_prior_counts_to_summary(&mut s, prior_s, prior_calls);
                }
            }
            out.insert(a.clone(), s);
        }
        out
    }

    fn chosen_fail_kinds_top_for(
        &self,
        backend: &str,
        datasets: Option<&[DatasetId]>,
        per_dataset: bool,
        top: usize,
    ) -> Option<Vec<FailKindCount>> {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        let mut saw_any = false;

        if per_dataset {
            if let Some(datasets) = datasets {
                for &ds in datasets {
                    let k = Self::dataset_key(backend, ds);
                    if let Some(buf) = self.fail_kinds.get(&k) {
                        saw_any = true;
                        for kind in buf.iter().flatten() {
                            *counts.entry(kind.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }
            if saw_any && !counts.is_empty() {
                return Some(top_counts(counts, top));
            }
        }

        if let Some(buf) = self.fail_kinds.get(backend) {
            for kind in buf.iter().flatten() {
                *counts.entry(kind.clone()).or_insert(0) += 1;
            }
        }
        if counts.is_empty() {
            None
        } else {
            Some(top_counts(counts, top))
        }
    }
}

fn top_counts(counts: BTreeMap<String, u64>, top: usize) -> Vec<FailKindCount> {
    let mut rows: Vec<(u64, String)> = counts.into_iter().map(|(k, v)| (v, k)).collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    rows.into_iter()
        .take(top.max(1))
        .map(|(count, kind)| FailKindCount { kind, count })
        .collect()
}

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

fn mab_config_from_env() -> MabConfig {
    mh::mab_config_from_env()
}

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
            let control_k = default_control_k_for_mode()
                .max(mh::control_k_from_env())
                .min(k)
                .min(candidates_in_order.len());
            let control = if control_k > 0 {
                mh::pick_random_subset(seed ^ 0xC0E1_1A11, candidates_in_order, control_k)
            } else {
                Vec::new()
            };
            let remaining_k = k.saturating_sub(control.len());
            let mut candidates_for_muxer: Vec<String> = candidates_in_order.to_vec();
            candidates_for_muxer.retain(|b| !control.contains(b));

            let per_dataset = mh::env_bool("ANNO_MUXER_PER_DATASET", true);

            // MAB selection (muxer): pick historically "best" arms (high ok_rate, low junk),
            // with exploration to avoid fixating too early.
            let cfg = mab_config_from_env();
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
                mh::novelty_from_env(),
                guard,
                |b| {
                    let s = history.observed_summary_for(b, datasets, per_dataset);
                    (s.calls, s.elapsed_ms_sum)
                },
                |eligible, remaining_k| {
                    let mk = muxer::select_mab_k_guardrailed_explain_full(
                        eligible,
                        |remaining| {
                            history.summaries_for(
                                prior,
                                remaining,
                                datasets,
                                per_dataset,
                                prior_calls,
                            )
                        },
                        cfg,
                        muxer::LatencyGuardrailConfig {
                            // Latency guardrail is applied on observed summaries above, so priors can't
                            // mask "require measured" or skew mean latency.
                            max_mean_ms: None,
                            require_measured: false,
                            allow_fewer: guard.allow_fewer,
                        },
                        remaining_k,
                    );
                    let chosen = mk.chosen.clone();
                    mk_opt = Some(mk);
                    chosen
                },
            );

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
                        let mut rows: Vec<(f64, &muxer::CandidateDebug)> = Vec::new();
                        for c in &d.selection.candidates {
                            let score = c.objective_success
                                - cfg.cost_weight * c.mean_cost_units
                                - cfg.latency_weight * c.mean_elapsed_ms
                                - cfg.hard_junk_weight * c.hard_junk_rate
                                - cfg.junk_weight * c.soft_junk_rate;
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

            if let (Some(ref p), Some(mk)) = (decisions_path.as_ref(), mk_opt.as_ref()) {
                let round_logs = muxer::log_mab_k_rounds_typed(mk, decisions_top);
                let ds: Vec<String> = datasets
                    .unwrap_or(&[])
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect();
                let profile = std::env::var("ANNO_MUXER_PROFILE").ok();
                for rl in round_logs {
                    let chosen_fail_kinds_top = rl.chosen.as_deref().and_then(|b| {
                        history.chosen_fail_kinds_top_for(b, datasets, per_dataset, 3)
                    });
                    append_jsonl(
                        p,
                        &DecisionLog {
                            schema_version: 6,
                            muxer_version: muxer::MUXER_VERSION.to_string(),
                            run_id: run_id.clone(),
                            strategy: "ml-only".to_string(),
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
                        },
                    );
                }
            }
            let mut out = control;
            out.extend(fill.chosen);
            out.truncate(k);
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
                    let top_candidates: muxer::LogTopCandidates = muxer::LogTopCandidates {
                        kind: "worst_first".to_string(),
                        rows: rows
                            .iter()
                            .take(decisions_top.max(1))
                            .map(|(score, arm, s)| muxer::LogTopCandidate {
                                arm: arm.clone(),
                                score: *score,
                                calls: Some(s.calls),
                                ok_rate: Some(s.ok_rate()),
                                junk_rate: Some(s.junk_rate()),
                                hard_junk_rate: Some(s.hard_junk_rate()),
                            })
                            .collect(),
                    };
                    let ds: Vec<String> = datasets
                        .unwrap_or(&[])
                        .iter()
                        .map(|d| format!("{d:?}"))
                        .collect();
                    let profile = std::env::var("ANNO_MUXER_PROFILE").ok();
                    append_jsonl(
                        p,
                        &DecisionLog {
                            schema_version: 6,
                            muxer_version: muxer::MUXER_VERSION.to_string(),
                            run_id: run_id.clone(),
                            strategy: "worst-first".to_string(),
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
                                top_candidates: muxer::LogTopCandidates {
                                    kind: "worst_first".to_string(),
                                    rows: rows
                                        .iter()
                                        .take(decisions_top.max(1))
                                        .map(|(score, arm, s)| muxer::LogTopCandidate {
                                            arm: arm.clone(),
                                            score: *score,
                                            calls: Some(s.calls),
                                            ok_rate: Some(s.ok_rate()),
                                            junk_rate: Some(s.junk_rate()),
                                            hard_junk_rate: Some(s.hard_junk_rate()),
                                        })
                                        .collect(),
                                },
                            }),
                        },
                    );
                }
                remaining.retain(|b| b != &pick);
                chosen.push(pick);
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
    let allow_ml = match std::env::var("ANNO_ML_IN_MATRIX")
        .ok()
        .map(|v| v.trim().to_string())
    {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        Some(_) => in_ci(),
        None => in_ci(),
    };

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
        // Known-unimplemented arms: keep them out of the matrix until implemented.
        // (If we keep them, worst-first will explore them first and waste the slice.)
        if b == "gliner_poly" {
            continue;
        }
        if b == "w2ner" && !has_hf_token {
            continue;
        }

        // Baselines: always include if present in this build.
        if matches!(b, "stacked" | "crf" | "hmm" | "heuristic") {
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

    let mut out: Vec<DatasetId> = Vec::new();
    let temporal_requested = tasks.contains(&Task::Temporal);
    for ds in candidates {
        // CI policy: avoid non-automatable sources unless explicitly requested.
        //
        // The registry contains many useful-but-not-directly-downloadable datasets (gated corpora,
        // dead links, “contact authors”, etc.). Those are still valuable, but running them in CI
        // wastes matrix budget and mostly produces hard-junk outcomes.
        if in_ci()
            && !require_cached
            && !mh::env_bool("ANNO_MATRIX_INCLUDE_NON_AUTOMATABLE", false)
            && !ds.is_automatable_download()
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
        candidates.retain(|d| {
            let lang_ok = pin_lang
                .as_ref()
                .is_none_or(|s| s.contains(&d.language().to_ascii_lowercase()));
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
    if fixed_backends_requested.is_some()
        && pinned_backends_requested.is_some()
        && mh::env_bool("ANNO_MUXER_VERBOSE", false)
    {
        eprintln!("matrix-muxer: both FIXED_BACKEND and PIN_BACKEND are set; FIXED will win for selection");
    }
    let mut chosen_datasets = choose_datasets_for_run(
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

    // Optional override: fixed datasets (comma-separated DatasetId debug names).
    //
    // This is the most “pandas groupby”-like pin: you are explicitly choosing the group.
    if let Ok(raw) = std::env::var("ANNO_MUXER_FIXED_DATASETS")
        .or_else(|_| std::env::var("ANNO_MUXER_FORCE_DATASETS"))
    {
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
        fixed.retain(|ds| dataset_is_usable(&loader, *ds, require_cached_for_run));
        fixed.retain(|ds| tasks.iter().all(|t| dataset_tasks(*ds).contains(t)));
        if fixed.is_empty() {
            eprintln!("matrix-muxer: no usable fixed datasets remained after filtering");
            return;
        }
        chosen_datasets = fixed;
    }

    if let Some(ref fixed) = fixed_backends_requested {
        if fixed.is_empty() {
            eprintln!("matrix-muxer: ANNO_MUXER_FIXED_BACKEND is set but empty");
            return;
        }
        // Prefer datasets compatible with the fixed backend(s). If the first draw includes
        // incompatible datasets, resample a larger pool and filter.
        let want = datasets_per_run;
        let mut filtered = chosen_datasets
            .into_iter()
            .filter(|d| {
                fixed
                    .iter()
                    .all(|b| TaskEvaluator::is_backend_compatible(b, *d))
            })
            .collect::<Vec<_>>();
        if filtered.len() < want {
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
            let also_fixed_ds = std::env::var("ANNO_MUXER_FIXED_DATASETS")
                .or_else(|_| std::env::var("ANNO_MUXER_FORCE_DATASETS"))
                .ok()
                .is_some();
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
    let slice_tag_for_muxer = mh::muxer_slice_tag(
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

    let per_dataset = mh::env_bool("ANNO_MUXER_PER_DATASET", true);
    let backends_per_run_default = if require_cached { 2 } else { 3 };
    let backends_per_run =
        mh::env_usize("ANNO_MUXER_BACKENDS_PER_RUN", backends_per_run_default).max(1);

    let mut exp3ix_state_for_update: Option<Exp3IxState> = None;
    let mut exp3ix_tickets_for_update: Vec<(String, f64)> = Vec::new();
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
                append_jsonl(
                    t,
                    &DecisionLog {
                        schema_version: 6,
                        muxer_version: muxer::MUXER_VERSION.to_string(),
                        run_id: format!(
                            "seed={} slice={} strategy=fixed fixed={}",
                            seed,
                            slice_tag_for_muxer,
                            fixed.join(",")
                        ),
                        strategy: "fixed".to_string(),
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
        let control_k = mh::control_k_from_env()
            .min(backends_per_run)
            .min(candidates.len());
        let control = if control_k > 0 {
            mh::pick_random_subset(seed ^ 0xC0E1_1A11, &candidates, control_k)
        } else {
            Vec::new()
        };
        let remaining_k = backends_per_run.saturating_sub(control.len());
        let mut candidates_for_policy = candidates.clone();
        candidates_for_policy.retain(|b| !control.contains(b));

        let decision_seed = mh::stable_hash64(seed, &format!("anno-exp3ix:{slice_tag_for_muxer}"));
        let mut exp3ix_explain: Option<muxer::Exp3IxKExplain> = None;
        let fill = mh::policy_fill_k_observed_with(
            seed ^ 0xE8D3_1A00,
            &candidates_for_policy,
            remaining_k,
            mh::novelty_from_env(),
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
                let ex = muxer::exp3ix_decide_k_persisted_guardrailed_explain_full(
                    exp3ix_config_from_env(seed),
                    history.exp3ix_state.clone(),
                    eligible,
                    &summaries,
                    muxer::LatencyGuardrailConfig {
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
                    .any(|n| matches!(n, muxer::DecisionNote::ExploreFirst))
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
                let exp3ix_rounds = muxer::log_exp3ix_k_rounds_typed(ex, decisions_top);
                let eligible_arms = ex
                    .rounds
                    .first()
                    .map(|r| r.guardrail.eligible.clone())
                    .unwrap_or_default();
                append_jsonl(
                    p,
                    &DecisionLog {
                        schema_version: 6,
                        muxer_version: muxer::MUXER_VERSION.to_string(),
                        run_id,
                        strategy: "ml-only".to_string(),
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
                    },
                );
            }
        }
        chosen
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

        let o = Outcome {
            ok,
            http_429: false,
            junk,
            hard_junk,
            // A crude “cost” proxy. Not currently weighted in the harness, but useful for future
            // tuning and for offline inspection.
            cost_units: r.num_examples as u64,
            elapsed_ms: dur_ms as u64,
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
            let run_id = format!(
                "seed={} slice={} strategy={:?}",
                seed, slice_tag_for_muxer, strategy
            );
            let f1 = r.primary_f1().unwrap_or(0.0);
            let thr = junk_f1_threshold(r.task);
            append_jsonl(
                p,
                &DecisionOutcomeLog {
                    schema_version: 1,
                    record_type: "outcome".to_string(),
                    muxer_version: muxer::MUXER_VERSION.to_string(),
                    run_id,
                    strategy: format!("{:?}", strategy).to_lowercase(),
                    slice: slice_tag_for_muxer.to_string(),
                    dataset: format!("{:?}", r.dataset),
                    backend: r.backend.clone(),
                    primary_f1: Some(f1),
                    junk_f1_threshold: Some(thr),
                    ok,
                    junk,
                    hard_junk,
                    fail_kind: fail_kind.clone(),
                    elapsed_ms: Some(dur_ms as u64),
                    cost_units: Some(r.num_examples as u64),
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
                    st = muxer::exp3ix_update_persisted(
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

    // Persist history best-effort for future runs.
    history.save(&hist_path);

    // If we got here, the harness executed. Failures are recorded, not fatal.
    // (This job is intended to find regressions over time, not block all merges on flaky data.)
}

#[cfg(test)]
#[test]
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
    };

    let mut wnut = Window::new(50);
    for _ in 0..10 {
        wnut.push(Outcome {
            ok: true,
            http_429: false,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1,
        });
    }
    prior.windows.insert(
        BackendHistory::dataset_key("stacked", DatasetId::Wnut17),
        wnut,
    );

    let mut de = Window::new(50);
    for _ in 0..10 {
        de.push(Outcome {
            ok: false,
            http_429: false,
            junk: true,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1,
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
    };
    let mut w = Window::new(50);
    for _ in 0..10 {
        w.push(Outcome {
            ok: true,
            http_429: false,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1,
        });
    }
    prior.windows.insert("stacked".to_string(), w);

    let current = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
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
    };
    let mut w_prior = Window::new(50);
    for _ in 0..10 {
        w_prior.push(Outcome {
            ok: true,
            http_429: false,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1, // fast
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
    };
    let mut w_obs = Window::new(50);
    for _ in 0..3 {
        w_obs.push(Outcome {
            ok: true,
            http_429: false,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1, // fast
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
    };
    let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let expected = mh::pick_random_subset(0 ^ 0xC0E1_1A11, &arms, 1)
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
    };
    let mut w = Window::new(50);
    for _ in 0..10 {
        w.push(Outcome {
            ok: true,
            http_429: false,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 1,
        });
    }
    prior.windows.insert("stacked".to_string(), w);

    let current = BackendHistory {
        version: 3,
        window_cap: 50,
        windows: BTreeMap::new(),
        fail_kinds: BTreeMap::new(),
        exp3ix_state: None,
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
