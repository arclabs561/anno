//! CI-friendly randomized matrix test (muxer-backed).
//!
//! This replaces the older archived test harness with something that:
//! - compiles under `--features eval-advanced` (no `cli` feature required)
//! - uses `muxer` for deterministic, windowed MAB-style backend selection
//! - respects `ANNO_CI_SEED` and `ANNO_SAMPLE_STRATEGY`
//! - only evaluates cached datasets when running in CI (fast + offline)
//!
//! Environment variables:
//! - `ANNO_CI_SEED`: u64 seed (default: 0)
//! - `ANNO_SAMPLE_STRATEGY`: `random` | `ml-only` | `worst-first` (default: `ml-only`)
//! - `ANNO_MATRIX_PERSPECTIVE`: `ner` | `coref` | `coalesce` | `relation` (default: `ner`)
//! - `ANNO_HISTORY_FILE`: optional JSON path for muxer history
//! - `ANNO_MAX_EXAMPLES`: max examples per dataset (default: 20 in CI, 50 locally)
//! - `ANNO_CACHE_DIR`: cache root (datasets live under `$ANNO_CACHE_DIR/datasets`)
//! - `ANNO_ML_IN_MATRIX`: include ML-ish backends in candidates (`1`/`true`)
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
//! - `ANNO_MUXER_VERBOSE`: print chosen slice + per-result outcomes (`1`/`true`)
//! - `ANNO_MUXER_DECISIONS_FILE`: optional path to write selection decisions as JSONL
//! - `ANNO_MUXER_DECISIONS_TOP`: max candidate rows to include per decision (default: 8)
//!
//! Worst-first tuning (regression hunting):
//! - `ANNO_WORST_EXPLORATION_C`: exploration coefficient for worst-first (default: 0.8)
//! - `ANNO_WORST_HARD_WEIGHT`: weight for hard failures in worst-first (default: 1.0)
//! - `ANNO_WORST_SOFT_WEIGHT`: weight for soft junk in worst-first (default: 0.0)
//!
//! Notes:
//! - “worst-first” here means “prioritize historically bad outcomes” where “bad” is
//!   an eval failure or low F1, recorded into a muxer window.

#![cfg(all(test, feature = "eval-advanced"))]

use crate::eval::backend_factory::BackendFactory;
use crate::eval::loader::{DatasetId, DatasetLoader};
use crate::eval::task_evaluator::{TaskEvalConfig, TaskEvaluator};
use crate::eval::task_mapping::{backend_tasks, dataset_tasks, get_task_backends, Task};
use crate::muxer_harness as mh;
use muxer::{MabConfig, Outcome, Summary, Window};
use std::collections::VecDeque;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
enum SampleStrategy {
    Random,
    MlOnly,
    WorstFirst,
}

impl SampleStrategy {
    fn from_env() -> Self {
        match std::env::var("ANNO_SAMPLE_STRATEGY")
            .ok()
            .unwrap_or_else(|| "ml-only".to_string())
            .to_lowercase()
            .as_str()
        {
            "random" => Self::Random,
            "worst-first" | "worstfirst" => Self::WorstFirst,
            // Historical default in the legacy harness.
            "ml-only" | "mlonly" | "ml" => Self::MlOnly,
            _ => Self::MlOnly,
        }
    }
}

fn in_ci() -> bool {
    std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok()
}

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

#[derive(Debug, Clone, Copy)]
struct WorstFirstConfig {
    exploration_c: f64,
    hard_weight: f64,
    soft_weight: f64,
}

fn worst_first_config_from_env() -> WorstFirstConfig {
    WorstFirstConfig {
        exploration_c: mh::env_f64("ANNO_WORST_EXPLORATION_C", 0.8).max(0.0),
        hard_weight: mh::env_f64("ANNO_WORST_HARD_WEIGHT", 1.0).max(0.0),
        soft_weight: mh::env_f64("ANNO_WORST_SOFT_WEIGHT", 0.0).max(0.0),
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct DecisionCandidate {
    arm: String,
    calls: u64,
    ok_rate: f64,
    junk_rate: f64,
    hard_junk_rate: f64,
    score: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DecisionLog {
    run_id: String,
    strategy: String,
    perspective: String,
    muxer_profile: Option<String>,
    latency_guardrail_max_mean_ms: Option<u64>,
    latency_guardrail_allow_fewer: Option<bool>,
    latency_guardrail_require_measured: Option<bool>,
    round: usize,
    datasets: Vec<String>,
    remaining: Vec<String>,
    chosen: String,
    explore_first: bool,
    constraints_fallback_used: Option<bool>,
    eligible_arms: Option<Vec<String>>,
    top_candidates: Vec<DecisionCandidate>,
}

fn append_jsonl(path: &str, v: &DecisionLog) {
    if path.trim().is_empty() {
        return;
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

#[derive(Debug, Clone, Copy)]
enum MatrixPerspective {
    Ner,
    Coref,
    Coalesce,
    Relation,
}

impl MatrixPerspective {
    fn from_env() -> Self {
        match std::env::var("ANNO_MATRIX_PERSPECTIVE")
            .ok()
            .unwrap_or_else(|| "ner".to_string())
            .to_lowercase()
            .as_str()
        {
            "coref" | "intra-coref" | "intracoref" => Self::Coref,
            "coalesce" | "inter-coref" | "intercoref" | "cdcr" => Self::Coalesce,
            "relation" | "rel" => Self::Relation,
            _ => Self::Ner,
        }
    }

    fn tasks(&self) -> Vec<Task> {
        match self {
            Self::Ner => vec![Task::NER],
            Self::Coref => vec![Task::IntraDocCoref],
            Self::Coalesce => vec![Task::InterDocCoref],
            Self::Relation => vec![Task::RelationExtraction],
        }
    }

    fn preferred_datasets(&self) -> &'static [DatasetId] {
        match self {
            Self::Ner => &[
                DatasetId::WikiGold,
                DatasetId::Wnut17,
                DatasetId::MitRestaurant,
                DatasetId::MitMovie,
                DatasetId::CoNLL2003Sample,
            ],
            // Prefer smaller/common coref datasets that are likely to be cached.
            Self::Coref => &[
                DatasetId::GAP,
                DatasetId::PreCo,
                DatasetId::OntoNotesCoref,
                DatasetId::LitBank,
            ],
            // Cross-doc / CDCR style.
            Self::Coalesce => &[DatasetId::ECBPlus, DatasetId::WikiCoref],
            Self::Relation => &[DatasetId::DocRED],
        }
    }

    fn tag(&self) -> &'static str {
        match self {
            Self::Ner => "ner",
            Self::Coref => "coref",
            Self::Coalesce => "coalesce",
            Self::Relation => "relation",
        }
    }
}

fn history_path(perspective: MatrixPerspective) -> PathBuf {
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
        None => format!("muxer_history.{}.json", perspective.tag()),
        Some(s) => format!("muxer_history.{}.salt={}.json", perspective.tag(), s),
    };

    // Prefer ANNO_CACHE_DIR so it matches CI caching.
    if let Ok(dir) = std::env::var("ANNO_CACHE_DIR") {
        return PathBuf::from(dir).join(suffix);
    }

    // Fallback: place next to default dataset cache root.
    // DatasetLoader::new() uses platform cache: ~/.cache/anno/datasets (linux).
    // We'll store history alongside `.../anno/`.
    #[cfg(feature = "eval")]
    {
        if let Some(base) = dirs::cache_dir() {
            return base.join("anno").join(suffix);
        }
    }
    PathBuf::from(".").join(suffix)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BackendHistory {
    #[serde(default)]
    version: u32,
    window_cap: usize,
    windows: BTreeMap<String, Window>,
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

                return Self {
                    version: 3,
                    window_cap: cap,
                    windows,
                };
            }
        }

        Self {
            version: 3,
            window_cap: window_cap.max(1),
            windows: BTreeMap::new(),
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

    fn dataset_key(backend: &str, dataset: DatasetId) -> String {
        // Stable, human-readable key. `{:?}` is the enum variant name.
        format!("{backend}@@{:?}", dataset)
    }

    fn summaries_for(
        &self,
        arms: &[String],
        datasets: Option<&[DatasetId]>,
        per_dataset: bool,
    ) -> BTreeMap<String, Summary> {
        let mut out = BTreeMap::new();
        for a in arms {
            // Prefer dataset-scoped windows when we have a dataset slice:
            // this avoids "objective bleed" across unrelated datasets while still allowing
            // a stable per-backend decision.
            if per_dataset {
                if let Some(datasets) = datasets {
                    let mut agg = Summary::default();
                    for &ds in datasets {
                        let k = Self::dataset_key(a, ds);
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
                            agg.elapsed_ms_sum =
                                agg.elapsed_ms_sum.saturating_add(s.elapsed_ms_sum);
                        }
                    }
                    if agg.calls > 0 {
                        out.insert(a.clone(), agg);
                        continue;
                    }
                }
            }

            // Fallback: global per-backend window (older history files, or when dataset-scoped windows
            // are not yet populated).
            let s = self.windows.get(a).map(|w| w.summary()).unwrap_or_default();
            out.insert(a.clone(), s);
        }
        out
    }
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
    history: &BackendHistory,
    candidates_in_order: &[String],
    datasets: Option<&[DatasetId]>,
    k: usize,
) -> Vec<String> {
    if candidates_in_order.is_empty() || k == 0 {
        return Vec::new();
    }

    match strategy {
        SampleStrategy::Random => pick_random_subset(seed, candidates_in_order, k),
        SampleStrategy::MlOnly => {
            // MAB selection (muxer): pick historically "best" arms (high ok_rate, low junk),
            // with exploration to avoid fixating too early.
            let cfg = mab_config_from_env();
            let per_dataset = mh::env_bool("ANNO_MUXER_PER_DATASET", true);
            let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
            let decisions_path = std::env::var("ANNO_MUXER_DECISIONS_FILE").ok();
            let decisions_top = mh::env_usize("ANNO_MUXER_DECISIONS_TOP", 8).max(1);
            let run_id = format!(
                "seed={} perspective={:?} strategy={:?}",
                seed,
                MatrixPerspective::from_env(),
                strategy
            );

            let mut remaining: Vec<String> = candidates_in_order.to_vec();
            let mut chosen: Vec<String> = Vec::new();
            for round in 0..k.min(remaining.len()) {
                let summaries = history.summaries_for(&remaining, datasets, per_dataset);

                // Optional latency guardrail: filter out arms whose mean latency (ms) is too high.
                // This is about “proper experience”: keep the default run fast and predictable.
                // If the filter eliminates all arms, either fall back or stop early (select <K),
                // depending on env.
                let guard = mh::latency_guardrail_from_env();
                let pick_from: Vec<String> = if let Some(ms) = guard.max_mean_ms {
                    let mut eligible: Vec<String> = remaining
                        .iter()
                        .filter(|a| {
                            let s = summaries.get(*a).copied().unwrap_or_default();
                            if guard.require_measured && s.calls == 0 {
                                return false;
                            }
                            s.mean_elapsed_ms() <= ms
                        })
                        .cloned()
                        .collect();
                    if eligible.is_empty() {
                        if guard.allow_fewer && !chosen.is_empty() {
                            if verbose {
                                eprintln!(
                                    "matrix-muxer: ml-only latency guardrail filtered all remaining arms (max_mean_ms={:.0}); stopping early (chosen={})",
                                    ms,
                                    chosen.len()
                                );
                            }
                            Vec::new()
                        } else {
                            if verbose {
                                eprintln!(
                                    "matrix-muxer: ml-only latency guardrail filtered all arms (max_mean_ms={:.0}); falling back",
                                    ms
                                );
                            }
                            remaining.clone()
                        }
                    } else {
                        eligible.sort();
                        eligible
                    }
                } else {
                    remaining.clone()
                };

                if pick_from.is_empty() {
                    break;
                }

                // Use muxer’s “explain” API so we can surface constraints/explore-first decisions
                // when debugging the harness.
                let d = muxer::select_mab_explain(&pick_from, &summaries, cfg);
                if verbose {
                    eprintln!(
                        "matrix-muxer: mab round={} remaining={} chosen={} explore_first={} constraints_fallback={}",
                        round + 1,
                        pick_from.len(),
                        d.selection.chosen,
                        d.explore_first,
                        d.constraints_fallback_used
                    );
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
                    let mut rows: Vec<(f64, &muxer::CandidateDebug)> = Vec::new();
                    for c in &d.selection.candidates {
                        let score = c.objective_success
                            - cfg.cost_weight * c.mean_cost_units
                            - cfg.latency_weight * c.mean_elapsed_ms
                            - cfg.hard_junk_weight * c.hard_junk_rate
                            - cfg.junk_weight * c.soft_junk_rate;
                        rows.push((score, c));
                    }
                    rows.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
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
                let pick = d.selection.chosen.clone();

                if let Some(ref p) = decisions_path {
                    let mut rows: Vec<(f64, &muxer::CandidateDebug)> = Vec::new();
                    for c in &d.selection.candidates {
                        let score = c.objective_success
                            - cfg.cost_weight * c.mean_cost_units
                            - cfg.latency_weight * c.mean_elapsed_ms
                            - cfg.hard_junk_weight * c.hard_junk_rate
                            - cfg.junk_weight * c.soft_junk_rate;
                        rows.push((score, c));
                    }
                    rows.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
                    let top_candidates: Vec<DecisionCandidate> = rows
                        .into_iter()
                        .take(decisions_top)
                        .map(|(score, c)| DecisionCandidate {
                            arm: c.name.clone(),
                            calls: c.calls,
                            ok_rate: c.ok_rate,
                            junk_rate: c.junk_rate,
                            hard_junk_rate: c.hard_junk_rate,
                            score,
                        })
                        .collect();
                    let ds: Vec<String> = datasets
                        .unwrap_or(&[])
                        .iter()
                        .map(|d| format!("{d:?}"))
                        .collect();
                    let profile = std::env::var("ANNO_MUXER_PROFILE").ok();
                    append_jsonl(
                        p,
                        &DecisionLog {
                            run_id: run_id.clone(),
                            strategy: "ml-only".to_string(),
                            perspective: format!("{:?}", MatrixPerspective::from_env()),
                            muxer_profile: profile,
                            latency_guardrail_max_mean_ms: guard
                                .max_mean_ms
                                .map(|x| x.round() as u64),
                            latency_guardrail_allow_fewer: Some(guard.allow_fewer),
                            latency_guardrail_require_measured: Some(guard.require_measured),
                            round: round + 1,
                            datasets: ds,
                            remaining: remaining.clone(),
                            chosen: pick.clone(),
                            explore_first: d.explore_first,
                            constraints_fallback_used: Some(d.constraints_fallback_used),
                            eligible_arms: Some(d.eligible_arms.clone()),
                            top_candidates,
                        },
                    );
                }
                remaining.retain(|b| b != &pick);
                chosen.push(pick);
            }
            chosen
        }
        SampleStrategy::WorstFirst => {
            // Worst-first selection is intentionally *not* muxer::select_mab:
            // muxer is designed to pick "good" providers. Here we want to bias toward
            // historically bad/flaky arms (to find regressions), while still exploring.
            //
            // Score: higher means "worse", with an exploration term favoring under-sampled arms.
            let wcfg = worst_first_config_from_env();
            let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
            let decisions_path = std::env::var("ANNO_MUXER_DECISIONS_FILE").ok();
            let decisions_top = mh::env_usize("ANNO_MUXER_DECISIONS_TOP", 8).max(1);
            let run_id = format!(
                "seed={} perspective={:?} strategy={:?}",
                seed,
                MatrixPerspective::from_env(),
                strategy
            );
            let mut remaining: Vec<String> = candidates_in_order.to_vec();
            let mut chosen: Vec<String> = Vec::new();

            for round in 0..k.min(remaining.len()) {
                let summaries = history.summaries_for(
                    &remaining,
                    datasets,
                    mh::env_bool("ANNO_MUXER_PER_DATASET", true),
                );

                // Explore unseen arms first (stable order), so we eventually saturate coverage.
                if let Some(unseen) = remaining
                    .iter()
                    .find(|b| summaries.get(*b).copied().unwrap_or_default().calls == 0)
                {
                    let pick = unseen.clone();
                    if verbose {
                        eprintln!(
                            "matrix-muxer: worst-first round={} remaining={} chosen={} explore_first=true",
                            round + 1,
                            remaining.len(),
                            pick
                        );
                    }
                    if let Some(ref p) = decisions_path {
                        let ds: Vec<String> = datasets
                            .unwrap_or(&[])
                            .iter()
                            .map(|d| format!("{d:?}"))
                            .collect();
                        let profile = std::env::var("ANNO_MUXER_PROFILE").ok();
                        append_jsonl(
                            p,
                            &DecisionLog {
                                run_id: run_id.clone(),
                                strategy: "worst-first".to_string(),
                                perspective: format!("{:?}", MatrixPerspective::from_env()),
                                muxer_profile: profile,
                                latency_guardrail_max_mean_ms: None,
                                latency_guardrail_allow_fewer: None,
                                latency_guardrail_require_measured: None,
                                round: round + 1,
                                datasets: ds,
                                remaining: remaining.clone(),
                                chosen: pick.clone(),
                                explore_first: true,
                                constraints_fallback_used: None,
                                eligible_arms: None,
                                top_candidates: Vec::new(),
                            },
                        );
                    }
                    remaining.retain(|b| b != &pick);
                    chosen.push(pick);
                    continue;
                }

                let total_calls: f64 = summaries
                    .values()
                    .map(|s| s.calls as f64)
                    .sum::<f64>()
                    .max(1.0);

                let mut rows: Vec<(f64, String, Summary)> = Vec::new();
                for b in &remaining {
                    let s = summaries.get(b).copied().unwrap_or_default();
                    let calls = (s.calls as f64).max(1.0);
                    let hard_junk = s.hard_junk_rate();
                    let soft_junk = s.soft_junk_rate();

                    // Regression hunting should prioritize *instability* (hard failures) over
                    // “low quality” (soft junk / low F1), otherwise weak-but-working baselines
                    // dominate the schedule forever.
                    let exploration = wcfg.exploration_c * ((total_calls.ln() / calls).sqrt());
                    let score =
                        wcfg.hard_weight * hard_junk + wcfg.soft_weight * soft_junk + exploration;
                    rows.push((score, b.clone(), s));
                }

                rows.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                let pick = rows
                    .first()
                    .map(|(_, b, _)| b.clone())
                    .unwrap_or_else(|| remaining[round].clone());
                if verbose {
                    eprintln!(
                        "matrix-muxer: worst-first round={} remaining={} chosen={} explore_first=false",
                        round + 1,
                        remaining.len(),
                        pick
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
                    let top_candidates: Vec<DecisionCandidate> = rows
                        .iter()
                        .take(decisions_top)
                        .map(|(score, arm, s)| DecisionCandidate {
                            arm: arm.clone(),
                            calls: s.calls,
                            ok_rate: s.ok_rate(),
                            junk_rate: s.junk_rate(),
                            hard_junk_rate: s.hard_junk_rate(),
                            score: *score,
                        })
                        .collect();
                    let ds: Vec<String> = datasets
                        .unwrap_or(&[])
                        .iter()
                        .map(|d| format!("{d:?}"))
                        .collect();
                    let profile = std::env::var("ANNO_MUXER_PROFILE").ok();
                    append_jsonl(
                        p,
                        &DecisionLog {
                            run_id: run_id.clone(),
                            strategy: "worst-first".to_string(),
                            perspective: format!("{:?}", MatrixPerspective::from_env()),
                            muxer_profile: profile,
                            latency_guardrail_max_mean_ms: None,
                            latency_guardrail_allow_fewer: None,
                            latency_guardrail_require_measured: None,
                            round: round + 1,
                            datasets: ds,
                            remaining: remaining.clone(),
                            chosen: pick.clone(),
                            explore_first: false,
                            constraints_fallback_used: None,
                            eligible_arms: None,
                            top_candidates,
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

fn backend_candidates(strategy: SampleStrategy, tasks: &[Task]) -> Vec<String> {
    // Start from what is actually feature-enabled in this build.
    let available: BTreeSet<String> = BackendFactory::available_backends()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Keep the matrix aligned with what the evaluator will actually run:
    // TaskEvaluator filters explicit backends through `get_task_backends(task)`.
    let allowed: BTreeSet<&'static str> =
        tasks.iter().flat_map(|t| get_task_backends(*t)).collect();

    // CI default: baselines only.
    //
    // Rationale: the default feature set enables `onnx`, so ML backends are *available*
    // but can be expensive and flaky if model caches are cold. Make ML opt-in via env:
    // `ANNO_ML_IN_MATRIX=1`.
    let allow_ml = std::env::var("ANNO_ML_IN_MATRIX")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

    let mut out: Vec<String> = Vec::new();
    // Some ML backends require gated HuggingFace models. If no token is present, they will always
    // be “skipped” and waste matrix budget.
    crate::env::load_dotenv();
    let has_hf_token = crate::env::has_hf_token();
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
        if matches!(b, "stacked" | "crf" | "heuristic") {
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
        if allow_ml && available.contains(b) {
            out.push(b.to_string());
        }
    }

    match strategy {
        SampleStrategy::Random => {
            // Include the full available set, but keep a stable order.
            let mut all: Vec<String> = available.into_iter().collect();
            all.sort();
            // Also include coref resolvers when coref tasks are requested.
            if tasks.iter().any(|t| t.is_coref_family()) {
                for b in ["coref_resolver", "mention_ranking", "box"] {
                    if !all.iter().any(|x| x == b) {
                        all.push(b.to_string());
                    }
                }
                all.sort();
                all.dedup();
            }
            all
        }
        SampleStrategy::MlOnly | SampleStrategy::WorstFirst => {
            out.sort();
            out.dedup();
            out
        }
    }
}

fn cached_dataset_candidates(loader: &DatasetLoader) -> Vec<DatasetId> {
    // Prefer cached datasets only; TaskEvaluator can filter further by task support.
    let mut out = Vec::new();
    for &ds in DatasetId::all() {
        let Ok(loadable) = crate::eval::LoadableDatasetId::try_from(ds) else {
            continue;
        };
        if loader.is_cached(loadable) {
            out.push(ds);
        }
    }
    out
}

#[test]
fn test_randomized_matrix_sample() {
    crate::env::load_dotenv();

    let seed = ci_seed();
    let strategy = SampleStrategy::from_env();
    let perspective = MatrixPerspective::from_env();
    let hist_path = history_path(perspective);
    let window_cap = env_usize("ANNO_MUXER_WINDOW_CAP", 50);

    let loader = DatasetLoader::new().expect("DatasetLoader::new");
    let cached_datasets = cached_dataset_candidates(&loader);

    // If CI cache is cold, do not fail the whole build; this job is intended to be best-effort.
    // (Other jobs provide strict compile/test guarantees.)
    if cached_datasets.is_empty() {
        eprintln!(
            "matrix-muxer: no cached datasets found; set ANNO_CACHE_DIR and cache it in CI (path={:?})",
            loader.cache_dir()
        );
        return;
    }

    let mut history = BackendHistory::load(&hist_path, window_cap);

    let tasks = perspective.tasks();

    // Datasets: choose a small number of cached datasets, biased toward small/common ones.
    // Keep this deterministic and stable across runs.
    //
    // Important: for coref-family tasks, "dataset file exists" is not sufficient:
    // `TaskEvaluator` loads coreference documents via `loader.load_coref()` and will
    // treat "empty docs" as "not cached" when `require_cached=1`. Some catalog
    // datasets have placeholder coref loaders and will return empty docs even when a
    // cache file exists. We filter those out here to avoid poisoning muxer history
    // with configuration/loader failures.
    let coref_tasks_requested = tasks.iter().any(|t| t.is_coref_family());
    let mut coref_dataset_ok: HashMap<DatasetId, bool> = HashMap::new();
    let eligible_cached: Vec<DatasetId> = cached_datasets
        .iter()
        .copied()
        .filter(|d| {
            let ts = dataset_tasks(*d);
            if !tasks.iter().any(|t| ts.contains(t)) {
                return false;
            }
            if coref_tasks_requested && d.is_coreference() {
                if let Some(ok) = coref_dataset_ok.get(d) {
                    return *ok;
                }
                let ok = match loader.load_coref(*d) {
                    Ok(docs) => !docs.is_empty(),
                    Err(_) => false,
                };
                coref_dataset_ok.insert(*d, ok);
                return ok;
            }
            true
        })
        .collect();
    if eligible_cached.is_empty() {
        eprintln!(
            "matrix-muxer: no cached datasets support tasks={:?} (cache_dir={:?})",
            tasks,
            loader.cache_dir()
        );
        return;
    }

    let datasets_per_run = env_usize("ANNO_MUXER_DATASETS_PER_RUN", 2).max(1);
    let mut chosen_datasets: Vec<DatasetId> = perspective
        .preferred_datasets()
        .iter()
        .copied()
        .filter(|d| eligible_cached.contains(d))
        .take(datasets_per_run)
        .collect();
    if chosen_datasets.len() < datasets_per_run {
        // Fallback: deterministic sample of whatever is cached.
        let ds_strings: Vec<String> = eligible_cached.iter().map(|d| format!("{:?}", d)).collect();
        let chosen_ds_strings =
            pick_random_subset(seed ^ 0xDADA_BEEF, &ds_strings, datasets_per_run);
        chosen_datasets.extend(
            eligible_cached
                .into_iter()
                .filter(|d| chosen_ds_strings.contains(&format!("{:?}", d))),
        );
        chosen_datasets.sort_by_key(|d| format!("{:?}", d));
        chosen_datasets.dedup();
        chosen_datasets.truncate(datasets_per_run);
    }

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
    if candidates.is_empty() {
        eprintln!(
            "matrix-muxer: no backend candidates support tasks={:?} for this feature set",
            tasks
        );
        return;
    }

    let per_dataset = mh::env_bool("ANNO_MUXER_PER_DATASET", true);
    let backends_per_run = mh::env_usize("ANNO_MUXER_BACKENDS_PER_RUN", 2).max(1);
    let chosen_backends = if per_dataset {
        select_backends(
            strategy,
            seed,
            &history,
            &candidates,
            Some(&chosen_datasets),
            backends_per_run,
        )
    } else {
        select_backends(
            strategy,
            seed,
            &history,
            &candidates,
            None,
            backends_per_run,
        )
    };

    let verbose = mh::env_bool("ANNO_MUXER_VERBOSE", false);
    if verbose {
        eprintln!(
            "matrix-muxer: perspective={} strategy={:?} seed={} per_dataset={} datasets={:?} backends={:?}",
            perspective.tag(),
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
        require_cached: true,
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
        // Update both:
        // - global per-backend window (back-compat + overall view)
        // - dataset-scoped per-backend window (preferred for selection within a slice)
        history.window_mut(&r.backend).push(o);
        if mh::env_bool("ANNO_MUXER_PER_DATASET", true) {
            let k = BackendHistory::dataset_key(&r.backend, r.dataset);
            history.window_mut(&k).push(o);
        }
    }

    // Persist history best-effort for future runs.
    history.save(&hist_path);

    // If we got here, the harness executed. Failures are recorded, not fatal.
    // (This job is intended to find regressions over time, not block all merges on flaky data.)
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
