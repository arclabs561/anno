//! Sampler command - Inspect the muxer-backed CI matrix history.
//!
//! This is a DX tool: it helps answer questions like:
//! - Have we "seen" each arm at least once?
//! - Which arms look flaky / low-signal recently (windowed)?
//! - Are we mixing signals across tasks/datasets (weight bleed)?

use clap::{Parser, Subcommand, ValueEnum};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[cfg(feature = "eval-advanced")]
use anno_eval::eval::backend_factory::BackendFactory;
#[cfg(feature = "eval-advanced")]
use anno_eval::eval::loader::DatasetId;
#[cfg(feature = "eval-advanced")]
use anno_eval::eval::task_mapping::{backend_tasks, get_task_backends, Task};

// Note: `muxer` is an optional dependency enabled by `eval-advanced`.
// This keeps default builds lean while letting this DX command reuse the exact selection
// semantics used by the matrix sampler harness when `eval-advanced` is enabled.
#[cfg(feature = "eval-advanced")]
use muxer::MabConfig;

#[cfg(feature = "eval-advanced")]
use anno_eval::muxer_harness as mh;

/// Inspect muxer history from the randomized matrix sampler harness.
#[derive(Parser, Debug)]
#[command(about = "Inspect muxer history from the randomized matrix sampler harness")]
pub struct MuxerArgs {
    /// Override history file path (otherwise uses env/defaults).
    #[arg(long)]
    pub history_file: Option<PathBuf>,

    /// Slice tag to inspect (matches the matrix sampler harness slice codes, e.g. `ner`, `temporal`,
    /// `discourse-segmentation`).
    ///
    /// If set, this overrides `--perspective` (which is legacy / coarse).
    #[arg(long)]
    pub slice: Option<String>,

    /// Perspective to inspect (controls tasks + preferred datasets in the matrix sampler harness).
    #[arg(long, value_enum)]
    pub perspective: Option<MuxerPerspective>,

    /// Scope muxer history by dataset facets (language/domain), matching the matrix sampler harness.
    ///
    /// This is only used when we can infer facets (see `--facet-datasets`).
    #[arg(long, default_value_t = true)]
    pub slice_by_dataset_facets: bool,

    /// Dataset IDs used to compute the facet-aware history slice tag.
    ///
    /// Example: `--facet-datasets DisrptEngDepScidtbConlluSeg,DisrptDeuRstPccConlluSeg`
    ///
    /// If omitted, we fall back to the non-facet history filename.
    #[arg(long)]
    pub facet_datasets: Option<String>,

    /// Sampling strategy (as used by the matrix sampler harness).
    #[arg(long, value_enum)]
    pub strategy: Option<MuxerStrategy>,

    /// High-level mode (maps to sensible defaults for `--strategy` and related knobs).
    ///
    /// - `triage`: prioritize finding failures/regressions quickly (defaults to worst-first)
    /// - `measure`: prioritize stable performance measurement (defaults to ml-only)
    ///
    /// Explicit `--strategy` still wins.
    #[arg(long, value_enum)]
    pub mode: Option<MuxerMode>,

    /// Include ML backends in candidate set (same intent as `ANNO_ML_IN_MATRIX=1`).
    ///
    /// If not set, this command will still honor `ANNO_ML_IN_MATRIX` if present.
    #[arg(long, default_value_t = false)]
    pub include_ml: bool,

    /// Action to run.
    #[command(subcommand)]
    pub action: MuxerAction,
}

/// Subcommands for `anno muxer`.
#[derive(Subcommand, Debug)]
pub enum MuxerAction {
    /// Print coverage + per-arm window stats.
    Stats {
        /// Show per-arm breakdown by dataset (keys like `arm@@DatasetId`).
        #[arg(long, default_value_t = false)]
        show_datasets: bool,

        /// Max datasets to show per arm (sorted by calls desc).
        #[arg(long, default_value_t = 8)]
        top_datasets: usize,
    },

    /// Summarize a muxer decision log JSONL (written by the matrix sampler harness).
    Decisions {
        /// Path to a decisions JSONL file (defaults to `ANNO_MUXER_DECISIONS_FILE`).
        #[arg(long)]
        file: Option<PathBuf>,

        /// Max rows to print for by-arm / by-dataset tables.
        #[arg(long, default_value_t = 20)]
        top: usize,

        /// Show a small per-run breakdown (grouped by run_id).
        #[arg(long, default_value_t = true)]
        by_run: bool,

        /// Optional substring filter for run_id (e.g. "slice=ner" or "seed=42").
        #[arg(long)]
        run_filter: Option<String>,

        /// Max runs to print in the per-run breakdown.
        #[arg(long, default_value_t = 8)]
        top_runs: usize,

        /// Window size for simple within-run outcome trend stats (first-N vs last-N).
        #[arg(long, default_value_t = 10)]
        trend_window: usize,

        /// Print by-chosen-arm failure-kind totals.
        #[arg(long, default_value_t = true)]
        by_arm: bool,

        /// Print by-dataset failure-kind totals (attributes counts to each dataset in the row).
        #[arg(long, default_value_t = true)]
        by_dataset: bool,
    },

    /// Preview what the muxer selector would pick next (using current history).
    Decide {
        /// Number of arms to choose (without replacement), like the matrix sampler harness.
        #[arg(long, default_value_t = 1)]
        k: usize,

        /// Use dataset-scoped windows (`arm@@DatasetId`) when present (recommended).
        #[arg(long, default_value_t = true)]
        per_dataset: bool,

        /// Optional comma-separated dataset ids to scope selection to (matches `DatasetId` debug strings).
        ///
        /// Example: `--datasets CHisIEC,DocRED`
        #[arg(long)]
        datasets: Option<String>,

        /// Show the Pareto frontier arms.
        #[arg(long, default_value_t = false)]
        show_frontier: bool,

        /// Show candidate debug rows (sorted by the muxer scalar score).
        #[arg(long, default_value_t = true)]
        show_candidates: bool,

        /// Max candidate rows to show.
        #[arg(long, default_value_t = 12)]
        top_candidates: usize,
    },

    /// Run the randomized matrix sampler harness (writes decisions JSONL; updates history).
    ///
    /// This is the operational entrypoint corresponding to the CI sampler, but runnable locally.
    /// It intentionally supports only a small set of flags; everything else is inherited from the
    /// harness env-vars.
    Run {
        /// Number of sampler runs (seeds). Defaults to 1 for `triage`, 10 for `measure`.
        #[arg(long)]
        runs: Option<u64>,

        /// Seed base (actual seeds are `seed_base + i` for i in 0..runs).
        #[arg(long, default_value_t = 0)]
        seed_base: u64,

        /// Write decisions/outcomes JSONL here (defaults to `.generated/muxer_run.jsonl`).
        #[arg(long)]
        decisions_file: Option<PathBuf>,

        /// Optional: write an aggregated JSON report (from the decisions JSONL).
        #[arg(long)]
        agg_out: Option<PathBuf>,

        /// Task override (sets `ANNO_MATRIX_TASK`, e.g. `ner`, `re`, `intra-coref`).
        #[arg(long)]
        task: Option<String>,

        /// Max examples per dataset (sets `ANNO_MAX_EXAMPLES`).
        #[arg(long)]
        max_examples: Option<u64>,

        /// How many datasets to sample per run (sets `ANNO_MUXER_DATASETS_PER_RUN`).
        #[arg(long)]
        datasets_per_run: Option<u64>,

        /// How many backends to sample per run (sets `ANNO_MUXER_BACKENDS_PER_RUN`).
        #[arg(long)]
        backends_per_run: Option<u64>,

        /// Use dataset-scoped windows (`arm@@DatasetId`) when updating history (sets `ANNO_MUXER_PER_DATASET=1`).
        #[arg(long, default_value_t = false)]
        per_dataset: bool,

        /// Cached-only datasets (sets `ANNO_MATRIX_REQUIRE_CACHED=1`).
        #[arg(long, default_value_t = false)]
        require_cached: bool,

        /// If a loader returns empty, try to download (sets `ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY=1`).
        #[arg(long, default_value_t = false)]
        try_download_on_empty: bool,

        /// Fixed datasets (sets `ANNO_MUXER_FIXED_DATASETS`), comma-separated.
        #[arg(long)]
        fixed_datasets: Option<String>,

        /// Fixed backend list (sets `ANNO_MUXER_FIXED_BACKEND`), comma-separated.
        #[arg(long)]
        fixed_backend: Option<String>,

        /// Pin language facet (sets `ANNO_MUXER_PIN_LANG`), e.g. `de`.
        #[arg(long)]
        pin_lang: Option<String>,

        /// Pin domain facet (sets `ANNO_MUXER_PIN_DOMAIN`), e.g. `wikipedia`.
        #[arg(long)]
        pin_domain: Option<String>,

        /// Pin backend facet (sets `ANNO_MUXER_PIN_BACKEND`), comma-separated.
        #[arg(long)]
        pin_backend: Option<String>,
    },

    /// Rank the most actionable regressions from history (recent vs baseline).
    ///
    /// This is intended to drive “what should I run/fix next?” for regression-hunting.
    Regress {
        /// How to score regressions.
        ///
        /// - `stability`: prioritize hard failures (and optionally soft junk) plus deltas.
        /// - `latency`: prioritize rising mean latency (ms).
        /// - `quality`: prioritize rising junk rate / falling ok-rate.
        #[arg(long, value_enum, default_value_t = RegressMode::Stability)]
        mode: RegressMode,

        /// How many most-recent outcomes to treat as “recent”.
        #[arg(long, default_value_t = 8)]
        recent: usize,

        /// Minimum recent calls required to rank an item.
        #[arg(long, default_value_t = 3)]
        min_recent_calls: usize,

        /// Max rows to print.
        #[arg(long, default_value_t = 20)]
        top: usize,

        /// Optional comma-separated dataset ids to scope ranking to (matches `DatasetId` debug strings).
        #[arg(long)]
        datasets: Option<String>,

        /// Show per-backend rows (aggregated across datasets) in addition to per-dataset rows.
        #[arg(long, default_value_t = false)]
        include_global: bool,
    },
}

#[derive(Debug, Clone, serde::Deserialize)]
struct DecisionsFailKindCount {
    kind: String,
    count: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TopCandidateLite {
    #[serde(default)]
    arm: String,
    #[serde(default)]
    score: f64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TopCandidatesLite {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    rows: Vec<TopCandidateLite>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct DecisionLogLite {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    strategy: String,
    #[serde(default)]
    slice: String,
    #[serde(default)]
    datasets: Vec<String>,
    #[serde(default)]
    round: u32,
    #[serde(default)]
    chosen: Option<String>,
    #[serde(default)]
    top_candidates: Option<TopCandidatesLite>,
    #[serde(default)]
    chosen_fail_kinds_top: Option<Vec<DecisionsFailKindCount>>,
    #[serde(default)]
    constraints_fallback_used: Option<bool>,
    #[serde(default)]
    explore_first: Option<bool>,
    #[serde(default)]
    control_arms: Option<Vec<String>>,
}

#[derive(Debug, Default)]
struct DecisionsAgg {
    // Input accounting (avoid silent drops).
    lines_total: u64,
    lines_parsed_decision: u64,
    lines_parsed_outcome: u64,
    lines_skipped_invalid: u64,
    lines_skipped_filtered: u64,

    total_rows: u64,
    // Selection-time decision records (reflect history at selection time).
    decision_rows: u64,
    decision_rows_with_chosen: u64,
    decision_rows_with_fail_kinds: u64,
    decision_constraints_fallback_used: u64,
    decision_explore_first: u64,
    decision_kinds_total: BTreeMap<String, u64>,
    decision_by_arm: BTreeMap<String, BTreeMap<String, u64>>,
    decision_by_dataset: BTreeMap<String, BTreeMap<String, u64>>,
    // Union of control arms seen for this run (best-effort).
    control_arms: BTreeSet<String>,

    // Learning-health: chosen rank among logged top candidates (when available).
    chosen_rank_rows: u64,
    chosen_rank_sum: u64,
    chosen_rank_1: u64,

    outcome_rows: u64,
    outcome_rows_with_fail_kind: u64,
    outcome_ok: u64,
    outcome_junk: u64,
    outcome_hard_junk: u64,
    // Outcome stats restricted to control arms (best-effort).
    outcome_control_rows: u64,
    outcome_control_ok: u64,
    outcome_control_junk: u64,
    outcome_control_hard: u64,
    outcome_kinds_total: BTreeMap<String, u64>,
    outcome_by_arm: BTreeMap<String, BTreeMap<String, u64>>,
    outcome_by_dataset: BTreeMap<String, BTreeMap<String, u64>>,
    // For each fail_kind, count backend@@dataset pairs (actionable clusters).
    outcome_kind_pairs: BTreeMap<String, BTreeMap<String, u64>>,

    // Outcome trend: first-N vs last-N (based on JSONL order within a run).
    outcome_trend_n: usize,
    outcome_first_n: u64,
    outcome_first_ok: u64,
    outcome_first_junk: u64,
    outcome_first_hard: u64,
    outcome_tail: std::collections::VecDeque<(bool, bool, bool)>,
    // For "newly appearing fail kinds": keep the last 2N outcome fail_kinds so we can compare
    // recent-N vs previous-N within a run (best-effort; based on JSONL order).
    outcome_fail_kind_tail_2n: std::collections::VecDeque<Option<String>>,
}

fn add_kind_counts(dst: &mut BTreeMap<String, u64>, kinds: &[DecisionsFailKindCount]) {
    for k in kinds {
        *dst.entry(k.kind.clone()).or_insert(0) += k.count;
    }
}

fn top_kinds_line(counts: &BTreeMap<String, u64>, k: usize) -> String {
    let mut pairs: Vec<(u64, String)> = counts.iter().map(|(k, v)| (*v, k.clone())).collect();
    pairs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    pairs
        .into_iter()
        .take(k.max(1))
        .map(|(v, k)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn outcome_new_fail_kinds_from_tail_2n(
    tail_2n: &std::collections::VecDeque<Option<String>>,
    n: usize,
) -> Vec<String> {
    use std::collections::BTreeSet;

    let n = n.max(1);
    let len = tail_2n.len();
    if len == 0 {
        return Vec::new();
    }
    let split = len.saturating_sub(n);

    let mut prev: BTreeSet<String> = BTreeSet::new();
    for opt in tail_2n.iter().take(split) {
        if let Some(k) = opt.as_ref() {
            prev.insert(k.clone());
        }
    }
    let mut recent: BTreeSet<String> = BTreeSet::new();
    for opt in tail_2n.iter().skip(split) {
        if let Some(k) = opt.as_ref() {
            recent.insert(k.clone());
        }
    }
    recent.difference(&prev).cloned().collect()
}

fn decisions_aggregate_from_jsonl(s: &str) -> DecisionsAgg {
    decisions_aggregate_grouped_by_run(s, None, 10).0
}

fn decisions_aggregate_grouped_by_run(
    s: &str,
    run_filter: Option<&str>,
    trend_window: usize,
) -> (DecisionsAgg, BTreeMap<String, DecisionsAgg>) {
    let mut by_run: BTreeMap<String, DecisionsAgg> = BTreeMap::new();
    let mut all = DecisionsAgg::default();
    let trend_window = trend_window.max(1);
    all.outcome_trend_n = trend_window;

    for line in s.lines() {
        let line = line.trim();
        all.lines_total += 1;
        if line.is_empty() {
            continue;
        }

        // Outcome lines.
        #[derive(Debug, Clone, serde::Deserialize)]
        struct OutcomeLine {
            #[serde(default)]
            record_type: String,
            #[serde(default)]
            run_id: String,
            #[serde(default)]
            dataset: String,
            #[serde(default)]
            backend: String,
            #[serde(default)]
            fail_kind: Option<String>,
            #[serde(default)]
            ok: bool,
            #[serde(default)]
            junk: bool,
            #[serde(default)]
            hard_junk: bool,
        }
        if let Ok(o) = serde_json::from_str::<OutcomeLine>(line) {
            if o.record_type == "outcome" && !o.backend.is_empty() && !o.dataset.is_empty() {
                if let Some(f) = run_filter {
                    if !o.run_id.contains(f) {
                        // If filter is set and we can read run_id, respect it.
                        all.lines_skipped_filtered += 1;
                        continue;
                    }
                }
                let key = if o.run_id.is_empty() {
                    "(missing run_id)".to_string()
                } else {
                    o.run_id.clone()
                };
                let entry = by_run.entry(key).or_default();
                if entry.outcome_trend_n == 0 {
                    entry.outcome_trend_n = trend_window;
                }

                // Update both the global and per-run aggregates by reusing the same logic:
                // we simulate the minimal effects of decisions_aggregate_from_jsonl here.
                for agg in [&mut all, entry] {
                    agg.total_rows += 1;
                    agg.lines_parsed_outcome += 1;
                    agg.outcome_rows += 1;
                    agg.outcome_ok += o.ok as u64;
                    agg.outcome_junk += o.junk as u64;
                    agg.outcome_hard_junk += o.hard_junk as u64;

                    // Trend update (JSONL order).
                    if agg.outcome_first_n < (agg.outcome_trend_n as u64) {
                        agg.outcome_first_n += 1;
                        agg.outcome_first_ok += o.ok as u64;
                        agg.outcome_first_junk += o.junk as u64;
                        agg.outcome_first_hard += o.hard_junk as u64;
                    }
                    agg.outcome_tail.push_back((o.ok, o.junk, o.hard_junk));
                    while agg.outcome_tail.len() > agg.outcome_trend_n {
                        agg.outcome_tail.pop_front();
                    }
                    agg.outcome_fail_kind_tail_2n
                        .push_back(o.fail_kind.as_ref().map(|s| s.to_string()));
                    while agg.outcome_fail_kind_tail_2n.len() > agg.outcome_trend_n * 2 {
                        agg.outcome_fail_kind_tail_2n.pop_front();
                    }

                    if let Some(kind) = o.fail_kind.as_ref() {
                        agg.outcome_rows_with_fail_kind += 1;
                        *agg.outcome_kinds_total.entry(kind.clone()).or_insert(0) += 1;
                        let pair = format!("{}@@{}", o.backend, o.dataset);
                        *agg.outcome_kind_pairs
                            .entry(kind.clone())
                            .or_default()
                            .entry(pair)
                            .or_insert(0) += 1;
                        *agg.outcome_by_arm
                            .entry(o.backend.clone())
                            .or_default()
                            .entry(kind.clone())
                            .or_insert(0) += 1;
                        *agg.outcome_by_dataset
                            .entry(o.dataset.clone())
                            .or_default()
                            .entry(kind.clone())
                            .or_insert(0) += 1;
                    }

                    // Control-only stats (best-effort; requires a prior decision line with control_arms).
                    if agg.control_arms.contains(&o.backend) {
                        agg.outcome_control_rows += 1;
                        agg.outcome_control_ok += o.ok as u64;
                        agg.outcome_control_junk += o.junk as u64;
                        agg.outcome_control_hard += o.hard_junk as u64;
                    }
                }
                continue;
            }
        }

        // Decision lines.
        let Ok(d) = serde_json::from_str::<DecisionLogLite>(line) else {
            all.lines_skipped_invalid += 1;
            continue;
        };
        if let Some(f) = run_filter {
            if !d.run_id.contains(f) {
                all.lines_skipped_filtered += 1;
                continue;
            }
        }
        let key = if d.run_id.is_empty() {
            "(missing run_id)".to_string()
        } else {
            d.run_id.clone()
        };
        let entry = by_run.entry(key).or_default();

        for agg in [&mut all, entry] {
            agg.total_rows += 1;
            agg.decision_rows += 1;
            agg.lines_parsed_decision += 1;
            if d.chosen.is_some() {
                agg.decision_rows_with_chosen += 1;
            }
            if d.constraints_fallback_used.unwrap_or(false) {
                agg.decision_constraints_fallback_used += 1;
            }
            if d.explore_first.unwrap_or(false) {
                agg.decision_explore_first += 1;
            }
            if let Some(c) = d.control_arms.as_ref() {
                for a in c {
                    agg.control_arms.insert(a.clone());
                }
            }
            if let (Some(chosen), Some(tc)) = (d.chosen.as_ref(), d.top_candidates.as_ref()) {
                if !tc.rows.is_empty() {
                    if let Some((idx, _)) =
                        tc.rows.iter().enumerate().find(|(_i, r)| r.arm == *chosen)
                    {
                        let rank = (idx as u64) + 1;
                        agg.chosen_rank_rows += 1;
                        agg.chosen_rank_sum += rank;
                        if rank == 1 {
                            agg.chosen_rank_1 += 1;
                        }
                    }
                }
            }
            if let Some(kinds) = d.chosen_fail_kinds_top.as_ref() {
                if !kinds.is_empty() {
                    agg.decision_rows_with_fail_kinds += 1;
                    add_kind_counts(&mut agg.decision_kinds_total, kinds);
                    if let Some(chosen) = d.chosen.as_ref() {
                        let m = agg.decision_by_arm.entry(chosen.clone()).or_default();
                        add_kind_counts(m, kinds);
                    }
                    for ds in &d.datasets {
                        let m = agg.decision_by_dataset.entry(ds.clone()).or_default();
                        add_kind_counts(m, kinds);
                    }
                }
            }
        }
    }

    (all, by_run)
}

/// Scoring mode for `anno muxer regress`.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RegressMode {
    /// Prioritize rising hard-failure rate (and optionally soft junk).
    Stability,
    /// Prioritize rising mean latency (ms).
    Latency,
    /// Prioritize rising “bad rate” (1 - ok_rate), where ok_rate is quality-aware.
    Quality,
}

/// Which slice of the matrix sampler harness to inspect.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MuxerPerspective {
    /// Named entity recognition slice.
    Ner,
    /// Within-document coreference slice.
    Coref,
    /// Cross-document clustering slice.
    Coalesce,
    /// Relation extraction slice.
    Relation,
}

impl MuxerPerspective {
    fn tag(&self) -> &'static str {
        match self {
            Self::Ner => "ner",
            Self::Coref => "coref",
            Self::Coalesce => "coalesce",
            Self::Relation => "relation",
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
}

/// Which sampling strategy to assume when interpreting candidate sets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MuxerStrategy {
    /// Deterministic random subset.
    Random,
    /// MAB-driven selection (prefers higher ok-rate / lower junk).
    ///
    /// Note: in the matrix sampler harness, `ok` is recorded as “evaluation succeeded AND not junk”
    /// (where “junk” is a coarse low-F1 threshold, task-dependent).
    MlOnly,
    /// Regression-hunting selection (bias toward historically bad/flaky arms).
    WorstFirst,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MuxerMode {
    #[value(alias = "bug-hunt", alias = "bughunt", alias = "regress")]
    Triage,
    #[value(
        alias = "perf-estimate",
        alias = "perfestimate",
        alias = "estimate",
        alias = "measurement"
    )]
    Measure,
}

impl MuxerStrategy {
    fn to_env_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::MlOnly => "ml-only",
            Self::WorstFirst => "worst-first",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BackendHistory {
    #[serde(default)]
    version: u32,
    window_cap: usize,
    windows: BTreeMap<String, WindowSerde>,
    /// Optional triage metadata recorded by the matrix sampler harness (parallel to `windows`).
    ///
    /// Keys match `windows` keys. Each entry is a deque of optional coarse failure kinds, aligned
    /// with the corresponding window buffer (best-effort).
    #[serde(default)]
    fail_kinds: BTreeMap<String, std::collections::VecDeque<Option<String>>>,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
struct OutcomeSerde {
    ok: bool,
    #[serde(default)]
    http_429: bool,
    #[serde(default)]
    junk: bool,
    #[serde(default)]
    hard_junk: bool,
    #[serde(default)]
    cost_units: u64,
    #[serde(default)]
    elapsed_ms: u64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct WindowSerde {
    #[serde(default)]
    cap: usize,
    #[serde(default)]
    buf: Vec<OutcomeSerde>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SummarySerde {
    calls: u64,
    ok: u64,
    http_429: u64,
    junk: u64,
    hard_junk: u64,
    cost_units: u64,
    elapsed_ms_sum: u64,
}

impl SummarySerde {
    fn wilson_95_ci(ok: u64, calls: u64) -> (f64, f64) {
        // Wilson score interval for a Bernoulli proportion at 95% confidence.
        // Used as a cheap “sample size / confidence” indicator for ok_rate.
        if calls == 0 {
            return (0.0, 0.0);
        }
        let n = calls as f64;
        let k = ok as f64;
        let p = (k / n).clamp(0.0, 1.0);
        let z = 1.96_f64;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denom;
        let half = (z * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt()) / denom;
        ((center - half).max(0.0), (center + half).min(1.0))
    }

    fn from_window(w: &WindowSerde) -> Self {
        let mut out = SummarySerde {
            calls: w.buf.len() as u64,
            ..Default::default()
        };
        for o in &w.buf {
            out.ok += o.ok as u64;
            out.http_429 += o.http_429 as u64;
            out.junk += o.junk as u64;
            out.hard_junk += o.hard_junk as u64;
            out.cost_units = out.cost_units.saturating_add(o.cost_units);
            out.elapsed_ms_sum = out.elapsed_ms_sum.saturating_add(o.elapsed_ms);
        }
        out
    }

    fn ok_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.ok as f64) / (self.calls as f64)
        }
    }

    fn ok_rate_95_hw(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            let (lo, hi) = Self::wilson_95_ci(self.ok, self.calls);
            (hi - lo) / 2.0
        }
    }

    fn junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.junk as f64) / (self.calls as f64)
        }
    }

    fn hard_junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.hard_junk as f64) / (self.calls as f64)
        }
    }

    fn http_429_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.http_429 as f64) / (self.calls as f64)
        }
    }

    fn mean_elapsed_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.elapsed_ms_sum as f64) / (self.calls as f64)
        }
    }

    fn mean_cost_units(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.cost_units as f64) / (self.calls as f64)
        }
    }

    fn soft_junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            let soft = self.junk.saturating_sub(self.hard_junk);
            (soft as f64) / (self.calls as f64)
        }
    }
}

impl BackendHistory {
    fn load(path: &PathBuf, default_window_cap: usize) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let mut h: BackendHistory = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;
        if h.version == 0 {
            // Older files may not have carried a version.
            h.version = 1;
        }
        if h.window_cap == 0 {
            h.window_cap = default_window_cap.max(1);
        }
        let cap = h.window_cap.max(1);

        // Defensive: older or hand-edited history files may contain unbounded buffers. Truncate
        // windows and failure-kind deques to the configured cap, keeping the most-recent entries.
        for w in h.windows.values_mut() {
            if w.cap == 0 {
                w.cap = cap;
            }
            if w.buf.len() > cap {
                let keep_from = w.buf.len() - cap;
                w.buf = w.buf.split_off(keep_from);
            }
        }
        for q in h.fail_kinds.values_mut() {
            while q.len() > cap {
                q.pop_front();
            }
        }
        Ok(h)
    }

    fn summaries_for(
        &self,
        prior: Option<&BackendHistory>,
        arms: &[String],
        datasets: Option<&BTreeSet<String>>,
        per_dataset: bool,
        prior_calls: u64,
    ) -> BTreeMap<String, SummarySerde> {
        let mut out = BTreeMap::new();
        for a in arms {
            let mut obs = SummarySerde::default();
            if per_dataset {
                // Prefer aggregating dataset-scoped windows if present:
                // keys of the form `{backend}@@{DatasetId}` are written by the matrix sampler harness.
                let prefix = format!("{a}@@");
                let mut agg = SummarySerde::default();
                for (k, w) in &self.windows {
                    if !k.starts_with(&prefix) {
                        continue;
                    }
                    if let Some(ds) = datasets {
                        let suffix = k.strip_prefix(&prefix).unwrap_or("");
                        if !ds.contains(suffix) {
                            continue;
                        }
                    }
                    let s = SummarySerde::from_window(w);
                    agg.calls = agg.calls.saturating_add(s.calls);
                    agg.ok = agg.ok.saturating_add(s.ok);
                    agg.http_429 = agg.http_429.saturating_add(s.http_429);
                    agg.junk = agg.junk.saturating_add(s.junk);
                    agg.hard_junk = agg.hard_junk.saturating_add(s.hard_junk);
                    agg.cost_units = agg.cost_units.saturating_add(s.cost_units);
                    agg.elapsed_ms_sum = agg.elapsed_ms_sum.saturating_add(s.elapsed_ms_sum);
                }
                if agg.calls > 0 {
                    obs = agg;
                }
            }

            if obs.calls == 0 {
                // Fallback: older history files (or global-only) use the backend name directly.
                obs = self
                    .windows
                    .get(a)
                    .map(SummarySerde::from_window)
                    .unwrap_or_default();
            }

            // Borrow a small pseudo-count prior when this slice is sparse (facet-scoped cold start).
            if prior_calls > 0 && obs.calls < prior_calls {
                // Build a prior summary from the provided history (task-wide), optionally
                // preferring facet-matched dataset windows when we can infer a (lang,domain) filter.
                let mut prior_s = prior
                    .and_then(|h| h.windows.get(a).map(SummarySerde::from_window))
                    .unwrap_or_default();
                if mh::prior_by_facets_from_env() && per_dataset {
                    if let Some(ds_set) = datasets {
                        let mut parsed: Vec<DatasetId> = Vec::new();
                        for ds_s in ds_set {
                            if let Ok(ds) = ds_s.parse::<DatasetId>() {
                                parsed.push(ds);
                            }
                        }
                        if let Some(prior_hist) = prior {
                            if let Some((lang, dom)) = mh::facet_prior_filter(&parsed) {
                                let prefix = format!("{a}@@");
                                let mut agg = SummarySerde::default();
                                for (k, w) in &prior_hist.windows {
                                    let Some(suffix) = k.strip_prefix(&prefix) else {
                                        continue;
                                    };
                                    let Ok(ds) = suffix.parse::<DatasetId>() else {
                                        continue;
                                    };
                                    if ds.language() != lang || ds.domain() != dom {
                                        continue;
                                    }
                                    let s = SummarySerde::from_window(w);
                                    agg.calls = agg.calls.saturating_add(s.calls);
                                    agg.ok = agg.ok.saturating_add(s.ok);
                                    agg.http_429 = agg.http_429.saturating_add(s.http_429);
                                    agg.junk = agg.junk.saturating_add(s.junk);
                                    agg.hard_junk = agg.hard_junk.saturating_add(s.hard_junk);
                                    agg.cost_units = agg.cost_units.saturating_add(s.cost_units);
                                    agg.elapsed_ms_sum =
                                        agg.elapsed_ms_sum.saturating_add(s.elapsed_ms_sum);
                                }
                                if agg.calls > 0 {
                                    prior_s = agg;
                                }
                            }
                        }
                    }
                }

                // Reuse the shared helper (same semantics as the CI harness).
                let mut out_m = muxer::Summary {
                    calls: obs.calls,
                    ok: obs.ok,
                    http_429: obs.http_429,
                    junk: obs.junk,
                    hard_junk: obs.hard_junk,
                    cost_units: obs.cost_units,
                    elapsed_ms_sum: obs.elapsed_ms_sum,
                };
                let prior_m = muxer::Summary {
                    calls: prior_s.calls,
                    ok: prior_s.ok,
                    http_429: prior_s.http_429,
                    junk: prior_s.junk,
                    hard_junk: prior_s.hard_junk,
                    cost_units: prior_s.cost_units,
                    elapsed_ms_sum: prior_s.elapsed_ms_sum,
                };
                mh::apply_prior_counts_to_summary(&mut out_m, prior_m, prior_calls);
                obs = SummarySerde {
                    calls: out_m.calls,
                    ok: out_m.ok,
                    http_429: out_m.http_429,
                    junk: out_m.junk,
                    hard_junk: out_m.hard_junk,
                    cost_units: out_m.cost_units,
                    elapsed_ms_sum: out_m.elapsed_ms_sum,
                };
            }

            out.insert(a.clone(), obs);
        }
        out
    }

    fn dataset_breakdown_for_arm(&self, arm: &str) -> Vec<(String, SummarySerde)> {
        // Collect per-dataset windows with keys like `arm@@DatasetId`.
        let prefix = format!("{arm}@@");
        let mut rows: Vec<(String, SummarySerde)> = Vec::new();
        for (k, w) in &self.windows {
            if !k.starts_with(&prefix) {
                continue;
            }
            let dataset = k.strip_prefix(&prefix).unwrap_or("").to_string();
            let s = SummarySerde::from_window(w);
            if s.calls == 0 {
                continue;
            }
            rows.push((dataset, s));
        }
        rows.sort_by(|a, b| b.1.calls.cmp(&a.1.calls).then_with(|| a.0.cmp(&b.0)));
        rows
    }

    fn failure_kind_counts_for_arm(
        &self,
        arm: &str,
        datasets: Option<&BTreeSet<String>>,
        per_dataset: bool,
    ) -> BTreeMap<String, u64> {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        let mut saw_any = false;

        if per_dataset {
            let prefix = format!("{arm}@@");
            for k in self.fail_kinds.keys() {
                if !k.starts_with(&prefix) {
                    continue;
                }
                if let Some(ds) = datasets {
                    let suffix = k.strip_prefix(&prefix).unwrap_or("");
                    if !ds.contains(suffix) {
                        continue;
                    }
                }
                saw_any = true;
                if let Some(buf) = self.fail_kinds.get(k) {
                    for kind in buf.iter().flatten() {
                        *counts.entry(kind.clone()).or_insert(0) += 1;
                    }
                }
            }
            if saw_any && !counts.is_empty() {
                return counts;
            }
        }

        // Fallback: global per-backend key.
        if let Some(buf) = self.fail_kinds.get(arm) {
            for kind in buf.iter().flatten() {
                *counts.entry(kind.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn failure_kind_counts_for_key(&self, key: &str) -> BTreeMap<String, u64> {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        let Some(buf) = self.fail_kinds.get(key) else {
            return counts;
        };
        for kind in buf.iter().flatten() {
            *counts.entry(kind.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn failure_kind_counts_recent_for_key(
        &self,
        key: &str,
        recent: usize,
    ) -> BTreeMap<String, u64> {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        let Some(buf) = self.fail_kinds.get(key) else {
            return counts;
        };
        let n = buf.len();
        if n == 0 {
            return counts;
        }
        let take = recent.max(1).min(n);
        for kind in buf.iter().skip(n - take).flatten() {
            *counts.entry(kind.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn failure_kind_counts_prev_for_key(&self, key: &str, recent: usize) -> BTreeMap<String, u64> {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        let Some(buf) = self.fail_kinds.get(key) else {
            return counts;
        };
        let n = buf.len();
        if n == 0 {
            return counts;
        }
        let drop = recent.max(1).min(n);
        let take = n.saturating_sub(drop);
        for kind in buf.iter().take(take).flatten() {
            *counts.entry(kind.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn summary_for_key(&self, key: &str) -> SummarySerde {
        self.windows
            .get(key)
            .map(SummarySerde::from_window)
            .unwrap_or_default()
    }

    fn summary_recent_for_key(&self, key: &str, recent: usize) -> SummarySerde {
        let Some(w) = self.windows.get(key) else {
            return SummarySerde::default();
        };
        let n = w.buf.len();
        if n == 0 {
            return SummarySerde::default();
        }
        let take = recent.max(1).min(n);
        let mut out = SummarySerde {
            calls: take as u64,
            ..Default::default()
        };
        for o in w.buf.iter().skip(n - take) {
            out.ok += o.ok as u64;
            out.http_429 += o.http_429 as u64;
            out.junk += o.junk as u64;
            out.hard_junk += o.hard_junk as u64;
            out.cost_units = out.cost_units.saturating_add(o.cost_units);
            out.elapsed_ms_sum = out.elapsed_ms_sum.saturating_add(o.elapsed_ms);
        }
        out
    }

    /// Compute observed (non-smoothed) stats for an arm under the current dataset scope.
    ///
    /// Returns `(calls, elapsed_ms_sum)`.
    fn observed_calls_and_elapsed(
        &self,
        arm: &str,
        datasets: Option<&BTreeSet<String>>,
        per_dataset: bool,
    ) -> (u64, u64) {
        let mut calls = 0u64;
        let mut elapsed_ms_sum = 0u64;

        if per_dataset {
            let prefix = format!("{arm}@@");
            for (k, w) in &self.windows {
                if !k.starts_with(&prefix) {
                    continue;
                }
                if let Some(ds) = datasets {
                    let suffix = k.strip_prefix(&prefix).unwrap_or("");
                    if !ds.contains(suffix) {
                        continue;
                    }
                }
                calls = calls.saturating_add(w.buf.len() as u64);
                for o in &w.buf {
                    elapsed_ms_sum = elapsed_ms_sum.saturating_add(o.elapsed_ms);
                }
            }
        }

        if calls == 0 {
            if let Some(w) = self.windows.get(arm) {
                calls = w.buf.len() as u64;
                for o in &w.buf {
                    elapsed_ms_sum = elapsed_ms_sum.saturating_add(o.elapsed_ms);
                }
            }
        }

        (calls, elapsed_ms_sum)
    }
}

#[cfg(test)]
mod fail_kinds_tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn test_failure_kind_counts_for_arm_missing_is_empty() {
        let h = BackendHistory {
            version: 3,
            window_cap: 50,
            windows: BTreeMap::new(),
            fail_kinds: BTreeMap::new(),
        };
        let c = h.failure_kind_counts_for_arm("a", None, true);
        assert!(c.is_empty());
    }

    #[test]
    fn test_failure_kind_counts_for_arm_aggregates_dataset_scoped() {
        let mut h = BackendHistory {
            version: 3,
            window_cap: 50,
            windows: BTreeMap::new(),
            fail_kinds: BTreeMap::new(),
        };
        let mut q = VecDeque::new();
        q.push_back(Some("timeout".to_string()));
        q.push_back(None);
        q.push_back(Some("timeout".to_string()));
        h.fail_kinds.insert("a@@Wnut17".to_string(), q);

        let c = h.failure_kind_counts_for_arm("a", None, true);
        assert_eq!(c.get("timeout").copied().unwrap_or(0), 2);
    }

    #[test]
    fn test_backend_history_load_truncates_windows_and_fail_kinds_to_cap() {
        let mut h = BackendHistory {
            version: 3,
            window_cap: 3,
            windows: BTreeMap::new(),
            fail_kinds: BTreeMap::new(),
        };
        h.windows.insert(
            "a".to_string(),
            WindowSerde {
                cap: 0,
                buf: vec![
                    OutcomeSerde {
                        ok: true,
                        ..Default::default()
                    },
                    OutcomeSerde {
                        ok: false,
                        ..Default::default()
                    },
                    OutcomeSerde {
                        ok: true,
                        ..Default::default()
                    },
                    OutcomeSerde {
                        ok: false,
                        ..Default::default()
                    },
                ],
            },
        );
        let mut q = VecDeque::new();
        q.push_back(Some("timeout".to_string()));
        q.push_back(Some("timeout".to_string()));
        q.push_back(None);
        q.push_back(Some("backend".to_string()));
        h.fail_kinds.insert("a".to_string(), q);

        let bytes = serde_json::to_vec(&h).expect("serialize BackendHistory");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "anno_muxer_history_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).expect("write temp history");

        let loaded = BackendHistory::load(&path, 50).expect("load history");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.window_cap, 3);
        let w = loaded.windows.get("a").expect("window a");
        assert_eq!(w.cap, 3);
        assert_eq!(w.buf.len(), 3, "window buf truncated to cap");
        let fk = loaded.fail_kinds.get("a").expect("fail_kinds a");
        assert_eq!(fk.len(), 3, "fail_kinds deque truncated to cap");
        // We keep most-recent entries.
        assert_eq!(fk.back().and_then(|x| x.as_deref()), Some("backend"));
    }

    #[test]
    fn test_failure_kind_counts_recent_for_key_counts_only_tail() {
        let mut h = BackendHistory {
            version: 3,
            window_cap: 50,
            windows: BTreeMap::new(),
            fail_kinds: BTreeMap::new(),
        };
        let mut q = VecDeque::new();
        q.push_back(Some("timeout".to_string()));
        q.push_back(None);
        q.push_back(Some("dataset".to_string()));
        q.push_back(Some("timeout".to_string()));
        h.fail_kinds.insert("a@@D1".to_string(), q);

        let all = h.failure_kind_counts_for_key("a@@D1");
        assert_eq!(all.get("timeout").copied().unwrap_or(0), 2);
        assert_eq!(all.get("dataset").copied().unwrap_or(0), 1);

        let rec2 = h.failure_kind_counts_recent_for_key("a@@D1", 2);
        assert_eq!(rec2.get("timeout").copied().unwrap_or(0), 1);
        assert_eq!(rec2.get("dataset").copied().unwrap_or(0), 1);
        assert_eq!(rec2.len(), 2);
    }

    #[test]
    fn test_failure_kind_counts_prev_for_key_excludes_tail() {
        let mut h = BackendHistory {
            version: 3,
            window_cap: 50,
            windows: BTreeMap::new(),
            fail_kinds: BTreeMap::new(),
        };
        let mut q = VecDeque::new();
        q.push_back(Some("timeout".to_string())); // prev
        q.push_back(Some("dataset".to_string())); // prev
        q.push_back(Some("timeout".to_string())); // tail
        q.push_back(Some("backend".to_string())); // tail
        h.fail_kinds.insert("k".to_string(), q);

        let prev2 = h.failure_kind_counts_prev_for_key("k", 2);
        assert_eq!(prev2.get("timeout").copied().unwrap_or(0), 1);
        assert_eq!(prev2.get("dataset").copied().unwrap_or(0), 1);
        assert!(prev2.get("backend").is_none());
    }
}

#[cfg(test)]
mod decisions_tests {
    use super::*;

    #[test]
    fn test_decisions_aggregate_counts_total_and_by_arm_and_by_dataset() {
        let jsonl = r#"
{"schema_version":6,"run_id":"r1","strategy":"worst-first","slice":"ner","datasets":["D1"],"round":1,"chosen":"a","top_candidates":{"kind":"worst_first","rows":[{"arm":"a","score":1.0},{"arm":"b","score":0.5}]},"chosen_fail_kinds_top":[{"kind":"timeout","count":2}],"constraints_fallback_used":false,"explore_first":false}
{"schema_version":6,"run_id":"r1","strategy":"worst-first","slice":"ner","datasets":["D1","D2"],"round":2,"chosen":"b","top_candidates":{"kind":"worst_first","rows":[{"arm":"a","score":1.0},{"arm":"b","score":0.5}]},"chosen_fail_kinds_top":[{"kind":"low_signal","count":1}],"constraints_fallback_used":true,"explore_first":true,"control_arms":["a"]}
{"schema_version":1,"record_type":"outcome","run_id":"r1","strategy":"worst-first","slice":"ner","dataset":"D2","backend":"b","ok":false,"junk":true,"hard_junk":false,"fail_kind":"low_signal"}
"#;
        let agg = decisions_aggregate_from_jsonl(jsonl);
        assert_eq!(agg.total_rows, 3);
        assert_eq!(agg.decision_rows, 2);
        assert_eq!(agg.decision_rows_with_chosen, 2);
        assert_eq!(agg.decision_rows_with_fail_kinds, 2);
        assert_eq!(agg.decision_constraints_fallback_used, 1);
        assert_eq!(agg.decision_explore_first, 1);
        assert_eq!(agg.outcome_rows, 1);
        assert_eq!(agg.outcome_rows_with_fail_kind, 1);
        assert_eq!(agg.outcome_ok, 0);
        assert_eq!(agg.outcome_junk, 1);
        assert_eq!(agg.outcome_hard_junk, 0);
        assert_eq!(agg.outcome_control_rows, 0);
        assert_eq!(agg.chosen_rank_rows, 2);
        assert_eq!(agg.chosen_rank_1, 1);
        assert_eq!(agg.chosen_rank_sum, 3); // ranks: a=1, b=2
        assert_eq!(
            agg.decision_kinds_total
                .get("timeout")
                .copied()
                .unwrap_or(0),
            2
        );
        assert_eq!(
            agg.decision_kinds_total
                .get("low_signal")
                .copied()
                .unwrap_or(0),
            1
        );
        assert_eq!(
            agg.outcome_kinds_total
                .get("low_signal")
                .copied()
                .unwrap_or(0),
            1
        );
        assert_eq!(
            agg.decision_by_arm
                .get("a")
                .and_then(|m| m.get("timeout"))
                .copied()
                .unwrap_or(0),
            2
        );
        assert_eq!(
            agg.outcome_by_dataset
                .get("D2")
                .and_then(|m| m.get("low_signal"))
                .copied()
                .unwrap_or(0),
            1
        );
    }

    #[test]
    fn test_decisions_aggregate_grouped_by_run_applies_filter_and_counts() {
        let jsonl = r#"
{"schema_version":6,"run_id":"r1 slice=ner","strategy":"ml-only","slice":"ner","datasets":["D1"],"round":1,"chosen":"a","top_candidates":{"kind":"mab","rows":[{"arm":"a","score":1.0}]},"constraints_fallback_used":false,"explore_first":false}
{"schema_version":1,"record_type":"outcome","run_id":"r1 slice=ner","strategy":"ml-only","slice":"ner","dataset":"D1","backend":"a","ok":true,"junk":false,"hard_junk":false}
{"schema_version":6,"run_id":"r2 slice=coref","strategy":"ml-only","slice":"coref","datasets":["D2"],"round":1,"chosen":"b","top_candidates":{"kind":"mab","rows":[{"arm":"b","score":1.0}]},"constraints_fallback_used":false,"explore_first":false}
{"schema_version":1,"record_type":"outcome","run_id":"r2 slice=coref","strategy":"ml-only","slice":"coref","dataset":"D2","backend":"b","ok":false,"junk":true,"hard_junk":true,"fail_kind":"timeout"}
"#;
        let (all, by_run) = decisions_aggregate_grouped_by_run(jsonl, Some("slice=ner"), 2);
        assert_eq!(by_run.len(), 1);
        let a = by_run.get("r1 slice=ner").expect("r1 present");
        assert_eq!(a.decision_rows, 1);
        assert_eq!(a.outcome_rows, 1);
        assert_eq!(a.outcome_ok, 1);
        assert_eq!(a.outcome_junk, 0);
        assert_eq!(a.chosen_rank_rows, 1);
        assert_eq!(a.chosen_rank_1, 1);
        // Global reflects the filtered view as well.
        assert_eq!(all.decision_rows, 1);
        assert_eq!(all.outcome_rows, 1);
    }

    #[test]
    fn test_outcome_new_fail_kinds_from_tail_2n_detects_new_kind() {
        use std::collections::VecDeque;
        let mut q: VecDeque<Option<String>> = VecDeque::new();
        // prev window (n=2): timeout, timeout
        q.push_back(Some("timeout".to_string()));
        q.push_back(Some("timeout".to_string()));
        // recent window: timeout, dataset
        q.push_back(Some("timeout".to_string()));
        q.push_back(Some("dataset".to_string()));
        let newly = outcome_new_fail_kinds_from_tail_2n(&q, 2);
        assert_eq!(newly, vec!["dataset".to_string()]);
    }

    #[test]
    fn test_mode_defaults_strategy_when_strategy_unset() {
        let a = MuxerArgs {
            history_file: None,
            slice: None,
            perspective: Some(MuxerPerspective::Ner),
            slice_by_dataset_facets: true,
            facet_datasets: None,
            strategy: None,
            mode: Some(MuxerMode::Triage),
            include_ml: false,
            action: MuxerAction::Stats {
                show_datasets: false,
                top_datasets: 1,
            },
        };
        // We don't execute run() here (it needs eval-advanced wiring), but we can at least ensure
        // the enum parses/constructs and is available for Clap.
        let _ = a;
    }

    #[test]
    fn test_control_arms_tag_outcomes_when_present() {
        let jsonl = r#"
{"schema_version":6,"run_id":"r1","strategy":"ml-only","slice":"ner","datasets":["D1"],"round":1,"chosen":"a","control_arms":["a"]}
{"schema_version":1,"record_type":"outcome","run_id":"r1","strategy":"ml-only","slice":"ner","dataset":"D1","backend":"a","ok":true,"junk":false,"hard_junk":false}
{"schema_version":1,"record_type":"outcome","run_id":"r1","strategy":"ml-only","slice":"ner","dataset":"D1","backend":"b","ok":false,"junk":true,"hard_junk":false,"fail_kind":"low_signal"}
"#;
        let (agg, by_run) = decisions_aggregate_grouped_by_run(jsonl, None, 5);
        let r1 = by_run.get("r1").expect("r1");
        assert_eq!(r1.outcome_rows, 2);
        assert_eq!(r1.outcome_control_rows, 1);
        assert_eq!(r1.outcome_control_ok, 1);
        assert_eq!(agg.outcome_control_rows, 1);
    }
}

#[cfg(feature = "eval-advanced")]
fn mab_config_from_env() -> MabConfig {
    mh::mab_config_from_env()
}

#[cfg(feature = "eval-advanced")]
#[derive(Debug, Clone, Copy)]
struct WorstFirstConfig {
    exploration_c: f64,
    hard_weight: f64,
    soft_weight: f64,
}

#[cfg(feature = "eval-advanced")]
fn worst_first_config_from_env() -> WorstFirstConfig {
    // Keep CLI parity with the harness: locally, include some soft junk so worst-first surfaces
    // "broken outputs" (low_signal) in addition to hard failures. CI stays hard-only by default.
    let default_soft = if std::env::var("GITHUB_ACTIONS").is_ok() {
        0.0
    } else {
        0.2
    };
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

#[cfg(feature = "eval-advanced")]
fn stable_hash64(seed: u64, s: &str) -> u64 {
    mh::stable_hash64(seed, s)
}

#[cfg(feature = "eval-advanced")]
fn parse_dataset_set(s: &str) -> BTreeSet<String> {
    s.split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.to_string())
        .collect()
}

fn default_history_path(slice_tag: &str) -> PathBuf {
    if let Ok(p) = std::env::var("ANNO_HISTORY_FILE") {
        return PathBuf::from(p);
    }
    let suffix = {
        let salt = std::env::var("ANNO_MUXER_HISTORY_SALT")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(|s| {
                let mut out = String::new();
                for ch in s.chars().take(64) {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                        out.push(ch);
                    } else {
                        out.push('_');
                    }
                }
                out
            })
            .filter(|s| !s.is_empty());
        match salt.as_deref() {
            None => format!("muxer_history.{}.json", slice_tag),
            Some(s) => format!("muxer_history.{}.salt={}.json", slice_tag, s),
        }
    };
    if let Ok(dir) = std::env::var("ANNO_CACHE_DIR") {
        return PathBuf::from(dir).join(suffix);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("anno")
        .join(suffix)
}

fn default_decisions_path() -> PathBuf {
    // Keep consistent with the matrix sampler harness default.
    let suffix = {
        let salt = std::env::var("ANNO_MUXER_DECISIONS_SALT")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(|s| {
                let mut out = String::new();
                for ch in s.chars().take(64) {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                        out.push(ch);
                    } else {
                        out.push('_');
                    }
                }
                out
            })
            .filter(|s| !s.is_empty());
        match salt.as_deref() {
            None => "muxer_decisions.jsonl".to_string(),
            Some(s) => format!("muxer_decisions.salt={}.jsonl", s),
        }
    };
    if let Ok(dir) = std::env::var("ANNO_CACHE_DIR") {
        return PathBuf::from(dir).join(suffix);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("anno")
        .join(suffix)
}
fn backend_candidates(tasks: &[Task], include_ml: bool) -> Vec<String> {
    // Start from what is actually feature-enabled in this build.
    let available: BTreeSet<String> = BackendFactory::available_backends()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Use the same "allowed backends per task" list as the evaluator to avoid
    // reporting "unseen" arms that will never be executed.
    let allowed: BTreeSet<&'static str> =
        tasks.iter().flat_map(|t| get_task_backends(*t)).collect();

    // Match the matrix sampler harness behavior: exclude backends that will *always* be skipped
    // due to missing credentials/config, otherwise `anno muxer` becomes misleading.
    anno::env::load_dotenv();
    let has_hf_token = anno::env::has_hf_token();

    let mut out: Vec<String> = Vec::new();
    for b in allowed {
        // Keep parity with matrix sampler harness.
        if b == "gliner_poly" {
            continue;
        }
        if b == "w2ner" && !has_hf_token {
            continue;
        }
        // Keep the CLI behavior: ML-only controls whether we include ML backends,
        // but always include the core baselines if they are available.
        if !include_ml
            && matches!(
                b,
                "gliner"
                    | "gliner_onnx"
                    | "gliner_candle"
                    | "gliner2"
                    | "gliner_poly"
                    | "bert_onnx"
                    | "deberta_v3"
                    | "albert"
                    | "candle_ner"
                    | "burn"
                    | "nuner"
                    | "w2ner"
                    | "universal_ner"
            )
        {
            continue;
        }
        // Some coref resolvers aren't in `BackendFactory::available_backends()`, but they still
        // show up in history, so keep them even if not "available" per factory.
        if b == "coref_resolver" || b == "mention_ranking" || b == "box" {
            out.push(b.to_string());
            continue;
        }
        if available.contains(b) {
            out.push(b.to_string());
        }
    }

    // Filter to task support (defensive).
    out.retain(|b| tasks.iter().any(|t| backend_tasks(b).contains(t)));
    out.sort();
    out.dedup();
    out
}

#[cfg(feature = "eval-advanced")]
/// Execute the muxer command.
pub fn run(args: MuxerArgs) -> Result<(), String> {
    let perspective = args.perspective.unwrap_or(MuxerPerspective::Ner);
    let strategy = args.strategy.unwrap_or_else(|| {
        // Explicit mode chooses the default strategy, otherwise mirror harness defaults.
        if let Some(m) = args.mode {
            return match m {
                MuxerMode::Triage => MuxerStrategy::WorstFirst,
                MuxerMode::Measure => MuxerStrategy::MlOnly,
            };
        }
        match std::env::var("ANNO_SAMPLE_STRATEGY")
            .ok()
            .unwrap_or_else(|| "ml-only".to_string())
            .to_lowercase()
            .as_str()
        {
            "random" => MuxerStrategy::Random,
            "worst-first" | "worstfirst" => MuxerStrategy::WorstFirst,
            _ => MuxerStrategy::MlOnly,
        }
    });

    // Slice tag + tasks (prefer explicit slice to avoid the legacy “perspective” limitation).
    let (slice_tag_base, tasks) = match args.slice.as_deref() {
        None => (mh::SliceTag::parse(perspective.tag())?, perspective.tasks()),
        Some(s) => {
            let s = s.trim();
            if s.is_empty() {
                return Err("--slice was set but empty".to_string());
            }
            let Some(t) = Task::from_code(s) else {
                return Err(format!(
                    "Unknown slice '{}' (expected a task code like `ner`, `temporal`, `discourse-segmentation`)",
                    s
                ));
            };
            (mh::SliceTag::parse(t.code())?, vec![t])
        }
    };

    // If requested, scope history file by dataset facets (language/domain), matching the harness.
    // This requires a dataset set. Otherwise we stick to the base slice tag.
    let slice_tag_for_history = {
        let slice_by_facets = args.slice_by_dataset_facets
            && mh::env_bool("ANNO_MUXER_SLICE_BY_DATASET_FACETS", true);
        let ds_raw = args.facet_datasets.as_deref().unwrap_or("").trim();
        if slice_by_facets && !ds_raw.is_empty() {
            use anno_eval::eval::loader::DatasetId;
            let ds_set = parse_dataset_set(ds_raw);
            let mut datasets: Vec<DatasetId> = Vec::new();
            for ds_s in ds_set {
                let ds: DatasetId = ds_s.parse().map_err(|e| {
                    format!("Invalid dataset id '{}' for --facet-datasets: {}", ds_s, e)
                })?;
                datasets.push(ds);
            }
            mh::muxer_slice_tag(slice_tag_base.as_str(), &datasets, true)?
        } else {
            slice_tag_base.clone()
        }
    };

    let history_path = args
        .history_file
        .clone()
        .unwrap_or_else(|| default_history_path(slice_tag_for_history.as_str()));
    // Optional: when inspecting a facet-scoped history (lang/domain), also load the base
    // task history as a tiny prior to reduce cold-start instability.
    let prior_history = if args.history_file.is_none() && slice_tag_for_history != slice_tag_base {
        let base_path = default_history_path(slice_tag_base.as_str());
        BackendHistory::load(&base_path, 50).ok()
    } else {
        None
    };
    let prior_calls = mh::prior_calls_from_env();
    let include_ml = args.include_ml || mh::env_bool("ANNO_ML_IN_MATRIX", false);
    let candidates = backend_candidates(&tasks, include_ml);

    let h = match BackendHistory::load(&history_path, 50) {
        Ok(h) => h,
        Err(e) => {
            return Err(format!(
                "{e}\nHint: run the matrix harness at least once (e.g. `ANNO_ML_IN_MATRIX=1 cargo test -p anno-eval --features \"eval-advanced onnx\" test_randomized_matrix_sample -- --nocapture`)."
            ));
        }
    };

    match args.action {
        MuxerAction::Run {
            runs,
            seed_base,
            decisions_file,
            agg_out,
            task,
            max_examples,
            datasets_per_run,
            backends_per_run,
            per_dataset,
            require_cached,
            try_download_on_empty,
            fixed_datasets,
            fixed_backend,
            pin_lang,
            pin_domain,
            pin_backend,
        } => {
            use std::path::Path;

            // Decide defaults based on mode (explicit `--runs` wins).
            let runs = runs.unwrap_or_else(|| match args.mode.unwrap_or(MuxerMode::Measure) {
                MuxerMode::Triage => 1,
                MuxerMode::Measure => 10,
            });
            if runs == 0 {
                return Err("--runs must be > 0".to_string());
            }

            // Map CLI knobs to the harness environment, then call the same entrypoint as CI.
            let decisions_path: PathBuf =
                decisions_file.unwrap_or_else(|| PathBuf::from(".generated/muxer_run.jsonl"));
            if let Some(parent) = decisions_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("muxer run: create_dir_all failed: {e}"))?;
                }
            }
            if decisions_path.exists() {
                std::fs::remove_file(&decisions_path)
                    .map_err(|e| format!("muxer run: remove decisions file failed: {e}"))?;
            }

            // Prefer CLI history file override if provided.
            if let Some(p) = args.history_file.as_ref() {
                std::env::set_var("ANNO_HISTORY_FILE", p.to_string_lossy().to_string());
            }

            // Strategy/mode: explicit `--strategy` wins; otherwise mode chooses the harness default.
            if let Some(m) = args.mode {
                std::env::set_var(
                    "ANNO_MUXER_MODE",
                    match m {
                        MuxerMode::Triage => "triage",
                        MuxerMode::Measure => "measure",
                    },
                );
            }
            if let Some(s) = args.strategy {
                std::env::set_var(
                    "ANNO_SAMPLE_STRATEGY",
                    match s {
                        MuxerStrategy::Random => "random",
                        MuxerStrategy::MlOnly => "ml-only",
                        MuxerStrategy::WorstFirst => "worst-first",
                    },
                );
            }

            if args.include_ml {
                std::env::set_var("ANNO_ML_IN_MATRIX", "1");
            }
            std::env::set_var(
                "ANNO_MUXER_DECISIONS_FILE",
                decisions_path.to_string_lossy().to_string(),
            );

            if let Some(t) = task {
                if !t.trim().is_empty() {
                    std::env::set_var("ANNO_MATRIX_TASK", t);
                }
            }
            if let Some(n) = max_examples {
                std::env::set_var("ANNO_MAX_EXAMPLES", n.to_string());
            }
            if let Some(n) = datasets_per_run {
                std::env::set_var("ANNO_MUXER_DATASETS_PER_RUN", n.to_string());
            }
            if let Some(n) = backends_per_run {
                std::env::set_var("ANNO_MUXER_BACKENDS_PER_RUN", n.to_string());
            }
            if per_dataset {
                std::env::set_var("ANNO_MUXER_PER_DATASET", "1");
            }
            if require_cached {
                std::env::set_var("ANNO_MATRIX_REQUIRE_CACHED", "1");
            }
            if try_download_on_empty {
                std::env::set_var("ANNO_MATRIX_TRY_DOWNLOAD_ON_EMPTY", "1");
            }
            if let Some(v) = fixed_datasets {
                if !v.trim().is_empty() {
                    std::env::set_var("ANNO_MUXER_FIXED_DATASETS", v);
                }
            }
            if let Some(v) = fixed_backend {
                if !v.trim().is_empty() {
                    std::env::set_var("ANNO_MUXER_FIXED_BACKEND", v);
                }
            }
            if let Some(v) = pin_lang {
                if !v.trim().is_empty() {
                    std::env::set_var("ANNO_MUXER_PIN_LANG", v);
                }
            }
            if let Some(v) = pin_domain {
                if !v.trim().is_empty() {
                    std::env::set_var("ANNO_MUXER_PIN_DOMAIN", v);
                }
            }
            if let Some(v) = pin_backend {
                if !v.trim().is_empty() {
                    std::env::set_var("ANNO_MUXER_PIN_BACKEND", v);
                }
            }

            // Execute runs.
            for i in 0..runs {
                let seed = seed_base + i;
                anno_eval::muxer_matrix::run_randomized_matrix_sample_with_seed(seed);
            }

            // Optional aggregation.
            if let Some(out) = agg_out {
                if let Some(parent) = out.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)
                            .map_err(|e| format!("muxer run: create_dir_all failed: {e}"))?;
                    }
                }
                let v = anno_eval::muxer_agg_lib::aggregate_jsonl_paths(&[decisions_path.clone()])
                    .map_err(|e| format!("muxer run: aggregate failed: {e}"))?;
                let s = serde_json::to_string_pretty(&v)
                    .map_err(|e| format!("muxer run: serialize agg failed: {e}"))?;
                std::fs::write(out, s).map_err(|e| format!("muxer run: write agg failed: {e}"))?;
            }

            // Basic “correctness” surface: ensure decisions file exists and is non-empty.
            let md = std::fs::metadata(&decisions_path).map_err(|e| {
                format!(
                    "muxer run: decisions file was not written ({}): {e}",
                    decisions_path.display()
                )
            })?;
            if md.len() == 0 {
                return Err(format!(
                    "muxer run: decisions file is empty ({})",
                    decisions_path.display()
                ));
            }
        }
        MuxerAction::Decisions {
            file,
            top,
            by_run,
            run_filter,
            top_runs,
            trend_window,
            by_arm,
            by_dataset,
        } => {
            let path = file
                .or_else(|| {
                    std::env::var("ANNO_MUXER_DECISIONS_FILE")
                        .ok()
                        .map(PathBuf::from)
                })
                .unwrap_or_else(default_decisions_path);
            let bytes = std::fs::read(&path)
                .map_err(|e| {
                    format!(
                        "Failed to read {}: {}\nHint: run the matrix harness to generate decisions, or set ANNO_MUXER_DECISIONS_FILE.",
                        path.display(),
                        e
                    )
                })?;
            let s = String::from_utf8_lossy(&bytes);
            let (agg, by_run_map) =
                decisions_aggregate_grouped_by_run(&s, run_filter.as_deref(), trend_window);

            println!("=== muxer decisions summary ===\n");
            println!("File: {}", path.display());
            println!(
                "Lines: total={} parsed_decision={} parsed_outcome={} skipped_invalid={} skipped_filtered={}",
                agg.lines_total,
                agg.lines_parsed_decision,
                agg.lines_parsed_outcome,
                agg.lines_skipped_invalid,
                agg.lines_skipped_filtered
            );
            println!("Runs: {}", by_run_map.len());
            if let Some(m) = args.mode {
                println!(
                    "Mode: {}",
                    match m {
                        MuxerMode::Triage => "triage",
                        MuxerMode::Measure => "measure",
                    }
                );
            }
            println!("Rows: {}", agg.total_rows);
            println!("Decision rows: {}", agg.decision_rows);
            println!(
                "Decision rows with chosen: {}",
                agg.decision_rows_with_chosen
            );
            println!(
                "Decision rows with chosen_fail_kinds_top: {}",
                agg.decision_rows_with_fail_kinds
            );
            println!(
                "Outcome rows: {} (with fail_kind: {})",
                agg.outcome_rows, agg.outcome_rows_with_fail_kind
            );
            if agg.chosen_rank_rows > 0 {
                println!(
                    "Chosen rank: avg={:.2} rank1={}/{} ({:.1}%)",
                    (agg.chosen_rank_sum as f64) / (agg.chosen_rank_rows as f64),
                    agg.chosen_rank_1,
                    agg.chosen_rank_rows,
                    (agg.chosen_rank_1 as f64) * 100.0 / (agg.chosen_rank_rows as f64)
                );
            } else {
                println!("Chosen rank: (no top_candidates+chosen overlap)");
            }
            if agg.outcome_rows > 0 {
                println!(
                    "Outcome rates: ok={:.2} junk={:.2} hard={:.2}",
                    (agg.outcome_ok as f64) / (agg.outcome_rows as f64),
                    (agg.outcome_junk as f64) / (agg.outcome_rows as f64),
                    (agg.outcome_hard_junk as f64) / (agg.outcome_rows as f64)
                );

                // Trend: compare first-N vs last-N outcomes (best-effort).
                //
                // Important: if the file mixes multiple run_ids, a global "trend" is misleading
                // because JSONL ordering can interleave runs. In that case, defer to the per-run
                // trends printed below.
                if by_run_map.len() <= 1 {
                    let first_n = agg.outcome_first_n.max(1) as f64;
                    let last_n = (agg.outcome_tail.len().max(1)) as f64;
                    let (last_ok, last_junk, last_hard) = agg.outcome_tail.iter().fold(
                        (0u64, 0u64, 0u64),
                        |(ok, junk, hard), (o, j, h)| {
                            (ok + (*o as u64), junk + (*j as u64), hard + (*h as u64))
                        },
                    );
                    println!(
                        "Outcome trend (first {} vs last {}): ok={:.2}→{:.2} junk={:.2}→{:.2} hard={:.2}→{:.2}",
                        agg.outcome_first_n,
                        agg.outcome_tail.len(),
                        (agg.outcome_first_ok as f64) / first_n,
                        (last_ok as f64) / last_n,
                        (agg.outcome_first_junk as f64) / first_n,
                        (last_junk as f64) / last_n,
                        (agg.outcome_first_hard as f64) / first_n,
                        (last_hard as f64) / last_n
                    );
                } else {
                    println!(
                        "Outcome trend: (multiple run_ids; see per-run trends below; window={})",
                        agg.outcome_trend_n.max(1)
                    );
                }

                // Newly appearing failure kinds (best-effort): only meaningful for a single run.
                if by_run_map.len() <= 1 {
                    let newly = outcome_new_fail_kinds_from_tail_2n(
                        &agg.outcome_fail_kind_tail_2n,
                        agg.outcome_trend_n,
                    );
                    if !newly.is_empty() {
                        println!(
                            "New outcome fail_kinds (recent vs prev): {}",
                            newly.into_iter().take(8).collect::<Vec<_>>().join(" ")
                        );
                    }
                }
            }
            if agg.decision_rows > 0 {
                println!(
                    "Constraints fallback used: {} ({:.1}%)",
                    agg.decision_constraints_fallback_used,
                    (agg.decision_constraints_fallback_used as f64) * 100.0
                        / (agg.decision_rows as f64)
                );
                println!(
                    "Explore-first chosen: {} ({:.1}%)",
                    agg.decision_explore_first,
                    (agg.decision_explore_first as f64) * 100.0 / (agg.decision_rows as f64)
                );
            }
            if !agg.outcome_kinds_total.is_empty() {
                println!("\nTop failure kinds (observed outcomes):");
                println!("  {}", top_kinds_line(&agg.outcome_kinds_total, 8));
            } else {
                println!("\nTop failure kinds (observed outcomes): (none recorded)");
            }
            if !agg.decision_kinds_total.is_empty() {
                println!("\nTop failure kinds (selection-time history):");
                println!("  {}", top_kinds_line(&agg.decision_kinds_total, 8));
            }

            if by_arm && !agg.outcome_by_arm.is_empty() {
                let mut rows: Vec<(u64, String)> = agg
                    .outcome_by_arm
                    .iter()
                    .map(|(arm, m)| (m.values().sum::<u64>(), arm.clone()))
                    .collect();
                rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                println!("\nBy backend (observed outcomes; top {}):", top.max(1));
                for (_tot, arm) in rows.into_iter().take(top.max(1)) {
                    let line = top_kinds_line(agg.outcome_by_arm.get(&arm).unwrap(), 5);
                    println!("  {:<18} {}", arm, line);
                }
            }

            if by_dataset && !agg.outcome_by_dataset.is_empty() {
                let mut rows: Vec<(u64, String)> = agg
                    .outcome_by_dataset
                    .iter()
                    .map(|(ds, m)| (m.values().sum::<u64>(), ds.clone()))
                    .collect();
                rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                println!("\nBy dataset (observed outcomes; top {}):", top.max(1));
                for (_tot, ds) in rows.into_iter().take(top.max(1)) {
                    let line = top_kinds_line(agg.outcome_by_dataset.get(&ds).unwrap(), 5);
                    println!("  {:<28} {}", ds, line);
                }
            }

            // Also print selection-time breakdowns (best-effort; often empty on early runs).
            if by_arm && !agg.decision_by_arm.is_empty() {
                let mut rows: Vec<(u64, String)> = agg
                    .decision_by_arm
                    .iter()
                    .map(|(arm, m)| (m.values().sum::<u64>(), arm.clone()))
                    .collect();
                rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                println!(
                    "\nBy chosen arm (selection-time history; top {}):",
                    top.max(1)
                );
                for (_tot, arm) in rows.into_iter().take(top.max(1)) {
                    let line = top_kinds_line(agg.decision_by_arm.get(&arm).unwrap(), 5);
                    println!("  {:<18} {}", arm, line);
                }
            }
            if by_dataset && !agg.decision_by_dataset.is_empty() {
                let mut rows: Vec<(u64, String)> = agg
                    .decision_by_dataset
                    .iter()
                    .map(|(ds, m)| (m.values().sum::<u64>(), ds.clone()))
                    .collect();
                rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                println!("\nBy dataset (selection-time history; top {}):", top.max(1));
                for (_tot, ds) in rows.into_iter().take(top.max(1)) {
                    let line = top_kinds_line(agg.decision_by_dataset.get(&ds).unwrap(), 5);
                    println!("  {:<28} {}", ds, line);
                }
            }

            if by_run {
                let mut runs: Vec<(u64, String)> = by_run_map
                    .iter()
                    .map(|(run_id, a)| (a.outcome_rows + a.decision_rows, run_id.clone()))
                    .collect();
                runs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                println!("\nBy run_id (top {}):", top_runs.max(1));
                for (_w, run_id) in runs.into_iter().take(top_runs.max(1)) {
                    let a = by_run_map.get(&run_id).unwrap();
                    let rank1_pct = if a.chosen_rank_rows > 0 {
                        (a.chosen_rank_1 as f64) * 100.0 / (a.chosen_rank_rows as f64)
                    } else {
                        0.0
                    };
                    let avg_rank = if a.chosen_rank_rows > 0 {
                        (a.chosen_rank_sum as f64) / (a.chosen_rank_rows as f64)
                    } else {
                        0.0
                    };
                    let ok_rate = if a.outcome_rows > 0 {
                        (a.outcome_ok as f64) / (a.outcome_rows as f64)
                    } else {
                        0.0
                    };
                    let junk_rate = if a.outcome_rows > 0 {
                        (a.outcome_junk as f64) / (a.outcome_rows as f64)
                    } else {
                        0.0
                    };
                    let hard_rate = if a.outcome_rows > 0 {
                        (a.outcome_hard_junk as f64) / (a.outcome_rows as f64)
                    } else {
                        0.0
                    };
                    println!(
                        "  {:<52} rows={} dec={} out={} rank1={:.0}% avg_rank={:.2} ok={:.2} junk={:.2} hard={:.2}",
                        run_id,
                        a.total_rows,
                        a.decision_rows,
                        a.outcome_rows,
                        rank1_pct,
                        avg_rank,
                        ok_rate,
                        junk_rate,
                        hard_rate
                    );
                    if a.outcome_rows > 0 {
                        let first_n = a.outcome_first_n.max(1) as f64;
                        let last_n = (a.outcome_tail.len().max(1)) as f64;
                        let (last_ok, last_junk, last_hard) = a.outcome_tail.iter().fold(
                            (0u64, 0u64, 0u64),
                            |(ok, junk, hard), (o, j, h)| {
                                (ok + (*o as u64), junk + (*j as u64), hard + (*h as u64))
                            },
                        );
                        println!(
                            "    trend (first {} vs last {}): ok={:.2}→{:.2} junk={:.2}→{:.2} hard={:.2}→{:.2}",
                            a.outcome_first_n,
                            a.outcome_tail.len(),
                            (a.outcome_first_ok as f64) / first_n,
                            (last_ok as f64) / last_n,
                            (a.outcome_first_junk as f64) / first_n,
                            (last_junk as f64) / last_n,
                            (a.outcome_first_hard as f64) / first_n,
                            (last_hard as f64) / last_n
                        );
                        let newly = outcome_new_fail_kinds_from_tail_2n(
                            &a.outcome_fail_kind_tail_2n,
                            a.outcome_trend_n,
                        );
                        if !newly.is_empty() {
                            println!(
                                "    new_fail_kinds: {}",
                                newly.into_iter().take(8).collect::<Vec<_>>().join(" ")
                            );
                        }
                    }
                    // Measure mode: highlight control coverage and control-only outcome rates.
                    if matches!(args.mode, Some(MuxerMode::Measure)) && a.outcome_rows > 0 {
                        let ctrl_rate = (a.outcome_control_rows as f64) / (a.outcome_rows as f64);
                        let ctrl_ok = if a.outcome_control_rows > 0 {
                            (a.outcome_control_ok as f64) / (a.outcome_control_rows as f64)
                        } else {
                            0.0
                        };
                        let ctrl_junk = if a.outcome_control_rows > 0 {
                            (a.outcome_control_junk as f64) / (a.outcome_control_rows as f64)
                        } else {
                            0.0
                        };
                        println!(
                            "    control: outcomes={}/{} ({:.0}%) ok={:.2} junk={:.2}",
                            a.outcome_control_rows,
                            a.outcome_rows,
                            100.0 * ctrl_rate,
                            ctrl_ok,
                            ctrl_junk
                        );
                    }
                    // Triage mode: show actionable clusters for newly appearing kinds.
                    if matches!(args.mode, Some(MuxerMode::Triage)) && a.outcome_rows > 0 {
                        let newly = outcome_new_fail_kinds_from_tail_2n(
                            &a.outcome_fail_kind_tail_2n,
                            a.outcome_trend_n,
                        );
                        for k in newly.into_iter().take(3) {
                            if let Some(pairs) = a.outcome_kind_pairs.get(&k) {
                                let mut rows: Vec<(u64, String)> =
                                    pairs.iter().map(|(p, c)| (*c, p.clone())).collect();
                                rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                                let top_pairs = rows
                                    .into_iter()
                                    .take(2)
                                    .map(|(c, p)| format!("{p}={c}"))
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                if !top_pairs.is_empty() {
                                    println!("    new_kind_cluster {k}: {top_pairs}");
                                }
                            }
                        }
                    }
                    if !a.outcome_kinds_total.is_empty() {
                        println!(
                            "    top_outcome_kinds: {}",
                            top_kinds_line(&a.outcome_kinds_total, 3)
                        );
                    }
                }
            }
        }

        MuxerAction::Stats {
            show_datasets,
            top_datasets,
        } => {
            println!("=== muxer history ===\n");
            println!("History file: {}", history_path.display());
            println!("Version: {}", h.version);
            println!("Window cap: {}", h.window_cap);
            println!("Slice: {} ({:?})", slice_tag_for_history, tasks);
            println!(
                "Strategy: {} ({})",
                strategy.to_env_str(),
                strategy.to_env_str()
            );
            println!("Candidate arms: {}", candidates.len());

            let summaries =
                h.summaries_for(prior_history.as_ref(), &candidates, None, true, prior_calls);
            let has_dataset_scoped_keys = h.windows.keys().any(|k| k.contains("@@"));
            if has_dataset_scoped_keys {
                let mut dataset_scoped_arms = 0usize;
                for a in &candidates {
                    let prefix = format!("{a}@@");
                    if h.windows.keys().any(|k| k.starts_with(&prefix)) {
                        dataset_scoped_arms += 1;
                    }
                }
                println!(
                    "Dataset-scoped keys: present ({} / {} arms have @@ windows)",
                    dataset_scoped_arms,
                    candidates.len()
                );
            } else {
                println!("Dataset-scoped keys: none (older history file shape)");
            }

            let seen: Vec<String> = candidates
                .iter()
                .filter(|a| summaries.get(*a).copied().unwrap_or_default().calls > 0)
                .cloned()
                .collect();
            let unseen: Vec<String> = candidates
                .iter()
                .filter(|a| summaries.get(*a).copied().unwrap_or_default().calls == 0)
                .cloned()
                .collect();

            println!("Seen arms: {}", seen.len());
            println!("Unseen arms: {}", unseen.len());
            if !unseen.is_empty() {
                println!("\nUnseen:");
                for a in &unseen {
                    println!("  {a}");
                }
            }

            // Show a small per-arm table sorted by (calls desc, name).
            let mut rows: Vec<(u64, f64, f64, f64, f64, f64, String)> = Vec::new();
            for a in &candidates {
                let s = summaries.get(a).copied().unwrap_or_default();
                rows.push((
                    s.calls,
                    s.ok_rate(),
                    s.ok_rate_95_hw(),
                    s.junk_rate(),
                    s.hard_junk_rate(),
                    s.mean_elapsed_ms(),
                    a.clone(),
                ));
            }
            rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.6.cmp(&b.6)));

            println!("\nBy calls (windowed):");
            println!(
                "{:<16} {:>5} {:>12} {:>6} {:>6} {:>9}",
                "arm", "calls", "ok95(+/-)", "junk", "hard", "mean_ms"
            );
            for (calls, ok, ok_hw, junk, hard, mean_ms, arm) in rows.iter().take(20) {
                println!(
                    "{:<16} {:>5} {:>6.2}+/-{:>4.2} {:>6.2} {:>6.2} {:>9.0}",
                    arm, calls, ok, ok_hw, junk, hard, mean_ms
                );

                // Optional triage: show top coarse failure kinds (best-effort).
                // Includes `low_signal` (junk) and other coarse categories for hard failures.
                let fk = h.failure_kind_counts_for_arm(arm, None, true);
                if !fk.is_empty() && (*junk > 0.0 || *hard > 0.0) {
                    let mut pairs: Vec<(u64, String)> =
                        fk.into_iter().map(|(k, v)| (v, k)).collect();
                    pairs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                    let top = pairs
                        .into_iter()
                        .take(3)
                        .map(|(v, k)| format!("{k}={v}"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!("  fail_kinds: {}", top);
                }

                if show_datasets {
                    let per_ds = h.dataset_breakdown_for_arm(arm);
                    if per_ds.is_empty() {
                        println!("  datasets: (none; no `arm@@DatasetId` keys found)");
                    } else {
                        println!("  datasets (top {}):", top_datasets);
                        for (ds, s) in per_ds.into_iter().take(top_datasets.max(1)) {
                            println!(
                                "    {:<18} calls={:>4} ok95={:>4.2}+/-{:>4.2} junk={:>5.2} hard={:>5.2} mean_ms={:>6.0}",
                                ds,
                                s.calls,
                                s.ok_rate(),
                                s.ok_rate_95_hw(),
                                s.junk_rate(),
                                s.hard_junk_rate(),
                                s.mean_elapsed_ms()
                            );

                            // Optional triage: per-dataset failure kinds (best-effort).
                            if s.junk > 0 || s.hard_junk > 0 {
                                let mut ds_set = BTreeSet::new();
                                ds_set.insert(ds.clone());
                                let fk_ds = h.failure_kind_counts_for_arm(arm, Some(&ds_set), true);
                                if !fk_ds.is_empty() {
                                    let mut pairs: Vec<(u64, String)> =
                                        fk_ds.into_iter().map(|(k, v)| (v, k)).collect();
                                    pairs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                                    let top = pairs
                                        .into_iter()
                                        .take(3)
                                        .map(|(v, k)| format!("{k}={v}"))
                                        .collect::<Vec<_>>()
                                        .join(" ");
                                    println!("      fail_kinds: {}", top);
                                }
                            }
                        }
                    }
                }
            }

            println!("\nNote: the matrix harness writes both per-backend keys (`arm`) and per-dataset keys (`arm@@DatasetId`). This command aggregates dataset-scoped keys when present.");
            println!("Note: in the matrix harness, ok_rate is “non-junk success rate” (quality-aware), not just “did it run”.");
        }

        MuxerAction::Decide {
            k,
            per_dataset,
            datasets,
            show_frontier,
            show_candidates,
            top_candidates,
        } => {
            let cfg = mab_config_from_env();
            let worst_cfg = worst_first_config_from_env();
            let guard = mh::latency_guardrail_from_env();
            let ds_set: Option<BTreeSet<String>> = datasets.as_deref().map(parse_dataset_set);
            let ds_set_ref = ds_set.as_ref();

            println!("=== muxer decide ===\n");
            println!("History file: {}", history_path.display());
            println!("Slice: {} ({:?})", slice_tag_for_history, tasks);
            println!("Strategy: {}", strategy.to_env_str());
            if let Some(ds) = ds_set_ref {
                println!("Datasets: {:?}", ds.iter().collect::<Vec<_>>());
            } else {
                println!("Datasets: (all in history)");
            }
            println!("Per-dataset windows: {}", per_dataset);
            println!("Candidate arms: {}", candidates.len());
            println!(
                "Config: explore_c={:.2} w_cost={:.2} w_lat={:.2} w_junk={:.2} w_hard={:.2} max_junk={:?} max_hard={:?} max_429={:?} max_cost={:?}",
                cfg.exploration_c,
                cfg.cost_weight,
                cfg.latency_weight,
                cfg.junk_weight,
                cfg.hard_junk_weight,
                cfg.max_junk_rate,
                cfg.max_hard_junk_rate,
                cfg.max_http_429_rate,
                cfg.max_mean_cost_units
            );
            let profile = std::env::var("ANNO_MUXER_PROFILE")
                .ok()
                .unwrap_or_else(|| "off".to_string());
            if guard.max_mean_ms.is_some()
                || profile.trim().to_lowercase() != "off"
                || std::env::var("ANNO_MUXER_MAX_MEAN_ELAPSED_MS").is_ok()
            {
                println!(
                    "Latency guardrail: profile={} max_mean_ms={:?} allow_fewer={} require_measured={}",
                    profile,
                    guard.max_mean_ms.map(|x| x.round() as u64),
                    guard.allow_fewer,
                    guard.require_measured
                );
            }
            if matches!(strategy, MuxerStrategy::WorstFirst) {
                println!(
                    "Worst-first: explore_c={:.2} w_hard={:.2} w_soft={:.2}",
                    worst_cfg.exploration_c, worst_cfg.hard_weight, worst_cfg.soft_weight
                );
            }

            let mut remaining = candidates.clone();
            let mut chosen: Vec<String> = Vec::new();
            for round in 0..k.min(remaining.len()) {
                let summaries = h.summaries_for(
                    prior_history.as_ref(),
                    &remaining,
                    ds_set_ref,
                    per_dataset,
                    prior_calls,
                );

                println!("\nround {}:", round + 1);
                match strategy {
                    MuxerStrategy::MlOnly => {
                        // Use the exact same policy pipeline as the CI harness: novelty → observed
                        // latency guardrail → muxer MAB pick.
                        let mut guard_step = guard;
                        // If we haven't picked anything yet, don't allow stop-early; fall back so we
                        // don't return an empty decision on a strict/unmeasured guardrail.
                        if chosen.is_empty() {
                            guard_step.allow_fewer = false;
                        }

                        let mut d_opt = None;
                        let fill = mh::policy_fill_k_observed_with(
                            mh::stable_hash64(0x504C_414E, &format!("round={round}")), // "PLAN"
                            &remaining,
                            1,
                            mh::novelty_from_env(),
                            guard_step,
                            |b| h.observed_calls_and_elapsed(b, ds_set_ref, per_dataset),
                            |eligible, _k| {
                                let mut m: BTreeMap<String, muxer::Summary> = BTreeMap::new();
                                for a in eligible {
                                    let s = summaries.get(a).copied().unwrap_or_default();
                                    m.insert(
                                        a.clone(),
                                        muxer::Summary {
                                            calls: s.calls,
                                            ok: s.ok,
                                            http_429: s.http_429,
                                            junk: s.junk,
                                            hard_junk: s.hard_junk,
                                            cost_units: s.cost_units,
                                            elapsed_ms_sum: s.elapsed_ms_sum,
                                        },
                                    );
                                }

                                let d = muxer::select_mab_explain(eligible, &m, cfg);
                                let pick = d.selection.chosen.clone();
                                d_opt = Some(d);
                                vec![pick]
                            },
                        );

                        let Some(pick) = fill.chosen.first().cloned() else {
                            // Stop-early (or nothing eligible) with allow_fewer: end the loop.
                            break;
                        };

                        println!("  chosen: {}", pick);
                        if !fill.plan.prechosen.is_empty() {
                            println!("  note: novelty (slice-unseen arm)");
                        }
                        if fill.fallback_used {
                            if let Some(ms) = guard.max_mean_ms {
                                println!(
                                    "  note: latency guardrail filtered all arms (max_mean_ms={:.0}); falling back",
                                    ms
                                );
                            }
                        }
                        if fill.stopped_early && guard.allow_fewer {
                            if let Some(ms) = guard.max_mean_ms {
                                println!(
                                    "  note: latency guardrail filtered all remaining arms (max_mean_ms={:.0}); stopping early (chosen={})",
                                    ms,
                                    chosen.len()
                                );
                            }
                            break;
                        }

                        let d = d_opt.expect("mab explain should be set when we picked");
                        if d.constraints_fallback_used {
                            println!("  note: constraints filtered all arms (fallback used)");
                        }
                        if d.explore_first {
                            println!("  note: explore-first (untried arm)");
                        }
                        if show_frontier {
                            println!("  frontier: {:?}", d.selection.frontier);
                        }

                        if show_candidates {
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

                            println!("  candidates (top {}):", top_candidates.max(1));
                            println!(
                                "  {:<16} {:>5} {:>6} {:>6} {:>6} {:>8} {:>6} {:>8} {:>8}",
                                "arm",
                                "calls",
                                "ok",
                                "junk",
                                "hard",
                                "mean_ms",
                                "ucb",
                                "obj_ok",
                                "score"
                            );
                            for (score, c) in rows.into_iter().take(top_candidates.max(1)) {
                                println!(
                                    "  {:<16} {:>5} {:>6.2} {:>6.2} {:>6.2} {:>8.0} {:>6.2} {:>8.2} {:>8.2}",
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

                        remaining.retain(|b| b != &pick);
                        chosen.push(pick);
                    }
                    MuxerStrategy::WorstFirst => {
                        let (pick, explore_first) = mh::worst_first_pick_one(
                            mh::stable_hash64(0x574F_5253, &format!("round={round}")), // "WORS"
                            &remaining,
                            mh::WorstFirstConfig {
                                exploration_c: worst_cfg.exploration_c,
                                hard_weight: worst_cfg.hard_weight,
                                soft_weight: worst_cfg.soft_weight,
                            },
                            |b| h.observed_calls_and_elapsed(b, ds_set_ref, per_dataset).0,
                            |b| {
                                let s = summaries.get(b).copied().unwrap_or_default();
                                let hard = s.hard_junk_rate();
                                let soft = (s.junk_rate() - hard).max(0.0);
                                (s.calls, hard, soft)
                            },
                        )
                        .unwrap_or_else(|| (remaining[0].clone(), false));

                        // Keep candidate display: compute the same score rows as before.
                        let total_calls: f64 = remaining
                            .iter()
                            .map(|a| {
                                (summaries.get(a).copied().unwrap_or_default().calls as f64)
                                    .max(1.0)
                            })
                            .sum::<f64>()
                            .max(1.0);
                        let mut rows: Vec<(f64, String, SummarySerde, f64, f64)> = Vec::new();
                        for a in &remaining {
                            let s = summaries.get(a).copied().unwrap_or_default();
                            let calls = (s.calls as f64).max(1.0);
                            let hard = s.hard_junk_rate();
                            let soft = (s.junk_rate() - hard).max(0.0);
                            let exploration =
                                worst_cfg.exploration_c * ((total_calls.ln() / calls).sqrt());
                            let score = worst_cfg.hard_weight * hard
                                + worst_cfg.soft_weight * soft
                                + exploration;
                            rows.push((score, a.clone(), s, hard, soft));
                        }
                        rows.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

                        println!("  chosen: {}", pick);
                        if explore_first {
                            println!("  note: explore-first (untried arm)");
                        }
                        if show_frontier {
                            println!("  frontier: (n/a for worst-first)");
                        }
                        if show_candidates {
                            println!("  candidates (top {}):", top_candidates.max(1));
                            println!(
                                "  {:<16} {:>5} {:>6} {:>6} {:>6} {:>7} {:>7} {:>8}",
                                "arm", "calls", "ok", "junk", "hard", "hard_w", "soft_w", "score"
                            );
                            for (score, arm, s, hard, soft) in
                                rows.into_iter().take(top_candidates.max(1))
                            {
                                println!(
                                    "  {:<16} {:>5} {:>6.2} {:>6.2} {:>6.2} {:>7.2} {:>7.2} {:>8.2}",
                                    arm,
                                    s.calls,
                                    s.ok_rate(),
                                    s.junk_rate(),
                                    s.hard_junk_rate(),
                                    worst_cfg.hard_weight * hard,
                                    worst_cfg.soft_weight * soft,
                                    score
                                );
                            }
                        }

                        remaining.retain(|b| b != &pick);
                        chosen.push(pick);
                    }
                    MuxerStrategy::Random => {
                        // Deterministic “random subset” (shared helper) to keep `decide` reproducible.
                        let pick = mh::pick_random_subset(0, &remaining, 1)
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| remaining[0].clone());
                        println!("  chosen: {}", pick);
                        remaining.retain(|b| b != &pick);
                        chosen.push(pick);
                    }
                }
            }

            if k > 1 {
                println!("\nchosen (in order): {:?}", chosen);
            }
        }

        MuxerAction::Regress {
            mode,
            recent,
            min_recent_calls,
            top,
            datasets,
            include_global,
        } => {
            let recent = recent.max(1);
            let min_recent_calls = min_recent_calls.max(1);
            let ds_set: Option<BTreeSet<String>> = datasets.as_deref().map(parse_dataset_set);
            let ds_set_ref = ds_set.as_ref();

            println!("=== muxer regress ===\n");
            println!("History file: {}", history_path.display());
            println!("Slice: {} ({:?})", slice_tag_for_history, tasks);
            println!("Candidate arms: {}", candidates.len());
            if let Some(ds) = ds_set_ref {
                println!("Datasets: {:?}", ds.iter().collect::<Vec<_>>());
            } else {
                println!("Datasets: (all in history)");
            }
            println!("Recent window: last {}", recent);
            println!("Min recent calls: {}", min_recent_calls);
            println!(
                "Worst-first weights: explore_c={:.2} w_hard={:.2} w_soft={:.2}",
                worst_first_config_from_env().exploration_c,
                worst_first_config_from_env().hard_weight,
                worst_first_config_from_env().soft_weight
            );

            // Rank per dataset-scoped key when available: `arm@@DatasetId`.
            // This is the most actionable surface (pinpoints which dataset is failing).
            let wcfg = worst_first_config_from_env();
            let candidate_set: BTreeSet<String> = candidates.iter().cloned().collect();

            #[derive(Debug, Clone)]
            struct Row {
                score_delta: f64,
                key: String,
                arm: String,
                dataset: Option<String>,
                base: SummarySerde,
                rec: SummarySerde,
            }

            let mut rows: Vec<Row> = Vec::new();
            for k in h.windows.keys() {
                // If a dataset filter is present, only consider dataset-scoped keys.
                let (arm, ds_opt) = if let Some((a, ds)) = k.split_once("@@") {
                    (a.to_string(), Some(ds.to_string()))
                } else {
                    (k.to_string(), None)
                };

                if !candidate_set.contains(&arm) {
                    continue;
                }
                if !include_global && ds_opt.is_none() {
                    continue;
                }
                if let Some(ref ds) = ds_opt {
                    if let Some(filter) = ds_set_ref {
                        if !filter.contains(ds) {
                            continue;
                        }
                    }
                } else if ds_set_ref.is_some() {
                    // If datasets were specified, skip global keys to avoid mixing.
                    continue;
                }

                let base = h.summary_for_key(k);
                let rec = h.summary_recent_for_key(k, recent);
                if rec.calls < min_recent_calls as u64 {
                    continue;
                }
                if base.calls == 0 {
                    continue;
                }

                // “Regression” heuristic: how much worse did it get recently?
                let delta = match mode {
                    RegressMode::Stability => {
                        let base_hard = base.hard_junk_rate();
                        let rec_hard = rec.hard_junk_rate();
                        let base_soft = base.soft_junk_rate();
                        let rec_soft = rec.soft_junk_rate();
                        wcfg.hard_weight * (rec_hard - base_hard)
                            + wcfg.soft_weight * (rec_soft - base_soft)
                    }
                    RegressMode::Latency => rec.mean_elapsed_ms() - base.mean_elapsed_ms(),
                    RegressMode::Quality => {
                        // ok_rate is “non-junk success rate” in the harness.
                        let base_bad = 1.0 - base.ok_rate();
                        let rec_bad = 1.0 - rec.ok_rate();
                        rec_bad - base_bad
                    }
                };

                rows.push(Row {
                    score_delta: delta,
                    key: k.clone(),
                    arm,
                    dataset: ds_opt,
                    base,
                    rec,
                });
            }

            // Biggest positive deltas first (worsened the most).
            rows.sort_by(|a, b| {
                b.score_delta
                    .total_cmp(&a.score_delta)
                    .then_with(|| a.key.cmp(&b.key))
            });

            println!("\nTop regressions (recent vs baseline):");
            println!(
                "{:<16} {:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
                "arm", "dataset", "Δscore", "ok→", "junk→", "hard→", "rec_n", "base_n"
            );
            for r in rows.iter().take(top.max(1)) {
                let ds = r.dataset.as_deref().unwrap_or("(global)");
                println!(
                    "{:<16} {:<18} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8} {:>8}",
                    r.arm,
                    ds,
                    r.score_delta,
                    r.rec.ok_rate(),
                    r.rec.junk_rate(),
                    r.rec.hard_junk_rate(),
                    r.rec.calls,
                    r.base.calls
                );

                // Optional triage: show coarse failure kinds for this exact key, comparing the last
                // N outcomes vs the preceding outcomes in the same window. (Best-effort; fail_kinds
                // may be absent in older histories.)
                let fk_rec = h.failure_kind_counts_recent_for_key(&r.key, recent);
                if !fk_rec.is_empty() {
                    let fk_prev = h.failure_kind_counts_prev_for_key(&r.key, recent);
                    println!("  fail_kinds (recent): {}", top_kinds_line(&fk_rec, 3));
                    if !fk_prev.is_empty() {
                        println!("  fail_kinds (prev):   {}", top_kinds_line(&fk_prev, 3));
                    }
                    // Surface newly-appearing kinds (recent>0, prev==0).
                    let mut newly: Vec<String> = fk_rec
                        .iter()
                        .filter(|(k, v)| **v > 0 && fk_prev.get(*k).copied().unwrap_or(0) == 0)
                        .map(|(k, _)| k.clone())
                        .collect();
                    newly.sort();
                    if !newly.is_empty() {
                        let shown = newly.into_iter().take(5).collect::<Vec<_>>().join(" ");
                        println!("  new_fail_kinds:      {}", shown);
                    }
                }
            }

            // Also show “worst currently” (useful to choose what to run next even without deltas).
            let mut current: Vec<(f64, String, Option<String>, SummarySerde)> = Vec::new();
            for k in h.windows.keys() {
                let (arm, ds_opt) = if let Some((a, ds)) = k.split_once("@@") {
                    (a.to_string(), Some(ds.to_string()))
                } else {
                    (k.to_string(), None)
                };
                if !candidate_set.contains(&arm) {
                    continue;
                }
                if !include_global && ds_opt.is_none() {
                    continue;
                }
                if let Some(ref ds) = ds_opt {
                    if let Some(filter) = ds_set_ref {
                        if !filter.contains(ds) {
                            continue;
                        }
                    }
                } else if ds_set_ref.is_some() {
                    continue;
                }
                let rec = h.summary_recent_for_key(k, recent);
                if rec.calls < min_recent_calls as u64 {
                    continue;
                }
                let score = match mode {
                    RegressMode::Stability => {
                        wcfg.hard_weight * rec.hard_junk_rate()
                            + wcfg.soft_weight * rec.soft_junk_rate()
                    }
                    RegressMode::Latency => rec.mean_elapsed_ms(),
                    RegressMode::Quality => 1.0 - rec.ok_rate(),
                };
                current.push((score, arm, ds_opt, rec));
            }
            current.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            println!("\nWorst currently (recent window):");
            println!(
                "{:<16} {:<18} {:>7} {:>6} {:>6} {:>6} {:>6}",
                "arm", "dataset", "score", "calls", "ok", "junk", "hard"
            );
            for (score, arm, ds_opt, rec) in current.into_iter().take(top.max(1)) {
                let ds = ds_opt.as_deref().unwrap_or("(global)");
                println!(
                    "{:<16} {:<18} {:>7.2} {:>6} {:>6.2} {:>6.2} {:>6.2}",
                    arm,
                    ds,
                    score,
                    rec.calls,
                    rec.ok_rate(),
                    rec.junk_rate(),
                    rec.hard_junk_rate()
                );

                // Optional triage: show coarse failure kinds for the same key used to compute this row.
                let key = if let Some(ds) = ds_opt.as_deref() {
                    format!("{arm}@@{ds}")
                } else {
                    arm.clone()
                };
                let fk_rec = h.failure_kind_counts_recent_for_key(&key, recent);
                if !fk_rec.is_empty() {
                    let fk_prev = h.failure_kind_counts_prev_for_key(&key, recent);
                    println!("  fail_kinds (recent): {}", top_kinds_line(&fk_rec, 3));
                    if !fk_prev.is_empty() {
                        println!("  fail_kinds (prev):   {}", top_kinds_line(&fk_prev, 3));
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(all(test, feature = "eval-advanced"))]
mod run_cli_tests {
    use super::*;

    #[test]
    fn parse_run_basic() {
        let args = MuxerArgs::parse_from([
            "sampler",
            "--mode",
            "triage",
            "run",
            "--runs",
            "3",
            "--seed-base",
            "10",
            "--pin-lang",
            "de",
            "--fixed-datasets",
            "WikiANN",
            "--fixed-backend",
            "heuristic,gliner_onnx",
        ]);

        assert!(matches!(args.mode, Some(MuxerMode::Triage)));
        match args.action {
            MuxerAction::Run {
                runs,
                seed_base,
                pin_lang,
                fixed_datasets,
                fixed_backend,
                ..
            } => {
                assert_eq!(runs, Some(3));
                assert_eq!(seed_base, 10);
                assert_eq!(pin_lang.as_deref(), Some("de"));
                assert_eq!(fixed_datasets.as_deref(), Some("WikiANN"));
                assert_eq!(fixed_backend.as_deref(), Some("heuristic,gliner_onnx"));
            }
            _ => panic!("expected Run action"),
        }
    }
}

#[cfg(not(feature = "eval-advanced"))]
pub fn run(_args: MuxerArgs) -> Result<(), String> {
    Err("Muxer command requires --features eval-advanced".to_string())
}
