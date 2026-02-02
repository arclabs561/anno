//! Muxer command - Inspect the muxer-backed CI matrix history.
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
// semantics used by the matrix harness when `eval-advanced` is enabled.
#[cfg(feature = "eval-advanced")]
use muxer::MabConfig;

#[cfg(feature = "eval-advanced")]
use anno_eval::muxer_harness as mh;

/// Inspect muxer history from the randomized matrix harness.
#[derive(Parser, Debug)]
#[command(about = "Inspect muxer history from the randomized matrix harness")]
pub struct MuxerArgs {
    /// Override history file path (otherwise uses env/defaults).
    #[arg(long)]
    pub history_file: Option<PathBuf>,

    /// Slice tag to inspect (matches the matrix harness slice codes, e.g. `ner`, `temporal`,
    /// `discourse-segmentation`).
    ///
    /// If set, this overrides `--perspective` (which is legacy / coarse).
    #[arg(long)]
    pub slice: Option<String>,

    /// Perspective to inspect (controls tasks + preferred datasets in the matrix harness).
    #[arg(long, value_enum)]
    pub perspective: Option<MuxerPerspective>,

    /// Scope muxer history by dataset facets (language/domain), matching the matrix harness.
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

    /// Sampling strategy (as used by the matrix harness).
    #[arg(long, value_enum)]
    pub strategy: Option<MuxerStrategy>,

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

    /// Preview what the muxer selector would pick next (using current history).
    Decide {
        /// Number of arms to choose (without replacement), like the matrix harness.
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

/// Which slice of the matrix harness to inspect.
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
    /// Note: in the matrix harness, `ok` is recorded as “evaluation succeeded AND not junk”
    /// (where “junk” is a coarse low-F1 threshold, task-dependent).
    MlOnly,
    /// Regression-hunting selection (bias toward historically bad/flaky arms).
    WorstFirst,
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
    /// Optional triage metadata recorded by the matrix harness (parallel to `windows`).
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
                // keys of the form `{backend}@@{DatasetId}` are written by the matrix harness.
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
    WorstFirstConfig {
        exploration_c: mh::env_f64("ANNO_WORST_EXPLORATION_C", 0.8).max(0.0),
        hard_weight: mh::env_f64("ANNO_WORST_HARD_WEIGHT", 1.0).max(0.0),
        soft_weight: mh::env_f64("ANNO_WORST_SOFT_WEIGHT", 0.0).max(0.0),
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

    // Match the matrix harness behavior: exclude backends that will *always* be skipped
    // due to missing credentials/config, otherwise `anno muxer` becomes misleading.
    anno::env::load_dotenv();
    let has_hf_token = anno::env::has_hf_token();

    let mut out: Vec<String> = Vec::new();
    for b in allowed {
        // Keep parity with matrix harness.
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
        // Mirror harness defaults.
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
            }
        }
    }

    Ok(())
}

#[cfg(not(feature = "eval-advanced"))]
pub fn run(_args: MuxerArgs) -> Result<(), String> {
    Err("Muxer command requires --features eval-advanced".to_string())
}
