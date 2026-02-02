//! Shared muxer history types for the CI matrix harness.
//!
//! This module provides persistent storage for MAB/Exp3-IX backend selection history.
//! It is used by `matrix_muxer_ci.rs` to learn from past evaluations.

use crate::eval::loader::DatasetId;
use crate::muxer_harness as mh;
use muxer::{Exp3IxState, Outcome, Summary, Window};
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::PathBuf;

/// A count of failure occurrences by kind.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FailKindCount {
    /// The failure kind (e.g., "timeout", "backend", "low_signal").
    pub kind: String,
    /// Number of occurrences.
    pub count: u64,
}

/// Persistent history of backend evaluation outcomes.
///
/// This stores windowed statistics per-backend (and optionally per-backend×dataset),
/// enabling MAB-style selection algorithms to learn from past performance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendHistory {
    /// Schema version for forward compatibility.
    #[serde(default)]
    pub version: u32,
    /// Maximum window size per arm.
    pub window_cap: usize,
    /// Per-arm outcome windows. Keys are backend names or `{backend}@@{DatasetId}`.
    pub windows: BTreeMap<String, Window>,
    /// Per-outcome failure kind strings (triage metadata, not used for selection).
    #[serde(default)]
    pub fail_kinds: BTreeMap<String, VecDeque<Option<String>>>,
    /// Persisted Exp3-IX state for contextual bandit selection.
    #[serde(default)]
    pub exp3ix_state: Option<Exp3IxState>,
}

impl BackendHistory {
    /// Load history from a JSON file, or return an empty history if not found.
    pub fn load(path: &PathBuf, window_cap: usize) -> Self {
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
                let _ = h.window_cap; // legacy; env-controlled cap is source of truth

                // Upgrade path for schema versions:
                // - v0/1: legacy; ok field was inverted or ambiguous
                // - v2: ok := "evaluation succeeded" (upgrade to quality-aware)
                // - v3+: ok := "succeeded AND not junk"
                let mut windows: BTreeMap<String, Window> = BTreeMap::new();
                let mut fail_kinds: BTreeMap<String, VecDeque<Option<String>>> = BTreeMap::new();
                for (k, w) in h.windows {
                    let _ = w.cap;
                    let mut out = Window::new(cap);
                    for mut o in w.buf {
                        match h.version {
                            0 | 1 => {
                                o.ok = !o.hard_junk && !o.junk;
                            }
                            2 => {
                                o.ok = o.ok && !o.hard_junk && !o.junk;
                            }
                            _ => {
                                o.ok = o.ok && !o.hard_junk && !o.junk;
                            }
                        }
                        out.push(o);
                    }
                    windows.insert(k, out);
                }
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

    /// Save history to a JSON file.
    pub fn save(&self, path: &PathBuf) {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec_pretty(self) {
            let _ = fs::write(path, bytes);
        }
    }

    /// Record an outcome with an optional failure kind.
    pub fn push_with_fail_kind(&mut self, key: &str, o: Outcome, fail_kind: Option<String>) {
        let w = self
            .windows
            .entry(key.to_string())
            .or_insert_with(|| Window::new(self.window_cap));
        w.push(o);

        let cap = self.window_cap;
        let fk = self
            .fail_kinds
            .entry(key.to_string())
            .or_insert_with(VecDeque::new);
        fk.push_back(fail_kind);
        while fk.len() > cap {
            fk.pop_front();
        }
    }

    /// Generate a stable key for dataset-scoped windows.
    pub fn dataset_key(backend: &str, dataset: DatasetId) -> String {
        format!("{backend}@@{:?}", dataset)
    }

    /// Get observed (non-smoothed) summary for a backend, optionally scoped to datasets.
    pub fn observed_summary_for(
        &self,
        backend: &str,
        datasets: Option<&[DatasetId]>,
        per_dataset: bool,
    ) -> Summary {
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

    /// Get summaries for arms, with optional prior smoothing.
    pub fn summaries_for(
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

    /// Get top failure kinds for a backend (for decision logging).
    pub fn chosen_fail_kinds_top_for(
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
        .map(|(v, k)| FailKindCount { kind: k, count: v })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn dataset_key_round_trips_variant() {
        let key = BackendHistory::dataset_key("stacked", DatasetId::Wnut17);
        let (_, suffix) = key
            .split_once("@@")
            .expect("dataset_key should contain @@ separator");
        let parsed = DatasetId::from_str(suffix).expect("dataset key should parse");
        assert_eq!(parsed, DatasetId::Wnut17);
    }
}
