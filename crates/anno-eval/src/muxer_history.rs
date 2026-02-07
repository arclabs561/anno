//! Shared muxer history types for the CI matrix sampler harness.
//!
//! This module provides persistent storage for MAB/Exp3-IX backend selection history.
//! It is used by `matrix_muxer_ci.rs` to learn from past evaluations.

use crate::eval::loader::DatasetId;
use crate::muxer_harness as mh;
use muxer::{Exp3IxState, LinUcbState, MonitoredWindow, Outcome, Summary};
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

/// A bounded outcome window for one arm.
///
/// This intentionally mirrors muxer’s window JSON shape (`{cap, buf}`) but keeps the buffer
/// directly accessible for tooling (e.g. “recent vs baseline” regressions).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistoryWindow {
    /// Maximum outcomes to retain.
    #[serde(default)]
    pub cap: usize,
    /// Outcome buffer (oldest → newest).
    #[serde(default)]
    pub buf: VecDeque<Outcome>,
}

impl HistoryWindow {
    /// Create a new empty window with capacity `cap` (minimum 1).
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            buf: VecDeque::new(),
        }
    }

    /// Push a new outcome, truncating to `cap` (keeps most-recent).
    pub fn push(&mut self, o: Outcome) {
        let cap = self.cap.max(1);
        self.buf.push_back(o);
        while self.buf.len() > cap {
            self.buf.pop_front();
        }
    }

    /// Summarize the current buffer (counts + sums).
    #[must_use]
    pub fn summary(&self) -> Summary {
        let mut s = Summary::default();
        for o in &self.buf {
            s.calls = s.calls.saturating_add(1);
            s.ok = s.ok.saturating_add(o.ok as u64);
            s.junk = s.junk.saturating_add(o.junk as u64);
            s.hard_junk = s.hard_junk.saturating_add(o.hard_junk as u64);
            s.cost_units = s.cost_units.saturating_add(o.cost_units);
            s.elapsed_ms_sum = s.elapsed_ms_sum.saturating_add(o.elapsed_ms);
        }
        s
    }
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
    pub windows: BTreeMap<String, HistoryWindow>,
    /// Per-outcome failure kind strings (triage metadata, not used for selection).
    #[serde(default)]
    pub fail_kinds: BTreeMap<String, VecDeque<Option<String>>>,
    /// Persisted Exp3-IX state for stochastic bandit selection.
    #[serde(default)]
    pub exp3ix_state: Option<Exp3IxState>,
    /// Persisted LinUCB state for contextual bandit selection.
    #[serde(default)]
    pub linucb_state: Option<LinUcbState>,
}

impl BackendHistory {
    /// Attempt to load history from a JSON file.
    ///
    /// Returns:
    /// - `Ok(empty)` if the file does not exist
    /// - `Err(_)` if the file exists but cannot be read or parsed
    pub fn try_load(path: &PathBuf, window_cap: usize) -> Result<Self, String> {
        #[derive(Debug, Clone, serde::Deserialize)]
        struct BackendHistorySerde {
            #[serde(default)]
            version: u32,
            #[serde(default)]
            window_cap: usize,
            #[serde(default)]
            windows: BTreeMap<String, HistoryWindow>,
            #[serde(default)]
            fail_kinds: BTreeMap<String, VecDeque<Option<String>>>,
            #[serde(default)]
            exp3ix_state: Option<Exp3IxState>,
            #[serde(default)]
            linucb_state: Option<LinUcbState>,
        }

        let cap = window_cap.max(1);
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self {
                    version: 3,
                    window_cap: cap,
                    windows: BTreeMap::new(),
                    fail_kinds: BTreeMap::new(),
                    exp3ix_state: None,
                    linucb_state: None,
                });
            }
            Err(e) => return Err(format!("muxer history: read {}: {e}", path.display())),
        };

        let h = serde_json::from_slice::<BackendHistorySerde>(&bytes)
            .map_err(|e| format!("muxer history: parse {}: {e}", path.display()))?;

        let _ = h.window_cap; // legacy; env-controlled cap is source of truth

        // Upgrade path for schema versions:
        // - v0/1: legacy; ok field was inverted or ambiguous
        // - v2: ok := "evaluation succeeded" (upgrade to quality-aware)
        // - v3+: ok := "succeeded AND not junk"
        let mut windows: BTreeMap<String, HistoryWindow> = BTreeMap::new();
        let mut fail_kinds: BTreeMap<String, VecDeque<Option<String>>> = BTreeMap::new();
        for (k, w) in h.windows {
            let mut out = HistoryWindow::new(cap);
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

        Ok(Self {
            version: 3,
            window_cap: cap,
            windows,
            fail_kinds,
            exp3ix_state: h.exp3ix_state,
            linucb_state: h.linucb_state,
        })
    }

    /// Load history from a JSON file, or return an empty history if not found.
    pub fn load(path: &PathBuf, window_cap: usize) -> Self {
        Self::try_load(path, window_cap).unwrap_or_else(|_e| Self {
            version: 3,
            window_cap: window_cap.max(1),
            windows: BTreeMap::new(),
            fail_kinds: BTreeMap::new(),
            exp3ix_state: None,
            linucb_state: None,
        })
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
            .or_insert_with(|| HistoryWindow::new(self.window_cap));
        w.push(o);

        let cap = self.window_cap;
        let fk = self.fail_kinds.entry(key.to_string()).or_default();
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

    /// Build monitored windows for backends using the backend-global history (not dataset-scoped).
    ///
    /// Rationale: per-dataset windows do not have a coherent cross-dataset temporal ordering; change
    /// monitors that depend on order (CUSUM) are best-effort at the backend-global level.
    pub fn monitored_for_backends(
        &self,
        backends: &[String],
        recent_cap: usize,
    ) -> BTreeMap<String, MonitoredWindow> {
        let baseline_cap = self.window_cap.max(1);
        let recent_cap = recent_cap.max(1).min(baseline_cap);
        let mut out: BTreeMap<String, MonitoredWindow> = BTreeMap::new();
        for b in backends {
            let mut mw = MonitoredWindow::new(baseline_cap, recent_cap);
            if let Some(w) = self.windows.get(b) {
                for o in &w.buf {
                    mw.push(*o);
                }
            }
            out.insert(b.clone(), mw);
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
    use std::path::PathBuf;
    use std::str::FromStr;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn dataset_key_round_trips_variant() {
        let key = BackendHistory::dataset_key("stacked", DatasetId::Wnut17);
        let (_, suffix) = key
            .split_once("@@")
            .expect("dataset_key should contain @@ separator");
        let parsed = DatasetId::from_str(suffix).expect("dataset key should parse");
        assert_eq!(parsed, DatasetId::Wnut17);
    }

    struct TempJsonFile {
        path: PathBuf,
    }

    impl TempJsonFile {
        fn new(tag: &str, json: serde_json::Value) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let mut path = std::env::temp_dir();
            path.push(format!(
                "anno-muxer-history-{tag}-pid{}-{nanos}.json",
                std::process::id()
            ));

            let content = serde_json::to_vec(&json).expect("temp json should serialize");
            std::fs::write(&path, content).expect("write temp json file");

            Self { path }
        }
    }

    impl Drop for TempJsonFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn history_upgrade_v0_overrides_ok_to_not_junk() {
        // v0/1: ok field was ambiguous; upgrade forces ok := !junk && !hard_junk.
        let tmp = TempJsonFile::new(
            "upgrade-v0",
            serde_json::json!({
                "version": 0,
                "windows": {
                    "a": {
                        "cap": 999,
                        "buf": [
                            // ok=false but not junk -> upgraded ok must become true
                            { "ok": false, "junk": false, "hard_junk": false, "cost_units": 1, "elapsed_ms": 2 }
                        ]
                    }
                }
            }),
        );

        let h = BackendHistory::try_load(&tmp.path, 10).expect("try_load");
        let w = h.windows.get("a").expect("window a");
        assert_eq!(h.version, 3);
        assert_eq!(h.window_cap, 10);
        assert_eq!(w.cap, 10);
        assert_eq!(w.buf.len(), 1);
        assert!(
            w.buf[0].ok,
            "v0 upgrade should set ok := !junk && !hard_junk"
        );
    }

    #[test]
    fn history_upgrade_v2_clamps_ok_by_junk_flags() {
        // v2: ok meant "evaluation succeeded"; upgrade clamps ok := ok && !junk && !hard_junk.
        let tmp = TempJsonFile::new(
            "upgrade-v2",
            serde_json::json!({
                "version": 2,
                "windows": {
                    "a": {
                        "buf": [
                            { "ok": true, "junk": true, "hard_junk": false, "cost_units": 0, "elapsed_ms": 0 },
                            { "ok": true, "junk": false, "hard_junk": false, "cost_units": 0, "elapsed_ms": 1 },
                            { "ok": false, "junk": false, "hard_junk": false, "cost_units": 0, "elapsed_ms": 2 }
                        ]
                    }
                }
            }),
        );

        let h = BackendHistory::try_load(&tmp.path, 10).expect("try_load");
        let w = h.windows.get("a").expect("window a");
        assert_eq!(w.buf.len(), 3);
        assert!(!w.buf[0].ok, "junk=true must force ok=false after upgrade");
        assert!(w.buf[1].ok, "clean ok should remain ok");
        assert!(!w.buf[2].ok, "explicit ok=false should remain false");
    }

    #[test]
    fn history_load_truncates_windows_and_fail_kinds_to_cap() {
        let tmp = TempJsonFile::new(
            "truncate-cap",
            serde_json::json!({
                "version": 3,
                "windows": {
                    "a": {
                        "cap": 999,
                        "buf": [
                            { "ok": true, "junk": false, "hard_junk": false, "cost_units": 0, "elapsed_ms": 1 },
                            { "ok": true, "junk": false, "hard_junk": false, "cost_units": 0, "elapsed_ms": 2 },
                            { "ok": true, "junk": false, "hard_junk": false, "cost_units": 0, "elapsed_ms": 3 },
                            { "ok": true, "junk": false, "hard_junk": false, "cost_units": 0, "elapsed_ms": 4 },
                            { "ok": true, "junk": false, "hard_junk": false, "cost_units": 0, "elapsed_ms": 5 }
                        ]
                    }
                },
                "fail_kinds": {
                    "a": [null, "timeout", "backend", "low_signal"]
                }
            }),
        );

        let h = BackendHistory::try_load(&tmp.path, 3).expect("try_load");
        let w = h.windows.get("a").expect("window a");
        assert_eq!(w.cap, 3);
        assert_eq!(w.buf.len(), 3);
        // Keep most-recent outcomes.
        assert_eq!(w.buf[0].elapsed_ms, 3);
        assert_eq!(w.buf[1].elapsed_ms, 4);
        assert_eq!(w.buf[2].elapsed_ms, 5);

        let fk = h.fail_kinds.get("a").expect("fail kinds a");
        assert_eq!(fk.len(), 3);
        assert_eq!(fk[0].as_deref(), Some("timeout"));
        assert_eq!(fk[1].as_deref(), Some("backend"));
        assert_eq!(fk[2].as_deref(), Some("low_signal"));
    }
}
