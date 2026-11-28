//! Compare two extraction settings on the same observations and gold items.
//!
//! Supply a versioned count report, or use [`PairedRow::from_eval_reports`] to
//! pair two streams of `anno eval --json` reports by document and gold identity.
//! Gained and lost count specific reference items that changed exact-match
//! status. Total suggestions include both exact and unmatched predictions.

use super::check_label;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::Read;

mod evidence;
mod render;
pub use evidence::EvalPairing;

/// Schema identifier for portable paired comparisons.
pub const PAIRED_SCHEMA: &str = "anno.paired/v1";

/// Baseline and candidate names apply to every cohort in the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Paired {
    /// Set to [`PAIRED_SCHEMA`].
    pub schema: String,
    /// Question or description shown on the chart.
    pub title: String,
    /// Plural unit for the number of observations, such as "sentences".
    pub unit: String,
    /// Name of the existing setting.
    pub baseline: String,
    /// Name of the setting compared to the baseline.
    pub candidate: String,
    /// Disjoint or overlapping cohorts, each evaluated independently.
    pub rows: Vec<PairedRow>,
}

/// Outcomes on one shared cohort. Rows are never added together implicitly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedRow {
    /// Name of the cohort.
    pub label: String,
    /// The cohort's role in the study, such as "fresh holdout".
    pub role: String,
    /// Count of distinct observations in this cohort, including empty outputs.
    pub observations: u64,
    /// Measured baseline outputs and costs.
    pub baseline: Arm,
    /// Measured candidate outputs and costs on the same observations.
    pub candidate: Arm,
    /// Gold items matched exactly by the candidate alone.
    pub gained: u64,
    /// Gold items matched exactly by the baseline alone.
    pub lost: u64,
    /// Optional measured interval for the signed change in exact matches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<Interval>,
}

/// One setting's observed predictions and optional operational measurements.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Arm {
    /// Exactly matched reference items.
    pub correct: u64,
    /// Additional predictions without an exact match in the scoring inventory.
    pub unmatched: u64,
    /// Number of measured calls to the component under study.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calls: Option<u64>,
    /// Measured median latency per observation in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_ms: Option<f64>,
}

/// Study-supplied interval bounds in net correct matches, on the original count scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
    /// Confidence level and resampling or estimation method used by the study.
    pub method: String,
}

impl Paired {
    /// Read and validate a portable comparison.
    pub fn from_json(reader: impl Read) -> Result<Self, String> {
        let report: Self = serde_json::from_reader(reader)
            .map_err(|e| format!("invalid paired comparison JSON: {e}"))?;
        report.validate()?;
        Ok(report)
    }

    /// Validate the schema, measurement units, costs and paired count identities.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PAIRED_SCHEMA {
            return Err(format!("expected schema {PAIRED_SCHEMA}"));
        }
        for (text, label) in [
            (&self.title, "title"),
            (&self.unit, "unit"),
            (&self.baseline, "baseline"),
            (&self.candidate, "candidate"),
        ] {
            check_label(text, label)?;
        }
        if self.rows.is_empty() {
            return Err("paired comparison requires at least one cohort".into());
        }
        let mut names = BTreeSet::new();
        for row in &self.rows {
            check_label(&row.label, "cohort label")?;
            check_label(&row.role, "cohort role")?;
            if !names.insert((&row.label, &row.role)) {
                return Err(format!("duplicate cohort and role: {}", row.label));
            }
            row.validate().map_err(|e| format!("{}: {e}", row.label))?;
        }
        Ok(())
    }

    /// Save an input which can be passed directly to this view again.
    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }
}

impl PairedRow {
    fn validate(&self) -> Result<(), String> {
        for arm in [&self.baseline, &self.candidate] {
            let suggestions = arm
                .correct
                .checked_add(arm.unmatched)
                .ok_or("suggestion count exceeds u64")?;
            if arm.median_ms.is_some_and(|n| !n.is_finite() || n < 0.0) {
                return Err("median_ms must be finite and nonnegative".into());
            }
            if self.observations == 0
                && (suggestions != 0 || arm.calls.unwrap_or(0) != 0 || arm.median_ms.is_some())
            {
                return Err("empty cohorts cannot have outputs, calls or a measured median".into());
            }
        }
        let retained = self
            .baseline
            .correct
            .checked_sub(self.lost)
            .ok_or("lost exceeds baseline correct")?;
        if retained.checked_add(self.gained) != Some(self.candidate.correct) {
            return Err(
                "baseline correct minus lost plus gained must equal candidate correct".into(),
            );
        }
        if let Some(interval) = &self.interval {
            check_label(&interval.method, "interval method")?;
            if self.observations == 0
                || !interval.lower.is_finite()
                || !interval.upper.is_finite()
                || interval.lower > interval.upper
            {
                return Err("interval requires observations and ordered, finite bounds".into());
            }
        }
        Ok(())
    }

    fn net(&self) -> i128 {
        i128::from(self.gained) - i128::from(self.lost)
    }
}
