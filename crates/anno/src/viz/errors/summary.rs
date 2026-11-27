//! The aggregate wire format has no per-mention records. It can be checked
//! and rendered directly when a report stream is unavailable.

use super::{check_label, EvalErrors, MaskReport, Metric, Source};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Version of the portable entity-error summary understood by this renderer.
pub const ERRORS_SCHEMA: &str = "anno.errors/v1";

#[derive(Serialize, Deserialize)]
struct Summary {
    schema: String,
    reports: u64,
    metric: Metric,
    #[serde(default)]
    prediction_source: Source,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    entities: Entities,
    same_span_label_confusions: Vec<LabelPair>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    char_masking: Option<MaskReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metadata: Option<Map<String, Value>>,
}

#[derive(Serialize, Deserialize)]
struct Entities {
    gold_count: u64,
    predicted_count: u64,
    exact: u64,
    same_span_wrong_label: u64,
    partial_overlap: u64,
    missed_gold: u64,
    extra_predictions: u64,
}

#[derive(Serialize, Deserialize)]
struct LabelPair {
    gold: String,
    predicted: String,
    pairs: u64,
}

impl EvalErrors {
    pub(super) fn from_summary(value: Value) -> Result<Self, String> {
        if value["schema"] != ERRORS_SCHEMA {
            return Err(format!(
                "unsupported error summary schema {}",
                value["schema"]
            ));
        }
        let summary: Summary =
            serde_json::from_value(value).map_err(|e| format!("invalid {ERRORS_SCHEMA}: {e}"))?;
        let e = summary.entities;
        if summary.reports == 0 {
            return Err("a summary requires at least one evaluation report".into());
        }
        let paired = e
            .exact
            .checked_add(e.same_span_wrong_label)
            .and_then(|n| n.checked_add(e.partial_overlap))
            .ok_or("entity counts overflow")?;
        if paired.checked_add(e.missed_gold) != Some(e.gold_count)
            || paired.checked_add(e.extra_predictions) != Some(e.predicted_count)
        {
            return Err(
                "summary entity counts do not reconcile with gold_count and predicted_count".into(),
            );
        }
        let mut label_pairs = BTreeMap::new();
        let mut labeled = 0_u64;
        for pair in summary.same_span_label_confusions {
            check_label(&pair.gold, "gold label")?;
            check_label(&pair.predicted, "predicted label")?;
            if pair.gold == pair.predicted || pair.pairs == 0 {
                return Err(
                    "a label confusion requires two different labels and a positive count".into(),
                );
            }
            if label_pairs
                .insert((pair.gold, pair.predicted), pair.pairs)
                .is_some()
            {
                return Err("duplicate label confusion in summary".into());
            }
            labeled = labeled
                .checked_add(pair.pairs)
                .ok_or("label confusion total exceeds u64")?;
        }
        if labeled != e.same_span_wrong_label {
            return Err("label confusion counts do not sum to same_span_wrong_label".into());
        }
        let chart = Self {
            reports: summary.reports,
            exact: e.exact,
            wrong_label: e.same_span_wrong_label,
            partial: e.partial_overlap,
            missed: e.missed_gold,
            extra: e.extra_predictions,
            label_pairs,
            masking: Self::masking(summary.metric, summary.char_masking)?,
            source: summary.prediction_source,
            title: summary.title,
            metadata: summary.metadata,
        };
        chart.check_context()?;
        Ok(chart)
    }

    /// Export pooled counts and label-pair frequencies for offline rendering.
    /// The summary omits event IDs, individual overlaps and source text.
    pub fn to_json(&self) -> Result<String, String> {
        let [gold_count, predicted_count] = self.entity_totals();
        let summary = Summary {
            schema: ERRORS_SCHEMA.into(),
            reports: self.reports,
            metric: if self.masking.is_some() {
                Metric::CharMasking
            } else {
                Metric::Entity
            },
            prediction_source: self.source.clone(),
            title: self.title.clone(),
            entities: Entities {
                gold_count,
                predicted_count,
                exact: self.exact,
                same_span_wrong_label: self.wrong_label,
                partial_overlap: self.partial,
                missed_gold: self.missed,
                extra_predictions: self.extra,
            },
            same_span_label_confusions: self
                .label_pairs()
                .into_iter()
                .map(|((gold, predicted), &pairs)| LabelPair {
                    gold: gold.clone(),
                    predicted: predicted.clone(),
                    pairs,
                })
                .collect(),
            char_masking: self.masking.clone().map(|counts| MaskReport { counts }),
            metadata: self.metadata.clone(),
        };
        serde_json::to_string_pretty(&summary)
            .map(|s| s + "\n")
            .map_err(|e| e.to_string())
    }
}
