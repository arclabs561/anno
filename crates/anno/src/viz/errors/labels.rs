//! Optional aggregate evidence from the labeled signals that the matcher paired.
use super::{check_label, count_share, escape, integer, svg_text, EvalErrors};
use crate::core::grounded::{EvalComparison, EvalMatch, SignalId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

/// Mutually exclusive one-to-one outcomes for gold mentions with one label.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct GoldLabelOutcomes {
    /// The paired prediction has the right span and type.
    pub exact: u64,
    /// The span is exact but the prediction has a different type.
    pub same_span_wrong_label: u64,
    /// The span overlaps with a different boundary. Its type may also differ.
    pub partial_overlap: u64,
    /// No prediction was paired with this mention.
    pub missed: u64,
}

impl GoldLabelOutcomes {
    fn values(&self) -> [u64; 4] {
        [
            self.exact,
            self.same_span_wrong_label,
            self.partial_overlap,
            self.missed,
        ]
    }

    // Validated outcomes partition the corresponding chart's gold total.
    fn total(&self) -> u64 {
        self.values().into_iter().sum()
    }
}

/// Type agreement among pairs whose boundaries differ.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PartialLabelAgreement {
    /// Partially overlapping pairs with the same type.
    pub same: u64,
    /// Partially overlapping pairs with different types.
    pub different: u64,
}

/// Per-gold-type outcomes and label agreement for the matcher's partial pairs.
/// Counts omit mention identifiers and source text.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct LabelOutcomes {
    by_gold_label: BTreeMap<String, GoldLabelOutcomes>,
    partial_labels: PartialLabelAgreement,
}

impl LabelOutcomes {
    /// Observed gold types and their exhaustive mention outcomes.
    #[must_use]
    pub fn by_gold_label(&self) -> &BTreeMap<String, GoldLabelOutcomes> {
        &self.by_gold_label
    }

    /// Label agreement for the matcher's paired partial spans.
    #[must_use]
    pub fn partial_labels(&self) -> &PartialLabelAgreement {
        &self.partial_labels
    }

    pub(super) fn from_comparison(
        comparison: &EvalComparison,
        ids: &[HashSet<SignalId>; 2],
    ) -> Result<Self, String> {
        let labels = [&comparison.gold, &comparison.predicted].map(|signals| {
            signals
                .iter()
                .map(|s| (s.id, s.label.as_str()))
                .collect::<HashMap<_, _>>()
        });
        for (i, (side, signals)) in [
            ("gold", &comparison.gold),
            ("predicted", &comparison.predicted),
        ]
        .into_iter()
        .enumerate()
        {
            if labels[i].len() != signals.len()
                || ids[i].len() != signals.len()
                || !ids[i].iter().all(|id| labels[i].contains_key(id))
            {
                return Err(format!("match IDs do not account for the {side} signals"));
            }
        }
        let mut result = Self::default();
        for event in &comparison.matches {
            let gold_id = match event {
                EvalMatch::Correct { gold_id, .. }
                | EvalMatch::TypeMismatch { gold_id, .. }
                | EvalMatch::BoundaryError { gold_id, .. }
                | EvalMatch::Missed { gold_id } => gold_id,
                EvalMatch::Spurious { .. } => continue,
            };
            let gold = labels[0][gold_id];
            check_label(gold, "gold label")?;
            let row = result.by_gold_label.entry(gold.to_owned()).or_default();
            match event {
                EvalMatch::Correct { pred_id, .. } => {
                    if labels[1][pred_id] != gold {
                        return Err("correct match disagrees with signal labels".into());
                    }
                    row.exact += 1;
                }
                EvalMatch::TypeMismatch {
                    pred_id,
                    gold_label,
                    pred_label,
                    ..
                } => {
                    if gold_label != gold || pred_label != labels[1][pred_id] {
                        return Err("type mismatch disagrees with signal labels".into());
                    }
                    row.same_span_wrong_label += 1;
                }
                EvalMatch::BoundaryError { pred_id, .. } => {
                    row.partial_overlap += 1;
                    if gold == labels[1][pred_id] {
                        result.partial_labels.same += 1;
                    } else {
                        result.partial_labels.different += 1;
                    }
                }
                EvalMatch::Missed { .. } => row.missed += 1,
                EvalMatch::Spurious { .. } => unreachable!(),
            }
        }
        Ok(result)
    }

    pub(super) fn validate(&self, chart: &EvalErrors) -> Result<(), String> {
        let mut wrong_by_gold = BTreeMap::new();
        for ((gold, _), count) in &chart.label_pairs {
            *wrong_by_gold.entry(gold).or_insert(0) += count;
        }
        let mut totals = [0_u64; 4];
        for (label, row) in &self.by_gold_label {
            check_label(label, "gold label")?;
            if row.values() == [0; 4]
                || row.same_span_wrong_label != wrong_by_gold.get(label).copied().unwrap_or(0)
            {
                return Err(
                    "gold label outcomes disagree with their observed label confusions".into(),
                );
            }
            for (sum, count) in totals.iter_mut().zip(row.values()) {
                *sum = sum.checked_add(count).ok_or("gold label counts overflow")?;
            }
        }
        if totals != [chart.exact, chart.wrong_label, chart.partial, chart.missed] {
            return Err("gold label outcomes do not sum to the entity counts".into());
        }
        if self
            .partial_labels
            .same
            .checked_add(self.partial_labels.different)
            != Some(chart.partial)
        {
            return Err("partial label agreement does not sum to partial_overlap".into());
        }
        Ok(())
    }

    pub(super) fn add(&mut self, other: Self) {
        // The caller has checked overflow of the combined gold total.
        for (label, other_row) in other.by_gold_label {
            let row = self.by_gold_label.entry(label).or_default();
            row.exact += other_row.exact;
            row.same_span_wrong_label += other_row.same_span_wrong_label;
            row.partial_overlap += other_row.partial_overlap;
            row.missed += other_row.missed;
        }
        self.partial_labels.same += other.partial_labels.same;
        self.partial_labels.different += other.partial_labels.different;
    }

    pub(super) fn partial_note(&self) -> String {
        let p = &self.partial_labels;
        match p.same + p.different {
            0 => "No partial span overlaps were observed.".into(),
            n => format!("Of {} partial span overlaps, {} also had a different label and {} kept the gold label.",
                         integer(n), integer(p.different), integer(p.same)),
        }
    }

    fn rows(&self) -> Vec<(&String, &GoldLabelOutcomes)> {
        let mut rows: Vec<_> = self.by_gold_label.iter().collect();
        rows.sort_by_key(|(label, row)| (std::cmp::Reverse(row.total() - row.exact), *label));
        rows
    }

    pub(super) fn leading_description(&self) -> Option<String> {
        let (label, row) = self.rows().into_iter().next()?;
        Some(format!("Of {} gold {label} mentions, {} were exact, {} had the right span and wrong type, {} overlapped partially and {} were missed.",
            integer(row.total()), integer(row.exact), integer(row.same_span_wrong_label),
            integer(row.partial_overlap), integer(row.missed)))
    }

    pub(super) fn svg(&self, svg: &mut String, mut y: usize) -> usize {
        if self.by_gold_label.is_empty() {
            return y;
        }
        let rows = self.rows();
        let visible = rows.len().min(12);
        y = svg_text(svg, "Outcomes by gold label", 20, y + 20, 460, 21, 700);
        let subtitle = if visible < rows.len() {
            format!("Top {visible} of {} gold labels by non-exact mentions. Shares use each label’s gold total.", integer(rows.len()))
        } else {
            "Ranked by non-exact mentions. Shares use each label’s gold total.".into()
        };
        y = svg_text(svg, &subtitle, 20, y, 460, 17, 400) + 7;
        for (label, row) in rows.into_iter().take(visible) {
            let n = row.total();
            let title = format!("{label} · {} gold", integer(n));
            y = svg_text(svg, &title, 20, y, 460, 19, 700);
            let mut x = 20.0;
            let colors = ["#17735f", "#a66315", "#56699b", "#a34646"];
            let bar_y = y - 13;
            for (v, color) in row.values().into_iter().zip(colors) {
                let width = 460.0 * v as f64 / n as f64;
                writeln!(
                    svg,
                    r#"<rect x="{x:.3}" y="{bar_y}" width="{width:.3}" height="6" fill="{color}"/>"#
                )
                .unwrap();
                x += width;
            }
            y += 13;
            for fields in [
                [
                    ("Exact", row.exact, colors[0]),
                    ("Wrong type", row.same_span_wrong_label, colors[1]),
                ],
                [
                    ("Partial", row.partial_overlap, colors[2]),
                    ("Missed", row.missed, colors[3]),
                ],
            ] {
                let mut end = y;
                for (x, (name, count, color)) in [20, 258].into_iter().zip(fields) {
                    writeln!(
                        svg,
                        r#"<rect x="{x}" y="{}" width="4" height="13" fill="{color}"/>"#,
                        y - 12
                    )
                    .unwrap();
                    end = end.max(svg_text(
                        svg,
                        &format!("{name} {}", count_share(count, n)),
                        x + 10,
                        y,
                        211,
                        17,
                        400,
                    ));
                }
                y = end;
            }
            y += 12;
        }
        y
    }

    pub(super) fn html(&self, output: &mut String) {
        if self.by_gold_label.is_empty() {
            return;
        }
        output.push_str("<table class=\"gold-labels\"><caption>Outcomes by gold label, with shares within each label</caption><thead><tr>");
        for head in [
            "Gold label",
            "Gold mentions",
            "Exact",
            "Same span, wrong type",
            "Partial span",
            "Missed",
        ] {
            write!(output, r#"<th scope="col">{head}</th>"#).unwrap();
        }
        output.push_str("</tr></thead><tbody>");
        for (label, row) in self.rows() {
            let n = row.total();
            write!(
                output,
                r#"<tr><th scope="row">{}</th><td data-label="Gold mentions">{}</td>"#,
                escape(label),
                integer(n)
            )
            .unwrap();
            for (field, v) in ["Exact", "Same span, wrong type", "Partial span", "Missed"]
                .into_iter()
                .zip(row.values())
            {
                write!(
                    output,
                    r#"<td data-label="{field}">{}</td>"#,
                    count_share(v, n)
                )
                .unwrap();
            }
            output.push_str("</tr>");
        }
        output.push_str("</tbody></table>\n");
    }
}
