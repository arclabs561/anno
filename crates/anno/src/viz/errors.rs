//! Visualize the one-to-one matches produced by the entity evaluator.
//!
//! The two columns have different denominators. A partially overlapping pair
//! can also disagree on its label. Only a same-span type mismatch records the
//! two labels in the portable evaluation report.

use super::{check_label, escape, html_document, integer, wrap};
use crate::core::grounded::{EvalComparison, EvalMatch, SignalId};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;
use std::io::Read;

mod summary;
pub use summary::ERRORS_SCHEMA;

#[derive(Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Metric {
    #[default]
    Entity,
    CharMasking,
}

#[derive(Deserialize)]
struct Report {
    #[serde(default)]
    metric: Metric,
    gold_count: usize,
    predicted_count: usize,
    correct: usize,
    errors: usize,
    matches: Vec<EvalMatch>,
    model: Option<String>,
    predictions: Option<PredictionSource>,
    char_masking: Option<MaskReport>,
}

#[derive(Deserialize)]
struct PredictionSource {
    document_id: String,
}

#[derive(Serialize, Deserialize)]
struct MaskReport {
    counts: MaskCounts,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct MaskCounts {
    true_positive_chars: u64,
    predicted_chars: u64,
    gold_chars: u64,
    non_pii_chars: u64,
    total_chars: u64,
}

impl MaskCounts {
    fn validate(&self) -> Result<(), String> {
        if self.gold_chars.checked_add(self.non_pii_chars) != Some(self.total_chars)
            || self.true_positive_chars > self.gold_chars
            || self.true_positive_chars > self.predicted_chars
            || self.predicted_chars - self.true_positive_chars > self.non_pii_chars
        {
            return Err("inconsistent char_masking.counts for the gold and predicted masks".into());
        }
        Ok(())
    }

    fn pool(&mut self, other: Self) -> Result<(), String> {
        // All subcounts are bounded by total_chars in each validated input.
        self.total_chars = self
            .total_chars
            .checked_add(other.total_chars)
            .ok_or("pooled character count exceeds u64")?;
        self.gold_chars += other.gold_chars;
        self.predicted_chars += other.predicted_chars;
        self.non_pii_chars += other.non_pii_chars;
        self.true_positive_chars += other.true_positive_chars;
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
enum Source {
    Model(String),
    Saved(String),
    SavedDocuments,
    Mixed,
    #[default]
    Unspecified,
}

/// A validated breakdown of one or more entity evaluations.
///
/// Read the output of `anno eval --json` with [`Self::from_json`], or pass an
/// [`EvalComparison`] directly to [`Self::from_comparison`]. An optional
/// character masking panel uses the union counts recorded by the evaluator.
#[derive(Debug, Clone)]
pub struct EvalErrors {
    reports: u64,
    exact: u64,
    wrong_label: u64,
    partial: u64,
    missed: u64,
    extra: u64,
    label_pairs: BTreeMap<(String, String), u64>,
    masking: Option<MaskCounts>,
    source: Source,
    title: Option<String>,
    metadata: Option<Map<String, Value>>,
}

impl EvalErrors {
    /// Read one report or a whitespace-separated stream of `anno eval --json`
    /// objects, or a saved `anno.errors/v1` summary. IDs are local to each
    /// report, including repeated evaluations of the same document. Counts
    /// are pooled before calculating percentages.
    pub fn from_json(reader: impl Read) -> Result<Self, String> {
        let reports = serde_json::Deserializer::from_reader(reader).into_iter::<Value>();
        let mut pooled: Option<Self> = None;
        for (i, item) in reports.enumerate() {
            let number = i + 1;
            let report =
                item.map_err(|e| format!("invalid `anno eval --json` report {number}: {e}"))?;
            let chart = if report.get("schema").is_some() {
                Self::from_summary(report)
            } else {
                serde_json::from_value(report)
                    .map_err(|e| format!("invalid `anno eval --json` report: {e}"))
                    .and_then(Self::from_report)
            }
            .map_err(|e| format!("evaluation report {number}: {e}"))?;
            pooled = Some(match pooled {
                None => chart,
                Some(prior) => prior
                    .pooled(chart)
                    .map_err(|e| format!("evaluation report {number}: {e}"))?,
            });
        }
        pooled.ok_or_else(|| "expected at least one `anno eval --json` report".into())
    }

    fn from_report(report: Report) -> Result<Self, String> {
        let (mut chart, ids) = Self::tally(&report.matches)?;
        for (field, reported, actual) in [
            ("gold_count", report.gold_count, ids[0].len()),
            ("predicted_count", report.predicted_count, ids[1].len()),
            ("correct", report.correct, chart.exact as usize),
            (
                "errors",
                report.errors,
                report.matches.len() - chart.exact as usize,
            ),
        ] {
            if reported != actual {
                return Err(format!(
                    "{field} is {reported}, but matches account for {actual}"
                ));
            }
        }
        chart.masking = Self::masking(report.metric, report.char_masking)?;
        chart.source = match (report.model, report.predictions) {
            (Some(model), _) => Source::Model(model),
            (_, Some(source)) => Source::Saved(source.document_id),
            _ => Source::Unspecified,
        };
        chart.check_context()?;
        Ok(chart)
    }

    fn masking(metric: Metric, counts: Option<MaskReport>) -> Result<Option<MaskCounts>, String> {
        Ok(match (metric, counts) {
            (Metric::CharMasking, Some(mask)) => {
                mask.counts.validate()?;
                Some(mask.counts)
            }
            (Metric::CharMasking, None) => {
                return Err("char-masking metric requires char_masking.counts".into())
            }
            (Metric::Entity, Some(_)) => {
                return Err("entity metric contains unexpected char_masking counts".into())
            }
            (Metric::Entity, None) => None,
        })
    }

    fn check_context(&self) -> Result<(), String> {
        if let Some(title) = &self.title {
            check_label(title, "summary title")?;
        }
        if let Some(source) = self.source_label() {
            check_label(&source, "prediction source")?;
        }
        Ok(())
    }

    /// Summarize existing matches without running a model or reassigning any pair.
    pub fn from_comparison(comparison: &EvalComparison) -> Result<Self, String> {
        let (chart, ids) = Self::tally(&comparison.matches)?;
        for (side, signals, seen) in [
            ("gold", &comparison.gold, &ids[0]),
            ("predicted", &comparison.predicted, &ids[1]),
        ] {
            let declared: HashSet<_> = signals.iter().map(|s| s.id).collect();
            if declared.len() != signals.len() || &declared != seen {
                return Err(format!("match IDs do not account for the {side} signals"));
            }
        }
        Ok(chart)
    }

    fn tally(matches: &[EvalMatch]) -> Result<(Self, [HashSet<SignalId>; 2]), String> {
        let mut chart = Self {
            reports: 1,
            exact: 0,
            wrong_label: 0,
            partial: 0,
            missed: 0,
            extra: 0,
            label_pairs: BTreeMap::new(),
            masking: None,
            source: Source::Unspecified,
            title: None,
            metadata: None,
        };
        let mut ids = [HashSet::new(), HashSet::new()];
        for event in matches {
            let (gold, pred) = match event {
                EvalMatch::Correct { gold_id, pred_id } => {
                    chart.exact += 1;
                    (Some(gold_id), Some(pred_id))
                }
                EvalMatch::TypeMismatch {
                    gold_id,
                    pred_id,
                    gold_label,
                    pred_label,
                } => {
                    check_label(gold_label, "gold_label")?;
                    check_label(pred_label, "pred_label")?;
                    if gold_label == pred_label {
                        return Err("type_mismatch has identical gold and predicted labels".into());
                    }
                    chart.wrong_label += 1;
                    *chart
                        .label_pairs
                        .entry((gold_label.clone(), pred_label.clone()))
                        .or_default() += 1;
                    (Some(gold_id), Some(pred_id))
                }
                EvalMatch::BoundaryError {
                    gold_id,
                    pred_id,
                    iou,
                } => {
                    if !(*iou > 0.0 && *iou <= 1.0) {
                        return Err(
                            "boundary_error iou must be greater than 0 and at most 1".into()
                        );
                    }
                    chart.partial += 1;
                    (Some(gold_id), Some(pred_id))
                }
                EvalMatch::Missed { gold_id } => {
                    chart.missed += 1;
                    (Some(gold_id), None)
                }
                EvalMatch::Spurious { pred_id } => {
                    chart.extra += 1;
                    (None, Some(pred_id))
                }
            };
            if let Some(&id) = gold {
                if !ids[0].insert(id) {
                    return Err(format!("duplicate gold ID {id} in matches"));
                }
            }
            if let Some(&id) = pred {
                if !ids[1].insert(id) {
                    return Err(format!("duplicate predicted ID {id} in matches"));
                }
            }
        }
        Ok((chart, ids))
    }

    /// Combine two independently validated evaluations without reassigning IDs
    /// or matches across their documents. Both must report the same metric.
    pub fn pooled(mut self, other: Self) -> Result<Self, String> {
        let other_totals = other.entity_totals();
        match (&mut self.masking, other.masking) {
            (Some(left), Some(right)) => left.pool(right)?,
            (None, None) => (),
            _ => {
                return Err("cannot pool entity and char-masking metrics in the same chart".into())
            }
        }
        // Gold and prediction totals bound all five entity outcome counts.
        for (first, second) in self.entity_totals().into_iter().zip(other_totals) {
            first
                .checked_add(second)
                .ok_or("pooled entity count exceeds u64")?;
        }
        self.reports = self
            .reports
            .checked_add(other.reports)
            .ok_or("too many evaluation reports")?;
        self.exact += other.exact;
        self.wrong_label += other.wrong_label;
        self.partial += other.partial;
        self.missed += other.missed;
        self.extra += other.extra;
        for (pair, count) in other.label_pairs {
            *self.label_pairs.entry(pair).or_default() += count;
        }
        if self.source != other.source {
            self.source = match (&self.source, &other.source) {
                (
                    Source::Saved(_) | Source::SavedDocuments,
                    Source::Saved(_) | Source::SavedDocuments,
                ) => Source::SavedDocuments,
                _ => Source::Mixed,
            };
        }
        // Metadata can contain study-specific counts. Their pooling rules
        // are unknown even when two summaries carry identical metadata.
        if self.title != other.title {
            self.title = None;
        }
        self.metadata = None;
        Ok(self)
    }

    fn entity_totals(&self) -> [u64; 2] {
        let paired = self.exact + self.wrong_label + self.partial;
        [paired + self.missed, paired + self.extra]
    }

    fn source_label(&self) -> Option<String> {
        let source = match &self.source {
            Source::Model(name) => Some(format!("Model · {name}")),
            Source::Saved(id) => Some(format!("Saved predictions · {id}")),
            Source::SavedDocuments => Some("Saved predictions".into()),
            Source::Mixed => Some("Multiple prediction sources".into()),
            Source::Unspecified => None,
        };
        if self.reports == 1 {
            return source;
        }
        let mut scope = format!("{} evaluation reports", integer(self.reports));
        if let Some(source) = source {
            write!(scope, " · {source}").unwrap();
        }
        Some(scope)
    }

    fn entity_panel(&self) -> Panel {
        Panel {
            title: "Entity matches",
            columns: ["Gold entities", "Predicted entities"],
            totals: self.entity_totals(),
            rows: vec![
                Row("Exact span and label", [self.exact; 2], "#17735f"),
                Row("Same span, wrong label", [self.wrong_label; 2], "#a66315"),
                Row("Partial span overlap", [self.partial; 2], "#56699b"),
                Row(
                    "Missed gold / extra prediction",
                    [self.missed, self.extra],
                    "#a34646",
                ),
            ],
            note: "A partial span overlap may also involve a different label.".into(),
        }
    }

    fn mask_panel(&self) -> Option<Panel> {
        self.masking.as_ref().map(|m| Panel {
            title: "Character masking",
            columns: ["Gold mask characters", "Predicted mask characters"],
            totals: [m.gold_chars, m.predicted_chars],
            rows: vec![
                Row("Covered by both masks", [m.true_positive_chars; 2], "#17735f"),
                Row("Missed gold / extra masking", [m.gold_chars - m.true_positive_chars, m.predicted_chars - m.true_positive_chars], "#a34646"),
            ],
            note: format!("{} Unicode scalar values in the text, {} outside the gold mask. Each position counts once.", integer(m.total_chars), integer(m.non_pii_chars)),
        })
    }

    fn label_pairs(&self) -> Vec<(&(String, String), &u64)> {
        let mut pairs: Vec<_> = self.label_pairs.iter().collect();
        pairs.sort_unstable_by(|(left, a), (right, b)| b.cmp(a).then_with(|| left.cmp(right)));
        pairs
    }

    /// Set a descriptive title for standalone exports.
    pub fn with_title(mut self, title: impl Into<String>) -> Result<Self, String> {
        let title = title.into();
        check_label(&title, "summary title")?;
        self.title = Some(title);
        Ok(self)
    }

    fn title(&self) -> &str {
        self.title
            .as_deref()
            .unwrap_or("Entity evaluation outcomes")
    }

    /// Render a standalone SVG with counts and within-column shares. The SVG
    /// names up to five same-span label confusions, ordered by frequency.
    #[must_use]
    pub fn render_svg(&self) -> String {
        let entities = self.entity_panel();
        let mask = self.mask_panel();
        let pairs = self.label_pairs();
        let pair_rows = pairs.len().min(5);
        let pair_height = if pair_rows == 0 {
            0
        } else {
            87 + 27 * pair_rows
        };
        let title = self.title();
        let title_lines = wrap(title, if title.is_ascii() { 52 } else { 32 });
        let title_extra = (title_lines.len() - 1) * 34;
        let height = 115
            + title_extra
            + entities.height()
            + mask.as_ref().map_or(0, Panel::height)
            + pair_height;
        let mut description = format!(
            "{} exact matches, {} same-span label errors, {} partial overlaps, {} missed gold entities and {} extra predictions. There are {} gold entities and {} predictions.",
            self.exact, self.wrong_label, self.partial, self.missed, self.extra,
            entities.totals[0], entities.totals[1]
        );
        if let Some(m) = &self.masking {
            write!(
                description,
                " The masks cover {} characters in common, out of {} gold and {} predicted masked characters.",
                m.true_positive_chars, m.gold_chars, m.predicted_chars
            ).unwrap();
        }
        if let Some(((gold, pred), count)) = pairs.first() {
            write!(
                description,
                " Most frequent same-span label change: {gold} to {pred} ({count})."
            )
            .unwrap();
        }
        let mut svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 {height}" width="980" height="{height}" role="img" aria-labelledby="eval-title eval-desc">
<title id="eval-title">{}</title>
<desc id="eval-desc">{}</desc>
<rect width="980" height="{height}" fill="white"/>
<g font-family="DejaVu Sans, sans-serif" fill="#183541">
"##,
            escape(title),
            escape(&description)
        );
        for (i, line) in title_lines.iter().enumerate() {
            writeln!(
                svg,
                r#"<text x="36" y="{}" font-size="28" font-weight="700">{}</text>"#,
                42 + i * 34,
                escape(line)
            )
            .unwrap();
        }
        writeln!(svg, r##"<text x="36" y="{}" font-size="15" fill="#456371">Each column shows a count and its share of that column. A dash means the column is empty.</text>"##, 68 + title_extra).unwrap();
        if let Some(source) = self.source_label() {
            writeln!(
                svg,
                r##"<text x="36" y="{}" font-size="14" fill="#456371"><title>{}</title>{}</text>"##,
                90 + title_extra,
                escape(&source),
                escape(&visible_label(&source, 58))
            )
            .unwrap();
        }
        let mut top = 119 + title_extra;
        for panel in std::iter::once(&entities).chain(mask.as_ref()) {
            panel.svg(&mut svg, top);
            top += panel.height();
        }
        if pair_rows > 0 {
            writeln!(svg, r#"<text x="36" y="{top}" font-size="20" font-weight="700">Same-span label confusions</text>"#).unwrap();
            let subtitle = if pairs.len() > pair_rows {
                format!(
                    "{} of {} label pairs, ranked by count",
                    pair_rows,
                    integer(pairs.len() as u64)
                )
            } else {
                "Label pairs ranked by count".to_owned()
            };
            writeln!(
                svg,
                r##"<text x="36" y="{}" font-size="14" fill="#456371">{subtitle}</text>"##,
                top + 23
            )
            .unwrap();
            for (x, label) in [(36, "Gold label"), (375, "Predicted label"), (720, "Pairs")] {
                writeln!(
                    svg,
                    r#"<text x="{x}" y="{}" font-size="14" font-weight="700">{label}</text>"#,
                    top + 50
                )
                .unwrap();
            }
            for (i, ((gold, pred), count)) in pairs.iter().take(pair_rows).enumerate() {
                let y = top + 77 + i * 27;
                for (x, label) in [(36, gold), (375, pred)] {
                    writeln!(
                        svg,
                        r#"<text x="{x}" y="{y}" font-size="15"><title>{}</title>{}</text>"#,
                        escape(label),
                        escape(&visible_label(label, 21))
                    )
                    .unwrap();
                }
                writeln!(
                    svg,
                    r#"<text x="720" y="{y}" font-size="15">{}</text>"#,
                    integer(**count)
                )
                .unwrap();
            }
        }
        svg.push_str("</g>\n</svg>\n");
        svg
    }

    /// Render self-contained HTML with the chart, both denominators in table
    /// headers, and every observed same-span label confusion ordered by frequency.
    #[must_use]
    pub fn render_html(&self) -> String {
        let mut tables = String::new();
        if let Some(source) = self.source_label() {
            writeln!(tables, "<p>{}</p>", escape(&source)).unwrap();
        }
        for panel in std::iter::once(self.entity_panel()).chain(self.mask_panel()) {
            panel.html(&mut tables);
        }
        if !self.label_pairs.is_empty() {
            tables.push_str("<table><caption>All same-span label confusions</caption><thead><tr><th scope=\"col\">Gold label</th><th scope=\"col\">Predicted label</th><th scope=\"col\">Pairs</th></tr></thead><tbody>\n");
            for ((gold, pred), count) in self.label_pairs() {
                writeln!(
                    tables,
                    "<tr><th scope=\"row\">{}</th><td>{}</td><td>{}</td></tr>",
                    escape(gold),
                    escape(pred),
                    integer(*count)
                )
                .unwrap();
            }
            tables.push_str("</tbody></table>\n");
        }
        html_document(self.title(), &self.render_svg(), &tables)
    }
}

// Label text can come from a saved report. SVG titles and HTML tables keep the full value.
fn visible_label(label: &str, max_chars: usize) -> String {
    let normalized = label.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let mut short: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        short.push('…');
    }
    short
}

struct Row(&'static str, [u64; 2], &'static str);

struct Panel {
    title: &'static str,
    columns: [&'static str; 2],
    totals: [u64; 2],
    rows: Vec<Row>,
    note: String,
}

impl Panel {
    fn height(&self) -> usize {
        90 + self.rows.len() * 46 + (wrap(&self.note, 110).len() - 1) * 18
    }

    fn header(&self, side: usize) -> String {
        format!("{} ({})", self.columns[side], integer(self.totals[side]))
    }

    fn svg(&self, svg: &mut String, top: usize) {
        let columns = [310, 642];
        writeln!(
            svg,
            r#"<text x="36" y="{top}" font-size="20" font-weight="700">{}</text>"#,
            self.title
        )
        .unwrap();
        for (side, x) in columns.iter().enumerate() {
            writeln!(svg, r#"<text x="{x}" y="{top}" font-size="15" font-weight="700">{}<tspan x="{x}" dy="19" font-size="14" font-weight="400">{} total</tspan></text>"#, self.columns[side], integer(self.totals[side])).unwrap();
        }
        writeln!(
            svg,
            r##"<path d="M36 {} H944" stroke="#d5e0e2"/>"##,
            top + 30
        )
        .unwrap();
        for (i, Row(label, counts, color)) in self.rows.iter().enumerate() {
            let y = top + 56 + i * 46;
            writeln!(svg, r#"<text x="36" y="{y}" font-size="15">{label}</text>"#).unwrap();
            for (side, x) in columns.iter().enumerate() {
                let value = count_share(counts[side], self.totals[side]);
                writeln!(
                    svg,
                    r#"<text x="{x}" y="{y}" font-size="15">{value}</text>"#
                )
                .unwrap();
                if self.totals[side] != 0 {
                    let bar_y = y + 6;
                    let width = 237.0 * counts[side] as f64 / self.totals[side] as f64;
                    writeln!(svg, r##"<rect x="{x}" y="{bar_y}" width="237" height="7" fill="#e6eeef"/><rect x="{x}" y="{bar_y}" width="{width:.3}" height="7" fill="{color}"/>"##).unwrap();
                }
            }
        }
        for (i, line) in wrap(&self.note, 110).iter().enumerate() {
            writeln!(
                svg,
                r##"<text x="36" y="{}" font-size="14" fill="#456371">{}</text>"##,
                top + 57 + self.rows.len() * 46 + i * 18,
                escape(line)
            )
            .unwrap();
        }
    }

    fn html(&self, table: &mut String) {
        writeln!(table, "<table><caption>{}</caption><thead><tr><th scope=\"col\">Outcome</th><th scope=\"col\">{}</th><th scope=\"col\">{}</th></tr></thead><tbody>", self.title, self.header(0), self.header(1)).unwrap();
        for Row(label, counts, _) in &self.rows {
            writeln!(
                table,
                "<tr><th scope=\"row\">{label}</th><td>{}</td><td>{}</td></tr>",
                count_share(counts[0], self.totals[0]),
                count_share(counts[1], self.totals[1])
            )
            .unwrap();
        }
        writeln!(table, "</tbody></table><p>{}</p>", self.note).unwrap();
    }
}

fn count_share(count: u64, total: u64) -> String {
    let share = if total == 0 {
        "—".into()
    } else {
        let percent = 100.0 * count as f64 / total as f64;
        if count != 0 && percent < 0.1 {
            "under 0.1%".into()
        } else if count != total && percent > 99.9 {
            "over 99.9%".into()
        } else {
            format!("{percent:.1}%")
        }
    };
    format!("{} ({share})", integer(count))
}
