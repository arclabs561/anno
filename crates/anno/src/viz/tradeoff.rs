//! Operating-point comparisons on a shared pair of rate axes.

use super::{check_label, escape, html_document, integer, lines, wrap};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt::Write;

/// Schema identifier for the version-one operating-point report.
pub const TRADEOFF_SCHEMA: &str = "anno.tradeoff/v1";

/// Counts for several ways of running an extractor on one or more cohorts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tradeoff {
    /// Set to [`TRADEOFF_SCHEMA`].
    pub schema: String,
    /// Question or description displayed above the chart.
    pub title: String,
    /// Plural name of the observation unit, such as `sentences` or `documents`.
    pub unit: String,
    /// Cohorts sharing a scoring unit and matching convention.
    pub series: Vec<TradeoffSeries>,
}

/// One cohort. Every operating point uses the same inputs and reference items.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TradeoffSeries {
    /// Cohort label.
    pub label: String,
    /// Number of input units evaluated, including units with no annotations.
    pub observations: u64,
    /// Number of annotated reference items in those inputs.
    pub gold: u64,
    /// Measured operating points in display order.
    pub points: Vec<OperatingPoint>,
}

/// Output counts for one configuration or threshold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatingPoint {
    /// Configuration or threshold label.
    pub label: String,
    /// Predicted items matched to a reference item under the study's scoring rule.
    pub correct: u64,
    /// Predicted items in the scoring inventory without a reference match.
    pub unmatched: u64,
}

const WIDTH: i32 = 560;
const LEFT: i32 = 18;
const PLOT_X: i32 = 69;
const PLOT_RIGHT: i32 = 299;
const LABEL_X: i32 = 325;

/// Quantity on an operating-point chart's horizontal axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TradeoffAxis {
    /// Predictions without an exact reference match, known after evaluation.
    #[default]
    Unmatched,
    /// All suggestions sent to review, including reference matches.
    Suggestions,
}

impl TradeoffAxis {
    fn count(self, correct: u64, unmatched: u64) -> u128 {
        match self {
            Self::Unmatched => u128::from(unmatched),
            Self::Suggestions => u128::from(correct) + u128::from(unmatched),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Unmatched => "Unmatched",
            Self::Suggestions => "Suggestions",
        }
    }

    fn caption(self) -> &'static str {
        match self {
            Self::Unmatched => "Horizontal position counts predictions without a reference match.",
            Self::Suggestions => "Horizontal position counts all suggestions sent to review.",
        }
    }
}

impl Tradeoff {
    /// Load a versioned JSON study and check its counts and labels.
    pub fn from_json(reader: impl std::io::Read) -> Result<Self, String> {
        let study: Self = serde_json::from_reader(reader)
            .map_err(|err| format!("invalid operating-point JSON: {err}"))?;
        study.validate()?;
        Ok(study)
    }

    /// Check schema, labels, denominators and matched counts before rendering.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != TRADEOFF_SCHEMA {
            return Err(format!(
                "expected schema {TRADEOFF_SCHEMA}, found {:?}",
                self.schema
            ));
        }
        check_label(&self.title, "title")?;
        check_label(&self.unit, "unit")?;
        if self.series.is_empty() {
            return Err("provide at least one series".into());
        }
        for (s, series) in self.series.iter().enumerate() {
            check_label(&series.label, &format!("series[{s}].label"))?;
            if series.points.is_empty() {
                return Err(format!("series[{s}] must contain an operating point"));
            }
            if series.observations == 0 && series.gold != 0 {
                return Err(format!("series[{s}] has gold items but zero observations"));
            }
            for (i, point) in series.points.iter().enumerate() {
                let path = format!("series[{s}].points[{i}]");
                check_label(&point.label, &format!("{path}.label"))?;
                if point.correct > series.gold {
                    return Err(format!("{path}.correct exceeds the cohort's gold count"));
                }
                if series.observations == 0 && point.unmatched != 0 {
                    return Err(format!("{path} has predictions but zero observations"));
                }
            }
        }
        Ok(())
    }

    /// Write validated counts as a replayable `anno.tradeoff/v1` JSON document.
    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        let json = serde_json::to_string_pretty(self).map_err(|err| err.to_string())?;
        Ok(format!("{json}\n"))
    }

    /// Draw correct versus unmatched predictions per 100 inputs on shared axes.
    pub fn render_svg(&self) -> Result<String, String> {
        self.render_svg_with_axis(TradeoffAxis::Unmatched)
    }

    /// Compare settings at an observed error count or a total review budget.
    pub fn render_svg_with_axis(&self, axis: TradeoffAxis) -> Result<String, String> {
        self.validate()?;
        let (mut max_x, mut max_y) = (0.0_f64, 0.0_f64);
        for cohort in self.series.iter().filter(|s| s.observations > 0) {
            for p in &cohort.points {
                max_x = max_x.max(
                    axis.count(p.correct, p.unmatched) as f64 * 100.0 / cohort.observations as f64,
                );
                max_y = max_y.max(rate(p.correct, cohort.observations));
            }
        }
        let axes = [Axis::new(max_x), Axis::new(max_y)];
        let mut content = String::new();
        let mut accessible = String::new();
        let mut y = 31;
        let title = wrap(&self.title, 36);
        lines(&mut content, &title, LEFT, y, 28, "heading");
        y += 28 * title.len() as i32;
        let explanation = format!("{} Rates are per 100 {}.", axis.caption(), self.unit);
        for note in [
            &explanation[..],
            "Filled points give the most correct matches at or below the stated budget.",
        ] {
            let rows = wrap(note, 66);
            lines(&mut content, &rows, LEFT, y, 19, "scope");
            y += rows.len() as i32 * 19;
        }
        for cohort in &self.series {
            y += 7;
            writeln!(content, r#"<path d="M{LEFT} {y}H542" class="rule"/>"#).unwrap();
            y += 26;
            let rows = wrap(&cohort.label, 44);
            lines(&mut content, &rows, LEFT, y, 23, "cohort");
            y += rows.len() as i32 * 23;
            let scope = format!(
                "{} {} · {} gold",
                integer(cohort.observations),
                self.unit,
                integer(cohort.gold)
            );
            let scope_rows = wrap(&scope, 66);
            lines(&mut content, &scope_rows, LEFT, y, 19, "scope");
            y += scope_rows.len() as i32 * 19 + 12;
            let groups = PointGroups::new(&cohort.points, axis);
            y = cohort.plot(&mut content, &groups, axes, axis, y);
            write!(
                accessible,
                " {} has {} {} and {} gold items.",
                cohort.label,
                integer(cohort.observations),
                self.unit,
                integer(cohort.gold)
            )
            .unwrap();
            for p in &cohort.points {
                write!(
                    accessible,
                    " {} has {} correct, {} unmatched, and {} total suggestions.",
                    p.label,
                    integer(p.correct),
                    integer(p.unmatched),
                    integer(TradeoffAxis::Suggestions.count(p.correct, p.unmatched))
                )
                .unwrap();
            }
        }
        y += 10;
        Ok(format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{y}" viewBox="0 0 {WIDTH} {y}" role="img" aria-labelledby="chart-title chart-desc">
<title id="chart-title">{}</title><desc id="chart-desc">{} Settings with identical counts share a mark.{} </desc>
<style>text{{font-family:"DejaVu Sans",sans-serif;fill:#183541;font-size:15px}}.heading{{font-size:24px;font-weight:700}}.cohort{{font-size:20px;font-weight:700}}.scope,.counts,.axis{{fill:#405963}}.label,.axis-title{{font-weight:700}}.axis{{font-size:16px}}.grid,.rule{{fill:none;stroke:#d8e2e3}}.frontier{{fill:none;stroke:#41877f;stroke-width:2}}.optimal{{fill:#006867;stroke:white;stroke-width:1.4}}.dominated{{fill:white;stroke:#425b63;stroke-width:1.6}}.leader{{fill:none;stroke:#9aaeb0;stroke-width:.9}}</style>
<rect width="{WIDTH}" height="{y}" fill="white"/>{content}</svg>
"##,
            escape(&self.title),
            escape(&explanation),
            escape(&accessible)
        ))
    }

    /// Render the chart, exact counts and rates as a self-contained HTML report.
    pub fn render_html(&self) -> Result<String, String> {
        self.render_html_with_axis(TradeoffAxis::Unmatched)
    }

    /// Use the same horizontal quantity for the SVG and the frontier in the table.
    pub fn render_html_with_axis(&self, axis: TradeoffAxis) -> Result<String, String> {
        let svg = self.render_svg_with_axis(axis)?;
        let mut tables = format!(
            r#"<section class="operating-points"><header class="mobile-intro"><h1>{}</h1><p>{} Filled points identify the greatest correct count observed at or below that budget.</p></header>"#,
            escape(&self.title),
            axis.caption()
        );
        let headings = [
            "Setting",
            "Suggestions",
            "Correct",
            "Unmatched",
            "Correct / 100",
            "Unmatched / 100",
            "Recall",
            "On frontier",
        ];
        for cohort in &self.series {
            let groups = PointGroups::new(&cohort.points, axis);
            write!(
                tables,
                "<table><caption>{} <span>· {} {} · {} gold</span></caption><thead><tr>",
                escape(&cohort.label),
                integer(cohort.observations),
                escape(&self.unit),
                integer(cohort.gold)
            )
            .unwrap();
            for heading in headings {
                write!(tables, r#"<th scope="col">{heading}</th>"#).unwrap();
            }
            tables.push_str("</tr></thead><tbody>");
            for (p, key) in cohort.points.iter().zip(&groups.keys) {
                let on_frontier = match (cohort.observations, groups.on_frontier[*key - 1]) {
                    (0, _) => "—",
                    (_, true) => "Yes",
                    _ => "No",
                };
                write!(tables, r#"<tr><th scope="row">{}</th>"#, escape(&p.label)).unwrap();
                let values = [
                    integer(TradeoffAxis::Suggestions.count(p.correct, p.unmatched)),
                    integer(p.correct),
                    integer(p.unmatched),
                    per_100(p.correct, cohort.observations),
                    per_100(p.unmatched, cohort.observations),
                    recall(p.correct, cohort.gold),
                    on_frontier.to_owned(),
                ];
                for (heading, value) in headings[1..].iter().zip(values) {
                    write!(tables, r#"<td data-label="{heading}">{value}</td>"#).unwrap();
                }
                tables.push_str("</tr>");
            }
            tables.push_str("</tbody></table>");
        }
        tables.push_str("</section>");
        Ok(html_document(&self.title, &svg, &tables))
    }
}

fn per_100(n: u64, denominator: u64) -> String {
    if denominator == 0 {
        return "—".into();
    }
    let value = rate(n, denominator);
    if n > 0 && value < 0.001 {
        format!("{value:.3e}")
    } else {
        format!("{value:.3}")
    }
}

fn rate(count: u64, denominator: u64) -> f64 {
    100.0 * count as f64 / denominator as f64
}

fn recall(correct: u64, gold: u64) -> String {
    if gold == 0 {
        "—".into()
    } else {
        let value = rate(correct, gold);
        if correct > 0 && value < 0.05 {
            format!("{value:.2e}%")
        } else {
            format!("{value:.1}%")
        }
    }
}

/// An axis with at most four intervals and a 1, 2 or 5 × 10^n step.
#[derive(Copy, Clone)]
struct Axis {
    step: f64,
    ticks: u32,
}

impl Axis {
    fn new(max: f64) -> Self {
        if max == 0.0 {
            return Self {
                step: 0.25,
                ticks: 4,
            };
        }
        let rough = max / 4.0;
        let power = 10_f64.powf(rough.log10().floor());
        let factor = [1.0, 2.0, 5.0, 10.0]
            .into_iter()
            .find(|k| k * power >= rough)
            .unwrap_or(10.0);
        let step = factor * power;
        Self {
            step,
            ticks: (max / step).ceil() as u32,
        }
    }

    fn max(self) -> f64 {
        f64::from(self.ticks) * self.step
    }

    fn label(self, index: u32) -> String {
        let n = f64::from(index) * self.step;
        if n == 0.0 {
            return "0".into();
        }
        if !(0.01..10_000.0).contains(&n) {
            return format!("{n:.1e}");
        }
        format!("{n:.2}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
}

/// Number distinct coordinates in first occurrence order so tied policies share a marker.
struct PointGroups {
    keys: Vec<usize>,
    unique: Vec<(u64, u64)>,
    frontier: Vec<usize>,
    on_frontier: Vec<bool>,
}

impl PointGroups {
    fn new(points: &[OperatingPoint], axis: TradeoffAxis) -> Self {
        let mut ids = BTreeMap::new();
        let mut unique = Vec::new();
        let keys = points
            .iter()
            .map(|p| {
                let counts = (p.unmatched, p.correct);
                *ids.entry((axis.count(p.correct, p.unmatched), Reverse(p.correct)))
                    .or_insert_with(|| {
                        unique.push(counts);
                        unique.len()
                    })
            })
            .collect();
        // First visit the most correct matches at each value of the budget axis. A
        // strictly higher correct count advances the observed Pareto frontier.
        let mut frontier = Vec::new();
        let mut on_frontier = vec![false; unique.len()];
        let mut best = None;
        for ((_, Reverse(correct)), key) in ids {
            if best.is_none_or(|previous| correct > previous) {
                frontier.push(key - 1);
                on_frontier[key - 1] = true;
                best = Some(correct);
            }
        }
        Self {
            keys,
            unique,
            frontier,
            on_frontier,
        }
    }
}

impl TradeoffSeries {
    fn plot(
        &self,
        svg: &mut String,
        groups: &PointGroups,
        axes: [Axis; 2],
        axis: TradeoffAxis,
        y: i32,
    ) -> i32 {
        let [x_axis, y_axis] = axes;
        let mut names = vec![Vec::new(); groups.unique.len()];
        for (point, &key) in self.points.iter().zip(&groups.keys) {
            names[key - 1].push(point.label.as_str());
        }
        let mut order: Vec<_> = (0..groups.unique.len()).collect();
        order.sort_by_key(|&i| {
            let (unmatched, correct) = groups.unique[i];
            (Reverse(correct), unmatched)
        });
        // The right column gives every distinct coordinate its own labeled row.
        // Tied settings share their measurements. No per-panel identifier is needed.
        let mut rows = Vec::with_capacity(order.len());
        let mut needed = 0;
        for i in order {
            let (unmatched, correct) = groups.unique[i];
            let label = wrap(&names[i].join(" / "), 28);
            let mut counts = wrap(
                &format!(
                    "{} correct · {} unmatched",
                    integer(correct),
                    integer(unmatched)
                ),
                29,
            );
            counts.extend(wrap(
                &format!(
                    "{} suggestions · {} recall",
                    integer(TradeoffAxis::Suggestions.count(correct, unmatched)),
                    recall(correct, self.gold)
                ),
                29,
            ));
            let height = (label.len() + counts.len()) as i32 * 18 + 10;
            needed += height;
            rows.push((i, label, counts, height));
        }
        let height = needed.max(144);
        let top = y + 14;
        let bottom = top + height;
        let x = |v: u128| {
            f64::from(PLOT_X)
                + f64::from(PLOT_RIGHT - PLOT_X) * (v as f64 / self.observations as f64 * 100.0)
                    / x_axis.max()
        };
        let py = |correct| {
            f64::from(bottom) - f64::from(height) * rate(correct, self.observations) / y_axis.max()
        };
        if self.observations == 0 {
            writeln!(svg, r#"<text x="{LEFT}" y="{y}" class="scope">Rates undefined for an empty cohort</text>"#).unwrap();
        } else {
            writeln!(
                svg,
                r#"<text x="{PLOT_X}" y="{y}" class="axis-title">Correct / 100</text>"#
            )
            .unwrap();
            for i in 0..=y_axis.ticks {
                let gy =
                    f64::from(bottom) - f64::from(i) * f64::from(height) / f64::from(y_axis.ticks);
                writeln!(svg, r#"<path d="M{PLOT_X} {gy:.2}H{PLOT_RIGHT}" class="grid"/><text x="{}" y="{:.2}" text-anchor="end" class="axis">{}</text>"#, PLOT_X - 7, gy + 5.0, y_axis.label(i)).unwrap();
            }
            for i in 0..=x_axis.ticks {
                let gx = f64::from(PLOT_X)
                    + f64::from(i) * f64::from(PLOT_RIGHT - PLOT_X) / f64::from(x_axis.ticks);
                writeln!(svg, r#"<path d="M{gx:.2} {top}V{bottom}" class="grid"/><text x="{gx:.2}" y="{}" text-anchor="middle" class="axis">{}</text>"#, bottom + 19, x_axis.label(i)).unwrap();
            }
            writeln!(svg, r#"<text x="{PLOT_RIGHT}" y="{}" text-anchor="end" class="axis-title">{} / 100</text>"#, bottom + 40, axis.label()).unwrap();
            let mut path = String::new();
            for &i in &groups.frontier {
                let (unmatched, correct) = groups.unique[i];
                let (px, cy) = (x(axis.count(correct, unmatched)), py(correct));
                if path.is_empty() {
                    write!(path, "M{px:.2} {cy:.2}").unwrap();
                } else {
                    write!(path, "H{px:.2}V{cy:.2}").unwrap();
                }
            }
            writeln!(svg, r#"<path d="{path}H{PLOT_RIGHT}" class="frontier"/>"#).unwrap();
        }
        let mut row_y = top + (height - needed) / 2 + 13;
        let mut marks = String::new();
        for (i, label, counts, row_height) in rows {
            let (unmatched, correct) = groups.unique[i];
            if self.observations > 0 {
                let (px, cy) = (x(axis.count(correct, unmatched)), py(correct));
                let symbol = if groups.on_frontier[i] {
                    "optimal"
                } else {
                    "dominated"
                };
                writeln!(
                    svg,
                    r#"<path d="M{px:.2} {cy:.2}L{} {}" class="leader"/>"#,
                    LABEL_X - 5,
                    row_y - 4
                )
                .unwrap();
                writeln!(marks, r#"<g><title>{} · {} correct · {} unmatched</title><circle cx="{px:.2}" cy="{cy:.2}" r="4.5" class="{symbol}"/></g>"#, escape(&names[i].join(" / ")), integer(correct), integer(unmatched)).unwrap();
            }
            lines(svg, &label, LABEL_X, row_y, 18, "label");
            lines(
                svg,
                &counts,
                LABEL_X,
                row_y + label.len() as i32 * 18,
                18,
                "counts",
            );
            row_y += row_height;
        }
        svg.push_str(&marks);
        bottom + 54
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn point(label: &str, correct: u64, unmatched: u64) -> OperatingPoint {
        OperatingPoint {
            label: label.into(),
            correct,
            unmatched,
        }
    }

    fn study(points: Vec<OperatingPoint>) -> Tradeoff {
        Tradeoff {
            schema: TRADEOFF_SCHEMA.into(),
            title: "How much does a higher threshold cost?".into(),
            unit: "sentences".into(),
            series: vec![TradeoffSeries {
                label: "Test".into(),
                observations: 10,
                gold: 5,
                points,
            }],
        }
    }

    #[test]
    fn frontier_respects_both_objectives_and_marks_equal_policies_together() {
        let points = [
            point("Costly tie", 3, 5),
            point("Early", 1, 0),
            point("Middle", 3, 2),
            point("Duplicate", 3, 2),
            point("Worse at same cost", 1, 2),
            point("Late", 4, 7),
        ];
        let groups = PointGroups::new(&points, TradeoffAxis::Unmatched);
        assert_eq!(groups.keys, [1, 2, 3, 3, 4, 5]);
        assert_eq!(groups.frontier, [1, 2, 4]);
        assert_eq!(groups.on_frontier, [false, true, true, false, true]);
        let html = study(points.into()).render_html().unwrap();
        assert_eq!(
            html.matches(r#"<td data-label="On frontier">Yes</td>"#)
                .count(),
            4
        );
        assert_eq!(
            html.matches(r#"<td data-label="On frontier">No</td>"#)
                .count(),
            2
        );
    }

    #[test]
    fn shared_axes_use_observations_and_recall_uses_gold() {
        let mut report = study(vec![point("Stock", 2, 1)]);
        report.series.push(TradeoffSeries {
            label: "Larger".into(),
            observations: 100,
            gold: 25,
            points: vec![point("Other policy", 20, 10)],
        });
        let html = report.render_html().unwrap();
        assert_eq!(html.matches(r#"<td data-label="Correct / 100">20.000</td><td data-label="Unmatched / 100">10.000</td>"#).count(), 2);
        assert!(
            html.contains(r#"<td data-label="Recall">40.0%</td>"#)
                && html.contains(r#"<td data-label="Recall">80.0%</td>"#)
        );
        let scatter_x: Vec<_> = html
            .lines()
            .filter_map(|l| l.split_once("<circle cx=\"").map(|(_, rest)| rest))
            .map(|l| l.split('"').next().unwrap())
            .collect();
        // Both cohorts have the same x-rate, with different raw counts.
        assert_eq!(scatter_x.len(), 2);
        assert_eq!(scatter_x[0], scatter_x[1]);
    }

    #[test]
    fn zero_denominators_and_full_integer_range_remain_interpretable() {
        let mut report = study(vec![point("Skip", 0, 0)]);
        report.series[0].observations = 0;
        report.series[0].gold = 0;
        report.series.push(TradeoffSeries {
            label: "Only negatives".into(),
            observations: u64::MAX,
            gold: 0,
            points: vec![point("Skip", 0, 0), point("Predict", 0, u64::MAX)],
        });
        let html = report.render_html().unwrap();
        assert!(html.contains("Rates undefined for an empty cohort"));
        assert!(html.contains(r#"<td data-label="Correct / 100">—</td><td data-label="Unmatched / 100">—</td><td data-label="Recall">—</td><td data-label="On frontier">—</td>"#));
        assert!(html.contains("18,446,744,073,709,551,615"));
        assert!(!html.contains("NaN") && !html.contains("infinity"));
        report.series[0].points[0].unmatched = 1;
        assert!(report
            .render_svg()
            .unwrap_err()
            .contains("zero observations"));
    }

    #[test]
    fn text_roundtrips_and_renders_as_text() {
        let label = "長い設定 & <tag> \"ø\"";
        let report = study(vec![point(label, 3, 1)]);
        let wire = serde_json::to_value(&report).unwrap();
        assert_eq!(
            wire["series"][0]["points"][0],
            json!({"label":label, "correct":3, "unmatched":1})
        );
        assert_eq!(serde_json::from_value::<Tradeoff>(wire).unwrap(), report);
        let svg = report.render_svg().unwrap();
        assert!(svg.contains("長い設定 &amp; &lt;tag&gt; &quot;ø&quot;"));
        assert!(!svg.contains("<tag>"));
        assert!(svg.contains("Test has 10 sentences and 5 gold items."));
        let cjk = "漢".repeat(150);
        assert_eq!(
            wrap(&cjk, 20)
                .iter()
                .map(|s| s.chars().count())
                .collect::<Vec<_>>(),
            vec![10; 15]
        );
        assert_eq!(wrap("Jose\u{301} del Río", 6)[0], "Jose\u{301}");
        // A wide all-capitals title would overrun the right edge if fitted by
        // raw character count. Wrap it while retaining every character.
        let capitals = "H".repeat(110);
        let headline = wrap(&capitals, 61);
        assert!(headline.iter().all(|line| line.len() <= 41));
        assert_eq!(headline.concat(), capitals);
        assert_eq!(report.render_svg().unwrap(), svg);
    }

    #[test]
    fn suggestion_budgets_rank_total_output_and_keep_all_tied_names() {
        let report = study(vec![
            point("A", 2, 1),
            point("B", 3, 1),
            point("Alias", 2, 1),
        ]);
        let error = PointGroups::new(&report.series[0].points, TradeoffAxis::Unmatched);
        let queue = PointGroups::new(&report.series[0].points, TradeoffAxis::Suggestions);
        assert_eq!(error.on_frontier, [false, true]);
        assert_eq!(queue.on_frontier, [true, true]);
        let svg = report
            .render_svg_with_axis(TradeoffAxis::Suggestions)
            .unwrap();
        assert!(svg.contains("A / Alias") && svg.contains("Suggestions / 100"));
        assert!(!svg.contains("plot-key"));
        let mut large = study(vec![point("all", u64::MAX, u64::MAX)]);
        large.series[0].gold = u64::MAX;
        let html = large
            .render_html_with_axis(TradeoffAxis::Suggestions)
            .unwrap();
        assert!(html.contains("36,893,488,147,419,103,230"));
    }

    #[test]
    fn rejects_unscorable_input_at_its_source() {
        let mut report = study(vec![point("Stock", 6, 0)]);
        assert!(report
            .render_html()
            .unwrap_err()
            .contains("series[0].points[0].correct"));
        report.series[0].points[0].correct = 0;
        report.series[0].label = "bad\u{0}".into();
        assert!(report.validate().unwrap_err().contains("series[0].label"));
        report.series[0].label = "Test".into();
        report.series[0].points.clear();
        assert!(report.validate().unwrap_err().contains("operating point"));
        report.schema = "anno.tradeoff/v2".into();
        assert!(report.validate().unwrap_err().contains(TRADEOFF_SCHEMA));
    }
}
