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

const LEFT: i32 = 32;
const PLOT_X: i32 = 80;
const PLOT_WIDTH: i32 = 310;
const KEY_X: i32 = 436;
const LABEL_X: i32 = 477;
const LABEL_COLUMNS: usize = 33;
const LABEL_WIDTH: i32 = 242;

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

    /// Draw a standalone SVG. Each panel uses the same two zero-based rate axes.
    pub fn render_svg(&self) -> Result<String, String> {
        self.validate()?;
        let mut x_max = 0.0_f64;
        let mut y_max = 0.0_f64;
        let mut correct_width = 70;
        let mut unmatched_width = 90;
        for s in &self.series {
            for p in &s.points {
                if s.observations > 0 {
                    x_max = x_max.max(rate(p.unmatched, s.observations));
                    y_max = y_max.max(rate(p.correct, s.observations));
                }
                correct_width = correct_width.max(integer(p.correct).len() as i32 * 9 + 18);
                unmatched_width = unmatched_width.max(integer(p.unmatched).len() as i32 * 9 + 18);
            }
        }
        let correct_x = LABEL_X + LABEL_WIDTH + correct_width;
        let unmatched_x = correct_x + unmatched_width;
        let recall_x = unmatched_x + 73;
        let width = recall_x + LEFT;
        let columns = [correct_x, unmatched_x, recall_x];
        let x_axis = Axis::new(x_max);
        let y_axis = Axis::new(y_max);
        let mut content = String::new();
        let mut accessible = String::new();
        let mut y = 38;
        let title = wrap(&self.title, ((width - 2 * LEFT) / 15) as usize);
        lines(&mut content, &title, LEFT, y, 29, "heading");
        y += title.len() as i32 * 29;
        let description = format!(
            "Correct matches and unmatched predictions per 100 {}.",
            self.unit
        );
        let subtitle = wrap(&description, ((width - 2 * LEFT) / 10) as usize);
        lines(&mut content, &subtitle, LEFT, y, 19, "subtitle");
        y += subtitle.len() as i32 * 19;
        let legend = "Filled points trace the best match count at each unmatched budget. Open points have a better alternative.";
        let legend_lines = wrap(legend, ((width - 2 * LEFT) / 7) as usize);
        lines(&mut content, &legend_lines, LEFT, y, 17, "scope");
        y += legend_lines.len() as i32 * 17 + 6;
        for s in &self.series {
            writeln!(
                content,
                r#"<g><title>{}</title><path d="M{LEFT} {y}H{recall_x}" class="rule"/>"#,
                escape(&s.label)
            )
            .unwrap();
            let cohort = wrap(&s.label, ((width - 2 * LEFT) / 12) as usize);
            y += 28;
            lines(&mut content, &cohort, LEFT, y, 22, "cohort");
            y += cohort.len() as i32 * 22;
            let scope = format!(
                "{} {} · {} gold",
                integer(s.observations),
                self.unit,
                integer(s.gold)
            );
            let scope_lines = wrap(&scope, ((width - 2 * LEFT) / 9) as usize);
            lines(&mut content, &scope_lines, LEFT, y, 18, "scope");
            y += scope_lines.len() as i32 * 18 + 15;
            let groups = PointGroups::new(&s.points);
            let table_bottom = s.svg_table(&mut content, &groups, y, columns);
            let plot_height = 146.max(groups.unique.len() as i32 * 17);
            if s.observations == 0 {
                writeln!(content, r#"<text x="{PLOT_X}" y="{}" class="scope">Rates undefined for an empty cohort</text>"#, y + 33).unwrap();
                y = table_bottom.max(y + 55);
            } else {
                s.plot(&mut content, &groups, [x_axis, y_axis], y, plot_height);
                y = table_bottom.max(y + plot_height + 57);
            }
            write!(
                accessible,
                " {} has {} {} and {} gold items.",
                s.label,
                integer(s.observations),
                self.unit,
                integer(s.gold)
            )
            .unwrap();
            for p in &s.points {
                let recall = if s.gold == 0 {
                    "undefined".into()
                } else {
                    recall(p.correct, s.gold)
                };
                write!(
                    accessible,
                    " {} has {} correct, {} unmatched, and {recall} recall.",
                    p.label,
                    integer(p.correct),
                    integer(p.unmatched)
                )
                .unwrap();
            }
            content.push_str("</g>\n");
            y += 22;
        }
        Ok(format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{y}" viewBox="0 0 {width} {y}" role="img" aria-labelledby="chart-title chart-desc">
<title id="chart-title">{}</title><desc id="chart-desc">{}. Each panel shares the correct and unmatched rate axes. The numbered points identify the exact counts and recall in its table. Identical counts share a point.{}</desc>
<style>text{{font-family:"DejaVu Sans",sans-serif;fill:#183541;font-size:13px}}.heading{{font-size:24px;font-weight:700}}.subtitle{{font-size:15px;fill:#456371}}.cohort{{font-size:19px;font-weight:700}}.scope,.axis{{fill:#456371}}.small,.axis{{font-size:11px}}.col{{font-weight:700;font-size:12px}}.num{{text-anchor:end;font-variant-numeric:tabular-nums}}.key{{font-size:12px;font-weight:700;fill:#006867}}.plot-key{{paint-order:stroke;stroke:white;stroke-width:2;stroke-linejoin:round}}.rule,.grid{{fill:none;stroke:#d5e0e2;stroke-width:1}}.frontier{{fill:none;stroke:#6da8a2;stroke-width:1.6}}.optimal{{fill:#006867;stroke:white;stroke-width:1.2}}.dominated{{fill:white;stroke:#456371;stroke-width:1.5}}.leader{{stroke:#658384;stroke-width:.8}}</style>
<rect width="{width}" height="{y}" fill="white"/>{content}</svg>
"#,
            escape(&self.title),
            escape(description.trim_end_matches('.')),
            escape(&accessible)
        ))
    }

    /// Draw the SVG and a semantic HTML table with the exact counts and rates.
    pub fn render_html(&self) -> Result<String, String> {
        let svg = self.render_svg()?;
        let mut table = String::from("<table><caption>Operating points</caption><thead><tr>");
        for name in [
            "Cohort",
            "Operating point",
            "Observations",
            "Gold",
            "Correct",
            "Unmatched",
            "Correct / 100",
            "Unmatched / 100",
            "Recall",
            "On frontier",
        ] {
            write!(table, r#"<th scope="col">{name}</th>"#).unwrap();
        }
        table.push_str("</tr></thead><tbody>");
        for s in &self.series {
            let groups = PointGroups::new(&s.points);
            for (p, &key) in s.points.iter().zip(&groups.keys) {
                let on_frontier = if s.observations == 0 {
                    "—"
                } else if groups.on_frontier[key - 1] {
                    "Yes"
                } else {
                    "No"
                };
                let per_100 = |n| {
                    if s.observations == 0 {
                        "—".into()
                    } else {
                        let value = rate(n, s.observations);
                        if n > 0 && value < 0.001 {
                            format!("{value:.3e}")
                        } else {
                            format!("{value:.3}")
                        }
                    }
                };
                write!(table, r#"<tr><td>{}</td><th scope="row">{}</th><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{on_frontier}</td></tr>"#,
                    escape(&s.label), escape(&p.label), integer(s.observations), integer(s.gold), integer(p.correct), integer(p.unmatched), per_100(p.correct), per_100(p.unmatched), recall(p.correct, s.gold)).unwrap();
            }
        }
        table.push_str("</tbody></table>");
        Ok(html_document(&self.title, &svg, &table))
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
    fn new(points: &[OperatingPoint]) -> Self {
        let mut ids = BTreeMap::new();
        let mut unique = Vec::new();
        let keys = points
            .iter()
            .map(|p| {
                let counts = (p.unmatched, p.correct);
                *ids.entry((p.unmatched, Reverse(p.correct)))
                    .or_insert_with(|| {
                        unique.push(counts);
                        unique.len()
                    })
            })
            .collect();
        // First visit the most correct matches at each unmatched count. A
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
    fn svg_table(&self, svg: &mut String, groups: &PointGroups, y: i32, columns: [i32; 3]) -> i32 {
        writeln!(
            svg,
            r#"<text x="{LABEL_X}" y="{y}" class="col">Operating point</text>"#
        )
        .unwrap();
        for (x, label) in columns.into_iter().zip(["Correct", "Unmatched", "Recall"]) {
            writeln!(
                svg,
                r#"<text x="{x}" y="{y}" class="col num">{label}</text>"#
            )
            .unwrap();
        }
        let mut row_y = y + 24;
        for (p, key) in self.points.iter().zip(&groups.keys) {
            let row_lines = wrap(&p.label, LABEL_COLUMNS);
            writeln!(svg, r#"<g><title>{}</title>"#, escape(&p.label)).unwrap();
            if self.observations > 0 {
                let symbol = if groups.on_frontier[key - 1] {
                    "optimal"
                } else {
                    "dominated"
                };
                writeln!(svg, r#"<circle cx="{}" cy="{}" r="4" class="{symbol}"/><text x="{KEY_X}" y="{row_y}" class="key">{key}</text>"#, KEY_X - 10, row_y - 4).unwrap();
            }
            lines(svg, &row_lines, LABEL_X, row_y, 17, "label");
            for (x, value) in columns.into_iter().zip([
                integer(p.correct),
                integer(p.unmatched),
                recall(p.correct, self.gold),
            ]) {
                writeln!(
                    svg,
                    r#"<text x="{x}" y="{row_y}" class="num">{value}</text>"#
                )
                .unwrap();
            }
            svg.push_str("</g>\n");
            row_y += row_lines.len() as i32 * 17 + 8;
        }
        row_y - 8
    }

    fn plot(&self, svg: &mut String, groups: &PointGroups, axes: [Axis; 2], y: i32, height: i32) {
        let [x_axis, y_axis] = axes;
        let top = y + 14;
        let bottom = top + height;
        let right = PLOT_X + PLOT_WIDTH;
        writeln!(
            svg,
            r#"<text x="{PLOT_X}" y="{y}" class="col">Correct / 100</text>"#
        )
        .unwrap();
        for i in 0..=y_axis.ticks {
            let gy = f64::from(bottom) - f64::from(i) * f64::from(height) / f64::from(y_axis.ticks);
            writeln!(svg, r#"<path d="M{PLOT_X} {gy:.2}H{right}" class="grid"/><text x="{}" y="{:.2}" class="axis num">{}</text>"#, PLOT_X - 9, gy + 4.0, y_axis.label(i)).unwrap();
        }
        for i in 0..=x_axis.ticks {
            let gx =
                f64::from(PLOT_X) + f64::from(i) * f64::from(PLOT_WIDTH) / f64::from(x_axis.ticks);
            writeln!(svg, r#"<path d="M{gx:.2} {top}V{bottom}" class="grid"/><text x="{gx:.2}" y="{}" class="axis" text-anchor="middle">{}</text>"#, bottom + 16, x_axis.label(i)).unwrap();
        }
        writeln!(
            svg,
            r#"<text x="{right}" y="{}" class="col num">Unmatched / 100</text>"#,
            bottom + 36
        )
        .unwrap();
        let coords: Vec<_> = groups
            .unique
            .iter()
            .map(|&(fp, tp)| {
                let x = f64::from(PLOT_X)
                    + f64::from(PLOT_WIDTH) * rate(fp, self.observations) / x_axis.max();
                let y = f64::from(bottom)
                    - f64::from(height) * rate(tp, self.observations) / y_axis.max();
                (x, y)
            })
            .collect();
        let (start_x, start_y) = coords[groups.frontier[0]];
        let mut path = format!("M{start_x:.2} {start_y:.2}");
        for &index in &groups.frontier[1..] {
            let (x, y) = coords[index];
            write!(path, "H{x:.2}V{y:.2}").unwrap();
        }
        writeln!(svg, r#"<path d="{path}H{right}" class="frontier"/>"#).unwrap();

        // Spread annotation baselines vertically in data order. Keep the marks
        // on their true coordinates and connect displaced labels to their mark.
        let mut ordered: Vec<_> = (0..coords.len()).collect();
        ordered.sort_by(|&a, &b| coords[a].1.total_cmp(&coords[b].1));
        let mut labels = vec![0.0; coords.len()];
        let mut previous = f64::from(top) - 17.0;
        for &i in &ordered {
            labels[i] = coords[i].1.max(previous + 17.0);
            previous = labels[i];
        }
        let mut next = f64::from(bottom) + 17.0;
        for &i in ordered.iter().rev() {
            labels[i] = labels[i].min(next - 17.0);
            next = labels[i];
        }
        let mut marks = String::new();
        let mut numbers = String::new();
        for (i, &(x, y)) in coords.iter().enumerate() {
            let key = (i + 1).to_string();
            let ly = labels[i];
            let label_width = key.len() as f64 * 9.0;
            let left = label_left(&coords, x, ly, label_width);
            let leader_end = if left > x {
                left - 2.0
            } else {
                (left + label_width + 2.0).min(x)
            };
            if (ly - y).abs() > 3.0 || (x - leader_end).abs() > 12.0 {
                writeln!(svg, r#"<line x1="{x:.2}" y1="{y:.2}" x2="{leader_end:.2}" y2="{ly:.2}" class="leader"/>"#).unwrap();
            }
            let (unmatched, correct) = groups.unique[i];
            let symbol = if groups.on_frontier[i] {
                "optimal"
            } else {
                "dominated"
            };
            writeln!(marks, r#"<g><title>Point {key} · {} correct · {} unmatched</title><circle cx="{x:.2}" cy="{y:.2}" r="4" class="{symbol}"/></g>"#, integer(correct), integer(unmatched)).unwrap();
            writeln!(
                numbers,
                r#"<text x="{left:.2}" y="{:.2}" class="key plot-key">{key}</text>"#,
                ly + 4.0
            )
            .unwrap();
        }
        svg.push_str(&marks);
        svg.push_str(&numbers);
    }
}

/// Find the nearest horizontal label position clear of every marker on this row.
/// Labels already have distinct vertical slots; a narrow outer gutter is used
/// only when the whole row is occupied.
fn label_left(marks: &[(f64, f64)], x: f64, y: f64, width: f64) -> f64 {
    let left = f64::from(PLOT_X);
    let right = f64::from(PLOT_X + PLOT_WIDTH);
    let preferred = if x + 10.0 + width <= right {
        x + 10.0
    } else {
        x - 10.0 - width
    };
    let mut blocked: Vec<_> = marks
        .iter()
        .filter(|(_, cy)| (cy - y).abs() <= 12.0)
        .map(|(cx, _)| (cx - 7.0 - width, cx + 7.0))
        .collect();
    blocked.sort_by(|a, b| a.0.total_cmp(&b.0));

    let last = right - width;
    let mut cursor = left;
    let mut best = None;
    for (start, end) in blocked.into_iter().chain(std::iter::once((last, right))) {
        if cursor <= start.min(last) {
            let candidate = preferred.clamp(cursor, start.min(last));
            if best.is_none_or(|previous: f64| {
                (candidate - preferred).abs() < (previous - preferred).abs()
            }) {
                best = Some(candidate);
            }
        }
        cursor = cursor.max(end);
        if cursor > last {
            break;
        }
    }
    best.unwrap_or(right + 8.0)
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
        let groups = PointGroups::new(&points);
        assert_eq!(groups.keys, [1, 2, 3, 3, 4, 5]);
        assert_eq!(groups.frontier, [1, 2, 4]);
        assert_eq!(groups.on_frontier, [false, true, true, false, true]);
        let html = study(points.into()).render_html().unwrap();
        assert_eq!(html.matches("<td>Yes</td>").count(), 4);
        assert_eq!(html.matches("<td>No</td>").count(), 2);
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
        assert_eq!(html.matches("<td>20.000</td><td>10.000</td>").count(), 2);
        assert!(html.contains("<td>40.0%</td>") && html.contains("<td>80.0%</td>"));
        let scatter_x: Vec<_> = html
            .lines()
            .filter_map(|l| l.split_once("<circle cx=\"").map(|(_, rest)| rest))
            .map(|l| l.split('"').next().unwrap())
            .collect();
        // Both cohorts have the same x-rate, with different raw counts.
        assert_eq!(scatter_x.len(), 4);
        assert_eq!(scatter_x[1], scatter_x[3]);
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
        assert!(html.contains("<td>—</td><td>—</td><td>—</td><td>—</td>"));
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
    fn point_keys_clear_their_neighbors_in_a_cluster_at_zero() {
        let at_zero = f64::from(PLOT_X);
        let marks = [(at_zero, 150.0), (at_zero + 14.0, 148.0)];
        let left = label_left(&marks, at_zero, 150.0, 9.0);
        for &(cx, _) in &marks {
            assert!(left >= cx + 7.0 || left + 9.0 <= cx - 7.0);
        }
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
