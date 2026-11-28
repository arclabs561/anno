use super::{Paired, PairedRow};
use crate::viz::{escape, integer, lines, wrap};
use std::fmt::Write;

const WIDTH: i32 = 520;
const LEFT: i32 = 18;

fn delta(value: i128) -> String {
    let sign = match value.cmp(&0) {
        std::cmp::Ordering::Greater => "+",
        std::cmp::Ordering::Less => "−",
        std::cmp::Ordering::Equal => "",
    };
    format!("{sign}{}", integer(value.unsigned_abs() as u64))
}

fn decimal(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    if value.abs() >= 1e12 || value.abs() < 0.001 {
        return format!("{value:.2e}");
    }
    format!("{value:.3}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

// Every string entering SVG text is escaped by lines(). Wrapped rows also
// handle long user-supplied cohort, setting, and interval-method names.
fn paragraph(svg: &mut String, y: &mut i32, text: &str, columns: usize, step: i32, class: &str) {
    let rows = wrap(text, columns);
    lines(svg, &rows, LEFT, *y, step, class);
    *y += rows.len() as i32 * step;
}

struct Measure {
    label: &'static str,
    baseline: String,
    candidate: String,
    change: String,
}

impl Measure {
    fn count(label: &'static str, base: Option<u64>, new: Option<u64>) -> Self {
        Self {
            label,
            baseline: base.map(integer).unwrap_or_else(|| "—".into()),
            candidate: new.map(integer).unwrap_or_else(|| "—".into()),
            change: base
                .zip(new)
                .map(|(b, n)| delta(i128::from(n) - i128::from(b)))
                .unwrap_or_else(|| "—".into()),
        }
    }
}

impl PairedRow {
    fn scope(&self, unit: &str) -> String {
        // A study's free-text unit can include irregular plurals or a qualifier.
        let sample = match (self.observations, unit) {
            (1, "documents" | "sentences") => format!("1 {}", unit.trim_end_matches('s')),
            (1, unit) => format!("{unit} (n = 1)"),
            (count, unit) => format!("{} {unit}", integer(count)),
        };
        format!("{} · {sample}", self.role)
    }

    fn net_caption(&self) -> String {
        let change = self.net();
        let noun = if change.abs() == 1 {
            "match"
        } else {
            "matches"
        };
        format!("{} net exact {noun}", delta(change))
    }

    fn measures(&self) -> Vec<Measure> {
        let (b, c) = (&self.baseline, &self.candidate);
        let mut measures = vec![
            Measure::count(
                "Total suggestions",
                Some(b.correct + b.unmatched),
                Some(c.correct + c.unmatched),
            ),
            Measure::count("Unmatched", Some(b.unmatched), Some(c.unmatched)),
        ];
        if b.calls.is_some() || c.calls.is_some() {
            measures.push(Measure::count("Model calls", b.calls, c.calls));
        }
        if b.median_ms.is_some() || c.median_ms.is_some() {
            let change = b.median_ms.zip(c.median_ms).map(|(old, new)| {
                let d = new - old;
                format!("{}{}", if d > 0.0 { "+" } else { "" }, decimal(d))
            });
            measures.push(Measure {
                label: "Median latency (ms)",
                baseline: b.median_ms.map(decimal).unwrap_or_else(|| "—".into()),
                candidate: c.median_ms.map(decimal).unwrap_or_else(|| "—".into()),
                change: change.unwrap_or_else(|| "—".into()),
            });
        }
        measures
    }

    // Sum of exact and unmatched gives the operational suggestion budget.
    // Pair related measurements in a single compact row where space permits.
    fn cost_lines(measures: &[Measure]) -> Vec<String> {
        let mut rows = Vec::new();
        let mut line = String::new();
        for m in measures {
            let field = format!(
                "{} {} → {} ({})",
                m.label, m.baseline, m.candidate, m.change
            );
            if !line.is_empty() {
                let together = format!("{line}  ·  {field}");
                if wrap(&together, 49).len() == 1 {
                    line = together;
                    continue;
                }
                rows.push(std::mem::take(&mut line));
            }
            line = field;
        }
        if !line.is_empty() {
            rows.push(line);
        }
        rows
    }

    fn retention(&self) -> String {
        let baseline = self.baseline.correct;
        if baseline == 0 {
            return "Retained 0 of 0 baseline matches (undefined share)".into();
        }
        let kept = baseline - self.lost;
        format!(
            "Retained {} of {} baseline matches ({:.1}%)",
            integer(kept),
            integer(baseline),
            100.0 * kept as f64 / baseline as f64
        )
    }

    fn interval_caption(&self) -> Option<String> {
        self.interval.as_ref().map(|i| {
            format!(
                "Interval {} to {} net matches · {}",
                decimal(i.lower),
                decimal(i.upper),
                i.method
            )
        })
    }

    // The dot is signed change in exact matches. Whiskers appear only when
    // provided by the study. All rows use the same linear count scale.
    fn plot(&self, svg: &mut String, y: i32, bound: f64, left: i32, right: i32, ticks: bool) {
        let middle = (right + left) / 2;
        let x = |v: f64| f64::from(middle) + v / bound * f64::from(right - left) / 2.0;
        let point = x(self.net() as f64);
        writeln!(
            svg,
            r##"<path d="M{left} {y}H{right}" stroke="#d2deda"/><path d="M{middle} {}v20" stroke="#688079"/>"##,
            y - 10
        )
        .unwrap();
        if let Some(i) = &self.interval {
            let (lo, hi) = (x(i.lower), x(i.upper));
            writeln!(svg, r##"<path d="M{lo:.2} {}v12M{lo:.2} {y}H{hi:.2}M{hi:.2} {}v12" fill="none" stroke="#356784" stroke-width="2.5"/>"##, y-6, y-6).unwrap();
        }
        writeln!(svg, r##"<circle cx="{point:.2}" cy="{y}" r="6" fill="#00675c" stroke="white" stroke-width="1.2"/>"##).unwrap();
        if !ticks {
            return;
        }
        let tick = decimal(bound);
        for (px, anchor, label) in [
            (left, "start", format!("−{tick}")),
            (middle, "middle", "0".into()),
            (right, "end", format!("+{tick}")),
        ] {
            writeln!(
                svg,
                r#"<text x="{px}" y="{}" text-anchor="{anchor}" class="tick">{label}</text>"#,
                y + 25
            )
            .unwrap();
        }
    }
}

impl Paired {
    fn bound(&self) -> f64 {
        self.rows.iter().fold(1.0f64, |greatest, row| {
            let mut limit = greatest.max((row.net() as f64).abs());
            if let Some(interval) = &row.interval {
                limit = limit.max(interval.lower.abs()).max(interval.upper.abs());
            }
            limit
        })
    }

    /// Render a narrow, directly labeled vector chart. Counts stay on the
    /// original cohort scale and every row has an accessible text description.
    pub fn render_svg(&self) -> Result<String, String> {
        self.validate()?;
        let bound = self.bound();
        let mut body = String::new();
        let mut y = 32;
        paragraph(&mut body, &mut y, &self.title, 34, 29, "heading");
        paragraph(
            &mut body,
            &mut y,
            &format!(
                "Baseline · {} → Candidate · {}",
                self.baseline, self.candidate
            ),
            58,
            20,
            "scope",
        );
        let limit = decimal(bound);
        paragraph(
            &mut body,
            &mut y,
            &format!("Net exact matches · −{limit} to +{limit} · center is zero"),
            58,
            20,
            "scope",
        );
        let mut methods = self
            .rows
            .iter()
            .filter_map(|r| r.interval.as_ref().map(|i| i.method.as_str()));
        let common_method = methods
            .next()
            .filter(|first| methods.all(|other| other == *first));
        if let Some(method) = common_method {
            paragraph(
                &mut body,
                &mut y,
                &format!("Intervals · {method}"),
                58,
                20,
                "scope",
            );
        }
        let mut description = format!(
            "Baseline {} compared with candidate {} on the same inputs. Total suggestions include exact and unmatched predictions.",
            self.baseline, self.candidate
        );
        for row in &self.rows {
            writeln!(body, r##"<path d="M18 {y}H502" stroke="#bdcec9"/>"##).unwrap();
            y += 25;
            paragraph(&mut body, &mut y, &row.label, 45, 24, "cohort");
            let scope = row.scope(&self.unit);
            paragraph(&mut body, &mut y, &scope, 58, 20, "scope");
            let change = format!(
                "{} · {} → {}",
                row.net_caption(),
                integer(row.baseline.correct),
                integer(row.candidate.correct)
            );
            y += 2;
            row.plot(&mut body, y - 5, bound, 390, 494, false);
            paragraph(&mut body, &mut y, &change, 40, 22, "score");
            if let Some(i) = &row.interval {
                let method = if common_method.is_none() {
                    format!(" · {}", i.method)
                } else {
                    String::new()
                };
                paragraph(
                    &mut body,
                    &mut y,
                    &format!(
                        "Interval {} to {} net matches{method}",
                        decimal(i.lower),
                        decimal(i.upper)
                    ),
                    58,
                    20,
                    "scope",
                );
            }
            let pairs = format!(
                "Gained {} · lost {}",
                integer(row.gained),
                integer(row.lost)
            );
            let kept = row.baseline.correct - row.lost;
            let share = if row.baseline.correct == 0 {
                "undefined share".to_owned()
            } else {
                format!("{:.1}%", 100.0 * kept as f64 / row.baseline.correct as f64)
            };
            let transitions = format!(
                "{pairs} · retained {}/{} ({share})",
                integer(kept),
                integer(row.baseline.correct)
            );
            paragraph(&mut body, &mut y, &transitions, 58, 20, "scope");
            let measures = row.measures();
            for pair in measures.chunks(2) {
                let mut bottom = y;
                for (i, m) in pair.iter().enumerate() {
                    let x = if i == 0 { LEFT } else { 267 };
                    let labels = wrap(&format!("{} {}", m.label, m.change), 27);
                    lines(&mut body, &labels, x, y, 19, "measure");
                    let next = y + labels.len() as i32 * 19;
                    let values = wrap(&format!("{} → {}", m.baseline, m.candidate), 27);
                    lines(&mut body, &values, x, next, 19, "scope");
                    bottom = bottom.max(next + values.len() as i32 * 19);
                }
                y = bottom + 3;
            }
            y += 2;
            write!(
                description,
                " {}. {scope}. {change}. {pairs}. {}.",
                row.label,
                row.retention()
            )
            .unwrap();
            if let Some(caption) = row.interval_caption() {
                write!(description, " {caption}.").unwrap();
            }
            for measure in measures {
                write!(
                    description,
                    " {} from {} to {}.",
                    measure.label, measure.baseline, measure.candidate
                )
                .unwrap();
            }
        }
        Ok(format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{y}" viewBox="0 0 {WIDTH} {y}" role="img" aria-labelledby="paired-title paired-desc">
<title id="paired-title">{}</title><desc id="paired-desc">{}</desc>
<style>text{{font-family:"DejaVu Sans",sans-serif;fill:#183541;font-size:17px}}.heading{{font-size:25px;font-weight:700}}.cohort{{font-size:20px;font-weight:700}}.score{{font-size:17px;font-weight:700;fill:#00675c}}.measure{{font-size:17px;font-weight:600}}.tick,.scope{{font-size:17px;fill:#415e66}}</style>
<rect width="100%" height="100%" fill="white"/>{body}</svg>
"##,
            escape(&self.title),
            escape(&description)
        ))
    }

    /// Render a standalone HTML comparison with semantic tables and text that
    /// reflows on narrow screens. No scripts, network or web fonts are required.
    pub fn render_html(&self) -> Result<String, String> {
        self.validate()?;
        let bound = self.bound();
        let mut cards = String::new();
        for row in &self.rows {
            write!(
                cards,
                "<section><h2>{}</h2><p>{}</p><h3>{} · {} → {}</h3>",
                escape(&row.label),
                escape(&row.scope(&self.unit)),
                row.net_caption(),
                integer(row.baseline.correct),
                integer(row.candidate.correct),
            )
            .unwrap();
            cards.push_str(r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 54" aria-hidden="true" focusable="false">"#);
            row.plot(&mut cards, 18, bound, 38, 482, true);
            cards.push_str("</svg>");
            if let Some(caption) = row.interval_caption() {
                write!(cards, "<p>{}</p>", escape(&caption)).unwrap();
            }
            write!(
                cards,
                "<p><strong>Gained {} · lost {}</strong><br>{}</p>",
                integer(row.gained),
                integer(row.lost),
                row.retention()
            )
            .unwrap();
            let mut measures = row.measures();
            for text in PairedRow::cost_lines(&measures) {
                write!(cards, "<p class=\"cost\">{}</p>", escape(&text)).unwrap();
            }
            measures.insert(
                1,
                Measure::count(
                    "Exact matches",
                    Some(row.baseline.correct),
                    Some(row.candidate.correct),
                ),
            );
            cards.push_str("<details><summary>All measurements</summary>");
            write!(cards,r#"<table><caption>Outputs and measured costs</caption><thead><tr><th scope="col">Measure</th><th scope="col">Baseline<br>{}</th><th scope="col">Candidate<br>{}</th><th scope="col">Change</th></tr></thead><tbody>"#,
                escape(&self.baseline),escape(&self.candidate)).unwrap();
            for measure in measures {
                write!(
                    cards,
                    r#"<tr><th scope="row">{}</th><td>{}</td><td>{}</td><td>{}</td></tr>"#,
                    measure.label, measure.baseline, measure.candidate, measure.change
                )
                .unwrap();
            }
            cards.push_str("</tbody></table></details></section>");
        }
        Ok(format!(
            r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
*{{box-sizing:border-box}}body{{margin:1.25rem auto;padding:0 1rem;max-width:44rem;color:#183541;background:white;font:1rem/1.45 "DejaVu Sans",sans-serif;overflow-wrap:anywhere}}
h1,h2,h3{{line-height:1.18}}h1{{font-size:1.8rem}}h2{{margin:0}}h3{{color:#00675c;margin-bottom:.3rem}}p{{margin:.4rem 0 .85rem}}section{{border-top:1px solid #bdcec9;padding:1.25rem 0}}
svg{{display:block;width:100%;max-width:32.5rem}}.tick{{font:18px "DejaVu Sans",sans-serif;fill:#415e66}}table{{border-collapse:collapse;table-layout:fixed;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:right;border-bottom:1px solid #d8e2df;padding:.4rem .25rem;vertical-align:top}}th:first-child{{width:30%;text-align:left}}details{{margin:.65rem 0}}summary{{color:#00675c;cursor:pointer}}.cost{{margin:.15rem 0}}caption{{text-align:left;font-weight:700;margin:.5rem 0}}@media(max-width:32rem){{body{{padding:0 .75rem}}th,td{{padding:.4rem .15rem}}}}
</style></head><body><main><h1>{title}</h1><p>Baseline · {baseline}<br>Candidate · {candidate}</p>{cards}</main></body></html>
"##,
            title = escape(&self.title),
            baseline = escape(&self.baseline),
            candidate = escape(&self.candidate)
        ))
    }
}
