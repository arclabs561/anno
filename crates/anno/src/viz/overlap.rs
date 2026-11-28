//! Count shared character positions in grounded extraction signals across documents.

use super::{check_label, escape, html_document, integer, lines, wrap};
use crate::GroundedDocument;
use serde::de::{self, value::MapAccessDeserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;
use std::fmt::{self, Write};
use std::io::Read;

/// Version of the portable source-free signal overlap summary.
pub const OVERLAP_SCHEMA: &str = "anno.overlap/v1";

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LabelCounts {
    signals: u64,
    signals_with_overlap: u64,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Coverage {
    zero: u64,
    one: u64,
    two: u64,
    three_or_more: u64,
}

impl Coverage {
    fn counts(&self) -> [u64; 4] {
        [self.zero, self.one, self.two, self.three_or_more]
    }

    fn add(&mut self, counts: [u64; 4]) {
        self.zero += counts[0];
        self.one += counts[1];
        self.two += counts[2];
        self.three_or_more += counts[3];
    }
}

/// Aggregate counts over whole source texts, using Unicode scalar offsets.
///
/// Read grounded extraction results, pipeline JSON/JSONL, or a saved summary
/// with [`Self::from_json`]. Distinct signals count independently, even when
/// their labels and locations coincide. Document and signal IDs stay local.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Overlap {
    schema: String,
    title: String,
    documents: u64,
    documents_with_overlap: u64,
    signals: u64,
    signals_with_overlap: u64,
    characters: u64,
    coverage: Coverage,
    labels: BTreeMap<String, LabelCounts>,
}

impl Default for Overlap {
    fn default() -> Self {
        Self {
            schema: OVERLAP_SCHEMA.into(),
            title: "Extraction signal overlap".into(),
            documents: 0,
            documents_with_overlap: 0,
            signals: 0,
            signals_with_overlap: 0,
            characters: 0,
            coverage: Coverage::default(),
            labels: BTreeMap::new(),
        }
    }
}

impl Overlap {
    /// Read one grounded document, a whitespace-separated stream of documents,
    /// a pipeline document array, or one previously exported overlap summary.
    pub fn from_json(input: impl Read) -> Result<Self, String> {
        let inputs = serde_json::Deserializer::from_reader(input).into_iter::<Input>();
        let mut pooled = Self::default();
        let mut saved = None;
        for (i, input) in inputs.enumerate() {
            match input.map_err(|e| format!("invalid overlap input {}: {e}", i + 1))? {
                Input::Summary(report) if i == 0 => saved = Some(report),
                Input::Documents(part) if saved.is_none() => pooled.merge(part)?,
                _ => return Err("a saved overlap summary must be the only input".into()),
            }
        }
        Ok(saved.unwrap_or(pooled))
    }

    /// Count an in-memory grounded document or reject invalid signal locations and IDs.
    pub fn from_document(doc: &GroundedDocument) -> Result<Self, String> {
        Self::from_documents(std::iter::once(doc))
    }

    /// Pool borrowed grounded documents, including documents that have no signals.
    /// Character offsets and signal IDs are interpreted within each document.
    pub fn from_documents<'a>(
        documents: impl IntoIterator<Item = &'a GroundedDocument>,
    ) -> Result<Self, String> {
        let mut report = Self::default();
        for doc in documents {
            if !doc.is_valid() || !doc.validate_invariants().is_empty() {
                return Err("invalid grounded document signal locations, surfaces or IDs".into());
            }
            report.add_document(doc)?;
        }
        Ok(report)
    }

    /// Set the title shown in charts and retained in portable JSON.
    pub fn set_title(&mut self, title: String) -> Result<(), String> {
        check_label(&title, "overlap report title")?;
        self.title = title;
        Ok(())
    }

    /// Serialize all counts and label names without source text, surfaces or document IDs.
    pub fn to_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map(|json| json + "\n")
            .map_err(|e| e.to_string())
    }

    fn add_document(&mut self, doc: &GroundedDocument) -> Result<(), String> {
        let length = doc.text().chars().count();
        self.add_totals(1, doc.signals().len() as u64, length as u64)?;
        let mut events = Vec::new();
        for (index, signal) in doc.signals().iter().enumerate() {
            let coverage = signal.location.coverage();
            for range in coverage.view().ranges() {
                events.push((range.start, true, index));
                events.push((range.end, false, index));
            }
        }
        // Ends sort before starts at the same position, preserving half-open ranges.
        events.sort_unstable();
        let mut counts = [0; 4];
        let mut shared = vec![false; doc.signals().len()];
        let mut previous = 0;
        let mut depth: usize = 0;
        // Only a newly active singleton can still need marking. Once depth reaches
        // two every member is already known to overlap, even after one leaves.
        let mut unmarked_singleton: Option<usize> = None;
        for (position, starts, index) in events {
            counts[depth.min(3)] += (position - previous) as u64;
            previous = position;
            if starts {
                if depth == 0 {
                    unmarked_singleton = Some(index);
                } else {
                    shared[index] = true;
                    if let Some(other) = unmarked_singleton.take() {
                        shared[other] = true;
                    }
                }
                depth += 1;
            } else {
                depth -= 1;
            }
        }
        counts[0] += (length - previous) as u64;
        self.coverage.add(counts);
        self.documents_with_overlap += u64::from(counts[2] != 0 || counts[3] != 0);
        for (signal, shared) in doc.signals().iter().zip(shared) {
            let label = signal.label.as_str();
            check_label(label, "signal label")?;
            let counts = self.labels.entry(label.to_owned()).or_default();
            counts.signals += 1;
            counts.signals_with_overlap += u64::from(shared);
            self.signals_with_overlap += u64::from(shared);
        }
        Ok(())
    }

    fn add_totals(&mut self, docs: u64, signals: u64, chars: u64) -> Result<(), String> {
        self.documents = self
            .documents
            .checked_add(docs)
            .ok_or("document count exceeds u64")?;
        self.signals = self
            .signals
            .checked_add(signals)
            .ok_or("signal count exceeds u64")?;
        self.characters = self
            .characters
            .checked_add(chars)
            .ok_or("character count exceeds u64")?;
        Ok(())
    }

    fn merge(&mut self, other: Self) -> Result<(), String> {
        // All merged subcounts are bounded by the corresponding totals.
        self.add_totals(other.documents, other.signals, other.characters)?;
        self.documents_with_overlap += other.documents_with_overlap;
        self.signals_with_overlap += other.signals_with_overlap;
        self.coverage.add(other.coverage.counts());
        for (label, counts) in other.labels {
            let row = self.labels.entry(label).or_default();
            row.signals += counts.signals;
            row.signals_with_overlap += counts.signals_with_overlap;
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != OVERLAP_SCHEMA {
            return Err(format!("expected overlap schema {OVERLAP_SCHEMA}"));
        }
        check_label(&self.title, "overlap report title")?;
        let total = self
            .coverage
            .counts()
            .into_iter()
            .try_fold(0u64, u64::checked_add);
        if total != Some(self.characters) {
            return Err("coverage counts must sum to all source characters".into());
        }
        let shared_chars = self.coverage.two + self.coverage.three_or_more;
        if self.documents_with_overlap > self.documents
            || self.signals_with_overlap > self.signals
            || self.documents_with_overlap > self.signals_with_overlap / 2
            || self.documents_with_overlap > shared_chars
            || (shared_chars > 0) != (self.documents_with_overlap > 0)
            || (self.signals_with_overlap > 0) != (shared_chars > 0)
            || (self.coverage.three_or_more > 0 && self.signals_with_overlap < 3)
            || (self.signals > 0) != (self.characters > self.coverage.zero)
            || (self.documents == 0 && self.characters != 0)
        {
            return Err("document, signal and character overlap totals disagree".into());
        }
        let mut signals = 0u64;
        let mut shared = 0u64;
        for (label, counts) in &self.labels {
            check_label(label, "signal label")?;
            if counts.signals == 0 || counts.signals_with_overlap > counts.signals {
                return Err("invalid per-label signal counts".into());
            }
            signals = signals
                .checked_add(counts.signals)
                .ok_or("label counts overflow")?;
            shared = shared
                .checked_add(counts.signals_with_overlap)
                .ok_or("label counts overflow")?;
        }
        if signals != self.signals || shared != self.signals_with_overlap {
            return Err("label counts must sum to the corresponding signal totals".into());
        }
        Ok(())
    }

    fn labels_by_count(&self) -> Vec<(&String, &LabelCounts)> {
        let mut rows: Vec<_> = self.labels.iter().collect();
        rows.sort_unstable_by(|(a, x), (b, y)| {
            y.signals_with_overlap
                .cmp(&x.signals_with_overlap)
                .then_with(|| y.signals.cmp(&x.signals))
                .then_with(|| a.cmp(b))
        });
        rows
    }

    fn labels_by_share(&self) -> Vec<(&String, &LabelCounts)> {
        let mut rows: Vec<_> = self.labels.iter().collect();
        rows.sort_unstable_by(|(a, x), (b, y)| {
            // Compare ratios exactly, including large u64 denominators and ties.
            let left = u128::from(x.signals_with_overlap) * u128::from(y.signals);
            let right = u128::from(y.signals_with_overlap) * u128::from(x.signals);
            right
                .cmp(&left)
                .then_with(|| y.signals_with_overlap.cmp(&x.signals_with_overlap))
                .then_with(|| a.cmp(b))
        });
        rows
    }
}

enum Input {
    Documents(Overlap),
    Summary(Overlap),
}

impl<'de> Deserialize<'de> for Input {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputVisitor;
        impl<'de> Visitor<'de> for InputVisitor {
            type Value = Input;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a GroundedDocument, a document array or an overlap summary")
            }

            fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Input, M::Error> {
                let value = serde_json::Value::deserialize(MapAccessDeserializer::new(map))?;
                if value.get("schema").is_some() {
                    let report: Overlap =
                        serde_json::from_value(value).map_err(de::Error::custom)?;
                    report.validate().map_err(de::Error::custom)?;
                    return Ok(Input::Summary(report));
                }
                let doc: GroundedDocument =
                    serde_json::from_value(value).map_err(de::Error::custom)?;
                let mut report = Overlap::default();
                report.add_document(&doc).map_err(de::Error::custom)?;
                Ok(Input::Documents(report))
            }

            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Input, S::Error> {
                let mut report = Overlap::default();
                while let Some(doc) = seq.next_element::<GroundedDocument>()? {
                    report.add_document(&doc).map_err(de::Error::custom)?;
                }
                Ok(Input::Documents(report))
            }
        }
        deserializer.deserialize_any(InputVisitor)
    }
}

const DEPTH_LABELS: [&str; 4] = ["0 signals", "1 signal", "2 signals", "3+ signals"];

fn percent(count: u64, total: u64) -> String {
    if total == 0 {
        return "—".into();
    }
    let value = 100.0 * count as f64 / total as f64;
    if count > 0 && value < 0.005 {
        "under 0.01%".into()
    } else if count < total && value >= 99.95 {
        "over 99.9%".into()
    } else if count > 0 && value < 10.0 {
        format!("{value:.2}%")
    } else {
        format!("{value:.1}%")
    }
}

impl Overlap {
    /// Render an accessible SVG with exact depth counts and the most exposed labels.
    pub fn render_svg(&self) -> Result<String, String> {
        self.validate()?;
        let mut body = String::new();
        // Reserve room for long aggregate counts while keeping typical reports narrow.
        let count_width = (integer(self.characters.max(self.signals)).len() as i32 * 11).max(84);
        let ratio_width = (integer(self.signals).len() as i32 * 22 + 25).max(170);
        let width = (420 + count_width).max(340 + ratio_width).max(570);
        let end = width - 26;
        let title = wrap(&self.title, ((width - 52) / 13) as usize);
        lines(&mut body, &title, 26, 39, 30, "title");
        let mut y = 39 + (title.len() as i32 - 1) * 30 + 37;
        let shared_chars = self.coverage.two + self.coverage.three_or_more;
        writeln!(
            body,
            r#"<text x="26" y="{y}" class="heading">Where signals share a character</text>"#
        )
        .unwrap();
        for (shared, total, noun) in [
            (self.documents_with_overlap, self.documents, "documents"),
            (self.signals_with_overlap, self.signals, "signals"),
            (shared_chars, self.characters, "source characters"),
        ] {
            let value = format!(
                "{} / {} {noun} ({})",
                integer(shared),
                integer(total),
                percent(shared, total)
            );
            let wrapped = wrap(&value, ((width - 52) / 9) as usize);
            y += 23;
            lines(&mut body, &wrapped, 26, y, 22, "value");
            y += (wrapped.len() as i32 - 1) * 22;
        }

        y += 41;
        writeln!(
            body,
            r#"<text x="26" y="{y}" class="heading">Coverage depth by Unicode character</text>"#
        )
        .unwrap();
        y += 31;
        let count_x = width - 282;
        let all_x = width - 159;
        writeln!(body, r#"<text x="26" y="{y}" class="col">Depth</text><text x="{count_x}" y="{y}" class="col num">Characters</text><text x="{all_x}" y="{y}" class="col num">Of all text</text><text x="{end}" y="{y}" class="col num">Of shared text</text>"#).unwrap();
        for (depth, (label, count)) in DEPTH_LABELS.iter().zip(self.coverage.counts()).enumerate() {
            y += 26;
            let within_shared = if depth >= 2 {
                percent(count, shared_chars)
            } else {
                "—".into()
            };
            writeln!(body, r#"<text x="26" y="{y}">{label}</text><text x="{count_x}" y="{y}" class="num">{}</text><text x="{all_x}" y="{y}" class="num">{}</text><text x="{end}" y="{y}" class="num">{within_shared}</text>"#, integer(count), percent(count, self.characters)).unwrap();
        }
        if shared_chars > 0 {
            y += 30;
            writeln!(body, r#"<text x="26" y="{y}" class="muted">Within positions shared by two or more signals</text>"#).unwrap();
            y += 10;
            let bar = width - 52;
            let two = bar as f64 * self.coverage.two as f64 / shared_chars as f64;
            writeln!(body, r##"<rect x="26" y="{y}" width="{bar}" height="12" fill="#805670"/><rect x="26" y="{y}" width="{two:.3}" height="12" fill="#216f72"/>"##).unwrap();
            y += 33;
            writeln!(body, r##"<circle cx="31" cy="{}" r="5" fill="#216f72"/><text x="43" y="{y}" class="muted">2 signals</text><circle cx="159" cy="{}" r="5" fill="#805670"/><text x="171" y="{y}" class="muted">3+ signals</text>"##, y-5, y-5).unwrap();
        }
        y += 40;
        writeln!(
            body,
            r#"<text x="26" y="{y}" class="heading">Overlap by signal label</text>"#
        )
        .unwrap();
        let by_share = self.labels_by_share();
        if by_share.is_empty() {
            y += 25;
            writeln!(
                body,
                r#"<text x="26" y="{y}" class="muted">No signals in these documents.</text>"#
            )
            .unwrap();
        } else {
            if self.signals_with_overlap > 0 {
                let by_count = self.labels_by_count();
                let (label, counts) = by_count[0];
                let note = format!(
                    "Largest affected count · {label} · {} of {} signals ({})",
                    integer(counts.signals_with_overlap),
                    integer(counts.signals),
                    percent(counts.signals_with_overlap, counts.signals)
                );
                let wrapped = wrap(&note, ((width - 52) / 8) as usize);
                y += 24;
                lines(&mut body, &wrapped, 26, y, 20, "muted");
                y += (wrapped.len() as i32 - 1) * 20;
            }
            y += 24;
            let note = if by_share.len() > 8 {
                format!(
                    "Top 8 of {} labels by affected share. Full rankings in HTML.",
                    by_share.len()
                )
            } else {
                "Ranked by the share of each label’s signals affected.".into()
            };
            writeln!(body, r#"<text x="26" y="{y}" class="muted">{note}</text>"#).unwrap();
            y += 31;
            let ratio_x = width - 141;
            writeln!(body, r#"<text x="26" y="{y}" class="col">Label</text><text x="{ratio_x}" y="{y}" class="col num">Affected / all</text><text x="{end}" y="{y}" class="col num">Of label</text>"#).unwrap();
            for (label, counts) in by_share.into_iter().take(8) {
                y += 26;
                let names = wrap(label, ((ratio_x - ratio_width - 48) / 9).max(8) as usize);
                lines(&mut body, &names, 26, y, 20, "");
                writeln!(body, r#"<text x="{ratio_x}" y="{y}" class="num">{} / {}</text><text x="{end}" y="{y}" class="num">{}</text>"#, integer(counts.signals_with_overlap), integer(counts.signals), percent(counts.signals_with_overlap, counts.signals)).unwrap();
                y += (names.len() as i32 - 1) * 20;
            }
        }
        let height = y + 28;
        Ok(format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="title desc">
<title id="title">{}</title><desc id="desc">{} of {} signals share at least one character. Of {} total characters, {} are covered by at least two signals. The depth table and bar report the exact count and share at each overlap depth.</desc>
<style>text{{font:17px "DejaVu Sans",sans-serif;fill:#183541}}.title{{font-size:25px;font-weight:700}}.heading{{font-size:20px;font-weight:700}}.muted,.col{{font-size:15px;fill:#456371}}.col{{font-weight:700}}.value,.num{{font-variant-numeric:tabular-nums}}.num{{text-anchor:end}}</style>
<rect width="{width}" height="{height}" fill="white"/>{body}</svg>
"##,
            escape(&self.title),
            integer(self.signals_with_overlap),
            integer(self.signals),
            integer(self.characters),
            integer(shared_chars)
        ))
    }

    /// Render the chart and complete count tables as a standalone HTML document.
    pub fn render_html(&self) -> Result<String, String> {
        let svg = self.render_svg()?;
        let shared_chars = self.coverage.two + self.coverage.three_or_more;
        let mut table = format!(
            r#"<section class="overlap-report"><h1 class="mobile-intro">{}</h1>"#,
            escape(&self.title)
        );
        // Column headers serve the desktop table and label each numeric value
        // when the same row reflows into a card on a narrow screen.
        fn table_header(table: &mut String, caption: &str, heads: [&str; 4]) {
            write!(table, "<table><caption>{caption}</caption><thead><tr>").unwrap();
            for head in heads {
                write!(table, r#"<th scope="col">{head}</th>"#).unwrap();
            }
            table.push_str("</tr></thead><tbody>");
        }
        fn table_row(table: &mut String, label: &str, heads: [&str; 4], values: [String; 3]) {
            write!(table, r#"<tr><th scope="row">{}</th>"#, escape(label)).unwrap();
            for (head, value) in heads[1..].iter().zip(values) {
                write!(table, r#"<td data-label="{head}">{value}</td>"#).unwrap();
            }
            table.push_str("</tr>");
        }

        let population = ["Population", "Sharing", "All", "Share"];
        table_header(&mut table, "Population with shared positions", population);
        for (label, shared, total) in [
            ("Documents", self.documents_with_overlap, self.documents),
            ("Signals", self.signals_with_overlap, self.signals),
            ("Source characters", shared_chars, self.characters),
        ] {
            table_row(
                &mut table,
                label,
                population,
                [integer(shared), integer(total), percent(shared, total)],
            );
        }
        table.push_str("</tbody></table>");

        let depths = ["Depth", "Characters", "Of all text", "Of shared text"];
        table_header(
            &mut table,
            &format!(
                "Coverage of all {} Unicode scalar positions",
                integer(self.characters)
            ),
            depths,
        );
        for (depth, (label, count)) in DEPTH_LABELS.iter().zip(self.coverage.counts()).enumerate() {
            let within_shared = if depth >= 2 {
                percent(count, shared_chars)
            } else {
                "—".into()
            };
            table_row(
                &mut table,
                label,
                depths,
                [
                    integer(count),
                    percent(count, self.characters),
                    within_shared,
                ],
            );
        }
        table.push_str("</tbody></table>");
        if !self.labels.is_empty() {
            let ranked = ["Label", "Sharing", "All signals", "Share of label"];
            for (i, (caption, rows)) in [
                ("Labels by affected share", self.labels_by_share()),
                ("Labels by affected count", self.labels_by_count()),
            ]
            .into_iter()
            .enumerate()
            {
                if i == 1 {
                    table.push_str("<details><summary>Rank all labels by affected count</summary>");
                }
                table_header(&mut table, caption, ranked);
                for (label, counts) in rows {
                    table_row(
                        &mut table,
                        label,
                        ranked,
                        [
                            integer(counts.signals_with_overlap),
                            integer(counts.signals),
                            percent(counts.signals_with_overlap, counts.signals),
                        ],
                    );
                }
                table.push_str("</tbody></table>");
                if i == 1 {
                    table.push_str("</details>");
                }
            }
        }
        table.push_str("</section>");
        Ok(html_document(&self.title, &svg, &table))
    }
}

#[cfg(test)]
mod tests {
    use super::percent;

    #[test]
    fn distinguish_small_covered_groups_from_zero_or_missing_denominators() {
        assert_eq!(percent(855, 3_103_238), "0.03%");
        assert_eq!(percent(1, 10_000_000), "under 0.01%");
        assert_eq!(percent(999_999, 1_000_000), "over 99.9%");
        assert_eq!(percent(1, 1), "100.0%");
        assert_eq!(percent(0, 3_103_238), "0.0%");
        assert_eq!(percent(0, 0), "—");
    }
}
