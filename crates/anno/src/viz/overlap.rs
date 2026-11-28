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

    fn sorted_labels(&self) -> Vec<(&String, &LabelCounts)> {
        let mut rows: Vec<_> = self.labels.iter().collect();
        rows.sort_unstable_by(|(a, x), (b, y)| {
            y.signals_with_overlap
                .cmp(&x.signals_with_overlap)
                .then_with(|| y.signals.cmp(&x.signals))
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
    } else if count > 0 && value < 0.1 {
        format!("{value:.2}%")
    } else {
        format!("{value:.1}%")
    }
}

impl Overlap {
    /// Render an accessible standalone SVG. The label panel shows up to eight rows.
    pub fn render_svg(&self) -> Result<String, String> {
        self.validate()?;
        let mut body = String::new();
        let title = wrap(&self.title, 42);
        lines(&mut body, &title, 34, 49, 34, "title");
        let mut y = 49 + (title.len() as i32 - 1) * 34;
        let documents = integer(self.documents);
        let signals = integer(self.signals);
        let characters = integer(self.characters);
        let shared_docs = integer(self.documents_with_overlap);
        let shared_signals = integer(self.signals_with_overlap);
        let doc_pct = percent(self.documents_with_overlap, self.documents);
        let signal_pct = percent(self.signals_with_overlap, self.signals);
        y += 40;
        writeln!(
            body,
            r#"<text x="34" y="{y}" class="heading">Signals that share a source character</text>"#
        )
        .unwrap();
        y += 27;
        writeln!(body, r#"<text x="34" y="{y}" class="value">{shared_docs} / {documents} documents ({doc_pct})</text><text x="420" y="{y}" class="value">{shared_signals} / {signals} signals ({signal_pct})</text>"#).unwrap();
        y += 50;
        writeln!(
            body,
            r#"<text x="34" y="{y}" class="heading">How many signals cover each character?</text>"#
        )
        .unwrap();
        y += 24;
        writeln!(body, r#"<text x="34" y="{y}" class="muted">Share of all {characters} Unicode scalar positions across {documents} documents.</text>"#).unwrap();
        let colors = ["#8299a3", "#327c93", "#b86232", "#773e71"];
        for ((label, count), color) in DEPTH_LABELS.iter().zip(self.coverage.counts()).zip(colors) {
            y += 32;
            let bar_y = y - 12;
            let width = if self.characters == 0 {
                0.0
            } else {
                400.0 * count as f64 / self.characters as f64
            };
            writeln!(body, r##"<text x="34" y="{y}">{label}</text><rect x="160" y="{bar_y}" width="400" height="12" rx="2" fill="#edf1f2"/><rect x="160" y="{bar_y}" width="{width:.3}" height="12" rx="2" fill="{color}"/><text x="766" y="{y}" text-anchor="end" class="value">{} ({})</text>"##, integer(count), percent(count, self.characters)).unwrap();
        }
        y += 52;
        writeln!(
            body,
            r#"<text x="34" y="{y}" class="heading">Overlap by signal label</text>"#
        )
        .unwrap();
        y += 25;
        let sorted = self.sorted_labels();
        if sorted.is_empty() {
            writeln!(
                body,
                r#"<text x="34" y="{y}" class="muted">No signals in these documents.</text>"#
            )
            .unwrap();
        } else {
            let note = if sorted.len() > 8 {
                format!("Top 8 of {} labels by signals sharing a character. JSON and HTML include every label.", sorted.len())
            } else {
                "Ranked by the number of signals sharing a character with another signal.".into()
            };
            writeln!(body, r#"<text x="34" y="{y}" class="muted">{note}</text>"#).unwrap();
            y += 30;
            writeln!(body, r#"<text x="34" y="{y}" class="muted">Label</text><text x="556" y="{y}" text-anchor="end" class="muted">Sharing / all signals</text><text x="766" y="{y}" text-anchor="end" class="muted">Share of label</text>"#).unwrap();
            for (label, counts) in sorted.into_iter().take(8) {
                y += 26;
                let names = wrap(label, 38);
                lines(&mut body, &names, 34, y, 20, "row");
                writeln!(body, r#"<text x="556" y="{y}" text-anchor="end" class="value">{} / {}</text><text x="766" y="{y}" text-anchor="end" class="value">{}</text>"#, integer(counts.signals_with_overlap), integer(counts.signals), percent(counts.signals_with_overlap, counts.signals)).unwrap();
                y += (names.len() as i32 - 1) * 20;
            }
        }
        let height = y + 32;
        Ok(format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 {height}" width="800" height="{height}" role="img" aria-labelledby="title desc">
<title id="title">{}</title><desc id="desc">{shared_docs} of {documents} documents and {shared_signals} of {signals} signals have overlapping annotations. Character coverage includes all {characters} Unicode scalar positions.</desc>
<style>text{{font:16px "DejaVu Sans",sans-serif;fill:#183541}}.title{{font-size:29px;font-weight:700}}.heading{{font-size:20px;font-weight:700}}.muted{{font-size:14px;fill:#456371}}.value{{font-variant-numeric:tabular-nums}}.row{{font-size:16px}}</style>
<rect width="800" height="{height}" fill="white"/>{body}</svg>
"##,
            escape(&self.title)
        ))
    }

    /// Render the chart and complete count tables as a standalone HTML document.
    pub fn render_html(&self) -> Result<String, String> {
        let svg = self.render_svg()?;
        let mut table = String::from("<table><caption>Signals sharing a source character</caption><thead><tr><th>Population</th><th>Sharing</th><th>All</th><th>Share</th></tr></thead><tbody>");
        for (label, shared, total) in [
            ("Documents", self.documents_with_overlap, self.documents),
            ("Signals", self.signals_with_overlap, self.signals),
        ] {
            write!(
                table,
                "<tr><th>{label}</th><td>{}</td><td>{}</td><td>{}</td></tr>",
                integer(shared),
                integer(total),
                percent(shared, total)
            )
            .unwrap();
        }
        write!(table, "</tbody></table><table><caption>Coverage of all {} Unicode scalar positions</caption><thead><tr><th>Depth</th><th>Characters</th><th>Share</th></tr></thead><tbody>", integer(self.characters)).unwrap();
        for (label, count) in DEPTH_LABELS.iter().zip(self.coverage.counts()) {
            write!(
                table,
                "<tr><th>{label}</th><td>{}</td><td>{}</td></tr>",
                integer(count),
                percent(count, self.characters)
            )
            .unwrap();
        }
        table.push_str("</tbody></table>");
        if !self.labels.is_empty() {
            table.push_str("<table><caption>All labels ranked by overlapping signals</caption><thead><tr><th>Label</th><th>Sharing</th><th>All signals</th><th>Share of label</th></tr></thead><tbody>");
            for (label, counts) in self.sorted_labels() {
                write!(
                    table,
                    "<tr><th>{}</th><td>{}</td><td>{}</td><td>{}</td></tr>",
                    escape(label),
                    integer(counts.signals_with_overlap),
                    integer(counts.signals),
                    percent(counts.signals_with_overlap, counts.signals)
                )
                .unwrap();
            }
            table.push_str("</tbody></table>");
        }
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
        assert_eq!(percent(0, 3_103_238), "0.0%");
        assert_eq!(percent(0, 0), "—");
    }
}
