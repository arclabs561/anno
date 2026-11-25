use std::collections::BTreeSet;
use std::fs;

use anno::core::grounded::html_source::{
    HtmlFragment, HtmlFragmentKind, HtmlRenderer, HtmlSource, HtmlSpan,
};
use anno::core::grounded::{GroundedDocument, Location};
use anno::offset::SpanConverter;
use deformat::html::{strip_to_text_with_spans, SpanKind, SpanMap};

use crate::cli::utils::{is_html_file, source_sha256};
use crate::cli::CliError;

pub(super) struct MappedHtml {
    source: String,
    pub(super) text: String,
    map: SpanMap,
}

impl MappedHtml {
    pub(super) fn read(path: &str) -> Result<Self, CliError> {
        let source = fs::read_to_string(path)
            .map_err(|e| format!("could not read UTF-8 HTML file '{path}': {e}"))?;
        if !is_html_file(path, &source) {
            return Err(CliError::from("--html-source-spans requires an HTML file"));
        }
        let (text, map) = strip_to_text_with_spans(&source);
        Ok(Self { source, text, map })
    }

    /// Source map for all rendered bytes, including regions without an extracted signal.
    pub(super) fn source_map(&self) -> Result<anno::SourceMap, CliError> {
        self.map
            .to_source_map(&self.source, &self.text)
            .map_err(CliError::from)
    }

    pub(super) fn record(self, doc: &mut GroundedDocument) -> Result<(), CliError> {
        // A map is valid only for the exact bytes emitted by its own renderer.
        if self.text != doc.text() {
            return Err(CliError::from("HTML rendering changed before projection"));
        }
        let mut ranges = BTreeSet::new();
        for signal in doc.signals() {
            match &signal.location {
                Location::Text { start, end } => {
                    ranges.insert((*start, *end));
                }
                Location::Discontinuous { segments } => ranges.extend(segments.iter().copied()),
            }
        }
        let offsets = SpanConverter::new(&self.text);
        let runs: Vec<_> = self.map.iter().collect();
        let mut spans = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            let start_byte = offsets.char_to_byte(start);
            let end_byte = offsets.char_to_byte(end);
            let first = runs.partition_point(|run| run.output_end <= start_byte);
            let mut fragments = Vec::new();
            let mut cursor = start_byte;
            for run in runs[first..]
                .iter()
                .take_while(|run| run.output_start < end_byte)
            {
                let lo = start_byte.max(run.output_start);
                let hi = end_byte.min(run.output_end);
                if cursor < lo {
                    fragments.push(unmapped(cursor..lo));
                }
                let kind = match run.kind {
                    SpanKind::Direct => HtmlFragmentKind::Direct,
                    SpanKind::EntityDecoded => HtmlFragmentKind::EntityDecoded,
                    SpanKind::WhitespaceCollapsed => HtmlFragmentKind::WhitespaceCollapsed,
                    SpanKind::Synthetic => HtmlFragmentKind::Synthetic,
                    _ => HtmlFragmentKind::Unknown,
                };
                let source_bytes = if kind == HtmlFragmentKind::Direct {
                    let a = run.source_start + (lo - run.output_start);
                    let b = run.source_start + (hi - run.output_start);
                    let source = self
                        .source
                        .get(a..b)
                        .ok_or("invalid direct HTML source range")?;
                    if self.text.get(lo..hi) != Some(source) {
                        return Err(CliError::from(
                            "direct HTML source fragment differs from rendered text",
                        ));
                    }
                    a..b
                } else {
                    run.source_start..run.source_end
                };
                fragments.push(HtmlFragment {
                    kind,
                    rendered_bytes: lo..hi,
                    source_bytes: Some(source_bytes),
                });
                cursor = hi;
            }
            if cursor < end_byte {
                fragments.push(unmapped(cursor..end_byte));
            }
            spans.push(HtmlSpan {
                rendered_chars: start..end,
                fragments,
            });
        }
        doc.record_html_source(HtmlSource {
            renderer: HtmlRenderer::DeformatStrip,
            source_sha256: source_sha256(&self.source),
            source_byte_length: self.source.len(),
            rendered_sha256: source_sha256(&self.text),
            spans,
        })
        .map_err(CliError::from)
    }
}

fn unmapped(rendered_bytes: std::ops::Range<usize>) -> HtmlFragment {
    HtmlFragment {
        kind: HtmlFragmentKind::Unmapped,
        rendered_bytes,
        source_bytes: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno::Signal;

    #[test]
    fn gaps_in_the_renderer_map_have_no_claimed_source_bytes() {
        let source = "<p>before <img alt='a β' src='photo.png'> after</p>".to_owned();
        let (text, map) = strip_to_text_with_spans(&source);
        assert_eq!(text, "before a β after");
        let mut doc = GroundedDocument::new("gap", &text);
        doc.add_signal(Signal::new(
            0,
            Location::text(0, text.chars().count()),
            &text,
            "test",
            1.0,
        ));
        MappedHtml { source, text, map }.record(&mut doc).unwrap();
        let fragments = &doc.html_source().unwrap().spans[0].fragments;
        let hole = fragments
            .iter()
            .find(|fragment| fragment.kind == HtmlFragmentKind::Unmapped)
            .unwrap();
        assert_eq!(hole.rendered_bytes, 11..12);
        assert_eq!(hole.source_bytes, None);
    }

    #[test]
    fn collapsed_whitespace_and_discontinuous_locations_retain_their_origins() {
        let original = "<p>A   B followed by <b>C</b></p>".to_owned();
        let (text, map) = strip_to_text_with_spans(&original);
        let mut doc = GroundedDocument::new("test", &text);
        doc.add_signal(Signal::new(0, Location::text(0, 3), "A B", "ORG", 1.0));
        doc.add_signal(Signal::new(
            1,
            Location::Discontinuous {
                segments: vec![(0, 1), (16, 17)],
            },
            "A C",
            "ORG",
            1.0,
        ));
        MappedHtml {
            source: original.clone(),
            text,
            map,
        }
        .record(&mut doc)
        .unwrap();
        let spans = &doc.html_source().unwrap().spans;
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].rendered_chars, 0..3);
        assert_eq!(
            spans[1].fragments[0].kind,
            HtmlFragmentKind::WhitespaceCollapsed
        );
        assert_eq!(spans[2].rendered_chars, 16..17);
        assert_eq!(
            &original[spans[2].fragments[0].source_bytes.clone().unwrap()],
            "C"
        );
    }
}
