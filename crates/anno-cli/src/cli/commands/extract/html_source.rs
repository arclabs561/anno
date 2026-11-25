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
    fn image_alt_separators_project_to_the_image_tag() {
        let source = "L<img alt='β'>R";
        let (text, map) = strip_to_text_with_spans(source);
        assert_eq!(text, "L β R");
        let mut doc = GroundedDocument::new("image", &text);
        for start in [1, 3] {
            doc.add_signal(Signal::new(
                0,
                Location::text(start, start + 1),
                " ",
                "test",
                1.0,
            ));
        }
        let html = MappedHtml {
            source: source.to_owned(),
            text,
            map,
        };
        let byte_map = html.source_map().unwrap();
        html.record(&mut doc).unwrap();
        let spans = &doc.html_source().unwrap().spans;
        assert_eq!(spans.len(), 2);
        for span in spans {
            let [fragment] = span.fragments.as_slice() else {
                panic!("expected the separator's image origin")
            };
            assert_eq!(fragment.kind, HtmlFragmentKind::Synthetic);
            let range = fragment.source_bytes.as_ref().unwrap();
            assert_eq!(&source[range.clone()], "<img alt='β'>");
            let runs = byte_map
                .project_chars(doc.text(), span.rendered_chars.clone())
                .unwrap();
            assert_eq!(runs.len(), 1);
            assert_eq!(runs[0].kind, anno::SourceKind::Synthetic);
            assert_eq!(runs[0].source_bytes.as_ref(), Some(range));
        }
    }

    #[test]
    fn incomplete_renderer_map_leaves_uncovered_output_unmapped() {
        let source = "<p>ok</p>";
        let (mut text, map) = strip_to_text_with_spans(source);
        // Model a renderer that supplies the prefix's origin but no run for its suffix.
        text.push('β');
        let mut doc = GroundedDocument::new("gap", &text);
        doc.add_signal(Signal::new(0, Location::text(0, 3), &text, "test", 1.0));
        let html = MappedHtml {
            source: source.to_owned(),
            text,
            map,
        };
        let byte_map = html.source_map().unwrap();
        html.record(&mut doc).unwrap();
        let fragments = &doc.html_source().unwrap().spans[0].fragments;
        assert_eq!(fragments.len(), 2);
        assert_eq!(fragments[0].kind, HtmlFragmentKind::Direct);
        assert_eq!(fragments[0].source_bytes, Some(3..5));
        assert_eq!(fragments[1].kind, HtmlFragmentKind::Unmapped);
        assert_eq!(fragments[1].rendered_bytes, 2..4);
        assert_eq!(fragments[1].source_bytes, None);
        let gap = byte_map.project_chars(doc.text(), 2..3).unwrap();
        assert_eq!(gap.len(), 1);
        assert_eq!(gap[0].kind, anno::SourceKind::Unmapped);
        assert_eq!(gap[0].source_bytes, None);
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
