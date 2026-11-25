//! Grounded documents retain validated source maps and legacy HTML projections.
use super::html_source::{HtmlFragmentKind, HtmlSource};
use super::GroundedDocument;
use std::ops::Range;
use textprep_source_map::checked_char_range;
pub use textprep_source_map::{SourceKind, SourceMap, SourceRun};

impl HtmlSource {
    pub(super) fn check_map_identity(&self, map: &SourceMap) -> Result<(), String> {
        if map.source_sha256.as_ref() != Some(&self.source_sha256)
            || map.source_byte_length != Some(self.source_byte_length)
            || map.analyzed_sha256 != self.rendered_sha256
        {
            return Err("HTML and generic source maps identify different source editions".into());
        }
        Ok(())
    }
}

impl GroundedDocument {
    /// Attach a map only when its analyzed coordinates and digest match this document.
    pub fn record_source_map(&mut self, map: SourceMap) -> Result<(), String> {
        map.validate_analyzed(self.text())?;
        if let Some(html) = self.html_source() {
            html.check_map_identity(&map)?;
        }
        self.source_map = Some(Box::new(map));
        Ok(())
    }

    /// Map for the entire analyzed source, when supplied by the producer.
    pub fn source_map(&self) -> Option<&SourceMap> {
        self.source_map.as_deref()
    }

    /// Look up original fragments for a document character span. A legacy HTML
    /// document may cover only individual stored spans. Missing evidence returns None.
    pub fn project_source_span(
        &self,
        chars: Range<usize>,
    ) -> Result<Option<Vec<SourceRun>>, String> {
        let bytes = checked_char_range(self.text(), chars.clone())?;
        // Both the text and attached map are immutable after validation.
        if let Some(map) = self.source_map() {
            return Ok(Some(map.overlaps(bytes).collect()));
        }
        let Some(html) = self.html_source() else {
            return Ok(None);
        };
        let Some(span) = html
            .spans
            .iter()
            .find(|s| s.rendered_chars.start <= chars.start && chars.end <= s.rendered_chars.end)
        else {
            return Ok(None);
        };
        let mut runs = Vec::new();
        for frag in &span.fragments {
            let kind = match frag.kind {
                HtmlFragmentKind::Direct => SourceKind::Direct,
                HtmlFragmentKind::EntityDecoded => SourceKind::Transformed,
                HtmlFragmentKind::WhitespaceCollapsed | HtmlFragmentKind::Unknown => {
                    SourceKind::Approximate
                }
                HtmlFragmentKind::Synthetic => SourceKind::Synthetic,
                HtmlFragmentKind::Unmapped => SourceKind::Unmapped,
            };
            let run = SourceRun::new(frag.rendered_bytes.clone(), frag.source_bytes.clone(), kind);
            runs.extend(run.intersection(bytes.clone()));
        }
        Ok(Some(runs))
    }
}

#[cfg(test)]
mod tests {
    use super::super::html_source::{source_sha256, HtmlFragment, HtmlRenderer, HtmlSpan};
    use super::*;

    #[test]
    fn deserialization_rejects_stale_maps_and_serves_existing_html_fragments() {
        let mut doc = GroundedDocument::new("doc", "éxy");
        let raw = "_éxy";
        let map = SourceMap::new(
            raw,
            doc.text(),
            vec![SourceRun::new(0..4, Some(1..5), SourceKind::Direct)],
        )
        .unwrap();
        doc.record_source_map(map.clone()).unwrap();
        let json = serde_json::to_value(&doc).unwrap();
        let restored: GroundedDocument = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(
            restored.project_source_span(0..1).unwrap().unwrap()[0].source_bytes,
            Some(1..3)
        );
        let mut wrong = json.clone();
        wrong["text"] = "éxz".into();
        assert!(serde_json::from_value::<GroundedDocument>(wrong).is_err());
        let mut wrong = json.clone();
        wrong["source_map"]["runs"][0]["source_ranges"] =
            serde_json::json!([{"start":0,"end":1},{"start":3,"end":4}]);
        assert!(serde_json::from_value::<GroundedDocument>(wrong).is_err());
        let mut old = json;
        old.as_object_mut().unwrap().remove("source_map");
        let mut legacy: GroundedDocument = serde_json::from_value(old).unwrap();
        assert!(legacy.project_source_span(0..2).unwrap().is_none());
        let html = HtmlSource {
            renderer: HtmlRenderer::DeformatStrip,
            source_sha256: source_sha256(raw),
            source_byte_length: raw.len(),
            rendered_sha256: source_sha256(doc.text()),
            spans: vec![HtmlSpan {
                rendered_chars: 0..2,
                fragments: vec![HtmlFragment {
                    kind: HtmlFragmentKind::Direct,
                    rendered_bytes: 0..3,
                    source_bytes: Some(1..4),
                }],
            }],
        };
        legacy.record_html_source(html.clone()).unwrap();
        assert_eq!(
            legacy.project_source_span(1..2).unwrap().unwrap()[0].source_bytes,
            Some(3..4)
        );
        assert!(legacy.project_source_span(1..3).unwrap().is_none());
        assert!(legacy
            .record_source_map(SourceMap::identity("éxy"))
            .is_err());
        legacy.record_source_map(map).unwrap();
        let mut wrong_html = html;
        wrong_html.source_sha256 = source_sha256("other");
        assert!(legacy.record_html_source(wrong_html).is_err());
        let mut wrong = serde_json::to_value(&legacy).unwrap();
        wrong["html_source"]["source_sha256"] = source_sha256("other").into();
        assert!(serde_json::from_value::<GroundedDocument>(wrong).is_err());
    }
}
