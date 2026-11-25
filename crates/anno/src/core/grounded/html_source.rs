//! Provenance for character spans extracted from rendered UTF-8 HTML.

use crate::offset::SpanConverter;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ops::Range;

/// Renderer used for both the extracted text and its source map.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HtmlRenderer {
    /// deformat's mapped HTML strip renderer.
    DeformatStrip,
}

/// How a rendered fragment relates to its originating HTML bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HtmlFragmentKind {
    /// Verbatim UTF-8 bytes at the indicated source range.
    Direct,
    /// The source range holds the entire HTML character reference.
    EntityDecoded,
    /// Origin of normalized whitespace. The source range is approximate.
    WhitespaceCollapsed,
    /// Originating markup of inserted text or a separator. The source range is approximate.
    Synthetic,
    /// Unrecognized renderer transformation with an approximate source range.
    Unknown,
    /// The renderer supplied no source range for these rendered bytes.
    Unmapped,
}

/// A single projected run. All ranges are half-open.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlFragment {
    /// How to interpret the source range.
    pub kind: HtmlFragmentKind,
    /// UTF-8 bytes in the rendered document.
    pub rendered_bytes: Range<usize>,
    /// UTF-8 bytes in the original file, or `None` for an unmapped fragment.
    pub source_bytes: Option<Range<usize>>,
}

/// Original source fragments for one contiguous rendered span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlSpan {
    /// Half-open Unicode scalar offsets in the grounded document text.
    pub rendered_chars: Range<usize>,
    /// Ordered fragments covering the span in rendered order.
    pub fragments: Vec<HtmlFragment>,
}

/// Source identities and the projected spans present when the extraction ran.
/// Spans are keyed by their rendered coordinates and may be shared by signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlSource {
    /// Renderer that produced the grounded document text.
    pub renderer: HtmlRenderer,
    /// SHA-256 of the original UTF-8 HTML file, in lowercase hexadecimal.
    pub source_sha256: String,
    /// Length of the original file in bytes.
    pub source_byte_length: usize,
    /// SHA-256 of the UTF-8 encoding of the grounded document text.
    pub rendered_sha256: String,
    /// Source projections for each distinct extracted span (or discontinuous component).
    pub spans: Vec<HtmlSpan>,
}

/// SHA-256 of the exact UTF-8 bytes of a source or rendered document.
#[must_use]
pub fn source_sha256(text: &str) -> String {
    bytes_sha256(text.as_bytes())
}

pub(super) fn bytes_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl HtmlSource {
    /// Check a persisted renderer projection against both exact document editions.
    pub fn verify(&self, original: &str, rendered: &str) -> Result<(), String> {
        self.validate(rendered)?;
        if original.len() != self.source_byte_length
            || source_sha256(original) != self.source_sha256
        {
            return Err("original HTML differs from the saved source edition".into());
        }
        for fragment in self.spans.iter().flat_map(|span| &span.fragments) {
            if let Some(range) = &fragment.source_bytes {
                let source = original
                    .get(range.clone())
                    .ok_or("original HTML fragment splits a UTF-8 scalar")?;
                if fragment.kind == HtmlFragmentKind::Direct
                    && rendered.get(fragment.rendered_bytes.clone()) != Some(source)
                {
                    return Err("direct HTML fragment differs from its source bytes".into());
                }
            }
        }
        Ok(())
    }

    pub(super) fn validate(&self, text: &str) -> Result<(), String> {
        if self.source_sha256.len() != 64
            || !self
                .source_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("invalid HTML source SHA-256".into());
        }
        if self.rendered_sha256 != source_sha256(text) {
            return Err("HTML source map does not match rendered text SHA-256".into());
        }
        let offsets = SpanConverter::new(text);
        let char_count = text.chars().count();
        let mut previous = None;
        for span in &self.spans {
            let chars = &span.rendered_chars;
            if chars.is_empty()
                || chars.end > char_count
                || previous.is_some_and(|key| key >= (chars.start, chars.end))
            {
                return Err("invalid or unsorted HTML span character offsets".into());
            }
            previous = Some((chars.start, chars.end));
            let mut cursor = offsets.char_to_byte(chars.start);
            for fragment in &span.fragments {
                let rendered = &fragment.rendered_bytes;
                let valid_source = match &fragment.source_bytes {
                    Some(source) => {
                        fragment.kind != HtmlFragmentKind::Unmapped
                            && source.start <= source.end
                            && source.end <= self.source_byte_length
                            && (fragment.kind != HtmlFragmentKind::Direct
                                || source.len() == rendered.len())
                    }
                    None => fragment.kind == HtmlFragmentKind::Unmapped,
                };
                if rendered.start != cursor
                    || rendered.is_empty()
                    || text.get(rendered.clone()).is_none()
                    || !valid_source
                {
                    return Err("invalid HTML source fragment ranges".into());
                }
                cursor = rendered.end;
            }
            if cursor != offsets.char_to_byte(chars.end) {
                return Err("HTML source fragments do not cover the rendered span".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_legacy_direct_fragment_against_original_even_when_digests_match() {
        let original = "<b>A</b>B";
        let mut saved = HtmlSource {
            renderer: HtmlRenderer::DeformatStrip,
            source_sha256: source_sha256(original),
            source_byte_length: original.len(),
            rendered_sha256: source_sha256("A"),
            spans: vec![HtmlSpan {
                rendered_chars: 0..1,
                fragments: vec![HtmlFragment {
                    kind: HtmlFragmentKind::Direct,
                    rendered_bytes: 0..1,
                    source_bytes: Some(3..4),
                }],
            }],
        };
        saved.verify(original, "A").unwrap();
        saved.spans[0].fragments[0].source_bytes = Some(8..9);
        assert!(saved.verify(original, "A").is_err());
    }
}
