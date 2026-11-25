//! Byte provenance for the exact text consumed by an extractor.

use super::html_source::{bytes_sha256, source_sha256, HtmlFragmentKind, HtmlSource};
use super::GroundedDocument;
use crate::offset::TextSpan;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// How an interval of analyzed text is supported by the original source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// The source contains the identical UTF-8 bytes; subranges can be clipped exactly.
    Direct,
    /// The source range participated in a known transformation of the complete run.
    Transformed,
    /// The range is a conservative location supplied by the producer.
    Approximate,
    /// Text inserted by the producer, optionally with a source anchor or originating markup.
    Synthetic,
    /// The producer could not establish an original location.
    Unmapped,
}

/// One nonempty analyzed UTF-8 interval and its optional original byte range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRun {
    /// Half-open bytes in the analyzed text, in document order.
    pub analyzed_bytes: Range<usize>,
    /// Half-open bytes in the original artifact, or an insertion point for synthetic text.
    pub source_bytes: Option<Range<usize>>,
    /// Multiple disjoint original ranges when a transformed run draws from separated
    /// bytes. Stored in source order, with no assumed alignment inside the output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_ranges: Vec<Range<usize>>,
    /// Interpretation of the coordinate correspondence.
    pub kind: SourceKind,
}

impl SourceRun {
    /// A run with one original range or no located source.
    #[must_use]
    pub fn new(
        analyzed_bytes: Range<usize>,
        source_bytes: Option<Range<usize>>,
        kind: SourceKind,
    ) -> Self {
        Self {
            analyzed_bytes,
            source_bytes,
            source_ranges: Vec::new(),
            kind,
        }
    }

    /// Construct the union of the contributing ranges, coalescing only overlaps
    /// and touching intervals. The map constructor checks bounds and kind.
    #[must_use]
    pub fn from_ranges(
        analyzed_bytes: Range<usize>,
        mut ranges: Vec<Range<usize>>,
        kind: SourceKind,
    ) -> Self {
        if ranges.iter().all(|r| r.start <= r.end) {
            ranges.sort_unstable_by_key(|r| (r.start, r.end));
            let mut union: Vec<Range<usize>> = Vec::new();
            for r in ranges {
                if let Some(prev) = union.last_mut() {
                    if r.start <= prev.end {
                        prev.end = prev.end.max(r.end);
                        continue;
                    }
                }
                union.push(r);
            }
            ranges = union;
        }
        if ranges.len() <= 1 {
            return Self::new(analyzed_bytes, ranges.pop(), kind);
        }
        Self {
            analyzed_bytes,
            source_bytes: None,
            source_ranges: ranges,
            kind,
        }
    }

    /// All known original byte ranges for this run.
    pub fn origins(&self) -> impl Iterator<Item = &Range<usize>> {
        self.source_bytes.iter().chain(&self.source_ranges)
    }

    fn valid_origins(&self, length: Option<usize>) -> bool {
        if !self.source_ranges.is_empty()
            && (self.source_ranges.len() < 2
                || self.source_bytes.is_some()
                || !self
                    .source_ranges
                    .windows(2)
                    .all(|pair| pair[0].end < pair[1].start))
        {
            return false;
        }
        if self
            .origins()
            .any(|r| r.start > r.end || length.is_none_or(|n| r.end > n))
        {
            return false;
        }
        match self.kind {
            SourceKind::Direct => self
                .source_bytes
                .as_ref()
                .is_some_and(|r| r.len() == self.analyzed_bytes.len()),
            SourceKind::Unmapped => self.origins().next().is_none(),
            SourceKind::Synthetic => true,
            SourceKind::Transformed | SourceKind::Approximate => {
                self.origins().next().is_some() && self.origins().all(|r| !r.is_empty())
            }
        }
    }

    fn clip(&self, clipped: Range<usize>) -> Self {
        let (kind, source) = match (self.kind, &self.source_bytes) {
            (SourceKind::Direct, Some(source)) => {
                let begin = source.start + (clipped.start - self.analyzed_bytes.start);
                (SourceKind::Direct, Some(begin..begin + clipped.len()))
            }
            (SourceKind::Transformed, source) if clipped != self.analyzed_bytes => {
                (SourceKind::Approximate, source.clone())
            }
            (kind, source) => (kind, source.clone()),
        };
        Self {
            analyzed_bytes: clipped,
            source_bytes: source,
            source_ranges: self.source_ranges.clone(),
            kind,
        }
    }
}

/// A partition of analyzed UTF-8 text. Original ranges may overlap or change order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceMap {
    /// SHA-256 of the exact original bytes, absent when the original is unavailable.
    pub source_sha256: Option<String>,
    /// Original byte length, present when the original digest is known.
    pub source_byte_length: Option<usize>,
    /// SHA-256 of the exact UTF-8 analyzed text.
    pub analyzed_sha256: String,
    /// Analyzed byte length.
    pub analyzed_byte_length: usize,
    /// Nonempty, ordered and gap-free across the analyzed text.
    pub runs: Vec<SourceRun>,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn checked_char_range(text: &str, chars: Range<usize>) -> Result<Range<usize>, String> {
    let span = TextSpan::from_chars(text, chars.start, chars.end);
    if span.char_start != chars.start || span.char_end != chars.end {
        return Err("analyzed character range is out of bounds".into());
    }
    Ok(span.byte_start..span.byte_end)
}

impl SourceMap {
    /// Verify a producer's complete UTF-8 map against both strings before storing it.
    pub fn new(original: &str, analyzed: &str, runs: Vec<SourceRun>) -> Result<Self, String> {
        let mut map = Self {
            source_sha256: Some(source_sha256(original)),
            source_byte_length: Some(original.len()),
            analyzed_sha256: source_sha256(analyzed),
            analyzed_byte_length: analyzed.len(),
            runs,
        };
        map.validate_structure()?;
        map.check_ranges(analyzed, Some(original))?;
        map.compact();
        Ok(map)
    }

    /// One direct run over unchanged text, or no runs for empty text.
    #[must_use]
    pub fn identity(text: &str) -> Self {
        let digest = source_sha256(text);
        Self {
            source_sha256: Some(digest.clone()),
            source_byte_length: Some(text.len()),
            analyzed_sha256: digest,
            analyzed_byte_length: text.len(),
            runs: if text.is_empty() {
                vec![]
            } else {
                vec![SourceRun {
                    analyzed_bytes: 0..text.len(),
                    source_bytes: Some(0..text.len()),
                    source_ranges: Vec::new(),
                    kind: SourceKind::Direct,
                }]
            },
        }
    }

    /// Text with no recoverable coordinates, even when the original binary bytes are known.
    #[must_use]
    pub fn unmapped(analyzed: &str, original: Option<&[u8]>) -> Self {
        Self {
            source_sha256: original.map(bytes_sha256),
            source_byte_length: original.map(<[u8]>::len),
            analyzed_sha256: source_sha256(analyzed),
            analyzed_byte_length: analyzed.len(),
            runs: if analyzed.is_empty() {
                vec![]
            } else {
                vec![SourceRun {
                    analyzed_bytes: 0..analyzed.len(),
                    source_bytes: None,
                    source_ranges: Vec::new(),
                    kind: SourceKind::Unmapped,
                }]
            },
        }
    }

    fn validate_structure(&self) -> Result<(), String> {
        if !valid_digest(&self.analyzed_sha256)
            || self.source_sha256.is_some() != self.source_byte_length.is_some()
            || self
                .source_sha256
                .as_ref()
                .is_some_and(|s| !valid_digest(s))
        {
            return Err("invalid source map digest or missing original byte length".into());
        }
        let mut cursor = 0;
        for run in &self.runs {
            if run.analyzed_bytes.start != cursor
                || run.analyzed_bytes.is_empty()
                || run.analyzed_bytes.end > self.analyzed_byte_length
            {
                return Err("source map runs must partition all analyzed bytes in order".into());
            }
            if !run.valid_origins(self.source_byte_length) {
                return Err("source map run has inconsistent original coordinates or kind".into());
            }
            cursor = run.analyzed_bytes.end;
        }
        if cursor != self.analyzed_byte_length {
            return Err("source map omits analyzed bytes".into());
        }
        Ok(())
    }

    fn check_ranges(&self, analyzed: &str, original: Option<&str>) -> Result<(), String> {
        for run in &self.runs {
            let output = analyzed
                .get(run.analyzed_bytes.clone())
                .ok_or("analyzed run splits a UTF-8 scalar")?;
            if let Some(source) = original {
                for range in run.origins() {
                    let input = source
                        .get(range.clone())
                        .ok_or("original run splits a UTF-8 scalar")?;
                    if run.kind == SourceKind::Direct && input != output {
                        return Err("direct source run does not contain the analyzed bytes".into());
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate the stored geometry and the analyzed text available in a grounded document.
    pub fn validate_analyzed(&self, analyzed: &str) -> Result<(), String> {
        self.validate_structure()?;
        if self.analyzed_byte_length != analyzed.len()
            || self.analyzed_sha256 != source_sha256(analyzed)
        {
            return Err("source map does not match the analyzed text edition".into());
        }
        let same_source = self.source_sha256.as_ref() == Some(&self.analyzed_sha256);
        if same_source && self.source_byte_length != Some(analyzed.len()) {
            return Err("source and analyzed digests agree but lengths differ".into());
        }
        self.check_ranges(analyzed, same_source.then_some(analyzed))
    }

    /// Verify the original artifact identity and original UTF-8 boundaries if it has mapped runs.
    pub fn verify_original(&self, original: &[u8]) -> Result<(), String> {
        self.validate_structure()?;
        let digest = self
            .source_sha256
            .as_deref()
            .ok_or("original source identity is unknown")?;
        if self.source_byte_length != Some(original.len()) || digest != bytes_sha256(original) {
            return Err("original source bytes do not match source map".into());
        }
        if self.runs.iter().any(|r| r.origins().next().is_some()) {
            let text =
                std::str::from_utf8(original).map_err(|_| "mapped original must be UTF-8")?;
            if self
                .runs
                .iter()
                .flat_map(SourceRun::origins)
                .any(|r| text.get(r.clone()).is_none())
            {
                return Err("original run splits a UTF-8 scalar".into());
            }
        }
        Ok(())
    }

    /// Verify both editions, UTF-8 slicing, and all claimed direct copies.
    pub fn verify(&self, original: &[u8], analyzed: &str) -> Result<(), String> {
        self.verify_original(original)?;
        self.validate_analyzed(analyzed)?;
        self.check_ranges(analyzed, std::str::from_utf8(original).ok())
    }

    fn overlaps(&self, range: Range<usize>) -> impl Iterator<Item = SourceRun> + '_ {
        let first = self
            .runs
            .partition_point(|r| r.analyzed_bytes.end <= range.start);
        self.runs[first..]
            .iter()
            .take_while(move |r| r.analyzed_bytes.start < range.end)
            .filter_map(move |r| {
                let start = r.analyzed_bytes.start.max(range.start);
                let end = r.analyzed_bytes.end.min(range.end);
                (start < end).then(|| r.clip(start..end))
            })
    }

    /// Resolve a half-open UTF-8 byte span, retaining each fragment's uncertainty.
    pub fn project_bytes(
        &self,
        analyzed: &str,
        range: Range<usize>,
    ) -> Result<Vec<SourceRun>, String> {
        self.validate_analyzed(analyzed)?;
        analyzed
            .get(range.clone())
            .ok_or("analyzed byte range is out of bounds or splits a UTF-8 scalar")?;
        Ok(self.overlaps(range).collect())
    }

    /// Resolve a half-open Unicode scalar span through the byte map.
    pub fn project_chars(
        &self,
        analyzed: &str,
        range: Range<usize>,
    ) -> Result<Vec<SourceRun>, String> {
        self.project_bytes(analyzed, checked_char_range(analyzed, range)?)
    }

    // An insertion exactly between runs has a single defensible source anchor
    // only when both surrounding pieces agree, or when the other side is EOF.
    fn direct_anchor(&self, point: usize) -> Option<Range<usize>> {
        if self.runs.is_empty() && point == 0 && self.source_byte_length == Some(0) {
            return Some(0..0);
        }
        let first = self.runs.partition_point(|r| r.analyzed_bytes.end < point);
        let mut anchor = None;
        for run in self.runs[first..]
            .iter()
            .take_while(|r| r.analyzed_bytes.start <= point)
        {
            if run.kind != SourceKind::Direct {
                return None;
            }
            let pos = run.source_bytes.as_ref()?.start + (point - run.analyzed_bytes.start);
            if anchor.is_some_and(|prev| prev != pos) {
                return None;
            }
            anchor = Some(pos);
        }
        anchor.map(|p| p..p)
    }

    /// Compose final→intermediate with intermediate→original. Intermediate
    /// text digests must match. Preserve disjoint original contributors for one
    /// indivisible output, including when they span a deleted markup region.
    pub fn compose(&self, upstream: &Self) -> Result<Self, String> {
        self.validate_structure()?;
        upstream.validate_structure()?;
        if self.source_sha256.as_ref() != Some(&upstream.analyzed_sha256)
            || self.source_byte_length != Some(upstream.analyzed_byte_length)
        {
            return Err("source maps refer to different intermediate editions".into());
        }
        let mut runs = Vec::new();
        for run in &self.runs {
            if run.kind == SourceKind::Direct {
                let middle = run.source_bytes.as_ref().expect("validated direct run");
                for mut part in upstream.overlaps(middle.clone()) {
                    let start =
                        run.analyzed_bytes.start + (part.analyzed_bytes.start - middle.start);
                    part.analyzed_bytes = start..start + part.analyzed_bytes.len();
                    runs.push(part);
                }
            } else if run.origins().next().is_none() {
                runs.push(run.clone());
            } else {
                runs.push(upstream.compose_indivisible(run));
            }
        }
        let mut result = Self {
            source_sha256: upstream.source_sha256.clone(),
            source_byte_length: upstream.source_byte_length,
            analyzed_sha256: self.analyzed_sha256.clone(),
            analyzed_byte_length: self.analyzed_byte_length,
            runs,
        };
        result.compact();
        result.validate_structure()?;
        Ok(result)
    }

    fn compose_indivisible(&self, downstream: &SourceRun) -> SourceRun {
        let mut origins = Vec::new();
        let mut incomplete = downstream.kind == SourceKind::Approximate;
        let mut all_synthetic = true;
        let mut any_synthetic = false;
        for middle in downstream.origins() {
            if middle.is_empty() {
                origins.extend(self.direct_anchor(middle.start));
                continue;
            }
            for fragment in self.overlaps(middle.clone()) {
                let synthetic = fragment.kind == SourceKind::Synthetic;
                all_synthetic &= synthetic;
                any_synthetic |= synthetic;
                incomplete |=
                    fragment.kind == SourceKind::Approximate || fragment.origins().next().is_none();
                origins.extend(fragment.origins().cloned());
            }
        }
        let kind = if downstream.kind == SourceKind::Synthetic || all_synthetic {
            SourceKind::Synthetic
        } else if origins.iter().all(Range::is_empty) {
            SourceKind::Unmapped
        } else if incomplete || any_synthetic {
            SourceKind::Approximate
        } else {
            SourceKind::Transformed
        };
        if kind != SourceKind::Synthetic {
            origins.retain(|r| !r.is_empty());
        }
        SourceRun::from_ranges(downstream.analyzed_bytes.clone(), origins, kind)
    }

    pub(super) fn check_html_identity(&self, html: &HtmlSource) -> Result<(), String> {
        if self.source_sha256.as_ref() != Some(&html.source_sha256)
            || self.source_byte_length != Some(html.source_byte_length)
            || self.analyzed_sha256 != html.rendered_sha256
        {
            return Err("HTML and generic source maps identify different source editions".into());
        }
        Ok(())
    }

    fn compact(&mut self) {
        let mut merged: Vec<SourceRun> = Vec::new();
        for run in self.runs.drain(..) {
            if let Some(prior) = merged.last_mut() {
                if prior.kind == SourceKind::Direct
                    && run.kind == SourceKind::Direct
                    && prior.source_bytes.as_ref().map(|r| r.end)
                        == run.source_bytes.as_ref().map(|r| r.start)
                {
                    prior.analyzed_bytes.end = run.analyzed_bytes.end;
                    prior.source_bytes.as_mut().expect("direct range").end =
                        run.source_bytes.expect("direct range").end;
                    continue;
                }
                if prior.kind == SourceKind::Unmapped && run.kind == SourceKind::Unmapped {
                    prior.analyzed_bytes.end = run.analyzed_bytes.end;
                    continue;
                }
            }
            merged.push(run);
        }
        self.runs = merged;
    }
}

impl GroundedDocument {
    /// Attach a map only when its analyzed coordinates and digest match this document.
    pub fn record_source_map(&mut self, map: SourceMap) -> Result<(), String> {
        map.validate_analyzed(self.text())?;
        if let Some(html) = self.html_source() {
            map.check_html_identity(html)?;
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
            let start = bytes.start.max(frag.rendered_bytes.start);
            let end = bytes.end.min(frag.rendered_bytes.end);
            if start < end {
                let kind = match frag.kind {
                    HtmlFragmentKind::Direct => SourceKind::Direct,
                    HtmlFragmentKind::EntityDecoded => SourceKind::Transformed,
                    HtmlFragmentKind::WhitespaceCollapsed | HtmlFragmentKind::Unknown => {
                        SourceKind::Approximate
                    }
                    HtmlFragmentKind::Synthetic => SourceKind::Synthetic,
                    HtmlFragmentKind::Unmapped => SourceKind::Unmapped,
                };
                runs.push(
                    SourceRun {
                        analyzed_bytes: frag.rendered_bytes.clone(),
                        source_bytes: frag.source_bytes.clone(),
                        source_ranges: Vec::new(),
                        kind,
                    }
                    .clip(start..end),
                );
            }
        }
        Ok(Some(runs))
    }
}

#[cfg(test)]
mod tests {
    use super::super::html_source::{HtmlFragment, HtmlRenderer, HtmlSpan};
    use super::*;

    #[test]
    fn a_transformed_scalar_retains_both_original_fragments_through_three_maps() {
        let html = "<b>e</b><i>&#x301;</i>!";
        let reference = html.find("&#x301;").unwrap();
        let end = html.len();
        let rendered = "e\u{301}!";
        let strip = SourceMap::new(
            html,
            rendered,
            vec![
                SourceRun::new(0..1, Some(3..4), SourceKind::Direct),
                SourceRun::new(
                    1..3,
                    Some(reference..reference + 7),
                    SourceKind::Transformed,
                ),
                SourceRun::new(3..4, Some(end - 1..end), SourceKind::Direct),
            ],
        )
        .unwrap();
        let nfc = SourceMap::new(
            rendered,
            "é!",
            vec![
                SourceRun::new(0..2, Some(0..3), SourceKind::Transformed),
                SourceRun::new(2..3, Some(3..4), SourceKind::Direct),
            ],
        )
        .unwrap();
        let expected = SourceRun::from_ranges(
            0..2,
            vec![3..4, reference..reference + 7],
            SourceKind::Transformed,
        );
        let composed = nfc.compose(&strip).unwrap();
        let third = SourceMap::new(
            "é!",
            "é",
            vec![SourceRun::new(0..2, Some(0..2), SourceKind::Direct)],
        )
        .unwrap();
        let left = third.compose(&composed).unwrap();
        let right = third.compose(&nfc).unwrap().compose(&strip).unwrap();
        for map in [left, right] {
            assert_eq!(map.project_chars("é", 0..1).unwrap(), [expected.clone()]);
            assert_eq!(map.runs[0].source_bytes, None);
            assert!(map.project_bytes("é", 0..1).is_err());
            map.verify(html.as_bytes(), "é").unwrap();
        }
    }

    #[test]
    fn transformed_intersections_and_incomplete_origins_remain_approximate() {
        let upstream = SourceMap::new(
            "a---b",
            "axxb",
            vec![
                SourceRun::new(0..1, Some(0..1), SourceKind::Direct),
                SourceRun::new(1..3, None, SourceKind::Unmapped),
                SourceRun::new(3..4, Some(4..5), SourceKind::Direct),
            ],
        )
        .unwrap();
        let folded = SourceMap::new(
            "axxb",
            "XY",
            vec![SourceRun::new(0..2, Some(0..4), SourceKind::Transformed)],
        )
        .unwrap();
        assert_eq!(
            folded.project_chars("XY", 1..2).unwrap()[0].kind,
            SourceKind::Approximate
        );
        let mixed = folded.compose(&upstream).unwrap();
        assert_eq!(
            mixed.runs[0],
            SourceRun::from_ranges(0..2, vec![0..1, 4..5], SourceKind::Approximate)
        );
        assert_eq!(mixed.compose(&SourceMap::identity("a---b")).unwrap(), mixed);

        let inserted = SourceMap::new(
            "",
            "x",
            vec![SourceRun::new(0..1, None, SourceKind::Synthetic)],
        )
        .unwrap();
        let transformed = SourceMap::new(
            "x",
            "X",
            vec![SourceRun::new(0..1, Some(0..1), SourceKind::Transformed)],
        )
        .unwrap();
        assert_eq!(
            transformed.compose(&inserted).unwrap().runs[0].kind,
            SourceKind::Synthetic
        );
    }

    #[test]
    fn validates_geometry_opaque_artifacts_hashes_and_clips_unicode_direct_spans() {
        let source = "🙂éa";
        let map = SourceMap::identity(source);
        assert_eq!(
            map.project_chars(source, 1..2).unwrap()[0].source_bytes,
            Some(4..6)
        );
        assert!(map.project_chars(source, 3..3).unwrap().is_empty());
        assert!(map.project_chars(source, 4..4).is_err());
        assert!(map.project_chars(source, 2..1).is_err());
        assert!(map.project_bytes(source, 1..4).is_err());
        assert!(SourceMap::new(
            "ab",
            "ax",
            vec![SourceRun::new(0..2, Some(0..2), SourceKind::Direct)]
        )
        .is_err());
        assert!(SourceMap::new(
            "é",
            "x",
            vec![SourceRun::new(0..1, Some(1..2), SourceKind::Transformed)]
        )
        .is_err());
        assert!(SourceMap::new(
            "abcdefgh",
            "xy",
            vec![SourceRun::from_ranges(
                0..2,
                vec![0..1, 4..5],
                SourceKind::Direct
            )]
        )
        .is_err());
        assert!(SourceMap::new(
            "ab",
            "ab",
            vec![SourceRun::new(1..2, Some(1..2), SourceKind::Direct)]
        )
        .is_err());
        let merged = SourceRun::from_ranges(0..1, vec![4..6, 0..2, 1..4], SourceKind::Transformed);
        assert_eq!(merged.source_bytes, Some(0..6));
        assert!(merged.source_ranges.is_empty());

        let opaque = SourceMap::unmapped("x", Some(&[0xff, 0x01]));
        opaque.verify(&[0xff, 0x01], "x").unwrap();
        assert!(opaque.verify_original(&[0xff, 0x02]).is_err());
        assert!(SourceMap::unmapped("x", None)
            .verify_original(b"x")
            .is_err());
        assert!(SourceMap::identity("y").compose(&opaque).is_err());
        let carried = SourceMap::identity("x").compose(&opaque).unwrap();
        assert_eq!(carried, opaque);
        assert_eq!(SourceMap::identity("").project_chars("", 0..0).unwrap(), []);
    }

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
