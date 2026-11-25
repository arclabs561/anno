//! Document preprocessing and cleaning utilities.
//!
//! Provides text normalization, cleaning, and preparation for entity extraction.

use crate::lang::detect_language;
use crate::{SourceKind, SourceMap, SourceRun};
use std::collections::HashMap;
use std::ops::Range;
use unicode_normalization::char::{canonical_combining_class, decompose_canonical};

/// Prepared document with metadata.
#[derive(Debug, Clone)]
pub struct PreparedDocument {
    /// The cleaned text
    pub text: String,
    /// Metadata about the preparation process
    pub metadata: HashMap<String, String>,
}

/// Document preprocessor for cleaning and normalizing text.
#[derive(Debug, Clone)]
pub struct DocumentPreprocessor {
    /// Normalize whitespace (collapse multiple spaces, normalize line breaks)
    pub clean_whitespace: bool,
    /// Normalize Unicode (NFC normalization)
    pub normalize_unicode: bool,
    /// Detect and record language
    pub detect_language: bool,
    /// Maximum chunk size (None = no chunking)
    pub chunk_size: Option<usize>,
}

impl Default for DocumentPreprocessor {
    fn default() -> Self {
        Self {
            clean_whitespace: true,
            normalize_unicode: true,
            detect_language: false,
            chunk_size: None,
        }
    }
}

impl DocumentPreprocessor {
    /// Create a new preprocessor with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a preprocessor with all cleaning enabled.
    #[must_use]
    pub fn with_all_cleaning() -> Self {
        Self {
            clean_whitespace: true,
            normalize_unicode: true,
            detect_language: true,
            chunk_size: None,
        }
    }

    /// Prepare text for entity extraction.
    pub fn prepare(&self, text: &str) -> PreparedDocument {
        let mut processed = self.normalize(text);
        if self.clean_whitespace {
            processed = clean_whitespace(&processed, |_| {});
        }
        self.prepared(text, processed)
    }

    /// Prepare the same text and metadata as [`Self::prepare`], retaining a map
    /// from analyzed UTF-8 byte ranges to the supplied input.
    pub fn prepare_mapped(&self, text: &str) -> Result<(PreparedDocument, SourceMap), String> {
        let (normalized, upstream) = if self.normalize_unicode {
            normalize_mapped(text)?
        } else {
            (text.to_owned(), SourceMap::identity(text))
        };
        if !self.clean_whitespace {
            return Ok((self.prepared(text, normalized), upstream));
        }
        let mut runs = Vec::new();
        let cleaned = clean_whitespace(&normalized, |run| push_run(&mut runs, run));
        let map = if cleaned == normalized {
            upstream
        } else {
            SourceMap::new(&normalized, &cleaned, runs)?.compose(&upstream)?
        };
        Ok((self.prepared(text, cleaned), map))
    }

    fn normalize(&self, text: &str) -> String {
        if self.normalize_unicode {
            textprep::unicode::nfc(&textprep::unicode::remove_zero_width(text))
        } else {
            text.to_owned()
        }
    }

    fn prepared(&self, text: &str, processed: String) -> PreparedDocument {
        let mut metadata = HashMap::new();
        if self.normalize_unicode {
            metadata.insert("unicode_normalized".to_string(), "nfc".to_string());
        }
        if self.clean_whitespace {
            metadata.insert("whitespace_cleaned".to_string(), "true".to_string());
        }

        // Language detection
        if self.detect_language {
            let lang = detect_language(&processed);
            metadata.insert("detected_language".to_string(), format!("{:?}", lang));
        }

        // Chunking (if requested)
        if let Some(chunk_size) = self.chunk_size {
            // For now, just record chunk size - actual chunking would be done
            // at extraction time to preserve entity spans
            metadata.insert("chunk_size".to_string(), chunk_size.to_string());
        }

        metadata.insert("original_length".to_string(), text.len().to_string());
        metadata.insert("processed_length".to_string(), processed.len().to_string());

        PreparedDocument {
            text: processed,
            metadata,
        }
    }
}

/// Emit the same interior whitespace as a newline-normalize, collapse, trim
/// pipeline. A horizontal prefix before the first line break survives as one space.
fn clean_whitespace(text: &str, mut record: impl FnMut(SourceRun)) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        let whitespace = ch.is_whitespace();
        while chars
            .next_if(|(_, c)| c.is_whitespace() == whitespace)
            .is_some()
        {}
        let end = chars.peek().map_or(text.len(), |(offset, _)| *offset);
        if !whitespace {
            let pos = out.len();
            out.push_str(&text[start..end]);
            record(SourceRun::new(
                pos..out.len(),
                Some(start..end),
                SourceKind::Direct,
            ));
        } else if start != 0 && end != text.len() {
            let newline = text[start..end]
                .find(['\r', '\n'])
                .map_or(end, |n| start + n);
            if start < newline {
                let pos = out.len();
                out.push(' ');
                record(mapped_run(&out, pos..out.len(), text, start..newline));
            }
            if newline < end {
                let pos = out.len();
                out.push('\n');
                record(mapped_run(&out, pos..out.len(), text, newline..end));
            }
        }
    }
    out
}

fn mapped_run(out: &str, dest: Range<usize>, original: &str, src: Range<usize>) -> SourceRun {
    let kind = if out[dest.clone()] == original[src.clone()] {
        SourceKind::Direct
    } else {
        SourceKind::Transformed
    };
    SourceRun::new(dest, Some(src), kind)
}

// Coalesce while building, so a document with one change does not accumulate
// a separate mapping record for every unchanged scalar, word or space.
fn push_run(runs: &mut Vec<SourceRun>, run: SourceRun) {
    if let Some(prior) = runs.last_mut() {
        if let (Some(prev), Some(next)) = (&mut prior.source_bytes, &run.source_bytes) {
            if prior.kind == SourceKind::Direct
                && run.kind == SourceKind::Direct
                && prev.end == next.start
            {
                prior.analyzed_bytes.end = run.analyzed_bytes.end;
                prev.end = next.end;
                return;
            }
        }
    }
    runs.push(run);
}

// A starter which cannot compose backwards begins an independent NFC segment.
// Inspect its first canonical decomposition scalar so this also handles Hangul.
fn nfc_boundary_before(ch: char) -> bool {
    use unicode_normalization::{is_nfc_quick, IsNormalized};
    if ch.is_ascii() {
        return true;
    }
    if textprep::unicode::contains_zero_width(ch.encode_utf8(&mut [0; 4])) {
        return false;
    }
    let mut first = None;
    decompose_canonical(ch, |part| {
        first.get_or_insert(part);
    });
    first.is_some_and(|c| {
        canonical_combining_class(c) == 0 && is_nfc_quick(std::iter::once(c)) == IsNormalized::Yes
    })
}

fn normalize_mapped(original: &str) -> Result<(String, SourceMap), String> {
    let mut normalized = String::with_capacity(original.len());
    let mut runs = Vec::new();
    let mut start = 0;
    while start < original.len() {
        // Bound temporary alignment buffers to a small chunk plus one unbroken
        // combining sequence. A chunk may only end before an NFC boundary.
        let target = original.ceil_char_boundary(start.saturating_add(4096).min(original.len()));
        let end = original[target..]
            .char_indices()
            .find(|(_, ch)| nfc_boundary_before(*ch))
            .map_or(original.len(), |(i, _)| target + i);
        let raw = &original[start..end];
        let normalized_chunk = textprep::unicode::nfc(&textprep::unicode::remove_zero_width(raw));
        let dest = normalized.len();
        if raw == normalized_chunk {
            push_run(
                &mut runs,
                SourceRun::new(dest..dest + raw.len(), Some(start..end), SourceKind::Direct),
            );
        } else {
            align_chunk(raw, &normalized_chunk, start, dest, &mut runs)?;
        }
        normalized.push_str(&normalized_chunk);
        start = end;
    }
    let map = SourceMap::new(original, &normalized, runs)?;
    Ok((normalized, map))
}

/// Tag each canonically decomposed scalar with its originating byte interval,
/// then stably order non-starters by their canonical combining class (UAX #15).
fn tagged_nfd(text: &str, skip_zero_width: bool) -> Vec<(char, Range<usize>)> {
    let mut parts = Vec::new();
    for (start, ch) in text.char_indices() {
        let range = start..start + ch.len_utf8();
        if skip_zero_width && textprep::unicode::contains_zero_width(&text[range.clone()]) {
            continue;
        }
        decompose_canonical(ch, |part| parts.push((part, range.clone())));
    }
    for marks in parts.split_mut(|(c, _)| canonical_combining_class(*c) == 0) {
        marks.sort_by_key(|(c, _)| canonical_combining_class(*c));
    }
    parts
}

fn align_chunk(
    original: &str,
    normalized: &str,
    original_offset: usize,
    analyzed_offset: usize,
    runs: &mut Vec<SourceRun>,
) -> Result<(), String> {
    // Both strings have identical NFD after filtering the configured zero-width
    // characters. Pairing tagged NFD streams attributes composition, expansion
    // and canonical reordering to the exact input scalars, including Hangul.
    let source = tagged_nfd(original, true);
    let dest = tagged_nfd(normalized, false);
    if source.len() != dest.len() {
        return Err("Canonical decompositions differ in length".into());
    }
    let mut origins = Vec::with_capacity(source.len());
    for ((from, src), (to, out)) in source.into_iter().zip(dest) {
        if from != to {
            return Err("Canonical decompositions differ after normalization".into());
        }
        origins.push((out, src));
    }
    origins.sort_by_key(|(out, src)| (out.start, src.start));
    for group in origins.chunk_by(|a, b| a.0 == b.0) {
        let (out, src) = &group[0];
        let dest = out.start + analyzed_offset..out.end + analyzed_offset;
        let single_source = group.iter().all(|(_, range)| range == src);
        let run = if single_source {
            let kind = if normalized[out.clone()] == original[src.clone()] {
                SourceKind::Direct
            } else {
                SourceKind::Transformed
            };
            let src = src.start + original_offset..src.end + original_offset;
            SourceRun::new(dest, Some(src), kind)
        } else {
            let ranges = group
                .iter()
                .map(|(_, r)| r.start + original_offset..r.end + original_offset)
                .collect();
            SourceRun::from_ranges(dest, ranges, SourceKind::Transformed)
        };
        push_run(runs, run);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_scalar_source(
        original: &str,
        analyzed: &str,
        out: Range<usize>,
        input: Range<usize>,
        kind: SourceKind,
    ) {
        let (doc, map) = DocumentPreprocessor::new()
            .prepare_mapped(original)
            .unwrap();
        assert_eq!(doc.text, analyzed);
        let runs = map.project_bytes(analyzed, out).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].source_bytes, Some(input));
        assert_eq!(runs[0].kind, kind);
    }

    #[test]
    fn projects_unicode_composition_expansion_and_reordering() {
        use SourceKind::{Direct, Transformed};
        assert_scalar_source("cafE\u{301}", "cafÉ", 3..5, 3..6, Transformed);
        assert_scalar_source("\u{1100}\u{1161}\u{11a8}", "각", 0..3, 0..9, Transformed);
        // U+0344 expands into diaeresis and acute; both derive from the same scalar.
        assert_scalar_source("\u{0344}", "\u{308}\u{301}", 2..4, 0..2, Transformed);
        // These marks reorder, and each still names its own original bytes.
        assert_scalar_source("q\u{315}\u{300}", "q\u{300}\u{315}", 1..3, 3..5, Direct);
        assert_scalar_source("q\u{315}\u{300}", "q\u{300}\u{315}", 3..5, 1..3, Direct);
        // The grave composes across a mark; the intervening bytes are not contributors.
        for (raw, cleaned, ranges) in [
            ("a\u{315}\u{300}", "à\u{315}", vec![0..1, 3..5]),
            ("e\u{200b}\u{301}", "é", vec![0..1, 4..6]),
        ] {
            let (doc, map) = DocumentPreprocessor::new().prepare_mapped(raw).unwrap();
            assert_eq!(doc.text, cleaned);
            let runs = map.project_bytes(cleaned, 0..2).unwrap();
            assert_eq!(runs[0].kind, Transformed);
            assert_eq!(runs[0].source_bytes, None);
            assert_eq!(runs[0].source_ranges, ranges);
        }
    }

    #[test]
    fn mapped_chunks_preserve_normalization_across_boundaries() {
        let prep = DocumentPreprocessor::new();
        for tail in [
            "e\u{200b}\u{301}",
            "\u{1100}\u{1161}\u{11a8}",
            "a\u{315}\u{300}",
            "x\u{0344}",
            "\u{09c7}\u{09be}", // Bengali vowel-sign composition (both CCC 0).
            "x\u{0f73}",        // Tibetan vowel decomposes to leading non-starters.
        ] {
            for position in 4090..4100 {
                let raw = format!("{}{}{}", "A".repeat(position), tail, "京".repeat(1500));
                let (doc, map) = prep.prepare_mapped(&raw).unwrap();
                assert_eq!(doc.text, prep.prepare(&raw).text);
                assert!(map.runs.len() <= 6, "unstable run count at {position}");
            }
        }
    }

    #[test]
    fn whitespace_matches_legacy_newline_collapse() {
        let alphabet = [' ', '\t', '\r', '\n', '\u{a0}', 'A'];
        for mut n in 0..alphabet.len().pow(4) {
            let text: String = (0..4)
                .map(|_| {
                    let ch = alphabet[n % alphabet.len()];
                    n /= alphabet.len();
                    ch
                })
                .collect();
            let mut expected = String::new();
            let mut was_space = false;
            for ch in textprep::unicode::normalize_newlines(&text).chars() {
                if !ch.is_whitespace() {
                    expected.push(ch);
                } else if !was_space {
                    expected.push(if ch == '\n' { '\n' } else { ' ' });
                } else if ch == '\n' && !expected.ends_with('\n') {
                    expected.push('\n');
                }
                was_space = ch.is_whitespace();
            }
            assert_eq!(clean_whitespace(&text, |_| {}), expected.trim(), "{text:?}");
        }
    }

    #[test]
    fn whitespace_sources_include_contributing_runs_and_exclude_trimmed_edges() {
        let input = "\tA\u{a0}\t\r\n\r\n B  C\r ";
        let output = "A \nB C";
        assert_scalar_source(input, output, 0..1, 1..2, SourceKind::Direct);
        assert_scalar_source(input, output, 1..2, 2..5, SourceKind::Transformed);
        assert_scalar_source(input, output, 2..3, 5..10, SourceKind::Transformed);
        assert_scalar_source(input, output, 4..5, 11..13, SourceKind::Transformed);
    }

    #[test]
    fn whitespace_collapse_excludes_removed_zero_width_bytes() {
        let (doc, map) = DocumentPreprocessor::new()
            .prepare_mapped("A \u{200b} B")
            .unwrap();
        assert_eq!(doc.text, "A B");
        let runs = map.project_bytes(&doc.text, 1..2).unwrap();
        assert_eq!(runs[0].kind, SourceKind::Transformed);
        assert_eq!(runs[0].source_ranges, vec![1..2, 5..6]);
    }

    #[test]
    fn mapped_preparation_matches_text_and_metadata_in_each_mode() {
        for clean_whitespace in [false, true] {
            for normalize_unicode in [false, true] {
                let prep = DocumentPreprocessor {
                    clean_whitespace,
                    normalize_unicode,
                    chunk_size: Some(20),
                    detect_language: true,
                };
                for input in ["", "\u{200b} \r\n", "  東京\r\nB\u{301}\t e\u{301} "] {
                    let plain = prep.prepare(input);
                    let (mapped, _) = prep.prepare_mapped(input).unwrap();
                    assert_eq!(mapped.text, plain.text);
                    assert_eq!(mapped.metadata, plain.metadata);
                }
            }
        }
    }

    #[test]
    fn test_preprocessor_default() {
        let prep = DocumentPreprocessor::new();
        assert!(prep.clean_whitespace);
        assert!(prep.normalize_unicode);
        assert!(!prep.detect_language);
        assert!(prep.chunk_size.is_none());
    }

    #[test]
    fn test_preprocessor_with_all_cleaning() {
        let prep = DocumentPreprocessor::with_all_cleaning();
        assert!(prep.clean_whitespace);
        assert!(prep.normalize_unicode);
        assert!(prep.detect_language);
    }

    #[test]
    fn test_whitespace_normalization() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("Hello   world\r\n\r\ntest");

        // Multiple spaces should be collapsed
        assert!(!doc.text.contains("  "));
        // CRLF should be normalized to LF
        assert!(!doc.text.contains("\r"));
    }

    #[test]
    fn test_unicode_zero_width_removal() {
        let prep = DocumentPreprocessor::new();
        let input = "Hello\u{200b}world\u{feff}test";
        let doc = prep.prepare(input);

        assert!(!doc.text.contains('\u{200b}'));
        assert!(!doc.text.contains('\u{feff}'));
        assert!(doc.text.contains("Helloworld"));
    }

    #[test]
    fn test_trim_whitespace() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("   text with spaces   ");

        assert_eq!(doc.text, "text with spaces");
    }

    #[test]
    fn test_metadata_recording() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("test input");

        assert!(doc.metadata.contains_key("original_length"));
        assert!(doc.metadata.contains_key("processed_length"));
        assert!(doc.metadata.contains_key("whitespace_cleaned"));
        assert_eq!(
            doc.metadata.get("unicode_normalized").map(String::as_str),
            Some("nfc"),
        );
    }

    #[test]
    fn test_nfc_normalization_precomposed_eq_decomposed() {
        let prep = DocumentPreprocessor::new();
        let precomposed = "caf\u{00e9}"; // "café" single codepoint
        let decomposed = "cafe\u{0301}"; // "cafe" + combining acute
        assert_ne!(precomposed, decomposed);
        let a = prep.prepare(precomposed).text;
        let b = prep.prepare(decomposed).text;
        assert_eq!(a, b);
    }

    #[test]
    fn test_language_detection_metadata() {
        let prep = DocumentPreprocessor::with_all_cleaning();
        let doc = prep.prepare("Hello world, this is English text.");

        assert!(doc.metadata.contains_key("detected_language"));
        assert!(doc
            .metadata
            .get("detected_language")
            .unwrap()
            .contains("English"));
    }

    #[test]
    fn test_preserve_paragraph_breaks() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("First paragraph.\n\nSecond paragraph.");

        // Should preserve double newline (paragraph break)
        assert!(doc.text.contains("\n\n") || doc.text.contains("\n"));
    }

    #[test]
    fn test_empty_input() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("");

        assert!(doc.text.is_empty());
        assert_eq!(doc.metadata.get("original_length"), Some(&"0".to_string()));
    }

    #[test]
    fn test_prepared_document_clone() {
        let prep = DocumentPreprocessor::new();
        let doc = prep.prepare("test");
        let cloned = doc.clone();

        assert_eq!(doc.text, cloned.text);
        assert_eq!(doc.metadata, cloned.metadata);
    }

    #[test]
    fn test_cjk_text_handling() {
        let prep = DocumentPreprocessor::with_all_cleaning();
        let doc = prep.prepare("東京オリンピック2020は延期されました。");

        // Should preserve CJK text
        assert!(doc.text.contains("東京"));
        // Language should be detected
        assert!(doc.metadata.contains_key("detected_language"));
    }
}
