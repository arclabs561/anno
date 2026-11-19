//! Captured from the same three-page PDF with Kreuzberg 4.9.2 and 4.10.4.
//! The sentence spans come from Aurelius LanguageRuntime::analyze at 1319bcd,
//! run on the unmarked extraction with its exported tokenizer and rules.
//!
//! To regenerate the page ranges, put each string in `with_pages.pages` on its own
//! PDF page (ReportLab Canvas, Bitstream Vera, 14pt, at x=70 and y=710). In
//! Python, pass `ExtractionConfig(pages=PageConfig(extract_pages=True,
//! insert_page_markers=marked))` to `kreuzberg.extract_file_sync` and read
//! `result.metadata["pages"]["boundaries"]` for each setting of `marked`.

use anno::backends::{chunking::ChunkConfig, regex::RegexNER};
use anno::rag::{preprocess, preprocess_with_hints, RagPreprocessConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct Extraction {
    text: String,
    pages: Vec<String>,
    bounds: Vec<[usize; 2]>,
}

#[derive(Deserialize)]
struct Fixture {
    with_pages: Extraction,
    marked: Extraction,
    sentences: Vec<[usize; 2]>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/rag_page_hints.json")).unwrap()
}

// A downstream caller can keep spelling every field in the original struct.
fn config() -> RagPreprocessConfig {
    RagPreprocessConfig {
        chunk_config: ChunkConfig {
            chunk_size: 40,
            overlap: 3,
            respect_sentences: false,
            ..Default::default()
        },
        coref: Default::default(),
    }
}

#[test]
fn borrowed_ends_from_pages_or_sentences_split_the_source_text() {
    let f = fixture();
    let doc = &f.with_pages;
    for (&[start, end], page) in doc.bounds.iter().zip(&doc.pages) {
        assert_eq!(doc.text.get(start..end), Some(page.as_str()));
    }
    for bounds in [&doc.bounds, &f.sentences] {
        let ends: Vec<_> = bounds.iter().map(|b| b[1]).collect();
        let chunks = preprocess_with_hints(&doc.text, &RegexNER, Some(config()), &ends).unwrap();
        let ranges: Vec<_> = chunks.iter().map(|c| (c.char_start, c.char_end)).collect();
        assert_eq!(ranges, [(0, 24), (21, 56), (53, 79)]);
        for chunk in chunks {
            let source: String = doc
                .text
                .chars()
                .skip(chunk.char_start)
                .take(chunk.char_end - chunk.char_start)
                .collect();
            assert_eq!(chunk.original_text, source);
        }
    }
    let regular = preprocess(&doc.text, &RegexNER, Some(config())).unwrap();
    let empty = preprocess_with_hints(&doc.text, &RegexNER, Some(config()), &[]).unwrap();
    let text = |chunks: Vec<anno::rag::RagChunk>| {
        chunks
            .into_iter()
            .map(|c| c.original_text)
            .collect::<Vec<_>>()
    };
    assert_eq!(text(regular), text(empty));
}

#[test]
fn page_markers_expose_inconsistent_upstream_ranges() {
    let doc = fixture().marked;
    for (&[start, end], page) in doc.bounds.iter().zip(&doc.pages) {
        assert_ne!(doc.text.get(start..end), Some(page.as_str()));
    }
    let ends: Vec<_> = doc.bounds.iter().map(|b| b[1]).collect();
    assert_eq!((doc.text.len(), ends.last()), (134, Some(&136)));
    assert!(preprocess_with_hints(&doc.text, &RegexNER, Some(config()), &ends).is_err());
}
