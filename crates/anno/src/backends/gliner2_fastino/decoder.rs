//! Span-score → Entity decoder. Converts ONNX output to char-offset entities.
#![allow(missing_docs)] // implementation internals; public API is on GLiNER2Fastino in mod.rs
//!
//! For each span (start_word, end_word) where score > threshold for label L,
//! look up the byte offsets of `start_word` and `end_word` in the original
//! text via the splitter's offset table, then convert byte offsets to char
//! offsets using `crate::offset::bytes_to_chars`.
//!
//! This is the porting hazard from the Phase 1 spec §6 risk #1: upstream's
//! gliner2-rs returns token offsets; we return char offsets in the original
//! input.

use crate::Entity;

/// One candidate span emitted by the model.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // standalone decoder path; pipeline.rs has its own span decode
pub struct Span {
    pub start_word: usize,
    pub end_word: usize,
    pub label_idx: usize,
    pub score: f32,
}

/// Decode spans into Entities with **character** offsets in the original text.
#[allow(dead_code)] // standalone decoder path; pipeline.rs has its own span decode
pub fn decode_spans(
    text: &str,
    word_offsets: &[(usize, usize)], // (byte_start, byte_end) per word
    labels: &[String],
    spans: &[Span],
    threshold: f32,
) -> Vec<Entity> {
    let mut out = Vec::new();
    for s in spans {
        if s.score < threshold {
            continue;
        }
        if s.start_word > s.end_word
            || s.end_word >= word_offsets.len()
            || s.label_idx >= labels.len()
        {
            continue;
        }
        let (byte_start, _) = word_offsets[s.start_word];
        let (_, byte_end) = word_offsets[s.end_word];
        let (char_start, char_end) = crate::offset::bytes_to_chars(text, byte_start, byte_end);
        let surface = &text[byte_start..byte_end];
        let etype = crate::schema::map_to_canonical(&labels[s.label_idx], None);
        out.push(Entity::new(surface, etype, char_start, char_end, s.score));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_two_spans_with_char_offsets() {
        let text = "Acme Corp in Paris.";
        // word_offsets: byte ranges for ["Acme","Corp","in","Paris","."]
        let words = [(0, 4), (5, 9), (10, 12), (13, 18), (18, 19)];
        let labels = vec!["organization".into(), "location".into()];
        let spans = vec![
            Span {
                start_word: 0,
                end_word: 1,
                label_idx: 0,
                score: 0.9,
            }, // "Acme Corp"
            Span {
                start_word: 3,
                end_word: 3,
                label_idx: 1,
                score: 0.8,
            }, // "Paris"
            Span {
                start_word: 0,
                end_word: 0,
                label_idx: 0,
                score: 0.1,
            }, // below threshold
        ];

        let ents = decode_spans(text, &words, &labels, &spans, 0.5);
        assert_eq!(ents.len(), 2);

        assert_eq!(ents[0].text, "Acme Corp");
        assert_eq!(ents[0].start(), 0);
        assert_eq!(ents[0].end(), 9);

        assert_eq!(ents[1].text, "Paris");
        assert_eq!(ents[1].start(), 13);
        assert_eq!(ents[1].end(), 18);
    }

    #[test]
    fn decodes_unicode_with_char_offsets() {
        // "田中" is 6 bytes / 2 chars; "Paris" is 5 bytes / 5 chars.
        let text = "田中 Paris";
        let words = [(0, 6), (7, 12)];
        let labels = vec!["person".into(), "location".into()];
        let spans = vec![
            Span {
                start_word: 0,
                end_word: 0,
                label_idx: 0,
                score: 0.9,
            },
            Span {
                start_word: 1,
                end_word: 1,
                label_idx: 1,
                score: 0.9,
            },
        ];
        let ents = decode_spans(text, &words, &labels, &spans, 0.5);
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].text, "田中");
        assert_eq!(ents[0].start(), 0);
        assert_eq!(ents[0].end(), 2); // chars, not bytes
        assert_eq!(ents[1].start(), 3); // 2 chars + 1 space
        assert_eq!(ents[1].end(), 8);
    }

    #[test]
    fn out_of_range_spans_are_dropped() {
        let text = "a b";
        let words = [(0, 1), (2, 3)];
        let labels = vec!["x".into()];
        let spans = vec![
            Span {
                start_word: 0,
                end_word: 99,
                label_idx: 0,
                score: 0.9,
            },
            Span {
                start_word: 0,
                end_word: 0,
                label_idx: 99,
                score: 0.9,
            },
            Span {
                start_word: 1,
                end_word: 0,
                label_idx: 0,
                score: 0.9,
            }, // start > end
        ];
        let ents = decode_spans(text, &words, &labels, &spans, 0.0);
        assert_eq!(ents.len(), 0);
    }
}
