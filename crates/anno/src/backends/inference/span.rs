//! Span representation types and the handshaking matrix for joint extraction.
//!
//! - `Span`, `SpanCandidate`, `SpanWindow`: character-offset span types
//! - `decode_span_logits_*`: GLiNER-style word-span decoding from flat logits
//! - `HandshakingMatrix`, `HandshakingCell`: sparse grid for W2NER / TPLinker

use super::registry::{LabelCategory, LabelDefinition, SemanticRegistry};
use crate::SpanCandidate;

// Span Scoring (GLiNER-style Span Heads)
// =============================================================================

/// A word-level span candidate scored by a GLiNER-style span head.
///
/// Word indices are inclusive: `start_word..=end_word`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScoredSpan {
    pub(crate) start_word: usize,
    pub(crate) end_word: usize,
    pub(crate) class_idx: usize,
    pub(crate) score: f32,
}

/// Derive the span width (`max_width`) encoded in a flattened span tensor.
///
/// GLiNER-style span heads enumerate `num_words * max_width` candidates in
/// row-major order (`start * max_width + width`). `max_width` is a property of
/// the **export**, not a runtime constant: a graph exported with
/// `max_width = 1` can only score single-word spans, while `max_width = 12`
/// scores spans up to 12 words long.
///
/// Returns `None` when the span count cannot be explained by a uniform width
/// (for example an export that omits padded spans).
pub(crate) fn span_width_from_flat_len(num_spans: usize, num_words: usize) -> Option<usize> {
    if num_words == 0 || num_spans < num_words || !num_spans.is_multiple_of(num_words) {
        return None;
    }
    Some(num_spans / num_words)
}

/// Decode span logits shaped `[1, num_words, max_width, num_classes]`.
///
/// The width is read from the tensor shape, so exports wider (or narrower)
/// than any compile-time default decode faithfully.
pub(crate) fn decode_span_logits_4d(
    shape: &[usize],
    logits: &[f32],
    num_words: usize,
    max_classes: usize,
    threshold: f32,
) -> Vec<ScoredSpan> {
    if shape.len() != 4 || shape[0] != 1 {
        return Vec::new();
    }
    let (out_words, out_width, out_classes) = (shape[1], shape[2], shape[3]);
    let num_classes = out_classes.min(max_classes);
    if out_width == 0 || num_classes == 0 {
        return Vec::new();
    }

    let mut spans = Vec::new();
    for word_idx in 0..out_words.min(num_words) {
        for width in 0..out_width {
            let end_word = word_idx + width;
            if end_word >= num_words {
                continue;
            }
            for class_idx in 0..num_classes {
                let idx = word_idx * out_width * out_classes + width * out_classes + class_idx;
                let Some(&logit) = logits.get(idx) else {
                    continue;
                };
                let score = sigmoid(logit);
                if score >= threshold {
                    spans.push(ScoredSpan {
                        start_word: word_idx,
                        end_word,
                        class_idx,
                        score,
                    });
                }
            }
        }
    }
    spans
}

/// Decode flattened span logits shaped `[1, num_spans, num_classes]`.
///
/// `fallback_width` is used only when the span count does not imply a uniform
/// width (`num_spans % num_words != 0`). Callers should prefer a width derived
/// from model metadata; the fallback preserves the historical layout.
pub(crate) fn decode_span_logits_3d(
    shape: &[usize],
    logits: &[f32],
    num_words: usize,
    max_classes: usize,
    threshold: f32,
    fallback_width: usize,
) -> Vec<ScoredSpan> {
    if shape.len() != 3 || shape[0] != 1 {
        return Vec::new();
    }
    let (num_spans, out_classes) = (shape[1], shape[2]);
    let num_classes = out_classes.min(max_classes);
    if num_classes == 0 {
        return Vec::new();
    }
    let width = span_width_from_flat_len(num_spans, num_words).unwrap_or(fallback_width);
    if width == 0 {
        return Vec::new();
    }

    let mut spans = Vec::new();
    for span_idx in 0..num_spans {
        let word_idx = span_idx / width;
        let span_width = span_idx % width;
        let end_word = word_idx + span_width;
        if word_idx >= num_words || end_word >= num_words {
            continue;
        }
        for class_idx in 0..num_classes {
            let idx = span_idx * out_classes + class_idx;
            let Some(&logit) = logits.get(idx) else {
                continue;
            };
            let score = sigmoid(logit);
            if score >= threshold {
                spans.push(ScoredSpan {
                    start_word: word_idx,
                    end_word,
                    class_idx,
                    score,
                });
            }
        }
    }
    spans
}

fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

// Handshaking Matrix (TPLinker-style Joint Extraction)
// =============================================================================

/// Result cell in a handshaking matrix.
#[derive(Debug, Clone, Copy)]
pub struct HandshakingCell {
    /// Row index (token i)
    pub i: u32,
    /// Column index (token j)
    pub j: u32,
    /// Predicted label index
    pub label_idx: u16,
    /// Confidence score
    pub score: f32,
}

/// Handshaking matrix for joint entity-relation extraction.
///
/// # Research Alignment (W2NER, AAAI 2022)
///
/// From the W2NER paper (arXiv:2112.10070):
/// > "We present a novel alternative by modeling the unified NER as word-word
/// > relation classification, namely W2NER. The architecture resolves the kernel
/// > bottleneck of unified NER by effectively modeling the neighboring relations
/// > between entity words with **Next-Neighboring-Word (NNW)** and
/// > **Tail-Head-Word-* (THW-*)** relations."
///
/// In TPLinker/W2NER, we don't just tag tokens - we tag token PAIRS.
/// The matrix M\[i,j\] contains the label for the span (i, j).
///
/// # Key Relations
///
/// | Relation | Description | Purpose |
/// |----------|-------------|---------|
/// | NNW | Next-Neighboring-Word | Links adjacent tokens within entity |
/// | THW-* | Tail-Head-Word | Links end of one entity to start of next |
///
/// # Benefits
///
/// - Overlapping entities (same token in multiple spans)
/// - Joint entity-relation extraction in one pass
/// - Explicit boundary modeling
/// - Handles flat, nested, AND discontinuous NER in one model
pub struct HandshakingMatrix {
    /// Non-zero cells (sparse representation)
    pub cells: Vec<HandshakingCell>,
    /// Sequence length
    pub seq_len: usize,
    /// Number of labels
    pub num_labels: usize,
}

impl HandshakingMatrix {
    /// Create from dense scores with thresholding.
    ///
    /// # Arguments
    /// * `scores` - Dense [seq_len, seq_len, num_labels] scores
    /// * `threshold` - Minimum score to keep
    pub fn from_dense(scores: &[f32], seq_len: usize, num_labels: usize, threshold: f32) -> Self {
        // Performance: Pre-allocate cells vec with estimated capacity
        // Most matrices have sparse cells (only high-scoring ones), so we estimate conservatively
        let estimated_capacity = (seq_len * seq_len / 10).min(1000); // ~10% of cells typically pass threshold
        let mut cells = Vec::with_capacity(estimated_capacity);

        for i in 0..seq_len {
            for j in i..seq_len {
                // Upper triangular (i <= j)
                for l in 0..num_labels {
                    let idx = i * seq_len * num_labels + j * num_labels + l;
                    if idx < scores.len() {
                        let score = scores[idx];
                        if score >= threshold {
                            cells.push(HandshakingCell {
                                i: i as u32,
                                j: j as u32,
                                label_idx: l as u16,
                                score,
                            });
                        }
                    }
                }
            }
        }

        Self {
            cells,
            seq_len,
            num_labels,
        }
    }

    /// Decode entities from handshaking matrix.
    ///
    /// In W2NER convention, cell (i, j) represents a span where:
    /// - j is the start token index
    /// - i is the end token index (inclusive, so we add 1 for exclusive end)
    pub fn decode_entities<'a>(
        &self,
        registry: &'a SemanticRegistry,
    ) -> Vec<(SpanCandidate, &'a LabelDefinition, f32)> {
        let mut entities = Vec::new();

        for cell in &self.cells {
            if let Some(label) = registry.labels.get(cell.label_idx as usize) {
                if label.category == LabelCategory::Entity {
                    // W2NER: j=start, i=end (inclusive), so span is [j, i+1)
                    entities.push((SpanCandidate::new(0, cell.j, cell.i + 1), label, cell.score));
                }
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort by position, then by score (descending)
        entities.sort_unstable_by(|a, b| {
            a.0.start
                .cmp(&b.0.start)
                .then_with(|| a.0.end.cmp(&b.0.end))
                .then_with(|| b.2.total_cmp(&a.2))
        });

        // Performance: Pre-allocate kept vec with estimated capacity
        // Non-maximum suppression
        let mut kept = Vec::with_capacity(entities.len().min(32));
        for (span, label, score) in entities {
            let overlaps = kept.iter().any(|(s, _, _): &(SpanCandidate, _, _)| {
                !(span.end <= s.start || s.end <= span.start)
            });
            if !overlaps {
                kept.push((span, label, score));
            }
        }

        kept
    }
}

// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::inference::registry::ModalityHint;
    use crate::Confidence;
    use std::collections::HashMap;

    #[test]
    fn handshaking_from_dense_thresholding() {
        // 2 tokens, 1 label => dense shape [2, 2, 1].
        // Only upper-triangular (i <= j) cells are visited.
        let scores = vec![
            0.1, // (0,0) -- below threshold
            0.9, // (0,1) -- above
            0.0, // (1,0) -- lower triangle, not visited
            0.5, // (1,1) -- at threshold
        ];
        let matrix = HandshakingMatrix::from_dense(&scores, 2, 1, 0.5);
        assert_eq!(matrix.cells.len(), 2);
        assert!((matrix.cells[0].score - 0.9).abs() < 1e-6);
        assert_eq!(matrix.cells[0].i, 0);
        assert_eq!(matrix.cells[0].j, 1);
        assert!((matrix.cells[1].score - 0.5).abs() < 1e-6);
        assert_eq!(matrix.cells[1].i, 1);
        assert_eq!(matrix.cells[1].j, 1);
    }

    #[test]
    fn handshaking_empty_when_all_below_threshold() {
        let scores = vec![0.1, 0.2, 0.0, 0.3];
        let matrix = HandshakingMatrix::from_dense(&scores, 2, 1, 0.5);
        assert!(matrix.cells.is_empty());
    }

    #[test]
    fn handshaking_decode_nms_removes_overlapping() {
        // Build a minimal registry with one Entity label.
        let registry = SemanticRegistry {
            embeddings: vec![0.0; 4],
            hidden_dim: 4,
            labels: vec![LabelDefinition {
                slug: "PER".to_string(),
                description: "Person".to_string(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: Confidence::ZERO,
            }],
            label_index: {
                let mut m = HashMap::new();
                m.insert("PER".to_string(), 0);
                m
            },
        };

        // Two overlapping cells: span [0,3) score 0.9 and span [1,4) score 0.8.
        // W2NER convention: cell.j = start, cell.i = end (inclusive), decoded as [j, i+1).
        let matrix = HandshakingMatrix {
            cells: vec![
                HandshakingCell {
                    i: 2,
                    j: 0,
                    label_idx: 0,
                    score: 0.9,
                },
                HandshakingCell {
                    i: 3,
                    j: 1,
                    label_idx: 0,
                    score: 0.8,
                },
            ],
            seq_len: 5,
            num_labels: 1,
        };

        let entities = matrix.decode_entities(&registry);
        // Spans [0,3) and [1,4) overlap; NMS keeps only the higher-score one.
        assert_eq!(entities.len(), 1, "NMS should suppress overlapping span");
        assert_eq!(entities[0].0.start, 0);
        assert_eq!(entities[0].0.end, 3);
        assert!((entities[0].2 - 0.9).abs() < 1e-6);
    }

    /// Raw scores can be NaN (cells are public); the decode sort must not
    /// panic and must still return a well-formed, non-overlapping result.
    #[test]
    fn handshaking_decode_with_nan_scores_does_not_panic() {
        let registry = SemanticRegistry {
            embeddings: vec![0.0; 4],
            hidden_dim: 4,
            labels: vec![LabelDefinition {
                slug: "PER".to_string(),
                description: "Person".to_string(),
                category: LabelCategory::Entity,
                modality: ModalityHint::TextOnly,
                threshold: Confidence::ZERO,
            }],
            label_index: HashMap::from([("PER".to_string(), 0)]),
        };
        // 60 cells sharing a few (start, end) keys so the score tie-break
        // is exercised, a third of them NaN.
        let cells = (0..60u32)
            .map(|k| HandshakingCell {
                i: 3 + k % 3,
                j: k % 2,
                label_idx: 0,
                score: if k.is_multiple_of(3) {
                    f32::NAN
                } else {
                    (k % 7) as f32 / 7.0
                },
            })
            .collect();
        let matrix = HandshakingMatrix {
            cells,
            seq_len: 8,
            num_labels: 1,
        };
        let entities = matrix.decode_entities(&registry);
        assert!(!entities.is_empty());
        for (a, b) in entities.iter().zip(entities.iter().skip(1)) {
            assert!(a.0.end <= b.0.start || b.0.end <= a.0.start, "overlap kept");
        }
    }

    // --- GLiNER span-head decoding -----------------------------------------

    fn span_words(spans: &[ScoredSpan]) -> Vec<(usize, usize)> {
        spans.iter().map(|s| (s.start_word, s.end_word)).collect()
    }

    #[test]
    fn span_width_is_derived_from_export_span_count() {
        assert_eq!(span_width_from_flat_len(36, 3), Some(12));
        assert_eq!(span_width_from_flat_len(3, 3), Some(1));
        assert_eq!(span_width_from_flat_len(8, 8), Some(1));
        assert_eq!(span_width_from_flat_len(4, 3), None);
    }

    #[test]
    fn decode_4d_keeps_two_word_entity() {
        // "New York City" with a 12-wide export: the span [0, 2] is scored.
        let num_words = 3;
        let width = 12;
        let mut logits = vec![-10.0f32; num_words * width];
        // start 0, width 2 -> "New York City".
        logits[2] = 10.0;
        let spans = decode_span_logits_4d(&[1, num_words, width, 1], &logits, num_words, 1, 0.5);
        assert_eq!(span_words(&spans), vec![(0, 2)]);
    }

    #[test]
    fn decode_4d_width_one_cannot_represent_multi_word_span() {
        // A `max_width=1` export has one span slot per word, so
        // "New York City" can only come back as three separate words. This is
        // the reported degradation; the decoder cannot recover a span the
        // graph never scored, hence the export requirement in docs/BACKENDS.md.
        let num_words = 3;
        let logits = vec![10.0f32; num_words];
        let spans = decode_span_logits_4d(&[1, num_words, 1, 1], &logits, num_words, 1, 0.5);
        assert_eq!(span_words(&spans), vec![(0, 0), (1, 1), (2, 2)]);
        assert!(
            !spans.iter().any(|s| s.end_word > s.start_word),
            "no multi-word span can exist in a max_width=1 export"
        );
    }

    #[test]
    fn decode_4d_honours_width_beyond_the_runtime_default() {
        // A 16-wide export scores a 14-word span; clamping enumeration to the
        // 12-wide runtime default would silently drop it.
        let num_words = 16;
        let width = 16;
        let mut logits = vec![-10.0f32; num_words * width];
        logits[13] = 10.0;
        let spans = decode_span_logits_4d(&[1, num_words, width, 1], &logits, num_words, 1, 0.5);
        assert_eq!(span_words(&spans), vec![(0, 13)]);
    }

    #[test]
    fn decode_3d_derives_width_instead_of_dividing_by_a_constant() {
        // Flat `max_width=1` export: `num_spans == num_words`. The old decoder
        // divided the flat index by the compile-time 12, so the confidently
        // scored span 6 ("New") was reported as words [0, 6] and every span
        // beyond word 0 was effectively lost. Deriving width = 8/8 = 1 maps it
        // back to [6, 6].
        let num_words = 8;
        let mut logits = vec![-10.0f32; num_words];
        logits[6] = 10.0;
        let spans = decode_span_logits_3d(&[1, num_words, 1], &logits, num_words, 1, 0.5, 12);
        assert_eq!(span_words(&spans), vec![(6, 6)]);
    }

    #[test]
    fn decode_3d_width_twelve_keeps_two_word_entity() {
        let num_words = 3;
        let width = 12;
        let num_spans = num_words * width;
        let mut logits = vec![-10.0f32; num_spans];
        // start * width + span_width = 0 * 12 + 2 -> "New York City".
        logits[2] = 10.0;
        let spans = decode_span_logits_3d(&[1, num_spans, 1], &logits, num_words, 1, 0.5, width);
        assert_eq!(span_words(&spans), vec![(0, 2)]);
    }
}
