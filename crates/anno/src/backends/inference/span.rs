//! Span representation types and the handshaking matrix for joint extraction.
//!
//! - `Span`, `SpanCandidate`, `SpanWindow`: character-offset span types
//! - `HandshakingMatrix`, `HandshakingCell`: sparse grid for W2NER / TPLinker

use super::registry::{LabelCategory, LabelDefinition, SemanticRegistry};
use anno_core::SpanCandidate;

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
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
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
}
