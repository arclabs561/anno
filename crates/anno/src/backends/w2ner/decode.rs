//! W2NER decoding algorithms.
//!
//! Standalone pure functions for decoding word-word relation grids into entity spans.
//! These are separated from the W2NER model struct so they can be used independently
//! with pre-computed grids (e.g., from external inference or testing).
//!
//! # Algorithm reference
//! - arXiv:2112.10070 §3.3 (Li et al., "Unified Named Entity Recognition as Word-Word
//!   Relation Classification", AAAI 2022)

use crate::backends::inference::{HandshakingCell, HandshakingMatrix};
use crate::{EntityCategory, EntityType};

/// Decoded row from the discontinuous entity algorithm:
/// `(entity_type_label, word_spans, score)` where each span is
/// `(word_start, word_end_exclusive)`.
pub type DiscontinuousDecodeRow = (String, Vec<(usize, usize)>, f64);

/// W2NER word-word relation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum W2NERRelation {
    /// Next-Neighboring-Word: tokens are adjacent in same entity.
    NNW,
    /// Tail-Head-Word: marks entity boundary (tail → head).
    THW,
    /// No relation between tokens.
    None,
}

impl W2NERRelation {
    /// Convert from label index (0=None, 1=NNW, 2=THW).
    #[must_use]
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::None,
            1 => Self::NNW,
            2 => Self::THW,
            _ => Self::None,
        }
    }

    /// Convert to label index.
    #[must_use]
    pub fn to_index(self) -> usize {
        match self {
            Self::None => 0,
            Self::NNW => 1,
            Self::THW => 2,
        }
    }
}

// =============================================================================
// Decode algorithms
// =============================================================================

/// Decode contiguous entity spans from a handshaking matrix.
///
/// Finds all THW(tail, head) cells of entity type `entity_type_idx` above
/// `threshold`, sorts by start position, and optionally removes nested spans
/// (outermost wins when `allow_nested` is false).
///
/// Label layout follows the reference implementation: 0 = None, 1 = NNW, and
/// `2 + k` = THW for entity type `k`. A grid with a single THW label (index 2)
/// therefore has exactly one entity type, index 0.
///
/// Returns `Vec<(word_start, word_end_exclusive, score)>`.
#[must_use]
pub fn decode_from_matrix(
    matrix: &HandshakingMatrix,
    tokens: &[&str],
    entity_type_idx: usize,
    threshold: f32,
    allow_nested: bool,
) -> Vec<(usize, usize, f64)> {
    let mut entities = Vec::with_capacity(16);

    for cell in &matrix.cells {
        let label = cell.label_idx as usize;
        if label >= 2 && label - 2 == entity_type_idx && cell.score >= threshold {
            let tail = cell.i as usize;
            let head = cell.j as usize;
            if head <= tail && head < tokens.len() && tail < tokens.len() {
                entities.push((head, tail + 1, cell.score as f64));
            }
        }
    }

    entities.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| (b.1 - b.0).cmp(&(a.1 - a.0))));

    if !allow_nested {
        entities = remove_nested(&entities);
    }

    entities
}

/// Decode discontinuous entity spans using the full NNW+THW algorithm (§3.3).
///
/// THW cells identify entity boundaries; NNW cells identify adjacent-word connections
/// within the same entity. Gaps in the NNW chain produce disjoint sub-spans.
///
/// `first_label` is used as the entity-type string when the model uses a single grid;
/// pass an empty string to fall back to `"ENTITY"`.
#[must_use]
pub fn decode_discontinuous_from_matrix(
    matrix: &HandshakingMatrix,
    tokens: &[&str],
    threshold: f32,
    first_label: &str,
) -> Vec<DiscontinuousDecodeRow> {
    let n = tokens.len();

    let mut entity_boundaries: Vec<(usize, usize, f64)> = Vec::new();
    for cell in &matrix.cells {
        if W2NERRelation::from_index(cell.label_idx as usize) == W2NERRelation::THW
            && cell.score >= threshold
        {
            let tail = cell.i as usize;
            let head = cell.j as usize;
            if head <= tail && tail < n {
                entity_boundaries.push((head, tail, cell.score as f64));
            }
        }
    }

    let mut nnw: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for cell in &matrix.cells {
        if W2NERRelation::from_index(cell.label_idx as usize) == W2NERRelation::NNW
            && cell.score >= threshold
        {
            let a = cell.i as usize;
            let b = cell.j as usize;
            nnw.insert((a, b));
            nnw.insert((b, a));
        }
    }

    let mut results: Vec<DiscontinuousDecodeRow> = Vec::new();
    let type_label = if first_label.is_empty() {
        "ENTITY".to_string()
    } else {
        first_label.to_string()
    };

    for (head, tail, score) in entity_boundaries {
        // An entity is a chain of NNW links from its head word to its tail
        // word; the links may skip words that are not part of the entity.
        let paths = nnw_paths(&nnw, head, tail);
        if paths.is_empty() {
            // No connecting chain: keep the boundary and split it wherever the
            // adjacent-word NNW link is missing.
            let mut segments: Vec<(usize, usize)> = Vec::new();
            let mut seg_start = head;
            for i in head..tail {
                let j = i + 1;
                if !nnw.contains(&(i, j)) {
                    segments.push((seg_start, i + 1));
                    seg_start = j;
                }
            }
            segments.push((seg_start, tail + 1));
            results.push((type_label.clone(), segments, score));
            continue;
        }
        for path in paths {
            results.push((type_label.clone(), words_to_segments(&path), score));
        }
    }

    results.sort_unstable_by(|a, b| {
        let a_start = a.1.first().map(|s| s.0).unwrap_or(usize::MAX);
        let b_start = b.1.first().map(|s| s.0).unwrap_or(usize::MAX);
        let a_len: usize = a.1.iter().map(|(s, e)| e - s).sum();
        let b_len: usize = b.1.iter().map(|(s, e)| e - s).sum();
        a_start.cmp(&b_start).then_with(|| b_len.cmp(&a_len))
    });

    results
}

/// Upper bound on the NNW paths enumerated for one THW boundary.
const MAX_NNW_PATHS: usize = 64;

/// All increasing word paths from `head` to `tail` along NNW links.
///
/// `nnw` holds each link in both directions; only forward steps (to a later
/// word) are followed, so every path is finite.
fn nnw_paths(
    nnw: &std::collections::HashSet<(usize, usize)>,
    head: usize,
    tail: usize,
) -> Vec<Vec<usize>> {
    if head == tail {
        return vec![vec![head]];
    }
    let mut next: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for &(a, b) in nnw {
        if a < b && a >= head && b <= tail {
            next.entry(a).or_default().push(b);
        }
    }
    for targets in next.values_mut() {
        targets.sort_unstable();
    }
    let mut paths = Vec::new();
    let mut stack = vec![vec![head]];
    while let Some(path) = stack.pop() {
        let last = *path.last().unwrap_or(&head);
        if last == tail {
            paths.push(path);
            if paths.len() >= MAX_NNW_PATHS {
                break;
            }
            continue;
        }
        if let Some(targets) = next.get(&last) {
            for &t in targets.iter().rev() {
                let mut p = path.clone();
                p.push(t);
                stack.push(p);
            }
        }
    }
    paths
}

/// Group sorted word indices into `(start, end_exclusive)` runs.
fn words_to_segments(words: &[usize]) -> Vec<(usize, usize)> {
    let mut segments: Vec<(usize, usize)> = Vec::new();
    for &w in words {
        match segments.last_mut() {
            Some(seg) if seg.1 == w => seg.1 = w + 1,
            _ => segments.push((w, w + 1)),
        }
    }
    segments
}

/// Convert a dense `[seq_len × seq_len × num_relations]` grid to a sparse matrix.
///
/// Each cell keeps only its argmax label, scored by its softmax probability.
/// Cells whose argmax is the None label (index 0) or whose probability is
/// below `threshold` are dropped. `grid` holds raw logits, as produced by
/// `scripts/export_w2ner_to_onnx.py`.
#[must_use]
pub fn grid_to_matrix(
    grid: &[f32],
    seq_len: usize,
    num_relations: usize,
    threshold: f32,
) -> HandshakingMatrix {
    let mut cells = Vec::new();
    if num_relations > 0 {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let base = (i * seq_len + j) * num_relations;
                let Some(logits) = grid.get(base..base + num_relations) else {
                    continue;
                };
                let mut best = 0;
                for (rel, &v) in logits.iter().enumerate() {
                    if v > logits[best] {
                        best = rel;
                    }
                }
                if best == 0 {
                    continue;
                }
                let max = logits[best];
                let denom: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
                let score = 1.0 / denom;
                if score >= threshold {
                    cells.push(HandshakingCell {
                        i: i as u32,
                        j: j as u32,
                        label_idx: best as u16,
                        score,
                    });
                }
            }
        }
    }
    HandshakingMatrix {
        cells,
        seq_len,
        num_labels: num_relations,
    }
}

/// Remove nested entities, keeping the outermost span at each position.
pub(crate) fn remove_nested(entities: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64)> {
    let mut result = Vec::new();
    let mut last_end = 0;
    for &(start, end, score) in entities {
        if start >= last_end {
            result.push((start, end, score));
            last_end = end;
        }
    }
    result
}

/// Map a label string to the canonical `EntityType`.
#[must_use]
pub fn map_label_to_entity_type(label: &str) -> EntityType {
    match label.to_uppercase().as_str() {
        "PER" | "PERSON" => EntityType::Person,
        "ORG" | "ORGANIZATION" => EntityType::Organization,
        "LOC" | "LOCATION" | "GPE" => EntityType::Location,
        "DATE" => EntityType::Date,
        "TIME" => EntityType::Time,
        "MONEY" => EntityType::Money,
        "PERCENT" => EntityType::Percent,
        "MISC" => EntityType::custom("MISC", EntityCategory::Misc),
        _ => EntityType::custom(label, EntityCategory::Misc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::inference::{HandshakingCell, HandshakingMatrix};

    fn cell(i: u32, j: u32, rel: W2NERRelation, score: f32) -> HandshakingCell {
        HandshakingCell {
            i,
            j,
            label_idx: rel.to_index() as u16,
            score,
        }
    }

    fn mat(cells: Vec<HandshakingCell>, seq_len: usize) -> HandshakingMatrix {
        HandshakingMatrix {
            cells,
            seq_len,
            num_labels: 3,
        }
    }

    #[test]
    fn decode_single_contiguous_entity() {
        // THW(tail=2, head=0) → entity spans words 0..=2
        let tokens = ["New", "York", "City"];
        let m = mat(vec![cell(2, 0, W2NERRelation::THW, 0.9)], 3);
        let result = decode_from_matrix(&m, &tokens, 0, 0.5, true);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0); // word start
        assert_eq!(result[0].1, 3); // word end (exclusive)
    }

    #[test]
    fn decode_removes_nested_when_disabled() {
        let tokens = ["The", "University", "of", "California"];
        // outer: THW(3,0), inner: THW(3,1)
        let m = mat(
            vec![
                cell(3, 0, W2NERRelation::THW, 0.8),
                cell(3, 1, W2NERRelation::THW, 0.9),
            ],
            4,
        );
        let nested = decode_from_matrix(&m, &tokens, 0, 0.5, true);
        assert_eq!(nested.len(), 2, "should keep both when nested=true");

        let flat = decode_from_matrix(&m, &tokens, 0, 0.5, false);
        assert_eq!(flat.len(), 1, "should keep only outer when nested=false");
    }

    #[test]
    fn decode_discontinuous_splits_on_nnw_gap() {
        // Entity: head=0, tail=3, but no NNW between words 1-2 → two segments
        let tokens = ["severe", "pain", "in", "abdomen"];
        let m = mat(
            vec![
                cell(3, 0, W2NERRelation::THW, 0.8),
                cell(0, 1, W2NERRelation::NNW, 0.8),
                // no NNW between 1-2
                cell(2, 3, W2NERRelation::NNW, 0.8),
            ],
            4,
        );
        let result = decode_discontinuous_from_matrix(&m, &tokens, 0.5, "SYMPTOM");
        assert_eq!(result.len(), 1);
        let (label, spans, _score) = &result[0];
        assert_eq!(label, "SYMPTOM");
        assert_eq!(
            spans.len(),
            2,
            "expected 2 disjoint segments; got {}",
            spans.len()
        );
        assert_eq!(spans[0], (0, 2)); // words 0-1
        assert_eq!(spans[1], (2, 4)); // words 2-3
    }

    #[test]
    fn grid_to_matrix_filters_none_and_below_threshold() {
        // 2×2×3 grid: only rel=2 (THW) at (0,1) with score 0.9 should survive
        let mut grid = vec![0.0f32; 2 * 2 * 3];
        grid[5] = 0.9; // (i=0,j=1,rel=2)
        grid[4] = 0.2; // below threshold
        let m = grid_to_matrix(&grid, 2, 3, 0.5);
        assert_eq!(m.cells.len(), 1);
        assert_eq!(m.cells[0].label_idx, 2);
    }

    #[test]
    fn decode_keeps_only_the_requested_thw_type() {
        // Labels: 0 None, 1 NNW, 2 THW-type0, 3 THW-type1.
        let tokens = ["Ada", "joined", "Acme"];
        let m = HandshakingMatrix {
            cells: vec![
                HandshakingCell {
                    i: 0,
                    j: 0,
                    label_idx: 2,
                    score: 0.9,
                },
                HandshakingCell {
                    i: 2,
                    j: 2,
                    label_idx: 3,
                    score: 0.9,
                },
            ],
            seq_len: 3,
            num_labels: 4,
        };
        assert_eq!(
            decode_from_matrix(&m, &tokens, 0, 0.5, true),
            vec![(0, 1, 0.9f32 as f64)]
        );
        assert_eq!(
            decode_from_matrix(&m, &tokens, 1, 0.5, true),
            vec![(2, 3, 0.9f32 as f64)]
        );
    }

    #[test]
    fn grid_to_matrix_keeps_argmax_label_only() {
        // One cell over 4 labels; logits favour label 3, label 2 is also > 0.5.
        let grid = [0.0f32, 0.0, 2.0, 5.0];
        let m = grid_to_matrix(&grid, 1, 4, 0.5);
        assert_eq!(m.cells.len(), 1);
        assert_eq!(m.cells[0].label_idx, 3);
    }

    #[test]
    fn map_label_person_org_loc() {
        use crate::EntityType;
        assert_eq!(map_label_to_entity_type("PER"), EntityType::Person);
        assert_eq!(map_label_to_entity_type("ORG"), EntityType::Organization);
        assert_eq!(map_label_to_entity_type("GPE"), EntityType::Location);
        assert!(matches!(
            map_label_to_entity_type("CUSTOM"),
            EntityType::Custom { .. }
        ));
    }
}
