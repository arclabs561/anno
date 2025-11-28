//! Visual NER evaluation metrics.
//!
//! # Overview
//!
//! Visual NER (VisualNER) extracts entities from images and documents,
//! producing entities with both text spans and bounding box locations.
//!
//! # Evaluation Metrics
//!
//! From VisualNER benchmarks (FUNSD, SROIE, CORD):
//!
//! | Metric | Description |
//! |--------|-------------|
//! | Text F1 | Standard NER F1 on extracted text |
//! | Box IoU | Intersection-over-Union of bounding boxes |
//! | End-to-End F1 | Correct text AND box (>50% IoU) |
//!
//! # Benchmark Datasets
//!
//! | Dataset | Domain | Text F1 | E2E F1 |
//! |---------|--------|---------|--------|
//! | FUNSD | Forms | ~85% | ~78% |
//! | SROIE | Receipts | ~94% | ~90% |
//! | CORD | Receipts | ~96% | ~92% |
//! | DocVQA | Documents | ~78% | N/A |
//!
//! # Research Alignment
//!
//! From LayoutLMv3 paper (arXiv:2204.08387):
//! > "Pre-training strategies that align text, layout, and image modalities
//! > significantly improve document understanding."

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// BOUNDING BOX TYPES
// =============================================================================

/// A bounding box in normalized coordinates (0.0-1.0).
///
/// Using normalized coordinates allows consistent evaluation
/// regardless of image resolution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Left coordinate (0.0-1.0)
    pub x1: f32,
    /// Top coordinate (0.0-1.0)
    pub y1: f32,
    /// Right coordinate (0.0-1.0)
    pub x2: f32,
    /// Bottom coordinate (0.0-1.0)
    pub y2: f32,
}

impl BoundingBox {
    /// Create a new bounding box.
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Calculate area of the bounding box.
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    /// Calculate Intersection-over-Union (IoU) with another box.
    ///
    /// IoU = intersection_area / union_area
    /// Range: 0.0 (no overlap) to 1.0 (identical boxes)
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    /// Check if this box substantially overlaps with another (IoU >= threshold).
    pub fn overlaps(&self, other: &BoundingBox, threshold: f32) -> bool {
        self.iou(other) >= threshold
    }
}

// =============================================================================
// VISUAL ENTITY TYPES
// =============================================================================

/// A gold standard visual entity with text and bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualGold {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: String,
    /// Bounding box (normalized coordinates)
    pub bbox: BoundingBox,
}

impl VisualGold {
    /// Create a new visual gold entity.
    pub fn new(text: impl Into<String>, entity_type: impl Into<String>, bbox: BoundingBox) -> Self {
        Self {
            text: text.into(),
            entity_type: entity_type.into(),
            bbox,
        }
    }
}

/// A predicted visual entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualPrediction {
    /// Extracted text
    pub text: String,
    /// Predicted entity type
    pub entity_type: String,
    /// Predicted bounding box
    pub bbox: BoundingBox,
    /// Confidence score
    pub confidence: f32,
}

// =============================================================================
// EVALUATION CONFIG
// =============================================================================

/// Configuration for visual NER evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEvalConfig {
    /// IoU threshold for box matching (default: 0.5)
    pub iou_threshold: f32,
    /// Whether text match should be case-insensitive
    pub case_insensitive: bool,
    /// Whether to normalize whitespace in text comparison
    pub normalize_whitespace: bool,
    /// Whether entity type must match for credit
    pub require_type_match: bool,
}

impl Default for VisualEvalConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            case_insensitive: false,
            normalize_whitespace: true,
            require_type_match: true,
        }
    }
}

// =============================================================================
// METRICS
// =============================================================================

/// Visual NER evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualNERMetrics {
    // Text-only metrics (ignoring boxes)
    /// Text-only precision
    pub text_precision: f64,
    /// Text-only recall
    pub text_recall: f64,
    /// Text-only F1
    pub text_f1: f64,

    // Box-only metrics (ignoring text)
    /// Mean IoU across matched boxes
    pub mean_iou: f64,
    /// Box precision (boxes above IoU threshold)
    pub box_precision: f64,
    /// Box recall
    pub box_recall: f64,
    /// Box F1
    pub box_f1: f64,

    // End-to-end metrics (text AND box must match)
    /// End-to-end precision
    pub e2e_precision: f64,
    /// End-to-end recall
    pub e2e_recall: f64,
    /// End-to-end F1
    pub e2e_f1: f64,

    // Per-type breakdown
    /// Metrics per entity type
    pub per_type: HashMap<String, VisualTypeMetrics>,

    // Counts
    /// Number of predicted entities
    pub num_predicted: usize,
    /// Number of gold entities
    pub num_gold: usize,
    /// Text matches
    pub text_matches: usize,
    /// Box matches (IoU >= threshold)
    pub box_matches: usize,
    /// End-to-end matches (text AND box)
    pub e2e_matches: usize,
}

/// Per-type visual metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualTypeMetrics {
    /// Entity type
    pub entity_type: String,
    /// Text F1
    pub text_f1: f64,
    /// Box F1
    pub box_f1: f64,
    /// End-to-end F1
    pub e2e_f1: f64,
    /// Support (gold count)
    pub support: usize,
}

// =============================================================================
// EVALUATION FUNCTION
// =============================================================================

/// Evaluate visual NER predictions against gold standard.
///
/// # Arguments
///
/// * `gold` - Gold standard entities
/// * `pred` - Predicted entities
/// * `config` - Evaluation configuration
///
/// # Returns
///
/// Comprehensive visual NER metrics.
pub fn evaluate_visual_ner(
    gold: &[VisualGold],
    pred: &[VisualPrediction],
    config: &VisualEvalConfig,
) -> VisualNERMetrics {
    let mut text_matches = 0;
    let mut box_matches = 0;
    let mut e2e_matches = 0;
    let mut iou_sum = 0.0f64;
    let mut iou_count = 0;

    let mut type_stats: HashMap<String, (usize, usize, usize, usize, usize)> = HashMap::new();

    // Track which gold entities have been matched
    let mut gold_text_matched = vec![false; gold.len()];
    let mut gold_box_matched = vec![false; gold.len()];
    let mut gold_e2e_matched = vec![false; gold.len()];

    // Initialize type stats
    for g in gold {
        type_stats.entry(g.entity_type.clone()).or_insert((0, 0, 0, 0, 0)).0 += 1;
    }
    for p in pred {
        type_stats.entry(p.entity_type.clone()).or_insert((0, 0, 0, 0, 0)).1 += 1;
    }

    // Match predictions to gold
    for p in pred {
        let pred_text = normalize_text(&p.text, config);

        for (g_idx, g) in gold.iter().enumerate() {
            // Check type match if required
            if config.require_type_match && p.entity_type != g.entity_type {
                continue;
            }

            let gold_text = normalize_text(&g.text, config);
            let text_match = pred_text == gold_text;
            let iou = p.bbox.iou(&g.bbox);
            let box_match = iou >= config.iou_threshold;

            // Update IoU stats for any overlapping boxes
            if iou > 0.0 {
                iou_sum += iou as f64;
                iou_count += 1;
            }

            // Text match
            if text_match && !gold_text_matched[g_idx] {
                gold_text_matched[g_idx] = true;
                text_matches += 1;
                if let Some(stats) = type_stats.get_mut(&g.entity_type) {
                    stats.2 += 1;
                }
            }

            // Box match
            if box_match && !gold_box_matched[g_idx] {
                gold_box_matched[g_idx] = true;
                box_matches += 1;
                if let Some(stats) = type_stats.get_mut(&g.entity_type) {
                    stats.3 += 1;
                }
            }

            // End-to-end match (both text AND box)
            if text_match && box_match && !gold_e2e_matched[g_idx] {
                gold_e2e_matched[g_idx] = true;
                e2e_matches += 1;
                if let Some(stats) = type_stats.get_mut(&g.entity_type) {
                    stats.4 += 1;
                }
                break; // Found a complete match, move to next prediction
            }
        }
    }

    // Calculate metrics
    let num_gold = gold.len();
    let num_pred = pred.len();

    let text_precision = if num_pred > 0 { text_matches as f64 / num_pred as f64 } else { 0.0 };
    let text_recall = if num_gold > 0 { text_matches as f64 / num_gold as f64 } else { 0.0 };
    let text_f1 = f1(text_precision, text_recall);

    let box_precision = if num_pred > 0 { box_matches as f64 / num_pred as f64 } else { 0.0 };
    let box_recall = if num_gold > 0 { box_matches as f64 / num_gold as f64 } else { 0.0 };
    let box_f1 = f1(box_precision, box_recall);

    let e2e_precision = if num_pred > 0 { e2e_matches as f64 / num_pred as f64 } else { 0.0 };
    let e2e_recall = if num_gold > 0 { e2e_matches as f64 / num_gold as f64 } else { 0.0 };
    let e2e_f1 = f1(e2e_precision, e2e_recall);

    let mean_iou = if iou_count > 0 { iou_sum / iou_count as f64 } else { 0.0 };

    // Per-type metrics
    let per_type: HashMap<_, _> = type_stats
        .into_iter()
        .map(|(et, (gold_count, pred_count, text_tp, box_tp, e2e_tp))| {
            let text_f1 = if gold_count > 0 && pred_count > 0 {
                let p = text_tp as f64 / pred_count as f64;
                let r = text_tp as f64 / gold_count as f64;
                f1(p, r)
            } else {
                0.0
            };
            let box_f1 = if gold_count > 0 && pred_count > 0 {
                let p = box_tp as f64 / pred_count as f64;
                let r = box_tp as f64 / gold_count as f64;
                f1(p, r)
            } else {
                0.0
            };
            let e2e_f1 = if gold_count > 0 && pred_count > 0 {
                let p = e2e_tp as f64 / pred_count as f64;
                let r = e2e_tp as f64 / gold_count as f64;
                f1(p, r)
            } else {
                0.0
            };
            (
                et.clone(),
                VisualTypeMetrics {
                    entity_type: et,
                    text_f1,
                    box_f1,
                    e2e_f1,
                    support: gold_count,
                },
            )
        })
        .collect();

    VisualNERMetrics {
        text_precision,
        text_recall,
        text_f1,
        mean_iou,
        box_precision,
        box_recall,
        box_f1,
        e2e_precision,
        e2e_recall,
        e2e_f1,
        per_type,
        num_predicted: num_pred,
        num_gold,
        text_matches,
        box_matches,
        e2e_matches,
    }
}

// =============================================================================
// HELPERS
// =============================================================================

fn normalize_text(text: &str, config: &VisualEvalConfig) -> String {
    let mut s = text.to_string();
    if config.case_insensitive {
        s = s.to_lowercase();
    }
    if config.normalize_whitespace {
        s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    s
}

fn f1(precision: f64, recall: f64) -> f64 {
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

// =============================================================================
// SYNTHETIC DATA
// =============================================================================

/// Generate synthetic visual NER examples for testing.
///
/// Returns examples with made-up bounding boxes for unit testing.
pub fn synthetic_visual_examples() -> Vec<(String, Vec<VisualGold>)> {
    vec![
        (
            "Invoice #12345".to_string(),
            vec![
                VisualGold::new(
                    "Invoice #12345",
                    "DOCUMENT_ID",
                    BoundingBox::new(0.1, 0.05, 0.4, 0.1),
                ),
            ],
        ),
        (
            "Total: $1,234.56\nDate: 2024-01-15".to_string(),
            vec![
                VisualGold::new(
                    "$1,234.56",
                    "MONEY",
                    BoundingBox::new(0.5, 0.8, 0.7, 0.85),
                ),
                VisualGold::new(
                    "2024-01-15",
                    "DATE",
                    BoundingBox::new(0.5, 0.7, 0.7, 0.75),
                ),
            ],
        ),
        (
            "Acme Corp\n123 Main St, City".to_string(),
            vec![
                VisualGold::new(
                    "Acme Corp",
                    "ORG",
                    BoundingBox::new(0.1, 0.1, 0.35, 0.15),
                ),
                VisualGold::new(
                    "123 Main St, City",
                    "ADDRESS",
                    BoundingBox::new(0.1, 0.16, 0.5, 0.21),
                ),
            ],
        ),
    ]
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_area() {
        let bbox = BoundingBox::new(0.0, 0.0, 0.5, 0.5);
        assert!((bbox.area() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_bounding_box_iou_identical() {
        let bbox1 = BoundingBox::new(0.1, 0.1, 0.5, 0.5);
        let bbox2 = BoundingBox::new(0.1, 0.1, 0.5, 0.5);
        assert!((bbox1.iou(&bbox2) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bounding_box_iou_no_overlap() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 0.2, 0.2);
        let bbox2 = BoundingBox::new(0.5, 0.5, 0.7, 0.7);
        assert!(bbox1.iou(&bbox2) < 0.001);
    }

    #[test]
    fn test_bounding_box_iou_partial() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 0.5, 0.5);
        let bbox2 = BoundingBox::new(0.25, 0.25, 0.75, 0.75);
        let iou = bbox1.iou(&bbox2);
        // Intersection: 0.25x0.25 = 0.0625
        // Union: 0.25 + 0.25 - 0.0625 = 0.4375
        // IoU: 0.0625 / 0.4375 ≈ 0.143
        assert!(iou > 0.1 && iou < 0.2);
    }

    #[test]
    fn test_evaluate_perfect_match() {
        let gold = vec![
            VisualGold::new("Invoice", "DOC", BoundingBox::new(0.1, 0.1, 0.3, 0.15)),
        ];
        let pred = vec![
            VisualPrediction {
                text: "Invoice".to_string(),
                entity_type: "DOC".to_string(),
                bbox: BoundingBox::new(0.1, 0.1, 0.3, 0.15),
                confidence: 0.95,
            },
        ];

        let config = VisualEvalConfig::default();
        let metrics = evaluate_visual_ner(&gold, &pred, &config);

        assert!((metrics.text_f1 - 1.0).abs() < 0.001);
        assert!((metrics.e2e_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_evaluate_text_only_match() {
        let gold = vec![
            VisualGold::new("Invoice", "DOC", BoundingBox::new(0.1, 0.1, 0.3, 0.15)),
        ];
        let pred = vec![
            VisualPrediction {
                text: "Invoice".to_string(),
                entity_type: "DOC".to_string(),
                bbox: BoundingBox::new(0.5, 0.5, 0.7, 0.6), // Different box
                confidence: 0.95,
            },
        ];

        let config = VisualEvalConfig::default();
        let metrics = evaluate_visual_ner(&gold, &pred, &config);

        assert!((metrics.text_f1 - 1.0).abs() < 0.001);
        assert!(metrics.e2e_f1 < 0.5); // E2E should fail due to box mismatch
    }

    #[test]
    fn test_synthetic_examples_valid() {
        let examples = synthetic_visual_examples();
        assert!(!examples.is_empty());

        for (text, entities) in &examples {
            assert!(!text.is_empty());
            for entity in entities {
                // Valid bounding box coordinates
                assert!(entity.bbox.x1 >= 0.0 && entity.bbox.x1 <= 1.0);
                assert!(entity.bbox.y1 >= 0.0 && entity.bbox.y1 <= 1.0);
                assert!(entity.bbox.x2 >= entity.bbox.x1 && entity.bbox.x2 <= 1.0);
                assert!(entity.bbox.y2 >= entity.bbox.y1 && entity.bbox.y2 <= 1.0);
            }
        }
    }
}

