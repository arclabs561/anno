//! NER evaluation modes following SemEval-2013 Task 9.1.
//!
//! # The Core Problem
//!
//! Suppose your model predicted "New York City" but the gold label was "New York":
//!
//! ```text
//! Text:       "I visited New York City yesterday"
//!                        ▼▼▼▼▼▼▼▼▼▼▼▼▼
//! Gold:       [====New York====]
//!                  0         8
//!
//! Predicted:  [=====New York City=====]
//!                  0              13
//!
//! Is this prediction correct? It depends on what you're measuring.
//! ```
//!
//! # Visual Guide to Each Mode
//!
//! ## Strict Mode (CoNLL Standard)
//!
//! "Did you get EXACTLY the right span AND the right type?"
//!
//! ```text
//! Case 1: Perfect match
//!   Gold:  [John]  type=PER
//!   Pred:  [John]  type=PER
//!   Result: ✓ TRUE POSITIVE
//!
//! Case 2: Wrong boundary
//!   Gold:  [New York]      type=LOC
//!   Pred:  [New York City] type=LOC
//!   Result: ✗ Both boundaries must match exactly
//!
//! Case 3: Wrong type
//!   Gold:  [Apple] type=ORG
//!   Pred:  [Apple] type=LOC
//!   Result: ✗ Type must match exactly
//! ```
//!
//! ## Partial Mode (Lenient Boundaries)
//!
//! "Did you find something that OVERLAPS the gold span with the RIGHT type?"
//!
//! ```text
//! Gold:     [====New York====]
//!                0         8
//!
//! Pred:     [=====New York City=====]
//!                0              13
//!
//!           |◄──overlap──►|
//!           Chars 0-8 are shared
//!
//! Overlap?  ✓ Yes (8 chars)
//! Type?     ✓ Both LOC
//! Result:   ✓ TRUE POSITIVE in Partial mode
//! ```
//!
//! ## Exact Mode (Boundary Detection)
//!
//! "Did you find the EXACT span, regardless of type?"
//!
//! ```text
//! Gold:  [Apple]  type=ORG
//! Pred:  [Apple]  type=LOC   (wrong type!)
//!
//! Boundaries match? ✓ Yes (both 0-5)
//! Result: ✓ TRUE POSITIVE in Exact mode
//!
//! Use case: "Can my model find entity boundaries at all?"
//! ```
//!
//! ## Type Mode (Classification Focus)
//!
//! "Did you identify the RIGHT TYPE somewhere in the overlapping region?"
//!
//! ```text
//! Gold:  [The Apple Company]  type=ORG
//! Pred:  [Apple]              type=ORG
//!
//! Overlap?  ✓ Yes ("Apple" is inside)
//! Type?     ✓ Both ORG
//! Result:   ✓ TRUE POSITIVE in Type mode
//!
//! Use case: "Can my model classify entity types correctly?"
//! ```
//!
//! # Diagnostic Patterns
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    What Your Scores Tell You                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  High Strict, Low Partial                                       │
//! │  ─────────────────────────                                      │
//! │  → Model finds exact spans but confuses types                   │
//! │  → Fix: Better type classification                              │
//! │                                                                 │
//! │  Low Strict, High Partial                                       │
//! │  ─────────────────────────                                      │
//! │  → Model finds general area but not exact boundaries            │
//! │  → Fix: Better boundary detection (tokenization? BIO tags?)     │
//! │                                                                 │
//! │  Low Strict, Low Partial                                        │
//! │  ─────────────────────────                                      │
//! │  → Model is missing entities entirely                           │
//! │  → Fix: More training data, lower threshold                     │
//! │                                                                 │
//! │  High Strict, High Partial (ideal!)                             │
//! │  ───────────────────────────────────                            │
//! │  → Model is working well                                        │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Summary Table
//!
//! | Mode | Boundary | Type | Use Case |
//! |------|----------|------|----------|
//! | **Strict** | Exact | Exact | Production benchmarks (CoNLL standard) |
//! | **Exact** | Exact | Any | Boundary detection evaluation |
//! | **Partial** | Overlap | Exact | Lenient type evaluation |
//! | **Type** | Any | Exact | Type classification evaluation |
//!
//! # Example
//!
//! ```rust
//! use anno::eval::modes::{EvalMode, evaluate_with_mode, MultiModeResults};
//! use anno::eval::GoldEntity;
//! use anno::{Entity, EntityType};
//!
//! let predicted = vec![
//!     Entity::new("New York City", EntityType::Location, 0, 13, 0.9),
//! ];
//! let gold = vec![
//!     GoldEntity::new("New York", EntityType::Location, 0),
//! ];
//!
//! // Strict mode (default) - requires exact boundary + type
//! let strict = evaluate_with_mode(&predicted, &gold, EvalMode::Strict);
//! println!("Strict F1: {:.1}%", strict.f1 * 100.0);
//!
//! // Partial mode - allows boundary overlap
//! let partial = evaluate_with_mode(&predicted, &gold, EvalMode::Partial);
//! println!("Partial F1: {:.1}%", partial.f1 * 100.0);
//!
//! // Get all modes at once
//! let all = MultiModeResults::compute(&predicted, &gold);
//! println!("Strict: {:.1}%, Partial: {:.1}%", 
//!     all.strict.f1 * 100.0, all.partial.f1 * 100.0);
//! ```

use crate::{Entity, EntityType};
use super::datasets::GoldEntity;
use serde::{Deserialize, Serialize};

// =============================================================================
// Evaluation Modes
// =============================================================================

/// Evaluation mode following SemEval-2013 Task 9.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum EvalMode {
    /// Strict: Exact boundary AND exact type (CoNLL standard).
    /// This is the default and most commonly reported metric.
    #[default]
    Strict,
    
    /// Exact boundary match only (type can differ).
    /// Useful for evaluating span detection separately from classification.
    Exact,
    
    /// Partial boundary overlap with exact type.
    /// More lenient than strict; gives credit for overlapping predictions.
    Partial,
    
    /// Any overlap with exact type.
    /// Most lenient; only requires some overlap and correct type.
    Type,
}

impl EvalMode {
    /// All available modes.
    pub fn all() -> &'static [EvalMode] {
        &[EvalMode::Strict, EvalMode::Exact, EvalMode::Partial, EvalMode::Type]
    }
    
    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            EvalMode::Strict => "Strict",
            EvalMode::Exact => "Exact",
            EvalMode::Partial => "Partial",
            EvalMode::Type => "Type",
        }
    }
    
    /// Description of what this mode evaluates.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            EvalMode::Strict => "Exact boundary + exact type (CoNLL standard)",
            EvalMode::Exact => "Exact boundary only (type can differ)",
            EvalMode::Partial => "Partial boundary overlap + exact type",
            EvalMode::Type => "Any overlap + exact type",
        }
    }
}

// =============================================================================
// Mode-specific Results
// =============================================================================

/// Results for a single evaluation mode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModeResults {
    /// Evaluation mode used
    pub mode: EvalMode,
    /// Precision (0.0-1.0)
    pub precision: f64,
    /// Recall (0.0-1.0)
    pub recall: f64,
    /// F1 score (0.0-1.0)
    pub f1: f64,
    /// True positives (matches)
    pub true_positives: usize,
    /// False positives (spurious predictions)
    pub false_positives: usize,
    /// False negatives (missed entities)
    pub false_negatives: usize,
}

impl ModeResults {
    /// Compute results for a specific mode.
    #[must_use]
    pub fn compute(predicted: &[Entity], gold: &[GoldEntity], mode: EvalMode) -> Self {
        let (tp, fp, fn_count) = count_matches(predicted, gold, mode);
        
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        
        Self {
            mode,
            precision,
            recall,
            f1,
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_count,
        }
    }
}

/// Results across all evaluation modes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiModeResults {
    /// Strict mode (exact boundary + exact type)
    pub strict: ModeResults,
    /// Exact mode (exact boundary only)
    pub exact: ModeResults,
    /// Partial mode (partial boundary + exact type)
    pub partial: ModeResults,
    /// Type mode (any overlap + exact type)
    pub type_mode: ModeResults,
}

impl MultiModeResults {
    /// Compute all modes at once.
    #[must_use]
    pub fn compute(predicted: &[Entity], gold: &[GoldEntity]) -> Self {
        Self {
            strict: ModeResults::compute(predicted, gold, EvalMode::Strict),
            exact: ModeResults::compute(predicted, gold, EvalMode::Exact),
            partial: ModeResults::compute(predicted, gold, EvalMode::Partial),
            type_mode: ModeResults::compute(predicted, gold, EvalMode::Type),
        }
    }
    
    /// Get results for a specific mode.
    #[must_use]
    pub fn get(&self, mode: EvalMode) -> &ModeResults {
        match mode {
            EvalMode::Strict => &self.strict,
            EvalMode::Exact => &self.exact,
            EvalMode::Partial => &self.partial,
            EvalMode::Type => &self.type_mode,
        }
    }
    
    /// Print summary table.
    pub fn print_summary(&self) {
        println!("Evaluation Mode Results:");
        println!("{:<10} {:>10} {:>10} {:>10}", "Mode", "Precision", "Recall", "F1");
        println!("{:-<43}", "");
        for mode in EvalMode::all() {
            let r = self.get(*mode);
            println!("{:<10} {:>9.1}% {:>9.1}% {:>9.1}%",
                mode.name(),
                r.precision * 100.0,
                r.recall * 100.0,
                r.f1 * 100.0);
        }
    }
}

// =============================================================================
// Matching Logic
// =============================================================================

/// Check if two entities match according to the given mode.
fn entities_match(pred: &Entity, gold: &GoldEntity, mode: EvalMode) -> bool {
    match mode {
        EvalMode::Strict => {
            // Exact boundary AND exact type
            pred.start == gold.start && pred.end == gold.end && types_match(&pred.entity_type, &gold.entity_type)
        }
        EvalMode::Exact => {
            // Exact boundary only (type can differ)
            pred.start == gold.start && pred.end == gold.end
        }
        EvalMode::Partial => {
            // Partial overlap AND exact type
            has_overlap(pred.start, pred.end, gold.start, gold.end) && types_match(&pred.entity_type, &gold.entity_type)
        }
        EvalMode::Type => {
            // Any overlap AND exact type
            has_overlap(pred.start, pred.end, gold.start, gold.end) && types_match(&pred.entity_type, &gold.entity_type)
        }
    }
}

/// Check if two entity types match.
fn types_match(a: &EntityType, b: &EntityType) -> bool {
    // Use the existing entity_type_matches logic
    super::entity_type_matches(a, b)
}

/// Check if two spans have any overlap.
fn has_overlap(start1: usize, end1: usize, start2: usize, end2: usize) -> bool {
    start1 < end2 && start2 < end1
}

/// Calculate overlap ratio (IoU) between two spans.
#[must_use]
pub fn overlap_ratio(start1: usize, end1: usize, start2: usize, end2: usize) -> f64 {
    let intersection_start = start1.max(start2);
    let intersection_end = end1.min(end2);
    
    if intersection_start >= intersection_end {
        return 0.0;
    }
    
    let intersection = (intersection_end - intersection_start) as f64;
    let union = ((end1 - start1) + (end2 - start2) - (intersection_end - intersection_start)) as f64;
    
    if union == 0.0 {
        1.0
    } else {
        intersection / union
    }
}

/// Count true positives, false positives, and false negatives.
fn count_matches(predicted: &[Entity], gold: &[GoldEntity], mode: EvalMode) -> (usize, usize, usize) {
    let mut gold_matched = vec![false; gold.len()];
    let mut tp = 0;
    let mut fp = 0;
    
    // For each prediction, try to find a matching gold entity
    for pred in predicted {
        let mut found_match = false;
        
        for (i, g) in gold.iter().enumerate() {
            if gold_matched[i] {
                continue;
            }
            
            if entities_match(pred, g, mode) {
                gold_matched[i] = true;
                found_match = true;
                tp += 1;
                break;
            }
        }
        
        if !found_match {
            fp += 1;
        }
    }
    
    let fn_count = gold_matched.iter().filter(|&&m| !m).count();
    
    (tp, fp, fn_count)
}

/// Evaluate with a specific mode.
#[must_use]
pub fn evaluate_with_mode(predicted: &[Entity], gold: &[GoldEntity], mode: EvalMode) -> ModeResults {
    ModeResults::compute(predicted, gold, mode)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn pred(text: &str, ty: EntityType, start: usize, end: usize) -> Entity {
        Entity::new(text, ty, start, end, 0.9)
    }
    
    fn gold(text: &str, ty: EntityType, start: usize) -> GoldEntity {
        GoldEntity::new(text, ty, start)
    }
    
    #[test]
    fn test_strict_exact_match() {
        let predicted = vec![pred("John", EntityType::Person, 0, 4)];
        let gold_entities = vec![gold("John", EntityType::Person, 0)];
        
        let results = ModeResults::compute(&predicted, &gold_entities, EvalMode::Strict);
        assert!((results.f1 - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_strict_wrong_boundary() {
        let predicted = vec![pred("John Smith", EntityType::Person, 0, 10)];
        let gold_entities = vec![gold("John", EntityType::Person, 0)];
        
        let results = ModeResults::compute(&predicted, &gold_entities, EvalMode::Strict);
        assert_eq!(results.f1, 0.0); // Strict mode fails
        
        // But partial mode should match
        let partial = ModeResults::compute(&predicted, &gold_entities, EvalMode::Partial);
        assert!((partial.f1 - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_strict_wrong_type() {
        let predicted = vec![pred("Apple", EntityType::Organization, 0, 5)];
        let gold_entities = vec![gold("Apple", EntityType::Location, 0)];
        
        let results = ModeResults::compute(&predicted, &gold_entities, EvalMode::Strict);
        assert_eq!(results.f1, 0.0); // Wrong type
        
        // But exact mode (boundary only) should match
        let exact = ModeResults::compute(&predicted, &gold_entities, EvalMode::Exact);
        assert!((exact.f1 - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_partial_overlap() {
        // "New York City" vs "New York"
        let predicted = vec![pred("New York City", EntityType::Location, 0, 13)];
        let gold_entities = vec![gold("New York", EntityType::Location, 0)];
        
        // Strict: fail (different boundary)
        let strict = ModeResults::compute(&predicted, &gold_entities, EvalMode::Strict);
        assert_eq!(strict.f1, 0.0);
        
        // Partial: pass (overlap + same type)
        let partial = ModeResults::compute(&predicted, &gold_entities, EvalMode::Partial);
        assert!((partial.f1 - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_no_overlap() {
        let predicted = vec![pred("John", EntityType::Person, 0, 4)];
        let gold_entities = vec![gold("Mary", EntityType::Person, 10)];
        
        for mode in EvalMode::all() {
            let results = ModeResults::compute(&predicted, &gold_entities, *mode);
            assert_eq!(results.f1, 0.0, "Mode {:?} should fail with no overlap", mode);
        }
    }
    
    #[test]
    fn test_multi_mode_results() {
        let predicted = vec![
            pred("John", EntityType::Person, 0, 4),
            pred("New York City", EntityType::Location, 10, 23),
        ];
        let gold_entities = vec![
            gold("John", EntityType::Person, 0),
            gold("New York", EntityType::Location, 10),
        ];
        
        let all = MultiModeResults::compute(&predicted, &gold_entities);
        
        // Strict: 1/2 (John matches, NYC doesn't)
        assert!((all.strict.precision - 0.5).abs() < 0.001);
        
        // Partial: 2/2 (both overlap)
        assert!((all.partial.f1 - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_overlap_ratio() {
        // Complete overlap
        assert!((overlap_ratio(0, 10, 0, 10) - 1.0).abs() < 0.001);

        // No overlap
        assert!((overlap_ratio(0, 5, 10, 15) - 0.0).abs() < 0.001);

        // Partial overlap: [0,10] and [5,15]
        // Intersection: [5,10] = 5 chars
        // Union: 10 + 10 - 5 = 15 chars
        // IoU = 5/15 = 0.333...
        assert!(
            (overlap_ratio(0, 10, 5, 15) - (5.0 / 15.0)).abs() < 0.001,
            "Expected IoU of 5/15 = {}, got {}",
            5.0 / 15.0,
            overlap_ratio(0, 10, 5, 15)
        );
    }
    
    #[test]
    fn test_empty_inputs() {
        let empty_pred: Vec<Entity> = vec![];
        let empty_gold: Vec<GoldEntity> = vec![];
        
        let results = ModeResults::compute(&empty_pred, &empty_gold, EvalMode::Strict);
        assert_eq!(results.f1, 0.0);
        assert_eq!(results.true_positives, 0);
        assert_eq!(results.false_positives, 0);
        assert_eq!(results.false_negatives, 0);
    }
}

