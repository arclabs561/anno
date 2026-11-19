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
//! "Did you find something that OVERLAPS the gold span?" Type is ignored.
//! An exact boundary earns full credit and an overlapping boundary half
//! credit (SemEval-2013 partial, Segura-Bedmar et al. 2013):
//! `P = (COR + 0.5 PAR) / ACT`, `R = (COR + 0.5 PAR) / POS`.
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
//! Boundary? Not exact
//! Result:   PARTIAL match in Partial mode (half credit)
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
//! | **Partial** | Overlap (half credit) | Any | Lenient boundary evaluation |
//! | **Type** | Any | Exact | Type classification evaluation |
//!
//! # Example
//!
//! ```rust
//! use anno_eval::eval::modes::{EvalMode, evaluate_with_mode, MultiModeResults};
//! use anno_eval::eval::GoldEntity;
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
//! // Partial mode - half credit for overlapping boundaries, type ignored
//! let partial = evaluate_with_mode(&predicted, &gold, EvalMode::Partial);
//! println!("Partial F1: {:.1}%", partial.f1 * 100.0);
//!
//! // Get all modes at once
//! let all = MultiModeResults::compute(&predicted, &gold);
//! println!("Strict: {:.1}%, Partial: {:.1}%",
//!     all.strict.f1 * 100.0, all.partial.f1 * 100.0);
//! ```

use super::datasets::GoldEntity;
use anno::{Entity, EntityType};
use serde::{Deserialize, Serialize};

// =============================================================================
// Evaluation Configuration
// =============================================================================

/// Configuration for NER evaluation.
///
/// Partial matching modes (Partial, Type) accept any overlap by default.
/// In practice, you may want to require a minimum overlap ratio to avoid
/// counting barely-touching spans as matches.
///
/// # Example
///
/// ```rust
/// use anno_eval::eval::modes::EvalConfig;
///
/// // Require at least 50% overlap for partial matches
/// let config = EvalConfig::new().with_min_overlap(0.5);
///
/// // Strict config (default behavior)
/// let strict = EvalConfig::strict();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Minimum overlap ratio (IoU) required for partial/type matches.
    ///
    /// Range: 0.0 to 1.0
    /// - 0.0: Any overlap counts (default)
    /// - 0.5: At least 50% overlap required
    /// - 1.0: Effectively requires exact boundaries
    pub min_overlap: f64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self { min_overlap: 0.0 }
    }
}

impl EvalConfig {
    /// Create a new configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a strict configuration (default overlap behavior).
    #[must_use]
    pub fn strict() -> Self {
        Self::default()
    }

    /// Set the minimum overlap threshold for partial matches.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum IoU (0.0-1.0) for partial matches
    #[must_use]
    pub fn with_min_overlap(mut self, threshold: f64) -> Self {
        self.min_overlap = threshold.clamp(0.0, 1.0);
        self
    }
}

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

    /// SemEval-2013 partial: type is ignored, an exact boundary earns full
    /// credit and an overlapping boundary half credit.
    ///
    /// Precision is `(COR + 0.5 PAR) / ACT` and recall `(COR + 0.5 PAR) / POS`,
    /// where COR counts exact-boundary matches, PAR overlapping ones, ACT the
    /// predictions and POS the gold entities. In [`ModeResults`],
    /// `true_positives` counts the matched pairs (COR + PAR).
    Partial,

    /// Any overlap with exact type.
    /// Most lenient; only requires some overlap and correct type.
    Type,
}

impl EvalMode {
    /// All available modes.
    pub fn all() -> &'static [EvalMode] {
        &[
            EvalMode::Strict,
            EvalMode::Exact,
            EvalMode::Partial,
            EvalMode::Type,
        ]
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
            EvalMode::Partial => {
                "Overlapping boundary for half credit, type ignored (SemEval partial)"
            }
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
    ///
    /// Formulas:
    /// - `Precision = TP / (TP + FP)`
    /// - `Recall = TP / (TP + FN)`
    /// - `F1 = 2 × (P × R) / (P + R)` (harmonic mean)
    ///
    /// Reference: Manning et al. (2008) [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
    #[must_use]
    pub fn compute(predicted: &[Entity], gold: &[GoldEntity], mode: EvalMode) -> Self {
        if mode == EvalMode::Partial {
            return semeval_partial_results(predicted, gold, 0.0);
        }
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
    /// Partial mode (SemEval partial: half credit for overlap, type ignored)
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
        println!(
            "{:<10} {:>10} {:>10} {:>10}",
            "Mode", "Precision", "Recall", "F1"
        );
        println!("{:-<43}", "");
        for mode in EvalMode::all() {
            let r = self.get(*mode);
            println!(
                "{:<10} {:>9.1}% {:>9.1}% {:>9.1}%",
                mode.name(),
                r.precision * 100.0,
                r.recall * 100.0,
                r.f1 * 100.0
            );
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
            pred.start() == gold.start
                && pred.end() == gold.end
                && types_match(&pred.entity_type, &gold.entity_type)
        }
        EvalMode::Exact => {
            // Exact boundary only (type can differ)
            pred.start() == gold.start && pred.end() == gold.end
        }
        EvalMode::Partial => {
            // Any overlap, type ignored. Scoring (half credit for inexact
            // boundaries) happens in `semeval_partial_results`.
            has_overlap(pred.start(), pred.end(), gold.start, gold.end)
        }
        EvalMode::Type => {
            // Any overlap AND exact type
            has_overlap(pred.start(), pred.end(), gold.start, gold.end)
                && types_match(&pred.entity_type, &gold.entity_type)
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

/// Check if overlap meets minimum threshold.
///
/// This allows requiring a minimum overlap ratio for partial matches,
/// useful when barely-touching spans shouldn't count as matches.
/// Default threshold of 0.0 accepts any overlap.
fn has_sufficient_overlap(
    start1: usize,
    end1: usize,
    start2: usize,
    end2: usize,
    min_threshold: f64,
) -> bool {
    if !has_overlap(start1, end1, start2, end2) {
        return false;
    }
    if min_threshold <= 0.0 {
        return true;
    }
    overlap_ratio(start1, end1, start2, end2) >= min_threshold
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
    let union =
        ((end1 - start1) + (end2 - start2) - (intersection_end - intersection_start)) as f64;

    if union == 0.0 {
        1.0
    } else {
        intersection / union
    }
}

/// Count true positives, false positives, and false negatives.
fn count_matches(
    predicted: &[Entity],
    gold: &[GoldEntity],
    mode: EvalMode,
) -> (usize, usize, usize) {
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

/// SemEval-2013 partial scoring (Segura-Bedmar et al. 2013).
///
/// Type is ignored. Pairs are assigned best match first across all gold
/// spans: exact-boundary pairs (COR) before overlapping ones (PAR), then
/// gold order, then prediction order, each span used at most once. An
/// overlapping pair must also reach `min_overlap` IoU.
fn semeval_partial_results(
    predicted: &[Entity],
    gold: &[GoldEntity],
    min_overlap: f64,
) -> ModeResults {
    let mut candidates: Vec<(bool, usize, usize)> = Vec::new();
    for (gold_idx, g) in gold.iter().enumerate() {
        for (pred_idx, p) in predicted.iter().enumerate() {
            if has_sufficient_overlap(p.start(), p.end(), g.start, g.end, min_overlap) {
                let exact = p.start() == g.start && p.end() == g.end;
                candidates.push((exact, gold_idx, pred_idx));
            }
        }
    }
    // Stable sort keeps gold order, then prediction order, within each class.
    candidates.sort_by_key(|&(exact, _, _)| std::cmp::Reverse(exact));

    let mut gold_matched = vec![false; gold.len()];
    let mut pred_matched = vec![false; predicted.len()];
    let (mut correct, mut partial) = (0usize, 0usize);
    for (exact, gold_idx, pred_idx) in candidates {
        if gold_matched[gold_idx] || pred_matched[pred_idx] {
            continue;
        }
        gold_matched[gold_idx] = true;
        pred_matched[pred_idx] = true;
        if exact {
            correct += 1;
        } else {
            partial += 1;
        }
    }

    let matched = correct + partial;
    let credit = correct as f64 + 0.5 * partial as f64;
    let precision = if predicted.is_empty() {
        0.0
    } else {
        credit / predicted.len() as f64
    };
    let recall = if gold.is_empty() {
        0.0
    } else {
        credit / gold.len() as f64
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    ModeResults {
        mode: EvalMode::Partial,
        precision,
        recall,
        f1,
        true_positives: matched,
        false_positives: predicted.len() - matched,
        false_negatives: gold.len() - matched,
    }
}

/// Evaluate with a specific mode.
#[must_use]
pub fn evaluate_with_mode(
    predicted: &[Entity],
    gold: &[GoldEntity],
    mode: EvalMode,
) -> ModeResults {
    ModeResults::compute(predicted, gold, mode)
}

/// Evaluate with a specific mode and configuration.
///
/// This allows customizing behavior like minimum overlap thresholds.
///
/// # Example
///
/// ```rust
/// use anno_eval::eval::modes::{EvalMode, EvalConfig, evaluate_with_config};
/// use anno_eval::eval::GoldEntity;
/// use anno::{Entity, EntityType};
///
/// let predicted = vec![Entity::new("New York", EntityType::Location, 0, 8, 0.9)];
/// let gold = vec![GoldEntity::new("New York City", EntityType::Location, 0)];
///
/// // Require 50% overlap for partial matches
/// let config = EvalConfig::new().with_min_overlap(0.5);
/// let results = evaluate_with_config(&predicted, &gold, EvalMode::Partial, &config);
/// ```
#[must_use]
pub fn evaluate_with_config(
    predicted: &[Entity],
    gold: &[GoldEntity],
    mode: EvalMode,
    config: &EvalConfig,
) -> ModeResults {
    if mode == EvalMode::Partial {
        return semeval_partial_results(predicted, gold, config.min_overlap);
    }
    let (tp, fp, fn_count) = count_matches_with_config(predicted, gold, mode, config);

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

    ModeResults {
        mode,
        precision,
        recall,
        f1,
        true_positives: tp,
        false_positives: fp,
        false_negatives: fn_count,
    }
}

/// Count matches with configuration.
fn count_matches_with_config(
    predicted: &[Entity],
    gold: &[GoldEntity],
    mode: EvalMode,
    config: &EvalConfig,
) -> (usize, usize, usize) {
    let mut gold_matched = vec![false; gold.len()];
    let mut tp = 0;
    let mut fp = 0;

    for pred in predicted {
        let mut found_match = false;

        for (i, g) in gold.iter().enumerate() {
            if gold_matched[i] {
                continue;
            }

            if entities_match_with_config(pred, g, mode, config) {
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

/// Check if entities match with configuration.
fn entities_match_with_config(
    pred: &Entity,
    gold: &GoldEntity,
    mode: EvalMode,
    config: &EvalConfig,
) -> bool {
    match mode {
        EvalMode::Strict => {
            pred.start() == gold.start
                && pred.end() == gold.end
                && types_match(&pred.entity_type, &gold.entity_type)
        }
        EvalMode::Exact => pred.start() == gold.start && pred.end() == gold.end,
        EvalMode::Partial => has_sufficient_overlap(
            pred.start(),
            pred.end(),
            gold.start,
            gold.end,
            config.min_overlap,
        ),
        EvalMode::Type => {
            has_sufficient_overlap(
                pred.start(),
                pred.end(),
                gold.start,
                gold.end,
                config.min_overlap,
            ) && types_match(&pred.entity_type, &gold.entity_type)
        }
    }
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

        // Partial mode gives the overlapping boundary half credit
        let partial = ModeResults::compute(&predicted, &gold_entities, EvalMode::Partial);
        assert!((partial.f1 - 0.5).abs() < 0.001); // one PAR pair: half credit
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

        // Partial: overlapping boundary earns half credit
        let partial = ModeResults::compute(&predicted, &gold_entities, EvalMode::Partial);
        assert!((partial.f1 - 0.5).abs() < 0.001); // one PAR pair: half credit
    }

    #[test]
    fn test_no_overlap() {
        let predicted = vec![pred("John", EntityType::Person, 0, 4)];
        let gold_entities = vec![gold("Mary", EntityType::Person, 10)];

        for mode in EvalMode::all() {
            let results = ModeResults::compute(&predicted, &gold_entities, *mode);
            assert_eq!(
                results.f1, 0.0,
                "Mode {:?} should fail with no overlap",
                mode
            );
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

        // Partial: John exact (COR), NYC overlapping (PAR)
        assert!((all.partial.f1 - 0.75).abs() < 0.001); // (1 COR + 0.5 PAR) / 2
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

    // === EvalConfig tests ===

    #[test]
    fn test_config_default() {
        let config = EvalConfig::default();
        assert_eq!(config.min_overlap, 0.0);
    }

    #[test]
    fn test_config_with_overlap() {
        let config = EvalConfig::new().with_min_overlap(0.5);
        assert_eq!(config.min_overlap, 0.5);
    }

    #[test]
    fn test_config_clamp() {
        // Values outside 0-1 should be clamped
        let config = EvalConfig::new().with_min_overlap(1.5);
        assert_eq!(config.min_overlap, 1.0);

        let config = EvalConfig::new().with_min_overlap(-0.5);
        assert_eq!(config.min_overlap, 0.0);
    }

    #[test]
    fn test_partial_with_zero_threshold() {
        // Default: any overlap counts
        let predicted = vec![pred("New York City", EntityType::Location, 0, 13)];
        let gold_entities = vec![gold("New York", EntityType::Location, 0)];

        let config = EvalConfig::default();
        let results = evaluate_with_config(&predicted, &gold_entities, EvalMode::Partial, &config);

        // Should match (overlap exists)
        assert!((results.f1 - 0.5).abs() < 0.001); // one PAR pair: half credit
    }

    #[test]
    fn test_partial_with_high_threshold() {
        // "New York City" [0,13] vs "New York" [0,8]
        // Overlap: 8 chars, Union: 13 chars
        // IoU = 8/13 ≈ 0.615
        let predicted = vec![pred("New York City", EntityType::Location, 0, 13)];
        let gold_entities = vec![gold("New York", EntityType::Location, 0)];

        // 50% threshold - should pass (0.615 > 0.5)
        let config = EvalConfig::new().with_min_overlap(0.5);
        let results = evaluate_with_config(&predicted, &gold_entities, EvalMode::Partial, &config);
        assert!(
            (results.f1 - 0.5).abs() < 0.001,
            "0.5 threshold should pass as a PAR pair"
        );

        // 70% threshold - should fail (0.615 < 0.7)
        let config = EvalConfig::new().with_min_overlap(0.7);
        let results = evaluate_with_config(&predicted, &gold_entities, EvalMode::Partial, &config);
        assert_eq!(results.f1, 0.0, "0.7 threshold should fail");
    }

    #[test]
    fn test_partial_barely_touching() {
        // Entities that barely touch: [0,5] and [4,10]
        // Overlap: 1 char, Union: 10 chars
        // IoU = 1/10 = 0.1
        let predicted = vec![pred("Apple", EntityType::Organization, 0, 5)];
        let gold_entities = vec![gold("Banana", EntityType::Organization, 4)];

        // Default (0%) should match
        let config = EvalConfig::default();
        let results = evaluate_with_config(&predicted, &gold_entities, EvalMode::Partial, &config);
        assert!((results.f1 - 0.5).abs() < 0.001); // one PAR pair: half credit

        // 20% threshold should fail (0.1 < 0.2)
        let config = EvalConfig::new().with_min_overlap(0.2);
        let results = evaluate_with_config(&predicted, &gold_entities, EvalMode::Partial, &config);
        assert_eq!(results.f1, 0.0);
    }

    #[test]
    fn test_strict_mode_ignores_threshold() {
        // Strict mode requires exact boundaries, threshold shouldn't matter
        let predicted = vec![pred("John", EntityType::Person, 0, 4)];
        let gold_entities = vec![gold("John", EntityType::Person, 0)];

        let config = EvalConfig::new().with_min_overlap(0.99);
        let results = evaluate_with_config(&predicted, &gold_entities, EvalMode::Strict, &config);

        // Should still pass (exact match)
        assert!((results.f1 - 1.0).abs() < 0.001);
    }

    /// SemEval-2013 partial scheme (Segura-Bedmar et al. 2013): type is
    /// ignored, an exact boundary is correct (COR), an overlapping boundary
    /// is partial (PAR) worth half credit, and
    /// P = (COR + 0.5 PAR) / ACT, R = (COR + 0.5 PAR) / POS.
    ///
    /// Gold: "John" PER [0,4), "New York" LOC [10,18), "Apple" ORG [30,35),
    /// "Paris" LOC [40,45). Predictions: "John" PER (COR), "New" LOC [10,13)
    /// (PAR), "Apple" PER (exact boundary, wrong type: COR), and a spurious
    /// "Rome" [50,54). "Paris" is missed.
    /// COR = 2, PAR = 1, ACT = 4, POS = 4, so P = R = F1 = 2.5 / 4 = 0.625.
    #[test]
    fn partial_mode_is_semeval_partial() {
        let gold_entities = vec![
            gold("John", EntityType::Person, 0),
            gold("New York", EntityType::Location, 10),
            gold("Apple", EntityType::Organization, 30),
            gold("Paris", EntityType::Location, 40),
        ];
        let predicted = vec![
            pred("John", EntityType::Person, 0, 4),
            pred("New", EntityType::Location, 10, 13),
            pred("Apple", EntityType::Person, 30, 35),
            pred("Rome", EntityType::Location, 50, 54),
        ];
        for r in [
            ModeResults::compute(&predicted, &gold_entities, EvalMode::Partial),
            evaluate_with_config(
                &predicted,
                &gold_entities,
                EvalMode::Partial,
                &EvalConfig::default(),
            ),
        ] {
            assert!((r.precision - 0.625).abs() < 1e-12, "{r:?}");
            assert!((r.recall - 0.625).abs() < 1e-12, "{r:?}");
            assert!((r.f1 - 0.625).abs() < 1e-12, "{r:?}");
            // Matched pairs (COR + PAR), unmatched predictions, unmatched gold.
            assert_eq!(
                (r.true_positives, r.false_positives, r.false_negatives),
                (3, 1, 1)
            );
        }

        // Type mode is unchanged: overlap and matching type required.
        let t = ModeResults::compute(&predicted, &gold_entities, EvalMode::Type);
        assert_eq!((t.true_positives, t.false_positives), (2, 2));
    }

    /// An overlapping prediction with the wrong type is still PAR, and an
    /// exact-boundary match is preferred over an earlier overlapping one.
    /// Gold "New York" [0,8) and "York" [4,8); predictions "New York" ORG
    /// [0,8) and "York" LOC [4,8). Matching "New York" pred to "York" gold
    /// first would leave a PAR pair; the best assignment is two COR pairs.
    #[test]
    fn partial_mode_ignores_type_and_prefers_exact_boundaries() {
        let gold_entities = vec![gold("York", EntityType::Location, 4)];
        let predicted = vec![pred("New York", EntityType::Organization, 0, 8)];
        let r = ModeResults::compute(&predicted, &gold_entities, EvalMode::Partial);
        // COR = 0, PAR = 1, ACT = POS = 1: P = R = F1 = 0.5.
        assert!((r.f1 - 0.5).abs() < 1e-12, "{r:?}");

        let gold_entities = vec![
            gold("York", EntityType::Location, 4),
            gold("New York", EntityType::Location, 0),
        ];
        let predicted = vec![
            pred("New York", EntityType::Organization, 0, 8),
            pred("York", EntityType::Location, 4, 8),
        ];
        let r = ModeResults::compute(&predicted, &gold_entities, EvalMode::Partial);
        assert!((r.f1 - 1.0).abs() < 1e-12, "{r:?}");
    }

    #[test]
    fn test_has_sufficient_overlap() {
        // Any overlap with 0% threshold
        assert!(has_sufficient_overlap(0, 10, 5, 15, 0.0));

        // Needs at least 50% IoU
        // [0,10] and [5,15]: IoU = 5/15 ≈ 0.33
        assert!(!has_sufficient_overlap(0, 10, 5, 15, 0.5));

        // [0,10] and [2,12]: IoU = 8/12 ≈ 0.67
        assert!(has_sufficient_overlap(0, 10, 2, 12, 0.5));

        // No overlap at all
        assert!(!has_sufficient_overlap(0, 5, 10, 15, 0.0));
    }
}
