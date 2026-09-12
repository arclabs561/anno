//! Robustness testing for NER models.
//!
//! Tests model behavior under various perturbations and distribution shifts.
//! A robust model should degrade gracefully rather than catastrophically fail.
//!
//! # Perturbation Types
//!
//! - **Typos**: Character-level noise (swaps, insertions, deletions)
//! - **Case changes**: UPPER, lower, Title, mIxEd
//! - **Whitespace**: Extra spaces, tabs, newlines
//! - **Punctuation**: Missing or extra punctuation
//! - **Unicode**: Homoglyphs, diacritics, combining characters
//!
//! # Research Background
//!
//! - Pacific AI (2024): "Robustness Testing of NER Models with LangTest"
//! - Perturbation-based evaluation reveals model brittleness
//! - Real-world data contains noise that test sets often lack
//!
//! # Example
//!
//! ```rust
//! use anno_eval::eval::robustness::{RobustnessEvaluator, Perturbation};
//!
//! let perturber = RobustnessEvaluator::default();
//! let original = "John Smith works at Google.";
//!
//! // Generate perturbed versions
//! let variants = perturber.generate_variants(original);
//! for (perturbation_type, text) in variants {
//!     println!("{:?}: {}", perturbation_type, text);
//! }
//! ```

use crate::eval::ner_metrics::evaluate_entities;
use crate::{Entity, Model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple deterministic pseudo-random number generator (xorshift).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn gen_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }

    fn gen_bool(&mut self) -> bool {
        #[allow(clippy::manual_is_multiple_of)]
        {
            self.next() % 2 == 0
        }
    }

    fn gen_range(&mut self, max: usize) -> usize {
        if max == 0 {
            0
        } else {
            (self.next() as usize) % max
        }
    }
}

// =============================================================================
// Perturbation Types
// =============================================================================

/// Types of perturbations for robustness testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Perturbation {
    /// No perturbation (baseline)
    None,
    /// Character swaps within words
    TypoSwap,
    /// Character insertions
    TypoInsert,
    /// Character deletions
    TypoDelete,
    /// Keyboard-adjacent character substitution
    TypoKeyboard,
    /// Convert to UPPERCASE
    CaseUpper,
    /// Convert to lowercase
    CaseLower,
    /// Convert to Title Case
    CaseTitle,
    /// Convert to mIxEd CaSe
    CaseMixed,
    /// Add extra whitespace
    WhitespaceExtra,
    /// Remove some whitespace
    WhitespaceRemove,
    /// Replace spaces with newlines
    WhitespaceNewline,
    /// Remove punctuation
    PunctuationRemove,
    /// Add extra punctuation
    PunctuationExtra,
    /// Unicode homoglyphs (e.g., 'а' vs 'a')
    UnicodeHomoglyph,
    /// Add diacritics (e.g., 'e' -> 'é')
    UnicodeDiacritics,
    /// Add zero-width characters
    UnicodeZeroWidth,
}

// =============================================================================
// Robustness Results
// =============================================================================

/// Results of robustness evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessResults {
    /// Baseline F1 (no perturbation)
    pub baseline_f1: f64,
    /// F1 score by perturbation type
    pub by_perturbation: HashMap<String, PerturbationMetrics>,
    /// Average F1 across all configured non-baseline perturbations.
    ///
    /// This includes 0.0 for a perturbation with no scored examples. It is comparable only when
    /// [`coverage_complete`](Self::coverage_complete) is true.
    pub avg_perturbed_f1: f64,
    /// Robustness score: avg_perturbed_f1 / baseline_f1 (1.0 = perfectly robust).
    ///
    /// Comparable only when [`coverage_complete`](Self::coverage_complete) is true.
    pub robustness_score: f64,
    /// Worst perturbation type
    pub worst_perturbation: String,
    /// Best perturbation type (often "None")
    pub best_perturbation: String,
    /// Total input examples before perturbation-specific alignment exclusions
    pub total_examples: usize,
    /// Whether every configured perturbation scored every input example.
    ///
    /// When false, score summaries are not comparable across runs; inspect each
    /// perturbation's `count`, `excluded_count`, and `exclusion_reasons`.
    #[serde(default)]
    pub coverage_complete: bool,
}

/// Metrics for a single perturbation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationMetrics {
    /// F1 score under this perturbation, or 0.0 when no examples could be scored
    pub f1: f64,
    /// Precision under this perturbation
    pub precision: f64,
    /// Recall under this perturbation
    pub recall: f64,
    /// Relative change from baseline: (perturbed - baseline) / baseline
    pub relative_change: f64,
    /// Number of examples scored after projecting gold into perturbed text
    pub count: usize,
    /// Number of examples excluded because their gold annotations could not be projected safely
    #[serde(default)]
    pub excluded_count: usize,
    /// Exclusion counts keyed by stable projection reason
    #[serde(default)]
    pub exclusion_reasons: HashMap<String, usize>,
}

// =============================================================================
// Robustness Evaluator
// =============================================================================

/// Evaluator for model robustness under perturbations.
#[derive(Debug, Clone)]
pub struct RobustnessEvaluator {
    /// Perturbation types to test
    pub perturbations: Vec<Perturbation>,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Perturbation intensity (0.0-1.0)
    pub intensity: f64,
}

impl Default for RobustnessEvaluator {
    fn default() -> Self {
        Self {
            perturbations: vec![
                Perturbation::None,
                Perturbation::TypoSwap,
                Perturbation::TypoDelete,
                Perturbation::CaseUpper,
                Perturbation::CaseLower,
                Perturbation::CaseMixed,
                Perturbation::WhitespaceExtra,
                Perturbation::PunctuationRemove,
                Perturbation::UnicodeHomoglyph,
            ],
            seed: 42,
            intensity: 0.1, // 10% of characters affected
        }
    }
}

impl RobustnessEvaluator {
    /// Create a new evaluator with custom perturbations.
    pub fn new(perturbations: Vec<Perturbation>) -> Self {
        Self {
            perturbations,
            ..Default::default()
        }
    }

    /// Generate perturbed variants of a text.
    pub fn generate_variants(&self, text: &str) -> Vec<(Perturbation, String)> {
        self.perturbations
            .iter()
            .map(|&p| (p, self.apply_perturbation(text, p)))
            .collect()
    }

    /// Apply a single perturbation to text.
    pub fn apply_perturbation(&self, text: &str, perturbation: Perturbation) -> String {
        let mut rng = SimpleRng::new(self.seed ^ (text.len() as u64));

        match perturbation {
            Perturbation::None => text.to_string(),

            Perturbation::TypoSwap => {
                let mut chars: Vec<char> = text.chars().collect();
                let num_swaps = ((chars.len() as f64 * self.intensity) as usize).max(1);
                for _ in 0..num_swaps {
                    if chars.len() >= 2 {
                        let idx = rng.gen_range(chars.len() - 1);
                        if chars[idx].is_alphabetic() && chars[idx + 1].is_alphabetic() {
                            chars.swap(idx, idx + 1);
                        }
                    }
                }
                chars.into_iter().collect()
            }

            Perturbation::TypoInsert => {
                let mut result = String::new();
                let chars: Vec<char> = text.chars().collect();
                for (i, c) in chars.iter().enumerate() {
                    result.push(*c);
                    if rng.gen_f64() < self.intensity && c.is_alphabetic() {
                        // Insert a random adjacent character
                        let adjacent = random_adjacent_char(*c, &mut rng);
                        result.push(adjacent);
                    }
                    // Ensure we don't insert too much
                    if i > 0 && i % 20 == 0 && rng.gen_f64() < 0.1 {
                        break;
                    }
                }
                result
            }

            Perturbation::TypoDelete => {
                let intensity = self.intensity;
                text.chars()
                    .filter(|c| !c.is_alphabetic() || rng.gen_f64() > intensity)
                    .collect()
            }

            Perturbation::TypoKeyboard => {
                let intensity = self.intensity;
                text.chars()
                    .map(|c| {
                        if c.is_alphabetic() && rng.gen_f64() < intensity {
                            keyboard_neighbor(c, &mut rng)
                        } else {
                            c
                        }
                    })
                    .collect()
            }

            Perturbation::CaseUpper => text.to_uppercase(),
            Perturbation::CaseLower => text.to_lowercase(),

            Perturbation::CaseTitle => text
                .split_whitespace()
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => first
                            .to_uppercase()
                            .chain(chars.flat_map(|c| c.to_lowercase()))
                            .collect(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" "),

            Perturbation::CaseMixed => text.chars().enumerate().fold(
                String::with_capacity(text.len()),
                |mut out, (i, c)| {
                    // Unicode-aware: case conversion can expand into multiple chars.
                    if i % 2 == 0 {
                        out.extend(c.to_uppercase());
                    } else {
                        out.extend(c.to_lowercase());
                    }
                    out
                },
            ),

            Perturbation::WhitespaceExtra => {
                let intensity = self.intensity;
                text.chars()
                    .flat_map(|c| {
                        if c == ' ' && rng.gen_f64() < intensity * 3.0 {
                            vec![' ', ' ']
                        } else {
                            vec![c]
                        }
                    })
                    .collect()
            }

            Perturbation::WhitespaceRemove => {
                let words: Vec<&str> = text.split_whitespace().collect();
                let mut result = String::new();
                for (i, word) in words.iter().enumerate() {
                    result.push_str(word);
                    if i < words.len() - 1 && rng.gen_f64() > self.intensity {
                        result.push(' ');
                    }
                }
                result
            }

            Perturbation::WhitespaceNewline => {
                let intensity = self.intensity;
                text.chars()
                    .map(|c| {
                        if c == ' ' && rng.gen_f64() < intensity {
                            '\n'
                        } else {
                            c
                        }
                    })
                    .collect()
            }

            Perturbation::PunctuationRemove => {
                text.chars().filter(|c| !c.is_ascii_punctuation()).collect()
            }

            Perturbation::PunctuationExtra => {
                let intensity = self.intensity;
                text.chars()
                    .flat_map(|c| {
                        if c.is_ascii_punctuation() && rng.gen_f64() < intensity * 3.0 {
                            vec![c, c]
                        } else {
                            vec![c]
                        }
                    })
                    .collect()
            }

            Perturbation::UnicodeHomoglyph => {
                let intensity = self.intensity;
                text.chars()
                    .map(|c| {
                        if rng.gen_f64() < intensity {
                            homoglyph(c)
                        } else {
                            c
                        }
                    })
                    .collect()
            }

            Perturbation::UnicodeDiacritics => {
                let intensity = self.intensity;
                text.chars()
                    .map(|c| {
                        if c.is_alphabetic() && rng.gen_f64() < intensity {
                            add_diacritic(c)
                        } else {
                            c
                        }
                    })
                    .collect()
            }

            Perturbation::UnicodeZeroWidth => {
                let zwsp = '\u{200B}'; // Zero-width space
                let intensity = self.intensity;
                text.chars()
                    .flat_map(|c| {
                        if rng.gen_f64() < intensity * 0.5 {
                            vec![c, zwsp]
                        } else {
                            vec![c]
                        }
                    })
                    .collect()
            }
        }
    }

    /// Evaluate model robustness on test cases.
    pub fn evaluate(
        &self,
        model: &dyn Model,
        test_cases: &[(String, Vec<Entity>)],
    ) -> RobustnessResults {
        #[derive(Default)]
        struct PerturbationSamples {
            metrics: Vec<(f64, f64, f64)>,
            exclusions: HashMap<String, usize>,
        }

        let mut by_perturbation: HashMap<String, PerturbationSamples> = self
            .perturbations
            .iter()
            .map(|perturbation| {
                (
                    format!("{:?}", perturbation),
                    PerturbationSamples::default(),
                )
            })
            .collect();

        for (text, gold_entities) in test_cases {
            for &perturbation in &self.perturbations {
                let perturbed = self.apply_perturbation(text, perturbation);
                let samples = by_perturbation
                    .entry(format!("{:?}", perturbation))
                    .or_default();
                // Gold offsets and surface forms belong to the original text. Project them
                // through the perturbation before scoring; otherwise changes before an entity
                // shift its offsets and changes inside it make its old surface form invalid.
                // Ambiguous or lossy edit alignments are deliberately excluded rather than
                // inventing a gold span that could reward or penalize the model incorrectly.
                let perturbed_gold = match project_gold_entities(text, &perturbed, gold_entities) {
                    Ok(gold) => gold,
                    Err(reason) => {
                        *samples
                            .exclusions
                            .entry(reason.as_str().to_string())
                            .or_default() += 1;
                        continue;
                    }
                };
                let predicted = match model.extract_entities(&perturbed, None) {
                    Ok(predicted) => predicted,
                    Err(_) => {
                        *samples
                            .exclusions
                            .entry("inference_error".to_string())
                            .or_default() += 1;
                        continue;
                    }
                };

                let (precision, recall, f1) = compute_simple_metrics(&predicted, &perturbed_gold);
                samples.metrics.push((precision, recall, f1));
            }
        }

        // Aggregate metrics
        let mut aggregated: HashMap<String, PerturbationMetrics> = HashMap::new();
        let baseline_f1 = by_perturbation
            .get("None")
            .and_then(|samples| average_metrics(&samples.metrics).map(|(_, _, f1)| f1))
            .unwrap_or(0.0);

        for (name, samples) in &by_perturbation {
            let (avg_precision, avg_recall, avg_f1) =
                average_metrics(&samples.metrics).unwrap_or((0.0, 0.0, 0.0));
            let relative_change = if baseline_f1 > 0.0 {
                (avg_f1 - baseline_f1) / baseline_f1
            } else {
                0.0
            };

            aggregated.insert(
                name.clone(),
                PerturbationMetrics {
                    f1: avg_f1,
                    precision: avg_precision,
                    recall: avg_recall,
                    relative_change,
                    count: samples.metrics.len(),
                    excluded_count: samples.exclusions.values().sum(),
                    exclusion_reasons: samples.exclusions.clone(),
                },
            );
        }

        let coverage_complete = aggregated
            .values()
            .all(|metrics| metrics.excluded_count == 0);

        // Find best/worst
        let (worst, _) = aggregated
            .iter()
            .filter(|(k, metrics)| k.as_str() != "None" && metrics.count > 0)
            .min_by(|a, b| {
                a.1.f1
                    .partial_cmp(&b.1.f1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, v)| (k.clone(), v.f1))
            .unwrap_or(("None".to_string(), baseline_f1));

        let (best, _) = aggregated
            .iter()
            .filter(|(_, metrics)| metrics.count > 0)
            .max_by(|a, b| {
                a.1.f1
                    .partial_cmp(&b.1.f1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, v)| (k.clone(), v.f1))
            .unwrap_or(("None".to_string(), baseline_f1));

        // Include every configured perturbation, including those with zero scored examples.
        // `coverage_complete` marks the resulting summary as non-comparable when exclusions
        // occurred; zero values prevent unscored perturbations from disappearing silently.
        let perturbed_f1s: Vec<f64> = aggregated
            .iter()
            .filter(|(k, _)| k.as_str() != "None")
            .map(|(_, v)| v.f1)
            .collect();
        let avg_perturbed_f1 = if perturbed_f1s.is_empty() {
            baseline_f1
        } else {
            perturbed_f1s.iter().sum::<f64>() / perturbed_f1s.len() as f64
        };

        let robustness_score = if baseline_f1 > 0.0 {
            avg_perturbed_f1 / baseline_f1
        } else {
            0.0
        };

        RobustnessResults {
            baseline_f1,
            by_perturbation: aggregated,
            avg_perturbed_f1,
            robustness_score,
            worst_perturbation: worst,
            best_perturbation: best,
            total_examples: test_cases.len(),
            coverage_complete,
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get a random character adjacent on a QWERTY keyboard.
fn keyboard_neighbor(c: char, rng: &mut SimpleRng) -> char {
    let keyboard: &[(&[char], &[char])] = &[
        (&['q'], &['w', 'a']),
        (&['w'], &['q', 'e', 's']),
        (&['e'], &['w', 'r', 'd']),
        (&['r'], &['e', 't', 'f']),
        (&['t'], &['r', 'y', 'g']),
        (&['a'], &['q', 's', 'z']),
        (&['s'], &['a', 'd', 'w', 'x']),
        (&['d'], &['s', 'f', 'e', 'c']),
        (&['f'], &['d', 'g', 'r', 'v']),
        (&['g'], &['f', 'h', 't', 'b']),
    ];

    // This helper is intentionally ASCII/QWERTY-only.
    let lower = c.to_ascii_lowercase();
    for (keys, neighbors) in keyboard {
        if keys.contains(&lower) && !neighbors.is_empty() {
            let idx = rng.gen_range(neighbors.len());
            let neighbor = neighbors[idx];
            return if c.is_uppercase() {
                neighbor.to_ascii_uppercase()
            } else {
                neighbor
            };
        }
    }
    c
}

/// Get a random adjacent character (simple version).
fn random_adjacent_char(c: char, rng: &mut SimpleRng) -> char {
    let offset: i32 = if rng.gen_bool() { 1 } else { -1 };
    char::from_u32((c as i32 + offset) as u32).unwrap_or(c)
}

/// Get a homoglyph for a character.
fn homoglyph(c: char) -> char {
    match c {
        'a' => 'а', // Cyrillic а
        'e' => 'е', // Cyrillic е
        'o' => 'о', // Cyrillic о
        'p' => 'р', // Cyrillic р
        'c' => 'с', // Cyrillic с
        'A' => 'А', // Cyrillic А
        'E' => 'Е', // Cyrillic Е
        'O' => 'О', // Cyrillic О
        'P' => 'Р', // Cyrillic Р
        'C' => 'С', // Cyrillic С
        _ => c,
    }
}

/// Add a diacritic to a character.
fn add_diacritic(c: char) -> char {
    match c {
        'a' => 'á',
        'e' => 'é',
        'i' => 'í',
        'o' => 'ó',
        'u' => 'ú',
        'n' => 'ñ',
        'A' => 'Á',
        'E' => 'É',
        'I' => 'Í',
        'O' => 'Ó',
        'U' => 'Ú',
        'N' => 'Ñ',
        _ => c,
    }
}

#[derive(Clone, Copy, Debug)]
enum ProjectionError {
    AlignmentTooLarge,
    AmbiguousAlignment,
    InvalidGoldSpan,
    GoldTextMismatch,
    EmptyProjection,
    NonContiguousProjection,
}

impl ProjectionError {
    fn as_str(self) -> &'static str {
        match self {
            Self::AlignmentTooLarge => "alignment_too_large",
            Self::AmbiguousAlignment => "ambiguous_alignment",
            Self::InvalidGoldSpan => "invalid_gold_span",
            Self::GoldTextMismatch => "gold_text_mismatch",
            Self::EmptyProjection => "empty_projection",
            Self::NonContiguousProjection => "non_contiguous_projection",
        }
    }
}

/// Project gold entities from the original text onto a perturbed version.
///
/// The projection uses a unique minimum-edit alignment over character offsets. A gold entity is
/// usable only when its source span agrees with its surface text and it retains a non-empty,
/// contiguous perturbed span. Ambiguous alignments and inputs over the bounded alignment budget
/// return an exclusion reason, causing the caller to record and exclude that example.
fn project_gold_entities(
    original_text: &str,
    perturbed_text: &str,
    gold: &[Entity],
) -> std::result::Result<Vec<Entity>, ProjectionError> {
    const MAX_ALIGNMENT_CELLS: usize = 4_000_000;

    if gold.is_empty() {
        return Ok(Vec::new());
    }

    let original: Vec<char> = original_text.chars().collect();
    if original_text == perturbed_text {
        validate_gold_entities(&original, gold)?;
        return Ok(gold.to_vec());
    }

    let perturbed: Vec<char> = perturbed_text.chars().collect();
    let rows = original
        .len()
        .checked_add(1)
        .ok_or(ProjectionError::AlignmentTooLarge)?;
    let cols = perturbed
        .len()
        .checked_add(1)
        .ok_or(ProjectionError::AlignmentTooLarge)?;
    if rows
        .checked_mul(cols)
        .ok_or(ProjectionError::AlignmentTooLarge)?
        > MAX_ALIGNMENT_CELLS
    {
        return Err(ProjectionError::AlignmentTooLarge);
    }
    #[derive(Clone, Copy)]
    enum Step {
        Diagonal,
        Delete,
        Insert,
    }

    let index = |i: usize, j: usize| i * cols + j;
    let mut costs = vec![0usize; rows * cols];
    let mut paths = vec![0u8; rows * cols];
    let mut steps = vec![None; rows * cols];
    paths[0] = 1;

    for i in 0..rows {
        for j in 0..cols {
            if i == 0 && j == 0 {
                continue;
            }

            let mut best_cost = usize::MAX;
            let mut path_count = 0u8;
            let mut chosen_step = None;
            let mut consider = |cost: usize, count: u8, step: Step| {
                if cost < best_cost {
                    best_cost = cost;
                    path_count = count;
                    chosen_step = Some(step);
                } else if cost == best_cost {
                    path_count = path_count.saturating_add(count).min(2);
                }
            };

            if i > 0 && j > 0 {
                let substitution_cost = usize::from(original[i - 1] != perturbed[j - 1]);
                let previous = index(i - 1, j - 1);
                consider(
                    costs[previous] + substitution_cost,
                    paths[previous],
                    Step::Diagonal,
                );
            }
            if i > 0 {
                let previous = index(i - 1, j);
                consider(costs[previous] + 1, paths[previous], Step::Delete);
            }
            if j > 0 {
                let previous = index(i, j - 1);
                consider(costs[previous] + 1, paths[previous], Step::Insert);
            }

            let current = index(i, j);
            costs[current] = best_cost;
            paths[current] = path_count;
            steps[current] = chosen_step;
        }
    }

    if paths[index(original.len(), perturbed.len())] != 1 {
        return Err(ProjectionError::AmbiguousAlignment);
    }

    let mut reversed_steps = Vec::with_capacity(original.len() + perturbed.len());
    let (mut i, mut j) = (original.len(), perturbed.len());
    while i > 0 || j > 0 {
        let step = steps[index(i, j)].ok_or(ProjectionError::AmbiguousAlignment)?;
        reversed_steps.push(step);
        match step {
            Step::Diagonal => {
                i -= 1;
                j -= 1;
            }
            Step::Delete => i -= 1,
            Step::Insert => j -= 1,
        }
    }
    reversed_steps.reverse();

    #[derive(Clone, Copy)]
    enum Owner {
        Source(usize),
        Gap(usize),
    }

    let mut owners = Vec::with_capacity(perturbed.len());
    let (mut source_index, mut target_index) = (0usize, 0usize);
    for step in reversed_steps {
        match step {
            Step::Diagonal => {
                owners.push(Owner::Source(source_index));
                source_index += 1;
                target_index += 1;
            }
            Step::Delete => source_index += 1,
            Step::Insert => {
                owners.push(Owner::Gap(source_index));
                target_index += 1;
            }
        }
    }
    debug_assert_eq!(target_index, perturbed.len());

    gold.iter()
        .map(|entity| {
            let start = entity.start();
            let end = entity.end();
            if start >= end || end > original.len() {
                return Err(ProjectionError::InvalidGoldSpan);
            }
            if original[start..end].iter().collect::<String>() != entity.text {
                return Err(ProjectionError::GoldTextMismatch);
            }

            let mut first = None;
            let mut last = 0usize;
            for (offset, owner) in owners.iter().enumerate() {
                let belongs_to_entity = match owner {
                    Owner::Source(index) => start <= *index && *index < end,
                    Owner::Gap(index) => start < *index && *index < end,
                };
                if belongs_to_entity {
                    first.get_or_insert(offset);
                    last = offset + 1;
                }
            }
            let first = first.ok_or(ProjectionError::EmptyProjection)?;
            if owners[first..last].iter().any(|owner| match owner {
                Owner::Source(index) => !(*index < end && start <= *index),
                Owner::Gap(index) => !(*index < end && start < *index),
            }) {
                return Err(ProjectionError::NonContiguousProjection);
            }

            Ok(Entity::new(
                perturbed[first..last].iter().collect::<String>(),
                entity.entity_type.clone(),
                first,
                last,
                entity.confidence,
            ))
        })
        .collect()
}

fn validate_gold_entities(
    original: &[char],
    gold: &[Entity],
) -> std::result::Result<(), ProjectionError> {
    for entity in gold {
        let start = entity.start();
        let end = entity.end();
        if start >= end || end > original.len() {
            return Err(ProjectionError::InvalidGoldSpan);
        }
        if original[start..end].iter().collect::<String>() != entity.text {
            return Err(ProjectionError::GoldTextMismatch);
        }
    }
    Ok(())
}

fn average_metrics(metrics: &[(f64, f64, f64)]) -> Option<(f64, f64, f64)> {
    (!metrics.is_empty()).then(|| {
        let count = metrics.len() as f64;
        (
            metrics.iter().map(|(p, _, _)| p).sum::<f64>() / count,
            metrics.iter().map(|(_, r, _)| r).sum::<f64>() / count,
            metrics.iter().map(|(_, _, f)| f).sum::<f64>() / count,
        )
    })
}

/// Compute strict span-and-type P/R/F1 using the canonical one-to-one NER matcher.
fn compute_simple_metrics(predicted: &[Entity], gold: &[Entity]) -> (f64, f64, f64) {
    let strict = evaluate_entities(gold, predicted).strict;
    (
        strict.precision_exact(),
        strict.recall_exact(),
        strict.f1_exact(),
    )
}

/// Grade robustness score.
pub fn robustness_grade(score: f64) -> &'static str {
    if score >= 0.95 {
        "Excellent robustness"
    } else if score >= 0.85 {
        "Good robustness"
    } else if score >= 0.70 {
        "Moderate robustness"
    } else if score >= 0.50 {
        "Poor robustness"
    } else {
        "Very poor robustness"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typo_swap() {
        let evaluator = RobustnessEvaluator {
            intensity: 0.5,
            ..Default::default()
        };

        let original = "hello world";
        let perturbed = evaluator.apply_perturbation(original, Perturbation::TypoSwap);

        // Should be different but similar length
        assert!(!perturbed.is_empty());
    }

    #[test]
    fn test_case_upper() {
        let evaluator = RobustnessEvaluator::default();
        let perturbed = evaluator.apply_perturbation("Hello World", Perturbation::CaseUpper);
        assert_eq!(perturbed, "HELLO WORLD");
    }

    #[test]
    fn test_case_lower() {
        let evaluator = RobustnessEvaluator::default();
        let perturbed = evaluator.apply_perturbation("Hello World", Perturbation::CaseLower);
        assert_eq!(perturbed, "hello world");
    }

    #[test]
    fn test_punctuation_remove() {
        let evaluator = RobustnessEvaluator::default();
        let perturbed =
            evaluator.apply_perturbation("Hello, World!", Perturbation::PunctuationRemove);
        assert_eq!(perturbed, "Hello World");
    }

    #[test]
    fn test_generate_variants() {
        let evaluator = RobustnessEvaluator::default();
        let variants = evaluator.generate_variants("Test text");

        assert!(!variants.is_empty());
        assert!(variants.iter().any(|(p, _)| *p == Perturbation::None));
    }

    #[test]
    fn test_homoglyph() {
        assert_eq!(homoglyph('a'), 'а'); // Cyrillic а
        assert_eq!(homoglyph('z'), 'z'); // No homoglyph
    }

    #[test]
    fn test_robustness_grades() {
        assert_eq!(robustness_grade(0.98), "Excellent robustness");
        assert_eq!(robustness_grade(0.90), "Good robustness");
        assert_eq!(robustness_grade(0.75), "Moderate robustness");
        assert_eq!(robustness_grade(0.60), "Poor robustness");
        assert_eq!(robustness_grade(0.30), "Very poor robustness");
    }

    #[test]
    fn projects_gold_surface_and_offsets_after_punctuation_removal() {
        let original = "Alice, works at Acme.";
        let gold = vec![
            Entity::new("Alice", crate::EntityType::Person, 0, 5, 1.0),
            Entity::new("Acme", crate::EntityType::Organization, 16, 20, 1.0),
        ];

        let projected = project_gold_entities(original, "Alice works at Acme", &gold).unwrap();

        assert_eq!(projected[0].text, "Alice");
        assert_eq!((projected[0].start(), projected[0].end()), (0, 5));
        assert_eq!(projected[1].text, "Acme");
        assert_eq!((projected[1].start(), projected[1].end()), (15, 19));
    }

    #[test]
    fn excludes_ambiguous_or_inconsistent_gold_projections() {
        let gold = vec![Entity::new("a", crate::EntityType::Person, 0, 1, 1.0)];
        assert!(matches!(
            project_gold_entities("a", "aa", &gold),
            Err(ProjectionError::AmbiguousAlignment)
        ));

        let inconsistent = vec![Entity::new("Bob", crate::EntityType::Person, 0, 3, 1.0)];
        assert!(matches!(
            project_gold_entities("Ann", "Ann", &inconsistent),
            Err(ProjectionError::GoldTextMismatch)
        ));
    }

    #[test]
    fn strict_metrics_do_not_credit_duplicate_predictions() {
        let gold = vec![Entity::new("Alice", crate::EntityType::Person, 0, 5, 1.0)];
        let predicted = vec![
            Entity::new("Alice", crate::EntityType::Person, 0, 5, 1.0),
            Entity::new("Alice", crate::EntityType::Person, 0, 5, 1.0),
        ];

        let (precision, recall, f1) = compute_simple_metrics(&predicted, &gold);

        assert_eq!(precision, 0.5);
        assert_eq!(recall, 1.0);
        assert!((f1 - 2.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn retains_an_entirely_unscoreable_perturbation_with_coverage_details() {
        let evaluator = RobustnessEvaluator {
            perturbations: vec![Perturbation::PunctuationExtra],
            intensity: 1.0,
            ..Default::default()
        };
        let model = anno::AnyModel::new("empty", "returns no entities", vec![], |_, _| Ok(vec![]));
        let test_cases = vec![(
            "!".to_string(),
            vec![Entity::new("!", crate::EntityType::Person, 0, 1, 1.0)],
        )];

        let results = evaluator.evaluate(&model, &test_cases);
        let metrics = &results.by_perturbation["PunctuationExtra"];

        assert_eq!(metrics.count, 0);
        assert_eq!(metrics.excluded_count, 1);
        assert_eq!(metrics.exclusion_reasons["ambiguous_alignment"], 1);
        assert_eq!(results.avg_perturbed_f1, 0.0);
        assert!(!results.coverage_complete);
    }

    #[test]
    fn scores_an_over_budget_unchanged_baseline_without_alignment() {
        let text = "a".repeat(2_000);
        let gold = vec![Entity::new(
            text.clone(),
            crate::EntityType::Person,
            0,
            text.chars().count(),
            1.0,
        )];
        let model = anno::AnyModel::new("echo", "returns the whole input", vec![], |text, _| {
            Ok(vec![Entity::new(
                text,
                crate::EntityType::Person,
                0,
                text.chars().count(),
                1.0,
            )])
        });
        let evaluator = RobustnessEvaluator::new(vec![Perturbation::None]);

        let results = evaluator.evaluate(&model, &[(text, gold)]);

        assert_eq!(results.by_perturbation["None"].count, 1);
        assert_eq!(results.baseline_f1, 1.0);
        assert!(results.coverage_complete);
    }

    #[test]
    fn bounds_alignment_for_changed_over_budget_text() {
        let original = "a".repeat(2_000);
        let perturbed = "b".repeat(2_000);
        let gold = vec![Entity::new(
            original.clone(),
            crate::EntityType::Person,
            0,
            original.chars().count(),
            1.0,
        )];

        assert!(matches!(
            project_gold_entities(&original, &perturbed, &gold),
            Err(ProjectionError::AlignmentTooLarge)
        ));
    }
}
