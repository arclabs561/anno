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
//! use anno::eval::robustness::{RobustnessEvaluator, Perturbation};
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
        self.next() % 2 == 0
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
    /// Average F1 across all perturbations
    pub avg_perturbed_f1: f64,
    /// Robustness score: avg_perturbed_f1 / baseline_f1 (1.0 = perfectly robust)
    pub robustness_score: f64,
    /// Worst perturbation type
    pub worst_perturbation: String,
    /// Best perturbation type (often "None")
    pub best_perturbation: String,
    /// Total examples tested
    pub total_examples: usize,
}

/// Metrics for a single perturbation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationMetrics {
    /// F1 score under this perturbation
    pub f1: f64,
    /// Precision under this perturbation
    pub precision: f64,
    /// Recall under this perturbation
    pub recall: f64,
    /// Relative change from baseline: (perturbed - baseline) / baseline
    pub relative_change: f64,
    /// Number of examples tested
    pub count: usize,
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

            Perturbation::CaseMixed => text
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if i % 2 == 0 {
                        c.to_uppercase().next().unwrap_or(c)
                    } else {
                        c.to_lowercase().next().unwrap_or(c)
                    }
                })
                .collect(),

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
        let mut by_perturbation: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new();

        for (text, gold_entities) in test_cases {
            for &perturbation in &self.perturbations {
                let perturbed = self.apply_perturbation(text, perturbation);
                let predicted = model.extract_entities(&perturbed, None).unwrap_or_default();

                // Compute metrics (simplified - just count matches)
                let (precision, recall, f1) =
                    compute_simple_metrics(&predicted, gold_entities, text, &perturbed);

                by_perturbation
                    .entry(format!("{:?}", perturbation))
                    .or_default()
                    .push((precision, recall, f1));
            }
        }

        // Aggregate metrics
        let mut aggregated: HashMap<String, PerturbationMetrics> = HashMap::new();
        let baseline_f1 = by_perturbation
            .get("None")
            .map(|v| v.iter().map(|(_, _, f1)| f1).sum::<f64>() / v.len() as f64)
            .unwrap_or(0.0);

        for (name, metrics) in &by_perturbation {
            let avg_precision =
                metrics.iter().map(|(p, _, _)| p).sum::<f64>() / metrics.len() as f64;
            let avg_recall = metrics.iter().map(|(_, r, _)| r).sum::<f64>() / metrics.len() as f64;
            let avg_f1 = metrics.iter().map(|(_, _, f)| f).sum::<f64>() / metrics.len() as f64;
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
                    count: metrics.len(),
                },
            );
        }

        // Find best/worst
        let (worst, _) = aggregated
            .iter()
            .filter(|(k, _)| k.as_str() != "None")
            .min_by(|a, b| {
                a.1.f1
                    .partial_cmp(&b.1.f1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, v)| (k.clone(), v.f1))
            .unwrap_or(("None".to_string(), baseline_f1));

        let (best, _) = aggregated
            .iter()
            .max_by(|a, b| {
                a.1.f1
                    .partial_cmp(&b.1.f1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, v)| (k.clone(), v.f1))
            .unwrap_or(("None".to_string(), baseline_f1));

        // Average F1 across perturbations (excluding baseline)
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

    let lower = c.to_lowercase().next().unwrap_or(c);
    for (keys, neighbors) in keyboard {
        if keys.contains(&lower) && !neighbors.is_empty() {
            let idx = rng.gen_range(neighbors.len());
            let neighbor = neighbors[idx];
            return if c.is_uppercase() {
                neighbor.to_uppercase().next().unwrap_or(neighbor)
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
    match c.to_lowercase().next().unwrap_or(c) {
        'a' => 'á',
        'e' => 'é',
        'i' => 'í',
        'o' => 'ó',
        'u' => 'ú',
        'n' => 'ñ',
        _ => c,
    }
}

/// Compute simple P/R/F1 metrics.
fn compute_simple_metrics(
    predicted: &[Entity],
    gold: &[Entity],
    _original_text: &str,
    _perturbed_text: &str,
) -> (f64, f64, f64) {
    // Simplified matching: count entities by type
    let mut correct = 0;

    for pred in predicted {
        if gold.iter().any(|g| {
            g.entity_type == pred.entity_type && g.text.to_lowercase() == pred.text.to_lowercase()
        }) {
            correct += 1;
        }
    }

    let precision = if predicted.is_empty() {
        0.0
    } else {
        correct as f64 / predicted.len() as f64
    };
    let recall = if gold.is_empty() {
        0.0
    } else {
        correct as f64 / gold.len() as f64
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
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
}
