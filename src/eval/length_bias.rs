//! Entity length bias evaluation for Named Entity Recognition.
//!
//! Measures performance differences based on entity text length.
//! Models often exhibit bias toward entity lengths common in training data,
//! performing worse on very short or very long entities.
//!
//! # Research Background
//!
//! - Jeong & Kang (2021): "Regularization for Long Named Entity Recognition"
//!   - Pre-trained language models tend to be biased toward dataset patterns
//!   - Length statistics of training data directly influence performance
//!
//! # Key Metrics
//!
//! - **Length Bucket Recognition Rate**: Performance by character/word length
//! - **Length Parity Gap**: Max difference across length buckets
//! - **Short Entity Bias**: Performance on 1-2 word entities vs longer
//!
//! # Example
//!
//! ```rust
//! use anno::eval::length_bias::{EntityLengthEvaluator, create_length_varied_dataset};
//!
//! let examples = create_length_varied_dataset();
//! let evaluator = EntityLengthEvaluator::default();
//! // let results = evaluator.evaluate(&model, &examples);
//! ```

use crate::{EntityType, Model};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Length Categories
// =============================================================================

/// Length bucket for entity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LengthBucket {
    /// Very short: 1-5 characters (e.g., "NYC", "IBM")
    VeryShort,
    /// Short: 6-15 characters (e.g., "John Smith")
    Short,
    /// Medium: 16-30 characters (e.g., "University of California")
    Medium,
    /// Long: 31-50 characters (e.g., "Massachusetts Institute of Technology")
    Long,
    /// Very long: 51+ characters (e.g., compound organization names)
    VeryLong,
}

impl LengthBucket {
    /// Classify a string by its character length.
    pub fn from_char_length(len: usize) -> Self {
        match len {
            0..=5 => LengthBucket::VeryShort,
            6..=15 => LengthBucket::Short,
            16..=30 => LengthBucket::Medium,
            31..=50 => LengthBucket::Long,
            _ => LengthBucket::VeryLong,
        }
    }

    /// Classify a string by its word count.
    pub fn from_word_count(words: usize) -> Self {
        match words {
            0..=1 => LengthBucket::VeryShort,
            2 => LengthBucket::Short,
            3..=4 => LengthBucket::Medium,
            5..=7 => LengthBucket::Long,
            _ => LengthBucket::VeryLong,
        }
    }
}

/// Word count bucket for finer-grained analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WordCountBucket {
    /// Single word (e.g., "Microsoft")
    SingleWord,
    /// Two words (e.g., "John Smith")
    TwoWords,
    /// Three words (e.g., "New York City")
    ThreeWords,
    /// Four or more words
    FourPlusWords,
}

impl WordCountBucket {
    /// Classify by word count.
    pub fn from_count(count: usize) -> Self {
        match count {
            0..=1 => WordCountBucket::SingleWord,
            2 => WordCountBucket::TwoWords,
            3 => WordCountBucket::ThreeWords,
            _ => WordCountBucket::FourPlusWords,
        }
    }
}

// =============================================================================
// Length Test Example
// =============================================================================

/// An entity example with length metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthTestExample {
    /// The entity text
    pub entity_text: String,
    /// Full sentence containing the entity
    pub sentence: String,
    /// Expected entity type
    pub entity_type: EntityType,
    /// Character length of entity
    pub char_length: usize,
    /// Word count of entity
    pub word_count: usize,
    /// Character length bucket
    pub char_bucket: LengthBucket,
    /// Word count bucket
    pub word_bucket: WordCountBucket,
}

impl LengthTestExample {
    /// Create a new length test example.
    pub fn new(entity: &str, entity_type: EntityType) -> Self {
        let sentence = format!("The entity {} was mentioned.", entity);
        let char_length = entity.chars().count();
        let word_count = entity.split_whitespace().count();

        Self {
            entity_text: entity.to_string(),
            sentence,
            entity_type,
            char_length,
            word_count,
            char_bucket: LengthBucket::from_char_length(char_length),
            word_bucket: WordCountBucket::from_count(word_count),
        }
    }

    /// Create with a custom sentence.
    pub fn with_sentence(entity: &str, sentence: &str, entity_type: EntityType) -> Self {
        let char_length = entity.chars().count();
        let word_count = entity.split_whitespace().count();

        Self {
            entity_text: entity.to_string(),
            sentence: sentence.to_string(),
            entity_type,
            char_length,
            word_count,
            char_bucket: LengthBucket::from_char_length(char_length),
            word_bucket: WordCountBucket::from_count(word_count),
        }
    }
}

// =============================================================================
// Evaluation Results
// =============================================================================

/// Results of entity length bias evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthBiasResults {
    /// Overall recognition rate
    pub overall_recognition_rate: f64,
    /// Recognition rate by character length bucket
    pub by_char_bucket: HashMap<String, f64>,
    /// Recognition rate by word count bucket
    pub by_word_bucket: HashMap<String, f64>,
    /// Recognition rate by entity type
    pub by_entity_type: HashMap<String, f64>,
    /// Maximum gap across character length buckets
    pub char_length_parity_gap: f64,
    /// Maximum gap across word count buckets
    pub word_count_parity_gap: f64,
    /// Short (1-2 words) vs long (4+ words) gap
    pub short_vs_long_gap: f64,
    /// Average character length of correctly recognized entities
    pub avg_recognized_char_length: f64,
    /// Average character length of missed entities
    pub avg_missed_char_length: f64,
    /// Total examples tested
    pub total_tested: usize,
}

// =============================================================================
// Evaluator
// =============================================================================

/// Evaluator for entity length bias.
#[derive(Debug, Clone, Default)]
pub struct EntityLengthEvaluator {
    /// Include detailed per-example results
    pub detailed: bool,
}

impl EntityLengthEvaluator {
    /// Create a new evaluator.
    pub fn new(detailed: bool) -> Self {
        Self { detailed }
    }

    /// Evaluate NER model for length bias.
    pub fn evaluate(&self, model: &dyn Model, examples: &[LengthTestExample]) -> LengthBiasResults {
        let mut by_char_bucket: HashMap<String, (usize, usize)> = HashMap::new();
        let mut by_word_bucket: HashMap<String, (usize, usize)> = HashMap::new();
        let mut by_entity_type: HashMap<String, (usize, usize)> = HashMap::new();
        let mut total_recognized = 0;
        let mut recognized_char_lengths: Vec<usize> = Vec::new();
        let mut missed_char_lengths: Vec<usize> = Vec::new();

        for example in examples {
            // Extract entities
            let entities = model
                .extract_entities(&example.sentence, None)
                .unwrap_or_default();

            // Check if entity was recognized with correct type
            let recognized = entities.iter().any(|e| {
                e.entity_type == example.entity_type
                    && example
                        .sentence
                        .get(e.start..e.end)
                        .map(|s| s.contains(&example.entity_text))
                        .unwrap_or(false)
            });

            if recognized {
                total_recognized += 1;
                recognized_char_lengths.push(example.char_length);
            } else {
                missed_char_lengths.push(example.char_length);
            }

            // Update char bucket stats
            let char_key = format!("{:?}", example.char_bucket);
            let char_entry = by_char_bucket.entry(char_key).or_insert((0, 0));
            char_entry.1 += 1;
            if recognized {
                char_entry.0 += 1;
            }

            // Update word bucket stats
            let word_key = format!("{:?}", example.word_bucket);
            let word_entry = by_word_bucket.entry(word_key).or_insert((0, 0));
            word_entry.1 += 1;
            if recognized {
                word_entry.0 += 1;
            }

            // Update entity type stats
            let type_key = format!("{:?}", example.entity_type);
            let type_entry = by_entity_type.entry(type_key).or_insert((0, 0));
            type_entry.1 += 1;
            if recognized {
                type_entry.0 += 1;
            }
        }

        // Convert counts to rates
        let to_rate = |counts: &HashMap<String, (usize, usize)>| -> HashMap<String, f64> {
            counts
                .iter()
                .map(|(k, (correct, total))| {
                    let rate = if *total > 0 {
                        *correct as f64 / *total as f64
                    } else {
                        0.0
                    };
                    (k.clone(), rate)
                })
                .collect()
        };

        let char_rates = to_rate(&by_char_bucket);
        let word_rates = to_rate(&by_word_bucket);
        let type_rates = to_rate(&by_entity_type);

        // Compute parity gaps
        let char_length_parity_gap = compute_max_gap(&char_rates);
        let word_count_parity_gap = compute_max_gap(&word_rates);

        // Short vs long gap
        let short_rate = word_rates
            .iter()
            .filter(|(k, _)| k.contains("SingleWord") || k.contains("TwoWords"))
            .map(|(_, v)| *v)
            .sum::<f64>()
            / 2.0;
        let long_rate = word_rates
            .get("FourPlusWords")
            .copied()
            .unwrap_or(short_rate);
        let short_vs_long_gap = (short_rate - long_rate).abs();

        // Average lengths
        let avg_recognized = if recognized_char_lengths.is_empty() {
            0.0
        } else {
            recognized_char_lengths.iter().sum::<usize>() as f64
                / recognized_char_lengths.len() as f64
        };
        let avg_missed = if missed_char_lengths.is_empty() {
            0.0
        } else {
            missed_char_lengths.iter().sum::<usize>() as f64 / missed_char_lengths.len() as f64
        };

        LengthBiasResults {
            overall_recognition_rate: if examples.is_empty() {
                0.0
            } else {
                total_recognized as f64 / examples.len() as f64
            },
            by_char_bucket: char_rates,
            by_word_bucket: word_rates,
            by_entity_type: type_rates,
            char_length_parity_gap,
            word_count_parity_gap,
            short_vs_long_gap,
            avg_recognized_char_length: avg_recognized,
            avg_missed_char_length: avg_missed,
            total_tested: examples.len(),
        }
    }
}

/// Compute maximum gap between any two rates.
fn compute_max_gap(rates: &HashMap<String, f64>) -> f64 {
    if rates.len() < 2 {
        return 0.0;
    }

    let values: Vec<f64> = rates.values().copied().collect();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    max - min
}

// =============================================================================
// Length-Varied Dataset
// =============================================================================

/// Create a dataset with entities of varying lengths.
pub fn create_length_varied_dataset() -> Vec<LengthTestExample> {
    vec![
        // === PERSON entities by length ===
        // Very short (abbreviations, initials)
        LengthTestExample::with_sentence(
            "JFK",
            "JFK gave a famous speech in Berlin.",
            EntityType::Person,
        ),
        LengthTestExample::with_sentence(
            "FDR",
            "FDR led the country through World War II.",
            EntityType::Person,
        ),
        // Short (typical names)
        LengthTestExample::with_sentence(
            "John Smith",
            "John Smith attended the meeting.",
            EntityType::Person,
        ),
        LengthTestExample::with_sentence(
            "Mary Johnson",
            "Mary Johnson won the award.",
            EntityType::Person,
        ),
        // Medium (names with middle name or title)
        LengthTestExample::with_sentence(
            "Dr. Martin Luther King",
            "Dr. Martin Luther King delivered a powerful speech.",
            EntityType::Person,
        ),
        LengthTestExample::with_sentence(
            "William Jefferson Clinton",
            "William Jefferson Clinton served as president.",
            EntityType::Person,
        ),
        // Long (full names with titles/suffixes)
        LengthTestExample::with_sentence(
            "His Royal Highness Prince William",
            "His Royal Highness Prince William visited the hospital.",
            EntityType::Person,
        ),
        // === ORGANIZATION entities by length ===
        // Very short
        LengthTestExample::with_sentence(
            "IBM",
            "IBM announced new products.",
            EntityType::Organization,
        ),
        LengthTestExample::with_sentence(
            "MIT",
            "MIT published research findings.",
            EntityType::Organization,
        ),
        LengthTestExample::with_sentence(
            "NASA",
            "NASA launched a new satellite.",
            EntityType::Organization,
        ),
        // Short
        LengthTestExample::with_sentence(
            "Google Inc",
            "Google Inc acquired the startup.",
            EntityType::Organization,
        ),
        LengthTestExample::with_sentence(
            "Apple Computer",
            "Apple Computer revolutionized mobile phones.",
            EntityType::Organization,
        ),
        // Medium
        LengthTestExample::with_sentence(
            "University of California",
            "University of California released the study.",
            EntityType::Organization,
        ),
        LengthTestExample::with_sentence(
            "World Health Organization",
            "World Health Organization issued guidelines.",
            EntityType::Organization,
        ),
        // Long
        LengthTestExample::with_sentence(
            "Massachusetts Institute of Technology",
            "Massachusetts Institute of Technology won the competition.",
            EntityType::Organization,
        ),
        LengthTestExample::with_sentence(
            "International Business Machines Corporation",
            "International Business Machines Corporation reported earnings.",
            EntityType::Organization,
        ),
        // Very long
        LengthTestExample::with_sentence(
            "United States Department of Health and Human Services",
            "United States Department of Health and Human Services announced the policy.",
            EntityType::Organization,
        ),
        LengthTestExample::with_sentence(
            "European Organization for Nuclear Research",
            "European Organization for Nuclear Research discovered the particle.",
            EntityType::Organization,
        ),
        // === LOCATION entities by length ===
        // Very short
        LengthTestExample::with_sentence(
            "NYC",
            "NYC is known for its skyline.",
            EntityType::Location,
        ),
        LengthTestExample::with_sentence("LA", "LA has beautiful weather.", EntityType::Location),
        // Short
        LengthTestExample::with_sentence(
            "New York",
            "New York is a bustling city.",
            EntityType::Location,
        ),
        LengthTestExample::with_sentence(
            "London",
            "London has many museums.",
            EntityType::Location,
        ),
        // Medium
        LengthTestExample::with_sentence(
            "San Francisco Bay Area",
            "San Francisco Bay Area is a tech hub.",
            EntityType::Location,
        ),
        LengthTestExample::with_sentence(
            "United Arab Emirates",
            "United Arab Emirates hosted the conference.",
            EntityType::Location,
        ),
        // Long
        LengthTestExample::with_sentence(
            "Democratic Republic of the Congo",
            "Democratic Republic of the Congo has vast resources.",
            EntityType::Location,
        ),
        LengthTestExample::with_sentence(
            "Saint Vincent and the Grenadines",
            "Saint Vincent and the Grenadines is in the Caribbean.",
            EntityType::Location,
        ),
        // Very long
        LengthTestExample::with_sentence(
            "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch",
            "Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is a town in Wales.",
            EntityType::Location,
        ),
    ]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_bucket_classification() {
        assert_eq!(LengthBucket::from_char_length(3), LengthBucket::VeryShort);
        assert_eq!(LengthBucket::from_char_length(10), LengthBucket::Short);
        assert_eq!(LengthBucket::from_char_length(25), LengthBucket::Medium);
        assert_eq!(LengthBucket::from_char_length(40), LengthBucket::Long);
        assert_eq!(LengthBucket::from_char_length(60), LengthBucket::VeryLong);
    }

    #[test]
    fn test_word_count_bucket() {
        assert_eq!(WordCountBucket::from_count(1), WordCountBucket::SingleWord);
        assert_eq!(WordCountBucket::from_count(2), WordCountBucket::TwoWords);
        assert_eq!(WordCountBucket::from_count(3), WordCountBucket::ThreeWords);
        assert_eq!(
            WordCountBucket::from_count(5),
            WordCountBucket::FourPlusWords
        );
    }

    #[test]
    fn test_create_length_dataset() {
        let examples = create_length_varied_dataset();

        // Should have examples in all length buckets
        let char_buckets: std::collections::HashSet<_> = examples
            .iter()
            .map(|e| format!("{:?}", e.char_bucket))
            .collect();

        assert!(
            char_buckets.contains("VeryShort"),
            "Should have very short entities"
        );
        assert!(char_buckets.contains("Short"), "Should have short entities");
        assert!(
            char_buckets.contains("Medium"),
            "Should have medium entities"
        );
        assert!(char_buckets.contains("Long"), "Should have long entities");
    }

    #[test]
    fn test_entity_type_coverage() {
        let examples = create_length_varied_dataset();

        let types: std::collections::HashSet<_> = examples
            .iter()
            .map(|e| format!("{:?}", e.entity_type))
            .collect();

        assert!(types.contains("Person"), "Should have PERSON entities");
        assert!(
            types.contains("Organization"),
            "Should have ORGANIZATION entities"
        );
        assert!(types.contains("Location"), "Should have LOCATION entities");
    }

    #[test]
    fn test_example_construction() {
        let example = LengthTestExample::new("John Smith", EntityType::Person);

        assert_eq!(example.entity_text, "John Smith");
        assert_eq!(example.char_length, 10);
        assert_eq!(example.word_count, 2);
        assert_eq!(example.char_bucket, LengthBucket::Short);
        assert_eq!(example.word_bucket, WordCountBucket::TwoWords);
    }
}
