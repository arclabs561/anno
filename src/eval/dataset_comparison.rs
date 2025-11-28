//! Dataset comparison for understanding distribution differences.
//!
//! Compares two NER datasets to understand:
//! - Entity type distribution differences
//! - Vocabulary overlap/divergence
//! - Entity length characteristics
//! - Difficulty estimation
//!
//! Useful for:
//! - Understanding domain gaps before cross-domain evaluation
//! - Selecting transfer learning source datasets
//! - Debugging performance differences across corpora
//!
//! # Example
//!
//! ```rust
//! use anno::eval::dataset_comparison::{DatasetStats, compare_datasets};
//! use anno::eval::synthetic::AnnotatedExample;
//!
//! let dataset_a = vec![
//!     AnnotatedExample::new("John works at Google.", vec![("John", "PER"), ("Google", "ORG")]),
//! ];
//! let dataset_b = vec![
//!     AnnotatedExample::new("Paris is beautiful.", vec![("Paris", "LOC")]),
//! ];
//!
//! let comparison = compare_datasets(&dataset_a, &dataset_b);
//! println!("Type distribution divergence: {:.3}", comparison.type_divergence);
//! println!("Vocabulary overlap: {:.1}%", comparison.vocab_overlap * 100.0);
//! ```

use crate::eval::synthetic::AnnotatedExample;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Dataset Statistics
// =============================================================================

/// Statistics about a single dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Number of examples
    pub num_examples: usize,
    /// Total number of entities
    pub num_entities: usize,
    /// Entity type distribution (type -> proportion)
    pub type_distribution: HashMap<String, f64>,
    /// Average entities per example
    pub avg_entities_per_example: f64,
    /// Vocabulary (unique tokens)
    pub vocab_size: usize,
    /// Entity length distribution stats
    pub entity_length_stats: LengthStats,
    /// Unique entity texts
    pub unique_entity_texts: usize,
    /// Entity text repetition rate (1.0 = all unique, lower = more repetition)
    pub entity_diversity: f64,
}

/// Statistics about entity lengths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthStats {
    /// Mean entity length in tokens
    pub mean: f64,
    /// Median entity length
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum length
    pub min: usize,
    /// Maximum length
    pub max: usize,
}

/// Comparison between two datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetComparison {
    /// Stats for dataset A
    pub stats_a: DatasetStats,
    /// Stats for dataset B
    pub stats_b: DatasetStats,
    /// Jensen-Shannon divergence of type distributions (0 = identical, 1 = disjoint)
    pub type_divergence: f64,
    /// Vocabulary overlap (Jaccard similarity)
    pub vocab_overlap: f64,
    /// Entity text overlap (Jaccard similarity)
    pub entity_text_overlap: f64,
    /// Types present in A but not B
    pub types_only_in_a: Vec<String>,
    /// Types present in B but not A
    pub types_only_in_b: Vec<String>,
    /// Estimated domain gap (heuristic combining multiple factors)
    pub estimated_domain_gap: f64,
    /// Recommendations for transfer learning
    pub recommendations: Vec<String>,
}

// =============================================================================
// Statistics Computation
// =============================================================================

/// Compute statistics for a dataset.
pub fn compute_stats(examples: &[AnnotatedExample]) -> DatasetStats {
    if examples.is_empty() {
        return DatasetStats {
            num_examples: 0,
            num_entities: 0,
            type_distribution: HashMap::new(),
            avg_entities_per_example: 0.0,
            vocab_size: 0,
            entity_length_stats: LengthStats {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                min: 0,
                max: 0,
            },
            unique_entity_texts: 0,
            entity_diversity: 1.0,
        };
    }

    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let mut vocab: HashSet<String> = HashSet::new();
    let mut entity_texts: HashSet<String> = HashSet::new();
    let mut entity_lengths: Vec<usize> = Vec::new();
    let mut total_entities = 0;

    for example in examples {
        // Collect vocabulary
        for token in example.text.split_whitespace() {
            vocab.insert(token.to_lowercase());
        }

        // Collect entity stats
        for entity in &example.entities {
            total_entities += 1;
            *type_counts.entry(entity.entity_type.to_string()).or_insert(0) += 1;
            entity_texts.insert(entity.text.to_lowercase());

            // Count tokens in entity
            let token_count = entity.text.split_whitespace().count().max(1);
            entity_lengths.push(token_count);
        }
    }

    // Compute type distribution
    let type_distribution: HashMap<String, f64> = type_counts
        .iter()
        .map(|(t, c)| (t.clone(), *c as f64 / total_entities.max(1) as f64))
        .collect();

    // Compute length stats
    let entity_length_stats = if entity_lengths.is_empty() {
        LengthStats {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0,
            max: 0,
        }
    } else {
        let mut sorted = entity_lengths.clone();
        sorted.sort_unstable();

        let mean = entity_lengths.iter().sum::<usize>() as f64 / entity_lengths.len() as f64;
        let median = sorted[sorted.len() / 2] as f64;
        let variance = entity_lengths
            .iter()
            .map(|&l| (l as f64 - mean).powi(2))
            .sum::<f64>()
            / entity_lengths.len() as f64;
        let std_dev = variance.sqrt();

        LengthStats {
            mean,
            median,
            std_dev,
            min: *sorted.first().unwrap_or(&0),
            max: *sorted.last().unwrap_or(&0),
        }
    };

    DatasetStats {
        num_examples: examples.len(),
        num_entities: total_entities,
        type_distribution,
        avg_entities_per_example: total_entities as f64 / examples.len() as f64,
        vocab_size: vocab.len(),
        entity_length_stats,
        unique_entity_texts: entity_texts.len(),
        entity_diversity: entity_texts.len() as f64 / total_entities.max(1) as f64,
    }
}

/// Compare two datasets.
pub fn compare_datasets(a: &[AnnotatedExample], b: &[AnnotatedExample]) -> DatasetComparison {
    let stats_a = compute_stats(a);
    let stats_b = compute_stats(b);

    // Collect vocabularies
    let vocab_a: HashSet<String> = a
        .iter()
        .flat_map(|e| e.text.split_whitespace().map(|t| t.to_lowercase()))
        .collect();
    let vocab_b: HashSet<String> = b
        .iter()
        .flat_map(|e| e.text.split_whitespace().map(|t| t.to_lowercase()))
        .collect();

    // Collect entity texts
    let entities_a: HashSet<String> = a
        .iter()
        .flat_map(|e| e.entities.iter().map(|ent| ent.text.to_lowercase()))
        .collect();
    let entities_b: HashSet<String> = b
        .iter()
        .flat_map(|e| e.entities.iter().map(|ent| ent.text.to_lowercase()))
        .collect();

    // Vocabulary overlap (Jaccard)
    let vocab_intersection = vocab_a.intersection(&vocab_b).count();
    let vocab_union = vocab_a.union(&vocab_b).count();
    let vocab_overlap = if vocab_union == 0 {
        1.0
    } else {
        vocab_intersection as f64 / vocab_union as f64
    };

    // Entity text overlap (Jaccard)
    let entity_intersection = entities_a.intersection(&entities_b).count();
    let entity_union = entities_a.union(&entities_b).count();
    let entity_text_overlap = if entity_union == 0 {
        1.0
    } else {
        entity_intersection as f64 / entity_union as f64
    };

    // Type distribution divergence (Jensen-Shannon)
    let type_divergence = jensen_shannon_divergence(&stats_a.type_distribution, &stats_b.type_distribution);

    // Types in one but not the other
    let types_a: HashSet<&String> = stats_a.type_distribution.keys().collect();
    let types_b: HashSet<&String> = stats_b.type_distribution.keys().collect();
    let types_only_in_a: Vec<String> = types_a.difference(&types_b).map(|s| (*s).clone()).collect();
    let types_only_in_b: Vec<String> = types_b.difference(&types_a).map(|s| (*s).clone()).collect();

    // Estimated domain gap (heuristic)
    let estimated_domain_gap = 0.4 * type_divergence
        + 0.3 * (1.0 - vocab_overlap)
        + 0.3 * (1.0 - entity_text_overlap);

    // Generate recommendations
    let recommendations = generate_recommendations(
        type_divergence,
        vocab_overlap,
        entity_text_overlap,
        &types_only_in_a,
        &types_only_in_b,
    );

    DatasetComparison {
        stats_a,
        stats_b,
        type_divergence,
        vocab_overlap,
        entity_text_overlap,
        types_only_in_a,
        types_only_in_b,
        estimated_domain_gap,
        recommendations,
    }
}

fn jensen_shannon_divergence(p: &HashMap<String, f64>, q: &HashMap<String, f64>) -> f64 {
    // Collect all keys
    let all_keys: HashSet<&String> = p.keys().chain(q.keys()).collect();

    if all_keys.is_empty() {
        return 0.0;
    }

    // Compute M = (P + Q) / 2
    let mut m: HashMap<&String, f64> = HashMap::new();
    for k in &all_keys {
        let p_val = p.get(*k).copied().unwrap_or(0.0);
        let q_val = q.get(*k).copied().unwrap_or(0.0);
        m.insert(*k, (p_val + q_val) / 2.0);
    }

    // Compute KL(P || M) and KL(Q || M)
    let kl_p_m: f64 = all_keys
        .iter()
        .map(|k| {
            let p_val = p.get(*k).copied().unwrap_or(0.0);
            let m_val = m.get(k).copied().unwrap_or(1e-10);
            if p_val > 0.0 {
                p_val * (p_val / m_val).ln()
            } else {
                0.0
            }
        })
        .sum();

    let kl_q_m: f64 = all_keys
        .iter()
        .map(|k| {
            let q_val = q.get(*k).copied().unwrap_or(0.0);
            let m_val = m.get(k).copied().unwrap_or(1e-10);
            if q_val > 0.0 {
                q_val * (q_val / m_val).ln()
            } else {
                0.0
            }
        })
        .sum();

    // JS divergence = (KL(P||M) + KL(Q||M)) / 2
    // Normalize to [0, 1] by dividing by ln(2)
    ((kl_p_m + kl_q_m) / 2.0) / 2.0_f64.ln()
}

fn generate_recommendations(
    type_div: f64,
    vocab_overlap: f64,
    entity_overlap: f64,
    types_only_a: &[String],
    types_only_b: &[String],
) -> Vec<String> {
    let mut recs = Vec::new();

    if type_div > 0.5 {
        recs.push("High type distribution divergence - consider domain adaptation".into());
    } else if type_div > 0.2 {
        recs.push("Moderate type divergence - transfer learning may require fine-tuning".into());
    }

    if vocab_overlap < 0.3 {
        recs.push("Low vocabulary overlap - domains use different terminology".into());
    }

    if entity_overlap < 0.1 {
        recs.push("Very few shared entities - gazetteer transfer unlikely to help".into());
    }

    if !types_only_a.is_empty() {
        recs.push(format!(
            "Types in source only: {:?} - target may not need these",
            types_only_a
        ));
    }

    if !types_only_b.is_empty() {
        recs.push(format!(
            "Types in target only: {:?} - source cannot help with these",
            types_only_b
        ));
    }

    if recs.is_empty() {
        recs.push("Datasets appear compatible for transfer learning".into());
    }

    recs
}

// =============================================================================
// Difficulty Estimation
// =============================================================================

/// Estimate relative difficulty of a dataset.
pub fn estimate_difficulty(stats: &DatasetStats) -> DifficultyEstimate {
    let mut factors = Vec::new();
    let mut score: f64 = 0.0;

    // More entity types = harder
    let num_types = stats.type_distribution.len();
    if num_types > 10 {
        factors.push("Many entity types (>10)".into());
        score += 0.2;
    } else if num_types > 5 {
        factors.push("Moderate entity types (5-10)".into());
        score += 0.1;
    }

    // Longer entities = harder
    if stats.entity_length_stats.mean > 3.0 {
        factors.push("Long average entity length (>3 tokens)".into());
        score += 0.2;
    }

    // High variance in entity length = harder
    if stats.entity_length_stats.std_dev > 2.0 {
        factors.push("High entity length variance".into());
        score += 0.1;
    }

    // Low entity diversity = easier (more repetition for learning)
    if stats.entity_diversity > 0.9 {
        factors.push("High entity diversity (few repeated entities)".into());
        score += 0.2;
    } else if stats.entity_diversity < 0.3 {
        factors.push("Low entity diversity (model can memorize)".into());
        score -= 0.1;
    }

    // Few entities per example = harder to learn context
    if stats.avg_entities_per_example < 1.0 {
        factors.push("Few entities per example (<1 avg)".into());
        score += 0.1;
    }

    let difficulty = match score {
        s if s < 0.2 => Difficulty::Easy,
        s if s < 0.4 => Difficulty::Medium,
        s if s < 0.6 => Difficulty::Hard,
        _ => Difficulty::VeryHard,
    };

    DifficultyEstimate {
        difficulty,
        score: score.clamp(0.0, 1.0),
        factors,
    }
}

/// Difficulty level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Difficulty {
    /// Simple examples with common entities
    Easy,
    /// Moderate complexity
    Medium,
    /// Complex examples requiring context
    Hard,
    /// Very challenging examples
    VeryHard,
}

/// Difficulty estimate with explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyEstimate {
    /// Overall difficulty
    pub difficulty: Difficulty,
    /// Numeric score (0-1, higher = harder)
    pub score: f64,
    /// Contributing factors
    pub factors: Vec<String>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example(text: &str, entities: Vec<(&str, &str)>) -> AnnotatedExample {
        use crate::eval::datasets::GoldEntity;
        use crate::eval::synthetic::{Domain, Difficulty};
        use crate::EntityType;

        let mut gold_entities = Vec::new();

        for (entity_text, entity_type_str) in entities {
            if let Some(start) = text.find(entity_text) {
                let entity_type = match entity_type_str {
                    "PER" => EntityType::Person,
                    "ORG" => EntityType::Organization,
                    "LOC" => EntityType::Location,
                    _ => EntityType::Other(entity_type_str.to_string()),
                };
                gold_entities.push(GoldEntity::new(entity_text, entity_type, start));
            }
        }

        AnnotatedExample {
            text: text.to_string(),
            entities: gold_entities,
            domain: Domain::News,
            difficulty: Difficulty::Easy,
        }
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.num_examples, 0);
        assert_eq!(stats.num_entities, 0);
    }

    #[test]
    fn test_compute_stats_basic() {
        let examples = vec![
            make_example("John works at Google.", vec![("John", "PER"), ("Google", "ORG")]),
            make_example("Paris is in France.", vec![("Paris", "LOC"), ("France", "LOC")]),
        ];

        let stats = compute_stats(&examples);

        assert_eq!(stats.num_examples, 2);
        assert_eq!(stats.num_entities, 4);
        assert_eq!(stats.avg_entities_per_example, 2.0);
        assert!(stats.type_distribution.contains_key("PER"));
        assert!(stats.type_distribution.contains_key("ORG"));
        assert!(stats.type_distribution.contains_key("LOC"));
    }

    #[test]
    fn test_compare_identical_datasets() {
        let examples = vec![make_example(
            "John works at Google.",
            vec![("John", "PER"), ("Google", "ORG")],
        )];

        let comparison = compare_datasets(&examples, &examples);

        assert!(comparison.type_divergence < 0.01);
        assert!((comparison.vocab_overlap - 1.0).abs() < 0.01);
        assert!((comparison.entity_text_overlap - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compare_different_datasets() {
        let a = vec![make_example("John works at Google.", vec![("John", "PER")])];
        let b = vec![make_example("Paris is beautiful.", vec![("Paris", "LOC")])];

        let comparison = compare_datasets(&a, &b);

        // Different types, different vocab, different entities
        assert!(comparison.type_divergence > 0.5);
        assert!(comparison.vocab_overlap < 0.5);
        assert!((comparison.entity_text_overlap - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_jensen_shannon_identical() {
        let mut p = HashMap::new();
        p.insert("A".into(), 0.5);
        p.insert("B".into(), 0.5);

        let js = jensen_shannon_divergence(&p, &p);
        assert!(js < 0.01);
    }

    #[test]
    fn test_jensen_shannon_disjoint() {
        let mut p = HashMap::new();
        p.insert("A".into(), 1.0);

        let mut q = HashMap::new();
        q.insert("B".into(), 1.0);

        let js = jensen_shannon_divergence(&p, &q);
        assert!(js > 0.9);
    }

    #[test]
    fn test_difficulty_estimation() {
        let easy_examples = vec![
            make_example("John works here.", vec![("John", "PER")]),
            make_example("John went home.", vec![("John", "PER")]),
        ];

        let hard_examples = vec![
            make_example(
                "International Business Machines Corporation announced.",
                vec![("International Business Machines Corporation", "ORG")],
            ),
        ];

        let easy_stats = compute_stats(&easy_examples);
        let hard_stats = compute_stats(&hard_examples);

        let easy_diff = estimate_difficulty(&easy_stats);
        let hard_diff = estimate_difficulty(&hard_stats);

        assert!(hard_diff.score >= easy_diff.score);
    }
}

