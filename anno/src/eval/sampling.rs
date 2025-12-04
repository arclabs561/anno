//! Sampling strategies for NER evaluation.
//!
//! # Research Context
//!
//! Random sampling introduces bias when entity types are imbalanced.
//!
//! | Problem | Effect | Solution |
//! |---------|--------|----------|
//! | Type skew | High F1 on frequent types, low on rare | Stratified sampling |
//! | Seed sensitivity | Results vary wildly across seeds | Multiple seeds + variance |
//! | Dataset size | Small samples → high variance | Report confidence intervals |
//!
//! # Stratified Sampling (Recommended)
//!
//! Maintains proportional entity type distribution from the full dataset.
//! Critical for domain-specific datasets where some types are rare.
//!
//! ```text
//! Full dataset:    PER (60%), ORG (30%), LOC (10%)
//! Random sample:   PER (75%), ORG (20%), LOC (5%)   ← Biased!
//! Stratified:      PER (60%), ORG (30%), LOC (10%)  ← Representative
//! ```
//!
//! # Example
//!
//! ```rust
//! use anno::eval::sampling::stratified_sample;
//! use anno::eval::datasets::GoldEntity;
//! use anno::EntityType;
//!
//! let cases: Vec<(String, Vec<GoldEntity>)> = vec![
//!     ("John works at Apple".into(), vec![
//!         GoldEntity::new("John", EntityType::Person, 0),
//!         GoldEntity::new("Apple", EntityType::Organization, 14),
//!     ]),
//!     // ... more cases
//! ];
//!
//! // Sample 100 cases, maintaining entity type proportions
//! let sample = stratified_sample(&cases, 100, 42);
//! ```

use crate::eval::datasets::GoldEntity;
use crate::TypeMapper;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Stratified sampling maintaining entity type proportions.
///
/// # Arguments
/// * `cases` - Input test cases (text, gold entities)
/// * `target_size` - Maximum number of cases to return
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// A subset of cases with proportional entity type distribution.
///
/// # Note on True Stratification
///
/// This is a simplified version using hash-based pseudo-random ordering.
/// For true stratified sampling by entity type, use `stratified_sample_ner`
/// which groups by type first.
pub fn stratified_sample(
    cases: &[(String, Vec<GoldEntity>)],
    target_size: usize,
    seed: u64,
) -> Vec<(String, Vec<GoldEntity>)> {
    if cases.len() <= target_size {
        return cases.to_vec();
    }

    // Hash-based deterministic shuffle
    let mut indexed: Vec<(usize, u64)> = cases
        .iter()
        .enumerate()
        .map(|(i, (text, _))| {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            text.hash(&mut hasher);
            (i, hasher.finish())
        })
        .collect();

    indexed.sort_by_key(|(_, hash)| *hash);
    indexed.truncate(target_size);
    indexed.sort_by_key(|(i, _)| *i); // Preserve relative order

    indexed.iter().map(|(i, _)| cases[*i].clone()).collect()
}

/// Stratified sampling with entity type awareness.
///
/// Groups cases by their primary entity type and samples proportionally
/// from each group to maintain the original type distribution.
///
/// # Arguments
/// * `cases` - Input test cases
/// * `target_size` - Maximum cases to return
/// * `seed` - Random seed for reproducibility
/// * `type_mapper` - Optional mapper for normalizing domain-specific types
///
/// # Example
///
/// ```rust
/// use anno::eval::sampling::stratified_sample_ner;
/// use anno::eval::datasets::GoldEntity;
/// use anno::EntityType;
///
/// let cases = vec![
///     ("John works at Apple".into(), vec![
///         GoldEntity::new("John", EntityType::Person, 0),
///     ]),
/// ];
///
/// let sample = stratified_sample_ner(&cases, 100, 42, None);
/// ```
pub fn stratified_sample_ner(
    cases: &[(String, Vec<GoldEntity>)],
    target_size: usize,
    seed: u64,
    type_mapper: Option<&TypeMapper>,
) -> Vec<(String, Vec<GoldEntity>)> {
    use std::collections::HashMap;

    if cases.len() <= target_size {
        return cases.to_vec();
    }

    // Group cases by dominant entity type
    let mut by_type: HashMap<String, Vec<usize>> = HashMap::new();

    for (idx, (_, entities)) in cases.iter().enumerate() {
        // Use the first entity's type as the "dominant" type for grouping
        let type_key = if let Some(e) = entities.first() {
            let mapped = if let Some(mapper) = type_mapper {
                mapper.normalize(&e.original_label)
            } else {
                e.entity_type.clone()
            };
            format!("{:?}", mapped)
        } else {
            "EMPTY".to_string()
        };

        by_type.entry(type_key).or_default().push(idx);
    }

    // Calculate proportional allocation
    let total_cases = cases.len();
    let mut result_indices = Vec::with_capacity(target_size);

    for indices in by_type.values_mut() {
        let proportion = indices.len() as f64 / total_cases as f64;
        let allocation = (proportion * target_size as f64).ceil() as usize;

        // Shuffle this group's indices using hash-based ordering
        hash_shuffle(indices, seed);

        // Take allocation from this group
        result_indices.extend(indices.iter().take(allocation.min(indices.len())).copied());
    }

    // If we over-allocated, trim with hash-based ordering
    if result_indices.len() > target_size {
        hash_shuffle(&mut result_indices, seed);
        result_indices.truncate(target_size);
    }

    // Sort to preserve relative order (better for debugging)
    result_indices.sort();

    result_indices.iter().map(|&i| cases[i].clone()).collect()
}

/// Hash-based deterministic shuffle (no external crate needed).
///
/// Sorts indices by their hash value, which produces a deterministic
/// pseudo-random ordering for the given seed.
fn hash_shuffle(indices: &mut [usize], seed: u64) {
    if indices.len() <= 1 {
        return;
    }

    // Compute hash for each index and sort by hash
    let mut hashed: Vec<(usize, u64)> = indices
        .iter()
        .map(|&idx| {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            idx.hash(&mut hasher);
            (idx, hasher.finish())
        })
        .collect();

    hashed.sort_by_key(|(_, hash)| *hash);

    // Copy back shuffled indices
    for (i, (idx, _)) in hashed.into_iter().enumerate() {
        indices[i] = idx;
    }
}

/// Run evaluation with multiple seeds and aggregate variance.
///
/// # Research Context
///
/// Single-seed evaluations are unreliable:
/// - F1 can vary ±5% across seeds on small datasets
/// - Always report mean ± CI, not point estimates
///
/// # Arguments
/// * `eval_fn` - Evaluation function that takes a seed and returns F1 score
/// * `seeds` - Seeds to run (recommend 5+)
///
/// # Returns
/// (mean, std_dev, min, max) of F1 scores
pub fn multi_seed_eval<F>(eval_fn: F, seeds: &[u64]) -> (f64, f64, f64, f64)
where
    F: Fn(u64) -> f64,
{
    if seeds.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let scores: Vec<f64> = seeds.iter().map(|&s| eval_fn(s)).collect();

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance = if scores.len() > 1 {
        scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (scores.len() - 1) as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    (mean, std_dev, min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno_core::EntityType;

    fn make_test_cases() -> Vec<(String, Vec<GoldEntity>)> {
        vec![
            (
                "John works at Apple".into(),
                vec![
                    GoldEntity::new("John", EntityType::Person, 0),
                    GoldEntity::new("Apple", EntityType::Organization, 14),
                ],
            ),
            (
                "Meeting on January 15".into(),
                vec![GoldEntity::new("January 15", EntityType::Date, 11)],
            ),
            (
                "Price is $500".into(),
                vec![GoldEntity::new("$500", EntityType::Money, 9)],
            ),
        ]
    }

    #[test]
    fn test_stratified_sample_smaller() {
        let cases = make_test_cases();
        let sample = stratified_sample(&cases, 10, 42);
        assert_eq!(sample.len(), cases.len()); // All returned if target > len
    }

    #[test]
    fn test_stratified_sample_deterministic() {
        let cases = make_test_cases();
        let s1 = stratified_sample(&cases, 2, 42);
        let s2 = stratified_sample(&cases, 2, 42);
        assert_eq!(s1.len(), s2.len());
        assert_eq!(s1[0].0, s2[0].0); // Same results for same seed
    }

    #[test]
    fn test_stratified_sample_different_seeds() {
        let cases: Vec<_> = (0..100)
            .map(|i| {
                (
                    format!("Text {}", i),
                    vec![GoldEntity::new("entity", EntityType::Person, 0)],
                )
            })
            .collect();

        let s1 = stratified_sample(&cases, 10, 42);
        let s2 = stratified_sample(&cases, 10, 123);

        // Different seeds should (usually) produce different orderings
        let texts1: Vec<_> = s1.iter().map(|(t, _)| t.clone()).collect();
        let texts2: Vec<_> = s2.iter().map(|(t, _)| t.clone()).collect();
        assert_ne!(texts1, texts2);
    }

    #[test]
    fn test_multi_seed_eval() {
        let (mean, std, min, max) =
            multi_seed_eval(|seed| 0.8 + (seed as f64 % 10.0) / 100.0, &[1, 2, 3, 4, 5]);

        assert!(mean > 0.8);
        assert!(mean < 0.9);
        assert!(std >= 0.0);
        assert!(min <= mean);
        assert!(max >= mean);
    }
}
