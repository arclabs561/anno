//! Synthetic NER Test Datasets
//!
//! This module provides access to synthetic NER datasets for testing and evaluation.
//! The actual data is organized in `dataset::synthetic` submodules by domain.
//!
//! # Research Context
//!
//! Synthetic data has known limitations (arXiv:2505.16814 "Does Synthetic Data Help NER"):
//!
//! | Issue | Mitigation |
//! |-------|------------|
//! | Entity type skew | Stratified sampling |
//! | Clean annotations | Add noise injection |
//! | Domain gap | Mix with real data |
//! | Label shift | Track via `LabelShift` |
//!
//! # What This Dataset IS Good For
//!
//! - **Unit testing**: Does the code work at all?
//! - **Pattern coverage**: Are regex patterns correct?
//! - **Edge cases**: Unicode, boundaries, special chars
//! - **Fast iteration**: Runs in <1s, no network
//!
//! # What This Dataset IS NOT Good For
//!
//! - **Zero-shot claims**: Label overlap with training ≈ 100%
//! - **Real-world performance**: Synthetic ≠ domain-specific noise
//! - **Model comparison**: Needs WikiGold/CoNLL/WNUT for fair eval
//!
//! # Usage
//!
//! ```rust
//! use anno::eval::synthetic::{all_datasets, datasets_by_domain, datasets_by_difficulty, Domain, Difficulty};
//!
//! // Get all datasets
//! let all = all_datasets();
//! assert!(!all.is_empty());
//!
//! // Filter by domain
//! let news = datasets_by_domain(Domain::News);
//!
//! // Filter by difficulty
//! let hard = datasets_by_difficulty(Difficulty::Hard);
//! ```

// Re-export types from dataset module (single source of truth)
pub use super::dataset::{AnnotatedExample, Difficulty, Domain};

// Re-export dataset functions from the modular structure
pub use super::dataset::synthetic::{
    // Domain-specific datasets
    academic_dataset,
    adversarial_dataset,
    biomedical_dataset,
    conversational_dataset,
    cybersecurity_dataset,
    ecommerce_dataset,
    entertainment_dataset,
    financial_dataset,
    food_dataset,
    globally_diverse_dataset,
    hard_domain_examples,
    historical_dataset,
    legal_dataset,
    multilingual_dataset,
    news_dataset,
    politics_dataset,
    real_estate_dataset,
    scientific_dataset,
    social_media_dataset,
    sports_dataset,
    structured_dataset,
    travel_dataset,
    weather_dataset,
    // Aggregate functions
    all_datasets,
};

use std::collections::HashMap;

// ============================================================================
// Backward Compatibility Aliases
// ============================================================================

/// Alias for `news_dataset()` - CoNLL-2003 style examples.
///
/// This is the traditional news/formal text dataset used in CoNLL shared tasks.
#[inline]
pub fn conll_style_dataset() -> Vec<AnnotatedExample> {
    news_dataset()
}

/// Filter datasets by domain.
///
/// # Example
///
/// ```rust
/// use anno::eval::synthetic::{datasets_by_domain, Domain};
///
/// let news = datasets_by_domain(Domain::News);
/// for ex in &news {
///     assert_eq!(ex.domain, Domain::News);
/// }
/// ```
pub fn datasets_by_domain(domain: Domain) -> Vec<AnnotatedExample> {
    super::dataset::synthetic::by_domain(domain)
}

/// Filter datasets by difficulty.
///
/// # Example
///
/// ```rust
/// use anno::eval::synthetic::{datasets_by_difficulty, Difficulty};
///
/// let hard = datasets_by_difficulty(Difficulty::Hard);
/// for ex in &hard {
///     assert_eq!(ex.difficulty, Difficulty::Hard);
/// }
/// ```
pub fn datasets_by_difficulty(difficulty: Difficulty) -> Vec<AnnotatedExample> {
    super::dataset::synthetic::by_difficulty(difficulty)
}

/// Statistics about the synthetic datasets.
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Total number of examples
    pub total_examples: usize,
    /// Total number of entities
    pub total_entities: usize,
    /// Examples per domain
    pub examples_per_domain: HashMap<String, usize>,
    /// Examples per difficulty
    pub examples_per_difficulty: HashMap<String, usize>,
}

/// Get statistics about all synthetic datasets.
pub fn dataset_stats() -> DatasetStats {
    let stats = super::dataset::synthetic::stats();
    DatasetStats {
        total_examples: stats.total_examples,
        total_entities: stats.total_entities,
        examples_per_domain: stats.domains,
        examples_per_difficulty: stats.difficulties,
    }
}

/// Extended quality dataset with diverse entity types and contexts.
///
/// This dataset focuses on quality over quantity, with carefully crafted
/// examples covering edge cases and challenging scenarios.
pub fn extended_quality_dataset() -> Vec<AnnotatedExample> {
    // Combine challenging examples from various sources
    let mut all = Vec::new();
    all.extend(hard_domain_examples());
    all.extend(globally_diverse_dataset());
    all.extend(adversarial_dataset());
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_datasets() {
        let all = all_datasets();
        assert!(!all.is_empty());
        assert!(all.len() >= 100, "Expected at least 100 examples");
    }

    #[test]
    fn test_conll_alias() {
        let conll = conll_style_dataset();
        let news = news_dataset();
        assert_eq!(conll.len(), news.len());
    }

    #[test]
    fn test_datasets_by_domain() {
        let news = datasets_by_domain(Domain::News);
        assert!(!news.is_empty());
        for ex in &news {
            assert_eq!(ex.domain, Domain::News);
        }
    }

    #[test]
    fn test_datasets_by_difficulty() {
        let hard = datasets_by_difficulty(Difficulty::Hard);
        for ex in &hard {
            assert_eq!(ex.difficulty, Difficulty::Hard);
        }
    }

    #[test]
    fn test_dataset_stats() {
        let stats = dataset_stats();
        assert!(stats.total_examples > 0);
        assert!(stats.total_entities > 0);
        assert!(!stats.examples_per_domain.is_empty());
    }

    #[test]
    fn test_extended_quality_dataset() {
        let extended = extended_quality_dataset();
        assert!(!extended.is_empty());
    }
}
