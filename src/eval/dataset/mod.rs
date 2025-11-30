//! Dataset API for NER evaluation.
//!
//! This module provides a unified interface for working with NER datasets,
//! whether loaded from files or synthetically generated.
//!
//! # Design Philosophy
//!
//! Following Rust idioms (burntsushi, shepmaster patterns):
//! - Use concrete types over trait objects where possible
//! - Implement standard traits (`IntoIterator`, `Index`, etc.)
//! - Keep the API surface small and predictable
//! - Prefer composition over inheritance
//!
//! # Quick Start
//!
//! ```rust
//! use anno::eval::dataset::{NERDataset, Domain, Difficulty};
//!
//! // Create from synthetic data
//! let dataset = NERDataset::synthetic();
//! println!("Total examples: {}", dataset.len());
//!
//! // Filter by domain
//! let news = dataset.filter_domain(Domain::News);
//! println!("News examples: {}", news.len());
//!
//! // Filter by difficulty
//! let hard = dataset.filter_difficulty(Difficulty::Hard);
//!
//! // Iterate
//! for example in &dataset {
//!     println!("{}: {} entities", example.text.len(), example.entity_count());
//! }
//! ```
//!
//! # Module Structure
//!
//! - `types`: Core types (`AnnotatedExample`, `Domain`, `Difficulty`)
//! - `synthetic`: Synthetic dataset generation by domain
//! - Main module: `NERDataset` struct and operations

pub mod synthetic;
pub mod types;

pub use types::{AnnotatedExample, Difficulty, Domain};

use crate::eval::GoldEntity;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Index;
use std::path::Path;

// ============================================================================
// NERDataset - Core Dataset Container
// ============================================================================

/// A collection of annotated NER examples with metadata and filtering.
///
/// `NERDataset` is the primary type for working with NER evaluation data.
/// It wraps a `Vec<AnnotatedExample>` with convenient methods for filtering,
/// statistics, and conversion to evaluation formats.
///
/// # Example
///
/// ```rust
/// use anno::eval::dataset::{NERDataset, Domain};
///
/// // Load synthetic data
/// let mut dataset = NERDataset::synthetic();
///
/// // Filter to specific domain
/// let biomedical = dataset.filter_domain(Domain::Biomedical);
///
/// // Get statistics
/// let stats = biomedical.stats();
/// println!("Examples: {}, Entities: {}", stats.total_examples, stats.total_entities);
///
/// // Convert to test cases for evaluation
/// let test_cases = biomedical.to_test_cases();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NERDataset {
    /// The underlying examples.
    examples: Vec<AnnotatedExample>,
    /// Dataset name/identifier.
    name: String,
    /// Optional source information (file path, URL, etc.).
    source: Option<String>,
}

impl NERDataset {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create an empty dataset with a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            examples: Vec::new(),
            name: name.into(),
            source: None,
        }
    }

    /// Create a dataset from a vector of examples.
    pub fn from_examples(name: impl Into<String>, examples: Vec<AnnotatedExample>) -> Self {
        Self {
            examples,
            name: name.into(),
            source: None,
        }
    }

    /// Create a dataset with source information.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Load the full synthetic dataset (all domains, all difficulties).
    ///
    /// This is the primary constructor for testing and development.
    /// The synthetic data covers many entity types and edge cases.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::eval::dataset::NERDataset;
    ///
    /// let dataset = NERDataset::synthetic();
    /// assert!(!dataset.is_empty());
    /// ```
    pub fn synthetic() -> Self {
        Self {
            examples: synthetic::all_datasets(),
            name: "synthetic".to_string(),
            source: Some("anno::eval::dataset::synthetic".to_string()),
        }
    }

    /// Load synthetic data for a specific domain.
    pub fn synthetic_domain(domain: Domain) -> Self {
        Self {
            examples: synthetic::by_domain(domain),
            name: format!("synthetic_{:?}", domain).to_lowercase(),
            source: Some("anno::eval::dataset::synthetic".to_string()),
        }
    }

    /// Load a dataset from a JSON file.
    ///
    /// Supports both JSON arrays and JSONL (one object per line).
    ///
    /// # Format
    ///
    /// ```json
    /// [
    ///   {
    ///     "text": "John works at Google.",
    ///     "entities": [
    ///       {"text": "John", "label": "PER", "start": 0, "end": 4},
    ///       {"text": "Google", "label": "ORG", "start": 14, "end": 20}
    ///     ]
    ///   }
    /// ]
    /// ```
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let test_cases = crate::eval::datasets::load_json_ner_dataset(path)?;
        let examples = test_cases
            .into_iter()
            .map(|(text, entities)| AnnotatedExample::new(text, entities))
            .collect();

        Ok(Self {
            examples,
            name,
            source: Some(path.display().to_string()),
        })
    }

    /// Load a dataset from CoNLL-2003 format.
    ///
    /// # Format
    ///
    /// Each line: `word POS chunk NER-tag`
    /// Empty lines separate sentences.
    pub fn from_conll<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let test_cases = crate::eval::load_conll2003(path)?;
        let examples = test_cases
            .into_iter()
            .map(|(text, entities)| AnnotatedExample::new(text, entities).with_domain(Domain::News))
            .collect();

        Ok(Self {
            examples,
            name,
            source: Some(path.display().to_string()),
        })
    }

    /// Auto-detect format and load from file.
    ///
    /// Tries CoNLL first, then JSON/JSONL.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let test_cases = crate::eval::datasets::load_ner_dataset(path)?;
        let examples = test_cases
            .into_iter()
            .map(|(text, entities)| AnnotatedExample::new(text, entities))
            .collect();

        Ok(Self {
            examples,
            name,
            source: Some(path.display().to_string()),
        })
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Returns the dataset name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the source (file path, URL, etc.) if known.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Returns the number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Returns true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get an example by index.
    pub fn get(&self, index: usize) -> Option<&AnnotatedExample> {
        self.examples.get(index)
    }

    /// Returns a slice of all examples.
    pub fn as_slice(&self) -> &[AnnotatedExample] {
        &self.examples
    }

    /// Returns a mutable slice of all examples.
    pub fn as_mut_slice(&mut self) -> &mut [AnnotatedExample] {
        &mut self.examples
    }

    /// Returns an iterator over the examples.
    pub fn iter(&self) -> impl Iterator<Item = &AnnotatedExample> {
        self.examples.iter()
    }

    /// Returns a mutable iterator over the examples.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut AnnotatedExample> {
        self.examples.iter_mut()
    }

    // ========================================================================
    // Filtering
    // ========================================================================

    /// Filter to a specific domain, returning a new dataset.
    pub fn filter_domain(&self, domain: Domain) -> Self {
        let examples = self
            .examples
            .iter()
            .filter(|ex| ex.domain == domain)
            .cloned()
            .collect();

        Self {
            examples,
            name: format!("{}_{:?}", self.name, domain).to_lowercase(),
            source: self.source.clone(),
        }
    }

    /// Filter to a specific difficulty, returning a new dataset.
    pub fn filter_difficulty(&self, difficulty: Difficulty) -> Self {
        let examples = self
            .examples
            .iter()
            .filter(|ex| ex.difficulty == difficulty)
            .cloned()
            .collect();

        Self {
            examples,
            name: format!("{}_{:?}", self.name, difficulty).to_lowercase(),
            source: self.source.clone(),
        }
    }

    /// Filter to examples containing a specific entity type.
    pub fn filter_entity_type(&self, entity_type: &crate::EntityType) -> Self {
        let examples = self
            .examples
            .iter()
            .filter(|ex| ex.entities.iter().any(|e| &e.entity_type == entity_type))
            .cloned()
            .collect();

        Self {
            examples,
            name: format!("{}_filtered", self.name),
            source: self.source.clone(),
        }
    }

    /// Filter using a custom predicate.
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&AnnotatedExample) -> bool,
    {
        let examples = self
            .examples
            .iter()
            .filter(|ex| predicate(ex))
            .cloned()
            .collect();

        Self {
            examples,
            name: format!("{}_filtered", self.name),
            source: self.source.clone(),
        }
    }

    /// Take the first n examples.
    pub fn take(&self, n: usize) -> Self {
        Self {
            examples: self.examples.iter().take(n).cloned().collect(),
            name: format!("{}_head_{}", self.name, n),
            source: self.source.clone(),
        }
    }

    /// Skip the first n examples.
    pub fn skip(&self, n: usize) -> Self {
        Self {
            examples: self.examples.iter().skip(n).cloned().collect(),
            name: format!("{}_tail", self.name),
            source: self.source.clone(),
        }
    }

    // ========================================================================
    // Mutation
    // ========================================================================

    /// Add an example to the dataset.
    pub fn push(&mut self, example: AnnotatedExample) {
        self.examples.push(example);
    }

    /// Extend with examples from another source.
    pub fn extend<I: IntoIterator<Item = AnnotatedExample>>(&mut self, iter: I) {
        self.examples.extend(iter);
    }

    /// Merge another dataset into this one.
    pub fn merge(&mut self, other: Self) {
        self.examples.extend(other.examples);
    }

    // ========================================================================
    // Conversion
    // ========================================================================

    /// Convert to test cases for evaluation functions.
    ///
    /// This is the bridge to `evaluate_ner_model()` and similar functions.
    pub fn to_test_cases(&self) -> Vec<(String, Vec<GoldEntity>)> {
        self.examples
            .iter()
            .map(|ex| (ex.text.clone(), ex.entities.clone()))
            .collect()
    }

    /// Consume and convert to test cases.
    pub fn into_test_cases(self) -> Vec<(String, Vec<GoldEntity>)> {
        self.examples
            .into_iter()
            .map(|ex| ex.into_test_case())
            .collect()
    }

    /// Convert to owned examples vec.
    pub fn into_inner(self) -> Vec<AnnotatedExample> {
        self.examples
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Compute dataset statistics.
    pub fn stats(&self) -> DatasetStats {
        let total_examples = self.examples.len();
        let total_entities: usize = self.examples.iter().map(|ex| ex.entities.len()).sum();

        let mut domains = HashMap::new();
        let mut difficulties = HashMap::new();
        let mut entity_types = HashMap::new();

        for ex in &self.examples {
            *domains.entry(ex.domain).or_insert(0) += 1;
            *difficulties.entry(ex.difficulty).or_insert(0) += 1;

            for entity in &ex.entities {
                let type_str = crate::eval::entity_type_to_string(&entity.entity_type);
                *entity_types.entry(type_str).or_insert(0) += 1;
            }
        }

        DatasetStats {
            total_examples,
            total_entities,
            avg_entities_per_example: if total_examples > 0 {
                total_entities as f64 / total_examples as f64
            } else {
                0.0
            },
            domains,
            difficulties,
            entity_types,
        }
    }
}

// ============================================================================
// Standard Trait Implementations
// ============================================================================

impl Default for NERDataset {
    fn default() -> Self {
        Self::new("default")
    }
}

impl Index<usize> for NERDataset {
    type Output = AnnotatedExample;

    fn index(&self, index: usize) -> &Self::Output {
        &self.examples[index]
    }
}

impl<'a> IntoIterator for &'a NERDataset {
    type Item = &'a AnnotatedExample;
    type IntoIter = std::slice::Iter<'a, AnnotatedExample>;

    fn into_iter(self) -> Self::IntoIter {
        self.examples.iter()
    }
}

impl IntoIterator for NERDataset {
    type Item = AnnotatedExample;
    type IntoIter = std::vec::IntoIter<AnnotatedExample>;

    fn into_iter(self) -> Self::IntoIter {
        self.examples.into_iter()
    }
}

impl FromIterator<AnnotatedExample> for NERDataset {
    fn from_iter<I: IntoIterator<Item = AnnotatedExample>>(iter: I) -> Self {
        Self {
            examples: iter.into_iter().collect(),
            name: "collected".to_string(),
            source: None,
        }
    }
}

impl Extend<AnnotatedExample> for NERDataset {
    fn extend<I: IntoIterator<Item = AnnotatedExample>>(&mut self, iter: I) {
        self.examples.extend(iter);
    }
}

// ============================================================================
// Dataset Statistics
// ============================================================================

/// Statistics about a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Total number of examples.
    pub total_examples: usize,
    /// Total number of entities across all examples.
    pub total_entities: usize,
    /// Average entities per example.
    pub avg_entities_per_example: f64,
    /// Count by domain.
    pub domains: HashMap<Domain, usize>,
    /// Count by difficulty.
    pub difficulties: HashMap<Difficulty, usize>,
    /// Count by entity type (as string labels).
    pub entity_types: HashMap<String, usize>,
}

impl DatasetStats {
    /// Format as a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Examples: {}\n", self.total_examples));
        s.push_str(&format!("Entities: {}\n", self.total_entities));
        s.push_str(&format!(
            "Avg entities/example: {:.1}\n",
            self.avg_entities_per_example
        ));

        s.push_str("\nDomains:\n");
        let mut domains: Vec<_> = self.domains.iter().collect();
        domains.sort_by(|a, b| b.1.cmp(a.1));
        for (domain, count) in domains.iter().take(5) {
            s.push_str(&format!("  {:?}: {}\n", domain, count));
        }
        if domains.len() > 5 {
            s.push_str(&format!("  ... and {} more\n", domains.len() - 5));
        }

        s.push_str("\nEntity Types:\n");
        let mut types: Vec<_> = self.entity_types.iter().collect();
        types.sort_by(|a, b| b.1.cmp(a.1));
        for (etype, count) in types.iter().take(10) {
            s.push_str(&format!("  {}: {}\n", etype, count));
        }
        if types.len() > 10 {
            s.push_str(&format!("  ... and {} more\n", types.len() - 10));
        }

        s
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_not_empty() {
        let dataset = NERDataset::synthetic();
        assert!(!dataset.is_empty());
        assert!(
            dataset.len() >= 50,
            "Should have substantial synthetic data"
        );
    }

    #[test]
    fn test_filter_domain() {
        let dataset = NERDataset::synthetic();
        let news = dataset.filter_domain(Domain::News);

        assert!(!news.is_empty());
        for ex in &news {
            assert_eq!(ex.domain, Domain::News);
        }
    }

    #[test]
    fn test_filter_difficulty() {
        let dataset = NERDataset::synthetic();
        let hard = dataset.filter_difficulty(Difficulty::Hard);

        assert!(!hard.is_empty());
        for ex in &hard {
            assert_eq!(ex.difficulty, Difficulty::Hard);
        }
    }

    #[test]
    fn test_to_test_cases() {
        let dataset = NERDataset::synthetic().take(5);
        let test_cases = dataset.to_test_cases();

        assert_eq!(test_cases.len(), 5);
        for (text, entities) in &test_cases {
            assert!(!text.is_empty());
            // Some might have no entities (negative examples)
            let _ = entities;
        }
    }

    #[test]
    fn test_stats() {
        let dataset = NERDataset::synthetic();
        let stats = dataset.stats();

        assert!(stats.total_examples > 0);
        assert!(stats.total_entities > 0);
        assert!(!stats.domains.is_empty());
        assert!(!stats.entity_types.is_empty());
    }

    #[test]
    fn test_indexing() {
        let dataset = NERDataset::synthetic();
        let first = &dataset[0];
        assert!(!first.text.is_empty());
    }

    #[test]
    fn test_into_iterator() {
        let dataset = NERDataset::synthetic().take(3);
        let mut count = 0;
        for _ex in &dataset {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_from_iterator() {
        let examples = vec![
            AnnotatedExample::from_tuples("John works", vec![("John", "PER")]),
            AnnotatedExample::from_tuples("At Google", vec![("Google", "ORG")]),
        ];

        let dataset: NERDataset = examples.into_iter().collect();
        assert_eq!(dataset.len(), 2);
    }
}
