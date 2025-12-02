//! Demonstration selection for few-shot NER.
//!
//! Implements CMAS-inspired demonstration selection (arXiv:2502.18702) that
//! selects helpful demonstrations based on similarity and quality.
//!
//! # Key Concepts
//!
//! From the CMAS paper:
//! 1. **Self-annotation**: Initial entity labeling
//! 2. **TRF (Type-Related Features)**: Context features around entities
//! 3. **Demonstration discriminator**: Evaluates helpfulness
//! 4. **Overall predictor**: Final ensemble
//!
//! This module provides the demonstration discriminator component.
//!
//! # Example
//!
//! ```rust
//! use anno::backends::demonstration::{DemonstrationBank, HelpfulnessConfig};
//!
//! let mut bank = DemonstrationBank::new();
//!
//! // Add demonstrations
//! bank.add("Steve Jobs founded Apple in 1976.", vec![
//!     ("Steve Jobs", "PER", 0, 10),
//!     ("Apple", "ORG", 19, 24),
//! ]);
//!
//! bank.add("Microsoft was founded by Bill Gates.", vec![
//!     ("Microsoft", "ORG", 0, 9),
//!     ("Bill Gates", "PER", 26, 36),
//! ]);
//!
//! // Select helpful demonstrations for a query
//! let demos = bank.select("Marie Curie worked at the Sorbonne.", 2);
//! assert_eq!(demos.len(), 2);
//! ```

use std::collections::HashMap;

/// Entity annotation: (text, entity_type, start, end).
pub type EntityAnnotation<'a> = (&'a str, &'a str, usize, usize);

/// Batch of demonstrations: (text, entities) pairs.
pub type DemoBatch<'a> = Vec<(&'a str, Vec<EntityAnnotation<'a>>)>;

/// Configuration for helpfulness scoring.
#[derive(Debug, Clone)]
pub struct HelpfulnessConfig {
    /// Weight for text similarity
    pub similarity_weight: f64,
    /// Weight for entity type overlap
    pub type_overlap_weight: f64,
    /// Weight for entity density similarity
    pub density_weight: f64,
    /// Minimum helpfulness score to include
    pub min_score: f64,
}

impl Default for HelpfulnessConfig {
    fn default() -> Self {
        Self {
            similarity_weight: 0.4,
            type_overlap_weight: 0.4,
            density_weight: 0.2,
            min_score: 0.1,
        }
    }
}

/// A single demonstration example.
#[derive(Debug, Clone)]
pub struct DemonstrationExample {
    /// Input text
    pub text: String,
    /// Annotated entities: (text, type, start, end)
    pub entities: Vec<(String, String, usize, usize)>,
    /// Precomputed features
    features: ExampleFeatures,
}

/// Precomputed features for efficient matching.
#[derive(Debug, Clone, Default)]
struct ExampleFeatures {
    /// Token set (lowercase words)
    tokens: Vec<String>,
    /// Entity types present
    entity_types: Vec<String>,
    /// Entity density (entities per 100 tokens)
    entity_density: f64,
}

impl DemonstrationExample {
    /// Create a new demonstration example.
    pub fn new(text: &str, entities: Vec<(&str, &str, usize, usize)>) -> Self {
        let entities: Vec<_> = entities
            .into_iter()
            .map(|(t, ty, s, e)| (t.to_string(), ty.to_string(), s, e))
            .collect();

        let features = Self::compute_features(text, &entities);

        Self {
            text: text.to_string(),
            entities,
            features,
        }
    }

    fn compute_features(
        text: &str,
        entities: &[(String, String, usize, usize)],
    ) -> ExampleFeatures {
        let tokens: Vec<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();

        let entity_types: Vec<String> = entities.iter().map(|(_, ty, _, _)| ty.clone()).collect();

        let entity_density = if tokens.is_empty() {
            0.0
        } else {
            (entities.len() as f64 / tokens.len() as f64) * 100.0
        };

        ExampleFeatures {
            tokens,
            entity_types,
            entity_density,
        }
    }
}

/// Bank of demonstrations for few-shot NER.
///
/// Stores demonstrations and selects the most helpful ones for a query.
#[derive(Debug, Clone, Default)]
pub struct DemonstrationBank {
    examples: Vec<DemonstrationExample>,
    config: HelpfulnessConfig,
}

impl DemonstrationBank {
    /// Create a new empty demonstration bank.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom helpfulness config.
    #[must_use]
    pub fn with_config(config: HelpfulnessConfig) -> Self {
        Self {
            examples: vec![],
            config,
        }
    }

    /// Add a demonstration to the bank.
    pub fn add(&mut self, text: &str, entities: Vec<(&str, &str, usize, usize)>) {
        self.examples
            .push(DemonstrationExample::new(text, entities));
    }

    /// Add multiple demonstrations at once.
    pub fn add_all(&mut self, demos: DemoBatch<'_>) {
        for (text, entities) in demos {
            self.add(text, entities);
        }
    }

    /// Number of demonstrations in the bank.
    #[must_use]
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if the bank is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Select the most helpful demonstrations for a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The input text to find demonstrations for
    /// * `k` - Maximum number of demonstrations to return
    ///
    /// # Returns
    ///
    /// Up to `k` demonstrations, sorted by helpfulness score (descending).
    #[must_use]
    pub fn select(&self, query: &str, k: usize) -> Vec<&DemonstrationExample> {
        if self.examples.is_empty() || k == 0 {
            return vec![];
        }

        let query_features = DemonstrationExample::compute_features(query, &[]);

        // Performance: Pre-allocate scored vec with estimated capacity
        // Score all demonstrations
        let mut scored: Vec<_> = Vec::with_capacity(self.examples.len().min(k * 2));
        scored.extend(
            self.examples
                .iter()
                .map(|ex| {
                    let score = self.helpfulness_score(&query_features, ex);
                    (ex, score)
                })
                .filter(|(_, score)| *score >= self.config.min_score),
        );

        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort by score descending
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        scored.into_iter().take(k).map(|(ex, _)| ex).collect()
    }

    /// Select demonstrations with their helpfulness scores.
    #[must_use]
    pub fn select_with_scores(&self, query: &str, k: usize) -> Vec<(&DemonstrationExample, f64)> {
        if self.examples.is_empty() || k == 0 {
            return vec![];
        }

        let query_features = DemonstrationExample::compute_features(query, &[]);

        // Performance: Pre-allocate scored vec with estimated capacity
        let mut scored: Vec<_> = Vec::with_capacity(self.examples.len().min(k * 2));
        scored.extend(
            self.examples
                .iter()
                .map(|ex| {
                    let score = self.helpfulness_score(&query_features, ex);
                    (ex, score)
                })
                .filter(|(_, score)| *score >= self.config.min_score),
        );

        // Performance: Use unstable sort (we don't need stable sort here)
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(k).collect()
    }

    /// Compute helpfulness score for a demonstration.
    ///
    /// Based on CMAS demonstration discriminator, combines:
    /// 1. Text similarity (token overlap)
    /// 2. Entity type overlap
    /// 3. Entity density similarity
    fn helpfulness_score(&self, query: &ExampleFeatures, demo: &DemonstrationExample) -> f64 {
        let sim = self.token_similarity(&query.tokens, &demo.features.tokens);
        let type_overlap = self.type_overlap(&query.entity_types, &demo.features.entity_types);
        let density_sim =
            self.density_similarity(query.entity_density, demo.features.entity_density);

        self.config.similarity_weight * sim
            + self.config.type_overlap_weight * type_overlap
            + self.config.density_weight * density_sim
    }

    /// Jaccard similarity between token sets.
    fn token_similarity(&self, a: &[String], b: &[String]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let set_a: std::collections::HashSet<_> = a.iter().collect();
        let set_b: std::collections::HashSet<_> = b.iter().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Overlap ratio for entity types.
    fn type_overlap(&self, query_types: &[String], demo_types: &[String]) -> f64 {
        // For queries without known types, all demonstrations are equally good
        if query_types.is_empty() {
            return 1.0;
        }
        if demo_types.is_empty() {
            return 0.0;
        }

        let query_set: std::collections::HashSet<_> = query_types.iter().collect();
        let demo_set: std::collections::HashSet<_> = demo_types.iter().collect();

        let overlap = query_set.intersection(&demo_set).count();
        overlap as f64 / query_set.len() as f64
    }

    /// Similarity based on entity density.
    fn density_similarity(&self, query_density: f64, demo_density: f64) -> f64 {
        // Exponential decay based on density difference
        let diff = (query_density - demo_density).abs();
        (-diff / 5.0).exp() // Scale factor of 5 entities per 100 tokens
    }
}

/// Type-Related Feature (TRF) extractor.
///
/// Extracts context features around entity mentions, as described in CMAS.
#[derive(Debug, Clone, Default)]
pub struct TRFExtractor {
    window_size: usize,
}

impl TRFExtractor {
    /// Create a new TRF extractor with default window size.
    #[must_use]
    pub fn new() -> Self {
        Self { window_size: 3 }
    }

    /// Create with custom window size.
    #[must_use]
    pub fn with_window(size: usize) -> Self {
        Self { window_size: size }
    }

    /// Extract type-related features from text.
    ///
    /// Returns context words around potential entity spans.
    #[must_use]
    pub fn extract(
        &self,
        text: &str,
        entities: &[(String, String, usize, usize)],
    ) -> HashMap<String, Vec<String>> {
        let mut features: HashMap<String, Vec<String>> = HashMap::new();
        let tokens: Vec<&str> = text.split_whitespace().collect();

        for (entity_text, entity_type, start, _end) in entities {
            // Find token index for entity start
            let mut char_pos = 0;
            let mut token_idx = None;

            for (i, token) in tokens.iter().enumerate() {
                if char_pos == *start || (char_pos <= *start && char_pos + token.len() > *start) {
                    token_idx = Some(i);
                    break;
                }
                char_pos += token.len() + 1; // +1 for space
            }

            if let Some(idx) = token_idx {
                // Extract window around entity
                let start_idx = idx.saturating_sub(self.window_size);
                let end_idx = (idx + self.window_size + 1).min(tokens.len());

                let context: Vec<String> = tokens[start_idx..end_idx]
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i + start_idx != idx) // Exclude entity itself
                    .map(|(_, &t)| t.to_lowercase())
                    .collect();

                features
                    .entry(entity_type.clone())
                    .or_default()
                    .extend(context);
            }

            // Also add the entity text as a feature (useful for learning patterns)
            features
                .entry(format!("{}_text", entity_type))
                .or_default()
                .push(entity_text.to_lowercase());
        }

        features
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demonstration_example_creation() {
        let demo = DemonstrationExample::new(
            "Steve Jobs founded Apple.",
            vec![("Steve Jobs", "PER", 0, 10), ("Apple", "ORG", 19, 24)],
        );

        assert_eq!(demo.entities.len(), 2);
        assert!(demo.features.entity_types.contains(&"PER".to_string()));
        assert!(demo.features.entity_types.contains(&"ORG".to_string()));
    }

    #[test]
    fn test_bank_add_and_len() {
        let mut bank = DemonstrationBank::new();
        assert!(bank.is_empty());

        bank.add("Test text.", vec![("Test", "MISC", 0, 4)]);
        assert_eq!(bank.len(), 1);
    }

    #[test]
    fn test_select_demonstrations() {
        let mut bank = DemonstrationBank::new();

        bank.add(
            "Steve Jobs founded Apple in California.",
            vec![
                ("Steve Jobs", "PER", 0, 10),
                ("Apple", "ORG", 19, 24),
                ("California", "LOC", 28, 38),
            ],
        );

        bank.add(
            "The weather in New York is nice today.",
            vec![("New York", "LOC", 15, 23)],
        );

        bank.add(
            "Bill Gates started Microsoft in Seattle.",
            vec![
                ("Bill Gates", "PER", 0, 10),
                ("Microsoft", "ORG", 19, 28),
                ("Seattle", "LOC", 32, 39),
            ],
        );

        // Query about companies (same domain as demos)
        let demos = bank.select("Steve Jobs founded Apple in Silicon Valley.", 3);

        // Should return all 3 demos
        assert_eq!(demos.len(), 3);

        // All demos should be returned - verify we have all three
        let demo_texts: Vec<_> = demos.iter().map(|d| d.text.as_str()).collect();
        assert!(demo_texts.contains(&"Steve Jobs founded Apple in California."));
        assert!(demo_texts.contains(&"Bill Gates started Microsoft in Seattle."));
        assert!(demo_texts.contains(&"The weather in New York is nice today."));
    }

    #[test]
    fn test_select_with_scores() {
        let mut bank = DemonstrationBank::new();

        bank.add("Apple is in Cupertino.", vec![("Apple", "ORG", 0, 5)]);
        bank.add("Google is in Mountain View.", vec![("Google", "ORG", 0, 6)]);

        let demos = bank.select_with_scores("Microsoft is in Redmond.", 2);

        assert_eq!(demos.len(), 2);
        // Both should have positive scores
        for (_, score) in &demos {
            assert!(*score > 0.0);
        }
    }

    #[test]
    fn test_select_empty_bank() {
        let bank = DemonstrationBank::new();
        let demos = bank.select("Test query.", 5);
        assert!(demos.is_empty());
    }

    #[test]
    fn test_trf_extractor() {
        let extractor = TRFExtractor::new();

        let features = extractor.extract(
            "The CEO Steve Jobs announced the new iPhone.",
            &[("Steve Jobs".to_string(), "PER".to_string(), 8, 18)],
        );

        assert!(features.contains_key("PER"));
        let per_context = features.get("PER").unwrap();
        // Should contain context words around "Steve Jobs"
        assert!(per_context.iter().any(|w| w == "ceo" || w == "announced"));
    }

    #[test]
    fn test_helpfulness_config() {
        let config = HelpfulnessConfig {
            similarity_weight: 0.5,
            type_overlap_weight: 0.3,
            density_weight: 0.2,
            min_score: 0.2,
        };

        let bank = DemonstrationBank::with_config(config);
        assert!(!bank.config.min_score.is_nan());
    }
}
