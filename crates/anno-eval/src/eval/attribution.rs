//! Feature attribution for NER predictions.
//!
//! Explains which input features contributed to entity predictions,
//! enabling interpretability and debugging.
//!
//! # Methods
//!
//! | Method | Type | Cost | Notes |
//! |--------|------|------|-------|
//! | **Attention** | Model-based | Low | Uses model's attention weights |
//! | **Ablation** | Perturbation | Medium | Mask tokens, observe change |
//! | **LIME** | Local surrogate | High | Fit interpretable model locally |
//! | **Gradient** | Gradient-based | Low | Input gradients (requires diff model) |
//!
//! # Example
//!
//! ```rust
//! use anno::eval::attribution::{AttributionAnalyzer, AblationConfig};
//!
//! let analyzer = AttributionAnalyzer::new(AblationConfig::default());
//!
//! // Analyze which tokens contribute to "Einstein" being Person
//! let text = "Albert Einstein was a physicist.";
//! let attributions = analyzer.ablation_analysis(text, 0, 15, "PERSON");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Token Attribution
// =============================================================================

/// Attribution score for a single token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAttribution {
    /// Token text
    pub token: String,
    /// Token position (0-indexed)
    pub position: usize,
    /// Attribution score (higher = more important)
    pub score: f64,
    /// Direction: positive = supports prediction, negative = against
    pub direction: f64,
    /// Normalized score (0-1)
    pub normalized_score: f64,
}

impl TokenAttribution {
    /// Create a new token attribution.
    pub fn new(token: &str, position: usize, score: f64) -> Self {
        Self {
            token: token.to_string(),
            position,
            score,
            direction: score.signum(),
            normalized_score: 0.0,
        }
    }
}

/// Full attribution result for an entity prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAttribution {
    /// Entity text
    pub entity_text: String,
    /// Entity type
    pub entity_type: String,
    /// Original confidence
    pub original_confidence: f64,
    /// Per-token attributions
    pub token_attributions: Vec<TokenAttribution>,
    /// Top positive contributors
    pub top_positive: Vec<TokenAttribution>,
    /// Top negative contributors (if any)
    pub top_negative: Vec<TokenAttribution>,
    /// Attribution method used
    pub method: AttributionMethod,
}

impl EntityAttribution {
    /// Get the most important token.
    #[must_use]
    pub fn most_important(&self) -> Option<&TokenAttribution> {
        self.token_attributions.iter().max_by(|a, b| {
            a.score
                .abs()
                .partial_cmp(&b.score.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get tokens sorted by importance.
    #[must_use]
    pub fn sorted_by_importance(&self) -> Vec<&TokenAttribution> {
        let mut sorted: Vec<_> = self.token_attributions.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .abs()
                .partial_cmp(&a.score.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Generate a text explanation.
    #[must_use]
    pub fn explain(&self) -> String {
        let mut parts = Vec::new();

        parts.push(format!(
            "Entity \"{}\" classified as {} (confidence: {:.1}%)",
            self.entity_text,
            self.entity_type,
            self.original_confidence * 100.0
        ));

        if !self.top_positive.is_empty() {
            let tokens: Vec<_> = self
                .top_positive
                .iter()
                .take(3)
                .map(|t| format!("\"{}\"", t.token))
                .collect();
            parts.push(format!("Key supporting tokens: {}", tokens.join(", ")));
        }

        if !self.top_negative.is_empty() {
            let tokens: Vec<_> = self
                .top_negative
                .iter()
                .take(3)
                .map(|t| format!("\"{}\"", t.token))
                .collect();
            parts.push(format!("Tokens reducing confidence: {}", tokens.join(", ")));
        }

        parts.join("\n")
    }
}

/// Attribution method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionMethod {
    /// Token ablation (masking)
    Ablation,
    /// Attention weights
    Attention,
    /// Gradient-based
    Gradient,
    /// LIME local surrogate
    LIME,
    /// Random baseline
    Random,
}

// =============================================================================
// Ablation Configuration
// =============================================================================

/// Configuration for ablation-based attribution.
#[derive(Debug, Clone)]
pub struct AblationConfig {
    /// Token to use for masking
    pub mask_token: String,
    /// Context window around entity
    pub context_window: usize,
    /// Minimum confidence drop to consider significant
    pub significance_threshold: f64,
    /// Whether to normalize scores
    pub normalize: bool,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            mask_token: "[MASK]".to_string(),
            context_window: 20,
            significance_threshold: 0.01,
            normalize: true,
        }
    }
}

// =============================================================================
// Attribution Analyzer
// =============================================================================

/// Analyzer for computing feature attributions.
#[derive(Debug, Clone)]
pub struct AttributionAnalyzer {
    /// Ablation config
    ablation_config: AblationConfig,
}

impl AttributionAnalyzer {
    /// Create a new analyzer.
    pub fn new(config: AblationConfig) -> Self {
        Self {
            ablation_config: config,
        }
    }

    /// Perform ablation analysis (simulated without actual model).
    ///
    /// In practice, this would call the model multiple times with masked tokens.
    /// Here we provide a heuristic-based simulation for demonstration.
    pub fn ablation_analysis(
        &self,
        text: &str,
        entity_start: usize,
        entity_end: usize,
        entity_type: &str,
    ) -> EntityAttribution {
        let chars: Vec<char> = text.chars().collect();
        let entity_text: String = chars[entity_start..entity_end].iter().collect();

        // Simple tokenization (word-level)
        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Heuristic attribution based on position and content
        let mut attributions: Vec<TokenAttribution> = Vec::new();

        for (pos, token) in tokens.iter().enumerate() {
            let score = self.heuristic_score(token, entity_type, pos, tokens.len(), &entity_text);
            attributions.push(TokenAttribution::new(token, pos, score));
        }

        // Normalize if configured
        if self.ablation_config.normalize {
            let max_abs = attributions
                .iter()
                .map(|a| a.score.abs())
                .fold(0.0f64, f64::max);
            if max_abs > 0.0 {
                for attr in &mut attributions {
                    attr.normalized_score = attr.score.abs() / max_abs;
                }
            }
        }

        // Compute top positive/negative
        let mut sorted = attributions.clone();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_positive: Vec<_> = sorted
            .iter()
            .filter(|a| a.score > 0.0)
            .take(5)
            .cloned()
            .collect();
        let top_negative: Vec<_> = sorted
            .iter()
            .rev()
            .filter(|a| a.score < 0.0)
            .take(5)
            .cloned()
            .collect();

        EntityAttribution {
            entity_text,
            entity_type: entity_type.to_string(),
            original_confidence: 0.85, // Placeholder
            token_attributions: attributions,
            top_positive,
            top_negative,
            method: AttributionMethod::Ablation,
        }
    }

    /// Heuristic scoring for demonstration.
    fn heuristic_score(
        &self,
        token: &str,
        entity_type: &str,
        position: usize,
        total_tokens: usize,
        entity_text: &str,
    ) -> f64 {
        let mut score = 0.0;
        let token_lower = token.to_lowercase();
        let entity_type_lower = entity_type.to_lowercase();

        // Tokens in entity itself are highly important
        if entity_text.to_lowercase().contains(&token_lower) {
            score += 0.8;
        }

        // Title tokens for PERSON
        if entity_type_lower.contains("person") || entity_type_lower == "per" {
            if [
                "dr", "dr.", "mr", "mr.", "ms", "ms.", "mrs", "mrs.", "prof", "prof.",
            ]
            .contains(&token_lower.as_str())
            {
                score += 0.5;
            }
            if token
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                score += 0.2;
            }
        }

        // Organization indicators
        if (entity_type_lower.contains("org") || entity_type_lower == "organization")
            && [
                "inc",
                "inc.",
                "corp",
                "corp.",
                "llc",
                "ltd",
                "company",
                "corporation",
                "group",
            ]
            .contains(&token_lower.as_str())
        {
            score += 0.6;
        }

        // Location indicators
        if (entity_type_lower.contains("loc") || entity_type_lower.contains("gpe"))
            && ["city", "country", "state", "capital", "in", "of", "near"]
                .contains(&token_lower.as_str())
        {
            score += 0.3;
        }

        // Position effect (closer to entity = more important)
        let entity_position = total_tokens / 2; // Approximate
        let distance = (position as i32 - entity_position as i32).unsigned_abs() as f64;
        let position_factor = 1.0 / (1.0 + distance * 0.1);
        score *= position_factor;

        // Function words have less attribution
        if [
            "the", "a", "an", "is", "was", "were", "are", "and", "or", "but",
        ]
        .contains(&token_lower.as_str())
        {
            score *= 0.3;
        }

        score
    }

    /// Compute aggregated feature importance across multiple entities.
    pub fn aggregate_importance(&self, attributions: &[EntityAttribution]) -> HashMap<String, f64> {
        let mut importance: HashMap<String, (f64, usize)> = HashMap::new();

        for attr in attributions {
            for token_attr in &attr.token_attributions {
                let entry = importance
                    .entry(token_attr.token.to_lowercase())
                    .or_insert((0.0, 0));
                entry.0 += token_attr.score.abs();
                entry.1 += 1;
            }
        }

        importance
            .into_iter()
            .map(|(k, (sum, count))| (k, sum / count as f64))
            .collect()
    }
}

impl Default for AttributionAnalyzer {
    fn default() -> Self {
        Self::new(AblationConfig::default())
    }
}

// =============================================================================
// Counterfactual Analysis
// =============================================================================

/// A counterfactual example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterfactual {
    /// Original text
    pub original: String,
    /// Modified text
    pub modified: String,
    /// What was changed
    pub change_description: String,
    /// Original prediction
    pub original_prediction: String,
    /// New prediction (or confidence change)
    pub new_prediction: String,
    /// Confidence delta
    pub confidence_delta: f64,
}

/// Generate counterfactual explanations.
#[derive(Debug, Clone, Default)]
pub struct CounterfactualGenerator {
    /// Maximum edits to try
    pub max_edits: usize,
}

impl CounterfactualGenerator {
    /// Create a new generator.
    pub fn new() -> Self {
        Self { max_edits: 5 }
    }

    /// Generate potential counterfactuals for an entity.
    ///
    /// Returns modifications that might change the prediction.
    pub fn generate(
        &self,
        text: &str,
        entity_start: usize,
        entity_end: usize,
        entity_type: &str,
    ) -> Vec<Counterfactual> {
        let chars: Vec<char> = text.chars().collect();
        let entity_text: String = chars[entity_start..entity_end].iter().collect();

        let mut counterfactuals = Vec::new();

        // Counterfactual 1: Remove title
        let tokens: Vec<&str> = text.split_whitespace().collect();
        for (i, token) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            if ["dr.", "mr.", "ms.", "prof.", "ceo"].contains(&lower.as_str()) {
                let modified: String = tokens
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, t)| *t)
                    .collect::<Vec<_>>()
                    .join(" ");

                counterfactuals.push(Counterfactual {
                    original: text.to_string(),
                    modified,
                    change_description: format!("Removed title '{}'", token),
                    original_prediction: entity_type.to_string(),
                    new_prediction: format!("{} (reduced confidence)", entity_type),
                    confidence_delta: -0.15,
                });
            }
        }

        // Counterfactual 2: Lowercase entity
        let lowercase_entity = entity_text.to_lowercase();
        if lowercase_entity != entity_text {
            // Important: lowercasing a single Unicode scalar can expand to multiple chars.
            // So we construct the modified string by splicing the char-range and inserting the
            // fully-lowercased substring.
            let prefix: String = text.chars().take(entity_start).collect();
            let suffix: String = text.chars().skip(entity_end).collect();
            let modified = format!("{prefix}{lowercase_entity}{suffix}");

            counterfactuals.push(Counterfactual {
                original: text.to_string(),
                modified,
                change_description: "Lowercased entity".to_string(),
                original_prediction: entity_type.to_string(),
                new_prediction: format!("{} (reduced confidence)", entity_type),
                confidence_delta: -0.25,
            });
        }

        // Counterfactual 3: Add context
        if entity_type.to_lowercase().contains("person") {
            let modified = format!("{} said that", text);
            counterfactuals.push(Counterfactual {
                original: text.to_string(),
                modified,
                change_description: "Added reporting context".to_string(),
                original_prediction: entity_type.to_string(),
                new_prediction: format!("{} (increased confidence)", entity_type),
                confidence_delta: 0.05,
            });
        }

        counterfactuals.truncate(self.max_edits);
        counterfactuals
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ablation_analysis() {
        let analyzer = AttributionAnalyzer::default();

        let text = "Dr. Albert Einstein was a physicist.";
        let attribution = analyzer.ablation_analysis(text, 4, 19, "PERSON");

        assert_eq!(attribution.entity_type, "PERSON");
        assert!(!attribution.token_attributions.is_empty());

        // Dr. should have high attribution for PERSON
        let dr_attr = attribution
            .token_attributions
            .iter()
            .find(|a| a.token == "Dr.");
        assert!(dr_attr.is_some());
        assert!(dr_attr.unwrap().score > 0.0);
    }

    #[test]
    fn test_token_attribution() {
        let attr = TokenAttribution::new("Einstein", 0, 0.8);
        assert_eq!(attr.token, "Einstein");
        assert!(attr.direction > 0.0);
    }

    #[test]
    fn test_entity_attribution_explain() {
        let attribution = EntityAttribution {
            entity_text: "Einstein".to_string(),
            entity_type: "PERSON".to_string(),
            original_confidence: 0.95,
            token_attributions: vec![
                TokenAttribution::new("Dr.", 0, 0.5),
                TokenAttribution::new("Einstein", 1, 0.9),
            ],
            top_positive: vec![TokenAttribution::new("Dr.", 0, 0.5)],
            top_negative: vec![],
            method: AttributionMethod::Ablation,
        };

        let explanation = attribution.explain();
        assert!(explanation.contains("PERSON"));
        assert!(explanation.contains("95.0%"));
    }

    #[test]
    fn test_counterfactual_generator() {
        let generator = CounterfactualGenerator::new();

        let text = "Dr. Smith is the CEO.";
        let counterfactuals = generator.generate(text, 4, 9, "PERSON");

        // Should generate counterfactual for removing "Dr."
        assert!(!counterfactuals.is_empty());
    }

    #[test]
    fn test_aggregate_importance() {
        let analyzer = AttributionAnalyzer::default();

        let attr1 = EntityAttribution {
            entity_text: "Einstein".to_string(),
            entity_type: "PERSON".to_string(),
            original_confidence: 0.9,
            token_attributions: vec![
                TokenAttribution::new("Dr.", 0, 0.5),
                TokenAttribution::new("Einstein", 1, 0.8),
            ],
            top_positive: vec![],
            top_negative: vec![],
            method: AttributionMethod::Ablation,
        };

        let importance = analyzer.aggregate_importance(&[attr1]);
        assert!(importance.contains_key("dr."));
        assert!(importance.contains_key("einstein"));
    }

    #[test]
    fn test_most_important() {
        let attribution = EntityAttribution {
            entity_text: "Test".to_string(),
            entity_type: "PERSON".to_string(),
            original_confidence: 0.9,
            token_attributions: vec![
                TokenAttribution::new("a", 0, 0.1),
                TokenAttribution::new("b", 1, 0.9),
                TokenAttribution::new("c", 2, 0.3),
            ],
            top_positive: vec![],
            top_negative: vec![],
            method: AttributionMethod::Ablation,
        };

        let most = attribution.most_important();
        assert!(most.is_some());
        assert_eq!(most.unwrap().token, "b");
    }
}
