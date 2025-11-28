//! Out-of-Distribution (OOD) detection for NER systems.
//!
//! Detects when models encounter entity patterns not seen during training,
//! enabling graceful degradation rather than confident incorrect predictions.
//!
//! # Research Background
//!
//! - Models often produce confident predictions on OOD inputs
//! - "Confident uncertainty" is dangerous in production
//! - OOD detection enables fallback strategies (human review, abstention)
//!
//! # Key Concepts
//!
//! - **Vocabulary OOD**: Entity surface forms not in training vocabulary
//! - **Distribution OOD**: Entity patterns statistically different from training
//! - **Confidence-based OOD**: Low model confidence as OOD signal
//!
//! # Example
//!
//! ```rust
//! use anno::eval::ood_detection::{OODDetector, OODConfig};
//!
//! let detector = OODDetector::new(OODConfig::default());
//!
//! // Build vocabulary from training entities
//! let training_entities = vec!["John Smith", "Google", "New York"];
//! let detector = detector.fit(&training_entities);
//!
//! // Check if test entities are OOD
//! let is_ood = detector.is_ood("Xiangjun Chen");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for OOD detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OODConfig {
    /// Confidence threshold below which predictions are flagged as OOD
    pub confidence_threshold: f64,
    /// Minimum character n-gram frequency to be considered in-distribution
    pub min_ngram_frequency: usize,
    /// N-gram size for vocabulary coverage
    pub ngram_size: usize,
    /// Whether to use subword tokenization for vocabulary
    pub use_subwords: bool,
    /// Threshold for vocabulary coverage (0.0-1.0)
    pub vocab_coverage_threshold: f64,
}

impl Default for OODConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_ngram_frequency: 1,
            ngram_size: 3,
            use_subwords: true,
            vocab_coverage_threshold: 0.5,
        }
    }
}

// =============================================================================
// OOD Detection Results
// =============================================================================

/// Results of OOD analysis on a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OODAnalysisResults {
    /// Total entities analyzed
    pub total_entities: usize,
    /// Number flagged as OOD
    pub ood_count: usize,
    /// OOD rate (ood_count / total_entities)
    pub ood_rate: f64,
    /// OOD breakdown by detection method
    pub by_method: HashMap<String, usize>,
    /// Average confidence of OOD entities
    pub avg_ood_confidence: f64,
    /// Average confidence of in-distribution entities
    pub avg_id_confidence: f64,
    /// Vocabulary coverage statistics
    pub vocab_stats: VocabCoverageStats,
    /// Sample OOD entities for inspection
    pub sample_ood_entities: Vec<String>,
}

/// Vocabulary coverage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabCoverageStats {
    /// Training vocabulary size (unique n-grams)
    pub train_vocab_size: usize,
    /// Test vocabulary size (unique n-grams)
    pub test_vocab_size: usize,
    /// N-grams in test but not in train
    pub unseen_ngrams: usize,
    /// Coverage ratio (seen / total test ngrams)
    pub coverage_ratio: f64,
}

/// OOD status for a single entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OODStatus {
    /// Entity text
    pub text: String,
    /// Whether entity is OOD
    pub is_ood: bool,
    /// OOD detection methods that flagged this entity
    pub flagged_by: Vec<String>,
    /// Vocabulary coverage score (0.0 = all unseen, 1.0 = all seen)
    pub vocab_coverage: f64,
    /// Model confidence (if available)
    pub confidence: Option<f64>,
}

// =============================================================================
// OOD Detector
// =============================================================================

/// Detector for out-of-distribution entities.
#[derive(Debug, Clone)]
pub struct OODDetector {
    /// Configuration
    config: OODConfig,
    /// Training vocabulary (character n-grams)
    train_ngrams: HashSet<String>,
    /// N-gram frequencies in training data
    ngram_frequencies: HashMap<String, usize>,
    /// Known entity surface forms
    known_entities: HashSet<String>,
    /// Entity type distributions from training
    type_distributions: HashMap<String, usize>,
}

impl OODDetector {
    /// Create a new OOD detector with given configuration.
    pub fn new(config: OODConfig) -> Self {
        Self {
            config,
            train_ngrams: HashSet::new(),
            ngram_frequencies: HashMap::new(),
            known_entities: HashSet::new(),
            type_distributions: HashMap::new(),
        }
    }

    /// Fit the detector on training entity texts.
    pub fn fit(mut self, training_entities: &[impl AsRef<str>]) -> Self {
        for entity in training_entities {
            let text = entity.as_ref();
            self.known_entities.insert(text.to_lowercase());

            // Extract and count n-grams
            for ngram in self.extract_ngrams(text) {
                self.train_ngrams.insert(ngram.clone());
                *self.ngram_frequencies.entry(ngram).or_insert(0) += 1;
            }
        }
        self
    }

    /// Fit with entity types for distribution tracking.
    pub fn fit_with_types(
        mut self,
        training_data: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Self {
        for (entity, entity_type) in training_data {
            let text = entity.as_ref();
            let etype = entity_type.as_ref();

            self.known_entities.insert(text.to_lowercase());
            *self.type_distributions.entry(etype.to_string()).or_insert(0) += 1;

            for ngram in self.extract_ngrams(text) {
                self.train_ngrams.insert(ngram.clone());
                *self.ngram_frequencies.entry(ngram).or_insert(0) += 1;
            }
        }
        self
    }

    /// Check if a single entity is OOD.
    pub fn is_ood(&self, entity_text: &str) -> bool {
        self.check_ood(entity_text, None).is_ood
    }

    /// Check OOD status with detailed information.
    pub fn check_ood(&self, entity_text: &str, confidence: Option<f64>) -> OODStatus {
        let mut flagged_by = Vec::new();

        // Check vocabulary coverage
        let vocab_coverage = self.compute_vocab_coverage(entity_text);
        if vocab_coverage < self.config.vocab_coverage_threshold {
            flagged_by.push("low_vocab_coverage".to_string());
        }

        // Check exact match
        if !self.known_entities.contains(&entity_text.to_lowercase()) {
            // Only flag if also has low vocab coverage (unknown but similar = OK)
            if vocab_coverage < 0.8 {
                flagged_by.push("unseen_entity".to_string());
            }
        }

        // Check confidence threshold
        if let Some(conf) = confidence {
            if conf < self.config.confidence_threshold {
                flagged_by.push("low_confidence".to_string());
            }
        }

        // Check for unusual character patterns
        if self.has_unusual_characters(entity_text) {
            flagged_by.push("unusual_characters".to_string());
        }

        OODStatus {
            text: entity_text.to_string(),
            is_ood: !flagged_by.is_empty(),
            flagged_by,
            vocab_coverage,
            confidence,
        }
    }

    /// Analyze OOD statistics for a test dataset.
    pub fn analyze(&self, test_entities: &[(impl AsRef<str>, Option<f64>)]) -> OODAnalysisResults {
        let mut ood_count = 0;
        let mut by_method: HashMap<String, usize> = HashMap::new();
        let mut ood_confidences = Vec::new();
        let mut id_confidences = Vec::new();
        let mut sample_ood = Vec::new();

        let mut test_ngrams = HashSet::new();

        for (entity, confidence) in test_entities {
            let text = entity.as_ref();
            let status = self.check_ood(text, *confidence);

            // Collect test n-grams
            for ngram in self.extract_ngrams(text) {
                test_ngrams.insert(ngram);
            }

            if status.is_ood {
                ood_count += 1;
                for method in &status.flagged_by {
                    *by_method.entry(method.clone()).or_insert(0) += 1;
                }
                if let Some(conf) = confidence {
                    ood_confidences.push(*conf);
                }
                if sample_ood.len() < 10 {
                    sample_ood.push(text.to_string());
                }
            } else if let Some(conf) = confidence {
                id_confidences.push(*conf);
            }
        }

        // Compute vocab stats
        let unseen: usize = test_ngrams
            .iter()
            .filter(|ng| !self.train_ngrams.contains(*ng))
            .count();

        let coverage_ratio = if test_ngrams.is_empty() {
            1.0
        } else {
            1.0 - (unseen as f64 / test_ngrams.len() as f64)
        };

        OODAnalysisResults {
            total_entities: test_entities.len(),
            ood_count,
            ood_rate: if test_entities.is_empty() {
                0.0
            } else {
                ood_count as f64 / test_entities.len() as f64
            },
            by_method,
            avg_ood_confidence: if ood_confidences.is_empty() {
                0.0
            } else {
                ood_confidences.iter().sum::<f64>() / ood_confidences.len() as f64
            },
            avg_id_confidence: if id_confidences.is_empty() {
                0.0
            } else {
                id_confidences.iter().sum::<f64>() / id_confidences.len() as f64
            },
            vocab_stats: VocabCoverageStats {
                train_vocab_size: self.train_ngrams.len(),
                test_vocab_size: test_ngrams.len(),
                unseen_ngrams: unseen,
                coverage_ratio,
            },
            sample_ood_entities: sample_ood,
        }
    }

    // --- Internal helpers ---

    fn extract_ngrams(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.to_lowercase().chars().collect();
        let n = self.config.ngram_size;

        if chars.len() < n {
            return vec![chars.iter().collect()];
        }

        (0..=chars.len() - n)
            .map(|i| chars[i..i + n].iter().collect())
            .collect()
    }

    fn compute_vocab_coverage(&self, text: &str) -> f64 {
        let ngrams = self.extract_ngrams(text);
        if ngrams.is_empty() {
            return 1.0;
        }

        let seen = ngrams
            .iter()
            .filter(|ng| self.train_ngrams.contains(*ng))
            .count();

        seen as f64 / ngrams.len() as f64
    }

    fn has_unusual_characters(&self, text: &str) -> bool {
        // Check for characters that might indicate non-standard text
        let unusual_count = text
            .chars()
            .filter(|c| {
                // Flag zero-width chars, unusual Unicode, etc.
                matches!(c, '\u{200B}'..='\u{200F}' | '\u{FEFF}' | '\u{2060}')
            })
            .count();

        unusual_count > 0
    }
}

impl Default for OODDetector {
    fn default() -> Self {
        Self::new(OODConfig::default())
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Grade OOD rate for interpretability.
pub fn ood_rate_grade(rate: f64) -> &'static str {
    if rate < 0.05 {
        "Very low OOD (well-covered domain)"
    } else if rate < 0.15 {
        "Low OOD (mostly covered)"
    } else if rate < 0.30 {
        "Moderate OOD (some gaps)"
    } else if rate < 0.50 {
        "High OOD (significant gaps)"
    } else {
        "Very high OOD (major domain shift)"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ood_detection() {
        let training = vec!["John Smith", "Jane Doe", "Google", "Microsoft"];
        let detector = OODDetector::default().fit(&training);

        // Known entity should not be OOD
        assert!(!detector.is_ood("John Smith"));

        // Similar entity should have high coverage
        let status = detector.check_ood("John Doe", None);
        assert!(status.vocab_coverage > 0.5);
    }

    #[test]
    fn test_unusual_characters() {
        let detector = OODDetector::default();

        // Normal text
        let status = detector.check_ood("John Smith", None);
        assert!(!status.flagged_by.contains(&"unusual_characters".to_string()));

        // Text with zero-width space
        let status = detector.check_ood("John\u{200B}Smith", None);
        assert!(status.flagged_by.contains(&"unusual_characters".to_string()));
    }

    #[test]
    fn test_vocab_coverage() {
        let training = vec!["apple", "banana", "orange"];
        let detector = OODDetector::default().fit(&training);

        // Similar text should have good coverage
        let status = detector.check_ood("apple", None);
        assert!(status.vocab_coverage > 0.9);

        // Very different text should have low coverage
        let status = detector.check_ood("xyz123", None);
        assert!(status.vocab_coverage < 0.5);
    }

    #[test]
    fn test_analyze_dataset() {
        let training = vec!["John Smith", "Jane Doe"];
        let detector = OODDetector::default().fit(&training);

        let test_data: Vec<(&str, Option<f64>)> = vec![
            ("John Smith", Some(0.9)),    // In-distribution
            ("Xiangjun Chen", Some(0.3)), // OOD
        ];

        let results = detector.analyze(&test_data);
        assert_eq!(results.total_entities, 2);
        assert!(results.ood_count >= 1);
    }

    #[test]
    fn test_confidence_threshold() {
        let detector = OODDetector::new(OODConfig {
            confidence_threshold: 0.7,
            ..Default::default()
        });

        // Low confidence should flag OOD
        let status = detector.check_ood("test", Some(0.5));
        assert!(status.flagged_by.contains(&"low_confidence".to_string()));

        // High confidence should not flag
        let status = detector.check_ood("test", Some(0.9));
        assert!(!status.flagged_by.contains(&"low_confidence".to_string()));
    }

    #[test]
    fn test_ood_rate_grades() {
        assert_eq!(ood_rate_grade(0.02), "Very low OOD (well-covered domain)");
        assert_eq!(ood_rate_grade(0.10), "Low OOD (mostly covered)");
        assert_eq!(ood_rate_grade(0.25), "Moderate OOD (some gaps)");
        assert_eq!(ood_rate_grade(0.40), "High OOD (significant gaps)");
        assert_eq!(ood_rate_grade(0.60), "Very high OOD (major domain shift)");
    }
}

