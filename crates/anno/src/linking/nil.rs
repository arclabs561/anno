//! NIL Detection for Entity Linking.
//!
//! Identifies mentions that cannot be linked to any KB entry.
//!
//! Based on insights from "Contrastive Entity Coreference and Disambiguation for
//! Historical Texts" (Arora et al. 2024):
//!
//! > Historical documents are replete with individuals not remembered in
//! > contemporary knowledgebases. [...] We use a threshold on the cosine
//! > similarity to the closest entity in the knowledgebase to identify
//! > out-of-knowledgebase individuals.
//!
//! # NIL Reasons
//!
//! - **No candidates**: Candidate generator found nothing
//! - **Low confidence**: Best candidate score below threshold
//! - **Type mismatch**: NER type incompatible with all candidates
//! - **Emerging entity**: Entity exists but not yet in KB
//! - **Out-of-KB**: Entity unlikely to be in any KB (historical/local figure)
//!
//! # Design
//!
//! NIL detection uses multiple signals:
//! 1. Score distribution analysis (primary)
//! 2. Margin between top candidates (uncertainty measure)
//! 3. Out-of-KB confidence threshold (embedding-based)
//! 4. Coverage heuristics (mention characteristics)
//! 5. Learned classifier (optional)

use serde::{Deserialize, Serialize};

use super::candidate::Candidate;

/// Reason for NIL classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NilReason {
    /// No candidates were generated
    NoCandidates,
    /// Top score is below the confidence threshold
    LowConfidence,
    /// All candidates have incompatible types
    TypeMismatch,
    /// Mention appears to be noise (too short, numeric, etc.)
    NoisyMention,
    /// Large margin between scores suggests uncertainty
    LargeMargin,
    /// Explicit NIL (manually marked as unlinkable)
    ExplicitNil,
    /// Out-of-knowledgebase entity (embedding similarity too low)
    ///
    /// This is especially common in historical documents where many
    /// individuals are not remembered in contemporary KBs like Wikipedia.
    OutOfKnowledgebase,
    /// Emerging/recent entity not yet in KB
    EmergingEntity,
}

impl std::fmt::Display for NilReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCandidates => write!(f, "no_candidates"),
            Self::LowConfidence => write!(f, "low_confidence"),
            Self::TypeMismatch => write!(f, "type_mismatch"),
            Self::NoisyMention => write!(f, "noisy_mention"),
            Self::LargeMargin => write!(f, "large_margin"),
            Self::ExplicitNil => write!(f, "explicit_nil"),
            Self::OutOfKnowledgebase => write!(f, "out_of_kb"),
            Self::EmergingEntity => write!(f, "emerging_entity"),
        }
    }
}

/// NIL detector for entity linking.
///
/// Implements threshold-based out-of-KB detection following Arora et al. (2024):
/// > We use a threshold on the cosine similarity to the closest entity in the
/// > knowledgebase to identify out-of-knowledgebase individuals.
#[derive(Debug, Clone)]
pub struct NilDetector {
    /// Minimum score for a valid link
    score_threshold: f64,
    /// Maximum margin between top-2 candidates (for uncertainty)
    margin_threshold: f64,
    /// Minimum mention length
    min_mention_length: usize,
    /// Minimum candidates required to link
    min_candidates: usize,
    /// Out-of-KB threshold for embedding similarity.
    ///
    /// If the best candidate's embedding similarity is below this threshold,
    /// the entity is classified as out-of-knowledgebase. This is critical
    /// for historical documents where many individuals never made it to
    /// Wikipedia/Wikidata.
    ///
    /// From Arora et al.: typical values are 0.5-0.7 for bi-encoder similarity.
    out_of_kb_threshold: f64,
    /// Whether to prefer creating new entities over skipping
    ///
    /// When true, out-of-KB entities are flagged for entity creation rather
    /// than skipping. This is useful for building local entity registries
    /// from historical documents.
    prefer_create_over_skip: bool,
}

impl Default for NilDetector {
    fn default() -> Self {
        Self {
            score_threshold: 0.3,
            margin_threshold: 0.8,
            min_mention_length: 2,
            min_candidates: 1,
            out_of_kb_threshold: 0.5, // Conservative default
            prefer_create_over_skip: false,
        }
    }
}

impl NilDetector {
    /// Create a new NIL detector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set score threshold.
    pub fn with_score_threshold(mut self, threshold: f64) -> Self {
        self.score_threshold = threshold;
        self
    }

    /// Set margin threshold.
    pub fn with_margin_threshold(mut self, threshold: f64) -> Self {
        self.margin_threshold = threshold;
        self
    }

    /// Set out-of-KB threshold for embedding similarity.
    ///
    /// If the best candidate's embedding similarity is below this threshold,
    /// the entity is classified as out-of-knowledgebase.
    ///
    /// From Arora et al. (2024): typical values are 0.5-0.7 for bi-encoder similarity.
    pub fn with_out_of_kb_threshold(mut self, threshold: f64) -> Self {
        self.out_of_kb_threshold = threshold;
        self
    }

    /// Set whether to prefer creating new entities over skipping.
    ///
    /// When true, out-of-KB entities are flagged for entity creation rather
    /// than skipping. Useful for building local entity registries.
    pub fn with_prefer_create(mut self, prefer: bool) -> Self {
        self.prefer_create_over_skip = prefer;
        self
    }

    /// Check if a mention should be classified as NIL.
    ///
    /// Returns `Some(NilReason)` if NIL, `None` if linkable.
    pub fn check_nil(
        &self,
        mention: &str,
        candidates: &[Candidate],
        ner_type: Option<&str>,
    ) -> Option<NilReason> {
        // Check for noisy mention
        if self.is_noisy_mention(mention) {
            return Some(NilReason::NoisyMention);
        }

        // Check for no candidates
        if candidates.len() < self.min_candidates {
            return Some(NilReason::NoCandidates);
        }

        // Check type mismatch
        if let Some(ner_t) = ner_type {
            let has_compatible = candidates.iter().any(|c| {
                c.kb_type
                    .as_ref()
                    .map(|kt| super::candidate::type_compatibility(Some(ner_t), Some(kt)) > 0.5)
                    .unwrap_or(true) // No type info = assume compatible
            });
            if !has_compatible {
                return Some(NilReason::TypeMismatch);
            }
        }

        // Get top candidate score
        let top_score = candidates
            .iter()
            .map(|c| c.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Check low confidence
        if top_score < self.score_threshold {
            return Some(NilReason::LowConfidence);
        }

        // Check margin (if multiple candidates)
        if candidates.len() >= 2 {
            let mut scores: Vec<f64> = candidates.iter().map(|c| c.score).collect();
            scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            let margin = scores[0] - scores[1];
            // If margin is too small (close competition), might be uncertain
            if margin < 0.1 && top_score < 0.6 {
                // Only flag if top score isn't very high
                return Some(NilReason::LargeMargin);
            }
        }

        None // Linkable
    }

    /// Check if a mention is likely noise.
    fn is_noisy_mention(&self, mention: &str) -> bool {
        let trimmed = mention.trim();

        // Too short
        if trimmed.len() < self.min_mention_length {
            return true;
        }

        // Pure numeric
        if trimmed.chars().all(|c| c.is_numeric() || c.is_whitespace()) {
            return true;
        }

        // Pure punctuation
        if trimmed
            .chars()
            .all(|c| c.is_ascii_punctuation() || c.is_whitespace())
        {
            return true;
        }

        // Single character (unless CJK)
        if trimmed.chars().count() == 1 && !trimmed.chars().next().map(is_cjk).unwrap_or(false) {
            return true;
        }

        false
    }
}

/// Check if a character is CJK.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF |   // CJK Unified Ideographs
        0x3400..=0x4DBF |   // CJK Unified Ideographs Extension A
        0x20000..=0x2A6DF | // CJK Unified Ideographs Extension B
        0xF900..=0xFAFF |   // CJK Compatibility Ideographs
        0x2F800..=0x2FA1F   // CJK Compatibility Ideographs Supplement
    )
}

/// Extended candidate with embedding similarity.
///
/// Used for out-of-KB detection when embeddings are available.
#[derive(Debug, Clone)]
pub struct CandidateWithEmbedding<'a> {
    /// Reference to the base candidate
    pub candidate: &'a Candidate,
    /// Embedding similarity (cosine) between mention and candidate
    pub embedding_similarity: f64,
}

impl NilDetector {
    /// Check for out-of-KB entity using embedding similarity.
    ///
    /// This is the core insight from Arora et al. (2024):
    /// > We use a threshold on the cosine similarity to the closest entity
    /// > in the knowledgebase to identify out-of-knowledgebase individuals.
    ///
    /// Returns `Some(OutOfKnowledgebase)` if the best embedding similarity
    /// is below the threshold, indicating the entity is likely not in any KB.
    pub fn check_out_of_kb(&self, candidates: &[CandidateWithEmbedding]) -> Option<NilReason> {
        if candidates.is_empty() {
            return None; // Will be caught by NoCandidates check
        }

        let best_similarity = candidates
            .iter()
            .map(|c| c.embedding_similarity)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        if best_similarity < self.out_of_kb_threshold {
            Some(NilReason::OutOfKnowledgebase)
        } else {
            None
        }
    }

    /// Full NIL check with embedding similarity.
    ///
    /// This combines standard candidate-based NIL detection with
    /// embedding-based out-of-KB detection.
    pub fn check_nil_with_embeddings(
        &self,
        mention: &str,
        candidates: &[CandidateWithEmbedding],
        ner_type: Option<&str>,
    ) -> Option<NilReason> {
        // First, check noisy mention (doesn't need candidates)
        if self.is_noisy_mention(mention) {
            return Some(NilReason::NoisyMention);
        }

        // Check for no candidates
        if candidates.is_empty() {
            return Some(NilReason::NoCandidates);
        }

        // Check out-of-KB using embedding threshold
        // This is the key insight from Arora et al.
        if let Some(reason) = self.check_out_of_kb(candidates) {
            return Some(reason);
        }

        // Extract base candidates for remaining checks
        let base_candidates: Vec<&Candidate> = candidates.iter().map(|c| c.candidate).collect();

        // Check type mismatch
        if let Some(ner_t) = ner_type {
            let has_compatible = base_candidates.iter().any(|c| {
                c.kb_type
                    .as_ref()
                    .map(|kt| super::candidate::type_compatibility(Some(ner_t), Some(kt)) > 0.5)
                    .unwrap_or(true)
            });
            if !has_compatible {
                return Some(NilReason::TypeMismatch);
            }
        }

        // Get top candidate score
        let top_score = base_candidates
            .iter()
            .map(|c| c.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Check low confidence
        if top_score < self.score_threshold {
            return Some(NilReason::LowConfidence);
        }

        // Check margin
        if base_candidates.len() >= 2 {
            let mut scores: Vec<f64> = base_candidates.iter().map(|c| c.score).collect();
            scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            let margin = scores[0] - scores[1];
            if margin < 0.1 && top_score < 0.6 {
                return Some(NilReason::LargeMargin);
            }
        }

        None
    }

    /// Analyze with embedding-based out-of-KB detection.
    ///
    /// Returns full analysis including suggested action.
    pub fn analyze_with_embeddings(
        &self,
        mention: &str,
        candidates: &[CandidateWithEmbedding],
        ner_type: Option<&str>,
    ) -> NilAnalysis {
        let nil_result = self.check_nil_with_embeddings(mention, candidates, ner_type);

        match nil_result {
            None => {
                let best_sim = candidates
                    .iter()
                    .map(|c| c.embedding_similarity)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);

                NilAnalysis {
                    is_nil: false,
                    reason: None,
                    confidence: best_sim,
                    action: NilAction::Link,
                }
            }
            Some(reason) => {
                let (confidence, action) = match &reason {
                    NilReason::NoCandidates => {
                        if is_likely_entity_name(mention) {
                            (
                                0.7,
                                if self.prefer_create_over_skip {
                                    NilAction::CreateEntry
                                } else {
                                    NilAction::Review
                                },
                            )
                        } else {
                            (0.9, NilAction::Skip)
                        }
                    }
                    NilReason::OutOfKnowledgebase => {
                        // Key case: entity exists but not in KB (common in historical docs)
                        // High confidence this is a real entity, just not in Wikipedia
                        if is_likely_entity_name(mention) {
                            (
                                0.8,
                                if self.prefer_create_over_skip {
                                    NilAction::CreateEntry
                                } else {
                                    NilAction::Review
                                },
                            )
                        } else {
                            (0.6, NilAction::Review)
                        }
                    }
                    NilReason::EmergingEntity => (0.7, NilAction::CreateEntry),
                    NilReason::LowConfidence => (0.6, NilAction::Review),
                    NilReason::TypeMismatch => (0.8, NilAction::Review),
                    NilReason::NoisyMention => (0.95, NilAction::Skip),
                    NilReason::LargeMargin => (0.5, NilAction::Review),
                    NilReason::ExplicitNil => (1.0, NilAction::Skip),
                };

                NilAnalysis {
                    is_nil: true,
                    reason: Some(reason),
                    confidence,
                    action,
                }
            }
        }
    }
}

/// Result of NIL analysis including calibrated score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NilAnalysis {
    /// Whether this is NIL
    pub is_nil: bool,
    /// Reason if NIL
    pub reason: Option<NilReason>,
    /// Confidence in the NIL decision (0-1)
    pub confidence: f64,
    /// Suggested action
    pub action: NilAction,
}

/// Suggested action for NIL mentions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NilAction {
    /// Link to best candidate (not NIL)
    Link,
    /// Skip this mention
    Skip,
    /// Flag for human review
    Review,
    /// Candidate for new KB entry
    CreateEntry,
}

impl NilDetector {
    /// Full NIL analysis with suggested action.
    pub fn analyze(
        &self,
        mention: &str,
        candidates: &[Candidate],
        ner_type: Option<&str>,
    ) -> NilAnalysis {
        let nil_result = self.check_nil(mention, candidates, ner_type);

        match nil_result {
            None => {
                // Linkable
                let top_score = candidates
                    .iter()
                    .map(|c| c.score)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);

                NilAnalysis {
                    is_nil: false,
                    reason: None,
                    confidence: top_score,
                    action: NilAction::Link,
                }
            }
            Some(reason) => {
                let (confidence, action) = match &reason {
                    NilReason::NoCandidates => {
                        // High confidence it's NIL, but might be new entity
                        if is_likely_entity_name(mention) {
                            (0.7, NilAction::CreateEntry)
                        } else {
                            (0.9, NilAction::Skip)
                        }
                    }
                    NilReason::LowConfidence => (0.6, NilAction::Review),
                    NilReason::TypeMismatch => (0.8, NilAction::Review),
                    NilReason::NoisyMention => (0.95, NilAction::Skip),
                    NilReason::LargeMargin => (0.5, NilAction::Review),
                    NilReason::ExplicitNil => (1.0, NilAction::Skip),
                    NilReason::OutOfKnowledgebase => (0.85, NilAction::CreateEntry),
                    NilReason::EmergingEntity => (0.75, NilAction::CreateEntry),
                };

                NilAnalysis {
                    is_nil: true,
                    reason: Some(reason),
                    confidence,
                    action,
                }
            }
        }
    }
}

/// Heuristic check if mention looks like a proper entity name.
fn is_likely_entity_name(mention: &str) -> bool {
    let trimmed = mention.trim();

    // Has uppercase start
    let has_upper = trimmed
        .chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false);

    // Multiple words with capitals
    let cap_words = trimmed
        .split_whitespace()
        .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        .count();

    has_upper && cap_words >= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nil_no_candidates() {
        let detector = NilDetector::new();
        let result = detector.check_nil("Unknown Entity", &[], None);
        assert_eq!(result, Some(NilReason::NoCandidates));
    }

    #[test]
    fn test_nil_noisy_mention() {
        let detector = NilDetector::new();
        assert_eq!(
            detector.check_nil("123", &[], None),
            Some(NilReason::NoisyMention)
        );
        assert_eq!(
            detector.check_nil(".", &[], None),
            Some(NilReason::NoisyMention)
        );
    }

    #[test]
    fn test_linkable() {
        let detector = NilDetector::new();
        let mut candidate = super::super::candidate::Candidate::new(
            "Q937",
            super::super::candidate::CandidateSource::Wikidata,
            "Albert Einstein",
        );
        candidate.score = 0.8;

        let result = detector.check_nil("Einstein", &[candidate], Some("PERSON"));
        assert_eq!(result, None); // Linkable
    }

    #[test]
    fn test_nil_analysis() {
        let detector = NilDetector::new();
        let analysis = detector.analyze("Unknown Entity", &[], None);

        assert!(analysis.is_nil);
        assert!(matches!(analysis.reason, Some(NilReason::NoCandidates)));
    }

    #[test]
    fn test_is_cjk() {
        assert!(is_cjk('中'));
        assert!(is_cjk('日'));
        assert!(!is_cjk('A'));
    }

    #[test]
    fn test_out_of_kb_detection() {
        let detector = NilDetector::new().with_out_of_kb_threshold(0.5);

        // Create candidate with low embedding similarity (historical figure)
        let mut candidate = super::super::candidate::Candidate::new(
            "Q12345",
            super::super::candidate::CandidateSource::Wikidata,
            "John Smith",
        );
        candidate.score = 0.6;

        let candidates_with_embeddings = vec![CandidateWithEmbedding {
            candidate: &candidate,
            embedding_similarity: 0.3, // Below threshold
        }];

        let result = detector.check_out_of_kb(&candidates_with_embeddings);
        assert_eq!(result, Some(NilReason::OutOfKnowledgebase));
    }

    #[test]
    fn test_out_of_kb_above_threshold() {
        let detector = NilDetector::new().with_out_of_kb_threshold(0.5);

        let mut candidate = super::super::candidate::Candidate::new(
            "Q937",
            super::super::candidate::CandidateSource::Wikidata,
            "Albert Einstein",
        );
        candidate.score = 0.9;

        let candidates_with_embeddings = vec![CandidateWithEmbedding {
            candidate: &candidate,
            embedding_similarity: 0.85, // Above threshold
        }];

        let result = detector.check_out_of_kb(&candidates_with_embeddings);
        assert_eq!(result, None); // Not out-of-KB
    }

    #[test]
    fn test_prefer_create_over_skip() {
        let detector = NilDetector::new()
            .with_out_of_kb_threshold(0.5)
            .with_prefer_create(true);

        // Historical figure not in KB
        let mut candidate = super::super::candidate::Candidate::new(
            "Q99999",
            super::super::candidate::CandidateSource::Wikidata,
            "Unknown Person",
        );
        candidate.score = 0.4;

        let candidates = vec![CandidateWithEmbedding {
            candidate: &candidate,
            embedding_similarity: 0.3,
        }];

        let analysis = detector.analyze_with_embeddings(
            "Mayor Thomas Jenkins", // Looks like entity name
            &candidates,
            Some("PERSON"),
        );

        assert!(analysis.is_nil);
        assert_eq!(analysis.reason, Some(NilReason::OutOfKnowledgebase));
        assert_eq!(analysis.action, NilAction::CreateEntry);
    }

    #[test]
    fn test_nil_reason_display() {
        assert_eq!(NilReason::OutOfKnowledgebase.to_string(), "out_of_kb");
        assert_eq!(NilReason::EmergingEntity.to_string(), "emerging_entity");
    }
}
