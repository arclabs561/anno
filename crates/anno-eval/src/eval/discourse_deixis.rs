//! Discourse Deixis Resolution.
//!
//! # Overview
//!
//! Discourse deixis refers to expressions that point to propositions, facts,
//! or discourse segments rather than entities. Unlike entity coreference,
//! the antecedent is not a noun phrase but a clause, sentence, or discourse unit.
//!
//! # Example
//!
//! ```text
//! "The stock crashed 40%. That was unexpected."
//!                         ^^^^ discourse deictic
//!                         antecedent: "The stock crashed 40%" (event/proposition)
//! ```
//!
//! # Discourse Deixis vs Entity Coreference
//!
//! | Aspect | Entity Coreference | Discourse Deixis |
//! |--------|-------------------|------------------|
//! | Antecedent | NP (entity) | Clause/proposition |
//! | Anaphor | pronouns, definite NPs | "this", "that", "it" |
//! | Semantic | Identity | Reference to content |
//!
//! # ARRAU Annotation
//!
//! ARRAU is one of the few resources that explicitly annotates discourse deixis
//! alongside identity coreference and bridging.
//!
//! # References
//!
//! - Webber (1991): "Structure and Ostension in the Interpretation of Discourse Deixis"
//! - Poesio et al. (2024): "ARRAU 3.0"

use anno::offset::TextSpan;
use serde::{Deserialize, Serialize};

/// Type of discourse deictic expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DeicticType {
    /// Demonstrative pronoun: "this", "that"
    #[default]
    Demonstrative,
    /// Pronoun "it" with propositional antecedent
    It,
    /// "So" in constructions like "I think so"
    So,
    /// Null complement (elided clause)
    NullComplement,
    /// Other deictic expression
    Other(String),
}

/// Type of antecedent for discourse deixis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DiscourseAntecedentType {
    /// Single clause
    #[default]
    Clause,
    /// Full sentence
    Sentence,
    /// Multiple sentences
    MultiSentence,
    /// Verb phrase
    VerbPhrase,
    /// Event description
    Event,
    /// Proposition/fact
    Proposition,
    /// Abstract entity (e.g., "the situation")
    AbstractEntity,
    /// Implicit (must be inferred from context)
    Implicit,
}

/// A discourse deictic expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseDeictic {
    /// The deictic expression (e.g., "that")
    pub text: String,
    /// Start offset
    pub start: usize,
    /// End offset
    pub end: usize,
    /// Type of deictic
    pub deictic_type: DeicticType,
    /// Sentence index containing the deictic
    pub sentence_idx: Option<usize>,
}

impl DiscourseDeictic {
    /// Create a new discourse deictic.
    pub fn new(text: &str, start: usize, end: usize) -> Self {
        let deictic_type = match text.to_lowercase().as_str() {
            "this" | "that" | "these" | "those" => DeicticType::Demonstrative,
            "it" => DeicticType::It,
            "so" => DeicticType::So,
            _ => DeicticType::Other(text.to_string()),
        };

        Self {
            text: text.to_string(),
            start,
            end,
            deictic_type,
            sentence_idx: None,
        }
    }

    /// Set the sentence index.
    pub fn with_sentence(mut self, idx: usize) -> Self {
        self.sentence_idx = Some(idx);
        self
    }
}

/// Antecedent for discourse deixis (typically a clause or proposition).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseAntecedent {
    /// The antecedent text
    pub text: String,
    /// Start offset
    pub start: usize,
    /// End offset
    pub end: usize,
    /// Type of antecedent
    pub antecedent_type: DiscourseAntecedentType,
    /// Sentence indices covered (for multi-sentence antecedents)
    pub sentence_indices: Vec<usize>,
}

impl DiscourseAntecedent {
    /// Create a new antecedent.
    pub fn new(text: &str, start: usize, end: usize) -> Self {
        Self {
            text: text.to_string(),
            start,
            end,
            antecedent_type: DiscourseAntecedentType::Clause,
            sentence_indices: Vec::new(),
        }
    }

    /// Set the antecedent type.
    pub fn with_type(mut self, ant_type: DiscourseAntecedentType) -> Self {
        self.antecedent_type = ant_type;
        self
    }

    /// Set sentence indices.
    pub fn with_sentences(mut self, indices: Vec<usize>) -> Self {
        self.sentence_indices = indices;
        self
    }
}

/// A resolved discourse deixis link.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseDeicticLink {
    /// The deictic expression
    pub deictic: DiscourseDeictic,
    /// The antecedent (discourse segment)
    pub antecedent: DiscourseAntecedent,
    /// Confidence in this resolution
    pub confidence: f64,
}

impl DiscourseDeicticLink {
    /// Create a new link.
    pub fn new(deictic: DiscourseDeictic, antecedent: DiscourseAntecedent) -> Self {
        Self {
            deictic,
            antecedent,
            confidence: 1.0,
        }
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Check if this is a demonstrative deixis.
    pub fn is_demonstrative(&self) -> bool {
        matches!(self.deictic.deictic_type, DeicticType::Demonstrative)
    }

    /// Check if antecedent spans multiple sentences.
    pub fn is_multi_sentence(&self) -> bool {
        matches!(
            self.antecedent.antecedent_type,
            DiscourseAntecedentType::MultiSentence
        ) || self.antecedent.sentence_indices.len() > 1
    }
}

/// Document with discourse deixis annotations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscourseDeicticDocument {
    /// Document ID
    pub id: String,
    /// Document text
    pub text: String,
    /// Discourse deixis links
    pub links: Vec<DiscourseDeicticLink>,
}

impl DiscourseDeicticDocument {
    /// Create a new document.
    pub fn new(id: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            links: Vec::new(),
        }
    }

    /// Add a link.
    pub fn add_link(&mut self, link: DiscourseDeicticLink) {
        self.links.push(link);
    }

    /// Number of links.
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Get demonstrative deixis links.
    pub fn demonstratives(&self) -> Vec<&DiscourseDeicticLink> {
        self.links.iter().filter(|l| l.is_demonstrative()).collect()
    }

    /// Get multi-sentence antecedent links.
    pub fn multi_sentence(&self) -> Vec<&DiscourseDeicticLink> {
        self.links
            .iter()
            .filter(|l| l.is_multi_sentence())
            .collect()
    }
}

/// Simple rule-based discourse deixis detector.
pub struct DiscourseDeicticDetector {
    /// Patterns that indicate propositional "it"
    propositional_it_contexts: Vec<&'static str>,
}

impl Default for DiscourseDeicticDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscourseDeicticDetector {
    /// Create a new detector.
    pub fn new() -> Self {
        Self {
            propositional_it_contexts: vec![
                "it is clear that",
                "it seems that",
                "it appears that",
                "it is obvious that",
                "it is surprising that",
                "it is important that",
                "it is true that",
                "it is a fact that",
                "it follows that",
                "it means that",
            ],
        }
    }

    /// Detect potential discourse deictics in text.
    ///
    /// Note: This is a heuristic detector. Full resolution requires
    /// syntactic parsing and semantic analysis.
    pub fn detect(&self, text: &str) -> Vec<DiscourseDeictic> {
        let mut deictics = Vec::new();

        // Detect demonstratives that likely refer to propositions
        // Pattern: "That + verb" at sentence start or after punctuation
        let sentence_initial_that =
            regex::Regex::new(r"(?i)(?:^|[.!?]\s+)(that)\s+(?:was|is|seems|appears|means|shows)")
                .ok();

        if let Some(re) = sentence_initial_that {
            for cap in re.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    let span = TextSpan::from_bytes(text, m.start(), m.end());
                    let original = span.extract(text);
                    deictics.push(DiscourseDeictic::new(
                        original,
                        span.char_start,
                        span.char_end,
                    ));
                }
            }
        }

        // Detect "this" that refers to prior discourse
        // Pattern: "This + verb" (not followed by noun)
        let this_propositional =
            regex::Regex::new(r"(?i)\b(this)\s+(?:is|was|means|suggests|shows|indicates|explains)")
                .ok();

        if let Some(re) = this_propositional {
            for cap in re.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    let span = TextSpan::from_bytes(text, m.start(), m.end());
                    let original = span.extract(text);
                    deictics.push(DiscourseDeictic::new(
                        original,
                        span.char_start,
                        span.char_end,
                    ));
                }
            }
        }

        // Detect propositional "it"
        for pattern in &self.propositional_it_contexts {
            let pattern_len = pattern.len();
            if pattern_len == 0 {
                continue;
            }
            // ASCII-only patterns; use index-preserving ASCII case-insensitive scan.
            for (idx, _) in text.char_indices() {
                let Some(hay) = text.get(idx..idx + pattern_len) else {
                    continue;
                };
                if !hay.eq_ignore_ascii_case(pattern) {
                    continue;
                }
                // Find "it" within the pattern (byte offsets; "it" is ASCII)
                if let Some(it_pos) = pattern.find("it") {
                    let it_start = idx + it_pos;
                    let it_end = it_start + 2;
                    let span = TextSpan::from_bytes(text, it_start, it_end);
                    let original = span.extract(text);
                    deictics.push(DiscourseDeictic::new(
                        original,
                        span.char_start,
                        span.char_end,
                    ));
                }
            }
        }

        // Sort by position and deduplicate
        deictics.sort_by_key(|d| d.start);
        deictics.dedup_by(|a, b| a.start == b.start);

        deictics
    }
}

/// Evaluation metrics for discourse deixis resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscourseDeicticMetrics {
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1: f64,
    /// Accuracy on deictic type classification
    pub type_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno::offset::TextSpan;

    #[test]
    fn test_deictic_type() {
        let deictic = DiscourseDeictic::new("that", 0, 4);
        assert_eq!(deictic.deictic_type, DeicticType::Demonstrative);

        let deictic = DiscourseDeictic::new("it", 0, 2);
        assert_eq!(deictic.deictic_type, DeicticType::It);
    }

    #[test]
    fn test_discourse_antecedent() {
        let antecedent = DiscourseAntecedent::new("The stock crashed 40%", 0, 21)
            .with_type(DiscourseAntecedentType::Event)
            .with_sentences(vec![0]);

        assert_eq!(antecedent.antecedent_type, DiscourseAntecedentType::Event);
        assert_eq!(antecedent.sentence_indices, vec![0]);
    }

    #[test]
    fn test_discourse_deictic_link() {
        let deictic = DiscourseDeictic::new("That", 23, 27).with_sentence(1);
        let antecedent = DiscourseAntecedent::new("The stock crashed 40%", 0, 21)
            .with_type(DiscourseAntecedentType::Event);

        let link = DiscourseDeicticLink::new(deictic, antecedent);

        assert!(link.is_demonstrative());
        assert!(!link.is_multi_sentence());
    }

    #[test]
    fn test_detector() {
        let detector = DiscourseDeicticDetector::new();

        let text = "The company went bankrupt. That was unexpected.";
        let deictics = detector.detect(text);

        // Should detect "That" as likely propositional
        assert!(!deictics.is_empty());
    }

    #[test]
    fn test_document() {
        let mut doc = DiscourseDeicticDocument::new("doc1", "Event happened. That was surprising.");

        let deictic = DiscourseDeictic::new("That", 16, 20);
        let antecedent = DiscourseAntecedent::new("Event happened", 0, 14);
        doc.add_link(DiscourseDeicticLink::new(deictic, antecedent));

        assert_eq!(doc.len(), 1);
        assert_eq!(doc.demonstratives().len(), 1);
    }

    #[test]
    fn test_detector_offsets_are_character_offsets_on_unicode_text() {
        // Mixed-script text ensures we don't accidentally treat byte offsets as char offsets.
        let text = "東京で事件が起きた。That was unexpected. Müller noted: This means delays.";
        let detector = DiscourseDeicticDetector::new();
        let deictics = detector.detect(text);
        assert!(!deictics.is_empty(), "expected at least one deictic");

        for d in deictics {
            let extracted = TextSpan::from_chars(text, d.start, d.end).extract(text);
            assert_eq!(
                extracted, d.text,
                "deictic span should round-trip via char offsets"
            );
        }
    }
}
