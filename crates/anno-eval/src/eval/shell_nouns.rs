//! Shell Noun Resolution.
//!
//! # Overview
//!
//! Shell nouns are abstract nouns (e.g., "fact," "issue," "problem," "possibility")
//! that require antecedent content to be interpreted. Unlike typical anaphora that
//! refers to entities, shell nouns refer to propositions, events, or discourse segments.
//!
//! # Example
//!
//! ```text
//! "The merger was blocked. This fact surprised analysts."
//!                          ^^^^^^^^^ shell noun
//!                          antecedent: "The merger was blocked" (proposition)
//! ```
//!
//! # Shell Noun Categories (Schmid's Taxonomy)
//!
//! | Category | Examples | Semantic Type |
//! |----------|----------|---------------|
//! | **Factual** | fact, truth, point | Propositions |
//! | **Linguistic** | statement, claim, argument | Speech acts |
//! | **Mental** | idea, thought, belief | Cognitive states |
//! | **Modal** | possibility, chance, risk | Modality |
//! | **Eventive** | event, situation, process | Events |
//! | **Circumstantial** | problem, issue, difficulty | States of affairs |
//!
//! # Theoretical Connection
//!
//! In the higher-order unification view of anaphora (Dalrymple et al. 1991),
//! shell nouns act as **type constraints** on the property P being recovered.
//! When we resolve "this problem" to an antecedent, the category of "problem"
//! (Circumstantial → states of affairs) constrains which discourse segments
//! are valid solutions.
//!
//! This is analogous to typed higher-order unification: the shell noun's
//! semantic category specifies the type of the variable we're solving for.
//! A "factual" shell noun (like "fact") requires a propositional antecedent;
//! an "eventive" shell noun (like "event") requires an event antecedent.
//!
//! # Corpora
//!
//! - **CSN**: Cataphoric shell nouns (pattern "this [shell noun]")
//! - **ASN**: 670 English shell nouns with crowdsourced antecedent annotations
//!
//! # References
//!
//! - Schmid (2000): "English Abstract Nouns as Conceptual Shells"
//! - Kolhatkar & Hirst (2014): "Resolving Shell Nouns"
//! - Dalrymple, Shieber & Pereira (1991): "Ellipsis and Higher-Order Unification"

use anno::offset::TextSpan;
use serde::{Deserialize, Serialize};

/// Category of shell noun (Schmid's taxonomy).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ShellNounCategory {
    /// Factual: fact, truth, point
    #[default]
    Factual,
    /// Linguistic: statement, claim, argument, news, report
    Linguistic,
    /// Mental: idea, thought, belief, notion
    Mental,
    /// Modal: possibility, chance, risk, need
    Modal,
    /// Eventive: event, situation, process, action
    Eventive,
    /// Circumstantial: problem, issue, difficulty, question
    Circumstantial,
    /// Unknown or unclassified
    Other(String),
}

impl ShellNounCategory {
    /// Create from string label.
    pub fn from_label(label: &str) -> Self {
        match label.to_lowercase().as_str() {
            "factual" | "fact" => Self::Factual,
            "linguistic" | "speech" => Self::Linguistic,
            "mental" | "cognitive" => Self::Mental,
            "modal" | "modality" => Self::Modal,
            "eventive" | "event" => Self::Eventive,
            "circumstantial" | "circumstance" => Self::Circumstantial,
            other => Self::Other(other.to_string()),
        }
    }

    /// Get canonical label.
    pub fn as_label(&self) -> &str {
        match self {
            Self::Factual => "factual",
            Self::Linguistic => "linguistic",
            Self::Mental => "mental",
            Self::Modal => "modal",
            Self::Eventive => "eventive",
            Self::Circumstantial => "circumstantial",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// Common shell nouns with their categories.
pub fn shell_noun_lexicon() -> &'static [(&'static str, ShellNounCategory)] {
    &[
        // Factual
        ("fact", ShellNounCategory::Factual),
        ("truth", ShellNounCategory::Factual),
        ("point", ShellNounCategory::Factual),
        ("matter", ShellNounCategory::Factual),
        ("reality", ShellNounCategory::Factual),
        // Linguistic
        ("statement", ShellNounCategory::Linguistic),
        ("claim", ShellNounCategory::Linguistic),
        ("argument", ShellNounCategory::Linguistic),
        ("news", ShellNounCategory::Linguistic),
        ("report", ShellNounCategory::Linguistic),
        ("announcement", ShellNounCategory::Linguistic),
        ("message", ShellNounCategory::Linguistic),
        ("story", ShellNounCategory::Linguistic),
        ("explanation", ShellNounCategory::Linguistic),
        ("conclusion", ShellNounCategory::Linguistic),
        // Mental
        ("idea", ShellNounCategory::Mental),
        ("thought", ShellNounCategory::Mental),
        ("belief", ShellNounCategory::Mental),
        ("notion", ShellNounCategory::Mental),
        ("view", ShellNounCategory::Mental),
        ("opinion", ShellNounCategory::Mental),
        ("impression", ShellNounCategory::Mental),
        ("feeling", ShellNounCategory::Mental),
        ("assumption", ShellNounCategory::Mental),
        ("hypothesis", ShellNounCategory::Mental),
        // Modal
        ("possibility", ShellNounCategory::Modal),
        ("chance", ShellNounCategory::Modal),
        ("risk", ShellNounCategory::Modal),
        ("need", ShellNounCategory::Modal),
        ("requirement", ShellNounCategory::Modal),
        ("ability", ShellNounCategory::Modal),
        ("tendency", ShellNounCategory::Modal),
        // Eventive
        ("event", ShellNounCategory::Eventive),
        ("situation", ShellNounCategory::Eventive),
        ("process", ShellNounCategory::Eventive),
        ("action", ShellNounCategory::Eventive),
        ("activity", ShellNounCategory::Eventive),
        ("development", ShellNounCategory::Eventive),
        ("change", ShellNounCategory::Eventive),
        ("movement", ShellNounCategory::Eventive),
        // Circumstantial
        ("problem", ShellNounCategory::Circumstantial),
        ("issue", ShellNounCategory::Circumstantial),
        ("difficulty", ShellNounCategory::Circumstantial),
        ("question", ShellNounCategory::Circumstantial),
        ("challenge", ShellNounCategory::Circumstantial),
        ("crisis", ShellNounCategory::Circumstantial),
        ("phenomenon", ShellNounCategory::Circumstantial),
        ("aspect", ShellNounCategory::Circumstantial),
    ]
}

/// A shell noun instance with its antecedent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellNounInstance {
    /// The shell noun phrase (e.g., "this fact")
    pub shell_phrase: String,
    /// The shell noun itself (e.g., "fact")
    pub shell_noun: String,
    /// Shell noun category
    pub category: ShellNounCategory,
    /// Start offset of shell phrase
    pub shell_start: usize,
    /// End offset of shell phrase
    pub shell_end: usize,
    /// The antecedent content (proposition/event/discourse segment)
    pub antecedent: Option<ShellNounAntecedent>,
    /// Whether this is cataphoric (shell noun before antecedent)
    pub is_cataphoric: bool,
}

impl ShellNounInstance {
    /// Create a new shell noun instance.
    pub fn new(shell_phrase: &str, shell_noun: &str, start: usize, end: usize) -> Self {
        let category = shell_noun_lexicon()
            .iter()
            .find(|(noun, _)| *noun == shell_noun.to_lowercase())
            .map(|(_, cat)| cat.clone())
            .unwrap_or_else(|| ShellNounCategory::Other(shell_noun.to_string()));

        Self {
            shell_phrase: shell_phrase.to_string(),
            shell_noun: shell_noun.to_string(),
            category,
            shell_start: start,
            shell_end: end,
            antecedent: None,
            is_cataphoric: false,
        }
    }

    /// Set the antecedent.
    pub fn with_antecedent(mut self, antecedent: ShellNounAntecedent) -> Self {
        self.antecedent = Some(antecedent);
        self
    }

    /// Mark as cataphoric.
    pub fn as_cataphoric(mut self) -> Self {
        self.is_cataphoric = true;
        self
    }

    /// Check if resolved (has antecedent).
    pub fn is_resolved(&self) -> bool {
        self.antecedent.is_some()
    }
}

/// Antecedent for a shell noun (typically a clause or proposition).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellNounAntecedent {
    /// The antecedent text
    pub text: String,
    /// Start offset in document
    pub start: usize,
    /// End offset in document
    pub end: usize,
    /// Type of antecedent
    pub antecedent_type: AntecedentType,
    /// Sentence index containing antecedent
    pub sentence_idx: Option<usize>,
}

impl ShellNounAntecedent {
    /// Create a new antecedent.
    pub fn new(text: &str, start: usize, end: usize) -> Self {
        Self {
            text: text.to_string(),
            start,
            end,
            antecedent_type: AntecedentType::Clause,
            sentence_idx: None,
        }
    }

    /// Set the antecedent type.
    pub fn with_type(mut self, ant_type: AntecedentType) -> Self {
        self.antecedent_type = ant_type;
        self
    }
}

/// Type of shell noun antecedent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AntecedentType {
    /// Full clause/sentence
    #[default]
    Clause,
    /// Verb phrase
    VerbPhrase,
    /// Noun phrase (rare for shell nouns)
    NounPhrase,
    /// Multiple sentences/discourse segment
    DiscourseSegment,
    /// Implicit (must be inferred)
    Implicit,
}

/// Shell noun resolution for a document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShellNounDocument {
    /// Document ID
    pub id: String,
    /// Document text
    pub text: String,
    /// Detected shell noun instances
    pub instances: Vec<ShellNounInstance>,
}

impl ShellNounDocument {
    /// Create a new document.
    pub fn new(id: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            instances: Vec::new(),
        }
    }

    /// Add a shell noun instance.
    pub fn add_instance(&mut self, instance: ShellNounInstance) {
        self.instances.push(instance);
    }

    /// Number of instances.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Get resolved instances only.
    pub fn resolved(&self) -> Vec<&ShellNounInstance> {
        self.instances.iter().filter(|i| i.is_resolved()).collect()
    }

    /// Get unresolved instances.
    pub fn unresolved(&self) -> Vec<&ShellNounInstance> {
        self.instances.iter().filter(|i| !i.is_resolved()).collect()
    }
}

/// Simple pattern-based shell noun detector.
pub struct ShellNounDetector {
    /// Shell noun lexicon
    lexicon: std::collections::HashSet<String>,
}

impl Default for ShellNounDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ShellNounDetector {
    /// Create a new detector with default lexicon.
    pub fn new() -> Self {
        let lexicon = shell_noun_lexicon()
            .iter()
            .map(|(noun, _)| noun.to_string())
            .collect();
        Self { lexicon }
    }

    /// Detect shell nouns in text.
    ///
    /// Looks for patterns like "this [shell noun]", "the [shell noun] that", etc.
    pub fn detect(&self, text: &str) -> Vec<ShellNounInstance> {
        let mut instances = Vec::new();
        // Use ASCII-only lowercasing so match indices stay aligned with the original text.
        // Unicode `to_lowercase()` can change string length (e.g., ß → ss), invalidating indices.
        // Shell noun patterns are English/ASCII, so this is the correct behavior here.
        let lower = text.to_ascii_lowercase();

        // Pattern: "this/that/the [shell noun]"
        for noun in &self.lexicon {
            // "this [noun]" pattern (cataphoric)
            let pattern = format!("this {}", noun);
            for (idx, _) in lower.match_indices(&pattern) {
                let end = idx + pattern.len();
                let span = TextSpan::from_bytes(text, idx, end);
                let shell_phrase = span.extract(text);
                let mut instance =
                    ShellNounInstance::new(shell_phrase, noun, span.char_start, span.char_end);
                instance.is_cataphoric = true;
                instances.push(instance);
            }

            // "that [noun]" pattern
            let pattern = format!("that {}", noun);
            for (idx, _) in lower.match_indices(&pattern) {
                let end = idx + pattern.len();
                let span = TextSpan::from_bytes(text, idx, end);
                let shell_phrase = span.extract(text);
                instances.push(ShellNounInstance::new(
                    shell_phrase,
                    noun,
                    span.char_start,
                    span.char_end,
                ));
            }

            // "the [noun] that" pattern (cataphoric)
            let pattern = format!("the {} that", noun);
            for (idx, _) in lower.match_indices(&pattern) {
                let end = idx + format!("the {}", noun).len();
                let span = TextSpan::from_bytes(text, idx, end);
                let shell_phrase = span.extract(text);
                let mut instance =
                    ShellNounInstance::new(shell_phrase, noun, span.char_start, span.char_end);
                instance.is_cataphoric = true;
                instances.push(instance);
            }
        }

        // Sort by position
        instances.sort_by_key(|i| i.shell_start);
        instances
    }

    /// Check if a word is a shell noun.
    pub fn is_shell_noun(&self, word: &str) -> bool {
        self.lexicon.contains(&word.to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno::offset::TextSpan;

    #[test]
    fn test_shell_noun_category() {
        assert_eq!(
            ShellNounCategory::from_label("factual"),
            ShellNounCategory::Factual
        );
        assert_eq!(
            ShellNounCategory::from_label("modal"),
            ShellNounCategory::Modal
        );
    }

    #[test]
    fn test_shell_noun_instance() {
        let instance = ShellNounInstance::new("this fact", "fact", 0, 9);
        assert_eq!(instance.category, ShellNounCategory::Factual);
        assert!(!instance.is_resolved());
    }

    #[test]
    fn test_shell_noun_detector() {
        let detector = ShellNounDetector::new();

        let text = "The merger was blocked. This fact surprised analysts.";
        let instances = detector.detect(text);

        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].shell_noun, "fact");
        assert!(instances[0].is_cataphoric);
    }

    #[test]
    fn test_detector_multiple_patterns() {
        let detector = ShellNounDetector::new();

        let text = "This problem is serious. The issue that concernos us is timing.";
        let instances = detector.detect(text);

        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].shell_noun, "problem");
        assert_eq!(instances[1].shell_noun, "issue");
    }

    #[test]
    fn test_shell_noun_detector_unicode_safe_indices() {
        // Regression: Unicode before the pattern must not break match indexing, and offsets
        // must be character offsets (not bytes).
        let detector = ShellNounDetector::new();
        let text = "Müller said: this fact matters.";
        let instances = detector.detect(text);
        let inst = instances
            .iter()
            .find(|i| i.shell_phrase == "this fact")
            .expect("expected to detect 'this fact'");
        let round_trip = TextSpan::from_chars(text, inst.shell_start, inst.shell_end).extract(text);
        assert_eq!(round_trip, inst.shell_phrase);
    }

    #[test]
    fn test_shell_noun_document() {
        let mut doc = ShellNounDocument::new("doc1", "This fact is important.");

        let antecedent = ShellNounAntecedent::new("The merger was blocked", 0, 22);
        let instance =
            ShellNounInstance::new("This fact", "fact", 0, 9).with_antecedent(antecedent);

        doc.add_instance(instance);

        assert_eq!(doc.len(), 1);
        assert_eq!(doc.resolved().len(), 1);
        assert_eq!(doc.unresolved().len(), 0);
    }
}
