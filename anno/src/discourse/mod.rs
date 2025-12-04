//! Discourse-level entities and event extraction for abstract anaphora resolution.
//!
//! # Overview
//!
//! This module extends NER beyond nominal entities to handle discourse-level
//! referents that abstract anaphors like "this" and "that" can refer to:
//!
//! - **Events**: "Russia invaded Ukraine" → the invasion
//! - **Propositions**: "She might resign" → the possibility
//! - **Facts**: "Water boils at 100C" → this fact
//! - **Situations**: "Prices rose while wages fell" → this state
//!
//! # Components
//!
//! - [`EventExtractor`] - Rule-based event trigger extraction
//! - [`EventMention`] - Extracted event with trigger, type, and arguments
//! - [`DiscourseReferent`] - Any entity that can be referred to (nominal or abstract)
//! - [`ShellNoun`] - Abstract nouns like "problem", "issue", "fact"
//! - [`DiscourseScope`] - Sentence/clause boundary detection
//!
//! # Example
//!
//! ```rust
//! use anno::discourse::{EventExtractor, DiscourseScope, ReferentType};
//!
//! let text = "Russia invaded Ukraine. This caused inflation.";
//!
//! // Extract events
//! let extractor = EventExtractor::default();
//! let events = extractor.extract(text);
//! assert!(!events.is_empty());
//! assert_eq!(events[0].trigger, "invaded");
//!
//! // Analyze discourse structure
//! let scope = DiscourseScope::analyze(text);
//! assert_eq!(scope.sentence_count(), 2);
//!
//! // Get candidate antecedent spans for "This" at position 24
//! let candidates = scope.candidate_antecedent_spans(24);
//! assert!(!candidates.is_empty());
//! ```
//!
//! # Research Background
//!
//! Based on:
//! - Asher (1993): "Reference to Abstract Objects in Discourse"
//! - Kolhatkar & Hirst (2012): "Resolving 'this-issue' anaphors"
//! - Marasović et al. (2017): "A Mention-Ranking Model for Abstract Anaphora"
//! - Schmid (2000): Shell noun taxonomy (~670 nouns)
//!
//! # Future Directions
//!
//! For production deployments, consider integrating:
//! - **GLiNER2** for multi-task extraction (NER + events + classification)
//! - **GLiREL** for zero-shot relation extraction
//! - Neural event extraction models trained on ACE2005

mod event_extractor;
mod types;

// Re-export from event_extractor
pub use event_extractor::{EventExtractor, EventExtractorConfig, EventTriggerLexicon};

// Re-export from types (the original discourse.rs content)
pub use types::{
    classify_shell_noun, is_shell_noun, DiscourseReferent, DiscourseScope, EventCluster,
    EventCorefResolver, EventMention, EventPolarity, EventTense, ReferentType, ShellNoun,
    ShellNounClass,
};
