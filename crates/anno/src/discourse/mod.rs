//! Discourse-level analysis helpers (optional).
//!
//! This module is feature-gated (`discourse`) and is primarily used by evaluation tooling and
//! experiments. It does **not** change `anno`'s primary extraction contract: localization remains
//! character-offset spans over the input text.
//!
//! # The Problem: Coreference Beyond String Matching
//!
//! Simple coreference systems match mentions by string overlap: "Barack Obama"
//! and "Obama" probably refer to the same person. But this fails on:
//!
//! ```text
//! "Russia invaded Ukraine. This caused inflation."
//! ```
//!
//! What does "This" refer to? Not "Russia" or "Ukraine"—it refers to the *event*
//! of invasion. Standard NER extracts entities; discourse analysis extracts
//! what those entities *do* and how pronouns can refer to actions, facts, and
//! propositions.
//!
//! Similarly:
//!
//! ```text
//! "John told Mary that he would resign. She was surprised by this news."
//! ```
//!
//! Here "this news" refers to the *proposition* that John will resign—an abstract
//! object that was never explicitly mentioned as a noun phrase.
//!
//! # Why Discourse-Level Analysis?
//!
//! Discourse analysis solves problems that sentence-level NLP cannot:
//!
//! 1. **Abstract anaphora**: "This", "that", "it" referring to events/propositions
//! 2. **Topic tracking**: What is the discourse *about* right now?
//! 3. **Coherence**: Is this text well-organized or jumping between topics?
//! 4. **Ambiguity**: Which of several candidates does a pronoun refer to?
//!
//! The key insight (Israel 1994) is that natural language interpretation requires
//! maintaining a "virtual machine" state—tracking which entities are active,
//! how salient they are, and what actions/events have been predicated of them.
//!
//! # This Module
//!
//! Provides infrastructure for:
//!
//! - **Abstract anaphora**: Pronouns referring to events, propositions, facts
//! - **Centering theory**: Tracking discourse focus and coherence
//! - **Uncertain reference**: Deferred resolution when antecedent is ambiguous
//! - **Shell nouns**: Abstract nouns like "problem", "issue", "fact"
//! - **Dialogue analysis**: Turn-taking, response tokens, speaker attribution
//!
//! # Submodules
//!
//! - `centering` — Centering theory implementation (Grosz, Joshi, Weinstein 1995)
//! - `uncertain_reference` — Epsilon-term semantics for deferred resolution
//! - `dialogue` — Dialogue turn types, speech acts, response token classification
//!
//! # Core Types
//!
//! - `EventExtractor` — Rule-based event trigger extraction
//! - `EventMention` — Extracted event with trigger, type, and arguments
//! - `DiscourseReferent` — Any entity that can be referred to (nominal or abstract)
//! - `ShellNoun` — Abstract nouns with semantic classification
//! - `DiscourseScope` — Sentence/clause boundary detection
//! - `DialogueTurn` — A single turn in dialogue with speaker and speech act
//! - `SpeechActType` — Classification of pragmatic function (continuer, acknowledgment, etc.)
//! - `DialogueContext` — Multi-turn dialogue state tracking
//!
//! # Centering Theory Example
//!
//! Track discourse coherence through forward/backward-looking centers:
//!
//! ```rust,ignore
//! use anno::discourse::centering::{
//!     ForwardCenter, CenteringConfig, track_centers, GrammaticalRole,
//! };
//!
//! // Build utterances as entity mention lists
//! let utterances = vec![
//!     vec![
//!         ForwardCenter::new(1, "John", 1.0).with_role(GrammaticalRole::Subject),
//!         ForwardCenter::new(2, "Mary", 0.8).with_role(GrammaticalRole::DirectObject),
//!     ],
//!     vec![
//!         ForwardCenter::new(1, "He", 0.9).with_role(GrammaticalRole::Subject),
//!     ],
//! ];
//!
//! let states = track_centers(&utterances, &CenteringConfig::default());
//! // states[1].cb == Some(1)  -- John is the backward-looking center
//! // states[1].transition == CenteringTransition::Continue
//! ```
//!
//! # Uncertain Reference Example
//!
//! Handle ambiguous pronouns with deferred resolution:
//!
//! ```rust,ignore
//! use anno::discourse::uncertain_reference::{
//!     UncertainReference, ReferenceCandidate, ResolutionStrategy,
//! };
//!
//! let mut reference = UncertainReference::new("he");
//! reference.add_candidate(ReferenceCandidate::new(1, "John", 0.6));
//! reference.add_candidate(ReferenceCandidate::new(2, "Bill", 0.4));
//!
//! // Later, evidence arrives
//! reference.update_evidence(1, 0.2);  // Boost John
//!
//! // Check uncertainty
//! if reference.is_ambiguous(0.3) {
//!     println!("Multiple high-probability candidates");
//! }
//!
//! // Resolve when needed
//! let resolved = reference.resolve();
//! ```
//!
//! # Event Extraction Example
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
//! # Dialogue Analysis Example
//!
//! Track dialogue turns, response tokens, and agent interactions:
//!
//! ```rust
//! use anno::discourse::{DialogueTurn, DialogueContext, SpeechActType, ParticipantType};
//!
//! let mut ctx = DialogueContext::new();
//!
//! // Agent greeting
//! ctx.add_turn(DialogueTurn::new("Bonjour!", "GPT")
//!     .with_participant_type(ParticipantType::Agent)
//!     .with_speech_act(SpeechActType::Greeting));
//!
//! // Human response token that triggers cutoff
//! ctx.add_turn(DialogueTurn::new("oui", "EMM")
//!     .with_participant_type(ParticipantType::Human)
//!     .with_speech_act(SpeechActType::Continuer)
//!     .with_triggered_cutoff(true));
//!
//! assert_eq!(ctx.cutoff_count(), 1);
//! ```
//!
//! # Theoretical Background
//!
//! This module implements ideas from:
//!
//! - **Israel (1994)**: "The Very Idea of Dynamic Semantics" — critique showing
//!   that discourse referent identity is often indeterminate until later context
//! - **Grosz, Joshi, Weinstein (1995)**: Centering theory for local coherence
//! - **Strube (1998)**: "Never Look Back" — S-list model simplifying centering
//! - **Jiang et al. (2022)**: CT + recency for neural coreference
//!
//! The key insight is that natural language processing requires maintaining
//! a "virtual machine" state tracking which entities are active, how salient
//! they are, and how uncertainty about reference should be resolved.
//!
//! # See Also
//!
//! - `anno-eval::eval::incremental_coref` — Incremental coreference with EntityMemory
//! - [`crate::backends::mention_ranking`] — Mention-ranking coreference
//! - `anno-eval::eval::coref_resolver` — Discourse-aware coreference resolution

pub mod centering;
pub mod dialogue;
mod event_extractor;
mod types;
pub mod uncertain_reference;

// Re-export from centering
pub use centering::{
    analyze_coherence, compute_transition, score_antecedents, track_centers, CenteringConfig,
    CenteringState, CenteringTransition, CoherenceAnalysis, ForwardCenter, GrammaticalRole,
    InformationStatus,
};

// Re-export from event_extractor
pub use event_extractor::{EventExtractor, EventExtractorConfig, EventTriggerLexicon};

// Re-export from types
pub use types::{
    classify_shell_noun, is_shell_noun, DiscourseReferent, DiscourseScope, EventCluster,
    EventCorefResolver, EventMention, EventPolarity, EventTense, ReferentType, ShellNoun,
    ShellNounClass,
};

// Re-export from uncertain_reference
pub use uncertain_reference::{
    resolve_uncertain, CandidateSource, ConstraintKind, DeferredResolutionContext,
    ReferenceCandidate, ReferenceConstraint, ResolutionStatistics, ResolutionStrategy,
    UncertainReference,
};

// Re-export from dialogue
pub use dialogue::{
    classify_response_token, DialogueContext, DialogueTurn, ParticipantType, SpeechActType,
};
