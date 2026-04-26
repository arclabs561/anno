//! Framing-divergent Event Coreference (FRECO) types.
//!
//! This module implements types for the FRECO task from Zhao et al. (EMNLP 2025):
//! "Seeing the Same Story Differently: Framing-Divergent Event Coreference
//! for Computational Framing Analysis"
//!
//! # Task Definition
//!
//! FRECO identifies pairs of event mentions that:
//! 1. Refer to the same real-world occurrence (coreferent events)
//! 2. Differ in framing (lexical choice, causal attribution, valence, perspective)
//!
//! Unlike traditional event coreference which treats variation as noise to be
//! normalized away, FRECO treats **framing divergence as the signal**. The task
//! is motivated by media analysis: different news sources describe the same events
//! in systematically different ways that shape reader perception.
//!
//! # Motivating Example
//!
//! Consider these two sentences describing the same police shooting:
//!
//! ```text
//! Source A: "The officer acted decisively to neutralize the threat,
//!            preventing further harm to civilians."
//!
//! Source B: "The officer opened fire on the unarmed man,
//!            who posed no immediate danger."
//! ```
//!
//! Both describe the same real-world event, but with starkly different framing:
//! - **Source A**: Security-focused, justificatory (supportive attitude)
//! - **Source B**: Justice-focused, critical (skeptical attitude)
//!
//! FRECO captures this divergence for downstream framing analysis.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐
//! │  Document A │     │  Document B │
//! └──────┬──────┘     └──────┬──────┘
//!        │                   │
//!        ▼                   ▼
//! ┌─────────────┐     ┌─────────────┐
//! │EventMention │     │EventMention │
//! │  trigger    │     │  trigger    │
//! │  arguments  │     │  arguments  │
//! │  attitude   │     │  attitude   │
//! └──────┬──────┘     └──────┬──────┘
//!        │                   │
//!        └───────┬───────────┘
//!                ▼
//!         ┌───────────┐
//!         │ FrecoPair │
//!         │  label    │
//!         │ divergence│
//!         └───────────┘
//! ```
//!
//! # Framing Divergence Types
//!
//! | Type | Example | Linguistic Marker |
//! |------|---------|-------------------|
//! | Lexical | "hunted down" vs "pursued" | Connotation shift |
//! | Valence | "victory" vs "setback" | Emotional tone |
//! | Granularity | "lost his job" vs "mass layoffs" | Specificity level |
//! | Abstraction | "challenged authority" vs "filed complaint" | Concrete vs abstract |
//! | Causal | "self-defense" vs "unprovoked attack" | Attribution |
//! | Agency | "was killed" vs "died" | Passive vs intransitive |
//! | Participant | "protesters" vs "rioters" | Role framing |
//!
//! # Usage
//!
//! ```rust
//! use crate::core::types::framing::{EventMention, FrecoPair, FramingAttitude};
//!
//! // Create event mentions from two different news sources
//! let event_a = EventMention::new(
//!     "evt_001",
//!     "nytimes_article",
//!     "raid",
//!     45, 49,
//!     "Israeli forces conducted a raid on the hospital complex."
//! ).with_attitude(FramingAttitude::Neutral);
//!
//! let event_b = EventMention::new(
//!     "evt_002",
//!     "aljazeera_article",
//!     "attack",
//!     23, 29,
//!     "Israeli forces launched an attack on the medical facility."
//! ).with_attitude(FramingAttitude::Skeptical);
//!
//! // Create a FRECO pair for classification
//! let pair = FrecoPair::new(event_a, event_b)
//!     .with_label(true)  // Gold annotation: this IS a FRECO pair
//!     .with_similarity_score(0.85);  // From CDEC cross-encoder
//!
//! assert!(pair.is_cross_document());
//! ```
//!
//! # Relationship to Other Coreference Tasks
//!
//! | Task | Focus | Framing Treatment |
//! |------|-------|-------------------|
//! | Within-doc coref | Entity chains | N/A |
//! | Cross-doc coref (CDCR) | Entity linking | Variation as noise |
//! | Event coref | Event identity | Variation as noise |
//! | **FRECO** | Event + framing | **Variation as signal** |
//!
//! # References
//!
//! - Zhao, Hu & Xue (2025): "Seeing the Same Story Differently" - FRECO task
//! - Zhao et al. (2024): Media attitude detection via framing analysis
//! - Mitamura et al. (2017): Event hoppers in TAC KBP
//! - Hovy et al. (2013): "Events are not simple" - quasi-identity
//! - O'Gorman et al. (2016): Richer Event Description corpus

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Framing Attitude
// =============================================================================

/// Attitude conveyed by an event mention toward a topic or actor.
///
/// Represents the stance a text takes when describing an event: whether it
/// frames the event/actor positively, negatively, or neutrally. Based on
/// Zhao et al. (2024) media attitude detection work.
///
/// # Variants
///
/// - [`Supportive`](Self::Supportive): Endorses, justifies, or frames positively
/// - [`Skeptical`](Self::Skeptical): Questions, criticizes, or frames negatively
/// - [`Neutral`](Self::Neutral): Factual reporting without apparent valence
///
/// # Examples
///
/// ```rust
/// use crate::core::types::framing::FramingAttitude;
///
/// // Parsing from strings (case-insensitive, multiple aliases)
/// let supportive: FramingAttitude = "positive".parse().unwrap();
/// let skeptical: FramingAttitude = "critical".parse().unwrap();
/// let neutral: FramingAttitude = "neutral".parse().unwrap();
///
/// // Checking contrast (supportive vs skeptical)
/// assert!(supportive.contrasts_with(&skeptical));
/// assert!(!neutral.contrasts_with(&supportive));
///
/// // Numeric polarity for scoring
/// assert_eq!(supportive.polarity(), 1.0);
/// assert_eq!(neutral.polarity(), 0.0);
/// assert_eq!(skeptical.polarity(), -1.0);
/// ```
///
/// # Serialization
///
/// Serializes to lowercase strings: `"supportive"`, `"skeptical"`, `"neutral"`.
///
/// ```rust
/// use crate::core::types::framing::FramingAttitude;
///
/// let json = serde_json::to_string(&FramingAttitude::Skeptical).unwrap();
/// assert_eq!(json, "\"skeptical\"");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FramingAttitude {
    /// Supportive framing: endorses actions, justifies outcomes, frames positively.
    ///
    /// Example: "The officer acted decisively to neutralize the threat."
    Supportive,

    /// Skeptical framing: questions legitimacy, highlights harm, frames critically.
    ///
    /// Example: "The officer opened fire on the unarmed man."
    Skeptical,

    /// Neutral framing: factual reporting without apparent emotional valence.
    ///
    /// Example: "The officer discharged their weapon, striking the individual."
    #[default]
    Neutral,
}

impl FramingAttitude {
    /// Check if this attitude contrasts with another (oppositional framing).
    ///
    /// Two attitudes contrast if one is [`Supportive`](Self::Supportive) and the
    /// other is [`Skeptical`](Self::Skeptical). [`Neutral`](Self::Neutral) does
    /// not contrast with either—it represents absence of valence rather than
    /// opposition.
    ///
    /// This is the core signal for FRECO: two event mentions that corefer but
    /// have contrasting attitudes indicate framing divergence.
    ///
    /// # Returns
    ///
    /// `true` if `self` and `other` form a supportive/skeptical pair (in either order).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::core::types::framing::FramingAttitude;
    ///
    /// // Opposites contrast
    /// assert!(FramingAttitude::Supportive.contrasts_with(&FramingAttitude::Skeptical));
    /// assert!(FramingAttitude::Skeptical.contrasts_with(&FramingAttitude::Supportive));
    ///
    /// // Same attitude doesn't contrast
    /// assert!(!FramingAttitude::Supportive.contrasts_with(&FramingAttitude::Supportive));
    ///
    /// // Neutral never contrasts (it's absence of stance, not a stance)
    /// assert!(!FramingAttitude::Neutral.contrasts_with(&FramingAttitude::Supportive));
    /// assert!(!FramingAttitude::Neutral.contrasts_with(&FramingAttitude::Skeptical));
    /// ```
    #[must_use]
    pub fn contrasts_with(&self, other: &FramingAttitude) -> bool {
        matches!(
            (self, other),
            (FramingAttitude::Supportive, FramingAttitude::Skeptical)
                | (FramingAttitude::Skeptical, FramingAttitude::Supportive)
        )
    }

    /// Convert attitude to numeric polarity for quantitative analysis.
    ///
    /// Maps attitudes to a [-1, +1] scale:
    /// - [`Supportive`](Self::Supportive): `+1.0` (positive framing)
    /// - [`Neutral`](Self::Neutral): `0.0` (no valence)
    /// - [`Skeptical`](Self::Skeptical): `-1.0` (negative framing)
    ///
    /// Useful for computing attitude divergence between event pairs:
    /// `|attitude_a.polarity() - attitude_b.polarity()|` gives a divergence
    /// score in [0, 2].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::core::types::framing::FramingAttitude;
    ///
    /// let supportive = FramingAttitude::Supportive;
    /// let skeptical = FramingAttitude::Skeptical;
    ///
    /// // Maximum divergence is 2.0 (supportive vs skeptical)
    /// let divergence = (supportive.polarity() - skeptical.polarity()).abs();
    /// assert!((divergence - 2.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn polarity(&self) -> f64 {
        match self {
            FramingAttitude::Supportive => 1.0,
            FramingAttitude::Neutral => 0.0,
            FramingAttitude::Skeptical => -1.0,
        }
    }
}

impl std::fmt::Display for FramingAttitude {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FramingAttitude::Supportive => write!(f, "supportive"),
            FramingAttitude::Skeptical => write!(f, "skeptical"),
            FramingAttitude::Neutral => write!(f, "neutral"),
        }
    }
}

impl std::str::FromStr for FramingAttitude {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "supportive" | "support" | "positive" | "+" => Ok(FramingAttitude::Supportive),
            "skeptical" | "skeptic" | "negative" | "critical" | "-" => {
                Ok(FramingAttitude::Skeptical)
            }
            "neutral" | "none" | "0" | "" => Ok(FramingAttitude::Neutral),
            _ => Err(format!("Unknown framing attitude: {}", s)),
        }
    }
}

// =============================================================================
// Framing Divergence Type
// =============================================================================

/// Type of framing divergence between coreferential event mentions.
///
/// Categorizes *how* two mentions of the same event differ in their framing.
/// A FRECO pair may exhibit one or more divergence types; when multiple apply,
/// use [`Mixed`](Self::Mixed).
///
/// # Categories
///
/// | Type | Mechanism | Example |
/// |------|-----------|---------|
/// | [`Lexical`](Self::Lexical) | Word connotation | "pursue" vs "hunt down" |
/// | [`Valence`](Self::Valence) | Emotional tone | "victory" vs "setback" |
/// | [`Agency`](Self::Agency) | Voice/transitivity | "was killed" vs "died" |
/// | [`Causal`](Self::Causal) | Blame attribution | "self-defense" vs "murder" |
/// | [`Participant`](Self::Participant) | Actor framing | "protesters" vs "rioters" |
/// | [`Granularity`](Self::Granularity) | Specificity level | "John's job" vs "layoffs" |
/// | [`Abstraction`](Self::Abstraction) | Concrete vs abstract | "filed complaint" vs "challenged" |
/// | [`Temporal`](Self::Temporal) | Duration framing | "crisis" vs "incident" |
/// | [`Scope`](Self::Scope) | Geographic extent | "local" vs "international" |
///
/// # Examples
///
/// ```rust
/// use crate::core::types::framing::FramingDivergenceType;
///
/// // Iterate over all types
/// assert_eq!(FramingDivergenceType::all().len(), 10);
///
/// // Display
/// assert_eq!(FramingDivergenceType::Agency.to_string(), "agency");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FramingDivergenceType {
    /// Lexical choice: synonyms with different connotations
    /// Example: "hunted down" vs "pursued"
    Lexical,

    /// Valence/emotional tone difference
    /// Example: "victory" vs "defeat" (for same outcome, different perspective)
    Valence,

    /// Level of specificity/granularity
    /// Example: "lost his job" vs "mass layoffs"
    Granularity,

    /// Abstraction level difference
    /// Example: "challenged authority" vs "demanded accountability"
    Abstraction,

    /// Causal attribution difference
    /// Example: "self-defense" vs "unprovoked attack"
    Causal,

    /// Participant portrayal difference
    /// Example: "protesters" vs "rioters"
    Participant,

    /// Temporal emphasis difference
    /// Example: "ongoing crisis" vs "isolated incident"
    Temporal,

    /// Location/scope emphasis
    /// Example: "local dispute" vs "international conflict"
    Scope,

    /// Agency attribution
    /// Example: "was killed" vs "died" (passive vs intransitive)
    Agency,

    /// Multiple or unspecified divergence types
    Mixed,
}

impl FramingDivergenceType {
    /// Get all divergence types for enumeration.
    #[must_use]
    pub fn all() -> &'static [FramingDivergenceType] {
        &[
            FramingDivergenceType::Lexical,
            FramingDivergenceType::Valence,
            FramingDivergenceType::Granularity,
            FramingDivergenceType::Abstraction,
            FramingDivergenceType::Causal,
            FramingDivergenceType::Participant,
            FramingDivergenceType::Temporal,
            FramingDivergenceType::Scope,
            FramingDivergenceType::Agency,
            FramingDivergenceType::Mixed,
        ]
    }
}

impl std::fmt::Display for FramingDivergenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FramingDivergenceType::Lexical => write!(f, "lexical"),
            FramingDivergenceType::Valence => write!(f, "valence"),
            FramingDivergenceType::Granularity => write!(f, "granularity"),
            FramingDivergenceType::Abstraction => write!(f, "abstraction"),
            FramingDivergenceType::Causal => write!(f, "causal"),
            FramingDivergenceType::Participant => write!(f, "participant"),
            FramingDivergenceType::Temporal => write!(f, "temporal"),
            FramingDivergenceType::Scope => write!(f, "scope"),
            FramingDivergenceType::Agency => write!(f, "agency"),
            FramingDivergenceType::Mixed => write!(f, "mixed"),
        }
    }
}

impl std::str::FromStr for FramingDivergenceType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lexical" | "lex" => Ok(FramingDivergenceType::Lexical),
            "valence" | "val" | "sentiment" => Ok(FramingDivergenceType::Valence),
            "granularity" | "gran" | "specificity" => Ok(FramingDivergenceType::Granularity),
            "abstraction" | "abs" | "abstract" => Ok(FramingDivergenceType::Abstraction),
            "causal" | "cause" | "attribution" => Ok(FramingDivergenceType::Causal),
            "participant" | "part" | "actor" => Ok(FramingDivergenceType::Participant),
            "temporal" | "temp" | "time" => Ok(FramingDivergenceType::Temporal),
            "scope" | "geographic" | "extent" => Ok(FramingDivergenceType::Scope),
            "agency" | "voice" | "transitivity" => Ok(FramingDivergenceType::Agency),
            "mixed" | "multiple" | "other" => Ok(FramingDivergenceType::Mixed),
            _ => Err(format!("Unknown framing divergence type: {}", s)),
        }
    }
}

// =============================================================================
// Event Mention
// =============================================================================

/// Semantic role argument for an event mention.
///
/// Represents a participant or circumstance of an event, following PropBank-style
/// semantic role labeling (SRL). FRECO uses SRL to enable structured comparison
/// of event mentions: two events may have the same trigger but different
/// arguments (e.g., different agents), or vice versa.
///
/// # Role Labels
///
/// | Role | Meaning | Example |
/// |------|---------|---------|
/// | `ARG0` | Agent/actor | "The officer" in "The officer shot..." |
/// | `ARG1` | Patient/theme | "the man" in "...shot the man" |
/// | `ARGM-LOC` | Location | "in the street" |
/// | `ARGM-TMP` | Temporal | "on Tuesday" |
/// | `ARGM-MNR` | Manner | "violently" |
/// | `ARGM-CAU` | Cause | "in self-defense" |
///
/// # Offset Convention
///
/// `start` and `end` are **character offsets relative to the containing sentence**,
/// not the document. To convert to document offsets, add the sentence's start offset.
///
/// # Example
///
/// ```rust
/// use crate::core::types::framing::EventArgument;
///
/// let agent = EventArgument {
///     role: "ARG0".to_string(),
///     text: "The officer".to_string(),
///     start: 0,
///     end: 11,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventArgument {
    /// Role label following PropBank conventions.
    ///
    /// Core roles: `ARG0` (agent), `ARG1` (patient), `ARG2`-`ARG5` (other).
    /// Modifiers: `ARGM-LOC`, `ARGM-TMP`, `ARGM-MNR`, `ARGM-CAU`, etc.
    pub role: String,

    /// Surface text of the argument span.
    pub text: String,

    /// Character start offset (inclusive), relative to containing sentence.
    pub start: usize,

    /// Character end offset (exclusive), relative to containing sentence.
    pub end: usize,
}

/// An event mention with trigger, arguments, and framing metadata.
///
/// Represents a single mention of an event in text, richer than a simple entity
/// span. An event mention captures:
///
/// - **Trigger**: The word/phrase that evokes the event ("shot", "agreement", "raid")
/// - **Arguments**: Semantic role participants (agent, patient, location, time)
/// - **Context**: The containing sentence for disambiguation
/// - **Provenance**: Source document for cross-document analysis
/// - **Framing**: Optional attitude annotation for FRECO
///
/// # Relationship to Entity Mentions
///
/// | Aspect | Entity Mention | Event Mention |
/// |--------|----------------|---------------|
/// | Core span | Named entity | Event trigger |
/// | Arguments | None | SRL arguments |
/// | Relations | Coreference chains | Event coreference + framing |
///
/// # Construction
///
/// Use the builder pattern for readable construction:
///
/// ```rust
/// use crate::core::types::framing::{EventMention, EventArgument, FramingAttitude};
///
/// let mention = EventMention::new(
///     "evt_001",           // unique ID
///     "nytimes_12345",     // document ID
///     "raid",              // trigger text
///     45, 49,              // trigger offsets (document-relative)
///     "Israeli forces conducted a raid on the hospital."
/// )
/// .with_arguments(vec![
///     EventArgument {
///         role: "ARG0".to_string(),
///         text: "Israeli forces".to_string(),
///         start: 0,
///         end: 14,
///     },
/// ])
/// .with_attitude(FramingAttitude::Neutral);
///
/// assert_eq!(mention.agent().map(|a| a.text.as_str()), Some("Israeli forces"));
/// ```
///
/// # Offset Convention
///
/// - `trigger_start`, `trigger_end`: Character offsets **in the document**
/// - `sentence_start`: Document offset where the sentence begins
/// - `EventArgument` offsets: Relative to **sentence start**, not document
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventMention {
    /// Unique identifier for this mention (e.g., `"evt_001"`, UUID).
    pub id: String,

    /// Document ID this mention comes from.
    ///
    /// Used for cross-document analysis—two mentions with different `doc_id`
    /// values are cross-document, which is typical for FRECO pairs.
    pub doc_id: String,

    /// Event trigger text (the word/phrase evoking the event).
    ///
    /// Examples: "shot", "agreement", "raid", "collapsed", "announced"
    pub trigger: String,

    /// Trigger start offset in document (character index, inclusive).
    pub trigger_start: usize,

    /// Trigger end offset in document (character index, exclusive).
    pub trigger_end: usize,

    /// Containing sentence text for context.
    pub sentence: String,

    /// Sentence start offset in document (used to convert argument offsets).
    pub sentence_start: usize,

    /// Semantic role arguments from SRL (may be empty if SRL not available).
    pub arguments: Vec<EventArgument>,

    /// Framing attitude toward the main topic (if annotated).
    ///
    /// This is the key FRECO annotation: how does this mention frame the event?
    pub attitude: Option<FramingAttitude>,

    /// Model confidence score (if this mention was extracted by a model).
    pub confidence: Option<f64>,
}

impl EventMention {
    /// Create a new event mention with minimal information.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        doc_id: impl Into<String>,
        trigger: impl Into<String>,
        trigger_start: usize,
        trigger_end: usize,
        sentence: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            doc_id: doc_id.into(),
            trigger: trigger.into(),
            trigger_start,
            trigger_end,
            sentence: sentence.into(),
            sentence_start: 0,
            arguments: Vec::new(),
            attitude: None,
            confidence: None,
        }
    }

    /// Add SRL arguments.
    #[must_use]
    pub fn with_arguments(mut self, arguments: Vec<EventArgument>) -> Self {
        self.arguments = arguments;
        self
    }

    /// Set framing attitude.
    #[must_use]
    pub fn with_attitude(mut self, attitude: FramingAttitude) -> Self {
        self.attitude = Some(attitude);
        self
    }

    /// Get agent (ARG0) if present.
    #[must_use]
    pub fn agent(&self) -> Option<&EventArgument> {
        self.arguments
            .iter()
            .find(|a| a.role == "ARG0" || a.role.to_lowercase() == "agent")
    }

    /// Get patient (ARG1) if present.
    #[must_use]
    pub fn patient(&self) -> Option<&EventArgument> {
        self.arguments
            .iter()
            .find(|a| a.role == "ARG1" || a.role.to_lowercase() == "patient")
    }

    /// Get location argument if present.
    #[must_use]
    pub fn location(&self) -> Option<&EventArgument> {
        self.arguments
            .iter()
            .find(|a| a.role == "ARGM-LOC" || a.role.to_lowercase().contains("loc"))
    }

    /// Get temporal argument if present.
    #[must_use]
    pub fn temporal(&self) -> Option<&EventArgument> {
        self.arguments
            .iter()
            .find(|a| a.role == "ARGM-TMP" || a.role.to_lowercase().contains("tmp"))
    }

    /// Check if this mention is from a different document than another.
    #[must_use]
    pub fn is_cross_document(&self, other: &EventMention) -> bool {
        self.doc_id != other.doc_id
    }
}

// =============================================================================
// FRECO Pair
// =============================================================================

/// A pair of event mentions for FRECO classification.
///
/// The core unit of the FRECO task: two event mentions that may or may not
/// refer to the same real-world event with contrastive framing. A pair is
/// a **positive FRECO pair** if:
///
/// 1. The events corefer (same real-world occurrence)
/// 2. The framing diverges (different lexical choice, attitude, causal attribution, etc.)
///
/// # Classification Task
///
/// Given a candidate pair, predict:
/// - **Binary**: Is this a FRECO pair? (`label`)
/// - **Fine-grained**: What type of divergence? (`divergence_type`)
///
/// # Typical Workflow
///
/// ```text
/// Candidate Generation → Pair Scoring → Classification → Divergence Typing
///      (CDEC)           (similarity)    (label)         (divergence_type)
/// ```
///
/// # Example
///
/// ```rust
/// use crate::core::types::framing::{EventMention, FrecoPair, FramingDivergenceType, FramingAttitude};
///
/// let e1 = EventMention::new("e1", "src_a", "raid", 10, 14, "Forces conducted a raid.")
///     .with_attitude(FramingAttitude::Neutral);
/// let e2 = EventMention::new("e2", "src_b", "attack", 8, 14, "Forces launched an attack.")
///     .with_attitude(FramingAttitude::Skeptical);
///
/// let pair = FrecoPair::new(e1, e2)
///     .with_label(true)  // Annotated as positive FRECO
///     .with_divergence_type(FramingDivergenceType::Lexical)
///     .with_similarity_score(0.87);
///
/// assert!(pair.is_cross_document());
/// assert_eq!(pair.has_attitude_contrast(), Some(false)); // Neutral doesn't contrast
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrecoPair {
    /// First event mention (order is arbitrary for symmetric evaluation).
    pub event_a: EventMention,

    /// Second event mention.
    pub event_b: EventMention,

    /// Gold label (if annotated).
    ///
    /// - `Some(true)`: This is a FRECO pair (coreferent + framing divergence)
    /// - `Some(false)`: Not a FRECO pair (either not coreferent or no divergence)
    /// - `None`: Unlabeled (e.g., candidate for annotation or inference)
    pub label: Option<bool>,

    /// Type of framing divergence (if positive and annotated).
    ///
    /// Only meaningful when `label == Some(true)`. Use [`Mixed`](FramingDivergenceType::Mixed)
    /// when multiple divergence types apply.
    pub divergence_type: Option<FramingDivergenceType>,

    /// Cross-encoder similarity score from CDEC candidate generation.
    ///
    /// Higher scores indicate more likely coreference. Typical threshold: 0.5-0.7.
    /// Used for candidate ranking before classification.
    pub similarity_score: Option<f64>,

    /// Model prediction confidence for the FRECO label.
    ///
    /// Range: [0, 1]. Distinct from `similarity_score` (coreference) vs
    /// this (coreference + framing divergence).
    pub prediction_confidence: Option<f64>,

    /// Free-text annotation notes (e.g., annotator comments, edge cases).
    pub notes: Option<String>,
}

impl FrecoPair {
    /// Create a new FRECO pair.
    #[must_use]
    pub fn new(event_a: EventMention, event_b: EventMention) -> Self {
        Self {
            event_a,
            event_b,
            label: None,
            divergence_type: None,
            similarity_score: None,
            prediction_confidence: None,
            notes: None,
        }
    }

    /// Set gold label.
    #[must_use]
    pub fn with_label(mut self, is_freco: bool) -> Self {
        self.label = Some(is_freco);
        self
    }

    /// Set divergence type.
    #[must_use]
    pub fn with_divergence_type(mut self, dtype: FramingDivergenceType) -> Self {
        self.divergence_type = Some(dtype);
        self
    }

    /// Set similarity score from CDEC model.
    #[must_use]
    pub fn with_similarity_score(mut self, score: f64) -> Self {
        self.similarity_score = Some(score);
        self
    }

    /// Check if this is a cross-document pair.
    #[must_use]
    pub fn is_cross_document(&self) -> bool {
        self.event_a.is_cross_document(&self.event_b)
    }

    /// Check if attitudes contrast (if both annotated).
    #[must_use]
    pub fn has_attitude_contrast(&self) -> Option<bool> {
        match (&self.event_a.attitude, &self.event_b.attitude) {
            (Some(a), Some(b)) => Some(a.contrasts_with(b)),
            _ => None,
        }
    }

    /// Get attitude polarity difference.
    #[must_use]
    pub fn attitude_divergence(&self) -> Option<f64> {
        match (&self.event_a.attitude, &self.event_b.attitude) {
            (Some(a), Some(b)) => Some((a.polarity() - b.polarity()).abs()),
            _ => None,
        }
    }
}

// =============================================================================
// FRECO Corpus
// =============================================================================

/// A corpus of FRECO-annotated event pairs, organized by topic.
///
/// The FRECO corpus structure groups pairs by news topics (e.g., "al-shifa",
/// "jan-6", "roe-v-wade"), enabling topic-stratified evaluation and
/// leave-one-topic-out cross-validation.
///
/// # Structure
///
/// ```text
/// FrecoCorpus
/// ├── topics/
/// │   ├── "al-shifa" → [FrecoPair, FrecoPair, ...]
/// │   ├── "jan-6"    → [FrecoPair, FrecoPair, ...]
/// │   └── "roe"      → [FrecoPair, FrecoPair, ...]
/// ├── mentions/      → {id → EventMention}  (for lookup)
/// └── metadata       → counts, IAA, citation
/// ```
///
/// # Example
///
/// ```rust
/// use crate::core::types::framing::{FrecoCorpus, FrecoPair, EventMention};
///
/// let mut corpus = FrecoCorpus::new("freco_v1");
///
/// let e1 = EventMention::new("e1", "doc1", "raid", 0, 4, "The raid began.");
/// let e2 = EventMention::new("e2", "doc2", "operation", 0, 9, "The operation started.");
///
/// corpus.add_pair("al-shifa", FrecoPair::new(e1, e2).with_label(true));
///
/// assert_eq!(corpus.metadata.num_pairs, 1);
/// assert_eq!(corpus.topic_names().len(), 1);
/// ```
///
/// # Evaluation Protocols
///
/// - **Standard**: Train on 80% of pairs, test on 20% (topic-mixed)
/// - **Cross-topic**: Leave-one-topic-out for generalization testing
/// - **Mining**: Rank candidates by similarity, evaluate P@K, R@K
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrecoCorpus {
    /// Corpus name/identifier (e.g., "freco_v1", "freco_expanded").
    pub name: String,

    /// Event pairs organized by topic.
    ///
    /// Keys are topic identifiers (e.g., "al-shifa", "jan-6").
    /// Each topic contains pairs from multiple news sources about that topic.
    pub topics: HashMap<String, Vec<FrecoPair>>,

    /// All event mentions indexed by ID for O(1) lookup.
    ///
    /// Populated automatically when pairs are added via [`add_pair`](Self::add_pair).
    pub mentions: HashMap<String, EventMention>,

    /// Corpus-level metadata (counts, agreement, citation).
    pub metadata: FrecoCorpusMetadata,
}

/// Metadata about a FRECO corpus.
///
/// Tracks corpus statistics and provenance information. Automatically updated
/// when pairs are added to the corpus.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrecoCorpusMetadata {
    /// Total number of annotated pairs (positive + negative).
    pub num_pairs: usize,

    /// Number of positive FRECO pairs (coreferent + framing divergence).
    pub num_positive: usize,

    /// Inter-annotator agreement (Cohen's kappa) if measured.
    ///
    /// Typical values for FRECO: κ = 0.6-0.8 (substantial to good agreement).
    pub inter_annotator_agreement: Option<f64>,

    /// Source description (e.g., "News articles from 2023-2024").
    pub source: Option<String>,

    /// Citation for the corpus (e.g., "Zhao et al. (2025)").
    pub citation: Option<String>,
}

impl FrecoCorpus {
    /// Create a new empty corpus.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            topics: HashMap::new(),
            mentions: HashMap::new(),
            metadata: FrecoCorpusMetadata::default(),
        }
    }

    /// Add a pair to a topic.
    pub fn add_pair(&mut self, topic: impl Into<String>, pair: FrecoPair) {
        // Index mentions
        self.mentions
            .insert(pair.event_a.id.clone(), pair.event_a.clone());
        self.mentions
            .insert(pair.event_b.id.clone(), pair.event_b.clone());

        // Add to topic
        self.topics.entry(topic.into()).or_default().push(pair);

        // Update metadata
        self.update_metadata();
    }

    /// Update metadata counts.
    fn update_metadata(&mut self) {
        self.metadata.num_pairs = self.topics.values().map(|v| v.len()).sum();
        self.metadata.num_positive = self
            .topics
            .values()
            .flat_map(|v| v.iter())
            .filter(|p| p.label == Some(true))
            .count();
    }

    /// Get all pairs across all topics.
    #[must_use]
    pub fn all_pairs(&self) -> Vec<&FrecoPair> {
        self.topics.values().flat_map(|v| v.iter()).collect()
    }

    /// Get positive FRECO pairs only.
    #[must_use]
    pub fn positive_pairs(&self) -> Vec<&FrecoPair> {
        self.all_pairs()
            .into_iter()
            .filter(|p| p.label == Some(true))
            .collect()
    }

    /// Get pairs for a specific topic.
    #[must_use]
    pub fn pairs_for_topic(&self, topic: &str) -> Option<&[FrecoPair]> {
        self.topics.get(topic).map(|v| v.as_slice())
    }

    /// Get topic names.
    #[must_use]
    pub fn topic_names(&self) -> Vec<&str> {
        self.topics.keys().map(|s| s.as_str()).collect()
    }

    /// Get positive rate (fraction of positive pairs).
    #[must_use]
    pub fn positive_rate(&self) -> f64 {
        if self.metadata.num_pairs == 0 {
            0.0
        } else {
            self.metadata.num_positive as f64 / self.metadata.num_pairs as f64
        }
    }
}

// =============================================================================
// Event Coreference Relation Types
// =============================================================================

/// Type of event coreference relation.
///
/// Events can be related in various ways beyond strict identity. This enum
/// captures the granularity of event coreference following Hovy et al. (2013)
/// "Events are Not Simple" and O'Gorman et al. (2016) RED annotation.
///
/// # Strict vs Hopper Coreference
///
/// - **Strict**: Only [`Full`](Self::Full) counts as coreference
/// - **Hopper**: Any non-[`None`](Self::None) relation counts (TAC KBP style)
///
/// FRECO uses hopper-style: any event relation can exhibit framing divergence.
///
/// # Relation Types
///
/// | Relation | Example |
/// |----------|---------|
/// | Full | "the shooting" = "when he opened fire" |
/// | Subevent | "the invasion" ⊃ "crossing the border" |
/// | Membership | "Tuesday's protest" ∈ "the week of protests" |
/// | ConceptInstance | "his murder" ~ "the killing of John Doe" |
///
/// # Example
///
/// ```rust
/// use crate::core::types::framing::EventCorefRelation;
///
/// let rel = EventCorefRelation::Subevent;
///
/// assert!(rel.is_coreferent());  // Counts for FRECO
/// assert!(!rel.is_strict());      // Not strict identity
/// assert!(rel.is_hopper());       // Hopper-style coreference
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventCorefRelation {
    /// Full coreference: same event, same participants, same time/place.
    ///
    /// The strongest relation—two mentions refer to exactly the same occurrence.
    Full,

    /// Subevent: one event is a part or phase of the other.
    ///
    /// Example: "the invasion" contains "crossing the border" as a subevent.
    Subevent,

    /// Superset: one event encompasses multiple instances of the other.
    ///
    /// Example: "the war" encompasses multiple "battles".
    Superset,

    /// Concept-instance: abstract event type vs concrete realization.
    ///
    /// Example: "a murder" (concept) vs "the killing of John Doe" (instance).
    ConceptInstance,

    /// Membership: one event is a member of a larger event set.
    ///
    /// Example: "Tuesday's protest" is one of "the week of protests".
    Membership,

    /// Causal: one event causes or enables the other.
    ///
    /// Not strict coreference but may exhibit framing divergence.
    Causal,

    /// Temporal: events share a temporal relationship (before/after/during).
    ///
    /// Useful for event timeline construction.
    Temporal,

    /// Not coreferent: events are unrelated.
    None,
}

impl EventCorefRelation {
    /// Check if this relation qualifies for FRECO (any non-None).
    #[must_use]
    pub fn is_coreferent(&self) -> bool {
        !matches!(self, EventCorefRelation::None)
    }

    /// Check if this is strict (full) coreference.
    #[must_use]
    pub fn is_strict(&self) -> bool {
        matches!(self, EventCorefRelation::Full)
    }

    /// Check if this is a hopper-style relaxed coreference.
    #[must_use]
    pub fn is_hopper(&self) -> bool {
        self.is_coreferent() && !self.is_strict()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framing_attitude_contrast() {
        assert!(FramingAttitude::Supportive.contrasts_with(&FramingAttitude::Skeptical));
        assert!(FramingAttitude::Skeptical.contrasts_with(&FramingAttitude::Supportive));
        assert!(!FramingAttitude::Neutral.contrasts_with(&FramingAttitude::Supportive));
        assert!(!FramingAttitude::Neutral.contrasts_with(&FramingAttitude::Skeptical));
    }

    #[test]
    fn test_framing_attitude_polarity() {
        assert!((FramingAttitude::Supportive.polarity() - 1.0).abs() < 0.001);
        assert!((FramingAttitude::Neutral.polarity() - 0.0).abs() < 0.001);
        assert!((FramingAttitude::Skeptical.polarity() - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_event_mention_arguments() {
        let mention = EventMention::new("e1", "doc1", "shot", 10, 14, "The officer shot the man.")
            .with_arguments(vec![
                EventArgument {
                    role: "ARG0".to_string(),
                    text: "The officer".to_string(),
                    start: 0,
                    end: 11,
                },
                EventArgument {
                    role: "ARG1".to_string(),
                    text: "the man".to_string(),
                    start: 15,
                    end: 22,
                },
            ]);

        assert_eq!(
            mention.agent().map(|a| &a.text),
            Some(&"The officer".to_string())
        );
        assert_eq!(
            mention.patient().map(|a| &a.text),
            Some(&"the man".to_string())
        );
        assert!(mention.location().is_none());
    }

    #[test]
    fn test_freco_pair_attitude_divergence() {
        let e1 = EventMention::new("e1", "doc1", "neutralized", 0, 11, "...")
            .with_attitude(FramingAttitude::Supportive);
        let e2 = EventMention::new("e2", "doc2", "killed", 0, 6, "...")
            .with_attitude(FramingAttitude::Skeptical);

        let pair = FrecoPair::new(e1, e2);

        assert_eq!(pair.has_attitude_contrast(), Some(true));
        assert!(
            (pair
                .attitude_divergence()
                .expect("attitude divergence defined when both attitudes present")
                - 2.0)
                .abs()
                < 0.001
        );
        assert!(pair.is_cross_document());
    }

    #[test]
    fn test_freco_corpus() {
        let mut corpus = FrecoCorpus::new("test");

        let e1 = EventMention::new("e1", "doc1", "raid", 0, 4, "The raid...");
        let e2 = EventMention::new("e2", "doc2", "operation", 0, 9, "The operation...");

        corpus.add_pair("al-shifa", FrecoPair::new(e1, e2).with_label(true));

        assert_eq!(corpus.metadata.num_pairs, 1);
        assert_eq!(corpus.metadata.num_positive, 1);
        assert!((corpus.positive_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_event_coref_relation() {
        assert!(EventCorefRelation::Full.is_coreferent());
        assert!(EventCorefRelation::Full.is_strict());
        assert!(!EventCorefRelation::Full.is_hopper());

        assert!(EventCorefRelation::Subevent.is_coreferent());
        assert!(!EventCorefRelation::Subevent.is_strict());
        assert!(EventCorefRelation::Subevent.is_hopper());

        assert!(!EventCorefRelation::None.is_coreferent());
    }

    #[test]
    fn test_serde_roundtrip() {
        let attitude = FramingAttitude::Skeptical;
        let json = serde_json::to_string(&attitude).expect("serialize FramingAttitude");
        let recovered: FramingAttitude =
            serde_json::from_str(&json).expect("deserialize FramingAttitude");
        assert_eq!(attitude, recovered);

        let dtype = FramingDivergenceType::Causal;
        let json = serde_json::to_string(&dtype).expect("serialize FramingDivergenceType");
        let recovered: FramingDivergenceType =
            serde_json::from_str(&json).expect("deserialize FramingDivergenceType");
        assert_eq!(dtype, recovered);
    }

    #[test]
    fn test_framing_attitude_from_str() {
        // Standard names
        assert_eq!(
            "supportive"
                .parse::<FramingAttitude>()
                .expect("parse 'supportive'"),
            FramingAttitude::Supportive
        );
        assert_eq!(
            "skeptical"
                .parse::<FramingAttitude>()
                .expect("parse 'skeptical'"),
            FramingAttitude::Skeptical
        );
        assert_eq!(
            "neutral"
                .parse::<FramingAttitude>()
                .expect("parse 'neutral'"),
            FramingAttitude::Neutral
        );

        // Aliases
        assert_eq!(
            "positive"
                .parse::<FramingAttitude>()
                .expect("parse 'positive'"),
            FramingAttitude::Supportive
        );
        assert_eq!(
            "critical"
                .parse::<FramingAttitude>()
                .expect("parse 'critical'"),
            FramingAttitude::Skeptical
        );
        assert_eq!(
            "+".parse::<FramingAttitude>().expect("parse '+'"),
            FramingAttitude::Supportive
        );
        assert_eq!(
            "-".parse::<FramingAttitude>().expect("parse '-'"),
            FramingAttitude::Skeptical
        );

        // Case insensitive
        assert_eq!(
            "SUPPORTIVE"
                .parse::<FramingAttitude>()
                .expect("parse 'SUPPORTIVE'"),
            FramingAttitude::Supportive
        );

        // Error case
        assert!("unknown".parse::<FramingAttitude>().is_err());
    }

    #[test]
    fn test_framing_divergence_type_from_str() {
        // Standard names
        assert_eq!(
            "lexical"
                .parse::<FramingDivergenceType>()
                .expect("parse 'lexical'"),
            FramingDivergenceType::Lexical
        );
        assert_eq!(
            "agency"
                .parse::<FramingDivergenceType>()
                .expect("parse 'agency'"),
            FramingDivergenceType::Agency
        );
        assert_eq!(
            "causal"
                .parse::<FramingDivergenceType>()
                .expect("parse 'causal'"),
            FramingDivergenceType::Causal
        );

        // Aliases
        assert_eq!(
            "voice"
                .parse::<FramingDivergenceType>()
                .expect("parse 'voice'"),
            FramingDivergenceType::Agency
        );
        assert_eq!(
            "sentiment"
                .parse::<FramingDivergenceType>()
                .expect("parse 'sentiment'"),
            FramingDivergenceType::Valence
        );
        assert_eq!(
            "actor"
                .parse::<FramingDivergenceType>()
                .expect("parse 'actor'"),
            FramingDivergenceType::Participant
        );

        // Case insensitive
        assert_eq!(
            "LEXICAL"
                .parse::<FramingDivergenceType>()
                .expect("parse 'LEXICAL'"),
            FramingDivergenceType::Lexical
        );

        // Error case
        assert!("unknown".parse::<FramingDivergenceType>().is_err());
    }

    #[test]
    fn test_divergence_type_all() {
        let all = FramingDivergenceType::all();
        assert_eq!(all.len(), 10);
        assert!(all.contains(&FramingDivergenceType::Lexical));
        assert!(all.contains(&FramingDivergenceType::Mixed));
    }
}
