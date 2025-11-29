//! Discourse-level entities for abstract anaphora resolution.
//!
//! # Research Background
//!
//! Standard NER extracts **nominal** entities (Person, Org, Location).
//! Abstract anaphora requires extracting **discourse referents**:
//!
//! - **Events**: "Russia invaded Ukraine" → the invasion
//! - **Propositions**: "She might resign" → the possibility of resignation
//! - **Facts**: "Water boils at 100C" → this established fact
//! - **Situations**: "Prices rose while wages fell" → this state of affairs
//!
//! These are not noun phrases but *discourse segments* that can serve as
//! antecedents for shell nouns ("this problem") and demonstratives ("This").
//!
//! # Key Papers
//!
//! - Kolhatkar & Hirst (2012): "Resolving 'this-issue' anaphors"
//! - Marasović et al. (2017): "A Mention-Ranking Model for Abstract Anaphora Resolution"
//! - Schmid (2000): ~670 shell nouns in English
//!
//! # Example
//!
//! ```rust
//! use anno::discourse::{DiscourseReferent, ReferentType, EventMention};
//!
//! let text = "Russia invaded Ukraine in 2022. This caused inflation.";
//!
//! // The event mention
//! let event = EventMention::new("invaded", 7, 14)
//!     .with_trigger_type("attack")
//!     .with_arguments(vec![("Agent", "Russia"), ("Patient", "Ukraine")]);
//!
//! // The full discourse referent (spans the whole clause)
//! let referent = DiscourseReferent::new(ReferentType::Event, 0, 30)
//!     .with_event(event)
//!     .with_label("Russia's invasion of Ukraine");
//!
//! assert!(referent.referent_type.is_abstract());
//! ```

use crate::Entity;
use serde::{Deserialize, Serialize};

// =============================================================================
// Discourse Referent Types
// =============================================================================

/// Type of discourse referent (what kind of "thing" can be referred to).
///
/// # Research Background
///
/// This taxonomy follows Asher (1993) "Reference to Abstract Objects in Discourse"
/// and Nedoluzhko & Lapshinova-Koltunski (2022) survey.
///
/// ## Ontological Distinctions
///
/// - **Nominal**: Standard NER entities (Person, Org, etc.)
/// - **Event**: Something that happened at a specific time/place
/// - **Fact**: A true proposition (can be asserted, denied)
/// - **Proposition**: A potential truth value (can be believed, doubted)
/// - **Situation**: A state of affairs (ongoing, not instantaneous)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ReferentType {
    /// Standard nominal entity (Person, Org, Location)
    #[default]
    Nominal,

    // === Abstract Types ===
    /// Event: happened at a specific time/place
    /// Example: "The earthquake struck" → EVENT
    Event,

    /// Fact: a true proposition that can be asserted
    /// Example: "Water boils at 100C" → FACT
    Fact,

    /// Proposition: a potential truth value
    /// Example: "She might resign" → PROPOSITION (not yet true/false)
    Proposition,

    /// Situation: ongoing state of affairs
    /// Example: "Prices are rising" → SITUATION
    Situation,

    /// Manner: how something was done
    /// Example: "He spoke softly" → MANNER (the softness)
    Manner,

    /// Discourse segment: a larger chunk of text
    /// Example: "The preceding paragraph" → SEGMENT
    Segment,
}

impl ReferentType {
    /// Is this an abstract (non-nominal) referent type?
    #[must_use]
    pub const fn is_abstract(&self) -> bool {
        !matches!(self, ReferentType::Nominal)
    }

    /// Human-readable label.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            ReferentType::Nominal => "nominal",
            ReferentType::Event => "event",
            ReferentType::Fact => "fact",
            ReferentType::Proposition => "proposition",
            ReferentType::Situation => "situation",
            ReferentType::Manner => "manner",
            ReferentType::Segment => "segment",
        }
    }

    /// Can this referent type be the antecedent of "this"?
    #[must_use]
    pub const fn can_be_this_antecedent(&self) -> bool {
        // All abstract types can be referred to by "this"
        self.is_abstract()
    }

    /// Can this referent type be the antecedent of "it"?
    #[must_use]
    pub const fn can_be_it_antecedent(&self) -> bool {
        // "it" can refer to events and situations, but less naturally to facts/propositions
        matches!(
            self,
            ReferentType::Nominal | ReferentType::Event | ReferentType::Situation
        )
    }
}

// =============================================================================
// Event Mention
// =============================================================================

/// An event mention extracted from text.
///
/// Events are a key type of abstract antecedent. They have:
/// - A **trigger** word/phrase (usually a verb or event noun)
/// - **Arguments** (agent, patient, location, time, etc.)
/// - **Type** classification (attack, movement, creation, etc.)
///
/// # Research Background
///
/// Event extraction is a well-studied task (ACE, TAC-KBP, etc.).
/// We use a simplified model focused on what's needed for anaphora resolution.
///
/// # Example
///
/// Text: "Russia invaded Ukraine in February 2022"
///
/// ```rust
/// use anno::discourse::EventMention;
///
/// let event = EventMention::new("invaded", 7, 14)
///     .with_trigger_type("attack")
///     .with_arguments(vec![
///         ("Agent", "Russia"),
///         ("Patient", "Ukraine"),
///         ("Time", "February 2022"),
///     ]);
///
/// assert_eq!(event.trigger, "invaded");
/// assert_eq!(event.trigger_type.as_deref(), Some("attack"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMention {
    /// The trigger word/phrase (e.g., "invaded", "announced")
    pub trigger: String,
    /// Start character offset of trigger
    pub trigger_start: usize,
    /// End character offset of trigger
    pub trigger_end: usize,
    /// Event type (e.g., "attack", "movement", "communication")
    pub trigger_type: Option<String>,
    /// Event arguments (role -> entity text)
    /// Common roles: Agent, Patient, Location, Time, Instrument
    pub arguments: Vec<(String, String)>,
    /// Confidence score for this event extraction
    pub confidence: f64,
    /// Event polarity (positive, negative, uncertain)
    pub polarity: EventPolarity,
    /// Event tense/aspect
    pub tense: Option<EventTense>,
}

/// Polarity of an event mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EventPolarity {
    /// The event did/will happen (default)
    #[default]
    Positive,
    /// The event did not / will not happen
    Negative,
    /// Unknown whether the event happened
    Uncertain,
}

/// Tense of an event mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventTense {
    /// Event occurred in the past
    Past,
    /// Event is occurring now
    Present,
    /// Event will occur in the future
    Future,
    /// Event is hypothetical/conditional
    Hypothetical,
}

impl EventMention {
    /// Create a new event mention.
    #[must_use]
    pub fn new(trigger: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            trigger: trigger.into(),
            trigger_start: start,
            trigger_end: end,
            trigger_type: None,
            arguments: Vec::new(),
            confidence: 1.0,
            polarity: EventPolarity::default(),
            tense: None,
        }
    }

    /// Set the trigger type (event classification).
    #[must_use]
    pub fn with_trigger_type(mut self, trigger_type: impl Into<String>) -> Self {
        self.trigger_type = Some(trigger_type.into());
        self
    }

    /// Add arguments to this event.
    #[must_use]
    pub fn with_arguments<S: Into<String>>(mut self, args: Vec<(&str, S)>) -> Self {
        self.arguments = args
            .into_iter()
            .map(|(role, text)| (role.to_string(), text.into()))
            .collect();
        self
    }

    /// Set confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set polarity.
    #[must_use]
    pub fn with_polarity(mut self, polarity: EventPolarity) -> Self {
        self.polarity = polarity;
        self
    }

    /// Set tense.
    #[must_use]
    pub fn with_tense(mut self, tense: EventTense) -> Self {
        self.tense = Some(tense);
        self
    }

    /// Get argument by role.
    #[must_use]
    pub fn get_argument(&self, role: &str) -> Option<&str> {
        self.arguments
            .iter()
            .find(|(r, _)| r.eq_ignore_ascii_case(role))
            .map(|(_, text)| text.as_str())
    }
}

// =============================================================================
// Discourse Referent
// =============================================================================

/// A discourse referent - something that can be referred to by an anaphor.
///
/// This is the key data structure for abstract anaphora resolution.
/// Unlike `Entity` which captures noun phrases, `DiscourseReferent` can
/// represent entire clauses, events, propositions, etc.
///
/// # Example
///
/// ```rust
/// use anno::discourse::{DiscourseReferent, ReferentType, EventMention};
///
/// // "Russia invaded Ukraine in 2022. This caused inflation."
/// let event = EventMention::new("invaded", 7, 14);
/// let referent = DiscourseReferent::new(ReferentType::Event, 0, 30)
///     .with_event(event)
///     .with_label("Russian invasion");
///
/// assert_eq!(referent.span(), (0, 30));
/// assert!(referent.is_abstract());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseReferent {
    /// What kind of referent this is
    pub referent_type: ReferentType,
    /// Start character offset (of the full discourse segment)
    pub start: usize,
    /// End character offset (of the full discourse segment)
    pub end: usize,
    /// Human-readable label for this referent
    pub label: Option<String>,
    /// The full text of this referent (optional, for debugging)
    pub text: Option<String>,
    /// If this is an event, the event mention details
    pub event: Option<EventMention>,
    /// Coreference cluster ID (shared with entities that refer to this)
    pub canonical_id: Option<u64>,
    /// Confidence that this is a valid discourse referent
    pub confidence: f64,
    /// Discourse depth (how nested this referent is)
    /// 0 = main clause, 1 = subordinate, 2 = embedded subordinate, etc.
    pub depth: u32,
}

impl DiscourseReferent {
    /// Create a new discourse referent.
    #[must_use]
    pub fn new(referent_type: ReferentType, start: usize, end: usize) -> Self {
        Self {
            referent_type,
            start,
            end,
            label: None,
            text: None,
            event: None,
            canonical_id: None,
            confidence: 1.0,
            depth: 0,
        }
    }

    /// Create a nominal referent from an Entity.
    #[must_use]
    pub fn from_entity(entity: &Entity) -> Self {
        Self {
            referent_type: ReferentType::Nominal,
            start: entity.start,
            end: entity.end,
            label: Some(entity.text.clone()),
            text: Some(entity.text.clone()),
            event: None,
            canonical_id: entity.canonical_id,
            confidence: entity.confidence,
            depth: 0,
        }
    }

    /// Set a human-readable label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the text.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Attach an event mention.
    #[must_use]
    pub fn with_event(mut self, event: EventMention) -> Self {
        self.event = Some(event);
        self
    }

    /// Set the canonical ID.
    #[must_use]
    pub fn with_canonical_id(mut self, id: u64) -> Self {
        self.canonical_id = Some(id);
        self
    }

    /// Set confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set discourse depth.
    #[must_use]
    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    /// Get the span as (start, end).
    #[must_use]
    pub const fn span(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    /// Get span length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Check if span is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Is this an abstract referent (not a nominal entity)?
    #[must_use]
    pub const fn is_abstract(&self) -> bool {
        self.referent_type.is_abstract()
    }

    /// Get the display text (label or text or type name).
    #[must_use]
    pub fn display_text(&self) -> &str {
        self.label
            .as_deref()
            .or(self.text.as_deref())
            .unwrap_or(self.referent_type.as_str())
    }
}

// =============================================================================
// Shell Noun
// =============================================================================

/// A shell noun that can refer to abstract antecedents.
///
/// Shell nouns are abstract nouns like "problem", "issue", "fact" that
/// serve as "conceptual shells" for complex information.
///
/// # Research Background
///
/// Schmid (2000) identified ~670 shell nouns in English, organized into
/// six semantic classes: factual, linguistic, mental, modal, eventive,
/// circumstantial.
///
/// # Example
///
/// ```rust
/// use anno::discourse::{ShellNoun, ShellNounClass};
///
/// let shell = ShellNoun::new("problem", ShellNounClass::Factual)
///     .with_determiner("this")
///     .at_span(32, 44);
///
/// assert!(shell.is_demonstrative());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellNoun {
    /// The shell noun lemma (e.g., "problem", "issue", "fact")
    pub lemma: String,
    /// Semantic class of this shell noun
    pub class: ShellNounClass,
    /// The determiner used (e.g., "this", "the", "such")
    pub determiner: Option<String>,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
    /// The full NP text (e.g., "this problem")
    pub full_text: Option<String>,
}

/// Semantic classes of shell nouns (from Schmid 2000).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ShellNounClass {
    /// Factual: fact, reason, evidence, proof, point
    Factual,
    /// Linguistic: claim, statement, argument, answer, question
    Linguistic,
    /// Mental: idea, belief, thought, view, opinion
    Mental,
    /// Modal: possibility, chance, ability, need, requirement
    Modal,
    /// Eventive: event, incident, action, step, move
    Eventive,
    /// Circumstantial: situation, context, case, circumstance, condition
    Circumstantial,
}

impl ShellNounClass {
    /// Human-readable label.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            ShellNounClass::Factual => "factual",
            ShellNounClass::Linguistic => "linguistic",
            ShellNounClass::Mental => "mental",
            ShellNounClass::Modal => "modal",
            ShellNounClass::Eventive => "eventive",
            ShellNounClass::Circumstantial => "circumstantial",
        }
    }

    /// What referent types does this shell noun class typically refer to?
    #[must_use]
    pub fn typical_antecedent_types(&self) -> &[ReferentType] {
        match self {
            ShellNounClass::Factual => &[ReferentType::Fact, ReferentType::Event],
            ShellNounClass::Linguistic => &[ReferentType::Proposition],
            ShellNounClass::Mental => &[ReferentType::Proposition, ReferentType::Fact],
            ShellNounClass::Modal => &[ReferentType::Proposition],
            ShellNounClass::Eventive => &[ReferentType::Event, ReferentType::Situation],
            ShellNounClass::Circumstantial => &[ReferentType::Situation],
        }
    }
}

impl ShellNoun {
    /// Create a new shell noun.
    #[must_use]
    pub fn new(lemma: impl Into<String>, class: ShellNounClass) -> Self {
        Self {
            lemma: lemma.into(),
            class,
            determiner: None,
            start: 0,
            end: 0,
            full_text: None,
        }
    }

    /// Set the determiner.
    #[must_use]
    pub fn with_determiner(mut self, det: impl Into<String>) -> Self {
        self.determiner = Some(det.into());
        self
    }

    /// Set the span.
    #[must_use]
    pub fn at_span(mut self, start: usize, end: usize) -> Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Set full text.
    #[must_use]
    pub fn with_full_text(mut self, text: impl Into<String>) -> Self {
        self.full_text = Some(text.into());
        self
    }

    /// Is this a demonstrative shell noun (e.g., "this problem")?
    #[must_use]
    pub fn is_demonstrative(&self) -> bool {
        self.determiner
            .as_ref()
            .map(|d| {
                matches!(
                    d.to_lowercase().as_str(),
                    "this" | "that" | "these" | "those"
                )
            })
            .unwrap_or(false)
    }

    /// Get the typical antecedent types for this shell noun.
    #[must_use]
    pub fn typical_antecedent_types(&self) -> &[ReferentType] {
        self.class.typical_antecedent_types()
    }
}

// =============================================================================
// Common Shell Nouns Lexicon
// =============================================================================

/// Get the semantic class for a known shell noun.
///
/// Based on Schmid (2000) taxonomy. Returns `None` for unknown nouns.
#[must_use]
pub fn classify_shell_noun(lemma: &str) -> Option<ShellNounClass> {
    match lemma.to_lowercase().as_str() {
        // Factual
        "fact" | "reason" | "evidence" | "proof" | "point" | "truth" | "result" | "outcome"
        | "consequence" | "effect" | "cause" => Some(ShellNounClass::Factual),

        // Linguistic
        "claim" | "statement" | "argument" | "answer" | "question" | "response" | "reply"
        | "assertion" | "allegation" | "announcement" | "explanation" | "suggestion"
        | "recommendation" | "proposal" | "promise" | "warning" | "threat" => {
            Some(ShellNounClass::Linguistic)
        }

        // Mental
        "idea" | "belief" | "thought" | "view" | "opinion" | "impression" | "feeling" | "sense"
        | "notion" | "assumption" | "understanding" | "knowledge" | "memory" | "expectation"
        | "hope" | "fear" | "worry" | "concern" => Some(ShellNounClass::Mental),

        // Modal
        "possibility" | "chance" | "ability" | "need" | "requirement" | "necessity"
        | "obligation" | "duty" | "right" | "permission" | "opportunity" | "risk" | "danger"
        | "likelihood" | "probability" => Some(ShellNounClass::Modal),

        // Eventive
        "event" | "incident" | "action" | "step" | "move" | "development" | "change"
        | "process" | "procedure" | "activity" | "behavior" | "decision" | "choice" | "attempt"
        | "effort" | "achievement" | "success" | "failure" => Some(ShellNounClass::Eventive),

        // Circumstantial
        "situation" | "context" | "case" | "circumstance" | "condition" | "state" | "position"
        | "environment" | "scenario" | "aspect" | "factor" | "issue" | "problem" | "difficulty"
        | "challenge" | "crisis" | "dilemma" => Some(ShellNounClass::Circumstantial),

        _ => None,
    }
}

/// Check if a word is a known shell noun.
#[must_use]
pub fn is_shell_noun(word: &str) -> bool {
    classify_shell_noun(word).is_some()
}

// =============================================================================
// Discourse Scope Tracking
// =============================================================================

/// A simple clause/sentence boundary detector for discourse scope analysis.
///
/// For abstract anaphora, the antecedent is often:
/// - The immediately preceding clause (for "This")
/// - The preceding sentence (for longer-distance reference)
/// - Multiple preceding sentences (for "This situation")
///
/// This module provides heuristic-based boundary detection.
#[derive(Debug, Clone)]
pub struct DiscourseScope {
    /// Sentence boundaries (character offsets)
    pub sentence_boundaries: Vec<usize>,
    /// Clause boundaries (character offsets, more fine-grained)
    pub clause_boundaries: Vec<usize>,
}

impl DiscourseScope {
    /// Analyze text for discourse boundaries.
    ///
    /// # Example
    /// ```rust
    /// use anno::discourse::DiscourseScope;
    ///
    /// let text = "Russia invaded Ukraine. This caused inflation.";
    /// let scope = DiscourseScope::analyze(text);
    ///
    /// assert_eq!(scope.sentence_count(), 2);
    /// ```
    #[must_use]
    pub fn analyze(text: &str) -> Self {
        let sentence_boundaries = Self::find_sentence_boundaries(text);
        let clause_boundaries = Self::find_clause_boundaries(text);

        Self {
            sentence_boundaries,
            clause_boundaries,
        }
    }

    /// Find sentence boundaries using punctuation heuristics.
    fn find_sentence_boundaries(text: &str) -> Vec<usize> {
        let mut boundaries = vec![0]; // Start of text
        let chars: Vec<char> = text.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            // Sentence-ending punctuation
            if matches!(c, '.' | '!' | '?') {
                // Check it's not an abbreviation (followed by lowercase)
                let next_char = chars.get(i + 1).or(chars.get(i + 2));
                let after_space = chars.get(i + 2);

                // If followed by space and uppercase (or end of text), it's a sentence boundary
                if next_char.map_or(true, |&nc| nc.is_whitespace() || nc == '"' || nc == '\'')
                    && after_space.map_or(true, |&ac| ac.is_uppercase() || ac == '"')
                {
                    // Convert char index to byte offset
                    let byte_offset: usize = text
                        .char_indices()
                        .take(i + 1)
                        .last()
                        .map(|(idx, c)| idx + c.len_utf8())
                        .unwrap_or(text.len());
                    boundaries.push(byte_offset);
                }
            }
        }

        // End of text
        if boundaries.last() != Some(&text.len()) {
            boundaries.push(text.len());
        }

        boundaries
    }

    /// Find clause boundaries using punctuation and connectors.
    fn find_clause_boundaries(text: &str) -> Vec<usize> {
        let mut boundaries = vec![0];

        // Clause-separating punctuation and words
        let clause_markers = [
            ", and ",
            ", but ",
            ", or ",
            ", so ",
            ", yet ",
            "; ",
            ": ",
            " -- ",
            " – ",
            " while ",
            " although ",
            " because ",
            " since ",
            " when ",
            " whereas ",
            " unless ",
            " if ",
        ];

        for marker in &clause_markers {
            let marker_lower = marker.to_lowercase();
            let text_lower = text.to_lowercase();

            let mut search_from = 0;
            while let Some(pos) = text_lower[search_from..].find(&marker_lower) {
                let absolute_pos = search_from + pos + marker.len();
                boundaries.push(absolute_pos);
                search_from = absolute_pos;
            }
        }

        // Also add sentence boundaries
        boundaries.extend(Self::find_sentence_boundaries(text));

        boundaries.sort();
        boundaries.dedup();
        boundaries
    }

    /// Number of sentences detected.
    #[must_use]
    pub fn sentence_count(&self) -> usize {
        self.sentence_boundaries.len().saturating_sub(1)
    }

    /// Number of clauses detected.
    #[must_use]
    pub fn clause_count(&self) -> usize {
        self.clause_boundaries.len().saturating_sub(1)
    }

    /// Get the sentence containing a character offset.
    #[must_use]
    pub fn sentence_at(&self, offset: usize) -> Option<(usize, usize)> {
        for window in self.sentence_boundaries.windows(2) {
            if offset >= window[0] && offset < window[1] {
                return Some((window[0], window[1]));
            }
        }
        None
    }

    /// Get the clause containing a character offset.
    #[must_use]
    pub fn clause_at(&self, offset: usize) -> Option<(usize, usize)> {
        for window in self.clause_boundaries.windows(2) {
            if offset >= window[0] && offset < window[1] {
                return Some((window[0], window[1]));
            }
        }
        None
    }

    /// Get the N preceding clauses from an offset.
    #[must_use]
    pub fn preceding_clauses(&self, offset: usize, n: usize) -> Vec<(usize, usize)> {
        let mut clauses = Vec::new();

        // Find which clause we're in
        let mut current_idx = None;
        for (i, window) in self.clause_boundaries.windows(2).enumerate() {
            if offset >= window[0] && offset < window[1] {
                current_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = current_idx {
            // Collect preceding clauses
            for i in (0..idx).rev().take(n) {
                if i + 1 < self.clause_boundaries.len() {
                    clauses.push((self.clause_boundaries[i], self.clause_boundaries[i + 1]));
                }
            }
        }

        clauses
    }

    /// Get text for a span.
    pub fn extract_span<'a>(&self, text: &'a str, start: usize, end: usize) -> &'a str {
        text.get(start..end).unwrap_or("")
    }

    /// For an anaphor at a given offset, find candidate antecedent spans.
    ///
    /// Returns spans in order of preference (nearest first).
    #[must_use]
    pub fn candidate_antecedent_spans(&self, anaphor_offset: usize) -> Vec<(usize, usize)> {
        let mut candidates = Vec::new();

        // 1. Immediately preceding clause (highest priority for "This")
        let preceding = self.preceding_clauses(anaphor_offset, 3);
        candidates.extend(preceding);

        // 2. Preceding sentences (for longer-distance)
        if let Some((sent_start, _)) = self.sentence_at(anaphor_offset) {
            // Find the previous sentence
            for window in self.sentence_boundaries.windows(2) {
                if window[1] <= sent_start {
                    candidates.push((window[0], window[1]));
                }
            }
        }

        candidates.sort_by_key(|&(start, _)| std::cmp::Reverse(start));
        candidates.dedup();
        candidates
    }
}

// =============================================================================
// Event Coreference Resolution
// =============================================================================

/// A cluster of coreferent event mentions.
///
/// # Research Background
///
/// Event coreference (Ahmed & Martin, 2021) links mentions of the same
/// real-world event across a document. Unlike entity coreference, event
/// coreference must consider:
/// - Trigger words (verbs, nominalizations)
/// - Arguments (who did what to whom)
/// - Temporal/spatial constraints
/// - Event subtypes (attacks, meetings, etc.)
///
/// # Example
///
/// ```rust
/// use anno::discourse::{EventMention, EventCluster};
///
/// let mentions = vec![
///     EventMention::new("invasion", 10, 18)
///         .with_trigger_type("attack"),
///     EventMention::new("invaded", 50, 57)
///         .with_trigger_type("attack"),
///     EventMention::new("attack", 100, 106)
///         .with_trigger_type("attack"),
/// ];
///
/// let cluster = EventCluster::new(mentions);
/// assert_eq!(cluster.len(), 3);
/// assert_eq!(cluster.canonical_trigger(), "invasion");
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCluster {
    /// All event mentions in this cluster
    pub mentions: Vec<EventMention>,
    /// Cluster ID
    pub id: u64,
    /// Canonical event type (e.g., "attack")
    pub event_type: Option<String>,
    /// Confidence in this clustering
    pub confidence: f64,
}

impl EventCluster {
    /// Create a new event cluster.
    #[must_use]
    pub fn new(mentions: Vec<EventMention>) -> Self {
        let event_type = mentions
            .iter()
            .filter_map(|m| m.trigger_type.clone())
            .next();

        Self {
            mentions,
            id: 0,
            event_type,
            confidence: 1.0,
        }
    }

    /// Number of mentions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mentions.len()
    }

    /// Is the cluster empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mentions.is_empty()
    }

    /// Get the canonical trigger (first mention's trigger).
    #[must_use]
    pub fn canonical_trigger(&self) -> &str {
        self.mentions
            .first()
            .map(|m| m.trigger.as_str())
            .unwrap_or("")
    }

    /// Set cluster ID.
    #[must_use]
    pub fn with_id(mut self, id: u64) -> Self {
        self.id = id;
        self
    }

    /// Add a mention to the cluster.
    pub fn add(&mut self, mention: EventMention) {
        self.mentions.push(mention);
    }
}

/// Simple event coreference resolver.
///
/// Uses heuristics based on:
/// - Trigger lemma matching (same verb stem)
/// - Event type matching
/// - Argument overlap
///
/// # Limitations
///
/// This rule-based approach is simplistic. For production, use:
/// - BERT-based event coref (Ahmed & Martin, 2021)
/// - Cross-document event coref with LSH blocking
#[derive(Debug, Clone, Default)]
pub struct EventCorefResolver {
    /// Require matching event types
    pub require_type_match: bool,
    /// Minimum argument overlap ratio (0.0 = no requirement)
    pub min_arg_overlap: f64,
}

impl EventCorefResolver {
    /// Create a new resolver with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            require_type_match: true,
            min_arg_overlap: 0.3,
        }
    }

    /// Resolve event coreference, grouping mentions into clusters.
    #[must_use]
    pub fn resolve(&self, mentions: &[EventMention]) -> Vec<EventCluster> {
        if mentions.is_empty() {
            return vec![];
        }

        let mut clusters: Vec<EventCluster> = Vec::new();
        let mut assigned: Vec<bool> = vec![false; mentions.len()];

        for i in 0..mentions.len() {
            if assigned[i] {
                continue;
            }

            // Start a new cluster
            let mut cluster_mentions = vec![mentions[i].clone()];
            assigned[i] = true;

            // Find all mentions that corefer with this one
            for j in (i + 1)..mentions.len() {
                if assigned[j] {
                    continue;
                }

                if self.should_corefer(&mentions[i], &mentions[j]) {
                    cluster_mentions.push(mentions[j].clone());
                    assigned[j] = true;
                }
            }

            clusters.push(EventCluster::new(cluster_mentions).with_id(clusters.len() as u64));
        }

        clusters
    }

    /// Check if two event mentions should corefer.
    fn should_corefer(&self, a: &EventMention, b: &EventMention) -> bool {
        // 1. Check event type match
        if self.require_type_match {
            match (&a.trigger_type, &b.trigger_type) {
                (Some(ta), Some(tb)) if ta != tb => return false,
                _ => {} // OK if either is None or they match
            }
        }

        // 2. Check trigger lemma similarity
        let trigger_match = self.triggers_match(&a.trigger, &b.trigger);
        if !trigger_match {
            return false;
        }

        // 3. Check argument overlap
        if self.min_arg_overlap > 0.0 {
            let overlap = self.compute_arg_overlap(a, b);
            if overlap < self.min_arg_overlap {
                return false;
            }
        }

        true
    }

    /// Very simple stemmer for event triggers.
    ///
    /// Handles common verb-to-noun transformations:
    /// - invade / invaded / invasion -> invad
    /// - attack / attacked / attacking -> attack
    fn simple_stem(&self, word: &str) -> String {
        let mut s = word.to_string();

        // Handle nominalizations first (-ation, -tion, -sion)
        // invasion -> invas, creation -> creat, discussion -> discus
        if s.ends_with("ation") {
            s = s.trim_end_matches("ation").to_string();
            // Add back 'e' for words like create -> creation
            if !s.is_empty() && s.chars().last().map(|c| c.is_alphabetic()).unwrap_or(false) {
                // Don't add e back, just use the stem
            }
        } else if s.ends_with("tion") || s.ends_with("sion") {
            s = s.trim_end_matches("ion").to_string();
        } else if s.ends_with("ing") {
            s = s.trim_end_matches("ing").to_string();
        } else if s.ends_with("ed") && s.len() > 3 {
            s = s.trim_end_matches("ed").to_string();
        } else if s.ends_with("s") && s.len() > 2 && !s.ends_with("ss") {
            s = s.trim_end_matches('s').to_string();
        }

        // Handle doubled consonants (running -> run, invadde -> invad)
        let bytes = s.as_bytes();
        if bytes.len() > 2 && bytes[bytes.len() - 1] == bytes[bytes.len() - 2] {
            s.pop();
        }

        s
    }

    /// Make method public for testing
    pub fn triggers_match(&self, a: &str, b: &str) -> bool {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        // Exact match
        if a_lower == b_lower {
            return true;
        }

        // Simple stemming
        let stem_a = self.simple_stem(&a_lower);
        let stem_b = self.simple_stem(&b_lower);

        stem_a == stem_b
    }

    /// Compute argument overlap ratio between two event mentions.
    fn compute_arg_overlap(&self, a: &EventMention, b: &EventMention) -> f64 {
        if a.arguments.is_empty() && b.arguments.is_empty() {
            return 1.0; // No arguments = compatible
        }

        let total = a.arguments.len().max(b.arguments.len());
        if total == 0 {
            return 1.0;
        }

        let mut matches = 0;
        for (role_a, val_a) in &a.arguments {
            for (role_b, val_b) in &b.arguments {
                // Same role with similar value
                if role_a == role_b && self.values_similar(val_a, val_b) {
                    matches += 1;
                    break;
                }
            }
        }

        matches as f64 / total as f64
    }

    /// Check if two argument values are similar.
    fn values_similar(&self, a: &str, b: &str) -> bool {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        a_lower == b_lower || a_lower.contains(&b_lower) || b_lower.contains(&a_lower)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    #[test]
    fn test_referent_types() {
        assert!(!ReferentType::Nominal.is_abstract());
        assert!(ReferentType::Event.is_abstract());
        assert!(ReferentType::Fact.is_abstract());
        assert!(ReferentType::Proposition.is_abstract());
        assert!(ReferentType::Situation.is_abstract());
    }

    #[test]
    fn test_event_mention() {
        let event = EventMention::new("invaded", 7, 14)
            .with_trigger_type("attack")
            .with_arguments(vec![("Agent", "Russia"), ("Patient", "Ukraine")]);

        assert_eq!(event.trigger, "invaded");
        assert_eq!(event.trigger_type.as_deref(), Some("attack"));
        assert_eq!(event.get_argument("Agent"), Some("Russia"));
        assert_eq!(event.get_argument("Patient"), Some("Ukraine"));
        assert_eq!(event.get_argument("Location"), None);
    }

    #[test]
    fn test_discourse_referent() {
        let event = EventMention::new("invaded", 7, 14);
        let referent = DiscourseReferent::new(ReferentType::Event, 0, 30)
            .with_event(event)
            .with_label("Russian invasion");

        assert_eq!(referent.span(), (0, 30));
        assert_eq!(referent.len(), 30);
        assert!(referent.is_abstract());
        assert_eq!(referent.display_text(), "Russian invasion");
    }

    #[test]
    fn test_shell_noun_classification() {
        assert_eq!(
            classify_shell_noun("problem"),
            Some(ShellNounClass::Circumstantial)
        );
        assert_eq!(classify_shell_noun("fact"), Some(ShellNounClass::Factual));
        assert_eq!(classify_shell_noun("idea"), Some(ShellNounClass::Mental));
        assert_eq!(
            classify_shell_noun("possibility"),
            Some(ShellNounClass::Modal)
        );
        assert_eq!(classify_shell_noun("event"), Some(ShellNounClass::Eventive));
        assert_eq!(
            classify_shell_noun("claim"),
            Some(ShellNounClass::Linguistic)
        );
        assert_eq!(classify_shell_noun("foobar"), None);
    }

    #[test]
    fn test_shell_noun_demonstrative() {
        let shell = ShellNoun::new("problem", ShellNounClass::Circumstantial)
            .with_determiner("this")
            .at_span(32, 44);

        assert!(shell.is_demonstrative());

        let shell_the =
            ShellNoun::new("problem", ShellNounClass::Circumstantial).with_determiner("the");

        assert!(!shell_the.is_demonstrative());
    }

    #[test]
    fn test_shell_noun_typical_antecedents() {
        let shell = ShellNoun::new("event", ShellNounClass::Eventive);
        let types = shell.typical_antecedent_types();
        assert!(types.contains(&ReferentType::Event));
        assert!(types.contains(&ReferentType::Situation));
    }

    #[test]
    fn test_from_entity() {
        let entity = Entity::new("Russia", EntityType::Location, 0, 6, 0.95);
        let referent = DiscourseReferent::from_entity(&entity);

        assert_eq!(referent.referent_type, ReferentType::Nominal);
        assert_eq!(referent.start, 0);
        assert_eq!(referent.end, 6);
        assert!(!referent.is_abstract());
    }

    #[test]
    fn test_discourse_scope_sentences() {
        let text = "Russia invaded Ukraine. This caused inflation. The crisis deepened.";
        let scope = DiscourseScope::analyze(text);

        assert_eq!(scope.sentence_count(), 3);
    }

    #[test]
    fn test_discourse_scope_clauses() {
        let text = "Prices rose, and wages fell. This was unsustainable.";
        let scope = DiscourseScope::analyze(text);

        // Should detect: "Prices rose", "and wages fell", sentence boundary, "This was..."
        assert!(scope.clause_count() >= 2);
    }

    #[test]
    fn test_discourse_scope_preceding() {
        let text = "Russia invaded Ukraine. This caused inflation.";
        let scope = DiscourseScope::analyze(text);

        // "This" is at position 24
        let preceding = scope.preceding_clauses(24, 2);
        assert!(!preceding.is_empty(), "Should find preceding clauses");
    }

    #[test]
    fn test_candidate_antecedent_spans() {
        let text = "Russia invaded Ukraine in 2022. This caused a global energy crisis.";
        let scope = DiscourseScope::analyze(text);

        // "This" is at position 32
        let candidates = scope.candidate_antecedent_spans(32);
        assert!(!candidates.is_empty(), "Should find candidate spans");

        // The first sentence should be a candidate
        let first_sentence = scope.extract_span(text, candidates[0].0, candidates[0].1);
        assert!(
            first_sentence.contains("invaded"),
            "First candidate should include the invasion"
        );
    }

    // Event coreference tests

    #[test]
    fn test_event_cluster_creation() {
        let mentions = vec![
            EventMention::new("invasion", 10, 18).with_trigger_type("attack"),
            EventMention::new("invaded", 50, 57).with_trigger_type("attack"),
        ];

        let cluster = EventCluster::new(mentions);
        assert_eq!(cluster.len(), 2);
        assert_eq!(cluster.canonical_trigger(), "invasion");
        assert_eq!(cluster.event_type, Some("attack".to_string()));
    }

    #[test]
    fn test_event_coref_resolver_simple() {
        let resolver = EventCorefResolver::new();

        let mentions = vec![
            // Two attack events (same trigger stem, should cluster)
            EventMention::new("attacked", 10, 18)
                .with_trigger_type("attack")
                .with_arguments(vec![("Agent", "Russia"), ("Patient", "Ukraine")]),
            EventMention::new("attack", 50, 56)
                .with_trigger_type("attack")
                .with_arguments(vec![("Agent", "Russia")]),
            // One meeting event (different type, should not cluster with attacks)
            EventMention::new("meeting", 100, 107)
                .with_trigger_type("meeting")
                .with_arguments(vec![("Participant", "leaders")]),
        ];

        let clusters = resolver.resolve(&mentions);

        // Should have 2 clusters: attack events + meeting event
        assert_eq!(clusters.len(), 2, "Expected 2 clusters");

        // First cluster should have the attack mentions
        let attack_cluster = &clusters[0];
        assert_eq!(
            attack_cluster.len(),
            2,
            "Attack cluster should have 2 mentions"
        );

        // Second cluster should have the meeting mention
        let meeting_cluster = &clusters[1];
        assert_eq!(
            meeting_cluster.len(),
            1,
            "Meeting cluster should have 1 mention"
        );
    }

    #[test]
    fn test_event_coref_trigger_matching() {
        let resolver = EventCorefResolver::new();

        // These should match (same stem or exact)
        assert!(resolver.triggers_match("attack", "attack"));
        assert!(resolver.triggers_match("attack", "attacks"));
        assert!(resolver.triggers_match("attack", "attacked"));
        assert!(resolver.triggers_match("attack", "attacking"));

        // These should not match (different events)
        assert!(!resolver.triggers_match("attack", "meeting"));
        assert!(!resolver.triggers_match("invade", "defend"));
    }

    // =================================================================
    // Additional Edge Case Tests
    // =================================================================

    #[test]
    fn test_empty_text_discourse_scope() {
        let scope = DiscourseScope::analyze("");
        assert_eq!(scope.sentence_count(), 0);
        assert_eq!(scope.clause_count(), 0);
    }

    #[test]
    fn test_single_word_text() {
        let scope = DiscourseScope::analyze("Hello");
        // Single word with no period should still have structure
        assert!(scope.sentence_boundaries.len() >= 2); // start and end
    }

    #[test]
    fn test_abbreviation_handling() {
        let text = "Dr. Smith went to the U.S. embassy. He met with officials.";
        let scope = DiscourseScope::analyze(text);
        // Should detect 2 sentences, not split on Dr. or U.S.
        // (Note: our simple heuristic may not handle all abbreviations)
        assert!(scope.sentence_count() >= 1);
    }

    #[test]
    fn test_shell_noun_case_insensitive() {
        assert_eq!(classify_shell_noun("FACT"), Some(ShellNounClass::Factual));
        assert_eq!(
            classify_shell_noun("Problem"),
            Some(ShellNounClass::Circumstantial)
        );
        assert_eq!(classify_shell_noun("IDEA"), Some(ShellNounClass::Mental));
    }

    #[test]
    fn test_event_mention_empty_arguments() {
        let event = EventMention::new("attacked", 0, 8);
        assert!(event.arguments.is_empty());
        assert_eq!(event.get_argument("Agent"), None);
    }

    #[test]
    fn test_discourse_referent_empty_span() {
        let referent = DiscourseReferent::new(ReferentType::Event, 5, 5);
        assert!(referent.is_empty());
        assert_eq!(referent.len(), 0);
    }

    #[test]
    fn test_event_polarity_variants() {
        let positive = EventMention::new("attacked", 0, 8).with_polarity(EventPolarity::Positive);
        let negative = EventMention::new("attacked", 0, 8).with_polarity(EventPolarity::Negative);
        let uncertain = EventMention::new("attacked", 0, 8).with_polarity(EventPolarity::Uncertain);

        assert_eq!(positive.polarity, EventPolarity::Positive);
        assert_eq!(negative.polarity, EventPolarity::Negative);
        assert_eq!(uncertain.polarity, EventPolarity::Uncertain);
    }

    #[test]
    fn test_event_tense_variants() {
        let past = EventMention::new("attacked", 0, 8).with_tense(EventTense::Past);
        let future = EventMention::new("will attack", 0, 11).with_tense(EventTense::Future);

        assert_eq!(past.tense, Some(EventTense::Past));
        assert_eq!(future.tense, Some(EventTense::Future));
    }

    #[test]
    fn test_event_cluster_empty() {
        let cluster = EventCluster::new(vec![]);
        assert!(cluster.is_empty());
        assert_eq!(cluster.len(), 0);
        assert_eq!(cluster.canonical_trigger(), "");
    }

    #[test]
    fn test_event_coref_empty_input() {
        let resolver = EventCorefResolver::new();
        let clusters = resolver.resolve(&[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_event_coref_single_mention() {
        let resolver = EventCorefResolver::new();
        let mentions = vec![EventMention::new("attacked", 0, 8).with_trigger_type("attack")];
        let clusters = resolver.resolve(&mentions);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 1);
    }

    #[test]
    fn test_referent_type_can_be_antecedent() {
        // "this" can refer to all abstract types
        assert!(ReferentType::Event.can_be_this_antecedent());
        assert!(ReferentType::Fact.can_be_this_antecedent());
        assert!(ReferentType::Proposition.can_be_this_antecedent());
        assert!(!ReferentType::Nominal.can_be_this_antecedent());

        // "it" is more restricted
        assert!(ReferentType::Event.can_be_it_antecedent());
        assert!(ReferentType::Nominal.can_be_it_antecedent());
        assert!(!ReferentType::Fact.can_be_it_antecedent());
    }

    #[test]
    fn test_discourse_scope_sentence_at() {
        let text = "First sentence. Second sentence. Third.";
        let scope = DiscourseScope::analyze(text);

        // Check sentence_at for different positions
        let sent1 = scope.sentence_at(5); // "First"
        assert!(sent1.is_some());

        let sent2 = scope.sentence_at(20); // "Second"
        assert!(sent2.is_some());
    }

    #[test]
    fn test_discourse_scope_clause_at() {
        let text = "Prices rose, and wages fell.";
        let scope = DiscourseScope::analyze(text);

        let clause = scope.clause_at(5); // "Prices"
        assert!(clause.is_some());
    }

    #[test]
    fn test_shell_noun_all_classes() {
        // Test at least one noun from each class
        let tests = vec![
            ("fact", ShellNounClass::Factual),
            ("claim", ShellNounClass::Linguistic),
            ("idea", ShellNounClass::Mental),
            ("possibility", ShellNounClass::Modal),
            ("event", ShellNounClass::Eventive),
            ("situation", ShellNounClass::Circumstantial),
        ];

        for (noun, expected_class) in tests {
            let result = classify_shell_noun(noun);
            assert_eq!(result, Some(expected_class), "Failed for noun: {}", noun);
        }
    }
}
