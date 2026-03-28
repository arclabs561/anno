//! Centering Theory Implementation.
//!
//! # The Problem: Pronoun Resolution is Hard
//!
//! Consider: "John called Bill. He was angry."
//!
//! Who is "he"? Both John and Bill are grammatically valid. Humans resolve this
//! effortlessly, but the cues they use are subtle:
//!
//! - **Grammatical role**: Subjects are more likely to be re-mentioned
//! - **Recency**: Recent entities are more salient
//! - **Coherence**: Texts that maintain focus are easier to process
//!
//! Simple heuristics (nearest antecedent, subject preference) work sometimes,
//! but fail on texts with multiple entities or topic shifts. Centering theory
//! provides a principled framework for tracking *what the discourse is about*.
//!
//! # Why Centering Theory?
//!
//! Centering theory (Grosz, Joshi, Weinstein 1995) captures the observation that
//! coherent discourse maintains a consistent "center of attention." When you read:
//!
//! > "John went to the store. He bought milk. He drove home."
//!
//! ...you effortlessly track that all three sentences are about John. The theory
//! formalizes this intuition through forward-looking centers (entities mentioned)
//! and backward-looking centers (what the utterance is "about").
//!
//! **Key insight**: The most salient entity in one utterance is likely to be
//! referenced in the next. Violations of this expectation (topic shifts) create
//! processing difficulty—and are marked by explicit referring expressions.
//!
//! # Core Concepts
//!
//! For each utterance U_n:
//!
//! - **Cf(U_n)**: Forward-looking centers — entities mentioned, ranked by salience
//! - **Cb(U_n)**: Backward-looking center — highest-ranked entity from Cf(U_{n-1})
//!   that is realized in U_n
//! - **Cp(U_n)**: Preferred center — highest-ranked member of Cf(U_n)
//!
//! The Cb answers: "What is this utterance about?"
//! The Cp predicts: "What will the next utterance likely be about?"
//!
//! # Transition Types
//!
//! | Transition | Cb(U_n) = Cb(U_{n-1}) | Cb(U_n) = Cp(U_n) |
//! |------------|----------------------|-------------------|
//! | CONTINUE   | Yes                  | Yes               |
//! | RETAIN     | Yes                  | No                |
//! | SMOOTH-SHIFT | No                 | Yes               |
//! | ROUGH-SHIFT | No                  | No                |
//!
//! **Preference ordering**: CONTINUE > RETAIN > SMOOTH-SHIFT > ROUGH-SHIFT
//!
//! This ordering predicts that CONTINUE transitions are easiest to process—
//! the discourse maintains focus on the same entity, which remains most salient.
//! ROUGH-SHIFTs are hardest: the topic changes to an entity that isn't even
//! the most prominent in the current utterance.
//!
//! # Centering + Recency: Modern Coreference
//!
//! Jiang et al. (2022) found vanilla CT provides little gain for neural coref,
//! but CT + recency captures more coreference signal. The combination works
//! because:
//!
//! 1. **CT captures structural salience** (grammatical role, information status)
//! 2. **Recency captures temporal decay** (recently mentioned = more accessible)
//!
//! This module implements both classical CT and the recency-augmented variant
//! via [`CenteringConfig::recency_decay`].
//!
//! # Connection to Israel (1994)
//!
//! Israel's critique of dynamic semantics noted that the "extent" of a discourse
//! referent is not statically determinable—you can't know at introduction time
//! how long an entity will remain relevant. Centering theory operationalizes this:
//! the extent of an entity is precisely how long it can serve as Cb. When it stops
//! appearing in Cf lists, it's effectively "garbage collected" from the discourse.
//!
//! # Example
//!
//! ```rust
//! use anno::discourse::centering::{
//!     CenteringState, ForwardCenter, track_centers, CenteringConfig
//! };
//!
//! let utterances = vec![
//!     vec![
//!         ForwardCenter::new(1, "John", 1.0),
//!         ForwardCenter::new(2, "Mary", 0.8),
//!     ],
//!     vec![
//!         ForwardCenter::new(1, "he", 0.9),  // Refers to John
//!         ForwardCenter::new(3, "the book", 0.7),
//!     ],
//! ];
//!
//! let config = CenteringConfig::default();
//! let states = track_centers(&utterances, &config);
//!
//! // First utterance has no Cb (discourse-initial)
//! assert!(states[0].cb.is_none());
//!
//! // Second utterance: Cb = John (highest Cf from U1 realized in U2)
//! assert_eq!(states[1].cb, Some(1));
//! ```
//!
//! # References
//!
//! - Grosz, Joshi, Weinstein (1995): "Centering: A Framework for Modeling
//!   the Local Coherence of Discourse"
//! - Brennan, Friedman, Pollard (1987): "A Centering Approach to Pronouns"
//! - Strube (1998): "Never Look Back: An Alternative to Centering"
//! - Jiang et al. (2022): "Investigating the Role of Centering Theory in
//!   the Context of Neural Coreference Resolution Systems"

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Core Types
// =============================================================================

/// A forward-looking center (member of Cf).
///
/// Represents an entity mentioned in an utterance, ranked by salience.
/// The Cf list for each utterance is sorted by [`effective_salience`](Self::effective_salience),
/// which combines the base salience with grammatical role and information status.
///
/// # Ranking Factors
///
/// Centering theory ranks Cf members by:
/// 1. **Grammatical role**: Subject > Object > Oblique (see [`GrammaticalRole`])
/// 2. **Information status**: Evoked > Unused > Inferrable > New (see [`InformationStatus`])
/// 3. **Base salience**: From external NER or mention detection scores
///
/// # Example
///
/// ```rust
/// use anno::discourse::centering::{ForwardCenter, GrammaticalRole, InformationStatus};
///
/// let fc = ForwardCenter::new(1, "John", 0.9)
///     .with_role(GrammaticalRole::Subject)
///     .with_info_status(InformationStatus::Evoked);
///
/// // Effective salience combines all factors
/// assert!(fc.effective_salience() > 0.9);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForwardCenter {
    /// Entity/cluster ID (links to coreference clusters).
    pub entity_id: u64,
    /// Surface realization (the mention text, e.g., "John", "he", "the CEO").
    pub realization: String,
    /// Base salience score (higher = more salient). Typically from NER confidence.
    pub salience: f64,
    /// Grammatical role in the utterance (affects Cf ranking).
    pub grammatical_role: Option<GrammaticalRole>,
    /// Information status (hearer-old vs hearer-new, affects Cf ranking).
    pub info_status: InformationStatus,
    /// Character offset in utterance (for span alignment).
    pub offset: usize,
}

/// Grammatical role of a mention in its clause.
///
/// Centering theory ranks Cf members partly by grammatical role, with
/// subjects ranking highest. This reflects the observation that subjects
/// are typically the "topic" of a clause and more likely to be referred
/// to in subsequent discourse.
///
/// # Ranking
///
/// The standard centering hierarchy is:
///
/// ```text
/// SUBJECT > EXISTENTIAL > OBJECT > INDIRECT OBJECT > OBLIQUE > ADJUNCT
/// ```
///
/// This module simplifies to: Subject > DirectObject > IndirectObject > Oblique > Other
///
/// # Example
///
/// ```rust
/// use anno::discourse::centering::GrammaticalRole;
///
/// assert!(GrammaticalRole::Subject.salience_weight() >
///         GrammaticalRole::DirectObject.salience_weight());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GrammaticalRole {
    /// Subject of clause — highest salience.
    Subject,
    /// Direct object — second highest.
    DirectObject,
    /// Indirect object (dative).
    IndirectObject,
    /// Oblique (prepositional object, adjunct).
    Oblique,
    /// Other/unknown — lowest salience.
    #[default]
    Other,
}

impl GrammaticalRole {
    /// Get salience weight for this role.
    ///
    /// Based on centering theory's Cf ranking:
    /// SUBJECT > EXISTENTIAL > OBJECT > INDIRECT OBJECT > OBLIQUE > ADJUNCT
    #[must_use]
    pub const fn salience_weight(&self) -> f64 {
        match self {
            GrammaticalRole::Subject => 1.0,
            GrammaticalRole::DirectObject => 0.8,
            GrammaticalRole::IndirectObject => 0.6,
            GrammaticalRole::Oblique => 0.4,
            GrammaticalRole::Other => 0.3,
        }
    }
}

/// Information status of a discourse entity.
///
/// Following Prince (1981) "Toward a Taxonomy of Given-New Information"
/// and Strube's (1998) hearer-old/hearer-new distinction.
///
/// # Hearer-Old vs Hearer-New
///
/// The key distinction is whether the entity is already in the hearer's
/// discourse model:
///
/// - **Hearer-old**: [`Evoked`](Self::Evoked), [`Unused`](Self::Unused),
///   [`Inferrable`](Self::Inferrable) — the hearer can identify the referent
/// - **Hearer-new**: [`New`](Self::New) — introduces a new discourse entity
///
/// Hearer-old entities rank higher in the S-list because they are already
/// salient to the hearer.
///
/// # Example
///
/// ```rust
/// use anno::discourse::centering::InformationStatus;
///
/// // "the man" after "a man walked in" — evoked
/// assert!(InformationStatus::Evoked.is_hearer_old());
///
/// // "a dog" first mention — new
/// assert!(!InformationStatus::New.is_hearer_old());
///
/// // Evoked entities get salience boost
/// assert!(InformationStatus::Evoked.salience_boost() >
///         InformationStatus::New.salience_boost());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum InformationStatus {
    /// First mention, indefinite NP ("a man walked in").
    /// Introduces a new entity to the discourse model.
    New,
    /// First explicit mention but inferrable from context.
    /// E.g., "the door" after "John entered a room" — the room implies a door.
    Inferrable,
    /// Previously mentioned in discourse ("the man" referring back to "a man").
    /// Most common status for anaphoric expressions.
    #[default]
    Evoked,
    /// Known from world knowledge or situational context.
    /// E.g., "the sun", "the president" — identifiable without prior mention.
    Unused,
}

impl InformationStatus {
    /// Is this hearer-old? (Known or inferrable by hearer)
    #[must_use]
    pub const fn is_hearer_old(&self) -> bool {
        matches!(
            self,
            InformationStatus::Evoked | InformationStatus::Unused | InformationStatus::Inferrable
        )
    }

    /// Salience boost for hearer-old entities.
    ///
    /// Strube (1998): "hearer-old entities are ranked higher than hearer-new"
    #[must_use]
    pub const fn salience_boost(&self) -> f64 {
        match self {
            InformationStatus::Evoked => 0.3,
            InformationStatus::Unused => 0.2,
            InformationStatus::Inferrable => 0.1,
            InformationStatus::New => 0.0,
        }
    }
}

impl ForwardCenter {
    /// Create a new forward center.
    #[must_use]
    pub fn new(entity_id: u64, realization: impl Into<String>, salience: f64) -> Self {
        Self {
            entity_id,
            realization: realization.into(),
            salience,
            grammatical_role: None,
            info_status: InformationStatus::default(),
            offset: 0,
        }
    }

    /// Set grammatical role.
    #[must_use]
    pub fn with_role(mut self, role: GrammaticalRole) -> Self {
        self.grammatical_role = Some(role);
        self
    }

    /// Set information status.
    #[must_use]
    pub fn with_info_status(mut self, status: InformationStatus) -> Self {
        self.info_status = status;
        self
    }

    /// Set character offset.
    #[must_use]
    pub fn at_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Compute effective salience including role and info status.
    #[must_use]
    pub fn effective_salience(&self) -> f64 {
        let role_weight = self.grammatical_role.map_or(0.5, |r| r.salience_weight());

        self.salience * role_weight + self.info_status.salience_boost()
    }
}

// =============================================================================
// Transition Types
// =============================================================================

/// Centering transition type between successive utterances.
///
/// Transitions describe how the discourse focus shifts (or doesn't) between
/// utterances. The preference ordering predicts processing difficulty:
///
/// ```text
/// CONTINUE > RETAIN > SMOOTH-SHIFT > ROUGH-SHIFT
/// ```
///
/// Texts with more CONTINUE transitions are easier to process because they
/// maintain a consistent topic focus.
///
/// # Transition Rules
///
/// | Transition | Cb(U_n) = Cb(U_{n-1})? | Cb(U_n) = Cp(U_n)? |
/// |------------|------------------------|-------------------|
/// | CONTINUE   | Yes                    | Yes               |
/// | RETAIN     | Yes                    | No                |
/// | SMOOTH-SHIFT | No                   | Yes               |
/// | ROUGH-SHIFT | No                    | No                |
///
/// # Example
///
/// ```rust
/// use anno::discourse::centering::CenteringTransition;
///
/// // CONTINUE is most coherent
/// assert!(CenteringTransition::Continue.coherence_score() >
///         CenteringTransition::Retain.coherence_score());
///
/// // Check transition type
/// assert!(CenteringTransition::Continue.is_continuing());
/// assert!(CenteringTransition::SmoothShift.is_shifting());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum CenteringTransition {
    /// Cb(U_n) = Cb(U_{n-1}) AND Cb(U_n) = Cp(U_n).
    /// The discourse continues about the same entity, which remains most salient.
    /// This is the most coherent transition.
    Continue,

    /// Cb(U_n) = Cb(U_{n-1}) AND Cb(U_n) != Cp(U_n).
    /// Same topic, but another entity is now more salient—signals an upcoming shift.
    Retain,

    /// Cb(U_n) != Cb(U_{n-1}) AND Cb(U_n) = Cp(U_n).
    /// Topic shift to a new entity, which is established as most salient.
    SmoothShift,

    /// Cb(U_n) != Cb(U_{n-1}) AND Cb(U_n) != Cp(U_n).
    /// Topic shift, but the new topic isn't even the most salient entity.
    /// This is the least coherent transition.
    RoughShift,

    /// No Cb exists (discourse-initial or no entity from previous utterance realized).
    #[default]
    Null,
}

impl CenteringTransition {
    /// Coherence score for this transition (higher = more coherent).
    #[must_use]
    pub const fn coherence_score(&self) -> f64 {
        match self {
            CenteringTransition::Continue => 1.0,
            CenteringTransition::Retain => 0.75,
            CenteringTransition::SmoothShift => 0.5,
            CenteringTransition::RoughShift => 0.25,
            CenteringTransition::Null => 0.0,
        }
    }

    /// Is this a continuing transition (same Cb)?
    #[must_use]
    pub const fn is_continuing(&self) -> bool {
        matches!(
            self,
            CenteringTransition::Continue | CenteringTransition::Retain
        )
    }

    /// Is this a shifting transition (different Cb)?
    #[must_use]
    pub const fn is_shifting(&self) -> bool {
        matches!(
            self,
            CenteringTransition::SmoothShift | CenteringTransition::RoughShift
        )
    }

    /// Human-readable label.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            CenteringTransition::Continue => "CONTINUE",
            CenteringTransition::Retain => "RETAIN",
            CenteringTransition::SmoothShift => "SMOOTH-SHIFT",
            CenteringTransition::RoughShift => "ROUGH-SHIFT",
            CenteringTransition::Null => "NULL",
        }
    }
}

// =============================================================================
// Centering State
// =============================================================================

/// Centering state for a single utterance.
///
/// Captures the centering configuration at one point in the discourse:
/// - **Cf**: Forward-looking centers (entities mentioned, ranked by salience)
/// - **Cb**: Backward-looking center (what the utterance is "about")
/// - **Cp**: Preferred center (predicted focus for next utterance)
///
/// # Example
///
/// ```rust
/// use anno::discourse::centering::{CenteringState, ForwardCenter};
///
/// let state = CenteringState::new(0)
///     .with_cf(vec![
///         ForwardCenter::new(1, "John", 1.0),
///         ForwardCenter::new(2, "Mary", 0.8),
///     ]);
///
/// // Cp is automatically set to the highest-ranked Cf
/// assert_eq!(state.cp, Some(1));
///
/// // Check if an entity is mentioned
/// assert!(state.mentions(1));
/// assert!(!state.mentions(99));
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CenteringState {
    /// Index of this utterance in the discourse (0-based).
    pub utterance_idx: usize,

    /// Forward-looking centers, ranked by effective salience (highest first).
    /// These are the entities mentioned in this utterance that could become
    /// the Cb of the next utterance.
    pub cf: Vec<ForwardCenter>,

    /// Backward-looking center — the most salient entity from the previous
    /// utterance that is realized in this utterance. `None` for discourse-initial
    /// utterances or when no entity carries over.
    pub cb: Option<u64>,

    /// Preferred center — the highest-ranked member of Cf. This predicts
    /// what the next utterance is likely to be about.
    pub cp: Option<u64>,

    /// Transition from previous utterance
    pub transition: CenteringTransition,

    /// Recency-weighted salience scores (entity_id -> score)
    /// This implements CT+recency from Jiang et al. (2022)
    pub recency_scores: HashMap<u64, f64>,
}

impl CenteringState {
    /// Create a new centering state.
    #[must_use]
    pub fn new(utterance_idx: usize) -> Self {
        Self {
            utterance_idx,
            cf: Vec::new(),
            cb: None,
            cp: None,
            transition: CenteringTransition::Null,
            recency_scores: HashMap::new(),
        }
    }

    /// Set forward-looking centers.
    #[must_use]
    pub fn with_cf(mut self, cf: Vec<ForwardCenter>) -> Self {
        // Sort by effective salience, descending
        self.cf = cf;
        self.cf.sort_by(|a, b| {
            b.effective_salience()
                .partial_cmp(&a.effective_salience())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update Cp (preferred center)
        self.cp = self.cf.first().map(|fc| fc.entity_id);

        self
    }

    /// Get entity IDs mentioned in this utterance.
    #[must_use]
    pub fn mentioned_entities(&self) -> Vec<u64> {
        self.cf.iter().map(|fc| fc.entity_id).collect()
    }

    /// Check if an entity is mentioned in this utterance.
    #[must_use]
    pub fn mentions(&self, entity_id: u64) -> bool {
        self.cf.iter().any(|fc| fc.entity_id == entity_id)
    }

    /// Get the forward center for an entity.
    #[must_use]
    pub fn get_fc(&self, entity_id: u64) -> Option<&ForwardCenter> {
        self.cf.iter().find(|fc| fc.entity_id == entity_id)
    }

    /// Get coherence score for this state.
    #[must_use]
    pub fn coherence_score(&self) -> f64 {
        self.transition.coherence_score()
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for centering computation.
#[derive(Debug, Clone)]
pub struct CenteringConfig {
    /// Decay rate for recency scoring (0-1, higher = faster decay)
    pub recency_decay: f64,

    /// Weight for grammatical role in Cf ranking
    pub role_weight: f64,

    /// Weight for information status in Cf ranking
    pub info_status_weight: f64,

    /// Whether to use recency-augmented centering (Jiang et al. 2022)
    pub use_recency: bool,

    /// Maximum number of utterances to look back for recency
    pub recency_window: usize,
}

impl Default for CenteringConfig {
    fn default() -> Self {
        Self {
            recency_decay: 0.5,
            role_weight: 1.0,
            info_status_weight: 1.0,
            use_recency: true, // CT+recency by default
            recency_window: 5,
        }
    }
}

// =============================================================================
// Centering Computation
// =============================================================================

/// Compute centering state for a sequence of utterances.
///
/// Each utterance is represented as a list of forward centers (entities mentioned).
///
/// # Example
///
/// ```rust
/// use anno::discourse::centering::{track_centers, ForwardCenter, CenteringConfig};
///
/// let utterances = vec![
///     vec![ForwardCenter::new(1, "John", 1.0)],
///     vec![ForwardCenter::new(1, "he", 0.9), ForwardCenter::new(2, "Mary", 0.8)],
/// ];
///
/// let states = track_centers(&utterances, &CenteringConfig::default());
/// assert_eq!(states.len(), 2);
/// ```
pub fn track_centers(
    utterances: &[Vec<ForwardCenter>],
    config: &CenteringConfig,
) -> Vec<CenteringState> {
    let mut states: Vec<CenteringState> = Vec::with_capacity(utterances.len());

    for (i, cf_list) in utterances.iter().enumerate() {
        let mut state = CenteringState::new(i).with_cf(cf_list.clone());

        if i == 0 {
            // Discourse-initial: no Cb, NULL transition
            state.cb = None;
            state.transition = CenteringTransition::Null;
        } else {
            let prev_state = &states[i - 1];

            // Compute Cb: highest-ranked member of Cf(U_{n-1}) realized in U_n
            state.cb = compute_cb(prev_state, &state);

            // Compute transition
            state.transition = compute_transition(prev_state, &state);
        }

        // Compute recency scores if enabled
        if config.use_recency {
            state.recency_scores = compute_recency_scores(&states, &state, config);
        }

        states.push(state);
    }

    states
}

/// Compute the backward-looking center.
///
/// Cb(U_n) is the highest-ranked member of Cf(U_{n-1}) that is realized in U_n.
fn compute_cb(prev_state: &CenteringState, current_state: &CenteringState) -> Option<u64> {
    // Cf is already sorted by salience (highest first)
    for fc in &prev_state.cf {
        if current_state.mentions(fc.entity_id) {
            return Some(fc.entity_id);
        }
    }
    None
}

/// Compute the centering transition between states.
///
/// Following Brennan, Friedman, Pollard (1987):
/// - If no previous Cb exists (discourse-initial), any transition establishing Cb
///   is treated as CONTINUE if Cb = Cp, otherwise RETAIN
/// - Subsequent transitions compare Cb to previous Cb
pub fn compute_transition(
    prev_state: &CenteringState,
    current_state: &CenteringState,
) -> CenteringTransition {
    let prev_cb = prev_state.cb;
    let curr_cb = current_state.cb;
    let curr_cp = current_state.cp;

    // No Cb in current utterance means NULL transition
    if curr_cb.is_none() {
        return CenteringTransition::Null;
    }

    let cb = curr_cb.expect("curr_cb.is_none() checked above");

    match (prev_cb, curr_cp) {
        // Cb(U_n) = Cb(U_{n-1}) AND Cb(U_n) = Cp(U_n)
        (Some(prev), Some(cp)) if prev == cb && cb == cp => CenteringTransition::Continue,

        // Cb(U_n) = Cb(U_{n-1}) AND Cb(U_n) != Cp(U_n)
        (Some(prev), Some(cp)) if prev == cb && cb != cp => CenteringTransition::Retain,

        // Cb(U_n) != Cb(U_{n-1}) AND Cb(U_n) = Cp(U_n)
        (Some(prev), Some(cp)) if prev != cb && cb == cp => CenteringTransition::SmoothShift,

        // Cb(U_n) != Cb(U_{n-1}) AND Cb(U_n) != Cp(U_n)
        (Some(prev), Some(cp)) if prev != cb && cb != cp => CenteringTransition::RoughShift,

        // Discourse-initial: no previous Cb
        // Following BFP, establishing a Cb from a discourse-initial context
        // is CONTINUE if Cb=Cp (smooth establishment), RETAIN otherwise
        (None, Some(cp)) if cb == cp => CenteringTransition::Continue,
        (None, Some(_)) => CenteringTransition::Retain,

        // Fallback
        _ => CenteringTransition::Null,
    }
}

/// Compute recency-weighted salience scores.
///
/// From Jiang et al. (2022): CT + recency captures more coreference signal.
fn compute_recency_scores(
    prev_states: &[CenteringState],
    current: &CenteringState,
    config: &CenteringConfig,
) -> HashMap<u64, f64> {
    let mut scores: HashMap<u64, f64> = HashMap::new();

    // Start with current utterance's entities
    for fc in &current.cf {
        scores.insert(fc.entity_id, fc.effective_salience());
    }

    // Add recency-decayed scores from previous utterances
    let start = prev_states.len().saturating_sub(config.recency_window);
    for (i, state) in prev_states[start..].iter().enumerate() {
        let age = prev_states.len() - start - i; // How far back
        let decay = config.recency_decay.powi(age as i32);

        for fc in &state.cf {
            let recency_score = fc.effective_salience() * decay;
            scores
                .entry(fc.entity_id)
                .and_modify(|s| *s += recency_score)
                .or_insert(recency_score);
        }
    }

    scores
}

// =============================================================================
// Discourse Coherence Analysis
// =============================================================================

/// Analyze discourse coherence based on centering transitions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoherenceAnalysis {
    /// Number of each transition type
    pub transition_counts: HashMap<String, usize>,
    /// Average coherence score (0-1)
    pub avg_coherence: f64,
    /// Proportion of continuing transitions (CONTINUE + RETAIN)
    pub continuity_ratio: f64,
    /// Number of center shifts
    pub shift_count: usize,
    /// Longest run of continuing transitions
    pub max_continuity_run: usize,
}

/// Analyze discourse coherence from centering states.
pub fn analyze_coherence(states: &[CenteringState]) -> CoherenceAnalysis {
    if states.is_empty() {
        return CoherenceAnalysis::default();
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut total_coherence = 0.0;
    let mut continuing = 0;
    let mut shifts = 0;
    let mut current_run = 0;
    let mut max_run = 0;

    for state in states {
        let key = state.transition.as_str().to_string();
        *counts.entry(key).or_default() += 1;

        total_coherence += state.transition.coherence_score();

        if state.transition.is_continuing() {
            continuing += 1;
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 0;
            if state.transition.is_shifting() {
                shifts += 1;
            }
        }
    }

    CoherenceAnalysis {
        transition_counts: counts,
        avg_coherence: total_coherence / states.len() as f64,
        continuity_ratio: continuing as f64 / states.len() as f64,
        shift_count: shifts,
        max_continuity_run: max_run,
    }
}

// =============================================================================
// Integration with Coreference
// =============================================================================

/// Use centering to score antecedent candidates.
///
/// Returns a map from entity_id to centering-based score.
/// Higher scores indicate better antecedent candidates.
pub fn score_antecedents(
    anaphor_utterance: usize,
    states: &[CenteringState],
    config: &CenteringConfig,
) -> HashMap<u64, f64> {
    let mut scores: HashMap<u64, f64> = HashMap::new();

    if anaphor_utterance == 0 || states.is_empty() {
        return scores;
    }

    let current_state = states.get(anaphor_utterance);

    // If we have recency scores, use them
    if let Some(state) = current_state {
        if config.use_recency && !state.recency_scores.is_empty() {
            return state.recency_scores.clone();
        }
    }

    // Otherwise, compute from Cf rankings
    for (i, state) in states[..anaphor_utterance].iter().enumerate().rev() {
        let age = anaphor_utterance - i;
        let decay = config.recency_decay.powi(age as i32);

        for fc in &state.cf {
            let score = fc.effective_salience() * decay;

            // Cb bonus: current Cb is preferred antecedent
            let cb_bonus = if Some(fc.entity_id) == state.cb {
                0.2
            } else {
                0.0
            };

            scores
                .entry(fc.entity_id)
                .and_modify(|s| *s = s.max(score + cb_bonus))
                .or_insert(score + cb_bonus);
        }
    }

    scores
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_center_salience() {
        let fc = ForwardCenter::new(1, "John", 0.8)
            .with_role(GrammaticalRole::Subject)
            .with_info_status(InformationStatus::Evoked);

        // salience * role_weight + info_status_boost
        // 0.8 * 1.0 + 0.3 = 1.1
        let expected = 0.8 * 1.0 + 0.3;
        assert!((fc.effective_salience() - expected).abs() < 0.001);
    }

    #[test]
    fn test_centering_continue() {
        let utterances = vec![
            vec![
                ForwardCenter::new(1, "John", 1.0).with_role(GrammaticalRole::Subject),
                ForwardCenter::new(2, "Mary", 0.8).with_role(GrammaticalRole::DirectObject),
            ],
            vec![
                ForwardCenter::new(1, "He", 0.9).with_role(GrammaticalRole::Subject),
                ForwardCenter::new(3, "the book", 0.7).with_role(GrammaticalRole::DirectObject),
            ],
        ];

        let config = CenteringConfig::default();
        let states = track_centers(&utterances, &config);

        // U1: Cb = None (discourse-initial)
        assert_eq!(states[0].cb, None);
        assert_eq!(states[0].cp, Some(1)); // John is Cp

        // U2: Cb = 1 (John from Cf(U1) realized as "He")
        assert_eq!(states[1].cb, Some(1));
        assert_eq!(states[1].cp, Some(1)); // He is Cp

        // Transition: CONTINUE (same Cb, Cb = Cp)
        assert_eq!(states[1].transition, CenteringTransition::Continue);
    }

    #[test]
    fn test_centering_retain() {
        let utterances = vec![
            vec![ForwardCenter::new(1, "John", 1.0).with_role(GrammaticalRole::Subject)],
            vec![
                // Mary is subject (higher salience), but John is mentioned
                ForwardCenter::new(2, "Mary", 1.0).with_role(GrammaticalRole::Subject),
                ForwardCenter::new(1, "him", 0.7).with_role(GrammaticalRole::DirectObject),
            ],
        ];

        let config = CenteringConfig::default();
        let states = track_centers(&utterances, &config);

        // U2: Cb = 1 (John), but Cp = 2 (Mary is highest Cf)
        assert_eq!(states[1].cb, Some(1));
        assert_eq!(states[1].cp, Some(2));

        // Transition: RETAIN (same Cb, but Cb != Cp)
        assert_eq!(states[1].transition, CenteringTransition::Retain);
    }

    #[test]
    fn test_centering_smooth_shift() {
        let utterances = vec![
            vec![ForwardCenter::new(1, "John", 1.0)],
            vec![
                // Only Mary mentioned, no John
                ForwardCenter::new(2, "Mary", 1.0).with_role(GrammaticalRole::Subject),
            ],
        ];

        let config = CenteringConfig::default();
        let states = track_centers(&utterances, &config);

        // U2: Cb = None (John not realized), but we have entities
        // Actually this should be NULL since no Cf from U1 is in U2
        assert_eq!(states[1].cb, None);
        assert_eq!(states[1].transition, CenteringTransition::Null);
    }

    #[test]
    fn test_coherence_analysis() {
        let utterances = vec![
            vec![ForwardCenter::new(1, "John", 1.0)],
            vec![ForwardCenter::new(1, "he", 0.9)],
            vec![ForwardCenter::new(1, "him", 0.8)],
            vec![ForwardCenter::new(2, "Mary", 1.0)], // Shift
        ];

        let config = CenteringConfig::default();
        let states = track_centers(&utterances, &config);
        let analysis = analyze_coherence(&states);

        assert!(analysis.avg_coherence > 0.0);
        // U1->U2: CONTINUE (establishing Cb=Cp=1)
        // U2->U3: CONTINUE (maintaining Cb=Cp=1)
        // U3->U4: NULL (no entity from U3 in U4)
        // So we have a run of 2 CONTINUEs (U2 and U3)
        // Actually U4 has no entity from U3, so its Cb is None
        assert!(analysis.max_continuity_run >= 2);
    }

    #[test]
    fn test_recency_scores() {
        let config = CenteringConfig {
            use_recency: true,
            recency_decay: 0.5,
            recency_window: 3,
            ..Default::default()
        };

        let utterances = vec![
            vec![ForwardCenter::new(1, "John", 1.0)],
            vec![ForwardCenter::new(2, "Mary", 1.0)],
            vec![
                ForwardCenter::new(1, "he", 0.9),
                ForwardCenter::new(2, "her", 0.8),
            ],
        ];

        let states = track_centers(&utterances, &config);

        // U3 should have recency scores for both John and Mary
        let scores = &states[2].recency_scores;
        assert!(scores.contains_key(&1));
        assert!(scores.contains_key(&2));
    }

    #[test]
    fn test_transition_coherence_ordering() {
        assert!(
            CenteringTransition::Continue.coherence_score()
                > CenteringTransition::Retain.coherence_score()
        );
        assert!(
            CenteringTransition::Retain.coherence_score()
                > CenteringTransition::SmoothShift.coherence_score()
        );
        assert!(
            CenteringTransition::SmoothShift.coherence_score()
                > CenteringTransition::RoughShift.coherence_score()
        );
    }

    #[test]
    fn test_grammatical_role_ordering() {
        assert!(
            GrammaticalRole::Subject.salience_weight()
                > GrammaticalRole::DirectObject.salience_weight()
        );
        assert!(
            GrammaticalRole::DirectObject.salience_weight()
                > GrammaticalRole::IndirectObject.salience_weight()
        );
        assert!(
            GrammaticalRole::IndirectObject.salience_weight()
                > GrammaticalRole::Oblique.salience_weight()
        );
    }

    #[test]
    fn test_information_status() {
        assert!(InformationStatus::Evoked.is_hearer_old());
        assert!(InformationStatus::Unused.is_hearer_old());
        assert!(InformationStatus::Inferrable.is_hearer_old());
        assert!(!InformationStatus::New.is_hearer_old());

        assert!(
            InformationStatus::Evoked.salience_boost() > InformationStatus::New.salience_boost()
        );
    }

    #[test]
    fn test_score_antecedents() {
        let utterances = vec![
            vec![
                ForwardCenter::new(1, "John", 1.0).with_role(GrammaticalRole::Subject),
                ForwardCenter::new(2, "Mary", 0.8),
            ],
            vec![ForwardCenter::new(3, "the book", 0.7)],
        ];

        let config = CenteringConfig::default();
        let states = track_centers(&utterances, &config);

        // Score antecedents for a pronoun in U2
        let scores = score_antecedents(1, &states, &config);

        // John should score higher than Mary (was subject)
        assert!(scores.get(&1).unwrap_or(&0.0) >= scores.get(&2).unwrap_or(&0.0));
    }
}
