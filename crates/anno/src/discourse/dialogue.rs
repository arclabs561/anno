//! Dialogue-specific types for conversational AI interaction analysis.
//!
//! # Why Dialogue Types?
//!
//! Most NLP evaluation datasets come from **written text** (news, Wikipedia, books).
//! Conversational AI systems operate in **spoken dialogue** where entirely different
//! phenomena emerge:
//!
//! 1. **Response tokens** like "uh huh", "oui", "d'accord" aren't questions—they're
//!    conversational grease that keeps dialogue flowing. But VAD-based systems treat
//!    any detected speech as requiring a response, causing interruptions.
//!
//! 2. **Aside sequences** where humans whisper or gesture to exclude the agent get
//!    picked up anyway, because "everything counts" to an always-listening system.
//!
//! 3. **Multi-party interactions** where speaker attribution and addressee tracking
//!    matter for understanding who said what to whom.
//!
//! # Theoretical Background
//!
//! ## Turn-Taking Models
//!
//! Voice agents use **silence-based turn-taking** (Voice Activity Detection):
//! any detected speech triggers processing. This differs fundamentally from human
//! turn-taking, which uses syntactic, prosodic, and pragmatic cues.
//!
//! Key research: Rudaz, Broth & Mlynář (2025) "Everything counts: the managed
//! omnirelevance of speech in human-voice agent interaction"
//!
//! ## Speech Acts vs Entities
//!
//! Standard NER extracts **entities** (Person, Org, Location).
//! Dialogue analysis requires extracting **speech acts**:
//!
//! | Type | Example | What It Is |
//! |------|---------|------------|
//! | Continuer | "uh huh", "oui" | Signals continued attention |
//! | Acknowledgment | "okay", "d'accord" | Confirms receipt |
//! | Assessment | "wow", "really" | Evaluates prior turn |
//! | BackChannel | "mm-hmm" | Non-intrusive attention signal |
//!
//! These are **not** entities, but they're crucial for understanding dialogue flow.
//!
//! # Example
//!
//! ```rust
//! use anno::discourse::dialogue::{DialogueTurn, SpeechActType, ParticipantType};
//!
//! let turn = DialogueTurn::new("oui", "EMM")
//!     .with_participant_type(ParticipantType::Human)
//!     .with_speech_act(SpeechActType::Continuer)
//!     .as_aside(false);
//!
//! assert!(turn.is_response_token());
//! ```

use serde::{Deserialize, Serialize};

// =============================================================================
// Participant Types
// =============================================================================

/// Type of participant in a dialogue.
///
/// This distinction matters because:
/// - **Human** participants use natural turn-taking cues
/// - **Agent** participants use VAD-based turn detection
/// - The mismatch causes the problems documented in HVA research
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ParticipantType {
    /// Human speaker (uses natural turn-taking)
    #[default]
    Human,
    /// AI agent (uses silence-based turn detection)
    Agent,
    /// Unknown participant type
    Unknown,
}

impl ParticipantType {
    /// Human-readable label.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            ParticipantType::Human => "human",
            ParticipantType::Agent => "agent",
            ParticipantType::Unknown => "unknown",
        }
    }
}

// =============================================================================
// Speech Act Types
// =============================================================================

/// Classification of pragmatic function in dialogue.
///
/// # Why Speech Act Types Matter
///
/// Response tokens are **not entities** (they don't refer to things in the world),
/// but they're crucial for dialogue understanding. A continuer like "uh huh" signals
/// continued attention, not a question requiring an answer.
///
/// VAD-based voice agents misinterpret these because they treat any detected speech
/// as requiring a response—"the managed omnirelevance of speech."
///
/// # Categories
///
/// Based on conversation analysis literature (Jefferson 1984, Gardner 2001):
///
/// - **Continuer**: Signals "go on" without claiming the floor ("mm-hmm", "uh huh")
/// - **Acknowledgment**: Confirms receipt of prior turn ("okay", "d'accord")
/// - **Assessment**: Evaluates prior turn content ("wow", "really", "interesting")
/// - **Alignment**: Indicates agreement or understanding ("yeah", "right")
/// - **BackChannel**: Non-intrusive attention signal during another's turn
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SpeechActType {
    /// Signals continued attention without taking the floor.
    /// Examples: "mm-hmm", "uh huh", "oui" (French)
    Continuer,

    /// Confirms receipt of information.
    /// Examples: "okay", "d'accord", "got it"
    Acknowledgment,

    /// Evaluates or reacts to prior turn content.
    /// Examples: "wow", "really", "interesting", "oh no"
    Assessment,

    /// Indicates agreement or shared understanding.
    /// Examples: "yeah", "right", "exactly"
    Alignment,

    /// Non-intrusive attention signal during another's turn.
    /// Overlaps with speaker's talk without claiming floor.
    BackChannel,

    /// Question requiring an answer.
    Question,

    /// Statement conveying information.
    Statement,

    /// Request for action.
    Request,

    /// Turn designed to close interaction.
    Farewell,

    /// Turn designed to open interaction.
    Greeting,

    /// Other/unclassified speech act.
    Other,
}

impl SpeechActType {
    /// Is this a response token (continuer, acknowledgment, etc.)?
    ///
    /// Response tokens are minimal turns that don't introduce new content
    /// but maintain dialogue flow. VAD-based systems often misinterpret these.
    #[must_use]
    pub const fn is_response_token(&self) -> bool {
        matches!(
            self,
            SpeechActType::Continuer
                | SpeechActType::Acknowledgment
                | SpeechActType::Assessment
                | SpeechActType::Alignment
                | SpeechActType::BackChannel
        )
    }

    /// Human-readable label.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            SpeechActType::Continuer => "continuer",
            SpeechActType::Acknowledgment => "acknowledgment",
            SpeechActType::Assessment => "assessment",
            SpeechActType::Alignment => "alignment",
            SpeechActType::BackChannel => "backchannel",
            SpeechActType::Question => "question",
            SpeechActType::Statement => "statement",
            SpeechActType::Request => "request",
            SpeechActType::Farewell => "farewell",
            SpeechActType::Greeting => "greeting",
            SpeechActType::Other => "other",
        }
    }
}

// =============================================================================
// Dialogue Turn
// =============================================================================

/// A single turn in a dialogue.
///
/// # Why Turn-Level Metadata?
///
/// Written text is speaker-agnostic. Dialogue requires tracking:
/// - Who said this?
/// - Who were they talking to?
/// - Was this meant to be heard by the agent?
/// - Did it trigger an unwanted agent response?
///
/// # Example
///
/// ```rust
/// use anno::discourse::dialogue::{DialogueTurn, SpeechActType, ParticipantType};
///
/// // A human says "oui" as a continuer
/// let turn = DialogueTurn::new("oui", "EMM")
///     .with_participant_type(ParticipantType::Human)
///     .with_speech_act(SpeechActType::Continuer)
///     .with_triggered_cutoff(true);
///
/// assert!(turn.is_response_token());
/// assert!(turn.triggered_cutoff);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTurn {
    /// The utterance text.
    pub text: String,

    /// Speaker identifier (e.g., "EMM", "GPT", "TOM").
    pub speaker: String,

    /// Type of participant (human or agent).
    pub participant_type: ParticipantType,

    /// Pragmatic function of this turn.
    pub speech_act: Option<SpeechActType>,

    /// Is this an aside (directed away from the agent)?
    ///
    /// Aside sequences are contributions designed to exclude the agent:
    /// - Whispered to co-participants
    /// - Gestured/mouthed rather than spoken
    /// - Explicitly directed at another human
    pub is_aside: bool,

    /// Did this turn trigger an agent interruption/cutoff?
    ///
    /// Response tokens often trigger unwanted agent responses because
    /// VAD detects them as speech requiring a response.
    pub triggered_cutoff: bool,

    /// Turn number in the dialogue (0-indexed).
    pub turn_number: usize,

    /// Who this turn is addressed to (if known).
    pub addressee: Option<String>,

    /// Language code (e.g., "fr", "en").
    pub language: Option<String>,

    /// Character offset in the full dialogue transcript.
    pub start: usize,

    /// End character offset.
    pub end: usize,
}

impl DialogueTurn {
    /// Create a new dialogue turn.
    #[must_use]
    pub fn new(text: impl Into<String>, speaker: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            speaker: speaker.into(),
            participant_type: ParticipantType::Unknown,
            speech_act: None,
            is_aside: false,
            triggered_cutoff: false,
            turn_number: 0,
            addressee: None,
            language: None,
            start: 0,
            end: 0,
        }
    }

    /// Set participant type.
    #[must_use]
    pub fn with_participant_type(mut self, pt: ParticipantType) -> Self {
        self.participant_type = pt;
        self
    }

    /// Set speech act type.
    #[must_use]
    pub fn with_speech_act(mut self, act: SpeechActType) -> Self {
        self.speech_act = Some(act);
        self
    }

    /// Mark as aside (not directed at agent).
    #[must_use]
    pub fn as_aside(mut self, is_aside: bool) -> Self {
        self.is_aside = is_aside;
        self
    }

    /// Mark whether this triggered an agent cutoff.
    #[must_use]
    pub fn with_triggered_cutoff(mut self, triggered: bool) -> Self {
        self.triggered_cutoff = triggered;
        self
    }

    /// Set turn number.
    #[must_use]
    pub fn with_turn_number(mut self, n: usize) -> Self {
        self.turn_number = n;
        self
    }

    /// Set addressee.
    #[must_use]
    pub fn with_addressee(mut self, addr: impl Into<String>) -> Self {
        self.addressee = Some(addr.into());
        self
    }

    /// Set language.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set character span.
    #[must_use]
    pub fn with_span(mut self, start: usize, end: usize) -> Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Is this a response token?
    #[must_use]
    pub fn is_response_token(&self) -> bool {
        self.speech_act.is_some_and(|act| act.is_response_token())
    }

    /// Is this from a human participant?
    #[must_use]
    pub fn is_human(&self) -> bool {
        matches!(self.participant_type, ParticipantType::Human)
    }

    /// Is this from an agent participant?
    #[must_use]
    pub fn is_agent(&self) -> bool {
        matches!(self.participant_type, ParticipantType::Agent)
    }
}

// =============================================================================
// Dialogue Context
// =============================================================================

/// Tracks the state of a multi-turn dialogue.
///
/// # Why Dialogue Context?
///
/// Coreference in dialogue requires knowing:
/// - Who are the active participants?
/// - What has been said recently?
/// - Who is the current addressee?
///
/// This context enables proper resolution of:
/// - Speaker pronouns ("I", "you")
/// - Addressee tracking
/// - Response token classification based on prior turn
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DialogueContext {
    /// All turns in the dialogue.
    pub turns: Vec<DialogueTurn>,

    /// Active participant IDs.
    pub participants: Vec<String>,

    /// Current addressee (who the last turn was directed at).
    pub current_addressee: Option<String>,

    /// Dialogue identifier.
    pub dialogue_id: Option<String>,
}

impl DialogueContext {
    /// Create a new empty dialogue context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a turn to the dialogue.
    pub fn add_turn(&mut self, mut turn: DialogueTurn) {
        turn.turn_number = self.turns.len();

        // Track participants
        if !self.participants.contains(&turn.speaker) {
            self.participants.push(turn.speaker.clone());
        }

        // Update addressee tracking
        if let Some(ref addr) = turn.addressee {
            self.current_addressee = Some(addr.clone());
        }

        self.turns.push(turn);
    }

    /// Get the last N turns.
    #[must_use]
    pub fn last_turns(&self, n: usize) -> &[DialogueTurn] {
        let start = self.turns.len().saturating_sub(n);
        &self.turns[start..]
    }

    /// Get turns from a specific speaker.
    #[must_use]
    pub fn turns_by_speaker(&self, speaker: &str) -> Vec<&DialogueTurn> {
        self.turns.iter().filter(|t| t.speaker == speaker).collect()
    }

    /// Count response tokens that triggered cutoffs.
    #[must_use]
    pub fn cutoff_count(&self) -> usize {
        self.turns
            .iter()
            .filter(|t| t.is_response_token() && t.triggered_cutoff)
            .count()
    }

    /// Count aside sequences.
    #[must_use]
    pub fn aside_count(&self) -> usize {
        self.turns.iter().filter(|t| t.is_aside).count()
    }

    /// Get the full dialogue text (concatenated turns).
    #[must_use]
    pub fn full_text(&self) -> String {
        self.turns
            .iter()
            .map(|t| format!("{}: {}", t.speaker, t.text))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// =============================================================================
// Response Token Lexicon
// =============================================================================

/// Classify a token as a response token type.
///
/// Based on conversation analysis literature and multilingual examples
/// from the Human-Voice Agent dataset.
#[must_use]
pub fn classify_response_token(token: &str, lang: Option<&str>) -> Option<SpeechActType> {
    let lower = token.to_lowercase();
    let lang = lang.unwrap_or("en");

    match lang {
        "fr" => match lower.as_str() {
            "oui" | "ouais" | "mm" | "mhm" => Some(SpeechActType::Continuer),
            "d'accord" | "ok" | "okai" | "okay" => Some(SpeechActType::Acknowledgment),
            "ah" | "oh" | "wow" => Some(SpeechActType::Assessment),
            "exactement" | "voilà" | "c'est ça" => Some(SpeechActType::Alignment),
            "salut" | "bonjour" => Some(SpeechActType::Greeting),
            "au revoir" | "à bientôt" => Some(SpeechActType::Farewell),
            _ => None,
        },
        _ => match lower.as_str() {
            // English defaults
            "uh huh" | "mm-hmm" | "mm" | "mhm" | "yeah" => Some(SpeechActType::Continuer),
            "okay" | "ok" | "got it" | "i see" => Some(SpeechActType::Acknowledgment),
            "wow" | "really" | "oh" | "interesting" => Some(SpeechActType::Assessment),
            "right" | "exactly" | "yes" => Some(SpeechActType::Alignment),
            "hello" | "hi" | "hey" => Some(SpeechActType::Greeting),
            "bye" | "goodbye" | "see you" => Some(SpeechActType::Farewell),
            _ => None,
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_participant_type() {
        assert_eq!(ParticipantType::Human.as_str(), "human");
        assert_eq!(ParticipantType::Agent.as_str(), "agent");
    }

    #[test]
    fn test_speech_act_response_token() {
        assert!(SpeechActType::Continuer.is_response_token());
        assert!(SpeechActType::Acknowledgment.is_response_token());
        assert!(SpeechActType::Assessment.is_response_token());
        assert!(SpeechActType::Alignment.is_response_token());
        assert!(SpeechActType::BackChannel.is_response_token());

        assert!(!SpeechActType::Question.is_response_token());
        assert!(!SpeechActType::Statement.is_response_token());
    }

    #[test]
    fn test_dialogue_turn() {
        let turn = DialogueTurn::new("oui", "EMM")
            .with_participant_type(ParticipantType::Human)
            .with_speech_act(SpeechActType::Continuer)
            .with_triggered_cutoff(true);

        assert!(turn.is_response_token());
        assert!(turn.is_human());
        assert!(!turn.is_agent());
        assert!(turn.triggered_cutoff);
    }

    #[test]
    fn test_aside() {
        let turn = DialogueTurn::new("°tu n'es pas très intelligent°", "ANA")
            .with_participant_type(ParticipantType::Human)
            .as_aside(true);

        assert!(turn.is_aside);
        assert!(turn.is_human());
    }

    #[test]
    fn test_dialogue_context() {
        let mut ctx = DialogueContext::new();

        ctx.add_turn(
            DialogueTurn::new("Bonjour", "GPT")
                .with_participant_type(ParticipantType::Agent)
                .with_speech_act(SpeechActType::Greeting),
        );

        ctx.add_turn(
            DialogueTurn::new("oui", "EMM")
                .with_participant_type(ParticipantType::Human)
                .with_speech_act(SpeechActType::Continuer)
                .with_triggered_cutoff(true),
        );

        assert_eq!(ctx.turns.len(), 2);
        assert_eq!(ctx.participants.len(), 2);
        assert_eq!(ctx.cutoff_count(), 1);
    }

    #[test]
    fn test_french_response_tokens() {
        assert_eq!(
            classify_response_token("oui", Some("fr")),
            Some(SpeechActType::Continuer)
        );
        assert_eq!(
            classify_response_token("d'accord", Some("fr")),
            Some(SpeechActType::Acknowledgment)
        );
        assert_eq!(
            classify_response_token("exactement", Some("fr")),
            Some(SpeechActType::Alignment)
        );
    }

    #[test]
    fn test_english_response_tokens() {
        assert_eq!(
            classify_response_token("uh huh", None),
            Some(SpeechActType::Continuer)
        );
        assert_eq!(
            classify_response_token("okay", None),
            Some(SpeechActType::Acknowledgment)
        );
        assert_eq!(
            classify_response_token("exactly", None),
            Some(SpeechActType::Alignment)
        );
    }
}
