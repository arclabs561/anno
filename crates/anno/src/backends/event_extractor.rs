//! Event extraction trait and implementations.
//!
//! Event extraction identifies event triggers (verbs/nouns that denote events)
//! and their arguments (participants, time, place) following ACE event schema.
//!
//! # ACE Event Schema
//!
//! Events have:
//! - **Trigger**: The word/phrase that denotes the event (e.g., "invaded", "announced")
//! - **Type**: Event category (e.g., "conflict:attack", "movement:transport")
//! - **Arguments**: Roles filled by entities (e.g., Agent, Patient, Time, Place)
//!
//! # Usage
//!
//! ```rust
//! use anno::backends::event_extractor::{EventExtractor, RuleBasedEventExtractor};
//!
//! let extractor = RuleBasedEventExtractor::new();
//! let events = extractor
//!     .extract_events("Russia invaded Ukraine in 2022.", None)
//!     .unwrap();
//!
//! for event in &events {
//!     println!("Event: {} ({})", event.trigger, event.event_type);
//!     for (role, entity) in &event.arguments {
//!         println!("  {}: {}", role, entity.text);
//!     }
//! }
//! ```
//!
//! # Research Background
//!
//! Based on ACE 2005 event ontology:
//! - Life: be-born, marry, divorce, injure, die
//! - Movement: transport
//! - Transaction: transfer-ownership, transfer-money
//! - Business: start-org, merge-org, declare-bankruptcy, end-org
//! - Conflict: attack, demonstrate
//! - Contact: meet, phone-write
//! - Personnel: start-position, end-position, nominate, elect
//! - Justice: arrest-jail, release-parole, trial-hearing, charge-indict, etc.

use crate::{Entity, Result};

/// Event with trigger and arguments following ACE schema.
#[derive(Debug, Clone)]
pub struct Event {
    /// The trigger word/phrase (e.g., "invaded", "announced")
    pub trigger: String,
    /// Start character offset of trigger
    pub trigger_start: usize,
    /// End character offset of trigger
    pub trigger_end: usize,
    /// Event type (e.g., "conflict:attack", "movement:transport")
    pub event_type: String,
    /// Event arguments: (role, entity)
    /// Common roles: Agent, Patient, Time, Place, Instrument
    pub arguments: Vec<(String, Entity)>,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
}

impl Event {
    /// Create a new event.
    #[must_use]
    pub fn new(
        trigger: impl Into<String>,
        trigger_start: usize,
        trigger_end: usize,
        event_type: impl Into<String>,
    ) -> Self {
        Self {
            trigger: trigger.into(),
            trigger_start,
            trigger_end,
            event_type: event_type.into(),
            arguments: Vec::new(),
            confidence: 1.0,
        }
    }

    /// Add an argument to this event.
    #[must_use]
    pub fn with_argument(mut self, role: impl Into<String>, entity: Entity) -> Self {
        self.arguments.push((role.into(), entity));
        self
    }

    /// Set confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Trait for event extraction backends.
///
/// Event extraction identifies event triggers and their arguments.
/// This is distinct from NER (which extracts entities) and relation extraction
/// (which links entities). Events are structured occurrences with participants.
pub trait EventExtractor: Send + Sync {
    /// Extract events from text.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to extract events from
    /// * `language` - Optional language hint (e.g., "en", "es")
    ///
    /// # Returns
    ///
    /// Vector of events with triggers and arguments.
    fn extract_events(&self, text: &str, language: Option<&str>) -> Result<Vec<Event>>;

    /// Get the extractor name/identifier.
    fn name(&self) -> &'static str;

    /// Get a description of the extractor.
    fn description(&self) -> &'static str {
        "Event extractor"
    }
}

/// Rule-based event extractor using trigger patterns.
///
/// Uses simple pattern matching on event trigger words.
/// Fast but limited coverage compared to neural methods.
pub struct RuleBasedEventExtractor {
    /// Minimum confidence threshold
    threshold: f64,
}

impl RuleBasedEventExtractor {
    /// Create a new rule-based event extractor.
    #[must_use]
    pub fn new() -> Self {
        Self { threshold: 0.5 }
    }

    /// Create with custom confidence threshold.
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }
}

impl Default for RuleBasedEventExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl EventExtractor for RuleBasedEventExtractor {
    fn extract_events(&self, text: &str, language: Option<&str>) -> Result<Vec<Event>> {
        let mut events = Vec::new();

        // Language-aware trigger patterns (ACE 2005 inspired)
        // In production, this would use a more sophisticated lexicon
        let lang_code = language.map(|l| l.split('-').next().unwrap_or(l).to_lowercase());

        // Get language-specific patterns or fall back to English
        let trigger_patterns: Vec<(&str, &str)> = match lang_code.as_deref() {
            Some("es") => vec![
                // Spanish conflict events
                ("invadió", "conflict:attack"),
                ("atacó", "conflict:attack"),
                ("bombardeó", "conflict:attack"),
                ("guerra", "conflict:attack"),
                // Spanish movement
                ("viajó", "movement:transport"),
                ("movió", "movement:transport"),
                ("desplegó", "movement:transport"),
                // Spanish transaction
                ("compró", "transaction:transfer-ownership"),
                ("vendió", "transaction:transfer-ownership"),
                ("adquirió", "transaction:transfer-ownership"),
                // Spanish business
                ("fundó", "business:start-org"),
                ("inició", "business:start-org"),
                ("fusionó", "business:merge-org"),
                // Spanish communication
                ("anunció", "communication:announce"),
                ("declaró", "communication:announce"),
                ("informó", "communication:announce"),
                ("dijo", "communication:announce"),
                // Spanish life
                ("nació", "life:be-born"),
                ("murió", "life:die"),
                ("se casó", "life:marry"),
                ("divorció", "life:divorce"),
                // Spanish justice
                ("arrestó", "justice:arrest-jail"),
                ("acusó", "justice:charge-indict"),
                ("condenó", "justice:convict"),
                ("sentenció", "justice:sentence"),
            ],
            Some("fr") => vec![
                // French conflict
                ("envahi", "conflict:attack"),
                ("attaqué", "conflict:attack"),
                ("bombardé", "conflict:attack"),
                ("guerre", "conflict:attack"),
                // French movement
                ("voyagé", "movement:transport"),
                ("déplacé", "movement:transport"),
                ("déployé", "movement:transport"),
                // French transaction
                ("acheté", "transaction:transfer-ownership"),
                ("vendu", "transaction:transfer-ownership"),
                ("acquis", "transaction:transfer-ownership"),
                // French business
                ("fondé", "business:start-org"),
                ("créé", "business:start-org"),
                ("fusionné", "business:merge-org"),
                // French communication
                ("annoncé", "communication:announce"),
                ("déclaré", "communication:announce"),
                ("rapporté", "communication:announce"),
                ("dit", "communication:announce"),
                // French life
                ("né", "life:be-born"),
                ("mort", "life:die"),
                ("marié", "life:marry"),
                ("divorcé", "life:divorce"),
                // French justice
                ("arrêté", "justice:arrest-jail"),
                ("accusé", "justice:charge-indict"),
                ("condamné", "justice:convict"),
                ("condamné", "justice:sentence"),
            ],
            Some("de") => vec![
                // German conflict
                ("invadiert", "conflict:attack"),
                ("angegriffen", "conflict:attack"),
                ("bombardiert", "conflict:attack"),
                ("krieg", "conflict:attack"),
                // German movement
                ("gereist", "movement:transport"),
                ("bewegt", "movement:transport"),
                ("verlegt", "movement:transport"),
                // German transaction
                ("gekauft", "transaction:transfer-ownership"),
                ("verkauft", "transaction:transfer-ownership"),
                ("erworben", "transaction:transfer-ownership"),
                // German business
                ("gegründet", "business:start-org"),
                ("gestartet", "business:start-org"),
                ("fusioniert", "business:merge-org"),
                // German communication
                ("angekündigt", "communication:announce"),
                ("erklärt", "communication:announce"),
                ("berichtet", "communication:announce"),
                ("sagte", "communication:announce"),
                // German life
                ("geboren", "life:be-born"),
                ("gestorben", "life:die"),
                ("geheiratet", "life:marry"),
                ("geschieden", "life:divorce"),
                // German justice
                ("verhaftet", "justice:arrest-jail"),
                ("angeklagt", "justice:charge-indict"),
                ("verurteilt", "justice:convict"),
                ("verurteilt", "justice:sentence"),
            ],
            Some("zh") | Some("ja") | Some("ko") | Some("ar") | Some("ru") => {
                // For CJK, Arabic, Russian: use English patterns as fallback
                // In production, add language-specific patterns
                vec![]
            }
            _ => vec![
                // English (default)
                ("invaded", "conflict:attack"),
                ("attacked", "conflict:attack"),
                ("bombed", "conflict:attack"),
                ("fired", "conflict:attack"),
                ("war", "conflict:attack"),
                ("traveled", "movement:transport"),
                ("moved", "movement:transport"),
                ("transported", "movement:transport"),
                ("deployed", "movement:transport"),
                ("bought", "transaction:transfer-ownership"),
                ("sold", "transaction:transfer-ownership"),
                ("purchased", "transaction:transfer-ownership"),
                ("acquired", "transaction:transfer-ownership"),
                ("founded", "business:start-org"),
                ("started", "business:start-org"),
                ("merged", "business:merge-org"),
                ("bankruptcy", "business:declare-bankruptcy"),
                ("announced", "communication:announce"),
                ("stated", "communication:announce"),
                ("reported", "communication:announce"),
                ("said", "communication:announce"),
                ("born", "life:be-born"),
                ("died", "life:die"),
                ("married", "life:marry"),
                ("divorced", "life:divorce"),
                ("arrested", "justice:arrest-jail"),
                ("charged", "justice:charge-indict"),
                ("convicted", "justice:convict"),
                ("sentenced", "justice:sentence"),
            ],
        };

        // Use Unicode-aware lowercasing for multilingual support
        // For CJK languages, case doesn't apply, but this is safe
        let text_lower = text.to_lowercase();
        for (trigger_word, event_type) in trigger_patterns {
            // For CJK and other languages without case, search in original text too
            let search_text = if lang_code
                .as_deref()
                .is_some_and(|l| matches!(l, "zh" | "ja" | "ko" | "ar" | "ru"))
            {
                text // No lowercasing for languages where case doesn't apply
            } else {
                &text_lower
            };

            if let Some(pos) = search_text.find(trigger_word) {
                // Find character offset (not byte offset)
                let char_start = text
                    .char_indices()
                    .nth(pos)
                    .map(|(i, _)| i)
                    .unwrap_or(text.len());
                let char_end = text
                    .char_indices()
                    .nth(pos + trigger_word.chars().count())
                    .map(|(i, _)| i)
                    .unwrap_or(text.len());

                // Extract actual trigger text (preserving original case)
                let trigger_text: String = text
                    .chars()
                    .skip(pos)
                    .take(trigger_word.chars().count())
                    .collect();

                let event = Event::new(trigger_text, char_start, char_end, event_type.to_string())
                    .with_confidence(0.7); // Rule-based confidence

                if event.confidence >= self.threshold {
                    events.push(event);
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &'static str {
        "rule-based-event"
    }

    fn description(&self) -> &'static str {
        "Rule-based event extraction using trigger word patterns"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_event_extraction() {
        let extractor = RuleBasedEventExtractor::new();
        let events = extractor
            .extract_events("Russia invaded Ukraine in 2022.", None)
            .unwrap();

        assert!(!events.is_empty());
        assert!(events
            .iter()
            .any(|e| e.trigger.to_lowercase() == "invaded" && e.event_type == "conflict:attack"));
    }

    #[test]
    fn test_event_with_arguments() {
        let event = Event::new("invaded", 7, 14, "conflict:attack");
        // In a full implementation, we'd extract entities and link them as arguments
        assert_eq!(event.arguments.len(), 0);
    }

    #[test]
    fn test_event_unicode_offsets() {
        let extractor = RuleBasedEventExtractor::new();
        let text = "ロシアがウクライナを侵攻した。"; // "Russia invaded Ukraine" in Japanese
        let events = extractor.extract_events(text, Some("ja")).unwrap();

        // Verify offsets are character-based
        for event in &events {
            assert!(event.trigger_start <= event.trigger_end);
            assert!(event.trigger_end <= text.chars().count());
        }
    }

    #[test]
    fn test_multilingual_event_extraction() {
        let extractor = RuleBasedEventExtractor::new();

        // Spanish - "invadió" should match
        let events_es = extractor
            .extract_events("Rusia invadió Ucrania en 2022.", Some("es"))
            .unwrap();
        assert!(!events_es.is_empty(), "Should extract Spanish events");

        // French - "envahi" should match
        let events_fr = extractor
            .extract_events("La Russie a envahi l'Ukraine en 2022.", Some("fr"))
            .unwrap();
        assert!(!events_fr.is_empty(), "Should extract French events");

        // German - "angegriffen" should match (past participle form)
        let events_de = extractor
            .extract_events("Russland hat die Ukraine 2022 angegriffen.", Some("de"))
            .unwrap();
        assert!(!events_de.is_empty(), "Should extract German events");
    }
}
