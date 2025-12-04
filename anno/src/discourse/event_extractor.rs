//! Event trigger extraction for abstract anaphora resolution.
//!
//! This module provides both rule-based and neural event extraction to enable
//! resolution of abstract anaphors like "this" and "that" to
//! event/proposition antecedents.
//!
//! # Backends
//!
//! - **Rule-based** (default): Fast, no model download, uses ACE 2005 lexicon
//! - **GLiNER** (with `candle` feature): Zero-shot neural extraction using bi-encoder
//!
//! # Research Background
//!
//! Based on ACE 2005 event ontology and TAC-KBP guidelines.
//! Event triggers are typically:
//! - Verbs: "invaded", "announced", "crashed"  
//! - Nominalizations: "invasion", "announcement", "crash"
//! - Event nouns: "war", "meeting", "earthquake"
//!
//! # Neural Integration (GLiNER)
//!
//! When the `candle` feature is enabled, you can use GLiNER for zero-shot
//! event detection. This matches spans to event type descriptions:
//!
//! ```rust,ignore
//! use anno::discourse::event_extractor::{EventExtractor, EventExtractorConfig};
//!
//! // Use GLiNER for neural event extraction
//! let config = EventExtractorConfig::default().with_gliner("urchade/gliner_small-v2.1");
//! let extractor = EventExtractor::with_config(config)?;
//!
//! let events = extractor.extract("Russia invaded Ukraine in 2022.");
//! ```
//!
//! # Example (Rule-based)
//!
//! ```rust
//! use anno::discourse::{EventExtractor, EventTriggerLexicon};
//!
//! let extractor = EventExtractor::default();
//! let events = extractor.extract("Russia invaded Ukraine in 2022.");
//!
//! assert_eq!(events.len(), 1);
//! assert_eq!(events[0].trigger, "invaded");
//! assert_eq!(events[0].trigger_type.as_deref(), Some("conflict:attack"));
//! ```

use super::{DiscourseReferent, EventMention, EventPolarity, EventTense, ReferentType};
use anno_core::Entity;
use std::collections::HashMap;

#[cfg(feature = "candle")]
use crate::backends::gliner_candle::GLiNERCandle;

// =============================================================================
// Event Trigger Lexicon
// =============================================================================

/// Event trigger lexicon based on ACE 2005 event types.
///
/// Event types follow the ACE hierarchy:
/// - Life: be-born, marry, divorce, injure, die
/// - Movement: transport
/// - Transaction: transfer-ownership, transfer-money
/// - Business: start-org, merge-org, declare-bankruptcy, end-org
/// - Conflict: attack, demonstrate
/// - Contact: meet, phone-write
/// - Personnel: start-position, end-position, nominate, elect
/// - Justice: arrest-jail, release-parole, trial-hearing, charge-indict, sue, convict, sentence, fine, execute, extradite, acquit, pardon, appeal
#[derive(Debug, Clone)]
pub struct EventTriggerLexicon {
    /// Map from trigger lemma to (event_type, polarity_hint)
    triggers: HashMap<String, (String, Option<EventPolarity>)>,
    /// Modal verbs that indicate hypothetical events
    modal_verbs: Vec<String>,
    /// Negation words
    negation_words: Vec<String>,
}

impl Default for EventTriggerLexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl EventTriggerLexicon {
    /// Create a new lexicon with default triggers.
    #[must_use]
    pub fn new() -> Self {
        let mut triggers = HashMap::new();

        // === Conflict Events ===
        for word in &[
            "attack",
            "attacked",
            "attacking",
            "attacks",
            "invade",
            "invaded",
            "invading",
            "invades",
            "invasion",
            "bomb",
            "bombed",
            "bombing",
            "bombs",
            "bombardment",
            "strike",
            "struck",
            "striking",
            "strikes",
            "assault",
            "assaulted",
            "assaulting",
            "assaults",
            "fight",
            "fought",
            "fighting",
            "fights",
            "battle",
            "battled",
            "battling",
            "battles",
            "war",
            "warfare",
            "kill",
            "killed",
            "killing",
            "kills",
            "murder",
            "murdered",
            "murdering",
            "murders",
            "shoot",
            "shot",
            "shooting",
            "shoots",
            "fire",
            "fired",
            "firing",
            "fires",
            "clash",
            "clashed",
            "clashing",
            "clashes",
            "destroy",
            "destroyed",
            "destroying",
            "destroys",
            "destruction",
        ] {
            triggers.insert(word.to_string(), ("conflict:attack".to_string(), None));
        }

        for word in &[
            "protest",
            "protested",
            "protesting",
            "protests",
            "demonstrate",
            "demonstrated",
            "demonstrating",
            "demonstrates",
            "demonstration",
            "rally",
            "rallied",
            "rallying",
            "rallies",
            "march",
            "marched",
            "marching",
            "marches",
            "riot",
            "rioted",
            "rioting",
            "riots",
        ] {
            triggers.insert(word.to_string(), ("conflict:demonstrate".to_string(), None));
        }

        // === Movement Events ===
        for word in &[
            "go",
            "went",
            "going",
            "goes",
            "gone",
            "move",
            "moved",
            "moving",
            "moves",
            "movement",
            "travel",
            "traveled",
            "travelling",
            "travels",
            "arrive",
            "arrived",
            "arriving",
            "arrives",
            "arrival",
            "depart",
            "departed",
            "departing",
            "departs",
            "departure",
            "leave",
            "left",
            "leaving",
            "leaves",
            "return",
            "returned",
            "returning",
            "returns",
            "flee",
            "fled",
            "fleeing",
            "flees",
            "escape",
            "escaped",
            "escaping",
            "escapes",
            "transport",
            "transported",
            "transporting",
            "transports",
            "ship",
            "shipped",
            "shipping",
            "ships",
            "shipment",
            "send",
            "sent",
            "sending",
            "sends",
            "deploy",
            "deployed",
            "deploying",
            "deploys",
            "deployment",
        ] {
            triggers.insert(word.to_string(), ("movement:transport".to_string(), None));
        }

        // === Transaction Events ===
        for word in &[
            "buy",
            "bought",
            "buying",
            "buys",
            "sell",
            "sold",
            "selling",
            "sells",
            "sale",
            "acquire",
            "acquired",
            "acquiring",
            "acquires",
            "acquisition",
            "purchase",
            "purchased",
            "purchasing",
            "purchases",
            "pay",
            "paid",
            "paying",
            "pays",
            "payment",
            "donate",
            "donated",
            "donating",
            "donates",
            "donation",
            "transfer",
            "transferred",
            "transferring",
            "transfers",
            "invest",
            "invested",
            "investing",
            "invests",
            "investment",
            "fund",
            "funded",
            "funding",
            "funds",
        ] {
            triggers.insert(word.to_string(), ("transaction:transfer".to_string(), None));
        }

        // === Business Events ===
        for word in &[
            "found",
            "founded",
            "founding",
            "founds",
            "foundation",
            "start",
            "started",
            "starting",
            "starts",
            "establish",
            "established",
            "establishing",
            "establishes",
            "create",
            "created",
            "creating",
            "creates",
            "creation",
            "launch",
            "launched",
            "launching",
            "launches",
            "open",
            "opened",
            "opening",
            "opens",
        ] {
            triggers.insert(word.to_string(), ("business:start-org".to_string(), None));
        }

        for word in &[
            "merge",
            "merged",
            "merging",
            "merges",
            "merger",
            "acquire",
            "acquired",
            "acquiring",
            "acquires",
            "acquisition",
            "takeover",
            "take over",
            "took over",
        ] {
            triggers.insert(word.to_string(), ("business:merge-org".to_string(), None));
        }

        for word in &[
            "bankrupt",
            "bankruptcy",
            "collapse",
            "collapsed",
            "collapsing",
            "collapses",
            "fail",
            "failed",
            "failing",
            "fails",
            "failure",
            "close",
            "closed",
            "closing",
            "closes",
            "closure",
            "shutdown",
            "shut down",
        ] {
            triggers.insert(
                word.to_string(),
                (
                    "business:end-org".to_string(),
                    Some(EventPolarity::Negative),
                ),
            );
        }

        // === Contact Events ===
        for word in &[
            "meet",
            "met",
            "meeting",
            "meets",
            "visit",
            "visited",
            "visiting",
            "visits",
            "talk",
            "talked",
            "talking",
            "talks",
            "discuss",
            "discussed",
            "discussing",
            "discusses",
            "discussion",
            "negotiate",
            "negotiated",
            "negotiating",
            "negotiates",
            "negotiation",
            "summit",
        ] {
            triggers.insert(word.to_string(), ("contact:meet".to_string(), None));
        }

        for word in &[
            "call",
            "called",
            "calling",
            "calls",
            "phone",
            "phoned",
            "phoning",
            "phones",
            "email",
            "emailed",
            "emailing",
            "emails",
            "write",
            "wrote",
            "writing",
            "writes",
            "announce",
            "announced",
            "announcing",
            "announces",
            "announcement",
            "say",
            "said",
            "saying",
            "says",
            "state",
            "stated",
            "stating",
            "states",
            "statement",
            "declare",
            "declared",
            "declaring",
            "declares",
            "declaration",
            "report",
            "reported",
            "reporting",
            "reports",
            "claim",
            "claimed",
            "claiming",
            "claims",
        ] {
            triggers.insert(word.to_string(), ("contact:communicate".to_string(), None));
        }

        // === Personnel Events ===
        for word in &[
            "hire",
            "hired",
            "hiring",
            "hires",
            "appoint",
            "appointed",
            "appointing",
            "appoints",
            "appointment",
            "promote",
            "promoted",
            "promoting",
            "promotes",
            "promotion",
            "elect",
            "elected",
            "electing",
            "elects",
            "election",
            "nominate",
            "nominated",
            "nominating",
            "nominates",
            "nomination",
        ] {
            triggers.insert(
                word.to_string(),
                ("personnel:start-position".to_string(), None),
            );
        }

        for word in &[
            "resign",
            "resigned",
            "resigning",
            "resigns",
            "resignation",
            "retire",
            "retired",
            "retiring",
            "retires",
            "retirement",
            "fire",
            "fired",
            "firing",
            "fires",
            "dismiss",
            "dismissed",
            "dismissing",
            "dismisses",
            "dismissal",
            // Multi-word triggers don't work with single-word tokenizer, so add individual forms
            "lay",
            "laid",
            "layoff",
            "layoffs",
            "quit",
            "quitting",
            "quits",
            "leave",
            "left",
            "leaving",
        ] {
            triggers.insert(
                word.to_string(),
                ("personnel:end-position".to_string(), None),
            );
        }

        // === Justice Events ===
        for word in &[
            "arrest",
            "arrested",
            "arresting",
            "arrests",
            "detain",
            "detained",
            "detaining",
            "detains",
            "detention",
            "jail",
            "jailed",
            "jailing",
            "jails",
            "imprison",
            "imprisoned",
            "imprisoning",
            "imprisons",
            "imprisonment",
            "capture",
            "captured",
            "capturing",
            "captures",
        ] {
            triggers.insert(word.to_string(), ("justice:arrest".to_string(), None));
        }

        for word in &[
            "charge",
            "charged",
            "charging",
            "charges",
            "indict",
            "indicted",
            "indicting",
            "indicts",
            "indictment",
            "accuse",
            "accused",
            "accusing",
            "accuses",
            "accusation",
            "prosecute",
            "prosecuted",
            "prosecuting",
            "prosecutes",
            "prosecution",
        ] {
            triggers.insert(word.to_string(), ("justice:charge".to_string(), None));
        }

        for word in &[
            "convict",
            "convicted",
            "convicting",
            "convicts",
            "conviction",
            "sentence",
            "sentenced",
            "sentencing",
            "sentences",
            "fine",
            "fined",
            "fining",
            "fines",
            "acquit",
            "acquitted",
            "acquitting",
            "acquits",
            "acquittal",
            "pardon",
            "pardoned",
            "pardoning",
            "pardons",
        ] {
            triggers.insert(word.to_string(), ("justice:convict".to_string(), None));
        }

        for word in &[
            "sue",
            "sued",
            "suing",
            "sues",
            "lawsuit",
            "litigate",
            "litigated",
            "litigating",
            "litigates",
            "litigation",
        ] {
            triggers.insert(word.to_string(), ("justice:sue".to_string(), None));
        }

        for word in &[
            "release",
            "released",
            "releasing",
            "releases",
            "free",
            "freed",
            "freeing",
            "frees",
            "parole",
            "paroled",
            "paroling",
            "paroles",
        ] {
            triggers.insert(word.to_string(), ("justice:release".to_string(), None));
        }

        // === Life Events ===
        for word in &[
            "born",
            "birth",
            "die",
            "died",
            "dying",
            "dies",
            "death",
            "kill",
            "killed",
            "killing",
            "kills",
            "injure",
            "injured",
            "injuring",
            "injures",
            "injury",
            "wound",
            "wounded",
            "wounding",
            "wounds",
            "marry",
            "married",
            "marrying",
            "marries",
            "marriage",
            "wedding",
            "divorce",
            "divorced",
            "divorcing",
            "divorces",
        ] {
            triggers.insert(word.to_string(), ("life:event".to_string(), None));
        }

        // === Natural Disaster Events ===
        for word in &[
            "earthquake",
            "quake",
            "flood",
            "flooded",
            "flooding",
            "floods",
            "hurricane",
            "typhoon",
            "cyclone",
            "storm",
            "tornado",
            "tornadoes",
            "wildfire",
            "fire",
            "fires",
            "tsunami",
            "drought",
            "eruption",
            "erupted",
            "erupting",
            "erupts",
            "landslide",
            "avalanche",
        ] {
            triggers.insert(word.to_string(), ("disaster:natural".to_string(), None));
        }

        // === Technical/System Events ===
        for word in &[
            "crash",
            "crashed",
            "crashing",
            "crashes",
            "fail",
            "failed",
            "failing",
            "fails",
            "failure",
            "break",
            "broke",
            "breaking",
            "breaks",
            "breakdown",
            "malfunction",
            "malfunctioned",
            "malfunctioning",
            "outage",
            "hack",
            "hacked",
            "hacking",
            "hacks",
            "breach",
            "breached",
            "breaching",
            "breaches",
        ] {
            triggers.insert(word.to_string(), ("technical:failure".to_string(), None));
        }

        // === Economic Events ===
        for word in &[
            "rise",
            "rose",
            "rising",
            "rises",
            "fall",
            "fell",
            "falling",
            "falls",
            "increase",
            "increased",
            "increasing",
            "increases",
            "decrease",
            "decreased",
            "decreasing",
            "decreases",
            "grow",
            "grew",
            "growing",
            "grows",
            "growth",
            "shrink",
            "shrank",
            "shrinking",
            "shrinks",
            "spike",
            "spiked",
            "spiking",
            "spikes",
            "plunge",
            "plunged",
            "plunging",
            "plunges",
            "surge",
            "surged",
            "surging",
            "surges",
            "drop",
            "dropped",
            "dropping",
            "drops",
        ] {
            triggers.insert(word.to_string(), ("economic:change".to_string(), None));
        }

        // Modal verbs for hypothetical detection
        let modal_verbs = vec!["might", "may", "could", "would", "should", "can", "will"]
            .into_iter()
            .map(String::from)
            .collect();

        // Negation words
        let negation_words = vec![
            "not",
            "never",
            "no",
            "none",
            "neither",
            "nobody",
            "nothing",
            "nowhere",
            "hardly",
            "scarcely",
            "barely",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "wouldn't",
            "couldn't",
            "shouldn't",
            "can't",
            "cannot",
            "hasn't",
            "haven't",
            "hadn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self {
            triggers,
            modal_verbs,
            negation_words,
        }
    }

    /// Look up a trigger in the lexicon.
    #[must_use]
    pub fn lookup(&self, word: &str) -> Option<(&str, Option<EventPolarity>)> {
        let lower = word.to_lowercase();
        self.triggers.get(&lower).map(|(t, p)| (t.as_str(), *p))
    }

    /// Check if a word is a modal verb.
    #[must_use]
    pub fn is_modal(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.modal_verbs.iter().any(|m| m == &lower)
    }

    /// Check if a word is a negation.
    #[must_use]
    pub fn is_negation(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.negation_words.iter().any(|n| n == &lower)
    }
}

// =============================================================================
// Event Extractor
// =============================================================================

/// Configuration for event extraction.
#[derive(Debug, Clone)]
pub struct EventExtractorConfig {
    /// Minimum confidence threshold for extraction
    pub min_confidence: f64,
    /// Maximum tokens to look back for arguments
    pub max_arg_distance: usize,
    /// Whether to extract nested events
    pub extract_nested: bool,
    /// GLiNER model ID for neural extraction (None = rule-based only)
    #[cfg(feature = "candle")]
    pub gliner_model: Option<String>,
    /// Event type labels for GLiNER (zero-shot)
    pub event_labels: Vec<String>,
}

impl Default for EventExtractorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_arg_distance: 10,
            extract_nested: false,
            #[cfg(feature = "candle")]
            gliner_model: None,
            event_labels: Self::default_event_labels(),
        }
    }
}

impl EventExtractorConfig {
    /// Default event type labels for zero-shot extraction.
    fn default_event_labels() -> Vec<String> {
        vec![
            "conflict event".into(),      // attack, war, bombing
            "movement event".into(),      // travel, transport, deploy
            "transaction event".into(),   // buy, sell, transfer
            "business event".into(),      // start, merge, bankruptcy
            "communication event".into(), // announce, state, report
            "personnel event".into(),     // hire, fire, resign
            "justice event".into(),       // arrest, charge, convict
            "life event".into(),          // birth, death, marriage
            "disaster event".into(),      // earthquake, flood, fire
            "economic change".into(),     // rise, fall, growth
        ]
    }

    /// Enable GLiNER neural extraction with a model ID.
    #[cfg(feature = "candle")]
    #[must_use]
    pub fn with_gliner(mut self, model_id: impl Into<String>) -> Self {
        self.gliner_model = Some(model_id.into());
        self
    }

    /// Set custom event labels for zero-shot extraction.
    #[must_use]
    pub fn with_event_labels(mut self, labels: Vec<String>) -> Self {
        self.event_labels = labels;
        self
    }

    /// Set minimum confidence threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.min_confidence = threshold;
        self
    }
}

/// Event extractor with optional neural backend.
///
/// **Rule-based** (default): Uses ACE 2005 trigger lexicon for fast extraction.
/// **Neural** (with `candle` feature): Uses GLiNER bi-encoder for zero-shot extraction.
///
/// The neural backend is preferred when available as it generalizes better
/// to unseen event types and handles ambiguity.
#[derive(Debug)]
pub struct EventExtractor {
    lexicon: EventTriggerLexicon,
    config: EventExtractorConfig,
    /// GLiNER model for neural extraction (when candle feature enabled)
    #[cfg(feature = "candle")]
    gliner: Option<GLiNERCandle>,
}

// Manual Clone implementation since GLiNERCandle doesn't implement Clone
impl Clone for EventExtractor {
    fn clone(&self) -> Self {
        Self {
            lexicon: self.lexicon.clone(),
            config: self.config.clone(),
            #[cfg(feature = "candle")]
            gliner: None, // GLiNER models are not cloneable, reset to None
        }
    }
}

impl Default for EventExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl EventExtractor {
    /// Create a new event extractor with default lexicon (rule-based only).
    #[must_use]
    pub fn new() -> Self {
        Self {
            lexicon: EventTriggerLexicon::new(),
            config: EventExtractorConfig::default(),
            #[cfg(feature = "candle")]
            gliner: None,
        }
    }

    /// Create with custom configuration.
    ///
    /// If `config.gliner_model` is set and `candle` feature is enabled,
    /// will attempt to load the GLiNER model for neural extraction.
    pub fn with_config(config: EventExtractorConfig) -> crate::Result<Self> {
        #[cfg(feature = "candle")]
        let gliner = if let Some(ref model_id) = config.gliner_model {
            log::info!("[EventExtractor] Loading GLiNER model: {}", model_id);
            Some(GLiNERCandle::from_pretrained(model_id)?)
        } else {
            None
        };

        Ok(Self {
            lexicon: EventTriggerLexicon::new(),
            config,
            #[cfg(feature = "candle")]
            gliner,
        })
    }

    /// Create with GLiNER model for neural extraction.
    #[cfg(feature = "candle")]
    pub fn with_gliner(model_id: &str) -> crate::Result<Self> {
        let config = EventExtractorConfig::default().with_gliner(model_id);
        Self::with_config(config)
    }

    /// Check if neural extraction is available.
    #[must_use]
    pub fn has_neural_backend(&self) -> bool {
        #[cfg(feature = "candle")]
        {
            self.gliner.is_some()
        }
        #[cfg(not(feature = "candle"))]
        {
            false
        }
    }

    /// Extract events from text.
    ///
    /// Uses GLiNER (neural) if available, otherwise falls back to rule-based extraction.
    #[must_use]
    pub fn extract(&self, text: &str) -> Vec<EventMention> {
        // Try neural extraction first if available
        #[cfg(feature = "candle")]
        if let Some(ref gliner) = self.gliner {
            if let Ok(events) = self.extract_with_gliner(gliner, text) {
                return events;
            }
            // Fall through to rule-based if GLiNER fails
            log::warn!("[EventExtractor] GLiNER extraction failed, falling back to rules");
        }

        // Rule-based extraction
        self.extract_rule_based(text)
    }

    /// Extract events using GLiNER zero-shot model.
    #[cfg(feature = "candle")]
    fn extract_with_gliner(
        &self,
        gliner: &GLiNERCandle,
        text: &str,
    ) -> crate::Result<Vec<EventMention>> {
        // Use GLiNER to extract spans matching event type labels
        let entities = gliner.extract(
            text,
            &self
                .config
                .event_labels
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            self.config.min_confidence as f32,
        )?;

        // Convert GLiNER entities to EventMentions
        let words = self.tokenize(text);
        let mut events = Vec::new();

        for entity in entities {
            // Map entity type to ACE event type
            let event_type = self.map_gliner_label_to_event_type(&entity.entity_type.to_string());

            // Find word index for polarity/tense detection
            let word_idx = words
                .iter()
                .position(|(_, start, end)| *start <= entity.start && *end >= entity.end)
                .unwrap_or(0);

            let polarity = self.detect_polarity(&words, word_idx, None);
            let tense = self.detect_tense(&words, word_idx, &entity.text);
            let arguments = self.extract_arguments(text, &words, word_idx);

            let mut event = EventMention::new(&entity.text, entity.start, entity.end)
                .with_trigger_type(event_type)
                .with_polarity(polarity)
                .with_confidence(entity.confidence);

            if let Some(t) = tense {
                event = event.with_tense(t);
            }

            if !arguments.is_empty() {
                event.arguments = arguments;
            }

            events.push(event);
        }

        Ok(events)
    }

    /// Map GLiNER label to ACE-style event type.
    #[cfg(feature = "candle")]
    fn map_gliner_label_to_event_type(&self, label: &str) -> &'static str {
        let label_lower = label.to_lowercase();

        if label_lower.contains("conflict") || label_lower.contains("attack") {
            "conflict:attack"
        } else if label_lower.contains("movement") || label_lower.contains("travel") {
            "movement:transport"
        } else if label_lower.contains("transaction") || label_lower.contains("transfer") {
            "transaction:transfer"
        } else if label_lower.contains("business") || label_lower.contains("company") {
            "business:event"
        } else if label_lower.contains("communication") || label_lower.contains("announce") {
            "contact:communicate"
        } else if label_lower.contains("personnel") || label_lower.contains("hire") {
            "personnel:event"
        } else if label_lower.contains("justice") || label_lower.contains("legal") {
            "justice:event"
        } else if label_lower.contains("life")
            || label_lower.contains("birth")
            || label_lower.contains("death")
        {
            "life:event"
        } else if label_lower.contains("disaster") || label_lower.contains("earthquake") {
            "disaster:natural"
        } else if label_lower.contains("economic") || label_lower.contains("change") {
            "economic:change"
        } else {
            "event:generic"
        }
    }

    /// Rule-based event extraction using trigger lexicon.
    fn extract_rule_based(&self, text: &str) -> Vec<EventMention> {
        let mut events = Vec::new();
        let words = self.tokenize(text);

        for (word_idx, (word, start, end)) in words.iter().enumerate() {
            if let Some((event_type, polarity_hint)) = self.lexicon.lookup(word) {
                // Determine polarity from context
                let polarity = self.detect_polarity(&words, word_idx, polarity_hint);

                // Determine tense from context
                let tense = self.detect_tense(&words, word_idx, word);

                // Extract arguments (entities near the trigger)
                let arguments = self.extract_arguments(text, &words, word_idx);

                let mut event = EventMention::new(word.clone(), *start, *end)
                    .with_trigger_type(event_type)
                    .with_polarity(polarity)
                    .with_confidence(0.8);

                if let Some(t) = tense {
                    event = event.with_tense(t);
                }

                if !arguments.is_empty() {
                    event.arguments = arguments;
                }

                events.push(event);
            }
        }

        events
    }

    /// Extract events and convert to discourse referents.
    #[must_use]
    pub fn extract_referents(&self, text: &str) -> Vec<DiscourseReferent> {
        let events = self.extract(text);

        events
            .into_iter()
            .map(|event| {
                // Expand the span to include the full clause
                // Note: find_clause_span returns byte offsets, but DiscourseReferent expects char offsets
                let (clause_byte_start, clause_byte_end) =
                    self.find_clause_span(text, event.trigger_start, event.trigger_end);

                // Convert byte offsets to character offsets
                let clause_char_start = text[..clause_byte_start].chars().count();
                let clause_text = &text[clause_byte_start..clause_byte_end];
                let clause_char_end = clause_char_start + clause_text.chars().count();

                DiscourseReferent::new(ReferentType::Event, clause_char_start, clause_char_end)
                    .with_event(event.clone())
                    .with_text(clause_text)
                    .with_confidence(event.confidence)
            })
            .collect()
    }

    /// Extract events with entity context (for argument filling).
    ///
    /// When entities are provided, they OVERRIDE heuristic argument extraction
    /// since NER entities are more reliable than capitalization heuristics.
    #[must_use]
    pub fn extract_with_entities(&self, text: &str, entities: &[Entity]) -> Vec<EventMention> {
        let mut events = self.extract(text);

        // Fill in arguments from entities (prefer over heuristics)
        for event in &mut events {
            let event_span = (event.trigger_start, event.trigger_end);

            // Find entities before the trigger (potential agents)
            let agents: Vec<_> = entities
                .iter()
                .filter(|e| e.end <= event_span.0 && event_span.0 - e.end < 50)
                .collect();

            // Find entities after the trigger (potential patients)
            let patients: Vec<_> = entities
                .iter()
                .filter(|e| e.start >= event_span.1 && e.start - event_span.1 < 50)
                .collect();

            // OVERRIDE heuristic Agent with entity-based Agent (entities are more reliable)
            if let Some(agent) = agents.last() {
                // Remove any existing Agent from heuristics
                event.arguments.retain(|(r, _)| r != "Agent");
                event
                    .arguments
                    .push(("Agent".to_string(), agent.text.clone()));
            }

            // OVERRIDE heuristic Patient with entity-based Patient
            if let Some(patient) = patients.first() {
                // Remove any existing Patient from heuristics
                event.arguments.retain(|(r, _)| r != "Patient");
                event
                    .arguments
                    .push(("Patient".to_string(), patient.text.clone()));
            }
        }

        events
    }

    /// Simple tokenizer that preserves character offsets.
    fn tokenize(&self, text: &str) -> Vec<(String, usize, usize)> {
        let mut tokens = Vec::new();
        let mut word_start = None;

        for (i, c) in text.char_indices() {
            if c.is_alphanumeric() || c == '\'' || c == '-' {
                if word_start.is_none() {
                    word_start = Some(i);
                }
            } else if let Some(start) = word_start {
                let word = &text[start..i];
                tokens.push((word.to_string(), start, i));
                word_start = None;
            }
        }

        // Handle last word
        if let Some(start) = word_start {
            let word = &text[start..];
            tokens.push((word.to_string(), start, text.len()));
        }

        tokens
    }

    /// Detect polarity from surrounding context.
    fn detect_polarity(
        &self,
        words: &[(String, usize, usize)],
        trigger_idx: usize,
        hint: Option<EventPolarity>,
    ) -> EventPolarity {
        // Check for negation in the preceding 3 words
        let start = trigger_idx.saturating_sub(3);
        for (word, _, _) in &words[start..trigger_idx] {
            if self.lexicon.is_negation(word) {
                return EventPolarity::Negative;
            }
        }

        // Check for modals (uncertain)
        for (word, _, _) in &words[start..trigger_idx] {
            if self.lexicon.is_modal(word) {
                return EventPolarity::Uncertain;
            }
        }

        hint.unwrap_or(EventPolarity::Positive)
    }

    /// Detect tense from verb form and context.
    fn detect_tense(
        &self,
        words: &[(String, usize, usize)],
        trigger_idx: usize,
        trigger: &str,
    ) -> Option<EventTense> {
        let trigger_lower = trigger.to_lowercase();

        // Check for future markers
        let start = trigger_idx.saturating_sub(3);
        for (word, _, _) in &words[start..trigger_idx] {
            let w = word.to_lowercase();
            if w == "will" || w == "going" || w == "shall" {
                return Some(EventTense::Future);
            }
            if w == "would" || w == "could" || w == "might" || w == "may" {
                return Some(EventTense::Hypothetical);
            }
        }

        // Past tense heuristics
        if trigger_lower.ends_with("ed")
            || matches!(
                trigger_lower.as_str(),
                "went"
                    | "came"
                    | "said"
                    | "took"
                    | "gave"
                    | "made"
                    | "got"
                    | "found"
                    | "knew"
                    | "thought"
                    | "felt"
                    | "became"
                    | "left"
                    | "held"
                    | "brought"
                    | "began"
                    | "kept"
                    | "put"
                    | "set"
                    | "saw"
                    | "heard"
                    | "told"
                    | "stood"
                    | "lost"
                    | "paid"
                    | "met"
                    | "ran"
                    | "sent"
                    | "built"
                    | "fell"
                    | "caught"
                    | "wrote"
                    | "sat"
                    | "led"
                    | "rose"
                    | "spoke"
                    | "won"
                    | "broke"
                    | "spent"
                    | "hit"
                    | "cut"
                    | "sold"
                    | "bought"
                    | "shot"
                    | "struck"
                    | "shut"
                    | "threw"
                    | "drove"
                    | "flew"
                    | "drew"
                    | "grew"
                    | "sang"
                    | "swam"
                    | "rang"
                    | "wore"
                    | "chose"
                    | "woke"
                    | "froze"
                    | "stole"
                    | "blew"
                    | "ate"
                    | "drank"
                    | "rode"
                    | "shook"
                    | "bit"
                    | "hid"
                    | "tore"
                    | "beat"
                    | "laid"
                    | "spread"
                    | "hurt"
                    | "fought"
                    | "hung"
                    | "slept"
                    | "swept"
                    | "bent"
                    | "dealt"
                    | "fed"
                    | "fled"
                    | "dug"
                    | "spun"
                    | "wove"
                    | "sank"
                    | "shone"
                    | "swung"
                    | "clung"
                    | "crept"
                    | "burnt"
                    | "leapt"
                    | "meant"
                    | "lent"
                    | "dwelt"
                    | "dreamt"
                    | "knelt"
                    | "split"
                    | "spit"
                    | "bid"
                    | "forbid"
                    | "shed"
                    | "rid"
                    | "burst"
                    | "stuck"
                    | "slid"
            )
        {
            return Some(EventTense::Past);
        }

        // Present tense (default for base form)
        if trigger_lower.ends_with("ing") {
            return Some(EventTense::Present);
        }

        None
    }

    /// Extract arguments using simple position heuristics.
    fn extract_arguments(
        &self,
        _text: &str,
        words: &[(String, usize, usize)],
        trigger_idx: usize,
    ) -> Vec<(String, String)> {
        let mut arguments = Vec::new();

        // Look for capitalized words before trigger (potential Agent)
        if trigger_idx > 0 {
            for (word, _, _) in words[..trigger_idx].iter().rev().take(5) {
                if word
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                    && word.len() > 1
                    && !Self::is_sentence_start_word(word)
                {
                    arguments.push(("Agent".to_string(), word.clone()));
                    break;
                }
            }
        }

        // Look for capitalized words after trigger (potential Patient/Theme)
        if trigger_idx + 1 < words.len() {
            for (word, _, _) in words[trigger_idx + 1..].iter().take(5) {
                if word
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                    && word.len() > 1
                {
                    arguments.push(("Patient".to_string(), word.clone()));
                    break;
                }
            }
        }

        arguments
    }

    /// Check if word is likely a sentence-starting word.
    fn is_sentence_start_word(word: &str) -> bool {
        matches!(
            word.to_lowercase().as_str(),
            "the"
                | "a"
                | "an"
                | "this"
                | "that"
                | "these"
                | "those"
                | "it"
                | "he"
                | "she"
                | "they"
                | "we"
                | "i"
        )
    }

    /// Find the clause span containing a trigger.
    ///
    /// # Arguments
    /// * `text` - The full text
    /// * `trigger_start` - Character offset of trigger start
    /// * `trigger_end` - Character offset of trigger end
    ///
    /// # Returns
    /// Byte offsets (start, end) of the clause containing the trigger
    fn find_clause_span(
        &self,
        text: &str,
        trigger_start: usize,
        trigger_end: usize,
    ) -> (usize, usize) {
        // Convert character offsets to byte offsets
        let trigger_byte_start = text
            .char_indices()
            .nth(trigger_start)
            .map(|(byte_idx, _)| byte_idx)
            .unwrap_or(text.len());
        let trigger_byte_end = text
            .char_indices()
            .nth(trigger_end)
            .map(|(byte_idx, _)| byte_idx)
            .unwrap_or(text.len());

        // Look backwards for clause boundary (using byte offsets)
        let mut clause_start = 0;
        for (i, c) in text[..trigger_byte_start].char_indices().rev() {
            if matches!(c, '.' | '!' | '?' | ';' | ':') {
                clause_start = i + 1;
                // Skip whitespace
                while clause_start < trigger_byte_start
                    && text[clause_start..].starts_with(char::is_whitespace)
                {
                    clause_start += 1;
                }
                break;
            }
        }

        // Look forwards for clause boundary (using byte offsets)
        let mut clause_end = text.len();
        for (i, c) in text[trigger_byte_end..].char_indices() {
            if matches!(c, '.' | '!' | '?' | ';' | ':' | ',') {
                clause_end = trigger_byte_end + i + 1;
                break;
            }
        }

        (clause_start, clause_end)
    }

    /// Get the lexicon for inspection or modification.
    #[must_use]
    pub fn lexicon(&self) -> &EventTriggerLexicon {
        &self.lexicon
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extraction() {
        let extractor = EventExtractor::new();
        let events = extractor.extract("Russia invaded Ukraine in 2022.");

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].trigger, "invaded");
        assert_eq!(events[0].trigger_type.as_deref(), Some("conflict:attack"));
        assert_eq!(events[0].tense, Some(EventTense::Past));
    }

    #[test]
    fn test_multiple_events() {
        let extractor = EventExtractor::new();
        let events = extractor.extract("The company announced layoffs. Stocks crashed.");

        // May extract "announced", "layoffs" (personnel event noun), and "crashed"
        assert!(
            events.len() >= 2,
            "Should extract at least 2 events, got {}",
            events.len()
        );
        assert!(events.iter().any(|e| e.trigger == "announced"));
        assert!(events.iter().any(|e| e.trigger == "crashed"));
    }

    #[test]
    fn test_negation_detection() {
        let extractor = EventExtractor::new();

        let events = extractor.extract("They did not attack the city.");
        assert_eq!(events[0].polarity, EventPolarity::Negative);

        let events = extractor.extract("They attacked the city.");
        assert_eq!(events[0].polarity, EventPolarity::Positive);
    }

    #[test]
    fn test_hypothetical_detection() {
        let extractor = EventExtractor::new();

        let events = extractor.extract("They might attack tomorrow.");
        assert_eq!(events[0].polarity, EventPolarity::Uncertain);
        assert_eq!(events[0].tense, Some(EventTense::Hypothetical));
    }

    #[test]
    fn test_argument_extraction() {
        let extractor = EventExtractor::new();
        let events = extractor.extract("Russia invaded Ukraine.");

        assert!(!events.is_empty());
        let event = &events[0];

        // Should extract Russia as Agent and Ukraine as Patient
        assert!(event
            .arguments
            .iter()
            .any(|(r, v)| r == "Agent" && v == "Russia"));
        assert!(event
            .arguments
            .iter()
            .any(|(r, v)| r == "Patient" && v == "Ukraine"));
    }

    #[test]
    fn test_discourse_referent_extraction() {
        let extractor = EventExtractor::new();
        let referents =
            extractor.extract_referents("The earthquake struck at dawn. This shocked everyone.");

        assert!(!referents.is_empty());
        let referent = &referents[0];

        assert_eq!(referent.referent_type, ReferentType::Event);
        assert!(referent.event.is_some());
        assert!(referent.text.as_ref().unwrap().contains("earthquake"));
    }

    #[test]
    fn test_lexicon_coverage() {
        let lexicon = EventTriggerLexicon::new();

        // Test various event types
        assert!(lexicon.lookup("invaded").is_some());
        assert!(lexicon.lookup("announced").is_some());
        assert!(lexicon.lookup("arrested").is_some());
        assert!(lexicon.lookup("earthquake").is_some());
        assert!(lexicon.lookup("crashed").is_some());

        // Test negation/modal detection
        assert!(lexicon.is_negation("not"));
        assert!(lexicon.is_negation("never"));
        assert!(lexicon.is_modal("might"));
        assert!(lexicon.is_modal("could"));
    }

    #[test]
    fn test_with_entities() {
        let extractor = EventExtractor::new();
        let text = "Apple Inc. announced record profits.";

        let entities = vec![Entity::new(
            "Apple Inc.",
            anno_core::EntityType::Organization,
            0,
            10,
            0.9,
        )];

        let events = extractor.extract_with_entities(text, &entities);

        assert!(!events.is_empty(), "Should extract at least one event");
        let announcement = events.iter().find(|e| e.trigger == "announced");
        assert!(announcement.is_some(), "Should find 'announced' event");

        let event = announcement.unwrap();
        // extract_with_entities should fill in entity arguments
        assert!(
            event
                .arguments
                .iter()
                .any(|(r, v)| r == "Agent" && v.contains("Apple")),
            "Should extract Apple Inc. as Agent, got: {:?}",
            event.arguments
        );
    }

    #[test]
    fn test_has_neural_backend() {
        let extractor = EventExtractor::new();

        // Without candle feature or GLiNER model, should be false
        #[cfg(not(feature = "candle"))]
        assert!(!extractor.has_neural_backend());

        // With candle but no model loaded, should still be false
        #[cfg(feature = "candle")]
        assert!(!extractor.has_neural_backend());
    }

    #[test]
    fn test_config_builder() {
        let config = EventExtractorConfig::default()
            .with_threshold(0.7)
            .with_event_labels(vec!["custom event".into()]);

        assert_eq!(config.min_confidence, 0.7);
        assert_eq!(config.event_labels, vec!["custom event"]);
    }
}
