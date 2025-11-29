//! Abstract anaphora evaluation infrastructure.
//!
//! Demonstrates the gap between nominal coreference (which we handle)
//! and abstract anaphora (which we don't).
//!
//! # The Problem
//!
//! Standard coref links noun phrases: "John" → "he"
//! Abstract anaphora links to events/propositions: "The invasion" → "This"
//!
//! # Research Context
//!
//! See `docs/ABSTRACT_ANAPHORA_RESEARCH.md` for full background.
//!
//! ## Key Papers
//!
//! - Kolhatkar & Hirst (2012): "Resolving 'this-issue' anaphors" - 92% antecedent recall
//! - Marasović et al. (2017): LSTM-Siamese model, EMNLP, outperforms on shell nouns
//! - Moosavi & Strube (2016): LEA metric addresses "mention identification effect"
//!
//! # Example
//!
//! ```rust
//! use anno::eval::abstract_anaphora::{
//!     AbstractAnaphoraDataset, AbstractAnaphoraEvaluator, AnaphoraType
//! };
//! use anno::eval::coref_resolver::SimpleCorefResolver;
//!
//! let dataset = AbstractAnaphoraDataset::default();
//! let resolver = SimpleCorefResolver::default();
//! let evaluator = AbstractAnaphoraEvaluator::new(resolver);
//!
//! let results = evaluator.evaluate(&dataset);
//! println!("Nominal accuracy: {:.1}%", results.nominal_accuracy * 100.0);
//! println!("Abstract accuracy: {:.1}%", results.abstract_accuracy * 100.0);
//! // Expected: Nominal ~70%, Abstract ~0%
//! ```

use crate::discourse::{classify_shell_noun, ReferentType, ShellNoun, ShellNounClass};
use crate::eval::coref::{CorefChain, Mention};
use crate::eval::coref_metrics::{lea_score, CorefScores};
use crate::eval::coref_resolver::{
    DiscourseAwareResolver, DiscourseCorefConfig, SimpleCorefResolver,
};
use crate::{Entity, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Anaphora Types
// =============================================================================

/// Type of anaphoric reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnaphoraType {
    /// Standard nominal coreference: "John" → "he"
    Nominal,
    /// Event anaphora: "The crash happened" → "This shocked everyone"
    Event,
    /// Fact anaphora: "He won" → "This is undeniable"
    Fact,
    /// Proposition anaphora: "She might leave" → "This worries me"
    Proposition,
    /// Situation anaphora: "Prices rose while wages fell" → "This was unsustainable"
    Situation,
}

impl AnaphoraType {
    /// Is this an abstract (non-nominal) anaphora type?
    #[must_use]
    pub const fn is_abstract(&self) -> bool {
        !matches!(self, AnaphoraType::Nominal)
    }

    /// Human-readable name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            AnaphoraType::Nominal => "nominal",
            AnaphoraType::Event => "event",
            AnaphoraType::Fact => "fact",
            AnaphoraType::Proposition => "proposition",
            AnaphoraType::Situation => "situation",
        }
    }
}

// =============================================================================
// Test Cases
// =============================================================================

/// A single anaphora test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnaphoraTestCase {
    /// Unique identifier
    pub id: String,
    /// Full text containing antecedent and anaphor
    pub text: String,
    /// The antecedent span (what the anaphor refers to)
    pub antecedent: AntecedentSpan,
    /// The anaphoric expression
    pub anaphor: AnaphorSpan,
    /// Type of anaphora
    pub anaphora_type: AnaphoraType,
    /// Expected: should resolver link these?
    pub should_resolve: bool,
    /// Notes on why this case is interesting
    pub notes: Option<String>,
}

/// The antecedent (what is being referred to).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntecedentSpan {
    /// Text of the antecedent
    pub text: String,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
    /// For events: the trigger verb/noun
    pub trigger: Option<String>,
}

/// The anaphoric expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnaphorSpan {
    /// Text of the anaphor ("this", "that", "it", etc.)
    pub text: String,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
}

impl AnaphoraTestCase {
    /// Create a new test case.
    pub fn new(
        id: impl Into<String>,
        text: impl Into<String>,
        antecedent: AntecedentSpan,
        anaphor: AnaphorSpan,
        anaphora_type: AnaphoraType,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            antecedent,
            anaphor,
            anaphora_type,
            should_resolve: true,
            notes: None,
        }
    }

    /// Add notes to explain the test case.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

// =============================================================================
// Dataset
// =============================================================================

/// Dataset of anaphora test cases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AbstractAnaphoraDataset {
    /// All test cases
    pub cases: Vec<AnaphoraTestCase>,
}

impl AbstractAnaphoraDataset {
    /// Create a new empty dataset.
    #[must_use]
    pub fn new() -> Self {
        Self { cases: Vec::new() }
    }

    /// Add a test case.
    pub fn add(&mut self, case: AnaphoraTestCase) {
        self.cases.push(case);
    }

    /// Get cases by type.
    #[must_use]
    pub fn by_type(&self, anaphora_type: AnaphoraType) -> Vec<&AnaphoraTestCase> {
        self.cases
            .iter()
            .filter(|c| c.anaphora_type == anaphora_type)
            .collect()
    }

    /// Get all nominal (non-abstract) cases.
    #[must_use]
    pub fn nominal_cases(&self) -> Vec<&AnaphoraTestCase> {
        self.by_type(AnaphoraType::Nominal)
    }

    /// Get all abstract cases.
    #[must_use]
    pub fn abstract_cases(&self) -> Vec<&AnaphoraTestCase> {
        self.cases
            .iter()
            .filter(|c| c.anaphora_type.is_abstract())
            .collect()
    }

    /// Create a standard evaluation dataset.
    ///
    /// Contains a balanced mix of nominal and abstract anaphora cases.
    #[must_use]
    pub fn standard() -> Self {
        let mut dataset = Self::new();

        // =================================================================
        // NOMINAL COREFERENCE (Baseline - should work)
        // =================================================================

        dataset.add(
            AnaphoraTestCase::new(
                "nom_01",
                "John Smith went to the store. He bought milk.",
                AntecedentSpan {
                    text: "John Smith".to_string(),
                    start: 0,
                    end: 10,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "He".to_string(),
                    start: 32,
                    end: 34,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Simple pronoun resolution - baseline case"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "nom_02",
                "Microsoft announced layoffs. The company cited economic conditions.",
                AntecedentSpan {
                    text: "Microsoft".to_string(),
                    start: 0,
                    end: 9,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "The company".to_string(),
                    start: 29,
                    end: 40,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Definite NP resolution"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "nom_03",
                "Dr. Sarah Chen published a paper. She presented it at EMNLP.",
                AntecedentSpan {
                    text: "Dr. Sarah Chen".to_string(),
                    start: 0,
                    end: 14,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "She".to_string(),
                    start: 35,
                    end: 38,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Pronoun with title prefix"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "nom_04",
                "The CEO of Nvidia is Jensen Huang. He co-founded the company.",
                AntecedentSpan {
                    text: "Jensen Huang".to_string(),
                    start: 20,
                    end: 32,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "He".to_string(),
                    start: 34,
                    end: 36,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Pronoun binds to proper name, not role description"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "nom_05",
                "Apple Inc. reported record earnings. Apple's stock rose 5%.",
                AntecedentSpan {
                    text: "Apple Inc.".to_string(),
                    start: 0,
                    end: 10,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "Apple's".to_string(),
                    start: 37,
                    end: 44,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Possessive form coreference"),
        );

        // =================================================================
        // EVENT ANAPHORA (Should fail - no event detection)
        // =================================================================

        dataset.add(
            AnaphoraTestCase::new(
                "event_01",
                "Russia invaded Ukraine in 2022. This caused a global energy crisis.",
                AntecedentSpan {
                    text: "Russia invaded Ukraine in 2022".to_string(),
                    start: 0,
                    end: 30,
                    trigger: Some("invaded".to_string()),
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 32,
                    end: 36,
                },
                AnaphoraType::Event,
            )
            .with_notes(
                "Classic event anaphora - 'This' refers to invasion EVENT, not Russia or Ukraine",
            ),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "event_02",
                "The earthquake struck at dawn. It destroyed thousands of homes.",
                AntecedentSpan {
                    text: "The earthquake struck at dawn".to_string(),
                    start: 0,
                    end: 29,
                    trigger: Some("struck".to_string()),
                },
                AnaphorSpan {
                    text: "It".to_string(),
                    start: 31,
                    end: 33,
                },
                AnaphoraType::Event,
            )
            .with_notes("'It' refers to the earthquake event, not just the noun 'earthquake'"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "event_03",
                "The merger was announced yesterday. This surprised investors.",
                AntecedentSpan {
                    text: "The merger was announced yesterday".to_string(),
                    start: 0,
                    end: 34,
                    trigger: Some("announced".to_string()),
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 36,
                    end: 40,
                },
                AnaphoraType::Event,
            )
            .with_notes("Announcement event, not the merger entity"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "event_04",
                "Scientists discovered a new species. This happened in the Amazon.",
                AntecedentSpan {
                    text: "Scientists discovered a new species".to_string(),
                    start: 0,
                    end: 35,
                    trigger: Some("discovered".to_string()),
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 37,
                    end: 41,
                },
                AnaphoraType::Event,
            )
            .with_notes("Discovery event"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "event_05",
                "The patient underwent surgery. This took six hours.",
                AntecedentSpan {
                    text: "The patient underwent surgery".to_string(),
                    start: 0,
                    end: 29,
                    trigger: Some("underwent".to_string()),
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 31,
                    end: 35,
                },
                AnaphoraType::Event,
            )
            .with_notes("Medical procedure event"),
        );

        // =================================================================
        // FACT ANAPHORA (Should fail)
        // =================================================================

        dataset.add(
            AnaphoraTestCase::new(
                "fact_01",
                "The Earth orbits the Sun. This is well established.",
                AntecedentSpan {
                    text: "The Earth orbits the Sun".to_string(),
                    start: 0,
                    end: 24,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 26,
                    end: 30,
                },
                AnaphoraType::Fact,
            )
            .with_notes("'This' refers to the FACT, not Earth or Sun"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fact_02",
                "Water boils at 100 degrees Celsius. This is basic chemistry.",
                AntecedentSpan {
                    text: "Water boils at 100 degrees Celsius".to_string(),
                    start: 0,
                    end: 34,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 36,
                    end: 40,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Scientific fact reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fact_03",
                "He lied under oath. This was proven in court.",
                AntecedentSpan {
                    text: "He lied under oath".to_string(),
                    start: 0,
                    end: 18,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 20,
                    end: 24,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Fact about past action"),
        );

        // =================================================================
        // PROPOSITION ANAPHORA (Should fail)
        // =================================================================

        dataset.add(
            AnaphoraTestCase::new(
                "prop_01",
                "She might resign. This worries the board.",
                AntecedentSpan {
                    text: "She might resign".to_string(),
                    start: 0,
                    end: 16,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 18,
                    end: 22,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("'This' refers to the POSSIBILITY of resignation"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "prop_02",
                "The company could go bankrupt. This scenario keeps investors awake.",
                AntecedentSpan {
                    text: "The company could go bankrupt".to_string(),
                    start: 0,
                    end: 29,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This scenario".to_string(),
                    start: 31,
                    end: 44,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Hypothetical proposition"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "prop_03",
                "Interest rates may rise again. This possibility concerns economists.",
                AntecedentSpan {
                    text: "Interest rates may rise again".to_string(),
                    start: 0,
                    end: 29,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This possibility".to_string(),
                    start: 31,
                    end: 47,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Modal proposition"),
        );

        // =================================================================
        // SITUATION ANAPHORA (Should fail)
        // =================================================================

        dataset.add(
            AnaphoraTestCase::new(
                "sit_01",
                "Prices rose while wages fell. This was unsustainable.",
                AntecedentSpan {
                    text: "Prices rose while wages fell".to_string(),
                    start: 0,
                    end: 28,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 30,
                    end: 34,
                },
                AnaphoraType::Situation,
            )
            .with_notes("'This' refers to the combined SITUATION, not prices or wages"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sit_02",
                "Traffic was gridlocked and tempers flared. This chaos lasted hours.",
                AntecedentSpan {
                    text: "Traffic was gridlocked and tempers flared".to_string(),
                    start: 0,
                    end: 41,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This chaos".to_string(),
                    start: 43,
                    end: 53,
                },
                AnaphoraType::Situation,
            )
            .with_notes("Complex situation with multiple aspects"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sit_03",
                "The server crashed, emails were lost, and backups failed. This disaster cost millions.",
                AntecedentSpan {
                    text: "The server crashed, emails were lost, and backups failed".to_string(),
                    start: 0,
                    end: 56,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This disaster".to_string(),
                    start: 58,
                    end: 71,
                },
                AnaphoraType::Situation,
            )
            .with_notes("Multi-clause situation"),
        );

        dataset
    }

    /// Extended dataset with ARRAU-style shell noun examples.
    ///
    /// ARRAU corpus annotates "discourse deixis" which includes shell nouns
    /// like "this issue", "the problem", "that decision".
    #[must_use]
    pub fn extended() -> Self {
        let mut dataset = Self::standard();

        // =================================================================
        // SHELL NOUN CASES (based on Schmid 2000 taxonomy)
        // =================================================================

        // Factual shell nouns
        dataset.add(
            AnaphoraTestCase::new(
                "shell_fact_01",
                "The GDP grew by 3%. This fact surprised analysts.",
                AntecedentSpan {
                    text: "The GDP grew by 3%".to_string(),
                    start: 0,
                    end: 18,
                    trigger: Some("grew".to_string()),
                },
                AnaphorSpan {
                    text: "This fact".to_string(),
                    start: 20,
                    end: 29,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Shell noun 'fact' - factual class (Schmid 2000)"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "shell_fact_02",
                "Prices doubled in one year. The reason was supply chain disruption.",
                AntecedentSpan {
                    text: "Prices doubled in one year".to_string(),
                    start: 0,
                    end: 26,
                    trigger: Some("doubled".to_string()),
                },
                AnaphorSpan {
                    text: "The reason".to_string(),
                    start: 28,
                    end: 38,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Shell noun 'reason' - factual class, cataphoric"),
        );

        // Linguistic shell nouns
        dataset.add(
            AnaphoraTestCase::new(
                "shell_ling_01",
                "The CEO promised higher wages. This claim was later retracted.",
                AntecedentSpan {
                    text: "The CEO promised higher wages".to_string(),
                    start: 0,
                    end: 29,
                    trigger: Some("promised".to_string()),
                },
                AnaphorSpan {
                    text: "This claim".to_string(),
                    start: 31,
                    end: 41,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Shell noun 'claim' - linguistic class"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "shell_ling_02",
                "We should invest in renewables. The argument convinced the board.",
                AntecedentSpan {
                    text: "We should invest in renewables".to_string(),
                    start: 0,
                    end: 30,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "The argument".to_string(),
                    start: 32,
                    end: 44,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Shell noun 'argument' - linguistic class"),
        );

        // Mental shell nouns
        dataset.add(
            AnaphoraTestCase::new(
                "shell_mental_01",
                "Automation will replace most jobs. This belief is controversial.",
                AntecedentSpan {
                    text: "Automation will replace most jobs".to_string(),
                    start: 0,
                    end: 33,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This belief".to_string(),
                    start: 35,
                    end: 46,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Shell noun 'belief' - mental class"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "shell_mental_02",
                "The new policy will fail. This view is shared by experts.",
                AntecedentSpan {
                    text: "The new policy will fail".to_string(),
                    start: 0,
                    end: 24,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This view".to_string(),
                    start: 26,
                    end: 35,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Shell noun 'view' - mental class"),
        );

        // Modal shell nouns
        dataset.add(
            AnaphoraTestCase::new(
                "shell_modal_01",
                "The system could crash under load. This possibility concerned engineers.",
                AntecedentSpan {
                    text: "The system could crash under load".to_string(),
                    start: 0,
                    end: 33,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This possibility".to_string(),
                    start: 35,
                    end: 51,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Shell noun 'possibility' - modal class"),
        );

        // Eventive shell nouns
        dataset.add(
            AnaphoraTestCase::new(
                "shell_event_01",
                "The company laid off 500 workers. This decision shocked employees.",
                AntecedentSpan {
                    text: "The company laid off 500 workers".to_string(),
                    start: 0,
                    end: 32,
                    trigger: Some("laid off".to_string()),
                },
                AnaphorSpan {
                    text: "This decision".to_string(),
                    start: 34,
                    end: 47,
                },
                AnaphoraType::Event,
            )
            .with_notes("Shell noun 'decision' - eventive class"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "shell_event_02",
                "A meteor struck the desert. The incident was witnessed by campers.",
                AntecedentSpan {
                    text: "A meteor struck the desert".to_string(),
                    start: 0,
                    end: 26,
                    trigger: Some("struck".to_string()),
                },
                AnaphorSpan {
                    text: "The incident".to_string(),
                    start: 28,
                    end: 40,
                },
                AnaphoraType::Event,
            )
            .with_notes("Shell noun 'incident' - eventive class"),
        );

        // Circumstantial shell nouns
        dataset.add(
            AnaphoraTestCase::new(
                "shell_circ_01",
                "Inflation is rising while wages stagnate. This situation is unsustainable.",
                AntecedentSpan {
                    text: "Inflation is rising while wages stagnate".to_string(),
                    start: 0,
                    end: 40,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This situation".to_string(),
                    start: 42,
                    end: 56,
                },
                AnaphoraType::Situation,
            )
            .with_notes("Shell noun 'situation' - circumstantial class"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "shell_circ_02",
                "The code has bugs and the deadline is tomorrow. This problem needs addressing.",
                AntecedentSpan {
                    text: "The code has bugs and the deadline is tomorrow".to_string(),
                    start: 0,
                    end: 46,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This problem".to_string(),
                    start: 48,
                    end: 60,
                },
                AnaphoraType::Situation,
            )
            .with_notes("Shell noun 'problem' - circumstantial class"),
        );

        // =================================================================
        // DISCOURSE DISTANCE CASES (testing longer-range anaphora)
        // =================================================================

        dataset.add(
            AnaphoraTestCase::new(
                "dist_01",
                "The protests began in March. Police deployed tear gas. Several arrests were made. This response drew international criticism.",
                AntecedentSpan {
                    text: "Police deployed tear gas. Several arrests were made".to_string(),
                    start: 29,
                    end: 80,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This response".to_string(),
                    start: 82,
                    end: 95,
                },
                AnaphoraType::Event,
            )
            .with_notes("Multi-sentence antecedent (2 sentences back)"),
        );

        dataset
    }

    // =================================================================
    // Domain-Specific Datasets
    // =================================================================

    /// Legal domain abstract anaphora dataset.
    ///
    /// Legal texts heavily use abstract reference for precedents,
    /// rulings, statutes, and legal principles.
    #[must_use]
    pub fn legal_domain() -> Self {
        let mut dataset = Self::new();

        dataset.add(
            AnaphoraTestCase::new(
                "legal_01",
                "The court ruled in favor of the plaintiff. This decision sets a precedent.",
                AntecedentSpan {
                    text: "The court ruled in favor of the plaintiff".to_string(),
                    start: 0,
                    end: 41,
                    trigger: Some("ruled".to_string()),
                },
                AnaphorSpan {
                    text: "This decision".to_string(),
                    start: 43,
                    end: 56,
                },
                AnaphoraType::Event,
            )
            .with_notes("Court ruling reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "legal_02",
                "The defendant violated the contract terms. This breach entitles the claimant to damages.",
                AntecedentSpan {
                    text: "The defendant violated the contract terms".to_string(),
                    start: 0,
                    end: 41,
                    trigger: Some("violated".to_string()),
                },
                AnaphorSpan {
                    text: "This breach".to_string(),
                    start: 43,
                    end: 54,
                },
                AnaphoraType::Event,
            )
            .with_notes("Legal violation reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "legal_03",
                "Corporations must disclose material information. Failure to do so constitutes fraud.",
                AntecedentSpan {
                    text: "Corporations must disclose material information".to_string(),
                    start: 0,
                    end: 47,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "Failure to do so".to_string(),
                    start: 49,
                    end: 65,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Obligation reference with negation"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "legal_04",
                "The statute requires prior notice. This requirement was not met.",
                AntecedentSpan {
                    text: "The statute requires prior notice".to_string(),
                    start: 0,
                    end: 33,
                    trigger: Some("requires".to_string()),
                },
                AnaphorSpan {
                    text: "This requirement".to_string(),
                    start: 35,
                    end: 51,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Legal requirement reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "legal_05",
                "The witness may have lied. If this is true, perjury charges apply.",
                AntecedentSpan {
                    text: "The witness may have lied".to_string(),
                    start: 0,
                    end: 25,
                    trigger: Some("lied".to_string()),
                },
                AnaphorSpan {
                    text: "this".to_string(),
                    start: 30,
                    end: 34,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Modal proposition in legal context"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "legal_06",
                "The parties agreed to arbitration. This agreement is binding.",
                AntecedentSpan {
                    text: "The parties agreed to arbitration".to_string(),
                    start: 0,
                    end: 33,
                    trigger: Some("agreed".to_string()),
                },
                AnaphorSpan {
                    text: "This agreement".to_string(),
                    start: 35,
                    end: 49,
                },
                AnaphoraType::Event,
            )
            .with_notes("Agreement event reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "legal_07",
                "The prosecution alleged embezzlement. The allegation was later withdrawn.",
                AntecedentSpan {
                    text: "The prosecution alleged embezzlement".to_string(),
                    start: 0,
                    end: 36,
                    trigger: Some("alleged".to_string()),
                },
                AnaphorSpan {
                    text: "The allegation".to_string(),
                    start: 38,
                    end: 52,
                },
                AnaphoraType::Event,
            )
            .with_notes("Allegation event reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "legal_08",
                "Evidence was obtained without a warrant. This fact renders it inadmissible.",
                AntecedentSpan {
                    text: "Evidence was obtained without a warrant".to_string(),
                    start: 0,
                    end: 39,
                    trigger: Some("obtained".to_string()),
                },
                AnaphorSpan {
                    text: "This fact".to_string(),
                    start: 41,
                    end: 50,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Factual shell noun in legal context"),
        );

        // Nominal case for contrast
        dataset.add(
            AnaphoraTestCase::new(
                "legal_nom_01",
                "The defendant hired a lawyer. He filed an appeal.",
                AntecedentSpan {
                    text: "a lawyer".to_string(),
                    start: 21,
                    end: 29,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "He".to_string(),
                    start: 31,
                    end: 33,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Standard nominal coreference (lawyer)"),
        );

        dataset
    }

    /// Medical/clinical domain abstract anaphora dataset.
    ///
    /// Medical texts reference diagnoses, procedures, symptoms,
    /// and treatment outcomes abstractly.
    #[must_use]
    pub fn medical_domain() -> Self {
        let mut dataset = Self::new();

        dataset.add(
            AnaphoraTestCase::new(
                "med_01",
                "The patient presented with chest pain. This symptom suggested cardiac involvement.",
                AntecedentSpan {
                    text: "The patient presented with chest pain".to_string(),
                    start: 0,
                    end: 37,
                    trigger: Some("presented".to_string()),
                },
                AnaphorSpan {
                    text: "This symptom".to_string(),
                    start: 39,
                    end: 51,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Symptom presentation reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "med_02",
                "Surgery was performed to remove the tumor. This procedure lasted four hours.",
                AntecedentSpan {
                    text: "Surgery was performed to remove the tumor".to_string(),
                    start: 0,
                    end: 41,
                    trigger: Some("performed".to_string()),
                },
                AnaphorSpan {
                    text: "This procedure".to_string(),
                    start: 43,
                    end: 57,
                },
                AnaphoraType::Event,
            )
            .with_notes("Surgical procedure reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "med_03",
                "Blood pressure normalized after treatment. This improvement was sustained.",
                AntecedentSpan {
                    text: "Blood pressure normalized after treatment".to_string(),
                    start: 0,
                    end: 41,
                    trigger: Some("normalized".to_string()),
                },
                AnaphorSpan {
                    text: "This improvement".to_string(),
                    start: 43,
                    end: 59,
                },
                AnaphoraType::Event,
            )
            .with_notes("Clinical improvement reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "med_04",
                "The medication may cause drowsiness. This side effect is usually temporary.",
                AntecedentSpan {
                    text: "The medication may cause drowsiness".to_string(),
                    start: 0,
                    end: 35,
                    trigger: Some("cause".to_string()),
                },
                AnaphorSpan {
                    text: "This side effect".to_string(),
                    start: 37,
                    end: 53,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Potential side effect reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "med_05",
                "The patient was diagnosed with diabetes. Managing this condition requires lifestyle changes.",
                AntecedentSpan {
                    text: "diabetes".to_string(),
                    start: 31,
                    end: 39,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "this condition".to_string(),
                    start: 51,
                    end: 65,
                },
                AnaphoraType::Situation,
            )
            .with_notes("Medical condition reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "med_06",
                "The biopsy revealed malignant cells. This finding necessitated further testing.",
                AntecedentSpan {
                    text: "The biopsy revealed malignant cells".to_string(),
                    start: 0,
                    end: 35,
                    trigger: Some("revealed".to_string()),
                },
                AnaphorSpan {
                    text: "This finding".to_string(),
                    start: 37,
                    end: 49,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Diagnostic finding reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "med_07",
                "The patient's fever spiked overnight. This development concerned the medical team.",
                AntecedentSpan {
                    text: "The patient's fever spiked overnight".to_string(),
                    start: 0,
                    end: 36,
                    trigger: Some("spiked".to_string()),
                },
                AnaphorSpan {
                    text: "This development".to_string(),
                    start: 38,
                    end: 54,
                },
                AnaphoraType::Event,
            )
            .with_notes("Clinical event reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "med_08",
                "Chemotherapy was discontinued due to adverse reactions. This decision was made by the oncologist.",
                AntecedentSpan {
                    text: "Chemotherapy was discontinued due to adverse reactions".to_string(),
                    start: 0,
                    end: 54,
                    trigger: Some("discontinued".to_string()),
                },
                AnaphorSpan {
                    text: "This decision".to_string(),
                    start: 56,
                    end: 69,
                },
                AnaphoraType::Event,
            )
            .with_notes("Treatment decision reference"),
        );

        // Nominal case for contrast
        dataset.add(
            AnaphoraTestCase::new(
                "med_nom_01",
                "The surgeon consulted a specialist. She recommended immediate intervention.",
                AntecedentSpan {
                    text: "a specialist".to_string(),
                    start: 23,
                    end: 35,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "She".to_string(),
                    start: 37,
                    end: 40,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Standard nominal coreference (specialist)"),
        );

        dataset
    }

    /// Financial/business domain abstract anaphora dataset.
    ///
    /// Financial texts reference market events, transactions,
    /// decisions, and economic phenomena.
    #[must_use]
    pub fn financial_domain() -> Self {
        let mut dataset = Self::new();

        dataset.add(
            AnaphoraTestCase::new(
                "fin_01",
                "The Fed raised interest rates. This move sent shockwaves through markets.",
                AntecedentSpan {
                    text: "The Fed raised interest rates".to_string(),
                    start: 0,
                    end: 29,
                    trigger: Some("raised".to_string()),
                },
                AnaphorSpan {
                    text: "This move".to_string(),
                    start: 31,
                    end: 40,
                },
                AnaphoraType::Event,
            )
            .with_notes("Policy decision reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fin_02",
                "The merger was approved by regulators. This development boosted investor confidence.",
                AntecedentSpan {
                    text: "The merger was approved by regulators".to_string(),
                    start: 0,
                    end: 37,
                    trigger: Some("approved".to_string()),
                },
                AnaphorSpan {
                    text: "This development".to_string(),
                    start: 39,
                    end: 55,
                },
                AnaphoraType::Event,
            )
            .with_notes("Regulatory approval reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fin_03",
                "Quarterly earnings exceeded expectations. This performance led to a stock rally.",
                AntecedentSpan {
                    text: "Quarterly earnings exceeded expectations".to_string(),
                    start: 0,
                    end: 40,
                    trigger: Some("exceeded".to_string()),
                },
                AnaphorSpan {
                    text: "This performance".to_string(),
                    start: 42,
                    end: 58,
                },
                AnaphoraType::Event,
            )
            .with_notes("Financial performance reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fin_04",
                "The company might default on its loans. This risk has alarmed bondholders.",
                AntecedentSpan {
                    text: "The company might default on its loans".to_string(),
                    start: 0,
                    end: 38,
                    trigger: Some("default".to_string()),
                },
                AnaphorSpan {
                    text: "This risk".to_string(),
                    start: 40,
                    end: 49,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Financial risk proposition"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fin_05",
                "Supply chain disruptions are causing inflation. This situation could persist for years.",
                AntecedentSpan {
                    text: "Supply chain disruptions are causing inflation".to_string(),
                    start: 0,
                    end: 46,
                    trigger: Some("causing".to_string()),
                },
                AnaphorSpan {
                    text: "This situation".to_string(),
                    start: 48,
                    end: 62,
                },
                AnaphoraType::Situation,
            )
            .with_notes("Economic situation reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fin_06",
                "The CEO announced a stock buyback program. The announcement pushed shares higher.",
                AntecedentSpan {
                    text: "The CEO announced a stock buyback program".to_string(),
                    start: 0,
                    end: 41,
                    trigger: Some("announced".to_string()),
                },
                AnaphorSpan {
                    text: "The announcement".to_string(),
                    start: 43,
                    end: 59,
                },
                AnaphoraType::Event,
            )
            .with_notes("Corporate announcement reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fin_07",
                "Revenue grew by 15% year-over-year. This growth outpaced analyst forecasts.",
                AntecedentSpan {
                    text: "Revenue grew by 15% year-over-year".to_string(),
                    start: 0,
                    end: 34,
                    trigger: Some("grew".to_string()),
                },
                AnaphorSpan {
                    text: "This growth".to_string(),
                    start: 36,
                    end: 47,
                },
                AnaphoraType::Event,
            )
            .with_notes("Revenue growth event reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "fin_08",
                "The acquisition was completed yesterday. This transaction creates the largest retailer.",
                AntecedentSpan {
                    text: "The acquisition was completed yesterday".to_string(),
                    start: 0,
                    end: 39,
                    trigger: Some("completed".to_string()),
                },
                AnaphorSpan {
                    text: "This transaction".to_string(),
                    start: 41,
                    end: 57,
                },
                AnaphoraType::Event,
            )
            .with_notes("Business transaction reference"),
        );

        // Nominal case for contrast
        dataset.add(
            AnaphoraTestCase::new(
                "fin_nom_01",
                "The CFO presented the report. She highlighted key metrics.",
                AntecedentSpan {
                    text: "The CFO".to_string(),
                    start: 0,
                    end: 7,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "She".to_string(),
                    start: 31,
                    end: 34,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Standard nominal coreference (CFO)"),
        );

        dataset
    }

    /// Scientific/technical domain abstract anaphora dataset.
    ///
    /// Scientific writing references experiments, observations,
    /// hypotheses, and theoretical concepts.
    #[must_use]
    pub fn scientific_domain() -> Self {
        let mut dataset = Self::new();

        dataset.add(
            AnaphoraTestCase::new(
                "sci_01",
                "The experiment failed to replicate earlier results. This failure suggests methodological issues.",
                AntecedentSpan {
                    text: "The experiment failed to replicate earlier results".to_string(),
                    start: 0,
                    end: 50,
                    trigger: Some("failed".to_string()),
                },
                AnaphorSpan {
                    text: "This failure".to_string(),
                    start: 52,
                    end: 64,
                },
                AnaphoraType::Event,
            )
            .with_notes("Experimental failure reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sci_02",
                "The data shows a correlation between diet and longevity. This finding aligns with previous studies.",
                AntecedentSpan {
                    text: "The data shows a correlation between diet and longevity".to_string(),
                    start: 0,
                    end: 55,
                    trigger: Some("shows".to_string()),
                },
                AnaphorSpan {
                    text: "This finding".to_string(),
                    start: 57,
                    end: 69,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Scientific finding reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sci_03",
                "Quantum entanglement may enable faster communication. If this is possible, it would revolutionize networking.",
                AntecedentSpan {
                    text: "Quantum entanglement may enable faster communication".to_string(),
                    start: 0,
                    end: 52,
                    trigger: Some("enable".to_string()),
                },
                AnaphorSpan {
                    text: "this".to_string(),
                    start: 57,
                    end: 61,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Scientific hypothesis reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sci_04",
                "The samples were contaminated during transport. This problem invalidated the study.",
                AntecedentSpan {
                    text: "The samples were contaminated during transport".to_string(),
                    start: 0,
                    end: 46,
                    trigger: Some("contaminated".to_string()),
                },
                AnaphorSpan {
                    text: "This problem".to_string(),
                    start: 48,
                    end: 60,
                },
                AnaphoraType::Event,
            )
            .with_notes("Experimental problem reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sci_05",
                "The protein folded incorrectly under high temperatures. This observation was unexpected.",
                AntecedentSpan {
                    text: "The protein folded incorrectly under high temperatures".to_string(),
                    start: 0,
                    end: 54,
                    trigger: Some("folded".to_string()),
                },
                AnaphorSpan {
                    text: "This observation".to_string(),
                    start: 56,
                    end: 72,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Observational fact reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sci_06",
                "The simulation predicted climate warming. This prediction matched observed data.",
                AntecedentSpan {
                    text: "The simulation predicted climate warming".to_string(),
                    start: 0,
                    end: 40,
                    trigger: Some("predicted".to_string()),
                },
                AnaphorSpan {
                    text: "This prediction".to_string(),
                    start: 42,
                    end: 57,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Model prediction reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sci_07",
                "The theory was disproven by new evidence. Despite this setback, research continues.",
                AntecedentSpan {
                    text: "The theory was disproven by new evidence".to_string(),
                    start: 0,
                    end: 40,
                    trigger: Some("disproven".to_string()),
                },
                AnaphorSpan {
                    text: "this setback".to_string(),
                    start: 50,
                    end: 62,
                },
                AnaphoraType::Event,
            )
            .with_notes("Scientific setback reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "sci_08",
                "The algorithm achieved 98% accuracy. This result represents state-of-the-art performance.",
                AntecedentSpan {
                    text: "The algorithm achieved 98% accuracy".to_string(),
                    start: 0,
                    end: 35,
                    trigger: Some("achieved".to_string()),
                },
                AnaphorSpan {
                    text: "This result".to_string(),
                    start: 37,
                    end: 48,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Experimental result reference"),
        );

        // Nominal case for contrast
        dataset.add(
            AnaphoraTestCase::new(
                "sci_nom_01",
                "The researcher published her findings. She received several awards.",
                AntecedentSpan {
                    text: "The researcher".to_string(),
                    start: 0,
                    end: 14,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "She".to_string(),
                    start: 40,
                    end: 43,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Standard nominal coreference (researcher)"),
        );

        dataset
    }

    /// News/journalism domain abstract anaphora dataset.
    ///
    /// News articles reference events, statements, developments,
    /// and reactions.
    #[must_use]
    pub fn news_domain() -> Self {
        let mut dataset = Self::new();

        dataset.add(
            AnaphoraTestCase::new(
                "news_01",
                "The president signed the bill into law. This action fulfilled a campaign promise.",
                AntecedentSpan {
                    text: "The president signed the bill into law".to_string(),
                    start: 0,
                    end: 38,
                    trigger: Some("signed".to_string()),
                },
                AnaphorSpan {
                    text: "This action".to_string(),
                    start: 40,
                    end: 51,
                },
                AnaphoraType::Event,
            )
            .with_notes("Political action reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "news_02",
                "Protests erupted across major cities. This unrest prompted a government response.",
                AntecedentSpan {
                    text: "Protests erupted across major cities".to_string(),
                    start: 0,
                    end: 36,
                    trigger: Some("erupted".to_string()),
                },
                AnaphorSpan {
                    text: "This unrest".to_string(),
                    start: 38,
                    end: 49,
                },
                AnaphoraType::Event,
            )
            .with_notes("Social unrest reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "news_03",
                "The minister denied any wrongdoing. This denial contradicted earlier statements.",
                AntecedentSpan {
                    text: "The minister denied any wrongdoing".to_string(),
                    start: 0,
                    end: 34,
                    trigger: Some("denied".to_string()),
                },
                AnaphorSpan {
                    text: "This denial".to_string(),
                    start: 36,
                    end: 47,
                },
                AnaphoraType::Event,
            )
            .with_notes("Statement/denial reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "news_04",
                "Peace talks collapsed after three days. The breakdown disappointed international observers.",
                AntecedentSpan {
                    text: "Peace talks collapsed after three days".to_string(),
                    start: 0,
                    end: 38,
                    trigger: Some("collapsed".to_string()),
                },
                AnaphorSpan {
                    text: "The breakdown".to_string(),
                    start: 40,
                    end: 53,
                },
                AnaphoraType::Event,
            )
            .with_notes("Diplomatic breakdown reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "news_05",
                "The hurricane devastated coastal towns. This disaster left thousands homeless.",
                AntecedentSpan {
                    text: "The hurricane devastated coastal towns".to_string(),
                    start: 0,
                    end: 38,
                    trigger: Some("devastated".to_string()),
                },
                AnaphorSpan {
                    text: "This disaster".to_string(),
                    start: 40,
                    end: 53,
                },
                AnaphoraType::Event,
            )
            .with_notes("Natural disaster reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "news_06",
                "The celebrity apologized publicly. This apology came after widespread backlash.",
                AntecedentSpan {
                    text: "The celebrity apologized publicly".to_string(),
                    start: 0,
                    end: 33,
                    trigger: Some("apologized".to_string()),
                },
                AnaphorSpan {
                    text: "This apology".to_string(),
                    start: 35,
                    end: 47,
                },
                AnaphoraType::Event,
            )
            .with_notes("Public apology reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "news_07",
                "The election results were contested. This controversy led to legal challenges.",
                AntecedentSpan {
                    text: "The election results were contested".to_string(),
                    start: 0,
                    end: 35,
                    trigger: Some("contested".to_string()),
                },
                AnaphorSpan {
                    text: "This controversy".to_string(),
                    start: 37,
                    end: 53,
                },
                AnaphoraType::Event,
            )
            .with_notes("Political controversy reference"),
        );

        dataset.add(
            AnaphoraTestCase::new(
                "news_08",
                "Unemployment fell to a historic low. This improvement boosted consumer spending.",
                AntecedentSpan {
                    text: "Unemployment fell to a historic low".to_string(),
                    start: 0,
                    end: 35,
                    trigger: Some("fell".to_string()),
                },
                AnaphorSpan {
                    text: "This improvement".to_string(),
                    start: 37,
                    end: 53,
                },
                AnaphoraType::Event,
            )
            .with_notes("Economic improvement reference"),
        );

        // Nominal case for contrast
        dataset.add(
            AnaphoraTestCase::new(
                "news_nom_01",
                "The mayor addressed the media. He promised immediate action.",
                AntecedentSpan {
                    text: "The mayor".to_string(),
                    start: 0,
                    end: 9,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "He".to_string(),
                    start: 32,
                    end: 34,
                },
                AnaphoraType::Nominal,
            )
            .with_notes("Standard nominal coreference (mayor)"),
        );

        dataset
    }

    /// Complex/challenging abstract anaphora cases.
    ///
    /// These cases test difficult scenarios: long-distance,
    /// cataphoric, ambiguous, and multi-clause antecedents.
    #[must_use]
    pub fn challenging_cases() -> Self {
        let mut dataset = Self::new();

        // Long-distance anaphora (multiple sentences back)
        dataset.add(
            AnaphoraTestCase::new(
                "chal_01",
                "The company reported strong earnings. Analysts praised the results. Investors celebrated. This success was unexpected.",
                AntecedentSpan {
                    text: "The company reported strong earnings".to_string(),
                    start: 0,
                    end: 36,
                    trigger: Some("reported".to_string()),
                },
                AnaphorSpan {
                    text: "This success".to_string(),
                    start: 91,
                    end: 103,
                },
                AnaphoraType::Event,
            )
            .with_notes("Long-distance (3 sentences back)"),
        );

        // Cataphoric reference (anaphor before antecedent)
        dataset.add(
            AnaphoraTestCase::new(
                "chal_02",
                "This much is clear: the policy has failed.",
                AntecedentSpan {
                    text: "the policy has failed".to_string(),
                    start: 20,
                    end: 41,
                    trigger: Some("failed".to_string()),
                },
                AnaphorSpan {
                    text: "This much".to_string(),
                    start: 0,
                    end: 9,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Cataphoric reference"),
        );

        // Complex multi-clause antecedent
        dataset.add(
            AnaphoraTestCase::new(
                "chal_03",
                "Inflation rose while wages stagnated and unemployment increased. This combination created economic hardship.",
                AntecedentSpan {
                    text: "Inflation rose while wages stagnated and unemployment increased".to_string(),
                    start: 0,
                    end: 63,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This combination".to_string(),
                    start: 65,
                    end: 81,
                },
                AnaphoraType::Situation,
            )
            .with_notes("Multi-clause conjunction antecedent"),
        );

        // Embedded clause antecedent
        dataset.add(
            AnaphoraTestCase::new(
                "chal_04",
                "The CEO said that layoffs were necessary. This claim angered workers.",
                AntecedentSpan {
                    text: "layoffs were necessary".to_string(),
                    start: 18,
                    end: 40,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This claim".to_string(),
                    start: 42,
                    end: 52,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Embedded clause antecedent"),
        );

        // Negated antecedent
        dataset.add(
            AnaphoraTestCase::new(
                "chal_05",
                "The witness did not appear in court. This absence was noted by the judge.",
                AntecedentSpan {
                    text: "The witness did not appear in court".to_string(),
                    start: 0,
                    end: 35,
                    trigger: Some("appear".to_string()),
                },
                AnaphorSpan {
                    text: "This absence".to_string(),
                    start: 37,
                    end: 49,
                },
                AnaphoraType::Event,
            )
            .with_notes("Negated event antecedent"),
        );

        // Disjunction antecedent
        dataset.add(
            AnaphoraTestCase::new(
                "chal_06",
                "Either the system crashed or data was corrupted. This problem halted operations.",
                AntecedentSpan {
                    text: "Either the system crashed or data was corrupted".to_string(),
                    start: 0,
                    end: 47,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This problem".to_string(),
                    start: 49,
                    end: 61,
                },
                AnaphoraType::Situation,
            )
            .with_notes("Disjunction antecedent"),
        );

        // Conditional antecedent
        dataset.add(
            AnaphoraTestCase::new(
                "chal_07",
                "If interest rates rise, housing prices will fall. This scenario worries homeowners.",
                AntecedentSpan {
                    text: "If interest rates rise, housing prices will fall".to_string(),
                    start: 0,
                    end: 48,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This scenario".to_string(),
                    start: 50,
                    end: 63,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Conditional antecedent"),
        );

        // Comparator antecedent
        dataset.add(
            AnaphoraTestCase::new(
                "chal_08",
                "Profits are higher than last year. This exceeds expectations.",
                AntecedentSpan {
                    text: "Profits are higher than last year".to_string(),
                    start: 0,
                    end: 33,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This".to_string(),
                    start: 35,
                    end: 39,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Comparative statement antecedent"),
        );

        // Question as antecedent
        dataset.add(
            AnaphoraTestCase::new(
                "chal_09",
                "Will the company survive? This question haunts investors.",
                AntecedentSpan {
                    text: "Will the company survive".to_string(),
                    start: 0,
                    end: 24,
                    trigger: None,
                },
                AnaphorSpan {
                    text: "This question".to_string(),
                    start: 27,
                    end: 40,
                },
                AnaphoraType::Proposition,
            )
            .with_notes("Interrogative clause antecedent"),
        );

        // Generic statement antecedent
        dataset.add(
            AnaphoraTestCase::new(
                "chal_10",
                "Power corrupts. This truth has been known for centuries.",
                AntecedentSpan {
                    text: "Power corrupts".to_string(),
                    start: 0,
                    end: 14,
                    trigger: Some("corrupts".to_string()),
                },
                AnaphorSpan {
                    text: "This truth".to_string(),
                    start: 16,
                    end: 26,
                },
                AnaphoraType::Fact,
            )
            .with_notes("Generic statement antecedent"),
        );

        dataset
    }

    /// Create a comprehensive dataset combining all domains.
    ///
    /// This is the most complete test set for abstract anaphora.
    #[must_use]
    pub fn comprehensive() -> Self {
        let mut dataset = Self::extended();

        // Merge in domain-specific cases
        for case in Self::legal_domain().cases {
            dataset.add(case);
        }
        for case in Self::medical_domain().cases {
            dataset.add(case);
        }
        for case in Self::financial_domain().cases {
            dataset.add(case);
        }
        for case in Self::scientific_domain().cases {
            dataset.add(case);
        }
        for case in Self::news_domain().cases {
            dataset.add(case);
        }
        for case in Self::challenging_cases().cases {
            dataset.add(case);
        }

        dataset
    }

    /// Statistics about the dataset.
    #[must_use]
    pub fn stats(&self) -> DatasetStats {
        let mut by_type: HashMap<AnaphoraType, usize> = HashMap::new();
        for case in &self.cases {
            *by_type.entry(case.anaphora_type).or_default() += 1;
        }

        DatasetStats {
            total: self.cases.len(),
            nominal: by_type.get(&AnaphoraType::Nominal).copied().unwrap_or(0),
            event: by_type.get(&AnaphoraType::Event).copied().unwrap_or(0),
            fact: by_type.get(&AnaphoraType::Fact).copied().unwrap_or(0),
            proposition: by_type
                .get(&AnaphoraType::Proposition)
                .copied()
                .unwrap_or(0),
            situation: by_type.get(&AnaphoraType::Situation).copied().unwrap_or(0),
        }
    }
}

/// Dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Total number of test cases
    pub total: usize,
    /// Nominal coreference cases
    pub nominal: usize,
    /// Event anaphora cases
    pub event: usize,
    /// Fact anaphora cases
    pub fact: usize,
    /// Proposition anaphora cases
    pub proposition: usize,
    /// Situation anaphora cases
    pub situation: usize,
}

impl DatasetStats {
    /// Total abstract (non-nominal) cases.
    #[must_use]
    pub fn abstract_total(&self) -> usize {
        self.event + self.fact + self.proposition + self.situation
    }
}

/// Analysis of shell noun distribution in a dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShellNounAnalysis {
    /// Total shell nouns detected
    pub total_shell_nouns: usize,
    /// Shell nouns by semantic class
    pub by_class: HashMap<ShellNounClass, usize>,
    /// How many are demonstrative ("this X")
    pub demonstrative_count: usize,
    /// How many have matching antecedent type for their class
    pub type_match_count: usize,
}

// =============================================================================
// Candidate Ranking Metrics (Marasović 2017 style)
// =============================================================================

/// Candidate ranking evaluation metrics.
///
/// These metrics match the evaluation approach from:
/// - Marasović et al. (2017): "A Mention-Ranking Model for Abstract Anaphora Resolution"
/// - Li & Ng (2022): "End-to-End Neural Discourse Deixis Resolution"
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandidateRankingMetrics {
    /// Accuracy@1: proportion where correct candidate ranked first
    pub accuracy_at_1: f64,
    /// Mean Reciprocal Rank: average of 1/rank of correct candidate
    pub mrr: f64,
    /// Antecedent containment: proportion where gold in candidate set
    pub containment: f64,
    /// Average number of candidates per case
    pub avg_candidates: f64,
    /// Total cases evaluated
    pub total_cases: usize,
}

impl CandidateRankingMetrics {
    /// Compute metrics from a list of (gold_rank, num_candidates) tuples.
    ///
    /// `gold_rank` is 1-indexed (1 = ranked first), 0 = not in candidates
    #[must_use]
    pub fn from_rankings(rankings: &[(usize, usize)]) -> Self {
        if rankings.is_empty() {
            return Self::default();
        }

        let total = rankings.len();
        let mut correct_at_1 = 0;
        let mut reciprocal_sum = 0.0;
        let mut contained = 0;
        let mut total_candidates = 0;

        for &(gold_rank, num_candidates) in rankings {
            total_candidates += num_candidates;

            if gold_rank > 0 {
                contained += 1;
                reciprocal_sum += 1.0 / gold_rank as f64;

                if gold_rank == 1 {
                    correct_at_1 += 1;
                }
            }
        }

        Self {
            accuracy_at_1: correct_at_1 as f64 / total as f64,
            mrr: reciprocal_sum / total as f64,
            containment: contained as f64 / total as f64,
            avg_candidates: total_candidates as f64 / total as f64,
            total_cases: total,
        }
    }

    /// Summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Candidate Ranking:\n  Accuracy@1: {:.1}%\n  MRR: {:.3}\n  Containment: {:.1}%\n  Avg candidates: {:.1}",
            self.accuracy_at_1 * 100.0,
            self.mrr,
            self.containment * 100.0,
            self.avg_candidates
        )
    }
}

impl ShellNounAnalysis {
    /// Percentage of shell nouns that are demonstrative.
    #[must_use]
    pub fn demonstrative_ratio(&self) -> f64 {
        if self.total_shell_nouns == 0 {
            0.0
        } else {
            self.demonstrative_count as f64 / self.total_shell_nouns as f64
        }
    }

    /// Percentage of shell nouns whose class matches the antecedent type.
    #[must_use]
    pub fn type_match_ratio(&self) -> f64 {
        if self.total_shell_nouns == 0 {
            0.0
        } else {
            self.type_match_count as f64 / self.total_shell_nouns as f64
        }
    }

    /// Summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Shell nouns: {} total, {:.0}% demonstrative, {:.0}% type-matched\n",
            self.total_shell_nouns,
            self.demonstrative_ratio() * 100.0,
            self.type_match_ratio() * 100.0
        );
        for (class, count) in &self.by_class {
            s.push_str(&format!("  {}: {}\n", class.as_str(), count));
        }
        s
    }
}

// =============================================================================
// Evaluator
// =============================================================================

/// Resolver backend for the evaluator.
#[derive(Debug, Clone)]
pub enum ResolverBackend {
    /// Simple rule-based nominal resolver (baseline)
    Simple(SimpleCorefResolver),
    /// Discourse-aware resolver with event extraction
    DiscourseAware,
}

impl Default for ResolverBackend {
    fn default() -> Self {
        ResolverBackend::Simple(SimpleCorefResolver::default())
    }
}

/// Evaluator for abstract anaphora resolution.
#[derive(Debug, Clone)]
pub struct AbstractAnaphoraEvaluator {
    resolver: SimpleCorefResolver,
    use_discourse: bool,
}

impl Default for AbstractAnaphoraEvaluator {
    fn default() -> Self {
        Self::new(SimpleCorefResolver::default())
    }
}

impl AbstractAnaphoraEvaluator {
    /// Create with a specific resolver.
    #[must_use]
    pub fn new(resolver: SimpleCorefResolver) -> Self {
        Self {
            resolver,
            use_discourse: false,
        }
    }

    /// Create an evaluator using discourse-aware resolution.
    ///
    /// This evaluator will use `DiscourseAwareResolver` with event extraction
    /// to attempt resolution of abstract anaphora.
    #[must_use]
    pub fn discourse_aware() -> Self {
        Self {
            resolver: SimpleCorefResolver::default(),
            use_discourse: true,
        }
    }

    /// Enable or disable discourse-aware resolution.
    #[must_use]
    pub fn with_discourse(mut self, enable: bool) -> Self {
        self.use_discourse = enable;
        self
    }

    /// Evaluate the resolver on a dataset.
    #[must_use]
    pub fn evaluate(&self, dataset: &AbstractAnaphoraDataset) -> EvaluationResults {
        let mut results = EvaluationResults::default();

        for case in &dataset.cases {
            let result = if self.use_discourse {
                self.evaluate_case_discourse(case)
            } else {
                self.evaluate_case(case)
            };
            results.case_results.push(result.clone());

            if case.anaphora_type == AnaphoraType::Nominal {
                results.nominal_total += 1;
                if result.resolved_correctly {
                    results.nominal_correct += 1;
                }
            } else {
                results.abstract_total += 1;
                if result.resolved_correctly {
                    results.abstract_correct += 1;
                }
                // Track by specific type
                results
                    .by_type
                    .entry(case.anaphora_type)
                    .or_insert_with(TypeResults::default)
                    .add(&result);
            }
        }

        results.compute_accuracy();
        results
    }

    /// Evaluate a single test case using the simple resolver.
    fn evaluate_case(&self, case: &AnaphoraTestCase) -> CaseResult {
        // Extract entities from the text
        // For nominal cases, we create Person/Org entities
        // For abstract cases, we also create entities but the resolver won't link events

        let entities = self.extract_entities_for_case(case);
        let resolved = self.resolver.resolve(&entities);

        // Check if antecedent and anaphor got the same canonical_id
        let antecedent_id = resolved
            .iter()
            .find(|e| {
                e.start == case.antecedent.start
                    || self.text_matches(&e.text, &case.antecedent.text)
            })
            .and_then(|e| e.canonical_id);

        let anaphor_id = resolved
            .iter()
            .find(|e| {
                e.start == case.anaphor.start || self.text_matches(&e.text, &case.anaphor.text)
            })
            .and_then(|e| e.canonical_id);

        let resolved_correctly = match (antecedent_id, anaphor_id) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        };

        CaseResult {
            case_id: case.id.clone(),
            anaphora_type: case.anaphora_type,
            resolved_correctly,
            antecedent_found: antecedent_id.is_some(),
            anaphor_found: anaphor_id.is_some(),
            antecedent_id,
            anaphor_id,
            failure_reason: if resolved_correctly {
                None
            } else {
                Some(self.diagnose_failure(case, antecedent_id, anaphor_id))
            },
        }
    }

    /// Evaluate a single test case using the discourse-aware resolver.
    fn evaluate_case_discourse(&self, case: &AnaphoraTestCase) -> CaseResult {
        let config = DiscourseCorefConfig::default();
        let resolver = DiscourseAwareResolver::new(config, &case.text);

        // For abstract cases, check if we can find the event antecedent
        if case.anaphora_type.is_abstract() {
            // Create an Entity from the anaphor span for the resolver
            let anaphor_entity = crate::Entity::new(
                &case.anaphor.text,
                crate::EntityType::Other("Anaphor".to_string()),
                case.anaphor.start,
                case.anaphor.end,
                1.0,
            );

            // Try to find the anaphor in the text and get its antecedent
            let resolved_correctly =
                if let Some(referent) = resolver.find_discourse_antecedent(&anaphor_entity) {
                    // Check if the found antecedent overlaps with the expected one
                    let spans_overlap = referent.start < case.antecedent.end
                        && referent.end > case.antecedent.start;

                    // Or check if the referent text contains the trigger
                    let trigger_found = case
                        .antecedent
                        .trigger
                        .as_ref()
                        .map(|t| referent.text.as_ref().is_some_and(|rt| rt.contains(t)))
                        .unwrap_or(false);

                    spans_overlap || trigger_found
                } else {
                    false
                };

            CaseResult {
                case_id: case.id.clone(),
                anaphora_type: case.anaphora_type,
                resolved_correctly,
                antecedent_found: true, // If we're here, there's an antecedent in the case
                anaphor_found: true,
                antecedent_id: Some(0),
                anaphor_id: if resolved_correctly { Some(0) } else { None },
                failure_reason: if resolved_correctly {
                    None
                } else {
                    Some("Discourse resolver couldn't find event antecedent".to_string())
                },
            }
        } else {
            // For nominal cases, use the standard resolver
            self.evaluate_case(case)
        }
    }

    /// Extract entities for evaluation.
    ///
    /// This simulates what a NER system would produce.
    fn extract_entities_for_case(&self, case: &AnaphoraTestCase) -> Vec<Entity> {
        let mut entities = Vec::new();

        // For nominal cases, create person/org entities
        if case.anaphora_type == AnaphoraType::Nominal {
            // Antecedent is typically a named entity
            entities.push(Entity::new(
                &case.antecedent.text,
                self.infer_entity_type(&case.antecedent.text),
                case.antecedent.start,
                case.antecedent.end,
                0.9,
            ));

            // Anaphor might be a pronoun or definite NP
            entities.push(Entity::new(
                &case.anaphor.text,
                self.infer_entity_type(&case.anaphor.text),
                case.anaphor.start,
                case.anaphor.end,
                0.85,
            ));
        } else {
            // For abstract anaphora, we still need to create some entities
            // The resolver will try to resolve, but should fail because
            // the antecedent is an EVENT, not an entity

            // Extract any named entities from the antecedent text
            let antecedent_entities =
                self.extract_named_entities(&case.antecedent.text, case.antecedent.start);
            entities.extend(antecedent_entities);

            // Add the anaphor (This/That/It)
            entities.push(Entity::new(
                &case.anaphor.text,
                EntityType::Other("abstract_anaphor".to_string()),
                case.anaphor.start,
                case.anaphor.end,
                0.8,
            ));
        }

        entities
    }

    /// Extract named entities from text (simple heuristic).
    fn extract_named_entities(&self, text: &str, offset: usize) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Simple capitalized word detection
        for (i, c) in text.char_indices() {
            if c.is_uppercase()
                && (i == 0 || text.chars().nth(i - 1).map_or(true, |p| p.is_whitespace()))
            {
                // Find end of capitalized word
                let end = text[i..]
                    .char_indices()
                    .find(|(_, c)| c.is_whitespace() || *c == '.' || *c == ',')
                    .map(|(j, _)| i + j)
                    .unwrap_or(text.len());

                let word = &text[i..end];
                if word.len() > 1 && !self.is_sentence_starter(word, i) {
                    entities.push(Entity::new(
                        word,
                        self.infer_entity_type(word),
                        offset + i,
                        offset + end,
                        0.7,
                    ));
                }
            }
        }

        entities
    }

    /// Check if word is just a sentence starter.
    fn is_sentence_starter(&self, word: &str, pos: usize) -> bool {
        pos == 0
            && matches!(
                word.to_lowercase().as_str(),
                "the" | "a" | "an" | "this" | "that" | "it" | "he" | "she" | "they"
            )
    }

    /// Infer entity type from text.
    fn infer_entity_type(&self, text: &str) -> EntityType {
        let lower = text.to_lowercase();

        // Pronouns
        if matches!(
            lower.as_str(),
            "he" | "him" | "his" | "she" | "her" | "hers" | "they" | "them" | "their"
        ) {
            return EntityType::Person;
        }

        // Definite NPs for organizations
        if lower.starts_with("the company")
            || lower.starts_with("the firm")
            || lower.starts_with("the organization")
        {
            return EntityType::Organization;
        }

        // Common org suffixes
        if text.ends_with("Inc.") || text.ends_with("Corp.") || text.ends_with("LLC") {
            return EntityType::Organization;
        }

        // Title prefixes suggest person
        if text.starts_with("Dr.")
            || text.starts_with("Mr.")
            || text.starts_with("Ms.")
            || text.starts_with("Prof.")
        {
            return EntityType::Person;
        }

        // Default to person for proper nouns
        if text.chars().next().is_some_and(|c| c.is_uppercase()) {
            return EntityType::Person;
        }

        EntityType::Other("unknown".to_string())
    }

    /// Check if texts match (case-insensitive, ignoring punctuation).
    fn text_matches(&self, a: &str, b: &str) -> bool {
        let normalize = |s: &str| {
            s.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect::<String>()
        };
        normalize(a) == normalize(b)
    }

    /// Detect if the anaphor is a shell noun phrase (e.g., "this problem").
    ///
    /// Shell nouns are abstract nouns that encapsulate propositions.
    /// Recognizing them is a first step toward abstract anaphora resolution.
    #[must_use]
    pub fn detect_shell_noun(&self, anaphor_text: &str) -> Option<ShellNoun> {
        let words: Vec<&str> = anaphor_text.split_whitespace().collect();

        // Check for "this/that/the + noun" patterns
        if words.len() >= 2 {
            let determiner = words[0].to_lowercase();
            if matches!(
                determiner.as_str(),
                "this" | "that" | "the" | "these" | "those"
            ) {
                // Safe: words.len() >= 2 ensures last() returns Some
                let noun = words
                    .last()
                    .expect("words has at least 2 elements")
                    .to_lowercase();
                // Remove trailing punctuation
                let noun = noun.trim_matches(|c: char| !c.is_alphanumeric());

                if let Some(class) = classify_shell_noun(noun) {
                    return Some(
                        ShellNoun::new(noun, class)
                            .with_determiner(&determiner)
                            .with_full_text(anaphor_text),
                    );
                }
            }
        }

        // Check single-word shell nouns
        if words.len() == 1 {
            let noun = words[0].to_lowercase();
            let noun = noun.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(class) = classify_shell_noun(noun) {
                return Some(ShellNoun::new(noun, class).with_full_text(anaphor_text));
            }
        }

        None
    }

    /// Analyze shell noun distribution in a dataset.
    #[must_use]
    pub fn analyze_shell_nouns(&self, dataset: &AbstractAnaphoraDataset) -> ShellNounAnalysis {
        let mut analysis = ShellNounAnalysis::default();

        for case in &dataset.cases {
            if let Some(shell) = self.detect_shell_noun(&case.anaphor.text) {
                analysis.total_shell_nouns += 1;
                *analysis.by_class.entry(shell.class).or_default() += 1;

                if shell.is_demonstrative() {
                    analysis.demonstrative_count += 1;
                }

                // Check if shell noun class matches anaphora type
                let expected_types = shell.typical_antecedent_types();
                let actual_type: ReferentType = match case.anaphora_type {
                    AnaphoraType::Nominal => ReferentType::Nominal,
                    AnaphoraType::Event => ReferentType::Event,
                    AnaphoraType::Fact => ReferentType::Fact,
                    AnaphoraType::Proposition => ReferentType::Proposition,
                    AnaphoraType::Situation => ReferentType::Situation,
                };

                if expected_types.contains(&actual_type) {
                    analysis.type_match_count += 1;
                }
            }
        }

        analysis
    }

    /// Diagnose why resolution failed.
    fn diagnose_failure(
        &self,
        case: &AnaphoraTestCase,
        antecedent_id: Option<u64>,
        anaphor_id: Option<u64>,
    ) -> String {
        // Check for shell noun
        let shell_info = if let Some(shell) = self.detect_shell_noun(&case.anaphor.text) {
            format!(" [shell noun: {} ({})]", shell.lemma, shell.class.as_str())
        } else {
            String::new()
        };

        if case.anaphora_type.is_abstract() {
            return format!(
                "Abstract anaphora ({}) - resolver cannot detect event/proposition antecedents{}",
                case.anaphora_type.as_str(),
                shell_info
            );
        }

        match (antecedent_id, anaphor_id) {
            (None, None) => "Neither antecedent nor anaphor was assigned a cluster".to_string(),
            (None, Some(_)) => "Antecedent was not assigned a cluster".to_string(),
            (Some(_), None) => "Anaphor was not assigned a cluster".to_string(),
            (Some(a), Some(b)) => format!("Assigned to different clusters: {} vs {}", a, b),
        }
    }
}

// =============================================================================
// Results
// =============================================================================

/// Results for a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    /// Test case identifier
    pub case_id: String,
    /// Type of anaphora being tested
    pub anaphora_type: AnaphoraType,
    /// Whether resolution was correct
    pub resolved_correctly: bool,
    /// Whether antecedent was found/assigned a cluster
    pub antecedent_found: bool,
    /// Whether anaphor was found/assigned a cluster
    pub anaphor_found: bool,
    /// Cluster ID assigned to antecedent
    pub antecedent_id: Option<u64>,
    /// Cluster ID assigned to anaphor
    pub anaphor_id: Option<u64>,
    /// Explanation of failure if resolution failed
    pub failure_reason: Option<String>,
}

/// Results aggregated by anaphora type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeResults {
    /// Total cases of this type
    pub total: usize,
    /// Correctly resolved cases
    pub correct: usize,
}

impl TypeResults {
    fn add(&mut self, result: &CaseResult) {
        self.total += 1;
        if result.resolved_correctly {
            self.correct += 1;
        }
    }

    /// Accuracy for this type.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
}

/// Full evaluation results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationResults {
    /// Individual case results
    pub case_results: Vec<CaseResult>,
    /// Total nominal coreference cases
    pub nominal_total: usize,
    /// Correctly resolved nominal cases
    pub nominal_correct: usize,
    /// Nominal accuracy (correct/total)
    pub nominal_accuracy: f64,
    /// Total abstract anaphora cases
    pub abstract_total: usize,
    /// Correctly resolved abstract cases
    pub abstract_correct: usize,
    /// Abstract accuracy (correct/total)
    pub abstract_accuracy: f64,
    /// Results by abstract type (event, fact, etc.)
    pub by_type: HashMap<AnaphoraType, TypeResults>,
}

/// LEA (Link-based Entity-Aware) analysis split by anaphora type.
///
/// LEA is the recommended metric (Moosavi & Strube 2016) because it:
/// - Is link-based (like MUC) so it measures resolution quality
/// - Is entity-aware (like B³) so it weights by entity importance
/// - Avoids the "mention identification effect" that inflates B³/CEAF
#[derive(Debug, Clone, Default)]
pub struct LeaAnalysis {
    /// LEA scores for nominal coreference cases
    pub nominal: CorefScores,
    /// LEA scores for abstract anaphora cases
    pub abstract_anaphora: CorefScores,
}

impl LeaAnalysis {
    /// The F1 gap between nominal and abstract LEA.
    #[must_use]
    pub fn f1_gap(&self) -> f64 {
        self.nominal.f1 - self.abstract_anaphora.f1
    }

    /// Summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "LEA Analysis:\n  Nominal:  P={:.1}% R={:.1}% F1={:.1}%\n  Abstract: P={:.1}% R={:.1}% F1={:.1}%\n  Gap: {:.1}pp",
            self.nominal.precision * 100.0,
            self.nominal.recall * 100.0,
            self.nominal.f1 * 100.0,
            self.abstract_anaphora.precision * 100.0,
            self.abstract_anaphora.recall * 100.0,
            self.abstract_anaphora.f1 * 100.0,
            self.f1_gap() * 100.0
        )
    }
}

impl EvaluationResults {
    fn compute_accuracy(&mut self) {
        self.nominal_accuracy = if self.nominal_total == 0 {
            0.0
        } else {
            self.nominal_correct as f64 / self.nominal_total as f64
        };

        self.abstract_accuracy = if self.abstract_total == 0 {
            0.0
        } else {
            self.abstract_correct as f64 / self.abstract_total as f64
        };
    }

    /// The gap between nominal and abstract accuracy.
    #[must_use]
    pub fn accuracy_gap(&self) -> f64 {
        self.nominal_accuracy - self.abstract_accuracy
    }

    /// Compute LEA scores from the case results.
    ///
    /// Converts our case-level results to coreference chains and
    /// computes the LEA metric (Moosavi & Strube, 2016).
    ///
    /// Returns LEA scores separately for nominal and abstract cases.
    ///
    /// Note: LEA requires the same mention set in both gold and predicted.
    /// When prediction is incorrect, we split the gold chain into singletons.
    #[must_use]
    pub fn compute_lea_scores(&self, dataset: &AbstractAnaphoraDataset) -> LeaAnalysis {
        let mut nominal_gold = Vec::new();
        let mut nominal_pred = Vec::new();
        let mut abstract_gold = Vec::new();
        let mut abstract_pred = Vec::new();

        for (case, result) in dataset.cases.iter().zip(self.case_results.iter()) {
            let antecedent_mention = Mention::new(
                &case.antecedent.text,
                case.antecedent.start,
                case.antecedent.end,
            );
            let anaphor_mention =
                Mention::new(&case.anaphor.text, case.anaphor.start, case.anaphor.end);

            // Create gold chain: antecedent + anaphor should corefer
            let gold_chain =
                CorefChain::new(vec![antecedent_mention.clone(), anaphor_mention.clone()]);

            // Create predicted chains based on resolution result
            // LEA requires same mention set, so we always include both mentions
            let pred_chains: Vec<CorefChain> = if result.resolved_correctly {
                // Correct: both in same chain
                vec![CorefChain::new(vec![
                    antecedent_mention.clone(),
                    anaphor_mention.clone(),
                ])]
            } else {
                // Incorrect: each mention in its own singleton chain
                vec![
                    CorefChain::new(vec![antecedent_mention.clone()]),
                    CorefChain::new(vec![anaphor_mention.clone()]),
                ]
            };

            if case.anaphora_type == AnaphoraType::Nominal {
                nominal_gold.push(gold_chain);
                nominal_pred.extend(pred_chains);
            } else {
                abstract_gold.push(gold_chain);
                abstract_pred.extend(pred_chains);
            }
        }

        let nominal_lea = lea_score(&nominal_pred, &nominal_gold);
        let abstract_lea = lea_score(&abstract_pred, &abstract_gold);

        LeaAnalysis {
            nominal: CorefScores::new(nominal_lea.0, nominal_lea.1),
            abstract_anaphora: CorefScores::new(abstract_lea.0, abstract_lea.1),
        }
    }

    /// Generate a summary report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str("=== Abstract Anaphora Evaluation Results ===\n\n");

        s.push_str(&format!(
            "Nominal Coreference: {}/{} ({:.1}%)\n",
            self.nominal_correct,
            self.nominal_total,
            self.nominal_accuracy * 100.0
        ));

        s.push_str(&format!(
            "Abstract Anaphora:   {}/{} ({:.1}%)\n",
            self.abstract_correct,
            self.abstract_total,
            self.abstract_accuracy * 100.0
        ));

        s.push_str(&format!(
            "\nAccuracy Gap: {:.1} percentage points\n",
            self.accuracy_gap() * 100.0
        ));

        s.push_str("\n--- By Abstract Type ---\n");
        for (atype, results) in &self.by_type {
            s.push_str(&format!(
                "  {}: {}/{} ({:.1}%)\n",
                atype.as_str(),
                results.correct,
                results.total,
                results.accuracy() * 100.0
            ));
        }

        s.push_str("\n--- Failure Analysis ---\n");
        let failures: Vec<_> = self
            .case_results
            .iter()
            .filter(|r| !r.resolved_correctly)
            .collect();
        for result in failures.iter().take(10) {
            s.push_str(&format!(
                "  [{}] {}: {}\n",
                result.case_id,
                result.anaphora_type.as_str(),
                result.failure_reason.as_deref().unwrap_or("unknown")
            ));
        }
        if failures.len() > 10 {
            s.push_str(&format!(
                "  ... and {} more failures\n",
                failures.len() - 10
            ));
        }

        s
    }

    /// Generate HTML report.
    #[must_use]
    pub fn to_html(&self, dataset: &AbstractAnaphoraDataset) -> String {
        let mut html = String::new();

        html.push_str(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Abstract Anaphora Evaluation</title>
    <style>
        :root {
            --bg: #0d1117;
            --fg: #c9d1d9;
            --accent: #58a6ff;
            --success: #3fb950;
            --failure: #f85149;
            --warning: #d29922;
            --border: #30363d;
            --card-bg: #161b22;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--fg);
            margin: 0;
            padding: 2rem;
            line-height: 1.6;
        }
        h1, h2, h3 { color: var(--accent); margin-top: 2rem; }
        h1 { font-size: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }
        .card-value {
            font-size: 2.5rem;
            font-weight: bold;
        }
        .card-label { color: #8b949e; margin-top: 0.5rem; }
        .success { color: var(--success); }
        .failure { color: var(--failure); }
        .warning { color: var(--warning); }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        th, td {
            border: 1px solid var(--border);
            padding: 0.75rem;
            text-align: left;
        }
        th { background: var(--card-bg); color: var(--accent); }
        tr:nth-child(even) { background: rgba(255,255,255,0.02); }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .badge-success { background: rgba(63,185,80,0.2); color: var(--success); }
        .badge-failure { background: rgba(248,81,73,0.2); color: var(--failure); }
        .badge-nominal { background: rgba(88,166,255,0.2); color: var(--accent); }
        .badge-abstract { background: rgba(210,153,34,0.2); color: var(--warning); }
        .case-text {
            font-family: 'SF Mono', Monaco, monospace;
            background: var(--card-bg);
            padding: 0.75rem;
            border-radius: 4px;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }
        .antecedent { background: rgba(63,185,80,0.3); padding: 2px 4px; border-radius: 2px; }
        .anaphor { background: rgba(248,81,73,0.3); padding: 2px 4px; border-radius: 2px; }
        .conclusion {
            background: var(--card-bg);
            border-left: 4px solid var(--failure);
            padding: 1rem 1.5rem;
            margin: 2rem 0;
        }
        .chart-bar {
            height: 24px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .chart-fill {
            height: 100%;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <h1>Abstract Anaphora Evaluation Report</h1>
    <p>Demonstrating the gap between nominal coreference and abstract anaphora resolution.</p>
"#,
        );

        // Summary cards
        html.push_str(
            r#"
    <div class="summary-cards">
        <div class="card">
            <div class="card-value success">"#,
        );
        html.push_str(&format!("{:.0}%", self.nominal_accuracy * 100.0));
        html.push_str(
            r#"</div>
            <div class="card-label">Nominal Accuracy</div>
        </div>
        <div class="card">
            <div class="card-value failure">"#,
        );
        html.push_str(&format!("{:.0}%", self.abstract_accuracy * 100.0));
        html.push_str(
            r#"</div>
            <div class="card-label">Abstract Accuracy</div>
        </div>
        <div class="card">
            <div class="card-value warning">"#,
        );
        html.push_str(&format!("{:.0}pp", self.accuracy_gap() * 100.0));
        html.push_str(
            r#"</div>
            <div class="card-label">Performance Gap</div>
        </div>
        <div class="card">
            <div class="card-value">"#,
        );
        html.push_str(&format!("{}", self.case_results.len()));
        html.push_str(
            r#"</div>
            <div class="card-label">Test Cases</div>
        </div>
    </div>
"#,
        );

        // Accuracy by type
        html.push_str(
            r#"
    <h2>Accuracy by Anaphora Type</h2>
    <table>
        <tr>
            <th>Type</th>
            <th>Correct</th>
            <th>Total</th>
            <th>Accuracy</th>
            <th>Visual</th>
        </tr>
        <tr>
            <td><span class="badge badge-nominal">Nominal</span></td>
            <td>"#,
        );
        html.push_str(&format!("{}", self.nominal_correct));
        html.push_str("</td><td>");
        html.push_str(&format!("{}", self.nominal_total));
        html.push_str("</td><td class=\"success\">");
        html.push_str(&format!("{:.1}%", self.nominal_accuracy * 100.0));
        html.push_str(
            r#"</td>
            <td><div class="chart-bar"><div class="chart-fill" style="width: "#,
        );
        html.push_str(&format!("{}%", (self.nominal_accuracy * 100.0) as u32));
        html.push_str(
            r#"; background: var(--success);"></div></div></td>
        </tr>"#,
        );

        for (atype, results) in &self.by_type {
            html.push_str(&format!(r#"
        <tr>
            <td><span class="badge badge-abstract">{}</span></td>
            <td>{}</td>
            <td>{}</td>
            <td class="failure">{:.1}%</td>
            <td><div class="chart-bar"><div class="chart-fill" style="width: {}%; background: var(--failure);"></div></div></td>
        </tr>"#,
                atype.as_str(),
                results.correct,
                results.total,
                results.accuracy() * 100.0,
                (results.accuracy() * 100.0) as u32
            ));
        }

        html.push_str("</table>");

        // Conclusion
        html.push_str(
            r#"
    <div class="conclusion">
        <h3 style="margin-top: 0;">Conclusion</h3>
        <p>The current <code>SimpleCorefResolver</code> achieves <strong class="success">"#,
        );
        html.push_str(&format!("{:.0}%", self.nominal_accuracy * 100.0));
        html.push_str(
            r#"</strong> accuracy on nominal coreference but 
        <strong class="failure">"#,
        );
        html.push_str(&format!("{:.0}%", self.abstract_accuracy * 100.0));
        html.push_str(
            r#"</strong> on abstract anaphora.</p>
        <p>This "#,
        );
        html.push_str(&format!("{:.0}", self.accuracy_gap() * 100.0));
        html.push_str(r#" percentage point gap demonstrates that:</p>
        <ul>
            <li>The resolver has <strong>no mechanism</strong> to detect event/proposition antecedents</li>
            <li>Abstract pronouns ("this", "that") are linked to the nearest <em>entity</em>, not to events</li>
            <li>Solving this requires event extraction + discourse structure modeling</li>
        </ul>
    </div>
"#);

        // Detailed case results
        html.push_str(
            r#"
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Type</th>
            <th>Result</th>
            <th>Text (highlighted)</th>
            <th>Failure Reason</th>
        </tr>"#,
        );

        for (case, result) in dataset.cases.iter().zip(self.case_results.iter()) {
            let badge_class = if result.resolved_correctly {
                "badge-success"
            } else {
                "badge-failure"
            };
            let result_text = if result.resolved_correctly {
                "PASS"
            } else {
                "FAIL"
            };

            // Highlight text
            let mut highlighted = case.text.clone();
            // Insert spans (do anaphor first if it comes after antecedent)
            if case.anaphor.start > case.antecedent.end {
                highlighted.insert_str(case.anaphor.end, "</span>");
                highlighted.insert_str(case.anaphor.start, "<span class=\"anaphor\">");
                highlighted.insert_str(case.antecedent.end, "</span>");
                highlighted.insert_str(case.antecedent.start, "<span class=\"antecedent\">");
            }

            html.push_str(&format!(
                r#"
        <tr>
            <td>{}</td>
            <td><span class="badge {}">{}</span></td>
            <td><span class="badge {}">{}</span></td>
            <td class="case-text">{}</td>
            <td>{}</td>
        </tr>"#,
                result.case_id,
                if result.anaphora_type.is_abstract() {
                    "badge-abstract"
                } else {
                    "badge-nominal"
                },
                result.anaphora_type.as_str(),
                badge_class,
                result_text,
                highlighted,
                result.failure_reason.as_deref().unwrap_or("-")
            ));
        }

        html.push_str("</table>");

        // Footer
        html.push_str(r#"
    <footer style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: #8b949e; font-size: 0.9rem;">
        <p>Generated by <code>anno::eval::abstract_anaphora</code></p>
        <p>See <code>docs/ABSTRACT_ANAPHORA_RESEARCH.md</code> for research background.</p>
    </footer>
</body>
</html>"#);

        html
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let dataset = AbstractAnaphoraDataset::standard();
        let stats = dataset.stats();

        assert!(stats.total > 0, "Dataset should have cases");
        assert!(stats.nominal > 0, "Should have nominal cases");
        assert!(stats.abstract_total() > 0, "Should have abstract cases");

        println!("Dataset stats: {:?}", stats);
    }

    #[test]
    fn test_anaphora_types() {
        assert!(!AnaphoraType::Nominal.is_abstract());
        assert!(AnaphoraType::Event.is_abstract());
        assert!(AnaphoraType::Fact.is_abstract());
        assert!(AnaphoraType::Proposition.is_abstract());
        assert!(AnaphoraType::Situation.is_abstract());
    }

    #[test]
    fn test_evaluation_runs() {
        let dataset = AbstractAnaphoraDataset::standard();
        let evaluator = AbstractAnaphoraEvaluator::default();
        let results = evaluator.evaluate(&dataset);

        println!("{}", results.summary());

        // Nominal should do better than abstract
        // (We expect nominal to be decent, abstract to be near 0)
        assert!(
            results.nominal_accuracy >= results.abstract_accuracy,
            "Nominal accuracy ({:.1}%) should be >= abstract ({:.1}%)",
            results.nominal_accuracy * 100.0,
            results.abstract_accuracy * 100.0
        );
    }

    #[test]
    fn test_accuracy_gap_exists() {
        let dataset = AbstractAnaphoraDataset::standard();
        let evaluator = AbstractAnaphoraEvaluator::default();
        let results = evaluator.evaluate(&dataset);

        // The gap should be substantial (this is the point of the research)
        let gap = results.accuracy_gap();
        println!("Accuracy gap: {:.1} percentage points", gap * 100.0);

        // We expect at least some gap - if nominal works at all
        // and abstract doesn't, gap should be positive
        if results.nominal_accuracy > 0.0 {
            assert!(
                gap > 0.0,
                "Expected positive accuracy gap, got {:.1}pp",
                gap * 100.0
            );
        }
    }

    #[test]
    fn test_html_generation() {
        let dataset = AbstractAnaphoraDataset::standard();
        let evaluator = AbstractAnaphoraEvaluator::default();
        let results = evaluator.evaluate(&dataset);

        let html = results.to_html(&dataset);
        assert!(html.contains("Abstract Anaphora Evaluation"));
        assert!(html.contains("Nominal Accuracy"));
        assert!(html.contains("Abstract Accuracy"));
    }

    // =================================================================
    // Domain-Specific Dataset Tests
    // =================================================================

    #[test]
    fn test_legal_domain_dataset() {
        let dataset = AbstractAnaphoraDataset::legal_domain();
        let stats = dataset.stats();

        assert!(
            stats.total >= 8,
            "Legal domain should have at least 8 cases"
        );
        assert!(stats.abstract_total() >= 7, "Most should be abstract");

        // Should have at least one nominal for contrast
        assert!(stats.nominal >= 1, "Should include nominal baseline case");
    }

    #[test]
    fn test_medical_domain_dataset() {
        let dataset = AbstractAnaphoraDataset::medical_domain();
        let stats = dataset.stats();

        assert!(
            stats.total >= 8,
            "Medical domain should have at least 8 cases"
        );
        assert!(
            stats.event >= 3,
            "Medical should have event cases (procedures)"
        );
    }

    #[test]
    fn test_financial_domain_dataset() {
        let dataset = AbstractAnaphoraDataset::financial_domain();
        let stats = dataset.stats();

        assert!(
            stats.total >= 8,
            "Financial domain should have at least 8 cases"
        );
        assert!(
            stats.event >= 4,
            "Financial should have event cases (transactions)"
        );
    }

    #[test]
    fn test_scientific_domain_dataset() {
        let dataset = AbstractAnaphoraDataset::scientific_domain();
        let stats = dataset.stats();

        assert!(
            stats.total >= 8,
            "Scientific domain should have at least 8 cases"
        );
        assert!(
            stats.fact >= 3,
            "Scientific should have fact cases (findings)"
        );
    }

    #[test]
    fn test_news_domain_dataset() {
        let dataset = AbstractAnaphoraDataset::news_domain();
        let stats = dataset.stats();

        assert!(stats.total >= 8, "News domain should have at least 8 cases");
        assert!(stats.event >= 5, "News should have many event cases");
    }

    #[test]
    fn test_challenging_cases_dataset() {
        let dataset = AbstractAnaphoraDataset::challenging_cases();
        let stats = dataset.stats();

        assert!(
            stats.total >= 10,
            "Challenging cases should have at least 10 cases"
        );

        // These are specifically hard cases, so no nominal baselines
        assert!(
            stats.abstract_total() == stats.total,
            "All challenging cases should be abstract"
        );
    }

    #[test]
    fn test_comprehensive_dataset() {
        let dataset = AbstractAnaphoraDataset::comprehensive();
        let stats = dataset.stats();

        // Comprehensive should include all other datasets
        let extended_stats = AbstractAnaphoraDataset::extended().stats();
        let legal_count = AbstractAnaphoraDataset::legal_domain().stats().total;
        let medical_count = AbstractAnaphoraDataset::medical_domain().stats().total;
        let financial_count = AbstractAnaphoraDataset::financial_domain().stats().total;
        let scientific_count = AbstractAnaphoraDataset::scientific_domain().stats().total;
        let news_count = AbstractAnaphoraDataset::news_domain().stats().total;
        let challenging_count = AbstractAnaphoraDataset::challenging_cases().stats().total;

        let expected_min = extended_stats.total
            + legal_count
            + medical_count
            + financial_count
            + scientific_count
            + news_count
            + challenging_count;

        assert!(
            stats.total >= expected_min,
            "Comprehensive should have at least {} cases, got {}",
            expected_min,
            stats.total
        );

        println!("Comprehensive dataset stats:");
        println!("  Total: {}", stats.total);
        println!("  Nominal: {}", stats.nominal);
        println!("  Event: {}", stats.event);
        println!("  Fact: {}", stats.fact);
        println!("  Proposition: {}", stats.proposition);
        println!("  Situation: {}", stats.situation);
    }

    #[test]
    fn test_domain_dataset_evaluation() {
        // Run evaluation on each domain dataset
        let evaluator = AbstractAnaphoraEvaluator::default();

        let domains = [
            ("Legal", AbstractAnaphoraDataset::legal_domain()),
            ("Medical", AbstractAnaphoraDataset::medical_domain()),
            ("Financial", AbstractAnaphoraDataset::financial_domain()),
            ("Scientific", AbstractAnaphoraDataset::scientific_domain()),
            ("News", AbstractAnaphoraDataset::news_domain()),
        ];

        for (name, dataset) in domains {
            let results = evaluator.evaluate(&dataset);
            println!(
                "{} domain: {:.1}% abstract accuracy",
                name,
                results.abstract_accuracy * 100.0
            );

            // Simple resolver should struggle with all domains
            assert!(
                results.abstract_accuracy < 0.5,
                "{} domain: Simple resolver shouldn't exceed 50% on abstract cases",
                name
            );
        }
    }
}
