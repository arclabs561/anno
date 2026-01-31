//! Cognitive error taxonomy for NER systems.
//!
//! # Beyond Surface Symptoms
//!
//! Traditional error analysis categorizes errors by their symptoms:
//! - Boundary error: wrong span
//! - Type error: wrong label
//! - False positive/negative: extra/missing prediction
//!
//! This module asks: **why** did the error happen?
//!
//! # Cognitive Error Causes
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                     COGNITIVE ERROR TAXONOMY                                │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │  SURFACE SYMPTOM                    POSSIBLE COGNITIVE CAUSES               │
//! │  ────────────────                   ────────────────────────                 │
//! │                                                                             │
//! │  Boundary Error      →  Context Window Insufficient                         │
//! │    "John" vs          │  Adjacent Token Confusion                           │
//! │    "John Smith"       │  Compositional Name Pattern Missing                 │
//! │                       │  Grammatical Structure Misread                      │
//! │                                                                             │
//! │  Type Error          →  World Knowledge Gap                                 │
//! │    "Apple" as LOC     │  Polysemy/Homonym Confusion                         │
//! │    instead of ORG     │  Type Hierarchy Confusion                           │
//! │                       │  Cultural Context Missing                           │
//! │                                                                             │
//! │  False Positive      →  Pattern Overfitting                                 │
//! │    "meeting" as       │  Capitalization Overreliance                        │
//! │    ORG                │  False Trigger Pattern                              │
//! │                       │  Entity-Like Non-Entity                             │
//! │                                                                             │
//! │  False Negative      →  Out-of-Vocabulary Entity                            │
//! │    missed "Zelensky"  │  Rare Name Pattern                                  │
//! │                       │  Novel Entity Type                                  │
//! │                       │  Low-Resource Language Pattern                      │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use anno::eval::cognitive_errors::{CognitiveErrorAnalyzer, ErrorCause, ErrorEvidence};
//! use anno::eval::error_analysis::{ErrorCategory, ErrorInstance, EntityInfo};
//!
//! let analyzer = CognitiveErrorAnalyzer::default();
//!
//! // Analyze why "Apple" was predicted as LOC instead of ORG
//! let error = ErrorInstance {
//!     category: ErrorCategory::TypeError,
//!     predicted: Some(EntityInfo {
//!         text: "Apple".to_string(),
//!         entity_type: "LOC".to_string(),
//!         span: (0, 5),
//!     }),
//!     gold: Some(EntityInfo {
//!         text: "Apple".to_string(),
//!         entity_type: "ORG".to_string(),
//!         span: (0, 5),
//!     }),
//!     description: "Type confusion".to_string(),
//! };
//!
//! let context = "Apple announced new products.";
//! let diagnosis = analyzer.diagnose(&error, context);
//!
//! // Might return ErrorCause::Polysemy { ambiguous_word: "Apple", ... }
//! ```
//!
//! # Research Background
//!
//! Based on:
//! - Ratinov & Roth (2009): "Design Challenges and Misconceptions in NER"
//! - Akbik et al. (2018): "Contextual String Embeddings for Sequence Labeling"
//! - Fu et al. (2020): "Rethinking Generalization of NER"

use super::error_analysis::{ErrorCategory, ErrorInstance};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Cognitive Error Causes
// =============================================================================

/// Root cause of a prediction error.
///
/// Unlike surface symptoms (boundary error, type error), these categorize
/// the underlying cognitive failure that led to the error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorCause {
    // =========================================================================
    // Context and Window Failures
    // =========================================================================

    /// Context window was too small to resolve the entity.
    ///
    /// Example: "He met Jobs" - need prior context to know "Jobs" = Steve Jobs
    InsufficientContext {
        /// How much context would be needed (estimated sentences)
        estimated_context_needed: usize,
        /// Evidence for this diagnosis
        evidence: String,
    },

    /// Adjacent tokens confused the boundary detection.
    ///
    /// Example: "New York Times" → "New York" (missed "Times")
    AdjacentTokenConfusion {
        /// The confusing adjacent tokens
        adjacent_tokens: Vec<String>,
        /// Whether left or right boundary was affected
        boundary_side: BoundarySide,
    },

    /// Document structure (headers, lists) affected extraction.
    ///
    /// Example: "CEO\nJohn Smith" - newline breaks entity
    DocumentStructureConfusion {
        /// Type of structure that caused confusion
        structure_type: StructureType,
    },

    // =========================================================================
    // World Knowledge Gaps
    // =========================================================================

    /// Word has multiple meanings (Apple = company/fruit).
    ///
    /// Common cause of type errors for ambiguous words.
    Polysemy {
        /// The ambiguous word
        ambiguous_word: String,
        /// Senses that were confused
        confused_senses: Vec<String>,
        /// Context cues that should have helped (but didn't)
        missed_context_cues: Vec<String>,
    },

    /// Missing background knowledge about entity.
    ///
    /// Example: Not knowing "NVIDIA" is a company, not a person
    WorldKnowledgeGap {
        /// What knowledge was missing
        missing_knowledge: String,
        /// Type of knowledge (company, historical figure, etc.)
        knowledge_type: KnowledgeType,
    },

    /// Confusion in entity type hierarchy.
    ///
    /// Example: "University of Michigan" - is it ORG or LOC?
    TypeHierarchyConfusion {
        /// The type that was predicted
        predicted_type: String,
        /// The type it should have been
        gold_type: String,
        /// Why they're often confused
        confusion_reason: String,
    },

    // =========================================================================
    // Pattern and Learning Failures
    // =========================================================================

    /// Model over-relied on surface patterns.
    ///
    /// Example: Predicting all capitalized words as entities
    PatternOverfitting {
        /// The pattern that was overfit
        pattern: String,
        /// Why the pattern doesn't generalize here
        failure_reason: String,
    },

    /// Entity appears in non-entity-like context.
    ///
    /// Example: "apple pie" - not the company
    ContextualDeactivation {
        /// The deactivating context
        deactivating_phrase: String,
    },

    /// Training data distribution mismatch.
    ///
    /// Example: Biomedical entities in news text
    DomainMismatch {
        /// Expected domain
        expected_domain: String,
        /// Actual domain
        actual_domain: Option<String>,
    },

    // =========================================================================
    // Linguistic and Cultural Failures
    // =========================================================================

    /// Entity from unfamiliar linguistic/cultural context.
    ///
    /// Example: Chinese names with different structure than Western names
    CulturalLinguisticGap {
        /// The cultural/linguistic context
        context: String,
        /// Why it's unfamiliar to the model
        unfamiliarity_reason: String,
    },

    /// Nested entity handling failed.
    ///
    /// Example: "Bank of America" contains "America" (LOC in ORG)
    NestedEntityFailure {
        /// The outer entity
        outer_entity: String,
        /// The inner entity that was incorrectly extracted
        inner_entity: String,
    },

    /// Rare or novel entity pattern.
    ///
    /// Example: New company name that doesn't match training patterns
    NovelEntityPattern {
        /// What makes this pattern novel
        novelty_description: String,
    },

    // =========================================================================
    // Coreference and Discourse Failures
    // =========================================================================

    /// Failed to link coreferent mentions.
    ///
    /// Example: "Microsoft" ... "the company" ... "it" - didn't link
    CoreferenceFailure {
        /// The mentions that should have been linked
        unlinked_mentions: Vec<String>,
    },

    /// Abstract anaphor resolution failed.
    ///
    /// Example: "The invasion shocked everyone. This was unexpected."
    AbstractAnaphoraFailure {
        /// The abstract anaphor
        anaphor: String,
        /// What it should have referred to
        intended_referent: String,
    },

    /// Cross-document entity not recognized.
    ///
    /// Example: Same person mentioned in different articles
    CrossDocumentFailure {
        /// Evidence for cross-document entity
        evidence: String,
    },

    // =========================================================================
    // Ambiguity and Uncertainty
    // =========================================================================

    /// Genuinely ambiguous - reasonable annotators might disagree.
    ///
    /// Example: "Washington" - person, state, or city?
    GenuineAmbiguity {
        /// The ambiguous text
        ambiguous_text: String,
        /// Plausible interpretations
        interpretations: Vec<String>,
    },

    /// Annotation error (gold label is wrong).
    ///
    /// Example: Gold says ORG but text clearly refers to a person
    PossibleAnnotationError {
        /// Evidence suggesting annotation error
        evidence: String,
    },

    // =========================================================================
    // Catch-all
    // =========================================================================

    /// Cause could not be determined.
    Unknown {
        /// Any available evidence
        evidence: String,
    },
}

/// Which boundary was affected in a span error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundarySide {
    /// Left boundary shifted
    Left,
    /// Right boundary shifted
    Right,
    /// Both boundaries affected
    Both,
}

/// Type of document structure causing errors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructureType {
    /// Newline or line break
    Newline,
    /// Section header
    Header,
    /// List item marker
    ListItem,
    /// Table cell boundary
    Table,
    /// Parenthetical expression
    Parenthetical,
    /// Other structure type
    Other(String),
}

/// Type of world knowledge involved in errors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeType {
    /// Company or business entity
    Company,
    /// Person or individual
    Person,
    /// Geographic location
    Location,
    /// Product or service
    Product,
    /// Non-commercial organization
    Organization,
    /// Event or occurrence
    Event,
    /// Abstract concept
    Concept,
    /// Other knowledge type
    Other(String),
}

// =============================================================================
// Error Evidence
// =============================================================================

/// Evidence supporting an error cause diagnosis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvidence {
    /// The diagnosed cause
    pub cause: ErrorCause,
    /// Confidence in this diagnosis (0-1)
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Suggested remediation
    pub remediation: Option<String>,
}

impl ErrorEvidence {
    /// Create new error evidence.
    pub fn new(cause: ErrorCause, confidence: f64) -> Self {
        Self {
            cause,
            confidence,
            evidence: vec![],
            remediation: None,
        }
    }

    /// Add evidence.
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence.push(evidence.into());
        self
    }

    /// Add remediation suggestion.
    pub fn with_remediation(mut self, remediation: impl Into<String>) -> Self {
        self.remediation = Some(remediation.into());
        self
    }
}

// =============================================================================
// Cognitive Error Analyzer
// =============================================================================

/// Analyzer that diagnoses cognitive causes of errors.
///
/// Uses heuristics and patterns to map surface errors to underlying causes.
#[derive(Debug, Clone, Default)]
pub struct CognitiveErrorAnalyzer {
    /// Known ambiguous words (word → possible types)
    pub known_ambiguous: HashMap<String, Vec<String>>,
    /// Known type confusion pairs
    pub known_confusions: Vec<(String, String)>,
}

impl CognitiveErrorAnalyzer {
    /// Create a new analyzer with default knowledge.
    #[must_use]
    pub fn new() -> Self {
        let mut analyzer = Self::default();
        analyzer.load_default_knowledge();
        analyzer
    }

    /// Load default knowledge about ambiguous words and confusions.
    pub fn load_default_knowledge(&mut self) {
        // Common ambiguous words
        self.known_ambiguous.insert(
            "apple".into(),
            vec!["ORG".into(), "MISC".into()], // company vs fruit
        );
        self.known_ambiguous.insert(
            "amazon".into(),
            vec!["ORG".into(), "LOC".into()], // company vs river/region
        );
        self.known_ambiguous.insert(
            "washington".into(),
            vec!["PER".into(), "LOC".into()], // person vs place
        );
        self.known_ambiguous.insert(
            "jordan".into(),
            vec!["PER".into(), "LOC".into()], // person vs country
        );
        self.known_ambiguous.insert(
            "china".into(),
            vec!["LOC".into(), "MISC".into()], // country vs porcelain
        );

        // Common type confusions
        self.known_confusions = vec![
            ("ORG".into(), "LOC".into()),   // Company HQs
            ("PER".into(), "ORG".into()),   // Named after founder
            ("LOC".into(), "ORG".into()),   // Government entities
            ("MISC".into(), "ORG".into()),  // Products vs companies
            ("PER".into(), "MISC".into()),  // Character names
        ];
    }

    /// Diagnose the cognitive cause of an error.
    ///
    /// # Arguments
    ///
    /// * `error` - The error instance to diagnose
    /// * `context` - The surrounding text context
    ///
    /// # Returns
    ///
    /// One or more possible causes with confidence scores.
    #[must_use]
    pub fn diagnose(&self, error: &ErrorInstance, context: &str) -> Vec<ErrorEvidence> {
        let mut diagnoses = Vec::new();

        match error.category {
            ErrorCategory::TypeError => {
                diagnoses.extend(self.diagnose_type_error(error, context));
            }
            ErrorCategory::BoundaryError => {
                diagnoses.extend(self.diagnose_boundary_error(error, context));
            }
            ErrorCategory::FalsePositive => {
                diagnoses.extend(self.diagnose_false_positive(error, context));
            }
            ErrorCategory::FalseNegative => {
                diagnoses.extend(self.diagnose_false_negative(error, context));
            }
            ErrorCategory::PartialMatch => {
                // Partial match is a combination of boundary and type issues
                diagnoses.extend(self.diagnose_type_error(error, context));
                diagnoses.extend(self.diagnose_boundary_error(error, context));
            }
        }

        // Sort by confidence
        diagnoses.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        diagnoses
    }

    fn diagnose_type_error(&self, error: &ErrorInstance, context: &str) -> Vec<ErrorEvidence> {
        let mut diagnoses = Vec::new();

        if let (Some(pred), Some(gold)) = (&error.predicted, &error.gold) {
            let text_lower = pred.text.to_lowercase();

            // Check for known ambiguous word
            if let Some(types) = self.known_ambiguous.get(&text_lower) {
                if types.contains(&pred.entity_type) && types.contains(&gold.entity_type) {
                    diagnoses.push(
                        ErrorEvidence::new(
                            ErrorCause::Polysemy {
                                ambiguous_word: pred.text.clone(),
                                confused_senses: types.clone(),
                                missed_context_cues: self.extract_context_cues(context),
                            },
                            0.85,
                        )
                        .with_evidence(format!(
                            "'{}' is known to be ambiguous between {:?}",
                            pred.text, types
                        ))
                        .with_remediation("Add disambiguation features or entity linking"),
                    );
                }
            }

            // Check for type hierarchy confusion
            let is_known_confusion = self.known_confusions.iter().any(|(t1, t2)| {
                (t1 == &pred.entity_type && t2 == &gold.entity_type)
                    || (t2 == &pred.entity_type && t1 == &gold.entity_type)
            });

            if is_known_confusion {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::TypeHierarchyConfusion {
                            predicted_type: pred.entity_type.clone(),
                            gold_type: gold.entity_type.clone(),
                            confusion_reason: format!(
                                "{} and {} are commonly confused",
                                pred.entity_type, gold.entity_type
                            ),
                        },
                        0.7,
                    )
                    .with_evidence("Known type confusion pair")
                    .with_remediation("Add type-distinguishing features or multi-task learning"),
                );
            }

            // Check for cultural/linguistic patterns
            if self.looks_non_western_name(&pred.text) {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::CulturalLinguisticGap {
                            context: pred.text.clone(),
                            unfamiliarity_reason: "Non-Western name pattern".into(),
                        },
                        0.6,
                    )
                    .with_evidence("Name pattern differs from Western convention")
                    .with_remediation("Add multilingual training data"),
                );
            }

            // Check for nested entity confusion
            if context.contains(&format!("{} of", &pred.text))
                || context.contains(&format!("of {}", &pred.text))
            {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::NestedEntityFailure {
                            outer_entity: context
                                .split_whitespace()
                                .take(5)
                                .collect::<Vec<_>>()
                                .join(" "),
                            inner_entity: pred.text.clone(),
                        },
                        0.55,
                    )
                    .with_evidence("Entity appears in 'X of Y' or 'of X' pattern")
                    .with_remediation("Use nested NER model"),
                );
            }
        }

        if diagnoses.is_empty() {
            diagnoses.push(ErrorEvidence::new(
                ErrorCause::Unknown {
                    evidence: error.description.clone(),
                },
                0.3,
            ));
        }

        diagnoses
    }

    fn diagnose_boundary_error(&self, error: &ErrorInstance, context: &str) -> Vec<ErrorEvidence> {
        let mut diagnoses = Vec::new();

        if let (Some(pred), Some(gold)) = (&error.predicted, &error.gold) {
            let pred_len = pred.span.1 - pred.span.0;
            let gold_len = gold.span.1 - gold.span.0;

            // Determine boundary side
            let boundary_side = if pred.span.0 != gold.span.0 && pred.span.1 != gold.span.1 {
                BoundarySide::Both
            } else if pred.span.0 != gold.span.0 {
                BoundarySide::Left
            } else {
                BoundarySide::Right
            };

            // Check for adjacent token confusion
            let adjacent_tokens: Vec<String> = context
                .split_whitespace()
                .filter(|w| {
                    let w_lower = w.to_lowercase();
                    gold.text.to_lowercase().contains(&w_lower)
                        && !pred.text.to_lowercase().contains(&w_lower)
                })
                .take(3)
                .map(String::from)
                .collect();

            if !adjacent_tokens.is_empty() {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::AdjacentTokenConfusion {
                            adjacent_tokens: adjacent_tokens.clone(),
                            boundary_side,
                        },
                        0.75,
                    )
                    .with_evidence(format!("Missed adjacent tokens: {:?}", adjacent_tokens))
                    .with_remediation("Use boundary-aware training or CRF layer"),
                );
            }

            // Check for document structure issues
            if context.contains('\n')
                || context.contains('\t')
                || context.contains("  ")
            {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::DocumentStructureConfusion {
                            structure_type: if context.contains('\n') {
                                StructureType::Newline
                            } else {
                                StructureType::Other("whitespace".into())
                            },
                        },
                        0.6,
                    )
                    .with_evidence("Document structure (newlines, tabs) in context")
                    .with_remediation("Normalize whitespace or use structure-aware model"),
                );
            }

            // Check if span is significantly shorter (context insufficient)
            if pred_len < gold_len / 2 {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::InsufficientContext {
                            estimated_context_needed: 2,
                            evidence: format!(
                                "Predicted span ({} chars) much shorter than gold ({} chars)",
                                pred_len, gold_len
                            ),
                        },
                        0.65,
                    )
                    .with_evidence("Model truncated entity significantly")
                    .with_remediation("Increase context window size"),
                );
            }
        }

        if diagnoses.is_empty() {
            diagnoses.push(ErrorEvidence::new(
                ErrorCause::Unknown {
                    evidence: error.description.clone(),
                },
                0.3,
            ));
        }

        diagnoses
    }

    fn diagnose_false_positive(&self, error: &ErrorInstance, context: &str) -> Vec<ErrorEvidence> {
        let mut diagnoses = Vec::new();

        if let Some(pred) = &error.predicted {
            let text_lower = pred.text.to_lowercase();

            // Check for capitalization overreliance
            if pred.text.chars().next().is_some_and(|c| c.is_uppercase())
                && context.starts_with(&pred.text)
            {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::PatternOverfitting {
                            pattern: "Sentence-initial capitalization".into(),
                            failure_reason: "Capitalized at sentence start, not entity".into(),
                        },
                        0.8,
                    )
                    .with_evidence("Entity at sentence start (capitalization misleading)")
                    .with_remediation("Add sentence position features"),
                );
            }

            // Check for contextual deactivation
            let deactivating_patterns = [
                " pie", " juice", " tree", " plant", // Food/nature
                " street", " road", " avenue",       // Address components
                " movie", " book", " show",          // Media
            ];

            for pattern in &deactivating_patterns {
                if context.to_lowercase().contains(&format!("{}{}", text_lower, pattern)) {
                    diagnoses.push(
                        ErrorEvidence::new(
                            ErrorCause::ContextualDeactivation {
                                deactivating_phrase: format!(
                                    "{}{}",
                                    pred.text,
                                    pattern
                                ),
                            },
                            0.75,
                        )
                        .with_evidence(format!("'{}{}' suggests non-entity usage", text_lower, pattern))
                        .with_remediation("Add contextual features or entity linking"),
                    );
                    break;
                }
            }

            // Check for pattern overfitting (common words)
            let common_non_entities = [
                "meeting", "conference", "project", "team", "group",
                "monday", "tuesday", "wednesday", "thursday", "friday",
                "january", "february", "march", "april", "may", "june",
            ];
            if common_non_entities.contains(&text_lower.as_str()) {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::PatternOverfitting {
                            pattern: "Common capitalized word".into(),
                            failure_reason: format!("'{}' is commonly capitalized but not an entity", text_lower),
                        },
                        0.7,
                    )
                    .with_evidence("Common word that's often capitalized")
                    .with_remediation("Add negative examples for common words"),
                );
            }
        }

        if diagnoses.is_empty() {
            diagnoses.push(ErrorEvidence::new(
                ErrorCause::Unknown {
                    evidence: error.description.clone(),
                },
                0.3,
            ));
        }

        diagnoses
    }

    fn diagnose_false_negative(&self, error: &ErrorInstance, context: &str) -> Vec<ErrorEvidence> {
        let mut diagnoses = Vec::new();

        if let Some(gold) = &error.gold {
            // Check for novel entity pattern
            if !gold.text.chars().next().is_some_and(|c| c.is_uppercase()) {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::NovelEntityPattern {
                            novelty_description: "Entity not capitalized".into(),
                        },
                        0.7,
                    )
                    .with_evidence("Entity lacks standard capitalization pattern")
                    .with_remediation("Train on case-insensitive examples"),
                );
            }

            // Check for cultural/linguistic unfamiliarity
            if self.looks_non_western_name(&gold.text) {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::CulturalLinguisticGap {
                            context: gold.text.clone(),
                            unfamiliarity_reason: "Non-Western name pattern unfamiliar to model".into(),
                        },
                        0.75,
                    )
                    .with_evidence("Name follows non-Western conventions")
                    .with_remediation("Add diverse multilingual training data"),
                );
            }

            // Check for world knowledge gap
            if gold.entity_type == "ORG" || gold.entity_type == "ORGANIZATION" {
                diagnoses.push(
                    ErrorEvidence::new(
                        ErrorCause::WorldKnowledgeGap {
                            missing_knowledge: format!("'{}' is an organization", gold.text),
                            knowledge_type: KnowledgeType::Organization,
                        },
                        0.6,
                    )
                    .with_evidence("Organization not in model's knowledge")
                    .with_remediation("Add entity gazetteer or knowledge base"),
                );
            }

            // Check for domain mismatch
            let domain_keywords = [
                ("biomedical", vec!["gene", "protein", "drug", "cell", "disease"]),
                ("legal", vec!["court", "plaintiff", "defendant", "statute"]),
                ("financial", vec!["stock", "fund", "portfolio", "dividend"]),
            ];

            for (domain, keywords) in &domain_keywords {
                if keywords.iter().any(|k| context.to_lowercase().contains(k)) {
                    diagnoses.push(
                        ErrorEvidence::new(
                            ErrorCause::DomainMismatch {
                                expected_domain: "general".into(),
                                actual_domain: Some((*domain).into()),
                            },
                            0.65,
                        )
                        .with_evidence(format!("Context suggests {} domain", domain))
                        .with_remediation(format!("Fine-tune on {} data", domain)),
                    );
                    break;
                }
            }
        }

        if diagnoses.is_empty() {
            diagnoses.push(ErrorEvidence::new(
                ErrorCause::Unknown {
                    evidence: error.description.clone(),
                },
                0.3,
            ));
        }

        diagnoses
    }

    fn looks_non_western_name(&self, text: &str) -> bool {
        // Check for CJK characters
        if text.chars().any(|c| {
            matches!(c, '\u{4E00}'..='\u{9FFF}' |  // CJK Unified
                        '\u{3040}'..='\u{309F}' |  // Hiragana
                        '\u{30A0}'..='\u{30FF}' |  // Katakana
                        '\u{AC00}'..='\u{D7AF}')   // Hangul
        }) {
            return true;
        }

        // Check for Arabic characters
        if text.chars().any(|c| matches!(c, '\u{0600}'..='\u{06FF}')) {
            return true;
        }

        // Check for Cyrillic
        if text.chars().any(|c| matches!(c, '\u{0400}'..='\u{04FF}')) {
            return true;
        }

        // Check for Devanagari
        if text.chars().any(|c| matches!(c, '\u{0900}'..='\u{097F}')) {
            return true;
        }

        false
    }

    fn extract_context_cues(&self, context: &str) -> Vec<String> {
        let mut cues = Vec::new();
        let lower = context.to_lowercase();

        // Organization cues
        let org_cues = ["announced", "company", "inc", "corp", "ceo", "founded"];
        for cue in &org_cues {
            if lower.contains(cue) {
                cues.push(format!("ORG cue: '{}'", cue));
            }
        }

        // Location cues
        let loc_cues = ["in", "at", "from", "located", "city", "country"];
        for cue in &loc_cues {
            if lower.contains(&format!(" {} ", cue)) {
                cues.push(format!("LOC cue: '{}'", cue));
            }
        }

        // Person cues
        let per_cues = ["said", "told", "mr", "ms", "dr", "president"];
        for cue in &per_cues {
            if lower.contains(cue) {
                cues.push(format!("PER cue: '{}'", cue));
            }
        }

        cues
    }
}

// =============================================================================
// Aggregate Statistics
// =============================================================================

/// Aggregated statistics across cognitive error causes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CognitiveErrorStats {
    /// Count by error cause category
    pub cause_counts: HashMap<String, usize>,
    /// Most common causes
    pub top_causes: Vec<(String, usize)>,
    /// Suggested priorities for improvement
    pub improvement_priorities: Vec<String>,
}

impl CognitiveErrorStats {
    /// Compute statistics from diagnoses.
    #[must_use]
    pub fn from_diagnoses(diagnoses: &[Vec<ErrorEvidence>]) -> Self {
        let mut cause_counts: HashMap<String, usize> = HashMap::new();

        for diagnosis_list in diagnoses {
            if let Some(top) = diagnosis_list.first() {
                let cause_name = Self::cause_name(&top.cause);
                *cause_counts.entry(cause_name).or_insert(0) += 1;
            }
        }

        let mut top_causes: Vec<_> = cause_counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
        top_causes.sort_by(|a, b| b.1.cmp(&a.1));
        top_causes.truncate(10);

        let improvement_priorities = Self::suggest_priorities(&top_causes);

        Self {
            cause_counts,
            top_causes,
            improvement_priorities,
        }
    }

    fn cause_name(cause: &ErrorCause) -> String {
        match cause {
            ErrorCause::InsufficientContext { .. } => "InsufficientContext".into(),
            ErrorCause::AdjacentTokenConfusion { .. } => "AdjacentTokenConfusion".into(),
            ErrorCause::DocumentStructureConfusion { .. } => "DocumentStructure".into(),
            ErrorCause::Polysemy { .. } => "Polysemy".into(),
            ErrorCause::WorldKnowledgeGap { .. } => "WorldKnowledgeGap".into(),
            ErrorCause::TypeHierarchyConfusion { .. } => "TypeHierarchyConfusion".into(),
            ErrorCause::PatternOverfitting { .. } => "PatternOverfitting".into(),
            ErrorCause::ContextualDeactivation { .. } => "ContextualDeactivation".into(),
            ErrorCause::DomainMismatch { .. } => "DomainMismatch".into(),
            ErrorCause::CulturalLinguisticGap { .. } => "CulturalLinguisticGap".into(),
            ErrorCause::NestedEntityFailure { .. } => "NestedEntityFailure".into(),
            ErrorCause::NovelEntityPattern { .. } => "NovelEntityPattern".into(),
            ErrorCause::CoreferenceFailure { .. } => "CoreferenceFailure".into(),
            ErrorCause::AbstractAnaphoraFailure { .. } => "AbstractAnaphoraFailure".into(),
            ErrorCause::CrossDocumentFailure { .. } => "CrossDocumentFailure".into(),
            ErrorCause::GenuineAmbiguity { .. } => "GenuineAmbiguity".into(),
            ErrorCause::PossibleAnnotationError { .. } => "AnnotationError".into(),
            ErrorCause::Unknown { .. } => "Unknown".into(),
        }
    }

    fn suggest_priorities(top_causes: &[(String, usize)]) -> Vec<String> {
        let mut priorities = Vec::new();

        for (cause, count) in top_causes.iter().take(3) {
            let suggestion = match cause.as_str() {
                "Polysemy" => format!(
                    "Priority: Add entity linking or disambiguation ({} errors)",
                    count
                ),
                "WorldKnowledgeGap" => {
                    format!("Priority: Integrate knowledge base or gazetteer ({} errors)", count)
                }
                "PatternOverfitting" => {
                    format!("Priority: Add negative examples and regularization ({} errors)", count)
                }
                "CulturalLinguisticGap" => {
                    format!("Priority: Add multilingual/diverse training data ({} errors)", count)
                }
                "InsufficientContext" => {
                    format!("Priority: Increase context window size ({} errors)", count)
                }
                "DomainMismatch" => {
                    format!("Priority: Fine-tune on domain-specific data ({} errors)", count)
                }
                "TypeHierarchyConfusion" => {
                    format!("Priority: Add type-distinguishing features ({} errors)", count)
                }
                _ => format!("Address {} errors ({} occurrences)", cause, count),
            };
            priorities.push(suggestion);
        }

        priorities
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::error_analysis::EntityInfo;

    #[test]
    fn test_polysemy_detection() {
        let analyzer = CognitiveErrorAnalyzer::new();

        let error = ErrorInstance {
            category: ErrorCategory::TypeError,
            predicted: Some(EntityInfo {
                text: "Apple".to_string(),
                entity_type: "MISC".to_string(),
                span: (0, 5),
            }),
            gold: Some(EntityInfo {
                text: "Apple".to_string(),
                entity_type: "ORG".to_string(),
                span: (0, 5),
            }),
            description: "Type error".to_string(),
        };

        let diagnoses = analyzer.diagnose(&error, "Apple announced new products");
        assert!(!diagnoses.is_empty());

        let has_polysemy = diagnoses
            .iter()
            .any(|d| matches!(d.cause, ErrorCause::Polysemy { .. }));
        assert!(has_polysemy, "Should detect polysemy for 'Apple'");
    }

    #[test]
    fn test_boundary_error_diagnosis() {
        let analyzer = CognitiveErrorAnalyzer::new();

        let error = ErrorInstance {
            category: ErrorCategory::BoundaryError,
            predicted: Some(EntityInfo {
                text: "John".to_string(),
                entity_type: "PER".to_string(),
                span: (0, 4),
            }),
            gold: Some(EntityInfo {
                text: "John Smith".to_string(),
                entity_type: "PER".to_string(),
                span: (0, 10),
            }),
            description: "Boundary error".to_string(),
        };

        let diagnoses = analyzer.diagnose(&error, "John Smith works at Google");
        assert!(!diagnoses.is_empty());
    }

    #[test]
    fn test_false_positive_capitalization() {
        let analyzer = CognitiveErrorAnalyzer::new();

        let error = ErrorInstance {
            category: ErrorCategory::FalsePositive,
            predicted: Some(EntityInfo {
                text: "Meeting".to_string(),
                entity_type: "ORG".to_string(),
                span: (0, 7),
            }),
            gold: None,
            description: "False positive".to_string(),
        };

        let diagnoses = analyzer.diagnose(&error, "Meeting scheduled for tomorrow");
        assert!(!diagnoses.is_empty());

        let has_pattern_overfit = diagnoses
            .iter()
            .any(|d| matches!(d.cause, ErrorCause::PatternOverfitting { .. }));
        assert!(has_pattern_overfit, "Should detect pattern overfitting");
    }

    #[test]
    fn test_cultural_linguistic_detection() {
        let analyzer = CognitiveErrorAnalyzer::new();

        let error = ErrorInstance {
            category: ErrorCategory::FalseNegative,
            predicted: None,
            gold: Some(EntityInfo {
                text: "習近平".to_string(),
                entity_type: "PER".to_string(),
                span: (0, 3),
            }),
            description: "Missed Chinese name".to_string(),
        };

        let diagnoses = analyzer.diagnose(&error, "習近平出席會議");
        assert!(!diagnoses.is_empty());

        let has_cultural = diagnoses
            .iter()
            .any(|d| matches!(d.cause, ErrorCause::CulturalLinguisticGap { .. }));
        assert!(has_cultural, "Should detect cultural/linguistic gap");
    }

    #[test]
    fn test_aggregate_stats() {
        let diagnoses = vec![
            vec![ErrorEvidence::new(
                ErrorCause::Polysemy {
                    ambiguous_word: "Apple".into(),
                    confused_senses: vec![],
                    missed_context_cues: vec![],
                },
                0.8,
            )],
            vec![ErrorEvidence::new(
                ErrorCause::Polysemy {
                    ambiguous_word: "Amazon".into(),
                    confused_senses: vec![],
                    missed_context_cues: vec![],
                },
                0.8,
            )],
            vec![ErrorEvidence::new(
                ErrorCause::PatternOverfitting {
                    pattern: "test".into(),
                    failure_reason: "test".into(),
                },
                0.7,
            )],
        ];

        let stats = CognitiveErrorStats::from_diagnoses(&diagnoses);

        assert_eq!(stats.cause_counts.get("Polysemy"), Some(&2));
        assert_eq!(stats.cause_counts.get("PatternOverfitting"), Some(&1));
        assert!(!stats.improvement_priorities.is_empty());
    }
}

