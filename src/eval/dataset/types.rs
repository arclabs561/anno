//! Core dataset types for NER evaluation.
//!
//! These types are the foundation of the dataset API, providing a uniform
//! representation for annotated NER examples regardless of source.

use crate::eval::GoldEntity;
use crate::EntityType;
use serde::{Deserialize, Serialize};

// ============================================================================
// Domain and Difficulty Classifications
// ============================================================================

/// Domain classification for NER examples.
///
/// Domains help organize datasets and enable domain-specific filtering/analysis.
/// A single example belongs to exactly one domain.
///
/// # Example
///
/// ```rust
/// use anno::eval::dataset::Domain;
///
/// let domain = Domain::News;
/// assert_eq!(format!("{:?}", domain), "News");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum Domain {
    /// News articles (CoNLL-2003 style)
    #[default]
    News,
    /// Social media text (WNUT style - noisy, informal)
    SocialMedia,
    /// Biomedical/clinical text (diseases, drugs, genes)
    Biomedical,
    /// Financial/business text (stocks, companies, money)
    Financial,
    /// Legal documents (contracts, court cases)
    Legal,
    /// Scientific/academic text (papers, abstracts)
    Scientific,
    /// Conversational text (dialogue, chat)
    Conversational,
    /// Technical documentation (code, manuals)
    Technical,
    /// Historical text (archaic language, historical figures)
    Historical,
    /// Sports reporting
    Sports,
    /// Entertainment news (movies, music, celebrities)
    Entertainment,
    /// Political news and discourse
    Politics,
    /// E-commerce (products, prices, brands)
    Ecommerce,
    /// Academic text (citations, institutions)
    Academic,
    /// Email communications
    Email,
    /// Weather reports and forecasts
    Weather,
    /// Travel content (destinations, hotels)
    Travel,
    /// Food and restaurant content
    Food,
    /// Real estate listings
    RealEstate,
    /// Cybersecurity (CVEs, malware, threat actors)
    Cybersecurity,
    /// Multilingual text with native scripts
    Multilingual,
}

impl Domain {
    /// Returns all available domain variants.
    pub fn all() -> &'static [Domain] {
        &[
            Domain::News,
            Domain::SocialMedia,
            Domain::Biomedical,
            Domain::Financial,
            Domain::Legal,
            Domain::Scientific,
            Domain::Conversational,
            Domain::Technical,
            Domain::Historical,
            Domain::Sports,
            Domain::Entertainment,
            Domain::Politics,
            Domain::Ecommerce,
            Domain::Academic,
            Domain::Email,
            Domain::Weather,
            Domain::Travel,
            Domain::Food,
            Domain::RealEstate,
            Domain::Cybersecurity,
            Domain::Multilingual,
        ]
    }

    /// Human-readable name for the domain.
    pub fn name(&self) -> &'static str {
        match self {
            Domain::News => "News",
            Domain::SocialMedia => "Social Media",
            Domain::Biomedical => "Biomedical",
            Domain::Financial => "Financial",
            Domain::Legal => "Legal",
            Domain::Scientific => "Scientific",
            Domain::Conversational => "Conversational",
            Domain::Technical => "Technical",
            Domain::Historical => "Historical",
            Domain::Sports => "Sports",
            Domain::Entertainment => "Entertainment",
            Domain::Politics => "Politics",
            Domain::Ecommerce => "E-commerce",
            Domain::Academic => "Academic",
            Domain::Email => "Email",
            Domain::Weather => "Weather",
            Domain::Travel => "Travel",
            Domain::Food => "Food",
            Domain::RealEstate => "Real Estate",
            Domain::Cybersecurity => "Cybersecurity",
            Domain::Multilingual => "Multilingual",
        }
    }
}

/// Difficulty level for NER examples.
///
/// Difficulty is subjective but useful for progressive evaluation:
/// - Start with Easy to verify basic functionality
/// - Progress to Medium for realistic performance estimates
/// - Use Hard/Adversarial for stress testing
///
/// # Example
///
/// ```rust
/// use anno::eval::dataset::Difficulty;
///
/// let difficulty = Difficulty::Hard;
/// assert!(difficulty.is_challenging());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Difficulty {
    /// Simple entities, clear context, unambiguous
    #[default]
    Easy,
    /// Multiple entities, some ambiguity, typical real-world
    Medium,
    /// Complex sentences, nested entities, domain jargon
    Hard,
    /// Edge cases, adversarial examples, intentionally tricky
    Adversarial,
}

impl Difficulty {
    /// Returns all difficulty levels in order.
    pub fn all() -> &'static [Difficulty] {
        &[
            Difficulty::Easy,
            Difficulty::Medium,
            Difficulty::Hard,
            Difficulty::Adversarial,
        ]
    }

    /// Returns true if this is Hard or Adversarial.
    pub fn is_challenging(&self) -> bool {
        matches!(self, Difficulty::Hard | Difficulty::Adversarial)
    }

    /// Numeric difficulty score (0-3) for sorting/filtering.
    pub fn score(&self) -> u8 {
        match self {
            Difficulty::Easy => 0,
            Difficulty::Medium => 1,
            Difficulty::Hard => 2,
            Difficulty::Adversarial => 3,
        }
    }
}

// ============================================================================
// Annotated Example
// ============================================================================

/// A single annotated NER example with text, entities, and metadata.
///
/// This is the canonical type for representing labeled NER data, whether
/// from synthetic generation, manual annotation, or loaded from files.
///
/// # Example
///
/// ```rust
/// use anno::eval::dataset::{AnnotatedExample, Domain, Difficulty};
/// use anno::eval::GoldEntity;
/// use anno::EntityType;
///
/// let example = AnnotatedExample::new(
///     "Microsoft announced earnings",
///     vec![GoldEntity::new("Microsoft", EntityType::Organization, 0)],
/// );
/// assert_eq!(example.text, "Microsoft announced earnings");
/// assert_eq!(example.entities.len(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedExample {
    /// The input text.
    pub text: String,
    /// Gold standard entity annotations.
    pub entities: Vec<GoldEntity>,
    /// Domain classification (e.g., News, Biomedical).
    pub domain: Domain,
    /// Difficulty level (e.g., Easy, Hard).
    pub difficulty: Difficulty,
}

impl AnnotatedExample {
    /// Create a new annotated example with default domain/difficulty.
    pub fn new(text: impl Into<String>, entities: Vec<GoldEntity>) -> Self {
        Self {
            text: text.into(),
            entities,
            domain: Domain::default(),
            difficulty: Difficulty::default(),
        }
    }

    /// Convenience constructor from text and entity tuples (alias for `from_tuples`).
    ///
    /// This is the same as `from_tuples` but with the shorter name commonly used
    /// in test code.
    #[must_use]
    pub fn simple(text: impl Into<String>, entities: Vec<(&str, &str)>) -> Self {
        Self::from_tuples(text, entities)
    }

    /// Create with explicit domain and difficulty.
    pub fn with_metadata(
        text: impl Into<String>,
        entities: Vec<GoldEntity>,
        domain: Domain,
        difficulty: Difficulty,
    ) -> Self {
        Self {
            text: text.into(),
            entities,
            domain,
            difficulty,
        }
    }

    /// Convenience constructor for doctests and simple examples.
    ///
    /// Entity positions are computed by finding each entity text within the input.
    ///
    /// # Panics
    ///
    /// Panics if an entity text is not found in the input text.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::eval::dataset::AnnotatedExample;
    ///
    /// let example = AnnotatedExample::from_tuples(
    ///     "John works at Google.",
    ///     vec![("John", "PER"), ("Google", "ORG")],
    /// );
    /// assert_eq!(example.entities.len(), 2);
    /// ```
    pub fn from_tuples(text: impl Into<String>, entities: Vec<(&str, &str)>) -> Self {
        let text = text.into();
        let gold_entities = entities
            .into_iter()
            .map(|(entity_text, entity_type_str)| {
                let start = text.find(entity_text).unwrap_or_else(|| {
                    panic!("Entity '{}' not found in text '{}'", entity_text, text)
                });
                let entity_type = EntityType::from_label(entity_type_str);
                GoldEntity::new(entity_text, entity_type, start)
            })
            .collect();

        Self {
            text,
            entities: gold_entities,
            domain: Domain::default(),
            difficulty: Difficulty::default(),
        }
    }

    /// Set the domain and return self (builder pattern).
    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.domain = domain;
        self
    }

    /// Set the difficulty and return self (builder pattern).
    pub fn with_difficulty(mut self, difficulty: Difficulty) -> Self {
        self.difficulty = difficulty;
        self
    }

    /// Returns true if this example has no entities (negative example).
    pub fn is_negative(&self) -> bool {
        self.entities.is_empty()
    }

    /// Returns the number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns unique entity types in this example.
    pub fn entity_types(&self) -> Vec<&EntityType> {
        let mut types: Vec<_> = self.entities.iter().map(|e| &e.entity_type).collect();
        types.sort_by_key(|t| format!("{:?}", t));
        types.dedup();
        types
    }

    /// Convert to (text, entities) tuple for evaluation functions.
    pub fn as_test_case(&self) -> (&str, &[GoldEntity]) {
        (&self.text, &self.entities)
    }

    /// Consume and convert to owned (text, entities) tuple.
    pub fn into_test_case(self) -> (String, Vec<GoldEntity>) {
        (self.text, self.entities)
    }
}

// ============================================================================
// Entity Helpers
// ============================================================================

/// Helper module for creating entities concisely in dataset definitions.
///
/// These functions are internal conveniences; external code should use
/// `GoldEntity::new()` directly.
pub(crate) mod helpers {
    use super::*;

    /// Create a standard entity.
    pub fn entity(text: &str, entity_type: EntityType, start: usize) -> GoldEntity {
        GoldEntity::new(text, entity_type, start)
    }

    /// Create a disease entity (biomedical domain).
    pub fn disease(text: &str, start: usize) -> GoldEntity {
        GoldEntity::new(
            text,
            EntityType::Custom {
                name: "DISEASE".to_string(),
                category: crate::EntityCategory::Misc,
            },
            start,
        )
    }

    /// Create a drug entity (biomedical domain).
    pub fn drug(text: &str, start: usize) -> GoldEntity {
        GoldEntity::new(
            text,
            EntityType::Custom {
                name: "DRUG".to_string(),
                category: crate::EntityCategory::Misc,
            },
            start,
        )
    }

    /// Create a gene entity (biomedical domain).
    pub fn gene(text: &str, start: usize) -> GoldEntity {
        GoldEntity::new(
            text,
            EntityType::Custom {
                name: "GENE".to_string(),
                category: crate::EntityCategory::Misc,
            },
            start,
        )
    }

    /// Create a chemical entity (biomedical domain).
    pub fn chemical(text: &str, start: usize) -> GoldEntity {
        GoldEntity::new(
            text,
            EntityType::Custom {
                name: "CHEMICAL".to_string(),
                category: crate::EntityCategory::Misc,
            },
            start,
        )
    }

    /// Create an email entity.
    pub fn entity_email(text: &str, start: usize) -> GoldEntity {
        GoldEntity::new(
            text,
            EntityType::Custom {
                name: "EMAIL".to_string(),
                category: crate::EntityCategory::Contact,
            },
            start,
        )
    }

    /// Create a URL entity.
    pub fn entity_url(text: &str, start: usize) -> GoldEntity {
        GoldEntity::new(
            text,
            EntityType::Custom {
                name: "URL".to_string(),
                category: crate::EntityCategory::Misc,
            },
            start,
        )
    }

    /// Create a phone number entity.
    pub fn entity_phone(text: &str, start: usize) -> GoldEntity {
        GoldEntity::new(
            text,
            EntityType::Custom {
                name: "PHONE".to_string(),
                category: crate::EntityCategory::Contact,
            },
            start,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_all() {
        let domains = Domain::all();
        assert!(domains.len() >= 20);
        assert!(domains.contains(&Domain::News));
        assert!(domains.contains(&Domain::Biomedical));
    }

    #[test]
    fn test_difficulty_ordering() {
        assert!(Difficulty::Easy.score() < Difficulty::Medium.score());
        assert!(Difficulty::Medium.score() < Difficulty::Hard.score());
        assert!(Difficulty::Hard.score() < Difficulty::Adversarial.score());
    }

    #[test]
    fn test_annotated_example_from_tuples() {
        let example = AnnotatedExample::from_tuples(
            "John works at Google in NYC.",
            vec![("John", "PER"), ("Google", "ORG"), ("NYC", "LOC")],
        );
        assert_eq!(example.entities.len(), 3);
        assert_eq!(example.entities[0].text, "John");
        assert_eq!(example.entities[0].start, 0);
        assert_eq!(example.entities[1].text, "Google");
        assert_eq!(example.entities[1].start, 14);
    }

    #[test]
    fn test_annotated_example_builder() {
        let example = AnnotatedExample::new("Test text", vec![])
            .with_domain(Domain::Biomedical)
            .with_difficulty(Difficulty::Hard);

        assert_eq!(example.domain, Domain::Biomedical);
        assert_eq!(example.difficulty, Difficulty::Hard);
    }

    #[test]
    fn test_is_negative() {
        let positive = AnnotatedExample::from_tuples("John is here", vec![("John", "PER")]);
        let negative = AnnotatedExample::new("No entities here", vec![]);

        assert!(!positive.is_negative());
        assert!(negative.is_negative());
    }
}
