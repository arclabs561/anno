//! Synthetic NER Test Datasets
//!
//! Comprehensive annotated datasets covering multiple domains:
//! - News (CoNLL-2003 style)
//! - Social Media (WNUT style)
//! - Biomedical, Financial, Legal, Scientific
//! - Multilingual/Unicode
//! - Adversarial edge cases
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::eval::synthetic::{all_datasets, Domain, Difficulty};
//!
//! // Get all datasets
//! let all = all_datasets();
//!
//! // Filter by domain
//! let news = datasets_by_domain(Domain::News);
//!
//! // Filter by difficulty
//! let hard = datasets_by_difficulty(Difficulty::Hard);
//! ```

use crate::EntityType;
use super::datasets::GoldEntity;
use serde::{Deserialize, Serialize};

// ============================================================================
// Dataset Structures
// ============================================================================

/// A single annotated example with text and gold entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedExample {
    /// The input text
    pub text: String,
    /// Gold standard entity annotations
    pub entities: Vec<GoldEntity>,
    /// Domain of the text
    pub domain: Domain,
    /// Difficulty level
    pub difficulty: Difficulty,
}

/// Domain classification for examples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    News,
    SocialMedia,
    Biomedical,
    Financial,
    Legal,
    Scientific,
    Conversational,
    Technical,
    Historical,
    Sports,
    Entertainment,
    Politics,
    Ecommerce,
    Academic,
    Email,
    Weather,
    Travel,
    Food,
    RealEstate,
}

/// Difficulty level for examples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Difficulty {
    /// Simple entities, clear context
    Easy,
    /// Multiple entities, some ambiguity
    Medium,
    /// Complex sentences, nested entities
    Hard,
    /// Edge cases, adversarial examples
    Adversarial,
}

// ============================================================================
// Helper for creating GoldEntity
// ============================================================================

fn entity(text: &str, entity_type: EntityType, start: usize) -> GoldEntity {
    GoldEntity::new(text, entity_type, start)
}

// ============================================================================
// CoNLL-2003 Style Dataset (News Domain)
// ============================================================================

/// News domain dataset (CoNLL-2003 style)
pub fn conll_style_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Microsoft Corp. reported strong quarterly earnings.".into(),
            entities: vec![entity("Microsoft Corp.", EntityType::Organization, 0)],
            domain: Domain::News,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "President Biden addressed the nation from Washington.".into(),
            entities: vec![
                entity("Biden", EntityType::Person, 10),
                entity("Washington", EntityType::Location, 42),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Apple CEO Tim Cook announced a partnership with Google in San Francisco.".into(),
            entities: vec![
                entity("Apple", EntityType::Organization, 0),
                entity("Tim Cook", EntityType::Person, 10),
                entity("Google", EntityType::Organization, 48),
                entity("San Francisco", EntityType::Location, 58),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "According to Reuters, Tesla's Elon Musk met with German Chancellor Olaf Scholz in Berlin.".into(),
            entities: vec![
                entity("Reuters", EntityType::Organization, 13),
                entity("Tesla", EntityType::Organization, 22),
                entity("Elon Musk", EntityType::Person, 30),
                entity("Olaf Scholz", EntityType::Person, 68),
                entity("Berlin", EntityType::Location, 83),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "The European Union reached an agreement with China on trade tariffs.".into(),
            entities: vec![
                entity("European Union", EntityType::Organization, 4),
                entity("China", EntityType::Location, 47),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Social Media Dataset (WNUT style)
// ============================================================================

/// Social media dataset (WNUT style - noisy text)
pub fn social_media_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Just saw @elonmusk at Tesla HQ in Palo Alto!".into(),
            entities: vec![
                entity("Tesla", EntityType::Organization, 21),
                entity("Palo Alto", EntityType::Location, 34),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Excited for #WWDC2024 in Cupertino! Apple is gonna announce something big".into(),
            entities: vec![
                entity("Cupertino", EntityType::Location, 25),
                entity("Apple", EntityType::Organization, 36),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "ChatGPT just dropped GPT-5 and its insane! OpenAI really did it".into(),
            entities: vec![entity("OpenAI", EntityType::Organization, 43)],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "NYC subway is delayed AGAIN smh heading to Times Square".into(),
            entities: vec![
                entity("NYC", EntityType::Location, 0),
                entity("Times Square", EntityType::Location, 43),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Biomedical Dataset
// ============================================================================

/// Biomedical/healthcare domain dataset
pub fn biomedical_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Pfizer's COVID-19 vaccine Comirnaty received FDA approval for boosters.".into(),
            entities: vec![
                entity("Pfizer", EntityType::Organization, 0),
                entity("FDA", EntityType::Organization, 46),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Researchers at Johns Hopkins University published findings on Alzheimer's treatment.".into(),
            entities: vec![entity("Johns Hopkins University", EntityType::Organization, 15)],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Mayo Clinic and Cleveland Clinic collaborated on the heart disease study.".into(),
            entities: vec![
                entity("Mayo Clinic", EntityType::Organization, 4),
                entity("Cleveland Clinic", EntityType::Organization, 20),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Financial Dataset
// ============================================================================

/// Financial/business domain dataset
pub fn financial_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "NVIDIA stock surged 15% after announcing Q4 earnings beat.".into(),
            entities: vec![
                entity("NVIDIA", EntityType::Organization, 0),
                entity("15%", EntityType::Percent, 20),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Goldman Sachs and Morgan Stanley led the $5 billion IPO.".into(),
            entities: vec![
                entity("Goldman Sachs", EntityType::Organization, 0),
                entity("Morgan Stanley", EntityType::Organization, 18),
                entity("$5 billion", EntityType::Money, 41),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Federal Reserve raised interest rates by 0.25%.".into(),
            entities: vec![
                entity("Federal Reserve", EntityType::Organization, 4),
                entity("0.25%", EntityType::Percent, 45),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Legal Dataset
// ============================================================================

/// Legal/regulatory domain dataset
pub fn legal_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "The Supreme Court ruled in favor of Apple in the Epic Games lawsuit.".into(),
            entities: vec![
                entity("Supreme Court", EntityType::Organization, 4),
                entity("Apple", EntityType::Organization, 36),
                entity("Epic Games", EntityType::Organization, 49),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Attorney General Merrick Garland announced the DOJ investigation.".into(),
            entities: vec![
                entity("Merrick Garland", EntityType::Person, 17),
                entity("DOJ", EntityType::Organization, 47),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Scientific Dataset
// ============================================================================

/// Scientific/academic domain dataset
pub fn scientific_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Dr. Jennifer Doudna won the Nobel Prize for CRISPR research at UC Berkeley.".into(),
            entities: vec![
                entity("Dr. Jennifer Doudna", EntityType::Person, 0),
                entity("UC Berkeley", EntityType::Organization, 63),
            ],
            domain: Domain::Scientific,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "NASA's James Webb Space Telescope captured images of the Carina Nebula.".into(),
            entities: vec![
                entity("NASA", EntityType::Organization, 0),
                entity("Carina Nebula", EntityType::Location, 57),
            ],
            domain: Domain::Scientific,
            difficulty: Difficulty::Hard,
        },
    ]
}

// ============================================================================
// Adversarial/Edge Case Dataset
// ============================================================================

/// Adversarial and edge case examples
pub fn adversarial_dataset() -> Vec<AnnotatedExample> {
    vec![
        // Ambiguous - is "Apple" the company or fruit?
        AnnotatedExample {
            text: "I bought an Apple at the Apple Store.".into(),
            entities: vec![entity("Apple Store", EntityType::Organization, 25)],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Nested entities
        AnnotatedExample {
            text: "The New York Times reported on the New York City subway.".into(),
            entities: vec![
                entity("New York Times", EntityType::Organization, 4),
                entity("New York City", EntityType::Location, 35),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Unusual capitalization
        AnnotatedExample {
            text: "mcdonald's announced partnership with UBER eats".into(),
            entities: vec![
                // Note: lowercase entities are challenging
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Unicode names
        AnnotatedExample {
            text: "CEO 田中太郎 announced expansion into München and São Paulo.".into(),
            entities: vec![
                entity("田中太郎", EntityType::Person, 4),
                entity("München", EntityType::Location, 38),
                entity("São Paulo", EntityType::Location, 50),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Empty text
        AnnotatedExample {
            text: "".into(),
            entities: vec![],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // No entities
        AnnotatedExample {
            text: "The quick brown fox jumps over the lazy dog.".into(),
            entities: vec![],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
    ]
}

// ============================================================================
// Aggregate Functions
// ============================================================================

/// Get all synthetic datasets combined
pub fn all_datasets() -> Vec<AnnotatedExample> {
    let mut all = Vec::new();
    all.extend(conll_style_dataset());
    all.extend(social_media_dataset());
    all.extend(biomedical_dataset());
    all.extend(financial_dataset());
    all.extend(legal_dataset());
    all.extend(scientific_dataset());
    all.extend(adversarial_dataset());
    all
}

/// Filter datasets by domain
pub fn datasets_by_domain(domain: Domain) -> Vec<AnnotatedExample> {
    all_datasets()
        .into_iter()
        .filter(|ex| ex.domain == domain)
        .collect()
}

/// Filter datasets by difficulty
pub fn datasets_by_difficulty(difficulty: Difficulty) -> Vec<AnnotatedExample> {
    all_datasets()
        .into_iter()
        .filter(|ex| ex.difficulty == difficulty)
        .collect()
}

/// Get dataset statistics
pub fn dataset_stats() -> DatasetStats {
    let all = all_datasets();
    let total_entities: usize = all.iter().map(|ex| ex.entities.len()).sum();
    
    DatasetStats {
        total_examples: all.len(),
        total_entities,
        domains: count_by_domain(&all),
        difficulties: count_by_difficulty(&all),
    }
}

/// Statistics about the synthetic datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_examples: usize,
    pub total_entities: usize,
    pub domains: std::collections::HashMap<String, usize>,
    pub difficulties: std::collections::HashMap<String, usize>,
}

fn count_by_domain(examples: &[AnnotatedExample]) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    for ex in examples {
        *counts.entry(format!("{:?}", ex.domain)).or_insert(0) += 1;
    }
    counts
}

fn count_by_difficulty(examples: &[AnnotatedExample]) -> std::collections::HashMap<String, usize> {
    let mut counts = std::collections::HashMap::new();
    for ex in examples {
        *counts.entry(format!("{:?}", ex.difficulty)).or_insert(0) += 1;
    }
    counts
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_datasets_not_empty() {
        let all = all_datasets();
        assert!(!all.is_empty(), "Should have synthetic examples");
        assert!(all.len() >= 20, "Should have at least 20 examples");
    }

    #[test]
    fn test_dataset_stats() {
        let stats = dataset_stats();
        assert!(stats.total_examples > 0);
        assert!(stats.total_entities > 0);
        assert!(!stats.domains.is_empty());
        assert!(!stats.difficulties.is_empty());
    }

    #[test]
    fn test_filter_by_domain() {
        let news = datasets_by_domain(Domain::News);
        assert!(!news.is_empty());
        assert!(news.iter().all(|ex| ex.domain == Domain::News));
    }

    #[test]
    fn test_filter_by_difficulty() {
        let easy = datasets_by_difficulty(Difficulty::Easy);
        assert!(!easy.is_empty());
        assert!(easy.iter().all(|ex| ex.difficulty == Difficulty::Easy));
    }

    #[test]
    fn test_entity_offsets_valid() {
        for example in all_datasets() {
            for entity in &example.entities {
                assert!(
                    entity.end <= example.text.len(),
                    "Entity '{}' end {} exceeds text length {} in: {}",
                    entity.text,
                    entity.end,
                    example.text.len(),
                    example.text
                );
            }
        }
    }
}

