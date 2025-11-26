//! Synthetic NER Test Datasets
#![allow(missing_docs)] // Internal evaluation types
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
//! ```rust
//! use anno::eval::synthetic::{all_datasets, datasets_by_domain, datasets_by_difficulty, Domain, Difficulty};
//!
//! // Get all datasets
//! let all = all_datasets();
//! assert!(!all.is_empty());
//!
//! // Filter by domain
//! let news = datasets_by_domain(Domain::News);
//!
//! // Filter by difficulty
//! let hard = datasets_by_difficulty(Difficulty::Hard);
//! ```

use super::datasets::GoldEntity;
use crate::EntityType;
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
                entity("Olaf Scholz", EntityType::Person, 67),
                entity("Berlin", EntityType::Location, 82),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "The European Union reached an agreement with China on trade tariffs.".into(),
            entities: vec![
                entity("European Union", EntityType::Organization, 4),
                entity("China", EntityType::Location, 45),
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
                entity("Tesla", EntityType::Organization, 22),
                entity("Palo Alto", EntityType::Location, 34),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Excited for #WWDC2024 in Cupertino! Apple is gonna announce something big"
                .into(),
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
                entity("FDA", EntityType::Organization, 45),
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
            text: "Dr. Jennifer Doudna won the Nobel Prize for CRISPR research at UC Berkeley."
                .into(),
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
                entity("München", EntityType::Location, 34),
                entity("São Paulo", EntityType::Location, 46),
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
// Structured Entities Dataset (for PatternNER testing)
// ============================================================================

/// Structured entities dataset - dates, times, money, percentages, emails, URLs, phones.
///
/// These are entities that PatternNER can reliably detect via regex patterns.
pub fn structured_dataset() -> Vec<AnnotatedExample> {
    vec![
        // Dates
        AnnotatedExample {
            text: "Meeting scheduled for 2024-01-15 at the office.".into(),
            entities: vec![entity("2024-01-15", EntityType::Date, 22)],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The deadline is January 15, 2024 for all submissions.".into(),
            entities: vec![entity("January 15, 2024", EntityType::Date, 16)],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Event on 12/31/2024 and follow-up on Jan 5, 2025.".into(),
            entities: vec![
                entity("12/31/2024", EntityType::Date, 9),
                entity("Jan 5, 2025", EntityType::Date, 37),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Medium,
        },
        // Times
        AnnotatedExample {
            text: "Call me at 3:30 PM or after 18:00 today.".into(),
            entities: vec![
                entity("3:30 PM", EntityType::Date, 11),
                entity("18:00", EntityType::Date, 28),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        // Money
        AnnotatedExample {
            text: "The project budget is $500,000 with a contingency of $50K.".into(),
            entities: vec![
                entity("$500,000", EntityType::Money, 22),
                entity("$50K", EntityType::Money, 53),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Revenue grew to €2.5 million from €1.8 million last year.".into(),
            entities: vec![
                entity("€2.5 million", EntityType::Money, 16),
                entity("€1.8 million", EntityType::Money, 34),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The acquisition cost 5 billion dollars in total.".into(),
            entities: vec![entity("5 billion dollars", EntityType::Money, 21)],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        // Percentages
        AnnotatedExample {
            text: "Sales increased by 15% while costs dropped 8.5%.".into(),
            entities: vec![
                entity("15%", EntityType::Percent, 19),
                entity("8.5%", EntityType::Percent, 43),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Completion rate: 95 percent with 5 pct margin of error.".into(),
            entities: vec![
                entity("95 percent", EntityType::Percent, 17),
                entity("5 pct", EntityType::Percent, 33),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Medium,
        },
        // Emails
        AnnotatedExample {
            text: "Contact support@example.com or sales@company.org for help.".into(),
            entities: vec![
                entity_other("support@example.com", "EMAIL", 8),
                entity_other("sales@company.org", "EMAIL", 31),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        // URLs
        AnnotatedExample {
            text: "Visit https://docs.example.com or http://api.test.io/v2 for documentation.".into(),
            entities: vec![
                entity_other("https://docs.example.com", "URL", 6),
                entity_other("http://api.test.io/v2", "URL", 34),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        // Phones
        AnnotatedExample {
            text: "Call (555) 123-4567 or +1-800-555-0199 for support.".into(),
            entities: vec![
                entity_other("(555) 123-4567", "PHONE", 5),
                entity_other("+1-800-555-0199", "PHONE", 23),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        // Mixed structured entities
        AnnotatedExample {
            text: "Invoice #123 dated 2024-03-15: $1,250.00 (10% discount). Contact: billing@corp.com".into(),
            entities: vec![
                entity("2024-03-15", EntityType::Date, 19),
                entity("$1,250.00", EntityType::Money, 31),
                entity("10%", EntityType::Percent, 42),
                entity_other("billing@corp.com", "EMAIL", 66),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Webinar on Jan 20 at 2pm EST. Register at https://events.co/webinar. Fee: $99.".into(),
            entities: vec![
                entity("Jan 20", EntityType::Date, 11),
                entity("2pm", EntityType::Date, 21),
                entity_other("https://events.co/webinar", "URL", 42),
                entity("$99", EntityType::Money, 74),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// Helper for creating Other entity types
fn entity_other(text: &str, label: &str, start: usize) -> GoldEntity {
    GoldEntity::new(text, EntityType::Other(label.to_string()), start)
}

// ============================================================================
// Sports Dataset
// ============================================================================

/// Sports domain dataset
pub fn sports_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "LeBron James scored 35 points as the Lakers defeated the Celtics 112-108.".into(),
            entities: vec![
                entity("LeBron James", EntityType::Person, 0),
                entity("Lakers", EntityType::Organization, 37),
                entity("Celtics", EntityType::Organization, 57),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Manchester United signed Cristiano Ronaldo for $15 million.".into(),
            entities: vec![
                entity("Manchester United", EntityType::Organization, 0),
                entity("Cristiano Ronaldo", EntityType::Person, 25),
                entity("$15 million", EntityType::Money, 47),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The 2024 Olympics in Paris saw 10,000 athletes from 200 countries.".into(),
            entities: vec![
                entity("Paris", EntityType::Location, 21),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Coach Bill Belichick led the Patriots to 6 Super Bowl wins.".into(),
            entities: vec![
                entity("Bill Belichick", EntityType::Person, 6),
                entity("Patriots", EntityType::Organization, 29),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Serena Williams won Wimbledon on July 14, 2018 with a 67% first serve rate.".into(),
            entities: vec![
                entity("Serena Williams", EntityType::Person, 0),
                entity("Wimbledon", EntityType::Location, 20),
                entity("July 14, 2018", EntityType::Date, 33),
                entity("67%", EntityType::Percent, 54),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Entertainment Dataset
// ============================================================================

/// Entertainment/media domain dataset
pub fn entertainment_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Taylor Swift's Eras Tour grossed $1 billion in North America.".into(),
            entities: vec![
                entity("Taylor Swift", EntityType::Person, 0),
                entity("$1 billion", EntityType::Money, 33),
                entity("North America", EntityType::Location, 47),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Netflix announced that Squid Game Season 2 premieres December 26, 2024.".into(),
            entities: vec![
                entity("Netflix", EntityType::Organization, 0),
                entity("December 26, 2024", EntityType::Date, 53),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Director Christopher Nolan filmed Oppenheimer in New Mexico.".into(),
            entities: vec![
                entity("Christopher Nolan", EntityType::Person, 9),
                entity("New Mexico", EntityType::Location, 49),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Disney acquired Marvel Studios for $4 billion in 2009.".into(),
            entities: vec![
                entity("Disney", EntityType::Organization, 0),
                entity("Marvel Studios", EntityType::Organization, 16),
                entity("$4 billion", EntityType::Money, 35),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Grammy Awards ceremony at Staples Center honored Beyoncé.".into(),
            entities: vec![
                entity("Staples Center", EntityType::Location, 30),
                entity("Beyoncé", EntityType::Person, 53),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Politics Dataset
// ============================================================================

/// Politics/government domain dataset
pub fn politics_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "President Obama visited China to meet President Xi Jinping.".into(),
            entities: vec![
                entity("Obama", EntityType::Person, 10),
                entity("China", EntityType::Location, 24),
                entity("Xi Jinping", EntityType::Person, 48),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The United Nations held a summit in Geneva on March 15, 2024.".into(),
            entities: vec![
                entity("United Nations", EntityType::Organization, 4),
                entity("Geneva", EntityType::Location, 36),
                entity("March 15, 2024", EntityType::Date, 46),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Senator Elizabeth Warren proposed a 2% wealth tax on billionaires.".into(),
            entities: vec![
                entity("Elizabeth Warren", EntityType::Person, 8),
                entity("2%", EntityType::Percent, 36),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The White House announced $50 billion in infrastructure funding.".into(),
            entities: vec![
                entity("White House", EntityType::Organization, 4),
                entity("$50 billion", EntityType::Money, 26),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "NATO members met in Brussels to discuss Ukraine security.".into(),
            entities: vec![
                entity("NATO", EntityType::Organization, 0),
                entity("Brussels", EntityType::Location, 20),
                entity("Ukraine", EntityType::Location, 40),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// E-commerce Dataset
// ============================================================================

/// E-commerce/retail domain dataset
pub fn ecommerce_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Amazon Prime Day starts July 15, 2024 with discounts up to 50%.".into(),
            entities: vec![
                entity("Amazon", EntityType::Organization, 0),
                entity("July 15, 2024", EntityType::Date, 24),
                entity("50%", EntityType::Percent, 59),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Order shipped via FedEx. Tracking: 1234567890. Arrives by 2024-12-25.".into(),
            entities: vec![
                entity("FedEx", EntityType::Organization, 18),
                entity("2024-12-25", EntityType::Date, 58),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Contact customer service at help@store.com or (800) 555-1234.".into(),
            entities: vec![
                entity_other("help@store.com", "EMAIL", 28),
                entity_other("(800) 555-1234", "PHONE", 46),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Shopify merchants processed $5.1 billion on Black Friday 2023.".into(),
            entities: vec![
                entity("Shopify", EntityType::Organization, 0),
                entity("$5.1 billion", EntityType::Money, 28),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "eBay auction ends Jan 31 at 11:59 PM. Current bid: $299.99.".into(),
            entities: vec![
                entity("eBay", EntityType::Organization, 0),
                entity("Jan 31", EntityType::Date, 18),
                entity("11:59 PM", EntityType::Date, 28),
                entity("$299.99", EntityType::Money, 51),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Travel Dataset
// ============================================================================

/// Travel/transportation domain dataset
pub fn travel_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Flight AA123 departs Los Angeles at 3:45 PM, arrives Tokyo at 9:30 PM.".into(),
            entities: vec![
                entity("Los Angeles", EntityType::Location, 21),
                entity("3:45 PM", EntityType::Date, 36),
                entity("Tokyo", EntityType::Location, 53),
                entity("9:30 PM", EntityType::Date, 62),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Hilton Paris Opera: Check-in 2024-06-15, checkout 2024-06-20. Rate: €250/night.".into(),
            entities: vec![
                entity("Hilton", EntityType::Organization, 0),
                entity("Paris", EntityType::Location, 7),
                entity("2024-06-15", EntityType::Date, 29),
                entity("2024-06-20", EntityType::Date, 50),
                entity("€250", EntityType::Money, 68),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "United Airlines offers 25% off flights to London this summer.".into(),
            entities: vec![
                entity("United Airlines", EntityType::Organization, 0),
                entity("25%", EntityType::Percent, 23),
                entity("London", EntityType::Location, 42),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Book at https://travel.example.com or call +1-888-555-0100.".into(),
            entities: vec![
                entity_other("https://travel.example.com", "URL", 8),
                entity_other("+1-888-555-0100", "PHONE", 43),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Uber ride from JFK Airport to Manhattan costs approximately $70.".into(),
            entities: vec![
                entity("Uber", EntityType::Organization, 0),
                entity("JFK Airport", EntityType::Location, 15),
                entity("Manhattan", EntityType::Location, 30),
                entity("$70", EntityType::Money, 60),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Weather Dataset
// ============================================================================

/// Weather/meteorology domain dataset
pub fn weather_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "NOAA forecasts 80% chance of rain in Seattle on December 10, 2024.".into(),
            entities: vec![
                entity("NOAA", EntityType::Organization, 0),
                entity("80%", EntityType::Percent, 15),
                entity("Seattle", EntityType::Location, 37),
                entity("December 10, 2024", EntityType::Date, 48),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Hurricane Maria hit Puerto Rico causing $90 billion in damage.".into(),
            entities: vec![
                entity("Puerto Rico", EntityType::Location, 20),
                entity("$90 billion", EntityType::Money, 40),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Weather Channel reports Tokyo at 35°C on August 1st.".into(),
            entities: vec![
                entity("Weather Channel", EntityType::Organization, 4),
                entity("Tokyo", EntityType::Location, 28),
                entity("August 1st", EntityType::Date, 45),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Academic Dataset
// ============================================================================

/// Academic/education domain dataset
pub fn academic_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Professor Stephen Hawking taught at Cambridge University until 2009.".into(),
            entities: vec![
                entity("Stephen Hawking", EntityType::Person, 10),
                entity("Cambridge University", EntityType::Organization, 36),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "MIT received $2.5 billion in research grants from NIH in 2023.".into(),
            entities: vec![
                entity("MIT", EntityType::Organization, 0),
                entity("$2.5 billion", EntityType::Money, 13),
                entity("NIH", EntityType::Organization, 50),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Stanford published research showing 45% improvement in AI accuracy.".into(),
            entities: vec![
                entity("Stanford", EntityType::Organization, 0),
                entity("45%", EntityType::Percent, 36),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Enrollment deadline: March 1, 2025. Apply at admissions@university.edu.".into(),
            entities: vec![
                entity("March 1, 2025", EntityType::Date, 21),
                entity_other("admissions@university.edu", "EMAIL", 45),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Dr. Marie Curie won Nobel Prizes in Physics and Chemistry.".into(),
            entities: vec![
                entity("Dr. Marie Curie", EntityType::Person, 0),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Easy,
        },
    ]
}

// ============================================================================
// Historical Dataset
// ============================================================================

/// Historical domain dataset
pub fn historical_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Abraham Lincoln delivered the Gettysburg Address on November 19, 1863.".into(),
            entities: vec![
                entity("Abraham Lincoln", EntityType::Person, 0),
                entity("Gettysburg", EntityType::Location, 30),
                entity("November 19, 1863", EntityType::Date, 52),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Roman Empire fell in 476 AD after ruling from Rome.".into(),
            entities: vec![
                entity("Roman Empire", EntityType::Organization, 4),
                entity("Rome", EntityType::Location, 50),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Napoleon Bonaparte was exiled to Elba in 1814.".into(),
            entities: vec![
                entity("Napoleon Bonaparte", EntityType::Person, 0),
                entity("Elba", EntityType::Location, 33),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The Wright Brothers flew at Kitty Hawk on December 17, 1903.".into(),
            entities: vec![
                entity("Wright Brothers", EntityType::Person, 4),
                entity("Kitty Hawk", EntityType::Location, 28),
                entity("December 17, 1903", EntityType::Date, 42),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Food/Restaurant Dataset  
// ============================================================================

/// Food and restaurant domain dataset
pub fn food_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Reservation at Noma Copenhagen for 8pm on Dec 31. Party of 4, $500 deposit.".into(),
            entities: vec![
                entity("Noma", EntityType::Organization, 15),
                entity("Copenhagen", EntityType::Location, 20),
                entity("8pm", EntityType::Date, 35),
                entity("Dec 31", EntityType::Date, 42),
                entity("$500", EntityType::Money, 62),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "McDonald's reported 5% sales growth in Q3 2024.".into(),
            entities: vec![
                entity("McDonald's", EntityType::Organization, 0),
                entity("5%", EntityType::Percent, 20),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Order online at https://food.delivery or call (555) 123-4567.".into(),
            entities: vec![
                entity_other("https://food.delivery", "URL", 16),
                entity_other("(555) 123-4567", "PHONE", 46),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Easy,
        },
    ]
}

// ============================================================================
// Real Estate Dataset
// ============================================================================

/// Real estate domain dataset
pub fn real_estate_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "3BR house in San Francisco listed at $1.2 million. Contact agent@realty.com.".into(),
            entities: vec![
                entity("San Francisco", EntityType::Location, 13),
                entity("$1.2 million", EntityType::Money, 37),
                entity_other("agent@realty.com", "EMAIL", 59),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Open house on Jan 15, 2025 at 2pm. Address: 123 Main St, Boston.".into(),
            entities: vec![
                entity("Jan 15, 2025", EntityType::Date, 14),
                entity("2pm", EntityType::Date, 30),
                entity("Boston", EntityType::Location, 57),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Zillow estimates 15% price increase in Austin since 2020.".into(),
            entities: vec![
                entity("Zillow", EntityType::Organization, 0),
                entity("15%", EntityType::Percent, 17),
                entity("Austin", EntityType::Location, 39),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Conversational Dataset
// ============================================================================

/// Conversational/chatbot style dataset
pub fn conversational_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Can you book me a flight to New York on March 15?".into(),
            entities: vec![
                entity("New York", EntityType::Location, 28),
                entity("March 15", EntityType::Date, 40),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "What's the weather like in Tokyo tomorrow?".into(),
            entities: vec![
                entity("Tokyo", EntityType::Location, 27),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Send $50 to John Smith at john@email.com please.".into(),
            entities: vec![
                entity("$50", EntityType::Money, 5),
                entity("John Smith", EntityType::Person, 12),
                entity_other("john@email.com", "EMAIL", 26),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Remind me to call Dr. Johnson at 3pm tomorrow.".into(),
            entities: vec![
                entity("Dr. Johnson", EntityType::Person, 18),
                entity("3pm", EntityType::Date, 33),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Aggregate Functions
// ============================================================================

/// Get all synthetic datasets combined
#[must_use]
pub fn all_datasets() -> Vec<AnnotatedExample> {
    let mut all = Vec::new();
    all.extend(conll_style_dataset());
    all.extend(social_media_dataset());
    all.extend(biomedical_dataset());
    all.extend(financial_dataset());
    all.extend(legal_dataset());
    all.extend(scientific_dataset());
    all.extend(adversarial_dataset());
    all.extend(structured_dataset());
    all.extend(sports_dataset());
    all.extend(entertainment_dataset());
    all.extend(politics_dataset());
    all.extend(ecommerce_dataset());
    all.extend(travel_dataset());
    all.extend(weather_dataset());
    all.extend(academic_dataset());
    all.extend(historical_dataset());
    all.extend(food_dataset());
    all.extend(real_estate_dataset());
    all.extend(conversational_dataset());
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
